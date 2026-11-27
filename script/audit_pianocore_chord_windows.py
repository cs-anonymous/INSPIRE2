#!/usr/bin/env python3
"""Count design1125 chord windows over the PianoCoRe-A refined splits."""

# ruff: noqa: E402

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.prepare_exp1123_measure_official import load_beat_mapper, score_geometry, windows, write_json
from src.data_process.dinr_chord_dataset import LIMIT, chord_layout, nonoverlapping_windows

SPLITS = ('train', 'test')
POLICIES = ('nonoverlapping', 'overlapping', 'overlapping_with_crops')


def count_geometry(root, score, weights):
    get_bars, _ = load_beat_mapper()
    geometry = score_geometry(root, score, get_bars)
    layout = chord_layout(geometry, np.ones(len(geometry[1]), dtype=bool))
    prefix = np.r_[0, np.cumsum(layout.notes)]

    def count(spans):
        total = Counter()
        for left, right in spans:
            if not 0 <= left < right <= len(layout.notes) or right - left > LIMIT:
                raise ValueError(f'{score}: invalid window {left}, {right}')
            total.update(windows=1, chords=right - left, notes=int(prefix[right] - prefix[left]))
        return total

    disjoint = count(nonoverlapping_windows(layout.boundaries))
    if disjoint['chords'] != len(layout.notes) or disjoint['notes'] != prefix[-1]:
        raise ValueError(f'{score}: nonoverlapping windows do not cover the score exactly')
    overlapped = list(windows(layout.boundaries, random.Random(42)))
    original = count((left, right) for left, right, _, _ in overlapped)
    crop_counts = {split: Counter() for split in weights}
    rng = {split: np.random.default_rng(int.from_bytes(hashlib.blake2b(
               f'42:{score}:{split}'.encode(), digest_size=8).digest(), 'little'))
           for split in weights}
    boundaries = layout.boundaries
    for left, right, middle, crop in overlapped:
        if crop is None:
            continue
        lo, hi = np.searchsorted(boundaries, [left, middle], side='left')
        for split, weight in weights.items():
            starts = boundaries[rng[split].integers(lo + 1, hi, size=weight)]
            crop_counts[split].update(windows=weight, chords=int(np.sum(right - starts)),
                                      notes=int(np.sum(prefix[right] - prefix[starts])))
    return weights, len(geometry[1]), disjoint, original, crop_counts


def count_job(job):
    return count_geometry(*job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pianocore', type=Path, default=ROOT.parent / 'backup/PianoCoRe')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/pianocore_chord_window_audit')
    parser.add_argument('--split-overlay', type=Path)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    source = args.pianocore / 'refined'
    historical = ROOT.parent / 'results/pianocoreA_slot_sum_pretrain4_asap_ft8_20261028/data_preflight.json'
    excluded = {row['source'] for row in json.loads(historical.read_text())['excluded_performances']}
    moved_ids = set()
    if args.split_overlay:
        moved_ids = {row['pianocore_id'] for row in json.loads(args.split_overlay.read_text())['matches']}
    groups = defaultdict(Counter)
    metadata, missing, known_exclusions = Counter(), Counter(), Counter()
    metadata_notes = Counter()
    with (args.pianocore / 'metadata.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['tier_a'] != 'True':
                continue
            original_split, score, perf, align = (row[key] for key in (
                'split', 'refined_score_midi_path', 'refined_performance_midi_path', 'refined_alignment_path'))
            if original_split not in SPLITS:
                raise ValueError(f'Unexpected split {original_split}')
            split = 'test' if row['id'] in moved_ids else original_split
            if split == 'test' and original_split == 'train':
                known_exclusions['moved_to_test'] += 1
            metadata[split] += 1
            if perf in excluded:
                known_exclusions[split] += 1
            elif not all(path and (source / path).is_file() for path in (score, perf, align)):
                missing[split] += 1
            else:
                groups[score][split] += 1
                metadata_notes[split] += int(row['refined_score_note_count'])

    jobs = [(source, score, groups[score]) for score in sorted(groups)]
    counts = {split: {policy: Counter() for policy in POLICIES} for split in SPLITS}
    score_counts, performances, normalized_notes = Counter(), Counter(), Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for weights, normalized, disjoint, original, crops in pool.map(count_job, jobs, chunksize=4):
            for split, weight in weights.items():
                score_counts[split] += 1
                performances[split] += weight
                normalized_notes[split] += weight * normalized
                for policy, base, extra in zip(POLICIES, (disjoint, original, original),
                                                (Counter(), Counter(), crops[split]), strict=True):
                    counts[split][policy].update(
                        original=weight * base['windows'], cropped=extra['windows'],
                        chords=weight * base['chords'] + extra['chords'],
                        notes=weight * base['notes'] + extra['notes'])

    result = {}
    for split in SPLITS:
        assert metadata[split] == known_exclusions[split] + missing[split] + performances[split]
        report = dict(metadata_pairs=metadata[split], moved_to_test=known_exclusions['moved_to_test'],
                      known_pitch_mismatches=known_exclusions[split],
                      missing_pairs=missing[split], score_paths=score_counts[split], performances=performances[split],
                      metadata_note_count=metadata_notes[split], normalized_note_count=normalized_notes[split], crop_seed=42,
                      windows={})
        if split == 'train':
            report['moved_from_train'] = known_exclusions['moved_to_test']
            del report['moved_to_test']
        for policy, stats in counts[split].items():
            total = stats['original'] + stats['cropped']
            report['windows'][policy] = dict(stats, total=total, mean_chords=stats['chords'] / total,
                                             mean_notes=stats['notes'] / total)
        result[split] = report
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'window_counts.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
