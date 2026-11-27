#!/usr/bin/env python3
"""Build shared, work-sharded PianoCoRe Chord targets and ASAP SFT splits."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.audit_pianocore_asap_overlap import PIANOCORE, OUTPUT, event_array, same_recording
from script.prepare_exp1123_measure_official import PT2, load_beat_mapper, score_geometry
from script.prepare_exp1125_chord import DATA, DEST, REFERENCES, save, score_chords, sha256, write_json
from src.data_process.dinr_chord_dataset import nonoverlapping_windows

RUN = ROOT / 'results/exp1126'
REFINED = PIANOCORE / 'refined'


def notes(midi):
    return sorted((note for track in midi.instruments if not track.is_drum for note in track.notes),
                  key=lambda note: (note.start, note.pitch, note.end, note.velocity))


def init_worker(reference_events):
    global mapper, references_by_count
    torch.set_num_threads(1)
    mapper, _ = load_beat_mapper()
    references_by_count = reference_events


def aligned_targets(row, pitch, order, starts, layout, leading, score):
    midi = MidiFile(str(REFINED / row['refined_performance_midi_path']))
    performance = notes(midi)
    with np.load(REFINED / row['refined_alignment_path']) as alignment:
        index = alignment['perf_idx'].astype(np.int64)
        observed = ~alignment['interpolated'].astype(bool)
    if index.shape != pitch.shape or observed.shape != pitch.shape or index.min() < 0 or index.max() >= len(performance):
        raise ValueError('invalid note alignment')
    performance = [performance[i] for i in index[order]]
    if not np.array_equal(pitch[order], [note.pitch for note in performance]):
        raise ValueError('pitch mismatch')
    observed = observed[order]
    ticks = np.asarray([(note.start, note.end, note.velocity) for note in performance])
    clock = np.asarray(midi.get_tick_to_time_mapping()) * 1000
    onset = clock[ticks[:, 0]]
    duration = clock[ticks[:, 1]] - onset
    membership = layout.membership
    earliest = np.minimum.reduceat(np.where(observed, onset, np.inf), starts)
    gap = np.zeros_like(earliest)
    adjacent = np.isfinite(earliest[1:]) & np.isfinite(earliest[:-1])
    gap[1:][adjacent] = earliest[1:][adjacent] - earliest[:-1][adjacent]
    ioi = np.where(leading, gap[layout.onsets], 0.)
    ioi_valid = (ioi > 0) & (score[:, 0] > 0)
    real_counts = np.bincount(membership[observed], minlength=len(layout.notes))
    velocity = np.bincount(membership[observed], weights=ticks[observed, 2], minlength=len(layout.notes))
    velocity /= np.maximum(real_counts, 1)
    sorted_notes = np.lexsort((duration[observed], membership[observed]))
    sorted_duration = duration[observed][sorted_notes]
    has_notes = real_counts > 0
    left = np.cumsum(real_counts)[has_notes] - real_counts[has_notes]
    count = real_counts[has_notes]
    chord_duration = np.ones(len(real_counts))
    chord_duration[has_notes] = np.maximum(1, (sorted_duration[left + (count - 1)//2] +
                                              sorted_duration[left + count//2]) / 2)
    targets = np.column_stack((np.where(ioi_valid, ioi, 0.), chord_duration, velocity.round()))
    return targets.astype(np.float32), np.column_stack((ioi_valid, has_notes, has_notes))


def build_work(job):
    directory, number, work, rows, holdout = job
    path = directory / f'{number:04d}.pt'
    report_path = path.with_suffix('.json')
    if path.is_file() and report_path.is_file():
        return json.loads(report_path.read_text())
    geometry = score_geometry(REFINED, work, mapper)
    pitch, order, _, _, starts, _ = geometry
    original_pitch = np.asarray([note.pitch for note in notes(MidiFile(str(REFINED / work)))])
    if not np.array_equal(original_pitch, pitch):
        raise ValueError(f'{work}: refined alignment and geometry have different note identities')
    layout, leading, arrays = score_chords(geometry, np.ones(len(pitch), dtype=bool))
    spans = list(nonoverlapping_windows(layout.boundaries))
    targets, labels_valid, performances, splits, failures, moved = [], [], [], [], [], []
    for row in rows:
        split = row['effective_split']
        performance = row['refined_performance_midi_path']
        try:
            candidates = references_by_count.get(int(row['performance_note_count'] or 0), ())
            if split == 'train' and candidates:
                events = event_array(PIANOCORE / 'raw' / row['performance_midi_path'])
                if any(same_recording(reference, events) for reference in candidates):
                    split = 'test'
                    moved.append(row['id'])
            target, valid = aligned_targets(row, pitch, order, starts, layout, leading, arrays['score'])
        except (OSError, ValueError, KeyError, IndexError) as error:
            failures.append(dict(id=row['id'], performance=performance, error=str(error)))
            continue
        targets.append(target)
        labels_valid.append(valid)
        performances.append(row['id'])
        splits.append('valid' if split == 'train' and holdout else split)
    if targets:
        data = {key: torch.from_numpy(value) for key, value in arrays.items()}
        data.update(targets=torch.from_numpy(np.stack(targets)), valid=torch.from_numpy(np.stack(labels_valid)),
                    spans=torch.tensor(spans, dtype=torch.int32))
        save(path, data)
    report = dict(path=str(path), score=work, notes=len(pitch), chords=len(layout.notes), windows=len(spans),
                  performances=performances, splits=splits, failures=failures, additional_test_matches=moved)
    write_json(report_path, report)
    return report


def prepare_pianocore(directory, workers, limit):
    directory.mkdir(parents=True, exist_ok=True)
    moved = {row['pianocore_id'] for row in json.loads(OUTPUT.read_text())['matches']}
    known = ROOT.parent / 'results/pianocoreA_slot_sum_pretrain4_asap_ft8_20261028/data_preflight.json'
    excluded = {row['source'] for row in json.loads(known.read_text())['excluded_performances']}
    groups = defaultdict(list)
    with (PIANOCORE / 'metadata.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['tier_a'] != 'True' or row['refined_performance_midi_path'] in excluded:
                continue
            row['effective_split'] = 'test' if row['id'] in moved else row['split']
            groups[row['refined_score_midi_path']].append(row)
    train = sorted(work for work, rows in groups.items() if any(row['effective_split'] == 'train' for row in rows))
    rng = random.Random(42)
    valid = set(rng.sample(train, max(1, round(len(train) * .02))))
    works = sorted(groups)
    if limit:
        works = sorted(rng.sample(sorted(valid), 3) + rng.sample(sorted(set(train) - valid), limit - 3))
    ref_events = defaultdict(list)
    for row in json.loads(REFERENCES['test'].read_text()):
        events = event_array(PT2 / 'asap-dataset-master' / row['performance'])
        ref_events[len(events)].append(events)
    jobs = [(directory, number, work, groups[work], work in valid) for number, work in enumerate(works)]
    reports = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(ref_events,)) as pool:
        for report in pool.map(build_work, jobs):
            reports.append(report)
            if len(reports) % 50 == 0:
                print(f'PianoCoRe scores {len(reports)}/{len(works)}', flush=True)
    counts = {}
    for split in ('train', 'valid', 'test'):
        counts[split] = {
            field: sum(row['splits'].count(split) * row[field] for row in reports)
            for field in ('windows', 'chords', 'notes')
        }
        counts[split]['performances'] = sum(row['splits'].count(split) for row in reports)
    write_json(directory / 'manifest.json', dict(counts=counts, works=reports, elapsed=time.monotonic() - started))
    print(counts, flush=True)


def canonical_work(path):
    score_dir = str(Path(path).parent)
    return re.sub(r'_(?:no_(?:first_|\d+_)?)?repeat$', '', score_dir, flags=re.IGNORECASE).casefold()


def prepare_asap(directory):
    directory.mkdir(parents=True, exist_ok=True)
    refs = sorted(json.loads(REFERENCES['train'].read_text()), key=lambda row: (row['score'], row['performance']))
    test_works = {canonical_work(row['score']) for row in json.loads(REFERENCES['test'].read_text())}
    works = sorted({canonical_work(row['score']) for row in refs} - test_works)
    valid = set(random.Random(42).sample(works, max(1, round(len(works) * .1))))
    source = torch.load(DATA / 'train.pt', map_location='cpu', weights_only=True, mmap=True)
    ledger = defaultdict(list)
    with (DEST / 'chord_windows.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split'] == 'train':
                key = row['score'], row['performance']
                ledger[key].append((int(row['start_chord']), int(row['end_chord_exclusive'])))
    splits = {'train': [], 'valid': []}
    offset = 0
    indices = []
    for row in refs:
        windows = ledger[row['score'], row['performance']]
        actual = source['spans'][len(indices):len(indices) + len(windows)]
        if not torch.equal(actual, torch.tensor(windows, dtype=actual.dtype) + offset):
            raise ValueError('ASAP sidecar does not match the window ledger')
        canonical = canonical_work(row['score'])
        if canonical not in test_works:
            splits['valid' if canonical in valid else 'train'].extend(range(len(indices), len(indices) + len(windows)))
        indices.extend(windows)
        offset += max(right for _, right in windows)
    if len(indices) != len(source['spans']) or offset != len(source['score']):
        raise ValueError('ASAP window ledger coverage differs')
    write_json(directory / 'asap_split.json', dict(splits, valid_works=sorted(valid),
                                                    train_sha256=sha256(DATA / 'train.pt'),
                                                    excluded_scores=sorted({row['score'] for row in refs
                                                                            if canonical_work(row['score']) in test_works})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare_asap(RUN)
    prepare_pianocore(RUN / ('smoke_data' if args.smoke else 'data'), args.workers, 9 if args.smoke else 0)
