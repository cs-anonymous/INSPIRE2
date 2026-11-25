#!/usr/bin/env python3
"""Compare the design1125 CHORD windows with the audited PT2 windows."""

# ruff: noqa: E402

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np

INSPIRE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSPIRE))

from script.prepare_exp1123_measure_official import (LIMIT, OUTPUT, PT2, PT2_EVAL, SEED, SPLITS, load_beat_mapper,
                                                    load_pt2_pairs, score_geometry, windows, write_json)
from src.data_process.dinr_chord_dataset import chord_layout, nonoverlapping_windows

DEST = INSPIRE / 'results/exp1125_chord/window_audit'
REFERENCES = {
    'train': INSPIRE / 'results/exp1123_measure_official_split_data_20261124/align_train/reference_manifest.json',
    'test': PT2_EVAL / 'reference_manifest.json',
}


def audit_split(split, pairs, get_bars, writer):
    manifest = REFERENCES[split]
    rows = json.loads(manifest.read_text())
    refs = {(row['score'], row['performance']): row for row in rows}
    if len(refs) != len(rows) or refs.keys() != pairs.keys():
        raise ValueError(f'{split}: reference identities differ from official PT2 pairs')
    rng = random.Random(SEED)
    totals = Counter()
    counts = defaultdict(Counter)
    chord_lengths = defaultdict(list)
    prior_work = None
    for work, performance in sorted(pairs):
        if work != prior_work:
            geometry = score_geometry(PT2 / 'asap-dataset-master', work, get_bars)
            prior_work = work
        pitch, order, *_ = geometry
        ref = refs[work, performance]
        path = manifest.parent / 'references' / Path(ref['path']).name
        with np.load(path) as source:
            np.testing.assert_array_equal(source['pitch'], pitch)
            segment = source['segment'][order]
        keep = segment >= 0
        if np.bincount(segment[keep]).tolist() != pairs[work, performance] or keep.sum() != ref['retained_notes']:
            raise ValueError(f'{performance}: retained note counts differ from PT2')
        notes, onsets, boundaries, oversized, _ = chord_layout(geometry, keep)
        note_prefix = np.r_[0, np.cumsum(notes)]
        if note_prefix[-1] != keep.sum():
            raise ValueError(f'{performance}: chord grouping changed note coverage')
        onset_counts = np.unique(onsets, return_counts=True)[1]
        totals.update(notes=int(note_prefix[-1]), chords=len(notes), onset_positions=len(onset_counts),
                      split_onset_positions=int((onset_counts > 1).sum()), oversized_measures=oversized,
                      nonoverlapping_windows=len(list(nonoverlapping_windows(boundaries))))
        last_end = 0
        used_starts = set()
        for parent, (left, right, middle, crop) in enumerate(windows(boundaries, rng), start=1):
            if left > last_end or right < last_end:
                raise ValueError(f'{performance}: uncovered chords or decreasing window end')
            last_end = right
            for start, kind in ((left, 'original'), (crop, 'crop')):
                if start is None:
                    continue
                length = right - start
                if (start in used_starts or not 0 < length <= LIMIT or
                        (kind == 'crop' and not (left < start < middle and 2 * length > right - left))):
                    raise ValueError(f'{performance}: invalid {kind} window')
                used_starts.add(start)
                note_count = int(note_prefix[right] - note_prefix[start])
                counts[kind].update(windows=1, chord_appearances=length, note_appearances=note_count)
                chord_lengths[kind].append(length)
                writer.writerow((split, work, performance, kind, parent, start, right, length, note_count))
        if last_end != len(notes):
            raise ValueError(f'{performance}: final chord is uncovered')

    counts['total'] = counts['original'] + counts['crop']
    chord_lengths['total'] = chord_lengths['original'] + chord_lengths['crop']
    windows_by_kind = {}
    for kind, counter in counts.items():
        windows_by_kind[kind] = dict(counter, mean_chords=counter['chord_appearances'] / counter['windows'],
                                     mean_notes=counter['note_appearances'] / counter['windows'],
                                     chords_per_window={name: float(value) for name, value in zip(
                                         ('min', 'p5', 'median', 'p95', 'max'),
                                         np.percentile(chord_lengths[kind], (0, 5, 50, 95, 100)), strict=True)})
    return dict(totals, score_paths=len({work for work, _ in pairs}), performances=len(pairs),
                retained_hmm_segments=sum(map(len, pairs.values())), by_window_kind=windows_by_kind)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    pt2_audit = json.loads((PT2 / 'dataset_audit.json').read_text())
    pairs, slides = load_pt2_pairs(pt2_audit)
    prior = json.loads((OUTPUT / 'comparison.json').read_text())
    get_bars, _ = load_beat_mapper()
    results = {}
    comparison = []
    with (DEST / 'chord_windows.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('split', 'score', 'performance', 'kind', 'original_window', 'start_chord',
                         'end_chord_exclusive', 'chords', 'notes'))
        for split in SPLITS:
            chord = audit_split(split, pairs[split], get_bars, writer)
            pt2 = prior[split]['pt2']
            if (pt2['original_windows'], pt2['original_note_appearances']) != (
                    slides[split]['windows'], slides[split]['note_appearances']):
                raise ValueError(f'{split}: audited PT2 original windows differ from raw rebuild')
            if tuple(chord[k] for k in ('score_paths', 'performances', 'retained_hmm_segments', 'notes')) != tuple(
                    pt2[k] for k in ('score_paths', 'performances', 'segments', 'aligned_notes')):
                raise ValueError(f'{split}: chord audit does not cover the same data as PT2')
            results[split] = dict(chord=chord, pt2=pt2)
            for kind, stats in chord['by_window_kind'].items():
                size = pt2[f'{kind}_windows']
                note_count = pt2[f'{kind}_note_appearances']
                comparison.append((split, 'PT2', kind, size, '', note_count, '', note_count / size))
                comparison.append((split, 'Chord', kind, stats['windows'], stats['chord_appearances'],
                                   stats['note_appearances'], stats['mean_chords'], stats['mean_notes']))
            print(split, chord, flush=True)
    with (DEST / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('split', 'model', 'kind', 'windows', 'chord_appearances', 'note_appearances',
                         'mean_chords_per_window', 'mean_notes_per_window'))
        writer.writerows(comparison)
    write_json(DEST / 'comparison.json', dict(window_cap=LIMIT, crop_seed=SEED, splits=results))


if __name__ == '__main__':
    main()
