#!/usr/bin/env python3
"""Build common design1125 chord labels and complete score inputs."""

# ruff: noqa: E402

from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.audit_exp1125_chord_windows import DEST, REFERENCES
from script.prepare_exp1123_measure_official import PT2, load_beat_mapper, score_geometry, sha256, write_json
from src.data_process.dinr_chord_dataset import LIMIT, chord_layout, nonoverlapping_windows

RUN = DEST.parent
DATA = RUN / 'data'
RAW = PT2 / 'asap-dataset-master'


def group_median(values, membership, sizes):
    ordered = values[np.lexsort((values, membership))]
    left = np.cumsum(sizes) - sizes
    return (ordered[left + (sizes - 1) // 2] + ordered[left + sizes // 2]) / 2


def score_chords(geometry, keep):
    _, _, measure, _, group_starts, fields = geometry
    layout = chord_layout(geometry, keep)
    notes, onsets, _, _, membership = layout
    leading = np.r_[True, np.diff(onsets) > 0]
    bar_start = np.r_[0., np.cumsum(measure['score_ms'])[:-1]]
    note_measure = measure['ids']
    note_ms = bar_start[note_measure] + measure['score_ms'][note_measure] * measure['time_ratio']
    source_gap = np.r_[0., np.diff(note_ms[group_starts])]
    ioi = np.where(leading, source_gap[onsets], 0.)
    pitches = np.zeros((len(notes), 128), dtype=np.uint8)
    pitch_list = fields[keep, 1].astype(np.int64)
    np.add.at(pitches, (membership, pitch_list), 1)
    if not np.array_equal(pitches.sum(axis=1), notes):
        raise ValueError('Pitch counts differ from chord membership')
    duration = np.zeros(len(notes))
    duration[membership] = fields[keep, 2]
    velocity = np.bincount(membership, weights=fields[keep, 4]) / notes
    score = np.column_stack((ioi, duration, velocity.round())).astype('float32')
    return layout, leading, dict(pitches=pitches, score=score)


def performance_chords(geometry, ref, manifest):
    pitch, order, _, _, groups, _ = geometry
    path = manifest.parent / 'references' / Path(ref['path']).name
    with np.load(path) as raw:
        np.testing.assert_array_equal(pitch, raw['pitch'])
        segment = raw['segment'][order]
        start, duration, velocity = (raw[key][order].astype(float) for key in ('start', 'duration', 'velocity'))
    keep = segment >= 0
    if keep.sum() != ref['retained_notes']:
        raise ValueError(f'Alignment coverage differs for {ref["performance"]}')
    layout, leading, arrays = score_chords(geometry, keep)
    notes, onsets, _, _, membership = layout
    earliest = np.minimum.reduceat(np.where(keep, start, np.inf), groups)
    left_segment = np.minimum.reduceat(np.where(keep, segment, len(keep)), groups)
    right_segment = np.maximum.reduceat(segment, groups)
    same = (left_segment == right_segment) & (right_segment >= 0)
    adjacent = same[1:] & same[:-1] & (right_segment[1:] == right_segment[:-1])
    raw_ioi = np.zeros(len(groups))
    raw_ioi[1:][adjacent] = earliest[1:][adjacent] - earliest[:-1][adjacent]
    ioi = np.where(leading, raw_ioi[onsets], 0.)
    ioi_valid = (ioi > 0) & (arrays['score'][:, 0] > 0)
    ioi[~ioi_valid] = 0.
    shared_duration = group_median(duration[keep], membership, notes)
    shared_velocity = np.bincount(membership, weights=velocity[keep]) / notes
    targets = np.column_stack((ioi, shared_duration.clip(1), shared_velocity.round())).astype('float32')
    return layout, dict(arrays, targets=targets, ioi_valid=ioi_valid)


def save(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def build_split(split, ledger, expected, get_bars):
    manifest = REFERENCES[split]
    entries = sorted(json.loads(manifest.read_text()), key=lambda ref: (ref['score'], ref['performance']))
    spans, data, offset = [], defaultdict(list), 0
    prior_work = None
    for ref in entries:
        work = ref['score']
        if work != prior_work:
            geometry = score_geometry(RAW, work, get_bars)
            prior_work = work
        layout, arrays = performance_chords(geometry, ref, manifest)
        note_prefix = np.r_[0, np.cumsum(layout.notes)]
        if split == 'train':
            windows = ledger[work, ref['performance']]
            if any(right - left != chords or note_prefix[right] - note_prefix[left] != notes
                   for left, right, chords, notes in windows):
                raise ValueError(f'Window counts differ for {ref["performance"]}')
            local_spans = [(left, right) for left, right, _, _ in windows]
        else:
            local_spans = nonoverlapping_windows(layout.boundaries)
        for left, right in local_spans:
            if not 0 <= left < right <= len(layout.notes) or right - left > LIMIT:
                raise ValueError(f'Invalid window {left, right} for {ref["performance"]}')
            spans.append((offset + left, offset + right))
        offset += len(layout.notes)
        for key, values in arrays.items():
            data[key].append(values)
    expected_windows = (expected['by_window_kind']['total']['windows'] if split == 'train'
                        else expected['nonoverlapping_windows'])
    if (offset, len(spans)) != (expected['chords'], expected_windows):
        raise ValueError(f'{split}: prepared sidecar differs from the window audit')
    payload = {key: torch.from_numpy(np.concatenate(values)) for key, values in data.items()}
    payload['spans'] = torch.tensor(spans, dtype=torch.int32)
    path = DATA / f'{split}.pt'
    save(path, payload)
    print(split, offset, len(spans), flush=True)
    score, targets = payload['score'], payload['targets']
    support = torch.column_stack((payload['ioi_valid'], torch.ones(len(score), dtype=torch.bool)))
    clips = {}
    for name, col in (('ioi', 0), ('duration', 1)):
        use = support[:, col]
        s, d = score[use, col], targets[use, col]
        dev = d.clamp_min(1).log() - s.clamp_min(1).log()
        clips[name] = dict(targets=int(use.sum()), abs5000_above=int((d.round() > 4999).sum()),
                           log_abs_above=int((d > 5000).sum()), log_dev_outside=int(((dev < -3) | (dev > 2)).sum()))
    return dict(path=str(path), sha256=sha256(path), chords=offset, windows=len(spans), clipping=clips)


def build_scores(works, get_bars):
    scores = {}
    for work in works:
        geometry = score_geometry(RAW, work, get_bars)
        pitch, order, _, ticks, group_starts, fields = geometry
        layout, _, arrays = score_chords(geometry, np.ones(len(order), dtype=bool))
        payload = dict(arrays, pitch=pitch[order], membership=layout.membership,
                       ticks=np.repeat(ticks, np.diff(group_starts, append=len(order))),
                       spans=np.asarray(list(nonoverlapping_windows(layout.boundaries)), dtype=np.int32))
        np.testing.assert_array_equal(payload['pitch'], fields[:, 1])
        scores[work] = {key: torch.from_numpy(value) for key, value in payload.items()}
    path = DATA / 'scores.pt'
    save(path, scores)
    return dict(path=str(path), sha256=sha256(path), works=len(scores),
                windows=sum(len(row['spans']) for row in scores.values()))


def main():
    torch.set_num_threads(1)
    DATA.mkdir(parents=True, exist_ok=True)
    audit = json.loads((DEST / 'comparison.json').read_text())['splits']
    ledger = defaultdict(list)
    with (DEST / 'chord_windows.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split'] == 'train':
                ledger[row['score'], row['performance']].append(
                    tuple(int(row[key]) for key in ('start_chord', 'end_chord_exclusive', 'chords', 'notes')))
    get_bars, _ = load_beat_mapper()
    results = {split: build_split(split, ledger, audit[split]['chord'], get_bars) for split in ('train', 'test')}
    works = sorted(json.loads((PT2 / 'score_split.json').read_text())['test'])
    results['scores'] = build_scores(works, get_bars)
    write_json(DATA / 'manifest.json', results)
    print(results, flush=True)


if __name__ == '__main__':
    main()
