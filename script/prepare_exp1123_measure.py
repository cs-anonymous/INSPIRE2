#!/usr/bin/env python3
"""Build overlapping 512-event MEASURE/NOTE sidecars from the frozen exp1122 cohort."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.analyze_exp1123_measure import FROZEN, grid, measure_edges, observe, score_time_ms
from src.data_process.beat_mapping import map_score_beats
from src.data_process.build_dinr_beat_sidecars import load_beat_mapper, score_notes
from src.data_process.dinr_beat_dataset import BeatTokenSpec

OUTPUT = ROOT.parent / 'data/ASAP_processed_dinr_measure_structural_earliest_e512_o50_v1'
SOURCE_AUDIT = ROOT / 'analysis/exp1122_structural_beat/fixed_event_data_audit_checked.json'
LENGTH = 512
STRIDE = 256


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def window_spans(start, end):
    for left in range(start, end, STRIDE):
        right = min(left + LENGTH, end)
        yield left, right
        if right == end:
            break


def score_stream(locked_events, payload, get_bars):
    score_path = ROOT.parent / 'data/ASAP_processed' / payload['meta']['score_source']
    midi = MidiFile(str(score_path))
    notes = score_notes(midi)
    ticks = np.asarray([note.start for note in notes], dtype=np.int64)
    pitches = np.asarray([note.pitch for note in notes], dtype=np.int64)
    np.testing.assert_array_equal(pitches, payload['score']['pitch'])
    np.testing.assert_array_equal(pitches, locked_events[:, 1])
    edges = measure_edges(midi, get_bars)
    mapping = np.asarray(midi.get_tick_to_time_mapping(), dtype=np.float64) * 1000
    measure = grid(edges, score_time_ms(midi, mapping, edges), score_time_ms(midi, mapping, ticks), ticks)
    group_ticks, group_starts = np.unique(ticks, return_index=True)
    num_measures, num_notes = len(measure['counts']), len(notes)
    measure_pos = np.arange(num_measures) + np.r_[0, measure['counts'].cumsum()[:-1]]
    note_pos = np.arange(num_notes) + measure['ids'] + 1
    events = np.empty((num_measures + num_notes, 6), dtype=np.float32)
    events[measure_pos] = [BeatTokenSpec.beat_type, -1, 0, -1, -1, 1]
    events[measure_pos, 2] = measure['score_ms']
    events[note_pos] = locked_events
    events[note_pos, 3] = measure['time_ratio']
    return measure, group_ticks, group_starts, measure_pos, note_pos, events


def performance_stream(source, geometry, locked_targets, locked_valid):
    measure, group_ticks, group_starts, measure_pos, note_pos, events = geometry
    raw = np.asarray(source['label_shared_raw'], dtype=np.float64)
    if raw.shape != (len(note_pos), 3) or not np.isfinite(raw).all():
        raise ValueError('Invalid frozen performance labels')
    np.testing.assert_allclose(locked_targets[:, 0], np.maximum(raw[:, 1], 1), rtol=1e-6, atol=1e-3)
    np.testing.assert_array_equal(locked_targets[:, 2], np.clip(np.rint(raw[:, 2]), 0, 127))
    onsets = np.cumsum(raw[:, 0])
    anchors, support = map_score_beats(measure['edges'], group_ticks, np.minimum.reduceat(onsets, group_starts))
    observed = observe(measure, anchors, support, onsets)
    valid_measure = observed['valid']
    valid_onset = valid_measure[measure['ids']]
    targets = np.empty((len(events), 3), dtype=np.float32)
    valid = np.zeros_like(targets, dtype=bool)
    targets[measure_pos] = [-1, -1, -1]
    targets[measure_pos, 0] = np.where(valid_measure, np.maximum(observed['duration'], 1), measure['score_ms'])
    valid[measure_pos, 0] = valid_measure
    targets[note_pos] = locked_targets
    targets[note_pos, 1] = np.where(valid_onset, observed['ratio'] - measure['time_ratio'], 0)
    valid[note_pos] = locked_valid
    valid[note_pos, 1] = valid_onset
    if not np.isfinite(targets).all():
        raise ValueError('Non-finite MEASURE/NOTE targets')
    return targets, valid


def prepare_split(split, output, expected, get_bars):
    source = FROZEN / f'dinr_beat_{split}_e512_v1.pt'
    source_hash = sha256(source)
    if source_hash != expected['sha256']:
        raise ValueError(f'{split}: frozen exp1122 sidecar hash differs')
    frozen = torch.load(source, map_location='cpu', weights_only=False)
    by_work = defaultdict(list)
    for row in frozen['performances']:
        by_work[row['work']].append(row)
    samples, performances = [], []
    event_rows, target_rows, mask_rows = [], [], []
    offset = 0
    coverage = defaultdict(int)
    for index, (work, rows) in enumerate(by_work.items(), start=1):
        [work_path] = {row['source_sidecar'] for row in rows}
        payload = torch.load(work_path, map_location='cpu', weights_only=False)
        sources = {row['performance_source']: row for row in payload['performances']}
        if set(sources) != {row['source'] for row in rows}:
            raise ValueError(f'{split}: performance cohort differs for {work}')
        begin, end = rows[0]['span']
        initial = frozen['events'][begin:end].numpy()
        locked_events = initial[initial[:, 0] == BeatTokenSpec.note_type]
        geometry = score_stream(locked_events, payload, get_bars)
        _, _, _, measure_pos, note_pos, events = geometry
        measure_mask = events[:, 0] == BeatTokenSpec.beat_type
        for row in rows:
            begin, end = row['span']
            original = frozen['events'][begin:end].numpy()
            note_mask = original[:, 0] == BeatTokenSpec.note_type
            np.testing.assert_array_equal(original[note_mask], locked_events)
            targets, valid = performance_stream(sources[row['source']], geometry,
                                                frozen['targets'][begin:end].numpy()[note_mask],
                                                frozen['target_valid'][begin:end].numpy()[note_mask])
            stop = offset + len(events)
            performances.append({**row, 'span': [offset, stop]})
            spans = list(window_spans(offset, stop))
            if spans[0][0] != offset or spans[-1][1] != stop:
                raise ValueError(f'{work}: window coverage misses a boundary')
            for previous, current in zip(spans, spans[1:]):
                if current[0] - previous[0] != STRIDE or previous[1] - current[0] != STRIDE:
                    raise ValueError(f'{work}: invalid window stride or overlap')
            for a, b in spans:
                if not offset <= a < b <= stop or b - a > LENGTH:
                    raise ValueError(f'{work}: window crosses a performance boundary')
                measures = int(measure_mask[a - offset:b - offset].sum())
                samples.append(dict(span=[a, b], work=work, performance=row['performance'],
                                    source=row['source'], notes=b-a-measures, measures=measures))
                coverage['padding_events'] += LENGTH - (b - a)
            event_rows.append(torch.from_numpy(events))
            target_rows.append(torch.from_numpy(targets))
            mask_rows.append(torch.from_numpy(valid))
            coverage['notes'] += len(note_pos)
            coverage['measures'] += len(measure_pos)
            coverage['valid_measures'] += int(valid[measure_pos, 0].sum())
            coverage['valid_onsets'] += int(valid[note_pos, 1].sum())
            upper_ms = np.exp(BeatTokenSpec.duration_abs_range[1])
            coverage['score_measures_above_input_range'] += int((events[measure_pos, 2] > upper_ms).sum())
            coverage['performance_measures_above_input_range'] += int(
                ((targets[measure_pos, 0] > upper_ms) & valid[measure_pos, 0]).sum())
            offset = stop
        if index % 50 == 0:
            print(f'{split}: {index}/{len(by_work)} works', flush=True)
    for key, actual in (('works', len(by_work)), ('performances', len(performances)), ('notes', coverage['notes'])):
        if actual != expected[key]:
            raise ValueError(f'{split}: {key} count differs from exp1122')
    metadata = {key: frozen[key] for key in ('sidecar_root', 'valid_work_manifest', 'event_fields',
                'target_fields', 'performance_time_normalization')}
    data = dict(**metadata, format='dinr_beat_events_v2', event_grid='measure',
                event_type_names={0: 'MEASURE', 1: 'NOTE'}, split=split,
                window_events=LENGTH, stride_events=STRIDE, overlap_ratio=0.5,
                window_policy='fixed_event_count_overlap',
                anchor_protocol='MEASURE edges mapped by earliest paired onset per score onset group, linear interpolation, no extrapolation',
                onset_target_protocol='performance measure time fraction minus score MIDI tempo-map time fraction',
                source_sidecar=str(source), source_sha256=source_hash,
                events=torch.cat(event_rows), targets=torch.cat(target_rows), target_valid=torch.cat(mask_rows),
                performances=performances, samples=samples)
    path = output / f'dinr_measure_{split}_e512_o50_v1.pt'
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(path)
    report = dict(path=str(path), sha256=sha256(path), source_sidecar=str(source), source_sha256=source_hash,
                  works=len(by_work), performances=len(performances), events=offset, windows=len(samples),
                  min_window_events=min(b-a for a,b in (row['span'] for row in samples)),
                  max_window_events=max(b-a for a,b in (row['span'] for row in samples)), **coverage)
    print(split, json.dumps(report), flush=True)
    return set(by_work), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    expected = json.loads(SOURCE_AUDIT.read_text())
    get_bars, _ = load_beat_mapper()
    works = set()
    audit = dict(status='passed', event_grid='measure', window_events=LENGTH,
                 stride_events=STRIDE, overlap_ratio=0.5, splits={})
    for split in ('train', 'valid', 'test'):
        split_works, report = prepare_split(split, output, expected['splits'][split], get_bars)
        if works & split_works:
            raise ValueError(f'{split}: score works overlap with another split')
        works.update(split_works)
        audit['splits'][split] = report
    temp = output / 'audit.tmp'
    temp.write_text(json.dumps(audit, indent=2) + '\n')
    temp.replace(output / 'audit.json')


if __name__ == '__main__':
    main()
