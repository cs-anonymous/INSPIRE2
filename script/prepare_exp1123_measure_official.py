#!/usr/bin/env python3
"""Build measure-boundary windows from the same retained ASAP pairs as PT2."""

# ruff: noqa: E402

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from miditoolkit import MidiFile

INSPIRE = Path(__file__).resolve().parents[1]
PROJECT = INSPIRE.parent
PT2 = PROJECT / 'results/pt_rendering_public_recipe_20261124'
PT2_EVAL = PT2 / 'official_split_measure'
RUN = PROJECT / 'results/exp1123_measure_official_split_data_20261124'
OUTPUT = PROJECT / 'data/ASAP_official_dinr_measure_structural_e512_snap50_crop_v1'
sys.path[:0] = [str(INSPIRE), str(PROJECT), str(PT2_EVAL)]

from score_official_split import intervals_inside_segments, representation
from script.analyze_exp1123_measure import grid, measure_edges, observe, score_time_ms
from script.prepare_exp1123_measure import sha256
from src.data_process.beat_mapping import map_score_beats
from src.data_process.build_dinr_beat_sidecars import load_beat_mapper
from src.data_process.dinr_beat_dataset import BeatTokenSpec, DINRBeatCollator, DINRBeatDataset

LIMIT = 512
SEED = 42
SPLITS = ('train', 'test')
EVENT_FIELDS = ['type', 'pitch', 'score_duration_ms', 'score_onset_ratio', 'score_velocity', 'measure_start']
TARGET_FIELDS = ['performance_duration_ms', 'performance_onset_dev', 'performance_velocity']


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_pt2_pairs(audit):
    source = PT2 / 'asap_aligned_sft.jsonl'
    if sha256(source) != audit['sha256']:
        raise ValueError('Source does not match the audited PT2 SFT dataset')
    pairs = {kind: defaultdict(list) for kind in SPLITS}
    slide = {kind: Counter(windows=0, note_appearances=0) for kind in SPLITS}
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            split = row['split']
            segments = pairs[split][row['score_source'], row['performance_source']]
            if len(row['x']) % 8 or len(row['x']) != len(row['label']) or row['cut'] != len(segments):
                raise ValueError('Bad token count or cut order in PT2 data')
            notes = len(row['x']) // 8
            segments.append(notes)
            # The pinned PT2 recipe windows each segment in 512-note steps of
            # 256, and stops as soon as a window reaches the segment end.
            for start in range(0, notes, 256):
                end = min(start + LIMIT, notes)
                slide[split].update(windows=1, note_appearances=end-start)
                if end == notes:
                    break
    for split, group in pairs.items():
        expected = audit['splits'][split]
        observed = (len(group), len({score for score, _ in group}), sum(map(len, group.values())),
                    sum(sum(notes) for notes in group.values()))
        if observed != tuple(expected[field] for field in ('performances', 'score_paths', 'segments', 'aligned_notes')):
            raise ValueError(f'Pair coverage differs from audited PT2 source for {split}')
    if {score for score, _ in pairs['train']} & {score for score, _ in pairs['test']}:
        raise ValueError('Training and testing score paths overlap')
    return pairs, slide


def cached_note_count(path, expected_samples):
    from pyarrow import ipc, memory_map
    from pyarrow.compute import list_value_length, sum as arrow_sum

    rows, tokens = 0, 0
    with memory_map(str(path), 'r') as source:
        stream = ipc.open_stream(source)
        if set(stream.schema.names) != {'input_ids', 'labels'}:
            raise ValueError(f'Expected PT2 window cache: {path}')
        i = stream.schema.get_field_index('input_ids')
        for batch in stream:
            rows += batch.num_rows
            tokens += arrow_sum(list_value_length(batch.column(i))).as_py()
    if rows != expected_samples or tokens % 8:
        raise ValueError(f'Wrong PT2 window count or token width in {path}')
    return tokens // 8


def score_geometry(raw_root, work, get_bars):
    score_path = raw_root / work
    midi = MidiFile(str(score_path))
    mapping, pitch, ticks, order, notes = representation(midi)

    def score_bars(*, midi):
        # Include a meter change on the final MIDI event. The generic mapper
        # excludes that event, which can shorten the preceding bar in raw ASAP.
        return get_bars(time_sigs=midi.time_signature_changes, ticks_per_beat=midi.ticks_per_beat,
                        max_tick=midi.max_tick)

    edges = measure_edges(midi, score_bars)
    measures = grid(edges, score_time_ms(midi, mapping, edges), score_time_ms(midi, mapping, ticks), ticks)
    group_ticks, group_starts = np.unique(ticks, return_index=True)
    fields = np.column_stack((np.ones(len(ticks)), pitch[order],
                              np.asarray([n.end - n.start for n in notes])[order],
                              measures['time_ratio'], np.asarray([n.velocity for n in notes])[order],
                              np.full(len(ticks), -1)))
    return pitch, order, measures, group_ticks, group_starts, fields


def performance_events(reference, geometry, expected_segments):
    pitch, order, measures, group_ticks, group_starts, score_fields = geometry
    with np.load(reference['path']) as source:
        if not np.array_equal(source['pitch'], pitch):
            raise ValueError(f'Score identities differ for {reference["performance"]}')
        raw_segments = source['segment']
        keep = raw_segments >= 0
        if np.bincount(raw_segments[keep]).tolist() != expected_segments:
            raise ValueError(f'Retained notes differ from PT2 for {reference["performance"]}')
        segments = raw_segments[order]
        onsets = source['start'][order].astype(np.float64)
        duration = source['duration'][order]
        velocity = source['velocity'][order]
    kept = segments >= 0
    bounds = measures['edges']
    within = intervals_inside_segments(bounds, group_ticks, group_starts, segments)
    anchors, support = map_score_beats(bounds, group_ticks, np.minimum.reduceat(onsets, group_starts))
    observed = observe(measures, anchors, support, onsets)
    valid_measure = within & observed['valid']
    ids = measures['ids']
    counts = np.bincount(ids[kept], minlength=len(bounds)-1)
    # Keep partial measures with retained notes and silent measures with
    # supported boundaries. A cut-out span cannot contribute timing labels.
    keep_measure = (counts > 0) | within
    merged_ids = np.cumsum(keep_measure)[ids[kept]] - 1
    note_count, measure_count = int(kept.sum()), int(keep_measure.sum())
    measure_pos = np.arange(measure_count) + np.r_[0, counts[keep_measure].cumsum()[:-1]]
    note_pos = np.arange(note_count) + merged_ids + 1
    size = note_count + measure_count
    events = np.empty((size, len(EVENT_FIELDS)), dtype=np.float32)
    events[measure_pos] = [BeatTokenSpec.beat_type, -1, 0, -1, -1, 1]
    events[measure_pos, 2] = measures['score_ms'][keep_measure]
    events[note_pos] = score_fields[kept]
    valid_onset = valid_measure[ids[kept]]
    targets = np.full((size, len(TARGET_FIELDS)), -1, dtype=np.float32)
    valid = np.zeros_like(targets, dtype=bool)
    duration_fallback = np.where(valid_measure, observed['duration'], measures['score_ms'])
    targets[measure_pos, 0] = np.maximum(duration_fallback[keep_measure], 1)
    valid[measure_pos, 0] = valid_measure[keep_measure]
    targets[note_pos, 0] = np.maximum(duration[kept], 1)
    dev = observed['ratio'][kept] - measures['time_ratio'][kept]
    targets[note_pos, 1] = np.where(valid_onset, dev, 0)
    targets[note_pos, 2] = velocity[kept]
    valid[note_pos] = np.column_stack((np.ones(note_count, bool), valid_onset, np.ones(note_count, bool)))
    if (not np.isfinite(targets).all() or not np.isfinite(events).all() or
            len(np.union1d(measure_pos, note_pos)) != size or measure_pos[0] != 0):
        raise ValueError(f'Malformed targets for {reference["performance"]}')
    return events, targets, valid, np.append(measure_pos, size)


def windows(boundaries, rng):
    """Yield original bounds, next start, and one strictly internal crop start."""
    start_index = 0
    final_index = len(boundaries) - 1
    while start_index < final_index:
        start = int(boundaries[start_index])
        end_index = int(np.searchsorted(boundaries, start + LIMIT, side='right') - 1)
        if end_index <= start_index:
            raise ValueError('Single retained measure exceeds the 512-event cap')
        end = int(boundaries[end_index])
        inner = boundaries[start_index+1:end_index]
        middle_index = end_index
        crop = None
        if len(inner):
            distances = np.abs(2*(inner-start)-(end-start))
            middle_index = start_index + 1 + int(np.argmin(distances))
            if middle_index > start_index + 1:
                crop = int(boundaries[rng.randrange(start_index+1, middle_index)])
                if 2*(end-crop) <= end-start:
                    raise ValueError('A sampled crop must retain strictly more than half of its original window')
        yield start, end, int(boundaries[middle_index]), crop
        if end_index == final_index:
            return
        start_index = middle_index


def build_split(split, pairs, manifest, raw_root, output, get_bars):
    entries = json.loads(manifest.read_text())
    references = {(row['score'], row['performance']): row for row in entries}
    if len(references) != len(entries) or set(references) != set(pairs):
        raise ValueError(f'{split}: recovered alignments do not match the official retained pairs')
    rng = random.Random(SEED)
    samples, performances, overlaps, lengths = [], [], [], []
    all_events, all_targets, all_valid = [], [], []
    coverage = Counter()
    base = Counter()
    crops = Counter()
    offset = 0
    work_prior = None
    for work, performance in sorted(pairs):
        if work != work_prior:
            geometry = score_geometry(raw_root, work, get_bars)
            work_prior = work
        ref = references[work, performance]
        events, targets, valid, boundaries = performance_events(ref, geometry, pairs[work, performance])
        is_note = events[:, 0] == BeatTokenSpec.note_type
        computed = Counter(notes=int(is_note.sum()), measures=int((~is_note).sum()),
                           valid_onsets=int(valid[:, 1].sum()), valid_measures=int((valid[:, 0] & ~is_note).sum()))
        coverage.update(computed)
        if computed['notes'] != ref['retained_notes']:
            raise ValueError(f'{performance}: source and sidecar note counts differ')
        meta = dict(work=work, score_source=work, performance=performance, source=performance)
        stop = offset + len(events)
        performances.append(dict(**meta, span=[offset, stop], alignment_file=ref['path'],
                                 original_alignment_segments=ref['segments']))
        events_count = np.r_[0, np.cumsum(is_note)]
        supervised = np.r_[0, np.cumsum(valid.sum(axis=1))]
        starts = set()
        last_end = 0
        for left, right, middle, crop in windows(boundaries, rng):
            if left > last_end or right < last_end:
                raise ValueError(f'{performance}: gapped coverage or backward window end')
            last_end = right
            if right < len(events):
                overlaps.append((right-middle)/(right-left))
            parent = len(samples)
            for start, kind, tally in ((left, 'original', base), (crop, 'crop', crops)):
                if start is None:
                    continue
                if start in starts or (kind == 'crop' and not left < start < middle):
                    raise ValueError(f'{performance}: repeated or out-of-bounds crop start')
                starts.add(start)
                length = right - start
                notes = int(events_count[right] - events_count[start])
                measure_count = length - notes
                samples.append(dict(**meta, span=[offset+start, offset+right], notes=notes,
                                    measures=measure_count, kind=kind, original_sample=parent))
                tally.update(windows=1, events=length, note_appearances=notes, measure_appearances=measure_count,
                             target_fields=int(supervised[right]-supervised[start]))
                lengths.append(length)
        if last_end != len(events):
            raise ValueError(f'{performance}: original windows do not cover the final measure')
        offset = stop
        all_events.append(events)
        all_targets.append(targets)
        all_valid.append(valid)
        if len(performances) % 100 == 0 or len(performances) == len(pairs):
            print(f'{split}: built {len(performances)}/{len(pairs)} performance streams', flush=True)
    path = output / f'dinr_measure_{split}_e512_snap50_crop_v1.pt'
    payload = dict(format='dinr_beat_events_v2', event_grid='measure', event_type_names={0: 'MEASURE', 1: 'NOTE'},
                   split=split, sidecar_root=str(raw_root), event_fields=EVENT_FIELDS, target_fields=TARGET_FIELDS,
                   window_events=LIMIT, overlap_ratio=.5, window_policy='nearest_measure_boundary',
                   random_cut_seed=SEED, performance_time_normalization='native HMM milliseconds',
                   source_sidecar=str(PT2 / 'asap_aligned_sft.jsonl'),
                   anchor_protocol='Earliest performance onset per score onset group; interpolate between groups. '
                                   'Both boundaries must be supported by the same retained PT2 segment.',
                   onset_target_protocol='performance measure time fraction minus score MIDI tempo-map time fraction',
                   events=torch.from_numpy(np.concatenate(all_events)), targets=torch.from_numpy(np.concatenate(all_targets)),
                   target_valid=torch.from_numpy(np.concatenate(all_valid)), performances=performances, samples=samples)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)
    # Check the artifact via its regular dataset and dynamic-padding entry point.
    del payload
    validate_dataset(path, samples, lengths)
    return dict(path=str(path), sha256=sha256(path), score_paths=len({work for work, _ in pairs}),
                performances=len(pairs), segments=sum(len(v) for v in pairs.values()), events=offset,
                original=dict(base), crop=dict(crops), total=dict(base+crops),
                missing_crop=base['windows']-crops['windows'], window_min=min(lengths), window_max=max(lengths),
                overlap_quantiles={str(q): float(v) for q, v in zip((0, 5, 50, 95, 100),
                                                                   np.percentile(overlaps, [0, 5, 50, 95, 100]))},
                **coverage)


def validate_dataset(path, samples, lengths):
    dataset = DINRBeatDataset(path, duration_target='log_dev', onset_target='dev')
    if len(dataset) != len(samples) or max(lengths) > LIMIT:
        raise ValueError(f'Invalid dataset shape for {path}')
    ids = [int(np.argmin(lengths)), int(np.argmax(lengths))]
    for index in ids:
        single = DINRBeatCollator()([dataset[index]])
        padded = (lengths[index]+7)//8*8
        labels = single['labels'][0]
        if (single['attention_mask'].shape != (1, padded) or
                int(single['attention_mask'].sum()) != lengths[index] or
                (labels[lengths[index]:] != -100).any() or
                int(single['event_types'][0, 0]) != BeatTokenSpec.beat_type):
            raise ValueError(f'Dynamic-batch checks failed for {path}')


def write_comparison(output, measures, source, slides, caches):
    comparison = {}
    for split in SPLITS:
        base = slides[split]
        cache = caches[split]
        original_notes = base['note_appearances']
        total_notes = cached_note_count(cache, 2*base['windows']) if cache else None
        pt2 = dict(**source['splits'][split], original_windows=base['windows'], crop_windows=base['windows'],
                   total_windows=2*base['windows'], original_note_appearances=original_notes,
                   crop_note_appearances=None if total_notes is None else total_notes-original_notes,
                   total_note_appearances=total_notes,
                   window_cache=None if cache is None else dict(path=str(cache), sha256=sha256(cache)))
        a2 = measures[split]
        comparison[split] = dict(pt2=pt2, measure_a2=a2)
        if (pt2['score_paths'], pt2['performances'], pt2['segments'], pt2['aligned_notes']) != (
                a2['score_paths'], a2['performances'], a2['segments'], a2['notes']):
            raise ValueError(f'{split}: Measure A2 data do not match retained PT2 identities')
    write_json(output / 'comparison.json', comparison)
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split', 'model', 'score_paths', 'performances', 'retained_hmm_segments', 'unique_notes',
                         'original_windows', 'crop_windows', 'total_windows', 'original_note_appearances',
                         'crop_note_appearances', 'total_note_appearances', 'total_measure_appearances'])
        for split, values in comparison.items():
            a, b = values['pt2'], values['measure_a2']
            row = [split, 'PT2', a['score_paths'], a['performances'], a['segments'], a['aligned_notes']]
            writer.writerow([*row, a['original_windows'], a['crop_windows'], a['total_windows'],
                             a['original_note_appearances'], a['crop_note_appearances'], a['total_note_appearances'], 0])
            writer.writerow([split, 'Measure A2', b['score_paths'], b['performances'], b['segments'], b['notes'],
                             *[b[k]['windows'] for k in ('original', 'crop', 'total')],
                             *[b[k]['note_appearances'] for k in ('original', 'crop', 'total')],
                             b['total']['measure_appearances']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--train-references', type=Path, default=RUN / 'align_train/reference_manifest.json')
    parser.add_argument('--test-references', type=Path, default=PT2_EVAL / 'reference_manifest.json')
    parser.add_argument('--pt2-train-arrow', type=Path)
    parser.add_argument('--pt2-test-arrow', type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    references = {'train': args.train_references, 'test': args.test_references}
    caches = {'train': args.pt2_train_arrow, 'test': args.pt2_test_arrow}
    source = json.loads((PT2 / 'dataset_audit.json').read_text())
    pairs, slides = load_pt2_pairs(source)
    get_bars, _ = load_beat_mapper()
    audit = dict(status='passed', event_grid='measure', window_events=LIMIT, nominal_overlap_ratio=.5,
                 dynamic_padding=True, split_for_validation='test', random_crop_seed=SEED,
                 source_sha256=source['sha256'], splits={})
    for split in SPLITS:
        audit['splits'][split] = build_split(split, pairs[split], references[split], PT2 / 'asap-dataset-master',
                                            output, get_bars)
    write_comparison(output, audit['splits'], source, slides, caches)
    write_json(output / 'audit.json', audit)
    print((output / 'comparison.csv').read_text(), end='')


if __name__ == '__main__':
    main()
