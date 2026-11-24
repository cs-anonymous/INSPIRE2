#!/usr/bin/env python3
"""Replay the exp1116 leave-one-out comparison under both beat rules."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import script.evaluate_exp1116_baselines as baseline
from script.analyze_exp1117_beat_median import load_current_references
from script.evaluate_exp1117_beat_dev import duration_log_coordinate
from script.evaluate_exp1117_five_main_metrics import matched_human_metrics
from script.evaluate_exp1122_a2_24proc import RUN, SIDECAR
from src.data_process.beat_mapping import map_score_beats, map_score_beats_structural_earliest

ARCHIVE = ROOT / 'analysis/exp1117_beat_median/result_analysis/five_main_metrics'
OUTPUT = ROOT / 'analysis/exp1122_structural_beat/correlation_comparison.json'
IDS = ('human_heldout', 'score_midi', 'pt_base', 'new_A2', 'exp1122_A2')


def load_earliest_references(references, old_sidecar):
    def groups(path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        grouped = {}
        for row in payload['performances']:
            grouped.setdefault(row['work'], []).append(row)
        return payload, grouped

    _, old_groups = groups(old_sidecar)
    payload, new_groups = groups(SIDECAR)
    result = deepcopy(references)
    for ref in result:
        old_rows = old_groups[ref['work']]
        new_rows = {row['source']: row for row in new_groups[ref['work']]}
        if set(new_rows) != {row['source'] for row in old_rows}:
            raise ValueError(f'Human cohort differs for {ref["work"]}')
        targets, masks = [], []
        for row in old_rows:
            a, b = new_rows[row['source']]['span']
            np.testing.assert_array_equal(payload['events'][a:b].numpy(), ref['events'])
            targets.append(payload['targets'][a:b].numpy())
            masks.append(payload['target_valid'][a:b].numpy())
        targets, masks = np.stack(targets).astype(float), np.stack(masks)
        beat, note = ref['beat_mask'], ref['note_mask']
        np.testing.assert_array_equal(targets[:, note, 2], ref['human_velocity'])
        ref.update(human_beat_log=np.log(np.maximum(targets[:, beat, 0], 1.)),
                   human_duration=np.log(np.maximum(targets[:, note, 0], 1.)) - ref['score_log_duration'],
                   human_onset=targets[:, note, 1], beat_valid=masks[:, beat, 0].all(0),
                   duration_valid=masks[:, note, 0].all(0), onset_valid=masks[:, note, 1].all(0),
                   velocity_valid=masks[:, note, 2].all(0))
    return result


def predictions(methods, references):
    by_work = {ref['work']: i for i, ref in enumerate(references)}
    result = {kind: [] for kind in ('mean', 'earliest', 'native')}
    expected = {(i, draw) for i in range(len(references)) for draw in range(3)}
    for method in methods:
        source = Path(method['source'])
        features = {kind: {} for kind in result}
        if method['family'] == 'BEAT':
            run = baseline.read_json(source / 'run_config.json')
            test_dir = source / ('test_24proc' if method['id'] == 'exp1122_A2' else 'test')
            inputs = [(by_work[baseline.read_json(work_dir / 'metrics.json')['work']], draw,
                       work_dir / f'autoregressive_{draw:02d}_tokens.npz')
                      for work_dir in sorted(test_dir.glob('work_*')) for draw in range(3)]
        else:
            rows = baseline.read_json(source)['rows']
            inputs = [(int(row['work']), int(row['sample']), Path(row['prediction'])) for row in rows]
            for row in rows:
                if row['score_source'] != references[int(row['work'])]['score_source']:
                    raise ValueError('PT-1 score identity differs')
        if len(inputs) != len(expected) or {(i, draw) for i, draw, _ in inputs} != expected:
            raise ValueError(f'Incomplete or duplicate predictions for {method["id"]}')
        for i, draw, path in inputs:
            ref = references[i]
            key = i, draw
            if method['family'] == 'BEAT':
                notes = baseline.beat_notes(path, run, ref)
            else:
                stream = baseline.read_note_on_stream(path)
                notes = baseline.align_note_ons(stream, ref['pitches'])
            features['mean'][key] = baseline.note_features(notes, ref, map_score_beats)
            features['earliest'][key] = baseline.note_features(notes, ref, map_score_beats_structural_earliest)
            if method['family'] == 'BEAT':
                with np.load(path) as data:
                    tokens = data['predictions'].reshape(-1, 3)
                mask = ref['beat_mask']
                events = ref['events'][mask]
                beat_log = duration_log_coordinate(tokens[mask, 0], events, run['beat_duration_target'])
                beat_log += np.log(np.maximum(events[:, 2], 1.))
                features['native'][key] = dict(features['earliest'][key], beat_log=beat_log,
                                               onset=tokens[ref['note_mask'], 1] * .01 - .995)
            else:
                features['native'][key] = features['earliest'][key]
        for kind in result:
            result[kind].append(dict(method=method, features=features[kind]))
    return result


def main():
    references, old_sidecar = load_current_references()
    new_references = load_earliest_references(references, old_sidecar)
    published = baseline.read_json(ARCHIVE / 'metrics.json')
    methods = [row for row in published['results'] if row['id'] in ('pt_base', 'new_A2')]
    methods.append(dict(id='exp1122_A2', family='BEAT', label='Exp1122 A2', source=str(RUN)))
    prediction_sets = predictions(methods, references)
    with np.load(ARCHIVE / 'common_masks.npz') as data:
        old_masks = [{name: data[f'{i}_{name}'] for name in ('beat', 'onset', 'duration', 'velocity')}
                     for i in range(len(references))]
    original = matched_human_metrics(references, old_masks, prediction_sets['mean'][:2])
    published_rows = {row['id']: row for row in published['human_matched']['rows']}
    for row in original['rows']:
        for kind, value in row['correlations'].items():
            np.testing.assert_allclose(value, published_rows[row['id']]['correlations'][kind], atol=1e-12, rtol=0)

    common = baseline.common_masks(prediction_sets['mean'] + prediction_sets['earliest'], new_references)
    for shared, old in zip(common, old_masks):
        for key in shared:
            shared[key] &= old[key]
    comparisons = {}
    for name, refs, mapper in (
        ('old_reference_mean', references, 'mean'),
        ('new_reference_mean', new_references, 'mean'),
        ('old_reference_earliest', references, 'earliest'),
        ('new_reference_earliest', new_references, 'earliest'),
        ('new_reference_native_beat', new_references, 'native'),
    ):
        result = matched_human_metrics(refs, common, prediction_sets[mapper])
        result['correlation_work_counts'] = {
            identifier: {kind: sum(row['metrics'][identifier]['correlations'][kind] is not None
                                   for row in result['per_work']) for kind in ('tempo', 'velocity')}
            for identifier in IDS
        }
        comparisons[name] = result
    # Ensure each row compares the same works across the two endpoint protocols.
    for old, new in zip(comparisons['old_reference_mean']['per_work'],
                        comparisons['new_reference_earliest']['per_work'], strict=True):
        assert old['work'] == new['work']
        for identifier in IDS:
            for kind in ('tempo', 'velocity'):
                assert (old['metrics'][identifier]['correlations'][kind] is None) == (
                    new['metrics'][identifier]['correlations'][kind] is None)
    result = dict(protocol='exp1116 section 7.11 leave-one-human-out; common valid masks '
                  'are the intersection of both sidecars, both note mappers, all compared '
                  'model draws, and the archived exp1116 common masks',
                  original_reproduced=original, comparisons=comparisons)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    for name, comparison in [('published', original), *comparisons.items()]:
        print(name, comparison['works'], comparison['human_performances'])
        for row in comparison['rows']:
            print(row['id'], *(f'{value:.6f}' for value in row['correlations'].values()),
                  comparison.get('correlation_work_counts', {}).get(row['id'], ''))


if __name__ == '__main__':
    main()
