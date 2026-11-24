#!/usr/bin/env python3
"""Compare saved A2 velocity predictions with shared score and human references."""

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script.analyze_exp1117_beat_median import load_current_references
from script.compare_exp1122_correlations import ARCHIVE, RUN, load_earliest_references, predictions
from script.evaluate_exp1116_baselines import common_masks, read_json
from script.evaluate_exp1117_five_main_metrics import beat_mean_curve, pearson
from src.evaluate.evaluate_dinr_beat import dataset_for

OLD = ROOT.parent / 'results/exp1117_beat_median_bs48/A2_seed42'
OUTPUT = ROOT / 'analysis/exp1122_structural_beat/a2_velocity_diagnosis.json'


def test_predictions(run, test_dir):
    return {read_json(path / 'metrics.json')['work']: path
            for path in (run / test_dir).glob('work_*')}


def score_work(path, ref, mask, expected_human):
    beat_mask, note_mask = mask['beat'], mask['velocity']
    membership = ref['membership']
    note_rows = ref['note_mask']
    human = ref['human_velocity']
    curves = [beat_mean_curve(row, membership, note_mask, len(beat_mask)) for row in human]
    result = dict(tf_same=[], tf_other=[], ar=[], ar_bias=[], ar_curve_std=[])
    for draw in range(3):
        for mode in ('teacher_forced', 'autoregressive'):
            with np.load(path / f'{mode}_{draw:02d}_tokens.npz') as data:
                np.testing.assert_array_equal(data['events'], ref['events'])
                np.testing.assert_array_equal(data['reference_continuous'], expected_human)
                velocity = data['predictions'][:, note_rows, 2]
            predicted = [beat_mean_curve(row, membership, note_mask, len(beat_mask)) for row in velocity]
            pairs = np.asarray([[pearson(pred, target, beat_mask) for target in curves]
                                for pred in predicted], dtype=float)
            if mode == 'teacher_forced':
                assert pairs.shape == (len(human), len(human))
                off_diag = ~np.eye(len(human), dtype=bool)
                result['tf_same'].append(float(np.nanmean(pairs.diagonal())))
                result['tf_other'].append(float(np.nanmean(pairs[off_diag])))
            else:
                assert pairs.shape == (1, len(human))
                result['ar'].append(float(np.nanmean(pairs)))
                result['ar_bias'].append(float(velocity[:, note_mask].mean() - human[:, note_mask].mean()))
                result['ar_curve_std'].append(float(np.nanstd(predicted[0][beat_mask])))
    return result


def train_tokens():
    old, new = [dataset_for(read_json(run / 'run_config.json')['train_sidecar'],
                            read_json(run / 'run_config.json')) for run in (OLD, RUN)]
    assert old.samples == new.samples
    counts = {name: np.zeros(2, dtype=int) for name in
              ('beat_duration_label', 'note_duration_label', 'onset_label', 'velocity_label',
               'beat_duration_history', 'note_duration_history', 'onset_history', 'velocity_history')}

    def accumulate(name, a, b, mask):
        counts[name] += [int(((a != b) & mask).sum()), int(mask.sum())]

    for i in range(len(old)):
        a, b = old[i], new[i]
        torch.testing.assert_close(a['encoder_tokens'], b['encoder_tokens'], rtol=0, atol=0)
        for col, name in enumerate(('beat_duration', 'note_duration', 'onset', 'velocity')):
            x, y = a['labels'][:, col], b['labels'][:, col]
            accumulate(name + '_label', x, y, (x >= 0) & (y >= 0))
        x, y = a['decoder_tokens'], b['decoder_tokens']
        torch.testing.assert_close(x[:, 0], y[:, 0], rtol=0, atol=0)
        for name, col, role in (('beat_duration', 2, 0), ('note_duration', 2, 1),
                                ('onset', 3, 1), ('velocity', 4, 1)):
            accumulate(name + '_history', x[:, col], y[:, col], x[:, 0] == role)
    return {name: dict(changed=int(n), total=int(total), fraction=float(n / total))
            for name, (n, total) in counts.items()}


def main():
    refs, old_sidecar = load_current_references()
    new_refs = load_earliest_references(refs, old_sidecar)
    published = read_json(ARCHIVE / 'metrics.json')
    methods = [row for row in published['results'] if row['id'] in ('pt_base', 'new_A2')]
    methods.append(dict(id='exp1122_A2', family='BEAT', source=str(RUN)))
    extracted = predictions(methods, refs)
    masks = common_masks(extracted['mean'] + extracted['earliest'], new_refs)
    directories = [test_predictions(OLD, 'test'), test_predictions(RUN, 'test_24proc')]
    per_work = []
    with np.load(ARCHIVE / 'common_masks.npz') as archived:
        for i, ref in enumerate(new_refs):
            if ref['humans'] < 2:
                continue
            for key in masks[i]:
                masks[i][key] &= archived[f'{i}_{key}']
            work = ref['work']
            row = dict(work=work, humans=ref['humans'])
            for version, dirs, reference in zip(('old', 'new'), directories, (refs[i], ref), strict=True):
                with np.load(dirs[work] / 'autoregressive_00_tokens.npz') as data:
                    expected = data['reference_continuous']
                np.testing.assert_array_equal(expected[:, ref['note_mask'], 2], reference['human_velocity'])
                row[version] = score_work(dirs[work], ref, masks[i], expected)
            per_work.append(row)

    summary = {}
    bootstrap = np.random.default_rng(42).integers(len(per_work), size=(10000, len(per_work)))
    for metric in per_work[0]['old']:
        old = np.asarray([row['old'][metric] for row in per_work])
        new = np.asarray([row['new'][metric] for row in per_work])
        diffs = (new - old).mean(axis=1)
        summary[metric] = dict(old=float(old.mean()), new=float(new.mean()),
            old_by_draw=old.mean(axis=0).tolist(), new_by_draw=new.mean(axis=0).tolist(),
            new_lower_works=int((diffs < 0).sum()),
            delta_work_bootstrap_95ci=np.percentile(diffs[bootstrap].mean(axis=1), [2.5, 97.5]).tolist())

    comparison = read_json(OUTPUT.with_name('correlation_comparison.json'))
    published_replay = {row['id']: row['correlations']['velocity']
                        for row in comparison['comparisons']['new_reference_earliest']['rows']}
    np.testing.assert_allclose([summary['ar']['old'], summary['ar']['new']],
                               [published_replay['new_A2'], published_replay['exp1122_A2']], rtol=0, atol=1e-12)
    result = dict(works=len(per_work), humans=sum(row['humans'] for row in per_work),
        protocol='Three saved draws, equal weight per work, common score-beat and note masks. '
                 'TF compares each history to its own and separately to the other performances. '
                 'AR compares its independent draw to all performances.',
        summary=summary, training_token_changes=train_tokens(), per_work=per_work)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('summary', 'training_token_changes')}, indent=2))


if __name__ == '__main__':
    main()
