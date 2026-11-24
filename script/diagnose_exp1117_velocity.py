#!/usr/bin/env python3
"""Diagnose beat velocity with loss curves, teacher forcing, and AR feedback."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
INSPIRE = PROJECT / 'INSPIRE2'
CAMPAIGN = PROJECT / 'results/exp1117_beat_median_bs48'
OUTPUT = INSPIRE / 'analysis/exp1117_beat_median/velocity_diagnostics'
SETTINGS = ('A1', 'A2', 'A3', 'C1', 'C2', 'C3')


def pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) <= 1e-12 or np.std(y[mask]) <= 1e-12:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def beat_curve(values, events):
    beat = events[:, 0].astype(int) == 0
    note = ~beat
    membership = np.cumsum(beat) - 1
    count = int(beat.sum())
    sums = np.bincount(membership[note], weights=np.asarray(values)[note], minlength=count)
    counts = np.bincount(membership[note], minlength=count)
    curve = np.full(count, np.nan)
    curve[counts > 0] = sums[counts > 0] / counts[counts > 0]
    return curve


def curve_diagnostics(setting):
    test = CAMPAIGN / f'{setting}_seed42/test'
    tf, ar = [], []
    generated_values, human_values = [], []
    for work_dir in sorted(test.glob('work_*')):
        metrics_path = work_dir / 'metrics.json'
        if not metrics_path.exists():
            continue
        for sample in range(3):
            tf_data = np.load(work_dir / f'teacher_forced_{sample:02d}_tokens.npz')
            ar_data = np.load(work_dir / f'autoregressive_{sample:02d}_tokens.npz')
            events = tf_data['events']
            human = tf_data['reference_continuous'][:, :, 2]
            human_curves = np.stack([beat_curve(row, events) for row in human])
            for destination, data in ((tf, tf_data['predictions']),
                                       (ar, ar_data['predictions'])):
                predictions = data[0:1] if data.ndim == 3 and data.shape[0] == 1 else data
                for prediction in predictions:
                    generated = beat_curve(prediction[:, 2], events)
                    values = [pearson(generated, target) for target in human_curves]
                    destination.append(float(np.mean([v for v in values if v is not None])))
            generated_values.extend(ar_data['predictions'][0, :, 2][events[:, 0].astype(int) == 1])
            human_values.extend(human[:, events[:, 0].astype(int) == 1].ravel())
    generated_values = np.asarray(generated_values, dtype=float)
    human_values = np.asarray(human_values, dtype=float)
    return {
        'teacher_forced_curve_pearson': float(np.mean(tf)),
        'autoregressive_curve_pearson': float(np.mean(ar)),
        'teacher_forced_pairs': len(tf),
        'autoregressive_pairs': len(ar),
        'autoregressive_velocity_mean': float(generated_values.mean()),
        'autoregressive_velocity_std': float(generated_values.std()),
        'human_velocity_mean': float(human_values.mean()),
        'human_velocity_std': float(human_values.std()),
        'autoregressive_signed_mean_error': float(generated_values.mean() - human_values.mean()),
    }


def read_history(setting):
    path = CAMPAIGN / f'{setting}_seed42/loss_history.jsonl'
    rows = []
    for line in path.read_text().splitlines():
        item = json.loads(line)
        if 'eval_loss' in item:
            rows.append({'setting': setting, 'split': 'valid', 'epoch': float(item['epoch']),
                         'loss': float(item['eval_loss'])})
        elif 'loss' in item:
            rows.append({'setting': setting, 'split': 'train', 'epoch': float(item['epoch']),
                         'loss': float(item['loss'])})
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = sum((read_history(setting) for setting in SETTINGS), [])
    with (OUTPUT / 'loss_curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['setting', 'split', 'epoch', 'loss'])
        writer.writeheader()
        writer.writerows(rows)

    diagnostics = {}
    for setting in SETTINGS:
        valid = json.loads((OUTPUT / setting / 'metrics.json').read_text())
        test = json.loads((CAMPAIGN / f'{setting}_seed42/test/metrics.json').read_text())
        rounds = {row['mode']: row for row in test['rounds'] if row['sample'] == 0}
        diagnostics[setting] = {
            'valid_total_loss': float(valid['loss']),
            'valid_head_ce': {name: float(valid['metrics'][f'{name}_ce']) for name in
                              ('beat_duration', 'note_duration', 'onset', 'velocity')},
            'valid_velocity_mae': float(valid['metrics']['velocity_mae']),
            'teacher_forced_velocity_mae': float(rounds['teacher_forced']['metrics']['velocity_mae']),
            'autoregressive_velocity_mae': float(rounds['autoregressive']['metrics']['velocity_mae']),
            'teacher_forced_velocity_signed_error': float(rounds['teacher_forced']['metrics']['velocity_signed_error']),
            'autoregressive_velocity_signed_error': float(rounds['autoregressive']['metrics']['velocity_signed_error']),
            **curve_diagnostics(setting),
        }
    (OUTPUT / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, constrained_layout=True)
    for axis, setting in zip(axes.flat, SETTINGS):
        history = [row for row in rows if row['setting'] == setting]
        train = [row for row in history if row['split'] == 'train']
        valid = [row for row in history if row['split'] == 'valid']
        axis.plot([r['epoch'] for r in train], [r['loss'] for r in train], lw=.7, alpha=.45, label='train')
        axis.plot([r['epoch'] for r in valid], [r['loss'] for r in valid], 'o-', ms=2.5, label='valid')
        axis.set_title(setting)
        axis.set_xlabel('epoch')
        axis.grid(alpha=.2)
    axes[0, 0].set_ylabel('total CE')
    axes[1, 0].set_ylabel('total CE')
    axes[0, 0].legend(frameon=False)
    fig.savefig(OUTPUT / 'loss_curves.png', dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    names = ('beat_duration', 'note_duration', 'onset', 'velocity')
    x = np.arange(len(SETTINGS))
    width = .2
    for i, name in enumerate(names):
        axis.bar(x + (i - 1.5) * width,
                 [diagnostics[s]['valid_head_ce'][name] for s in SETTINGS], width, label=name)
    axis.set_xticks(x, SETTINGS)
    axis.set_ylabel('validation cross entropy')
    axis.set_title('Best-checkpoint validation loss by head')
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis='y', alpha=.2)
    fig.savefig(OUTPUT / 'head_ce.png', dpi=180)
    plt.close(fig)

    lines = [
        '### Beat velocity diagnosis', '',
        'The total loss is the sum of four separately averaged categorical cross-entropies. The loss curves and best-checkpoint head losses are in `loss_curves.png`, `head_ce.png`, `loss_curves.csv`, and `diagnostics.json`.', '',
        '| Setting | Valid velocity CE | Teacher-forced velocity MAE | Autoregressive velocity MAE | Teacher-forced beat-velocity corr | Autoregressive beat-velocity corr | AR signed velocity error |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for setting in SETTINGS:
        d = diagnostics[setting]
        lines.append(f"| {setting} | {d['valid_head_ce']['velocity']:.3f} | {d['teacher_forced_velocity_mae']:.2f} | {d['autoregressive_velocity_mae']:.2f} | {d['teacher_forced_curve_pearson']:.3f} | {d['autoregressive_curve_pearson']:.3f} | {d['autoregressive_velocity_signed_error']:.2f} |")
    lines += ['',
        'The velocity head is learning a useful one-step conditional distribution: validation velocity CE is about 3.69–3.72 nats for all settings, versus ln(128)=4.85 nats for a uniform 128-class predictor. Teacher-forced beat velocity correlation is about 0.72, close to the Human leave-one-out value 0.755.', '',
        'The failure appears after autoregressive feedback. The same models fall to roughly 0.07–0.15 beat-velocity correlation and their velocity MAE increases sharply. During generation, each predicted velocity is written into the next decoder input; training instead uses ground-truth history with 50% random decoder masking. This exposure mismatch lets early velocity errors alter later hidden states. The positive AR signed error also shows a systematic loudness drift, rather than only random fluctuation.', '',
        'Therefore the current beat-velocity result does indicate a weakness in free-running dynamics, but it does not show that the velocity head cannot learn the beat-level pattern. The first fixes to test are reducing velocity feedback exposure bias (scheduled/generated-history training or removing velocity from decoder feedback) and adding an ordinal/continuous velocity loss alongside CE. A velocity-only loss reweighting is not the first diagnosis: its teacher-forced CE is already comparable across settings.',
    ]
    (OUTPUT / 'diagnosis.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
