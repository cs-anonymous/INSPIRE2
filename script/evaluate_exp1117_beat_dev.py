#!/usr/bin/env python3
"""Score-event metrics in the same target coordinates used by exp1117."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_process.dinr_beat_dataset import BeatTokenSpec as S


def decode_numpy(tokens, bounds, bins):
    low, high = bounds
    return low + (np.clip(np.asarray(tokens, dtype=np.float64), 0, bins - 1) + 0.5) * ((high - low) / bins)


def duration_log_coordinate(tokens, events, mode):
    tokens = np.asarray(tokens, dtype=np.int64)
    events = np.asarray(events, dtype=np.float64)
    bounds = S.duration_abs_range if mode == 'absolute_log' else S.duration_dev_range
    value = decode_numpy(tokens, bounds, S.duration_bins)
    if mode == 'absolute_log':
        value = value - np.log(np.maximum(events[:, 2], 1.0))
    return value


def work_metrics(npz_path: Path, run: dict) -> dict:
    data = np.load(npz_path)
    events = data['events'].astype(np.float64)
    predictions = data['predictions']
    references = data['references']
    if predictions.ndim == 3:
        predictions = predictions[0]
    is_beat = events[:, 0].astype(np.int64) == S.beat_type
    is_note = ~is_beat
    pred_duration = duration_log_coordinate(predictions[:, 0], events, run['duration_target'])
    score_log_duration = np.log(np.maximum(events[:, 2], 1.0))
    pred_log_duration = pred_duration + score_log_duration
    if 'reference_continuous' not in data or 'reference_valid' not in data:
        raise ValueError(f'{npz_path}: continuous reference labels and validity masks are required')
    continuous = data['reference_continuous'].astype(np.float64)
    valid = data['reference_valid'].astype(bool)
    ref_log_duration = np.log(np.maximum(continuous[:, :, 0], 1.0))
    ref_duration = ref_log_duration - score_log_duration[None, :]
    ref_valid_duration = valid[:, :, 0]
    if run.get('onset_target', 'dev') == 'dev':
        pred_onset = decode_numpy(predictions[:, 1], S.onset_dev_range, S.onset_dev_bins)
    else:
        pred_onset = (decode_numpy(predictions[:, 1], S.onset_input_range, S.onset_input_bins)
                      - events[:, 3])
    ref_onset = continuous[:, :, 1]
    ref_valid_onset = valid[:, :, 1]

    note_indices = np.flatnonzero(is_note)
    onset_ok = np.all(ref_valid_onset[:, note_indices], axis=0)
    onset_errors = np.abs(ref_onset[:, note_indices][:, onset_ok]
                          - pred_onset[note_indices][None, onset_ok])
    onset_w1 = float(onset_errors.mean()) if onset_errors.size else float('nan')

    duration_ok = np.all(ref_valid_duration[:, note_indices], axis=0)
    duration_errors = np.abs(ref_duration[:, note_indices][:, duration_ok]
                             - pred_duration[note_indices][None, duration_ok])
    duration_w1 = float(duration_errors.mean()) if duration_errors.size else float('nan')

    ref_velocity = continuous[:, :, 2]
    velocity_ok = np.all(valid[:, note_indices, 2], axis=0)
    velocity_errors = np.abs(ref_velocity[:, note_indices][:, velocity_ok]
                             - predictions[note_indices, 2][None, velocity_ok])
    velocity_w1 = float(velocity_errors.mean()) if velocity_errors.size else float('nan')

    beat_indices = np.flatnonzero(is_beat)
    tempo_ok = np.all(ref_valid_duration[:, beat_indices], axis=0)
    predicted_tempo = -pred_log_duration[beat_indices][tempo_ok]
    human_tempo = -np.nanmedian(ref_log_duration[:, beat_indices][:, tempo_ok], axis=0)
    if len(predicted_tempo) >= 3 and np.std(predicted_tempo) > 1e-12 and np.std(human_tempo) > 1e-12:
        tempo = float(pearsonr(predicted_tempo, human_tempo).statistic)
    else:
        tempo = float('nan')
    return {
        'tempo_curve_pearson': tempo,
        'onset_w1': onset_w1,
        'duration_w1': duration_w1,
        'velocity_w1': velocity_w1,
        'onset_notes': int(onset_ok.sum()),
        'duration_notes': int(duration_ok.sum()),
        'tempo_beats': int(tempo_ok.sum()),
    }


def evaluate_run(run_dir: Path, test_dir_name='test') -> dict:
    run = json.loads((run_dir / 'run_config.json').read_text())
    if '_seed' in run_dir.name:
        setting, seed = run_dir.name.rsplit('_seed', 1)
        training_seed = int(seed)
    else:
        setting = run.get('setting', run_dir.name)
        training_seed = int(run.get('seed', 42))
    rows = []
    for sample in range(3):
        works = []
        for work_dir in sorted((run_dir / test_dir_name).glob('work_*')):
            path = work_dir / f'autoregressive_{sample:02d}_tokens.npz'
            if not path.exists():
                continue
            metrics = work_metrics(path, run)
            metrics['work'] = work_dir.name
            works.append(metrics)
        row = {'sample': sample, 'works': len(works)}
        if len(works) != 19:
            raise ValueError(f'{run_dir}: incomplete sample {sample}: {len(works)}/19 works')
        for name in ('tempo_curve_pearson', 'onset_w1', 'duration_w1', 'velocity_w1'):
            row[name] = float(np.nanmean([item[name] for item in works]))
        rows.append(row)
    metrics = {}
    for name in ('tempo_curve_pearson', 'onset_w1', 'duration_w1', 'velocity_w1'):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        metrics[name] = {
            'mean': float(np.nanmean(values)),
            'std': float(np.nanstd(values, ddof=1)),
            'sampling_values': values.tolist(),
        }
    return {'setting': setting, 'training_seed': training_seed,
            'metrics': metrics, 'per_sample': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path,
                        default=ROOT.parent / 'results/exp1117_beat_dev_bs48')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'analysis/exp1116_beat_dev/metrics.json')
    args = parser.parse_args()
    result = {
        'protocol': {
            'onset': 'Per-note mean W1 between predicted note onset dev (200 bins of width 0.01) and unquantized aligned human dev targets; mask unsupported intervals.',
            'duration': 'Per-note mean W1 between predicted and human log(performance duration / score duration).',
            'velocity': 'Per-note mean W1 on MIDI velocity.',
            'tempo': 'Pearson correlation of predicted and human median beat-level negative log duration, retaining aligned score beat indices.',
            'aggregation': 'Within each sampling draw, macro-average works; report mean and sample std over three AR draws.',
        },
        'settings': [],
    }
    for run_dir in sorted(args.run_root.glob('*_seed*')):
        if not (run_dir / 'training_summary.json').exists():
            continue
        if not (run_dir / 'test/metrics.json').exists():
            continue
        row = evaluate_run(run_dir)
        result['settings'].append(row)
        print(row['setting'], {k: f"{v['mean']:.3f} ± {v['std']:.3f}"
                               for k, v in row['metrics'].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
