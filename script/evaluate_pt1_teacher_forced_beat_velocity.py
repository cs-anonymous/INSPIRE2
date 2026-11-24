#!/usr/bin/env python3
"""Evaluate PT-1 teacher-forced beat velocity on the current 19-work references."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(PROJECT))

from script.analyze_exp1117_beat_median import load_current_references
from script.evaluate_exp1117_five_main_metrics import beat_mean_curve, pearson
from src.model.pianoformer import PianoT5Gemma
from src.utils.midi import midi_to_ids


CHECKPOINT = PROJECT / 'results/halfepoch_earlystop_20261112/pt_base/checkpoints/sft_2026-11-12-00-58-12/checkpoint-643'
OUTPUT = ROOT / 'analysis/exp1117_beat_median/pt1_teacher_forced_velocity'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_TOKENS = 4096
SAMPLES = 3


def stable_seed(work, history, sample):
    return int.from_bytes(hashlib.sha256(f'pt1-tf:{work}:{history}:{sample}'.encode()).digest()[:4], 'little')


def sample_top_p(logits, generator, top_p=.9):
    values, indices = logits.float().sort(dim=-1, descending=True)
    probs = values.softmax(dim=-1)
    remove = probs.cumsum(-1) - probs > top_p
    remove[..., 0] = False
    selected = torch.multinomial(values.masked_fill(remove, -torch.inf).softmax(-1), 1, generator=generator)
    return indices.gather(-1, selected).squeeze(-1)


def clamp(value, lo, hi):
    return min(hi - 1, max(lo, int(round(float(value)))))


def target_ids(reference, targets, score_ids, config):
    """Build PT's eight fields from the current continuous BEAT human target."""
    events = reference['events']
    beat = events[:, 0].astype(bool) == 0
    note = ~beat
    membership = reference['membership']
    beat_durations = np.asarray(targets[beat, 0], dtype=float)
    anchors = np.r_[0., np.cumsum(beat_durations)[:-1]]
    positions = events[note, 3] + targets[note, 1]
    starts = anchors[membership] + positions * beat_durations[membership]
    durations = np.maximum(np.asarray(targets[note, 0], dtype=float), 1.)
    velocities = np.clip(np.asarray(targets[note, 2], dtype=float), 1., 127.)
    iois = np.diff(starts, prepend=0.)
    ids = []
    for i, (ioi, duration, velocity) in enumerate(zip(iois, durations, velocities)):
        row = [
            int(score_ids[i * 8]),
            clamp(config.timing_start + ioi, *config.valid_id_range[1]),
            clamp(config.velocity_start + velocity, *config.valid_id_range[2]),
            clamp(config.timing_start + duration, *config.valid_id_range[3]),
            config.pedal_start, config.pedal_start, config.pedal_start, config.pedal_start,
        ]
        ids.extend(row)
    return np.asarray(ids, dtype=np.int64), starts, durations, velocities


@torch.inference_mode()
def predict_tf(model, score_ids, target_ids_array, seed):
    """Predict each token with true preceding target tokens, in independent windows."""
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    output = []
    for start in range(0, len(score_ids), MAX_TOKENS):
        end = min(start + MAX_TOKENS, len(score_ids))
        score = torch.as_tensor(score_ids[start:end], device=DEVICE).view(1, -1)
        target = torch.as_tensor(target_ids_array[start:end], device=DEVICE).view(1, -1)
        bos = torch.full((1, 1), int(model.config.bos_token_id), dtype=torch.long, device=DEVICE)
        decoder = torch.cat([bos, target[:, :-1]], dim=1)
        result = model(input_ids=score, decoder_input_ids=decoder)
        logits = result.logits[0]
        rows = torch.empty(logits.shape[0], dtype=torch.long, device=DEVICE)
        for field in range(8):
            positions = torch.arange(field, logits.shape[0], 8, device=DEVICE)
            lo, hi = model.config.valid_id_range[field]
            if field == 0:
                rows[positions] = score[0, positions]
            else:
                rows[positions] = sample_top_p(logits[positions, lo:hi], generator) + lo
        output.append(rows.cpu().numpy())
    return np.concatenate(output)


def notes_from_prediction(prediction, config):
    fields = prediction.reshape(-1, 8)
    iois = fields[:, 1] - config.timing_start
    starts = np.cumsum(iois.astype(float))
    durations = np.maximum(fields[:, 3] - config.timing_start, 1.)
    velocities = np.clip(fields[:, 2] - config.velocity_start, 1., 127.)
    return starts, durations, velocities


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    references, reference_path = load_current_references()
    model = PianoT5Gemma.from_pretrained(str(CHECKPOINT), torch_dtype=torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32).to(DEVICE).eval()
    rows = []
    for work_index, reference in enumerate(references):
        score = MidiFile(reference['score_path'])
        score_ids = np.asarray(midi_to_ids(model.config, score), dtype=np.int64)
        payload = torch.load(reference_path, map_location='cpu', weights_only=False)
        performance_rows = [r for r in payload['performances'] if r['work'] == reference['work']]
        work_rows = []
        for history_index, performance in enumerate(performance_rows):
            left, right = performance['span']
            targets = payload['targets'][left:right].numpy().astype(float)
            target_ids_array, _, _, _ = target_ids(reference, targets, score_ids, model.config)
            for sample in range(SAMPLES):
                prediction = predict_tf(model, score_ids, target_ids_array, stable_seed(work_index, history_index, sample))
                starts, durations, velocities = notes_from_prediction(prediction, model.config)
                beat_count = int(reference['beat_mask'].sum())
                beat_log = np.full(beat_count, np.nan)
                # Predicted beat intervals are recovered by mapping note starts to the score beat grid.
                from src.data_process.beat_mapping import map_score_beats, valid_beat_intervals
                anchors, supported = map_score_beats(reference['edges'], reference['ticks'], starts)
                valid, periods = valid_beat_intervals(anchors, supported)
                beat_log[valid] = np.log(np.maximum(periods[valid], 1.))
                membership = reference['membership']
                onset = np.full(reference['notes'], np.nan)
                ok = valid[membership]
                onset[ok] = ((starts[ok] - anchors[membership[ok]]) / periods[membership[ok]] - reference['score_ratio'][ok])
                features = dict(beat_log=beat_log, onset=onset,
                                duration=np.log(durations) - reference['score_log_duration'], velocity=velocities)
                human_velocity = np.stack([beat_mean_curve(reference['human_velocity'][h], membership,
                    reference['velocity_valid'], beat_count) for h in range(reference['humans'])])
                generated_curve = beat_mean_curve(velocities, membership, reference['velocity_valid'], beat_count)
                correlations = [pearson(generated_curve, row, reference['beat_valid']) for row in human_velocity]
                correlations = [v for v in correlations if v is not None]
                work_rows.append(dict(work=reference['work'], history=history_index, sample=sample,
                    velocity_curve_corr=float(np.mean(correlations)),
                    velocity_mae=float(np.abs(velocities[None, :] - reference['human_velocity']).mean()),
                    velocity_mean=float(velocities.mean()), velocity_std=float(velocities.std()),
                    signed_velocity_error=float(velocities.mean() - reference['human_velocity'].mean())))
        rows.extend(work_rows)
        print(f'completed {work_index + 1}/{len(references)} {reference["work"]}', flush=True)
    per_work = []
    for reference in references:
        selected = [r for r in rows if r['work'] == reference['work']]
        per_work.append(dict(work=reference['work'], velocity_curve_corr=float(np.mean([r['velocity_curve_corr'] for r in selected])),
                             velocity_mae=float(np.mean([r['velocity_mae'] for r in selected])),
                             velocity_mean=float(np.mean([r['velocity_mean'] for r in selected])),
                             velocity_std=float(np.mean([r['velocity_std'] for r in selected])),
                             signed_velocity_error=float(np.mean([r['signed_velocity_error'] for r in selected]))))
    result = dict(checkpoint=str(CHECKPOINT), works=len(references), histories=len(rows) // SAMPLES,
                  samples=SAMPLES, per_work=per_work,
                  metrics={key: float(np.mean([r[key] for r in per_work])) for key in
                           ('velocity_curve_corr', 'velocity_mae', 'velocity_mean', 'velocity_std', 'signed_velocity_error')},
                  protocol='PT-1 teacher forcing with each human target history; 4096-token independent windows; top-p=.9; current beat-median references')
    (OUTPUT / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (OUTPUT / 'rows.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
