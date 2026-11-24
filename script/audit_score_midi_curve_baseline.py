#!/usr/bin/env python3
"""Independent audit of the Score MIDI curve baseline."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
from script.analyze_exp1117_beat_median import load_current_references
from script.evaluate_exp1117_five_main_metrics import beat_mean_curve, pearson

OUTPUT = ROOT / 'analysis/exp1117_beat_median/result_analysis/five_main_metrics/score_midi_audit.json'


def main():
    references, _ = load_current_references()
    common_masks = np.load(ROOT / 'analysis/exp1117_beat_median/result_analysis/five_main_metrics/common_masks.npz')
    rows = []
    velocity_values, tempo_values = [], []
    mismatches = []
    for work_index, reference in enumerate(references):
        score = MidiFile(reference['score_path'])
        score_notes = sorted(
            [note for instrument in score.instruments if not instrument.is_drum for note in instrument.notes],
            key=lambda note: (note.start, note.pitch, note.end, note.velocity),
        )
        score_velocity = np.asarray([note.velocity for note in score_notes], dtype=float)
        event_velocity = reference['events'][reference['note_mask'], 4].astype(float)
        identity = np.array_equal(score_velocity, event_velocity)
        if not identity:
            mismatches.append(reference['work'])

        beat_count = int(reference['beat_mask'].sum())
        velocity_mask = common_masks[f'{work_index}_velocity']
        beat_mask = common_masks[f'{work_index}_beat']
        score_curve = beat_mean_curve(event_velocity, reference['membership'], velocity_mask, beat_count)
        human_velocity = np.stack([
            beat_mean_curve(reference['human_velocity'][h], reference['membership'], velocity_mask, beat_count)
            for h in range(reference['humans'])
        ])
        velocity_corr = [pearson(score_curve, row, beat_mask) for row in human_velocity]
        velocity_corr = [value for value in velocity_corr if value is not None]

        score_tempo = -np.log(np.maximum(reference['events'][reference['beat_mask'], 2], 1.0))
        human_tempo = -reference['human_beat_log']
        tempo_corr = [pearson(score_tempo, row, beat_mask) for row in human_tempo]
        tempo_corr = [value for value in tempo_corr if value is not None]
        rows.append(dict(work=reference['work'], score_velocity_identity=bool(identity),
                         velocity_corr=float(np.mean(velocity_corr)),
                         tempo_corr=(float(np.mean(tempo_corr)) if tempo_corr else None),
                         velocity_references=len(velocity_corr), tempo_references=len(tempo_corr),
                         score_velocity_mean=float(event_velocity.mean()),
                         score_velocity_std=float(event_velocity.std())))
        velocity_values.extend(velocity_corr)
        tempo_values.extend(tempo_corr)

    tempo_work_values = [row['tempo_corr'] for row in rows if row['tempo_corr'] is not None]
    result = dict(protocol={
        'score_velocity': 'events[note,4] compared with score MIDI note.velocity after the same canonical note sort',
        'velocity_curve': 'arithmetic mean score velocity per score-onset beat; current common valid velocity and beat masks',
        'correlation': 'Pearson against every individual human curve; average within work, then macro-average works',
        'tempo_curve': 'negative log score beat duration vs negative log human beat duration',
    }, works=len(rows), score_velocity_identity_mismatches=mismatches,
        macro_velocity_corr=float(np.mean([row['velocity_corr'] for row in rows])),
        macro_tempo_corr=float(np.mean(tempo_work_values)),
        pooled_velocity_pair_corr=float(np.mean(velocity_values)),
        pooled_tempo_pair_corr=float(np.mean(tempo_values)), per_work=rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_work'}, indent=2))


if __name__ == '__main__':
    main()
