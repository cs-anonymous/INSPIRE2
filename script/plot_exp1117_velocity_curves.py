#!/usr/bin/env python3
"""Plot beat velocity and beat duration for Human, PT-1, and A2."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
from script.analyze_exp1117_beat_median import load_current_references
from script.evaluate_exp1117_five_main_metrics import beat_mean_curve
import script.evaluate_exp1116_baselines as baseline

OUTPUT = ROOT / 'analysis/exp1117_beat_median/result_analysis/five_works_curves'
SELECTED = (0, 3, 5, 9, 13)


def slug(value):
    return ''.join(c.lower() if c.isalnum() else '_' for c in value).strip('_')[:90]


def curve_corr(predicted, human_curves, mask):
    values = []
    for human in human_curves:
        use = mask & np.isfinite(predicted) & np.isfinite(human)
        if use.sum() >= 3 and np.std(predicted[use]) > 1e-12 and np.std(human[use]) > 1e-12:
            values.append(float(np.corrcoef(predicted[use], human[use])[0, 1]))
    return float(np.mean(values)) if values else float('nan')


def fmt_corr(value):
    return 'undefined' if not np.isfinite(value) else f'{value:.3f}'


def make_curves(reference, index, features, masks, attribute):
    events = reference['events']
    note_mask = reference['note_mask']
    beat_count = int(reference['beat_mask'].sum())
    membership = reference['membership']
    common_note = masks[f'{index}_velocity']
    common_beat = masks[f'{index}_beat']
    if attribute == 'velocity':
        score = beat_mean_curve(events[note_mask, 4], membership, common_note, beat_count)
        human = np.stack([beat_mean_curve(reference['human_velocity'][h], membership, common_note, beat_count)
                          for h in range(reference['humans'])])
        models = {name: np.stack([beat_mean_curve(features[name][(index, sample)]['velocity'], membership,
                                                  common_note, beat_count) for sample in range(3)])
                  for name in ('pt_base', 'new_A2')}
        ylabel, limit = 'Mean note-on velocity per beat', (0, 127)
    elif attribute == 'duration':
        score = np.log(np.maximum(events[reference['beat_mask'], 2], 1.0))
        human = np.asarray(reference['human_beat_log'], dtype=float)
        models = {name: np.stack([features[name][(index, sample)]['beat_log'] for sample in range(3)])
                  for name in ('pt_base', 'new_A2')}
        ylabel, limit = r'ln(beat duration / ms)', None
    else:
        raise ValueError(attribute)
    return dict(score=score, human=human, models=models, mask=common_beat,
                ylabel=ylabel, limit=limit, beats=beat_count)


def draw_panel(ax, curves, panel, show_legend=False):
    x = np.arange(1, curves['beats'] + 1)
    mask = curves['mask']
    human = curves['human']
    ax.plot(x, np.where(mask, curves['score'], np.nan), color='black', lw=1.25, label='Score MIDI')
    if panel == 'Human':
        colors = plt.get_cmap('Blues')(np.linspace(.4, .9, len(human)))
        for i, row in enumerate(human):
            ax.plot(x, np.where(mask, row, np.nan), color=colors[i], lw=.8, alpha=.78,
                    label=f'Human {i + 1}')
        title_corr = curve_corr(curves['score'], human, mask)
    else:
        generated = curves['models']['pt_base' if panel == 'PT-1' else 'new_A2']
        color = '#e45756' if panel == 'PT-1' else '#7a5195'
        for i, row in enumerate(generated):
            ax.plot(x, np.where(mask, row, np.nan), color=color, lw=.85, alpha=.85,
                    label=f'{panel} sample {i + 1}')
        title_corr = float(np.mean([curve_corr(row, human, mask) for row in generated]))
    ax.set_title(f'{panel} | curve-vs-Human r={fmt_corr(title_corr)}', fontsize=9)
    ax.set_xlim(.5, curves['beats'] + .5)
    if curves['limit'] is not None:
        ax.set_ylim(*curves['limit'])
    ax.set_xlabel('Score beat index', fontsize=8)
    ax.set_ylabel(curves['ylabel'], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.18)
    if show_legend:
        ax.legend(loc='upper right', fontsize=6, ncol=2, framealpha=.9)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    references, _ = load_current_references()
    masks = np.load(ROOT / 'analysis/exp1117_beat_median/result_analysis/five_main_metrics/common_masks.npz')
    baseline._REFERENCES = references
    pt = next(method for method in baseline.discover_methods() if method['id'] == 'pt_base')
    metrics = json.loads((ROOT / 'analysis/exp1117_beat_median/result_analysis/five_main_metrics/metrics.json').read_text())
    a2_row = next(row for row in metrics['results'] if row['id'] == 'new_A2')
    a2 = {key: a2_row[key] for key in ('id', 'family', 'label', 'mask', 'representation', 'source')}
    extracted = {method['id']: baseline.extract_method(method) for method in (pt, a2)}
    features = {key: value['features'] for key, value in extracted.items()}
    all_data = {'selection_indices': list(SELECTED), 'works': {}}

    for attribute in ('velocity', 'duration'):
        overview, axes = plt.subplots(5, 3, figsize=(20, 15), constrained_layout=True, sharey=False)
        for row_index, work_index in enumerate(SELECTED):
            curves = make_curves(references[work_index], work_index, features, masks, attribute)
            work = references[work_index]['work']
            all_data['works'].setdefault(work, {})[attribute] = dict(
                score=curves['score'].tolist(), human=curves['human'].tolist(),
                models={key: value.tolist() for key, value in curves['models'].items()},
                mask=curves['mask'].tolist(), ylabel=curves['ylabel'])
            for column, panel in enumerate(('Human', 'PT-1', 'A2')):
                standalone = OUTPUT / f'{work_index:02d}_{slug(work.split("/")[-1])}_{attribute}_{panel.lower().replace("-", "")}.png'
                fig, ax = plt.subplots(figsize=(15, 4.5), constrained_layout=True)
                draw_panel(ax, curves, panel, show_legend=True)
                fig.savefig(standalone, dpi=170)
                plt.close(fig)
                draw_panel(axes[row_index, column], curves, panel, show_legend=(row_index == 0))
                axes[row_index, column].set_ylabel(f'{work.split("/")[-1][:28]}\n{curves["ylabel"]}')
        overview.suptitle(f'Beat {attribute} curves on five test works\nBlack = Score MIDI; Human = individual performances; model panels = three samples', fontsize=14)
        overview.savefig(OUTPUT / f'overview_5works_3methods_{attribute}.png', dpi=180)
        plt.close(overview)

    (OUTPUT / 'curves.json').write_text(json.dumps(all_data, indent=2, ensure_ascii=False) + '\n')
    (OUTPUT / 'README.md').write_text(
        '# Beat velocity and beat duration curves\n\n'
        'Each panel plots individual curves only. Human has one line per performance; PT-1 and A2 each have three generated samples. Score MIDI is always the black reference line. Beat duration is plotted as natural-log milliseconds, matching the tempo-curve coordinate.\n\n'
        '[Velocity overview](overview_5works_3methods_velocity.png) · '
        '[Duration overview](overview_5works_3methods_duration.png) · [Raw curves](curves.json)\n')
    print(OUTPUT / 'overview_5works_3methods_velocity.png')
    print(OUTPUT / 'overview_5works_3methods_duration.png')


if __name__ == '__main__':
    main()
