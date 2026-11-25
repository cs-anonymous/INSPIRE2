#!/usr/bin/env python3
"""Plot the three completed chord runs on a shared optimizer-step axis."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(__file__).resolve().parents[1] / 'results/exp1125_chord'
MODELS = {'abs5000': ('abs 5000', '#1965ac'), 'log_abs': ('log abs', '#c95b17'), 'log_dev': ('log dev', '#00846e')}
SMOOTH = 20


def main():
    fig, (ax_train, ax_valid) = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    fig.subplots_adjust(left=.075, right=.98, bottom=.22, top=.76, wspace=.26)
    fig.text(.075, .94, 'Chord training and validation loss', fontsize=19, weight='bold', color='#172334')
    for mode, (label, color) in MODELS.items():
        path = RUN / mode
        history = [json.loads(row) for row in (path / 'loss_history.jsonl').read_text().splitlines()]
        completion = json.loads((path / 'training_complete.json').read_text())
        train = [row for row in history if 'train_loss' in row]
        valid = [row for row in history if 'valid_loss' in row]
        best = min(valid, key=lambda row: row['valid_loss'])
        assert (best['step'], best['valid_loss']) == (completion['best_step'], completion['best_valid_loss'])
        train_x = [row['step'] for row in train]
        train_y = [row['train_loss'] for row in train]
        ax_train.plot(train_x, train_y, color=color, alpha=.13, linewidth=.75)
        moving = np.convolve(train_y, np.repeat(1 / SMOOTH, SMOOTH), mode='valid')
        ax_train.plot(train_x[SMOOTH - 1:], moving, label=label, color=color, linewidth=1.7)
        ax_valid.plot([row['step'] for row in valid], [row['valid_loss'] for row in valid],
                      color=color, linewidth=1.6, marker='.', markersize=3)
        ax_valid.scatter(best['step'], best['valid_loss'], color=color, edgecolor='white', s=44, zorder=4)
        ax_valid.annotate(f"Step {best['step']}", (best['step'], best['valid_loss']), xytext=(3, 8),
                          textcoords='offset points', color=color, fontsize=8)
    ax_train.set(title='Train', ylabel='Sum of head CE')
    ax_valid.set(title='Valid', ylabel='Sum of head CE')
    for ax in (ax_train, ax_valid):
        ax.set(xlabel='Optimizer step', xlim=(0, 1140))
        ax.grid(axis='y', alpha=.25)
        for end in (285, 570, 855):
            ax.axvline(end, color='#8592a0', alpha=.32, linestyle='--', linewidth=.8)
        for edge in ('top', 'right'):
            ax.spines[edge].set_visible(False)
    fig.legend(*ax_train.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.072, .89),
               ncols=3, frameon=False)
    fig.text(.075, .064, 'Solid train lines show a trailing 20-step mean. Labeled valid points mark the selected checkpoints.',
             fontsize=9, color='#46566b')
    fig.savefig(RUN / 'train_valid_loss.png', dpi=185)
    plt.close(fig)


if __name__ == '__main__':
    main()
