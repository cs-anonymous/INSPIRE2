#!/usr/bin/env python3
"""Audit measure-relative timing on the frozen exp1122 ASAP cohort."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_process.beat_mapping import map_score_beats, valid_beat_intervals
from src.data_process.build_dinr_beat_sidecars import beat_grid, load_beat_mapper, score_notes
from src.data_process.dinr_beat_dataset import BeatTokenSpec

DATA = ROOT.parent / 'data'
FROZEN = DATA / 'ASAP_processed_dinr_beat_structural_earliest_e512_v1'
OUTPUT = ROOT / 'analysis/design1123_measure'
EXP1122 = ROOT / 'analysis/exp1122_structural_beat/statistics.json'
SPLITS = ('train', 'valid', 'test')


def describe(values):
    data = np.asarray(values)
    data = data[np.isfinite(data)]
    quantiles = (.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
    return dict(count=int(data.size), mean=float(data.mean()), std=float(data.std()),
                minimum=float(data.min()), maximum=float(data.max()),
                percentiles={str(p): float(value) for p, value in zip(quantiles, np.percentile(data, quantiles))})


def measure_edges(midi, get_bars):
    bars, _ = get_bars(midi=midi)
    last = midi.time_signature_changes[-1]
    step = midi.ticks_per_beat * 4 * last.numerator // last.denominator
    edges = np.append(bars, bars[-1] + step).astype(np.int64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError('Non-increasing measure edges')
    return edges


def score_time_ms(midi, mapping, ticks):
    last_tick = len(mapping) - 1
    tail_rate = 60000 / (midi.tempo_changes[-1].tempo * midi.ticks_per_beat)
    return mapping[np.minimum(ticks, last_tick)] + np.maximum(ticks - last_tick, 0) * tail_rate


def grid(edges, times, tick_times, ticks):
    ids = np.searchsorted(edges, ticks, side='right') - 1
    if ids.min() < 0 or ids.max() >= len(edges) - 1:
        raise ValueError('Score note is outside the grid')
    score_ms = np.diff(times)
    if np.any(score_ms <= 0):
        raise ValueError('Non-positive score duration')
    counts = np.bincount(ids, minlength=len(score_ms))
    ratio = (ticks - edges[ids]) / np.diff(edges)[ids]
    time_ratio = (tick_times - times[ids]) / score_ms[ids]
    return dict(edges=edges, ids=ids, counts=counts, score_ms=score_ms, ratio=ratio, time_ratio=time_ratio)


def anchor_geometry(edges, onsets):
    unique = np.unique(onsets)
    inside = edges[(edges >= unique[0]) & (edges <= unique[-1])]
    right = np.searchsorted(unique, inside)
    exact = unique[right] == inside
    gaps = unique[right[~exact]] - unique[right[~exact] - 1]
    return len(inside), int(exact.sum()), gaps


def observe(g, anchors, support, onsets):
    valid, durations = valid_beat_intervals(anchors, support)
    duration_log = np.full(len(valid), np.nan)
    duration_log[valid] = np.log(durations[valid] / g['score_ms'][valid])
    local_ms = durations[g['ids']]
    delta_ms = onsets - anchors[g['ids']]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(valid[g['ids']], delta_ms / local_ms, np.nan)
    dev = ratio - g['ratio']
    return dict(valid=valid, duration=durations, duration_log=duration_log, ratio=ratio, dev=dev,
                dev_ms=dev * local_ms, time_dev_ms=(ratio - g['time_ratio']) * local_ms,
                local_ms=local_ms, adjacent_step=np.abs(np.diff(duration_log)))


def mean_velocity(velocity, measure):
    counts = measure['counts']
    sums = np.bincount(measure['ids'], weights=velocity, minlength=len(counts))
    return np.divide(sums, counts, out=np.full(len(counts), np.nan), where=counts > 0)


def add_observation(stats, prefix, g, observed, anchors, old_anchors, supported, scale):
    valid = observed['valid']
    stats[prefix + '_anchor_shift_raw_ms'].append((anchors[supported] - old_anchors[supported]) / scale)
    stats[prefix + '_score_duration_ms'].append(g['score_ms'][valid])
    stats[prefix + '_duration_ms'].append(observed['duration'][valid])
    stats[prefix + '_duration_log_ratio'].append(observed['duration_log'][valid])
    stats[prefix + '_adjacent_abs_log_step'].append(observed['adjacent_step'])
    for name in ('ratio', 'dev', 'dev_ms', 'time_dev_ms'):
        stats[prefix + '_onset_' + name].append(observed[name][valid[g['ids']]])


def process_work(path, rows, payload, get_bars, stats, show_curve):
    work = torch.load(path, weights_only=False, map_location='cpu')
    midi = MidiFile(str(DATA / 'ASAP_processed' / work['meta']['score_source']))
    notes = score_notes(midi)
    ticks = np.asarray([note.start for note in notes], dtype=np.int64)
    group_ticks, group_starts = np.unique(ticks, return_index=True)
    if [note.pitch for note in notes] != list(work['score']['pitch']):
        raise ValueError(f'{path}: MIDI and sidecar pitch order differ')
    by_source = {item['performance_source']: item for item in work['performances']}
    if set(by_source) != {row['source'] for row in rows}:
        raise ValueError(f'{path}: performance cohort changed')

    bar_edges = measure_edges(midi, get_bars)
    beat_edges, _, beat_score_ms, _, _ = beat_grid(midi, get_bars)
    mapping = np.asarray(midi.get_tick_to_time_mapping(), dtype=np.float64) * 1000
    tick_times = score_time_ms(midi, mapping, ticks)
    bar = grid(bar_edges, score_time_ms(midi, mapping, bar_edges), tick_times, ticks)
    beat = grid(beat_edges, score_time_ms(midi, mapping, beat_edges), tick_times, ticks)
    beat['score_ms'] = beat_score_ms
    edge_union = np.union1d(bar_edges, beat_edges)
    indices = {name: np.searchsorted(edge_union, g['edges']) for name, g in [('bar', bar), ('beat', beat)]}
    q_per_bar = np.diff(bar_edges) / midi.ticks_per_beat
    score_values = np.asarray(work['score']['score_raw'], dtype=float)
    score_velocity = np.clip(np.rint(score_values[:, 2]), 0, 127)
    score_onset_ms = bar['time_ratio'] * bar['score_ms'][bar['ids']]
    timing_change_notes = np.abs(bar['ratio'] - bar['time_ratio']) > 1e-6
    bar_lengths = np.diff(bar_edges)
    exact, supported, gaps = {}, {}, {}
    for name, edges in [('bar', bar_edges), ('beat', beat_edges)]:
        supported[name], exact[name], gaps[name] = anchor_geometry(edges, ticks)
        stats[name + '_interpolation_gap_quarters'].append(gaps[name] / midi.ticks_per_beat)
    stats['notes_per_bar'].append(bar['counts'])
    stats['quarter_notes_per_bar'].append(q_per_bar)
    stats['score_onset_ratio'].append(bar['ratio'])
    containing = np.searchsorted(bar_edges, beat_edges[:-1], side='right') - 1
    safe_indices = np.clip(containing, 0, len(bar_lengths) - 1)
    contained = ((containing >= 0) & (containing < len(bar_lengths))
                 & (beat_edges[1:] <= bar_edges[safe_indices + 1]))
    summary = dict(work=rows[0]['work'], score_source=work['meta']['score_source'], performances=len(rows),
                   score_notes=len(ticks), beats=len(beat_edges) - 1, bars=len(bar_lengths),
                   measures_without_onsets=int((bar['counts'] == 0).sum()),
                   bar_starts_in_beat_grid=int(np.isin(bar_edges[:-1], beat_edges[:-1]).sum()),
                   score_anchor_geometry={name: dict(supported=supported[name], exact=exact[name]) for name in exact},
                   note_instances=len(ticks) * len(rows), possible_beat_intervals=(len(beat_edges) - 1) * len(rows),
                   possible_measure_intervals=len(bar_lengths) * len(rows), beat_events=0, measure_events=0,
                   beat_windows=0, measure_windows=0, valid_beats=0, valid_measures=0,
                   beat_valid_notes=0, measure_valid_notes=0, common_valid_notes=0,
                   score_tempo_changes_inside_measure=sum(change.time not in bar_edges for change in midi.tempo_changes),
                   notes_with_different_score_time_fraction=int(timing_change_notes.sum()))
    per_human = {name: [] for name in ('beat', 'bar')}
    curves = dict(work=summary['work'], score_bpm=60000 * q_per_bar / bar['score_ms'],
                  score_velocity=mean_velocity(score_velocity, bar), human_bpm=[], human_velocity=[])
    for row in rows:
        raw = np.asarray(by_source[row['source']]['label_shared_raw'], dtype=float)
        if raw.shape != (len(ticks), 3) or not np.isfinite(raw).all():
            raise ValueError(f'{path}: invalid performance labels')
        onsets = np.cumsum(raw[:, 0])
        # Collapse sorted note onsets once per group before using the existing
        # interpolation. A singleton group has the same mean and minimum.
        first_onset = np.minimum.reduceat(onsets, group_starts)
        anchors, supported_union = map_score_beats(edge_union, group_ticks, first_onset)
        old_anchors, old_support = map_score_beats(edge_union, ticks, onsets)
        if not np.array_equal(supported_union, old_support):
            raise ValueError(f'{path}: anchor supports differ by rule')
        scale = float(row['global_timing_scale']['scale'])
        both = {}
        for name, g in [('beat', beat), ('bar', bar)]:
            edge_ids = indices[name]
            a, s, old = anchors[edge_ids], supported_union[edge_ids], old_anchors[edge_ids]
            both[name] = observe(g, a, s, onsets)
            add_observation(stats, name, g, both[name], a, old, s, scale)
            per_human[name].append(both[name]['dev_ms'])
        observed_beat, observed_bar = both['beat'], both['bar']
        start, end = row['span']
        event_type = payload['events'][start:end, 0].numpy()
        frozen_masks = payload['target_valid'][start:end].numpy()
        np.testing.assert_array_equal(observed_beat['valid'], frozen_masks[event_type == 0, 0])
        np.testing.assert_array_equal(observed_beat['valid'][beat['ids']], frozen_masks[event_type == 1, 1])
        frozen_durations = payload['targets'][start:end, 0].numpy()[event_type == 0]
        np.testing.assert_allclose(np.maximum(observed_beat['duration'][observed_beat['valid']], 1),
                                   frozen_durations[observed_beat['valid']], rtol=1e-6, atol=1e-3)
        valid_notes = observed_beat['valid'][beat['ids']]
        note_anchors = anchors[indices['beat']][beat['ids']]
        frozen_onset = payload['targets'][start:end, 1].numpy()[event_type == 1]
        replay_onset = ((onsets[valid_notes] - note_anchors[valid_notes]) /
                        np.maximum(observed_beat['local_ms'][valid_notes], 1) - beat['ratio'][valid_notes])
        np.testing.assert_allclose(replay_onset, frozen_onset[valid_notes], rtol=1e-6, atol=1e-6)
        if payload['split'] == 'test':
            bar_valid_notes = observed_bar['valid'][bar['ids']]
            note_duration_valid = frozen_masks[event_type == 1, 0]
            stats['note_score_duration_ms'].append(score_values[note_duration_valid, 1])
            stats['note_duration_ms'].append(raw[note_duration_valid, 1])
            stats['bar_score_onset_ms'].append(score_onset_ms[bar_valid_notes])
            stats['bar_onset_ms'].append((observed_bar['ratio'] * observed_bar['local_ms'])[bar_valid_notes])
            stats['bar_score_onset_percent'].append(100 * bar['time_ratio'][bar_valid_notes])
            stats['bar_onset_percent'].append(100 * observed_bar['ratio'][bar_valid_notes])
        note_common = np.isfinite(observed_beat['ratio']) & np.isfinite(observed_bar['ratio'])
        stats['common_bar_ratio'].append(observed_bar['ratio'][note_common])
        for name, observed in both.items():
            for key in ('dev', 'dev_ms', 'time_dev_ms', 'local_ms'):
                stats[f'common_{name}_{key}'].append(observed[key][note_common])
        selected = contained & observed_beat['valid'] & observed_bar['valid'][safe_indices]
        inherited = observed_bar['duration_log'][safe_indices[selected]]
        stats['within_measure_beat_abs_log_residual'].append(np.abs(observed_beat['duration_log'][selected] - inherited))
        summary['beat_events'] += end - start
        summary['measure_events'] += len(bar_lengths) + len(ticks)
        summary['beat_windows'] += int(np.ceil((end - start) / 512))
        summary['measure_windows'] += int(np.ceil((len(bar_lengths) + len(ticks)) / 512))
        summary['valid_beats'] += int(observed_beat['valid'].sum())
        summary['valid_measures'] += int(observed_bar['valid'].sum())
        summary['beat_valid_notes'] += int(np.isfinite(observed_beat['ratio']).sum())
        summary['measure_valid_notes'] += int(np.isfinite(observed_bar['ratio']).sum())
        summary['common_valid_notes'] += int(note_common.sum())
        if show_curve:
            bpm = np.full(len(bar_lengths), np.nan)
            valid = observed_bar['valid']
            bpm[valid] = 60000 * q_per_bar[valid] / observed_bar['duration'][valid]
            vel = np.clip(np.rint(raw[:, 2]), 0, 127)
            curves['human_bpm'].append(bpm)
            curves['human_velocity'].append(mean_velocity(vel, bar))
    if len(rows) > 1:
        beat_values, bar_values = (np.stack(per_human[name]) for name in ('beat', 'bar'))
        both_valid = (np.isfinite(beat_values) & np.isfinite(bar_values)).all(axis=0)
        stats['across_human_beat_onset_std_ms'].append(beat_values[:, both_valid].std(axis=0, ddof=1))
        stats['across_human_bar_onset_std_ms'].append(bar_values[:, both_valid].std(axis=0, ddof=1))
    return summary, curves


def bin_midpoints(values, bounds, bins):
    low, high = bounds
    width = (high - low) / bins
    encoded = np.clip(np.floor((values - low) / width), 0, bins - 1)
    return low + (encoded + 0.5) * width


def quantization(values, durations, bounds):
    bins = BeatTokenSpec.onset_dev_bins
    low, high = bounds
    decoded = bin_midpoints(values, bounds, bins)
    return dict(bounds=list(bounds), bins=bins,
                clipped_fraction=float(np.mean((values < low) | (values > high))),
                error_ms=describe(np.abs(decoded - values) * durations))


def absolute_onset_quantization(ratios, durations, max_ms):
    """Quantize a measure's elapsed time with one exact-zero class in each variant."""
    bins = BeatTokenSpec.onset_dev_bins
    offsets = ratios * durations
    positive = offsets > 0
    log_offsets = np.log1p(np.maximum(offsets, 0))
    linear_ms = bin_midpoints(offsets, (0, max_ms), bins - 1)
    log_ms = np.expm1(bin_midpoints(log_offsets, (0, np.log1p(max_ms)), bins - 1))
    ratio_ms = bin_midpoints(ratios, (0, 1), bins - 1) * durations
    variants = (
        ('linear_ms', linear_ms, offsets > max_ms, max_ms),
        ('log1p_ms', log_ms, offsets > max_ms, max_ms),
        ('ratio', ratio_ms, ratios > 1, None),
    )
    results = {}
    for name, decoded, above_cap, cap in variants:
        errors = np.abs(np.where(positive, decoded, 0) - offsets)
        results[name] = dict(
            tokens=bins, offset_cap_ms=cap, exact_zero_fraction=float(np.mean(offsets == 0)),
            clipped_fraction=float(np.mean((offsets < 0) | above_cap)),
            error_ms=describe(errors),
            positive_first_quarter_mae_ms=float(errors[positive & (ratios <= .25)].mean()),
            last_quarter_mae_ms=float(errors[ratios > .75].mean()),
        )
    return results


def plot_distributions(test):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    distributions = [
        ('Local duration deviation', 'ln(performance duration / score duration)',
         test['beat_duration_log_ratio'], test['bar_duration_log_ratio']),
        ('Note onset deviation', 'Fraction of the local interval',
         test['common_beat_dev'], test['common_bar_dev']),
        ('Note onset deviation', 'Onset deviation (ms)',
         test['common_beat_dev_ms'], test['common_bar_dev_ms']),
        ('Sensitivity to chord anchor rule', 'Earliest minus mean anchor (ms, before scaling)',
         test['beat_anchor_shift_raw_ms'], test['bar_anchor_shift_raw_ms']),
    ]
    for ax, (title, label, beat, bar) in zip(axes.flat, distributions, strict=True):
        low = min(np.percentile(beat, 1), np.percentile(bar, 1))
        high = max(np.percentile(beat, 99), np.percentile(bar, 99))
        edges = np.linspace(low, high, 90)
        for vals, name, color in [(beat, 'Beat', '#457498'), (bar, 'Measure', '#b45136')]:
            heights = np.histogram(vals, bins=edges)[0] / len(vals)
            ax.stairs(heights, edges, label=name, lw=1.5, color=color)
        ax.axvline(0, color='0.35', lw=.7)
        ax.set(title=title, xlabel=label, ylabel='Fraction per bin')
        ax.grid(alpha=.18)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('ASAP test: beat and measure timing (1st–99th percentile display)')
    fig.savefig(OUTPUT / 'measure_statistics.png', dpi=165)
    plt.close(fig)


def plot_timing_pair(axes, label, unit, score, perf, position_range=None, show_legend=False):
    left, right = axes
    delta = perf - score
    left.set_title(label, loc='left', fontweight='semibold')
    right.set_title('Paired deviation', loc='left', fontweight='semibold')
    panels = (
        (left, [('Score', score, '#303030'), ('Performance', perf, '#467ca8')], unit),
        (right, [('Deviation', delta, '#a44d34')],
         'Performance − score (percentage points)' if position_range else f'Performance − score ({unit})'),
    )
    summary = {}
    for ax, traces, xlabel in panels:
        limits = [np.percentile(values, [.5, 99.5]) for _, values, _ in traces]
        low = min(interval[0] for interval in limits)
        high = max(interval[1] for interval in limits)
        if ax is right:
            low, high = min(low, 0), max(high, 0)
            ax.axvline(0, color='0.4', lw=.8, ls='--')
        elif position_range:
            low, high = position_range
        edges = np.linspace(low, high, 81)
        omitted = []
        for name, values, color in traces:
            counts = np.histogram(values, bins=edges)[0]
            ax.stairs(100 * counts / len(values), edges, label=name, color=color, lw=1.5)
            outside = len(values) - int(counts.sum())
            summary[name.lower()] = dict(
                **describe(values), displayed_range=[float(low), float(high)],
                outside_display_fraction=outside / len(values), outside_display_count=outside)
            outside_label = f'n={outside:,}' if ax is left and position_range else f'{outside / len(values):.2%}'
            omitted.append(f'{name} {outside_label}')
        ax.text(.99, .96, 'Outside view  ' + '  ·  '.join(omitted), transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='0.36')
        ax.set(xlabel=xlabel, ylabel='Samples per bin (%)', xlim=(low, high))
        ax.grid(axis='y', alpha=.18)
    if show_legend:
        left.legend(frameon=False, loc='upper right', bbox_to_anchor=(1, .87))
    return summary


def plot_absolute_timing(test, totals):
    variables = (
        ('measure_duration', 'Measure duration', test['bar_score_duration_ms'], test['bar_duration_ms'],
         totals['valid_measures']),
        ('note_duration', 'Note duration', test['note_score_duration_ms'], test['note_duration_ms'],
         totals['note_instances']),
        ('measure_onset', 'Onset from measure start', test['bar_score_onset_ms'], test['bar_onset_ms'],
         totals['measure_valid_notes']),
    )
    fig, axes = plt.subplots(6, 2, figsize=(13, 16), layout='constrained')
    summary = {}
    for row, (key, label, score_ms, perf_ms, expected_count) in enumerate(variables):
        if score_ms.shape != perf_ms.shape or len(score_ms) != expected_count:
            raise ValueError(f'{key}: score and performance pairs differ')
        if not (np.isfinite(score_ms).all() and np.isfinite(perf_ms).all()):
            raise ValueError(f'{key}: non-finite timing')
        if key == 'measure_onset':
            if np.any(score_ms < 0) or np.any(perf_ms < 0):
                raise ValueError('Expected nonnegative onset intervals for log(max(1, ms))')
            score_log = np.log(np.maximum(1, score_ms))
            perf_log = np.log(np.maximum(1, perf_ms))
            log_unit = 'ln(max(1, ms))'
        else:
            if np.any(score_ms <= 0) or np.any(perf_ms <= 0):
                raise ValueError(f'{key}: non-positive duration before logarithm')
            score_log = np.log(score_ms)
            perf_log = np.log(perf_ms)
            log_unit = 'ln(ms)'
        summary[key] = {}
        for scale, (identifier, unit, score, perf) in enumerate((
            ('ms', 'ms', score_ms, perf_ms), ('log_ms', log_unit, score_log, perf_log),
        )):
            summary[key][identifier] = plot_timing_pair(
                axes[2 * row + scale], f'{label} · {unit}', unit, score, perf, show_legend=row == scale == 0)
    fig.suptitle('ASAP test · Paired measure and note timing', fontsize=16, fontweight='semibold')
    fig.savefig(OUTPUT / 'timing_ms_log_distributions.png', dpi=170)
    plt.close(fig)
    return summary


def plot_onset_percent(test, expected_count):
    score = test['bar_score_onset_percent']
    perf = test['bar_onset_percent']
    if score.shape != perf.shape or len(score) != expected_count:
        raise ValueError('Measure onset percentages do not cover the paired onset sample')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    summary = plot_timing_pair(axes, 'Score and performance', 'Position within measure (%)', score, perf,
                               position_range=(0, 100), show_legend=True)
    fig.suptitle('ASAP test · Onset position within the measure', fontsize=14, fontweight='semibold')
    fig.savefig(OUTPUT / 'measure_onset_percent.png', dpi=170)
    plt.close(fig)
    return summary


def display_title(work):
    author, *pieces = work.split('/')
    return author.split(',')[0] + ' | ' + ' · '.join(piece.replace('_', ' ') for piece in pieces)


def plot_curves(rows, name, y_label, color):
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 2.55 * len(rows)), layout='constrained')
    for ax, row in zip(axes, rows, strict=True):
        score = row['score_' + name]
        humans = np.stack(row['human_' + name])
        indices = np.arange(1, len(score) + 1)
        has_human = np.isfinite(humans).any(axis=0)
        median = np.full(len(score), np.nan)
        median[has_human] = np.nanmedian(humans[:, has_human], axis=0)
        ax.plot(indices, score, color='#242424', linewidth=1.05, label='Score')
        for index, human in enumerate(humans):
            ax.plot(indices, human, color=color, alpha=.28, linewidth=.75,
                    label='Individual performances' if index == 0 else None)
        ax.plot(indices, median, color=color, linewidth=1.45, label='Human median')
        ax.set(title=f"{display_title(row['work'])} ({len(humans)} performances)", ylabel=y_label,
               xlim=(1, len(score)))
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False, ncols=3, loc='best')
    axes[-1].set_xlabel('Score measure index')
    fig.savefig(OUTPUT / ('measure_tempo_curves.png' if name == 'bpm' else 'measure_velocity_curves.png'), dpi=170)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with EXP1122.open() as source:
        frozen_stats = json.load(source)
    selected_works = frozen_stats['selected_works']
    get_bars, _ = load_beat_mapper()
    result = dict(reference='exp1122 frozen ASAP structural-earliest cohort and beat-normalized performances',
                  selected_works=selected_works, splits={})
    plots = {}
    for split in SPLITS:
        path = FROZEN / f'dinr_beat_{split}_e512_v1.pt'
        payload = torch.load(path, weights_only=False, map_location='cpu')
        groups = defaultdict(list)
        for row in payload['performances']:
            groups[row['source_sidecar']].append(row)
        stats, works = defaultdict(list), []
        for index, (path, rows) in enumerate(sorted(groups.items()), start=1):
            show_curve = split == 'test' and rows[0]['work'] in selected_works
            summary, curves = process_work(path, rows, payload, get_bars, stats, show_curve)
            works.append(summary)
            if show_curve:
                plots[summary['work']] = curves
            if index % 50 == 0:
                print(f'{split}: {index}/{len(groups)} works', flush=True)
        stats = {name: np.concatenate(parts) for name, parts in stats.items()}
        totals = {key: sum(row[key] for row in works) for key, value in works[0].items() if type(value) is int}
        totals['works'] = len(works)
        totals['works_with_multiple_performances'] = sum(row['performances'] > 1 for row in works)
        totals['anchor_geometry'] = {name: {key: sum(row['score_anchor_geometry'][name][key] for row in works)
                                           for key in ('supported', 'exact')} for name in ('beat', 'bar')}
        expected = frozen_stats['split_audit'][split]
        assert (totals['works'], totals['performances'], totals['note_instances'], totals['possible_beat_intervals'],
                totals['beat_windows']) == tuple(expected[key] for key in ('works', 'performances', 'notes', 'beats', 'windows'))
        result['splits'][split] = dict(totals=totals, works=works, distributions={
            name: describe(values) for name, values in stats.items() if np.isfinite(values).any()
        })
        by_scale = result['splits'][split]
        by_scale['onset_range_fractions'] = {}
        for name in ('beat', 'bar'):
            ratios = stats[f'{name}_onset_ratio']
            by_scale['onset_range_fractions'][name] = dict(
                below_zero=float(np.mean(ratios < 0)), at_zero=float(np.mean(ratios == 0)),
                above_one=float(np.mean(ratios > 1)))
            for key in ('dev_ms', 'time_dev_ms'):
                by_scale['distributions'][f'common_{name}_abs_{key}'] = describe(np.abs(stats[f'common_{name}_{key}']))
        by_scale['measure_duration_range_fractions'] = dict(
            score_log_above_8=float(np.mean(stats['bar_score_duration_ms'] > np.exp(8))),
            perf_log_above_8=float(np.mean(stats['bar_duration_ms'] > np.exp(8))),
            log_dev_outside_current=float(np.mean(
                (stats['bar_duration_log_ratio'] < BeatTokenSpec.duration_dev_range[0]) |
                (stats['bar_duration_log_ratio'] > BeatTokenSpec.duration_dev_range[1]))))
        if split == 'train':
            q = np.percentile(stats['bar_onset_dev'], [.1, 99.9])
            candidate = (np.floor(q[0] * 100) / 100, np.ceil(q[1] * 100) / 100)
            train_offsets = stats['common_bar_ratio'] * stats['common_bar_local_ms']
            interval_cap_ms = float(np.ceil(np.percentile(train_offsets, 99.9) / 100) * 100)
        by_scale['onset_quantization_on_common_notes'] = dict(
            beat=quantization(stats['common_beat_dev'], stats['common_beat_local_ms'], BeatTokenSpec.onset_dev_range),
            measure_same_range=quantization(stats['common_bar_dev'], stats['common_bar_local_ms'], BeatTokenSpec.onset_dev_range),
            measure_train_range=quantization(stats['common_bar_dev'], stats['common_bar_local_ms'], candidate),
        )
        by_scale['absolute_measure_onset_quantization'] = absolute_onset_quantization(
            stats['common_bar_ratio'], stats['common_bar_local_ms'], interval_cap_ms)
        print(split, json.dumps(totals, ensure_ascii=False), flush=True)
        if split == 'test':
            plot_distributions(stats)
            by_scale['absolute_timing_ms_log'] = plot_absolute_timing(stats, totals)
            by_scale['measure_onset_percent'] = plot_onset_percent(stats, totals['measure_valid_notes'])
    ordered = [plots[work] for work in selected_works]
    plot_curves(ordered, 'bpm', 'Quarter-note BPM', '#457498')
    plot_curves(ordered, 'velocity', 'Mean MIDI velocity', '#b45136')
    (OUTPUT / 'statistics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
