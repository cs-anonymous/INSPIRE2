#!/usr/bin/env python3
"""Build beat sidecars using the earliest performance onset at structural beats."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
from miditoolkit import MidiFile

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_process.beat_mapping import (
    map_score_beats, map_score_beats_structural_earliest, valid_beat_intervals,
)
from src.data_process.beat_normalization import normalize_performance
from src.data_process.build_dinr_beat_sidecars import build, beat_grid, load_beat_mapper, score_notes
from script.prepare_exp1116_beat import prepare
from script.audit_dinr_beat import audit

PROJECT = ROOT.parent
SOURCE = PROJECT / 'data/ASAP_processed'
OLD_WORKS = PROJECT / 'data/ASAP_processed_beat_median_v1'
WORKS = PROJECT / 'data/ASAP_processed_beat_structural_earliest_v1'
EVENTS = PROJECT / 'data/ASAP_processed_dinr_beat_structural_earliest_v1'
FIXED = PROJECT / 'data/ASAP_processed_dinr_beat_structural_earliest_e512_v1'
OLD_FIXED = PROJECT / 'data/ASAP_processed_dinr_beat_median_e512_v1'
OUTPUT = ROOT / 'analysis/exp1122_structural_beat'
METHOD = 'beat_structural_earliest_median_log_ratio'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0)
    q = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])
    return dict(count=len(values), mean=float(values.mean()), std=float(values.std()),
                percentiles=dict(zip(('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'), q.tolist())))


def compare_anchor_protocol(score_ticks, beat_edges, perf_onsets):
    old_a, old_s = map_score_beats(beat_edges, score_ticks, perf_onsets)
    new_a, new_s = map_score_beats_structural_earliest(beat_edges, score_ticks, perf_onsets)
    common = old_s & new_s & np.isfinite(old_a) & np.isfinite(new_a)
    old_v, old_d = valid_beat_intervals(old_a, old_s)
    new_v, new_d = valid_beat_intervals(new_a, new_s)
    common_i = old_v & new_v
    return dict(old_anchors=old_a, new_anchors=new_a,
                anchor_diff=new_a[common] - old_a[common],
                duration_log_ratio=np.log(new_d[common_i] / old_d[common_i]),
                old_valid=int(old_v.sum()), new_valid=int(new_v.sum()),
                common_valid=int(common_i.sum()), common_anchors=int(common.sum()),
                total_beats=len(old_v))


def plot_statistics(stats, curve_rows):
    local_cjk_font = Path('/usr/share/fonts/truetype/winfonts/msyhbd.ttc')
    if local_cjk_font.is_file():
        font_manager.fontManager.addfont(local_cjk_font)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    anchor = np.asarray(stats['anchor_shift_ms']['values'])
    duration = np.asarray(stats['beat_duration_log_ratio']['values'])
    anchor_lo, anchor_hi = np.percentile(anchor, [1, 99])
    duration_lo, duration_hi = np.percentile(duration, [1, 99])
    axes[0].hist(anchor, bins=100, range=(anchor_lo, anchor_hi), color='#4776a8', alpha=.82)
    axes[0].axvline(0, color='black', lw=1)
    axes[0].axvline(np.median(anchor), color='#e45756', ls='--', label=f'中位数 {np.median(anchor):.1f} ms')
    axes[0].set_xlim(anchor_lo, anchor_hi)
    axes[0].set(title='锚点位移：最早起音减同组均值', xlabel='锚点位移（ms；第 1 至第 99 百分位）', ylabel='数量')
    anchor_outside = np.mean((anchor < anchor_lo) | (anchor > anchor_hi))
    axes[0].text(.02, .96, f'完整范围 [{anchor.min():.1f}, {anchor.max():.1f}] ms\n视野外样本：{anchor_outside:.1%}',
                 transform=axes[0].transAxes, va='top', fontsize=8)
    axes[0].legend(frameon=False)
    axes[1].hist(duration, bins=100, range=(duration_lo, duration_hi), color='#65a765', alpha=.82)
    axes[1].axvline(0, color='black', lw=1)
    axes[1].axvline(np.median(duration), color='#e45756', ls='--',
                    label=f'中位数 {np.median(duration):.3f}')
    axes[1].set_xlim(duration_lo, duration_hi)
    axes[1].set(title='共同有效拍的拍长变化',
                xlabel='ln(新拍长 / 旧拍长)（第 1 至第 99 百分位）', ylabel='数量')
    duration_outside = np.mean((duration < duration_lo) | (duration > duration_hi))
    axes[1].text(.02, .96, f'完整范围 [{duration.min():.2f}, {duration.max():.2f}]\n视野外样本：{duration_outside:.1%}',
                 transform=axes[1].transAxes, va='top', fontsize=8)
    axes[1].legend(frameon=False)
    labels = [r['work'].split('/')[-1][:27] for r in curve_rows]
    vals = [r['common_valid_fraction'] for r in curve_rows]
    axes[2].barh(np.arange(len(vals)), vals, color='#f0a35e')
    axes[2].set(yticks=np.arange(len(vals)), yticklabels=labels, xlim=(0, 1.02),
                title='结构性规则的 beat 覆盖率（5 首作品）', xlabel='共同有效拍比例')
    axes[2].invert_yaxis()
    fig.savefig(OUTPUT / 'anchor_and_duration_statistics.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(len(curve_rows), 2, figsize=(16, 3.2 * len(curve_rows)), squeeze=False,
                             constrained_layout=True)
    for i, row in enumerate(curve_rows):
        x = np.arange(len(row['score']))
        score_log = np.log(np.maximum(row['score'], 1))
        visible = [score_log]
        axes[i, 0].plot(x, score_log, color='black', lw=1.2, label='Score')
        axes[i, 1].plot(x, score_log, color='black', lw=1.2, label='Score')
        for j, vals in enumerate(row['old_human']):
            curve = np.where(row['old_mask'][j], np.log(np.maximum(vals, 1)), np.nan)
            visible.append(curve[np.isfinite(curve)])
            axes[i, 0].plot(x, curve,
                            color='#4776a8', alpha=.24, lw=.65)
        for j, vals in enumerate(row['new_human']):
            curve = np.where(row['new_mask'][j], np.log(np.maximum(vals, 1)), np.nan)
            visible.append(curve[np.isfinite(curve)])
            axes[i, 1].plot(x, curve,
                            color='#e45756', alpha=.24, lw=.65)
        visible = np.concatenate([v for v in visible if len(v)])
        low, high = np.percentile(visible, [1, 99])
        axes[i, 0].set_title(f"{row['work'].split('/')[-1]} — onset group 均值")
        axes[i, 1].set_title(f"{row['work'].split('/')[-1]} — 结构性最早起音")
        for ax in axes[i]:
            ax.set_ylim(low, high)
            ax.set_ylabel('ln(beat duration / ms)')
            ax.set_xlabel('Score beat 序号')
            ax.grid(alpha=.2)
        axes[i, 0].text(.01, .96, f'纵轴显示 1%–99%：[{low:.2f}, {high:.2f}]',
                        transform=axes[i, 0].transAxes, va='top', fontsize=8)
    fig.suptitle('两种 beat 锚点规则下的人类 beat duration 曲线；黑线 = Score')
    fig.savefig(OUTPUT / 'five_works_duration_curve_comparison.png', dpi=170)
    plt.close(fig)


def load_curve_comparison():
    old = torch.load(OLD_FIXED / 'dinr_beat_test_e512_v1.pt', map_location='cpu', weights_only=False)
    new = torch.load(FIXED / 'dinr_beat_test_e512_v1.pt', map_location='cpu', weights_only=False)
    def grouped(data):
        result = {}
        for row in data['performances']:
            a, b = row['span']
            event = data['events'][a:b]
            beat = event[:, 0].eq(0)
            target = data['targets'][a:b, 0]
            valid = data['target_valid'][a:b, 0] & beat
            result.setdefault(row['work'], []).append((event[beat, 2].numpy(), target[beat].numpy(), valid[beat].numpy()))
        return result
    old_g, new_g = grouped(old), grouped(new)
    works = sorted(set(old_g) & set(new_g))
    # Use the same fixed five test examples as the previous curve audit indices.
    selected = [works[i] for i in (0, 3, 5, 9, 13)]
    rows = []
    for work in selected:
        score = old_g[work][0][0]
        if any(not np.array_equal(score, item[0]) for item in old_g[work] + new_g[work]):
            raise ValueError(f'Score beat topology changed for {work}')
        old_h = np.stack([item[1] for item in old_g[work]])
        new_h = np.stack([item[1] for item in new_g[work]])
        old_m = np.stack([item[2] for item in old_g[work]])
        new_m = np.stack([item[2] for item in new_g[work]])
        rows.append(dict(work=work, score=score, old_human=old_h, new_human=new_h,
                         old_mask=old_m, new_mask=new_m,
                         common_valid_fraction=float((old_m & new_m).sum() / old_m.size)))
    return rows


def collect_plot_statistics(old_paths):
    get_beats, _ = load_beat_mapper()
    anchor_shifts, duration_changes = [], []
    for index, path in enumerate(sorted(old_paths)):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        source_root = PROJECT / 'data/ASAP_processed'
        score_path = source_root / payload['meta']['score_source']
        midi = MidiFile(str(score_path))
        ticks = np.asarray([note.start for note in score_notes(midi)], dtype=np.int64)
        edges, _, _, _, _ = beat_grid(midi, get_beats)
        for performance in payload['performances']:
            values = np.asarray(performance['label_shared_raw'], dtype=float)
            prior = performance.get('global_timing_scale')
            if prior:
                values = values.copy()
                values[:, :2] /= float(prior['scale'])
            result = compare_anchor_protocol(ticks, edges, np.cumsum(values[:, 0]))
            anchor_shifts.extend(result['anchor_diff'].tolist())
            duration_changes.extend(result['duration_log_ratio'].tolist())
        if index % 50 == 0:
            print(f'collected plot statistics for {index+1}/{len(old_paths)} works', flush=True)
    return anchor_shifts, duration_changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only', action='store_true', help='Regenerate figures without rebuilding sidecars')
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old_paths = set()
    for split in ('train', 'valid', 'test'):
        old = torch.load(PROJECT / f'data/ASAP_processed_dinr_beat_dev_v2/dinr_beat_{split}_b512_o50.pt',
                         weights_only=False)
        old_paths.update(Path(row['source_sidecar']) for row in old['performances'])
    if args.plots_only:
        anchor_diffs, duration_diffs = collect_plot_statistics(old_paths)
        plot_statistics(dict(anchor_shift_ms=dict(values=anchor_diffs),
                             beat_duration_log_ratio=dict(values=duration_diffs)),
                        load_curve_comparison())
        return
    get_beats, _ = load_beat_mapper()
    overrides, records, sources = {}, [], []
    anchor_diffs, duration_diffs = [], []
    for index, old_path in enumerate(sorted(old_paths)):
        raw_path = old_path.with_name(old_path.name.replace('ASAP_GROUPED_MEDIAN', 'ASAP_MUSICAL51'))
        payload = torch.load(raw_path, weights_only=False)
        old_payload = torch.load(old_path, weights_only=False)
        if payload['score']['pitch'] != old_payload['score']['pitch']:
            raise ValueError(f'Raw and current score pitch differ: {raw_path}')
        payload['score'] = copy.deepcopy(old_payload['score'])
        old_by_source = {p['performance_source']: p for p in old_payload['performances']}
        midi = MidiFile(str(SOURCE / payload['meta']['score_source']))
        notes = score_notes(midi)
        ticks = np.asarray([note.start for note in notes], dtype=np.int64)
        edges, _, score_ms, _, _ = beat_grid(midi, get_beats)
        converted = []
        for performance in payload['performances']:
            if performance['performance_source'] not in old_by_source:
                continue
            old_perf = old_by_source[performance['performance_source']]
            old_values = np.asarray(old_perf['label_shared_raw'], dtype=float).copy()
            prior = old_perf.get('global_timing_scale')
            previous_scale = float(prior['scale']) if prior else 1.0
            old_values[:, :2] /= previous_scale
            if not np.allclose(old_values, performance['label_shared_raw'], atol=1e-7, rtol=1e-10):
                raise ValueError(f'Raw label identity differs: {performance["performance_source"]}')
            comparison = compare_anchor_protocol(ticks, edges, np.cumsum(old_values[:, 0]))
            anchor_diffs.extend(comparison['anchor_diff'].tolist())
            duration_diffs.extend(comparison['duration_log_ratio'].tolist())
            normalized = normalize_performance(performance, ticks, edges, score_ms,
                anchor_method='structural_earliest', method=METHOD)
            converted.append(normalized)
            records.append(dict(work=old_path.parent.relative_to(SOURCE).as_posix(),
                performance=performance['performance_id'],
                valid_beats_old=comparison['old_valid'], valid_beats_new=comparison['new_valid'],
                valid_beats_common=comparison['common_valid'], beat_count=comparison['total_beats'],
                **normalized['global_timing_scale']))
        if len(converted) != len(old_payload['performances']):
            raise ValueError(f'Raw performance cohort mismatch: {old_path}')
        payload['performances'] = converted
        payload['performances_by_source'] = {p['performance_source']: p for p in converted}
        payload['label_cache'], payload['perf_style_cache'] = {}, {}
        payload.pop('_cache_signature', None)
        payload['performance_time_normalization'] = METHOD
        payload['meta'] = copy.deepcopy(payload['meta'])
        payload['meta'].update(time_normalization=METHOD, performance_count=len(converted))
        payload['normalization_provenance'] = dict(raw_source=str(raw_path), raw_sha256=digest(raw_path),
            prior_sidecar=str(old_path), prior_sha256=digest(old_path),
            protocol='per-performance median log beat-duration ratio; earliest paired performance onset within each same-score-onset group; interpolate score beat edges between observed groups; no endpoint extrapolation; standalone grace-note MIDI flags unavailable')
        destination = WORKS / old_path.relative_to(SOURCE)
        destination = destination.with_name(destination.name.replace('ASAP_GROUPED_MEDIAN', 'ASAP_STRUCTURAL_EARLIEST').replace('ASAP_MUSICAL51', 'ASAP_STRUCTURAL_EARLIEST'))
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, destination)
        overrides[str(old_path.resolve())] = str(destination.resolve())
        sources.append(dict(path=str(destination), sha256=digest(destination), **payload['normalization_provenance']))
        if index % 20 == 0:
            print(f'normalized {index+1}/{len(old_paths)} works', flush=True)
    (OUTPUT / 'normalization_manifest.json').write_text(json.dumps(dict(method=METHOD,
        anchor_method='structural_earliest', works=sources, performances=records), indent=2) + '\n')

    build(SOURCE, EVENTS, ROOT / 'configs/asap_valid_heldout_12works.json', 512, .5,
          sidecar_overrides=overrides, normalization_description=METHOD,
          anchor_method='structural_earliest')
    fixed_audit = prepare(EVENTS, FIXED, output_tag='e512_v1')
    (OUTPUT / 'fixed_event_data_audit.json').write_text(json.dumps(fixed_audit, indent=2) + '\n')
    checked = audit(FIXED, 'e512_v1')
    for split in ('train', 'valid', 'test'):
        current = torch.load(FIXED / f'dinr_beat_{split}_e512_v1.pt', weights_only=False)
        beat = current['events'][:, 0] == 0
        valid = beat & current['target_valid'][:, 0]
        if not torch.isfinite(current['targets'][valid, 0]).all() or (current['targets'][valid, 0] <= 0).any():
            raise ValueError(f'{split}: invalid supported beat durations')
        residuals = []
        for p in current['performances']:
            a, b = p['span']
            valid_p = beat[a:b] & current['target_valid'][a:b, 0]
            ratios = torch.log(current['targets'][a:b, 0][valid_p] / current['events'][a:b, 2][valid_p])
            if valid_p.any():
                residuals.append(float(np.median(ratios.numpy())))
        if max(map(abs, residuals)) > 1e-5:
            raise ValueError(f'{split}: new anchor normalization residual too large')
        checked['splits'][split]['max_abs_median_log_beat_ratio'] = max(map(abs, residuals))
    checked['checks'] += ['structural-earliest beat anchors', 'all supported durations positive finite',
                          'per-performance median log beat ratio normalized to zero']
    (OUTPUT / 'fixed_event_data_audit_checked.json').write_text(json.dumps(checked, indent=2) + '\n')

    curve_rows = load_curve_comparison()
    plot_statistics(dict(anchor_shift_ms=dict(values=anchor_diffs),
                         beat_duration_log_ratio=dict(values=duration_diffs)), curve_rows)
    summary = dict(method=METHOD, anchor_method='structural_earliest', works=len(sources),
        performances=len(records), scale_distribution=pct([r['scale'] for r in records]),
        anchor_shift_ms=pct(anchor_diffs), beat_duration_log_ratio=pct(duration_diffs),
        beat_support=dict(old=sum(r['valid_beats_old'] for r in records),
            new=sum(r['valid_beats_new'] for r in records), common=sum(r['valid_beats_common'] for r in records),
            possible=sum(r['beat_count'] for r in records)),
        split_audit=checked['splits'], selected_works=[r['work'] for r in curve_rows])
    (OUTPUT / 'statistics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('split_audit',)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
