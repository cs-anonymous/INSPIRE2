#!/usr/bin/env python3
"""Compare completed beat-median runs with saved PT-1 under one observation."""
from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
import script.evaluate_exp1116_baselines as baseline
from script.evaluate_exp1117_beat_dev import work_metrics

OUTPUT = ROOT / 'analysis/exp1117_beat_median/result_analysis'
CAMPAIGN = PROJECT / 'results/exp1117_beat_median_bs48'


def render_analysis(data):
    rows = {r['id']: r for r in data['results']}
    def mean(identifier, name):
        return rows[identifier]['metrics'][name]['mean']
    def cells(identifier):
        return [f'{rows[identifier]["metrics"][n]["mean"]:.3f} ± {rows[identifier]["metrics"][n]["std"]:.3f}'
                for n in baseline.NAMES]
    gaps = {n: (mean('new_A2', n) / mean('pt_base', n) - 1) * 100
            for n in ('onset_w1', 'duration_w1', 'velocity_w1')}
    velocity_gain = (1 - mean('new_A2', 'velocity_w1') / mean('beat_A2', 'velocity_w1')) * 100
    coverage = {k: sum(r[k] for r in data['coverage']) for k in ('notes', 'beats', 'beat_valid',
                'onset_valid', 'duration_valid', 'velocity_valid')}
    lines = ['### 7.5 2026-11-19 结果分析与 PT-1 同口径复核', '',
        '**A2 是本轮六项设置中最均衡的一组。** 它在原生主表的 tempo、duration、velocity 上均领先；onset 与 A1/A3 接近，并非四项均严格最优。当前适合保留的组合是 beat/note log_dev、onset dev、shared duration 输入、separate duration 输出以及 measure_start。', '',
        '**先统一比较口径。** 第 4.1 节的 PT-1 是旧人类参考＋音符重提取，第 7.3 节是新归一化参考＋原生 BEAT 事件；不能直接用两表数值判断差距。本次复用已保存输出，对 PT-1、旧 A2、新六项设置统一使用本轮 beat-median 人类参考，从预测音符起音以 onset-group 均值＋线性插值重提取 beat/onset，保留 score 音符身份；不重训、不重新采样、不额外缩放主比较预测。', '',
        f'共同有效范围按这 8 项设置 × 3 轮确定：tempo {coverage["beat_valid"]:,}/{coverage["beats"]:,} beats；onset {coverage["onset_valid"]:,}/{coverage["notes"]:,} notes；duration、velocity 均为 {coverage["notes"]:,}/{coverage["notes"]:,} notes。没有纳入 PT-scratch，因此 duration/velocity 覆盖比第 4.1 节更完整。每轮 macro-average 19 首作品，报告三轮 mean ± sample std。', '',
        '| 设置 | Tempo Pearson ↑ | Onset dev W1 ↓ | Duration log-ratio W1 ↓ | Velocity W1 ↓ |',
        '|---|---:|---:|---:|---:|']
    for identifier, label in [('pt_base', 'PT-1'), ('beat_A2', '旧 A2'),
                              *[(f'new_{s}', f'新 {s}') for s in ('A1','A2','A3','C1','C2','C3')]]:
        lines.append('| '+' | '.join([label]+cells(identifier))+' |')
    lines += ['', f'**A2 与 PT-1 的差距集中在三个 W1 指标。** 新 A2 相比 PT-1：onset 误差高 {gaps["onset_w1"]:.1f}%，duration 高 {gaps["duration_w1"]:.1f}%，velocity 高 {gaps["velocity_w1"]:.1f}%；tempo 高 {mean("new_A2", "tempo_curve_pearson")-mean("pt_base", "tempo_curve_pearson"):.3f}。Tempo Pearson 对全曲统一时间缩放不敏感，它衡量相对速度曲线形状，不能代表音符时长或绝对速度均更好。', '',
        '| 指标 | A2 − PT-1 | 按作品配对 bootstrap 95% 区间 | A2 更好的作品数 |',
        '|---|---:|---:|---:|']
    for n in baseline.NAMES:
        r = data['a2_vs_pt'][n]
        low, high = r['paired_work_bootstrap_95ci']
        lines.append(f'| {n} | {r["a2_minus_pt"]:.4f} | [{low:.4f}, {high:.4f}] | {r["a2_better_works"]}/19 |')
    lines += ['', '以上先对每首作品的三轮取均值，再对 19 首作品配对 bootstrap 10,000 次。Tempo 优势的区间跨零，尚不能认定 A2 稳定优于 PT-1；三项 W1 的差距则在多数作品上同向。该区间只衡量作品抽样，表内 std 只衡量生成采样，两者都不包含重复训练 seed 的不确定性。', '',
        f'额外诊断中，对每份 PT-1 输出再做一次全曲 beat-median 缩放，其 duration W1 为 {data["posthoc_pt_rescaled"]["metrics"]["duration_w1"]["mean"]:.3f}，原值为 {mean("pt_base", "duration_w1"):.3f}；tempo/onset/velocity 不变。因此时长差距主要不能由全曲典型速度偏置解释。此项只是后处理诊断，不属于主比较，也不是重训 PT。', '',
        '**主要消融结论。** A2 相比 A1，仅将 beat/note duration 改为 log_dev，tempo 从 0.141 升至 0.332、duration 从 0.980 降至 0.783，说明显式使用 score 时长作为参照有帮助。A3 去掉 measure_start 后没有优于 A2；A1/A3 对照中则存在指标取舍，不能宣称小节标记对每项都提升。C1/C2 的 abs onset 明显劣于 dev；C3 的 shared_conditioned duration head 明显劣于 separate，现阶段应保留独立输出头。', '',
        '**Teacher forcing 与 AR 的差距揭示后续重点。** 下面均为本轮原生事件评估，两个模式都做三轮采样；teacher forcing 为真实历史，AR 为生成历史，因此差异主要反映条件历史变化与误差传播，并非 greedy 与 sampling 的区别。', '',
        '| 设置 | Beat log MAE TF→AR | Note duration log MAE TF→AR | Onset MAE TF→AR | Velocity MAE TF→AR |',
        '|---|---:|---:|---:|---:|']
    for s in ('A1', 'A2', 'C1', 'C2', 'C3'):
        d=data['native_teacher_forced_vs_ar'][s]
        cells2 = [f'{d["teacher_forced"][k]:.3f} → {d["autoregressive"][k]:.3f}'
                  for k in ('beat_duration_log_mae','note_duration_log_mae','onset_mae','velocity_mae')]
        lines.append('| '+' | '.join([s]+cells2)+' |')
    lines += ['', 'A2 的 beat duration 对生成历史相对稳健，velocity 和 note duration 仍明显退化。C1/C2 在真实历史下 onset 已较差，AR 又将误差放大；残差 dev 让模型围绕已知 score 位置预测偏移，可能比重新预测拍内绝对位置更容易，但目前 abs/dev 同时使用不同 bins/range，不能把效应完全归因于表示形式。较低的 valid 总 CE 也不能跨不同 token 表示直接当作更好音乐表现：A1 valid CE 低于 A2，但 AR 指标落后。', '',
        f'**新归一化不是本轮 timing 提升的主要证据。** 在相同新参考与观测下，旧 A2→新 A2 的 tempo 为 {mean("beat_A2", "tempo_curve_pearson"):.3f}→{mean("new_A2", "tempo_curve_pearson"):.3f}，duration 为 {mean("beat_A2", "duration_w1"):.3f}→{mean("new_A2", "duration_w1"):.3f}，几乎持平；velocity 为 {mean("beat_A2", "velocity_w1"):.3f}→{mean("new_A2", "velocity_w1"):.3f}，降低 {velocity_gain:.1f}%。velocity 标签未改变，时间标签变化会影响联合学习，但单个训练 seed 不足以把这项提升稳定归因于归一化。统一归一化仍有明确的数据一致性价值，尤其消除了两个 raw MS-score fallback。', '',
        f'**指标本身也需对照简单基线。** 在同一共同范围内，将所有 onset dev 设为 0（照 score 拍内位置演奏）的 W1 为 {data["score_onset_dev_zero_w1"]:.4f}，A2 为 {mean("new_A2", "onset_w1"):.4f}，PT-1 为 {mean("pt_base", "onset_w1"):.4f}。说明当前 onset W1 不能单独证明学到了更好的微时值表达。W1(singleton, human) 偏好人类逐音符中位数；使用测试人类标签构造的理论最小值约为 onset {data["oracle_lower_bound"]["onset_w1"]:.4f}、duration {data["oracle_lower_bound"]["duration_w1"]:.3f}、velocity {data["oracle_lower_bound"]["velocity_w1"]:.3f}。这只是不可部署的参考下界，不是可达到性承诺，也不衡量生成多样性。', '',
        '**建议的后续顺序：** 先固定 A2，用多个训练 seed 确认优势；在 valid 上对 sampling temperature / top-p 与 greedy 做小范围比较，检查 velocity 的 AR 退化及动态范围，避免直接按 test 调参。再考虑改善生成历史鲁棒性、给 velocity 单独建模；暂不扩大 abs onset 或共享 duration 输出头的搜索。PT-1 使用预训练初始化，而本轮 A2 为随机初始化，残余差距同时混合了预训练、架构、mask 和表示因素，不能直接判定 BEAT 架构存在上限。', '',
        '![A2 与 PT-1 的逐作品指标差值](../analysis/exp1117_beat_median/result_analysis/a2_vs_pt_per_work.png)', '',
        '[完整复核 JSON、逐作品差值与 bootstrap](../analysis/exp1117_beat_median/result_analysis/analysis.json) · [复核脚本](../script/analyze_exp1117_beat_median.py)']
    (OUTPUT/'analysis.md').write_text('\n'.join(lines)+'\n')

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    labels = []
    for item in data['a2_vs_pt']['onset_w1']['per_work']:
        parts = item['work'].split('/')
        labels.append(parts[0].split(',')[0]+' / '+parts[-1][:26])
    for ax, n in zip(axes.flat, baseline.NAMES):
        differences = [r['delta'] for r in data['a2_vs_pt'][n]['per_work']]
        colors = ['#2563eb' if (v > 0 if n=='tempo_curve_pearson' else v < 0) else '#e68a2e' for v in differences]
        ax.barh(np.arange(19), differences, color=colors, height=.7)
        ax.set_yticks(np.arange(19), labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color='black', lw=.8)
        ax.set(title=n, xlabel='A2 - PT-1 (mean of 3 draws)')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Common observation and new human references; blue = A2 better, orange = PT-1 better',fontsize=12)
    fig.savefig(OUTPUT/'a2_vs_pt_per_work.png',dpi=160)
    plt.close(fig)


def read(path):
    return json.loads(path.read_text())


def load_current_references():
    """Load the new continuous human targets with canonical score identities."""
    old_reference = PROJECT / 'data/ASAP_processed_dinr_beat_dev_e512_v2/dinr_beat_test_e512_v2.pt'
    new_reference = PROJECT / 'data/ASAP_processed_dinr_beat_median_e512_v1/dinr_beat_test_e512_v1.pt'
    references = baseline.load_references(old_reference)
    payload = torch.load(new_reference, map_location='cpu', weights_only=False)
    for reference in references:
        rows = [row for row in payload['performances'] if row['work'] == reference['work']]
        targets, masks = [], []
        for row in rows:
            a, b = row['span']
            assert np.array_equal(payload['events'][a:b].numpy(), reference['events'])
            targets.append(payload['targets'][a:b].numpy().astype(float))
            masks.append(payload['target_valid'][a:b].numpy())
        targets, masks = np.stack(targets), np.stack(masks)
        note, beat = reference['note_mask'], reference['beat_mask']
        reference.update(human_onset=targets[:, note, 1],
            human_duration=np.log(np.maximum(targets[:, note, 0], 1)) - reference['score_log_duration'],
            human_velocity=targets[:, note, 2], human_beat_log=np.log(np.maximum(targets[:, beat, 0], 1)),
            onset_valid=masks[:, note, 1].all(0), duration_valid=masks[:, note, 0].all(0),
            velocity_valid=masks[:, note, 2].all(0), beat_valid=masks[:, beat, 0].all(0))
    return references, new_reference


def main():
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    references, new_reference = load_current_references()
    baseline._REFERENCES = references
    methods = [m for m in baseline.discover_methods() if m['id'] in ('pt_base', 'beat_A2')]
    methods[1]['label'] = 'Old BEAT A2'
    for setting in ('A1', 'A2', 'A3', 'C1', 'C2', 'C3'):
        directory = CAMPAIGN / f'{setting}_seed42'
        run = read(directory / 'run_config.json')
        methods.append(dict(id='new_'+setting, family='BEAT', label='New BEAT '+setting,
            mask='timestep', representation=run['duration_target'], source=str(directory)))
    extracted = []
    for method in methods:
        print('Extract', method['id'], flush=True)
        extracted.append(baseline.extract_method(method))
    masks = baseline.common_masks(extracted, references)
    results = baseline.summarize(extracted, references, masks)
    coverage = [dict(work=r['work'], notes=r['notes'], beats=len(r['beat_ticks']),
        **{k+'_valid': int(v.sum()) for k, v in mask.items()}) for r, mask in zip(references, masks)]

    # A diagnostic only: retain PT's local timing, but remove its own typical
    # tempo offset, using the exact same beat-median rule as data preparation.
    pt_rescaled = copy.deepcopy(extracted[0])
    pt_rescaled['method'].update(id='pt_base_rescaled', label='PT-1 (post-hoc beat-median)')
    scales = []
    for (work, sample), features in pt_rescaled['features'].items():
        reference = references[work]
        score_log = np.log(reference['events'][reference['beat_mask'], 2])
        valid = reference['beat_valid'] & np.isfinite(features['beat_log'])
        shift = float(-np.median(features['beat_log'][valid] - score_log[valid]))
        features['beat_log'] += shift
        features['duration'] += shift
        scales.append(dict(work=work, sample=sample, scale=float(np.exp(shift))))
    rescaled = baseline.summarize([pt_rescaled], references, masks)[0]

    # Paired work comparisons: average 3 draws per work before resampling works.
    # These intervals do not represent independent training-seed uncertainty.
    by_id = {r['id']: r for r in results}
    a2, pt = by_id['new_A2'], by_id['pt_base']
    rng = np.random.default_rng(20261119)
    indices = rng.integers(0, 19, size=(10000, 19))
    comparisons = {}
    for name in baseline.NAMES:
        a = np.array([np.mean([x[name] for x in a2['per_work'] if x['work'] == w]) for w in range(19)])
        b = np.array([np.mean([x[name] for x in pt['per_work'] if x['work'] == w]) for w in range(19)])
        diff = a - b
        comparisons[name] = dict(a2_minus_pt=float(diff.mean()),
            paired_work_bootstrap_95ci=np.percentile(diff[indices].mean(1), [2.5, 97.5]).tolist(),
            a2_better_works=int((diff > 0).sum() if name == 'tempo_curve_pearson' else (diff < 0).sum()),
            per_work=[dict(work=references[w]['work'], a2=float(a[w]), pt=float(b[w]), delta=float(diff[w])) for w in range(19)])

    # W1(singleton,human) is minimized at the per-note human median.
    oracle, score_onset = [], []
    for reference, mask in zip(references, masks):
        row = dict(work=reference['work'])
        for name in ('onset', 'duration', 'velocity'):
            values = reference['human_'+name][:, mask[name]]
            row[name+'_w1'] = float(np.abs(values - np.median(values, axis=0)).mean())
        oracle.append(row)
        score_onset.append(float(np.abs(reference['human_onset'][:, mask['onset']]).mean()))

    diagnostics, native = {}, {}
    for setting in ('A1', 'A2', 'A3', 'C1', 'C2', 'C3'):
        directory = CAMPAIGN / f'{setting}_seed42'
        run, test = read(directory/'run_config.json'), read(directory/'test/metrics.json')
        diagnostics[setting] = {mode: {name: float(np.mean([r['metrics'][name]
            for r in test['rounds'] if r['mode'] == mode]))
            for name in ('beat_duration_log_mae', 'note_duration_log_mae', 'onset_mae', 'velocity_mae')}
            for mode in ('teacher_forced', 'autoregressive')}
        works = []
        for wd in sorted((directory/'test').glob('work_*')):
            for sample in range(3):
                works.append(dict(work=read(wd/'metrics.json')['work'], sample=sample,
                    **work_metrics(wd/f'autoregressive_{sample:02d}_tokens.npz', run)))
        native[setting] = works
    native_old = read(ROOT / 'analysis/exp1116_beat_dev/metrics.json')
    native_new = read(ROOT / 'analysis/exp1117_beat_median/metrics.json')
    data = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        protocol=dict(reference=str(new_reference), sha256=hashlib.sha256(new_reference.read_bytes()).hexdigest(),
            observation='All 8 methods: onset-group mean + linear beat mapping from predicted notes; supported intervals only',
            normalization='New human beat-median labels; saved predictions unchanged; no new training or sampling',
            coverage='Common support over all 8 methods and 3 draws; separate from prior 32-method masks',
            aggregation='Macro average 19 works per draw; mean and sample std ddof=1 over 3 AR draws',
            diagnostic_pt_rescaled='Separate diagnostic: one extra factor per saved PT draw/work removes its typical beat-tempo offset; not a retrained PT result',
            bootstrap='Paired 19-work bootstrap after averaging 3 draws, 10000 resamples; excludes training-seed uncertainty'),
        coverage=coverage, results=results, a2_vs_pt=comparisons,
        oracle_lower_bound={k:float(np.mean([r[k] for r in oracle])) for k in ('onset_w1','duration_w1','velocity_w1')},
        score_onset_dev_zero_w1=float(np.mean(score_onset)), per_work_oracle=oracle,
        posthoc_pt_rescaled=rescaled, posthoc_pt_scales=scales,
        native_teacher_forced_vs_ar=diagnostics, native_per_work=native,
        native_old_a2=next(r for r in native_old['settings'] if r['setting']=='A2'),
        native_new_a2=next(r for r in native_new['settings'] if r['setting']=='A2'))
    (OUTPUT/'analysis.json').write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    render_analysis(data)
    for r in results+[rescaled]:
        print(r['id'], {k: f'{v["mean"]:.6f} ± {v["std"]:.6f}' for k,v in r['metrics'].items()})
    print('A2 vs PT', json.dumps({k:{f:v for f,v in r.items() if f!='per_work'} for k,r in comparisons.items()},indent=2))
    print('Oracle lower bounds',data['oracle_lower_bound'],'score onset dev=0',data['score_onset_dev_zero_w1'])


if __name__ == '__main__':
    main()
