#!/usr/bin/env python3
"""Leave one human performance out; compare it and models to remaining humans."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from script.analyze_exp1117_beat_median import load_current_references, OUTPUT, CAMPAIGN
import script.evaluate_exp1116_baselines as baseline


def leave_one_out(values):
    """One draw compared to the empirical distribution of all other humans."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Expected at least two finite, aligned human performances')
    errors = np.abs(values[:, None, :] - values[None, :, :]).mean(axis=-1)
    return (errors.sum(axis=1) / (len(values) - 1)).tolist()


def main():
    torch.set_num_threads(2)
    references, reference_path = load_current_references()
    previous = json.loads((OUTPUT / 'analysis.json').read_text())
    baseline._REFERENCES = references
    # Retain the 8-method common support used in Section 7.5, including abs
    # variants. This avoids silently changing note/beat coverage per method.
    methods = [{k: r[k] for k in ('id','family','label','mask','representation','source')}
               for r in previous['results']]
    extracted = [baseline.extract_method(method) for method in methods]
    masks = baseline.common_masks(extracted, references)
    by_id = {r['method']['id']: r for r in extracted}
    output, excluded = [], []
    for work, (reference, mask) in enumerate(zip(references, masks)):
        human_count = len(reference['human_onset'])
        if human_count < 2:
            excluded.append(reference['work'])
            continue
        human_rows = []
        feature_names = ('onset', 'duration', 'velocity')
        errors = {name: leave_one_out(reference['human_'+name][:, mask[name]]) for name in feature_names}
        model_rows = {name: [] for name in ('pt_base', 'new_A2')}
        human_log_beat = reference['human_beat_log'][:, mask['beat']]
        for heldout in range(human_count):
            others = np.arange(human_count) != heldout
            curve = -human_log_beat[heldout]
            median_curve = -np.median(human_log_beat[others], axis=0)
            if min(curve.std(), median_curve.std()) <= 1e-12:
                raise ValueError(f'Constant human tempo curve: {reference["work"]}')
            values = dict(heldout=heldout, tempo_curve_pearson=float(np.corrcoef(curve, median_curve)[0, 1]),
                **{name+'_w1': errors[name][heldout] for name in feature_names})
            human_rows.append(values)
            remaining = {**reference, **{key: value[others] for key, value in reference.items() if key.startswith('human_')}}
            for identifier in model_rows:
                for sample in range(3):
                    model_rows[identifier].append(dict(heldout=heldout, sample=sample,
                        **baseline.score_features(by_id[identifier]['features'][(work, sample)], remaining, mask)))
        # Pointwise reference medians include the entire human set; useful only
        # as an in-sample mathematical lower bound for singleton W1.
        oracle = {name+'_w1': float(np.abs(reference['human_'+name][:, mask[name]]
            - np.median(reference['human_'+name][:, mask[name]], axis=0)).mean()) for name in feature_names}
        human_summary = {n: float(np.mean([r[n] for r in human_rows])) for n in baseline.NAMES}
        model_summary = {identifier: {n: float(np.mean([r[n] for r in rows])) for n in baseline.NAMES}
                         for identifier, rows in model_rows.items()}
        per_sample = {identifier: [{n: float(np.mean([r[n] for r in rows if r['sample']==sample]))
                     for n in baseline.NAMES} for sample in range(3)] for identifier, rows in model_rows.items()}
        onset = reference['human_onset'][:, mask['onset']]
        summary = dict(work=reference['work'], humans=human_count, human=human_summary,
            human_leave_one_out_rows=human_rows, models=model_summary, models_per_sample=per_sample,
            score_zero_onset_w1=float(np.abs(onset).mean()), in_sample_median_lower_bound=oracle,
            coverage={k: int(v.sum()) for k, v in mask.items()})
        output.append(summary)
    names = ('human','pt_base','new_A2')
    metrics = {identifier: {n: float(np.mean([r['human'][n] if identifier=='human' else r['models'][identifier][n]
                                          for r in output])) for n in baseline.NAMES} for identifier in names}
    # For W1, averaging the H leave-one-out reference sets must equal using
    # all H references once; tempo medians need not have that property.
    for identifier in ('pt_base', 'new_A2'):
        stored = next(r for r in previous['results'] if r['id'] == identifier)
        for name in ('onset_w1', 'duration_w1', 'velocity_w1'):
            expected = np.mean([r[name] for r in stored['per_work']
                                if r['work_path'] not in excluded])
            assert abs(metrics[identifier][name] - expected) < 1e-10
    model_sampling = {identifier: {n: dict(mean=metrics[identifier][n], std=float(np.std([
        np.mean([r['models_per_sample'][identifier][sample][n] for r in output]) for sample in range(3)], ddof=1)))
        for n in baseline.NAMES} for identifier in ('pt_base','new_A2')}
    result = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        protocol=dict(reference=str(reference_path),
            human='Each complete human performance compared to all remaining humans of its work; no self-reference',
            models='Same 12 works, exact 8-method common masks; compare each of 3 draws to the same remaining-human references for every heldout human',
            aggregation='Mean heldout performances within work, then macro-average works; model draws also averaged',
            tempo='Human negative log beat duration Pearson vs pointwise median of other humans; models use those same leave-one-out median curves',
            wasserstein='Singleton to remaining-human distribution = mean absolute difference; lower does not imply more human-like diversity',
            units='Onset: dimensionless beat position residual; duration: log ratio; velocity: MIDI value',
            exclusions='Single-performance works cannot support leave-one-out; excluded from all compared rows'),
        works=len(output), human_performances=sum(r['humans'] for r in output),
        excluded_single_performance_works=excluded, metrics=metrics, model_sampling=model_sampling,
        score_zero_onset_w1=float(np.mean([r['score_zero_onset_w1'] for r in output])),
        in_sample_median_lower_bound={n:float(np.mean([r['in_sample_median_lower_bound'][n] for r in output]))
            for n in ('onset_w1','duration_w1','velocity_w1')}, per_work=output)
    (OUTPUT/'human_baseline.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_work','protocol','excluded_single_performance_works')}, indent=2))
    render(result)


def render(data):
    lines = ['### 7.6 2026-11-19 人类留一基线与 Onset Dev 指标解释', '',
        f'补算 human→human：每次拿一个完整人类演奏作为预测，仅与该作品其余人类演奏比较，排除自身。19 首 test work 中有 7 首只有一个演奏，无法留一，因此下表全部限定在其余 **{data["works"]} works、{data["human_performances"]} performances**；PT-1/A2 也按完全相同的作品、共同 mask 和留一人类参考重算。先平均每首作品内留一演奏，再对作品 macro-average；模型再平均三轮 AR。它与第 7.5 节的 19-work 数字不同。', '',
        '| 对照 | Tempo Pearson ↑ | Onset dev W1 ↓ | Duration log-ratio W1 ↓ | Velocity W1 ↓ |',
        '|---|---:|---:|---:|---:|']
    for identifier, label in [('human','Human→其他 Human（留一）'),('pt_base','PT-1'),('new_A2','A2')]:
        m = data['metrics'][identifier]
        lines.append('| '+' | '.join([label, f'{m["tempo_curve_pearson"]:.3f}', f'{m["onset_w1"]:.4f}',
                                     f'{m["duration_w1"]:.3f}', f'{m["velocity_w1"]:.3f}'])+' |')
    oracle = data['in_sample_median_lower_bound']
    lines += [f'| 固定 onset dev=0 | — | {data["score_zero_onset_w1"]:.4f} | — | — |', '',
        f'逐音符人类中位数对全部同作品人类的 W1 最小值为 onset {oracle["onset_w1"]:.4f}、duration {oracle["duration_w1"]:.3f}、velocity {oracle["velocity_w1"]:.3f}。这是使用参考标签构造的样本内数学下界，**不是 human→human 指标**。此前 7.5 节下界还包含 7 首单人演奏作品，单人位置的样本内误差为零，因此不能拿来充当人类表现。', '',
        '**需要区分 onset dev 表示与当前标量评分。** Onset dev 保留音符相对局部 beat 的细微位置偏移，能描述拍内伸缩和同起音组的不同步；全曲、段落层面的 rubato 大部分已经被 beat curve 吸收。它作为训练 target 或微时值特征仍然合理，但不能单独评价整体节奏表达。', '',
        '例如每拍只有一个音、且都位于 score 拍点时，以这些起音映射出的 beat anchor 会经过每个音符：即使演奏中的相邻 beat 时长变化很大，onset dev 仍可全为零。这些 rubato 应由 tempo curve 衡量。', '',
        r'当前分数是 $W_1(\delta_{\hat d_i},P_i^{human})=\frac{1}{H}\sum_h|\hat d_i-d_i^{(h)}|$，然后对 note、work、draw 平均。它等价于逐音符平均绝对误差，被人类该音符的中位数最小化。即使模型完整复现真实演奏分布，随机抽到的演奏也可能比中位数模板分数更差；先逐 draw 算再平均，仍然不会奖励生成的多样性。**同一问题也适用于当前 duration/velocity W1，不仅是 onset。**', '',
        '**因此建议保留 onset dev，但将当前 onset W1 定位为偏差诊断。** 分开评价：(1) tempo curve 的形状；(2) 拍内 onset dev 的位置、幅度和分布；(3) 同 score onset group 的异步起音/声部关系。分布可比较每个音符多次生成与多人参考的 W1，或使用 CRPS/energy score 这类适当评分；当前仅 3 次生成、部分作品仅 1–2 个参考，分布估计会有较大有限样本误差，需配套同采样数的人类基线和区间。不要为了降低此项误差而直接把表达压向零。', '',
        '[逐作品、逐演奏、采样波动与人类覆盖审计](../analysis/exp1117_beat_median/result_analysis/human_baseline.json) · [计算脚本](../script/evaluate_exp1117_human_baseline.py)']
    (OUTPUT/'human_baseline.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
