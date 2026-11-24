#!/usr/bin/env python3
"""Exact per-note Wasserstein distances between generated and human samples."""
from __future__ import annotations

from datetime import datetime
from itertools import combinations
import csv
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from script.analyze_exp1117_beat_median import load_current_references, OUTPUT
import script.evaluate_exp1116_baselines as baseline

DESTINATION = OUTPUT / 'distribution_w1'
ATTRIBUTES = ('onset', 'duration', 'velocity')


def empirical_w1_columns(predictions, references):
    """Exact W1 for uniform empirical distributions, independently per column.

    Arrays are [samples, aligned notes]. Unequal sample counts are supported:
    integrate |F_pred^-1(q) - F_ref^-1(q)| over merged quantile boundaries.
    Notes never get pooled or re-paired with a different score note.
    """
    p, q = np.asarray(predictions, dtype=np.float64), np.asarray(references, dtype=np.float64)
    if p.ndim != 2 or q.ndim != 2 or p.shape[1] != q.shape[1] or min(len(p), len(q)) < 1:
        raise ValueError('Expected nonempty [samples, aligned notes] matrices')
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('W1 inputs must be finite after applying the shared mask')
    boundaries = np.unique(np.r_[np.arange(len(p)+1) / len(p), np.arange(len(q)+1) / len(q)])
    widths = np.diff(boundaries)
    midpoints = (boundaries[:-1] + boundaries[1:]) * .5
    pi = np.minimum((midpoints * len(p)).astype(int), len(p)-1)
    qi = np.minimum((midpoints * len(q)).astype(int), len(q)-1)
    return (np.abs(np.sort(p, axis=0)[pi] - np.sort(q, axis=0)[qi]) * widths[:, None]).sum(axis=0)


def distribution_metrics(prediction, reference, mask):
    result = {}
    for name in ATTRIBUTES:
        p, q = prediction[name][:, mask[name]], reference['human_'+name][:, mask[name]]
        w1 = empirical_w1_columns(p, q)
        cross_mae = np.abs(p[:, None, :] - q[None, :, :]).mean(axis=(0, 1))
        assert np.all(w1 <= cross_mae + 1e-10)
        if name == 'onset':
            score = reference['score_ratio'][mask[name]]
            # The exact same predicted notes must score identically in the
            # absolute beat-relative and score-relative residual coordinates.
            absolute = empirical_w1_columns(p + score, q + score)
            np.testing.assert_allclose(absolute, w1, rtol=1e-10, atol=1e-12)
        result[name+'_w1'] = float(w1.mean())
        result[name+'_cross_mae'] = float(cross_mae.mean())
    return result


def matched_human_baseline(references, masks, predictions):
    """Every disjoint 3-human vs remaining-human split, with matched model refs."""
    rows, excluded = [], []
    for work, (reference, mask) in enumerate(zip(references, masks)):
        count = reference['humans']
        if count < 4:
            excluded.append(reference['work'])
            continue
        splits = list(combinations(range(count), 3))
        values = {n: reference['human_'+n][:, mask[n]] for n in ATTRIBUTES}
        models = {identifier: {n: predictions[identifier][work][n][:, mask[n]] for n in ATTRIBUTES}
                  for identifier in ('pt_base', 'new_A2')}
        accumulated = {identifier: {n+'_w1': [] for n in ATTRIBUTES}
                       for identifier in ('human', 'pt_base', 'new_A2')}
        for heldout in splits:
            selected = np.array(heldout)
            remaining = np.setdiff1d(np.arange(count), selected)
            for name in ATTRIBUTES:
                q = values[name][remaining]
                accumulated['human'][name+'_w1'].append(float(empirical_w1_columns(values[name][selected], q).mean()))
                for identifier in models:
                    accumulated[identifier][name+'_w1'].append(float(empirical_w1_columns(models[identifier][name], q).mean()))
        rows.append(dict(work=reference['work'], humans=count, prediction_samples=3,
            reference_samples=count-3, disjoint_splits=len(splits),
            metrics={identifier: {n: float(np.mean(v)) for n,v in metrics.items()}
                     for identifier, metrics in accumulated.items()}))
        print('Human matched splits', reference['work'], len(splits), flush=True)
    return dict(works=len(rows), human_performances=sum(r['humans'] for r in rows),
        split_count=sum(r['disjoint_splits'] for r in rows), excluded=excluded,
        metrics={identifier: {name+'_w1': float(np.mean([r['metrics'][identifier][name+'_w1'] for r in rows]))
                 for name in ATTRIBUTES} for identifier in ('human', 'pt_base', 'new_A2')},
        per_work=rows,
        protocol='All disjoint subsets of 3 complete human performances vs remaining H-3; model 3 draws vs exactly the same remaining references; average splits within work then macro-average works; H>=4')


def onset_controls(references, masks, predictions):
    """Zero/mean/median controls and within-note, across-performance variance."""
    per_work=[]
    for work,(reference,mask) in enumerate(zip(references,masks)):
        human=reference['human_onset'][:,mask['onset']]
        zero=float(empirical_w1_columns(np.zeros((3,human.shape[1])),human).mean())
        assert abs(zero-np.abs(human).mean())<1e-12
        variance={identifier:float(np.var(predictions[identifier][work]['onset'][:,mask['onset']],axis=0,ddof=1).mean())
                  for identifier in ('pt_base','new_A2')}
        if len(human)>=2:
            variance['human']=float(np.var(human,axis=0,ddof=1).mean())
        per_work.append(dict(work=reference['work'],humans=len(human),
            zero_w1=zero,
            in_sample_human_mean_w1=float(empirical_w1_columns(human.mean(0,keepdims=True),human).mean()),
            in_sample_human_median_w1=float(empirical_w1_columns(np.median(human,axis=0,keepdims=True),human).mean()),
            mean_sample_variance=variance))
    groups={}
    for name,minimum in [('all',1),('humans_ge2',2),('humans_ge4',4),('humans_ge6',6)]:
        chosen=[r for r in per_work if r['humans']>=minimum]
        if not chosen:
            continue
        groups[name]=dict(works=len(chosen),humans=sum(r['humans'] for r in chosen),
            **{n:float(np.mean([r[n] for r in chosen])) for n in ('zero_w1','in_sample_human_mean_w1','in_sample_human_median_w1')})
        if minimum>=2:
            groups[name]['rms_within_note_spread']={identifier:float(np.sqrt(np.mean([
                r['mean_sample_variance'][identifier] for r in chosen]))) for identifier in ('human','pt_base','new_A2')}
    return dict(groups=groups,per_work=per_work,
        protocol='Same 38-method common masks; zero is three repeated zeros; human mean/median use reference labels and are in-sample diagnostic controls, not deployable predictors; spread = sqrt(macro-work mean of note-mean unbiased across-performance sample variance), only H>=2')


def render_zero_onset(data):
    controls=data['onset_controls']
    models={r['id']:r for r in data['results']}
    lines=['### 7.8 2026-11-19 Onset dev=0 与多样性诊断', '',
        'W1 是通用的分布距离。若真实分布可直接获得，W1(P,Q)=0 当且仅当两分布相同，因此总体目标不会专门偏好均值或中位数；但它衡量的是分布是否匹配，不是多样性越大越好。过窄、过宽、整体偏移都可能产生距离。', '',
        '实际比较的是仅 3 次生成和有限人类参考构成的经验分布，不能保证完全消除偏好收缩的现象。特别是单个人类参考时，W1 退化成到该参考的平均绝对误差；随机人类参考下，期望仍由确定性中位数最小化。此前关于“不偏好均值”的总体性质不能直接套到这种小样本评分上。', '',
        '一个可穷举的例子：真实分布在 −1、0、1 上等概率。模型若从真实分布独立生成 3 次，与独立抽到的 1 个人类参考计算经验 W1，期望为 8/9；模型若永远输出 0，期望为 2/3，反而更低。但如果直接比较真实分布，正确模型的 W1 为 0，固定零分布为 2/3。问题来自有限样本评分，不能据此声称总体 W1 偏好固定均值。', '',
        '下表均使用各作品的全部人类参考和同一共同 mask。多人子集排除单演奏作品，可单独检查零偏移对照：', '',
        '| 同一范围 | 固定 dev=0 W1 ↓ | PT-1 分布 W1 ↓ | A2 分布 W1 ↓ |',
        '|---|---:|---:|---:|']
    for group,model_group in [('all','all_19'),('humans_ge2','multi_human_12'),('humans_ge4','human_ge4_8'),('humans_ge6','human_ge6_5')]:
        c=controls['groups'][group]
        lines.append(f'| {c["works"]} works / {c["humans"]} humans | {c["zero_w1"]:.4f} | {models["pt_base"]["groups"][model_group]["metrics"]["onset_w1"]:.4f} | {models["new_A2"]["groups"][model_group]["metrics"]["onset_w1"]:.4f} |')
    matched=data['human_matched']['metrics']
    c=controls['groups']['humans_ge4']
    lines+=['', '在第 7.7 节的 8 首作品、811 个不重叠人类三样本划分中，零偏移对照也必须对相同剩余人类参考评价。对全部划分平均时，每个人类演奏的权重相同，所以零偏移平均 W1 恰好等于这 8 首作品使用全部人类参考的值：', '',
        '| 8-work、相同剩余参考 | Onset dev W1 ↓ |', '|---|---:|',
        f'| 3 Human → 其余 Human | {matched["human"]["onset_w1"]:.4f} |',
        f'| PT-1：3 次生成 | {matched["pt_base"]["onset_w1"]:.4f} |',
        f'| 固定 dev=0 | {c["zero_w1"]:.4f} |',
        f'| A2：3 次生成 | {matched["new_A2"]["onset_w1"]:.4f} |', '',
        '**另报生成多样性，而不是只看 W1。** 下表固定每个 score note，计算不同演奏/生成之间的无偏样本方差（ddof=1），逐 note、work 平均，再开平方。单位仍为 beat fraction；它表示同一音符在不同演奏中的 RMS 变化幅度，不是把不同音符混在一起的方差。单演奏作品无法估计人类方差，已排除。', '',
        '| 范围 | Human RMS spread | PT-1 RMS spread | A2 RMS spread | 固定 dev=0 |',
        '|---|---:|---:|---:|---:|']
    for group in ('humans_ge2','humans_ge4','humans_ge6'):
        c=controls['groups'][group];v=c['rms_within_note_spread']
        lines.append(f'| {c["works"]} works | {v["human"]:.4f} | {v["pt_base"]:.4f} | {v["new_A2"]:.4f} | 0 |')
    lines+=['', '按这个总体幅度统计，两模型在多人作品上的 onset 跨演奏变化均小于人类；这不支持把 A2 的距离较大简单解释为“生成太多样”。模型每音符只有 3 次生成，方差估计噪声较大；汇总 RMS 也会受少数大偏差影响，应同时看逐作品结果。零偏移分数低说明“保守的微时值”是有竞争力的对照；不能据此认定它具有更好的人类表达。W1 同时受位置与散布影响，spread 单独也不能验证变化是否在正确的音符、方向和结构上。当前仍只评价逐音符边缘分布，无法检测完整风格、跨音符相关性或短语连贯性。', '',
        '[零偏移、样本内均值/中位数对照及逐作品方差](../analysis/exp1117_beat_median/result_analysis/distribution_w1/metrics.json)']
    (OUTPUT/'zero_onset_diagnostics.md').write_text('\n'.join(lines)+'\n')


def render(data):
    rows = {row['id']: row for row in data['results']}
    lines = ['### 7.7 2026-11-19 追加：多次生成分布与多人参考分布的真正 W1', '',
        '对每个 score note，把 3 次生成视作各占 1/3 权重的经验分布，把该作品的 H 个人类演奏视作各占 1/H 权重的经验分布，计算两者的 W1，先逐 note 平均，再 macro-average works。采用精确一维分位数积分，支持两侧样本数不相等；不把不同音符混在一起，也不逐生成轮次算距离后再平均。', '',
        '本节三项距离使用与第 7.5 节相同的新 beat-median 人类参考与音符重提取方式；复用已有三轮 AR 预测，未重训或重新采样。覆盖扩展到 exp1112 的 24 项 DINR、2 项 PT、旧六项 BEAT 和新六项 BEAT，共 38 项，统一所有方法三轮的共同有效位置，因此覆盖与 7.5 节略有差异。旧表保留，旧三项数字应按“逐生成样本对人类的平均绝对误差”理解。', '',
        f'19 works、82 human performances；共同有效 onset {data["coverage"]["onset"]:,}/{data["coverage"]["notes"]:,} notes，duration {data["coverage"]["duration"]:,}/{data["coverage"]["notes"]:,} notes，velocity {data["coverage"]["velocity"]:,}/{data["coverage"]["notes"]:,} notes。7 首单演奏作品仍包含在完整表中；当人类参考只有 1 个时，经验分布 W1 就等于三次生成对该参考的平均绝对误差，无法据此判断真实演奏分布。', '',
        '| 设置 | Onset dev 分布 W1 ↓ | Duration log-ratio 分布 W1 ↓ | Velocity 分布 W1 ↓ |',
        '|---|---:|---:|---:|']
    selected = [('pt_base','PT-1'), ('pt_scratch','PT-scratch'), ('beat_A2','旧 A2'),
                *[(f'new_{s}',f'新 {s}') for s in ('A1','A2','A3','C1','C2','C3')]]
    for identifier, label in selected:
        row = rows[identifier]['metrics']
        lines.append('| '+' | '.join([label]+[f'{row[n+"_w1"]:.4f}' for n in ATTRIBUTES])+' |')
    lines += ['', '| exp1112 log dev 设置 | Onset dev 分布 W1 ↓ | Duration log-ratio 分布 W1 ↓ | Velocity 分布 W1 ↓ |',
        '|---|---:|---:|---:|']
    for row in data['results']:
        if row['family']=='1112' and row['representation'].replace('_',' ')=='log dev':
            label = row['label']+' / '+row['mask']
            lines.append('| '+' | '.join([label]+[f'{row["metrics"][n+"_w1"]:.4f}' for n in ATTRIBUTES])+' |')
    gaps={n:(rows['new_A2']['metrics'][n+'_w1']/rows['pt_base']['metrics'][n+'_w1']-1)*100 for n in ATTRIBUTES}
    lines += ['', f'改成多次生成分布 W1 后，A2 仍是新六项设置中 duration、velocity 最好的一项；onset 上 A1/A3 略低。A2 相比 PT-1 的 onset、duration、velocity 距离分别高 {gaps["onset"]:.1f}%、{gaps["duration"]:.1f}%、{gaps["velocity"]:.1f}%，因此主要排序结论没有反转；逐作品配对结果与区间见 JSON。', '',
        '三次生成合成一个经验分布后只有一个距离，不能再把三轮当作三个 W1 结果报告 std。JSON 提供条件于这批固定预测的作品 bootstrap 区间，以及 19 首、12 首多人演奏作品、8 首至少四演奏作品等分组汇总；这些区间不包含重复训练及新增生成样本的不确定性。Tempo Pearson 的定义没有改变，本节不把它改成分布距离。', '',
        '**A2 / PT-1：旧聚合方式与真正分布 W1，在本节同一覆盖上比较。**', '',
        '| 设置 | Onset：旧→分布 W1 | Duration：旧→分布 W1 | Velocity：旧→分布 W1 |',
        '|---|---:|---:|---:|']
    for identifier in ('pt_base','new_A2'):
        row=rows[identifier]
        lines.append('| '+' | '.join([row['label']]+[f'{row["metrics"][n+"_cross_mae"]:.4f} → {row["metrics"][n+"_w1"]:.4f}' for n in ATTRIBUTES])+' |')
    lines += ['', f'同范围的固定 onset dev=0 基线为 {data["zero_onset_w1"]:.4f}。该确定性分布到人类分布的 W1 与旧平均绝对误差相等。所有方法均数值验证：把同一 note 的预测与参考一起加回 score 拍内位置，abs onset W1 与 onset dev W1 一致。', '',
        '**相同三样本数的人类分布对照：不与自身比较。**', '',
        f'要取 3 个人类演奏组成预测分布且参考侧仍非空，只能使用 H≥4 的 **{data["human_matched"]["works"]} works、{data["human_matched"]["human_performances"]} human performances**。穷举每首作品所有三人子集，将剩余 H−3 人作为参考；PT-1/A2 的三个生成也对完全相同的剩余参考计算 W1。共 {data["human_matched"]["split_count"]} 个互不重叠的人类预测/参考划分，先在作品内平均划分，再 macro-average works。', '',
        '| 同一 H≥4 子集 | Onset dev 分布 W1 ↓ | Duration log-ratio 分布 W1 ↓ | Velocity 分布 W1 ↓ |',
        '|---|---:|---:|---:|']
    for identifier,label in [('human','3 Human → 其余 Human'),('pt_base','PT-1：3 次生成'),('new_A2','A2：3 次生成')]:
        m=data['human_matched']['metrics'][identifier]
        lines.append('| '+' | '.join([label]+[f'{m[n+"_w1"]:.4f}' for n in ATTRIBUTES])+' |')
    lines += ['', '这是有限样本的人类参照，不是零误差下界。当前只有三个生成样本，增加采样数可以改善分布估计，但不保证数值单调下降；本节没有重新生成 greedy 输出，样本中位数也不能冒充 greedy。逐音符分布 W1 评价各音符的边缘分布，不直接评价跨音符的相关结构或完整演奏风格一致性。', '',
        '[全部 38 项、逐作品 W1、配对 bootstrap 与人类分布对照](../analysis/exp1117_beat_median/result_analysis/distribution_w1/metrics.json) · [完整 CSV](../analysis/exp1117_beat_median/result_analysis/distribution_w1/metrics.csv) · [计算脚本](../script/evaluate_exp1117_distribution_w1.py)']
    (OUTPUT/'distribution_w1.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(2)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    references, reference_path = load_current_references()
    baseline._REFERENCES = references
    methods = baseline.discover_methods()
    previous = json.loads((OUTPUT/'analysis.json').read_text())
    methods += [{k:r[k] for k in ('id','family','label','mask','representation','source')}
                for r in previous['results'] if r['id'].startswith('new_')]
    assert len(methods)==38 and len({m['id'] for m in methods})==38
    extracted = []
    for method in methods:
        print('Extract',method['id'],flush=True)
        extracted.append(baseline.extract_method(method))
    masks = baseline.common_masks(extracted, references)
    np.savez_compressed(DESTINATION/'common_masks.npz',**{
        f'{work}_{name}':value for work,mask in enumerate(masks) for name,value in mask.items()})
    predictions = {}
    for entry in extracted:
        predictions[entry['method']['id']] = [{name: np.stack([entry['features'][(work,s)][name]
            for s in range(3)]) for name in ATTRIBUTES} for work in range(len(references))]
    rng = np.random.default_rng(20261119)
    bootstrap_indices = rng.integers(0, 19, size=(10000, 19))
    results = []
    for entry in extracted:
        identifier=entry['method']['id']
        per_work=[]
        for work,(reference,mask) in enumerate(zip(references,masks)):
            values=distribution_metrics(predictions[identifier][work],reference,mask)
            per_work.append(dict(work=reference['work'],humans=reference['humans'],**values))
        fields=tuple(sorted(per_work[0].keys()-{'work','humans'}))
        groups={}
        for group,minimum in [('all_19',1),('multi_human_12',2),('human_ge4_8',4),('human_ge6_5',6)]:
            chosen=[r for r in per_work if r['humans']>=minimum]
            groups[group]=dict(works=len(chosen),metrics={n:float(np.mean([r[n] for r in chosen])) for n in fields})
        intervals={n:np.percentile(np.array([r[n] for r in per_work])[bootstrap_indices].mean(axis=1),[2.5,97.5]).tolist()
                   for n in fields if n.endswith('_w1')}
        results.append(dict(**entry['method'], metrics=groups['all_19']['metrics'], bootstrap_95ci=intervals,
                            groups=groups, per_work=per_work, sources=entry['sources']))
        print(identifier,groups['all_19']['metrics'],flush=True)
    by_id={r['id']:r for r in results}
    comparisons={}
    for name in ATTRIBUTES:
        field=name+'_w1'
        differences=np.array([a[field]-b[field] for a,b in zip(by_id['new_A2']['per_work'],by_id['pt_base']['per_work'])])
        comparisons[field]=dict(a2_minus_pt=float(differences.mean()),
            paired_work_bootstrap_95ci=np.percentile(differences[bootstrap_indices].mean(axis=1),[2.5,97.5]).tolist(),
            a2_better_works=int((differences<0).sum()))
    human=matched_human_baseline(references,masks,predictions)
    result=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        protocol=dict(reference=str(reference_path),reference_sha256=baseline.sha256(reference_path),
            samples=3, methods=38, works=19, human_performances=82,
            observation='Common onset-group mean + linear interpolation from saved predicted note onsets; original score-note identities retained',
            normalization='New human beat-median labels; saved model predictions unchanged',
            wasserstein='Exact per-score-note W1 between uniform empirical distributions of 3 generated samples and H human references, then macro-average notes within work and works',
            legacy='For audit only: mean absolute difference over every generated/human cross pair on exactly the same mask',
            uncertainty='10000 paired work bootstrap resamples conditional on fixed samples; no sampling-draw std for a pooled 3-sample distribution',
            invariance='Verified per-note beat-relative abs onset and onset dev W1 equality after adding same score position'),
        coverage=dict(notes=sum(r['notes'] for r in references),
                      **{n:sum(int(m[n].sum()) for m in masks) for n in ('beat','onset','duration','velocity')}),
        per_work_coverage=[dict(work=r['work'],humans=r['humans'],**{n:int(v.sum()) for n,v in m.items()})
                           for r,m in zip(references,masks)],
        results=results,a2_vs_pt=comparisons,human_matched=human,
        onset_controls=onset_controls(references,masks,predictions),
        zero_onset_w1=float(np.mean([np.abs(r['human_onset'][:,m['onset']]).mean() for r,m in zip(references,masks)])))
    (DESTINATION/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (DESTINATION/'metrics.csv').open('w',newline='') as stream:
        fields=['id','family','label','mask','representation']+[n+s for n in ATTRIBUTES for s in ('_w1','_cross_mae')]
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({**{k:r[k] for k in fields[:5]},**r['metrics']})
    render(result)
    render_zero_onset(result)
    print('A2 vs PT:',json.dumps(comparisons,indent=2),flush=True)
    print('Human matched:',json.dumps(human['metrics'],indent=2),flush=True)


if __name__=='__main__':
    main()
