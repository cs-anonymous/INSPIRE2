#!/usr/bin/env python3
"""Separate mean prediction error from across-draw diversity per score note."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from script.analyze_exp1117_beat_median import OUTPUT,load_current_references
from script.evaluate_exp1117_distribution_w1 import ATTRIBUTES,DESTINATION as W1_OUTPUT
import script.evaluate_exp1116_baselines as baseline

DESTINATION=OUTPUT/'mean_variance'


def mean_variance_columns(predictions, humans):
    """Compare sample means; variance is across draws at a fixed note."""
    p,q=np.asarray(predictions,dtype=float),np.asarray(humans,dtype=float)
    if p.ndim!=2 or q.ndim!=2 or p.shape[1]!=q.shape[1] or len(p)<2 or len(q)<1:
        raise ValueError('Expected aligned sample matrices with at least 2 predictions and 1 human')
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Inputs must be finite after masking')
    result=dict(mean_mae=float(np.abs(p.mean(axis=0)-q.mean(axis=0)).mean()),
        signed_mean_error=float((p.mean(axis=0)-q.mean(axis=0)).mean()),
        generated_var=float(np.var(p,axis=0,ddof=1).mean()))
    if len(q)>=2:
        result['human_var']=float(np.var(q,axis=0,ddof=1).mean())
    return result


def render(data):
    models={r['id']:r for r in data['results']}
    lines=['### 7.9 2026-11-19 改用均值 MAE＋生成 VAR 分开报告', '',
        '为提高可解释性，本节分开报告“预测中心是否准确”和“生成变化是否足够”。每个 score note 的生成均值来自 3 次生成，人类均值来自同作品 H 个演奏；均值 MAE 是二者差值的绝对值，先逐 note 平均再 macro-average works。它不同于此前每个生成值分别对人类参考计算的 MAE，也不等于先平均整首作品再计算差值。', '',
        'VAR 固定同一 score note，在三次生成之间计算无偏样本方差（ddof=1），再逐 note、work 平均；人类 VAR 也固定同一音符，在不同人类演奏间计算。只在人类 H≥2 的作品上比较两边 VAR，不把单个人类演奏的方差当成零。VAR 单位为属性单位的平方。', '',
        '目标是均值 MAE 更小，并保留与人类相称的生成 VAR。VAR 不应无限增大：随机噪声也能使其增大，合理做法是同时列出 Human VAR 和 Generated VAR / Human VAR；比值 1 表示平均变化幅度匹配，比值大于 1 表示更分散，小于 1 表示更集中。比值为聚合方差之比，不平均各音符方差比，避免人类方差接近零时数值爆炸。', '',
        '**主对照：12 首多人演奏作品、75 个演奏，MAE 与 VAR 使用同一范围。** 使用第 7.7 节 38 项共同有效 mask；仍保留 score note 身份，模型输出、归一化和音符重提取方式均不变。', '']
    for name,title in [('onset','Onset dev（beat fraction）'),('duration','Duration log-ratio'),('velocity','Velocity（MIDI value）')]:
        human=data['human']['multi_human']['metrics'][name+'_human_var']
        lines += [f'**{title}**', '',
            '| 设置 | 均值 MAE ↓ | 生成 VAR | VAR / Human VAR |', '|---|---:|---:|---:|',
            f'| Human 参考 | — | {human:.6g} | 1.000 |']
        for identifier,label in [('pt_base','PT-1'),*[(f'new_{s}',s) for s in ('A1','A2','A3','C1','C2','C3')]]:
            m=models[identifier]['groups']['multi_human']['metrics']
            lines.append(f'| {label} | {m[name+"_mean_mae"]:.5f} | {m[name+"_generated_var"]:.6g} | {m[name+"_var_ratio"]:.3f} |')
        if name=='onset':
            zero=data['zero_onset']['groups']['multi_human']
            lines.append(f'| 固定 onset dev=0 | {zero["mean_mae"]:.5f} | 0 | 0 |')
        lines.append('')
    a2=models['new_A2']['groups']['multi_human']['metrics']
    lines += [f'这组拆分给出更直接的诊断：A2 的 onset VAR 是 Human 的 {a2["onset_var_ratio"]:.1%}，而 duration、velocity VAR 已分别达到 {a2["duration_var_ratio"]:.1%}、{a2["velocity_var_ratio"]:.1%}。因此不能统一要求所有属性 VAR 更大；需要先降低均值误差，再看变化幅度是否与人类匹配。PT-1 三项均值 MAE 均小于 A2。A2 的 velocity 平均有符号偏差为 {a2["velocity_signed_mean_error"]:+.2f} 个 MIDI velocity 单位，提示存在整体偏响；该总体偏差并不代表每首作品、每个音符都偏响。', '',
        '**完整 19 首的均值 MAE：单演奏作品也可以计算均值误差，但不能估计人类 VAR。**', '',
        '| 设置 | Onset dev 均值 MAE ↓ | Duration log-ratio 均值 MAE ↓ | Velocity 均值 MAE ↓ |',
        '|---|---:|---:|---:|']
    for identifier,label in [('pt_base','PT-1'),('new_A2','A2')]:
        m=models[identifier]['groups']['all']['metrics']
        lines.append('| '+' | '.join([label]+[f'{m[n+"_mean_mae"]:.5f}' for n in ATTRIBUTES])+' |')
    lines += [f'| 固定 onset dev=0 | {data["zero_onset"]["groups"]["all"]["mean_mae"]:.5f} | — | — |', '',
        '固定零偏移的均值 MAE 是 abs(mean(human onset dev))，不再是 mean(abs(human onset dev))，因此与第 7.8 节固定零偏移 W1 不相同。人类不同演奏正负偏移可以在均值中抵消，这正是“接近人类平均表达”这个目标的含义。', '',
        '这两个量容易解释，但均值＋方差并不确定整个分布，仍可能遗漏双峰、多峰或跨音符相关性。每个音符仅 3 次生成，样本均值与方差都不稳定，均值 MAE 也会受到采样噪声影响；两者应并列报告，不合成“MAE 减 VAR”的无限奖励。Human 参考均值对自身误差为零只是定义，不是独立 human→human 基线。', '',
        '[全部 38 项、逐作品、分组与定义 JSON](../analysis/exp1117_beat_median/result_analysis/mean_variance/metrics.json) · [CSV](../analysis/exp1117_beat_median/result_analysis/mean_variance/metrics.csv) · [计算脚本](../script/evaluate_exp1117_mean_variance.py)']
    (OUTPUT/'mean_variance.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(2)
    DESTINATION.mkdir(parents=True,exist_ok=True)
    references,reference_path=load_current_references()
    baseline._REFERENCES=references
    w1=json.loads((W1_OUTPUT/'metrics.json').read_text())
    with np.load(W1_OUTPUT/'common_masks.npz') as stored:
        masks=[{name:stored[f'{work}_{name}'].copy() for name in ('beat',*ATTRIBUTES)} for work in range(19)]
    for reference,coverage in zip(references,w1['per_work_coverage']):
        assert reference['work']==coverage['work']
    groups=[('all',1),('multi_human',2),('human_ge4',4),('human_ge6',6)]
    results=[]
    for source in w1['results']:
        method={k:source[k] for k in ('id','family','label','mask','representation','source')}
        print('Compute',method['id'],flush=True)
        entry=baseline.extract_method(method)
        per_work=[]
        for work,(reference,mask) in enumerate(zip(references,masks)):
            metrics={}
            for name in ATTRIBUTES:
                p=np.stack([entry['features'][(work,s)][name][mask[name]] for s in range(3)])
                q=reference['human_'+name][:,mask[name]]
                values=mean_variance_columns(p,q)
                metrics.update({name+'_'+k:v for k,v in values.items()})
            per_work.append(dict(work=reference['work'],humans=reference['humans'],metrics=metrics))
        summary={}
        for group,minimum in groups:
            selected=[r for r in per_work if r['humans']>=minimum]
            fields=set.intersection(*(set(r['metrics']) for r in selected))
            metrics={n:float(np.mean([r['metrics'][n] for r in selected])) for n in sorted(fields)}
            if minimum>=2:
                metrics.update({n+'_var_ratio':metrics[n+'_generated_var']/metrics[n+'_human_var'] for n in ATTRIBUTES})
            summary[group]=dict(works=len(selected),humans=sum(r['humans'] for r in selected),metrics=metrics)
        results.append(dict(**method,groups=summary,per_work=per_work))
    zeros=[]
    for reference,mask in zip(references,masks):
        human=reference['human_onset'][:,mask['onset']]
        values=mean_variance_columns(np.zeros((3,human.shape[1])),human)
        zeros.append(dict(work=reference['work'],humans=len(human),**values))
    zero_groups={group:dict(works=sum(r['humans']>=minimum for r in zeros),
        mean_mae=float(np.mean([r['mean_mae'] for r in zeros if r['humans']>=minimum])),generated_var=0.) for group,minimum in groups}
    human={group:dict(works=results[0]['groups'][group]['works'],
        metrics={n+'_human_var':results[0]['groups'][group]['metrics'][n+'_human_var'] for n in ATTRIBUTES}) for group,minimum in groups if minimum>=2}
    # Human variances are independent of which model was evaluated.
    for result in results:
        for group in human:
            for n,v in human[group]['metrics'].items():assert result['groups'][group]['metrics'][n]==v
    data=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        protocol=dict(reference=str(reference_path),reference_sha256=baseline.sha256(reference_path),
            mask_source=str(W1_OUTPUT/'common_masks.npz'),mask_sha256=baseline.sha256(W1_OUTPUT/'common_masks.npz'),
            methods=38,generated_samples=3,mean_mae='abs(mean over generated draws - mean over human performances) per score note; mean notes within work then macro-average works',
            variance='Unbiased sample variance ddof=1 across draws/human performances at fixed note; mean notes within work then macro-average works',
            primary_scope='12 works/75 human performances with H>=2; same scope for mean MAE and both variances',
            variance_ratio='Ratio of macro-averaged variances, not macro-average of per-note ratios',
            interpretation='Lower mean MAE; generated variance should be compared with human variance rather than maximized without limit; only 3 generated draws'),
        human=human,zero_onset=dict(groups=zero_groups,per_work=zeros),results=results)
    (DESTINATION/'metrics.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    fields=['id','family','label','mask','representation','group','works','humans']+[n+'_'+s for n in ATTRIBUTES for s in ('mean_mae','generated_var','human_var','var_ratio')]
    with (DESTINATION/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for result in results:
            for group,row in result['groups'].items():
                writer.writerow({**{k:result[k] for k in fields[:5]},'group':group,'works':row['works'],'humans':row['humans'],
                    **{k:v for k,v in row['metrics'].items() if k in fields}})
    render(data)
    for identifier in ('pt_base','new_A2'):
        row=next(r for r in results if r['id']==identifier)
        print(identifier,json.dumps(row['groups']['multi_human'],indent=2),flush=True)
    print('zero',zero_groups,flush=True)


if __name__=='__main__':
    main()
