#!/usr/bin/env python3
"""Consolidate score, human and model MAE/VAR; separate self from held-out human."""
from __future__ import annotations

import csv
from datetime import datetime
from itertools import combinations
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from script.analyze_exp1117_beat_median import OUTPUT,load_current_references
from script.evaluate_exp1117_mean_variance import DESTINATION as MEAN_OUTPUT,mean_variance_columns
from script.evaluate_exp1117_distribution_w1 import ATTRIBUTES,DESTINATION as W1_OUTPUT
import script.evaluate_exp1116_baselines as baseline

DESTINATION=OUTPUT/'human_score_mean_variance'
MODEL_ORDER=('pt_base','new_A1','new_A2','new_A3','new_C1','new_C2','new_C3')


def score_attribute_values(reference,name):
    if name in ('onset','duration'):
        # No beat-relative timing deviation; D_score / D_score = 1.
        return np.zeros(reference['notes'],dtype=float)
    return reference['events'][reference['note_mask'],4].copy()


def render(data):
    lines=['### 7.10 2026-11-19 Human、Score MIDI 与模型的 MAE / VAR 统一表', '',
        '**Human 的均值 MAE=0 仅指参考均值对自身。** 把全部人类参考的逐音符均值作为预测，再与同一均值比较，误差必为 0；这一行用于标注目标中心，Human VAR 仍来自真实不同演奏，不代表存在一条确定性 MIDI 同时具有这个均值和人类方差。独立人类演奏或独立三人均值与其余人类比较，MAE 通常不为 0。', '',
        '**主表统一使用 12 首多人作品、75 个演奏。** 每个模型用 3 次生成的均值与全部人类参考均值比较；VAR 为固定同一 score note 的跨演奏/跨生成样本方差（ddof=1），先平均 note，再 macro-average works。Human 参考行列出目标中心 MAE=0 和人类 VAR。', '',
        '**Score MIDI 基线使用完整乐谱属性。** Onset dev=0；duration log-ratio=ln(D_score/D_score)=0；velocity 使用 score MIDI 自身的逐音符 velocity，不设为零或人类均值。重复同一 score 三次，三个属性的跨生成 VAR 均为 0；这不表示乐曲内部所有音符的数值都相同。', '',
        '| 设置 | Onset 均值 MAE ↓ | Onset VAR | Duration 均值 MAE ↓ | Duration VAR | Velocity 均值 MAE ↓ | Velocity VAR |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in data['primary']['rows']:
        m=row['metrics']
        cells=[row['label']]
        for name in ATTRIBUTES:
            cells += [f'{m[name+"_mean_mae"]:.5f}',f'{m[name+"_generated_var"]:.6g}']
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'Onset 单位为 beat fraction；duration 使用自然对数比；velocity 为 MIDI 数值，VAR 单位分别为这些单位的平方。Human 行的 0 不应被当作可独立实现的生成模型性能。', '',
        '**独立 Human 对照：8 首至少四演奏作品、65 个演奏，不能与上面的 12-work 数字直接混比。** 每首作品穷举选择 3 个完整人类演奏，把剩余 H−3 个演奏作为参考。Human MAE 是三人均值对剩余人类均值；所有模型和 Score MIDI 对完全相同的剩余参考算均值 MAE。先平均作品内划分，再 macro-average works，共 811 个划分；预测侧和参考侧没有重复人类演奏。', '',
        '| 设置（同一剩余参考） | Onset 均值 MAE ↓ | Onset VAR | Duration 均值 MAE ↓ | Duration VAR | Velocity 均值 MAE ↓ | Velocity VAR |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in data['heldout']['rows']:
        m=row['metrics'];cells=[row['label']]
        for name in ATTRIBUTES:
            cells += [f'{m[name+"_mean_mae"]:.5f}',f'{m[name+"_generated_var"]:.6g}']
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '穷举三人子集后，人类三样本的无偏 VAR 平均值等于该作品全部人类的无偏 VAR；因此 Human 的 VAR 可作为同样本数参照，但它的独立均值 MAE 不再为零。Score 的 VAR=0 只表示完全确定，不代表其表达优于有变化的人类演奏。', '',
        '[统一主表、独立 Human 对照及逐作品 JSON](../analysis/exp1117_beat_median/result_analysis/human_score_mean_variance/metrics.json) · [CSV](../analysis/exp1117_beat_median/result_analysis/human_score_mean_variance/metrics.csv) · [计算脚本](../script/evaluate_exp1117_human_score_mean_variance.py)']
    (OUTPUT/'human_score_mean_variance.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(2)
    DESTINATION.mkdir(parents=True,exist_ok=True)
    refs,reference_path=load_current_references()
    previous=json.loads((MEAN_OUTPUT/'metrics.json').read_text())
    by_id={r['id']:r for r in previous['results']}
    with np.load(W1_OUTPUT/'common_masks.npz') as stored:
        masks=[{name:stored[f'{work}_{name}'].copy() for name in ('beat',*ATTRIBUTES)} for work in range(19)]
    human_metrics={};score_work=[]
    for reference,mask in zip(refs,masks):
        m={}
        for name in ATTRIBUTES:
            p=np.repeat(score_attribute_values(reference,name)[None,mask[name]],3,axis=0)
            values=mean_variance_columns(p,reference['human_'+name][:,mask[name]])
            m.update({name+'_'+k:v for k,v in values.items()})
        score_work.append(dict(work=reference['work'],humans=reference['humans'],metrics=m))
    for name in ATTRIBUTES:
        human_metrics[name+'_mean_mae']=0.
        human_metrics[name+'_generated_var']=previous['human']['multi_human']['metrics'][name+'_human_var']
    score_metrics={key:float(np.mean([r['metrics'][key] for r in score_work if r['humans']>=2]))
        for key in sorted(set.intersection(*(set(r['metrics']) for r in score_work if r['humans']>=2)))}
    primary=[dict(id='human_reference',label='Human（参考中心，对自身）',metrics=human_metrics),
             dict(id='score_midi',label='Score MIDI',metrics=score_metrics)]
    for identifier in MODEL_ORDER:
        primary.append(dict(id=identifier,label=by_id[identifier]['label'].replace('New BEAT ','') ,
                            metrics=by_id[identifier]['groups']['multi_human']['metrics']))

    baseline._REFERENCES=refs
    predictions={}
    for identifier in MODEL_ORDER:
        method={k:by_id[identifier][k] for k in ('id','family','label','mask','representation','source')}
        extracted=baseline.extract_method(method)
        predictions[identifier]={work:{name:np.stack([extracted['features'][(work,s)][name][masks[work][name]] for s in range(3)])
            for name in ATTRIBUTES} for work,r in enumerate(refs) if r['humans']>=4}
        print('Loaded',identifier,flush=True)
    heldout_work=[]
    for work,(reference,mask) in enumerate(zip(refs,masks)):
        if reference['humans']<4:
            continue
        splits=list(combinations(range(reference['humans']),3))
        accum={identifier:{name+'_'+s:[] for name in ATTRIBUTES for s in ('mean_mae','generated_var')}
               for identifier in ('human_heldout','score_midi',*MODEL_ORDER)}
        human={name:reference['human_'+name][:,mask[name]] for name in ATTRIBUTES}
        for selected in splits:
            remaining=np.setdiff1d(np.arange(reference['humans']),selected)
            for name in ATTRIBUTES:
                q=human[name][remaining]
                candidates=dict(human_heldout=human[name][list(selected)],
                    score_midi=np.repeat(score_attribute_values(reference,name)[None,mask[name]],3,axis=0),
                    **{identifier:predictions[identifier][work][name] for identifier in MODEL_ORDER})
                for identifier,p in candidates.items():
                    values=mean_variance_columns(p,q)
                    for suffix in ('mean_mae','generated_var'):
                        accum[identifier][name+'_'+suffix].append(values[suffix])
        metrics={identifier:{key:float(np.mean(values)) for key,values in attrs.items()} for identifier,attrs in accum.items()}
        for name in ATTRIBUTES:
            np.testing.assert_allclose(metrics['human_heldout'][name+'_generated_var'],np.var(human[name],axis=0,ddof=1).mean(),rtol=1e-12,atol=1e-12)
        heldout_work.append(dict(work=reference['work'],humans=reference['humans'],splits=len(splits),metrics=metrics))
    heldout_rows=[]
    for identifier,label in [('human_heldout','Human（3 人对其余人）'),('score_midi','Score MIDI'),
                              *[(r['id'],r['label']) for r in primary[2:]]]:
        metrics={key:float(np.mean([r['metrics'][identifier][key] for r in heldout_work]))
                 for key in heldout_work[0]['metrics'][identifier]}
        heldout_rows.append(dict(id=identifier,label=label,metrics=metrics))
    result=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        protocol=dict(reference=str(reference_path),reference_sha256=baseline.sha256(reference_path),
            mask_sha256=baseline.sha256(W1_OUTPUT/'common_masks.npz'),
            primary='Human reference centre vs itself = zero MAE by construction; models and score vs all human means; H>=2, 12 works/75 humans',
            heldout='All disjoint 3-human vs H-3 subsets; models and score compared to same heldout reference means; H>=4, 8 works/65 humans',
            score='Score onset dev=0; note duration log-ratio=0; score MIDI velocity; all across-draw variances=0',
            variance='Per-note sample variance ddof=1 across samples, mean notes within work then macro-average works',
            mean_mae='abs(predicted sample mean - human reference sample mean), mean notes within work then macro-average works'),
        primary=dict(works=12,human_performances=75,rows=primary),
        heldout=dict(works=len(heldout_work),human_performances=sum(r['humans'] for r in heldout_work),
                     splits=sum(r['splits'] for r in heldout_work),rows=heldout_rows,per_work=heldout_work),
        score_per_work=score_work)
    (DESTINATION/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    fields=['scope','id','label']+[n+'_'+s for n in ATTRIBUTES for s in ('mean_mae','generated_var')]
    with (DESTINATION/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for scope in ('primary','heldout'):
            for row in result[scope]['rows']:
                writer.writerow(dict(scope=scope,id=row['id'],label=row['label'],**{k:v for k,v in row['metrics'].items() if k in fields}))
    render(result)
    print(json.dumps({scope:result[scope]['rows'][:4] for scope in ('primary','heldout')},indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
