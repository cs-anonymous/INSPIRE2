#!/usr/bin/env python3
"""Evaluate all saved methods with two pairwise-human curve correlations and 3 MAEs."""
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

DESTINATION=OUTPUT/'five_main_metrics'
SETTINGS=('A1','A2','A3','C1','C2','C3')


def beat_mean_curve(note_values, membership, note_mask, beat_count):
    values=np.asarray(note_values,dtype=np.float64)
    groups=np.asarray(membership,dtype=np.int64)
    mask=np.asarray(note_mask,dtype=bool)
    if values.shape!=groups.shape or mask.shape!=groups.shape:
        raise ValueError('Note values, beat memberships and mask must align')
    curve=np.full(beat_count,np.nan,dtype=np.float64)
    selected=mask & np.isfinite(values)
    counts=np.bincount(groups[selected],minlength=beat_count)
    sums=np.bincount(groups[selected],weights=values[selected],minlength=beat_count)
    valid=counts>0
    curve[valid]=sums[valid]/counts[valid]
    return curve


def pearson(x,y,mask):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    use=np.asarray(mask,dtype=bool)&np.isfinite(x)&np.isfinite(y)
    if use.sum()<3 or np.std(x[use])<=1e-12 or np.std(y[use])<=1e-12:
        return None
    return float(np.corrcoef(x[use],y[use])[0,1])


def average_reference_correlations(predicted_curve, human_curves, beat_mask):
    values=[pearson(predicted_curve,row,beat_mask) for row in human_curves]
    valid=[value for value in values if value is not None]
    return dict(mean=float(np.mean(valid)) if valid else None,
        references=len(human_curves), valid_references=len(valid),
        undefined_references=len(human_curves)-len(valid))


def centered_mae(predicted_samples,human_samples):
    p,q=np.asarray(predicted_samples,dtype=float),np.asarray(human_samples,dtype=float)
    if p.ndim!=2 or q.ndim!=2 or p.shape[1]!=q.shape[1]:
        raise ValueError('Prediction and human arrays must align by score note')
    return float(np.abs(p.mean(axis=0)-q.mean(axis=0)).mean())


def score_attribute_values(reference,name):
    if name in ('onset','duration'):
        return np.zeros(int(reference['note_mask'].sum()),dtype=float)
    return np.asarray(reference['events'][reference['note_mask'],4],dtype=float)


def work_metrics(predictions, reference, masks):
    beat_mask=masks['beat']
    note_mask=masks['velocity']
    humans=int(reference['humans'])
    human_tempo=-np.asarray(reference['human_beat_log'],dtype=float)
    human_velocity_curves=np.stack([
        beat_mean_curve(reference['human_velocity'][h],reference['membership'],note_mask,len(beat_mask))
        for h in range(humans)])
    output={}
    for sample in range(3):
        rows={}
        for kind,values in (('tempo',-predictions[sample]['beat_log']),
                            ('velocity',beat_mean_curve(predictions[sample]['velocity'],reference['membership'],note_mask,len(beat_mask)))):
            rows[kind]=average_reference_correlations(values,
                human_tempo if kind=='tempo' else human_velocity_curves,beat_mask)
        output[sample]=rows
    correlations={}
    for kind in ('tempo','velocity'):
        by_sample=[output[s][kind]['mean'] for s in range(3)]
        if any(value is None for value in by_sample):
            valid=[v for v in by_sample if v is not None]
        else:
            valid=by_sample
        correlations[kind]=dict(mean=float(np.mean(valid)) if valid else None,
            std=float(np.std(valid,ddof=1)) if len(valid)>1 else 0.,
            sampling_values=by_sample,
            valid_reference_pairs=sum(output[s][kind]['valid_references'] for s in range(3)),
            possible_reference_pairs=3*humans)
    maes={}
    for name in ATTRIBUTES:
        generated=np.stack([predictions[s][name] for s in range(3)])[:,masks[name]]
        human=reference['human_'+name][:,masks[name]]
        maes[name]=centered_mae(generated,human)
    return dict(correlations=correlations,mae=maes,
                per_sample_human_correlations=output,
                beats=int(beat_mask.sum()),velocity_notes=int(note_mask.sum()))


def score_prediction(reference):
    events=reference['events']
    beat=reference['beat_mask'];note=reference['note_mask']
    return dict(beat_log=np.log(np.maximum(events[beat,2],1.)),
                onset=np.zeros(int(note.sum())),
                duration=np.zeros(int(note.sum())),
                velocity=events[note,4].astype(float))


def model_curves_and_human_baseline(references,masks,extracted):
    """Pairwise human tempo/velocity correlation baseline, excluding self."""
    work_rows=[]
    for work,(reference,mask) in enumerate(zip(references,masks)):
        h=int(reference['humans'])
        if h<2:
            continue
        tempo=-reference['human_beat_log']
        velocity=np.stack([beat_mean_curve(reference['human_velocity'][i],reference['membership'],
            mask['velocity'],len(mask['beat'])) for i in range(h)])
        metrics={}
        for name,curves in (('tempo',tempo),('velocity',velocity)):
            pairs=[pearson(curves[i],curves[j],mask['beat'])
                   for i in range(h) for j in range(h) if i!=j]
            valid=[v for v in pairs if v is not None]
            metrics[name]=dict(mean=float(np.mean(valid)) if valid else None,
                valid_pairs=len(valid),possible_pairs=len(pairs))
        work_rows.append(dict(work=reference['work'],humans=h,metrics=metrics))
    result={}
    for name in ('tempo','velocity'):
        eligible=[r['metrics'][name] for r in work_rows if r['metrics'][name]['mean'] is not None]
        result[name]=dict(mean=float(np.mean([r['mean'] for r in eligible])),
            works=len(eligible),valid_pairs=sum(r['valid_pairs'] for r in eligible),
            possible_pairs=sum(r['possible_pairs'] for r in eligible))
    return dict(metrics=result,per_work=work_rows,
        protocol='For each eligible work, average Pearson across all ordered distinct human-performance pairs, then macro-average works; H>=2; identical common masks as model evaluation')


def matched_human_metrics(references,masks,extracted):
    """Leave one human out; compare the held-out performance and methods to H-1 refs."""
    by_id={entry['method']['id']:entry for entry in extracted}
    human_work_rows=[]
    for work,(reference,mask) in enumerate(zip(references,masks)):
        h=int(reference['humans'])
        if h<2:
            continue
        beat_mask=mask['beat']
        velocity_mask=mask['velocity']
        human_tempo=-reference['human_beat_log']
        human_velocity=np.stack([beat_mean_curve(reference['human_velocity'][i],reference['membership'],
            velocity_mask,len(beat_mask)) for i in range(h)])
        score=score_prediction(reference)
        candidates={'score_midi':[[score,score,score]]}
        for identifier,entry in by_id.items():
            candidates[identifier]=[[entry['features'][(work,s)] for s in range(3)]]
        accum={identifier:{'correlations':{'tempo':[],'velocity':[]},'mae':{name:[] for name in ATTRIBUTES}}
               for identifier in ('human_heldout',*candidates)}
        split_count=0
        for held_index in range(h):
            held=np.asarray([held_index],dtype=int)
            rest=np.setdiff1d(np.arange(h),held)
            split_count+=1
            # Human prediction is one held-out performance; models retain all
            # three generated draws and average their prediction/reference pairs.
            for kind,human_curves in (('tempo',human_tempo),('velocity',human_velocity)):
                for candidate in ('human_heldout',*candidates):
                    if candidate=='human_heldout':
                        p_curves=human_curves[held]
                    elif candidate=='score_midi':
                        score_curve=(-np.log(np.maximum(reference['events'][reference['beat_mask'],2],1.))
                                     if kind=='tempo' else
                                     beat_mean_curve(score['velocity'],reference['membership'],velocity_mask,len(beat_mask)))
                        p_curves=np.repeat(score_curve[None,:],3,axis=0)
                    else:
                        entry=by_id[candidate]
                        p_curves=[]
                        for sample in range(3):
                            feature=entry['features'][(work,sample)]
                            p_curves.append(-feature['beat_log'] if kind=='tempo' else
                                beat_mean_curve(feature['velocity'],reference['membership'],velocity_mask,len(beat_mask)))
                        p_curves=np.stack(p_curves)
                    coefficients=[pearson(curve,human_curves[ref],beat_mask)
                                  for curve in p_curves for ref in rest]
                    valid=[v for v in coefficients if v is not None]
                    if valid:
                        accum[candidate]['correlations'][kind].append(float(np.mean(valid)))
            for name in ATTRIBUTES:
                target=reference['human_'+name]
                held_mae=centered_mae(target[held][:,mask[name]],target[rest][:,mask[name]])
                accum['human_heldout']['mae'][name].append(held_mae)
                for candidate,prediction_rows in candidates.items():
                    if candidate=='score_midi':
                        values=np.repeat(score_attribute_values(reference,name)[None,:],3,axis=0)
                    else:
                        entry=by_id[candidate]
                        values=np.stack([entry['features'][(work,s)][name] for s in range(3)])
                    accum[candidate]['mae'][name].append(centered_mae(values[:,mask[name]],target[rest][:,mask[name]]))
        row_metrics={}
        for candidate,attrs in accum.items():
            row_metrics[candidate]=dict(
                correlations={kind:float(np.mean(values)) if values else None for kind,values in attrs['correlations'].items()},
                mae={name:float(np.mean(values)) if values else None for name,values in attrs['mae'].items()})
        human_work_rows.append(dict(work=reference['work'],humans=h,splits=split_count,metrics=row_metrics))
    identifiers=['human_heldout','score_midi']+[entry['method']['id'] for entry in extracted]
    labels={'human_heldout':'Human：留一演奏','score_midi':'Score MIDI'}
    labels.update({entry['method']['id']:entry['method']['label'] for entry in extracted})
    rows=[]
    for identifier in identifiers:
        rows.append(dict(id=identifier,label=labels[identifier],
            correlations={kind:float(np.mean([r['metrics'][identifier]['correlations'][kind]
                    for r in human_work_rows if r['metrics'][identifier]['correlations'][kind] is not None]))
                for kind in ('tempo','velocity')},
            mae={name:float(np.mean([r['metrics'][identifier]['mae'][name] for r in human_work_rows]))
                for name in ATTRIBUTES}))
    return dict(works=len(human_work_rows),human_performances=sum(r['humans'] for r in human_work_rows),
        splits=sum(r['splits'] for r in human_work_rows),rows=rows,per_work=human_work_rows,
        protocol='For each H>=2 work, hold out one complete human performance and compare it with the other H-1 performances. Score and models use the same H-1 reference subset. Correlations average over every prediction/reference performance pair; MAE compares the held-out note value or model sample mean to the remaining-human note mean. Average held-out performances within work, then macro-average works.')


def render(data):
    rows=data['results']
    lines=['### 7.11 2026-11-21 当前五项主指标：两条曲线相关与三项 MAE', '',
        '范围按最新确认的对照集合：Human、Score MIDI、PT-1、A1–A3、C1–C3。PT-1 与六个新 BEAT 设置复用保存好的预测，不重新训练或采样；使用新 beat-median 人类参考、score-note 对齐和这七个模型的共同有效 mask。此前 38 项的探索性对照不纳入本节。', '',
        '**两项 Pearson 均按逐个人类演奏计算后取平均。** 每个生成结果分别与同一作品的每个人类演奏比较，再在人类演奏间取均值；不先合成人类中位数曲线。先对作品内人类相关取平均，再对作品 macro-average。每个生成样本各算一次，表中相关报告三次 AR 的 mean ± sample std（ddof=1）。Human 基线逐次留出一个完整演奏，与其余演奏比较，排除自身。', '',
        'Pearson 对曲线整体加常数、乘正数不敏感；它衡量逐拍形状，不衡量力度绝对水平。统一力度增减由 Velocity MAE 反映。', '',
        '**Velocity curve 定义：** 按 score onset 所属 beat 聚合该拍新起音音符，取有效 note velocity 的算术平均。Beat 内无有效 note 时该 beat 记缺失，不填零、不延续前值；相关只用相同的有效 beat。主定义用平均值；median/max 作为不同统计量，不与此结果混用。Score MIDI 的 velocity 使用 score note velocity。', '',
        '**三项 MAE 是中心误差。** 对每个 score note，先平均同一 checkpoint 的 3 次生成，再与该 note 的人类演奏均值比较绝对误差，逐 note、逐作品平均。Onset 用拍内 dev，duration 用 ln(D_perf/D_score)，velocity 用 MIDI velocity。它们不是三次 MAE 再平均。表中 MAE 无 sampling std，因为已在三次生成上先求均值。', '',
        f'完整覆盖：{data["coverage"]["works"]} works、{data["coverage"]["human_performances"]} human performances、{data["coverage"]["notes"]:,} score notes；velocity curve 使用 {data["coverage"]["velocity_active_beats"]:,} 个有共同有效 note 的 beat；tempo curve 使用 {data["coverage"]["tempo_beats"]:,} 个有效 beat。', '',
        '| 方法 | Tempo curve Pearson ↑ | Beat velocity curve Pearson ↑ | Onset dev MAE ↓ | Duration log-ratio MAE ↓ | Velocity MAE ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        c=row['correlations'];m=row['mae']
        lines.append('| '+' | '.join([row['label'],
            f'{c["tempo"]["mean"]:.3f} ± {c["tempo"]["std"]:.3f}',
            f'{c["velocity"]["mean"]:.3f} ± {c["velocity"]["std"]:.3f}',
            f'{m["onset"]:.4f}',f'{m["duration"]:.4f}',f'{m["velocity"]:.3f}'])+' |')
    support=data['correlation_work_support']
    lines += ['', f'Score MIDI 使用 onset dev=0、duration log-ratio=0、score MIDI 自身 velocity。其力度曲线按每拍 score note velocity 均值计算。为保证相关数值使用相同作品集合，Tempo corr 在 {support["tempo"]["works"]} 首曲目上计算，Velocity corr 在 {support["velocity"]["works"]} 首曲目上计算；Tempo 的 score curve 在 {19-support["tempo"]["works"]} 首曲目上为常数，因此 Pearson 无定义，这些作品对所有方法一并排除。', '',
        f'**留一演奏 Human 基线。** 取至少有 2 个演奏的 {data["human_matched"]["works"]} 首作品、{data["human_matched"]["human_performances"]} 个演奏。每次留出 1 人作为 Human prediction，其余 H−1 人作为 reference；Score、PT-1、A1–A3、C1–C3 在相同留出划分下与剩余人类比较。共 {data["human_matched"]["splits"]} 个 leave-one-performance-out 拆分。Human 的 MAE 是独立留出演奏对其余演奏均值的误差，不是自比的零。', '',
        '| 方法 | Tempo curve Pearson ↑ | Beat velocity curve Pearson ↑ | Onset dev MAE ↓ | Duration log-ratio MAE ↓ | Velocity MAE ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for row in data['human_matched']['rows']:
        c,m=row['correlations'],row['mae']
        lines.append('| '+' | '.join([row['label'],f'{c["tempo"]:.3f}',f'{c["velocity"]:.3f}',
            f'{m["onset"]:.4f}',f'{m["duration"]:.4f}',f'{m["velocity"]:.3f}'])+' |')
    lines += ['', f'留一基线 Human prediction 只有 1 个演奏；模型 MAE 仍用 3 次生成均值。这两个估计量的样本数不同，Human 行用于表示真实演奏对其他人类的可预测程度，不等于模型对同等样本数基线。Human–Human corr 排除了自身配对；Pearson 无定义的平坦曲线配对会排除并计数。', '',
        f'[所选方法逐作品/逐人类明细与 mask 审计](../analysis/exp1117_beat_median/result_analysis/five_main_metrics/metrics.json) · [CSV](../analysis/exp1117_beat_median/result_analysis/five_main_metrics/metrics.csv) · [计算脚本](../script/evaluate_exp1117_five_main_metrics.py)']
    (OUTPUT/'five_main_metrics.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(2)
    DESTINATION.mkdir(parents=True,exist_ok=True)
    references,reference_path=load_current_references()
    baseline._REFERENCES=references
    methods=[m for m in baseline.discover_methods() if m['id']=='pt_base']
    previous=json.loads((OUTPUT/'analysis.json').read_text())
    methods += [{**{k:r[k] for k in ('id','family','label','mask','representation','source')},
                 'label':r['label'].replace('New BEAT ','')}
        for r in previous['results'] if r['id'] in {f'new_{s}' for s in SETTINGS}]
    if len(methods)!=7 or len({m['id'] for m in methods})!=7:
        raise ValueError(f'Expected PT-1 and six current BEAT settings, got {len(methods)}')
    extracted=[]
    for method in methods:
        print('Extract',method['id'],flush=True)
        extracted.append(baseline.extract_method(method))
    masks=baseline.common_masks(extracted,references)
    np.savez_compressed(DESTINATION/'common_masks.npz',**{
        f'{i}_{name}':values for i,row in enumerate(masks) for name,values in row.items()})
    results=[]
    for entry in extracted:
        per_work=[]
        for work,(reference,mask) in enumerate(zip(references,masks)):
            predictions=[entry['features'][(work,sample)] for sample in range(3)]
            per_work.append(dict(work=reference['work'],humans=reference['humans'],
                                 **work_metrics(predictions,reference,mask)))
        correlations={}
        for kind in ('tempo','velocity'):
            sample_values=[]
            sample_work_counts=[]
            for sample in range(3):
                work_values=[r['per_sample_human_correlations'][sample][kind]['mean'] for r in per_work]
                valid=[v for v in work_values if v is not None]
                sample_values.append(float(np.mean(valid)) if valid else None)
                sample_work_counts.append(len(valid))
            valid=[v for v in sample_values if v is not None]
            correlations[kind]=dict(mean=float(np.mean(valid)) if valid else None,
                std=float(np.std(valid,ddof=1)) if len(valid)>1 else 0.,sampling_values=sample_values,
                works_per_draw=sample_work_counts,works=min(sample_work_counts))
        maes={name:float(np.mean([row['mae'][name] for row in per_work])) for name in ATTRIBUTES}
        results.append(dict(**entry['method'],correlations=correlations,mae=maes,per_work=per_work,
            source_audit=entry['sources']))
        print(entry['method']['id'],correlations,maes,flush=True)

    score_results=[]
    for reference,mask in zip(references,masks):
        score=score_prediction(reference)
        predictions=[score,score,score]
        row=work_metrics(predictions,reference,mask)
        score_results.append(dict(work=reference['work'],humans=reference['humans'],**row))
    score_correlations={}
    for kind in ('tempo','velocity'):
        sample_values=[]
        for sample in range(3):
            valid=[r['per_sample_human_correlations'][sample][kind]['mean'] for r in score_results
                   if r['per_sample_human_correlations'][sample][kind]['mean'] is not None]
            sample_values.append(float(np.mean(valid)) if valid else None)
        finite=[v for v in sample_values if v is not None]
        score_correlations[kind]=dict(mean=float(np.mean(finite)),std=float(np.std(finite,ddof=1)) if len(finite)>1 else 0.,sampling_values=sample_values,
            works_per_draw=[sum(r['per_sample_human_correlations'][s][kind]['mean'] is not None for r in score_results) for s in range(3)],
            works=min(sum(r['per_sample_human_correlations'][s][kind]['mean'] is not None for r in score_results) for s in range(3)))
    score_row=dict(id='score_midi',family='BASELINE',label='Score MIDI',correlations=score_correlations,
        mae={name:float(np.mean([row['mae'][name] for row in score_results])) for name in ATTRIBUTES},
        per_work=score_results)
    all_rows=[score_row,*results]
    coverage=dict(works=len(references),human_performances=sum(r['humans'] for r in references),
        notes=sum(r['notes'] for r in references),tempo_beats=sum(int(r['beat'].sum()) for r in masks),
        velocity_notes=sum(int(r['velocity'].sum()) for r in masks),
        velocity_active_beats=sum(int(np.unique(ref['membership'][mask['velocity']]).size)
                                  for ref,mask in zip(references,masks)),
        common_mask_sha256=baseline.sha256(DESTINATION/'common_masks.npz'))
    correlation_work_support={}
    for kind in ('tempo','velocity'):
        selected=[]
        for work in range(len(references)):
            available=all(row['per_work'][work]['per_sample_human_correlations'][sample][kind]['mean'] is not None
                          for row in all_rows for sample in range(3))
            if available:
                selected.append(work)
        if not selected:
            raise ValueError(f'No shared work support for {kind} correlation')
        correlation_work_support[kind]=dict(works=len(selected),indices=selected,
            work_names=[references[i]['work'] for i in selected])
        for row in all_rows:
            sample_values=[]
            sample_work_counts=[]
            for sample in range(3):
                values=[row['per_work'][i]['per_sample_human_correlations'][sample][kind]['mean'] for i in selected]
                values=[value for value in values if value is not None]
                sample_values.append(float(np.mean(values)) if values else None)
                sample_work_counts.append(len(values))
            valid=[value for value in sample_values if value is not None]
            row['correlations'][kind]=dict(mean=float(np.mean(valid)),
                std=float(np.std(valid,ddof=1)) if len(valid)>1 else 0.,
                sampling_values=sample_values,works_per_draw=sample_work_counts,
                works=min(sample_work_counts))
    human_human=model_curves_and_human_baseline(references,masks,extracted)
    human_matched=matched_human_metrics(references,masks,extracted)
    result=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        protocol=dict(reference=str(reference_path),reference_sha256=baseline.sha256(reference_path),
            methods=len(results),included_methods=['PT-1',*SETTINGS],score_baseline=True,sampling_draws=3,
            tempo_correlation='Each generated negative log beat-duration curve vs every individual normalized human performance curve; Pearson per human; average humans within work, then macro-average works',
            velocity_curve='Arithmetic mean note-on velocity per score-onset beat; empty beats masked; each generated curve vs every individual human curve; Pearson per human; average humans within work, then macro-average works',
            mae='Per score note, absolute difference between mean of three generated values and mean of aligned human values; note mean within work, then macro-average works',
            uncertainties='Correlation std is sample std across 3 saved draws. MAE pools the 3 saved draws before scoring and has no sampling std.',
            human_pairwise='All ordered distinct human-performance pairs; no self-comparison; macro-average eligible works',
            undefined='Pearson pairs with fewer than 3 shared beats or a constant curve are excluded and counted per work/method'),
        coverage=coverage,correlation_work_support=correlation_work_support,
        human_human=human_human,human_matched=human_matched,results=all_rows)
    (DESTINATION/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    fields=['id','family','label','tempo_corr_mean','tempo_corr_std','velocity_corr_mean','velocity_corr_std',
        'onset_mae','duration_mae','velocity_mae','tempo_valid_pairs','tempo_possible_pairs',
        'velocity_valid_pairs','velocity_possible_pairs']
    with (DESTINATION/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in all_rows:
            tempo_valid=tempo_possible=velocity_valid=velocity_possible=0
            for work in row['per_work']:
                for sample in work['per_sample_human_correlations'].values():
                    for kind in ('tempo','velocity'):
                        tempo_valid+=sample[kind]['valid_references'] if kind=='tempo' else 0
                        tempo_possible+=sample[kind]['references'] if kind=='tempo' else 0
                        velocity_valid+=sample[kind]['valid_references'] if kind=='velocity' else 0
                        velocity_possible+=sample[kind]['references'] if kind=='velocity' else 0
            writer.writerow(dict(id=row['id'],family=row['family'],label=row['label'],
                tempo_corr_mean=row['correlations']['tempo']['mean'],tempo_corr_std=row['correlations']['tempo']['std'],
                velocity_corr_mean=row['correlations']['velocity']['mean'],velocity_corr_std=row['correlations']['velocity']['std'],
                onset_mae=row['mae']['onset'],duration_mae=row['mae']['duration'],velocity_mae=row['mae']['velocity'],
                tempo_valid_pairs=tempo_valid,tempo_possible_pairs=tempo_possible,
                velocity_valid_pairs=velocity_valid,velocity_possible_pairs=velocity_possible))
    render(result)
    print('Human-human:',json.dumps(human_human['metrics'],indent=2),flush=True)


if __name__=='__main__':
    main()
