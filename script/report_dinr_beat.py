#!/usr/bin/env python3
"""Refresh Chinese beat experiment report, CSV and grouped 2-row loss plots."""
import argparse
import csv
import json
from pathlib import Path
import sys
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
GROUPS={'A':['A1','A2','A3'],'B':['B1','B2'],'C':['C2','C3'],
        'D':['D1','D2','D3','D4','D5']}


def history(directory):
    path=directory/'loss_history.jsonl'
    result=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:result.append(json.loads(line))
            except json.JSONDecodeError:pass
    return result


def fmt(values):
    values=[v for v in values if v is not None and np.isfinite(v)]
    if not values:return '—'
    if len(values)==1:return f'{values[0]:.3f}（n=1）'
    return f'{np.mean(values):.3f} ± {np.std(values,ddof=1):.3f}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,default=ROOT.parent/'results/exp1115_beat')
    args=parser.parse_args()
    manifest=json.loads((args.campaign/'manifest.json').read_text())
    output=ROOT/'analysis/exp1115_beat'
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,4,figsize=(21,8),squeeze=False,constrained_layout=True)
    colors=plt.get_cmap('tab10')
    seed_styles={42:'-',43:'--',44:':',45:'-',46:'--',47:':'}
    by_setting={}
    all_rows=[]
    for job in manifest['jobs']:
        by_setting.setdefault(job['setting'],[]).append(job)
    for col,(group,settings) in enumerate(GROUPS.items()):
        observed=[]
        for color_index,setting in enumerate(settings):
            for job in by_setting.get(setting,[]):
                rows=history(Path(job['run_dir']))
                for row,key in enumerate(('loss','eval_loss')):
                    selected=[r for r in rows if key in r and r.get('epoch',0)>0]
                    if selected:
                        x=[r['epoch'] for r in selected]; y=[r[key] for r in selected]
                        axes[row,col].plot(x,y,color=colors(color_index),alpha=.8,
                           linestyle=seed_styles[job['seed']],linewidth=1,
                           label=f"{setting} s{job['seed']}")
                        observed.extend(x)
                        if row:
                            best=int(np.argmin(y))
                            axes[row,col].scatter(x[best],y[best],marker='*',s=50,color=colors(color_index))
        for row in range(2):
            ax=axes[row,col]
            ax.set(title=f'Group {group} — '+('Train' if row==0 else 'Valid'),
                   xlabel='Epoch',ylabel='Summed head CE')
            ax.set_xlim(0,max(observed) if observed else 1)
            ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=7,ncol=2,frameon=False)
    tmp=output/'loss_groups.tmp.png'
    fig.savefig(tmp,dpi=150)
    tmp.replace(output/'loss_groups.png')
    plt.close(fig)
    summary=[]
    for setting,jobs in by_setting.items():
        record=dict(setting=setting,trained=0,tested=0)
        records=[]
        for job in jobs:
            directory=Path(job['run_dir'])
            row=dict(setting=setting,seed=job['seed'],status=job['status'])
            train=directory/'training_summary.json'
            valid=directory/'validation/metrics.json'
            test=directory/'test/metrics.json'
            if train.exists():
                record['trained']+=1
                data=json.loads(train.read_text())
                row.update(best_valid_loss=data['best_valid_loss'],stop_epoch=data['stopped_epoch'])
            if valid.exists():
                data=json.loads(valid.read_text())
                row.update({'valid_'+k:v for k,v in data['metrics'].items()})
            if test.exists():
                record['tested']+=1
                data=json.loads(test.read_text())
                for mode,short in [('teacher_forced','tf'),('autoregressive','ar')]:
                    draws=[r['metrics'] for r in data['rounds'] if r['mode']==mode]
                    for key in draws[0]:
                        row[short+'_'+key]=float(np.mean([d[key] for d in draws]))
                        row[short+'_'+key+'_sampling_std']=float(np.std([d[key] for d in draws],ddof=1))
            records.append(row)
            all_rows.append(row)
        record['values']={key:[r.get(key) for r in records] for key in set().union(*(r.keys() for r in records))
                          if key not in ('setting','seed','status')}
        summary.append(record)
    keys=['setting','seed','status']+sorted(set().union(*(r.keys() for r in all_rows))-{'setting','seed','status'})
    temporary=output/'metrics.tmp.csv'
    with temporary.open('w',newline='') as f:
        writer=csv.DictWriter(f,keys,lineterminator='\n');writer.writeheader();writer.writerows(all_rows)
    temporary.replace(output/'metrics.csv')
    completed=sum(j['status']=='completed' for j in manifest['jobs'])
    failed=[j for j in manifest['jobs'] if j['status']=='failed']
    active=[j for j in manifest['jobs'] if j['status'] in ('training','validating','testing')]
    lines=['# exp1115：Beat / Note 双级时间表示实验','',
           '## 1. 协议与进度','',
           '[设计文档](design1115_beat.md)；[冻结数据审计](design1115_beat_data_audit.json)。A0 已有结果，不重跑。',
           '',f"更新时间：{manifest.get('updated_at')}。完整计划为 12 项设置 × 3 个训练种子 = 36 次训练；当前已建立 {len(manifest['jobs'])} 个任务，完成训练及 TF/AR 评估 {completed} 个。后续配置按验证集依赖分阶段加入。",'',
           '每次从随机权重开始；最大 16 epoch，每半 epoch 验证一次，patience=4（连续四次验证未改善停止），仅保留最低 valid CE 的 checkpoint。AdamW，起始学习率 3e-4，线性衰减，有效 batch=32（8×4）。decoder 训练时以 0.5 概率遮住整个历史事件向量，BOS/PAD 不遮；验证和生成关闭遮蔽。',
           '',
           '每个设置使用三个独立训练种子，且每个最佳模型进行三轮采样（temperature=1、top-p=0.9）。表中先在每个训练种子内平均采样轮次，再对训练种子报告 mean ± sample std，保留三位小数；不足两个已完成训练种子时只展示数值与 n，不伪造标准差。CSV 另存各模型的采样标准差。',
           '',
           'C/D 的自适应配置只使用 valid 贪心预测的物理误差：四项（beat duration log MAE、note duration log MAE、onset ratio MAE、velocity MAE/127）的均值，再跨三个训练种子平均；test 不参与选择。checkpoint 仍按总 valid CE 选择。',
           '',
           '测试为 19 首作品、82 个参考演奏（以冻结审计为准）；其中两首 MS-score 沿用旧评估的 MUSICAL51 raw 标签兜底，其余使用 grouped-median 标签。HR 保留固定 human anchor，使用生成后按实际起音时间排序的 MIDI note-index W1；token JS / onset / duration 误差保留 score event 身份，二者不能视为同一个对齐协议。PP 报告原始 W1，不额外构造未校准的 PP HR。',
           '',
           '生成分为不重叠的完整 beat 窗口（最多 512 note），窗口间延续累计 beat 时间，BOS 与 decoder cache 在窗口开始重置。AR 的独立窗口和三轮采样按固定任务顺序以 batch=16 解码，随机流由作品标识固定；原始 token/事件存 NPZ，MIDI 仅用于播放和 legacy 指标。',
           '', '## 2. 分组训练与验证曲线','',
           '四列分别为 A/B/C/D，首行为细粒度 train CE，次行为半 epoch valid CE；星号标记各训练种子的最低 valid 点。每列横轴随该组实际最大 epoch 自动扩展。空列表示依赖尚未满足。','',
           '![Beat DINR 分组 train/valid 曲线](../analysis/exp1115_beat/loss_groups.png)','',
           '## 3. 结果表','',
           '| 设置 | beat target | note target | duration 输入表 | duration head | measure_start | 训练/评估 n | stop ep | best valid CE | valid 物理误差 | TF HR | AR HR |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for row in summary:
        s=manifest['settings'][row['setting']];v=row['values']
        values=[row['setting'],s['beat_duration_target'],s['note_duration_target'],s['duration_input_tables'],
                s['duration_output_head'],str(int(s['measure_start'])),f"{row['trained']}/{row['tested']}",
                fmt(v.get('stop_epoch',[])),fmt(v.get('best_valid_loss',[])),fmt(v.get('valid_selection_score',[])),
                fmt(v.get('tf_pn_hr',[])),fmt(v.get('ar_pn_hr',[]))]
        lines.append('| '+' | '.join(values)+' |')
    lines.extend(['','### 3.1 Token 与物理误差','',
                  '| 设置 | 解码 | beat JS | note dur JS | onset JS | velocity JS | beat log MAE | note dur log MAE | onset ratio MAE | beat anchor MAE (ms) |',
                  '|---|---|---|---|---|---|---|---|---|---|'])
    for row in summary:
        for short in ('tf','ar'):
            v=row['values']
            keys=('beat_duration_js','note_duration_js','onset_js','velocity_js','beat_duration_log_mae',
                  'note_duration_log_mae','onset_mae','beat_anchor_mae_ms')
            lines.append('| '+' | '.join([row['setting'],short.upper()]+[fmt(v.get(short+'_'+k,[])) for k in keys])+' |')
    lines.extend(['','### 3.2 MIDI PN / PP Wasserstein','',
                  '| 设置 | 解码 | IOI PN W1 (ms) | duration PN W1 (ms) | velocity PN W1 | IOI PP W1 (ms) | duration PP W1 (ms) | velocity PP W1 |',
                  '|---|---|---|---|---|---|---|---|'])
    for row in summary:
        for short in ('tf','ar'):
            v=row['values']
            keys=('ioi_pn_w1','duration_pn_w1','velocity_pn_w1','ioi_pp_w1','duration_pp_w1','velocity_pp_w1')
            lines.append('| '+' | '.join([row['setting'],short.upper()]+[fmt(v.get(short+'_'+k,[])) for k in keys])+' |')
    lines.extend(['','[全部逐模型指标 CSV](../analysis/exp1115_beat/metrics.csv)。其中另含 accuracy、signed error、ratio MAE、各属性 HR 及模型内的采样标准差。',
                  '', '## 4. 当前运行与自动依赖决策',''])
    for job in active:
        lines.append(f"- GPU{job.get('gpu')}：`{job['id']}`，{job['status']}，子进程 PID {job.get('child_pid')}。")
    for key,decision in manifest['decisions'].items():
        lines.append(f"- {key}：选取 `{decision['chosen']}`，依据三训练种子 valid 物理误差均值；细分分数记录在 manifest。")
    for job in failed:
        lines.append(f"- **失败** `{job['id']}`：{job.get('error')}")
    if not active and not failed:lines.append('当前没有正在执行的子任务；以 manifest 的总状态为准。')
    lines.extend(['','## 5. 可恢复队列','',
                  '队列入口：[`script/run_dinr_beat_queue.py`](../script/run_dinr_beat_queue.py)。每张卡一次一个任务；训练后依次验证、TF/AR 测试、保存指标和刷新图表。OOM 时减小 microbatch 并保持有效 batch=32；评估失败可从已完成作品恢复。失败会明确记入 manifest，依赖阶段停止，不把缺失实验当成完成。',
                  '', f'运行目录：`{args.campaign.resolve()}`。此目录保存 manifest、每任务日志、best checkpoint、sidecar hashes、原始 token、MIDI 和逐作品指标。',
                  '', '```bash','cd INSPIRE2','python script/run_dinr_beat_queue.py --gpus 0 1 2','```',''])
    if manifest['status']=='completed':lines.extend(['全部 36 次训练及其 TF/AR 评估已完成；请按多种子均值和标准差判断差异。',''])
    elif manifest['status']=='failed':lines.extend(['**队列遇到失败，尚未完成。** 详见 manifest 和失败任务日志。',''])
    doc=ROOT/'docs/exp1115_BEAT.md'
    temp=doc.with_suffix('.md.tmp');temp.write_text('\n'.join(lines));temp.replace(doc)


if __name__=='__main__':main()
