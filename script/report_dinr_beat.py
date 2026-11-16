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


def normalized_train_history(directory, rows):
    """Undo the frozen Trainer 4.54 partial-accumulation log scaling.

    This changes only plotted values, never the original history or gradients.
    With logging_steps=1 the last update sums r mean losses divided by the
    configured accumulation G, so multiply its log by G/r for display.
    """
    config_path = directory / 'run_config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    accumulation = int(config.get('gradient_accumulation_steps', 1))
    windows = config.get('train_windows')
    microbatch = config.get('batch_size')
    remainder = math.ceil(windows / microbatch) % accumulation if windows and microbatch else 0
    factor = accumulation / remainder if remainder and not config.get('normalize_partial_accumulation') else 1.0
    selected, corrections = [], []
    for row in rows:
        if 'loss' not in row or row.get('epoch', 0) <= 0:
            continue
        point = dict(row)
        epoch = float(row['epoch'])
        if factor != 1 and abs(epoch - round(epoch)) < 1e-7:
            point['loss'] = row['loss'] * factor
            corrections.append(dict(epoch=epoch, step=row.get('step'),
                                    raw_loss=row['loss'], plotted_loss=point['loss'],
                                    factor=factor, remaining_microbatches=remainder))
        selected.append(point)
    return selected, corrections


def fmt(values):
    values=[v for v in values if v is not None and np.isfinite(v)]
    if not values:return '—'
    if len(values)==1:return f'{values[0]:.3f}（n=1）'
    return f'{np.mean(values):.3f} ± {np.std(values,ddof=1):.3f}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,default=ROOT.parent/'results/exp1115_beat')
    parser.add_argument('--experiment',choices=['exp1115','exp1116'],default='exp1115')
    args=parser.parse_args()
    new_protocol=args.experiment=='exp1116'
    manifest=json.loads((args.campaign/'manifest.json').read_text())
    output=ROOT/f'analysis/{args.experiment}_beat'
    output.mkdir(parents=True,exist_ok=True)
    groups={k:v for k,v in GROUPS.items() if not new_protocol or k!='C'}
    fig,axes=plt.subplots(2,len(groups),figsize=(5.25*len(groups),8),squeeze=False,constrained_layout=True)
    colors=plt.get_cmap('tab10')
    seed_styles={42:'-',43:'--',44:':',45:'-',46:'--',47:':'}
    by_setting={}
    all_rows=[]
    loss_corrections=[]
    for job in manifest['jobs']:
        by_setting.setdefault(job['setting'],[]).append(job)
    for col,(group,settings) in enumerate(groups.items()):
        observed=[]
        for color_index,setting in enumerate(settings):
            for job in by_setting.get(setting,[]):
                directory=Path(job['run_dir'])
                rows=history(directory)
                train_rows, corrections=normalized_train_history(directory, rows)
                loss_corrections.extend(dict(job=job['id'], **r) for r in corrections)
                for row,key in enumerate(('loss','eval_loss')):
                    selected=train_rows if key == 'loss' else [r for r in rows if key in r and r.get('epoch',0)>0]
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
                   xlabel='Epoch',ylabel='Train CE per optimizer update' if row == 0 else 'Half-epoch valid CE')
            ax.set_xlim(0,max(observed) if observed else 1)
            ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=7,ncol=2,frameon=False)
    tmp=output/'loss_groups.tmp.png'
    fig.savefig(tmp,dpi=150)
    tmp.replace(output/'loss_groups.png')
    plt.close(fig)
    (output/'train_loss_normalization.json').write_text(json.dumps({
        'description': 'Display-only correction of Trainer 4.54 final partial accumulation logs; raw histories and trained weights are unchanged.',
        'corrections': loss_corrections}, indent=2)+'\n')
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
    def display(values,key):
        if new_protocol and key.startswith(('tf_','ar_')):
            means=values.get(key,[]);stds=values.get(key+'_sampling_std',[])
            if len(means)==1 and len(stds)==1:
                return f'{means[0]:.3f} ± {stds[0]:.3f}'
        result=fmt(values.get(key,[]))
        return result.replace('（n=1）','') if new_protocol else result

    lines=['# exp1115：Beat / Note 双级时间表示实验','',
           '## 1. 协议与进度','',
           '[设计文档](design1115_beat.md)；[冻结数据审计](design1115_beat_data_audit.json)。A0 已有结果，不重跑。',
           '',f"更新时间：{manifest.get('updated_at')}。完整计划为 12 项设置 × 3 个训练种子 = 36 次训练；当前已建立 {len(manifest['jobs'])} 个任务，完成训练及 TF/AR 评估 {completed} 个。后续配置按验证集依赖分阶段加入。",'',
           '每次从随机权重开始；最大 16 epoch，每半 epoch 验证一次，patience=4（连续四次验证未改善停止），仅保留最低 valid CE 的 checkpoint。AdamW，起始学习率 3e-4，线性衰减，有效 batch=32（8×4）。decoder 训练时以 0.5 概率遮住整个历史 BEAT/NOTE 事件向量，BOS 不遮；padding 不参与有效预测和 loss，验证和生成关闭遮蔽。',
           '',
           '每个设置使用三个独立训练种子，且每个最佳模型进行三轮采样（temperature=1、top-p=0.9）。表中先在每个训练种子内平均采样轮次，再对训练种子报告 mean ± sample std，保留三位小数；不足两个已完成训练种子时只展示数值与 n，不伪造标准差。CSV 另存各模型的采样标准差。',
           '',
           'C/D 的自适应配置只使用 valid 贪心预测的物理误差：四项（beat duration log MAE、note duration log MAE、onset ratio MAE、velocity MAE/127）的均值，再跨三个训练种子平均；test 不参与选择。checkpoint 仍按总 valid CE 选择。',
           '',
           '测试为 19 首作品、82 个参考演奏（以冻结审计为准）；其中两首 MS-score 沿用旧评估的 MUSICAL51 raw 标签兜底，其余使用 grouped-median 标签。HR 保留固定 human anchor，使用生成后按实际起音时间排序的 MIDI note-index W1；token JS / onset / duration 误差保留 score event 身份，二者不能视为同一个对齐协议。PP 报告原始 W1，不额外构造未校准的 PP HR。',
           '',
           '生成分为不重叠的完整 beat 窗口（最多 512 note），窗口间延续累计 beat 时间，BOS 与 decoder cache 在窗口开始重置。AR 的独立窗口和三轮采样按固定任务顺序以 batch=16 解码，随机流由作品标识固定；原始 token/事件存 NPZ，MIDI 仅用于播放和 legacy 指标。',
           '', '## 2. 分组训练与验证曲线','',
           '四列分别为 A/B/C/D，首行为细粒度 train CE，次行为半 epoch valid CE；星号标记各训练种子的最低 valid 点。每列横轴随该组实际最大 epoch 自动扩展。train 曲线保留 epoch 边界点，并按实际累积 microbatch 数恢复其显示尺度；原始日志不修改，详见第 6.1 节。','',
           '![Beat DINR 分组 train/valid 曲线](../analysis/exp1115_beat/loss_groups.png)','',
           '## 3. 结果表','',
           '| 设置 | beat target | note target | duration 输入表 | duration head | measure_start | 训练/评估 n | stop ep | best valid CE | valid 物理误差 | TF HR | AR HR |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for row in summary:
        s=manifest['settings'][row['setting']];v=row['values']
        values=[row['setting'],s['beat_duration_target'],s['note_duration_target'],s['duration_input_tables'],
                s['duration_output_head'],str(int(s['measure_start'])),f"{row['trained']}/{row['tested']}",
                display(v,'stop_epoch'),display(v,'best_valid_loss'),display(v,'valid_selection_score'),
                display(v,'tf_pn_hr'),display(v,'ar_pn_hr')]
        lines.append('| '+' | '.join(values)+' |')
    lines.extend(['','### 3.1 Token 与物理误差','',
                  '| 设置 | 解码 | beat JS | note dur JS | onset JS | velocity JS | beat log MAE | note dur log MAE | onset ratio MAE | beat anchor MAE (ms) |',
                  '|---|---|---|---|---|---|---|---|---|---|'])
    for row in summary:
        for short in ('tf','ar'):
            v=row['values']
            keys=('beat_duration_js','note_duration_js','onset_js','velocity_js','beat_duration_log_mae',
                  'note_duration_log_mae','onset_mae','beat_anchor_mae_ms')
            lines.append('| '+' | '.join([row['setting'],short.upper()]+[display(v,short+'_'+k) for k in keys])+' |')
    lines.extend(['','### 3.2 MIDI PN / PP Wasserstein','',
                  '| 设置 | 解码 | IOI PN W1 (ms) | duration PN W1 (ms) | velocity PN W1 | IOI PP W1 (ms) | duration PP W1 (ms) | velocity PP W1 |',
                  '|---|---|---|---|---|---|---|---|'])
    for row in summary:
        for short in ('tf','ar'):
            v=row['values']
            keys=('ioi_pn_w1','duration_pn_w1','velocity_pn_w1','ioi_pp_w1','duration_pp_w1','velocity_pp_w1')
            lines.append('| '+' | '.join([row['setting'],short.upper()]+[display(v,short+'_'+k) for k in keys])+' |')
    lines.extend(['','[全部逐模型指标 CSV](../analysis/exp1115_beat/metrics.csv)。其中另含 accuracy、signed error、ratio MAE、各属性 HR 及模型内的采样标准差。',
                  '', '## 4. 当前运行与自动依赖决策',''])
    for job in active:
        lines.append(f"- GPU{job.get('gpu')}：`{job['id']}`，{job['status']}，子进程 PID {job.get('child_pid')}。")
    for key,decision in manifest['decisions'].items():
        lines.append(f"- {key}：选取 `{decision['chosen']}`，依据 valid 物理误差（{'单次训练' if new_protocol else '三训练种子均值'}）；细分分数记录在 manifest。")
    for job in failed:
        lines.append(f"- **失败** `{job['id']}`：{job.get('error')}")
    if not active and not failed:lines.append('当前没有正在执行的子任务；以 manifest 的总状态为准。')
    if manifest['status']=='paused':
        paused=[j['id'] for j in manifest['jobs'] if j['status']=='paused']
        lines.extend(['', '**队列已按要求暂停，尚未全部完成。** 已完成 30 次训练及评估；暂停中的任务：'+', '.join(f'`{j}`' for j in paused)+'；D5 尚未建立。此轮仅分析和更新报告，没有恢复训练。'])
    lines.extend(['','## 5. 可恢复队列','',
                  '队列入口：[`script/run_dinr_beat_queue.py`](../script/run_dinr_beat_queue.py)。每张卡一次一个任务；训练后依次验证、TF/AR 测试、保存指标和刷新图表。OOM 时减小 microbatch 并保持有效 batch=32；评估失败可从已完成作品恢复。失败会明确记入 manifest，依赖阶段停止，不把缺失实验当成完成。',
                  '', f'运行目录：`{args.campaign.resolve()}`。此目录保存 manifest、每任务日志、best checkpoint、sidecar hashes、原始 token、MIDI 和逐作品指标。',
                  '', '```bash','cd INSPIRE2','python script/run_dinr_beat_queue.py --gpus 0 1 2','```',''])
    if manifest['status']=='completed':lines.extend(['全部 36 次训练及其 TF/AR 评估已完成；请按多种子均值和标准差判断差异。',''])
    elif manifest['status']=='failed':lines.extend(['**队列遇到失败，尚未完成。** 详见 manifest 和失败任务日志。',''])
    if new_protocol:
        replacement=[]
        for line in lines:
            if line.startswith('# exp1115'):
                line='# exp1116：512 总事件定长 Beat / Note 实验'
            elif line.startswith('[设计文档]'):
                line='沿用 [1115 设计](design1115_beat.md) 的数值表示；本轮修改以本文为准。[固定切分数据审计](exp1116_BEAT_data_audit.json)。1115 队列保持暂停。'
            elif line.startswith('更新时间：'):
                line=f"更新时间：{manifest.get('updated_at')}。完整计划为 A/B/D 共 10 项设置、每项一次训练；当前已建立 {len(manifest['jobs'])} 个任务，训练及 TF/AR 评估完成 {completed} 个。队列状态：`{manifest['status']}`。C 组不运行。"
            elif line.startswith('每次从随机权重开始'):
                line='每次随机初始化，不加载 PT 或 1115 checkpoint；最大 16 epoch，每半 epoch 验证，patience=4，仅保留最低 valid CE checkpoint。AdamW，学习率 3e-4 线性衰减，有效 batch=32（固定 16×2），BF16。BEAT/NOTE 合计每 512 事件直接切分，步长 512、无重叠，允许 NOTE 开头或拍内切分；每条演奏独立切分，尾块 padding 到 512。encoder/decoder 训练张量长度均为 512。decoder 历史 BEAT/NOTE 仍各以 0.5 概率整向量 mask，验证/生成不 mask。末尾不足两个 microbatch 的更新按实际数量归一化梯度和 loss，避免 1115 的 epoch 边界减半。'
            elif line.startswith('每个设置使用三个独立训练种子'):
                line='每项仅训练一次；普通配置 seed=42，D3/D5 延续独立复验用途使用 seed=45，但每项仍只有一次训练。最佳模型做三轮采样（temperature=1、top-p=0.9）；所有测试指标按三轮采样报告 mean ± sample std，三位小数。这不是训练随机性的标准差；stop epoch、valid CE 与 valid 物理误差为单次训练数值。'
            elif line.startswith('C/D 的自适应配置'):
                line='D 组依赖只使用单次训练的 valid 贪心物理误差：beat duration log MAE、note duration log MAE、onset ratio MAE、velocity MAE/127 四项均值。缺少某类事件的尾窗不伪造该属性指标；跨含该类事件的窗口平均后再合成分数。test 不参与设置选择；checkpoint 仍按 valid CE 选择。'
            elif line.startswith('生成分为不重叠的完整 beat 窗口'):
                line='测试也按每条演奏每 512 个总事件直接切分，尾部 padding；每窗重置 BOS/cache，允许 NOTE 开头。预测后先按原始事件顺序拼回完整作品，再累计 beat duration、重建 note 起音；不会把 NOTE-first 窗口误当成一个新 beat。AR 任务按 batch=16 解码，保留三轮原始 token NPZ、MIDI 和逐作品指标。'
            elif line.startswith('四列分别为'):
                line='三列分别为 A/B/D，两行为每个 optimizer update 的 train CE 与半 epoch valid CE。横轴按各组实际最大 epoch 展开；星号为最低 valid CE。'
            elif line.startswith('![Beat DINR') or 'analysis/exp1115_beat/' in line:
                line=line.replace('exp1115_beat','exp1116_beat')
            elif line.startswith('队列入口：'):
                line='队列入口：[`run_exp1116_beat_queue.py`](../script/run_exp1116_beat_queue.py)。GPU0/1/2 各运行一个任务；训练后依次完成验证、三轮 TF/AR 评估并刷新报告。保持 16×2，不自动改变 batch；失败明确记入 manifest，依赖阶段停止。'
            elif line=='python script/run_dinr_beat_queue.py --gpus 0 1 2':
                line='python script/run_exp1116_beat_queue.py --gpus 0 1 2'
            elif line.startswith('全部 36 次训练'):
                line='全部 10 次训练及其三轮 TF/AR 评估已完成。'
            replacement.append(line)
        lines=replacement
    doc=ROOT/f'docs/{args.experiment}_BEAT.md'
    analysis = ''
    if doc.exists():
        previous = doc.read_text()
        marker = '\n## 6.'
        if marker in previous:
            analysis = previous[previous.index(marker):]
    temp=doc.with_suffix('.md.tmp');temp.write_text(('\n'.join(lines) + analysis).rstrip() + '\n');temp.replace(doc)


if __name__=='__main__':main()
