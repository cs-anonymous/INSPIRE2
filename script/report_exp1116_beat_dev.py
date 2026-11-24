#!/usr/bin/env python3
"""Write the corrected six-setting report to docs/exp1116_BEAT.md."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from script.evaluate_exp1117_beat_dev import evaluate_run

SETTING_ORDER = ('A1', 'A2', 'A3', 'B1', 'B2', 'D4')
GROUPS = {'A': ('A1', 'A2', 'A3'), 'B': ('B1', 'B2'), 'D': ('D4',)}
METRICS = ('tempo_curve_pearson', 'onset_w1', 'duration_w1', 'velocity_w1')


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def replace_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def report(campaign, output, doc):
    manifest = read_json(campaign / 'manifest.json')
    if manifest is None:
        raise FileNotFoundError(campaign / 'manifest.json')
    output.mkdir(parents=True, exist_ok=True)
    jobs = {row['setting']: row for row in manifest['jobs']}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8), constrained_layout=True)
    colors = plt.get_cmap('tab10')
    for col, (group, settings) in enumerate(GROUPS.items()):
        maximum_epoch = 0.0
        for index, setting in enumerate(settings):
            job = jobs.get(setting)
            if not job:
                continue
            history_path = Path(job['run_dir']) / 'loss_history.jsonl'
            history = []
            if history_path.exists():
                for line in history_path.read_text().splitlines():
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            for row, key in enumerate(('loss', 'eval_loss')):
                points = [x for x in history if key in x and x.get('epoch', 0) > 0]
                if not points:
                    continue
                xs, ys = [x['epoch'] for x in points], [x[key] for x in points]
                axes[row, col].plot(xs, ys, color=colors(index), lw=1,
                                    alpha=.8, label=setting)
                maximum_epoch = max(maximum_epoch, max(xs))
                if row == 1:
                    best = min(range(len(ys)), key=lambda i: ys[i])
                    axes[row, col].scatter(xs[best], ys[best], marker='*', s=65,
                                           color=colors(index), zorder=3)
        for row in range(2):
            ax = axes[row, col]
            ax.set(title=f'Group {group} / ' + ('Train' if row == 0 else 'Valid'),
                   xlabel='Epoch', ylabel='Train CE per optimizer update' if row == 0
                   else 'Half-epoch valid CE', xlim=(0, max(maximum_epoch, .5)))
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(frameon=False)
    image = output / 'loss_groups.png'
    temporary = output / 'loss_groups.tmp.png'
    fig.savefig(temporary, dpi=150)
    temporary.replace(image)
    plt.close(fig)

    metric_path = output / 'metrics.json'
    cache = read_json(metric_path) or {'settings': []}
    metric_rows = {row['setting']: row for row in cache['settings']}
    for setting in SETTING_ORDER:
        job = jobs.get(setting)
        if not job:
            continue
        directory = Path(job['run_dir'])
        if setting not in metric_rows and (directory / 'test/metrics.json').exists():
            metric_rows[setting] = evaluate_run(directory)
    metrics_payload = {
        'protocol': {
            'tempo': 'Beat-token negative log duration Pearson vs aligned human median curve; valid beat targets only.',
            'onset': 'Per-note W1 of generated onset dev vs unquantized aligned human onset dev; valid paired intervals only.',
            'duration': 'Per-note W1 on log(performance duration / score duration), unquantized human references.',
            'velocity': 'Per-note W1 on MIDI velocity.',
            'aggregation': 'Macro-average notes within work then works; mean and sample std over three AR draws; one training seed.',
        },
        'settings': [metric_rows[s] for s in SETTING_ORDER if s in metric_rows],
    }
    replace_text(metric_path, json.dumps(metrics_payload, indent=2, allow_nan=False) + '\n')
    trained = sum((Path(j['run_dir']) / 'training_summary.json').exists() for j in jobs.values())
    evaluated = len(metric_rows)
    image_link = os.path.relpath(image, doc.parent)
    metrics_link = os.path.relpath(metric_path, doc.parent)
    audit = ROOT / 'analysis/exp1117_beat_dev/fixed_event_data_audit_checked.json'
    audit_link = os.path.relpath(audit, doc.parent)
    lines = [
        '# exp1116：统一 Beat 映射与 Note Onset Dev 实验', '',
        '## 1. 当前协议', '',
        '本轮仅运行 A1、A2、A3、B1、B2、D4，共六个设置；每项随机初始化训练一次（seed=42），最佳 checkpoint 做三轮 sampling。', '',
        '训练与评估共用 onset-group 均值与线性插值的 score-to-performance beat 映射，不作单调过滤或端点外推。sidecar 已重新生成；无法确定完整 beat interval 的 beat duration/onset 监督使用有效性 mask，保留事件拓扑，不把兜底值当成真实监督。', '',
        'Beat 与 note duration target 始终一致：absolute_log 使用 ln(duration in ms)、区间 [3,8]；log_dev 使用 ln(D_perf/D_score)、区间 [-3,2]。两者均为 500 bins、宽度 0.01，解码使用 bin 中心。Note onset target 统一为 dev=u_perf-u_score，区间 [-1,1]、200 bins、宽度 0.01。', '',
        'Encoder 与 decoder 历史 duration 使用绝对 log duration；onset 历史使用 u_score+dev 还原的绝对拍内位置。dev 输出头参数与 encoder/decoder 输入 embedding 独立；shared/separate 输入表仅指 beat/note duration 是否共用输入表。', '',
        '连续 512 个 BEAT/NOTE 总事件直接切分，步长 512、无重叠，尾部 padding；microbatch=24、gradient accumulation=2，有效 batch=48，BF16。GPU0/1/2 各运行一个任务。最大 16 epoch，每半 epoch valid 一次，patience=4，只保存最低 valid CE 的最佳模型。AdamW，初始 LR=3e-4、线性衰减，decoder 历史 timestep mask 概率 0.5。', '',
        f'[sidecar 审计]({audit_link})：train/valid/test 为 6561/329/974 个窗口；测试为 19 首作品、82 个参考演奏。', '',
        '## 2. 配置与训练进度', '',
        f'更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}。训练完成 {trained}/6，四指标评估完成 {evaluated}/6。队列状态：`{manifest["status"]}`。', '',
        '| 设置 | beat/note duration target | onset target | duration 输入表 | duration head | measure_start | 当前 ep | stop ep | best valid CE |',
        '|---|---|---|---|---|---|---:|---:|---:|',
    ]
    for setting in SETTING_ORDER:
        job = jobs.get(setting)
        if not job:
            continue
        spec = job['config']
        directory = Path(job['run_dir'])
        summary = read_json(directory / 'training_summary.json')
        history_path = directory / 'loss_history.jsonl'
        history = []
        if history_path.exists():
            for line in history_path.read_text().splitlines():
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        epoch = max([float(x.get('epoch', 0)) for x in history] or [0.])
        validation = [float(x['eval_loss']) for x in history if 'eval_loss' in x]
        best = summary['best_valid_loss'] if summary else min(validation) if validation else None
        stop = f"{summary['stopped_epoch']:.3f}" if summary else '—'
        best_text = f'{best:.3f}' if best is not None else '—'
        lines.append(f'| {setting} | {spec["duration_target"]} | dev | {spec["duration_input_tables"]} | '
                     f'{spec["duration_output_head"]} | {int(spec["measure_start"])} | {epoch:.3f} | {stop} | {best_text} |')
    lines += ['', '## 3. 分组 Train / Valid 曲线', '',
              'A/B/D 三列、train/valid 两行；train 按每次 optimizer update 记录，valid 每半 epoch 记录。各列横轴自适应当前实际最大 epoch，星号标记最低 valid CE。', '',
              f'![exp1116 分组 Train / Valid 曲线]({image_link})', '',
              '## 4. 四项结果指标', '',
              'Tempo 使用生成的 BEAT duration，onset 使用生成的 note onset dev；直接按 score 事件身份与同一 sidecar 中的人类连续标签对齐，不从 MIDI 再次反推 beat。三项 W1 为逐 note 比较后对 note、作品平均；Tempo 为逐作品 Pearson 后平均。只报告三轮 AR sampling 的 mean ± sample std，保留三位小数；该 std 不表示训练随机性。', '',
              '| 设置 | Tempo curve Pearson ↑ | Onset dev W1 ↓ | Duration log-ratio W1 ↓ | Velocity W1 ↓ |',
              '|---|---:|---:|---:|---:|']
    for setting in SETTING_ORDER:
        row = metric_rows.get(setting)
        values = [f"{row['metrics'][key]['mean']:.3f} ± {row['metrics'][key]['std']:.3f}"
                  if row else '—' for key in METRICS]
        lines.append('| ' + ' | '.join([setting] + values) + ' |')
    lines += ['', f'[逐 sampling 指标]({metrics_link})。未完成的评估以“—”显示。', '',
              '## 5. 运行与归档', '', f'本轮正式结果目录：`{campaign.resolve()}`。', '',
              '结果文档固定为 `INSPIRE2/docs/exp1116_BEAT.md`；先前中断的试运行不纳入本轮结果。', '',
              '队列入口：[`run_exp1117_beat_dev_queue.py`](../script/run_exp1117_beat_dev_queue.py)，报告入口：[`report_exp1116_beat_dev.py`](../script/report_exp1116_beat_dev.py)。', '']
    for job in jobs.values():
        if job['status'] == 'failed':
            lines.append(f"- `{job['id']}` 失败：{job.get('error', '见任务日志')}。")
    beat_figure = ROOT / 'analysis/beat_duration_curves/test15_valid10_beat_duration_5x5.png'
    if beat_figure.exists():
        beat_link = os.path.relpath(beat_figure, doc.parent)
        beat_data_link = os.path.relpath(beat_figure.with_name('test15_valid10_curves.json'), doc.parent)
        lines += ['', '## 6. Score 与人类演奏的 Log Beat Duration 曲线', '',
                  '一张 5×5 矩阵：前 3 行为 15 个 test work（T01–T15，78 个演奏），后 2 行为 10 个 valid work（V01–V10，41 个演奏）。先覆盖各作曲家，再优先选择演奏数量更多的作品；每项保留该 work 的全部演奏，不按曲线形态筛选。', '',
                  '横轴为从 0 开始的 score beat 序号，纵轴为 $\\ln(D_{\\mathrm{beat}}/\\mathrm{ms})$，与训练 absolute_log 使用的自然对数坐标一致；绘制量化前连续值，不裁剪至 $[3,8]$。黑色虚线为 score，不同颜色实线为不同人类 performance。相邻小节用浅蓝/浅橙背景交替标识，边界来自 score 的 measure_start。曲线未经平滑，无法可靠映射的 performance beat interval 留空。此图沿用 sidecar 的既有时间归一化，不表示原始录音的绝对 BPM。', '',
                  f'![15 个 test 与 10 个 valid work 的 log beat duration 5×5 总图]({beat_link})', '',
                  f'[完整作品名、performance ID 与逐 beat 数据]({beat_data_link})。']
    # Keep the separately computed cross-model audit when refreshing training.
    comparison = output / 'baseline_comparison/comparison.md'
    if comparison.is_file():
        boundary = lines.index('## 5. 运行与归档')
        lines[boundary:boundary] = [comparison.read_text().strip(), '']
    # A later campaign owns only its appended block; preserve it on refresh.
    appended = ''
    if doc.exists() and '<!-- EXP1117_BEAT_MEDIAN_BEGIN -->' in doc.read_text():
        appended = '\n' + doc.read_text().split('<!-- EXP1117_BEAT_MEDIAN_BEGIN -->', 1)[1]
        appended = '\n<!-- EXP1117_BEAT_MEDIAN_BEGIN -->' + appended[1:]
    replace_text(doc, '\n'.join(lines).rstrip() + '\n' + appended)
    return manifest['status']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT.parent / 'results/exp1117_beat_dev_bs48')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/exp1116_beat_dev')
    parser.add_argument('--doc', type=Path, default=ROOT / 'docs/exp1116_BEAT.md')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        status = report(args.campaign, args.output, args.doc)
        print(datetime.now().astimezone().isoformat(), status, args.doc, flush=True)
        if not args.watch or status in ('completed', 'failed', 'paused'):
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
