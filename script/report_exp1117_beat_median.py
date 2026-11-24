#!/usr/bin/env python3
"""Append or refresh only the new beat-median experiment block in exp1116."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from script.evaluate_exp1117_beat_dev import evaluate_run

BEGIN = '<!-- EXP1117_BEAT_MEDIAN_BEGIN -->'
END = '<!-- EXP1117_BEAT_MEDIAN_END -->'
ORDER = ('A1', 'A2', 'A3', 'C1', 'C2', 'C3')
METRICS = ('tempo_curve_pearson', 'onset_w1', 'duration_w1', 'velocity_w1')


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def report(campaign, output, document):
    manifest = json.loads((campaign / 'manifest.json').read_text())
    statistics = json.loads((output / 'statistics.json').read_text())
    jobs = {j['setting']: j for j in manifest['jobs']}
    metrics_path = output / 'metrics.json'
    metrics = {r['setting']: r for r in json.loads(metrics_path.read_text())['settings']} if metrics_path.exists() else {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, group in enumerate((ORDER[:3], ORDER[3:])):
        for setting in group:
            if setting not in jobs:
                continue
            run = Path(jobs[setting]['run_dir'])
            if setting not in metrics and (run / 'test/metrics.json').exists():
                metrics[setting] = evaluate_run(run)
            history = []
            if (run / 'loss_history.jsonl').exists():
                for line in (run / 'loss_history.jsonl').read_text().splitlines():
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            for row, key in enumerate(('loss', 'eval_loss')):
                points = [h for h in history if key in h and h.get('epoch', 0) > 0]
                if points:
                    axes[row, col].plot([p['epoch'] for p in points], [p[key] for p in points], label=setting, lw=1)
        for row, label in enumerate(('Train CE', 'Half-epoch valid CE')):
            axes[row, col].set(xlabel='Epoch', ylabel=label, title='/'.join(group))
            axes[row, col].grid(alpha=.2)
            if axes[row, col].lines:
                axes[row, col].legend(frameon=False)
    image = output / 'loss_groups.png'
    temporary = image.with_name('loss_groups.tmp.png')
    fig.savefig(temporary, dpi=150)
    temporary.replace(image)
    plt.close(fig)
    write(metrics_path, json.dumps(dict(protocol=dict(
        normalization='beat_median_log_ratio; one factor per entire human performance',
        tempo='Pearson of negative log beat duration vs pointwise human log-duration median',
        onset='per-note W1 in onset dev; abs predictions subtract score beat-relative position',
        duration='per-note W1 in ln(D_perf/D_score)', velocity='per-note W1 in MIDI velocity',
        aggregation='macro average 19 works within each AR draw, then mean and sample std ddof=1 across 3 draws'),
        settings=[metrics[s] for s in ORDER if s in metrics]), indent=2, allow_nan=False) + '\n')
    trained = sum((Path(j['run_dir']) / 'training_summary.json').exists() for j in jobs.values())
    now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    lines = [BEGIN, '## 7. 2026-11-17 追加：全曲 Beat 典型速度归一化重跑', '',
        f'更新时间：{now}。状态：`{manifest["status"]}`；完成训练 {trained}/6，完成三轮测试与四指标 {len(metrics)}/6。', '',
        '本节使用独立数据与结果目录；第 1–6 节为之前实验，保留原始设置、指标和曲线。', '',
        '### 7.1 归一化定义与数据', '',
        r'以每个完整 performance 的有效 beat interval 为样本，计算 $g=\operatorname{median}_b\ln(D_b^{perf}/D_b^{score})$，令 $\alpha=\exp(-g)$。所有 performance IOI、onset、note duration、beat duration 共用这一个缩放因子，因此归一化后 $\operatorname{median}_b\ln(D_b^{perf,new}/D_b^{score})=0$。score tempo 变化保留在逐拍分母中；段落与局部速度变化由模型学习。', '',
        '从 `ASAP_MUSICAL51` 原始未缩放标签重建，核对其与旧 grouped-median 标签撤销原因子后的数值一致；两个原 raw MS-score 测试作品也采用同一新归一化。映射仍为 score-onset-group 均值＋线性插值，不作单调过滤或端点外推，只用两端受支持且时长为正的完整 beat interval 估计因子。', '',
        'score 输入、velocity、数据划分、样本窗口和有效性 mask 均与上一轮逐项核对一致。连续事件时长仍保留既有 1 ms 下限；不作分段归一化。', '',
        f'共 {statistics["works"]} works、{statistics["performances"]} performances。原始连续标签归一化残差最大绝对值 {statistics["max_abs_raw_normalized_median_log_ratio"]:.3g}。', '',
        '| 划分 | Works | Performances | 固定 512-event 窗口 | Events | 最大 median log beat-ratio 绝对值 |',
        '|---|---:|---:|---:|---:|---:|']
    for split, row in statistics['splits'].items():
        lines.append(f'| {split} | {row["works"]} | {row["performances"]} | {row["windows"]} | {row["events"]:,} | {row["max_abs_median_log_beat_ratio"]:.3g} |')
    scale_stats = statistics['scale_percentiles']
    lines += ['', f'Raw→normalized 缩放因子：min={scale_stats["min"]:.4f}，P25={scale_stats["p25"]:.4f}，median={scale_stats["median"]:.4f}，P75={scale_stats["p75"]:.4f}，max={scale_stats["max"]:.4f}。', '',
        '[归一化因子与来源哈希](../analysis/exp1117_beat_median/normalization_manifest.json) · [统计与裁剪比例](../analysis/exp1117_beat_median/statistics.json) · [数据审计](../analysis/exp1117_beat_median/fixed_event_data_audit_checked.json)', '',
        '![新数据分布：有效连续标签](../analysis/exp1117_beat_median/normalized_distributions.png)', '',
        '归一化 work sidecar：`data/ASAP_processed_beat_median_v1`；连续事件：`data/ASAP_processed_dinr_beat_median_v1`；固定事件窗口：`data/ASAP_processed_dinr_beat_median_e512_v1`。', '',
        '### 7.2 六项配置与训练协议', '',
        '所有 duration 输入表统一为 **shared**。原 B1、B2、D4 分别改名为 C1、C2、C3；C1、C2 使用 **abs** onset target（拍内绝对位置 $u_{perf}$，不是毫秒 onset）；其余使用 dev=$u_{perf}-u_{score}$。', '',
        '| 设置 | 原编号 | Beat/Note duration target | Duration input | Duration output head | Onset target | measure_start |',
        '|---|---|---|---|---|---|---|']
    old_ids = dict(C1='B1', C2='B2', C3='D4')
    for s in ORDER:
        cfg = manifest['settings'][s]
        lines.append('| ' + ' | '.join((s, old_ids.get(s, s), cfg['duration_target'], cfg['duration_input_tables'],
            cfg['duration_output_head'], cfg['onset_target'], str(cfg['measure_start']))) + ' |')
    lines += ['', 'Duration absolute_log：ln(ms)，[3,8]、500 bins；log_dev：ln(D_perf/D_score)，[-3,2]、500 bins，均宽 0.01。Onset dev：[-1,1]、200 bins、宽 0.01；abs：[-0.25,1]、500 bins、宽 0.0025。Decoder duration/onset history 均使用绝对输入坐标。v2 连续 sidecar 统一存 onset dev，训练和评估按配置显式转换。', '',
        '其余沿用上一轮：随机初始化 seed=42，E10/D2、hidden=768；固定 512 个 BEAT/NOTE 总事件，stride=512，无重叠；microbatch=24 × accumulation=2，有效 batch=48，尾批梯度按实际累积数归一化；BF16，LR=3e-4、linear；最多 16 epochs，每半个 epoch 验证，patience=4 次验证，只保留最佳 valid checkpoint。三个 GPU 调度六项设置。', '',
        '### 7.3 进度与结果', '',
        '| 设置 | 状态 | Best valid CE | 停止 epoch | Tempo Pearson ↑ | Onset dev W1 ↓ | Duration log-ratio W1 ↓ | Velocity W1 ↓ |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in ORDER:
        j = jobs[s]
        summary_path = Path(j['run_dir']) / 'training_summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        values = [f'{metrics[s]["metrics"][n]["mean"]:.3f} ± {metrics[s]["metrics"][n]["std"]:.3f}' if s in metrics else '待完成' for n in METRICS]
        lines.append('| ' + ' | '.join([s, j['status'], f'{summary["best_valid_loss"]:.3f}' if summary else '—',
            f'{summary["stopped_epoch"]:.2f}' if summary else '—'] + values) + ' |')
        if j.get('error'):
            lines += ['', f'{s} 失败原因：`{j["error"]}`。']
    lines += ['', '同一最佳 checkpoint 做 3 轮 AR sampling，先逐作品聚合再 macro-average 19 首作品，报告三轮 mean ± sample std（ddof=1）。四指标沿用第 4 节原生事件身份：tempo 比较负 log beat duration 与人类逐拍 log-duration 中位数；三项 W1 使用连续人类参考与有效性 mask。C1、C2 的 abs 预测先减去 score 拍内位置，再用统一 onset-dev 指标评价。', '',
        '![本轮训练与验证 loss](../analysis/exp1117_beat_median/loss_groups.png)', '',
        '[四指标 JSON](../analysis/exp1117_beat_median/metrics.json) · [队列入口](../script/run_exp1117_beat_median_queue.py) · [数据重建入口](../script/prepare_exp1117_beat_median.py)', '',
        f'结果目录：`{campaign}`。队列在训练、验证、测试时自动刷新本节；全部完成才标记 completed。', '',
        '### 7.4 归一化后的 Score / Human Beat Duration 曲线', '',
        '使用与第 6 节相同的作品选择规则（15 test＋10 valid），展示新连续标签的 ln(beat duration/ms)，不按 token 范围裁剪。黑色虚线为 score，彩色线为各演奏，小节背景交替；无有效支持的 interval 留空。', '',
        '![归一化后的 beat duration 曲线](../analysis/exp1117_beat_median/test15_valid10_beat_duration_5x5.png)', '',
        '[完整曲线数据](../analysis/exp1117_beat_median/test15_valid10_curves.json)']
    analysis = output / 'result_analysis/analysis.md'
    if analysis.exists():
        lines += ['', analysis.read_text().strip()]
    human_baseline = output / 'result_analysis/human_baseline.md'
    if human_baseline.exists():
        lines += ['', human_baseline.read_text().strip()]
    distribution_w1 = output / 'result_analysis/distribution_w1.md'
    if distribution_w1.exists():
        lines += ['', distribution_w1.read_text().strip()]
    zero_onset_diagnostics = output / 'result_analysis/zero_onset_diagnostics.md'
    if zero_onset_diagnostics.exists():
        lines += ['', zero_onset_diagnostics.read_text().strip()]
    mean_variance = output / 'result_analysis/mean_variance.md'
    if mean_variance.exists():
        lines += ['', mean_variance.read_text().strip()]
    human_score_mean_variance = output / 'result_analysis/human_score_mean_variance.md'
    if human_score_mean_variance.exists():
        lines += ['', human_score_mean_variance.read_text().strip()]
    five_main_metrics = output / 'result_analysis/five_main_metrics.md'
    if five_main_metrics.exists():
        lines += ['', five_main_metrics.read_text().strip()]
    lines.append(END)
    block = '\n'.join(lines) + '\n'
    write(output / 'report.md', block)
    previous = document.read_text()
    if BEGIN in previous:
        a = previous.index(BEGIN)
        b = previous.index(END, a) + len(END)
        updated = previous[:a] + block.rstrip() + previous[b:]
    else:
        updated = previous + '\n' + block
    write(document, updated)
    print(now, manifest['status'], f'trained={trained}/6 metrics={len(metrics)}/6', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT.parent / 'results/exp1117_beat_median_bs48')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/exp1117_beat_median')
    parser.add_argument('--doc', type=Path, default=ROOT / 'docs/exp1116_BEAT.md')
    args = parser.parse_args()
    report(args.campaign, args.output, args.doc)


if __name__ == '__main__':
    main()
