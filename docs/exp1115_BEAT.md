# exp1115：512 总事件定长 Beat / Note 实验

## 1. 实验协议与进度

本轮使用 [design1115_beat.md](design1115_beat.md) 中的 Beat/Note 双级表示，固定每个窗口为连续 512 个 BEAT/NOTE 总事件，步长 512、无重叠，尾窗 padding 到 512。窗口允许从 NOTE 开始，也允许在 beat 内切分；padding 不参与 attention 或 loss。训练从随机权重开始，不加载 PT rendering checkpoint。

最大训练轮数为 16，每半个 epoch 验证一次，early stopping patience=4，只保存最低 valid CE 的 checkpoint。优化器为 AdamW，初始学习率 `3e-4`、线性衰减、weight decay=0；microbatch=16、gradient accumulation=2，有效 batch=32，使用 BF16。每个设置训练一次，最佳 checkpoint 做三轮 sampling（temperature=1、top-p=0.9）。

固定协议下 A/B/D 共 10 个设置均已完成训练和三轮 sampling。指标结果使用已生成的 AR MIDI 离线计算；1112 的两个代表方法也按同一套 score-event per-note 指标重新计算并合并到下表。

## 2. 分组训练曲线

下图按 A、B、D 三组绘制 train CE 与半 epoch valid CE。横轴按各组实际最大 epoch 自适应，星号表示最低 valid CE。

![Beat DINR 分组 train/valid 曲线](../analysis/exp1116_beat/loss_groups.png)

## 3. 训练设置与 valid 结果

| 设置 | beat target | note target | duration 输入表 | duration head | measure_start | stop ep | best valid CE |
|---|---|---|---|---|---:|---:|---:|
| A1 | absolute_log | absolute_log | shared | separate | 1 | 9.000 | 16.903 |
| A2 | log_dev | log_dev | shared | separate | 1 | 10.000 | 17.252 |
| A3 | absolute_log | absolute_log | shared | separate | 0 | 8.000 | 16.919 |
| B1 | absolute_log | absolute_log | separate | separate | 1 | 9.000 | 16.896 |
| B2 | log_dev | log_dev | separate | separate | 1 | 10.000 | 17.149 |
| D1 | absolute_log | log_dev | shared | separate | 1 | 9.000 | 16.980 |
| D2 | log_dev | absolute_log | shared | separate | 1 | 8.501 | 17.182 |
| D3 | absolute_log | absolute_log | shared | separate | 1 | 8.501 | 17.017 |
| D4 | absolute_log | absolute_log | separate | shared_conditioned | 1 | 8.000 | 17.618 |
| D5 | absolute_log | absolute_log | separate | separate | 0 | 7.000 | 16.980 |

## 4. 四项 score-event 指标

四项指标均保留 score note 身份，并在同一 note 上比较生成值与人类参考值：

- **Tempo curve Pearson**：逐作品比较 beat-level log-tempo 曲线与人类参考中位曲线的 Pearson 相关。
- **Onset W1**：逐 note 比较 `u_perf-u_score` 的 PN Wasserstein-1。
- **Duration W1**：逐 note 比较 `ln(D_perf/D_score)` 的 PN Wasserstein-1。
- **Velocity W1**：逐 note 比较原始 MIDI velocity 的 PN Wasserstein-1。

每个 1115 设置只有一次训练，表中 `mean ± std` 来自三轮 sampling，因此 std 表示 sampling 波动，不表示训练 seed 波动。1112 两个代表方法使用已有三轮 AR MIDI，未重新生成。所有数字保留三位小数。

| 方法 | Tempo curve Pearson | Onset W1 (dev) | Duration W1 (log ratio) | Velocity W1 |
|---|---:|---:|---:|---:|
| 1115 A1 | 0.115 ± 0.024 | 0.171 ± 0.001 | 0.959 ± 0.028 | 20.515 ± 0.698 |
| 1115 A2 | 0.235 ± 0.020 | 0.194 ± 0.001 | 0.950 ± 0.005 | 23.786 ± 0.190 |
| 1115 A3 | 0.137 ± 0.016 | 0.175 ± 0.002 | 0.952 ± 0.029 | 18.519 ± 0.462 |
| 1115 B1 | 0.113 ± 0.005 | 0.171 ± 0.002 | 0.902 ± 0.020 | 20.479 ± 0.145 |
| 1115 B2 | 0.240 ± 0.012 | 0.213 ± 0.002 | 1.016 ± 0.009 | 20.517 ± 0.521 |
| 1115 D1 | 0.124 ± 0.041 | 0.204 ± 0.003 | 1.049 ± 0.010 | 23.006 ± 0.296 |
| 1115 D2 | 0.246 ± 0.020 | 0.182 ± 0.003 | 1.020 ± 0.011 | 23.404 ± 0.711 |
| 1115 D3 | 0.081 ± 0.019 | 0.198 ± 0.005 | 1.042 ± 0.015 | 22.567 ± 0.031 |
| 1115 D4 | 0.022 ± 0.029 | 0.241 ± 0.014 | 1.882 ± 0.015 | 19.829 ± 0.053 |
| 1115 D5 | 0.074 ± 0.021 | 0.212 ± 0.008 | 0.990 ± 0.022 | 26.784 ± 0.024 |
| 1112 separate_timestep_abs5000 | -0.241 ± 0.005 | 0.086 ± 0.001 | 1.070 ± 0.013 | 27.020 ± 0.353 |
| 1112 pure_coord_mlp64 | -0.263 ± 0.004 | 0.100 ± 0.002 | 0.940 ± 0.004 | 16.062 ± 0.240 |

Onset 使用 `u_perf-u_score`。如果改用 onset ratio，单个 note 的预测值和全部参考值都会共同减去同一个 `u_score`，所以 PN W1 数值不变；逐 note 校验最大差异为 `1.1e-16`。Duration 使用 `ln(D_perf/D_score)`，velocity 使用 MIDI velocity 原值。逐作品、逐 sampling 的明细见 [pn_metrics.json](../analysis/exp1115_beat/pn_metrics.json) 和 [six_metrics_representatives.json](../analysis/exp1112/six_metrics_representatives.json)。

## 5. 当前配置说明

### 5.1 固定长度与事件类型

BEAT 与 NOTE 合计 512 个事件直接切分，窗口不要求从 BEAT 开始，也不在 beat 边界对齐。生成后按原始事件顺序拼回完整作品，再累计 beat duration 重建 note onset。BEAT 位置预测 beat duration；NOTE 位置预测 note duration、onset 和 velocity，输出 head 由已知事件类型路由。

BEAT 输入为

$$e_{BEAT}=E_{beat}[m]+E_{duration}[d].$$

有 `measure_start` 时 $m\in\{0,1\}$，无区分时只有一个 beat token；不叠加另一份独立的 BEAT 类型向量。duration 的 absolute-log 区间为 `[3,8]`，log-dev 区间为 `[-2,3]`，onset 区间为 `[-0.25,1]`，均使用 500 个 bins。

### 5.2 数据与运行目录

固定切分审计见 [fixed_event_data_audit.json](../analysis/exp1115_beat/fixed_event_data_audit.json)。运行目录为 `/home/kaititech/EPR/PianistTransformer/results/exp1116_beat`，保存 manifest、checkpoint、原始 sampling 输出、MIDI 和逐作品指标。队列入口为 [`run_exp1116_beat_queue.py`](../script/run_exp1116_beat_queue.py)：

```bash
cd INSPIRE2
python script/run_exp1116_beat_queue.py --gpus 0 1 2
```

当前 10 个设置均已完成，不需要重新训练；本文件只保留当前四项指标作为结果比较依据。
