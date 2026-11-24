# exp1116：统一 Beat 映射与 Note Onset Dev 实验

## 1. 当前协议

本轮仅运行 A1、A2、A3、B1、B2、D4，共六个设置；每项随机初始化训练一次（seed=42），最佳 checkpoint 做三轮 sampling。

训练与评估共用 onset-group 均值与线性插值的 score-to-performance beat 映射，不作单调过滤或端点外推。sidecar 已重新生成；无法确定完整 beat interval 的 beat duration/onset 监督使用有效性 mask，保留事件拓扑，不把兜底值当成真实监督。

Beat 与 note duration target 始终一致：absolute_log 使用 ln(duration in ms)、区间 [3,8]；log_dev 使用 ln(D_perf/D_score)、区间 [-3,2]。两者均为 500 bins、宽度 0.01，解码使用 bin 中心。Note onset target 统一为 dev=u_perf-u_score，区间 [-1,1]、200 bins、宽度 0.01。

Encoder 与 decoder 历史 duration 使用绝对 log duration；onset 历史使用 u_score+dev 还原的绝对拍内位置。dev 输出头参数与 encoder/decoder 输入 embedding 独立；shared/separate 输入表仅指 beat/note duration 是否共用输入表。

连续 512 个 BEAT/NOTE 总事件直接切分，步长 512、无重叠，尾部 padding；microbatch=24、gradient accumulation=2，有效 batch=48，BF16。GPU0/1/2 各运行一个任务。最大 16 epoch，每半 epoch valid 一次，patience=4，只保存最低 valid CE 的最佳模型。AdamW，初始 LR=3e-4、线性衰减，decoder 历史 timestep mask 概率 0.5。

[sidecar 审计](../analysis/exp1117_beat_dev/fixed_event_data_audit_checked.json)：train/valid/test 为 6561/329/974 个窗口；测试为 19 首作品、82 个参考演奏。

## 2. 配置与训练进度

更新时间：2026-11-17T01:12:43+08:00。训练完成 6/6，四指标评估完成 6/6。队列状态：`completed`。

| 设置 | beat/note duration target | onset target | duration 输入表 | duration head | measure_start | 当前 ep | stop ep | best valid CE |
|---|---|---|---|---|---|---:|---:|---:|
| A1 | absolute_log | dev | shared | separate | 1 | 10.000 | 10.000 | 15.468 |
| A2 | log_dev | dev | shared | separate | 1 | 10.000 | 10.000 | 15.930 |
| A3 | absolute_log | dev | shared | separate | 0 | 9.000 | 9.000 | 15.477 |
| B1 | absolute_log | dev | separate | separate | 1 | 10.000 | 10.000 | 15.549 |
| B2 | log_dev | dev | separate | separate | 1 | 10.000 | 10.000 | 15.905 |
| D4 | absolute_log | dev | separate | shared_conditioned | 1 | 9.504 | 9.504 | 15.981 |

## 3. 分组 Train / Valid 曲线

A/B/D 三列、train/valid 两行；train 按每次 optimizer update 记录，valid 每半 epoch 记录。各列横轴自适应当前实际最大 epoch，星号标记最低 valid CE。

![exp1116 分组 Train / Valid 曲线](../analysis/exp1116_beat_dev/loss_groups.png)

## 4. 四项结果指标

Tempo 使用生成的 BEAT duration，onset 使用生成的 note onset dev；直接按 score 事件身份与同一 sidecar 中的人类连续标签对齐，不从 MIDI 再次反推 beat。三项 W1 为逐 note 比较后对 note、作品平均；Tempo 为逐作品 Pearson 后平均。只报告三轮 AR sampling 的 mean ± sample std，保留三位小数；该 std 不表示训练随机性。

| 设置 | Tempo curve Pearson ↑ | Onset dev W1 ↓ | Duration log-ratio W1 ↓ | Velocity W1 ↓ |
|---|---:|---:|---:|---:|
| A1 | 0.128 ± 0.008 | 0.039 ± 0.000 | 0.927 ± 0.016 | 20.195 ± 0.280 |
| A2 | 0.327 ± 0.006 | 0.040 ± 0.000 | 0.782 ± 0.004 | 22.223 ± 0.483 |
| A3 | 0.149 ± 0.019 | 0.039 ± 0.000 | 0.861 ± 0.033 | 18.856 ± 0.406 |
| B1 | 0.136 ± 0.016 | 0.042 ± 0.001 | 0.935 ± 0.014 | 20.680 ± 0.510 |
| B2 | 0.319 ± 0.009 | 0.039 ± 0.000 | 0.769 ± 0.008 | 19.147 ± 0.415 |
| D4 | 0.049 ± 0.002 | 0.045 ± 0.001 | 1.270 ± 0.009 | 19.792 ± 0.149 |

[逐 sampling 指标](../analysis/exp1116_beat_dev/metrics.json)。未完成的评估以“—”显示。

<!-- EXP1116_BASELINE_COMPARISON_BEGIN -->
### 4.1 exp1112、PT 与 BEAT 的统一观测指标对照

复用已保存的最佳 checkpoint 预测：exp1112 的 24 个 DINR 设置、PT-1（即 PT base-1）、PT-scratch，以及本轮 6 个 BEAT 设置。每项均为 19 首作品 × 3 轮 AR sampling；不重训、不重新采样、不增加全曲或分段时间缩放。

**本节采用统一的音符起音重提取协议，与第 4 节上表的原生 BEAT-token 协议分开报告。** 所有模型都用同一个 onset-group 均值＋线性插值映射，从预测音符起音重新估计 beat duration 与 note onset dev。旧模型读取已保存的 MIDI；BEAT 模型由原始 token NPZ 重建连续音符时间，保留 score 事件身份，避免按实际起音排序重新配对。

人类参考固定为当前冻结 sidecar 中的 19 首作品、82 个连续演奏标签；不使用各旧方法原先导出的不同人类 MIDI。Tempo 为负 log beat duration 对人类逐拍 log-duration 中位数曲线的 Pearson；三项 W1 均逐 score note 比较单个预测与该音符的人类参考分布，再对音符、作品平均。Duration 使用 ln(D_perf/D_score)，onset 使用拍内 onset dev，velocity 使用原始 MIDI velocity；duration 的 1 ms 下限与主表一致，不按 token 范围裁剪数值。

为保证本节 32 个设置及三轮采样的覆盖完全相同，各指标使用所有预测与人类参考的共同有效位置：tempo 13,373/13,444 个 beat；onset 64,924/65,039 个 note；duration 63,169/65,039 个 note；velocity 64,145/65,039 个 note。该共同范围仅用于本节，不改变上表主结果。

PT-scratch 的原始 MIDI 保留完整的正 velocity note-on 序列，但部分 note-on 没有常规 FIFO 解码可得到的正时长结束事件。对齐使用完整 note-on 的 pitch 与同 pitch 内起音次序，起音与力度不因缺失 duration 而错位；duration 不可用的位置在本节所有模型中统一排除。JSON 保留每份预测的配对与覆盖审计。

若 MIDI 中同一 pitch 的多个 note-on 落在同一时刻，导出时的排序无法恢复它们各自的 score 身份：相同起音仍可用于 beat/onset，但这些音符的 duration/velocity 在所有模型中共同排除，不按结束时间或力度任意配对。

以下 mean ± sample std（ddof=1）仅反映同一 checkpoint 的三轮采样变化，全部保留三位小数。

下表合并 PT、exp1112 与 BEAT：DINR 和 BEAT 仅展示 log dev 表示，PT 保留原始表示作为参考。共同有效范围仍按全部 32 个设置确定；完整设置见 JSON/CSV。

| 设置 | 训练 mask | 表示 | Tempo curve Pearson ↑ | Onset dev W1 ↓ | Duration log-ratio W1 ↓ | Velocity W1 ↓ |
|---|---|---|---:|---:|---:|---:|
| PT-1 | non-pitch token | abs 5000 token | 0.304 ± 0.014 | 0.034 ± 0.000 | 0.670 ± 0.011 | 15.715 ± 0.524 |
| PT-scratch | non-pitch token | abs 5000 token | -0.128 ± 0.003 | 0.151 ± 0.001 | 2.688 ± 0.014 | 38.019 ± 0.012 |
| 1112 DINR separate | timestep 1+pitch | log dev | 0.224 ± 0.005 | 0.062 ± 0.000 | 0.894 ± 0.003 | 24.452 ± 0.100 |
| 1112 DINR separate | timestep | log dev | 0.214 ± 0.012 | 0.060 ± 0.000 | 0.868 ± 0.004 | 25.433 ± 0.025 |
| BEAT A2 | timestep | log dev | 0.336 ± 0.007 | 0.040 ± 0.000 | 0.781 ± 0.004 | 22.175 ± 0.485 |
| BEAT B2 | timestep | log dev | 0.326 ± 0.010 | 0.039 ± 0.000 | 0.768 ± 0.008 | 19.092 ± 0.419 |

[完整定义、逐作品/逐轮结果与输入审计](../analysis/exp1116_beat_dev/baseline_comparison/metrics.json) · [汇总 CSV](../analysis/exp1116_beat_dev/baseline_comparison/metrics.csv)
<!-- EXP1116_BASELINE_COMPARISON_END -->


## 5. 运行与归档

本轮正式结果目录：`/home/kaititech/EPR/PianistTransformer/results/exp1117_beat_dev_bs48`。

结果文档固定为 `INSPIRE2/docs/exp1116_BEAT.md`；先前中断的试运行不纳入本轮结果。

队列入口：[`run_exp1117_beat_dev_queue.py`](../script/run_exp1117_beat_dev_queue.py)，报告入口：[`report_exp1116_beat_dev.py`](../script/report_exp1116_beat_dev.py)。


## 6. Score 与人类演奏的 Log Beat Duration 曲线

一张 5×5 矩阵：前 3 行为 15 个 test work（T01–T15，78 个演奏），后 2 行为 10 个 valid work（V01–V10，41 个演奏）。先覆盖各作曲家，再优先选择演奏数量更多的作品；每项保留该 work 的全部演奏，不按曲线形态筛选。

横轴为从 0 开始的 score beat 序号，纵轴为 $\ln(D_{\mathrm{beat}}/\mathrm{ms})$，与训练 absolute_log 使用的自然对数坐标一致；绘制量化前连续值，不裁剪至 $[3,8]$。黑色虚线为 score，不同颜色实线为不同人类 performance。相邻小节用浅蓝/浅橙背景交替标识，边界来自 score 的 measure_start。曲线未经平滑，无法可靠映射的 performance beat interval 留空。此图沿用 sidecar 的既有时间归一化，不表示原始录音的绝对 BPM。

![15 个 test 与 10 个 valid work 的 log beat duration 5×5 总图](../analysis/beat_duration_curves/test15_valid10_beat_duration_5x5.png)

[完整作品名、performance ID 与逐 beat 数据](../analysis/beat_duration_curves/test15_valid10_curves.json)。

<!-- EXP1117_BEAT_MEDIAN_BEGIN -->
## 7. 2026-11-17 追加：全曲 Beat 典型速度归一化重跑

更新时间：2026-11-21T17:08:20+08:00。状态：`completed`；完成训练 6/6，完成三轮测试与四指标 6/6。

本节使用独立数据与结果目录；第 1–6 节为之前实验，保留原始设置、指标和曲线。

### 7.1 归一化定义与数据

以每个完整 performance 的有效 beat interval 为样本，计算 $g=\operatorname{median}_b\ln(D_b^{perf}/D_b^{score})$，令 $\alpha=\exp(-g)$。所有 performance IOI、onset、note duration、beat duration 共用这一个缩放因子，因此归一化后 $\operatorname{median}_b\ln(D_b^{perf,new}/D_b^{score})=0$。score tempo 变化保留在逐拍分母中；段落与局部速度变化由模型学习。

从 `ASAP_MUSICAL51` 原始未缩放标签重建，核对其与旧 grouped-median 标签撤销原因子后的数值一致；两个原 raw MS-score 测试作品也采用同一新归一化。映射仍为 score-onset-group 均值＋线性插值，不作单调过滤或端点外推，只用两端受支持且时长为正的完整 beat interval 估计因子。

score 输入、velocity、数据划分、样本窗口和有效性 mask 均与上一轮逐项核对一致。连续事件时长仍保留既有 1 ms 下限；不作分段归一化。

共 206 works、966 performances。原始连续标签归一化残差最大绝对值 3.32e-13。

| 划分 | Works | Performances | 固定 512-event 窗口 | Events | 最大 median log beat-ratio 绝对值 |
|---|---:|---:|---:|---:|---:|
| train | 175 | 837 | 6561 | 3,154,305 | 1.04e-07 |
| valid | 12 | 47 | 329 | 159,599 | 1.02e-07 |
| test | 19 | 82 | 974 | 480,332 | 4.91e-08 |

Raw→normalized 缩放因子：min=0.2143，P25=0.8685，median=0.9795，P75=1.0568，max=1.9666。

[归一化因子与来源哈希](../analysis/exp1117_beat_median/normalization_manifest.json) · [统计与裁剪比例](../analysis/exp1117_beat_median/statistics.json) · [数据审计](../analysis/exp1117_beat_median/fixed_event_data_audit_checked.json)

![新数据分布：有效连续标签](../analysis/exp1117_beat_median/normalized_distributions.png)

归一化 work sidecar：`data/ASAP_processed_beat_median_v1`；连续事件：`data/ASAP_processed_dinr_beat_median_v1`；固定事件窗口：`data/ASAP_processed_dinr_beat_median_e512_v1`。

### 7.2 六项配置与训练协议

所有 duration 输入表统一为 **shared**。原 B1、B2、D4 分别改名为 C1、C2、C3；C1、C2 使用 **abs** onset target（拍内绝对位置 $u_{perf}$，不是毫秒 onset）；其余使用 dev=$u_{perf}-u_{score}$。

| 设置 | 原编号 | Beat/Note duration target | Duration input | Duration output head | Onset target | measure_start |
|---|---|---|---|---|---|---|
| A1 | A1 | absolute_log | shared | separate | dev | True |
| A2 | A2 | log_dev | shared | separate | dev | True |
| A3 | A3 | absolute_log | shared | separate | dev | False |
| C1 | B1 | absolute_log | shared | separate | abs | True |
| C2 | B2 | log_dev | shared | separate | abs | True |
| C3 | D4 | absolute_log | shared | shared_conditioned | dev | True |

Duration absolute_log：ln(ms)，[3,8]、500 bins；log_dev：ln(D_perf/D_score)，[-3,2]、500 bins，均宽 0.01。Onset dev：[-1,1]、200 bins、宽 0.01；abs：[-0.25,1]、500 bins、宽 0.0025。Decoder duration/onset history 均使用绝对输入坐标。v2 连续 sidecar 统一存 onset dev，训练和评估按配置显式转换。

其余沿用上一轮：随机初始化 seed=42，E10/D2、hidden=768；固定 512 个 BEAT/NOTE 总事件，stride=512，无重叠；microbatch=24 × accumulation=2，有效 batch=48，尾批梯度按实际累积数归一化；BF16，LR=3e-4、linear；最多 16 epochs，每半个 epoch 验证，patience=4 次验证，只保留最佳 valid checkpoint。三个 GPU 调度六项设置。

### 7.3 进度与结果

| 设置 | 状态 | Best valid CE | 停止 epoch | Tempo Pearson ↑ | Onset dev W1 ↓ | Duration log-ratio W1 ↓ | Velocity W1 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | completed | 15.496 | 8.00 | 0.141 ± 0.009 | 0.040 ± 0.000 | 0.980 ± 0.031 | 22.581 ± 0.623 |
| A2 | completed | 15.962 | 10.00 | 0.332 ± 0.016 | 0.040 ± 0.000 | 0.783 ± 0.006 | 18.814 ± 0.322 |
| A3 | completed | 15.513 | 8.00 | 0.151 ± 0.009 | 0.040 ± 0.000 | 0.929 ± 0.010 | 24.160 ± 0.179 |
| C1 | completed | 16.924 | 9.00 | 0.127 ± 0.021 | 0.207 ± 0.003 | 0.915 ± 0.006 | 20.051 ± 0.180 |
| C2 | completed | 17.558 | 8.50 | 0.307 ± 0.009 | 0.264 ± 0.003 | 1.006 ± 0.021 | 21.663 ± 0.367 |
| C3 | completed | 15.976 | 8.50 | 0.060 ± 0.021 | 0.042 ± 0.001 | 1.463 ± 0.037 | 20.938 ± 0.152 |

同一最佳 checkpoint 做 3 轮 AR sampling，先逐作品聚合再 macro-average 19 首作品，报告三轮 mean ± sample std（ddof=1）。四指标沿用第 4 节原生事件身份：tempo 比较负 log beat duration 与人类逐拍 log-duration 中位数；三项 W1 使用连续人类参考与有效性 mask。C1、C2 的 abs 预测先减去 score 拍内位置，再用统一 onset-dev 指标评价。

![本轮训练与验证 loss](../analysis/exp1117_beat_median/loss_groups.png)

[四指标 JSON](../analysis/exp1117_beat_median/metrics.json) · [队列入口](../script/run_exp1117_beat_median_queue.py) · [数据重建入口](../script/prepare_exp1117_beat_median.py)

结果目录：`/home/kaititech/EPR/PianistTransformer/results/exp1117_beat_median_bs48`。队列在训练、验证、测试时自动刷新本节；全部完成才标记 completed。

### 7.4 归一化后的 Score / Human Beat Duration 曲线

使用与第 6 节相同的作品选择规则（15 test＋10 valid），展示新连续标签的 ln(beat duration/ms)，不按 token 范围裁剪。黑色虚线为 score，彩色线为各演奏，小节背景交替；无有效支持的 interval 留空。

![归一化后的 beat duration 曲线](../analysis/exp1117_beat_median/test15_valid10_beat_duration_5x5.png)

[完整曲线数据](../analysis/exp1117_beat_median/test15_valid10_curves.json)

### 7.5 2026-11-19 结果分析与 PT-1 同口径复核

**A2 是本轮六项设置中最均衡的一组。** 它在原生主表的 tempo、duration、velocity 上均领先；onset 与 A1/A3 接近，并非四项均严格最优。当前适合保留的组合是 beat/note log_dev、onset dev、shared duration 输入、separate duration 输出以及 measure_start。

**先统一比较口径。** 第 4.1 节的 PT-1 是旧人类参考＋音符重提取，第 7.3 节是新归一化参考＋原生 BEAT 事件；不能直接用两表数值判断差距。本次复用已保存输出，对 PT-1、旧 A2、新六项设置统一使用本轮 beat-median 人类参考，从预测音符起音以 onset-group 均值＋线性插值重提取 beat/onset，保留 score 音符身份；不重训、不重新采样、不额外缩放主比较预测。

共同有效范围按这 8 项设置 × 3 轮确定：tempo 13,372/13,444 beats；onset 64,897/65,039 notes；duration、velocity 均为 65,039/65,039 notes。没有纳入 PT-scratch，因此 duration/velocity 覆盖比第 4.1 节更完整。每轮 macro-average 19 首作品，报告三轮 mean ± sample std。

| 设置 | Tempo Pearson ↑ | Onset dev W1 ↓ | Duration log-ratio W1 ↓ | Velocity W1 ↓ |
|---|---:|---:|---:|---:|
| PT-1 | 0.306 ± 0.016 | 0.034 ± 0.000 | 0.665 ± 0.014 | 15.723 ± 0.534 |
| 旧 A2 | 0.337 ± 0.006 | 0.040 ± 0.000 | 0.785 ± 0.003 | 22.223 ± 0.483 |
| 新 A1 | 0.142 ± 0.010 | 0.039 ± 0.000 | 0.980 ± 0.031 | 22.581 ± 0.623 |
| 新 A2 | 0.338 ± 0.017 | 0.039 ± 0.000 | 0.783 ± 0.006 | 18.814 ± 0.322 |
| 新 A3 | 0.152 ± 0.008 | 0.039 ± 0.000 | 0.929 ± 0.010 | 24.160 ± 0.179 |
| 新 C1 | 0.089 ± 0.025 | 0.179 ± 0.002 | 0.915 ± 0.006 | 20.051 ± 0.180 |
| 新 C2 | 0.229 ± 0.008 | 0.216 ± 0.001 | 1.006 ± 0.021 | 21.663 ± 0.367 |
| 新 C3 | 0.062 ± 0.021 | 0.042 ± 0.000 | 1.463 ± 0.037 | 20.938 ± 0.152 |

**A2 与 PT-1 的差距集中在三个 W1 指标。** 新 A2 相比 PT-1：onset 误差高 17.1%，duration 高 17.7%，velocity 高 19.7%；tempo 高 0.032。Tempo Pearson 对全曲统一时间缩放不敏感，它衡量相对速度曲线形状，不能代表音符时长或绝对速度均更好。

| 指标 | A2 − PT-1 | 按作品配对 bootstrap 95% 区间 | A2 更好的作品数 |
|---|---:|---:|---:|
| tempo_curve_pearson | 0.0320 | [-0.0430, 0.0935] | 11/19 |
| onset_w1 | 0.0058 | [0.0033, 0.0086] | 2/19 |
| duration_w1 | 0.1177 | [0.0427, 0.1784] | 1/19 |
| velocity_w1 | 3.0910 | [1.8855, 4.2904] | 2/19 |

以上先对每首作品的三轮取均值，再对 19 首作品配对 bootstrap 10,000 次。Tempo 优势的区间跨零，尚不能认定 A2 稳定优于 PT-1；三项 W1 的差距则在多数作品上同向。该区间只衡量作品抽样，表内 std 只衡量生成采样，两者都不包含重复训练 seed 的不确定性。

额外诊断中，对每份 PT-1 输出再做一次全曲 beat-median 缩放，其 duration W1 为 0.656，原值为 0.665；tempo/onset/velocity 不变。因此时长差距主要不能由全曲典型速度偏置解释。此项只是后处理诊断，不属于主比较，也不是重训 PT。

**主要消融结论。** A2 相比 A1，仅将 beat/note duration 改为 log_dev，tempo 从 0.141 升至 0.332、duration 从 0.980 降至 0.783，说明显式使用 score 时长作为参照有帮助。A3 去掉 measure_start 后没有优于 A2；A1/A3 对照中则存在指标取舍，不能宣称小节标记对每项都提升。C1/C2 的 abs onset 明显劣于 dev；C3 的 shared_conditioned duration head 明显劣于 separate，现阶段应保留独立输出头。

**Teacher forcing 与 AR 的差距揭示后续重点。** 下面均为本轮原生事件评估，两个模式都做三轮采样；teacher forcing 为真实历史，AR 为生成历史，因此差异主要反映条件历史变化与误差传播，并非 greedy 与 sampling 的区别。

| 设置 | Beat log MAE TF→AR | Note duration log MAE TF→AR | Onset MAE TF→AR | Velocity MAE TF→AR |
|---|---:|---:|---:|---:|
| A1 | 0.161 → 0.320 | 0.657 → 0.980 | 0.039 → 0.040 | 11.555 → 22.581 |
| A2 | 0.188 → 0.209 | 0.657 → 0.783 | 0.039 → 0.040 | 11.324 → 18.814 |
| C1 | 0.166 → 0.287 | 0.646 → 0.915 | 0.070 → 0.207 | 11.245 → 20.051 |
| C2 | 0.183 → 0.221 | 0.650 → 1.006 | 0.081 → 0.264 | 11.272 → 21.663 |
| C3 | 0.341 → 0.521 | 0.834 → 1.463 | 0.038 → 0.042 | 11.415 → 20.938 |

A2 的 beat duration 对生成历史相对稳健，velocity 和 note duration 仍明显退化。C1/C2 在真实历史下 onset 已较差，AR 又将误差放大；残差 dev 让模型围绕已知 score 位置预测偏移，可能比重新预测拍内绝对位置更容易，但目前 abs/dev 同时使用不同 bins/range，不能把效应完全归因于表示形式。较低的 valid 总 CE 也不能跨不同 token 表示直接当作更好音乐表现：A1 valid CE 低于 A2，但 AR 指标落后。

**新归一化不是本轮 timing 提升的主要证据。** 在相同新参考与观测下，旧 A2→新 A2 的 tempo 为 0.337→0.338，duration 为 0.785→0.783，几乎持平；velocity 为 22.223→18.814，降低 15.3%。velocity 标签未改变，时间标签变化会影响联合学习，但单个训练 seed 不足以把这项提升稳定归因于归一化。统一归一化仍有明确的数据一致性价值，尤其消除了两个 raw MS-score fallback。

**指标本身也需对照简单基线。** 在同一共同范围内，将所有 onset dev 设为 0（照 score 拍内位置演奏）的 W1 为 0.0331，A2 为 0.0395，PT-1 为 0.0337。说明当前 onset W1 不能单独证明学到了更好的微时值表达。W1(singleton, human) 偏好人类逐音符中位数；使用测试人类标签构造的理论最小值约为 onset 0.0116、duration 0.162、velocity 3.776。这只是不可部署的参考下界，不是可达到性承诺，也不衡量生成多样性。

**建议的后续顺序：** 先固定 A2，用多个训练 seed 确认优势；在 valid 上对 sampling temperature / top-p 与 greedy 做小范围比较，检查 velocity 的 AR 退化及动态范围，避免直接按 test 调参。再考虑改善生成历史鲁棒性、给 velocity 单独建模；暂不扩大 abs onset 或共享 duration 输出头的搜索。PT-1 使用预训练初始化，而本轮 A2 为随机初始化，残余差距同时混合了预训练、架构、mask 和表示因素，不能直接判定 BEAT 架构存在上限。

![A2 与 PT-1 的逐作品指标差值](../analysis/exp1117_beat_median/result_analysis/a2_vs_pt_per_work.png)

[完整复核 JSON、逐作品差值与 bootstrap](../analysis/exp1117_beat_median/result_analysis/analysis.json) · [复核脚本](../script/analyze_exp1117_beat_median.py)

### 7.6 2026-11-19 人类留一基线与 Onset Dev 指标解释

补算 human→human：每次拿一个完整人类演奏作为预测，仅与该作品其余人类演奏比较，排除自身。19 首 test work 中有 7 首只有一个演奏，无法留一，因此下表全部限定在其余 **12 works、75 performances**；PT-1/A2 也按完全相同的作品、共同 mask 和留一人类参考重算。先平均每首作品内留一演奏，再对作品 macro-average；模型再平均三轮 AR。它与第 7.5 节的 19-work 数字不同。

| 对照 | Tempo Pearson ↑ | Onset dev W1 ↓ | Duration log-ratio W1 ↓ | Velocity W1 ↓ |
|---|---:|---:|---:|---:|
| Human→其他 Human（留一） | 0.769 | 0.0311 | 0.439 | 10.230 |
| PT-1 | 0.349 | 0.0340 | 0.665 | 15.819 |
| A2 | 0.403 | 0.0385 | 0.782 | 18.962 |
| 固定 onset dev=0 | — | 0.0331 | — | — |

逐音符人类中位数对全部同作品人类的 W1 最小值为 onset 0.0184、duration 0.256、velocity 5.979。这是使用参考标签构造的样本内数学下界，**不是 human→human 指标**。此前 7.5 节下界还包含 7 首单人演奏作品，单人位置的样本内误差为零，因此不能拿来充当人类表现。

**需要区分 onset dev 表示与当前标量评分。** Onset dev 保留音符相对局部 beat 的细微位置偏移，能描述拍内伸缩和同起音组的不同步；全曲、段落层面的 rubato 大部分已经被 beat curve 吸收。它作为训练 target 或微时值特征仍然合理，但不能单独评价整体节奏表达。

例如每拍只有一个音、且都位于 score 拍点时，以这些起音映射出的 beat anchor 会经过每个音符：即使演奏中的相邻 beat 时长变化很大，onset dev 仍可全为零。这些 rubato 应由 tempo curve 衡量。

当前分数是 $W_1(\delta_{\hat d_i},P_i^{human})=\frac{1}{H}\sum_h|\hat d_i-d_i^{(h)}|$，然后对 note、work、draw 平均。它等价于逐音符平均绝对误差，被人类该音符的中位数最小化。即使模型完整复现真实演奏分布，随机抽到的演奏也可能比中位数模板分数更差；先逐 draw 算再平均，仍然不会奖励生成的多样性。**同一问题也适用于当前 duration/velocity W1，不仅是 onset。**

**因此建议保留 onset dev，但将当前 onset W1 定位为偏差诊断。** 分开评价：(1) tempo curve 的形状；(2) 拍内 onset dev 的位置、幅度和分布；(3) 同 score onset group 的异步起音/声部关系。分布可比较每个音符多次生成与多人参考的 W1，或使用 CRPS/energy score 这类适当评分；当前仅 3 次生成、部分作品仅 1–2 个参考，分布估计会有较大有限样本误差，需配套同采样数的人类基线和区间。不要为了降低此项误差而直接把表达压向零。

[逐作品、逐演奏、采样波动与人类覆盖审计](../analysis/exp1117_beat_median/result_analysis/human_baseline.json) · [计算脚本](../script/evaluate_exp1117_human_baseline.py)

### 7.7 2026-11-19 追加：多次生成分布与多人参考分布的真正 W1

对每个 score note，把 3 次生成视作各占 1/3 权重的经验分布，把该作品的 H 个人类演奏视作各占 1/H 权重的经验分布，计算两者的 W1，先逐 note 平均，再 macro-average works。采用精确一维分位数积分，支持两侧样本数不相等；不把不同音符混在一起，也不逐生成轮次算距离后再平均。

本节三项距离使用与第 7.5 节相同的新 beat-median 人类参考与音符重提取方式；复用已有三轮 AR 预测，未重训或重新采样。覆盖扩展到 exp1112 的 24 项 DINR、2 项 PT、旧六项 BEAT 和新六项 BEAT，共 38 项，统一所有方法三轮的共同有效位置，因此覆盖与 7.5 节略有差异。旧表保留，旧三项数字应按“逐生成样本对人类的平均绝对误差”理解。

19 works、82 human performances；共同有效 onset 64,885/65,039 notes，duration 63,169/65,039 notes，velocity 64,145/65,039 notes。7 首单演奏作品仍包含在完整表中；当人类参考只有 1 个时，经验分布 W1 就等于三次生成对该参考的平均绝对误差，无法据此判断真实演奏分布。

| 设置 | Onset dev 分布 W1 ↓ | Duration log-ratio 分布 W1 ↓ | Velocity 分布 W1 ↓ |
|---|---:|---:|---:|
| PT-1 | 0.0308 | 0.6092 | 14.6870 |
| PT-scratch | 0.1463 | 2.6697 | 37.8082 |
| 旧 A2 | 0.0367 | 0.7312 | 21.3462 |
| 新 A1 | 0.0363 | 0.9120 | 21.1617 |
| 新 A2 | 0.0364 | 0.7292 | 17.6552 |
| 新 A3 | 0.0362 | 0.8677 | 23.1377 |
| 新 C1 | 0.1741 | 0.8485 | 18.4912 |
| 新 C2 | 0.2112 | 0.9521 | 20.3594 |
| 新 C3 | 0.0391 | 1.4297 | 19.8676 |

| exp1112 log dev 设置 | Onset dev 分布 W1 ↓ | Duration log-ratio 分布 W1 ↓ | Velocity 分布 W1 ↓ |
|---|---:|---:|---:|
| DINR separate / timestep 1+pitch | 0.0560 | 0.8576 | 23.7853 |
| DINR separate / timestep | 0.0540 | 0.8289 | 24.7741 |

改成多次生成分布 W1 后，A2 仍是新六项设置中 duration、velocity 最好的一项；onset 上 A1/A3 略低。A2 相比 PT-1 的 onset、duration、velocity 距离分别高 18.3%、19.7%、20.2%，因此主要排序结论没有反转；逐作品配对结果与区间见 JSON。

三次生成合成一个经验分布后只有一个距离，不能再把三轮当作三个 W1 结果报告 std。JSON 提供条件于这批固定预测的作品 bootstrap 区间，以及 19 首、12 首多人演奏作品、8 首至少四演奏作品等分组汇总；这些区间不包含重复训练及新增生成样本的不确定性。Tempo Pearson 的定义没有改变，本节不把它改成分布距离。

**A2 / PT-1：旧聚合方式与真正分布 W1，在本节同一覆盖上比较。**

| 设置 | Onset：旧→分布 W1 | Duration：旧→分布 W1 | Velocity：旧→分布 W1 |
|---|---:|---:|---:|
| PT-1 | 0.0337 → 0.0308 | 0.6613 → 0.6092 | 15.7151 → 14.6870 |
| New BEAT A2 | 0.0395 → 0.0364 | 0.7824 → 0.7292 | 18.7787 → 17.6552 |

同范围的固定 onset dev=0 基线为 0.0331。该确定性分布到人类分布的 W1 与旧平均绝对误差相等。所有方法均数值验证：把同一 note 的预测与参考一起加回 score 拍内位置，abs onset W1 与 onset dev W1 一致。

**相同三样本数的人类分布对照：不与自身比较。**

要取 3 个人类演奏组成预测分布且参考侧仍非空，只能使用 H≥4 的 **8 works、65 human performances**。穷举每首作品所有三人子集，将剩余 H−3 人作为参考；PT-1/A2 的三个生成也对完全相同的剩余参考计算 W1。共 811 个互不重叠的人类预测/参考划分，先在作品内平均划分，再 macro-average works。

| 同一 H≥4 子集 | Onset dev 分布 W1 ↓ | Duration log-ratio 分布 W1 ↓ | Velocity 分布 W1 ↓ |
|---|---:|---:|---:|
| 3 Human → 其余 Human | 0.0251 | 0.3187 | 7.9588 |
| PT-1：3 次生成 | 0.0314 | 0.5761 | 14.0279 |
| A2：3 次生成 | 0.0362 | 0.6906 | 18.4177 |

这是有限样本的人类参照，不是零误差下界。当前只有三个生成样本，增加采样数可以改善分布估计，但不保证数值单调下降；本节没有重新生成 greedy 输出，样本中位数也不能冒充 greedy。逐音符分布 W1 评价各音符的边缘分布，不直接评价跨音符的相关结构或完整演奏风格一致性。

[全部 38 项、逐作品 W1、配对 bootstrap 与人类分布对照](../analysis/exp1117_beat_median/result_analysis/distribution_w1/metrics.json) · [完整 CSV](../analysis/exp1117_beat_median/result_analysis/distribution_w1/metrics.csv) · [计算脚本](../script/evaluate_exp1117_distribution_w1.py)

### 7.8 2026-11-19 Onset dev=0 与多样性诊断

W1 是通用的分布距离。若真实分布可直接获得，W1(P,Q)=0 当且仅当两分布相同，因此总体目标不会专门偏好均值或中位数；但它衡量的是分布是否匹配，不是多样性越大越好。过窄、过宽、整体偏移都可能产生距离。

实际比较的是仅 3 次生成和有限人类参考构成的经验分布，不能保证完全消除偏好收缩的现象。特别是单个人类参考时，W1 退化成到该参考的平均绝对误差；随机人类参考下，期望仍由确定性中位数最小化。此前关于“不偏好均值”的总体性质不能直接套到这种小样本评分上。

一个可穷举的例子：真实分布在 −1、0、1 上等概率。模型若从真实分布独立生成 3 次，与独立抽到的 1 个人类参考计算经验 W1，期望为 8/9；模型若永远输出 0，期望为 2/3，反而更低。但如果直接比较真实分布，正确模型的 W1 为 0，固定零分布为 2/3。问题来自有限样本评分，不能据此声称总体 W1 偏好固定均值。

下表均使用各作品的全部人类参考和同一共同 mask。多人子集排除单演奏作品，可单独检查零偏移对照：

| 同一范围 | 固定 dev=0 W1 ↓ | PT-1 分布 W1 ↓ | A2 分布 W1 ↓ |
|---|---:|---:|---:|
| 19 works / 82 humans | 0.0331 | 0.0308 | 0.0364 |
| 12 works / 75 humans | 0.0331 | 0.0294 | 0.0336 |
| 8 works / 65 humans | 0.0354 | 0.0304 | 0.0350 |
| 5 works / 51 humans | 0.0389 | 0.0313 | 0.0367 |

在第 7.7 节的 8 首作品、811 个不重叠人类三样本划分中，零偏移对照也必须对相同剩余人类参考评价。对全部划分平均时，每个人类演奏的权重相同，所以零偏移平均 W1 恰好等于这 8 首作品使用全部人类参考的值：

| 8-work、相同剩余参考 | Onset dev W1 ↓ |
|---|---:|
| 3 Human → 其余 Human | 0.0251 |
| PT-1：3 次生成 | 0.0314 |
| 固定 dev=0 | 0.0354 |
| A2：3 次生成 | 0.0362 |

**另报生成多样性，而不是只看 W1。** 下表固定每个 score note，计算不同演奏/生成之间的无偏样本方差（ddof=1），逐 note、work 平均，再开平方。单位仍为 beat fraction；它表示同一音符在不同演奏中的 RMS 变化幅度，不是把不同音符混在一起的方差。单演奏作品无法估计人类方差，已排除。

| 范围 | Human RMS spread | PT-1 RMS spread | A2 RMS spread | 固定 dev=0 |
|---|---:|---:|---:|---:|
| 12 works | 0.0399 | 0.0311 | 0.0279 | 0 |
| 8 works | 0.0440 | 0.0283 | 0.0287 | 0 |
| 5 works | 0.0513 | 0.0279 | 0.0277 | 0 |

按这个总体幅度统计，两模型在多人作品上的 onset 跨演奏变化均小于人类；这不支持把 A2 的距离较大简单解释为“生成太多样”。模型每音符只有 3 次生成，方差估计噪声较大；汇总 RMS 也会受少数大偏差影响，应同时看逐作品结果。零偏移分数低说明“保守的微时值”是有竞争力的对照；不能据此认定它具有更好的人类表达。W1 同时受位置与散布影响，spread 单独也不能验证变化是否在正确的音符、方向和结构上。当前仍只评价逐音符边缘分布，无法检测完整风格、跨音符相关性或短语连贯性。

[零偏移、样本内均值/中位数对照及逐作品方差](../analysis/exp1117_beat_median/result_analysis/distribution_w1/metrics.json)

### 7.9 2026-11-19 改用均值 MAE＋生成 VAR 分开报告

为提高可解释性，本节分开报告“预测中心是否准确”和“生成变化是否足够”。每个 score note 的生成均值来自 3 次生成，人类均值来自同作品 H 个演奏；均值 MAE 是二者差值的绝对值，先逐 note 平均再 macro-average works。它不同于此前每个生成值分别对人类参考计算的 MAE，也不等于先平均整首作品再计算差值。

VAR 固定同一 score note，在三次生成之间计算无偏样本方差（ddof=1），再逐 note、work 平均；人类 VAR 也固定同一音符，在不同人类演奏间计算。只在人类 H≥2 的作品上比较两边 VAR，不把单个人类演奏的方差当成零。VAR 单位为属性单位的平方。

目标是均值 MAE 更小，并保留与人类相称的生成 VAR。VAR 不应无限增大：随机噪声也能使其增大，合理做法是同时列出 Human VAR 和 Generated VAR / Human VAR；比值 1 表示平均变化幅度匹配，比值大于 1 表示更分散，小于 1 表示更集中。比值为聚合方差之比，不平均各音符方差比，避免人类方差接近零时数值爆炸。

**主对照：12 首多人演奏作品、75 个演奏，MAE 与 VAR 使用同一范围。** 使用第 7.7 节 38 项共同有效 mask；仍保留 score note 身份，模型输出、归一化和音符重提取方式均不变。

**Onset dev（beat fraction）**

| 设置 | 均值 MAE ↓ | 生成 VAR | VAR / Human VAR |
|---|---:|---:|---:|
| Human 参考 | — | 0.00159113 | 1.000 |
| PT-1 | 0.02503 | 0.000969046 | 0.609 |
| A1 | 0.03032 | 0.000907394 | 0.570 |
| A2 | 0.02949 | 0.000776788 | 0.488 |
| A3 | 0.03029 | 0.000983725 | 0.618 |
| C1 | 0.14545 | 0.0572138 | 35.958 |
| C2 | 0.17122 | 0.160886 | 101.114 |
| C3 | 0.03262 | 0.00134909 | 0.848 |
| 固定 onset dev=0 | 0.02840 | 0 | 0 |

**Duration log-ratio**

| 设置 | 均值 MAE ↓ | 生成 VAR | VAR / Human VAR |
|---|---:|---:|---:|
| Human 参考 | — | 0.190123 | 1.000 |
| PT-1 | 0.50748 | 0.268895 | 1.414 |
| A1 | 0.78723 | 0.719585 | 3.785 |
| A2 | 0.64507 | 0.275845 | 1.451 |
| A3 | 0.78547 | 0.554761 | 2.918 |
| C1 | 0.71754 | 0.459537 | 2.417 |
| C2 | 0.83444 | 0.505645 | 2.660 |
| C3 | 1.41599 | 0.62002 | 3.261 |

**Velocity（MIDI value）**

| 设置 | 均值 MAE ↓ | 生成 VAR | VAR / Human VAR |
|---|---:|---:|---:|
| Human 参考 | — | 91.7605 | 1.000 |
| PT-1 | 13.28984 | 88.1363 | 0.961 |
| A1 | 18.58411 | 267.996 | 2.921 |
| A2 | 16.03383 | 138.351 | 1.508 |
| A3 | 21.55749 | 217.844 | 2.374 |
| C1 | 15.83788 | 256.861 | 2.799 |
| C2 | 17.11335 | 198.057 | 2.158 |
| C3 | 18.33260 | 136.272 | 1.485 |

这组拆分给出更直接的诊断：A2 的 onset VAR 是 Human 的 48.8%，而 duration、velocity VAR 已分别达到 145.1%、150.8%。因此不能统一要求所有属性 VAR 更大；需要先降低均值误差，再看变化幅度是否与人类匹配。PT-1 三项均值 MAE 均小于 A2。A2 的 velocity 平均有符号偏差为 +10.70 个 MIDI velocity 单位，提示存在整体偏响；该总体偏差并不代表每首作品、每个音符都偏响。

**完整 19 首的均值 MAE：单演奏作品也可以计算均值误差，但不能估计人类 VAR。**

| 设置 | Onset dev 均值 MAE ↓ | Duration log-ratio 均值 MAE ↓ | Velocity 均值 MAE ↓ |
|---|---:|---:|---:|
| PT-1 | 0.02729 | 0.54545 | 13.73158 |
| A2 | 0.03201 | 0.67115 | 16.28873 |
| 固定 onset dev=0 | 0.03013 | — | — |

固定零偏移的均值 MAE 是 abs(mean(human onset dev))，不再是 mean(abs(human onset dev))，因此与第 7.8 节固定零偏移 W1 不相同。人类不同演奏正负偏移可以在均值中抵消，这正是“接近人类平均表达”这个目标的含义。

这两个量容易解释，但均值＋方差并不确定整个分布，仍可能遗漏双峰、多峰或跨音符相关性。每个音符仅 3 次生成，样本均值与方差都不稳定，均值 MAE 也会受到采样噪声影响；两者应并列报告，不合成“MAE 减 VAR”的无限奖励。Human 参考均值对自身误差为零只是定义，不是独立 human→human 基线。

[全部 38 项、逐作品、分组与定义 JSON](../analysis/exp1117_beat_median/result_analysis/mean_variance/metrics.json) · [CSV](../analysis/exp1117_beat_median/result_analysis/mean_variance/metrics.csv) · [计算脚本](../script/evaluate_exp1117_mean_variance.py)

### 7.10 2026-11-19 Human、Score MIDI 与模型的 MAE / VAR 统一表

**Human 的均值 MAE=0 仅指参考均值对自身。** 把全部人类参考的逐音符均值作为预测，再与同一均值比较，误差必为 0；这一行用于标注目标中心，Human VAR 仍来自真实不同演奏，不代表存在一条确定性 MIDI 同时具有这个均值和人类方差。独立人类演奏或独立三人均值与其余人类比较，MAE 通常不为 0。

**主表统一使用 12 首多人作品、75 个演奏。** 每个模型用 3 次生成的均值与全部人类参考均值比较；VAR 为固定同一 score note 的跨演奏/跨生成样本方差（ddof=1），先平均 note，再 macro-average works。Human 参考行列出目标中心 MAE=0 和人类 VAR。

**Score MIDI 基线使用完整乐谱属性。** Onset dev=0；duration log-ratio=ln(D_score/D_score)=0；velocity 使用 score MIDI 自身的逐音符 velocity，不设为零或人类均值。重复同一 score 三次，三个属性的跨生成 VAR 均为 0；这不表示乐曲内部所有音符的数值都相同。

| 设置 | Onset 均值 MAE ↓ | Onset VAR | Duration 均值 MAE ↓ | Duration VAR | Velocity 均值 MAE ↓ | Velocity VAR |
|---|---:|---:|---:|---:|---:|---:|
| Human（参考中心，对自身） | 0.00000 | 0.00159113 | 0.00000 | 0.190123 | 0.00000 | 91.7605 |
| Score MIDI | 0.02840 | 0 | 0.69909 | 0 | 20.92694 | 0 |
| PT-1 | 0.02503 | 0.000969046 | 0.50748 | 0.268895 | 13.28984 | 88.1363 |
| A1 | 0.03032 | 0.000907394 | 0.78723 | 0.719585 | 18.58411 | 267.996 |
| A2 | 0.02949 | 0.000776788 | 0.64507 | 0.275845 | 16.03383 | 138.351 |
| A3 | 0.03029 | 0.000983725 | 0.78547 | 0.554761 | 21.55749 | 217.844 |
| C1 | 0.14545 | 0.0572138 | 0.71754 | 0.459537 | 15.83788 | 256.861 |
| C2 | 0.17122 | 0.160886 | 0.83444 | 0.505645 | 17.11335 | 198.057 |
| C3 | 0.03262 | 0.00134909 | 1.41599 | 0.62002 | 18.33260 | 136.272 |

Onset 单位为 beat fraction；duration 使用自然对数比；velocity 为 MIDI 数值，VAR 单位分别为这些单位的平方。Human 行的 0 不应被当作可独立实现的生成模型性能。

**独立 Human 对照：8 首至少四演奏作品、65 个演奏，不能与上面的 12-work 数字直接混比。** 每首作品穷举选择 3 个完整人类演奏，把剩余 H−3 个演奏作为参考。Human MAE 是三人均值对剩余人类均值；所有模型和 Score MIDI 对完全相同的剩余参考算均值 MAE。先平均作品内划分，再 macro-average works，共 811 个划分；预测侧和参考侧没有重复人类演奏。

| 设置（同一剩余参考） | Onset 均值 MAE ↓ | Onset VAR | Duration 均值 MAE ↓ | Duration VAR | Velocity 均值 MAE ↓ | Velocity VAR |
|---|---:|---:|---:|---:|---:|---:|
| Human（3 人对其余人） | 0.01921 | 0.00193763 | 0.24794 | 0.177562 | 6.07547 | 96.3686 |
| Score MIDI | 0.03096 | 0 | 0.67251 | 0 | 20.69577 | 0 |
| PT-1 | 0.02677 | 0.00080187 | 0.50159 | 0.278404 | 13.05765 | 80.5402 |
| A1 | 0.03243 | 0.000945131 | 0.78733 | 0.705322 | 18.89422 | 268.049 |
| A2 | 0.03164 | 0.000821748 | 0.63442 | 0.276226 | 17.18004 | 144.547 |
| A3 | 0.03213 | 0.000960876 | 0.76675 | 0.571574 | 22.05338 | 225.673 |
| C1 | 0.14935 | 0.0707362 | 0.73147 | 0.479404 | 16.22863 | 284.825 |
| C2 | 0.16890 | 0.21165 | 0.79474 | 0.489056 | 16.60161 | 201.14 |
| C3 | 0.03462 | 0.00126895 | 1.39197 | 0.61536 | 19.21092 | 149.174 |

穷举三人子集后，人类三样本的无偏 VAR 平均值等于该作品全部人类的无偏 VAR；因此 Human 的 VAR 可作为同样本数参照，但它的独立均值 MAE 不再为零。Score 的 VAR=0 只表示完全确定，不代表其表达优于有变化的人类演奏。

[统一主表、独立 Human 对照及逐作品 JSON](../analysis/exp1117_beat_median/result_analysis/human_score_mean_variance/metrics.json) · [CSV](../analysis/exp1117_beat_median/result_analysis/human_score_mean_variance/metrics.csv) · [计算脚本](../script/evaluate_exp1117_human_score_mean_variance.py)

### 7.11 2026-11-21 当前五项主指标：两条曲线相关与三项 MAE

范围按最新确认的对照集合：Human、Score MIDI、PT-1、A1–A3、C1–C3。PT-1 与六个新 BEAT 设置复用保存好的预测，不重新训练或采样；使用新 beat-median 人类参考、score-note 对齐和这七个模型的共同有效 mask。此前 38 项的探索性对照不纳入本节。

**两项 Pearson 均按逐个人类演奏计算后取平均。** 每个生成结果分别与同一作品的每个人类演奏比较，再在人类演奏间取均值；不先合成人类中位数曲线。先对作品内人类相关取平均，再对作品 macro-average。每个生成样本各算一次，表中相关报告三次 AR 的 mean ± sample std（ddof=1）。Human 基线逐次留出一个完整演奏，与其余演奏比较，排除自身。

Pearson 对曲线整体加常数、乘正数不敏感；它衡量逐拍形状，不衡量力度绝对水平。统一力度增减由 Velocity MAE 反映。

**Velocity curve 定义：** 按 score onset 所属 beat 聚合该拍新起音音符，取有效 note velocity 的算术平均。Beat 内无有效 note 时该 beat 记缺失，不填零、不延续前值；相关只用相同的有效 beat。主定义用平均值；median/max 作为不同统计量，不与此结果混用。Score MIDI 的 velocity 使用 score note velocity。

**三项 MAE 是中心误差。** 对每个 score note，先平均同一 checkpoint 的 3 次生成，再与该 note 的人类演奏均值比较绝对误差，逐 note、逐作品平均。Onset 用拍内 dev，duration 用 ln(D_perf/D_score)，velocity 用 MIDI velocity。它们不是三次 MAE 再平均。表中 MAE 无 sampling std，因为已在三次生成上先求均值。

完整覆盖：19 works、82 human performances、65,039 score notes；velocity curve 使用 12,380 个有共同有效 note 的 beat；tempo curve 使用 13,372 个有效 beat。

| 方法 | Tempo curve Pearson ↑ | Beat velocity curve Pearson ↑ | Onset dev MAE ↓ | Duration log-ratio MAE ↓ | Velocity MAE ↓ |
|---|---:|---:|---:|---:|---:|
| Score MIDI | 0.561 ± 0.000 | 0.493 ± 0.000 | 0.0301 | 0.7092 | 21.969 |
| PT-1 | 0.436 ± 0.027 | 0.405 ± 0.021 | 0.0273 | 0.5493 | 13.743 |
| A1 | 0.198 ± 0.011 | 0.062 ± 0.014 | 0.0322 | 0.7960 | 19.178 |
| A2 | 0.451 ± 0.006 | 0.100 ± 0.006 | 0.0320 | 0.6715 | 16.332 |
| A3 | 0.212 ± 0.015 | 0.071 ± 0.009 | 0.0321 | 0.7885 | 21.413 |
| C1 | 0.136 ± 0.027 | 0.055 ± 0.016 | 0.1521 | 0.7471 | 16.299 |
| C2 | 0.327 ± 0.005 | 0.063 ± 0.017 | 0.1786 | 0.8559 | 18.214 |
| C3 | 0.112 ± 0.026 | 0.058 ± 0.014 | 0.0344 | 1.3733 | 18.531 |

Score MIDI 使用 onset dev=0、duration log-ratio=0、score MIDI 自身 velocity。其力度曲线按每拍 score note velocity 均值计算。为保证相关数值使用相同作品集合，Tempo corr 在 12 首曲目上计算，Velocity corr 在 19 首曲目上计算；Tempo 的 score curve 在 7 首曲目上为常数，因此 Pearson 无定义，这些作品对所有方法一并排除。

**留一演奏 Human 基线。** 取至少有 2 个演奏的 12 首作品、75 个演奏。每次留出 1 人作为 Human prediction，其余 H−1 人作为 reference；Score、PT-1、A1–A3、C1–C3 在相同留出划分下与剩余人类比较。共 75 个 leave-one-performance-out 拆分。Human 的 MAE 是独立留出演奏对其余演奏均值的误差，不是自比的零。

| 方法 | Tempo curve Pearson ↑ | Beat velocity curve Pearson ↑ | Onset dev MAE ↓ | Duration log-ratio MAE ↓ | Velocity MAE ↓ |
|---|---:|---:|---:|---:|---:|
| Human：留一演奏 | 0.713 | 0.755 | 0.0254 | 0.3667 | 8.361 |
| Score MIDI | 0.610 | 0.514 | 0.0291 | 0.7100 | 21.033 |
| PT-1 | 0.326 | 0.405 | 0.0259 | 0.5262 | 13.494 |
| A1 | 0.154 | 0.074 | 0.0310 | 0.8040 | 18.789 |
| A2 | 0.377 | 0.116 | 0.0302 | 0.6569 | 16.301 |
| A3 | 0.157 | 0.079 | 0.0310 | 0.7978 | 21.732 |
| C1 | 0.090 | 0.062 | 0.1457 | 0.7321 | 16.055 |
| C2 | 0.250 | 0.103 | 0.1714 | 0.8428 | 17.343 |
| C3 | 0.059 | 0.089 | 0.0333 | 1.4233 | 18.551 |

留一基线 Human prediction 只有 1 个演奏；模型 MAE 仍用 3 次生成均值。这两个估计量的样本数不同，Human 行用于表示真实演奏对其他人类的可预测程度，不等于模型对同等样本数基线。Human–Human corr 排除了自身配对；Pearson 无定义的平坦曲线配对会排除并计数。

[所选方法逐作品/逐人类明细与 mask 审计](../analysis/exp1117_beat_median/result_analysis/five_main_metrics/metrics.json) · [CSV](../analysis/exp1117_beat_median/result_analysis/five_main_metrics/metrics.csv) · [计算脚本](../script/evaluate_exp1117_five_main_metrics.py)
<!-- EXP1117_BEAT_MEDIAN_END -->

### 7.12 2026-11-21 Beat velocity 诊断：问题主要出现在 autoregressive feedback

针对 7.11 中较低的 beat velocity curve Pearson，重新检查了六个当前 BEAT checkpoint 的 loss。原训练日志只有四个属性相加后的 total loss，因此在相同 valid sidecar 上用最佳 checkpoint 离线重算四个独立的 categorical cross-entropy；这不会改变模型或原实验结果。

| 设置 | Valid velocity CE | Teacher-forced velocity MAE | Autoregressive velocity MAE | Teacher-forced beat-velocity corr | Autoregressive beat-velocity corr | AR signed velocity error |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 3.710 | 11.56 | 22.06 | 0.726 | 0.063 | 13.42 |
| A2 | 3.691 | 11.31 | 18.54 | 0.721 | 0.098 | 8.66 |
| A3 | 3.713 | 11.66 | 24.37 | 0.725 | 0.072 | 17.16 |
| C1 | 3.716 | 11.25 | 20.09 | 0.728 | 0.056 | 8.11 |
| C2 | 3.710 | 11.26 | 21.56 | 0.728 | 0.063 | 11.72 |
| C3 | 3.708 | 11.42 | 20.82 | 0.721 | 0.059 | 13.07 |

Velocity CE 约为 3.69–3.72；128 类均匀预测的 CE 是 ln(128)=4.85，所以 velocity head 已经学到有用的条件分布。Teacher-forced beat velocity curve corr 约 0.72，也接近 Human leave-one-out 的 0.755。相反，autoregressive 相关全部降到 0.056–0.098，velocity MAE 明显增大，并出现正的 signed error。以 A2 为例，测试生成的 velocity 均值约 73.80，而人类均值约 66.71；标准差约 15.81，而人类约 18.60。这是整体偏响并伴随动态范围收缩的漂移。

当前实现会把每个生成的 velocity 写入下一个 decoder input；训练时 decoder history 仍以 ground-truth 为主，只进行 50% 的随机 mask。因此早期 velocity 误差会改变后续 hidden state，造成 exposure mismatch 和累积漂移。这个诊断说明力度曲线差确实是自由生成性能的问题，但不能归因于 velocity head 单步学习能力不足。优先应测试减少 velocity feedback 的 exposure bias（scheduled/generated-history training，或不把 velocity 反馈给下一步），再考虑在 CE 之外加入 ordinal/continuous velocity loss；仅提高 velocity loss 权重不是当前证据支持的首要修复。

[训练与 valid total loss 曲线](../analysis/exp1117_beat_median/velocity_diagnostics/loss_curves.png) · [四个 head 的 valid CE](../analysis/exp1117_beat_median/velocity_diagnostics/head_ce.png) · [分头 loss 与 teacher-forced/AR 诊断](../analysis/exp1117_beat_median/velocity_diagnostics/diagnostics.json) · [诊断说明](../analysis/exp1117_beat_median/velocity_diagnostics/diagnosis.md) · [计算脚本](../script/diagnose_exp1117_velocity.py)

### 7.13 2026-11-22 PT-1 与 A2 的 teacher-forced beat velocity 对照

为了判断低 beat velocity curve 是否主要来自 AR 误差传播，对 PT-1 checkpoint 也做了同口径的 teacher-forced 推理。PT-1 没有现成的 BEAT 事件输出，因此直接用 PT-1 checkpoint，在每首作品的每个人类演奏 history 下前向预测；预测结果按当前 beat-median score identity 聚合为每拍新起音 velocity 均值，再逐人类演奏计算 Pearson，最后对作品取 macro-average。PT-1 使用 3 次 top-p=0.9 TF sampling；A2 数字来自 7.12 的 3 次 TF sampling。两者都使用 19 首作品和同一批当前人类参考。

| 方法 | Beat velocity curve Pearson ↑ | Velocity MAE ↓ | 生成 velocity mean | 生成 velocity std | Signed mean error |
|---|---:|---:|---:|---:|---:|
| Human leave-one-out | 0.755 | 8.361 | 66.71 | 18.60 | 0 |
| PT-1 TF | **0.832** | **8.917** | 64.07 | 15.24 | 1.20 |
| A2 TF | 0.721 | 11.31 | 67.00 | 16.90 | 0.29 |

PT-1 TF 的 beat velocity curve corr 比 A2 TF 高约 **0.111**，velocity MAE 低约 **2.39**。因此“只要使用真实历史，A2 就已经达到 PT-1 水平”的解释不成立：即使在 TF 条件下，A2 的力度曲线和逐音符力度仍明显弱于 PT-1。结论需要区分两点：

1. A2 的 AR 退化仍然真实存在：A2 的 velocity curve corr 从 TF 的 0.721 降到 AR 的约 0.098，说明 exposure bias 会进一步放大问题。
2. TF 下 PT-1 仍优于 A2，说明还存在模型本身的差异。PT-1 使用预训练初始化和 8-token-per-note 表示，A2 使用随机初始化和 beat/event 表示；本次结果不能把差距唯一归因于 beat 表示，但可以排除“问题完全只是 TF/AR 切换”的说法。

PT-1 TF 结果与逐作品明细保存在 [metrics.json](../analysis/exp1117_beat_median/pt1_teacher_forced_velocity/metrics.json) 和 [rows.json](../analysis/exp1117_beat_median/pt1_teacher_forced_velocity/rows.json)，计算脚本为 [evaluate_pt1_teacher_forced_beat_velocity.py](../script/evaluate_pt1_teacher_forced_beat_velocity.py)。

### 7.14 2026-11-22 Score MIDI 曲线基线独立审计

针对 Score MIDI 的 tempo/velocity Pearson 高于 PT-1 和 A2，独立复算了 Score MIDI 曲线，并逐音符检查 score velocity 的来源与对齐。19 首作品中，BEAT sidecar 的每个 note `events[...,4]` 都与对应 refined score MIDI 按相同 `(start, pitch, end, velocity)` 顺序排列后的 `note.velocity` 完全相同，**velocity identity mismatch 为 0**。Beat membership 则由 score onset tick 映射，并由参考加载逻辑验证与 BEAT score-onset ratio 一致。复算使用主表同一个 common mask、每拍 score note velocity 算术均值、逐人类演奏 Pearson，再先对作品内人类取均值、最后对作品 macro-average。

| 曲线 | 独立审计 | 主表 | 覆盖作品 |
|---|---:|---:|---:|
| Score MIDI tempo Pearson | 0.561266 | 0.561266 | 12 |
| Score MIDI beat velocity Pearson | 0.492504 | 0.492504 | 19 |

因此当前查到的证据不支持“Score MIDI 指标计算错了”：同一 Score MIDI 曲线经独立实现复算得到完全相同的结果，且力度输入逐音符核对通过。原始输入 score MIDI 本身含有非恒定的 velocity/dynamic 信息；19 首中多首有明显的 beat-level 动态起伏。因此“无预测能力时输出输入”并非无信息基线：它直接保留了谱面已有的动态轮廓。Pearson 不要求输出绝对力度正确，只要拍间形状吻合就会高；Score MIDI 的 velocity MAE 仍为 21.969，明显比 PT-1 的 13.743 和 A2 的 16.332 大。它是一个有信息的曲线形状基线，不能解释为 score velocities 等于人类力度或绝对响度正确。

Tempo 也是同理：Score MIDI 直接使用谱面 BEAT duration 曲线；若 score tempo 变化与人类段内速度轮廓相关，直接保留 score 曲线可以获得较高 corr。这个结果表明 PT-1 和 A2 尚未稳定学出比谱面 tempo 标记更贴近人类的相对速度形状。Score tempo curve 在 7 首作品上为常数，Pearson 未定义；为公平比较，这 7 首从所有方法共同排除，余下 12 首的复算与主表完全一致。

本审计没有改动五项主指标或既有结果。逐作品 identity、Pearson 和分布统计见 [score_midi_audit.json](../analysis/exp1117_beat_median/result_analysis/five_main_metrics/score_midi_audit.json)，审计脚本为 [audit_score_midi_curve_baseline.py](../script/audit_score_midi_curve_baseline.py)。

### 7.15 2026-11-22 五首 test 作品的 beat velocity 与 beat duration 曲线

重新绘图后，每个作品固定为三列：左侧 Human（每条人类演奏单独绘制），中间 PT-1（3 个生成样本分别绘制），右侧 A2（3 个生成样本分别绘制）。不绘制 Human mean。三列都保留 Score MIDI 黑线作为共同参照。Beat duration 使用与 tempo curve 相同的 `ln(duration_ms)` 坐标。

#### Beat velocity

![Five works beat velocity curves](../analysis/exp1117_beat_median/result_analysis/five_works_curves/overview_5works_3methods_velocity.png)

#### Beat duration

![Five works beat duration curves](../analysis/exp1117_beat_median/result_analysis/five_works_curves/overview_5works_3methods_duration.png)

单作品图、原始曲线和 mask 审计保存在 [five_works_curves](../analysis/exp1117_beat_median/result_analysis/five_works_curves/README.md)。绘图脚本为 [plot_exp1117_velocity_curves.py](../script/plot_exp1117_velocity_curves.py)。
