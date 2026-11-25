# 基于 Measure 的双级时间表示设计

## 1. 设计动机

将 `BEAT + NOTE` 改为 `MEASURE + NOTE`，每小节只生成一个时长，音符预测相对小节起点的时间。这样可以减少时间锚点和序列长度，由逐音符 onset 表达小节内的速度变化。

在 1122 的同一批对齐演奏上，小节级时长曲线更平滑，适合建模乐句速度轮廓。毫秒起音残差衡量偏离谱面线性拉伸的程度，其中也有需要模型生成的 rubato。优先训练 `MEASURE + NOTE` 验证这个假设，以 1122 A2 作为已有对照，用同一套指标决定采用哪种表示。

## 2. 双级表示与时间锚点

由 refined score MIDI 的拍号生成小节线，沿用 1115 的 `get_bar_beat_ticks`，按实际拍号变化截断前一小节。每小节保留一个 `MEASURE` 事件，即使该小节没有新起音。最后一个小节补齐谱面上的右边界。演奏侧沿用 1122 的最早起音规则。同一谱面 onset group 取最早的对齐演奏起音，小节线落在两组之间时线性插值。首尾无观测的边界不外推，非正时长不计入相关时间目标。弱起和不规则小节以 refined MIDI 的网格为准。

令谱面小节边界为 $S_m$，演奏锚点为 $A_m$，两者各自的相邻锚点时长以毫秒表示为 $D_m^s, D_m^p$。`MEASURE` 的时长目标为

$$
D_m^p = A_{m+1}-A_m,\quad d_m^{\mathrm{dur}}=\ln(D_m^p/D_m^s).
$$

音符按**谱面起音**归属小节。令 $s_i$ 为谱面起音 tick，$t_i$ 为对齐后的演奏起音毫秒值。音符从演奏小节锚点开始计时。记毫秒间隔为 $\Delta_i$，小节内比例为 $u_i^p$，相对谱面比例的偏差为 $\delta_i$。

$$
\Delta_i=t_i-A_m,\qquad u_i^p=\frac{\Delta_i}{D_m^p},\qquad
u_i^s=\frac{s_i-S_m}{S_{m+1}-S_m},\qquad \delta_i=u_i^p-u_i^s.
$$

预测 $\Delta_i$ 时用 $t_i=A_m+\Delta_i$ 还原，预测 $u_i^p$ 时用 $t_i=A_m+u_i^pD_m^p$ 还原。比例表示会把预测的小节时长误差传给小节内音符，靠近小节末尾时影响更大。毫秒表示可独立重建本小节音符相对起点的时间，时长误差仍会通过累加影响后续小节锚点。两种方案都应评估起音误差与小节边界跨越率。

对齐数据中，$\Delta_i$ 可以恰为零。可用 $\ln(\max(1,\Delta_i/1\,\mathrm{ms}))$ 统一处理。这个变换把 0 至 1 ms 的起音映到 0。若需要在解码时精确区分这段范围，可另设零点 token，或使用 $\ln(1+\Delta_i/1\,\mathrm{ms})$。小节时长与音符时长均为正，可以直接取 log 毫秒。评分时将预测值还原到毫秒。

诊断部分还使用带符号的 $\delta_i$，并用 $\delta_i D_m^p$ 查看毫秒残差。若小节内谱面 tempo 改变，以谱面毫秒计算的位置会与 tick 比例不同，下文另做这一口径的复核。

## 3. 样本与覆盖率

使用 1122 冻结的 206 首 ASAP 作品和 966 份演奏。同一份演奏的 beat、measure 都沿用 1122 的全曲 beat-median 时间缩放，避免另做小节归一化影响比较。统计脚本逐份复核 1122 的 beat 有效范围、时长和起音偏差标签。下表的区间数和起音数按演奏累计。

| 划分 | 作品 / 演奏 | 有效拍区间 | 有效小节区间 | 有效拍内起音 | 有效小节内起音 |
|---|---:|---:|---:|---:|---:|
| Train | 175 / 837 | 411,408 / 414,749（99.19%） | 139,038 / 140,765（98.77%） | 2,734,544 / 2,739,556（99.82%） | 2,730,651 / 2,739,556（99.67%） |
| Valid | 12 / 47 | 23,967 / 24,132（99.32%） | 8,335 / 8,428（98.90%） | 135,265 / 135,467（99.85%） | 135,085 / 135,467（99.72%） |
| Test | 19 / 82 | 95,044 / 95,378（99.65%） | 31,749 / 31,908（99.50%） | 384,559 / 384,954（99.90%） | 384,204 / 384,954（99.81%） |

测试集谱面共有 4,624 个小节，每小节新起音音符数的中位数为 12，P5/P95 为 3 / 32。其中 81 个小节没有新起音。直接替换事件后，Train 总事件数从 3,154,305 降到 2,880,321（−8.69%），按 512 个事件连续切窗从 6,561 窗降到 6,042 窗。Test 总事件数从 480,332 降到 416,862（−13.21%）。

### 3.1 小节线的可观测性

测试集中 4,624 个小节起点都在既有拍点网格上。同样的映射规则会在同一个 tick 得到同一个锚点。小节线的区别在于选了哪些拍点。小节边界在有观测支持时有 **91.82%** 直接命中谱面起音组，拍边界为 **86.36%**（按谱面统计）。直接命中跳过插值，仍会受滚奏和对齐误差影响。

把起音组锚点规则从均值改为最早起音，未做全曲缩放的测试集锚点位移中位数为 **−6.0 ms**（小节边界）和 **−5.7 ms**（拍边界），P5 为 **−45.0 ms** 和 **−37.0 ms**。小节线较少依赖插值，但对这条和弦锚点规则并未显得更稳；也没有独立演奏拍点标注可用于判断绝对准确性。

### 3.2 时长与局部速度

| 测试集有效区间 | Beat | Measure |
|---|---:|---:|
| 谱面时长中位数（ms） | 267.9 | 769.2 |
| 演奏时长中位数（ms） | 257.2 | 745.8 |
| 演奏时长 P5 / P95（ms） | 129.3 / 916.7 | 420.5 / 2,367.7 |
| `ln(D_perf / D_score)` P5 / P95 | −0.545 / 0.590 | −0.438 / 0.565 |
| `ln(D_perf / D_score)` 标准差 | 0.354 | 0.292 |
| 相邻有效区间的绝对 log-ratio 变化，中位数 / P95 | 0.104 / 0.682 | 0.092 / 0.508 |

把同一小节内每拍的谱面时长乘以小节整体伸缩比，再和实际转写的拍长比较。在 94,974 个完整落于有效小节内的有效拍中，绝对对数误差的中位数为 **0.066**，P95 为 **0.403**，对应较长值与较短值的比约为 **1.07× / 1.50×**。小节曲线越平滑，按单一小节速率重建时越需要 NOTE onset 承担小节内速度变化。

### 3.3 起音偏差与量化

以下比较使用在两种表示中都有效的同一批 **384,204** 个演奏音符。多演奏作品的最后一行按同一谱面音符跨演奏计算样本标准差，再取所有有效音符的中位数。

| 测试集共同音符 | Beat | Measure |
|---|---:|---:|
| 相对起音偏差 $\delta$ 的标准差 | 0.098 | 0.059 |
| 毫秒起音偏差 P5 / P95 | −27.9 / 62.1 | −78.8 / 120.1 |
| 毫秒起音偏差的平均绝对值 | 19.8 ms | 42.5 ms |
| 毫秒起音偏差标准差 | 39.1 ms | 87.1 ms |
| 12 首多演奏作品、43,259 个谱面音符的跨演奏标准差中位数 | 10.9 ms | 22.3 ms |

小节内 $u_i^p$ 的中位数为 0.404，P5/P95 为 0.000/0.871。测试集没有 $u_i^p<0$，有 1 个音符 $u_i^p>1$。换成谱面时间比例后，小节毫秒偏差的标准差仍为 **84.6 ms**，拍级为 **38.6 ms**。小节分母更大，可以压缩比例偏差，同时把小节内 rubato 和微时值一起留给音符目标。

| Onset dev 量化，共同音符上的重建误差 | bins | 范围 | 测试裁剪比例 | 平均绝对误差 | P95 绝对误差 |
|---|---:|---:|---:|---:|---:|
| 1122 Beat 范围 | 200 | `[-1, 1]` | 0.019% | 1.31 ms | 3.60 ms |
| Measure 沿用 Beat 范围 | 200 | `[-1, 1]` | 0.000% | 3.57 ms | 9.49 ms |
| Measure 按 Train P0.1/P99.9 向外取整 | 200 | `[-0.24, 0.24]` | 0.885% | 1.45 ms | 2.39 ms |

量化表使用真实锚点和时长，只测离散化及裁剪造成的误差。若做首轮训练，可将窄范围作为验证集对照，并保留尾部裁剪计数。时长输出可从 1122 的 500-bin log-dev 设定开始，Train 小节目标超出 `[-3, 2]` 的比例为 0.16%。对绝对 log 时长，Train 有 12.00% 的谱面小节时长和 11.65% 的演奏小节时长超过旧上限 8，`MEASURE` 的输入与 decoder 历史需要重新确定量化范围。

![Beat 与 measure 的时间统计对照](../analysis/design1123_measure/measure_statistics.png)

分布图的横轴覆盖两种表示各自第 1 至第 99 百分位的联合范围。表格和 JSON 使用全部有效值。

### 3.4 绝对小节位置的三种编码

Train、Valid、Test 的有效 $\Delta_i$ 中均未观察到负值。恰为零的比例依次为 **5.33% / 6.43% / 8.59%**。这是按 1122 最早起音规则测得的可用范围。

在同一批 384,204 个测试音符上，比较三种绝对位置编码。各用 200 个 token，其中 1 个表示精确零点，其余 199 个等宽分箱。两种毫秒编码都用 Train 第 99.9 百分位向上取整得到的 6,900 ms 上限，比例上限为 1。所有误差都还原为毫秒，包含裁剪误差。

| `MEASURE` onset 输出 | 测试裁剪比例 | MAE | P95 绝对误差 | 小节末四分之一 MAE |
|---|---:|---:|---:|---:|
| 比例 $\Delta_i/D_m^p$ | 0.00026% | **1.53 ms** | **4.46 ms** | **1.89 ms** |
| 线性毫秒 $\Delta_i$ | 0.038% | 8.26 ms | 16.34 ms | 10.44 ms |
| $\ln(1+\Delta_i/1\,\mathrm{ms})$ | 0.038% | 6.53 ms | 23.38 ms | 16.17 ms |

比例量化的毫秒步长随小节时长变化。`log1p` 将更多分辨率分给靠近零点的间隔，在首四分之一的非零起音上约为 1.56 ms MAE，比例约为 1.68 ms。到了小节末尾，同一个 log bin 对应更宽的毫秒区间。本表用真实锚点和小节时长，衡量的是量化误差。训练仍需比较比例解码下小节时长误差的传播，以及毫秒解码下的跨小节行为。

### 3.5 小节时长、音符时长与小节内起音的毫秒及对数分布

使用 1122 的 19 首测试作品和 82 份对齐演奏。三项分别包含 **31,749** 对有效小节时长、**384,954** 对音符时长和 **384,204** 对有效小节内起音。Score 值按每份演奏的有效对齐位置取样，与 Performance 在每行使用相同样本数。Score onset 从谱面小节线起按谱面 tempo map 计毫秒，Performance onset 从对应演奏小节锚点起计毫秒。

全图六行，每个变量依次使用 ms 和 log ms。左列叠加 Score 与 Performance，右列逐对计算 `Performance − Score`。毫秒差直接在 ms 值之间做减法，对数差在取对数后做减法。两项时长用 `ln(ms)`，onset 用 `ln(max(1, ms))`。因此时长的 log deviation 是 `ln(Perf / Score)`，onset 则在分子和分母各自应用 1 ms 下限。

![三项时间变量的 Score、Performance 和逐对偏差分布](../analysis/design1123_measure/timing_ms_log_distributions.png)

每个面板显示对应序列的 P0.5 至 P99.5 联合范围，并标出视野外样本比例。每个 bin 仍以该序列全部样本为分母。完整分位数和极值见 [统计 JSON 的 `test.absolute_timing_ms_log`](../analysis/design1123_measure/statistics.json)。

### 3.6 小节内起音百分比与百分点偏差

沿用上节的 384,204 对音符。Score 和 Performance 各自用小节内起音间隔除以本侧的小节时长，再乘以 100。0 对应小节起点，100 对应下一条小节线。谱面侧沿用上节的 tempo map 时间口径，小节内速度恒定时与 tick 比例相同。

$$
p_i^s = 100\frac{\Delta_i^s}{D_m^s},\qquad
p_i^p = 100\frac{\Delta_i^p}{D_m^p},\qquad
\mathrm{dev}_i = p_i^p - p_i^s.
$$

`dev` 为正表示演奏相对其小节进度更靠后，为负表示更靠前，单位是百分点。若预测 `dev`，将它加回谱面百分比，再按演奏小节时长还原毫秒。相对位置有明确零点，直接相减避免了在小节起点附近用很小的 elapsed time 作分母。

![小节内起音百分比和逐音符百分点偏差](../analysis/design1123_measure/measure_onset_percent.png)

测试集偏差的中位数为 **+0.60** 个百分点，P5/P95 为 **−6.49 / +11.15**。左图显示 0–100，唯一超出范围的演奏起音位于 103.95%，仍参与右图和 [完整统计](../analysis/design1123_measure/statistics.json)。右图使用偏差的 P0.5 至 P99.5 显示范围。

## 4. 小节级速度与力度曲线

沿用 1122 选出的相同五首测试作品。黑线为 Score，浅线为每份演奏，彩色实线为逐小节的人类中位数。

**Tempo** 用四分音符等效 BPM。若小节长 $q_m$ 个四分音符，则速度为 $60000q_m/D_m$。这让不同拍号或不等长小节仍可比较。演奏锚点缺失时曲线留空。三拍子作品若习惯听作一小节一拍，图中的四分音符 BPM 会是该感知速度的三倍。

![五首作品的小节级速度曲线](../analysis/design1123_measure/measure_tempo_curves.png)

**Velocity** 对同一谱面小节内所有新起音的音符取 MIDI velocity 算术均值。延音归其起音小节，空小节跳过，无需小节时长有效。

![五首作品的小节级力度曲线](../analysis/design1123_measure/measure_velocity_curves.png)

## 5. Token、解码与首轮对照

```text
MEASURE(score_measure_duration)
NOTE(pitch, score_note_duration, score_measure_onset, score_velocity)
NOTE(...)
MEASURE(score_measure_duration)
...
```

首轮对照将小节时长与音符时长固定为相对谱面的 log-dev，沿用 1122 A2 的时长目标思路。`NOTE` 可先预测 §3.6 的线性 onset dev，用直接预测小节内百分比作对照。两者共享还原公式，可比较把谱面位置显式移到目标基准的效果。绝对毫秒间隔可作另一对照，检验对预测小节时长的依赖。力度仍用 MIDI velocity。训练的时长与 onset 标签分别使用对应的有效 mask。生成时累加小节时长得到锚点，再还原音符起音。对边界超出谱面首末起音范围的首尾小节，保留事件并用谱面时长回退。这个判断在推理时只需要谱面。

首轮对照可沿用 1122 A2 的划分、归一化、早停和 512-event 窗口，固定更新预算并记录每步有效音符数。两种模型都输出 MIDI，在相同的 1122 拍网格和共同有效范围上比较 Tempo Pearson、Velocity Pearson 及三项 MAE；补充报告起音毫秒误差、小节级曲线和推理步数。小节比例 MAE 与拍内比例 MAE 使用不同分母，应在同一个重建坐标下比较。

可用 `python INSPIRE2/script/analyze_exp1123_measure.py` 从仓库根目录重算。 [逐划分和逐作品统计 JSON](../analysis/design1123_measure/statistics.json) · [统计与绘图脚本](../script/analyze_exp1123_measure.py) · [1115 拍级设计](design1115_beat.md) · [1122 锚点规则与参考](exp1122.md)

## 6. 官方 ASAP 划分与小节级评估

### 6.1 数据划分

按官方脚本对 `midi_score` 路径排序，再以 seed 42 打乱，前 10% 划为 Test。表中谱面数按 MIDI 路径计。HMM 对齐后保留有可用片段的演奏。

| 划分 | 对齐前谱面路径 | 对齐前演奏 | 对齐后谱面路径 | 对齐后演奏 |
|---|---:|---:|---:|---:|
| Train | 212 | 966 | 206 | 918 |
| Test | 23 | 101 | 23 | 101 |

训练期间的 Valid loss 在同一批 23 首 Test 谱面上计算。PT2 从官方 PT base 初始化，按官方 SFT 配方固定训练两轮并使用最终权重。路径划分中有两个测试乐章的不同反复版本落在训练侧。 [划分清单](../../results/pt_rendering_public_recipe_20261124/score_split.json) · [数据审计](../../results/pt_rendering_public_recipe_20261124/dataset_audit.json) · [PT2 训练记录](../../results/pt_rendering_public_recipe_20261124/official_ddp_3gpu/completion.json)

### 6.2 小节级五项指标

PT base、官方 PT rendering、PT2 和 Score MIDI 均使用完整的 23 首测试谱面和 101 份人类参考。三个 PT 模型每首各生成三次，Pearson 的 `±` 为三次采样的样本标准差，MAE 比较三次生成的逐音符均值和人类参考均值。人类参考按 exp1122 做全曲 beat-median 归一化。Human 使用其中至少有两份演奏的 19 首、97 份演奏，每次留出一份演奏与该曲剩余演奏比较，先在曲目内平均。所有指标按曲目等权汇总。

Tempo Pearson 使用小节四分音符等效 BPM，Velocity Pearson 使用小节内新起音音符的平均力度。Onset MAE 以小节内位置的百分点计，Duration MAE 以音符时长的自然对数比值计，Velocity MAE 以 MIDI 力度单位计。Human 的两项 Pearson 均覆盖 19 首，三个 PT 均覆盖 23 首。Score MIDI 的恒定曲线对应的 Pearson 无定义，Tempo 和 Velocity Pearson 分别由 6 首和 20 首汇总，三项 MAE 仍覆盖 23 首。

| 方法 | 测试谱面 | 小节 Tempo Pearson ↑ | 小节 Velocity Pearson ↑ | 小节内 Onset MAE ↓（百分点） | 音符 Duration MAE ↓（log ratio） | 音符 Velocity MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| PT base | 23 | 0.004 ± 0.012 | 0.348 ± 0.014 | 2.220 | 0.603 | 19.750 |
| PT rendering | 23 | 0.331 ± 0.019 | 0.497 ± 0.015 | 1.817 | 0.525 | 9.508 |
| PT2 | 23 | 0.346 ± 0.039 | 0.479 ± 0.042 | 1.771 | 0.545 | 9.060 |
| Score MIDI | 23 | 0.313 | 0.489 | 1.980 | 0.580 | 22.419 |
| Human（留一） | 19 | 0.612 | 0.750 | 1.760 | 0.375 | 8.738 |

[完整测试指标 CSV](../../results/pt_rendering_public_recipe_20261124/official_split_measure/five_metrics_measure.csv) · [逐谱面指标 JSON](../../results/pt_rendering_public_recipe_20261124/official_split_measure/five_metrics_measure.json) · [PT2 推理清单](../../results/pt_rendering_public_recipe_20261124/official_split_measure/rendering_reproduced/prediction_manifest.json)

### 6.3 Measure A2 与 PT2 的窗口对比

Measure A2 与 PT2 使用相同的官方划分和 HMM 保留音符。Train 有 206 份谱面、918 份演奏、2,838,143 个保留音符。Test / Valid 有 23 份谱面、101 份演奏、314,477 个保留音符。每窗平均音符数为原窗和裁剪窗内的音符出现次数之和除以总窗口数，计数单位为 `NOTE` 事件。

| 划分 | 模型 | 原窗 | 裁剪窗 | 总窗口 | 窗口内音符出现次数 | 每窗平均音符数 |
|---|---|---:|---:|---:|---:|---:|
| Train | PT2 | 10,490 | 10,490 | 20,980 | 7,469,754 | 356.04 |
| Train | Measure A2 | 11,534 | 11,534 | 23,068 | 9,384,326 | 406.81 |
| Test / Valid | PT2 | 1,164 | 1,164 | 2,328 | 858,428 | 368.74 |
| Test / Valid | Measure A2 | 1,284 | 1,284 | 2,568 | 1,036,399 | 403.58 |

[窗口对比 CSV](../../data/ASAP_official_dinr_measure_structural_e512_snap50_crop_v1/comparison.csv) · [Measure A2 数据审计](../../data/ASAP_official_dinr_measure_structural_e512_snap50_crop_v1/audit.json)

### 6.4 Measure A2 的直接小节速度评估

Measure A2 实验固定使用 §6.1 的官方 Train/Test 划分。评估直接用输出 `MEASURE` 的时长 $D_m$ 计算 $60000q_m/D_m$。PT 和 Human 的小节边界仍由各自 MIDI 的起音映射。

第 700 步 checkpoint 在 23 首 Test 上直接计算得到 Tempo Pearson **0.079**。旧评估把生成音符的起音重新映射到小节，得到 **0.080**。按谱面速度分组，17 首谱面速度恒定的曲目平均为 **0.015**，6 首谱面自带速度变化的曲目平均为 **0.263**。每首曲目先对各次生成和各份参考取平均，再对曲目等权平均。 [直接评估的五项指标](../results/exp1123_a2_degradation_audit_20261124/native_measure_tempo/five_metrics_measure.csv) · [逐曲目诊断](../results/exp1123_a2_degradation_audit_20261124/tempo_diagnostic.json)

保持整曲自回归，只将 `MEASURE` 的 log-duration deviation 从采样改成预测分布的均值，Tempo Pearson 为 **0.112**。在谱面速度恒定的 17 首上为 **0.029**。用真人演奏历史诊断时，`MEASURE` 时长预测分布的均值给出 **0.266**。在同样的诊断里，只保留真人 `NOTE` 历史时为 **0.260**，只保留真人 `MEASURE` 历史时为 **0.141**，其余历史字段均用谱面填充。 [整曲生成对照](../results/exp1123_a2_degradation_audit_20261124/tempo_sampling_comparison.json) · [历史替换诊断](../results/exp1123_a2_degradation_audit_20261124/tempo_diagnostic.json)

下图选自官方 Test 的六首谱面。§4 原来的五首在当前划分中都属于 Train。Human、PT2 和 A2 曲线分别取各份演奏或三次生成的小节 BPM 中位数。A2 使用输出小节时长，PT2 使用生成起音映射得到的小节时长。图中的相关系数按单次生成与单份真人演奏计算，然后取平均。缺少共同对齐支持的小节留空。 [绘图数据 CSV](../results/exp1123_a2_degradation_audit_20261124/tempo_curve_comparison/measure_tempo_curves.csv)

![官方 Test 六首的 Score MIDI、Human、PT2 和 A2 小节速度曲线](../results/exp1123_a2_degradation_audit_20261124/tempo_curve_comparison/measure_tempo_curves.png)
