# Design1125 · Chord 极简基线

## 1. 实验问题

固定 [design1123 §6.1](design1123_measure.md#61-数据划分) 的官方 ASAP Train/Test。可用 Train 有 206 份谱面、918 份演奏，Test 有 23 份谱面、101 份演奏。本轮用一个 `CHORD` 预测一组共享的 IOI、duration 和 velocity。三张 GPU 各训练一个模型，只比较 timing 表示，所有模型使用相同的 chord、窗口、模型结构和训练预算。

## 2. 从音符到 CHORD

在官方对齐使用的规范化谱面上，按 **`(谱面起音 tick, 谱面时长 ms)`** 分组。这里的时长是生成 timing token 前的谱面时长，沿用 design1123 的官方 score 表示。用谱面音符 ID 连接 HMM 对齐结果。`CHORD` 序列先按谱面起音排序。同起音的 `CHORD` 按各自最低 pitch 从低到高排，最低 pitch 相同则按时长、原始音符 ID 排。组内 pitch 也从低到高排。这个顺序用于 Transformer 的 timestep embedding。

例如，同一 tick 上 C4、E4 都是 400 ms，G4 是 200 ms，则依次得到 `CHORD([C4, E4], 400 ms)` 和 `CHORD([G4], 200 ms)`。Score 和演奏目标都把这两个 `CHORD` 放在同一个起音时刻。最终 C4 与 E4 的起音、结束和力度完全相同，G4 与它们同时开始并使用自己的 duration 和 velocity。

| 字段 | 谱面输入 | 演奏训练目标 |
|---|---|---|
| Pitch | 成员 MIDI pitch 列表 | 输出 MIDI 复制这些 pitch |
| IOI | 同起音位置的第一个 `CHORD` 持有到上一谱面起音位置的间隔 | 取该起音位置对应的第一个演奏起音，即所有该位置保留音符中的最早起音，再减去上一位置的同类锚点 |
| Duration | 组内统一的谱面音符时长 | 在原始毫秒域取保留演奏音符的时长中位数，作为全组共享时长 |
| Velocity | 组成员的谱面力度算术均值，取整 | 组成员的演奏力度算术均值，取整 |

先在毫秒域确定每个 `CHORD` 的唯一演奏时长，再按 §4 生成各实验的时长标签。偶数个成员取中间两个时长的算术均值。例如 100 ms 和 400 ms 对应同一个 250 ms 目标，再分别做 `abs`、`log abs` 或 `log dev` 量化。预测后的共享时长广播给组内全部音符。

**每个谱面起音位置只推进一次时钟。** 排序后的首个 `CHORD` 承载组间 IOI，其余 `CHORD` 的 `score IOI=0`。曲首从时间 0 开始，首个位置也用 `score IOI=0`。在量化前记录这个零掩码，三种表示遇到这些位置都跳过 IOI 预测及损失，直接设 `perf IOI=0`。低音到高音只定义序列顺序，演奏目标把这一位置的所有成员对齐到同一个演奏锚点。原始人类起音用于最后评估。

训练中按 HMM 保留的成员确定该位置的首个 `CHORD`，用同一 HMM 片段内相邻谱面起音位置的正间隔作 IOI 目标。遇到缺失锚点、跨 HMM 片段或观测间隔非正时，该位置只训练有效的 duration 和 velocity。

## 3. Pitch 编码与预测头

编码器接收每个 `CHORD` 的 pitch 列表，将它转换为 **128 维 MIDI pitch 向量**，每一维记录对应 pitch 在组内出现的次数。双层 MLP `Linear(128, h) → GELU → Linear(h, d_model)` 把该向量投影到 Transformer 隐藏维度，再与 score IOI、duration、velocity 的 embedding 相加。三卡使用同一套 pitch 编码。

双向 Transformer 编码器直接接三个并行输出头。Duration 和 velocity head 用于所有有效 `CHORD`，IOI head 只读取量化前 `score IOI>0` 的位置。每个 head 对自己的有效标签求平均 CE，再把三个 head 的损失等权相加。首轮生成使用各 head 的后验中位类别，导出时从谱面复制 pitch 列表，并将预测的起音、时长和力度广播给组内音符。

## 4. 三种统一的 timing 表示

两种 timing 属性在每个模型内共用下表所列的变换和量化范围，各自使用独立的 embedding 和分类头。三组沿用 [exp1112](exp1112_DINR_VAR.md#exp1112-dinr-abs5000-与坐标表示对照) 的 **5,000 个数值类别**。下表的 $s$ 是量化前的谱面毫秒值，$d$ 是按 §2 构造的共同演奏目标毫秒值。定义 $L(x)=\ln(\max(x,1))$。

| GPU | 表示 | Score IOI / duration 输入 | 演奏 IOI / duration 目标 | 解码 |
|---:|---|---|---|---|
| 0 | `abs 5000 token` | 毫秒整数类别 | $d$ 取最近整数，范围 0–4,999 ms | 类别代表的毫秒值 |
| 1 | `log abs` | $L(s)$，在 $[0,\ln(5000)]$ 内均匀量化 | $L(d)$，在 $[0,\ln(5000)]$ 内均匀量化 | $\exp(\hat x)$ |
| 2 | `log dev` | 同 `log abs` | $L(d)-L(s)$，在 $[-3,2]$ 内均匀量化 | $\exp(L(s)+\hat x)$ |

连续范围含两个端点，使用最近类别，各端截断比例按 IOI、duration 分开记录。解码时，推进时钟的 IOI 和音符 duration 至少为 1 ms。结构性零由谱面指定，直接回放为 0 ms。MIDI 导出时对每个 `CHORD` 只舍入一次起音和时长，再广播给成员。

## 5. 窗口、训练与端到端评估

- 三张 GPU 各跑一项独立单卡训练。每项使用相同随机初始化种子 42、固定 4 epoch、LR `3e-4` 线性衰减。以每次 16 个窗口、累积 3 次作为首轮设置，有效 batch 为 48 个窗口。相同的数据顺序、裁剪种子和更新步数。
- 每窗最多 **512 个 `CHORD`**，动态 padding。每份演奏仍按谱面顺序拼接 HMM 保留片段。训练窗尽量在小节边界切，下一窗从离半窗最近的小节边界开始，额外裁剪从两起点之间的小节边界起步并共用原窗终点。短窗口跳过没有合法起点的额外裁剪。单个小节超出上限时，在完整的同起音位置之间切。组间 IOI 在全序列上先计算，切窗后沿用这些标签。
- 每 50 个 optimizer step 和最后一个 step 用 Test 的非重叠原窗记录监督 CE。验证与推理均只输入谱面并行预测。每项训练固定跑满 4 轮，最终推理使用各自 **valid loss 最低**的 checkpoint。
- 训练完成后，每个 checkpoint 在 **完整的 23 首 Test 谱面**上用非重叠窗口预测，拼接结果后用全曲时钟展开全部 pitch。核对所有 score 音符 ID 覆盖一次、同起音的多个 chord 同时落下、每个 chord 内起音和结束及力度一致。
- 五项指标为 **Tempo Pearson、Velocity Pearson、IOI MAE（ms）、Duration MAE（ms）、Velocity MAE**。速度与力度相关沿用 [design1123 §6.2](design1123_measure.md#62-小节级五项指标) 的小节网格。IOI 比较相邻不同谱面起音位置的组间隔，真人锚点取最早保留的对齐起音，并要求相邻锚点在同一有效 HMM 片段。Duration 和 velocity 按展开后的谱面音符对齐。两项时间 MAE 都用生成和 HMM 参考的原始毫秒值。每个位置先分别对模型生成和真人参考求平均，然后取绝对差。曲目等权，Human 基线用多演奏曲目留一。三个 chord 版本与 PT2、Measure A2、Score MIDI 使用同一评分口径。
- 数据审计记录按同起音拆出的 chord 数、有效 IOI 标签数、同起音零 IOI 数、截断率、窗数与每窗平均 `CHORD` / 原始音符数。训练日志、checkpoint、loss 曲线、预测、指标和代表曲目叠图放在 `INSPIRE2/results/exp1125_chord/` 下，各表示用独立子目录。

## 6. Chord 与 PT2 窗口对比

两种表示覆盖相同的官方谱面、演奏和 HMM 保留音符。切窗前，Train 的 **2,838,143** 个音符合成 **1,834,717** 个 chord，Test 的 **314,477** 个音符合成 **209,655** 个 chord。同起音而时长不同的音符按 §2 拆分，Train 和 Test 分别有 271,450 和 32,926 个起音位置拆成了多个 chord。

沿用 [design1123 的对照口径](design1123_measure.md#63-measure-a2-与-pt2-的窗口对比)，Train 和 Test 都统计原窗与裁剪窗。PT2 每窗最多 512 个音符，Chord 每窗最多 512 个 `CHORD`。同一个音符或 chord 出现在多个窗口时，按出现次数累加。Chord 的音符数按窗口内各组包含的原始音符数计算。

| 划分 | 模型 | 原窗 | 裁剪窗 | 总窗口 | 窗内 chord 出现次数 | 每窗平均 chord | 窗内音符出现次数 | 每窗平均音符 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Train | PT2 | 10,490 | 10,490 | 20,980 | — | — | 7,469,754 | 356.04 |
| Train | Chord | 6,825 | 6,825 | 13,650 | 5,822,969 | 426.59 | 8,973,408 | 657.39 |
| Test | PT2 | 1,164 | 1,164 | 2,328 | — | — | 858,428 | 368.74 |
| Test | Chord | 769 | 769 | 1,538 | 664,744 | 432.21 | 976,789 | 635.10 |

| 划分 | Chord 窗口类型 | chord 出现次数 | 每窗平均 chord | 每窗 chord 中位数 | 每窗 chord 最小–最大 |
|---|---|---:|---:|---:|---:|
| Train | 原窗 | 3,324,835 | 487.16 | 504 | 251–512 |
| Train | 裁剪窗 | 2,498,134 | 366.03 | 366 | 146–509 |
| Test | 原窗 | 378,751 | 492.52 | 506 | 256–512 |
| Test | 裁剪窗 | 285,993 | 371.90 | 371 | 151–511 |

Chord 的 Train 窗口数比 PT2 少 **34.94%**，Test 少 **33.93%**。Train 在窗口中累计处理的时序位置从 PT2 的 746.98 万个音符位置降到 582.30 万个 chord 位置，少 **22.05%**。展开后，Chord 的 Train 每窗平均覆盖 657.39 个音符，PT2 为 356.04 个。实际监督验证按 §5 统计 Test 非重叠原窗，共 **449 窗**。

[逐窗口 chord 和音符个数](../results/exp1125_chord/window_audit/chord_windows.csv) · [对照汇总 CSV](../results/exp1125_chord/window_audit/comparison.csv) · [分位数和数据审计 JSON](../results/exp1125_chord/window_audit/comparison.json) · [复算脚本](../script/audit_exp1125_chord_windows.py)

## 7. 端到端测试结果

三种表示都完成了 4 轮、1,140 次 optimizer update，各自在 449 个验证窗上选出最低 valid loss 的权重。`abs5000`、`log_abs`、`log_dev` 分别选出第 **250 / 350 / 550** 步，valid loss 为 **14.8279 / 15.0333 / 16.9602**。验证窗来自同一官方 Test。每项使用选中权重在 23 首完整谱面上生成一份 MIDI，共 69 份。时长和 IOI 在原始毫秒域取绝对误差。

| 方法 | 小节 Tempo r ↑ | 小节 Vel r ↑ | Beat Tempo r ↑ | Beat Vel r ↑ | 起音组 Tempo r ↑ | Chord Vel r ↑ | IOI MAE ↓（ms） | Duration MAE ↓（ms） | Vel MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chord abs 5000 | −0.002 | 0.490 | −0.029 | 0.465 | 0.022 | 0.472 | 70.17 | 152.26 | 10.04 |
| Chord log abs | 0.077 | 0.530 | 0.036 | 0.491 | 0.017 | 0.497 | 71.62 | 141.56 | 9.94 |
| Chord log dev（并行中位数） | 0.279 | 0.539 | 0.197 | 0.504 | 0.104 | 0.487 | 60.12 | 118.50 | 9.92 |
| PT base | 0.004 | 0.348 | 0.011 | 0.307 | 0.013 | 0.247 | 62.91 | 179.12 | 19.75 |
| PT2 | 0.346 | 0.479 | 0.266 | 0.463 | 0.130 | 0.473 | 57.02 | 118.15 | 9.06 |
| PT-scratch | −0.065 | 0.062 | −0.043 | 0.056 | −0.019 | 0.079 | 101.77 | 189.63 | 21.47 |
| Measure A2 | 0.079 | −0.002 | 0.039 | −0.006 | 0.013 | 0.002 | 133.26 | 168.11 | 19.22 |
| Score MIDI | 0.313 | 0.489 | 0.222 | 0.438 | 0.067 | 0.353 | 58.42 | 130.74 | 22.42 |
| Human 留一 | 0.612 | 0.750 | 0.496 | 0.709 | 0.192 | 0.645 | 31.03 | 76.64 | 8.74 |

三个粒度的 Score MIDI Tempo r 和 Vel r 分别在曲线有方差的 6 首和 20 首上定义。Human 留一覆盖 19 首有多份演奏的曲目。其他模型六项相关均覆盖 23 首。在谱面速度恒定的 17 首上，Chord log dev 的小节 Tempo r 为 **0.223**，PT2 为 **0.305**，Measure A2 为 **0.015**。

PT base 使用现有的 23 首、每首三次官方测试推理。将它加入统一评分后，共同有效的时长位置由 57,359 个变为 57,315 个，表中的 Duration MAE 均按这个范围复算。 [PT base 预测清单](../../results/pt_rendering_public_recipe_20261124/official_split_measure/base/prediction_manifest.json)

[完整指标 CSV](../results/exp1125_pt_scratch/multilevel_metrics.csv) · [逐曲目指标与权重清单](../results/exp1125_pt_scratch/multilevel_metrics.json) · [三个 Chord 实验的训练与验证损失](../results/exp1125_chord/train_valid_loss.png) · [代表曲目速度曲线](../results/exp1125_chord/measure_tempo_curves.png) · [端到端运行记录](../results/exp1125_chord/runs.jsonl)

### 7.1 不同粒度的相关性口径

三个粒度共用 23 份谱面、101 份人类参考和已生成的模型预测。Pearson 先按生成样本和人类参考配对计算，再在曲目内平均，最后对曲目等权平均。

- **小节 Tempo / Vel** 按谱面小节计算四分音符等效 BPM 和小节内新起音音符的平均力度。Measure A2 的小节 BPM 直接使用其小节时长输出。
- **Beat Tempo / Vel** 将谱面拍边界映射到对齐后的最早起音曲线，按拍长计算四分音符等效 BPM。Measure A2 在此粒度用生成音符的起音映射拍边界。力度取该拍内新起音音符的平均值。共同有效范围内有 9,364 个拍长和 9,245 个拍级力度位置。
- **起音组 Tempo** 将相邻不同谱面起音的距离换算为四分音符数，然后计算 `60000 × 四分音符数 / 演奏 IOI(ms)`。同起音、异时长的 chord 共用该起音锚点。留出各方法中 IOI 非正的位置后，共 23,490 个区间参与对照。
- **Chord Velocity** 在同起音且同谱面时长的 chord 内，只对所有方法共同对齐的音符取 MIDI 力度均值。曲线每个 chord 占一个位置，共 38,854 个 chord。

### 7.2 Chord log dev 的并行与 AR 输出对照

四项均为 Chord log dev Cat5000，在同一批 23 首完整谱面上评分。并行版和 AR 版分别加载各自的权重，各用中位数生成一次或随机采样三次。两种采样均使用 `temperature=1.0` 和 `top_p=0.95`。

| Chord log dev 模型 | 解码 | 小节 Tempo r ↑ | 小节 Vel r ↑ | Beat Tempo r ↑ | Beat Vel r ↑ | IOI MAE ↓（ms） | Duration MAE ↓（ms） | Vel MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 并行 Cat5000 | 中位数 | 0.279 | 0.539 | 0.197 | 0.504 | 60.12 | 118.50 | 9.92 |
| 并行 Cat5000 | 采样 ×3 | 0.213 | 0.497 | 0.111 | 0.427 | 68.74 | 129.70 | 10.70 |
| AR Cat5000 | 中位数 | 0.225 | 0.500 | 0.146 | 0.462 | 56.02 | 118.31 | 10.45 |
| AR Cat5000 | 采样 ×3 | 0.168 | 0.412 | 0.083 | 0.361 | 64.09 | 131.59 | 10.71 |

[四种配置的完整九项指标](../results/exp1125_chord_ar/decoding_audit/multilevel_metrics.csv) · [逐曲目指标与权重清单](../results/exp1125_chord_ar/decoding_audit/multilevel_metrics.json)

### 7.3 PT-scratch 对照

PT-scratch 复用 PT2 已执行的官方 rendering 训练入口、对齐数据、窗口、BFloat16、三卡 DDP、有效 batch 33、两轮、cosine LR 和末轮 checkpoint。模型只读 PT base 的架构配置，所有参数在 seed 42 下随机初始化。验证每 50 个 optimizer step 执行一次，使用原 PT2 的 2,328 个 Test 窗口。训练后按 PT2 的三个推理种子渲染相同的 23 首谱面，并在共同支持上复算 §7 各项指标。

PT-scratch 共训练 1,272 步。第 1,250 步的定期 valid loss 为 **2.1237**。用第 1,272 步的权重完成 23 首、每首 3 次的测试推理。小节 Tempo r 为 **−0.065**，小节 Vel r 为 **0.062**。共用相同评分范围时，PT2 的两项相关分别为 **0.346** 和 **0.479**。

[PT-scratch 配置](../results/exp1125_pt_scratch/official_sft_config.json) · [训练完成记录](../results/exp1125_pt_scratch/completion.json) · [69 份 MIDI 清单](../results/exp1125_pt_scratch/test_official/prediction_manifest.json) · [日志与运行记录](../results/exp1125_pt_scratch/runs.jsonl) · [端到端入口](../script/run_exp1125_pt_scratch.py)

## 8. Chord log dev 的连续输入与输出分布实验设计

§7.2 同时列出原始并行 `Chord log dev` 和后续消融使用的 AR Cat5000 基线。各项沿用按谱面起音和时长分组的 chord、10 层双向 encoder、`128 → 256 → 768` pitch MLP 和 HMM 有效位置。IOI 的零掩码继续由**原始谱面 IOI**确定。谱面 IOI 为零的 chord 不计算 IOI loss，导出时直接用 0 ms。目标仍为演奏 timing 对原始 score timing 的 log dev，力度仍为绝对 MIDI velocity。

### 8.1 输入编码

记 score timing 为毫秒值 $s_I,s_D$，力度为 $v$。两个归一化设置共用线性映射

$$
x=\bigl(\operatorname{clip}(s_I,0,5000)/5000,\;
        \operatorname{clip}(s_D,0,5000)/5000,\;
        \operatorname{clip}(v,0,127)/127\bigr).
$$

| 输入 | IOI、duration、velocity 的编码 | 与 pitch 的融合 |
|---|---|---|
| 基线 lookup | timing 查各自的 5,000 类 log-absolute 表，velocity 查 128 类表 | 三个 embedding 与 pitch 向量相加 |
| 原值独立 MLP | 原始毫秒和 velocity 各进一个 `Linear(1,64) → GELU → Linear(64,768)`。原值直接送入 MLP | 三个输出与 pitch 向量相加 |
| 归一化独立 MLP | $x$ 的三个标量各进一个 `1 → 64 → 768` MLP | 三个输出与 pitch 向量相加 |
| 归一化联合 MLP | $x$ 整体进入 `Linear(3,192) → GELU → Linear(192,768)` | 一个联合输出与 pitch 向量相加 |

归一化独立 MLP 是控制项。它与原值独立 MLP 的差异是输入尺度，与归一化联合 MLP 的差异是字段何时融合。两个归一化模型的数值编码参数量接近，分别是 **150,144** 和 **148,992**，当前三张输入表共 **7,778,304** 个参数。输入归一化只用于编码，反量化继续用 sidecar 中的原始谱面毫秒。训练集 1,834,717 个 chord 中，score IOI 和 duration 超过 5,000 ms 的分别有 59 和 856 个。原值分支记录三个 MLP 输出和 pitch 向量各自的 RMS，以判断毫秒值是否在相加时压过力度和音高。

### 8.2 Timing 目标、500 类输出和 DLM

沿用 $L(t)=\ln(\max(t,1))$。有效 IOI 和 duration 各自的目标偏差为 $\delta=L(d)-L(s)$。500 类模型与 DLM 共用一组桶边界 $b_k=-3+0.01k$，$k=0,\ldots,500$。内部区间左闭右开，恰好等于上端点或超出范围的目标进入首尾相应桶。分类头用桶中心 $b_k+0.005$ 还原偏差，再计算 $\max(s,1)\exp(\hat\delta)$。单次桶内量化引入的相对时间误差最多约 0.5%，另计超出支持范围的截断误差。

| 输出 | IOI 和 duration | Velocity | 训练损失 |
|---|---|---|---|
| 基线 Cat5000 | 独立 5,000 类 log dev 头，保留 §4 的原类别网格 | 128 类 | 三项 CE |
| Cat500 | 独立 `Linear(768,500)` | 128 类，沿用基线 | 三项 CE |
| Timing DLM | 两个单分量 DLM 头，各输出位置和尺度 | 128 类，沿用基线 | 两项 DLM NLL 加力度 CE |
| Timing + Vel DLM | 两个单分量 DLM 头，各输出位置和尺度 | 一个单分量 DLM，边界 `−0.5, 0.5, …, 127.5` | 三项 DLM NLL |

DLM 头为 `Linear(768,2)`。Timing 的位置参数用 sigmoid 映射到 $[-3,2]$，velocity 的位置参数映射到 $[0,127]$。尺度用 `softplus` 加半个桶宽的下限，timing 为 0.005，velocity 为 0.5。对真实值所在的桶计算概率质量

$$
p_k=F(b_{k+1})-F(b_k),\qquad
F(x)=\operatorname{sigmoid}((x-\mu)/\sigma).
$$

首尾桶分别收纳下尾和上尾的概率，和越界目标的归桶规则一致。用 FP32 计算真实桶的稳定 log 概率，loss 为 $-\log p_k$。连续的是分布的位置和尺度。训练仍以宽度 0.01 的 timing 桶和整数 velocity 桶计算似然，每个位置只需计算真实桶的两侧 CDF。所有设置继续按各字段有效位置求平均 loss，再将三项等权相加。

主表统一取分布的中位类别，Cat500 与 DLM 使用同一套 timing 桶中心。DLM 另用同一 checkpoint 直接取 $\mu$ 还原连续 timing，单列结果，用来量化导出时省去桶中心的影响。Velocity 导出为 `1–127` 的 MIDI 整数。 [论文中的 DLM 定义](../../INSPIRE-AAAI2027/technical_supplement.tex#L240)

### 8.3 三卡调度与评估

| 批次 | GPU 0 | GPU 1 | GPU 2 |
|---|---|---|---|
| 1 · 输入消融，输出固定 Cat5000 | 原值独立 MLP | 归一化独立 MLP | 归一化联合 MLP |
| 2 · 输出消融，输入固定基线 lookup | Cat500 | Timing DLM + Velocity CE | Timing + Vel DLM |
| 3 · 检查输入与输出组合 | 联合 MLP + Cat500 | 联合 MLP + Timing DLM | 联合 MLP + Timing + Vel DLM |

第一批用现有 **Chord AR log dev lookup** 做对照。三项共用 lookup AR 的 encoder、GRU、演奏反馈、5,000 类 timing 输出及有效位置掩码，仅替换 score 数值编码。开窗时重置 AR 历史。每项分别生成 **AR 中位数每首一份**和 **AR 随机采样每首三份**。采样沿用 `temperature=1.0` 和 `top_p=0.95`。每卡 8 个常驻进程按 chord 数从长到短处理这 92 个任务。两套解码各与 lookup 的相同解码设置对比。 [AR 输入实验目录](../results/exp1125_chord_ar_inputs/) · [AR 中位数 lookup 对照](../results/exp1125_chord_ar/decoding_audit/log_dev/ar.json) · [AR 采样 lookup 对照](../results/exp1125_chord_ar/log_dev/test_official/prediction_manifest.json)

第二批继续用 **Chord AR log dev lookup** 对照。Score 的 IOI、duration 输入表各保留 5,000 类，velocity 保留 128 类。三项都改用 500 个演奏 timing 桶，AR feedback 也统一读取这 500 个 timing 类别。两种 DLM 在中位数生成时复用同一条 AR 轨迹，分别用桶中心和连续位置参数导出 timing，历史反馈使用相同的中位类别。 [AR 输出实验目录](../results/exp1125_chord_ar_outputs/)

前两批回答各项改动的效果，第三批检验预先指定的归一化联合 MLP 与三种输出的组合。基线复用现有同口径权重。训练复用 13,650 个 Train 窗和 449 个验证原窗，单卡 BF16、seed 42、有效 batch 48、AdamW、`3e-4` 线性衰减、4 轮共 1,140 步。每 50 步及末步验证，每项取自己最低 valid loss 的 checkpoint。第二批三个 500 桶模型的验证 loss 都是相同目标桶的负对数概率，5,000 桶基线用原分辨率选权重。主表采用同一份端到端指标。

每项在 23 首完整官方 Test 谱面上生成，使用相同的 101 份真人参考和有效位置。报告小节与 beat Tempo/Vel Pearson、起音组 Tempo 和 chord Vel Pearson，以及原始毫秒的 IOI/Duration MAE 和 Vel MAE。保留逐曲目指标、权重步数、总参数量和输出分布的边界命中率。如果增加随机评估，基线和变体都各采样 3 次，单列采样结果。配置、日志、权重、预测和复评均存入 `INSPIRE2/results/`。

### 8.4 第一批 AR 输入编码结果

三组均完成 4 轮、1,140 步。原值独立、归一化独立、归一化联合 MLP 的最低验证 loss 权重分别在 **1,100、550、500 步**，对应验证 CE 为 **17.7762、18.0554、18.0983**。三卡各用 8 个进程，逐首生成一份 AR 中位数和三份 AR 采样。每组 92 份，共核对 276 份完整生成。

| Score 输入 | AR 解码 | 小节 Tempo r ↑ | 小节 Vel r ↑ | Beat Tempo r ↑ | Beat Vel r ↑ | IOI MAE ↓（ms） | Duration MAE ↓（ms） | Vel MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Lookup | 中位数 | 0.225 | 0.500 | 0.146 | 0.462 | 56.02 | 118.31 | 10.45 |
| 原值独立 MLP | 中位数 | 0.169 | 0.294 | 0.159 | 0.277 | 61.97 | 139.00 | 11.46 |
| 归一化独立 MLP | 中位数 | 0.255 | 0.413 | 0.166 | 0.385 | 60.72 | 121.57 | 11.66 |
| 归一化联合 MLP | 中位数 | 0.257 | 0.406 | 0.177 | 0.380 | 56.49 | 116.64 | 11.47 |
| Lookup | 采样 ×3 | 0.168 | 0.412 | 0.083 | 0.361 | 64.09 | 131.59 | 10.71 |
| 原值独立 MLP | 采样 ×3 | 0.116 | 0.216 | 0.076 | 0.191 | 64.17 | 132.53 | 12.49 |
| 归一化独立 MLP | 采样 ×3 | 0.095 | 0.321 | 0.053 | 0.291 | 65.77 | 128.16 | 12.45 |
| 归一化联合 MLP | 采样 ×3 | 0.129 | 0.337 | 0.051 | 0.311 | 70.55 | 130.66 | 12.22 |

两种归一化 MLP 都提高了 AR 中位数的小节和 beat Tempo r。联合 MLP 的小节 Tempo r 从 lookup 的 0.225 到 0.257，Duration MAE 从 118.31 到 116.64 ms。力度相关在两种解码下均以 lookup 更高。采样下三个 MLP 的小节 Tempo r 都低于同口径 lookup。

[完整九项指标](../results/exp1125_chord_ar_inputs/multilevel_metrics.csv) · [逐曲目指标与权重清单](../results/exp1125_chord_ar_inputs/multilevel_metrics.json) · [训练、生成和评估日志](../results/exp1125_chord_ar_inputs/launcher.log)

### 8.5 第二批 AR 输出分布结果

各组固定 score lookup 输入，均完成 4 轮、1,140 步。Cat500、Timing DLM + Velocity CE、Timing + Vel DLM 的最低验证 loss 权重分别在 **250、950、1,140 步**，验证 loss 为 **13.7597、16.2296、15.7048**。每卡 8 个进程对 23 首完整谱面生成并核对。主表共 276 份新 MIDI，两个 DLM 另导出 46 份连续 timing MIDI。

| 输出头 | AR 解码 | 小节 Tempo r ↑ | 小节 Vel r ↑ | Beat Tempo r ↑ | Beat Vel r ↑ | IOI MAE ↓（ms） | Duration MAE ↓（ms） | Vel MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AR Cat5000 基线 | 中位数 | 0.225 | 0.500 | 0.146 | 0.462 | 56.02 | 118.31 | 10.45 |
| Cat500 | 中位数 | 0.219 | 0.423 | 0.145 | 0.400 | 60.16 | 117.61 | 10.43 |
| Timing DLM + Velocity CE | 中位数 | 0.127 | 0.471 | 0.068 | 0.440 | 189.60 | 155.26 | 11.20 |
| Timing + Vel DLM | 中位数 | 0.181 | 0.419 | 0.129 | 0.389 | 61.40 | 228.86 | 14.05 |
| AR Cat5000 基线 | 采样 ×3 | 0.168 | 0.412 | 0.083 | 0.361 | 64.09 | 131.59 | 10.71 |
| Cat500 | 采样 ×3 | 0.174 | 0.318 | 0.091 | 0.291 | 71.89 | 125.19 | 11.14 |
| Timing DLM + Velocity CE | 采样 ×3 | −0.031 | 0.425 | −0.014 | 0.390 | 187.22 | 139.47 | 11.19 |
| Timing + Vel DLM | 采样 ×3 | 0.126 | 0.345 | 0.074 | 0.313 | 74.44 | 244.50 | 13.32 |

Cat500 在两种解码下都维持了与 Cat5000 相近的小节与 beat Tempo r，Duration MAE 更低，IOI MAE 更高。Timing DLM + Velocity CE 在中位数解码时把 **34,454/34,454** 个推进时钟的 IOI 都解到最左桶 `[-3, -2.99)`。Timing + Vel DLM 把 **41,704/41,704** 个 chord duration 都解到最左桶。随机采样仍分别有 **52.83%** 的 IOI 和 **52.44%** 的 duration 落入最左桶。改用同一轨迹的连续位置后，相应 IOI MAE 为 **189.64 ms**，Duration MAE 为 **228.92 ms**。测得的时间误差对应模型预测贴近下界，桶中心与连续位置之间的差异很小。

[完整九项指标](../results/exp1125_chord_ar_outputs/multilevel_metrics.csv) · [分字段边界桶计数](../results/exp1125_chord_ar_outputs/edge_bin_summary.json) · [逐曲目指标与权重清单](../results/exp1125_chord_ar_outputs/multilevel_metrics.json) · [训练、生成和评估日志](../results/exp1125_chord_ar_outputs/launcher.log)

### 8.6 直接位置参数的 DLM 重训

Score 继续用 lookup。沿用第二批的数据、500 个 timing 桶、AR 类别反馈、训练预算和最低验证 loss 选权重。此次 DLM 的 head 直接输出位置 $\mu$，logistic CDF 继续使用 sigmoid。首尾桶收纳超出支持的概率，连续位置导出限制到 $[-3,2]$。尺度继续用 softplus，timing 与 velocity 的尺度下限分别为 0.005 和 0.5。

| GPU | IOI、duration | Velocity |
|---:|---|---|
| 0 | 每个字段各用三分量 DLM | 128 类 CE |
| 1 | 每个字段各用单分量 DLM | 128 类 CE |
| 2 | 每个字段各用单分量 DLM | 单分量 DLM |

各 DLM 头的权重从零初始化，位置和尺度的 bias 从有效 Train 标签估计。单分量使用中位数和 `IQR / (2 ln 3)`。三个分量按标签排序等分成三组，用各组中位数、尺度和样本占比初始化。计算 IOI 统计时排除结构性零和无效真人对齐。评估按每首一份 AR 中位类别、三份 AR 随机采样执行，另外基于同一中位数轨迹导出连续 timing。三分量的连续中位数由混合 CDF 的 50% 分位数确定。 [重训目录](../results/exp1125_chord_ar_dlm_direct/)

三组都完成 4 轮、1,140 步。GPU 0、1、2 的最佳验证权重分别为 **550、500、800 步**，对应验证 NLL 为 **13.7443、13.9274、13.9393**。每卡 8 个进程完成 23 首的中位数和三次采样，共核对 276 份主评估 MIDI 与 69 份连续 timing MIDI。

| 输出头 | AR 解码 | 小节 Tempo r ↑ | 小节 Vel r ↑ | Beat Tempo r ↑ | Beat Vel r ↑ | IOI MAE ↓（ms） | Duration MAE ↓（ms） | Vel MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AR Cat500 | 中位数 | 0.219 | 0.423 | 0.145 | 0.400 | 60.16 | 117.61 | 10.43 |
| AR Timing DLM K3 + Velocity CE | 中位数 | 0.188 | 0.432 | 0.120 | 0.401 | 66.72 | 130.23 | 12.37 |
| AR Timing DLM K1 + Velocity CE | 中位数 | 0.230 | 0.392 | 0.164 | 0.375 | 68.58 | 145.20 | 11.39 |
| AR Timing + Velocity DLM K1 | 中位数 | 0.175 | 0.442 | 0.118 | 0.432 | 67.93 | 132.42 | 13.19 |
| AR Cat500 | 采样 ×3 | 0.174 | 0.318 | 0.091 | 0.291 | 71.89 | 125.19 | 11.14 |
| AR Timing DLM K3 + Velocity CE | 采样 ×3 | 0.156 | 0.375 | 0.089 | 0.334 | 65.67 | 131.50 | 11.93 |
| AR Timing DLM K1 + Velocity CE | 采样 ×3 | 0.177 | 0.362 | 0.109 | 0.333 | 70.47 | 140.72 | 11.21 |
| AR Timing + Velocity DLM K1 | 采样 ×3 | 0.180 | 0.390 | 0.102 | 0.369 | 69.45 | 138.21 | 11.71 |

三组中位数生成的 **34,454** 个推进时钟 IOI 和 **41,704** 个 duration 都没有落入最左桶。Timing DLM K1 的 IOI MAE 从旧版的 **189.60 降到 68.58 ms**，全字段 DLM K1 的 Duration MAE 从 **228.86 降到 132.42 ms**。三分量 timing 的两项时间 MAE 均低于对应单分量，单分量的小节和 Beat Tempo r 更高。同一轨迹改用连续 timing 后，三组小节 Tempo r 的改变量均小于 0.001。

[§7.2 Cat5000 对照](#72-chord-log-dev-的并行与-ar-输出对照) · [完整九项指标与旧版对照](../results/exp1125_chord_ar_dlm_direct/multilevel_metrics.csv) · [边界桶统计](../results/exp1125_chord_ar_dlm_direct/edge_bin_summary.json) · [逐曲目指标与权重清单](../results/exp1125_chord_ar_dlm_direct/multilevel_metrics.json) · [训练、生成和评估日志](../results/exp1125_chord_ar_dlm_direct/launcher.log)
