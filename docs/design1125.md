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

记 score timing 为毫秒值 $s_I,s_D$，力度为 $v$，$L(s)=\ln(\max(s,1))$。第一批 MLP 使用连续 log timing。两个归一化设置共用映射

$$
x=\bigl(\operatorname{clip}(L(s_I),0,\ln 5000)/\ln 5000,\;
        \operatorname{clip}(L(s_D),0,\ln 5000)/\ln 5000,\;
        \operatorname{clip}(v,0,127)/127\bigr).
$$

| 输入 | IOI、duration、velocity 的编码 | 与 pitch 的融合 |
|---|---|---|
| 基线 lookup | timing 查各自的 5,000 类 log-absolute 表，velocity 查 128 类表 | 三个 embedding 与 pitch 向量相加 |
| 未归一化 log 独立 MLP | $L(s_I),L(s_D),v$ 各进一个 `Linear(1,64) → GELU → Linear(64,768)` | 三个输出与 pitch 向量相加 |
| 归一化 log 独立 MLP | $x$ 的三个标量各进一个 `1 → 64 → 768` MLP | 三个输出与 pitch 向量相加 |
| 归一化 log 联合 MLP | $x$ 整体进入 `Linear(3,192) → GELU → Linear(192,768)` | 一个联合输出与 pitch 向量相加 |

归一化独立 MLP 是控制项。它与未归一化独立 MLP 的差异是输入尺度，与归一化联合 MLP 的差异是字段何时融合。两个归一化模型的数值编码参数量接近，分别是 **150,144** 和 **148,992**，当前三张输入表共 **7,778,304** 个参数。输入变换只用于编码，反量化继续用 sidecar 中的原始谱面毫秒。训练集 1,834,717 个 chord 中，score IOI 和 duration 超过 5,000 ms 的分别有 59 和 856 个。记录各组数值向量和 pitch 向量的 RMS，以观察它们相加时的相对幅度。

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
| 1 · log timing 输入消融，输出固定 Cat5000 | 未归一化独立 MLP | 归一化独立 MLP | 归一化联合 MLP |
| 2 · 输出消融，输入固定基线 lookup | Cat500 | Timing DLM + Velocity CE | Timing + Vel DLM |
| 3 · 检查输入与输出组合 | 联合 MLP + Cat500 | 联合 MLP + Timing DLM | 联合 MLP + Timing + Vel DLM |

第一批用现有 **Chord AR log dev lookup** 做对照。三项共用 lookup AR 的 encoder、GRU、演奏反馈、5,000 类 timing 输出及有效位置掩码，仅替换 score 数值编码。开窗时重置 AR 历史。每项分别生成 **AR 中位数每首一份**和 **AR 随机采样每首三份**。采样沿用 `temperature=1.0` 和 `top_p=0.95`。每卡 8 个常驻进程按 chord 数从长到短处理这 92 个任务。两套解码各与 lookup 的相同解码设置对比。 [log 输入重训目录](../results/exp1125_chord_ar_log_inputs/) · [AR 中位数 lookup 对照](../results/exp1125_chord_ar/decoding_audit/log_dev/ar.json) · [AR 采样 lookup 对照](../results/exp1125_chord_ar/log_dev/test_official/prediction_manifest.json)

第二批继续用 **Chord AR log dev lookup** 对照。Score 的 IOI、duration 输入表各保留 5,000 类，velocity 保留 128 类。三项都改用 500 个演奏 timing 桶，AR feedback 也统一读取这 500 个 timing 类别。两种 DLM 在中位数生成时复用同一条 AR 轨迹，分别用桶中心和连续位置参数导出 timing，历史反馈使用相同的中位类别。 [AR 输出实验目录](../results/exp1125_chord_ar_outputs/)

前两批回答各项改动的效果，第三批检验预先指定的归一化联合 MLP 与三种输出的组合。基线复用现有同口径权重。训练复用 13,650 个 Train 窗和 449 个验证原窗，单卡 BF16、seed 42、有效 batch 48、AdamW、`3e-4` 线性衰减、4 轮共 1,140 步。每 50 步及末步验证，每项取自己最低 valid loss 的 checkpoint。第二批三个 500 桶模型的验证 loss 都是相同目标桶的负对数概率，5,000 桶基线用原分辨率选权重。主表采用同一份端到端指标。

每项在 23 首完整官方 Test 谱面上生成，使用相同的 101 份真人参考和有效位置。报告小节与 beat Tempo/Vel Pearson、起音组 Tempo 和 chord Vel Pearson，以及原始毫秒的 IOI/Duration MAE 和 Vel MAE。保留逐曲目指标、权重步数、总参数量和输出分布的边界命中率。如果增加随机评估，基线和变体都各采样 3 次，单列采样结果。配置、日志、权重、预测和复评均存入 `INSPIRE2/results/`。

### 8.4 第一批 AR log timing 输入结果

按 §8.1 的定义重训三组。三组均完成 4 轮、1,140 步。未归一化独立、归一化独立、归一化联合 MLP 的最低验证 loss 权重分别在 **1,100、500、500 步**，对应验证 CE 为 **17.6979、18.2279、18.2587**。三卡各用 8 个进程，逐首生成一份 AR 中位数和三份 AR 采样。每组 92 份，共核对 276 份完整生成。

| Score 输入 | AR 解码 | 小节 Tempo r ↑ | 小节 Vel r ↑ | Beat Tempo r ↑ | Beat Vel r ↑ | IOI MAE ↓（ms） | Duration MAE ↓（ms） | Vel MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Lookup | 中位数 | 0.225 | 0.500 | 0.146 | 0.462 | 56.02 | 118.31 | 10.45 |
| 未归一化 log 独立 MLP | 中位数 | 0.172 | 0.359 | 0.140 | 0.341 | 62.99 | 139.82 | 11.25 |
| 归一化 log 独立 MLP | 中位数 | 0.264 | 0.431 | 0.216 | 0.400 | 57.60 | 120.42 | 10.78 |
| 归一化 log 联合 MLP | 中位数 | 0.252 | 0.373 | 0.180 | 0.343 | 58.46 | 118.95 | 11.51 |
| Lookup | 采样 ×3 | 0.168 | 0.412 | 0.083 | 0.361 | 64.09 | 131.59 | 10.71 |
| 未归一化 log 独立 MLP | 采样 ×3 | 0.119 | 0.267 | 0.068 | 0.243 | 66.70 | 141.45 | 11.87 |
| 归一化 log 独立 MLP | 采样 ×3 | 0.167 | 0.311 | 0.093 | 0.279 | 65.05 | 130.24 | 11.73 |
| 归一化 log 联合 MLP | 采样 ×3 | 0.162 | 0.294 | 0.084 | 0.257 | 65.88 | 129.26 | 12.04 |

归一化 log 独立 MLP 的中位数小节和 beat Tempo r 分别为 **0.264** 和 **0.216**，高于 lookup 的 **0.225** 和 **0.146**。两种归一化 MLP 的三次采样小节 Tempo r 为 **0.167 / 0.162**，接近 lookup 的 **0.168**。同口径复算的旧线性毫秒版本分别为 **0.095 / 0.129**。两种解码下，力度相关仍以 lookup 更高。

[完整九项指标含线性对照](../results/exp1125_chord_ar_log_inputs/multilevel_metrics.csv) · [逐曲目指标与权重清单](../results/exp1125_chord_ar_log_inputs/multilevel_metrics.json) · [训练、生成和评估日志](../results/exp1125_chord_ar_log_inputs/launcher.log) · [旧线性毫秒实验](../results/exp1125_chord_ar_inputs/multilevel_metrics.csv)

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

### 8.7 力度表绑权与独立 MLP 扩宽

以 §8.4 的归一化 log 独立 MLP 为对照。三组共用 Cat5000 timing 头、128 类力度 CE、独立 AR 历史反馈和既有的训练与评估口径。

| GPU | Score IOI 与 duration 输入 | Score velocity 输入 | 力度输出权重 |
|---:|---|---|---|
| 0 | 归一化 log 独立 MLP，宽度 64 | `128 × 768` 查表 | 与谱面力度输入共用一个参数，输出有独立 bias |
| 1 | 归一化 log 独立 MLP，宽度 64 | `128 × 768` 查表 | 独立 `128 × 768` 权重和 bias |
| 2 | 归一化 log 独立 MLP，宽度 128 | 归一化力度独立 MLP，宽度 128 | 独立 `128 × 768` 权重和 bias |

GPU 0 和 GPU 1 的力度读入表与输出权重从相同数值起步。前者将梯度汇入同一个参数，后者各自更新，因此初始前向一致。GPU 2 保留三个独立的双层 MLP，每个隐藏层由 64 个单元加宽到 128 个。整模型参数量分别为 **107,658,640、107,756,944、107,856,528**。每个变体训练 4 轮，使用最低验证 loss 的权重在 23 首上分别生成单次 AR 中位数和三次随机采样。 [实验目录](../results/exp1125_chord_ar_velocity_inputs/)

三组均完成 1,140 步，最低验证 loss 权重分别在 **500、600、650 步**。每卡 8 个进程，共核对 276 份新 MIDI。

| Score 输入 | AR 解码 | 小节 Tempo r ↑ | 小节 Vel r ↑ | Beat Tempo r ↑ | Beat Vel r ↑ | IOI MAE ↓（ms） | Duration MAE ↓（ms） | Vel MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 全 Lookup Cat5000 | 中位数 | 0.225 | 0.500 | 0.146 | 0.462 | 56.02 | 118.31 | 10.45 |
| 归一化 log 独立 MLP 64 | 中位数 | 0.264 | 0.431 | 0.216 | 0.400 | 57.60 | 120.42 | 10.78 |
| 力度查表，绑权 | 中位数 | 0.246 | 0.247 | 0.179 | 0.242 | 57.25 | 116.94 | 12.76 |
| 力度查表，独立权重 | 中位数 | 0.299 | 0.337 | 0.230 | 0.318 | 54.89 | 113.73 | 11.83 |
| 归一化 log 独立 MLP 128 | 中位数 | 0.290 | 0.447 | 0.223 | 0.415 | 58.82 | 124.88 | 11.18 |
| 全 Lookup Cat5000 | 采样 ×3 | 0.168 | 0.412 | 0.083 | 0.361 | 64.09 | 131.59 | 10.71 |
| 归一化 log 独立 MLP 64 | 采样 ×3 | 0.167 | 0.311 | 0.093 | 0.279 | 65.05 | 130.24 | 11.73 |
| 力度查表，绑权 | 采样 ×3 | 0.144 | 0.231 | 0.078 | 0.209 | 67.25 | 127.01 | 12.82 |
| 力度查表，独立权重 | 采样 ×3 | 0.161 | 0.285 | 0.086 | 0.257 | 63.96 | 125.94 | 11.83 |
| 归一化 log 独立 MLP 128 | 采样 ×3 | 0.206 | 0.367 | 0.115 | 0.330 | 65.97 | 129.21 | 11.45 |

独立力度表在两种解码下均比绑权有更高的 Vel r，力度相关仍低于宽度 64 的全 MLP。将三个 MLP 加宽，采样小节 Tempo r 由 0.167 到 0.206，Vel r 由 0.311 到 0.367。全 Lookup 的力度相关仍最高。这个宽度对照同时改变了三个输入字段的容量。

[完整九项指标](../results/exp1125_chord_ar_velocity_inputs/multilevel_metrics.csv) · [逐曲目指标与权重清单](../results/exp1125_chord_ar_velocity_inputs/multilevel_metrics.json) · [训练、生成和评估日志](../results/exp1125_chord_ar_velocity_inputs/launcher.log)

### 8.8 输入宽度 256 与字段专属输出 MLP

以 §8.7 的归一化 log 独立 MLP 128 和 §7.2 的全 Lookup Cat5000 为对照。字段专属头对同一个 768 维共享状态分别拟合三个残差 $h_a=h+\operatorname{MLP}_a(h)$。每个 MLP 用 `768 → 256 → 768` 和 GELU，末层初始化为零，再接回各自原来的输出分类层。IOI、duration、velocity 三个类别数仍为 5,000、5,000、128。训练使用 CE，AR 历史反馈仍用同一套类别与反馈表。

| GPU | Score 输入 | 输出头 | 总参数 |
|---:|---|---|---:|
| 0 | 归一化 log 独立 MLP，宽度 256 | 三个线性头 | 108,152,208 |
| 1 | 全 Lookup Cat5000 | 三个字段专属残差 MLP 加原分类层 | 116,519,568 |
| 2 | 归一化 log 独立 MLP，宽度 256 | 三个字段专属残差 MLP 加原分类层 | 109,334,928 |

用已有 MLP 128 对比 GPU 0 可检验进一步加宽的效果。用全 Lookup 对比 GPU 1、用 GPU 0 对比 GPU 2，可分别在两种输入上检验字段专属头。三组用相同 AR 训练与评估口径，每项固定四轮并按最低验证 loss 选权重。评估 23 首完整谱面，每首用一次中位数和三次随机采样。 [实验目录](../results/exp1125_chord_ar_field_heads/)

三组均完成 1,140 步，最低验证 loss 权重依次为 **450、500、850 步**，验证 CE 为 **18.1910、16.5107、17.8676**。每卡 8 个进程，核对 276 份新 MIDI。

| 模型 | AR 解码 | 小节 Tempo r ↑ | 小节 Vel r ↑ | Beat Tempo r ↑ | Beat Vel r ↑ | IOI MAE ↓（ms） | Duration MAE ↓（ms） | Vel MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Lookup Cat5000 + 线性头 | 中位数 | 0.225 | 0.500 | 0.146 | 0.462 | 56.02 | 118.31 | 10.45 |
| MLP 128 + 线性头 | 中位数 | 0.290 | 0.447 | 0.223 | 0.415 | 58.82 | 124.88 | 11.18 |
| MLP 256 + 线性头 | 中位数 | 0.243 | 0.379 | 0.190 | 0.346 | 59.42 | 119.77 | 11.91 |
| Lookup Cat5000 + 专属头 | 中位数 | 0.273 | 0.466 | 0.185 | 0.440 | 56.16 | 114.46 | 11.27 |
| MLP 256 + 专属头 | 中位数 | 0.212 | 0.384 | 0.174 | 0.374 | 63.11 | 119.45 | 12.60 |
| Lookup Cat5000 + 线性头 | 采样 ×3 | 0.168 | 0.412 | 0.083 | 0.361 | 64.09 | 131.59 | 10.71 |
| MLP 128 + 线性头 | 采样 ×3 | 0.206 | 0.367 | 0.115 | 0.330 | 65.97 | 129.21 | 11.45 |
| MLP 256 + 线性头 | 采样 ×3 | 0.162 | 0.331 | 0.091 | 0.302 | 68.18 | 129.92 | 11.82 |
| Lookup Cat5000 + 专属头 | 采样 ×3 | 0.206 | 0.366 | 0.118 | 0.332 | 60.66 | 123.54 | 11.70 |
| MLP 256 + 专属头 | 采样 ×3 | 0.159 | 0.305 | 0.108 | 0.281 | 72.16 | 135.05 | 12.48 |

Lookup 加专属头后，两种解码的小节 Tempo r 都提高，采样的 IOI 和 duration MAE 同时降低，力度相关下降。输入 MLP 从 128 到 256 后，两种解码的小节 Tempo 与 Vel r 均下降。MLP 256 再加专属头后，验证 CE 下降，但采样的三项 MAE 均变大。

[完整九项指标](../results/exp1125_chord_ar_field_heads/multilevel_metrics.csv) · [逐曲目指标与权重清单](../results/exp1125_chord_ar_field_heads/multilevel_metrics.json) · [训练、生成和评估日志](../results/exp1125_chord_ar_field_heads/launcher.log)

### 8.9 固定 MLP 128 的输出建模实验

统一使用归一化 log 独立 MLP 128，三个字段各走 `1 → 128 → 768`，和 pitch 向量相加。基线直接复用 §8.7 的 `MLP 128 + 线性头` 权重。IOI 和 duration 各为 5,000 类 log dev，velocity 为 128 类，三项等权 CE。Encoder、GRU、掩码和数据按前面的 AR 实验设置执行。

| 方法 | 输出变化 | 这一组回答的问题 |
|---|---|---|
| Cat5000 线性头 | 现有基线 | 后续各组与同一输入的基线比较 |
| Cat5000 + 顺序损失 | 三个分类头和反馈表照旧。在 CE 外加离散 CRPS | 给相邻数值连续的梯度，是否改善毫秒误差或力度曲线 |
| Cat500 | IOI 与 duration 改为 500 类，velocity 仍为 128 类 CE | MLP 128 下粗化 timing 桶的代价，也作为 DLM 同网格对照 |
| 同组 timing 条件化力度 | 先从 $h$ 得到本组 IOI、duration，再据此预测本组 velocity | 用本组已经选出的 timing，能否改善力度和 timing 的联合生成 |
| Timing DLM K1 或 K3 + Velocity CE | timing 在同一 500 桶上用单分量或三分量 logistic，力度仍用 CE | 同网格下，带数值结构的分布有何收益，增加分量是否有帮助 |
| Timing + Velocity DLM K1 | 在 Timing DLM K1 的设置上，将力度换为 128 桶的 logistic 似然 | 力度从自由分类分布改为位置与尺度模型后，Vel r 与速度误差如何变化 |

**顺序损失。** 令 $p_k$ 表示第 $k$ 类的概率、$F_k=\sum_{j=0}^kp_j$，$y$ 为真实类别。每个字段计算

$$
\mathrm{RPS}=\frac{1}{K-1}\sum_{k=0}^{K-2}\bigl(F_k-\mathbf{1}\{y\leq k\}\bigr)^2.
$$

首轮令每项 loss 为 `CE + RPS`，三个字段按原掩码各自平均，再等权相加。Timing 的距离对应等宽 log dev 网格，velocity 的距离对应整数力度。用 FP32 累计概率，并独立记录 CE 和 RPS。这一组保留类别数量、头宽度、输出反馈和解码规则。 [CDF 型概率评分原理](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)

**同组 timing 条件化力度。** 用 $p(I,D,V\mid h)=p(I\mid h)p(D\mid h)p(V\mid h,I,D)$，保留 IOI 与 duration 原头。将本组量化的 timing 中心和 IOI 的结构零、无效标签两个指示量组成四维向量，经 `4 → 64 → 768` MLP 加到力度头的 $h$ 上。结构零或缺失的 IOI 使用 0 作为占位偏差，由指示量区分。训练传入本组真人 timing 标签，生成传入这一步已选取的 timing 类别。主表用完整 free-running AR 输出。在同一验证批次的真人历史下，可以分别给力度头真人 timing 和模型 timing，观察它对本组 timing 误差的敏感度。

**500 桶与 DLM。** 统一使用 §8.2 的 500 个 timing 桶和 500 类 AR timing 反馈。DLM 沿用 §8.6 的直接位置、训练标签初始化尺度和位置、首尾桶收纳区间外概率。三分量 timing 的分量权重、位置和尺度按训练标签三等分初始化。比较 Cat500 与 DLM 可固定输出网格，K1 与 K3 可比较分量数。记录边界桶占比和 DLM 尺度分布。若导出连续中位数，沿用相同反馈类别和同一条 AR 轨迹，单列导出对照。

| 批次 | GPU 0 | GPU 1 | GPU 2 |
|---|---|---|---|
| 1 | Cat5000 + RPS | Cat500 | Cat5000 同组 timing 条件化力度 |
| 2 | Timing DLM K1 + Velocity CE | Timing DLM K3 + Velocity CE | Timing + Velocity DLM K1 |

各组训练 4 轮、1,140 步，每 50 步及末步验证，取本组最低验证 loss 的权重。用已有 23 首与 101 份真人参考，按共同可用位置评估每首一份中位数和三份随机采样。采样沿用 `temperature=1.0`、`top_p=0.95`。每卡 8 个进程按 chord 序列长度从长到短生成，指标表保留 §7.1 的九项指标。CE、RPS 与 NLL 的跨组对照按相同网格和损失定义解读。 [基线训练与评估记录](../results/exp1125_chord_ar_velocity_inputs/multilevel_metrics.csv)

更强的联合输出可在这个对照之后考虑。例如将一组演奏的 timing 和力度编码成小 latent，并用条件模型预测其分布，以便一次抽样同时改变多个字段。适合检验共同变化的假设，训练与生成流程也会随之调整。

#### 8.9.1 第一批结果

输入均为归一化 log 独立 MLP 128。三组以 seed 42 各训练 4 轮、1,140 步，共用 13,650 个训练窗与同一训练顺序。沿用 PT 官方切分，用 test 的 449 个非重叠演奏窗每 50 步及末步验证，再按最低验证 loss 选权重。每组在 23 份完整 test 谱面生成 23 份 AR 中位数和 69 份 AR 随机样本，用 101 份真人演奏作参考。已核对 276 份新预测、207 个独立采样 seed、每卡 8 个工作进程和 checkpoint SHA-256。

| 输出 | 参数量 | 选中步数 | 最低验证 loss | 第 1,140 步 loss | 选权重口径 |
|---|---:|---:|---:|---:|---|
| Cat5000 线性头（基线） | 107,856,528 | 650 | 18.1567 | 18.2862 | 三项 CE |
| Cat5000 + RPS | 107,856,528 | 500 | 18.2345 | 18.4442 | 三项 CE + RPS |
| Cat500 | 99,783,528 | 700 | 13.2849 | 13.4038 | 三项 CE，timing 500 桶 |
| Cat5000 同组 timing 条件化力度 | 107,906,768 | 650 | 18.0771 | 18.3057 | 三项 CE，力度头接收真人 timing |

| 输出 | AR 解码 | Measure Tempo r ↑ | Measure Vel r ↑ | Beat Tempo r ↑ | Beat Vel r ↑ | Onset-group Tempo r ↑ | Chord Vel r ↑ | IOI MAE (ms) ↓ | Duration MAE (ms) ↓ | Vel MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cat5000 线性头（基线） | 中位数 | 0.290 | 0.447 | 0.223 | 0.415 | 0.079 | 0.435 | 58.82 | 124.88 | 11.18 |
| Cat5000 + RPS | 中位数 | 0.282 | 0.406 | 0.213 | 0.386 | 0.085 | 0.410 | 57.98 | 122.64 | 11.04 |
| Cat500 | 中位数 | 0.294 | 0.491 | 0.231 | 0.475 | 0.082 | 0.500 | 57.42 | 113.59 | 10.24 |
| Cat5000 同组 timing 条件化力度 | 中位数 | 0.240 | 0.435 | 0.195 | 0.402 | 0.074 | 0.419 | 58.97 | 127.18 | 11.64 |
| Cat5000 线性头（基线） | 采样 ×3 | 0.206 | 0.367 | 0.115 | 0.330 | 0.030 | 0.309 | 65.97 | 129.21 | 11.45 |
| Cat5000 + RPS | 采样 ×3 | 0.162 | 0.274 | 0.091 | 0.256 | 0.042 | 0.249 | 63.95 | 128.05 | 11.75 |
| Cat500 | 采样 ×3 | 0.202 | 0.365 | 0.141 | 0.341 | 0.058 | 0.343 | 60.85 | 121.57 | 10.99 |
| Cat5000 同组 timing 条件化力度 | 采样 ×3 | 0.191 | 0.347 | 0.100 | 0.318 | 0.040 | 0.297 | 64.94 | 131.07 | 11.94 |

- **Cat500** 的中位数九项都优于 Cat5000 线性基线，力度相关在三个粒度都提高，duration MAE 下降 11.29 ms。采样时小节两个 r 基本持平，beat 和 chord 粒度的四项 r 与三项 MAE 均改善。该配置同时粗化了 timing 输出桶和历史反馈类别，参数少了 807 万。这组 Cat500 可用于第二批 DLM 的同网格对照。
- **Cat5000 + RPS** 的中位数三项 MAE 小幅下降，采样的 Measure Vel r 从 0.367 降到 0.274。在相同的真人历史和验证窗上重算，三字段归一化 RPS 之和从基线 0.14896 到 0.15024，CE 从 18.15674 到 18.08429。本轮所用的 RPS 权重为 1。
- **同组 timing 条件化力度** 的力度相关和力度 MAE 在两种 AR 解码下均退步。在相同的真人历史上，把条件头的本组 timing 从真人标签换成模型中位数，力度 CE 从 3.6061 到 3.6085，弦组力度 MAE 从 7.116 到 7.131。这个诊断固定了历史，测量的是只更换本组 timing 时的差距。

[九项指标 CSV](../results/exp1125_chord_ar_mlp128_outputs/multilevel_metrics.csv) · [逐曲目指标](../results/exp1125_chord_ar_mlp128_outputs/multilevel_metrics.json) · [覆盖与权重核对](../results/exp1125_chord_ar_mlp128_outputs/batch1_audit.json) · [固定真人历史诊断](../results/exp1125_chord_ar_mlp128_outputs/teacher_forced_diagnostics.json) · [运行日志](../results/exp1125_chord_ar_mlp128_outputs/launcher.log)

#### 8.9.2 第二批 DLM 结果

固定归一化 log 独立 MLP 128 输入，以及与 Cat500 相同的 500 桶 timing 网格和 AR 反馈类别。三组沿用第一批的训练顺序和 seed 42，各训练 4 轮、1,140 步。每首生成一份 AR 中位数和三份随机样本，并沿同一条中位数轨迹额外导出连续 timing。23 首谱面共生成 345 份新 MIDI，使用同一批 101 份真人参考评分。Cat500 复用第一批的权重和预测。

| 输出 | 选中步数 | 最低验证 loss |
|---|---:|---:|
| Cat500 | 700 | 13.2849 |
| Timing DLM K1 + Velocity CE | 1,140 | 13.3534 |
| Timing DLM K3 + Velocity CE | 1,140 | 13.2436 |
| Timing + Velocity DLM K1 | 1,100 | 13.3967 |

| 输出 | AR 解码 | Measure Tempo r ↑ | Measure Vel r ↑ | Beat Tempo r ↑ | Beat Vel r ↑ | Onset-group Tempo r ↑ | Chord Vel r ↑ | IOI MAE (ms) ↓ | Duration MAE (ms) ↓ | Vel MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cat500 | 中位数 | 0.294 | 0.491 | 0.231 | 0.475 | 0.082 | 0.500 | 57.42 | 113.59 | 10.24 |
| Timing DLM K1 + Velocity CE | 中位数 | 0.318 | 0.443 | 0.235 | 0.432 | 0.083 | 0.459 | 60.93 | 123.63 | 10.74 |
| Timing DLM K3 + Velocity CE | 中位数 | 0.259 | 0.420 | 0.205 | 0.399 | 0.065 | 0.432 | 66.43 | 125.34 | 11.24 |
| Timing + Velocity DLM K1 | 中位数 | 0.266 | 0.458 | 0.227 | 0.460 | 0.094 | 0.489 | 61.40 | 120.10 | 10.46 |
| Cat500 | 采样 ×3 | 0.202 | 0.365 | 0.141 | 0.341 | 0.058 | 0.343 | 60.85 | 121.57 | 10.99 |
| Timing DLM K1 + Velocity CE | 采样 ×3 | 0.233 | 0.342 | 0.160 | 0.326 | 0.043 | 0.328 | 62.74 | 127.58 | 11.26 |
| Timing DLM K3 + Velocity CE | 采样 ×3 | 0.190 | 0.333 | 0.150 | 0.304 | 0.047 | 0.298 | 69.73 | 126.80 | 11.49 |
| Timing + Velocity DLM K1 | 采样 ×3 | 0.270 | 0.404 | 0.197 | 0.382 | 0.098 | 0.372 | 59.59 | 121.67 | 10.53 |

- **随机采样。** 全字段 DLM K1 相比 Cat500 提高了全部六项相关性，并降低 IOI 和力度 MAE。Duration MAE 增加 0.10 ms。Timing DLM K1 只替换时间字段时，小节和拍级节奏相关提高，三项 MAE 均增加。
- **中位数。** Timing DLM K1 的小节 Tempo r 最高。Cat500 保持最低的三项 MAE 和最高的三个粒度的力度相关。Timing DLM K3 的验证 loss 最低，整曲结果中仅采样 Beat Tempo r 高于 Cat500。
- **连续导出。** 保持反馈类别和生成轨迹，改用连续 timing 中位数后，三组的小节 Tempo r 变化均小于 0.0003。每组中位数的 34,454 个有效 IOI 和 41,704 个 duration 均无最左桶命中。

结果来自单训练 seed。最低验证 loss 的 checkpoint 仍在同一批 Test 的 449 个窗口上选取。

[九项指标 CSV](../results/exp1125_chord_ar_mlp128_dlm/multilevel_metrics.csv) · [逐曲目指标](../results/exp1125_chord_ar_mlp128_dlm/multilevel_metrics.json) · [覆盖、边界桶和验证尺度核对](../results/exp1125_chord_ar_mlp128_dlm/batch2_audit.json) · [运行日志](../results/exp1125_chord_ar_mlp128_dlm/launcher.log)

#### 8.9.3 50% 历史 mask 对照

固定 §8.9.2 的归一化 log 独立 MLP 128 和全字段单分量 DLM。对每个具有有效上一位置的训练 Chord，以 0.5 概率将上一 Chord 的 IOI、duration、velocity 合并反馈向量置零。每个位置独立抽取，谱面与音高条件、BOS、当前监督标签保持完整。mask 使用独立随机数流。验证沿用完整右移真人历史，整曲推理沿用模型的自回归反馈。

复用 §8.9.2 的 0% 权重。从相同初始化和数据顺序训练 50% 组四轮，在相同的 449 窗、相同步数用最低 NLL 选权重。对 23 份测试谱面生成一次中位数和三次采样，复用 0% 组的采样种子、温度 1.0 与 top-p 0.95。以共同位置的九项指标同时比较 50% 对 0% 的采样变化，以及每组采样与中位数的差距。相关性对采样逐份评分，MAE 先对三份采样逐位置取均值。

自由生成时，采样值会进入后续 Chord 的历史，可能放大预测噪声。即使固定相同历史，随机选值也会改变当前步误差。另在相同的 449 个有真人标签的验证窗固定完整历史，比较逐 Chord 的中位数和三次采样，检查历史反馈之外的误差来源。整曲评分用于判定 mask 是否改善应用中的采样表现。

两组各训练 1,140 步。0% 组在第 1,100 步的验证 NLL 最低，为 13.3967。50% 组在第 1,050 步最低，为 13.5585。已核对两组数据哈希、初始化统计和 69 个整曲采样种子。50% 组由 8 个进程生成 23 份桶中位数、69 份随机采样，并沿中位数轨迹额外导出 23 份连续 timing MIDI。

| 历史 mask | AR 解码 | Measure Tempo r ↑ | Measure Vel r ↑ | Beat Tempo r ↑ | Beat Vel r ↑ | Onset-group Tempo r ↑ | Chord Vel r ↑ | IOI MAE (ms) ↓ | Duration MAE (ms) ↓ | Vel MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 中位数 | 0.266 | 0.458 | 0.227 | 0.460 | 0.094 | 0.489 | 61.40 | 120.10 | 10.46 |
| 0% | 采样 ×3 | 0.270 | 0.404 | 0.197 | 0.382 | 0.098 | 0.372 | 59.59 | 121.67 | 10.53 |
| 50% | 中位数 | 0.305 | 0.544 | 0.253 | 0.525 | 0.109 | 0.526 | 60.48 | 117.48 | 11.03 |
| 50% | 采样 ×3 | 0.274 | 0.476 | 0.201 | 0.440 | 0.100 | 0.416 | 59.30 | 120.03 | 11.00 |

- **50% 对 0% 的采样变化。** 六项相关性均提高，力度相关在小节、拍、Chord 分别提高 0.071、0.058、0.045。IOI 和 duration MAE 分别降低 0.30 和 1.65 ms，力度 MAE 增加 0.47。
- **50% 组内的解码差距。** 中位数的六项相关和 duration MAE 更好，采样的 IOI MAE 低 1.18 ms，力度 MAE 低 0.02。相较 0% 组，采样的 Chord Vel r 落差从 0.118 缩到 0.110，小节和拍的 Tempo r 与 Vel r 落差扩大。
- **固定真人历史。** 0% 组的 IOI、duration、velocity MAE 从中位数的 29.70 ms、64.42 ms、6.80 上升到三次采样取均值的 35.17 ms、73.07 ms、7.54。50% 组从 31.46 ms、65.78 ms、6.95 上升到 35.85 ms、74.59 ms、7.65。两组的输出均在读同一条真实历史时产生，说明单步采样本身已经带来误差。这个逐 Chord 诊断和整曲曲目等权指标的统计单位不同。

本轮基于一个训练 seed。沿用 1125 的选权重协议，449 个验证窗来自这批测试谱面。结果支持 50% 历史 mask 改善当前采样的相关性，对 exposure bias 的贡献仍缺少直接测量。

[九项指标 CSV](../results/exp1125_chord_ar_mlp128_dlm_mask50/multilevel_metrics.csv) · [逐曲目结果](../results/exp1125_chord_ar_mlp128_dlm_mask50/multilevel_metrics.json) · [0% 固定历史](../results/exp1125_chord_ar_mlp128_dlm_mask50/all_dlm_k1_mlp128_teacher_forced.json) · [50% 固定历史](../results/exp1125_chord_ar_mlp128_dlm_mask50/all_dlm_k1_mlp128_mask50_teacher_forced.json) · [运行记录](../results/exp1125_chord_ar_mlp128_dlm_mask50/runs.jsonl)

#### 8.9.4 可学习历史 mask 向量

三组复用 §8.9.3 的模型和训练设置。训练时使用同一随机数流，以 50% 概率选中有下一有效位置的历史 Chord。可学习 mask 向量从零初始化。验证与整曲生成使用完整谱面，历史协议与 §8.9.3 相同。

| 50% mask 方式 | 被选中位置的处理 |
|---|---|
| 单向量，保留 pitch | 将上一 Chord 的三字段合并反馈替换成一个 768 维向量。 |
| 单向量，遮掉 pitch | 使用相同的单向量反馈，同时在 Encoder 输入层将该历史 Chord 对应的谱面 pitch embedding 置零。保留这组的谱面数值。当前 Chord 的谱面 pitch 也可能因它是下一步的历史而被遮掉。 |
| 三个字段专属向量 | 在反馈投影前同时替换 IOI、duration、velocity 的三个 128 维 embedding。三者各有独立参数，共用同一组遮盖位置，保留谱面 pitch。 |

已有 0% 和 50% 置零组的最低验证 NLL 都出现在第 4 轮，分别是第 1,100 步的 13.3967 和第 1,050 步的 13.5585。第 3 轮内最优分别为 13.4472 和 13.5929。三轮可用于降低初筛成本，单轮 285 步，相当于少约 25% 的训练步数。现有日志只有三轮阶段的 NLL，缺少对应的整曲生成结果。四轮内截取第三轮的学习率轨迹也和按三轮重新设定线性衰减不同。本轮三卡并行，各组沿用四轮和相同的 69 个采样种子。

| 50% mask 方式 | 第 3 轮以内最佳 NLL | 四轮最佳步数 | 四轮最佳 NLL |
|---|---:|---:|---:|
| 整组置零 | 13.5929 | 1,050 | 13.5585 |
| 单向量，保留 pitch | 13.6038 | 1,100 | 13.5596 |
| 单向量，遮掉 pitch | 13.5765 | 1,050 | 13.5383 |
| 三个字段专属向量 | 13.5866 | 1,100 | 13.5820 |

| 50% mask 方式 | AR 解码 | Measure Tempo r ↑ | Measure Vel r ↑ | Beat Tempo r ↑ | Beat Vel r ↑ | Onset-group Tempo r ↑ | Chord Vel r ↑ | IOI MAE (ms) ↓ | Duration MAE (ms) ↓ | Vel MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 整组置零 | 中位数 | 0.305 | 0.544 | 0.253 | 0.525 | 0.109 | 0.526 | 60.48 | 117.48 | 11.03 |
| 整组置零 | 采样 ×3 | 0.274 | 0.476 | 0.201 | 0.440 | 0.100 | 0.416 | 59.30 | 120.03 | 11.00 |
| 单向量，保留 pitch | 中位数 | 0.295 | 0.531 | 0.236 | 0.516 | 0.136 | 0.517 | 62.64 | 126.47 | 10.69 |
| 单向量，保留 pitch | 采样 ×3 | 0.263 | 0.460 | 0.176 | 0.421 | 0.098 | 0.389 | 61.39 | 126.01 | 10.83 |
| 单向量，遮掉 pitch | 中位数 | 0.356 | 0.516 | 0.284 | 0.495 | 0.100 | 0.491 | 60.00 | 127.45 | 11.38 |
| 单向量，遮掉 pitch | 采样 ×3 | 0.268 | 0.445 | 0.194 | 0.406 | 0.074 | 0.378 | 61.59 | 123.82 | 11.26 |
| 三个字段专属向量 | 中位数 | 0.246 | 0.525 | 0.203 | 0.512 | 0.108 | 0.508 | 66.16 | 131.79 | 11.05 |
| 三个字段专属向量 | 采样 ×3 | 0.269 | 0.464 | 0.197 | 0.436 | 0.089 | 0.406 | 62.85 | 132.49 | 11.07 |

- **整曲采样。** 置零组在六项相关和两项 timing MAE 上领先三种可学习方式。单向量保留 pitch 的力度 MAE 更低，为 10.83，置零组为 11.00。三种可学习方式的中位数都在六项相关上领先各自的采样。
- **三轮与四轮。** 三种可学习方式在第 3 轮以内都选中第 850 步。训完四轮后，三组的采样时长 MAE 分别从 160.37、127.13、144.70 ms 降到 126.01、123.82、132.49 ms。三组的采样 IOI MAE 也都下降。遮掉 pitch 组的力度相关下降。差异来自同一条四轮学习率轨迹上的 checkpoint，对当前配置保留四轮更有依据。

本轮每组使用一个训练 seed，449 个选权重验证窗来自最终评分的 23 首谱面，沿用 §8.9.3 的协议。

[四轮九项指标](../results/exp1125_chord_ar_mlp128_dlm_mask_embeddings/multilevel_metrics.csv) · [第三轮与第四轮同口径比较](../results/exp1125_chord_ar_mlp128_dlm_mask_embeddings/epoch3_comparison/multilevel_metrics.csv) · [逐曲目结果](../results/exp1125_chord_ar_mlp128_dlm_mask_embeddings/multilevel_metrics.json) · [运行记录](../results/exp1125_chord_ar_mlp128_dlm_mask_embeddings/runs.jsonl)

#### 8.9.5 整曲采样 baseline

后续 ASAP 对照固定使用归一化 log 独立 MLP 128、Chord AR GRU 和全字段单分量 DLM。训练时以 50% 概率将上一 Chord 的 IOI、duration、velocity 合并反馈置零，保留谱面条件和 BOS。验证用完整右移真人历史，生成用模型自己的完整历史，采样温度为 1.0，top-p 为 0.95。现有 ASAP baseline 采用第 1,050 步选出的权重。 [配置](../results/exp1125_chord_ar_mlp128_dlm_mask50/all_dlm_k1_mlp128_mask50/run_config.json) · [训练记录](../results/exp1125_chord_ar_mlp128_dlm_mask50/all_dlm_k1_mlp128_mask50/training_complete.json)

## 9. PianoCoRe 预训练与 ASAP 微调

PianoCoRe-A 的窗口统计、PT 预训练切窗核查和两阶段训练计划见 [design1126](design1126.md)。
