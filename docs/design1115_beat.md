# 基于 Beat 的双级时间表示设计

## 1. 设计动机

当前逐音符时间目标用相邻音符或起音组之间的 IOI 表示时间。这样同一组和弦中的音符需要特殊的零 IOI 规则，而下一个起音组的时间还可能依赖音符排序。当前 score MIDI 在和弦内按音高排序，因此最低音可能意外地成为时间参照。

新设计把 score 的节拍结构显式加入输入，并让 BEAT 同时承担 beat-duration 输出：**BEAT token 提供 score beat 网格、score beat duration 和小节起始标记，并在 beat 状态上预测 performance beat duration**；**NOTE token 预测演奏音符属性**。note onset 相对所属 performance beat 起点表示；弱起、切分或演奏提前都可能令它为负，因此比例必须允许负值，也不应硬限制在 `[0, 1]`。

## 2. 双级表示

记 `b` 为 score 中的拍编号，`T0_b` 为 score 中该拍的标准时长（毫秒），`B_b` 为演奏中按确定性规则选出的拍点时间。

每个 BEAT token 表示一个 score beat。输入的 `score_duration` 是当前 score beat 到下一个 score beat 的 score 时长；对应的输出目标是 performance beat duration。它不是相邻音符的 IOI。

每个拍 token 对应从当前拍点到下一拍点的间隔：

$$
r_b^{\mathrm{beat}} = \frac{B_{b+1}-B_b}{T^0_b},\qquad r_b^{\mathrm{beat}}>0.
$$

`r=1` 表示 performance beat 时长等于 score beat 时长，`r>1` 表示放慢，`r<1` 表示加快。模型可以预测 `ln(r)` 或 ratio 的离散 token；输入的 score beat duration 仍可编码为 `ln(T^0_b)`。

**二级：NOTE token 相对所属拍起点的起音比例。**

音符 `i` 属于 score 拍 `b_i`，其演奏起音为 `t_i`：

$$
u_i^{\mathrm{note}} = \frac{t_i-B_{b_i}}{B_{b_i+1}-B_{b_i}},\qquad
t_i=B_{b_i}+u_i^{\mathrm{note}}(B_{b_i+1}-B_{b_i}).
$$

其中 `B_b` 是模型还原出的 performance beat 起点。`u=0.10` 表示拍点后 10% performance beat duration，`u=-0.05` 表示早 5%，`u=1.10` 表示落在下一 beat 起点之后 10%，但音符仍按 score 对齐关系归属原 beat。`u` 是 note 相对 performance beat 的位置。

## 3. 双级比例统计（ScorePerformer beat 映射的初步结果）

拍不是从演奏音符里挑一个“第一个音”。理论上的拍网格应由 score 的小节线、拍号和拍内分组确定：例如 4/4 通常以四分音符为拍，6/8 通常以附点四分音符为复合拍。即使该拍没有音符，beat token 仍存在；休止拍也不能删除，否则 score beat 的时间轴会断掉。

初步分析覆盖 19 首 ASAP test 乐曲、384,954 个对齐的演奏音符。当前数据没有独立的演奏 beat 标注，因此拍网格直接由 score MIDI 的拍号生成，采用仓库已有的 ScorePerformer 流程：先将相同 score 起音的音符合并成一个 onset group，取该组所有已演奏音符的 performance 起音**均值**形成 `(score_onset, performance_onset)` pair；然后由 `get_performance_beats` 对 score beat 边界做 exact hit 或相邻 onset pair 之间的线性插值，并启用单调时间过滤。因此 beat 锚点不是同 musical onset 的第一颗低音，也不携带 pitch 排序；同一 onset 组只贡献一个聚合时间。

这个流程是已有的确定性 score-to-performance beat 映射，不训练 beat detector，也不额外拟合模型。需要注意，均值仍会把和弦内部的起音差异压成一个组时间，单调过滤也可能删除局部反向 pair；因此 note-level residual 仍混合演奏偏移、和弦内不同步及对齐噪声。下方比例应看作该标准流程下的诊断，不直接等同于演奏者提前/延后的真实发生率。

### 3.1 Beat duration

这里的 `beat duration` 明确定义为**当前 beat anchor 到下一个 beat anchor 的时间间距**。性能侧使用相邻 `get_performance_beats` 输出的差，score 侧使用相邻 score beat ticks 按 score tempo 换算出的时长。它不是 note-to-note IOI。

| 统计量 | 比例 |
|---|---:|
| Beat duration 样本数 | 95,378 |
| 均值 / 中位数 | 1.105 / 1.019 |
| 5 / 25 / 75 / 95 分位数 | 0.595 / 0.895 / 1.197 / 1.810 |
| `[0, 1]` | 45.57% |
| `>1` | 54.43% |

这里大于 1 代表实际拍间时长长于 score 标准拍时长，也就是该拍局部较慢；小于 1 则较快。`r` 严格来说不会小于等于零。统计中位数略高于 1，反映该批 test 性能的估计拍长总体略长于 score 时值。

对训练目标而言，ratio 的对数更合适：`ln(r)=0` 是不偏离 score tempo，正值是放慢，负值是加快。当前 `ln(r)` 的中位数为 0.019，5/95 分位数为 -0.520/0.594。

下面比较 score 和 performance 的绝对 beat duration 以及其自然对数。两边按各自样本数归一化，log 图使用 `ln(duration in ms)`。

| 统计量 | Score beat duration | Performance beat duration |
|---|---:|---:|
| 样本数 | 95,378 | 95,378 |
| 均值 / 中位数（ms） | 338.740 / 267.857 | 375.719 / 260.330 |
| 5 / 95 分位数（ms） | 150.000 / 789.474 | 129.753 / 953.428 |
| 均值 / 中位数（ln ms） | 5.659 / 5.590 | 5.694 / 5.562 |

![Beat duration ratio and log deviation](../analysis/design1115_beat_duration_ratio_logdev.png)

![Score and performance beat duration, absolute and log](../analysis/design1115_score_perf_beat_duration_abs_log.png)

### 3.2 Note-token onset / 实际局部拍长

| 统计量 | 比例 |
|---|---:|
| Note onset 样本数 | 384,954 |
| 早于估计拍点 | 95,412（24.79%）|
| 与估计拍点重合 | 8.39% |
| 位于 `[0, 1]` 当前拍范围 | 75.20% |
| 晚于下一拍边界（`>1`） | 0.0099%（38 个音符）|
| 中位数 | 0.040（4.0%）|
| 5 / 95 分位数 | -0.061 / 0.747 |
| 1 / 99 分位数 | -0.174 / 0.861 |

负值的幅度并不都很大。按实际局部拍长分箱，`-1%–0%` 有 **7.08%**，`-10%–-1%` 有 **15.11%**，`-25%–-10%` 有 **2.08%**，小于 `-25%` 有 **0.51%**；零点占 **8.39%**。负值可能来自音符起音早于映射拍点，也可能来自同和弦各音的滚奏差异及对齐噪声，不能统称为“提前演奏”。

按区间查看，下面的比例均以全部音符为分母；`0–10%` 区间包含恰好位于拍点的音符：

| Note onset 比例区间 | 音符比例 |
|---|---:|
| `<0` | 24.79% |
| `0–10%` | 30.91% |
| `10–25%` | 6.52% |
| `25–50%` | 15.85% |
| `50–75%` | 17.21% |
| `75–100%` | 4.71% |
| `100–125%` | 0.0119% |
| `>125%` | 0.0018% |

![Note-token onset ratio distribution](../analysis/design1115_note_token_onset_ratio.png)

负值表示 note 起音早于 ScorePerformer 映射出的该拍边界，不表示 MIDI 无效。正值大多落在当前拍内。图中横轴裁切在 note `[-1,2]`、beat `[0,3]`；被裁切样本数会标在图内，精确数据保存在统计 JSON 中。使用 onset-group 均值和已有线性插值流程后，超过 1 的比例为 38/384,954。

`u<0` 表示 note 起音早于 ScorePerformer 映射出的所属 score beat 边界；`u>1` 表示它晚于下一拍边界。这些值是**未环绕的连续拍坐标**，不是限制在单拍内的相位角。大于 1 不会自动把音符重新归入下一拍，因为 note 仍按 score 对齐关系归属原拍。负值还包含同一 score onset group 内的和弦滚奏差异。

### 3.3 Note onset 范围建议

不要把 onset 强制限制在 `[0,1]`：当前约 24.79% 的样本小于 0，零点前后的负值是有意义的。首轮实验把 `[-0.25,1]` 均匀划分为 500 个 token，区间外值裁到最近端点并记录裁剪比例。该区间覆盖约 99.48% 样本；1%–99% 分位数为 `[-0.174,0.861]`。裁剪策略会损失少量尾部差异，后续可单独比较 tail-token 方案。

## 4. Duration 统计

duration 以逐音符对齐的 score/performance 时值计算。定义时值比例和自然对数偏差：

$$
q_i^{\mathrm{dur}}=\frac{D_i^{\mathrm{perf}}}{D_i^{\mathrm{score}}},\qquad
d_i^{\mathrm{dur}}=\ln(q_i^{\mathrm{dur}}).
$$

`q=1` 或 `d=0` 表示演奏时值等于 score；负的 log deviation 表示演奏时值较短，正值表示较长。log deviation 将“缩短一半”和“延长一倍”表示为相反数（`-ln 2` 和 `+ln 2`）。

| 统计量 | Duration ratio | Duration log deviation |
|---|---:|---:|
| 样本数 | 384,954 | 384,954 |
| 均值 / 中位数 | 0.716 / 0.559 | -0.628 / -0.582 |
| 5 / 95 分位数 | 0.144 / 1.703 | -1.935 / 0.533 |
| 小于 score 时值 | 77.66% | 77.66%（`d<0`）|
| 大于 score 时值 | 22.19% | 22.19%（`d>0`）|

ratio 图横轴显示 `[0,3]`，log deviation 图显示 `[-2,2]`；图内标注了视野外样本数，完整极值见统计 JSON。

下面分别叠加 score 和 performance 的绝对时值分布、自然对数时值分布。为便于比较，两类直方图分别按各自样本数归一化；log 图使用 `ln(duration in ms)`，横轴平移单位不会影响分布形状。

![Note duration ratio and log deviation](../analysis/design1115_note_duration_ratio_logdev.png)

![Score and performance note duration, absolute and log](../analysis/design1115_score_perf_duration_abs_log.png)

## 5. Beat duration 与 note duration 的统一数值设计

BEAT 输入中的 score beat duration、BEAT 输出中的 performance beat duration、NOTE 输入中的 score note duration 和 NOTE 输出中的 performance note duration 都是正值时长，可以采用统一的 absolute-log 或 score-relative log-dev 数值设计。输入 embedding 仍使用 score-side absolute-log，输出 head 分别预测 beat 与 note 的 performance duration。

可以比较以下两个统一协议。绝对 log duration 使用自然对数毫秒值 `ln(D_ms)`。两种 duration 输出都使用 500 个数值 token；note onset 也固定使用 500 个数值 token，区间为 `[-0.25,1]`。

| 协议 | encoder 输入 | decoder 历史输入 | beat/note 输出目标 | 建议主体区间 |
|---|---|---|---|---|
| Absolute-log | `ln(D_score)`，包括 beat 和 note duration | `ln(D_perf_previous)` | `ln(D_perf)` | `[3,8]` |
| Score-relative log-dev | `ln(D_score)`，包括 beat 和 note duration | `ln(D_perf_previous)` | `ln(D_perf)-ln(D_score)` | `[-2,3]` |

Absolute-log 的优点是 encoder、decoder 输入和 duration 输出处于完全相同的语义空间，输出可以直接量化后回馈给下一步 decoder。Score-relative log-dev 的优点是输出只表达相对于 score duration 的演奏偏差，减少曲目速度和时值本身的变化；但它的输出不能直接作为下一步 decoder 的输入。生成时应先还原：

$$
\ln D^{\mathrm{perf}}_i
=\ln D^{\mathrm{score}}_i+d_i^{\mathrm{dur}},
\qquad
D^{\mathrm{perf}}_i=\exp\left(\ln D^{\mathrm{score}}_i+d_i^{\mathrm{dur}}\right),
$$

再把还原后的 `ln(D_perf)` 量化并作为历史 decoder 输入。beat head 和 note duration head 都可以使用同一还原公式；beat head 的输出用于累积得到 `B_b`，note onset 再相对这些 beat anchors 计算。

首轮协议严格使用 500 个 bins：`[3,8]`、`[-2,3]` 和 `[-0.25,1]` 均匀分箱，区间外样本裁到最近端点。裁剪比例必须单独记录；后续可以增加 tail-token 版本，但不与首轮 500-token 结果混合比较。

输入的 absolute-log 时长词表可以采用两种消融：

1. **Shared duration value table**：beat duration 与 note duration 共用一张数值 embedding 表，使用类型/slot projection 区分上下文。
2. **Separate duration value tables**：两者使用独立表，参数量增加但不会强迫两种 duration 共享 embedding 几何。

score/performance absolute-log 输入词表可以在 beat 与 note duration 之间 shared 或 separate。若使用 log-dev 输出，其输出 token 表表示偏差，应与 absolute-log 输入表 separate；shared/separate 以及 absolute-log/log-dev 都应通过相同训练协议进行消融。

统计协议、分位数和逐曲目结果见 [`../analysis/design1115_beat_stats.json`](../analysis/design1115_beat_stats.json)，可由 [`../../script/analyze_beat_relative_onsets.py`](../../script/analyze_beat_relative_onsets.py) 重算。当前分析直接使用 score MIDI 的 time signature 生成 beat 网格。

## 6. Token 与解码结构

不额外使用 `BAR` token。小节边界直接编码为 beat 的二值属性 `measure_start`，因此 encoder 输入和 decoder 目标序列都包含 beat 和 note 两种事件：

```text
BEAT(score_duration, measure_start)
NOTE(pitch, score_duration, score_onset, score_velocity)
NOTE(...)
BEAT(score_duration, measure_start)
NOTE(...)
...
```

两类事件使用不同的属性组合，但都采用属性 embedding 相加：

$$
h^{\mathrm{beat}}=e_{\mathrm{beat}}+e_{\mathrm{score\_duration}}+e_{\mathrm{measure\_start}},
$$

$$
h^{\mathrm{note}}=e_{\mathrm{pitch}}+e_{\mathrm{score\_duration}}+e_{\mathrm{score\_onset}}+e_{\mathrm{score\_velocity}}.
$$

事件类型必须显式编码。实际输入向量还应加上 type embedding，或使用彼此独立且有明确 type ID 的 `BEAT` 与 `NOTE` 向量；不能从 duration 数值或 hidden state 猜 token 类型。经过 encoder/decoder 后，按原序列的 type mask 将 beat hidden states 和 note hidden states 分别路由到对应输出目标：beat 状态只监督 performance beat duration，note 状态监督 note duration、onset 和 velocity。这样同一长度事件序列中的两类位置有确定的输出对齐关系。

建议首版**共享 duration 输入数值 embedding 表，使用独立的 beat-duration 与 note-duration 输出 head**。共享输入表让相同的 log-duration 数值具有相同表示；独立 head 则允许 beat tempo 与 note articulation 学到不同的条件分布。若输出均采用同一种 duration 数值定义和相同 token 边界，也可以共享 duration 输出投影矩阵，但必须保留 type-conditioned bias 或小型 role projection；完全不区分 beat/note 的单一输出 head 不作为首选。可将“共享输出 head”与“独立输出 head”作为后续消融，单独判断参数共享的影响。

BEAT 的 `score_duration` 是当前 score beat 到下一个 score beat 的 score 时长；BEAT head 输出对应的 performance beat duration。NOTE 的 `score_duration` 是 score 音符时值；NOTE decoder 输出 performance duration/onset/velocity。`measure_start=1` 表示该 beat 是小节第一拍，其他 beat 为 0。拍号、拍内位置等上下文可以继续作为 score-side feature，不需要单独的 BAR 事件。

同一 beat 下的所有 note 按既有确定性顺序排列，但不能在窗口中间切开 beat。一个训练 window 满足：

1. 第一个 token 必须是 `BEAT`；
2. 后面跟随该 beat 及后续 beat 的 note，直到达到 note 数量上限；
3. 最后一个 token 必须是 `NOTE`；下一个 token 要么是下一个 `BEAT`，要么是序列结束；
4. 不能在同一 beat 的 note 集合中间截断。

窗口长度约束应以 note 数量为主，beat token 作为额外开销。这样每个样本都拥有完整的 beat 起点，且 beat duration 不会因窗口边界被拆开。

encoder 输入和 decoder 目标都包含 beat 与 note，因此一个 window 的事件序列长度都是 `N_note + N_beat`（teacher-forcing 的 decoder 输入通常是右移后的序列，另加 BOS）。每个位置根据显式 type mask 路由到对应输出：BEAT 位置预测 performance beat duration，NOTE 位置预测 performance note duration/onset/velocity。BEAT 输出累积成 performance beat anchors，再用于还原 note onset。这样 beat 与 note 共享事件拓扑，但使用不同的输出 head。

同一拍内的和弦音共享拍坐标，不再以最低音或前一个和弦音作起音锚点。note onset 标签彼此独立地相对 beat 起点计算；生成后按还原出的绝对起音时间排序 MIDI。

## 7. 采用的现有流程

不另行实现 beat detector。复用仓库已有的 ScorePerformer `get_bar_beat_ticks` 与 `get_performance_beats`：前者按 score 拍号生成拍网格；上游把同 musical onset 的已演奏音符聚合成一个 onset pair，当前 ScorePerformer SPMuple 实现取这些音符 performance onset 的均值；后者在 beat 边界精确命中 onset pair 时取对应时间，否则在相邻 pair 之间线性插值，并可启用单调时间过滤。

因此它不会在一个 chord 中挑最低音或第一条 note 来代表整拍。onset 组聚合的均值仍可能受和弦滚奏影响，但确定且无 pitch-order bias。对于 score beat 落在两个 onset groups 之间、或该拍没有音符的情况，线性映射给出其 performance 时间，不需要为每拍找一个真实 note。

在当前方案中，beat-duration head 预测 beat duration，note onset 头使用 500-bin 有序离散分布预测带符号的 `u`，note-duration 头预测 `ln(D^{perf}/D^{score})` 并按 score 时值还原。onset 固定使用 `[-0.25,1]`，区间外值裁到端点并记录比例；若后续使用连续头，仍需对错配标签做稳健处理。

## 8. 相关符号生成方法

- **REMI / Pop Music Transformer**（Huang & Yang, 2020）用 `Bar`、`Position`、`Tempo` 等事件显式表示小节、拍内位置和速度，再生成音符。它支持把 metrical structure 明确放进序列；REMI 的 `Position` 是小节内量化位置，本设计的 `u` 则表达对齐演奏拍点后的连续、可带符号偏移。
- **Compound Word Transformer**（Hsiao et al., 2021）将相关符号字段组合为 compound event，并建模小节结构。它支持在保留结构上下文的同时组合音符属性，但本身并未提出这里的带符号演奏起音比例。
- **ScorePerformer / SPMuple** 使用 score-performance 对齐的音符 tuple，并包含 beat-position 上下文，是 EPR 中较近的先例。本设计把这种 beat-position 上下文显式编码为输入 beat token，再让 note decoder 预测逐音符拍内起音比例。

这些工作支持“显式表示 beat 结构”的设计动机，但不能证明本方案已经有效。需要实验检验的假设是：把 score beat 结构作为输入上下文，是否能提升 note-level 演奏属性的建模并简化起音目标学习。

## 9. 实验队列与比较设置

当前 IOI 以相邻音符或起音组为参照；新方案以 score beat 网格作为显式输入，同时预测 beat-level performance duration 和 note 相对 beat 的起音位置。它让同拍和弦使用统一坐标，并显式建模 performance tempo curve，但不保证 HR、Wasserstein 或听感一定变好。

### 9.1 固定协议

所有队列使用相同的数据划分、窗口上限、优化器、学习率、训练 epoch、early stopping 和随机种子组。窗口以 note 数量为主约束，首 token 为 `BEAT`，末 token 为 `NOTE`，不能在一个 beat 的 note 集合中间切割。

数值目标固定为 500 个 bins：

| 目标 | 数值空间 | bins | 说明 |
|---|---|---:|---|
| Beat/note duration absolute-log | `[3,8]` | 500 | 输出 `ln(D_perf)`，输入 score 和 decoder 历史仍为 absolute-log |
| Beat/note duration score-relative log-dev | `[-2,3]` | 500 | 输出 `ln(D_perf)-ln(D_score)`，生成后加回 score log |
| Note onset | `[-0.25,1]` | 500 | 相对 performance beat 的 signed onset ratio |

这里的“统一 500 token”表示 bin 数和离散化协议一致，不表示三种语义必须共享同一张 embedding 表。duration 的 beat/note 表共享是一个实验因素；onset 语义不同，默认使用独立表。区间外值首轮裁到端点，并报告裁剪比例。

### 9.2 队列 A：结构与输出目标

| 编号 | Beat token | `measure_start` | Duration target | Duration 输入表 | Duration 输出 head | 目的 |
|---|---|---|---|---|---|---|
| A1 | 有 | 有 | absolute-log | beat/note shared | beat/note separate | 默认 beat 结构基线 |
| A2 | 有 | 有 | log-dev | beat/note shared | beat/note separate | 比较 absolute-log 与 score-relative target |
| A3 | 有 | 无 | absolute-log | beat/note shared | beat/note separate | `measure_start` 消融 |

A0 已有 legacy note-only 结果，本轮不重新训练；旧目标与 beat 目标不同，不作为严格的单因素对照。A1 与 A2 判断输出采用 absolute-log 还是 log-dev；A1 与 A3 判断小节第一拍标记是否提供额外信息。A1、A2、A3 均使用相同的三个随机种子组。

### 9.3 队列 B：duration 输入表共享

在 A1 和 A2 中分别把 duration 输入表从 shared 改成 separate：

| 编号 | 结构 | Duration target | Duration 输入表 | 输出 head |
|---|---|---|---|---|
| B1 | beat + `measure_start` | absolute-log | beat/note separate | separate |
| B2 | beat + `measure_start` | log-dev | beat/note separate | separate |

B1 对比 A1，B2 对比 A2。separate 表示 beat 与 note 各有自己的 500-token 输入表；score 输入与 decoder 历史输入在各自事件类型内使用相同表。该组只检验 beat duration 与 note duration 是否应共享输入 embedding；输出 head 保持 separate，避免混淆参数共享因素。

### 9.4 队列 C：duration 输出 head 共享

固定 A1 或 A2 中表现更好的 target 和输入表，比较：

| 编号 | Duration 输出 head | 类型条件 | 目的 |
|---|---|---|---|
| C1 | beat/note separate | 无需额外条件 | 默认方案 |
| C2 | shared projection | 保留 type-conditioned bias/role projection | 检验输出参数共享 |
| C3 | 完全 shared projection | 无 role projection | 极限参数共享对照 |

C2 是可行的共享方案；C3 只作为诊断，不作为默认实现，因为 beat tempo 和 note articulation 的条件分布不同。

### 9.5 队列 D：第二阶段交互实验

只有在 A–C 得到稳定结果后再运行以下组合，避免在第一轮形成过大的笛卡尔积：

1. **D1：** beat duration 使用 absolute-log、note duration 使用 log-dev；
2. **D2：** beat duration 使用 log-dev、note duration 使用 absolute-log；
3. **D3：** shared duration 输入表与 separate 输出 head 的最佳组合；
4. **D4：** separate duration 输入表与 shared type-conditioned 输出 head 的最佳组合；
5. **D5：** 加入或移除 `measure_start` 的最佳组合复验。

这组用于判断 beat tempo 与 note articulation 是否需要不同的 target 表示；若 A1/A2 已显示统一 target 明显更好，D1/D2 可以取消。

### 9.6 评估与选择规则

每个设置至少使用三个随机种子。报告 valid loss、beat duration log/ratio 误差、note onset token accuracy 与 signed error、note duration 误差、生成后的 token-level timing JS、PP/PN 和 HR；同时检查 teacher-forced 与 autoregressive 输出。

模型选择不只看总 valid loss。先确认 beat duration、note onset、note duration 三个 head 均没有明显退化，再比较 autoregressive timing JS、beat anchor 重建误差和 HR。若 valid loss 与 HR 或 JS 方向不一致，保留全部指标，不用单一指标强行排序。

窗口重建必须验证 score beat 到 note 的归属、performance beat anchors 的连续性、note 绝对起音、跨 beat 顺序、音符对齐、拍号变化、弱起、休止，以及 `u<0` 或 `u>1` 的音符。

不能只凭 valid loss 选型；此前实验已观察到 loss 与 HR 不总是同向。首要检验点是两级时间表示是否更清晰且更易学习，质量结论仍需多随机种子及 held-out 指标支持。

## 10. 实现协议与实际运行（2026-11-16）

### 10.1 已实现的数据与模型

实现入口为 [`build_dinr_beat_sidecars.py`](../src/data_process/build_dinr_beat_sidecars.py)、[`dinr_beat_dataset.py`](../src/data_process/dinr_beat_dataset.py)、[`dinr_beat.py`](../src/model/dinr_beat.py) 和 [`train_dinr_beat.py`](../src/train/train_dinr_beat.py)。BEAT/NOTE 共享事件拓扑，输入为属性向量相加；decoder 历史序列右移一位，第一位是 BOS，输出类型直接由已知的 score 事件类型决定。默认 BEAT 预测 duration，NOTE 预测 duration/onset/velocity，不生成 pitch 或事件类型。

四个 head 的 CE 分别在其有效事件上求均值，再等权相加：beat duration、note duration、note onset、note velocity。PAD 不参与 loss。模型及所有数值表从随机权重初始化，不加载 PT rendering checkpoint。duration 输出可逐类型独立设置，覆盖 D1/D2；log-dev 输出加回未量化的 score log duration，再量化成 absolute-log 后用于下一步 decoder 历史输入。所有模型使用相同的 500-token 数值范围和量化器。

本轮 decoder mask 沿用 timestep 的完整事件遮蔽：训练时每个非 BOS/PAD 的历史事件以 0.5 概率由一个可学习 mask 向量替换，包含该历史 note 的 pitch；验证与推理不遮蔽。score encoder 信息始终完整。duration shared 输入表按 beat/note 类型使用同一张 500 行表；separate 使用两张独立表。默认分开 beat/note 输出 head，C2 为共享投影加类型 bias，C3 无额外类型 bias/role projection。

### 10.2 冻结 sidecar 与完整性审计

数据写到外层 `data/ASAP_processed_dinr_beat_v1/`；sidecar 保存连续值，训练按配置量化，不为每项设置复制整份数据。work-level held-out 12 首验证集与旧协议一致，三种 split 在作品级互不重叠。

| split | 作品 | 演奏 | 唯一 note | BEAT 事件 | 窗口 |
|---|---:|---:|---:|---:|---:|
| train | 175 | 837 | 2,739,556 | 414,749 | 10,447 |
| valid | 12 | 47 | 135,467 | 24,132 | 507 |
| test | 19 | 82 | 384,954 | 95,378 | 1,479 |

窗口上限为 512 note，首事件是 BEAT、末事件是 NOTE，不能拆分同一 beat；完整保留窗口内部的无音符拍。相邻训练窗口约 50% note 重叠，实际起点落在 beat 边界。所有 note 均至少被一个窗口覆盖。最大事件长度分别是 train 811、valid 747、test 771；它大于 note 上限是因为还包含 BEAT 事件。

Beat mapper 复用 ScorePerformer 的 onset-group performance 均值和线性插值；**调用 mapper 时将毫秒转成秒**，返回后转回毫秒，以匹配其 tempo-600 单调过滤阈值。这修正了初步统计脚本直接传毫秒导致的单位不一致，因此本轮裁剪比例以冻结 sidecar 的审计结果为准，不能直接把前文初步统计当成完全相同的数据版本。

测试沿用旧评估中两首 MS-score 的 MUSICAL51 fallback，避免漏掉测试作品。其标签未经过 grouped-median 归一化；其他作品保留已有 grouped-median 标签。每个来源的时间归一化元信息保存在 sidecar 中；本轮不悄悄改变这两首的归一化规则。完整裁剪统计与文件 SHA256 见 [数据审计](design1115_beat_data_audit.json)，重算入口为 [`audit_dinr_beat.py`](../script/audit_dinr_beat.py)。

500-token 端点裁剪比例（包括下界与上界，连续值仍保存在 sidecar）：

| split | beat abs-log | beat log-dev | note abs-log | note log-dev | note onset |
|---|---:|---:|---:|---:|---:|
| train | 0.574% | 0.004% | 0.647% | 3.794% | 0.072% |
| valid | 0.224% | 0.004% | 0.512% | 4.191% | 0.054% |
| test | 0.070% | 0.008% | 0.380% | 4.166% | 0.767% |

这说明 note log-dev 的 `[-2,3]` 仍会裁掉约 4% 的尾部；首轮遵守统一约定区间并报告裁剪，不根据 test 临时扩大词表。输入 score absolute-log 的裁剪比例单列保存在审计 JSON 中。

### 10.3 三卡完整队列与依赖

A0 不运行。每项设置三个独立训练种子，因此计划共 **12 项设置、36 次训练**；每个最佳模型额外做三轮测试采样。A1/A2/A3/B1/B2/C2/C3/D1/D2/D4 使用训练种子 42、43、44；D3/D5 为独立复验，使用 45、46、47。采样标准差和训练标准差分别记录，不把“三次采样”当成“三次训练”。

1. 第一阶段运行 A1/A2/A3/B1/B2，共 15 次训练。
2. C1 复用对应的 A1/A2 结果；C2/C3 在 A1/A2 中选择更好的 duration target，共 6 次训练。
3. D1/D2 使用 shared 输入表、separate 输出 head，分别比较 beat absolute-log + note log-dev 与 beat log-dev + note absolute-log，共 6 次训练；本轮两个组合都运行。
4. 在 A1/A2/D1/D2 中选择 duration target 组合。D3 对该 shared 输入表 + separate head 组合独立复验；D4 使用相同 duration target，改为 separate 输入表 + shared conditioned head，共 6 次训练。
5. 在 A–D4 中选择表现更好的设置，D5 翻转该设置的 `measure_start` 属性作独立复验，共 3 次训练。

为了使自动队列可复现，以上配置选择只采用 **valid 贪心预测的物理误差**，不查看 test：beat duration log MAE、note duration log MAE、note onset ratio MAE、velocity MAE/127 四项的平均值，再跨三个训练种子求均值。模型 checkpoint 的选择仍按最低总 valid CE。这个分数是预先约定的队列控制规则，不宣称取代全部生成质量指标；各 head、JS、HR 仍完整报告。

全部设置最大 16 epoch；每半 epoch 验证一次；patience=4；只保留最低 valid checkpoint。AdamW，初始 LR 3e-4、线性衰减、无 warmup，weight decay=0，有效 batch=32（microbatch=8、累积=4）。若出现显存不足，队列将 microbatch 减为 4、累积增为 8，保持有效 batch 不变并记录。每次 optimizer update 保存 train loss，验证点单独记录。

### 10.4 评估、生成与结果归档

[`evaluate_dinr_beat.py`](../src/evaluate/evaluate_dinr_beat.py) 包含 valid 贪心诊断、teacher-forced 与 autoregressive 采样。测试保留 19 首作品和全部 82 个对齐参考演奏；每个模型三轮采样，temperature=1、top-p=0.9。TF 使用每个真实参考演奏作为历史；AR 只读取 score 和生成历史，绝不使用真实演奏属性。

自回归使用不重叠的完整 beat 窗口，每窗最多 512 note；各窗起点重置 BOS/cache，累计 performance beat 时间跨窗延续。把不同窗口与采样轮次按固定顺序做 batch=16 解码以减少串行开销；随机流以作品标识固定，固定任务顺序和 batch 大小可重放。输出保存原始 token NPZ、事件身份、MIDI、逐作品和逐轮指标。

Token JS 在 500-bin duration、500-bin onset 和 128-bin velocity 上计算，单位为 nats；误差与 beat anchor 还原始终保留 score event 对齐。播放 MIDI 按还原后的实际 onset 排序；若起音小于零，对整个片段做共同平移，不单独把提前音压到零。Legacy PN/PP W1 与 HR 另按 time-sorted MIDI note index 计算，并沿用旧的固定 human anchor；它与 event-aligned token 指标属于不同对齐协议，报告中明确区分。PP 保留 W1，不虚构未校准的 PP HR。

[实验结果与曲线](exp1115_BEAT.md) 由队列自动刷新。主表对三个独立训练种子报告 mean ± sample std（小数三位），每个种子的 HR 先平均其三轮采样；CSV 另保留各模型采样标准差。曲线按 A/B/C/D 分为四列两行，横轴自适应该组实际最大 epoch。README 直接嵌入同一张图。

### 10.5 监督运行与恢复

队列入口为 [`run_dinr_beat_queue.py`](../script/run_dinr_beat_queue.py)，运行在 GPU0/1/2；每张卡一次一个任务，依次执行训练、valid、TF/AR 测试、指标归档和图表更新。manifest 保存 PID、心跳、阶段、退出码和自适应配置选择依据；禁止重复启动同一队列。评估可从已完成作品恢复，失败任务保留日志，依赖阶段不会把缺失结果当成完成。

```bash
cd INSPIRE2
python -m src.data_process.build_dinr_beat_sidecars
python script/audit_dinr_beat.py
python -m unittest discover -s tests -p test_dinr_beat.py -v
python script/run_dinr_beat_queue.py --gpus 0 1 2
```

运行目录为外层 `results/exp1115_beat/`；数据、checkpoint 和运行日志保存在本地，仅 INSPIRE2 内层仓库包含实现、审计、设计及结果文档，不提交外层仓库。
