# 基于 Beat 的双级时间表示设计

## 设计动机

当前逐音符时间目标用相邻音符或起音组之间的 IOI 表示时间。这样同一组和弦中的音符需要特殊的零 IOI 规则，而下一个起音组的时间还可能依赖音符排序。当前 score MIDI 在和弦内按音高排序，因此最低音可能意外地成为时间参照。

新设计把 score 的节拍结构显式加入输入，并让 BEAT 同时承担 beat-duration 输出：**BEAT token 提供 score beat 网格、score beat duration 和小节起始标记，并在 beat 状态上预测 performance beat duration**；**NOTE token 预测演奏音符属性**。note onset 相对所属 performance beat 起点表示；弱起、切分或演奏提前都可能令它为负，因此比例必须允许负值，也不应硬限制在 `[0, 1]`。

## 双级表示

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

## 双级比例统计（ScorePerformer beat 映射的初步结果）

拍不是从演奏音符里挑一个“第一个音”。理论上的拍网格应由 score 的小节线、拍号和拍内分组确定：例如 4/4 通常以四分音符为拍，6/8 通常以附点四分音符为复合拍。即使该拍没有音符，beat token 仍存在；休止拍也不能删除，否则 score beat 的时间轴会断掉。

初步分析覆盖 19 首 ASAP test 乐曲、384,954 个对齐的演奏音符。当前数据没有独立的演奏 beat 标注，因此拍网格直接由 score MIDI 的拍号生成，采用仓库已有的 ScorePerformer 流程：先将相同 score 起音的音符合并成一个 onset group，取该组所有已演奏音符的 performance 起音**均值**形成 `(score_onset, performance_onset)` pair；然后由 `get_performance_beats` 对 score beat 边界做 exact hit 或相邻 onset pair 之间的线性插值，并启用单调时间过滤。因此 beat 锚点不是同 musical onset 的第一颗低音，也不携带 pitch 排序；同一 onset 组只贡献一个聚合时间。

这个流程是已有的确定性 score-to-performance beat 映射，不训练 beat detector，也不额外拟合模型。需要注意，均值仍会把和弦内部的起音差异压成一个组时间，单调过滤也可能删除局部反向 pair；因此 note-level residual 仍混合演奏偏移、和弦内不同步及对齐噪声。下方比例应看作该标准流程下的诊断，不直接等同于演奏者提前/延后的真实发生率。

### 一级：Beat duration

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

### 二级：Note-token onset / 实际局部拍长

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

### Note onset 范围建议

不要把 onset 强制限制在 `[0,1]`：当前约 24.79% 的样本小于 0，零点前后的负值是有意义的。若使用有界分类表示，建议主体区间设为 `[-0.25,1.0]`，两端各保留 underflow/overflow tail 类，不要把超界值直接裁到边界。该区间覆盖约 99.48% 样本；1%–99% 分位数为 `[-0.174,0.861]`。超界极值很少，但可能混有 beat 插值或对齐异常，保留 tail 可避免模型把异常极值当成常见边界值。若使用连续回归，则不必硬设有限输出范围，可用稳健损失控制极端残差。

## Duration 统计

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

## Beat duration 与 note duration 的统一数值设计

BEAT 输入中的 score beat duration、BEAT 输出中的 performance beat duration、NOTE 输入中的 score note duration 和 NOTE 输出中的 performance note duration 都是正值时长，可以采用统一的 absolute-log 或 score-relative log-dev 数值设计。输入 embedding 仍使用 score-side absolute-log，输出 head 分别预测 beat 与 note 的 performance duration。

可以比较以下两个统一协议：

| 协议 | encoder 输入 | decoder 历史输入 | beat/note 输出目标 | 建议主体区间 |
|---|---|---|---|---|
| Absolute-log | `ln(D_score)`，包括 beat 和 note duration | `ln(D_perf_previous)` | `ln(D_perf)` | `[3,8]` |
| Score-relative log-dev | `ln(D_score)`，包括 beat 和 note duration | `ln(D_perf_previous)` | `ln(D_perf)-ln(D_score)` | `[-3,2]` |

Absolute-log 的优点是 encoder、decoder 输入和 duration 输出处于完全相同的语义空间，输出可以直接量化后回馈给下一步 decoder。Score-relative log-dev 的优点是输出只表达相对于 score duration 的演奏偏差，减少曲目速度和时值本身的变化；但它的输出不能直接作为下一步 decoder 的输入。生成时应先还原：

$$
\ln D^{\mathrm{perf}}_i
=\ln D^{\mathrm{score}}_i+d_i^{\mathrm{dur}},
\qquad
D^{\mathrm{perf}}_i=\exp\left(\ln D^{\mathrm{score}}_i+d_i^{\mathrm{dur}}\right),
$$

再把还原后的 `ln(D_perf)` 量化并作为历史 decoder 输入。beat head 和 note duration head 都可以使用同一还原公式；beat head 的输出用于累积得到 `B_b`，note onset 再相对这些 beat anchors 计算。

两种协议都应保留 underflow/overflow tail，而不是把区间外样本静默裁到边界。`[3,8]` 与 `[-3,2]` 可以先作为主体区间；最终边界应只由 train split 决定，并在 valid/test 报告 tail 比例。

输入的 absolute-log 时长词表可以采用两种消融：

1. **Shared duration value table**：beat duration 与 note duration 共用一张数值 embedding 表，使用类型/slot projection 区分上下文。
2. **Separate duration value tables**：两者使用独立表，参数量增加但不会强迫两种 duration 共享 embedding 几何。

score/performance absolute-log 输入词表可以在 beat 与 note duration 之间 shared 或 separate。若使用 log-dev 输出，其输出 token 表表示偏差，应与 absolute-log 输入表 separate；shared/separate 以及 absolute-log/log-dev 都应通过相同训练协议进行消融。

统计协议、分位数和逐曲目结果见 [`../analysis/design1115_beat_stats.json`](../analysis/design1115_beat_stats.json)，可由 [`../../script/analyze_beat_relative_onsets.py`](../../script/analyze_beat_relative_onsets.py) 重算。当前分析直接使用 score MIDI 的 time signature 生成 beat 网格。

## Token 与解码结构

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

## 采用的现有流程

不另行实现 beat detector。复用仓库已有的 ScorePerformer `get_bar_beat_ticks` 与 `get_performance_beats`：前者按 score 拍号生成拍网格；上游把同 musical onset 的已演奏音符聚合成一个 onset pair，当前 ScorePerformer SPMuple 实现取这些音符 performance onset 的均值；后者在 beat 边界精确命中 onset pair 时取对应时间，否则在相邻 pair 之间线性插值，并可启用单调时间过滤。

因此它不会在一个 chord 中挑最低音或第一条 note 来代表整拍。onset 组聚合的均值仍可能受和弦滚奏影响，但确定且无 pitch-order bias。对于 score beat 落在两个 onset groups 之间、或该拍没有音符的情况，线性映射给出其 performance 时间，不需要为每拍找一个真实 note。

在当前方案中，beat-duration head 预测 beat duration，note onset 头使用连续分布或有序离散分布预测带符号的 `u`，note-duration 头预测 `ln(D^{perf}/D^{score})` 并按 score 时值还原。若 onset 使用 DINR 分类表，主体 bin 建议覆盖 `[-0.25,1]` 并显式包含两侧 tail；其他 bin 边界只能由 train split 决定。若使用 CINR 连续头，仍需对错配标签做稳健处理。

## 相关符号生成方法

- **REMI / Pop Music Transformer**（Huang & Yang, 2020）用 `Bar`、`Position`、`Tempo` 等事件显式表示小节、拍内位置和速度，再生成音符。它支持把 metrical structure 明确放进序列；REMI 的 `Position` 是小节内量化位置，本设计的 `u` 则表达对齐演奏拍点后的连续、可带符号偏移。
- **Compound Word Transformer**（Hsiao et al., 2021）将相关符号字段组合为 compound event，并建模小节结构。它支持在保留结构上下文的同时组合音符属性，但本身并未提出这里的带符号演奏起音比例。
- **ScorePerformer / SPMuple** 使用 score-performance 对齐的音符 tuple，并包含 beat-position 上下文，是 EPR 中较近的先例。本设计把这种 beat-position 上下文显式编码为输入 beat token，再让 note decoder 预测逐音符拍内起音比例。

这些工作支持“显式表示 beat 结构”的设计动机，但不能证明本方案已经有效。需要实验检验的假设是：把 score beat 结构作为输入上下文，是否能提升 note-level 演奏属性的建模并简化起音目标学习。

## 与当前 IOI 的关系及实验计划

当前 IOI 以相邻音符或起音组为参照；新方案以 score beat 网格作为显式输入，同时预测 beat-level performance duration 和 note 相对 beat 的起音位置。它让同拍和弦使用统一坐标，并显式建模 performance tempo curve，但不保证 HR、Wasserstein 或听感一定变好。

1. **标签审计：** 复用现有 score-to-performance beat 映射，检查两级比例的分布、异常样例及不同演奏间的稳定性。
2. **匹配对照：** 当前 timestep 模型与两级 beat 模型使用相同 split、初始化策略、训练预算和采样种子。
3. **消融：** 无 beat token；加入 beat token；加入或移除 `measure_start`；beat/note duration 使用 shared 或 separate 输入词表；absolute-log 与 log-dev 输出目标。
4. **评估：** 报告 valid loss、beat duration 误差、signed onset 误差、note duration 误差、还原后的 token-level timing JS、PP/PN 和 HR；同时检查 teacher-forced 与 autoregressive 输出。
5. **重建检查：** 验证 score beat 到 note 的归属、note 绝对起音、跨拍顺序、音符对齐、拍号变化、弱起、休止，以及 `u<0` 或 `u>1` 的音符。

不能只凭 valid loss 选型；此前实验已观察到 loss 与 HR 不总是同向。首要检验点是两级时间表示是否更清晰且更易学习，质量结论仍需多随机种子及 held-out 指标支持。
