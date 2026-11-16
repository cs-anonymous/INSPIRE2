# exp1112 DINR abs5000 与坐标表示对照

本实验独立于 exp1111，PT base-1 与 PT scratch 沿用 exp1111 第7节的既有结果，不在本轮重跑，也不计入本轮24个 DINR 任务。
本轮 24 个 DINR 任务全部从随机初始化开始：不加载 PT rendering checkpoint，backbone、pitch/velocity/timing lookup、coordinate 映射和输出头均按模型初始化规则生成。非-coordinate lookup 任务采用 timing 5000 bins、1ms/bin、0–4999ms；共同协议为 held-out-12-work、seed42、effective batch32、LR3e-4 linear decay、最多16 epoch、每半 epoch validation、patience4、best-valid-only checkpoint。

纯 coordinate 组属于 DINR separate：IOI、duration、velocity 分别使用独立的 MLP/linear/Fourier 映射替代数值 lookup，保留 pitch lookup；timing 输出仍使用5000类目标。表格 method 统一写为 DINR separate，time表示中注明 pure coordinate 和映射类型。原四项 coordinate 实验使用 timestep 1+pitch；新增 MLP32、MLP64、Linear 三项改用 timestep，其他配置与各自原实验一致，均重新随机初始化训练。

后续实验固定使用 timestep mask，即 decoder_mask_mode=note、decoder_mask_probability=0.5：训练时以50%概率将一个历史 decoder note 的 pitch、IOI、velocity、duration 整体替换为一个 mask 向量。后续不再使用 attribute 或 timestep 1+pitch/3+pitch；本表保留既有实验的实际配置。

DINR 端到端训练与 HR 评估已完成 24/24 项；新增 E4D3 与批次对照直接评估19首 test work × 3 samples。

三轮采样评估（18项 DINR + 2项 PT）已完成 20/20 项。复用各自已保存的最佳 checkpoint，保留原 sample 0、1，新增 sample 2，每项共19首曲目×3个样例。每轮先对19首曲目取平均，再报告三轮 HR 的 mean ± std（n=3，样本标准差 ddof=1），统一保留3位小数。这衡量固定模型的采样波动，不包含重复训练的波动。尚未完成三轮评估的指标显示 —。 PT scratch 沿用共同 note-index 范围评价规则；加入第三样例后，5首曲目的共同前缀合计缩短10个音符，三轮均在该共同范围重算。覆盖变化与原双样例校验值见 hr_mean_std.json，原始结果不覆盖。

## PT 参考结果

| method | 训练mask | time表示 | IOI HR | Dur HR | Vel HR | PN HR |
|---|---|---|---:|---:|---:|---:|
| PT base-1 | non-pitch token | abs 5000 token | 2.038 ± 0.042 | 2.054 ± 0.010 | 1.586 ± 0.040 | 1.892 ± 0.023 |
| PT scratch | non-pitch token | abs 5000 token | 10.251 ± 0.065 | 83.443 ± 1.538 | 3.603 ± 0.001 | 32.432 ± 0.507 |

## DINR 任务

| method | 训练mask | time表示 | IOI HR | Dur HR | Vel HR | PN HR |
|---|---|---|---:|---:|---:|---:|
| DINR separate | attribute | abs 5000 token | 3.964 ± 0.056 | 2.480 ± 0.081 | 2.676 ± 0.024 | 3.040 ± 0.051 |
| DINR separate | none | abs 5000 token | 3.618 ± 0.014 | 2.406 ± 0.005 | 2.311 ± 0.036 | 2.778 ± 0.008 |
| DINR separate | timestep | abs 5000 token | 3.546 ± 0.015 | 2.374 ± 0.006 | 2.554 ± 0.034 | 2.825 ± 0.010 |
| DINR separate | timestep 1+pitch | abs 5000 token | 3.818 ± 0.066 | 2.442 ± 0.018 | 2.830 ± 0.019 | 3.030 ± 0.029 |
| DINR shared | attribute | abs 5000 token | 4.347 ± 0.022 | 2.525 ± 0.048 | 2.833 ± 0.088 | 3.235 ± 0.008 |
| DINR shared | timestep 1+pitch | abs 5000 token | 4.021 ± 0.158 | 2.439 ± 0.059 | 2.994 ± 0.075 | 3.151 ± 0.064 |
| DINR shared | none | abs 5000 token | 3.718 ± 0.035 | 2.400 ± 0.013 | 2.305 ± 0.023 | 2.808 ± 0.013 |
| DINR shared | timestep | abs 5000 token | 3.609 ± 0.014 | 2.409 ± 0.003 | 2.169 ± 0.012 | 2.729 ± 0.005 |
| DINR separate | timestep 1+pitch | log abs | 5.218 ± 0.159 | 2.882 ± 0.085 | 2.930 ± 0.047 | 3.677 ± 0.067 |
| DINR separate | timestep 1+pitch | log dev | 3.051 ± 0.025 | 3.269 ± 0.039 | 2.309 ± 0.010 | 2.876 ± 0.019 |
| DINR separate | timestep | log dev | 2.799 ± 0.008 | 3.046 ± 0.044 | 2.401 ± 0.002 | 2.749 ± 0.013 |
| DINR separate | timestep 1+pitch | pure coordinate MLP32 | 3.554 ± 0.003 | 2.423 ± 0.004 | 1.941 ± 0.014 | 2.639 ± 0.004 |
| DINR separate | timestep 1+pitch | pure coordinate MLP64 | 3.598 ± 0.003 | 2.287 ± 0.005 | 1.518 ± 0.022 | 2.468 ± 0.008 |
| DINR separate | timestep 1+pitch | pure coordinate Linear | 3.541 ± 0.003 | 2.404 ± 0.001 | 1.609 ± 0.033 | 2.518 ± 0.010 |
| DINR separate | timestep 1+pitch | pure coordinate Fourier | 4.082 ± 0.044 | 2.473 ± 0.007 | 2.427 ± 0.078 | 2.994 ± 0.012 |
| DINR separate | timestep | pure coordinate MLP32 | 3.499 ± 0.014 | 2.369 ± 0.008 | 1.925 ± 0.058 | 2.598 ± 0.014 |
| DINR separate | timestep | pure coordinate MLP64 | 3.512 ± 0.018 | 2.389 ± 0.008 | 2.268 ± 0.091 | 2.723 ± 0.022 |
| DINR separate | timestep | pure coordinate Linear | 3.664 ± 0.006 | 2.434 ± 0.008 | 2.264 ± 0.032 | 2.787 ± 0.010 |
| DINR separate | timestep E4D3 | abs 5000 token | 3.543 ± 0.025 | 2.397 ± 0.002 | 2.255 ± 0.043 | 2.732 ± 0.006 |
| DINR separate | timestep E4D3 | pure coordinate MLP32 | 3.648 ± 0.007 | 2.564 ± 0.017 | 1.692 ± 0.004 | 2.635 ± 0.006 |
| DINR separate | timestep E4D3 | pure coordinate MLP64 | 3.496 ± 0.027 | 2.529 ± 0.021 | 1.770 ± 0.078 | 2.599 ± 0.028 |
| DINR separate | timestep E4D3 | pure coordinate Linear | 3.566 ± 0.020 | 2.397 ± 0.018 | 1.570 ± 0.088 | 2.511 ± 0.016 |
| DINR separate | timestep (batch seed43) | pure coordinate MLP32 | 3.669 ± 0.019 | 2.462 ± 0.006 | 1.929 ± 0.106 | 2.686 ± 0.027 |
| DINR separate | timestep (batch seed43) | pure coordinate MLP64 | 3.848 ± 0.013 | 2.472 ± 0.004 | 1.472 ± 0.007 | 2.597 ± 0.004 |

## 训练记录（单次训练）

每行对应一次训练；Best epoch、Best valid 和验证次数来自该次训练，不构造训练标准差。E4D3 与批次对照为新增训练，既有实验沿用原 checkpoint。

| method | 训练mask | time表示 | Best epoch | Best valid | Stop epoch | Valid 次数 |
|---|---|---|---:|---:|---:|---:|
| PT base-1 | non-pitch token | abs 5000 token | 1.000 | 1.628 | 3.000 | 6 |
| PT scratch | non-pitch token | abs 5000 token | 1.000 | 3.046 | 3.000 | 6 |
| DINR separate | attribute | abs 5000 token | 5.000 | 18.380 | 7.000 | 14 |
| DINR separate | none | abs 5000 token | 2.000 | 18.449 | 4.000 | 8 |
| DINR separate | timestep | abs 5000 token | 5.000 | 18.452 | 7.000 | 14 |
| DINR separate | timestep 1+pitch | abs 5000 token | 4.000 | 18.393 | 6.000 | 12 |
| DINR shared | attribute | abs 5000 token | 5.000 | 18.375 | 7.000 | 14 |
| DINR shared | timestep 1+pitch | abs 5000 token | 5.000 | 18.398 | 7.000 | 14 |
| DINR shared | none | abs 5000 token | 2.000 | 18.448 | 4.000 | 8 |
| DINR shared | timestep | abs 5000 token | 4.500 | 18.519 | 6.500 | 13 |
| DINR separate | timestep 1+pitch | log abs | 3.500 | 18.328 | 5.500 | 11 |
| DINR separate | timestep 1+pitch | log dev | 4.500 | 20.525 | 6.500 | 13 |
| DINR separate | timestep | log dev | 6.000 | 20.681 | 8.000 | 16 |
| DINR separate | timestep 1+pitch | pure coordinate MLP32 | 6.000 | 18.395 | 8.000 | 16 |
| DINR separate | timestep 1+pitch | pure coordinate MLP64 | 7.000 | 18.351 | 9.000 | 18 |
| DINR separate | timestep 1+pitch | pure coordinate Linear | 7.500 | 18.260 | 9.500 | 19 |
| DINR separate | timestep 1+pitch | pure coordinate Fourier | 3.500 | 18.100 | 5.500 | 11 |
| DINR separate | timestep | pure coordinate MLP32 | 10.000 | 18.233 | 12.000 | 24 |
| DINR separate | timestep | pure coordinate MLP64 | 8.500 | 18.202 | 10.500 | 21 |
| DINR separate | timestep | pure coordinate Linear | 15.500 | 18.181 | 16.000 | 32 |
| DINR separate | timestep E4D3 | abs 5000 token | 5.000 | 18.534 | 7.000 | 14 |
| DINR separate | timestep E4D3 | pure coordinate MLP32 | 10.500 | 18.050 | 12.500 | 25 |
| DINR separate | timestep E4D3 | pure coordinate MLP64 | 11.500 | 17.926 | 13.500 | 27 |
| DINR separate | timestep E4D3 | pure coordinate Linear | 13.000 | 18.054 | 15.000 | 30 |
| DINR separate | timestep (batch seed43) | pure coordinate MLP32 | 8.500 | 18.259 | 10.500 | 21 |
| DINR separate | timestep (batch seed43) | pure coordinate MLP64 | 12.000 | 18.150 | 14.000 | 28 |

## Coordinate 与 mask 对照审计

已对新旧 MLP32、MLP64、Linear 六项执行权重逐张量核对、实际 mask 输入检查、完整验证集分头 CE 重算，以及342份生成记录的 checkpoint 来源核对，未发现这两项新实验加载错模型或遗漏 coordinate 的问题。
MLP64 与 Linear 的生成 HR 退化主要来自 velocity；三轮采样不能替代多个训练 seed 的验证。详细证据与解释限制见 [coordinate/mask 审计](../analysis/exp1112/coordinate_mask_audit.md)。

## timestep E4D3 与训练批次对照

E4D3 保持 encoder 的 pitch、IOI、velocity、duration 四项输入；decoder 所有位置只接收 IOI、velocity、duration 三项（包括三项 BOS），训练及自回归生成均不输入 pitch。三个字段分别映射并相加；50%的历史 decoder note 整体替换为一个 mask 向量。生成 MIDI 的 pitch 仍复制乐谱，不预测 pitch。lookup、MLP32、MLP64、Linear 均为 DINR separate、随机初始化，沿用16 epoch、半 epoch valid、patience4、LR3e-4协议。

两项额外 MLP32/64 重训保持初始化及训练随机 seed42，仅把数据批次 seed 从42改为43；batch size32和数据划分不变。这检验批次顺序的影响，不是独立初始化的多 seed 实验。

新六项各使用最佳 valid checkpoint 采样19首曲目×3轮，HR 报告三轮 mean ± std（3位小数）；不将不同训练模型的结果混作采样标准差。训练目录的 training_randomness_audit.json 记录初始权重指纹、实际 sampler seed 和首批索引。

本组训练已完成 6/6 项；三轮采样与 HR 评估已完成 6/6 项（正在评估 0 项，训练结束待评估 0 项）。

下面两列分别展示 E4D3 与训练批次对照，上排 train、下排 valid；附各自原 timestep 结果作为对照，横轴按实际 epoch 自适应。

![E4D3 and training batch order train/valid curves](../analysis/exp1112/e4d3_batch_groups.png)

[新实验曲线 PDF](../analysis/exp1112/e4d3_batch_groups.pdf)

### 六项新增实验的结果对照

下表均为 PN HR（越低越好），mean ± std 仅对应同一模型的三轮采样。

| 表示 | 原 timestep（批次seed42） | timestep E4D3（批次seed42） | 原 timestep（批次seed43） |
|---|---:|---:|---:|
| lookup | 2.825 ± 0.010 | 2.732 ± 0.006 | — |
| MLP32 | 2.598 ± 0.014 | 2.635 ± 0.006 | 2.686 ± 0.027 |
| MLP64 | 2.723 ± 0.022 | 2.599 ± 0.028 | 2.597 ± 0.004 |
| Linear | 2.787 ± 0.010 | 2.511 ± 0.016 | — |

E4D3 改善 lookup、MLP64、Linear 的 PN HR，MLP32 略差，尚未显示统一优势。本组 E4D3 Linear 的 PN HR 最低（2.511 ± 0.016）。MLP64 和 Linear 的改善主要来自 velocity，分别从2.268降至1.770、从2.264降至1.570；MLP32 的 velocity 也改善，但 IOI/duration 变差，抵消了力度收益。

仅改变批次顺序，原 timestep MLP32 从2.598 ± 0.014变为2.686 ± 0.027，MLP64 从2.723 ± 0.022变为2.597 ± 0.004，二者排名反转。差异大于各自三轮采样的标准差，说明至少存在明显的训练顺序敏感性，不能把此前差异全部归因于生成采样。当前只有两个批次seed，仍不足以确立宽度或mask的稳定排名。

更低 valid CE 仍不保证更低生成 HR：E4D3 MLP32 的 valid 从18.233降至18.050，PN HR反而从2.598升至2.635；E4D3 MLP64 的 valid 为17.926，低于Linear的18.054，但PN HR为2.599，高于Linear的2.511。总CE选择checkpoint与自回归生成质量的差异仍然存在。

## 曲线和数据

原四组 train/valid 曲线（4列×2行）：上排为逐 optimizer step 的 train loss，下排为每半个 epoch 的 valid loss；横轴按各组实际最大 epoch 自适应。E4D3 与训练批次对照单独绘制于上方。

![exp1112 DINR grouped train and validation loss](../analysis/exp1112/dinr_groups.png)

[PDF](../analysis/exp1112/dinr_groups.pdf)

[逐 step 与半 epoch loss CSV](../analysis/exp1112/all_losses.csv) · [原始双样例实验汇总 CSV](../analysis/exp1112/all_experiments.csv)

[三轮采样 mean/std 汇总 CSV](../analysis/exp1112/sampling3_summary.csv) · [每轮原始指标 CSV](../analysis/exp1112/sampling3_rounds.csv)

[第三样例队列、生成结果及 mean/std JSON](../../results/exp1112_sampling3/)

运行状态、命令、日志、表示审计、checkpoint、测试 MIDI 与 HR JSON：[exp1112 运行目录](../../results/exp1112_dinr_abs5000/)

预计算文件使用 held-out-12-work manifest，timing bins=5000、bin width=1ms；训练和验证窗口与 exp1111 使用相同划分。

## MLP64：逐 epoch 的 loss、TF HR 与自回归 HR

配置为 DINR separate、timestep 1+pitch、pure coordinate MLP64；随机初始化 seed42、批次seed42，沿用LR3e-4线性衰减、最多16 epoch、每半epoch验证、early stopping patience4。此诊断保留所有半epoch验证点的模型权重；评估完整epoch及最终半epoch（如有），不按测试HR选择模型或停止训练。

Train loss 是该epoch内各训练batch原始loss按样本数加权的平均值（训练mask开启），不是从训练开始累计的平均；半epoch终点对应当前epoch已完成部分。Valid loss来自固定完整验证集（mask关闭）。

Test TF HR：真实演奏严格右移一位作为历史，当前音符由模型采样；每首测试曲目的所有test演奏分别作为历史，先在曲目内平均，再等权平均19首曲目。Test HR：历史使用模型生成的数值。两者均对同一组人类参考计算原有note-index W1/固定human anchor HR，temperature1、top-p0.9；各做3轮采样，报告mean ± sample std（3位小数）。TF保留完整历史，分块只控制logits内存；生成MIDI的pitch均复制乐谱。

训练状态：completed；已有 8 个epoch评估点，TF与自回归均完成 8 个。

| Epoch | Train loss（本epoch平均） | Valid loss | Test TF PN HR | Test AR PN HR |
|---:|---:|---:|---:|---:|
| 1.000 | 19.232 | 18.887 | 2.069 ± 0.003 | 2.480 ± 0.003 |
| 2.000 | 18.231 | 18.633 | 2.091 ± 0.003 | 2.567 ± 0.002 |
| 3.000 | 17.916 | 18.507 | 2.062 ± 0.002 | 2.602 ± 0.007 |
| 4.000 | 17.711 | 18.415 | 2.046 ± 0.004 | 2.668 ± 0.007 |
| 5.000 | 17.544 | 18.360 | 2.025 ± 0.004 | 2.610 ± 0.007 |
| 6.000 | 17.401 | 18.285 | 2.064 ± 0.002 | 2.563 ± 0.007 |
| 7.000 | 17.279 | 18.313 | 2.052 ± 0.005 | 2.562 ± 0.004 |
| 8.000 | 17.149 | 18.289 | 2.028 ± 0.003 | 2.557 ± 0.005 |

截至已完成的 8 个共同评估点，自回归PN HR最低出现在epoch 1（2.480 ± 0.003）。从epoch 1到8，valid loss从18.887变为18.289，TF PN HR从2.069变为2.028，AR PN HR从2.480变为2.557；AR−TF差值从0.411变为0.530。

CE与TF HR之间的关系用于检查分类目标和数值HR是否一致；TF与AR HR的差距用于观察真实历史和生成历史条件下的差异。差距本身不能证明唯一原因。未完成的epoch不能纳入最终排名，以上只是当前训练轨迹的描述。

完整评估已结束：最低valid位于epoch 6（18.285，对应AR HR 2.563 ± 0.007）；最低TF HR位于epoch 5（2.025 ± 0.004）；最低AR HR位于epoch 1。本次8个epoch的valid loss与AR HR的Spearman相关系数为0.048，接近零。

从epoch 1到8，AR IOI HR从3.410变为3.762，AR velocity HR从1.652变为1.518；后期力度改善被时间表现的恶化抵消，不能概括为所有属性都越训练越差。本轮结果支持在独立验证集上诊断自回归HR与选模指标的关系；测试集轨迹只用于本次分析，不据此选择最终模型。

![Epoch train/valid loss and test TF/AR HR](../analysis/exp1112/epoch_diagnostic/epoch_metrics.png)

![Loss versus TF/AR HR](../analysis/exp1112/epoch_diagnostic/loss_hr_relationships.png)

散点颜色与标注表示epoch，相关系数只描述本次训练轨迹，不代表独立重复训练的统计结论。

[逐epoch与分属性指标CSV](../analysis/exp1112/epoch_diagnostic/epoch_metrics.csv) · [关系图PDF](../analysis/exp1112/epoch_diagnostic/loss_hr_relationships.pdf) · [Checkpoint及评估队列](../../results/exp1112_mlp64_epoch_diagnostic/manifest.json)

## 逐 epoch 补充 rebuttal 指标：KL/JS 与逐拍相关

复用8个epoch的已保存MIDI，未重新训练或采样。按 revision/rebuttal-short.md 的A2/C4及其原实现补充：IOI/duration/velocity的PP KL、JS，以及beat tempo/velocity Pearson。时间使用score-relative log deviation，256 bins；零score IOI使用[0,5]，其余时间[-2,1]；velocity128 bins，[-0.5,127.5]；自然对数、伪计数1e-6、端点截断。KL方向为generated || human。引用原rebuttal已按score onset span归一化的人类参考MIDI，预测不额外缩放。

主表每轮单独计算PP指标，先在TF历史内平均，再对19首曲目等权平均；三轮报告mean ± std。原rebuttal把三个生成样例先合并再计算PP直方图，本次也原样补算，另存pooled3_pp.csv；该合并估计只有一个值，不虚构采样标准差。逐拍Pearson复用原A2实现；人类基准复现为tempo 0.303、velocity 0.760。

这里的mean ± std只衡量固定checkpoint的三轮采样变动，不是三次独立训练。所有8个epoch、两种生成模式、三轮采样的八项指标均覆盖19首曲目，TF的全部历史也都有有限值。

### 实测结论

替换指标没有消除loss与生成质量的不一致。ep1→最佳valid的ep6，valid从18.887降到18.285，但AR IOI JS从0.145升到0.192；AR duration JS从0.192降到0.164、velocity JS从0.102降到0.092，说明不同属性方向不一致。到ep8，AR三项JS分别为0.202、0.216、0.137，均高于ep1；总loss不能代替逐属性诊断。

AR tempo Pearson随loss总体改善，Spearman ρ=-0.833，方向符合预期；但其绝对值从ep1的-0.047到ep6的-0.034，仍接近零且为负，不能据此称为良好的节奏一致性。TF velocity Pearson在ep1/6/8分别为0.732/0.733/0.717，AR对应为-0.035/-0.025/-0.010。这提示使用真实历史与生成历史存在显著差距；尚不能仅凭这些曲线确证具体实现错误或唯一原因。

KL/JS越低越好；Pearson越高越好。和valid loss的相关系数对KL/JS期望为正、对Pearson期望为负；这只是一次训练中8个相关联epoch的描述，不能视为跨seed的稳定相关性，未按相关性筛选或删除指标。

| 指标 | ρ(valid loss, TF指标) | ρ(valid loss, AR指标) | 期望方向 |
|---|---:|---:|---|
| ioi_kl | -0.333 | -0.952 | 正 |
| ioi_js | -0.667 | -0.905 | 正 |
| duration_kl | 0.500 | 0.000 | 正 |
| duration_js | 0.619 | -0.048 | 正 |
| velocity_kl | -0.286 | 0.405 | 正 |
| velocity_js | -0.286 | 0.476 | 正 |
| tempo_pearson | 0.405 | -0.833 | 负 |
| velocity_pearson | 0.500 | -0.119 | 负 |

### 固定原HR参考的对照

rebuttal人类参考按score onset span归一化，原exp1112 HR参考使用本轮评估导出的MIDI；两者的时间尺度不同，因此以上绝对值不能直接归因于指标替换。另补算一版，PP KL/JS使用与原HR完全相同的参考MIDI，其他定义不变。逐拍Pearson仍使用A2原实现的人类曲线。

| 指标 | 原HR参考：ρ(valid, TF) | 原HR参考：ρ(valid, AR) |
|---|---:|---:|
| ioi_kl | -0.286 | -0.952 |
| ioi_js | -0.476 | -0.905 |
| duration_kl | 0.619 | 0.000 |
| duration_js | 0.571 | -0.048 |
| velocity_kl | -0.286 | 0.405 |
| velocity_js | -0.286 | 0.476 |
| tempo_pearson | 0.405 | -0.833 |
| velocity_pearson | 0.500 | -0.119 |

原HR参考下AR IOI JS在ep1/6/8为0.157/0.206/0.219，ρ仍为-0.905。两套参考下AR三项JS的相关系数完全一致；归一化改变绝对值，但没有解释掉当前的趋势冲突。按原rebuttal把三轮样例先合并的pooled3估计也保留AR IOI JS的ρ=-0.905、duration JS的ρ=-0.048、velocity JS的ρ=0.476，结论不依赖先合并还是先计算。

[原HR参考对照CSV](../analysis/exp1112/epoch_diagnostic/rebuttal_metrics_current_refs/epoch_metrics.csv) · [原HR参考对照JSON](../analysis/exp1112/epoch_diagnostic/rebuttal_metrics_current_refs/summary.json)

自回归的逐epoch补充指标（KL另见CSV与图）：

| Epoch | Valid loss | IOI JS ↓ | Dur JS ↓ | Vel JS ↓ | Tempo Pearson ↑ | Velocity Pearson ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 18.887 | 0.145 ± 0.001 | 0.192 ± 0.001 | 0.102 ± 0.002 | -0.047 ± 0.001 | -0.035 ± 0.021 |
| 2.000 | 18.633 | 0.165 ± 0.001 | 0.203 ± 0.001 | 0.200 ± 0.002 | -0.061 ± 0.002 | -0.016 ± 0.016 |
| 3.000 | 18.507 | 0.161 ± 0.000 | 0.256 ± 0.002 | 0.172 ± 0.002 | -0.057 ± 0.011 | 0.012 ± 0.017 |
| 4.000 | 18.415 | 0.174 ± 0.001 | 0.252 ± 0.002 | 0.184 ± 0.006 | -0.050 ± 0.009 | -0.003 ± 0.013 |
| 5.000 | 18.360 | 0.176 ± 0.002 | 0.210 ± 0.006 | 0.156 ± 0.001 | -0.038 ± 0.013 | 0.008 ± 0.033 |
| 6.000 | 18.285 | 0.192 ± 0.003 | 0.164 ± 0.003 | 0.092 ± 0.002 | -0.034 ± 0.002 | -0.025 ± 0.007 |
| 7.000 | 18.313 | 0.192 ± 0.001 | 0.321 ± 0.002 | 0.140 ± 0.005 | -0.041 ± 0.007 | 0.012 ± 0.023 |
| 8.000 | 18.289 | 0.202 ± 0.003 | 0.216 ± 0.001 | 0.137 ± 0.004 | -0.037 ± 0.006 | -0.010 ± 0.012 |

TF的逐epoch补充指标（KL另见CSV与图）：

| Epoch | Valid loss | IOI JS ↓ | Dur JS ↓ | Vel JS ↓ | Tempo Pearson ↑ | Velocity Pearson ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 18.887 | 0.103 ± 0.001 | 0.062 ± 0.001 | 0.028 ± 0.000 | -0.006 ± 0.014 | 0.732 ± 0.002 |
| 2.000 | 18.633 | 0.108 ± 0.001 | 0.054 ± 0.000 | 0.020 ± 0.000 | -0.002 ± 0.004 | 0.743 ± 0.001 |
| 3.000 | 18.507 | 0.103 ± 0.001 | 0.055 ± 0.000 | 0.017 ± 0.000 | 0.013 ± 0.000 | 0.749 ± 0.004 |
| 4.000 | 18.415 | 0.109 ± 0.001 | 0.053 ± 0.000 | 0.019 ± 0.000 | 0.009 ± 0.010 | 0.740 ± 0.003 |
| 5.000 | 18.360 | 0.106 ± 0.001 | 0.050 ± 0.000 | 0.021 ± 0.000 | 0.014 ± 0.028 | 0.738 ± 0.001 |
| 6.000 | 18.285 | 0.121 ± 0.000 | 0.051 ± 0.001 | 0.022 ± 0.000 | -0.011 ± 0.002 | 0.733 ± 0.003 |
| 7.000 | 18.313 | 0.119 ± 0.001 | 0.056 ± 0.001 | 0.029 ± 0.000 | -0.006 ± 0.004 | 0.725 ± 0.002 |
| 8.000 | 18.289 | 0.106 ± 0.000 | 0.051 ± 0.000 | 0.024 ± 0.000 | -0.012 ± 0.004 | 0.717 ± 0.002 |

### 指标解释与后续选择

当前PN HR每轮是单个生成值与人类参考集合的W1，即平均绝对距离；三轮再取平均。它会偏好靠近人类中位数的输出，不直接奖励恢复完整人类分布。这解释了它和似然目标可能不一致，但不能证明模型的生成质量已经很好。PP直方图KL/JS和逐拍相关衡量不同方面，也不保证与交叉熵同向；不能仅因某指标更符合loss趋势就认定它更可靠。

CE在固定数据及真实历史条件下对应data || model的条件KL（相差固定的数据熵）；C4计算generated || human的曲目边缘直方图KL，方向、条件和离散化均不同。CE降低不保证采样误差、top-p后的分布、自由生成轨迹或另一个数据划分上的任何单项指标同步改善。总CE下降也不保证每个属性CE都下降。

固定区间会截断超范围值。本次AR预测中，IOI截断比例ep1/6/8为4.565%/9.447%/9.019%，duration为25.675%/20.993%/25.350%；velocity为0。这些比例按预测值计数汇总，含三轮（TF时还含各历史），不是按曲目等权。KL/JS会把尾部差异折叠到边界，不能单独评价全部时间误差；逐行截断比例保存在summary.json。

建议保留valid分头CE作为概率拟合与checkpoint选择依据，同时固定报告AR的三项JS与tempo/velocity Pearson，保留HR作为距离补充，避免只用总分掩盖属性冲突。若增加考虑数值顺序的验证指标，可在原始输出概率上计算CRPS/RPS；它直接评估条件CDF与真实值的距离，无需生成采样，也无需把三轮合并为稀疏直方图，但与CE仍不保证同步。当前test逐epoch结果只作诊断，不据此反复选择最佳epoch或调整指标；生成指标用于选模时应在valid集预先固定协议。

![Rebuttal metrics across epochs](../analysis/exp1112/epoch_diagnostic/rebuttal_metrics/epoch_metrics.png)

![Validation loss versus rebuttal metrics](../analysis/exp1112/epoch_diagnostic/rebuttal_metrics/valid_loss_relations.png)

[全部逐轮均值与标准差CSV](../analysis/exp1112/epoch_diagnostic/rebuttal_metrics/epoch_metrics.csv) · [原rebuttal三样例合并PP估计](../analysis/exp1112/epoch_diagnostic/rebuttal_metrics/pooled3_pp.csv) · [完整定义、覆盖与结果JSON](../analysis/exp1112/epoch_diagnostic/rebuttal_metrics/summary.json)

## 直接从 MIDI 计算与 token 同粒度的 JS

直接复用原8个epoch的TF/AR归档MIDI和原HR人类参考，不重新采样、不改变归一化。时间按1ms取整并截到0–4999ms（5000 bins），velocity按0–127（128 bins）。这里IOI是最终MIDI的相邻音符间隔，区别于转换前训练目标中的grouped interval；这版用于评价最终输出。JS为自然对数divergence，不加伪计数，不取平方根。每轮先在TF历史内平均，再对19首曲目等权平均，三轮mean ± std只表示固定模型的采样变动。

自回归：

| Epoch | Valid loss | IOI JS ↓ | Duration JS ↓ | Velocity JS ↓ |
|---:|---:|---:|---:|---:|
| 1.000 | 18.887 | 0.151 ± 0.001 | 0.274 ± 0.000 | 0.102 ± 0.002 |
| 2.000 | 18.633 | 0.170 ± 0.001 | 0.283 ± 0.001 | 0.200 ± 0.002 |
| 3.000 | 18.507 | 0.169 ± 0.000 | 0.342 ± 0.001 | 0.172 ± 0.002 |
| 4.000 | 18.415 | 0.179 ± 0.001 | 0.339 ± 0.003 | 0.184 ± 0.006 |
| 5.000 | 18.360 | 0.186 ± 0.002 | 0.287 ± 0.006 | 0.156 ± 0.001 |
| 6.000 | 18.285 | 0.199 ± 0.002 | 0.245 ± 0.004 | 0.092 ± 0.002 |
| 7.000 | 18.313 | 0.204 ± 0.001 | 0.425 ± 0.001 | 0.140 ± 0.005 |
| 8.000 | 18.289 | 0.210 ± 0.002 | 0.298 ± 0.001 | 0.137 ± 0.004 |

TF：

| Epoch | Valid loss | IOI JS ↓ | Duration JS ↓ | Velocity JS ↓ |
|---:|---:|---:|---:|---:|
| 1.000 | 18.887 | 0.098 ± 0.000 | 0.120 ± 0.000 | 0.028 ± 0.000 |
| 2.000 | 18.633 | 0.108 ± 0.001 | 0.109 ± 0.001 | 0.020 ± 0.000 |
| 3.000 | 18.507 | 0.109 ± 0.001 | 0.113 ± 0.001 | 0.017 ± 0.000 |
| 4.000 | 18.415 | 0.110 ± 0.001 | 0.108 ± 0.001 | 0.019 ± 0.000 |
| 5.000 | 18.360 | 0.107 ± 0.000 | 0.105 ± 0.000 | 0.021 ± 0.000 |
| 6.000 | 18.285 | 0.120 ± 0.000 | 0.109 ± 0.001 | 0.022 ± 0.000 |
| 7.000 | 18.313 | 0.125 ± 0.000 | 0.111 ± 0.000 | 0.029 ± 0.000 |
| 8.000 | 18.289 | 0.105 ± 0.000 | 0.109 ± 0.000 | 0.024 ± 0.000 |

| 指标 | ρ(valid, TF) | ρ(valid, AR) | 三轮合并 AR ρ |
|---|---:|---:|---:|
| ioi_js | -0.452 | -0.905 | -0.905 |
| duration_js | 0.405 | -0.048 | -0.048 |
| velocity_js | -0.286 | 0.476 | 0.476 |

两者越低越好，正相关为期望方向；仅为单次训练8个epoch的描述。

![Final MIDI JS at token granularity](../analysis/exp1112/epoch_diagnostic/midi_token_granularity_js/token_js.png)

[逐epoch CSV](../analysis/exp1112/epoch_diagnostic/midi_token_granularity_js/epoch_metrics.csv) · [三轮合并CSV](../analysis/exp1112/epoch_diagnostic/midi_token_granularity_js/pooled3.csv) · [定义和相关JSON](../analysis/exp1112/epoch_diagnostic/midi_token_granularity_js/summary.json)

## Grouped IOI 冲突与延后概率检查

输出规则保持不变，JS仍评价延后后的最终MIDI。模型在score IOI=0处预测相邻间隔，在score IOI>0处预测相对上一组起点的组间隔G。若当前组内累计时间A大于G，实际相邻间隔为max(G−A,0)，实际组间隔为max(G,A)，因此音符会延后A−G ms。分组本身不是实现错误；不可逆只发生在截断等导出处理影响到的值。

先用所有归档MIDI计算上界：在score IOI>0位置出现零相邻间隔的比例。由于G=A也会产生零间隔，这不是实际延后比例。排除首音符，每份MIDI均核对完整音符数量。先在TF历史内平均，再对19首曲目等权平均，报告三轮采样mean ± std。

| Epoch | TF 延后比例上界 | AR 延后比例上界 |
|---:|---:|---:|
| 1.000 | 0.449% ± 0.012% | 0.477% ± 0.017% |
| 2.000 | 0.939% ± 0.019% | 0.182% ± 0.005% |
| 3.000 | 0.850% ± 0.025% | 0.249% ± 0.021% |
| 4.000 | 1.923% ± 0.045% | 0.849% ± 0.100% |
| 5.000 | 1.640% ± 0.015% | 1.948% ± 0.059% |
| 6.000 | 2.903% ± 0.080% | 4.846% ± 0.142% |
| 7.000 | 2.428% ± 0.033% | 3.074% ± 0.065% |
| 8.000 | 2.008% ± 0.022% | 3.390% ± 0.059% |

严格的转换前token重放不是最终指标所必需的，因此不把它作为本次结果的前置条件。上述MIDI零间隔上界已经足以判断延后并非极低频事件；最终JS固定使用延后后的MIDI。

[上界完整JSON](../analysis/exp1112/epoch_diagnostic/grouped_delay_audit/summary.json)

### 当前 chord IOI 的建模语义审计

当前实现并不是所有 IOI 都相对于上一组起点：score IOI>0 使用 `grouped_ioi`，表示当前组起点相对上一组起点的间隔；score IOI=0 使用 `chord_ioi`，实际按序列中前一个音符的相邻间隔解码。score 音符来自 `(onset, pitch, end, velocity)` 排序，因此同时起音组的第一个序列音符总是最低音。19首测试曲目共检查16,098个多音符组、46,822个组内音符，16,098/16,098个组均为最低音先出现。

这说明当前设计隐含了“低音作为组锚点”和“组内按低到高传递间隔”两个假设；它们是序列化约定，不是音乐结构本身。更稳健的改法是两层表示：先预测组起点间隔，再对组内每个音符预测相对于组起点的非负 onset offset，并显式输入 group id、组内位置或声部/音域特征。这样组内 offset 不再依赖前一个音符的预测，也不会因低音先后顺序改变而累积误差。另一条更简单但成本更高的路线是每个音符直接预测绝对 onset 或 score-relative onset；它避免组内锚点，但需要处理更大的时间范围。当前实验的 JS 和 HR 不能证明哪种新表示更好，应该先做同 checkpoint、同 token budget 的 target ablation。
