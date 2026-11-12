# INSPIRE2 / PianistTransformer 实验总览

第 7 节记录了最新的半 epoch validation / early stopping 重跑结果。上一轮协议和结果仍保留用于历史对照。

本文记录当前 INSPIRE2 实验线的统一设置、训练曲线、checkpoint 选择、test 采样评估和最终结论。所有实验的 loss 都只能在相同目标、属性和数据划分内比较；PT 的 token-level CE 与 DINR 的属性级 loss 不具有直接可比的绝对数值。

## 1. 数据与评估协议

训练使用 `data/ASAP_processed` 及其预处理 sidecar。固定 validation 使用 held-out-12-work split，训练和 validation 使用 512-note window、50% overlap、global batch size 32、seed 42。PT 和 DINR 的 validation 都使用完整 teacher-forcing 历史，validation 不做随机 decoder mask。

Test 评估固定使用 19 首 work。每首 work 生成 2 个随机 sample，因此每个模型总计 38 个生成。推理顺序按 score note 数从大到小：10090、5988、5435、4170、4012、3958、3482、3322、3230、3140、2816、2630、2313、2027、1945、1866、1814、1508、1293 notes。

DINR 的 test HR 使用 IOI、duration、velocity 三个 PN Wasserstein 指标，分别除以 human leave-one-out anchor 后取平均。该 HR 定义是 **越低越好**，HR=1 表示达到 human anchor 的量级。详细 test 顺序见 [`test_order.json`](../analysis/dinr_six_model_hr_comparison/test_order.json)。

## 2. PT 训练线

### 2.1 PT rendering 初始化

该实验从已经经过 rendering SFT 的 `pianist-transformer-rendering` 初始化，不能作为 PT base 的首次 rendering SFT 结论。训练时保留 50% 的非 pitch decoder-history token，pitch 和 BOS 始终保留；validation 使用完整、无 mask 的 decoder history。

完整 8 epoch 结果：

| Epoch | Train loss | Valid loss |
|---:|---:|---:|
| 1 | 1.606221 | 1.607974 |
| 2 | 1.510266 | 1.622704 |
| 3 | 1.466891 | 1.659681 |
| 4 | 1.420756 | 1.709060 |
| 5 | 1.366159 | 1.813223 |
| 6 | 1.315059 | 1.911755 |
| 7 | 1.287821 | 1.949552 |
| 8 | 1.282481 | 1.958380 |

valid 最佳是 epoch 1；训练 loss 持续下降而 valid 从第 1 epoch 后持续上升，说明在这个小规模 held-out 诊断上已经出现明显过拟合。

### 2.2 PT base-1

PT base-1 从仅完成 pretraining、尚未进行 rendering SFT 的 `models/modelscope_pianist_transformer_base` 初始化。训练配置与上面一致：8 epoch、训练时 50% 非 pitch history mask、validation 无 mask。

| Epoch | Train loss | Valid loss |
|---:|---:|---:|
| 1 | 1.673908 | **1.630077** |
| 2 | 1.527575 | 1.636972 |
| 3 | 1.484384 | 1.667575 |
| 4 | 1.441169 | 1.717068 |
| 5 | 1.392494 | 1.800303 |
| 6 | 1.347691 | 1.893631 |
| 7 | 1.323197 | 1.928106 |
| 8 | 1.318306 | 1.936294 |

PT base-1 的最佳 checkpoint 是 **epoch 1**；更细的 first-epoch validation 中，step 560（约 epoch 0.8716）达到 1.627733。也就是说，PT base 上进行 rendering SFT 时，前期 valid 很快达到最低点，随后即使 train loss 继续下降，valid 也持续恶化。

最佳 checkpoint 的 test 采样使用单卡多进程评测：1 张 GPU、8 个 inference worker、19 首 work、每首 2 samples。结果写入 `results/pt_base_median_heldout12_8ep_mask50_evalnomask/test_sampling_2/`，对应的 MIDI manifest 和 Wasserstein 汇总将在该目录保存。

### 2.3 PT init train

PT init train 使用与 PT base-1 相同的 SFT 数据、mask、batch、学习率和 validation 协议，但 `pretrained_model=null`，即从随机初始化的 PT architecture 开始训练。最多训练 8 epoch；每个完整 epoch 后做一次无 mask validation，连续 3 个 epoch 的 valid loss 上升时停止。epoch 0 baseline 不计入连续上升次数，epoch 内不额外验证。启动配置为 [`pt_init_median_heldout12_8ep_earlystop.json`](../../configs/pt_init_median_heldout12_8ep_earlystop.json)，使用两张 GPU 的 DDP：

```bash
CUDA_VISIBLE_DEVICES=1,2 \
torchrun --standalone --nproc_per_node=2 -m src.train.sft \
  --config configs/pt_init_median_heldout12_8ep_earlystop.json
```

| Epoch | Train loss | Valid loss |
|---:|---:|---:|
| 0 (random init) | - | 7.680868 |
| 1 | 2.438339 | 2.533902 |
| 2 | 2.000288 | **2.525610** |
| 3 | 1.797044 | 2.733587 |
| 4 | 1.704975 | 2.919032 |
| 5 | 1.649091 | 2.997557 |

epoch 5 后连续 3 次 valid loss 上升而早停；best checkpoint 是 **epoch 2**，而不是 epoch 1。train loss 持续下降，valid 从 epoch 2 起连续恶化。训练曲线和 epoch loss 表见 [`loss_curves.png`](../../results/pt_init_median_heldout12_8ep_earlystop/analysis/loss_curves.png) 与 [`losses.csv`](../../results/pt_init_median_heldout12_8ep_earlystop/analysis/losses.csv)。

best checkpoint 使用双卡、每卡 4 workers 评测 19 首 test work × 2 samples，共 **38 个生成**，顺序从长到短。DINR 统一口径的分项 HR 为 IOI 7.0568、duration 17.3287、velocity 2.0844，PN HR **8.8233**，见 [`pn_hr_dinr_protocol.json`](../../results/pt_init_median_heldout12_8ep_earlystop/test_sampling_2gpu/pn_hr_dinr_protocol.json)。原 PT evaluator 的配对聚合另报 PN HR 7.1172，因算法不同不用于统一表。

## 3. DINR absolute 表示与 Mask

DINR 使用 note-level absolute timing 目标。无 pedal 诊断阶段输出 `[pitch, IOI, velocity, duration]`，IOI 和 duration 量化为 10 ms bins，velocity 量化为 MIDI 0–127。score encoder 不 mask；decoder performance history 只在训练时 mask，validation 和 test 不 mask。

修正后的 `attribute` / `timestep 3+pitch` 实验使用三个独立、可学习的属性 mask embedding，分别对应 IOI、velocity、duration；三者都以 `initializer_range`（约 0.02）正态初始化。mask 直接替换对应槽位的 embedding，不使用普通词表中的 MASK token ID 1。首个 BOS timestep 不参与 mask。

- **attribute mask**：三个属性各自独立以 0.5 概率替换为自己的属性专属 mask embedding；pitch 不 mask。
- **timestep 3+pitch**：每个 decoder timestep 以 0.5 概率同时 mask IOI、velocity、duration，并分别使用各自属性专属 embedding；pitch 保留。
- **timestep 1+pitch**：每个 decoder timestep 以 0.5 概率将 IOI、velocity、duration 的合计输入替换为一个 `decoder_mask_embedding`，再加回原 pitch embedding；mask vector 按 `initializer_range=0.02` 初始化。验证和测试不 mask。
- **no mask**：保留完整 decoder history，作为无正则对照。

旧版 **timestep**（总表单独列出）每个历史 timestep 以 0.5 概率将完整 note embedding 替换成一个 learned mask embedding，因此 pitch、IOI、velocity、duration 一起被 mask。旧版 timestep 的训练和 19 首 × 2 次 test 结果有效，保留为 legacy baseline。`timestep 3+pitch` 的 forward 输入可写成 `pitch_embedding + effective_mask`，其中三个向量在被 mask timestep 上总是一起相加；但三个独立参数的初始化、梯度更新和 Adam 状态不同于一个可学习向量。最佳 checkpoint 的三向量之和范数为 1.098（separate），本次 single-vector checkpoint 的 mask 范数为 0.714，legacy single-vector checkpoint 为 0.568。范数是训练后的结果，不能单独解释 HR 差异。

本次 `timestep 1+pitch` 与`timestep 3+pitch` 使用相同当前模型代码、seed、PT 初始化和训练方案，二者直接比较反映了 mask 参数化方式对结果的影响。相比之下，legacy timestep checkpoint 来自较早的模型实现；其 single-vector + pitch HR 对照有参考价值，但不是完全匹配实现的单变量因果实验。旧 attribute pilot 和第一次以 ID 1 初始化的 pitch-retaining pilot 不纳入最终结果。

六组 mask/time 重跑和本次 single-vector follow-up 均为 8 epoch、seed 42、batch 32、learning rate `3e-4`，从相同 PT rendering checkpoint 初始化。每 epoch 保存 checkpoint，以无 mask validation loss 选择最佳 checkpoint；test 也不 mask。

2026-11-12 补齐 `timestep 1+pitch` 的另外三组配置：GPU0 为 shared + abs 500 token，GPU1 为 separate + log abs，GPU2 为 separate + log dev。三组均沿用上述训练参数和 held-out-12-work split；训练后自动使用 best valid checkpoint 做 19 首 × 2 samples、每卡 8 workers 的无 mask 推理，并更新第 5 节统一表及 CSV。表中的 `—` 表示尚未完成，不作为实验结果。已有 separate + abs 500 token 结果复用。

```bash
bash script/run_dinr_single_pitch_missing_3gpu.sh 0 1 2
```

三组输出分别位于 `models/dinr_mask_revised_heldout12_8ep/` 下的 `shared_retain_pitch_single_vector_abs500/`、`separate_retain_pitch_single_vector_log_abs/`、`separate_retain_pitch_single_vector_log_dev/`。每个目录的 `status.txt` 记录阶段，`train.log` 和 `test_inference.log` 记录训练和测试。启动记录见 `models/dinr_mask_revised_heldout12_8ep/single_pitch_followup_20261112/launch.json` 与同目录 `launcher.log`。

## 4. Shared 与 separate timing table

Shared 方案使用同一组 500-bin timing value embedding，但 IOI 和 duration 在求和前经过独立 slot projection。Separate 方案为 IOI 和 duration 使用两张独立 `[500, 768]` token table，均从相同 PT timing embedding 初始化，之后独立接收梯度。

八 epoch validation 结果。`Time 表示` 区分 absolute、log absolute 和 score-relative log target；所有验证均使用无 mask 输入。

| Mask | Table | Time 表示 | Best epoch | Best valid | Epoch 8 valid |
|---|---|---|---:|---:|---:|
| attribute | separate | abs 500 token | 5 | 10.504103 | 10.815738 |
| attribute | shared | abs 500 token | 5 | 10.500670 | 10.843790 |
| timestep | separate | abs 500 token | 7 | 10.760382 | 10.777897 |
| timestep | shared | abs 500 token | 6 | 10.585932 | 10.633567 |
| no mask | separate | abs 500 token | 3 | 10.840699 | 11.834352 |
| no mask | shared | abs 500 token | 4 | **10.447963** | 11.870809 |
| timestep 3+pitch | separate | abs 500 token | 5 | 10.538136 | 10.926798 |
| timestep 3+pitch | shared | abs 500 token | 4 | 10.447859 | 10.840719 |
| timestep 3+pitch | separate | log abs | 4 | 10.402810 | 10.734770 |
| timestep 3+pitch | separate | log dev | 5 | 11.256263 | 11.440074 |
| timestep 1+pitch | separate | abs 500 token | 5 | 10.497801 | 10.830093 |
| timestep 1+pitch | shared | abs 500 token | — | — | — |
| timestep 1+pitch | separate | log abs | — | — | — |
| timestep 1+pitch | separate | log dev | — | — | — |

Validation loss 单独看并不能决定最终配置：它偏好 shared no-mask，但该模型 test HR 较高，说明 validation loss 与生成质量不完全一致。所有 mask 组的最佳 valid epoch 都早于 epoch 8。这里的 `timestep` 是旧版整-note mask；普通 `timestep 3+pitch` 使用三个属性 mask vector，`timestep 1+pitch` 使用一个 mask vector 并保留 pitch。

## 5. 所有实验的统一 test HR 大表

共 16 个实验配置，已完成 13 个，待完成 3 个：PT 两个实验固定放在前两行；DINR 包含 2 组 attribute、2 组 legacy timestep、2 组 no-mask、4 组 timestep 3+pitch 和 4 组 timestep 1+pitch。已完成的每行使用 **best valid epoch** checkpoint，在相同 19 首 test work × 2 samples（38 个生成）协议上评估，按 test work 从长到短推理。

表内 HR 都按 IOI、duration、velocity 的 W1 分别除以固定 human anchors（22.772456、75.085651、10.580359），再对三个分项取算术平均；越低越好。所有行采用同一 `evaluate_dinr_absolute.py` 的逐 note-index W1 算法。PT 的原始 `pn_hr.json` 使用不同的配对聚合，不可直接比较；PT 两行已用相同 DINR 算法对相同 38 个生成重算，结果保存在各自 test 目录的 `pn_hr_dinr_protocol.json`。

| method | 训练mask | time表示 | Best valid epoch | Best valid loss | IOI HR | Dur HR | Vel HR | PN HR |
|---|---|---|---:|---:|---:|---:|---:|---:|
| PT base-1 | non-pitch token | abs 5000 token | 0.871595 (step 560) | 1.627733 | 1.8847 | 2.0883 | 1.4504 | **1.8078** |
| PT scratch | non-pitch token | abs 5000 token | 2 | 2.525610 | 7.0568 | 17.3287 | 2.0844 | **8.8233** |
| DINR separate | attribute | abs 500 token | 5 | 10.504103 | 3.7085 | 2.3880 | 1.4958 | **2.5308** |
| DINR shared | attribute | abs 500 token | 5 | 10.500670 | 3.8063 | 2.8055 | 1.9103 | **2.8407** |
| DINR separate | timestep | abs 500 token | 7 | 10.760382 | 2.6905 | 2.2249 | 1.5618 | **2.1591** |
| DINR shared | timestep | abs 500 token | 6 | 10.585932 | 3.2274 | 2.4014 | 1.8376 | **2.4888** |
| DINR separate | none | abs 500 token | 3 | 10.840699 | 3.7024 | 2.3735 | 2.6160 | **2.8973** |
| DINR shared | none | abs 500 token | 4 | 10.447963 | 4.4274 | 2.8612 | 2.9568 | **3.4152** |
| DINR separate | timestep 3+pitch | abs 500 token | 5 | 10.538136 | 2.9221 | 2.2456 | 1.6431 | **2.2702** |
| DINR shared | timestep 3+pitch | abs 500 token | 4 | 10.447859 | 3.3086 | 2.5026 | 2.3302 | **2.7138** |
| DINR separate | timestep 3+pitch | log abs | 4 | 10.402810 | 2.8578 | 2.2296 | 2.5025 | **2.5300** |
| DINR separate | timestep 3+pitch | log dev | 5 | 11.256263 | 2.2562 | 2.3685 | 1.7990 | **2.1412** |
| DINR separate | timestep 1+pitch | abs 500 token | 5 | 10.497801 | 2.4576 | 2.1363 | 1.4816 | **2.0252** |
| DINR shared | timestep 1+pitch | abs 500 token | — | — | — | — | — | — |
| DINR separate | timestep 1+pitch | log abs | — | — | — | — | — | — |
| DINR separate | timestep 1+pitch | log dev | — | — | — | — | — | — |

`log dev` 使用 log absolute 输入和 score-relative log target；`log abs` 使用 log absolute 输入和 target。PT valid loss 是 token CE，DINR valid loss 是属性 head loss，量纲不同，不能横向比较。PT 使用 abs 5000 token；DINR 的 abs 500 token 是 500 类、每类 10 ms。统一 CSV 见 [`all_experiments.csv`](../analysis/dinr_six_model_hr_comparison/all_experiments.csv)。

旧版 `timestep` 两行使用整-note learned mask，包含 pitch，且按统一 HR 口径保留。absolute 500 下，`timestep 1+pitch` 的 PN HR 为 2.0252，低于 legacy separate timestep 的 2.1591；由于 legacy checkpoint 来自较早模型实现，这个差值不能完全归因于 pitch。更严格的当前实现对照是 `timestep 1+pitch`（2.0252）与`timestep 3+pitch`（2.2702）：同样保留 pitch 时，single-vector 低约 10.8%，说明 mask 参数化方式会影响训练结果。旧 attribute pilot 的 ID 1 mask 尺度异常，未并入当前 attribute 行。第一次 pitch-retaining pilot 的 ID 1 初始化结果（PN HR 约 4.55 / 7.43）不纳入表格。retain-pitch 各行排在 DINR 结果末尾。

## 6. 结论

1. **DINR 当前最低 PN HR 是 timestep 1+pitch + abs 500，为 2.0252。** log dev 为 2.1412；single-vector abs500 在 epoch 5 达到 best valid loss 10.4978。
2. **Mask 参数化方式会影响 retain-pitch 结果。** `timestep 1+pitch` 的 PN HR 为 2.0252，`timestep 3+pitch` 为 2.2702；两者使用相同当前实现、seed 和训练方案。三向量之和虽然在 forward 上等效于一个向量，但独立参数的优化轨迹并不等价于一个向量。
3. **与 legacy timestep 相比，当前 `timestep 1+pitch` 的 PN HR 更低（2.0252 vs. 2.1591）。** 方向上支持保留 pitch，但 legacy checkpoint 来自较早模型实现；该差异不是完全匹配实现的单变量因果证明。所有结论仍限于单 seed 和每首 2 次采样。
4. **最佳 checkpoint 不是最后一个 epoch。** 六组修正实验加本次 follow-up 的最佳 valid epoch 为 4 或 5；旧版 timestep 分别为 epoch 7 和 6；两个 no-mask 组分别是 epoch 4（shared）和 epoch 3（separate）。应按无 mask validation loss 选 checkpoint。
5. **PT base-1 的 PN HR 低于 PT scratch（1.8078 vs. 8.8233）。** PT scratch 行采用本次最多 8 epoch、连续 3 次 valid 上升早停的完整 run，best 是 epoch 2；两行均使用同一 DINR HR 函数重算，PT 原始评估脚本的配对汇总不用于此表。

## 7. 半 epoch validation / early stopping 重跑

本轮重新运行除 `timestep 3+pitch` 外的全部 12 个配置：PT 2 组、DINR attribute 2 组、DINR timestep 2 组、DINR none 2 组、DINR `timestep 1+pitch` 4 组。`timestep 3+pitch` 本轮不重跑，Section 5 中的对应数值仍是历史结果。

训练规则统一为：最多 8 epoch；每个 epoch 在中点和结束时各验证一次，中点不能整除 optimizer step 时使用其后的第一个 step；valid loss 严格低于历史最低值才算改善，持平也计为未改善；连续 4 次未改善即早停。只保存最低 valid loss 对应的 checkpoint，每个实验最终只保留一个最佳 checkpoint，不保存 optimizer state。Train loss 每个 optimizer step 记录一次，valid loss 保留每 epoch 两次的全部点。

沿用 held-out-12-work split、seed 42 和 effective batch 32。DINR 使用单卡 batch 32、学习率 `3e-4` 和 linear decay；PT 使用单卡 batch 2 × gradient accumulation 16、学习率 `5e-4` 和 cosine decay。PT 上一轮的双卡 batch 2 × accumulation 8 改为单卡队列，effective batch 仍为 32。DINR timestep 使用当前代码的整 note mask（`decoder_mask_mode=note`），timestep 1+pitch 使用 `timestep_single`。

本轮所有配置均完成 19 首 test work × 2 samples（38 个生成）。HR 使用本文件第 1 节定义的逐 note-index Wasserstein 口径。新的训练、valid 曲线、最佳 checkpoint、停止 epoch 和 HR 汇总位于 [`epx1111_halfepoch`](../analysis/epx1111_halfepoch/)。Section 7 的五列两行总图仅含本轮 12 项实验；五列依次为 PT、DINR attribute、DINR timestep、DINR none、DINR `timestep 1+pitch`，第一行是逐 optimizer step 的 train loss，第二行是每半个 epoch 的 valid loss。后续实验统一另绘 K 列两行图。

![五组 train/valid 曲线，五列两行](../analysis/epx1111_halfepoch/all_groups.png)

[总图 PDF](../analysis/epx1111_halfepoch/all_groups.pdf) · [本轮结果 CSV（含 checkpoint 路径）](../analysis/epx1111_halfepoch/all_experiments.csv) · [原始 train/valid loss CSV](../analysis/epx1111_halfepoch/all_losses.csv)

| method | 训练mask | time表示 | Best valid point | Best valid loss | Stop epoch | IOI HR | Dur HR | Vel HR | PN HR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| PT base-1 | non-pitch token | abs 5000 token | 1.0 | 1.628303 | 3.0 | 2.0598 | 2.0549 | 1.6014 | **1.9053** |
| PT scratch | non-pitch token | abs 5000 token | 1.0 | 3.045935 | 3.0 | 10.2280 | 82.9304 | 3.6032 | **32.2539** |
| DINR separate | attribute | abs 500 token | 5.0 | 10.556693 | 7.0 | 3.8375 | 2.3808 | 1.6076 | **2.6086** |
| DINR shared | attribute | abs 500 token | 4.5 | 10.507156 | 6.5 | 3.9974 | 2.9456 | 1.9862 | **2.9764** |
| DINR separate | timestep | abs 500 token | 6.0 | 10.642722 | 8.0 | 3.5809 | 2.4194 | 2.1074 | **2.7026** |
| DINR shared | timestep | abs 500 token | 6.0 | 10.582067 | 8.0 | 3.8383 | 2.6406 | 2.2459 | **2.9082** |
| DINR separate | none | abs 500 token | 3.0 | 10.949567 | 5.0 | 3.5455 | 2.2517 | 2.5002 | **2.7658** |
| DINR shared | none | abs 500 token | 4.0 | 10.665311 | 6.0 | 3.0398 | 2.2748 | 2.9645 | **2.7597** |
| DINR separate | timestep 1+pitch | abs 500 token | 4.5 | 10.454540 | 6.5 | 2.4867 | 2.1589 | 1.3977 | **2.0144** |
| DINR shared | timestep 1+pitch | abs 500 token | 4.0 | 10.437893 | 6.0 | 2.9933 | 2.2065 | 1.9703 | **2.3900** |
| DINR separate | timestep 1+pitch | log abs | 4.0 | 10.375171 | 6.0 | 2.9767 | 2.3997 | 2.3087 | **2.5617** |
| DINR separate | timestep 1+pitch | log dev | 6.0 | 11.280642 | 8.0 | 2.2263 | 2.3196 | 1.5526 | **2.0328** |

在这轮重跑中，DINR 的最低 PN HR 是 `timestep 1+pitch` 的 separate + abs 500 token，PN HR **2.0144**；PT base-1 的 PN HR 为 **1.9053**。PT scratch 的 duration HR 明显偏高，PN HR 为 **32.2539**。这些数值只与本轮相同协议下的结果直接比较，不能与 Section 5 中不同训练/验证协议的数值混为一谈。

<!-- PT_SCRATCH_FULL8_BEGIN -->
### 7.1 PT scratch：关闭早停，双卡完整训练 8 epoch

状态：**训练与测试评估全部完成**。仅 PT scratch 重新从随机初始化训练；使用 GPU0、GPU1 的 DDP，每卡 batch 2 × gradient accumulation 8，全局 batch 32。关闭 early stopping，固定训练 8 epoch；每半个 epoch 验证一次，共 16 次，仍只保留 valid loss 最低的 checkpoint。学习率 `5e-4`、cosine decay、seed 42 和 held-out split 沿用本轮设置。

旧 scratch 在 epoch 1 的 valid loss 为 3.045935；epoch 2、2.5、3 分别为 3.307507、3.224152、3.075577，最后两次正在下降但未超过历史最佳，因此触发 patience 4。完整 8 epoch 实验用来检查后续能否再次改善。上方原 12 项表格和五列两行总图只保留早停版；完整 8 epoch 版放在后续实验的独立 K 列两行图中。

已记录 validation：16/16。

| method | Best valid epoch | Best valid loss | Final epoch | IOI HR | Dur HR | Vel HR | PN HR |
|---|---:|---:|---:|---:|---:|---:|---:|
| PT scratch full 8ep | 1.5011673151750973 | 3.114427 | 8.0 | 34.1373 | 81.7223 | 3.3218 | 39.7271 |

训练后自动选择最佳 valid checkpoint，使用双卡、每卡 4 workers 完成 19 首 × 2 samples；temperature 1.0、top-p 0.95、seed 42，按相同 DINR note-index W1 和 human anchors 计算分项 HR。

运行产物：[`pt_scratch_halfepoch_full8_2gpu`](../../results/pt_scratch_halfepoch_full8_2gpu/)，包括 `config.json`、`status.json`、`train.log`、完整 loss 历史、最佳 checkpoint、测试 MIDI 和 `hr.json`。

训练与进程退出状态：完整 loss 历史及最终 trainer state 均确认 **epoch 8、5144 optimizer steps、16 次 validation**；训练结束时 DDP/NCCL 的 ALLGATHER 同步超时，使 torchrun 返回失败，原自动测试因此未触发。已核验保存的唯一最佳 checkpoint（step 965，epoch 1.5012），保留原始报错记录，单独恢复最佳模型测试，未重新训练。

**训练曲线结论：本次跑满 8 epoch 没有在后期刷新 valid 最佳值。** epoch 1.5012 最低为 3.114427；epoch 3 为 3.376163，3.5012 短暂降到 3.293414，随后整体升高，epoch 8 为 3.879771。即使中间有局部下降，后续也未超过历史最佳；本次结果不支持延长到 8 epoch 能改善该配置的 best valid。双卡运行的随机数消耗和数据分配与前一轮单卡不同，不能把两轮不同的最低 loss 单独归因于早停。
<!-- PT_SCRATCH_FULL8_END -->

<!-- DINR_COORD_QUEUE_BEGIN -->
### 7.2 timestep 1+pitch：lookup + coordinate 附加编码对照

**实现范围澄清：本节保留数值 lookup，再加坐标分支；不是用坐标映射替代 lookup。其结果不能直接回答纯 coordinate 编码是否有效；纯映射替代实验见第 7.3 节。**

GPU2 单卡顺序队列：**MLP → linear → Fourier**。固定 separate + abs 500 token、PT rendering 初始化、held-out-12 split、batch 32、LR `3e-4`、seed 42、linear scheduler、训练 decoder timestep mask 0.5。最多 8 epoch，每半 epoch 验证一次，连续 4 次未低于历史最低 valid 则早停；只保存最佳 valid checkpoint，train loss 每 optimizer step 记录。

三个输入属性均采用 `lookup(token) + coordinate(value)`：IOI 与 Dur 为 `bin × 0.01` 秒，Vel 为 `value / 127`。**IOI、Dur、Vel 各有独立的数值映射**，score encoder 与 decoder history 共用对应属性表。pitch 不加坐标，PAD/BOS 和被 mask 属性不泄漏坐标。输出仍是原来的独立 linear heads + hard CE；本实验比较输入数值编码，不改变输出损失或引入 prototype head。

| Encoding | 数值分支 | 三属性数值分支参数量 |
|---|---|---:|
| linear | `Linear(1, 768)`，含 bias | 4,608 |
| MLP | `Linear(1,32) → GELU → Linear(32,768)`，含 bias | 76,224 |
| Fourier | `[x, sin(kπx), cos(kπx)]`，`k=1..16`，再 `Linear(33,768,bias=False)` | 76,032 |

MLP 与 Fourier 数值分支容量接近；linear 刻意保留为低容量线性对照。每组从相同 PT checkpoint 独立初始化，沿用已有无 coordinate 的 separate + abs500 对照：best valid `10.454540`（epoch 4.5），PN HR `2.0144`。

| Encoding | 状态 | Latest epoch | Valid 次数 | Best epoch | Best valid | IOI HR | Dur HR | Vel HR | PN HR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp | 已完成 | 8.000 | 16 | 6.000 | 10.637312 | 2.7696 | 2.3965 | 1.8535 | 2.3399 |
| linear | 已完成 | 8.000 | 16 | 8.000 | 11.249613 | 3.0432 | 2.2141 | 2.4649 | 2.5741 |
| fourier | 已完成 | 7.000 | 14 | 5.000 | 11.180223 | 4.3324 | 2.5972 | 2.4231 | 3.1175 |

训练后自动使用 best valid checkpoint 对 19 首 × 2 samples 测试（GPU2、8 workers，temperature 1、top-p 0.9），计算相同 DINR note-index W1 / human-anchor HR。三种 coordinate 编码的曲线放在后续实验独立 K 列两行图的 `Lookup + coordinate` 组；不并入第 7 节原始五列总图。

配置、命令、状态与完整日志：[`dinr_single_pitch_coord_queue_gpu2`](../../results/dinr_single_pitch_coord_queue_gpu2/)。

#### 7.2.1 完成结果与结论

三组均完成训练及 **19 首 × 2 samples，共 38 个生成**。MLP 和 linear 训练到 epoch 8；Fourier 在 epoch 5 达到 best valid，随后连续 4 次验证未改善，在 epoch 7 早停。

**三种 coordinate 编码都没有超过无 coordinate 基线。** 按 test PN HR（越低越好）排序：无 coordinate **2.0144** < MLP **2.3399** < linear **2.5741** < Fourier **3.1175**。相对基线，MLP、linear、Fourier 的 PN HR 分别升高 **16.2%、27.8%、54.8%**；三组各自的 IOI、duration、velocity 分项 HR 也都高于基线。当前配置继续保留 lookup-only 作为优选。

**MLP 是本轮最好的坐标编码，但不能据此说增加非线性就一定更好。** MLP 的 best valid 为 10.637312，优于 Fourier 的 11.180223 和 linear 的 11.249613；test PN HR 相比 linear 降低约 9.1%，相比 Fourier 降低约 24.9%。Fourier 的 best valid 略优于 linear，test HR 却更差，说明该小范围比较中 teacher-forcing valid loss 与生成 HR 的排序并不完全一致。

**曲线与适用范围。** linear 到 epoch 8 仍刷新 valid 最低值，因此这里只能比较当前最多 8 epoch 的预算，不能断言它训练更久仍不会改善。MLP 在 epoch 6 后 valid 回升；Fourier 在 epoch 5 后不再刷新最低值。三组都只运行 seed 42，MLP 与 Fourier 数值分支参数量接近，而 linear 参数更少；映射结构变化也会改变随机数消耗及新增参数初始化。当前结果不足以断言 Fourier 普遍无效或 MLP 普遍优越。本轮只比较输入端 `lookup + coordinate`，输出仍是原来的 hard CE，因此也没有检验带坐标的输出 prototypes、ordinal loss 或距离感知 loss 的效果。

原始曲线与汇总：[五列两行总图](../analysis/epx1111_halfepoch/all_groups.png) · [逐步 loss CSV](../analysis/epx1111_halfepoch/all_losses.csv) · [实验结果 CSV](../analysis/epx1111_halfepoch/all_experiments.csv)。后续曲线见[独立 K 列两行图](../analysis/epx1111_halfepoch/followup_groups.png)及其 [PDF](../analysis/epx1111_halfepoch/followup_groups.pdf) 和 [CSV](../analysis/epx1111_halfepoch/followup_all_losses.csv)。当前后续实验分为 K=4 组，因此图为四列两行。
<!-- DINR_COORD_QUEUE_END -->

<!-- DINR_PURE_COORD_QUEUE_BEGIN -->
### 7.3 timestep 1+pitch：纯 coordinate 替代数值 lookup

**定义：`e_IOI=f_IOI(bin×0.01秒)`、`e_Dur=f_Dur(bin×0.01秒)`、`e_Vel=f_Vel(value/127)`，三个映射独立；数值 embedding 完全由映射计算，没有 lookup 残差，也没有 IOI、Dur、Vel 的可学习 lookup 行。** 仅 pitch 与特殊符号保留 `133×768` 的 compact embedding 表；PAD 为零，BOS 沿用特殊符号表示。score encoder 与 decoder history 使用同一套对应属性映射；`timestep 1+pitch` mask 行只保留 pitch 加单个 mask vector。

GPU2 队列原计划包含 Fourier；训练开始后决定取消该项，并改为单独运行 linear 16 epoch。已完成的 pure-coordinate 对照仍为 separate + abs500、10 ms bins、PT rendering backbone/pitch 初始化、seed42、batch32、LR3e-4、linear decay、无 warmup、最多8 epoch、每半 epoch验证、patience4、只保存 best valid checkpoint，train 每 optimizer step 记录。linear 16 epoch 沿用该配置，仅将 epoch 上限改为16。数值映射为随机初始化，不能从 PT 的独立数值 lookup 精确复制；未增加预拟合或蒸馏。输出仍是 grouped/chord IOI、Dur、Vel 的原分类 heads + hard CE，分类头参数仍依赖 bins 数。

| 方法 | 每属性数值映射 | 三属性映射参数量 | 数值 lookup 参数量 |
|---|---|---:|---:|
| MLP | `Linear(1,32) → GELU → Linear(32,768)` | 76,224 | **0** |
| linear | `Linear(1,768)`，含 bias | 4,608 | **0** |

本轮纯 coordinate 比较聚焦 MLP 与 linear；中止的 Fourier 训练不作为结果，不纳入图表和比较。每组 `representation_audit.json` 记录实际数值 lookup 行数与映射参数量；测试从 checkpoint 权重自动识别 compact lookup 和 coordinate 结构。

| 方法 | 状态 | Latest epoch | Valid 次数 | Best epoch | Best valid | IOI HR | Dur HR | Vel HR | PN HR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp | 已完成 | 8.000 | 16 | 8.000 | 11.196815 | 3.3451 | 2.2988 | 2.2827 | 2.6422 |
| linear | 已完成 | 8.000 | 16 | 8.000 | 11.392002 | 3.5701 | 2.3292 | 2.0883 | 2.6625 |
| linear (16ep) | 已完成 | 13.500 | 27 | 11.500 | 11.318710 | 3.6204 | 2.3713 | 2.7579 | 2.9165 |

8 epoch 对照与 linear 16 epoch 任务使用 best valid checkpoint 完成 **19首×2 samples**测试，GPU2、8 workers、temperature1、top-p0.9；按相同 note-index W1/human anchors 计算 HR。曲线放入后续独立 K 列两行图的 `Pure coordinate replacement` 组，不并入第 7 节原始总图。无 coordinate 的 lookup-only 基线 PN HR 为 **2.0144**；第7.2节 lookup+coord 的 MLP/linear/Fourier 为已完成的独立实验，PN HR 为 **2.3399 / 2.5741 / 3.1175**。

运行目录：[纯 coordinate 队列](../../results/dinr_single_pitch_pure_coord_queue_gpu2/)，包含命令、结构审计、完整loss、最佳权重、测试MIDI及HR。
<!-- DINR_PURE_COORD_QUEUE_END -->

<!-- DINR_PURE_MLP_16EP_BEGIN -->
### 7.4 纯 MLP coordinate：16 epoch 与隐层宽度对照

在第7.3节纯 coordinate 实验上继续验证训练预算和映射容量。GPU0 使用 `1→32→768`，GPU1 使用 `1→128→768`；均为每属性独立两层 MLP、无 timing/velocity lookup。其余配置一致：separate + abs500、`timestep 1+pitch`、seed42、batch32、LR `5e-4`、cosine scheduler、warmup0、最多16 epoch、每半 epoch validation、patience4、仅保存 best valid checkpoint。

调度器按最多16 epoch对应的 5152 optimizer steps 预先构造。若 patience 4 触发，则在早停点结束；若未触发，计划末尾 LR 按 cosine 降到接近零。训练数据和 held-out split 沿用前两轮。

| GPU | 每属性映射 | 三属性映射参数 | 状态 | Epoch | Validation | Best epoch | Best valid | IOI HR | Dur HR | Vel HR | PN HR |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `1→32→768` | 76,224 | 已完成 | 9.500 | 19 | 7.500 | 11.216751 | 3.0580 | 2.2367 | 2.2943 | 2.5296 |
| 1 | `1→128→768` | 297,984 | 已完成 | 11.000 | 22 | 9.000 | 11.203219 | 3.7883 | 2.3316 | 2.9842 | 3.0347 |

第7.3节 pure MLP-32（LR3e-4、linear decay、8 epoch）best valid 为 11.196815（epoch 8.000），test PN HR 2.6422。由于本节同时把学习率从3e-4改为5e-4并改用cosine，与第7.3节的差异不能单独归因为训练时长或 MLP 宽度；GPU0/1 两组之间才是相同训练协议下的宽度对照。

本节单独绘制 16 epoch 的 32/128 宽 MLP 对照图，不并入第7节五组总图：[训练/valid 曲线](../../results/dinr_single_pitch_pure_mlp_16ep_gpu01/width_comparison.png) · [PDF](../../results/dinr_single_pitch_pure_mlp_16ep_gpu01/width_comparison.pdf) · [逐步 CSV](../../results/dinr_single_pitch_pure_mlp_16ep_gpu01/all_losses.csv)。

运行配置、表示结构审计、日志、best checkpoint、测试 MIDI 和 HR：[32/128 宽 MLP 双卡实验](../../results/dinr_single_pitch_pure_mlp_16ep_gpu01/)。
<!-- DINR_PURE_MLP_16EP_END -->

## 8. 结果文件索引

- 修正后的 DINR mask 实验：`INSPIRE2/models/dinr_mask_revised_heldout12_8ep/`
- 全部模型统一 HR：`INSPIRE2/analysis/dinr_six_model_hr_comparison/`
- DINR mask loss：`INSPIRE2/analysis/dinr_stage1_mask_ablation/`
- DINR separate table loss：`INSPIRE2/analysis/dinr_separate_ioi_duration_heldout12_8ep/`
- DINR coordinate loss：`INSPIRE2/analysis/dinr_nonpitch_coordinates_heldout12_8ep_summary.md`
- PT rendering SFT：`INSPIRE2/analysis/pt_original_baseline_8ep_complete/`
- PT base-1：`INSPIRE2/analysis/pt_base_baseline_8ep/`
- PT init train：`results/pt_init_median_heldout12_8ep/`

