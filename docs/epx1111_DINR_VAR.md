# INSPIRE2 / PianistTransformer 实验总览

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

## 7. 结果文件索引

- 修正后的 DINR mask 实验：`INSPIRE2/models/dinr_mask_revised_heldout12_8ep/`
- 全部模型统一 HR：`INSPIRE2/analysis/dinr_six_model_hr_comparison/`
- DINR mask loss：`INSPIRE2/analysis/dinr_stage1_mask_ablation/`
- DINR separate table loss：`INSPIRE2/analysis/dinr_separate_ioi_duration_heldout12_8ep/`
- DINR coordinate loss：`INSPIRE2/analysis/dinr_nonpitch_coordinates_heldout12_8ep_summary.md`
- PT rendering SFT：`INSPIRE2/analysis/pt_original_baseline_8ep_complete/`
- PT base-1：`INSPIRE2/analysis/pt_base_baseline_8ep/`
- PT init train：`results/pt_init_median_heldout12_8ep/`

