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

PT init train 使用与 PT base-1 完全相同的 SFT 数据、mask、batch、学习率和 validation 协议，但 `pretrained_model=null`，即从随机初始化的 PT architecture 开始训练 8 epoch。启动配置为 [`pt_init_median_heldout12_8ep.json`](../../configs/pt_init_median_heldout12_8ep.json)，使用两张 GPU 的 DDP：

```bash
CUDA_VISIBLE_DEVICES=1,2 \
torchrun --standalone --nproc_per_node=2 -m src.train.sft \
  --config configs/pt_init_median_heldout12_8ep.json
```

训练完成后将在本节补充每个 epoch 的 train/valid loss、最佳 checkpoint 及与 PT base-1 的对比。启动阶段已确认随机初始化的 token CE 明显高于 pretrained PT，前几个 validation 点用于判断它是否先下降再过拟合，还是从一开始就不具备泛化能力。

## 3. DINR absolute 表示与 Mask

DINR 使用 note-level absolute timing 目标。无 pedal 诊断阶段输出 `[pitch, IOI, velocity, duration]`，IOI 和 duration 量化为 10 ms bins，velocity 量化为 MIDI 0–127。score encoder 不 mask；decoder performance history 只在训练时 mask，validation 和 test 不 mask。

本次重跑统一使用三个独立、可学习的属性 mask embedding，分别对应 IOI、velocity、duration；三者都以 `initializer_range`（约 0.02）正态初始化。mask 直接替换对应槽位的 embedding，不使用普通词表中的 MASK token ID 1。首个 BOS timestep 不参与 mask。

- **attribute mask**：三个属性各自独立以 0.5 概率替换为自己的属性专属 mask embedding；pitch 不 mask。
- **timestep retain pitch**：每个 decoder timestep 以 0.5 概率同时 mask IOI、velocity、duration，并分别使用各自属性专属 embedding；pitch 保留。
- **no mask**：保留完整 decoder history，作为无正则对照。

旧版 **timestep**（总表单独列出）每个历史 timestep 以 0.5 概率将完整 note embedding 替换成一个 learned mask embedding，因此 pitch、IOI、velocity、duration 一起被 mask。该整-note mask 向量范数约 0.55；它与 attribute pilot 中误用的词表 ID 1（范数约 28.3）不是同一个 embedding。旧版 timestep 的训练和 19 首 × 2 次 test 结果有效，保留为 legacy baseline；它与 timestep retain pitch 是两种不同 mask 配置，不作为只改变 pitch 是否保留的严格单变量对照。旧 attribute pilot 和第一次以 ID 1 初始化的 pitch-retaining pilot 不纳入最终结果。本轮六组修正实验使用属性专属 embedding 和统一的小尺度初始化。

六组训练均为 8 epoch、seed 42、batch 32、learning rate `3e-4`，从相同 PT rendering checkpoint 初始化。每 epoch 保存 checkpoint，以无 mask validation loss 选择最佳 checkpoint；test 也不 mask。

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
| timestep retain pitch | separate | abs 500 token | 5 | 10.538136 | 10.926798 |
| timestep retain pitch | shared | abs 500 token | 4 | 10.447859 | 10.840719 |
| timestep retain pitch | separate | log abs | 4 | 10.402810 | 10.734770 |
| timestep retain pitch | separate | log dev | 5 | 11.256263 | 11.440074 |

Validation loss 单独看并不能决定最终配置：它偏好 shared no-mask，但该模型 test HR 较高，说明 validation loss 与生成质量不完全一致。所有 mask 组的最佳 valid epoch 都早于 epoch 8。这里的 `timestep` 是旧版整-note mask；`timestep retain pitch` 是修正后的按属性 mask。

## 5. 所有实验的统一 test HR 大表

共 12 个实验：PT 两个实验固定放在前两行；DINR 包含 2 组 attribute、2 组 legacy timestep、2 组 no-mask 和 4 组 timestep retain pitch 实验。每行使用 **best valid epoch** checkpoint，在相同 19 首 test work × 2 samples（38 个生成）协议上评估，按 test work 从长到短推理。

表内 HR 都按 IOI、duration、velocity 的 W1 分别除以固定 human anchors（22.772456、75.085651、10.580359），再对三个分项取算术平均；越低越好。所有行采用同一 `evaluate_dinr_absolute.py` 的逐 note-index W1 算法。PT 的原始 `pn_hr.json` 使用不同的配对聚合，不可直接比较；PT 两行已用相同 DINR 算法对相同 38 个生成重算，结果保存在各自 test 目录的 `pn_hr_dinr_protocol.json`。

| method | 训练mask | time表示 | Best valid epoch | Best valid loss | IOI HR | Dur HR | Vel HR | PN HR |
|---|---|---|---:|---:|---:|---:|---:|---:|
| PT base-1 | non-pitch token | abs 5000 token | 0.871595 (step 560) | 1.627733 | 1.8847 | 2.0883 | 1.4504 | **1.8078** |
| PT scratch | non-pitch token | abs 5000 token | 1 | 2.567680 | 16.3307 | 17.9762 | 3.0993 | **12.4687** |
| DINR separate | attribute | abs 500 token | 5 | 10.504103 | 3.7085 | 2.3880 | 1.4958 | **2.5308** |
| DINR shared | attribute | abs 500 token | 5 | 10.500670 | 3.8063 | 2.8055 | 1.9103 | **2.8407** |
| DINR separate | timestep | abs 500 token | 7 | 10.760382 | 2.6905 | 2.2249 | 1.5618 | **2.1591** |
| DINR shared | timestep | abs 500 token | 6 | 10.585932 | 3.2274 | 2.4014 | 1.8376 | **2.4888** |
| DINR separate | none | abs 500 token | 3 | 10.840699 | 3.7024 | 2.3735 | 2.6160 | **2.8973** |
| DINR shared | none | abs 500 token | 4 | 10.447963 | 4.4274 | 2.8612 | 2.9568 | **3.4152** |
| DINR separate | timestep retain pitch | abs 500 token | 5 | 10.538136 | 2.9221 | 2.2456 | 1.6431 | **2.2702** |
| DINR shared | timestep retain pitch | abs 500 token | 4 | 10.447859 | 3.3086 | 2.5026 | 2.3302 | **2.7138** |
| DINR separate | timestep retain pitch | log abs | 4 | 10.402810 | 2.8578 | 2.2296 | 2.5025 | **2.5300** |
| DINR separate | timestep retain pitch | log dev | 5 | 11.256263 | 2.2562 | 2.3685 | 1.7990 | **2.1412** |

`log dev` 使用 log absolute 输入和 score-relative log target；`log abs` 使用 log absolute 输入和 target。PT valid loss 是 token CE，DINR valid loss 是属性 head loss，量纲不同，不能横向比较。PT 使用 abs 5000 token；DINR 的 abs 500 token 是 500 类、每类 10 ms。统一 CSV 见 [`all_experiments.csv`](../analysis/dinr_six_model_hr_comparison/all_experiments.csv)。

表中的旧版 `timestep` 两行使用整-note learned mask，包含 pitch；它们按统一 HR 口径保留。旧 attribute pilot 的 ID 1 mask 尺度异常，因此没有把该 pilot 结果并入当前 attribute 行。第一次 pitch-retaining pilot 的 ID 1 初始化结果（PN HR 约 4.55 / 7.43）不纳入表格，已由修正初始化后的四组 timestep-retain-pitch 结果替代。`timestep retain pitch` 四行排在 DINR 结果末尾，和旧版 `timestep` 清楚区分。

## 6. 结论

1. **Separate timing tables 在 attribute 和 timestep retain pitch 两类修正 mask 下都优于 shared。** attribute 的 PN HR 为 2.5308（separate）与 2.8407（shared）；timestep retain pitch + abs 500 为 2.2702（separate）与 2.7138（shared）。旧版整-note timestep 的 PN HR 为 2.1591（separate）和 2.4888（shared），作为独立 legacy 结果列出。
2. **六组修正实验中，log dev 的 PN HR 最低，为 2.1412。** 它使用 separate tables、保留 pitch，并以 score-relative log 作为 target；这是当前结果，不代表已对多随机种子做统计检验。
3. **旧版 timestep 与 retain-pitch timestep 是两种不同配置。** 前者用一个 learned embedding 替换整条 note（包括 pitch），后者用三个属性专属 embedding mask IOI、velocity、duration 并保留 pitch。两组结果都在表中，但不能当作仅改变 pitch 是否 mask 的严格单变量实验；错误使用 ID 1 初始化的 pitch-retaining pilot 不纳入比较。
4. **最佳 checkpoint 不是最后一个 epoch。** 六个修正实验的最佳 valid epoch 为 4 或 5；旧版 timestep 分别为 epoch 7 和 6；两个 no-mask 组分别是 epoch 4（shared）和 epoch 3（separate）。应按无 mask validation loss 选 checkpoint。
5. **PT base-1 的 PN HR 低于 PT scratch。** 两行均使用同一 DINR HR 函数重算；PT 原始评估脚本的配对汇总不用于此表。

## 7. 结果文件索引

- 修正后的 DINR mask 实验：`INSPIRE2/models/dinr_mask_revised_heldout12_8ep/`
- 全部模型统一 HR：`INSPIRE2/analysis/dinr_six_model_hr_comparison/`
- DINR mask loss：`INSPIRE2/analysis/dinr_stage1_mask_ablation/`
- DINR separate table loss：`INSPIRE2/analysis/dinr_separate_ioi_duration_heldout12_8ep/`
- DINR coordinate loss：`INSPIRE2/analysis/dinr_nonpitch_coordinates_heldout12_8ep_summary.md`
- PT rendering SFT：`INSPIRE2/analysis/pt_original_baseline_8ep_complete/`
- PT base-1：`INSPIRE2/analysis/pt_base_baseline_8ep/`
- PT init train：`results/pt_init_median_heldout12_8ep/`

