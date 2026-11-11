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

Mask 对照为：

- **attribute mask**：IOI、velocity、duration 三个非 pitch 属性分别以 0.5 概率替换为共享 MASK；pitch 不 mask。
- **timestep mask**：以 0.5 概率将整个 decoder timestep 的输入 embedding 替换为 learned mask embedding。
- **no mask**：保留完整 decoder history，作为无正则对照。

所有 8 epoch DINR 训练都保存每个 epoch 的 checkpoint，以 valid loss 选择 best，而不是默认使用最后一个 epoch。这一点非常关键：no-mask 的 train loss 下降最快，但 valid 在最佳点以后快速上升；timestep 和 attribute mask 虽然也继续降低 train loss，但后期 valid 回升明显减弱。

这里描述的是早期 mask ablation 和六模型比较采用的旧版 timestep mask，pitch 也随整条 embedding 一起被遮挡。新增时间表示实验采用修订定义：保留 pitch，只对 IOI、velocity、duration 使用同一个 Bernoulli 决策共同 mask。两种定义不同，不能把它们的结果当成完全相同 mask 配置的重复实验。

## 4. Shared 与 separate timing table

Shared 方案使用同一组 500-bin timing value embedding，但 IOI 和 duration 在求和前经过独立 slot projection。Separate 方案为 IOI 和 duration 使用两张独立 `[500, 768]` token table，均从相同 PT timing embedding 初始化，之后独立接收梯度。

八 epoch validation 结果：

| Mask | Table | Best epoch | Best valid | Epoch 8 valid |
|---|---|---:|---:|---:|
| attribute | shared | 7 | 11.120880 | 11.137168 |
| attribute | separate | 7 | 10.991883 | 11.005986 |
| timestep | shared | 6 | 10.585932 | 10.633567 |
| timestep | separate | 7 | 10.760382 | 10.777897 |
| no mask | shared | 4 | **10.447963** | 11.870809 |
| no mask | separate | 3 | 10.840699 | 11.834352 |

Validation loss 单独看并不能决定最终配置：它偏好 shared no-mask，但该模型 test HR 最差，说明 validation CE 与生成质量并不完全一致。

## 5. 所有实验的统一 test HR 大表

共 12 个实验：PT 两个实验固定放在前两行，之后是原六组 DINR 和四个新增 DINR 实验。每行使用 **best valid epoch** checkpoint，在相同 19 首 test work × 2 samples（38 个生成）协议上评估。

表内 HR 都按 IOI、duration、velocity 的 W1 分别除以固定 human anchors（22.772456、75.085651、10.580359），再对三个分项取算术平均；越低越好。DINR 六模型、新时间表示实验、新 shared/pitch-retained 实验均采用同一 `evaluate_dinr_absolute.py` 的逐 note-index W1 算法。PT 的原始 `pn_hr.json` 使用不同的配对聚合，不可直接比较；PT 两行已用相同 DINR 算法对相同 38 个生成重算，结果保存在各自 test 目录的 `pn_hr_dinr_protocol.json`。

| method | 训练mask | time表示 | Best valid epoch | Best valid loss | IOI HR | Dur HR | Vel HR | PN HR |
|---|---|---|---:|---:|---:|---:|---:|---:|
| PT base-1 | non-pitch token | abs 5000 token | 0.8716 (step 560) | 1.627733 | 1.8847 | 2.0883 | 1.4504 | **1.8078** |
| PT scratch | non-pitch token | abs 5000 token | 1 | 2.567680 | 16.3307 | 17.9762 | 3.0993 | **12.4687** |
| DINR separate | timestep | abs 500 token | 7 | 10.760382 | 2.6905 | 2.2249 | 1.5618 | **2.1591** |
| DINR shared | timestep | abs 500 token | 6 | 10.585932 | 3.2274 | 2.4014 | 1.8376 | **2.4888** |
| DINR shared | attribute | abs 500 token | 7 | 11.120880 | 3.3769 | 2.4736 | 2.3669 | **2.7391** |
| DINR separate | attribute | abs 500 token | 7 | 10.991883 | 3.7774 | 2.4016 | 2.2515 | **2.8102** |
| DINR separate | none | abs 500 token | 3 | 10.840699 | 3.7024 | 2.3735 | 2.6160 | **2.8973** |
| DINR shared | none | abs 500 token | 4 | 10.447963 | 4.4274 | 2.8612 | 2.9568 | **3.4152** |
| DINR separate | timestep retain pitch | log dev | 8 | 11.403992 | 2.3754 | 2.4437 | 1.7900 | **2.2031** |
| DINR separate | timestep retain pitch | log abs | 8 | 10.932355 | 7.0211 | 3.9305 | 2.5677 | **4.5064** |
| DINR separate | timestep retain pitch | abs 500 token | 6 | 11.136314 | 6.8848 | 4.1512 | 2.6267 | **4.5542** |
| DINR shared | timestep retain pitch | abs 500 token | 8 | 10.987569 | 11.8375 | 7.7103 | 2.7323 | **7.4267** |

`log dev` 使用 log absolute 输入和 score-relative log target；`log abs` 使用 log absolute 输入和 target。PT valid loss 是 token CE，DINR valid loss 是属性 head loss，量纲不同，不能横向比较。PT 使用 abs 5000 token；DINR 的 abs 500 token 是 500 类、每类 10 ms。表内 HR 使用同一组 anchors 和 DINR `pn_metrics` 评估算法；PT 结果已按此算法重算，明细见 [`all_experiments.csv`](../analysis/dinr_six_model_hr_comparison/all_experiments.csv)。

需要注意，旧六组的 `timestep` 与新增的 `timestep retain pitch` 训练操作不只差 pitch：旧版把整步 decoder embedding 替换成 learned mask embedding；新版以一个共同 Bernoulli 决策，将 IOI、velocity、duration token 替换为 MASK token 并保留 pitch。因此两行的 HR 数值按同一评估口径计算，但不能据此把差异解释为单独的 pitch 效应。

## 6. 结论

1. **在原六模型比较中，旧版 timestep mask 取得前两名。** 它在 shared 和 separate table 下都取得前两名；pitch-retaining timestep 使用不同的 masking 操作，当前结果不能单独归因于 pitch 是否保留。
2. **最佳 table 是 separate IOI/duration table。** timestep + separate 的 PN HR 为 2.1591，优于 timestep + shared 的 2.4888；这是当前六组实验中最好的组合。
3. **最佳 checkpoint 不是最后一个 epoch。** 六组模型都应从 valid 曲线选择 checkpoint。尤其 no-mask 的 valid 最佳分别是 epoch 4（shared）和 epoch 3（separate），而不是 epoch 8；timestep shared 最佳是 epoch 6，attribute shared/separate 和 timestep separate 最佳是 epoch 7。
4. **PT base-1 也在第一个 epoch 达到最佳 valid。** PT train loss 持续下降，但 valid 从第 1 epoch 起上升，这与 DINR no-mask 的过拟合形态一致，说明“训练到最后一轮”不是可靠的 checkpoint 选择规则。
5. **PT scratch 的 PN HR 高于 PT base-1。** 两行均使用同一 DINR HR 函数重算；PT 原始评估脚本的配对汇总不用于此表。

## 7. 结果文件索引

- DINR 六组 HR：`INSPIRE2/analysis/dinr_six_model_hr_comparison/`
- DINR mask loss：`INSPIRE2/analysis/dinr_stage1_mask_ablation/`
- DINR separate table loss：`INSPIRE2/analysis/dinr_separate_ioi_duration_heldout12_8ep/`
- DINR coordinate loss：`INSPIRE2/analysis/dinr_nonpitch_coordinates_heldout12_8ep_summary.md`
- PT rendering SFT：`INSPIRE2/analysis/pt_original_baseline_8ep_complete/`
- PT base-1：`INSPIRE2/analysis/pt_base_baseline_8ep/`
- PT init train：`results/pt_init_median_heldout12_8ep/`

