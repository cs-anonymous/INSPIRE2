# DINR-Absolute

这是一个独立的 note-level 离散基线。原始 PT 的 token-level 模型和训练脚本保持不变。

## 数据表示

训练和评估只读取 grouped-median 预处理产生的
`score_ASAP_refined.ASAP_GROUPED_MEDIAN.pt`。IOI 和 duration 使用预处理后的绝对毫秒值；正 score-IOI 音符预测 grouped IOI，总 grouped IOI 在重建时再展开为 note-level IOI；score-IOI 为零的音符使用 chord IOI。时间值量化为 10 ms，使用 500 个类别，范围为 `0..4990 ms`。velocity 量化为 MIDI 值 `0..127`，四个 pedal 采样点以 64 为阈值转成二值标签。

DINR 的紧凑词表共有 763 个 ID：pitch 为 `5..132`，velocity 为 `133..260`，IOI 和 duration 共用 timing ID `261..760`，pedal 二值 ID 为 `761..762`。grouped IOI 与 chord IOI 使用不同的输出 head；四个 pedal 位置共享两个二值 token。score 和 shifted performance 的 8 个属性 token embedding 在求和前分别经过独立的 slot-specific `768→768` 残差线性映射，再求和得到每个 note 的一个输入 embedding。这样不会把 IOI/duration 或四个 pedal 位置视为可交换属性。模型不使用 coordinate；coordinate 变体另外对 timing 和 velocity token 加入 Fourier 数值坐标。

损失为：

```text
L_ioi = L_grouped_ioi + L_chord_ioi
L_pedal = mean(L_pedal_1, L_pedal_2, L_pedal_3, L_pedal_4)
L_total = L_ioi + L_duration + L_velocity + L_pedal
```

`--pedal-mode ce` 使用每个 pedal 两个 logits 的二分类 CE；`--pedal-mode bce` 使用一个 logit 的 BCEWithLogitsLoss；`--pedal-mode none` 移除 pedal 输入、输出 head 和 pedal loss，此时不包含 `L_pedal`。

## 第一阶段：无 pedal 的属性级诊断

当前第一阶段只诊断 IOI、duration 和 velocity。设置 `--pedal-mode none` 时，运行时的 score encoder 与 shifted-performance decoder 输入均为 `[B,N,4]`，槽位顺序为 `[pitch, IOI, velocity, duration]`；输出 loss 只包含 grouped/chord IOI、duration、velocity 三项。预计算 sidecar 仍可存 8 槽位，数据集会在取样时裁成 4 槽位。

第一阶段采用**共享 value vocabulary + 属性专属 slot projection**。IOI 与 duration 可使用相同的 500 个 timing value ID，但在求和前分别经过 IOI、duration 的独立投影；因此相同 bin 不会被当成同一属性。pitch 和 velocity 也各自保留槽位投影。decoder 的历史 performance 输入可用 `--decoder-mask-mode attribute --decoder-mask-probability 0.5` 对 IOI、velocity、duration 独立采样 mask；三种属性共用 MASK ID `1`，随后仍经过各自的 slot projection。pitch 不 mask，encoder score 输入不 mask。`--decoder-mask-mode note` 保留旧的整 note mask 行为，整 timestep 的 mask 在 slot-sum 后将完整 note embedding 替换为 learned mask embedding。

三个输入表示方案的取舍：

1. **共享词表、共享 MASK ID、属性专属投影（第一阶段采用）**：参数较少、保留 PT value embedding 初始化，也能显式区分槽位。主要风险是类型区分依赖 slot projection 学出来，因此训练初期相同数值的 IOI/duration 表示相近。
2. **IOI 与 duration 使用独立词表**：类型身份最直接，但会复制 timing embedding 参数，并需要为新表设计 PT 初始化；这会同时改变参数量和初始化条件，适合作为后续消融。
3. **共享词表后拼接四个槽位，再用 MLP 融合**：槽位身份明确且融合更灵活，但参数和表达能力都会增加，也偏离 slot-sum。当前正研究过拟合来源，不建议把它作为第一阶段基线；若后续测试，应单独作为融合结构消融。

第一阶段的 mask 对照使用相同的 4-slot、无 pedal 模型、held-out-12-work split、PT 初始化、seed 42、global batch 32 和 8 epochs，只改变 decoder 历史输入的 mask 方式：

| Run | Mask 方式 | 配置 | 目的 |
|---|---|---|---|
| `attribute_random` | 属性级独立随机 mask | 每个历史 timestep 的 IOI、velocity、duration 各以 0.5 概率替换为共享 MASK ID；pitch 不 mask | 检验保留部分属性历史是否足以抑制 teacher-forcing 依赖 |
| `timestep_random` | 整个 timestep 随机 mask | 每个历史 timestep 以 0.5 概率将完整 4-slot embedding 替换为 learned mask embedding | 与属性级 mask 比较 mask 粒度 |
| `no_mask` | 不 mask | mask probability 为 0 | 提供同一新结构与 split 下的无正则对照，判断 mask 是否带来收益 |

三组每个 epoch 都保存 checkpoint，以 `eval_loss` 选择最佳模型，并通过 `save_total_limit=2` 只保留 valid 最佳与最新 epoch 两份 checkpoint；不在输出根目录另存一份模型。每步训练日志、epoch-end validation loss 和模型配置单独保存。训练和验证曲线使用 `trainer_log_history.json` 汇总；每个 epoch 的 train loss 用该 epoch 内 step loss 的均值近似，valid loss 是完整 held-out validation 集的 epoch-end loss。
训练结束后用 `python -m src.evaluate.plot_dinr_stage1_mask_ablation` 生成 `analysis/dinr_stage1_mask_ablation/loss_curves.png` 和 `losses.csv`。

启动三卡对照：

```bash
bash script/run_dinr_stage1_mask_ablation_3gpu.sh 0 1 2
```

输出位于 `models/dinr_stage1_mask_ablation_heldout12/{attribute_random,timestep_random,no_mask}/`。
另一个有价值的 mask 变体是**连续时间段 mask**：按 note 序列采样连续 span，使 decoder 经历成段缺失的 performance feedback。它测试与独立随机 mask 不同的时序缺失模式；先完成上述三组并查看训练/验证曲线，再决定是否增加该实验，避免缺少无 mask 对照。

### 第一阶段 mask 对比结果

三组于 2026-11-04 在 GPU0/1/2 并行完成 8 epochs。所有设置使用相同的 held-out-12-work 数据、PT 初始化、seed 42、global batch 32；只改变 decoder mask。完整曲线见 [loss_curves.png](../analysis/dinr_stage1_mask_ablation/loss_curves.png)，原始表格见 [losses.csv](../analysis/dinr_stage1_mask_ablation/losses.csv)。Train 值是每 epoch 所有 step-level loss 日志的均值，valid 值是完整 validation 集的 epoch-end loss。

| Epoch | Attribute train | Attribute valid | Timestep train | Timestep valid | No-mask train | No-mask valid |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.094 | 11.742 | 11.502 | 11.289 | 11.124 | 11.169 |
| 2 | 11.302 | 11.526 | 10.909 | 11.178 | 10.268 | 10.846 |
| 3 | 11.047 | 11.370 | 10.712 | 11.093 | 9.791 | 10.540 |
| 4 | 10.848 | 11.270 | 10.491 | 10.897 | 9.232 | **10.448** |
| 5 | 10.677 | 11.255 | 10.215 | 10.641 | 8.793 | 10.656 |
| 6 | 10.448 | 11.177 | 9.951 | **10.586** | 8.351 | 11.036 |
| 7 | 10.192 | **11.121** | 9.762 | 10.616 | 7.936 | 11.503 |
| 8 | 9.986 | 11.137 | 9.632 | 10.634 | 7.590 | 11.871 |

| Mask | Best valid | Epoch 8 valid | Epoch 8 minus best | Train loss ep1→8 |
|---|---:|---:|---:|---:|
| Attribute random | 11.121 (ep7) | 11.137 | +0.016 | 12.094→9.986 |
| Timestep random | 10.586 (ep6) | 10.634 | +0.048 | 11.502→9.632 |
| No mask | 10.448 (ep4) | 11.871 | +1.423 | 11.124→7.590 |

在这次单 seed 对照中，no-mask 训练 loss 下降最快，但 valid loss 从 ep4 起持续回升，符合明显过拟合；属性级和整 timestep mask 都显著减小 ep8 相对各自最佳 valid 的回升。整 timestep mask 的最低 valid 比属性级 mask 低 0.535，ep8 也低 0.503；属性级组则在 ep7 才达到最低点。no-mask 的 ep4 最低 valid 仍略低于两个 mask 组，但继续训练会快速恶化，因此早停 epoch 的选择很重要。

三种配置可用相同 loss 作横向参考，但 mask 保留的信息量不同：属性级 mask 总是保留 decoder pitch，并平均保留 1.5/3 个 performance 属性；整 timestep mask 会一并遮掉 pitch 和全部 performance 属性；no-mask 保留所有历史输入。因此当前结果支持“decoder feedback masking 能抑制后期过拟合”，但还不能归结为纯 mask 粒度的因果差异，也不能据单 seed 宣称性能差异具有统计显著性。暂未做 test MIDI 指标评估；本轮目的是比较 train/valid 轨迹。

保存策略为每 epoch checkpoint、`eval_loss` 选 best、`save_total_limit=2` 保留 best 与 latest。完成后的 checkpoint 为：attribute `checkpoint-2254`（best ep7）与 `checkpoint-2576`（last ep8）；timestep `checkpoint-1932`（best ep6）与 `checkpoint-2576`（last ep8）；no-mask `checkpoint-1288`（best ep4）与 `checkpoint-2576`（last ep8）。输出根目录不另存重复模型权重。

单配置训练命令示例使用 held-out-12-work validation 和 8 epochs：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train.dinr_absolute \
  --sidecar-root ../data/ASAP_processed \
  --pedal-mode none \
  --decoder-mask-mode attribute \
  --decoder-mask-probability 0.5 \
  --epochs 8 --batch-size 32 --learning-rate 3e-4 \
  --eval-split valid \
  --pretrained-pt ../models/official_redownload_20260711/pianist-transformer-rendering \
  --precomputed-train ../data/ASAP_processed_dinr_tokens_heldout12_v1/dinr_absolute_train_train_b512_o50.pt \
  --precomputed-valid ../data/ASAP_processed_dinr_tokens_heldout12_v1/dinr_absolute_train_valid_b512_o50.pt \
  --output-dir models/dinr_stage1_no_pedal_attrmask
```

这只是新的训练配置；此前 CE/BCE/none 的 16-epoch 结果仍对应 8 槽位和旧整音符输入 mask，不能当作本阶段结果。

## Work、Perf 和 Window

三种对象必须区分：

- **Work**：一个 score sidecar，对应一首曲目或一个乐谱。
- **Perf**：该 work 下的一条完整 aligned performance。
- **Window**：从一条 perf 对应的 note 序列中切出的训练片段。本实验使用 512 notes、stride 256，也就是 50% overlap。

这里的 window 数量指 **perf-window instance**：同一个 score window 在同一 work 的不同 perf 上会形成不同训练样本。因此 window 数不能直接解释为 work 数或 perf 数。

## 数据划分

Test 保持 work-disjoint。当前 grouped-median 数据共有 204 个 work：

| Split | Work | Perf | Perf-window instances |
|---|---:|---:|---:|
| Train 原始候选集 | 187 | 884 | 10,787 |
| Valid | 138 个 selected train works | 335 | 335 |
| Train 实际使用 | 187 | 884 | 9,831 |
| 排除的 overlap 实例 | selected train works 中的部分 perf | - | 621 |
| Test | 17 | 77 | 1,412 |

外层项目的固定验证方案是
`train_valid_asap3_nonasap05_edge50_v1`。它只从少量 selected train works 中选择验证样本，而不是为所有 train works 建立 valid。当前 grouped-median sidecar 中有一个外层方案选中的 score sidecar 不可用，因此实际覆盖 138 个 work，而不是外层摘要中的 139 个。

对每个可用 selected work，固定选择该 work 的最后一个 canonical window 作为 valid。这个 window 在该 work 的每条 train perf 上各产生一个 valid perf-window instance。与 valid window 相交的 canonical train windows 会被排除，以避免 50% overlap 造成 train/valid note 内容泄漏。未被外层方案选中的 train works 的所有 canonical windows 都保留在 train。

因此，valid 与 train 可以属于同一个 work、甚至同一条 perf，但它们的 note 范围不重叠；test 则与 train/valid 完全 work-disjoint。

### 固定 work-level validation split

为避免 edge-window validation 的曲尾位置偏差，新增固定的 work-disjoint 诊断划分：
`configs/asap_valid_heldout_12works.json`。它从原 train works 中预先留出 12 首完整作品，valid 使用这些作品的全部 canonical windows，train 完全排除这些作品的 windows；test 保持原有的 17 个 works 不变。

这 12 首作品包含 test 中的 Beethoven、Chopin、Debussy、Haydn、Liszt、Mozart，并补充 Bach、Ravel、Schubert、Rachmaninoff。使用 512 notes、50% overlap 时，valid 为 507 个 perf-window instances，train 为 10,280 个 instances。对应的预计算 cache 位于
`../data/ASAP_processed_dinr_tokens_heldout12_v1/`，训练入口为
`script/run_dinr_absolute_asap_heldout12.sh`。

其中 Debussy 在原 train split 中只有一首 work；留出它之后，train 不再包含 Debussy work，而 test 仍有一首 Debussy work。因此该作者的 valid 项是较严格的跨作品/跨作者检验，解读总体 valid loss 时应注意这一点。

此前完成的 16-epoch edge-window 诊断使用旧 cache：9,831 个 train windows 和 335 个曲尾 valid windows。它的 loss 结果对应旧划分，不能与本节 held-out-12-work 的 loss 曲线直接比较。

## Held-out-12-work 对照实验

在固定的 12-work split 上，使用 CE pedal、global batch 32、PT 初始化和相同的 50% overlap，分别训练 6 epochs：

- `dinr_absolute_ce_heldout12_nocoord_6ep`：slot embedding，无数值 coordinate。
- `dinr_absolute_ce_heldout12_coord_6ep`：slot embedding，并对 timing/velocity token 加 Fourier 数值 coordinate。

两组都在每个 epoch 结束后计算 valid loss，并在训练结束后使用相同的 17 个 test works、3 个随机 sample 和 PN Wasserstein 评估。

本次 slot-sum 使用 8 个独立的 residual `768→768` 线性映射，分别作用于 pitch、IOI、velocity、duration 和四个 pedal slot 的 value embedding，之后求和。映射矩阵零初始化，所以训练开始时不改变 PT value embedding；它避免了不同属性共享 token id 时出现的置换不变性。Coordinate 组在输入 timing token 使用秒为单位的坐标 `bin × 0.01`，velocity 使用 `bin / 127`，两者用 16 个 Fourier 频率投影到 hidden size 后加入 categorical embedding。

### 6-epoch 结果

Train/valid loss 曲线和 CSV 分别为 `analysis/dinr_heldout12_6ep_loss.png` 与 `analysis/dinr_heldout12_6ep_loss.csv`。两组 train loss 持续下降，valid loss 都先下降后小幅回升：

| Epoch | No-coord train | No-coord valid | Coordinate train | Coordinate valid |
|---:|---:|---:|---:|---:|
| 1 | 11.449 | 11.320 | 11.580 | 11.545 |
| 2 | 10.417 | 10.975 | 10.817 | 11.312 |
| 3 | 9.988 | **10.948** | 10.446 | 11.194 |
| 4 | 9.645 | 10.999 | 10.129 | **11.155** |
| 5 | 9.353 | 11.160 | 9.883 | 11.189 |
| 6 | 9.082 | 11.324 | 9.665 | 11.268 |

因此，修正为 held-out works 后仍有过拟合迹象，但幅度比旧 edge-window validation 温和：no-coord 从 epoch 3 到 6 回升 0.377；coordinate 从 epoch 4 到 6 回升 0.113。六个 epoch 内 coordinate valid loss 略好于 no-coord 的最低值，但 test PN Wasserstein 没有同步改善。

两组 17 个 test works、每首 3 个随机 sample 的采样评估结果：

| Variant | PN IOI W1 (ms) | PN Duration W1 (ms) | PN Velocity W1 | PN shared-3 mean |
|---|---:|---:|---:|---:|
| No coordinate | **121.95** | **241.54** | 26.77 | **130.09** |
| Coordinate | 171.50 | 334.94 | **26.03** | 177.49 |

Coordinate 仅在 velocity 上略好，IOI 和 duration 明显更差，shared-3 mean 比 no-coordinate 高约 36%。当前结果不支持将 coordinate 纳入这个 DINR-Absolute 变体。该评估脚本报告的 pedal Wasserstein 为 MIDI CC 原始 `0..127` 值，不与 shared-3 mean 混合用于上述比较。

## 预计算 token cache

为了避免训练时反复读取原始 sidecar 和动态构造 token，先生成共享 token cache：

```bash
python -m src.data_process.build_dinr_token_sidecars \
  --sidecar-root ../data/ASAP_processed \
  --output-dir ../data/ASAP_processed_dinr_tokens \
  --fixed-window-summary ../data/train_valid_asap3_nonasap05_edge50_v1_summary.json \
  --block-notes 512 --overlap-ratio 0.5
```

生成的主要文件为：

```text
../data/ASAP_processed_dinr_tokens/dinr_absolute_train_train_b512_o50.pt
../data/ASAP_processed_dinr_tokens/dinr_absolute_train_valid_b512_o50.pt
../data/ASAP_processed_dinr_tokens/dinr_absolute_test_b512_o50.pt
```

## 训练和测试评估

以下脚本在指定 GPU 上完成一种 pedal 模式的训练，然后执行 test inference、绝对时间 MIDI 重建和 PN Wasserstein 评估：

```bash
bash script/run_dinr_absolute_asap.sh <ce|bce|none> <gpu-index>
```

三个模式使用相同的 sidecar、window 划分、PT 初始化、batch size 32 和 16 epochs，只改变 pedal 输入、head 和 loss。训练期间每个 epoch 计算 valid loss；test 只在训练结束后运行。

训练输出目录分别为：

```text
models/dinr_absolute_ce_validlast/
models/dinr_absolute_bce_validlast/
models/dinr_absolute_none_validlast/
```

test 评估结果保存在各自的 `test_eval/metrics.json`，同时保存 prediction MIDI 和每条 test reference 的 GT MIDI。

