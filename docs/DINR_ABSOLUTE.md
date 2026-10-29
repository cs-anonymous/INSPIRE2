# DINR-Absolute

这是一个独立的 note-level 离散基线。原始 PT 的 token-level 模型和训练脚本保持不变。

## 数据表示

训练和评估只读取 grouped-median 预处理产生的
`score_ASAP_refined.ASAP_GROUPED_MEDIAN.pt`。IOI 和 duration 使用预处理后的绝对毫秒值；正 score-IOI 音符预测 grouped IOI，总 grouped IOI 在重建时再展开为 note-level IOI；score-IOI 为零的音符使用 chord IOI。时间值量化为 10 ms，使用 500 个类别，范围为 `0..4990 ms`。velocity 量化为 MIDI 值 `0..127`，四个 pedal 采样点以 64 为阈值转成二值标签。

DINR 的紧凑词表共有 763 个 ID：pitch 为 `5..132`，velocity 为 `133..260`，IOI 和 duration 共用 timing ID `261..760`，pedal 二值 ID 为 `761..762`。grouped IOI 与 chord IOI 使用不同的输出 head；四个 pedal 位置共享两个二值 token。score 和 shifted performance 的 8 个属性 token embedding 分别求和，得到每个 note 的一个输入 embedding。模型不使用 coord。

损失为：

```text
L_ioi = L_grouped_ioi + L_chord_ioi
L_pedal = mean(L_pedal_1, L_pedal_2, L_pedal_3, L_pedal_4)
L_total = L_ioi + L_duration + L_velocity + L_pedal
```

`--pedal-mode ce` 使用每个 pedal 两个 logits 的二分类 CE；`--pedal-mode bce` 使用一个 logit 的 BCEWithLogitsLoss；`--pedal-mode none` 移除 pedal 输入、输出 head 和 pedal loss，此时不包含 `L_pedal`。

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

训练使用 9,831 个 train perf-window instances，valid 使用 335 个 valid perf-window instances，并在每个 epoch 结束后计算 valid loss。17 个 test works、77 条 test perfs 不参与训练，也不参与 epoch 内的 valid loss；训练结束后才进行完整 perf 的 test inference、MIDI 重建和 PN Wasserstein 评估。

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

