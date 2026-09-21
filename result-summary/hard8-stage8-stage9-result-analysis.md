# Smoke Test 1 Scheme 1: Hard8 Result Analysis

生成时间：2026-09-21

## 1. 实验定位

本结果对应 `/home/chw/code/packages/OpenWAM/note/smoketest1-scheme1-task-plan.md` 中的 Smoke Test 1 方案一：不重新训练，只使用 5 个已发布的 RoboTwin fine-tuned OpenWAM checkpoint，在 RoboTwin `demo_clean` 上做快速现象验证。

原计划 Stage 8 写的是 `5 checkpoint x 10 task x 32 rollout = 1600`。本次实际执行的是后续困难任务筛选后的 **Hard8** 版本：

```text
5 checkpoint x 8 task x 32 rollout = 1280 episodes
```

这仍然满足计划的核心要求：每个 checkpoint、每个任务都有 32 条 episode-level 成败记录，可以严格计算 `success@1/2/4/8/16/32`。

结果目录：

```text
/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/result/hard8_stage8_20260917_072352
```

核心文件：

```text
episodes/*.jsonl
summaries/per_task_passk.csv
summaries/backbone_average_passk.csv
notes/stage9-hard8-analysis.md
videos/video_sources.json
```

## 2. 数据完整性

本次评测已经完整跑完：

```text
jobs complete: 40 / 40
episodes:      1280 / 1280
episode files: 40
```

每个 `(backbone, task)` 都有 32 条 JSONL 记录。此前调度中出现过“job 未满 32 条就进入下一阶段”的问题，但后续通过 resume 补跑已经补齐，最终 CSV 是完整的。

注意：`videos/video_sources.json` 显示 8 个任务的成功视频均为 `missing`。这不影响 pass@k 和 success rate 统计，但说明“每个任务保留 1 个成功视频”的交付项还没有完成。

## 3. 任务与模型

本次 8 个困难任务：

```text
put_bottles_dustbin
open_microwave
stack_blocks_three
stack_bowls_three
blocks_ranking_rgb
hanging_mug
put_object_cabinet
handover_block
```

5 个 checkpoint：

```text
wan21_vace_1_3b
cosmos25
cosmos3
wan22_ti2v_5b
wan21_i2v_14b
```

这些任务比最初 Stage 7 里的简单候选更难，覆盖了容器放置、开合、堆叠、排序、悬挂、柜体交互和双臂交接。这个选择符合后续“需要拉开区分度”的判断。

## 4. Backbone 平均表现

按 8 个任务、每任务 32 次 rollout 汇总。下表按 WM/backbone 参数规模从大到小排序，不按性能排序：

| backbone | WM/backbone 参数规模 | episodes | successes | success rate | success@1 | success@2 | success@4 | success@8 | success@16 | success@32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wan21_i2v_14b | 14B | 256 | 218 | 85.16% | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| wan22_ti2v_5b | 5B | 256 | 205 | 80.08% | 0.875 | 0.875 | 1.00 | 1.00 | 1.00 | 1.00 |
| cosmos3 | 4B | 256 | 183 | 71.48% | 0.875 | 0.875 | 1.00 | 1.00 | 1.00 | 1.00 |
| cosmos25 | 2.5B | 256 | 202 | 78.91% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| wan21_vace_1_3b | 1.3B | 256 | 212 | 82.81% | 0.625 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

如果只看逐 episode 成功率，最高的是 `wan21_i2v_14b`，最低的是 `cosmos3`，最高与最低相差约 13.67 个百分点。但这个性能排序并不等于参数规模排序：`wan21_vace_1_3b` 虽然参数最小，平均成功率排到第二；`cosmos3` 参数规模大于 `cosmos25` 和 `wan21_vace_1_3b`，但平均成功率最低。

## 5. pass@k 曲线解读

原计划 Stage 9 的核心问题是：

- 大 backbone 是否在 `success@1` 或小 `k` 上更强？
- `k` 增大后，backbone 之间差距是否缩小？
- 大 backbone 是否在 `success@32` 下仍能解决更多任务？

本次结果对这三个问题的回答比较明确。

第一，`success@32` 没有区分度：

```text
5 个 backbone 的 success@32 全部为 1.0
```

也就是说，在这 8 个任务上，只要允许每个任务采样 32 次，所有 backbone 都能在每个任务上至少成功一次。因此本实验 **不支持** “大 backbone 在 success@32 下解决更多任务” 这个能力边界扩展结论。

第二，`success@4` 也已经完全饱和：

```text
5 个 backbone 的 success@4 全部为 1.0
```

这说明本次 Hard8 虽然比早期候选任务难，但从 pass@k 的角度看仍然不够难：多数差异在前 1-2 次采样内就被抹平了。

第三，`success@1` 与 `success@2` 有一些差异，但方向不完全符合“大模型必然更好”：

```text
success@1:
cosmos25        1.000
wan22_ti2v_5b   0.875
cosmos3         0.875
wan21_i2v_14b   0.750
wan21_vace_1_3b 0.625
```

`cosmos25` 在 `success@1` 上最好，但它的逐 episode 平均成功率不是最高。`wan21_i2v_14b` 的总成功率最高，但 `success@1` 只有 0.75。这说明 `success@1` 很受每个任务首个 seed 偶然性的影响，不适合单独作为 backbone 能力排序依据。

更稳妥的解释是：本次结果支持“多次采样很快能消除任务级失败”，但不支持“模型大小单调决定小 k 表现”。

## 6. 任务难度分析

按 5 个 backbone 的平均逐 episode 成功率排序：

| task | successes / 160 | mean success rate | success@1 mean | 难度判断 |
|---|---:|---:|---:|---|
| hanging_mug | 71 / 160 | 44.38% | 0.20 | 最难，最有区分度 |
| open_microwave | 106 / 160 | 66.25% | 1.00 | 中等偏难 |
| put_object_cabinet | 118 / 160 | 73.75% | 0.80 | 中等 |
| stack_bowls_three | 125 / 160 | 78.12% | 0.80 | 中等 |
| stack_blocks_three | 146 / 160 | 91.25% | 1.00 | 偏易 |
| put_bottles_dustbin | 147 / 160 | 91.88% | 1.00 | 偏易 |
| handover_block | 149 / 160 | 93.12% | 0.80 | 偏易 |
| blocks_ranking_rgb | 158 / 160 | 98.75% | 1.00 | 过易 |

任务层面最关键的结论：

- `hanging_mug` 明显最难，平均成功率只有 44.38%，也是最能拉开模型差异的任务。
- `blocks_ranking_rgb` 几乎饱和，160 次里成功 158 次，不适合继续作为区分 backbone 的主任务。
- `put_bottles_dustbin`、`handover_block`、`stack_blocks_three` 的平均成功率都超过 90%，也偏容易。
- `open_microwave`、`put_object_cabinet`、`stack_bowls_three` 处在较合理的中间难度。

如果后续要进一步提高区分度，应该保留或增加类似 `hanging_mug` 的任务，并减少 `blocks_ranking_rgb` 这类接近饱和的任务。

## 7. 各任务最佳 backbone

按逐 episode 成功率看，每个任务的最高者如下：

| task | best backbone | success rate |
|---|---|---:|
| blocks_ranking_rgb | cosmos25 / cosmos3 / wan21_i2v_14b / wan22_ti2v_5b 并列 | 100.00% |
| handover_block | wan22_ti2v_5b | 100.00% |
| hanging_mug | wan21_i2v_14b | 59.38% |
| open_microwave | wan21_vace_1_3b | 81.25% |
| put_bottles_dustbin | wan21_i2v_14b / wan22_ti2v_5b 并列 | 96.88% |
| put_object_cabinet | cosmos25 | 90.62% |
| stack_blocks_three | wan21_i2v_14b | 100.00% |
| stack_bowls_three | wan21_i2v_14b | 87.50% |

`wan21_i2v_14b` 在 3 个非饱和任务上表现最好或并列最好，且总体成功率最高。这支持它在这组任务上具有较强的平均 rollout 成功能力。

但也有反例：

- `open_microwave` 最好的是 `wan21_vace_1_3b`。
- `put_object_cabinet` 最好的是 `cosmos25`。
- `handover_block` 最好的是 `wan22_ti2v_5b`。

因此不能简单得出“大 backbone 在所有任务都更好”的结论。差异更像是 backbone 类型、训练状态和任务类型共同作用。

## 8. 对原计划判断标准的回应

### 是否支持“WM 主要提高采样效率”？

部分支持，但证据不强。

计划中预期的现象是：大模型在 `success@1/2/4` 更好，到 `success@16/32` 差距缩小。本次确实看到 `success@32` 完全收敛，`success@4` 也完全收敛，说明多次采样后模型差异会很快被抹平。

但是小 k 上不是大模型单调更好。`cosmos25` 的 `success@1=1.0`，反而高于 `wan21_i2v_14b=0.75`。所以只能说“多采样抹平差异”成立，不能说“大 WM 明显提高小 k 采样效率”成立。

### 是否支持“WM 扩展能力边界”？

不支持。

所有模型在所有任务上 `success@32=1.0`，没有任何任务是只有大 backbone 能解决、小 backbone 完全不能解决的。因此这组任务没有测出能力边界扩展。

### 任务是否足够难？

从逐 episode 成功率看，任务比 `adjust_bottle` 这类 smoke 任务难很多；但从 pass@k 看，仍然偏容易。

最明显的问题是：

```text
success@4 全部饱和
success@32 全部饱和
```

这意味着本任务集适合比较平均成功率，但不适合强力比较 pass@k 曲线。

## 9. 方法学注意事项

本次是方案一，不能单独证明因果关系。原因包括：

- 5 个 checkpoint 不只改变 video backbone 尺寸，也改变 backbone 类型和预训练分布。
- 每个任务只有 32 次 rollout，`success@1` 对 seed 顺序很敏感。
- RoboTwin 环境中存在初始化或 planner fallback 异常，虽然最终补齐了 episode，但这会增加 wall-clock 和 seed 分布复杂度。
- `success@k` 是按每个 `(backbone, task)` 的前 k 条 episode 判断，若 episode 初始状态分布本身有偶然性，小 k 指标需要谨慎解释。
- 成功视频没有收集到，缺少直观行为审查证据。

因此本次结果应定位为“现象性 smoke test”，而不是严格消融实验。

## 10. 后续建议

### 10.1 继续保留的任务

建议保留：

```text
hanging_mug
open_microwave
put_object_cabinet
stack_bowls_three
```

这些任务成功率没有完全饱和，能提供更多区分度。

### 10.2 建议替换或降权的任务

建议替换或降权：

```text
blocks_ranking_rgb
handover_block
put_bottles_dustbin
stack_blocks_three
```

其中 `blocks_ranking_rgb` 最接近天花板，平均成功率 98.75%，继续使用的区分度很低。

### 10.3 下一轮实验设计

如果目标是比较 backbone 能力边界，下一轮应：

1. 继续选择更难任务，特别是能让部分模型 `success@32=0` 或明显低于 1 的任务。
2. 或切换到 `demo_randomized`，提高初始状态和视觉扰动难度。
3. 保留 episode-level JSONL，并额外保存成功/失败视频，便于判断失败是否来自 policy、planner 还是环境初始化。
4. 除 pass@k 外，继续报告逐 episode success rate，因为本次证明在任务较容易时 pass@k 很快饱和。
5. 若要验证 WM 尺寸因果，应进入更严格的方案二：尽量固定训练数据、policy head、action space、训练步数，只改变 video backbone 或 WM 相关变量。

## 11. 总结结论

本次 Hard8 完整评测已经满足 Smoke Test 1 方案一的核心数据要求：5 个 checkpoint、8 个任务、每组 32 次 rollout，共 1280 条 episode-level 记录。

最重要的结论是：

```text
1. wan21_i2v_14b 的平均逐 episode 成功率最高，为 85.16%。
2. cosmos3 平均最低，为 71.48%。
3. success@4 和 success@32 全部饱和，pass@k 曲线区分度不足。
4. 没有证据表明大 backbone 在 success@32 上扩展了任务能力边界。
5. hanging_mug 是本组任务中最有区分度的任务。
6. blocks_ranking_rgb 等任务偏容易，后续应替换或降权。
```

因此，本次结果支持继续做更严格实验，但下一轮任务集需要更难，或者切到 randomized 设置；否则 pass@k 会过早饱和，难以回答 backbone 能力边界问题。
