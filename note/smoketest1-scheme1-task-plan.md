# Smoke Test 1 方案一任务规划

目标：在 RoboTwin 上比较 5 个 OpenWAM video backbone checkpoint 的推理表现，并统计同一任务上的 `success@k/pass@k` 曲线。

本计划只覆盖 Smoke Test 1 的方案一：不重新训练，只使用 OpenWAM 已公开的 RoboTwin fine-tuned checkpoint 做快速现象验证。

## 0. 当前已知状态

本地环境：

- OpenWAM 仓库：`/home/chw/code/packages/OpenWAM/OpenWAM`
- RoboTwin 仓库：`/home/chw/code/packages/RoboTwin/RoboTwin`
- OpenWAM conda 环境：`openwam`
- RoboTwin conda 环境：`RoboTwin`
- 模型基础权重目录：`/mnt/data/chw/model/openwam`

已确认的基础 video backbone 权重：

- `/mnt/data/chw/model/openwam/Wan2.1-VACE-1.3B`
- `/mnt/data/chw/model/openwam/Cosmos-Predict2.5-2B`
- `/mnt/data/chw/model/openwam/Cosmos3-Edge`
- `/mnt/data/chw/model/openwam/Wan2.2-TI2V-5B`
- `/mnt/data/chw/model/openwam/Wan2.1-I2V-14B-480P`

需要特别注意：

- 这些是基础 video backbone 权重，不等同于可直接部署到 RoboTwin 的 OpenWAM policy checkpoint。
- RoboTwin 当前本地仓库 commit 与 OpenWAM README 中标注的 verified commit 不一致。
- OpenWAM 的 RoboTwin wrapper 期望 RoboTwin 里存在 `script/eval_policy.py`，当前本地 RoboTwin 更像是新版 `scripts/` 布局，需要先对齐版本或做兼容确认。

## 1. 总体执行顺序

推荐顺序：

1. 环境和仓库版本检查。
2. 下载 5 个 RoboTwin fine-tuned OpenWAM checkpoint。
3. 验证单个 checkpoint 能启动 policy server。
4. 验证单任务、少 episode 的 RoboTwin 闭环推理。
5. 并行跑通 5 个不同 backbone 的单任务 smoke test。
6. 实现并验证单任务 `success@k/pass@k` 统计。
7. 了解 RoboTwin 任务全集并选择最终 10 个任务。
8. 对 10 个任务运行 5 个 backbone 的 `success@k/pass@k`。
9. 汇总结果，判断是否值得进入更严格实验。

## 2. Stage 1：环境和仓库版本检查

目的：确认 OpenWAM server 环境、RoboTwin client 环境和 GPU 资源都能用于后续实验。

需要检查：

- `openwam` 环境可 import OpenWAM、torch、diffusers、transformers、websockets。
- `RoboTwin` 环境可 import torch、sapien、mplib、warp、websockets、yaml。
- GPU 空闲显存足够。
- RoboTwin 版本是否与 OpenWAM README 的 verified commit 对齐。
- RoboTwin 是否存在 OpenWAM wrapper 需要的 eval 入口。

推荐检查命令：

```bash
conda run -n openwam python -c "import torch, openwam, transformers, diffusers, websockets; print(torch.__version__, torch.cuda.is_available())"

conda run -n RoboTwin python -c "import torch, sapien, mplib, warp, websockets, yaml; print(torch.__version__, torch.cuda.is_available())"

nvidia-smi

git -C /home/chw/code/packages/RoboTwin/RoboTwin rev-parse HEAD

ls /home/chw/code/packages/RoboTwin/RoboTwin/script/eval_policy.py
```

完成检查点：

- `openwam` 环境 import 无错误。
- `RoboTwin` 环境 import 无错误。
- 至少 1 张 GPU 有足够空闲显存用于最小模型 smoke test。
- 明确 RoboTwin 是否已切到 verified commit。
- 明确 `script/eval_policy.py` 是否存在。

失败判据：

- `sapien`、`mplib`、`warp` 任一无法 import，则 RoboTwin 环境未完成。
- `script/eval_policy.py` 不存在且不切换 RoboTwin 版本，则 OpenWAM 当前 wrapper 大概率不能直接跑。

## 3. Stage 2：补齐 5 个 OpenWAM RoboTwin checkpoint

目的：下载可直接 `scripts/deploy.sh` 部署的 policy checkpoint。

需要下载的不是基础 backbone，而是 OpenWAM Study Video Backbone 里的 RoboTwin fine-tuned checkpoint：

- `robotwin_dual_system_joint_self_attention_wan21_vace_1_3b`
- `robotwin_dual_system_joint_self_attention_cosmos25`
- `robotwin_dual_system_joint_self_attention_cosmos3`
- `robotwin_dual_system_joint_self_attention`
- `robotwin_dual_system_joint_self_attention_wan21_i2v_14b`

推荐保存目录：

```text
/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/
```

建议让 OpenWAM repo 内的 `assets/openwam_ckpt` 指向上述大盘目录，避免 checkpoint 占用代码目录空间。

完成检查点：

- 每个 checkpoint 目录都存在。
- 每个 checkpoint 目录里至少包含：
  - `config.yaml`
  - 模型权重文件，例如 `.safetensors` 或 `.pt`
  - 推理所需 normalization/tokenizer 等随 checkpoint 发布的文件
- `du -sh` 能看到每个 checkpoint 体积大致合理。

失败判据：

- 只有 `/mnt/data/chw/model/openwam/*` 基础 backbone，没有 `config.yaml` 形式的 OpenWAM checkpoint，则不能进入 RoboTwin policy 推理。

## 4. Stage 3：单 checkpoint policy server 启动验证

目的：先验证一个最小闭环中的 OpenWAM server 可以加载模型。

建议优先选择体积较小的 checkpoint：

```text
robotwin_dual_system_joint_self_attention_wan21_vace_1_3b
```

推荐启动方式：

```bash
cd /home/chw/code/packages/OpenWAM/OpenWAM
conda activate openwam
CUDA_VISIBLE_DEVICES=2 bash scripts/deploy.sh /mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention_wan21_vace_1_3b --port 8848
```

完成检查点：

- server 成功启动。
- 没有 checkpoint 缺文件错误。
- 没有基础 backbone 路径缺失错误。
- WebSocket server 监听 `8848`。
- GPU 显存被正常占用。

失败判据：

- 报 `config.yaml` 不存在：说明下载的是基础 backbone，不是 OpenWAM checkpoint。
- 报某个 `/mnt/data/chw/model/openwam/...` 路径不存在：说明基础 backbone 缺文件或路径名不一致。
- 14B checkpoint OOM：先用 1.3B 验证链路，14B 后续单独调。

## 5. Stage 4：单任务 RoboTwin 闭环 smoke test

目的：验证 RoboTwin client 能连上 OpenWAM server，并完成最小 episode。

建议任务：

```text
adjust_bottle
```

建议先只跑 1 到 2 个 episode：

```bash
cd /home/chw/code/packages/OpenWAM/OpenWAM

ROBOTWIN_PATH=/home/chw/code/packages/RoboTwin/RoboTwin \
ROBOTWIN_PYTHON=/home/chw/miniconda3/envs/RoboTwin/bin/python \
ROBOTWIN_TEST_NUM=1 \
bash benchmarks/robotwin/single_eval.sh adjust_bottle demo_clean openwam_smoke 3 8848 127.0.0.1
```

完成检查点：

- RoboTwin client 能连接 `ws://127.0.0.1:8848`。
- SAPIEN 能正常 offscreen render。
- episode 能跑完。
- 日志中出现 `Success rate`。
- RoboTwin 的 `eval_result/` 或 OpenWAM 日志目录中产生结果记录。

失败判据：

- 连接超时：server 未启动、端口不对、server crash。
- `script/eval_policy.py` 找不到：RoboTwin 版本或 wrapper 路径不匹配。
- SAPIEN/EGL/Vulkan 报错：需要优先处理 RoboTwin 图形后端。
- cuRobo 报错：先尝试 verified RoboTwin commit，再考虑 fallback。

## 6. Stage 5：并行跑通 5 个 backbone 的单任务 smoke test

目的：确认 5 个 checkpoint 都能独立完成同一任务的少量 episode。

建议策略：

- 每个 checkpoint 单独启动一个 OpenWAM server。
- 每个 server 使用不同 GPU 和端口。
- RoboTwin client 逐个连接端口跑同一个任务。

建议端口规划：

| Backbone | Checkpoint 名称 | GPU | Port |
|---|---|---:|---:|
| Wan2.1-VACE-1.3B | `robotwin_dual_system_joint_self_attention_wan21_vace_1_3b` | 2 | 8848 |
| Cosmos-Predict2.5-2B | `robotwin_dual_system_joint_self_attention_cosmos25` | 3 | 8849 |
| Cosmos3-Edge-4B | `robotwin_dual_system_joint_self_attention_cosmos3` | 4 | 8850 |
| Wan2.2-TI2V-5B | `robotwin_dual_system_joint_self_attention` | 5 | 8851 |
| Wan2.1-I2V-14B | `robotwin_dual_system_joint_self_attention_wan21_i2v_14b` | 6 | 8852 |

完成检查点：

- 5 个 server 都能分别启动。
- 5 个 checkpoint 都能跑完 `adjust_bottle` 的 `ROBOTWIN_TEST_NUM=1`。
- 每个结果日志中都有 `Success rate`。
- 无端口冲突、无 GPU OOM。

失败判据：

- 某个 checkpoint 独立加载失败：先定位该 checkpoint 的 `config.yaml` 与基础模型路径。
- 多 server 并行失败但单 server 成功：优先排查显存、端口和 CPU/IO 压力。
- 14B 单独失败：先不阻塞其他 4 个 backbone，记录为高显存专项问题。

## 7. Stage 6：单任务 success@k/pass@k 验证

目的：确认统计逻辑正确。

建议先只对一个任务做：

```text
task = adjust_bottle
k = 1, 2, 4, 8, 16, 32
```

统计定义：

- 对同一个任务、同一个 checkpoint 连续采样最多 32 次。
- `success@k = 1` 表示前 `k` 次中至少有一次成功。
- 如果需要跨多个随机种子或多个任务汇总，则先保存 episode-level 成败，再由 episode-level 结果聚合，避免只保存平均成功率。

完成检查点：

- 每个 checkpoint 至少有 32 次 episode-level 成败记录。
- 能从记录中计算：
  - `success@1`
  - `success@2`
  - `success@4`
  - `success@8`
  - `success@16`
  - `success@32`
- 计算结果可人工抽查，例如前 4 次只要有一次成功，则 `success@4=1`。

失败判据：

- 只有 RoboTwin 聚合后的 `Success rate`，没有 episode-level 成败，则不能严格计算 `success@k`，需要解析更细日志或补充记录方式。
- 每次 rollout 初始状态不受控，则 `success@k` 更接近“多 episode 成功概率”，需要在报告中明确说明。

## 8. Stage 7：选择最终 10 个 RoboTwin 任务

目的：从 50 个 RoboTwin 任务里选择一组适合 smoke test 的任务。

选择原则：

- 覆盖不同操作类型：点击、抓取、放置、堆叠、开合、双臂协作。
- 避免只选过易或过难任务。
- 优先选择能在单任务 smoke test 中稳定运行、不频繁因为仿真异常失败的任务。
- 保持 `demo_clean` 优先，必要时再做 `demo_randomized`。

初始候选任务：

- `adjust_bottle`
- `click_bell`
- `open_laptop`
- `pick_dual_bottles`
- `place_can_basket`
- `place_object_stand`
- `move_stapler_pad`
- `stack_blocks_two`
- `handover_block`
- `turn_switch`

完成检查点：

- 形成 10 个任务的固定列表。
- 每个任务至少用 1 个 checkpoint 跑过 `ROBOTWIN_TEST_NUM=1`。
- 没有任务因为环境资源、资产缺失或仿真初始化稳定失败而保留在最终列表中。

失败判据：

- 某任务频繁卡在初始化、资产加载或 planner，而不是 policy 表现，则应替换任务。

## 9. Stage 8：10 任务完整 pass@k 实验

目的：产出 Smoke Test 1 方案一的核心对比结果。

实验矩阵：

```text
5 checkpoint x 10 task x 32 rollout
```

总 rollout 数：

```text
5 * 10 * 32 = 1600
```

建议先跑 `demo_clean`，确认趋势后再考虑 `demo_randomized`。

完成检查点：

- 每个 checkpoint 每个任务都有 32 次 episode-level 成败记录。
- 每个 checkpoint 都能得到 10 任务平均的：
  - `success@1`
  - `success@2`
  - `success@4`
  - `success@8`
  - `success@16`
  - `success@32`
- 同时保留 per-task 曲线，避免平均值掩盖任务差异。

失败判据：

- 任何 checkpoint 缺失超过 1 个任务结果，应先补齐再比较平均值。
- 只有不同 checkpoint 的总体成功率，没有 `k` 曲线，则不能回答当前 smoke test 的核心问题。

## 10. Stage 9：结果判断标准

主要观察：

- 大 backbone 是否在 `success@1` 或小 `k` 上更强。
- `k` 增大后，backbone 之间差距是否缩小。
- 大 backbone 是否在 `success@32` 下仍能解决更多任务。

支持“WM 主要提高采样效率”的现象：

- 大模型在 `success@1/2/4` 明显更好。
- 到 `success@16/32` 时，不同模型差距缩小。
- 各模型最终能解决的任务集合差异不大。

支持“WM 可能扩展能力边界”的现象：

- 大模型在 `success@32` 仍明显领先。
- 大模型能稳定解决小模型从未成功的任务。
- 差异集中在需要更长时序推理或复杂几何理解的任务上。

需要谨慎解释的情况：

- 不同 checkpoint 不只改变 WM 大小，也改变 backbone 类型、预训练权重、VAE/visual latent 和训练状态。
- 因此方案一只能作为快速现象观察，不能单独证明因果关系。

## 11. 最终交付物

推荐最终保存：

- 每个 checkpoint 的 server 启动日志。
- 每个任务的 RoboTwin eval 日志。
- episode-level 成败表。
- per-task `success@k` 表。
- 10 任务平均 `success@k` 表。
- 一张 5 个 backbone 的 `success@k` 曲线图。
- 一段结论：现象是否值得继续做 Smoke Test 1 方案二。

