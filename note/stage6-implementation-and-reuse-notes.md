# Stage 6 实现与复用说明

本文档记录 `smoketest1-scheme1-task-plan.md` 中 Stage 6 的实际实现状态、运行产物、可复用代码与后续迭代建议。目标是后续继续做 Stage 7/Stage 8 或扩展更多任务时，可以直接复用当前环境和脚本，而不需要重新摸索。

## 1. Stage 6 已实现什么

Stage 6 的目标是：

- 在 RoboTwin 单任务上跑 5 个 OpenWAM video backbone。
- 每个 backbone 采集 32 条 episode-level 成败记录。
- 基于 episode-level 结果计算 `success@k/pass@k`。
- 验证后续多任务实验所需的推理服务、RoboTwin 评测、日志记录和汇总链路。

本次实际跑通的任务为：

```text
task_name   = adjust_bottle
task_config = demo_clean
k           = 1, 2, 4, 8, 16, 32
```

已跑通的 5 个 backbone：

```text
wan21_vace_1_3b
cosmos25
cosmos3
wan22_ti2v_5b
wan21_i2v_14b
```

最终结果目录：

```text
/home/chw/code/packages/OpenWAM/chw-code/runs/stage6_adjust_bottle_20260916_085151
```

核心结果文件：

```text
/home/chw/code/packages/OpenWAM/chw-code/runs/stage6_adjust_bottle_20260916_085151/passk_summary.csv
/home/chw/code/packages/OpenWAM/chw-code/runs/stage6_adjust_bottle_20260916_085151/passk_summary.json
```

最终汇总结果：

| backbone | episodes | successes | success_rate | success@1 | success@2 | success@4 | success@8 | success@16 | success@32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cosmos25 | 32 | 32 | 1.0 | 1 | 1 | 1 | 1 | 1 | 1 |
| cosmos3 | 32 | 32 | 1.0 | 1 | 1 | 1 | 1 | 1 | 1 |
| wan21_i2v_14b | 32 | 32 | 1.0 | 1 | 1 | 1 | 1 | 1 | 1 |
| wan21_vace_1_3b | 32 | 32 | 1.0 | 1 | 1 | 1 | 1 | 1 | 1 |
| wan22_ti2v_5b | 32 | 31 | 0.96875 | 1 | 1 | 1 | 1 | 1 | 1 |

Stage 6 的验收点已经满足：

- 5 个 backbone 都有 32 条 episode-level 记录。
- 每条记录包含 `task_name`、`task_config`、`ckpt_setting`、`episode_id`、`seed`、`success`。
- 已生成 `success@1/2/4/8/16/32` 汇总。
- 评测结束后无残留 `single_eval.sh`、`eval_policy_wrapper.py`、OpenWAM `deploy.py` 服务进程。

## 2. 运行产物结构

Stage 6 结果目录下主要文件分为四类。

### 2.1 Episode 记录

每个 `*_episodes.jsonl` 文件是一组 episode-level 成败记录。例如：

```text
wan21_vace_1_3b_episodes.jsonl
wan21_vace_1_3b_resume_episodes.jsonl
cosmos25_episodes.jsonl
cosmos25_resume_episodes.jsonl
cosmos3_episodes.jsonl
cosmos3_resume_episodes.jsonl
wan22_ti2v_5b_episodes.jsonl
wan21_i2v_14b_episodes.jsonl
```

部分 backbone 有 `_resume_episodes.jsonl`，这是因为中途断点续跑过。汇总脚本会把同一个 backbone 的主文件和 resume 文件合并统计。

### 2.2 RoboTwin 评测日志

```text
<backbone>.log
<backbone>_resume.log
```

这些文件记录 RoboTwin 侧的环境初始化、每个 episode 的成功/失败、当前 seed、累计成功率等信息。

### 2.3 OpenWAM 服务端日志

```text
<backbone>_server.log
```

这些文件记录 OpenWAM checkpoint 加载、模型架构、dtype、normalizer、WebSocket 服务启动、推理过程等信息。排查模型加载失败、显存不足、端口未监听时优先看这里。

### 2.4 pass@k 汇总

```text
passk_summary.csv
passk_summary.json
```

这是当前 Stage 6 最关键的结果文件。

## 3. 已复用和新增的代码

遵循“不修改原始源代码”的约束，本次 Stage 6 使用的新增/调整内容都在：

```text
/home/chw/code/packages/OpenWAM/chw-code
```

### 3.1 OpenWAM 服务启动脚本

```text
/home/chw/code/packages/OpenWAM/chw-code/start_openwam_server.sh
```

作用：

- 激活 `openwam` conda 环境。
- 进入 `/home/chw/code/packages/OpenWAM/OpenWAM`。
- 设置 `CUDA_VISIBLE_DEVICES`。
- 调用 `python scripts/deploy.py` 启动 WebSocket policy server。
- 将服务端日志写到指定文件。

调用形式：

```bash
/home/chw/code/packages/OpenWAM/chw-code/start_openwam_server.sh CKPT_DIR GPU PORT LOG_FILE
```

后续可以继续复用它启动任意 OpenWAM checkpoint，只要 checkpoint 目录结构和当前 5 个 backbone 一致。

### 3.2 Stage 6 顺序断点续跑脚本

```text
/home/chw/code/packages/OpenWAM/chw-code/run_stage6_remaining_sequential.sh
```

作用：

- 固定任务为 `adjust_bottle demo_clean`。
- 顺序跑 5 个 backbone。
- 每个 backbone 检查当前已有 episode 数。
- 如果已有记录数达到 `TARGET_N=32`，则跳过。
- 如果不足 32，则启动对应 OpenWAM server，补齐缺失 episode。
- 每个 backbone 结束后关闭对应 server。
- 全部完成后自动调用 `summarize_passk.py`。

常用运行方式：

```bash
RUN_ROOT=/home/chw/code/packages/OpenWAM/chw-code/runs/stage6_adjust_bottle_20260916_085151 \
SERVER_GPU=2 \
SIM_GPU=7 \
PORT=8848 \
/home/chw/code/packages/OpenWAM/chw-code/run_stage6_remaining_sequential.sh
```

建议后续优先复用这个脚本的结构，因为它有断点续跑逻辑，适合长任务和大模型。

### 3.3 Stage 6 初始批量脚本

```text
/home/chw/code/packages/OpenWAM/chw-code/run_stage6_adjust_bottle.sh
```

作用：

- 创建新的 run 目录。
- 依次调用 RoboTwin 单任务评测。
- 支持通过环境变量覆盖 `TASK_NAME`、`TASK_CONFIG`、`TEST_NUM`、`SIM_GPU` 等。

注意：

- 这个脚本更像初版批量入口。
- 当前更推荐继续使用 `run_stage6_remaining_sequential.sh`，因为它能检查已有 episode 数并 resume。

### 3.4 pass@k 汇总脚本

```text
/home/chw/code/packages/OpenWAM/chw-code/summarize_passk.py
```

作用：

- 扫描某个 run 目录下所有 `*_episodes.jsonl`。
- 自动把 `_resume`、`_part2`、`_part3` 后缀归并到同一个 backbone。
- 计算：
  - `episodes`
  - `successes`
  - `success_rate`
  - `success@1`
  - `success@2`
  - `success@4`
  - `success@8`
  - `success@16`
  - `success@32`
- 输出：
  - `passk_summary.json`
  - `passk_summary.csv`

手动重算命令：

```bash
python /home/chw/code/packages/OpenWAM/chw-code/summarize_passk.py \
  /home/chw/code/packages/OpenWAM/chw-code/runs/stage6_adjust_bottle_20260916_085151
```

### 3.5 RoboTwin 兼容副本

```text
/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat_copy
```

作用：

- 作为 RoboTwin runtime 兼容副本使用。
- 避免修改原始 RoboTwin/OpenWAM 源代码。
- Stage 6 使用这个副本完成 episode-level 日志写入和兼容修补。

当前关键点：

- `task_config/demo_clean.yml` 中关闭了 eval video log，用于降低视频写入开销。
- `script/eval_policy.py` 中支持通过 `ROBOTWIN_EPISODE_LOG` 追加 episode-level JSONL 记录。
- `ROBOTWIN_EXPERT_CHECK` 保持为 1。不要随意关掉它；之前关掉会导致 instruction 生成依赖的 `episode_info` 缺失。

## 4. 当前环境和模型路径

### 4.1 Conda 环境

OpenWAM 推理环境：

```text
openwam
```

RoboTwin 仿真环境：

```text
/home/chw/miniconda3/envs/RoboTwin/bin/python
```

Stage 6 脚本默认使用：

```bash
ROBOTWIN_PYTHON=/home/chw/miniconda3/envs/RoboTwin/bin/python
ROBOTWIN_PATH=/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat_copy
OPENWAM_ROOT=/home/chw/code/packages/OpenWAM/OpenWAM
```

### 4.2 模型 checkpoint

Stage 6 使用的 checkpoint base：

```text
/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone
```

具体映射：

```text
wan21_vace_1_3b -> robotwin_dual_system_joint_self_attention_wan21_vace_1_3b
cosmos25        -> robotwin_dual_system_joint_self_attention_cosmos25
cosmos3         -> robotwin_dual_system_joint_self_attention_cosmos3
wan22_ti2v_5b   -> robotwin_dual_system_joint_self_attention
wan21_i2v_14b   -> robotwin_dual_system_joint_self_attention_wan21_i2v_14b
```

## 5. 如何复用 Stage 6 成果继续往下跑

### 5.1 继续补跑或重算当前 Stage 6

如果只是想确认或修复当前 `adjust_bottle` 的 Stage 6，可以直接运行：

```bash
RUN_ROOT=/home/chw/code/packages/OpenWAM/chw-code/runs/stage6_adjust_bottle_20260916_085151 \
SERVER_GPU=2 \
SIM_GPU=7 \
PORT=8848 \
/home/chw/code/packages/OpenWAM/chw-code/run_stage6_remaining_sequential.sh
```

如果 5 个 backbone 都已有 32 条记录，脚本会全部 skip，然后重新生成 `passk_summary.json/csv`。

### 5.2 换任务继续跑单任务 pass@k

当前 `run_stage6_remaining_sequential.sh` 里任务名写死为：

```text
adjust_bottle demo_clean
```

如果后续要跑其他任务，建议不要改原始 OpenWAM/RoboTwin 代码，而是在 `chw-code` 下复制一份新的 runner，例如：

```text
/home/chw/code/packages/OpenWAM/chw-code/run_passk_single_task.sh
```

建议改造成环境变量形式：

```bash
TASK_NAME=place_can_basket
TASK_CONFIG=demo_clean
RUN_ROOT=/home/chw/code/packages/OpenWAM/chw-code/runs/stage6_place_can_basket_YYYYMMDD_HHMMSS
TARGET_N=32
SERVER_GPU=2
SIM_GPU=7
PORT=8848
```

这样 Stage 7/Stage 8 可以复用同一套 runner，只替换任务名和输出目录。

### 5.3 扩展到 Stage 8 的 10 任务实验

Stage 8 可以组织成：

```text
task x backbone x 32 episodes
```

建议目录结构：

```text
/home/chw/code/packages/OpenWAM/chw-code/runs/stage8_10tasks_YYYYMMDD_HHMMSS/
  adjust_bottle/
    passk_summary.csv
    *_episodes.jsonl
  place_can_basket/
    passk_summary.csv
    *_episodes.jsonl
  ...
  all_tasks_summary.csv
```

建议实现方式：

1. 先在 `chw-code` 下做一个通用单任务 runner。
2. 每个任务单独一个 `RUN_ROOT` 子目录。
3. 每个任务跑完后调用 `summarize_passk.py`。
4. 最后再写一个跨任务汇总脚本，把每个任务的 `passk_summary.csv` 合并为 `all_tasks_summary.csv`。

## 6. 快速检查命令

### 6.1 检查 episode 数量

```bash
wc -l /home/chw/code/packages/OpenWAM/chw-code/runs/stage6_adjust_bottle_20260916_085151/*episodes.jsonl
```

### 6.2 重算 pass@k

```bash
python /home/chw/code/packages/OpenWAM/chw-code/summarize_passk.py \
  /home/chw/code/packages/OpenWAM/chw-code/runs/stage6_adjust_bottle_20260916_085151
```

### 6.3 查看汇总结果

```bash
cat /home/chw/code/packages/OpenWAM/chw-code/runs/stage6_adjust_bottle_20260916_085151/passk_summary.csv
```

### 6.4 检查是否有残留评测进程

```bash
ps -eo pid,ppid,stat,etime,pcpu,pmem,args \
  | awk '/run_stage6_remaining|eval_policy_wrapper.py|single_eval.sh|scripts\\/deploy.py|start_openwam_server/ && !/awk/ {print}'
```

如果没有输出，说明没有当前 Stage 6 残留进程。

### 6.5 检查端口

```bash
ss -ltnp | egrep ':8848\\b|:8849\\b|:8850\\b|:8851\\b|:8852\\b' || true
```

## 7. 已知注意事项

### 7.1 断点续跑导致部分重复 seed

本次 Stage 6 中，部分 backbone 因为中途断点续跑，RoboTwin 的 `single_eval.sh` 默认从 `--seed 0` 派生起始 seed，所以部分模型存在重复 seed。

当前检查结果：

```text
cosmos25:        episodes=32 unique_seeds=25 successes=32 duplicate_seeds=7
cosmos3:         episodes=32 unique_seeds=29 successes=32 duplicate_seeds=3
wan21_i2v_14b:   episodes=32 unique_seeds=32 successes=32 duplicate_seeds=0
wan21_vace_1_3b: episodes=32 unique_seeds=22 successes=32 duplicate_seeds=10
wan22_ti2v_5b:   episodes=32 unique_seeds=32 successes=31 duplicate_seeds=0
```

这不影响“每个 backbone 有 32 条 episode-level 记录并能计算 pass@k”的 Stage 6 基本验收。但如果后续论文/正式实验要求 32 个唯一随机种子，建议在 `chw-code` 下补一个支持 seed offset 的 runner，或者直接调用 `eval_policy_wrapper.py` 传不同 `--seed` 补齐唯一 seed。

### 7.2 不建议关闭 expert check

`ROBOTWIN_EXPERT_CHECK=1` 应保持开启。关闭后可能导致 instruction 生成依赖的 `episode_info` 未定义，从而触发错误。

### 7.3 14B 显存情况

`wan21_i2v_14b` 已在当前机器上单卡加载成功。运行时 OpenWAM server 侧约使用 57GB 显存，当前脚本使用：

```text
SERVER_GPU=2
SIM_GPU=7
PORT=8848
```

后续跑多任务时，建议先使用顺序执行，确认稳定后再考虑多 GPU 并行。

### 7.4 不要修改原始源代码

后续所有实验性改动建议继续放在：

```text
/home/chw/code/packages/OpenWAM/chw-code
```

不要直接改：

```text
/home/chw/code/packages/OpenWAM/OpenWAM
```

除非明确决定要把某个稳定改动上游化。

## 8. 下一步建议

建议按如下顺序继续：

1. Stage 7：梳理 RoboTwin 任务列表，选 10 个任务。
2. 在 `chw-code` 下抽象通用单任务 runner，支持 `TASK_NAME`、`TASK_CONFIG`、`RUN_ROOT`、`TARGET_N`。
3. 对候选任务先每个跑 1 个 backbone、少量 episode，排除容易仿真异常的任务。
4. 对最终 10 个任务跑完整 `5 backbone x 32 episodes`。
5. 生成跨任务总表，统计每个 backbone 在 10 个任务上的 `success@k` 表现。

当前 Stage 6 已证明：

- 5 个 backbone checkpoint 都能加载。
- OpenWAM server 和 RoboTwin client 能稳定通信。
- `adjust_bottle demo_clean` 可以完成完整 pass@k 采样。
- `chw-code` 下的 runner 和 summary 脚本可以作为后续 Stage 8 的基础模板。
