# Stage 8/9 困难 8 任务并行评测实施计划

更新时间：2026-09-17

目标：在不修改 OpenWAM/RoboTwin 原始代码的前提下，用当前可用的 3 张 GPU 尽快完成 Smoke Test 1 方案一的困难 8 任务完整评测，并产出 Stage 9 分析结果。

新增脚本、兼容逻辑、日志和结果只放在：

```text
/home/chw/code/packages/OpenWAM/chw-code
```

正式结果放在：

```text
/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/result
```

不直接修改：

```text
/home/chw/code/packages/OpenWAM/OpenWAM
/home/chw/code/packages/RoboTwin/RoboTwin
```

## Stage 8.0：固定任务与实验矩阵

困难 8 任务：

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

5 个 OpenWAM RoboTwin checkpoint：

```text
wan21_vace_1_3b
cosmos25
cosmos3
wan22_ti2v_5b
wan21_i2v_14b
```

完整评测规模：

```text
5 backbone x 8 task x 32 rollout = 1280 episodes
```

Stage 8 输出：

- 每个 `(backbone, task)` 一个 JSONL，共 40 个。
- 每个 JSONL 目标 32 条 episode 记录。
- 每个困难任务保留 1 个成功视频，共 8 个 mp4。
- 汇总 `success@1/2/4/8/16/32`。

检查点：

- `result/<run_name>/config/hard8_tasks.txt` 有 8 个任务。
- `result/<run_name>/episodes/*.jsonl` 最终有 40 个文件。
- `wc -l result/<run_name>/episodes/*.jsonl` 每个目标为 32 行。

## Stage 8.1：GPU slot smoke test

用户更正后，当前只按 3 张可用 GPU 执行：

```text
GPU 1
GPU 2
GPU 7
```

smoke 目的不是测最终成功率，而是判断每张卡最多能同时承载多少个 `server + RoboTwin simulator` worker，并且每个 worker 至少能完成 1 条真实 episode。

已启动 smoke：

```bash
/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/smoke_gpu_slots.sh
```

smoke 结果目录：

```text
/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/result/hard8_gpu_slot_smoke_20260917_063604
```

2026-09-17 已完成 smoke，结果：

```text
GPU 1: 1 slot ok, 2 slots ok, 3 slots ok, best=3
GPU 2: 1 slot ok, 2 slots ok, 3 slots ok, best=3
GPU 7: 1 slot ok, 2 slots ok, 3 slots ok, best=3
```

检查点：

- `gpu_slot_results.csv` 每张 GPU 都有 `best` 行。
- 每个通过的 slot 都有 1 行 JSONL episode 记录。
- 每个通过的 slot 都有独立 server log 和 eval log。
- 通过标准必须是真实 episode 完成，不能只看 server 端口已启动。

## Stage 8.2：正式并行策略

调度脚本：

```text
/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/run_hard8_parallel_stage8.py
```

已启动正式 tmux run：

```text
tmux session: stage8_hard8_parallel
run_name: hard8_stage8_20260917_072352
run_root: /home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/result/hard8_stage8_20260917_072352
```

正式评测采用按 backbone 分阶段执行：

1. 小/中模型使用 smoke 得出的多 slot 并发。
2. 大模型使用保守并发，避免 5B/14B 在同一张卡多开导致 OOM。
3. 所有阶段写入同一个 `run_root`，因此可以统一汇总和断点续跑。

当前预案：

```text
wan21_vace_1_3b: 使用 smoke 得出的最大 slot，即 1,1,1,2,2,2,7,7,7
cosmos25:         使用 smoke 得出的最大 slot，即 1,1,1,2,2,2,7,7,7
cosmos3:          使用 smoke 得出的最大 slot，即 1,1,1,2,2,2,7,7,7
wan22_ti2v_5b:    先按每张 GPU 1 slot
wan21_i2v_14b:    先按每张 GPU 1 slot
```

如果 5B 实测显存和吞吐都允许，可以后续单独 resume 提高并发；如果 14B OOM，则保留已完成 JSONL，降并发后续跑未完成 job。

检查点：

- `config/gpu_plan.json` 记录默认 slot。
- `config/gpu_slot_overrides.json` 记录 5B/14B 的保守 slot。
- `status.csv` 能看到 40 个 job 的完成行数。
- 中断后重新运行不会覆盖已有 episode，只补不足 32 行的 job。

## Stage 8.3：tmux 启动方式

tmux session 名称：

```text
stage8_hard8_parallel
```

待 smoke 完成后启动示例：

```bash
cd /home/chw/code/packages/OpenWAM
rm -rf /home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/chw-choose-sample
tmux new-session -d -s stage8_hard8_parallel \
  'python /home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/run_hard8_parallel_stage8.py \
    --phase-by-backbone \
    --gpu-slots "1,1,1,2,2,2,7,7,7" \
    --gpu-slots-override "wan22_ti2v_5b=1,2,7" \
    --gpu-slots-override "wan21_i2v_14b=1,2,7" \
    --base-port 9100 \
    --target-n 32'
```

如果 smoke 发现某张卡只能跑 2 slot，则把默认 slot 改成类似：

```text
1,1,1,2,2,7,7,7
```

如果某张卡只能跑 1 slot，则只保留一次该 GPU id。

检查点：

- `tmux ls` 能看到 `stage8_hard8_parallel`。
- `result/hard8_stage8_<timestamp>/status.csv` 持续更新。
- `logs/worker_*.log` 持续追加。
- `nvidia-smi` 能看到 1/2/7 三张卡被使用。

## Stage 8.4：断点续跑规则

每个 job 的目标文件：

```text
episodes/<backbone>__<task>.jsonl
```

resume 规则：

- 文件不存在：从 0 开始跑。
- 已有 N 行且 `N < 32`：只补 `32 - N` 条。
- 已有 N 行且 `N >= 32`：跳过。

每条 episode 至少需要包含：

```json
{
  "task_name": "...",
  "task_config": "demo_clean",
  "ckpt_setting": "...",
  "backbone": "...",
  "episode_id": 0,
  "seed": 0,
  "success": true
}
```

检查点：

- 任意 worker 被杀后，重跑同一命令可继续补齐。
- 非 policy 基础设施失败不应被误记为 task failure。
- 失败 job 可从对应 `<backbone>__<task>.log` 和 server log 定位。

## Stage 8.5：成功视频保留

目标：

```text
result/<run_name>/videos/<task>.mp4
```

共 8 个：

```text
put_bottles_dustbin.mp4
open_microwave.mp4
stack_blocks_three.mp4
stack_bowls_three.mp4
blocks_ranking_rgb.mp4
hanging_mug.mp4
put_object_cabinet.mp4
handover_block.mp4
```

视频来源优先级：

1. OpenWAM 正式评测中任意 backbone 的成功 episode 视频。
2. 如果某任务 OpenWAM 全失败，则生成 RoboTwin expert 成功视频作为任务说明样例。

检查点：

- `videos/` 下有 8 个 mp4。
- 每个 mp4 可以被 `ffprobe` 读取。
- Stage 9 文档说明每个视频来自 OpenWAM 成功还是 expert 成功。

## Stage 8.6：完成标准

Stage 8 完成需要满足：

- 40 个 job 全部完成。
- 1280 条 episode-level 记录齐全。
- 汇总 CSV/JSON 已生成。
- 8 个成功视频已保留。
- 无残留 `deploy.py`、`single_eval.sh`、`eval_policy_wrapper.py`。

检查命令：

```bash
find result/hard8_stage8_<timestamp>/episodes -name '*.jsonl' | wc -l
wc -l result/hard8_stage8_<timestamp>/episodes/*.jsonl
find result/hard8_stage8_<timestamp>/videos -name '*.mp4' | wc -l
ps -eo pid,cmd | rg 'deploy.py|single_eval.sh|eval_policy_wrapper.py'
```

## Stage 9：结果分析

汇总脚本：

```text
/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/summarize_hard8_passk.py
```

Stage 9 输出：

```text
result/<run_name>/notes/stage9-hard8-analysis.md
```

当前正式 tmux 脚本会在 40 个 job 跑完后自动执行：

```text
summaries/per_task_passk.csv
summaries/per_task_passk.json
summaries/backbone_average_passk.csv
summaries/backbone_average_passk.json
videos/video_sources.json
notes/stage9-hard8-analysis.md
```

分析内容：

- 每个 backbone 的 8-task 平均 `success@k`。
- 每个任务上 5 个 backbone 的 `success@k` 对比。
- 小 k 差距是否明显。
- 大 k 时差距是否收敛。
- 是否存在只有大 backbone 在 `success@32` 下能解决的任务。
- 是否存在全模型失败或全模型成功的任务。

判断逻辑：

- 大 backbone 在 `success@1/2/4` 更好但 `success@16/32` 收敛：支持“主要提高采样效率”。
- 大 backbone 在 `success@32` 仍解决更多任务：说明可能扩展能力边界。
- 全任务全模型接近 0：说明任务过难，需要补中等难度任务。
- 全任务全模型接近 1：说明任务不够难，需要继续筛选。

## 当前耗时预估

基于 3 张可用卡：

- 小/中模型如果每卡 3 slot 稳定：总耗时可能落在 12-24 小时。
- 如果只能每卡 2 slot：约 18-36 小时。
- 5B/14B 如果必须每卡 1 slot：整体可能接近 1-2 天。
- 困难任务长时序较多，实际耗时会由 `put_bottles_dustbin`、`open_microwave`、`stack_blocks_three`、`stack_bowls_three` 等任务主导。

最终以 smoke 的 `gpu_slot_results.csv` 和正式 run 的 `status.csv` 为准。

## 最终交付物

最终目录：

```text
/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/result/hard8_stage8_<timestamp>
```

必须包含：

- `config/hard8_tasks.txt`
- `config/backbone_ckpts.json`
- `config/gpu_plan.json`
- `config/gpu_slot_overrides.json`
- `episodes/*.jsonl`
- `logs/*.log`
- `videos/*.mp4`
- `summaries/per_task_passk.csv`
- `summaries/per_task_passk.json`
- `summaries/backbone_average_passk.csv`
- `summaries/backbone_average_passk.json`
- `notes/stage9-hard8-analysis.md`
