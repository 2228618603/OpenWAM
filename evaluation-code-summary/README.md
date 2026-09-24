# OpenWAM RoboTwin 评测代码迁移说明

这个目录是从当前 OpenWAM 工作区提取出来的轻量评测复用包，目标是在另一台机器上尽快复现 RoboTwin 上的 OpenWAM policy server 评测流程。它不包含大模型权重、RoboTwin 仿真资产、历史日志、第三方源码或 conda 环境，只保留调度脚本、RoboTwin adapter 快照、配置模板和结果汇总工具。

## 目录结构

```text
evaluation-code-summary/
  env.example                         # 新机器路径配置模板
  configs/
    backbone_ckpts.json               # 5 个 OpenWAM RoboTwin checkpoint 路径映射
    hard8_tasks.txt                   # Hard8 任务列表
    policy_config.yml                 # RoboTwin client 连接 OpenWAM server 的配置
    step_limits.yml                   # 可选 task step limit override
  openwam-robotwin-adapter/
    single_eval.sh
    eval_policy_wrapper.py
    openwam2robotwin_interface.py
    prompt_template.py
  scripts/
    install_robotwin_adapter.sh       # 把 adapter 同步到 OpenWAM/benchmarks/robotwin/
    start_openwam_server.sh           # 启动 OpenWAM WebSocket policy server
    run_one_robotwin_job.sh           # 单个 backbone x task 的可断点续跑 job
    run_parallel_robotwin_eval.py     # 多任务/多 backbone/GPU slot 并行调度器
    smoke_gpu_slots.sh                # 快速测单卡可并发几个 job
    summarize_passk.py                # Stage6 风格单任务 pass@k 汇总
    summarize_hard8_passk.py          # Hard8 风格 per-task/backbone-average 汇总
    download_openwam_robotwin_video_backbone_ckpts.py
```

## 需要的环境

有两套 Python 环境：

1. OpenWAM server 环境：用于运行 `OPENWAM_ROOT/scripts/deploy.py`，需要 PyTorch/CUDA、OpenWAM 依赖、模型后端依赖和 checkpoint 所需组件。
2. RoboTwin client 环境：用于运行 RoboTwin 仿真和 `benchmarks/robotwin/single_eval.sh`，需要 RoboTwin、SAPIEN、cuRobo/运动规划相关依赖。

硬件建议：

- 至少 1 张 NVIDIA GPU。`wan21_i2v_14b` 显存需求最高，当前机器历史记录显示 server 侧单卡约 57GB 显存。
- 多 GPU 时建议先跑 `smoke_gpu_slots.sh`，确认每张卡能承载几个并发 job。
- OpenWAM server 和 RoboTwin simulator 当前脚本使用同一个 `GPU` 参数。若要 server/client 分卡，需要进一步拆 `run_one_robotwin_job.sh`。

## 最快配置流程

1. 准备 OpenWAM 仓库和 RoboTwin 仓库。

   `OPENWAM_ROOT` 必须指向包含 `scripts/deploy.py` 和 `benchmarks/robotwin/` 的 OpenWAM 目录。`ROBOTWIN_PATH` 必须指向包含 `script/eval_policy.py`、`envs/`、`task_config/` 的 RoboTwin 目录，或当前项目里的 legacy-compatible RoboTwin copy。

2. 配置环境变量。

   ```bash
   cd /path/to/evaluation-code-summary
   cp env.example env.local
   # 编辑 env.local，把 /path/to/... 改成新机器真实路径
   source env.local
   ```

3. 同步 RoboTwin adapter 到 OpenWAM。

   ```bash
   bash scripts/install_robotwin_adapter.sh
   ```

   这会覆盖 `${OPENWAM_ROOT}/benchmarks/robotwin/` 下的 `single_eval.sh`、`eval_policy_wrapper.py`、`openwam2robotwin_interface.py`、`prompt_template.py`、`policy_config.yml` 和 `step_limits.yml`。如果目标 OpenWAM 仓库已有本地改动，先备份或用 git diff 检查。

4. 配置 checkpoint。

   编辑 `configs/backbone_ckpts.json`，把 5 个值改成新机器上的实际 checkpoint 目录。目录里应包含可被 `scripts/deploy.py --ckpt-dir` 加载的 OpenWAM checkpoint 配置和权重。

   如果需要下载公开的 RoboTwin video-backbone study checkpoint，可先看：

   ```bash
   python scripts/download_openwam_robotwin_video_backbone_ckpts.py --help
   ```

## 快速自检

先确认 OpenWAM server 能单独启动：

```bash
source env.local
bash scripts/start_openwam_server.sh \
  /path/to/one/openwam/checkpoint \
  0 \
  8848 \
  ./eval-result/server_smoke.log
```

看到端口监听后，在另一个 shell 停掉该进程。`start_openwam_server.sh` 会打印 server PID。

然后跑一个 1-episode RoboTwin job：

```bash
source env.local
TARGET_N=1 bash scripts/run_one_robotwin_job.sh \
  ./eval-result/single_smoke \
  wan21_vace_1_3b \
  click_bell \
  0 \
  8848 \
  /path/to/openwam/checkpoint
```

成功后会生成：

```text
eval-result/single_smoke/
  episodes/wan21_vace_1_3b__click_bell.jsonl
  logs/wan21_vace_1_3b__click_bell.log
  logs/wan21_vace_1_3b__click_bell__server.log
```

## 运行 Hard8 并行评测

编辑 `configs/hard8_tasks.txt` 可调整任务集合。默认是当前 Hard8：

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

单卡、每个 backbone-task 跑 32 episode：

```bash
source env.local
python scripts/run_parallel_robotwin_eval.py \
  --result-root ./eval-result \
  --run-name hard8_reuse_$(date +%Y%m%d_%H%M%S) \
  --gpu-slots 0 \
  --target-n 32 \
  --phase-by-backbone
```

多卡并行示例：

```bash
source env.local
python scripts/run_parallel_robotwin_eval.py \
  --result-root ./eval-result \
  --gpu-slots 0,1,2 \
  --base-port 9100 \
  --target-n 32 \
  --phase-by-backbone
```

`--phase-by-backbone` 会一次只跑一个 backbone 的全部任务，便于显存较高的 backbone 使用不同 GPU slot。可以给某个 backbone 单独指定 slot：

```bash
python scripts/run_parallel_robotwin_eval.py \
  --result-root ./eval-result \
  --gpu-slots 0,1,2 \
  --gpu-slots-override wan21_i2v_14b=0 \
  --target-n 32 \
  --phase-by-backbone
```

调度器支持断点续跑：它按 `episodes/<backbone>__<task>.jsonl` 统计已有记录数，少于 `target-n` 才补跑。

## 结果汇总

Hard8 评测结束后，调度器会自动调用：

```bash
python scripts/summarize_hard8_passk.py ./eval-result/<run_name>
```

输出：

```text
summaries/per_task_passk.csv
summaries/per_task_passk.json
summaries/backbone_average_passk.csv
summaries/backbone_average_passk.json
status.csv
status.json
```

如果只跑单任务、每个 backbone 一个或多个 `*_episodes.jsonl`，可以用：

```bash
python scripts/summarize_passk.py ./eval-result/<single_task_run>
```

它会输出 `passk_summary.csv` 和 `passk_summary.json`。

## 常见问题

- `ROBOTWIN_PATH must be set`：没有 `source env.local`，或路径写错。
- `ROBOTWIN_PYTHON is not executable`：RoboTwin conda env 路径不对。
- `port did not open in time`：OpenWAM server 加载 checkpoint 失败、显存不足、CUDA 环境不对，先看 `logs/*__server.log`。
- SAPIEN/EGL 报错：优先在 `env.local` 里设置 `VK_ICD_FILENAMES` 和 `__EGL_VENDOR_LIBRARY_FILENAMES`，指到 RoboTwin 环境里的 SAPIEN vulkan JSON。
- episode 数不足：看对应 `logs/<backbone>__<task>.log`，常见原因是仿真不稳定、planner 失败或任务 step limit 不够。
- OpenWAM/RoboTwin 版本不一致：`eval_policy_wrapper.py` 会打印 RoboTwin commit 和 `script/eval_policy.py` hash。为了快速跑通可保留 `ROBOTWIN_SKIP_VERSION_CHECK=1`，正式复现实验建议记录实际 commit。

## 迁移边界

这个目录只解决“怎么启动、调度、记录和汇总评测”。它不替代：

- OpenWAM 本体代码和依赖安装；
- RoboTwin 本体、仿真资产和任务数据；
- OpenWAM checkpoint 下载与校验；
- CUDA/驱动/conda 环境兼容性调试。

推荐在新机器先跑 `TARGET_N=1` 的单任务闭环，再跑 `smoke_gpu_slots.sh`，最后启动完整 Hard8。
