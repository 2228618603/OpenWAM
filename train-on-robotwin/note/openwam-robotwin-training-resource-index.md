# OpenWAM RoboTwin 训练资源索引与启动说明

更新时间：2026-09-24

目标：为 `train-on-robotwin/note/wm_action_learning_experiment_plan.md` 中的 WM / No-WM 对比实验建立一份可迭代维护的资源说明，先把数据、权重、代码入口、环境和当前缺口固定下来。

## 结论先行

- 代码根目录：`/home/chw/code/packages/OpenWAM`
- OpenWAM 实际子项目目录：`/home/chw/code/packages/OpenWAM/OpenWAM`
- Conda 环境：`openwam`
- 官方 RoboTwin 2.0 原始数据候选：`/mnt/data/embodied_datasets/public_datasets_raw/RoboTwin2.0`
- 当前已整理的 OpenWAM 训练数据目录：`/mnt/data/chw/robotwin-dataset/dataset`
- 当前已完成数据口径：
  - `aloha-agilex_clean_50`：50 个任务，每任务 50 条 HDF5 episode，总计 2500 条。
  - `aloha-agilex_randomized_500`：50 个任务，每任务 500 条 HDF5 episode，总计 25000 条。
- 当前任务选择说明：`/mnt/data/chw/robotwin-dataset/robotwin_task_choices.txt`
- OpenWAM 训练期望的数据目录：`<dataset_dir>/<task>/aloha-agilex_clean_50/data/episode*.hdf5` 和/或 `<dataset_dir>/<task>/aloha-agilex_randomized_500/data/episode*.hdf5`
- 当前本机状态：`raw/RoboTwin2.0/dataset` 下有 50 个 `aloha-agilex_clean_50.zip` 和 50 个 `aloha-agilex_randomized_500.zip`。已将 50 个任务的 `aloha-agilex_clean_50` 和 `aloha-agilex_randomized_500` 都解压到 `/mnt/data/chw/robotwin-dataset/dataset`，OpenWAM dataloader 已确认可发现 50 个 clean root。
- `public_datasets_staging/geek_data` 是 LeRobot/parquet 风格数据，但任务不是 RoboTwin 2.0 的 50 个官方任务，不应当作为本次 OpenWAM-on-RoboTwin 主训练数据。

## 官方核对

OpenWAM 官方 README/下载脚本说明：

- RoboTwin eval 支持全部 50 个任务，入口在 `OpenWAM/benchmarks/robotwin/`。
- OpenWAM benchmark downloader 支持 `RoboTwin2.0`，HuggingFace repo 是 `TianxingChen/RoboTwin2.0`，默认只下载 `dataset/*aloha-agilex*.zip`，估算约 415 GB。
- 下载后目标结构是 `assets/benchmark_data/robotwin2.0/dataset`，并会更新 `configs/dataloader/robotwin.yaml` 的 `dataset_dir`。
- OpenWAM `robotwin` dataloader 直接读取 HDF5：`episode*.hdf5`，不是直接读取 zip，也不是 LeRobot parquet。

RoboTwin 官方 README 说明：

- RoboTwin 2.0 是当前默认版本，包含 50 个任务。
- 官方推荐下载 HuggingFace 的预采集 XPolicyLab-format trajectories，路径形态是每个任务下的 `data/episode*.hdf5`。
- 官方也提供 LeRobot 转换脚本，但这不是 OpenWAM 当前 `robotwin.py` reader 的直接输入。

## 本地数据盘判断

### 1. `public_datasets_raw`

主候选：

```text
/mnt/data/embodied_datasets/public_datasets_raw/RoboTwin2.0
```

已观察到的关键内容：

```text
RoboTwin2.0/
  README.md
  background_texture.zip
  embodiments.zip
  objects.zip
  dataset/
    <task>/
      aloha-agilex_clean_50.zip
      aloha-agilex_randomized_500.zip
      ...
  act_ckpt/
  DP3_ckpt/
  rdt_ckpt/
```

抽查 `adjust_bottle/aloha-agilex_clean_50.zip`，zip 内部包含：

```text
aloha-agilex_clean_50/
  scene_info.json
  data/episode0.hdf5 ... episode49.hdf5
  _traj_data/episode0.pkl ... episode49.pkl
```

这与 OpenWAM `discover_robotwin_roots()` 期望的目录结构一致，但前提是 zip 已解压到任务目录下。

当前缺口：

```bash
find /mnt/data/embodied_datasets/public_datasets_raw/RoboTwin2.0/dataset \
  -maxdepth 3 -type d -path '*/aloha-agilex_*_*/data' | wc -l
# 当前结果：0
```

原始数据目录仍保持 zip 形态。当前没有直接在 raw 目录解压，而是将训练数据整理到了用户目录：

```text
/mnt/data/chw/robotwin-dataset/dataset/<task>/aloha-agilex_clean_50/data/episode*.hdf5
/mnt/data/chw/robotwin-dataset/dataset/<task>/aloha-agilex_randomized_500/data/episode*.hdf5
```

当前 clean 校验结果：

```text
tasks: 50
clean_zips: 50
clean_dirs: 50
clean_hdf5_total: 2500
bad_clean_tasks_count: 0
OpenWAM discover_robotwin_roots(..., "aloha-agilex", "clean_50"): 50
```

当前 randomized 校验结果：

```text
randomized_zips: 50
randomized_dirs: 50
randomized_hdf5_total: 25000
randomized_pkl_total: 25000
per_task_hdf5: 500
per_task_pkl: 500
bad_randomized_tasks_count: 0
解压目标：/mnt/data/chw/robotwin-dataset/dataset/<task>/aloha-agilex_randomized_500/
并行解压日志：/tmp/robotwin_parallel_unzip_20260924T091209
中断残留 .tmp_extract 备份：/tmp/robotwin_tmp_extract_residue_20260924T092339
```

### 2. `public_datasets_staging`

当前只看到：

```text
/mnt/data/embodied_datasets/public_datasets_staging/geek_data/
  agilex_empty_the_box_all_470/
  agilex_make_breakfast_410_90/
```

这些目录包含 `meta/info.json`、`data/chunk-*.parquet`、`t5_embedding/`，是 LeRobot/parquet 风格，但不是 RoboTwin 2.0 50-task 训练集。因此本次先不作为主训练数据。

## 需要的模型权重

### 视频 backbone 权重

OpenWAM 配置已经指向这些本地路径：

```text
/mnt/data/chw/model/openwam/Wan2.2-TI2V-5B
/mnt/data/chw/model/openwam/Wan2.1-VACE-1.3B
/mnt/data/chw/model/openwam/Wan2.1-I2V-14B-480P
/mnt/data/chw/model/openwam/Cosmos-Predict2.5-2B
/mnt/data/chw/model/openwam/Cosmos-Reason1-7B
/mnt/data/chw/model/openwam/Cosmos3-Edge
```

默认训练配置使用：

```text
OpenWAM/configs/model/video_backbone/wan22_ti2v_5b.yaml
model.video_backbone.model_path=/mnt/data/chw/model/openwam/Wan2.2-TI2V-5B
```

### 已有 RoboTwin study checkpoint

本机已有以下 OpenWAM Study / video backbone checkpoint，可用于复现实验、warm-start 或评测对照：

```text
/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention
/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention_cosmos25
/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention_cosmos3
/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention_wan21_i2v_14b
/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention_wan21_vace_1_3b
```

注意：这些 checkpoint 的 `config.yaml` 多数仍写着 `/path/to/...` 占位符。直接 `deploy.sh <ckpt_dir>` 前要确认运行时是否会使用 checkpoint 自带 config，必要时改为本机 `/mnt/data/chw/model/openwam/...` 路径或用 CLI 覆盖。

## 代码入口

### 训练

```text
OpenWAM/scripts/train.sh
OpenWAM/scripts/train.py
OpenWAM/configs/train.yaml
OpenWAM/configs/dataloader/robotwin.yaml
OpenWAM/openwam/dataloader/robotwin.py
OpenWAM/openwam/dataloader/utils/stats_computation/robotwin_stats_computation.py
```

`OpenWAM/scripts/train.sh` 会自动 `cd` 到 `OpenWAM/` 子项目目录，然后用 `torchrun` 启动 `scripts/train.py`。

### 推理与评测

```text
OpenWAM/scripts/deploy.sh
OpenWAM/scripts/deploy.py
OpenWAM/configs/deploy.yaml
OpenWAM/scripts/inference_test/inference_single_test.py
OpenWAM/scripts/inference_test/inference_continuous_test.py
OpenWAM/benchmarks/robotwin/single_eval.sh
OpenWAM/benchmarks/robotwin/multi_eval.sh
OpenWAM/benchmarks/robotwin/openwam2robotwin_interface.py
OpenWAM/benchmarks/robotwin/eval_policy_wrapper.py
OpenWAM/benchmarks/robotwin/prompt_template.py
```

历史可复用代码：

```text
evaluation-code-summary/
chw-code/robotwin_legacy_compat/
chw-code/robotwin_legacy_compat_copy/
```

这些更适合复用已有评测调度、Hard8 smoke 和多 checkpoint 对比；正式训练入口仍以 `OpenWAM/scripts/train.sh` 为准。

## 环境

OpenWAM 环境已存在：

```bash
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam
python -V
# Python 3.10.21
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
# torch 2.7.1+cu128, CUDA 12.8, cuda available True
```

RoboTwin 评测环境也存在：

```text
/home/chw/miniconda3/envs/RoboTwin
```

RoboTwin 代码候选：

```text
/home/chw/code/packages/RoboTwin
/home/chw/code/packages/RoboTwin/RoboTwin
/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat_copy
```

正式评测时建议使用 OpenWAM 官方 README 中记录的验证组合，或至少记录实际 RoboTwin commit、`eval_policy.py` hash 和本地改动。

## 训练前准备命令

### 1. 解压 RoboTwin aloha-agilex 数据

当前已按 clean + randomized 口径完成整理，不需要重复执行。训练目录：

```text
/mnt/data/chw/robotwin-dataset/dataset
```

当前每个任务目录结构：

```text
/mnt/data/chw/robotwin-dataset/dataset/<task>/aloha-agilex_clean_50/data/episode0.hdf5 ... episode49.hdf5
/mnt/data/chw/robotwin-dataset/dataset/<task>/aloha-agilex_randomized_500/data/episode0.hdf5 ... episode499.hdf5
```

### 2. 校验 OpenWAM 可发现数据

```bash
cd /home/chw/code/packages/OpenWAM/OpenWAM

python - <<'PY'
from openwam.dataloader.robotwin import discover_robotwin_roots
root = "/mnt/data/chw/robotwin-dataset/dataset"
roots = discover_robotwin_roots(root, "aloha-agilex", "clean_50")
print("clean_50", len(roots))
print(roots[:3])
PY
```

期望：

```text
clean_50 50
```

### 3. 生成或确认 normalization stats

OpenWAM 多任务 reader 会在训练首次加载时自动生成：

```text
<dataset_dir>/meta/robotwin_clean_50_normalization_stats.npy
```

也可以训练前显式生成：

```bash
cd /home/chw/code/packages/OpenWAM/OpenWAM

python -m openwam.dataloader.utils.stats_computation.robotwin_stats_computation \
  --dataset_dir /mnt/data/chw/robotwin-dataset/dataset \
  --embodiment aloha-agilex \
  --variant clean_50
```

## 启动训练建议

### Smoke debug

先跑 20 step，确认 dataloader、权重路径、DeepSpeed、显存都没问题：

```bash
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam
cd /home/chw/code/packages/OpenWAM/OpenWAM

NPROC_PER_NODE=1 bash scripts/train.sh \
  dataloader=robotwin \
  dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset \
  dataloader.embodiment=aloha-agilex \
  dataloader.variant=clean_50 \
  model=dual_system \
  model/video_backbone=wan22_ti2v_5b \
  model.architecture.variant=joint_self_attn \
  model.architecture.attention_mask_mode=mutual \
  training.debug=true
```

### WM 正式训练

实验计划里 WM = action loss + video loss。对应 OpenWAM 默认 `lambda_action=1.0`、`lambda_video=1.0`。

```bash
cd /home/chw/code/packages/OpenWAM/OpenWAM

CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash /home/chw/code/packages/OpenWAM/train-on-robotwin/code/train_robotwin_split.sh \
  dataloader=robotwin \
  dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset \
  dataloader.embodiment=aloha-agilex \
  dataloader.variant=clean_50 \
  model=dual_system \
  model/video_backbone=wan22_ti2v_5b \
  model.architecture.variant=joint_self_attn \
  model.architecture.attention_mask_mode=mutual \
  training.batch_size=4 \
  training.gradient_accumulation_steps=2 \
  training.max_steps=50000 \
  training.num_epochs=null \
  training.save_steps=5000 \
  training.keep_last_k_ckpts=5 \
  training.lambda_action=1.0 \
  training.lambda_video=1.0 \
  training.output_path=/mnt/data/chw/model/wm-function/robotwin_clean40_wm_50k \
  project.wandb.run_name=robotwin_clean40_wm_50k
```

上面 `batch_size=4 x 4 GPU x gradient_accumulation_steps=2 = global batch 32`，符合当前 4 卡设置。

### No-WM / Action-only 正式训练

实验计划里 No-WM = action loss only。最小差异建议先只改 loss：

```bash
cd /home/chw/code/packages/OpenWAM/OpenWAM

CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash /home/chw/code/packages/OpenWAM/train-on-robotwin/code/train_robotwin_split.sh \
  dataloader=robotwin \
  dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset \
  dataloader.embodiment=aloha-agilex \
  dataloader.variant=clean_50 \
  model=dual_system \
  model/video_backbone=wan22_ti2v_5b \
  model.architecture.variant=joint_self_attn \
  model.architecture.attention_mask_mode=mutual \
  training.batch_size=4 \
  training.gradient_accumulation_steps=2 \
  training.max_steps=50000 \
  training.num_epochs=null \
  training.save_steps=5000 \
  training.keep_last_k_ckpts=5 \
  training.lambda_action=1.0 \
  training.lambda_video=0.0 \
  training.output_path=/mnt/data/chw/model/wm-function/robotwin_clean40_action_only_no_wm_50k \
  project.wandb.run_name=robotwin_clean40_action_only_no_wm_50k
```

需要注意：这只是关闭显式 video loss。Action-only / No-WM 仍保留完整 Video DiT、原始 Video/Action 交互路径和完全相同 trainable/frozen 参数；不得通过冻结、隔离 attention 或删除 Video 分支构造 baseline。

## 与实验计划的差距

实验计划要求：

- 40 个训练任务
- 每任务 45 条 demonstration 训练
- 每任务 5 条 demonstration 训练过程验证
- 10 个留出任务 OOD

当前 OpenWAM `robotwin` dataloader 默认按磁盘中存在的任务全量发现，不内置 40/10 task split，也不内置每任务 45/5 episode split。需要补一个显式 split 机制，或先用目录级软链接构造训练目录：

```text
robotwin_splits/
  train40/
    <40 tasks copied or symlinked>
  unseen10/
    <10 held-out tasks copied or symlinked>
```

并且每个 task 内部需要区分前 45 条 episode 用于 train、后 5 条用于 val。当前 reader 的 `split=train/val` 只影响 window 和 prompt 选择逻辑，不会自动做 episode 45/5 切分。

建议下一步先确定：

```text
train_tasks_40.txt
unseen_tasks_10.txt
val_episode_policy: episode45-49 或固定 seed 抽样 5 条
```

## 推理 / 评测启动

启动 OpenWAM policy server：

```bash
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam
cd /home/chw/code/packages/OpenWAM/OpenWAM

bash scripts/deploy.sh /mnt/data/chw/model/wm-function/robotwin_clean40_wm_50k
```

单任务 RoboTwin 评测：

```bash
ROBOTWIN_PATH=/home/chw/code/packages/RoboTwin/RoboTwin \
ROBOTWIN_PYTHON=/home/chw/miniconda3/envs/RoboTwin/bin/python \
bash benchmarks/robotwin/single_eval.sh adjust_bottle demo_clean openwam 0
```

Smoke 时加：

```bash
ROBOTWIN_TEST_NUM=1
```

## 待办清单

- [x] 复制并解压 `RoboTwin2.0/dataset/*/aloha-agilex_clean_50.zip` 到 `/mnt/data/chw/robotwin-dataset/dataset`
- [x] 解压并校验 `RoboTwin2.0/dataset/*/aloha-agilex_randomized_500.zip` 到 `/mnt/data/chw/robotwin-dataset/dataset`
- [x] 运行 `discover_robotwin_roots` 校验 clean_50 = 50
- [ ] 生成或确认 `meta/robotwin_clean_50_normalization_stats.npy`
- [ ] 固定 40 train / 10 unseen task split
- [ ] 固定每任务 45 train / 5 val episode split，并实现到 dataloader 或目录结构
- [ ] 先跑 `training.debug=true`
- [ ] 跑 WM 50k
- [ ] 跑 No-WM 50k
- [ ] 每 10k checkpoint 做固定 initial state rollout 评测

## 迭代记录

### 2026-09-24

- 确认 `raw/RoboTwin2.0` 是本次数据主候选。
- 确认 `staging/geek_data` 不是 RoboTwin 2.0 50-task 训练数据。
- 确认本机已有 OpenWAM video backbone 权重和 5 个 RoboTwin video-backbone study checkpoint。
- 确认 `openwam` 环境 Python 3.10.21，Torch 2.7.1+cu128，CUDA 可用。
- 已将 50 个任务的 `aloha-agilex_clean_50` 复制并解压到 `/mnt/data/chw/robotwin-dataset/dataset`，共 2500 个 HDF5。
- 已将 50 个任务的 `aloha-agilex_randomized_500` 解压到 `/mnt/data/chw/robotwin-dataset/dataset`，共 25000 个 HDF5、25000 个 PKL；每任务 500 条 HDF5 episode。
- 已生成任务选择说明：`/mnt/data/chw/robotwin-dataset/robotwin_task_choices.txt`。
- 已确认 OpenWAM `discover_robotwin_roots()` 可发现 50 个 clean root。
- 当前阻塞项：40/10 task split 和 45/5 episode split 尚未落到代码或目录结构；normalization stats 尚未生成。
