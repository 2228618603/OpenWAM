# OpenWAM RoboTwin WM / Action-only 训练跑通规划

更新时间：2026-09-24

目标：在不先修改训练代码的前提下，规划一条从数据检查、split 固定、debug smoke 到 50k 正式训练的分阶段路径。实验只比较显式 future-video / world-modeling loss 是否改变 action learning。

参考：

- `train-on-robotwin/note/openwam-robotwin-training-resource-index.md`
- `train-on-robotwin/note/wm_action_learning_experiment_plan.md`

## 0. 实验合同

两组训练必须共享同一份 model code、同一份 forward、同一份 optimizer 构建逻辑、同一份数据与采样口径。唯一核心区别是 `training.lambda_video`。

| 组别 | `lambda_action` | `lambda_video` | 总 loss |
| --- | ---: | ---: | --- |
| Action-only / No-WM | 1.0 | 0.0 | `L_action` |
| WM | 1.0 | 1.0 | `L_action + L_video` |

共同结构：

- `model=dual_system`
- `model/video_backbone=wan22_ti2v_5b`
- `model.architecture.variant=joint_self_attn`
- `model.architecture.attention_mask_mode=mutual`
- 保留 Action DiT、Video DiT、OpenWAM 原本 Video/Action 信息交互机制。
- 不删除 Video DiT，不绕过 Video DiT，不切断 Video -> Action 信息流，不把 No-WM 改成 `isolated`。

共同冻结策略：

- trainable：Action DiT 全量训练，Video DiT 全量训练，不使用 LoRA。
- frozen：VAE、Text Encoder、Image Encoder / Video Encoder、OpenWAM 原本作为 frozen pretrained encoder 使用的等价感知模块。
- No-WM 中 Video DiT 也必须 `requires_grad=True`，且不得从 optimizer 中移除。

正式训练共同超参：

```text
dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset
dataloader.embodiment=aloha-agilex
dataloader.variant=clean_50
training.batch_size=4
training.gradient_accumulation_steps=2
NPROC_PER_NODE=4
global batch size = 32
training.max_steps=50000
training.num_epochs=null
training.save_steps=5000
training.keep_last_k_ckpts=5
training.mixed_precision=bf16
training.zero_stage=2
training.learning_rate=1e-4
training.lr_scheduler=cosine
training.warmup_ratio=0.05
training.max_grad_norm=1.0
project.seed=42
```

## 1. 任务划分固定

OOD / unseen 任务使用用户指定的 10 个任务，训练阶段完全不使用。

| 用户名称 | RoboTwin task name |
| --- | --- |
| Adjust Bottle | `adjust_bottle` |
| Grab Roller | `grab_roller` |
| Place Container Plate | `place_container_plate` |
| Move Pillbottle Pad | `move_pillbottle_pad` |
| Move Can Pot | `move_can_pot` |
| Open Microwave | `open_microwave` |
| Press Stapler | `press_stapler` |
| Stack Bowls Three | `stack_bowls_three` |
| Handover Block | `handover_block` |
| Turn Switch | `turn_switch` |

Train 40 任务：

```text
beat_block_hammer
blocks_ranking_rgb
blocks_ranking_size
click_alarmclock
click_bell
dump_bin_bigbin
handover_mic
hanging_mug
lift_pot
move_playingcard_away
move_stapler_pad
open_laptop
pick_diverse_bottles
pick_dual_bottles
place_a2b_left
place_a2b_right
place_bread_basket
place_bread_skillet
place_burger_fries
place_can_basket
place_cans_plasticbox
place_dual_shoes
place_empty_cup
place_fan
place_mouse_pad
place_object_basket
place_object_scale
place_object_stand
place_phone_stand
place_shoe
put_bottles_dustbin
put_object_cabinet
rotate_qrcode
scan_object
shake_bottle
shake_bottle_horizontally
stack_blocks_three
stack_blocks_two
stack_bowls_two
stamp_seal
```

检查点：

- `train_tasks_40.txt` 有 40 行。
- `ood_tasks_10.txt` 有 10 行。
- 两个文件无交集。
- 两个文件并集等于 clean_50 的 50 个任务。

## 2. Stage A：数据完整性检查

目标：确认 clean-only 训练数据已经满足 OpenWAM dataloader 的基础发现条件。

操作：

```bash
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam
cd /home/chw/code/packages/OpenWAM/OpenWAM

python - <<'PY'
from openwam.dataloader.robotwin import discover_robotwin_roots
root = "/mnt/data/chw/robotwin-dataset/dataset"
roots = discover_robotwin_roots(root, "aloha-agilex", "clean_50")
print("clean_50 roots:", len(roots))
print(roots[:3])
PY

find /mnt/data/chw/robotwin-dataset/dataset \
  -path '*/aloha-agilex_clean_50/data/episode*.hdf5' | wc -l
```

通过检查点：

- `clean_50 roots: 50`
- HDF5 总数为 `2500`
- 本轮所有训练命令都显式使用 `dataloader.variant=clean_50`，避免误读残留 randomized 数据。

失败处理：

- 如果 roots 少于 50，先回到资源索引文档的数据整理步骤。
- 如果 HDF5 不是 2500，先定位缺失任务，不进入训练。

## 3. Stage B：构造 40 任务 x 45 train episodes 的训练目录

目标：在不改 OpenWAM 训练代码的前提下，用目录级数据 split 保证 dataloader 只能看到 40 个训练任务、每任务 45 条训练 episode。

建议目录：

```text
/mnt/data/chw/robotwin-dataset/splits/wm_action_v1/
  train40_45eps/
    <40 train tasks>/aloha-agilex_clean_50/data/episode0.hdf5 ... episode44.hdf5
  val40_5eps/
    <40 train tasks>/aloha-agilex_clean_50/data/episode45.hdf5 ... episode49.hdf5
  unseen10_50eps/
    <10 ood tasks>/aloha-agilex_clean_50/data/episode0.hdf5 ... episode49.hdf5
  train_tasks_40.txt
  ood_tasks_10.txt
```

实现建议：

- 不复制 HDF5，也不创建 symlink；训练入口通过任务目录名和 episode 文件名做运行时过滤。
- 正式训练用原始数据根目录 `dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset`。
- `train_tasks_40.txt` 和 `ood_tasks_10.txt` 只作为可审计的 split 清单。
- OOD 任务不参与训练，只用于 OOD eval。

检查点：

```bash
wc -l /home/chw/code/packages/OpenWAM/train-on-robotwin/code/splits/wm_action_v1/train_tasks_40.txt
wc -l /home/chw/code/packages/OpenWAM/train-on-robotwin/code/splits/wm_action_v1/ood_tasks_10.txt
python /home/chw/code/packages/OpenWAM/train-on-robotwin/code/check_robotwin_split_configs.py
```

通过检查点：

- `train_tasks_40.txt` 为 `40` 行。
- `ood_tasks_10.txt` 为 `10` 行。
- Hydra config 中 `dataloader.tasks` 为 40 个训练任务。
- `experiment_split.ood_tasks` 为 10 个 OOD 任务。
- 训练任务与 OOD 任务无交集。

## 4. Stage C：normalization stats

目标：避免正式训练启动时 rank0 临时全量计算 stats，导致启动阶段不可控。

操作：

```bash
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam
cd /home/chw/code/packages/OpenWAM/OpenWAM

python /home/chw/code/packages/OpenWAM/train-on-robotwin/code/compute_robotwin_split_stats.py
```

检查点：

```bash
ls -lh /home/chw/code/packages/OpenWAM/train-on-robotwin/code/stats/wm_action_v1/train40_45eps_robotwin_clean_50_normalization_stats.npy
```

通过检查点：

- stats 文件存在。
- 后续 Action-only 与 WM 都使用同一个 `train40_45eps` 目录和同一个 stats 文件。

失败处理：

- 如果 stats 计算失败，先用 `training.debug=true dataloader.normalize_mode=null` 跑 dataloader smoke，只验证路径和 HDF5 schema；正式训练仍应回到 stats 生成。

## 5. Stage D：Hydra 配置合成检查

目标：不加载大模型，只确认两组训练命令的配置只有 `lambda_video` 与输出目录不同。

共同 override：

```text
dataloader=robotwin
dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset
dataloader.embodiment=aloha-agilex
dataloader.variant=clean_50
model=dual_system
model/video_backbone=wan22_ti2v_5b
model.architecture.variant=joint_self_attn
model.architecture.attention_mask_mode=mutual
training.batch_size=4
training.gradient_accumulation_steps=2
training.max_steps=50000
training.num_epochs=null
training.save_steps=5000
training.keep_last_k_ckpts=5
training.lambda_action=1.0
project.seed=42
```

No-WM 额外 override：

```text
training.lambda_video=0.0
training.output_path=/mnt/data/chw/model/wm-function/robotwin_clean40_action_only_no_wm_50k
project.wandb.run_name=robotwin_clean40_action_only_no_wm_50k
```

WM 额外 override：

```text
training.lambda_video=1.0
training.output_path=/mnt/data/chw/model/wm-function/robotwin_clean40_wm_50k
project.wandb.run_name=robotwin_clean40_wm_50k
```

检查点：

- Hydra compose 能成功。
- 两份 resolved config diff 只允许以下字段不同：
  - `training.lambda_video`
  - `training.output_path`
  - `project.wandb.run_name`
- `model.architecture.attention_mask_mode` 两组均为 `mutual`。
- `model.freeze` 两组完全一致。

## 6. Stage E：参数冻结与可训练参数审计

目标：在训练前验证两组 trainable/frozen parameter 集合完全一致。

需要检查：

1. Action-only 与 WM 的 trainable parameter 数量完全一致。
2. 两组 frozen parameter 数量完全一致。
3. Action-only 中 Video DiT 仍然 `requires_grad=True`。
4. VAE、Text Encoder、Image Encoder / Video Encoder 均 `requires_grad=False`。
5. optimizer 参数组不因 `lambda_video=0.0` 少包含 Video DiT。

建议审计输出：

```text
audit/
  action_only_trainable_params.txt
  wm_trainable_params.txt
  action_only_frozen_params.txt
  wm_frozen_params.txt
  param_count_summary.json
```

通过检查点：

- trainable 参数 name set 完全一致。
- frozen 参数 name set 完全一致。
- trainable 参数总数完全一致。
- Video DiT 相关参数出现在两组 trainable list 中。

失败处理：

- 如果两组参数集合不同，不进入训练。
- 不允许用 freeze / detach / attention mask 差异修补 No-WM；只能回到配置和 trainer 审计。

## 7. Stage F：loss 路径与梯度审计

目标：确认 No-WM 和 WM 只在 loss weight 上不同，且 No-WM 的 action loss 仍能沿原始计算图更新相关 Video DiT 参数。

需要验证：

1. No-WM：`loss_total == loss_action`，`loss_video` 日志可以存在但乘以 `lambda_video=0.0` 后不参与 total。
2. WM：`loss_total == loss_action + loss_video`。
3. 两组都执行完整 forward。
4. No-WM backward 后，若 OpenWAM 原始 joint graph 存在 Action Loss -> Video DiT 的梯度路径，Video DiT 参数 grad 非空且非全 0。
5. 两组除 `lambda_video` 外没有分支差异。

建议做法：

- 用 1 个 batch、`NPROC_PER_NODE=1`、小输出目录做本地 audit。
- 暂时不跑 20 step，只做 forward/backward probe。
- 记录 Video DiT grad norm、Action DiT grad norm、frozen module grad 状态。

通过检查点：

- No-WM 中 Video DiT `requires_grad=True`。
- No-WM 中至少一部分 Video DiT trainable 参数在 backward 后有 grad，除非审计证明 OpenWAM 原始计算图本身不存在从 action loss 到该参数的路径。
- frozen 模块 grad 为 `None` 或 0。
- No-WM 的 total loss 不包含 video loss 权重。

失败处理：

- 如果 `lambda_video=0.0` 导致跳过影响 action forward 的 video branch，不进入正式训练，需要先修正实现。
- 如果 Video DiT 被 freeze 或 optimizer 排除，不进入正式训练。

## 8. Stage G：单卡 debug smoke

目标：确认 dataloader、stats、模型权重、DeepSpeed/Accelerate、loss 日志、checkpoint 写入都能跑通。

No-WM debug：

```bash
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam
cd /home/chw/code/packages/OpenWAM/OpenWAM

NPROC_PER_NODE=1 bash /home/chw/code/packages/OpenWAM/train-on-robotwin/code/train_robotwin_split.sh \
  dataloader=robotwin \
  dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset \
  dataloader.embodiment=aloha-agilex \
  dataloader.variant=clean_50 \
  model=dual_system \
  model/video_backbone=wan22_ti2v_5b \
  model.architecture.variant=joint_self_attn \
  model.architecture.attention_mask_mode=mutual \
  training.debug=true \
  training.lambda_action=1.0 \
  training.lambda_video=0.0 \
  training.output_path=/mnt/data/chw/model/wm-function/debug_action_only_no_wm
```

WM debug：

```bash
NPROC_PER_NODE=1 bash /home/chw/code/packages/OpenWAM/train-on-robotwin/code/train_robotwin_split.sh \
  dataloader=robotwin \
  dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset \
  dataloader.embodiment=aloha-agilex \
  dataloader.variant=clean_50 \
  model=dual_system \
  model/video_backbone=wan22_ti2v_5b \
  model.architecture.variant=joint_self_attn \
  model.architecture.attention_mask_mode=mutual \
  training.debug=true \
  training.lambda_action=1.0 \
  training.lambda_video=1.0 \
  training.output_path=/mnt/data/chw/model/wm-function/debug_wm
```

通过检查点：

- 两组都完成 20 step。
- 两组都在 step 10 写 checkpoint。
- No-WM 日志中 `loss_video` 对 total 的贡献为 0。
- WM 日志中 `loss_video` 非 0 且参与 total。
- 两组 `config.yaml` 除允许字段外一致。

## 9. Stage H：8 卡短训 smoke

目标：正式 4 卡配置启动前，验证 NCCL、DeepSpeed ZeRO、DataLoader 多进程和 checkpoint 管理。

设置：

```text
NPROC_PER_NODE=4
training.batch_size=4
training.gradient_accumulation_steps=2
training.max_steps=200
training.num_epochs=null
training.save_steps=100
training.keep_last_k_ckpts=2
```

通过检查点：

- 两组都能完成 200 step。
- 每组都产生 step100 和 step200 checkpoint。
- 显存没有 OOM。
- 每步耗时稳定，没有长时间卡在 stats 或 dataloader 初始化。
- 两组 checkpoint 目录结构一致。

失败处理：

- 如果 OOM，优先保持 global batch 32 不变，通过 `training.gradient_accumulation_steps` 调整 per-GPU batch，而不是改变两组不一致的训练条件。
- 如果 DataLoader 卡住，先降 `training.dataset_num_workers=0` 做诊断；正式设置两组必须一致。

## 10. Stage I：正式 50k 训练

建议先启动 Action-only，再启动 WM，或在资源允许时并行启动。两组使用相同 seed 和同一份 train split。

Action-only / No-WM：

```bash
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam
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
  project.seed=42 \
  project.wandb.run_name=robotwin_clean40_action_only_no_wm_50k
```

WM：

```bash
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam
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
  project.seed=42 \
  project.wandb.run_name=robotwin_clean40_wm_50k
```

通过检查点：

- 两组都保存 5k / 10k / 15k / 20k / 25k / 30k / 35k / 40k / 45k / 50k checkpoint。
- `config.yaml` diff 只包含允许字段。
- 两组训练 step 数、global batch、学习率曲线、seed、数据目录完全一致。
- No-WM 的 total loss 始终只由 action loss 贡献。
- WM 的 total loss 同时包含 action 和 video loss。

## 11. Stage J：训练后最小评测闭环

目标：先确认 checkpoint 能 deploy，并能在 RoboTwin 中完成最小 rollout，再进入大规模实验。

Policy server：

```bash
source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam
cd /home/chw/code/packages/OpenWAM/OpenWAM

bash scripts/deploy.sh /mnt/data/chw/model/wm-function/robotwin_clean40_wm_50k
```

单任务 smoke：

```bash
ROBOTWIN_PATH=/home/chw/code/packages/RoboTwin/RoboTwin \
ROBOTWIN_PYTHON=/home/chw/miniconda3/envs/RoboTwin/bin/python \
ROBOTWIN_TEST_NUM=1 \
bash benchmarks/robotwin/single_eval.sh beat_block_hammer demo_clean openwam 0
```

通过检查点：

- Action-only 和 WM 的 50k checkpoint 都能启动 policy server。
- seen clean smoke 至少能完成环境 reset、模型请求和 rollout 写盘。
- unseen clean smoke 对 OOD 任务也能完成同样闭环。

## 12. Stage K：正式实验评测排期

训练完成后按实验规划展开：

- Experiment 1：40 个 seen task 中选 10 个代表任务，每任务 20 个 fixed initial states，每 state 每模型 32 rollouts。
- Experiment 2A/2B：选择 5 个多 mode 任务，每任务 10 个 fixed initial states，每 state 每模型 128 rollouts，后处理轨迹 mode。
- Experiment 3：Seen / Unseen x Clean / Randomized。

当前注意：

- 本轮训练数据只准备了 clean_50。
- 如果要做 `Seen + Randomized` 和 `Unseen + Randomized`，需要另行准备 randomized eval 环境或 randomized 数据/资产；这不是当前 clean-only 训练启动的阻塞项。

## 13. 总阻塞清单

正式 50k 训练前必须完成：

- [ ] 生成 `train_tasks_40.txt` 和 `ood_tasks_10.txt`
- [ ] 构造 `train40_45eps` / `val40_5eps` / `unseen10_50eps`
- [ ] 校验 `train40_45eps` 返回 40 个 clean root
- [ ] 校验 train/val/unseen HDF5 数分别为 1800 / 200 / 500
- [ ] 生成 `robotwin_clean_50_normalization_stats.npy`
- [ ] Hydra config diff 只允许 `lambda_video`、输出目录和 run name 不同
- [ ] 参数冻结/可训练集合审计通过
- [ ] No-WM loss 和梯度路径审计通过
- [ ] 单卡 debug 两组都跑通
- [ ] 8 卡 200 step smoke 两组都跑通

满足以上检查点后，再启动两组 50k 正式训练。
