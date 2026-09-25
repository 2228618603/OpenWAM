#!/usr/bin/env bash
set -euo pipefail

source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam

cd /home/chw/code/packages/OpenWAM/OpenWAM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export WANDB_MODE=online
unset PYTORCH_CUDA_ALLOC_CONF
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

COMMON_ARGS=(
  dataloader=robotwin
  dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset
  dataloader.embodiment=aloha-agilex
  dataloader.variant=clean_50
  model=dual_system
  model/video_backbone=wan22_ti2v_5b
  model.architecture.variant=joint_self_attn
  model.architecture.attention_mask_mode=mutual
  training.debug=false
  training.batch_size=4
  training.gradient_accumulation_steps=1
  training.max_steps=50000
  training.num_epochs=null
  training.save_steps=5000
  training.keep_last_k_ckpts=5
  training.dataset_num_workers=0
  training.use_gradient_checkpointing_offload=false
  training.lambda_action=1.0
  project.seed=42
)

echo "[$(date -Is)] starting Action-only / No-WM formal training (8 GPUs, per-GPU bs=4, global bs=32)"
ROBOTWIN_SPLIT_RANK_LOG_DIR=/home/chw/code/packages/OpenWAM/train-on-robotwin/code/audit/rank_logs/formal_action_only_no_wm_50k_8gpu_${RUN_STAMP} \
MASTER_PORT=29630 \
bash /home/chw/code/packages/OpenWAM/train-on-robotwin/code/train_robotwin_split.sh \
  "${COMMON_ARGS[@]}" \
  training.lambda_video=0.0 \
  training.output_path=/mnt/data/chw/model/wm-function/robotwin_clean40_action_only_no_wm_50k \
  project.wandb.run_name=robotwin_clean40_action_only_no_wm_50k_8gpu_bs32

echo "[$(date -Is)] Action-only / No-WM formal training finished"
