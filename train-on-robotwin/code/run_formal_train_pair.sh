#!/usr/bin/env bash
set -euo pipefail

source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OPENWAM_PROJECT_ROOT="${REPO_ROOT}/OpenWAM"

cd "${OPENWAM_PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
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
  training.gradient_accumulation_steps=2
  training.max_steps=50000
  training.num_epochs=null
  training.save_steps=5000
  training.keep_last_k_ckpts=5
  training.dataset_num_workers=0
  training.use_gradient_checkpointing_offload=false
  training.lambda_action=1.0
  project.seed=42
)

echo "[$(date -Is)] starting Action-only / No-WM formal training"
ROBOTWIN_SPLIT_RANK_LOG_DIR="${REPO_ROOT}/train-on-robotwin/code/audit/rank_logs/formal_action_only_no_wm_50k_${RUN_STAMP}" \
MASTER_PORT=29620 \
bash "${SCRIPT_DIR}/train_robotwin_split.sh" \
  "${COMMON_ARGS[@]}" \
  training.lambda_video=0.0 \
  training.output_path=/mnt/data/chw/model/wm-function/robotwin_clean40_action_only_no_wm_50k \
  project.wandb.run_name=robotwin_clean40_action_only_no_wm_50k

echo "[$(date -Is)] starting WM formal training"
ROBOTWIN_SPLIT_RANK_LOG_DIR="${REPO_ROOT}/train-on-robotwin/code/audit/rank_logs/formal_wm_50k_${RUN_STAMP}" \
MASTER_PORT=29621 \
bash "${SCRIPT_DIR}/train_robotwin_split.sh" \
  "${COMMON_ARGS[@]}" \
  training.lambda_video=1.0 \
  training.output_path=/mnt/data/chw/model/wm-function/robotwin_clean40_wm_50k \
  project.wandb.run_name=robotwin_clean40_wm_50k

echo "[$(date -Is)] formal training pair finished"
