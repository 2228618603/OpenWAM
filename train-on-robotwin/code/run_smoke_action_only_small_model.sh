#!/usr/bin/env bash
set -euo pipefail

source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENWAM_STEP_DIAG="${OPENWAM_STEP_DIAG:-1}"
export OPENWAM_STEP_DIAG_EVERY="${OPENWAM_STEP_DIAG_EVERY:-5}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MAX_STEPS="${MAX_STEPS:-30}"
TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH:-${REPO_ROOT}/train-on-robotwin/runs/smoke_action_only_wan21_vace_1_3b_${RUN_STAMP}}"

COMMON_ARGS=(
  dataloader=robotwin
  dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset
  dataloader.embodiment=aloha-agilex
  dataloader.variant=clean_50
  model=dual_system
  model/video_backbone=wan21_vace_1_3b
  model.architecture.variant=joint_self_attn
  model.architecture.attention_mask_mode=action_sees_video
  training.debug=false
  training.batch_size=1
  training.gradient_accumulation_steps=4
  training.max_steps="${MAX_STEPS}"
  training.num_epochs=null
  training.zero_stage=2
  training.offload_optimizer_device=cpu
  +training.adamw_foreach=false
  +training.freeze_video_backbone=true
  +training.skip_zero_weight_video_loss=true
  training.save_steps=0
  training.dataset_num_workers=0
  training.use_gradient_checkpointing_offload=false
  training.lambda_action=1.0
  training.lambda_video=0.0
  training.output_path="${TRAIN_OUTPUT_PATH}"
  project.seed=42
  project.wandb.run_name=smoke_action_only_wan21_vace_1_3b
)

echo "[$(date -Is)] starting small-model action-only smoke (${NPROC_PER_NODE} GPUs, max_steps=${MAX_STEPS})"
ROBOTWIN_SPLIT_RANK_LOG_DIR="${REPO_ROOT}/train-on-robotwin/code/audit/rank_logs/smoke_action_only_wan21_vace_1_3b_${RUN_STAMP}" \
ROBOTWIN_SPLIT_ERROR_DIR="${REPO_ROOT}/train-on-robotwin/code/audit/error_logs/smoke_action_only_wan21_vace_1_3b_${RUN_STAMP}" \
MASTER_PORT="${MASTER_PORT:-29631}" \
bash "${SCRIPT_DIR}/train_robotwin_split.sh" "${COMMON_ARGS[@]}"

echo "[$(date -Is)] small-model action-only smoke finished"
