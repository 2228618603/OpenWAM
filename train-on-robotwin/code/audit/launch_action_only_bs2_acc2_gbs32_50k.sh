#!/usr/bin/env bash
set -euo pipefail

source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam

REPO_ROOT=/home/chw/code/packages/OpenWAM
OPENWAM_PROJECT_ROOT="${REPO_ROOT}/OpenWAM"
SCRIPT_DIR="${REPO_ROOT}/train-on-robotwin/code"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-/mnt/data/chw/model/wm-function/cache/robotwin_text_embeddings/wan22_ti2v_5b_umt5_xxl_bf16_clean50}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/chw/model/openwam_train_runs/robotwin_clean40_action_only_no_wm_50k_8gpu_bs2_acc2_gbs32_${RUN_STAMP}}"
RANK_LOG_DIR="${RANK_LOG_DIR:-${REPO_ROOT}/train-on-robotwin/code/audit/rank_logs/robotwin_action_only_bs2_acc2_gbs32_50k_${RUN_STAMP}}"
ERROR_DIR="${ERROR_DIR:-${REPO_ROOT}/train-on-robotwin/code/audit/error_logs/robotwin_action_only_bs2_acc2_gbs32_50k_${RUN_STAMP}}"
RUN_NAME="${RUN_NAME:-robotwin_clean40_action_only_no_wm_50k_8gpu_bs2_acc2_gbs32_${RUN_STAMP}}"

mkdir -p "${OUTPUT_DIR}" "${RANK_LOG_DIR}" "${ERROR_DIR}"

cd "${OPENWAM_PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export WANDB_MODE=online
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN
export OPENWAM_DISTRIBUTED_TIMEOUT_SEC=1800
export OPENWAM_STEP_DIAG=0
export OPENWAM_PHASE_DIAG=0
export OPENWAM_TRAIN_LOG_EVERY=50
export ROBOTWIN_SPLIT_RANK_LOG_DIR="${RANK_LOG_DIR}"
export ROBOTWIN_SPLIT_ERROR_DIR="${ERROR_DIR}"
export MASTER_PORT="${MASTER_PORT:-29651}"
unset PYTORCH_CUDA_ALLOC_CONF

printf '[%s] starting OpenWAM action-only training bs2/acc2/gbs32/50k\n' "$(date -Is)"
printf 'output_dir=%s\nrank_log_dir=%s\nerror_dir=%s\nrun_name=%s\n' \
  "${OUTPUT_DIR}" "${RANK_LOG_DIR}" "${ERROR_DIR}" "${RUN_NAME}"

bash "${SCRIPT_DIR}/train_robotwin_split.sh" \
  dataloader=robotwin \
  dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset \
  dataloader.embodiment=aloha-agilex \
  dataloader.variant=clean_50 \
  model=dual_system \
  model/video_backbone=wan22_ti2v_5b \
  model.video_backbone.load_text_encoder=false \
  model.video_backbone.use_cached_text_embeddings=true \
  model.video_backbone.text_embedding_cache_dir="${TEXT_EMBED_CACHE_DIR}" \
  model.architecture.variant=joint_self_attn \
  model.architecture.attention_mask_mode=mutual \
  training.debug=false \
  training.batch_size=2 \
  training.gradient_accumulation_steps=2 \
  training.max_steps=50000 \
  training.num_epochs=null \
  training.save_steps=5000 \
  training.keep_last_k_ckpts=5 \
  training.dataset_num_workers=0 \
  training.use_gradient_checkpointing_offload=false \
  training.use_cached_text_embeddings=true \
  training.text_embedding_cache_dir="${TEXT_EMBED_CACHE_DIR}" \
  training.lambda_action=1.0 \
  training.lambda_video=0.0 \
  training.output_path="${OUTPUT_DIR}" \
  project.seed=42 \
  project.wandb.run_name="${RUN_NAME}"

printf '[%s] training finished\n' "$(date -Is)"
