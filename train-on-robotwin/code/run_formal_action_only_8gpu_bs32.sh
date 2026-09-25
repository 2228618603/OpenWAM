#!/usr/bin/env bash
set -euo pipefail

source /home/chw/miniconda3/etc/profile.d/conda.sh
conda activate openwam

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OPENWAM_PROJECT_ROOT="${REPO_ROOT}/OpenWAM"

cd "${OPENWAM_PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export WANDB_MODE=online
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV,COLL}"
export OPENWAM_DISTRIBUTED_TIMEOUT_SEC="${OPENWAM_DISTRIBUTED_TIMEOUT_SEC:-1800}"
export OPENWAM_STEP_DIAG="${OPENWAM_STEP_DIAG:-1}"
export OPENWAM_STEP_DIAG_EVERY="${OPENWAM_STEP_DIAG_EVERY:-10}"
TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-/mnt/data/chw/model/wm-function/cache/robotwin_text_embeddings/wan22_ti2v_5b_umt5_xxl_bf16_clean50}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-/mnt/data/chw/model/openwam_train_runs}"
TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH:-${TRAIN_RUN_ROOT}/robotwin_clean40_action_only_no_wm_50k_8gpu_bs1_accum4_cached_t5_cpuoffload_noforeach}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login "${WANDB_API_KEY}"
fi

COMMON_ARGS=(
  dataloader=robotwin
  dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset
  dataloader.embodiment=aloha-agilex
  dataloader.variant=clean_50
  model=dual_system
  model/video_backbone=wan22_ti2v_5b
  model.video_backbone.load_text_encoder=false
  model.video_backbone.use_cached_text_embeddings=true
  model.video_backbone.text_embedding_cache_dir="${TEXT_EMBED_CACHE_DIR}"
  model.architecture.variant=joint_self_attn
  model.architecture.attention_mask_mode=action_sees_video
  training.debug=false
  training.batch_size=1
  training.gradient_accumulation_steps=4
  training.max_steps=50000
  training.num_epochs=null
  training.zero_stage=2
  training.offload_optimizer_device=cpu
  +training.adamw_foreach=false
  +training.freeze_video_backbone=true
  +training.skip_zero_weight_video_loss=true
  training.save_steps=5000
  training.keep_last_k_ckpts=5
  training.dataset_num_workers=0
  training.use_gradient_checkpointing_offload=false
  training.use_cached_text_embeddings=true
  training.text_embedding_cache_dir="${TEXT_EMBED_CACHE_DIR}"
  training.lambda_action=1.0
  project.seed=42
)

echo "[$(date -Is)] starting Action-only / No-WM formal training (8 GPUs, per-GPU bs=1, grad_accum=4, effective global bs=32, optimizer CPU offload, AdamW foreach disabled)"
ROBOTWIN_SPLIT_RANK_LOG_DIR="${REPO_ROOT}/train-on-robotwin/code/audit/rank_logs/formal_action_only_no_wm_50k_8gpu_${RUN_STAMP}" \
ROBOTWIN_SPLIT_ERROR_DIR="${REPO_ROOT}/train-on-robotwin/code/audit/error_logs/formal_action_only_no_wm_50k_8gpu_${RUN_STAMP}" \
MASTER_PORT=29630 \
bash "${SCRIPT_DIR}/train_robotwin_split.sh" \
  "${COMMON_ARGS[@]}" \
  training.lambda_video=0.0 \
  training.output_path="${TRAIN_OUTPUT_PATH}" \
  project.wandb.run_name=robotwin_clean40_action_only_no_wm_50k_8gpu_bs1_accum4_cached_t5_cpuoffload_noforeach

echo "[$(date -Is)] Action-only / No-WM formal training finished"
