#!/usr/bin/env bash
set -euo pipefail

cd /home/chw/code/packages/OpenWAM

CACHE_DIR=/mnt/data/chw/model/wm-function/cache/robotwin_text_embeddings/wan22_ti2v_5b_umt5_xxl_bf16_clean50
PIPELINE_SESSION=robotwin_t5_no_wm_8gpu_5w

while tmux has-session -t "${PIPELINE_SESSION}" 2>/dev/null; do
  echo "===== $(date -Is) ====="
  echo "[gpu]"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

  echo "[cache_count]"
  find "${CACHE_DIR}/embeddings" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l || true

  latest_shard=$(ls -td train-on-robotwin/code/audit/tmux_logs/preencode_shards_* 2>/dev/null | head -1 || true)
  if [ -n "${latest_shard}" ]; then
    echo "[preencode_tail] ${latest_shard}"
    for f in "${latest_shard}"/shard_*.log "${latest_shard}"/manifest.log; do
      [ -f "${f}" ] || continue
      printf -- '--- %s\n' "$(basename "${f}")"
      tail -n 2 "${f}" || true
    done
  fi

  latest_rank=$(ls -td train-on-robotwin/code/audit/rank_logs/formal_action_only_no_wm_50k_8gpu_* 2>/dev/null | head -1 || true)
  if [ -n "${latest_rank}" ]; then
    echo "[rank_scan] ${latest_rank}"
    rg -i 'CUDA out of memory|outofmemory|\boom\b|Using cached text|models_t5|openwam_step_diag|step=17[0-9]|step=180|step=300' "${latest_rank}" || true
  fi

  sleep 60
done

echo "===== $(date -Is) pipeline tmux ended ====="
