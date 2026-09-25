#!/usr/bin/env bash
set -euo pipefail

cd /home/chw/code/packages/OpenWAM/OpenWAM

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

exec torchrun \
  --nnodes "${NNODES:-1}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --node_rank "${NODE_RANK:-0}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" \
  --master_port "${MASTER_PORT:-29500}" \
  /home/chw/code/packages/OpenWAM/train-on-robotwin/code/train_robotwin_split.py \
  "$@"
