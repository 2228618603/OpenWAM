#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 CKPT_DIR GPU PORT LOG_FILE" >&2
    exit 2
fi

CKPT_DIR="$1"
GPU="$2"
PORT="$3"
LOG_FILE="$4"

mkdir -p "$(dirname "${LOG_FILE}")"

setsid bash -c "
    source /home/chw/miniconda3/etc/profile.d/conda.sh
    conda activate openwam
    cd /home/chw/code/packages/OpenWAM/OpenWAM
    export CUDA_VISIBLE_DEVICES='${GPU}'
    exec python scripts/deploy.py \
        --ckpt-dir '${CKPT_DIR}' \
        --device cuda:0 \
        --port '${PORT}' \
        --compile-enabled false
" > "${LOG_FILE}" 2>&1 < /dev/null &

echo $!
