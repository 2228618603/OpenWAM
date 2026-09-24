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

OPENWAM_ROOT="${OPENWAM_ROOT:?set OPENWAM_ROOT to the OpenWAM repository root}"
OPENWAM_CONDA_SH="${OPENWAM_CONDA_SH:-}"
OPENWAM_CONDA_ENV="${OPENWAM_CONDA_ENV:-openwam}"
OPENWAM_PYTHON="${OPENWAM_PYTHON:-}"

mkdir -p "$(dirname "${LOG_FILE}")"

if [[ -n "${OPENWAM_PYTHON}" ]]; then
    PY_BOOTSTRAP="exec '${OPENWAM_PYTHON}' scripts/deploy.py"
else
    [[ -n "${OPENWAM_CONDA_SH}" ]] || {
        echo "set OPENWAM_PYTHON or OPENWAM_CONDA_SH" >&2
        exit 2
    }
    PY_BOOTSTRAP="source '${OPENWAM_CONDA_SH}'; conda activate '${OPENWAM_CONDA_ENV}'; exec python scripts/deploy.py"
fi

setsid bash -c "
    set -euo pipefail
    cd '${OPENWAM_ROOT}'
    export CUDA_VISIBLE_DEVICES='${GPU}'
    ${PY_BOOTSTRAP} \
        --ckpt-dir '${CKPT_DIR}' \
        --device cuda:0 \
        --port '${PORT}' \
        --compile-enabled false
" > "${LOG_FILE}" 2>&1 < /dev/null &

echo $!
