#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 RUN_ROOT BACKBONE TASK_NAME GPU PORT CKPT_DIR" >&2
    exit 2
fi

RUN_ROOT="$1"
BACKBONE="$2"
TASK_NAME="$3"
GPU="$4"
PORT="$5"
CKPT_DIR="$6"

TARGET_N="${TARGET_N:-32}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
OPENWAM_ROOT="${OPENWAM_ROOT:-/home/chw/code/packages/OpenWAM/OpenWAM}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat_copy}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/home/chw/miniconda3/envs/RoboTwin/bin/python}"
START_SERVER="${START_SERVER:-/home/chw/code/packages/OpenWAM/chw-code/start_openwam_server.sh}"
HOST="${HOST:-127.0.0.1}"

EP_DIR="${RUN_ROOT}/episodes"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${EP_DIR}" "${LOG_DIR}"

JOB_ID="${BACKBONE}__${TASK_NAME}"
EPISODE_LOG="${EP_DIR}/${JOB_ID}.jsonl"
EVAL_LOG="${LOG_DIR}/${JOB_ID}.log"
SERVER_LOG="${LOG_DIR}/${JOB_ID}__server.log"

count_records() {
    [[ -f "${EPISODE_LOG}" ]] || { echo 0; return 0; }
    wc -l < "${EPISODE_LOG}" | tr -d ' '
}

wait_port() {
    local port="$1"
    local deadline=$((SECONDS + 900))
    until ss -ltn | awk '{print $4}' | grep -q ":${port}$"; do
        if (( SECONDS > deadline )); then
            echo "[job:${JOB_ID}] ERROR: port ${port} did not open in time" >&2
            return 1
        fi
        sleep 3
    done
}

stop_pid() {
    local pid="${1:-}"
    [[ -n "${pid}" ]] || return 0
    kill "${pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${pid}" 2>/dev/null || true
}

have="$(count_records)"
if (( have >= TARGET_N )); then
    echo "[job:${JOB_ID}] skip: already have ${have}/${TARGET_N}"
    exit 0
fi

need=$((TARGET_N - have))
echo "[job:${JOB_ID}] start gpu=${GPU} port=${PORT} have=${have} need=${need}"

server_pid="$("${START_SERVER}" "${CKPT_DIR}" "${GPU}" "${PORT}" "${SERVER_LOG}")"
trap 'stop_pid "${server_pid}"' EXIT
wait_port "${PORT}"

export ROBOTWIN_PATH
export ROBOTWIN_PYTHON
export ROBOTWIN_SKIP_VERSION_CHECK=1
export ROBOTWIN_ENABLE_PLANNER_FALLBACK=1
export ROBOTWIN_EXPERT_CHECK="${ROBOTWIN_EXPERT_CHECK:-1}"
export ROBOTWIN_TEST_NUM="${need}"
export ROBOTWIN_MAX_SEED_ATTEMPTS="${ROBOTWIN_MAX_SEED_ATTEMPTS:-$((need * 30 + 200))}"
export ROBOTWIN_EPISODE_LOG="${EPISODE_LOG}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-openwam}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/home/chw/miniconda3/envs/RoboTwin/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/home/chw/miniconda3/envs/RoboTwin/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json}"

set +e
timeout "${JOB_TIMEOUT:-21600}" \
    bash "${OPENWAM_ROOT}/benchmarks/robotwin/single_eval.sh" \
    "${TASK_NAME}" "${TASK_CONFIG}" "hard8_${BACKBONE}" "${GPU}" "${PORT}" "${HOST}" \
    >> "${EVAL_LOG}" 2>&1
eval_status=$?
set -e

final_have="$(count_records)"
echo "[job:${JOB_ID}] eval_status=${eval_status} records=${final_have}/${TARGET_N}"

stop_pid "${server_pid}"
trap - EXIT

if (( eval_status != 0 )); then
    exit "${eval_status}"
fi
if (( final_have < TARGET_N )); then
    echo "[job:${JOB_ID}] ERROR: incomplete records ${final_have}/${TARGET_N}" >&2
    exit 1
fi
