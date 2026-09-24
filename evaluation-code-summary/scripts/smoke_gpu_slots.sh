#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULT_ROOT="${RESULT_ROOT:-./eval-result}"
SMOKE_ROOT="${SMOKE_ROOT:-${RESULT_ROOT}/gpu_slot_smoke_$(date +%Y%m%d_%H%M%S)}"
GPUS_CSV="${GPUS_CSV:-0}"
MAX_SLOTS="${MAX_SLOTS:-1}"
BASE_PORT="${BASE_PORT:-9300}"
TASK_NAME="${TASK_NAME:-click_bell}"
BACKBONE="${BACKBONE:-wan21_vace_1_3b}"
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to one OpenWAM RoboTwin checkpoint directory}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/run_one_robotwin_job.sh}"

mkdir -p "${SMOKE_ROOT}/logs"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"

echo "smoke_root=${SMOKE_ROOT}"
echo "gpu,slots,status" > "${SMOKE_ROOT}/gpu_slot_results.csv"

run_trial() {
    local gpu="$1"
    local slots="$2"
    local trial_root="${SMOKE_ROOT}/gpu${gpu}_slots${slots}"
    mkdir -p "${trial_root}/logs"
    echo "[smoke] gpu=${gpu} slots=${slots}"

    local pids=()
    for ((i=0; i<slots; i++)); do
        local port=$((BASE_PORT + gpu * 20 + i))
        local run_root="${trial_root}/slot_${i}"
        TARGET_N=1 TASK_CONFIG="${TASK_CONFIG:-demo_clean}" \
            "${JOB_SCRIPT}" "${run_root}" "${BACKBONE}" "${TASK_NAME}" "${gpu}" "${port}" "${CKPT_DIR}" \
            > "${trial_root}/logs/slot_${i}.log" 2>&1 &
        pids+=("$!")
        sleep 8
    done

    local ok=1
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            ok=0
        fi
    done

    if (( ok == 1 )); then
        echo "${gpu},${slots},ok" | tee -a "${SMOKE_ROOT}/gpu_slot_results.csv"
        return 0
    fi
    echo "${gpu},${slots},fail" | tee -a "${SMOKE_ROOT}/gpu_slot_results.csv"
    return 1
}

for gpu in "${GPUS[@]}"; do
    best=0
    for ((slots=1; slots<=MAX_SLOTS; slots++)); do
        if run_trial "${gpu}" "${slots}"; then
            best="${slots}"
        else
            break
        fi
        sleep 5
    done
    echo "${gpu},best,${best}" | tee -a "${SMOKE_ROOT}/gpu_slot_results.csv"
done

echo "[smoke] complete: ${SMOKE_ROOT}"
