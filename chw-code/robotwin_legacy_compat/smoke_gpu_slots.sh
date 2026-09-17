#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${RESULT_ROOT:-/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/result}"
SMOKE_ROOT="${SMOKE_ROOT:-${RESULT_ROOT}/hard8_gpu_slot_smoke_$(date +%Y%m%d_%H%M%S)}"
GPUS_CSV="${GPUS_CSV:-1,2,7}"
MAX_SLOTS="${MAX_SLOTS:-3}"
BASE_PORT="${BASE_PORT:-9300}"
TASK_NAME="${TASK_NAME:-click_bell}"
BACKBONE="${BACKBONE:-wan21_vace_1_3b}"
CKPT_DIR="${CKPT_DIR:-/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention_wan21_vace_1_3b}"
JOB_SCRIPT="${JOB_SCRIPT:-/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/run_one_hard8_job.sh}"

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
        TARGET_N=1 TASK_CONFIG=demo_clean \
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
