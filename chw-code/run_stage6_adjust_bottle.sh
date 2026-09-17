#!/usr/bin/env bash
set -euo pipefail

OPENWAM_ROOT="${OPENWAM_ROOT:-/home/chw/code/packages/OpenWAM/OpenWAM}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat_copy}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/home/chw/miniconda3/envs/RoboTwin/bin/python}"
RUN_ROOT="${RUN_ROOT:-/home/chw/code/packages/OpenWAM/chw-code/runs/stage6_adjust_bottle_$(date +%Y%m%d_%H%M%S)}"
TASK_NAME="${TASK_NAME:-adjust_bottle}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
SIM_GPU="${SIM_GPU:-7}"
TEST_NUM="${TEST_NUM:-32}"
MAX_SEED_ATTEMPTS="${MAX_SEED_ATTEMPTS:-200}"
HOST="${HOST:-127.0.0.1}"

mkdir -p "${RUN_ROOT}"
echo "${RUN_ROOT}" > /home/chw/code/packages/OpenWAM/chw-code/runs/latest_stage6_dir.txt

run_one() {
    local name="$1"
    local port="$2"
    local log="${RUN_ROOT}/${name}.log"
    local episode_log="${RUN_ROOT}/${name}_episodes.jsonl"

    echo "[stage6] ${name} port=${port} test_num=${TEST_NUM}"
    ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
    ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON}" \
    ROBOTWIN_SKIP_VERSION_CHECK=1 \
    ROBOTWIN_ENABLE_PLANNER_FALLBACK=1 \
    ROBOTWIN_TEST_NUM="${TEST_NUM}" \
    ROBOTWIN_MAX_SEED_ATTEMPTS="${MAX_SEED_ATTEMPTS}" \
    ROBOTWIN_EPISODE_LOG="${episode_log}" \
    timeout 14400 bash "${OPENWAM_ROOT}/benchmarks/robotwin/single_eval.sh" \
        "${TASK_NAME}" "${TASK_CONFIG}" "stage6_${name}" "${SIM_GPU}" "${port}" "${HOST}" \
        > "${log}" 2>&1
    echo "[stage6] finished ${name}"
    tail -8 "${log}" || true
}

run_one wan21_vace_1_3b 8848
run_one cosmos25 8849
run_one cosmos3 8850
run_one wan22_ti2v_5b 8851
run_one wan21_i2v_14b 8852

python /home/chw/code/packages/OpenWAM/chw-code/summarize_passk.py "${RUN_ROOT}"
echo "[stage6] done: ${RUN_ROOT}"
