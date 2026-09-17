#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/chw/code/packages/OpenWAM/chw-code/runs/stage6_adjust_bottle_20260916_085151}"
SERVER_GPU="${SERVER_GPU:-2}"
SIM_GPU="${SIM_GPU:-7}"
PORT="${PORT:-8848}"
TARGET_N="${TARGET_N:-32}"
OPENWAM_ROOT="${OPENWAM_ROOT:-/home/chw/code/packages/OpenWAM/OpenWAM}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat_copy}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/home/chw/miniconda3/envs/RoboTwin/bin/python}"

count_records() {
    local name="$1"
    find "${RUN_ROOT}" -maxdepth 1 -type f -name "${name}*episodes.jsonl" -print0 \
        | xargs -0 -r wc -l \
        | awk 'END {print ($1 == "" ? 0 : $1)}'
}

wait_port() {
    local port="$1"
    local deadline=$((SECONDS + 600))
    until ss -ltn | awk '{print $4}' | grep -q ":${port}$"; do
        if (( SECONDS > deadline )); then
            echo "[stage6] ERROR: port ${port} did not open in time" >&2
            return 1
        fi
        sleep 5
    done
}

stop_pid() {
    local pid="${1:-}"
    [[ -n "${pid}" ]] || return 0
    kill "${pid}" 2>/dev/null || true
    sleep 5
    kill -9 "${pid}" 2>/dev/null || true
}

run_model() {
    local name="$1"
    local ckpt="$2"
    local have
    have="$(count_records "${name}")"
    if (( have >= TARGET_N )); then
        echo "[stage6] skip ${name}: already have ${have}/${TARGET_N} records"
        return 0
    fi

    local need=$((TARGET_N - have))
    local server_log="${RUN_ROOT}/${name}_server.log"
    local eval_log="${RUN_ROOT}/${name}.log"
    local episode_log="${RUN_ROOT}/${name}_episodes.jsonl"
    if (( have > 0 )); then
        eval_log="${RUN_ROOT}/${name}_resume.log"
        episode_log="${RUN_ROOT}/${name}_resume_episodes.jsonl"
    fi

    echo "[stage6] start server ${name} gpu=${SERVER_GPU} port=${PORT}"
    local server_pid
    server_pid="$("/home/chw/code/packages/OpenWAM/chw-code/start_openwam_server.sh" "${ckpt}" "${SERVER_GPU}" "${PORT}" "${server_log}")"
    trap 'stop_pid "${server_pid}"' RETURN
    wait_port "${PORT}"
    echo "[stage6] server ready ${name}; running ${need} episode(s)"

    ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
    ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON}" \
    ROBOTWIN_SKIP_VERSION_CHECK=1 \
    ROBOTWIN_ENABLE_PLANNER_FALLBACK=1 \
    ROBOTWIN_EXPERT_CHECK="${ROBOTWIN_EXPERT_CHECK:-1}" \
    ROBOTWIN_TEST_NUM="${need}" \
    ROBOTWIN_MAX_SEED_ATTEMPTS="$((need * 20 + 100))" \
    ROBOTWIN_EPISODE_LOG="${episode_log}" \
    timeout 14400 bash "${OPENWAM_ROOT}/benchmarks/robotwin/single_eval.sh" \
        adjust_bottle demo_clean "stage6_${name}" "${SIM_GPU}" "${PORT}" 127.0.0.1 \
        > "${eval_log}" 2>&1

    echo "[stage6] finished ${name}: $(count_records "${name}")/${TARGET_N}"
    stop_pid "${server_pid}"
    trap - RETURN
}

BASE="/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone"

run_model wan21_vace_1_3b "${BASE}/robotwin_dual_system_joint_self_attention_wan21_vace_1_3b"
run_model cosmos25 "${BASE}/robotwin_dual_system_joint_self_attention_cosmos25"
run_model cosmos3 "${BASE}/robotwin_dual_system_joint_self_attention_cosmos3"
run_model wan22_ti2v_5b "${BASE}/robotwin_dual_system_joint_self_attention"
run_model wan21_i2v_14b "${BASE}/robotwin_dual_system_joint_self_attention_wan21_i2v_14b"

python /home/chw/code/packages/OpenWAM/chw-code/summarize_passk.py "${RUN_ROOT}"
