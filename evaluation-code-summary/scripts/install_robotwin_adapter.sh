#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPENWAM_ROOT="${OPENWAM_ROOT:?set OPENWAM_ROOT to the OpenWAM repository root}"
TARGET_DIR="${OPENWAM_ROOT}/benchmarks/robotwin"
SOURCE_DIR="${PACKAGE_ROOT}/openwam-robotwin-adapter"

[[ -d "${TARGET_DIR}" ]] || {
    echo "target not found: ${TARGET_DIR}" >&2
    exit 2
}

install -m 0644 "${SOURCE_DIR}/eval_policy_wrapper.py" "${TARGET_DIR}/eval_policy_wrapper.py"
install -m 0644 "${SOURCE_DIR}/openwam2robotwin_interface.py" "${TARGET_DIR}/openwam2robotwin_interface.py"
install -m 0644 "${SOURCE_DIR}/prompt_template.py" "${TARGET_DIR}/prompt_template.py"
install -m 0755 "${SOURCE_DIR}/single_eval.sh" "${TARGET_DIR}/single_eval.sh"
install -m 0644 "${PACKAGE_ROOT}/configs/policy_config.yml" "${TARGET_DIR}/policy_config.yml"
install -m 0644 "${PACKAGE_ROOT}/configs/step_limits.yml" "${TARGET_DIR}/step_limits.yml"

echo "installed RoboTwin adapter files to ${TARGET_DIR}"
