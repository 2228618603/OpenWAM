#!/usr/bin/env python3
"""Compose and compare Action-only / WM Hydra configs for the split run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENWAM_PROJECT_ROOT = REPO_ROOT / "OpenWAM"
LOCAL_CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(OPENWAM_PROJECT_ROOT))
sys.path.insert(0, str(LOCAL_CODE_ROOT))

from robotwin_split_runtime import apply_train40_45eps_config


COMMON_OVERRIDES = [
    "dataloader=robotwin",
    "dataloader.dataset_dir=/mnt/data/chw/robotwin-dataset/dataset",
    "dataloader.embodiment=aloha-agilex",
    "dataloader.variant=clean_50",
    "model=dual_system",
    "model/video_backbone=wan22_ti2v_5b",
    "model.architecture.variant=joint_self_attn",
    "model.architecture.attention_mask_mode=mutual",
    "training.batch_size=4",
    "training.gradient_accumulation_steps=2",
    "training.max_steps=50000",
    "training.num_epochs=null",
    "training.save_steps=5000",
    "training.keep_last_k_ckpts=5",
    "training.lambda_action=1.0",
    "project.seed=42",
]


def _compose(extra: list[str]):
    with initialize_config_dir(version_base=None, config_dir=str(OPENWAM_PROJECT_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=COMMON_OVERRIDES + extra)
    apply_train40_45eps_config(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def _diff(a, b, prefix=""):
    if type(a) is not type(b):
        yield prefix, a, b
        return
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            new_prefix = f"{prefix}.{key}" if prefix else key
            if key not in a or key not in b:
                yield new_prefix, a.get(key), b.get(key)
            else:
                yield from _diff(a[key], b[key], new_prefix)
    elif isinstance(a, list):
        if a != b:
            yield prefix, a, b
    elif a != b:
        yield prefix, a, b


def main() -> None:
    no_wm = _compose(
        [
            "training.lambda_video=0.0",
            "training.output_path=/mnt/data/chw/model/wm-function/robotwin_clean40_action_only_no_wm_50k",
            "project.wandb.run_name=robotwin_clean40_action_only_no_wm_50k",
        ]
    )
    wm = _compose(
        [
            "training.lambda_video=1.0",
            "training.output_path=/mnt/data/chw/model/wm-function/robotwin_clean40_wm_50k",
            "project.wandb.run_name=robotwin_clean40_wm_50k",
        ]
    )
    allowed = {
        "training.lambda_video",
        "training.output_path",
        "project.wandb.run_name",
    }
    diffs = list(_diff(no_wm, wm))
    unexpected = [item for item in diffs if item[0] not in allowed]

    out_dir = LOCAL_CODE_ROOT / "audit" / "config"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "action_only_config.json").write_text(json.dumps(no_wm, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "wm_config.json").write_text(json.dumps(wm, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "diffs.json").write_text(json.dumps(diffs, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"diff_count={len(diffs)}")
    for path, left, right in diffs:
        print(f"{path}: {left!r} -> {right!r}")
    if unexpected:
        raise SystemExit(f"unexpected diffs: {[item[0] for item in unexpected]}")
    print("CONFIG_DIFF_OK")


if __name__ == "__main__":
    main()
