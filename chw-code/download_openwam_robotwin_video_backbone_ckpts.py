#!/usr/bin/env python3
"""Download OpenWAM RoboTwin video-backbone study checkpoints.

This helper intentionally lives outside the upstream OpenWAM source tree. It
downloads directly into /mnt/data so large checkpoints do not occupy the repo.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


CHECKPOINTS = {
    "wan21_vace_1_3b": "OpenWAM/robotwin_dual_system_joint_self_attention_wan21_vace_1_3b",
    "cosmos25": "OpenWAM/robotwin_dual_system_joint_self_attention_cosmos25",
    "cosmos3": "OpenWAM/robotwin_dual_system_joint_self_attention_cosmos3",
    "wan22_ti2v_5b": "OpenWAM/robotwin_dual_system_joint_self_attention",
    "wan21_i2v_14b": "OpenWAM/robotwin_dual_system_joint_self_attention_wan21_i2v_14b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone",
        help="Directory where checkpoint folders will be created.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(CHECKPOINTS),
        default=None,
        help="Optional subset to download.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="Optional Hugging Face endpoint, for example https://hf-mirror.com.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    keys = args.only or list(CHECKPOINTS)
    for key in keys:
        repo_id = CHECKPOINTS[key]
        target = output_root / repo_id.split("/", 1)[1]
        print(f"[download] {key}: {repo_id} -> {target}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        config = target / "config.yaml"
        if not config.is_file():
            raise FileNotFoundError(f"Downloaded checkpoint is missing config.yaml: {target}")
        print(f"[ok] {key}: found {config}", flush=True)


if __name__ == "__main__":
    main()
