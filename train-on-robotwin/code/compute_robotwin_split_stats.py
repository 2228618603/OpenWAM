#!/usr/bin/env python3
"""Compute normalization stats for the filtered RoboTwin train40_45eps split."""

from __future__ import annotations

import sys
from pathlib import Path

OPENWAM_PROJECT_ROOT = Path("/home/chw/code/packages/OpenWAM/OpenWAM")
LOCAL_CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(OPENWAM_PROJECT_ROOT))
sys.path.insert(0, str(LOCAL_CODE_ROOT))

from robotwin_split_runtime import TRAIN_TASKS_40, install_robotwin_split_filter, write_split_files


def main() -> None:
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import (
        atomic_save_stats_npy,
        cleanup_partial_stats_checkpoint,
        compute_multitask_robotwin_stats,
    )

    dataset_dir = "/mnt/data/chw/robotwin-dataset/dataset"
    output = (
        "/home/chw/code/packages/OpenWAM/train-on-robotwin/code/stats/wm_action_v1/"
        "train40_45eps_robotwin_clean_50_normalization_stats.npy"
    )
    write_split_files("/home/chw/code/packages/OpenWAM/train-on-robotwin/code/splits/wm_action_v1")
    install_robotwin_split_filter(
        dataset_dir=dataset_dir,
        tasks=TRAIN_TASKS_40,
        episode_spec="0-44",
    )
    stats = compute_multitask_robotwin_stats(
        dataset_dir=dataset_dir,
        embodiment="aloha-agilex",
        variant="clean_50",
        tasks=TRAIN_TASKS_40,
        checkpoint_path=None,
    )
    atomic_save_stats_npy(output, stats)
    cleanup_partial_stats_checkpoint(output)
    print(f"saved {output}")
    print(f"num_timesteps={stats.get('num_timesteps')}")
    for mode in ("joint", "eef"):
        if mode in stats:
            print(f"{mode}_dim={len(stats[mode]['mean'])}")


if __name__ == "__main__":
    main()
