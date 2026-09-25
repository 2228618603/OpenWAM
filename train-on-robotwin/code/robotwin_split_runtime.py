"""Runtime filters for OpenWAM RoboTwin split experiments.

This module deliberately lives outside the upstream OpenWAM package. It
monkey-patches RoboTwin discovery and episode globbing at process startup so
the original trainer can run unchanged while seeing only the requested task
and episode subset.
"""

from __future__ import annotations

import glob as _stdlib_glob
import os
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf

_ORIGINAL_GLOB = _stdlib_glob.glob
_INSTALLED = False
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CODE_ROOT = Path(__file__).resolve().parent

TRAIN_TASKS_40 = [
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_two",
    "stamp_seal",
]

OOD_TASKS_10 = [
    "adjust_bottle",
    "grab_roller",
    "place_container_plate",
    "move_pillbottle_pad",
    "open_microwave",
    "press_stapler",
    "stack_bowls_three",
    "handover_block",
    "move_can_pot",
    "turn_switch",
]


def parse_episode_spec(spec: str) -> set[str] | None:
    """Return allowed episode basenames for specs like ``0-44`` or ``all``."""
    spec = str(spec).strip().lower()
    if spec in {"", "all", "none", "null"}:
        return None
    allowed: set[str] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"bad episode range: {chunk}")
            allowed.update(f"episode{i}.hdf5" for i in range(start, end + 1))
        else:
            allowed.add(f"episode{int(chunk)}.hdf5")
    return allowed


def write_split_files(out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_tasks_40.txt").write_text("\n".join(TRAIN_TASKS_40) + "\n", encoding="utf-8")
    (out / "ood_tasks_10.txt").write_text("\n".join(OOD_TASKS_10) + "\n", encoding="utf-8")


def _filter_episode_files(files: Iterable[str], allowed: set[str] | None) -> list[str]:
    if allowed is None:
        return list(files)
    return [path for path in files if os.path.basename(path) in allowed]


def install_robotwin_split_filter(
    *,
    dataset_dir: str,
    tasks: list[str],
    episode_spec: str,
) -> None:
    """Patch OpenWAM RoboTwin discovery and stats file iteration."""
    global _INSTALLED
    if _INSTALLED:
        return
    import openwam.dataloader.robotwin as robotwin
    import openwam.dataloader.utils.stats_computation.robotwin_stats_computation as stats_mod

    dataset_dir_abs = os.path.abspath(dataset_dir)
    split_tasks = list(tasks)
    tasks_set = set(split_tasks)
    allowed_episodes = parse_episode_spec(episode_spec)
    original_discover = robotwin.discover_robotwin_roots

    def discover_filtered(dataset_dir_arg, embodiment, variant="clean_50", tasks=None):
        selected_tasks = split_tasks if tasks is None else [task for task in tasks if task in tasks_set]
        return original_discover(dataset_dir_arg, embodiment, variant, selected_tasks)

    def glob_filtered(pattern, *args, **kwargs):
        files = _ORIGINAL_GLOB(pattern, *args, **kwargs)
        if pattern.endswith("episode*.hdf5"):
            abs_pattern = os.path.abspath(pattern)
            if abs_pattern.startswith(dataset_dir_abs + os.sep):
                files = _filter_episode_files(files, allowed_episodes)
        return files

    robotwin.discover_robotwin_roots = discover_filtered
    robotwin.glob.glob = glob_filtered
    stats_mod.glob.glob = glob_filtered
    stats_mod.discover_robotwin_roots = discover_filtered
    _INSTALLED = True


def apply_train40_45eps_config(cfg) -> None:
    """Record the runtime split in the Hydra config and install filters."""
    dataset_dir = OmegaConf.select(cfg, "dataloader.dataset_dir")
    if not dataset_dir:
        raise ValueError("cfg.dataloader.dataset_dir is required")

    split_root = LOCAL_CODE_ROOT / "splits" / "wm_action_v1"
    stats_path = (
        LOCAL_CODE_ROOT
        / "stats"
        / "wm_action_v1"
        / "train40_45eps_robotwin_clean_50_normalization_stats.npy"
    )
    write_split_files(split_root)

    OmegaConf.update(cfg, "dataloader.tasks", list(TRAIN_TASKS_40), force_add=True)
    OmegaConf.update(cfg, "dataloader.episode_filter", "0-44", force_add=True)
    OmegaConf.update(cfg, "dataloader.normalization_stats_path", str(stats_path), force_add=True)
    OmegaConf.update(cfg, "experiment_split.name", "wm_action_v1_train40_45eps", force_add=True)
    OmegaConf.update(cfg, "experiment_split.ood_tasks", list(OOD_TASKS_10), force_add=True)

    install_robotwin_split_filter(
        dataset_dir=dataset_dir,
        tasks=TRAIN_TASKS_40,
        episode_spec="0-44",
    )
