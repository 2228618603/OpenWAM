#!/usr/bin/env python3
"""Create file-copy RoboTwin clean_50 splits for the WM/action experiment.

The mounted data filesystem used here does not support symlink or hardlink
creation reliably, so this helper intentionally copies only the clean_50 HDF5
files needed by each split.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


OOD_TASKS = [
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

ALL_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


def _safe_reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _link_file_force(src: Path, dst: Path, resume: bool = False) -> None:
    if resume and dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copyfile(src, dst)


def _copytree_force(src: Path, dst: Path, resume: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        if resume and dst.is_dir() and not dst.is_symlink():
            pass
        elif dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        out = dst / rel
        if item.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            _link_file_force(item, out, resume=resume)


def _link_variant_metadata(src_variant: Path, dst_variant: Path, resume: bool = False) -> None:
    scene_info = src_variant / "scene_info.json"
    if scene_info.exists():
        _link_file_force(scene_info, dst_variant / "scene_info.json", resume=resume)
    instructions = src_variant / "instructions"
    if instructions.exists():
        _copytree_force(instructions, dst_variant / "instructions", resume=resume)


def _link_task(src_root: Path, dst_root: Path, task: str, episodes: range, resume: bool = False) -> None:
    src_variant = src_root / task / "aloha-agilex_clean_50"
    src_data = src_variant / "data"
    if not src_data.is_dir():
        raise FileNotFoundError(f"missing source data dir: {src_data}")

    dst_variant = dst_root / task / "aloha-agilex_clean_50"
    dst_data = dst_variant / "data"
    dst_data.mkdir(parents=True, exist_ok=True)
    _link_variant_metadata(src_variant, dst_variant, resume=resume)

    for ep in episodes:
        src_file = src_data / f"episode{ep}.hdf5"
        if not src_file.exists():
            raise FileNotFoundError(f"missing source episode: {src_file}")
        _link_file_force(src_file, dst_data / src_file.name, resume=resume)


def _count_hdf5(root: Path) -> int:
    return sum(1 for _ in root.glob("*/aloha-agilex_clean_50/data/episode*.hdf5"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/mnt/data/chw/robotwin-dataset/dataset")
    parser.add_argument("--dst", default="/mnt/data/chw/robotwin-dataset/splits/wm_action_v1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    train_tasks = [task for task in ALL_TASKS if task not in set(OOD_TASKS)]

    if len(train_tasks) != 40 or len(OOD_TASKS) != 10:
        raise RuntimeError("bad task split cardinality")
    if set(train_tasks) & set(OOD_TASKS):
        raise RuntimeError("train/OOD overlap")

    if args.force and args.resume:
        raise ValueError("--force and --resume are mutually exclusive")
    if dst.exists() and not (args.force or args.resume):
        raise FileExistsError(f"{dst} exists; pass --force to replace or --resume to continue")

    dst.mkdir(parents=True, exist_ok=True)
    if args.force:
        for name in ("train40_45eps", "val40_5eps", "unseen10_50eps"):
            _safe_reset_dir(dst / name)
    else:
        for name in ("train40_45eps", "val40_5eps", "unseen10_50eps"):
            (dst / name).mkdir(parents=True, exist_ok=True)

    (dst / "train_tasks_40.txt").write_text("\n".join(train_tasks) + "\n", encoding="utf-8")
    (dst / "ood_tasks_10.txt").write_text("\n".join(OOD_TASKS) + "\n", encoding="utf-8")

    for task in train_tasks:
        _link_task(src, dst / "train40_45eps", task, range(45), resume=args.resume)
        _link_task(src, dst / "val40_5eps", task, range(45, 50), resume=args.resume)
    for task in OOD_TASKS:
        _link_task(src, dst / "unseen10_50eps", task, range(50), resume=args.resume)

    print(f"dst={dst}")
    print(f"train_tasks={len(train_tasks)} ood_tasks={len(OOD_TASKS)}")
    print(f"train_hdf5={_count_hdf5(dst / 'train40_45eps')}")
    print(f"val_hdf5={_count_hdf5(dst / 'val40_5eps')}")
    print(f"unseen_hdf5={_count_hdf5(dst / 'unseen10_50eps')}")


if __name__ == "__main__":
    main()
