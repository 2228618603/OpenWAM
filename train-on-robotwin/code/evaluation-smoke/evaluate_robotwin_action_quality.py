#!/usr/bin/env python3
"""Smoke-evaluate RoboTwin action quality for an OpenWAM checkpoint.

This script is intentionally small and close to the training path:

* datasets are built through ``openwam.dataloader.registry.build_dataset``;
* model loading uses the deploy checkpoint loader;
* optional diffusion-loss probing uses ``architecture.prepare_inputs`` and
  ``architecture.compute_loss``;
* generation probing calls ``architecture.generate`` and compares the generated
  raw action chunk against the ground-truth raw action chunk.

The default probes are the three splits from the training plan:

* train_seen: trained tasks, trained episodes;
* val_seen: trained tasks, held-out episodes;
* unseen_task: OOD tasks, all clean episodes if the split dir is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
OPENWAM_PROJECT_ROOT = REPO_ROOT / "OpenWAM"
LOCAL_CODE_ROOT = REPO_ROOT / "train-on-robotwin" / "code"

sys.path.insert(0, str(OPENWAM_PROJECT_ROOT))
sys.path.insert(0, str(LOCAL_CODE_ROOT))

from robotwin_split_runtime import OOD_TASKS_10, TRAIN_TASKS_40  # noqa: E402


DEFAULT_SPLIT_ROOT = Path("/mnt/data/chw/robotwin-dataset/splits/wm_action_v1")
DEFAULT_RAW_DATASET_DIR = Path("/mnt/data/chw/robotwin-dataset/dataset")
DEFAULT_STATS_PATH = (
    REPO_ROOT
    / "train-on-robotwin"
    / "code"
    / "stats"
    / "wm_action_v1"
    / "train40_45eps_robotwin_clean_50_normalization_stats.npy"
)


@dataclass
class MetricAccumulator:
    count: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    max_abs: float = 0.0
    per_dim_abs: np.ndarray | None = None
    per_dim_sq: np.ndarray | None = None
    per_dim_count: np.ndarray | None = None
    extra: dict[str, list[float]] = field(default_factory=dict)

    def update(self, pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> None:
        pred = np.asarray(pred, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if pred.shape != target.shape:
            raise ValueError(f"prediction shape {pred.shape} != target shape {target.shape}")

        err = pred - target
        if mask is None:
            mask = np.ones(err.shape, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != err.shape:
                mask = np.broadcast_to(mask, err.shape)

        valid = err[mask]
        if valid.size == 0:
            return

        abs_err = np.abs(valid)
        sq_err = valid * valid
        self.count += int(valid.size)
        self.sum_abs += float(abs_err.sum())
        self.sum_sq += float(sq_err.sum())
        self.max_abs = max(self.max_abs, float(abs_err.max()))

        dim_mask = mask.reshape(-1, mask.shape[-1])
        dim_err = err.reshape(-1, err.shape[-1])
        if self.per_dim_abs is None:
            dim = err.shape[-1]
            self.per_dim_abs = np.zeros(dim, dtype=np.float64)
            self.per_dim_sq = np.zeros(dim, dtype=np.float64)
            self.per_dim_count = np.zeros(dim, dtype=np.float64)
        self.per_dim_abs += (np.abs(dim_err) * dim_mask).sum(axis=0)
        self.per_dim_sq += ((dim_err * dim_err) * dim_mask).sum(axis=0)
        self.per_dim_count += dim_mask.sum(axis=0)

    def add_extra(self, name: str, value: float) -> None:
        if math.isfinite(float(value)):
            self.extra.setdefault(name, []).append(float(value))

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            base = {"num_values": 0, "mae": None, "mse": None, "rmse": None, "max_abs": None}
        else:
            mse = self.sum_sq / self.count
            base = {
                "num_values": self.count,
                "mae": self.sum_abs / self.count,
                "mse": mse,
                "rmse": math.sqrt(mse),
                "max_abs": self.max_abs,
            }
        for key, values in self.extra.items():
            base[key] = float(np.mean(values)) if values else None

        if self.per_dim_count is not None:
            denom = np.maximum(self.per_dim_count, 1.0)
            base["per_dim_mae"] = (self.per_dim_abs / denom).tolist()
            base["per_dim_mse"] = (self.per_dim_sq / denom).tolist()
        return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenWAM RoboTwin checkpoint with action MSE/MAE smoke metrics."
    )
    parser.add_argument("checkpoint_dir", help="Run/checkpoint directory containing config.yaml and checkpoint_step_*.safetensors.")
    parser.add_argument("--ckpt-name", default=None, help="Specific checkpoint filename. Defaults to latest step.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("generation", "loss", "both"), default="generation")
    parser.add_argument("--max-samples", type=int, default=8, help="Samples per split.")
    parser.add_argument("--denoise-steps", type=int, default=8, help="Generation denoising steps for smoke eval.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_RAW_DATASET_DIR), help="Original RoboTwin dataset root.")
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT), help="Split root with train40/val40/unseen dirs.")
    parser.add_argument("--stats-path", default=str(DEFAULT_STATS_PATH), help="Shared train normalization stats.")
    parser.add_argument("--out-dir", default=None, help="Directory for JSON/CSV outputs. Defaults under this script.")
    parser.add_argument("--no-csv", action="store_true", help="Skip per-sample CSV output.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nonempty_robotwin_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*/aloha-agilex_clean_50/data/episode*.hdf5"))


def build_eval_configs(saved_cfg, args: argparse.Namespace) -> dict[str, Any]:
    base_dl = deepcopy(saved_cfg.dataloader)
    OmegaConf.update(base_dl, "split", "val", force_add=True)
    OmegaConf.update(base_dl, "variant", "clean_50", force_add=True)
    OmegaConf.update(base_dl, "normalization_stats_path", args.stats_path, force_add=True)

    split_root = Path(args.split_root)
    train_dir = split_root / "train40_45eps"
    val_dir = split_root / "val40_5eps"
    unseen_dir = split_root / "unseen10_50eps"

    configs = {}
    if nonempty_robotwin_dir(train_dir):
        cfg = deepcopy(base_dl)
        OmegaConf.update(cfg, "dataset_dir", str(train_dir), force_add=True)
        OmegaConf.update(cfg, "tasks", list(TRAIN_TASKS_40), force_add=True)
        configs["train_seen"] = cfg

    if nonempty_robotwin_dir(val_dir):
        cfg = deepcopy(base_dl)
        OmegaConf.update(cfg, "dataset_dir", str(val_dir), force_add=True)
        OmegaConf.update(cfg, "tasks", list(TRAIN_TASKS_40), force_add=True)
        configs["val_seen"] = cfg

    if nonempty_robotwin_dir(unseen_dir):
        cfg = deepcopy(base_dl)
        OmegaConf.update(cfg, "dataset_dir", str(unseen_dir), force_add=True)
        OmegaConf.update(cfg, "tasks", list(OOD_TASKS_10), force_add=True)
        configs["unseen_task"] = cfg
    else:
        cfg = deepcopy(base_dl)
        OmegaConf.update(cfg, "dataset_dir", args.dataset_dir, force_add=True)
        OmegaConf.update(cfg, "tasks", list(OOD_TASKS_10), force_add=True)
        configs["unseen_task"] = cfg

    if not configs:
        raise FileNotFoundError(
            f"No usable split datasets under {split_root}; train_seen/val_seen are missing or empty."
        )
    return configs


def select_indices(length: int, max_samples: int) -> list[int]:
    if length <= 0:
        return []
    n = min(max_samples, length)
    if n == length:
        return list(range(length))
    return sorted(set(int(x) for x in np.linspace(0, length - 1, num=n)))


def raw_action_from_sample(dataset, sample: dict, key: str) -> np.ndarray:
    value = sample[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(dataset.denormalize_action(value), dtype=np.float32)


def build_valid_mask(sample: dict, raw_dim: int) -> np.ndarray | None:
    mask = sample.get("action_mask")
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 2 and mask.shape[-1] != raw_dim:
        # Unified-space mask. Gather the active dimensions via the same inverse
        # path used for actions; nonzero means the raw dim was active.
        gathered = raw_action_mask_from_unified(mask)
        if gathered is not None:
            mask = gathered
    if mask.ndim == 1:
        mask = np.expand_dims(mask, axis=-1)
    if mask.shape[-1] == 1:
        mask = np.broadcast_to(mask, (mask.shape[0], raw_dim))
    if mask.shape[-1] != raw_dim:
        return None
    return mask


def raw_action_mask_from_unified(mask: np.ndarray) -> np.ndarray | None:
    # The RoboTwin split config maps raw EEF dims to unified slots
    # [0-9, 34-43]. Keep this helper local so the output metrics stay in raw
    # 20-D space even when the dataloader emits 80-D masks.
    raw_slots = list(range(0, 10)) + list(range(34, 44))
    if mask.shape[-1] <= max(raw_slots):
        return None
    return mask[..., raw_slots]


def run_generation_probe(architecture, dataset, sample: dict, args: argparse.Namespace, sample_seed: int) -> np.ndarray:
    from openwam.deploy.denoise_schedule import make_schedule

    raw_proprio = raw_action_from_sample(dataset, sample, "proprio")
    raw_proprio = np.asarray(raw_proprio, dtype=np.float32)
    if raw_proprio.ndim == 2 and raw_proprio.shape[0] == 1:
        raw_proprio = raw_proprio[0]

    video_num_frames = len(sample["video"])
    action_num_frames = int(sample["action"].shape[0]) + 1
    schedule = make_schedule(
        "sync",
        video_scheduler=architecture.video_scheduler,
        action_scheduler=architecture.action_scheduler,
        num_steps=int(args.denoise_steps),
        shift=getattr(architecture.action_backbone, "shift_action", None) or 5.0,
        shift_video=getattr(architecture.video_backbone, "shift_video", None),
    )

    result = architecture.generate(
        schedule,
        sample["prompt"],
        first_frame_image=sample.get("first_frame_image"),
        num_frames=video_num_frames,
        action_num_frames=action_num_frames,
        height=int(OmegaConf.select(args.saved_cfg, "dataloader.height", default=384)),
        width=int(OmegaConf.select(args.saved_cfg, "dataloader.width", default=320)),
        seed=int(sample_seed),
        tiled=True,
        num_inference_steps=int(args.denoise_steps),
        decode_video=False,
        proprio=torch.from_numpy(raw_proprio),
    )
    return np.asarray(result["actions"], dtype=np.float32)


def run_loss_probe(architecture, sample: dict, args: argparse.Namespace) -> dict[str, float]:
    architecture.init_training_schedulers(1000)
    architecture.set_training_runtime(
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    )
    with torch.no_grad():
        inputs = architecture.prepare_inputs([sample])
        result = architecture.compute_loss(**inputs, lambda_video=0.0, lambda_action=1.0)
    return {
        "loss_action": float(result["loss_action"].detach().float().cpu().item()),
        "loss_total_action_only": float(result["loss"].detach().float().cpu().item()),
    }


def evaluate_split(name: str, dataset, architecture, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    indices = select_indices(len(dataset), int(args.max_samples))
    metrics = MetricAccumulator()
    rows: list[dict[str, Any]] = []

    t0 = time.time()
    for ordinal, idx in enumerate(indices):
        sample_seed = int(args.seed) + ordinal
        seed_everything(sample_seed)
        sample = dataset[idx]
        target = raw_action_from_sample(dataset, sample, "action")
        mask = build_valid_mask(sample, target.shape[-1])

        row: dict[str, Any] = {
            "split": name,
            "sample_ordinal": ordinal,
            "dataset_index": idx,
            "task_name": sample.get("task_name"),
            "episode_path": sample.get("episode_path"),
            "start_frame": int(sample.get("start_frame", -1)),
            "seed": sample_seed,
        }

        if args.mode in ("generation", "both"):
            pred = run_generation_probe(architecture, dataset, sample, args, sample_seed)
            pred = pred[: target.shape[0], : target.shape[1]]
            target_cmp = target[: pred.shape[0], : pred.shape[1]]
            mask_cmp = None if mask is None else mask[: pred.shape[0], : pred.shape[1]]
            metrics.update(pred, target_cmp, mask_cmp)
            diff = pred - target_cmp
            if mask_cmp is not None:
                diff = diff[mask_cmp]
            row.update(
                {
                    "generation_mae": float(np.mean(np.abs(diff))),
                    "generation_mse": float(np.mean(diff * diff)),
                    "generation_rmse": float(math.sqrt(float(np.mean(diff * diff)))),
                }
            )

        if args.mode in ("loss", "both"):
            loss_values = run_loss_probe(architecture, sample, args)
            for key, value in loss_values.items():
                metrics.add_extra(key, value)
                row[key] = value

        rows.append(row)
        print(f"[{name}] {ordinal + 1}/{len(indices)} idx={idx} task={row['task_name']}")

    summary = metrics.summary()
    summary.update(
        {
            "split": name,
            "num_samples": len(indices),
            "dataset_len": len(dataset),
            "elapsed_s": round(time.time() - t0, 3),
        }
    )
    return summary, rows


def write_outputs(out_dir: Path, result: dict[str, Any], rows: list[dict[str, Any]], write_csv: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "action_quality_summary.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {json_path}")

    if write_csv and rows:
        csv_path = out_dir / "action_quality_samples.csv"
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {csv_path}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    from openwam.dataloader.registry import build_dataset
    from openwam.deploy.model_loader import load_from_checkpoint_dir

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    cfg, architecture = load_from_checkpoint_dir(str(ckpt_dir), device=args.device, ckpt_name=args.ckpt_name)
    args.saved_cfg = cfg

    configs = build_eval_configs(cfg, args)
    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for split_name, dl_cfg in configs.items():
        print("=" * 80)
        print(f"Building dataset: {split_name}")
        print(OmegaConf.to_yaml(dl_cfg))
        dataset = build_dataset(dl_cfg, split="val")
        summary, split_rows = evaluate_split(split_name, dataset, architecture, args)
        summaries[split_name] = summary
        rows.extend(split_rows)
        print(json.dumps(summary, indent=2, sort_keys=True))

    result = {
        "checkpoint_dir": str(ckpt_dir),
        "mode": args.mode,
        "device": args.device,
        "max_samples": args.max_samples,
        "denoise_steps": args.denoise_steps,
        "seed": args.seed,
        "splits": summaries,
    }

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "outputs" / ckpt_dir.name
    write_outputs(out_dir, result, rows, write_csv=not args.no_csv)


if __name__ == "__main__":
    main()
