#!/usr/bin/env python3
"""Diagnose RoboTwin batch reads for the local OpenWAM split wrapper."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf


OPENWAM_PROJECT_ROOT = Path("/home/chw/code/packages/OpenWAM/OpenWAM")
LOCAL_CODE_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(OPENWAM_PROJECT_ROOT))
sys.path.insert(0, str(LOCAL_CODE_ROOT))


def _describe_index(dataset, idx: int) -> str:
    ds_idx = None
    local_idx = idx
    sub = dataset
    cumulative = getattr(dataset, "_cumulative_lengths", None)
    if cumulative is not None:
        lo, hi = 0, len(cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < cumulative[mid]:
                hi = mid
            else:
                lo = mid + 1
        ds_idx = lo
        local_idx = idx if ds_idx == 0 else idx - cumulative[ds_idx - 1]
        sub = dataset._sub_datasets[ds_idx]

    ep_idx, start = sub._window_index[local_idx]
    path = sub._episode_files[ep_idx]
    task = getattr(sub, "task_name", "?")
    return (
        f"global_idx={idx} ds={ds_idx} local_idx={local_idx} "
        f"task={task} episode={Path(path).name} start={start} path={path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/mnt/data/chw/robotwin-dataset/dataset")
    parser.add_argument("--embodiment", default="aloha-agilex")
    parser.add_argument("--variant", default="clean_50")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--start-base-batch", type=int, default=0)
    args = parser.parse_args()

    from openwam.dataloader.registry import build_dataset
    from openwam.train.utils.seeding import dataloader_worker_init_fn, make_dataloader_generator
    from robotwin_split_runtime import apply_train40_45eps_config

    cfg = OmegaConf.create(
        {
            "dataloader": {
                "type": "robotwin",
                "dataset_dir": args.dataset_dir,
                "embodiment": args.embodiment,
                "variant": args.variant,
            },
            "project": {"seed": args.seed},
        }
    )
    apply_train40_45eps_config(cfg)
    OmegaConf.update(cfg.dataloader, "seed", args.seed, force_add=True)

    dataset = build_dataset(cfg.dataloader, split="train")
    generator = make_dataloader_generator(args.seed, rank=0)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=list,
        pin_memory=False,
        generator=generator,
        worker_init_fn=dataloader_worker_init_fn if args.num_workers else None,
    )

    sampler_iter = iter(loader.batch_sampler)
    max_base_batch = args.world_size * args.steps + args.rank
    for base_batch_idx, indices in enumerate(sampler_iter):
        if base_batch_idx < args.start_base_batch:
            continue
        if base_batch_idx > max_base_batch:
            break
        if base_batch_idx % args.world_size != args.rank:
            continue

        train_step = base_batch_idx // args.world_size
        indices = list(indices)
        print(f"\n[rank {args.rank}] train_step={train_step} base_batch={base_batch_idx} indices={indices}", flush=True)
        for idx in indices:
            print("  will_read " + _describe_index(dataset, int(idx)), flush=True)

        t0 = time.monotonic()
        batch = [dataset[int(idx)] for idx in indices]
        dt = time.monotonic() - t0
        print(f"[rank {args.rank}] read_ok train_step={train_step} samples={len(batch)} seconds={dt:.3f}", flush=True)


if __name__ == "__main__":
    main()
