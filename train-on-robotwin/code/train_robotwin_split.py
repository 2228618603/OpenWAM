#!/usr/bin/env python3
"""Hydra training entry for the RoboTwin WM/action split experiment."""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import shutil
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENWAM_PROJECT_ROOT = REPO_ROOT / "OpenWAM"
LOCAL_CODE_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(OPENWAM_PROJECT_ROOT))
sys.path.insert(0, str(LOCAL_CODE_ROOT))

faulthandler.enable(file=sys.stderr, all_threads=True)


def _force_flush_excepthook(exc_type, exc_value, exc_tb):
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
    sys.stderr.write(f"\n===== UNHANDLED EXCEPTION ON RANK {rank} =====\n")
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()


sys.excepthook = _force_flush_excepthook


def _format_exception_chain(exc: BaseException) -> str:
    lines = []
    seen = set()
    current = exc
    label = "exception"
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        lines.append(f"{label}: {type(current).__module__}.{type(current).__name__}: {current!r}")
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if cause is not None:
            current = cause
            label = "cause"
        elif context is not None:
            current = context
            label = "context"
        else:
            current = None
    return "\n".join(lines)


def _write_rank_error_file(exc: BaseException) -> None:
    error_dir = os.environ.get("ROBOTWIN_SPLIT_ERROR_DIR")
    if not error_dir:
        return
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    Path(error_dir).mkdir(parents=True, exist_ok=True)
    path = Path(error_dir) / f"rank{rank}_error.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"rank={rank}\n")
        f.write(f"pid={os.getpid()}\n")
        f.write(_format_exception_chain(exc))
        f.write("\n\n--- traceback ---\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def fileno(self):
        return self._streams[-1].fileno()


def _install_rank_file_logging() -> None:
    log_dir = os.environ.get("ROBOTWIN_SPLIT_RANK_LOG_DIR")
    if not log_dir:
        return
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(log_dir) / f"rank{rank}.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)


def _install_hang_trace_dump() -> None:
    timeout = os.environ.get("ROBOTWIN_SPLIT_TRACE_TIMEOUT")
    if not timeout:
        return
    faulthandler.dump_traceback_later(int(timeout), repeat=True, file=sys.stderr)


class _StartupNoiseFilter(logging.Filter):
    _HIDDEN_PREFIXES = ("gcc -pthread ", "NCCL version ")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (record.name == "root" and message.startswith(self._HIDDEN_PREFIXES))


def _install_startup_noise_filter() -> None:
    noise_filter = _StartupNoiseFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(noise_filter)


def _copytree_data_only(src: str, dst: str) -> str:
    """Copy directory contents without preserving metadata.

    Some mounted OSS paths allow writing file contents but reject chmod/copystat
    during shutil.copytree. OpenWAM only needs deploy asset bytes here.
    """
    src_path = Path(src)
    dst_path = Path(dst)
    for item in src_path.rglob("*"):
        rel = item.relative_to(src_path)
        target = dst_path / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
    return str(dst_path)


def _install_copytree_metadata_fallback() -> None:
    original_copytree = shutil.copytree

    def copytree_with_fallback(src, dst, *args, **kwargs):
        try:
            return original_copytree(src, dst, *args, **kwargs)
        except (PermissionError, shutil.Error) as exc:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
            if rank == 0:
                print(
                    "[robotwin_split] shutil.copytree failed while preserving metadata; "
                    f"retrying data-only copy: {src} -> {dst} ({exc})"
                )
            return _copytree_data_only(src, dst)

    shutil.copytree = copytree_with_fallback


def _install_wandb_action_loss_alias() -> None:
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb, "_robotwin_action_loss_alias_installed", False):
        return

    original_init = wandb.init

    def init_with_action_loss_alias(*args, **kwargs):
        run = original_init(*args, **kwargs)
        original_log = run.log

        def log_with_action_loss_alias(data=None, *log_args, **log_kwargs):
            if isinstance(data, dict) and "train/loss_action" in data and "train/action_loss" not in data:
                data = dict(data)
                data["train/action_loss"] = data["train/loss_action"]
            return original_log(data, *log_args, **log_kwargs)

        run.log = log_with_action_loss_alias
        return run

    wandb.init = init_with_action_loss_alias
    wandb._robotwin_action_loss_alias_installed = True


def _build_accelerator(cfg: DictConfig):
    import accelerate
    from accelerate.utils import InitProcessGroupKwargs

    t = cfg.training
    grad_accum = int(t.gradient_accumulation_steps)
    max_grad_norm = getattr(t, "max_grad_norm", None)
    mixed_precision = str(t.mixed_precision)
    timeout_sec = int(os.environ.get("OPENWAM_DISTRIBUTED_TIMEOUT_SEC", "600"))

    plugin = accelerate.DeepSpeedPlugin(
        zero_stage=int(t.zero_stage),
        gradient_accumulation_steps=grad_accum,
        gradient_clipping=float(max_grad_norm) if max_grad_norm else 0.0,
        offload_optimizer_device=str(t.offload_optimizer_device),
    )
    return accelerate.Accelerator(
        gradient_accumulation_steps=grad_accum,
        deepspeed_plugin=plugin,
        mixed_precision=mixed_precision,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_sec))],
    )


def _inject_project_seed(cfg: DictConfig) -> None:
    project_seed = OmegaConf.select(cfg, "project.seed", default=None)
    if project_seed is not None and cfg.get("dataloader", None) is not None:
        OmegaConf.update(cfg.dataloader, "seed", int(project_seed), force_add=True)


def _train_openwam(cfg: DictConfig) -> None:
    from openwam.dataloader.registry import build_dataset
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.seeding import seed_everything

    _inject_project_seed(cfg)

    project_seed = OmegaConf.select(cfg, "project.seed", default=None)
    if project_seed is not None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        seed_everything(int(project_seed), rank=rank)

    accelerator = _build_accelerator(cfg)
    dataset = build_dataset(cfg.dataloader, split="train")
    trainer = OpenWAMTrainer(cfg, accelerator=accelerator, dataset=dataset)
    trainer.train()


@hydra.main(version_base=None, config_path=str(OPENWAM_PROJECT_ROOT / "configs"), config_name="train")
def main(cfg: DictConfig) -> None:
    _install_rank_file_logging()
    _install_hang_trace_dump()
    _install_startup_noise_filter()
    _install_copytree_metadata_fallback()
    _install_wandb_action_loss_alias()
    from robotwin_split_runtime import apply_train40_45eps_config

    apply_train40_45eps_config(cfg)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("=" * 60)
        print("OpenWAM RoboTwin Split Training")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 60)
    else:
        import builtins

        builtins.print = lambda *a, **kw: None

    try:
        _train_openwam(cfg)
    except BaseException as exc:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
        sys.stderr.write(f"\n===== RANK {rank} EXCEPTION (pre-destroy) =====\n")
        sys.stderr.write(_format_exception_chain(exc) + "\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        _write_rank_error_file(exc)
        raise
    finally:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        from torch.distributed.elastic.multiprocessing.errors import record
    except Exception:
        record = lambda fn: fn

    @record
    def _main_with_elastic_record():
        main()

    _main_with_elastic_record()
