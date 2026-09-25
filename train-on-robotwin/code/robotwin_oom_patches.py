"""Local memory patches for RoboTwin OpenWAM experiments.

This file intentionally lives under train-on-robotwin so the upstream OpenWAM
package stays untouched.
"""

from __future__ import annotations

import logging
import os

import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _cfg_bool(cfg, key: str, default: bool) -> bool:
    value = OmegaConf.select(cfg, key, default=default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def install_robotwin_memory_patches(cfg) -> None:
    """Install RoboTwin-only training patches controlled by cfg.training."""

    _patch_adamw_foreach(cfg)
    _patch_zero_weight_video_loss(cfg)


def _patch_adamw_foreach(cfg) -> None:
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    if getattr(OpenWAMTrainer, "_robotwin_adamw_patch_installed", False):
        return

    def _maybe_freeze_video_backbone(self) -> None:
        if not _cfg_bool(self.cfg, "training.freeze_video_backbone", False):
            return
        arch = self.architecture
        video_backbone = getattr(arch, "video_backbone", None)
        if video_backbone is None or getattr(arch, "_robotwin_video_backbone_frozen", False):
            return
        before = sum(p.numel() for p in video_backbone.parameters() if p.requires_grad)
        video_backbone.requires_grad_(False)
        for module in video_backbone.modules():
            module.training = False
        arch._robotwin_video_backbone_frozen = True
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info("RoboTwin memory patch: froze video_backbone trainable params=%s", before)
            print(f"[robotwin_memory_patch] freeze_video_backbone=True frozen_params={before}", flush=True)

    def build_optimizer(self) -> torch.optim.Optimizer:
        t = self.cfg.training
        _maybe_freeze_video_backbone(self)
        params = build_trainable_parameters(
            self,
            action_lr=float(t.action_lr) if getattr(t, "action_lr", None) else None,
            video_lr=float(t.video_lr) if getattr(t, "video_lr", None) else None,
        )
        betas = tuple(getattr(t, "adam_betas", [0.9, 0.95]))
        foreach = _cfg_bool(self.cfg, "training.adamw_foreach", False)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info("RoboTwin memory patch: torch.optim.AdamW(foreach=%s)", foreach)
            print(f"[robotwin_memory_patch] AdamW foreach={foreach}", flush=True)
        return torch.optim.AdamW(
            params,
            lr=float(t.learning_rate),
            weight_decay=float(t.weight_decay),
            betas=betas,
            foreach=foreach,
        )

    OpenWAMTrainer.build_optimizer = build_optimizer
    OpenWAMTrainer._robotwin_adamw_patch_installed = True


def _patch_zero_weight_video_loss(cfg) -> None:
    if not _cfg_bool(cfg, "training.skip_zero_weight_video_loss", True):
        return

    from openwam.model.architectures.base import BaseWAMArchitecture
    from openwam.train.openwam_trainer import OpenWAMTrainer

    if not getattr(BaseWAMArchitecture, "_robotwin_video_loss_patch_installed", False):
        original_compute_video_loss = BaseWAMArchitecture._compute_video_loss

        def _compute_video_loss(self, noise_pred, target, timestep_ids, inputs, device):
            if getattr(self, "_robotwin_skip_video_loss_once", False):
                return torch.zeros((), dtype=torch.float32, device=device)
            return original_compute_video_loss(self, noise_pred, target, timestep_ids, inputs, device)

        BaseWAMArchitecture._compute_video_loss = _compute_video_loss
        BaseWAMArchitecture._robotwin_video_loss_patch_installed = True

    if getattr(OpenWAMTrainer, "_robotwin_compute_loss_patch_installed", False):
        return

    original_compute_loss = OpenWAMTrainer.compute_loss

    def compute_loss(self, batch) -> dict:
        skip_video = self.lambda_video == 0 and _cfg_bool(self.cfg, "training.skip_zero_weight_video_loss", True)
        modules = [self.architecture]
        if self.accelerator is not None:
            try:
                unwrapped = self.accelerator.unwrap_model(self.architecture)
            except Exception:
                unwrapped = None
            if unwrapped is not None and unwrapped is not self.architecture:
                modules.append(unwrapped)

        previous = [(module, getattr(module, "_robotwin_skip_video_loss_once", False)) for module in modules]
        try:
            for module in modules:
                setattr(module, "_robotwin_skip_video_loss_once", skip_video)
            return original_compute_loss(self, batch)
        finally:
            for module, value in previous:
                setattr(module, "_robotwin_skip_video_loss_once", value)

    OpenWAMTrainer.compute_loss = compute_loss
    OpenWAMTrainer._robotwin_compute_loss_patch_installed = True
