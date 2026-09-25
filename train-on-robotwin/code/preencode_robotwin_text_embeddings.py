#!/usr/bin/env python3
"""Precompute Wan T5 text embeddings for RoboTwin training prompts."""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


OPENWAM_PROJECT_ROOT = Path("/home/chw/code/packages/OpenWAM/OpenWAM")
LOCAL_CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = Path(
    "/mnt/data/chw/model/wm-function/cache/robotwin_text_embeddings/"
    "wan22_ti2v_5b_umt5_xxl_bf16_clean50"
)

sys.path.insert(0, str(OPENWAM_PROJECT_ROOT))
sys.path.insert(0, str(LOCAL_CODE_ROOT))


def _instruction_candidates(instructions: dict, ep_file: str, split: str, task_name: str) -> list[str]:
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference

    ep_basename = os.path.basename(ep_file)
    ep_num = ep_basename.replace("episode", "").replace(".hdf5", "")
    instr_key = f"episode{ep_num}.json"

    base_prompts: list[str] = []
    instr = instructions.get(instr_key)
    if isinstance(instr, dict):
        pool = instr.get("seen") or instr.get("unseen") or []
        if pool:
            base_prompts.extend(str(p) for p in pool)
        elif "instruction" in instr:
            base_prompts.append(str(instr["instruction"]))
    elif isinstance(instr, str):
        base_prompts.append(instr)
    elif isinstance(instr, list) and instr:
        base_prompts.extend(str(p) for p in instr if p is not None)

    if not base_prompts:
        base_prompts.append(f"The bimanual robot is performing a {task_name} task.")

    # Val uses the first candidate at runtime, but encoding all candidates is
    # harmless and prevents misses if the same cache is reused for train.
    return [format_prompt_for_inference(p) for p in base_prompts]


def _iter_robotwin_prompts(dataset) -> Iterable[str]:
    sub_datasets = getattr(dataset, "_sub_datasets", None)
    if sub_datasets is None:
        sub_datasets = [dataset]
    for ds in sub_datasets:
        window_index = getattr(ds, "_window_index", None)
        if window_index is None:
            for i in range(len(ds)):
                yield ds[i]["prompt"]
            continue
        ep_indices = sorted({int(ep_idx) for ep_idx, _start in window_index})
        instructions = getattr(ds, "_instructions", {})
        episode_files = getattr(ds, "_episode_files", [])
        split = str(getattr(ds, "split", "train"))
        task_name = str(getattr(ds, "task_name", "manipulation"))
        for ep_idx in ep_indices:
            for prompt in _instruction_candidates(instructions, episode_files[ep_idx], split, task_name):
                yield prompt


def _unique_in_order(values: Iterable[str]) -> list[str]:
    unique = OrderedDict()
    for value in values:
        unique[str(value)] = None
    return list(unique.keys())


def _model_config_label(model_config) -> str:
    if getattr(model_config, "path", None) is not None:
        return str(model_config.path)
    if getattr(model_config, "origin_file_pattern", None) is not None:
        return str(model_config.origin_file_pattern)
    return str(model_config)


def _load_text_encoder_and_tokenizer(cfg: DictConfig, *, device: str):
    from openwam.model.video_backbone.wan.loader import load_wan_components
    from openwam.model.video_backbone.wan.pipeline_builder import _filter_text_encoder_configs, discover_model_files

    model_dir = str(OmegaConf.select(cfg, "model.video_backbone.model_path"))
    model_configs, tokenizer_config = discover_model_files(model_dir)
    text_configs = _filter_text_encoder_configs(model_configs)
    text_config_ids = {_model_config_label(c) for c in text_configs}
    text_configs = [c for c in model_configs if _model_config_label(c) not in text_config_ids]
    if not text_configs:
        raise RuntimeError(f"No Wan T5 text encoder model file discovered under {model_dir}")
    holder = load_wan_components(text_configs, tokenizer_config, device=device, torch_dtype=torch.bfloat16)
    if holder.text_encoder is None or holder.tokenizer is None:
        raise RuntimeError("Failed to load Wan text_encoder/tokenizer for preencoding.")
    holder.text_encoder.eval()
    return holder.text_encoder, holder.tokenizer, model_dir


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


@hydra.main(version_base=None, config_path=str(OPENWAM_PROJECT_ROOT / "configs"), config_name="train")
def main(cfg: DictConfig) -> None:
    from openwam.dataloader.registry import build_dataset
    from openwam.model.video_backbone.wan import encode as wan_encode
    from openwam.model.video_backbone.wan.text_embedding_cache import (
        CACHE_VERSION,
        DEFAULT_HASH_IDENTITY,
        TextEmbeddingCache,
        prompt_cache_key,
    )
    from robotwin_split_runtime import apply_train40_45eps_config

    apply_train40_45eps_config(cfg)

    cache_dir = Path(
        os.environ.get(
            "ROBOTWIN_T5_CACHE_DIR",
            str(OmegaConf.select(cfg, "training.text_embedding_cache_dir", default=None) or DEFAULT_CACHE_DIR),
        )
    )
    batch_size = int(os.environ.get("ROBOTWIN_T5_PREENCODE_BATCH_SIZE", "8"))
    max_prompts_raw = os.environ.get("ROBOTWIN_T5_PREENCODE_MAX_PROMPTS")
    max_prompts = int(max_prompts_raw) if max_prompts_raw else None
    shard_count = int(os.environ.get("ROBOTWIN_T5_PREENCODE_SHARD_COUNT", "1"))
    shard_index = int(os.environ.get("ROBOTWIN_T5_PREENCODE_SHARD_INDEX", "0"))
    manifest_only = os.environ.get("ROBOTWIN_T5_PREENCODE_MANIFEST_ONLY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    device = os.environ.get("ROBOTWIN_T5_PREENCODE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if shard_count < 1:
        raise ValueError(f"ROBOTWIN_T5_PREENCODE_SHARD_COUNT must be >= 1, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"ROBOTWIN_T5_PREENCODE_SHARD_INDEX must be in [0, {shard_count}), got {shard_index}"
        )

    dataset = build_dataset(cfg.dataloader, split="train")
    all_prompts = _unique_in_order(_iter_robotwin_prompts(dataset))
    total_samples = len(dataset)
    if max_prompts is not None:
        all_prompts = all_prompts[:max_prompts]
    prompts = all_prompts[shard_index::shard_count]

    hash_identity = dict(DEFAULT_HASH_IDENTITY)
    hash_identity["model_path"] = str(OmegaConf.select(cfg, "model.video_backbone.model_path"))
    hash_identity["variant"] = str(OmegaConf.select(cfg, "dataloader.variant", default=""))
    hash_identity["embodiment"] = str(OmegaConf.select(cfg, "dataloader.embodiment", default=""))
    hash_identity["storage"] = "cropped_to_seq_len"

    manifest = {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "dataset_type": str(OmegaConf.select(cfg, "dataloader.type", default="robotwin")),
        "dataset_dir": str(OmegaConf.select(cfg, "dataloader.dataset_dir", default="")),
        "split": "train",
        "total_samples": total_samples,
        "total_unique_prompts": len(all_prompts),
        "unique_prompts": len(prompts),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "hash_identity": hash_identity,
    }

    embeddings_dir = cache_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    if manifest_only:
        manifest["unique_prompts"] = len(all_prompts)
        manifest["shard_count"] = 1
        manifest["shard_index"] = 0
        manifest["manifest_only"] = True
        _write_json(cache_dir / "manifest.json", manifest)
        print(f"[preencode] manifest refreshed: dir={cache_dir} total_unique_prompts={len(all_prompts)}")
        return

    if shard_count == 1:
        _write_json(cache_dir / "manifest.json", manifest)

    if not prompts:
        print(f"[preencode] cache ready but no prompts were selected: dir={cache_dir}")
        return

    text_encoder, tokenizer, model_dir = _load_text_encoder_and_tokenizer(cfg, device=device)
    manifest["model_path"] = model_dir

    index_records = []
    encoded = 0
    skipped = 0
    for start in tqdm(range(0, len(prompts), batch_size), desc="Encoding prompts", unit="batch"):
        batch_prompts = prompts[start : start + batch_size]
        batch_keys = [prompt_cache_key(prompt, hash_identity) for prompt in batch_prompts]
        missing_pairs = [
            (prompt, key)
            for prompt, key in zip(batch_prompts, batch_keys)
            if not (embeddings_dir / f"{key}.pt").is_file()
        ]
        skipped += len(batch_prompts) - len(missing_pairs)
        if missing_pairs:
            with torch.no_grad():
                context, seq_lens = wan_encode.encode_text(
                    [prompt for prompt, _key in missing_pairs],
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    device=torch.device(device),
                )
            context = context.detach().cpu().to(torch.bfloat16)
            seq_lens = seq_lens.detach().cpu().long()
            for i, (prompt, key) in enumerate(missing_pairs):
                seq_len = int(seq_lens[i].item())
                record = {
                    "context": context[i, :seq_len].clone().contiguous(),
                    "seq_len": seq_len,
                    "prompt": prompt,
                    "meta": {
                        "cache_version": CACHE_VERSION,
                        "hash_identity": hash_identity,
                        "model_path": model_dir,
                        "storage": "cropped_to_seq_len",
                    },
                }
                torch.save(record, embeddings_dir / f"{key}.pt")
                encoded += 1
        for prompt, key in zip(batch_prompts, batch_keys):
            index_records.append({"key": key, "path": f"embeddings/{key}.pt", "prompt": prompt})

    index_name = "index.jsonl" if shard_count == 1 else f"index.shard{shard_index:03d}-of-{shard_count:03d}.jsonl"
    with (cache_dir / index_name).open("w", encoding="utf-8") as f:
        for record in index_records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest["encoded_files"] = len(index_records)
    manifest["newly_encoded"] = encoded
    manifest["existing_files_reused"] = skipped
    if shard_count == 1:
        _write_json(cache_dir / "manifest.json", manifest)
    else:
        print(
            f"[preencode] shard ready: dir={cache_dir} shard={shard_index}/{shard_count} "
            f"unique_prompts={len(prompts)} newly_encoded={encoded} existing_files_reused={skipped}"
        )
        return

    cache = TextEmbeddingCache(cache_dir)
    verify_count = min(int(os.environ.get("ROBOTWIN_T5_VERIFY_PROMPTS", "10")), len(prompts))
    verify_prompts = prompts[:verify_count]
    sample_context, sample_seq_lens = cache.load_batch(
        prompts[: min(10, len(prompts))],
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    if verify_count:
        verify_context, verify_seq_lens = cache.load_batch(
            verify_prompts,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            online_context, online_seq_lens = wan_encode.encode_text(
                verify_prompts,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                device=torch.device(device),
            )
        online_context = online_context.detach().cpu().to(torch.bfloat16)
        online_seq_lens = online_seq_lens.detach().cpu().long()
        online_context = online_context[:, : verify_context.shape[1]]
        diff = (online_context.float() - verify_context.float()).abs()
        print(
            "[preencode] verify: "
            f"context_shape={tuple(verify_context.shape)} "
            f"seq_lens_match={bool(torch.equal(online_seq_lens, verify_seq_lens.cpu()))} "
            f"max_abs_diff={float(diff.max().item()) if diff.numel() else 0.0:.6g} "
            f"mean_abs_diff={float(diff.mean().item()) if diff.numel() else 0.0:.6g}"
        )
    print(
        f"[preencode] cache ready: dir={cache_dir} unique_prompts={len(prompts)} "
        f"context_shape={tuple(sample_context.shape)} seq_lens={sample_seq_lens.tolist()}"
    )


if __name__ == "__main__":
    main()
