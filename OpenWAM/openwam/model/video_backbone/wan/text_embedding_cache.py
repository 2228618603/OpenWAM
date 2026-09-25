"""Prompt-keyed Wan T5 embedding cache helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch


CACHE_VERSION = 1
DEFAULT_HASH_IDENTITY = {
    "cache_version": CACHE_VERSION,
    "tokenizer": "google/umt5-xxl",
    "text_encoder": "models_t5_umt5-xxl-enc-bf16",
    "max_length": 512,
    "padding": "max_length",
    "truncation": True,
    "add_special_tokens": True,
    "clean": "whitespace",
    "dtype": "torch.bfloat16",
}


def normalize_hash_identity(identity: dict[str, Any] | None = None) -> dict[str, Any]:
    result = dict(DEFAULT_HASH_IDENTITY)
    if identity:
        result.update(identity)
    return result


def prompt_cache_key(prompt: str, identity: dict[str, Any] | None = None) -> str:
    payload = {
        "prompt": prompt,
        "identity": normalize_hash_identity(identity),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(cache_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"text embedding cache manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if int(manifest.get("cache_version", -1)) != CACHE_VERSION:
        raise ValueError(
            f"Unsupported text embedding cache version {manifest.get('cache_version')!r}; "
            f"expected {CACHE_VERSION}."
        )
    return manifest


def _read_record(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class TextEmbeddingCache:
    """Read prompt-keyed cached ``context`` / ``seq_lens`` tensors."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.manifest = load_manifest(self.cache_dir)
        self.hash_identity = normalize_hash_identity(self.manifest.get("hash_identity"))
        if not self.embeddings_dir.is_dir():
            raise FileNotFoundError(f"text embedding cache embeddings dir not found: {self.embeddings_dir}")

    def key_for_prompt(self, prompt: str) -> str:
        return prompt_cache_key(prompt, self.hash_identity)

    def path_for_prompt(self, prompt: str) -> Path:
        return self.embeddings_dir / f"{self.key_for_prompt(prompt)}.pt"

    def load_one(self, prompt: str) -> tuple[torch.Tensor, int]:
        path = self.path_for_prompt(prompt)
        if not path.is_file():
            raise FileNotFoundError(f"Cached text embedding miss for prompt key {path.stem}: {path}")
        record = _read_record(path)
        cached_prompt = record.get("prompt")
        if cached_prompt != prompt:
            raise ValueError(f"Prompt hash collision or stale cache at {path}: prompt text does not match.")
        context = record.get("context")
        if not torch.is_tensor(context):
            raise TypeError(f"{path} missing tensor field 'context'.")
        if context.ndim == 3 and context.shape[0] == 1:
            context = context[0]
        if context.ndim != 2:
            raise ValueError(f"{path} context must have shape [S, D] or [1, S, D], got {tuple(context.shape)}.")
        seq_len = record.get("seq_len", record.get("seq_lens"))
        if torch.is_tensor(seq_len):
            seq_len = int(seq_len.reshape(-1)[0].item())
        if seq_len is None:
            raise KeyError(f"{path} missing 'seq_len' or 'seq_lens'.")
        return context.contiguous(), int(seq_len)

    def load_batch(
        self,
        prompts: Iterable[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contexts = []
        seq_lens = []
        max_seq = 0
        for prompt in prompts:
            context, seq_len = self.load_one(str(prompt))
            contexts.append(context)
            seq_lens.append(seq_len)
            max_seq = max(max_seq, int(context.shape[0]))
        if not contexts:
            raise ValueError("Cannot load cached text embeddings for an empty prompt batch.")
        padded = []
        dim = int(contexts[0].shape[-1])
        for context in contexts:
            if int(context.shape[-1]) != dim:
                raise ValueError(f"Cached text context dim mismatch: expected {dim}, got {context.shape[-1]}.")
            if int(context.shape[0]) < max_seq:
                pad = torch.zeros((max_seq - context.shape[0], dim), dtype=context.dtype)
                context = torch.cat([context, pad], dim=0)
            padded.append(context)
        batch_context = torch.stack(padded, dim=0).to(device=device, dtype=dtype, non_blocking=True)
        batch_seq_lens = torch.tensor(seq_lens, dtype=torch.long, device=device)
        return batch_context, batch_seq_lens
