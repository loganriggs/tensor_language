#!/usr/bin/env python3
"""Data-minimal, cache-only runtime for prospective bilin18 evaluation.

Importing this module reads exactly the pinned config and checkpoint blobs.  It does
not import experiment scripts, datasets, row caches, fitted artifacts, or results.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import jacclust.tt_model as TT

REPOSITORY = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
REVISION = "ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240"
CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
CONFIG_SHA256 = "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c"
DEV = "cuda"
D = 1152
T = 256


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8*1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def local_blob(filename, expected):
    path = Path(hf_hub_download(REPOSITORY, filename, revision=REVISION,
                                local_files_only=True))
    if sha(path) != expected:
        raise ValueError(f"pinned checkpoint blob mismatch: {filename}")
    return path


CONFIG_PATH = None
CHECKPOINT_PATH = None
m = None
H = None


def initialize():
    """Explicit, idempotent initialization after caller-owned resource preflight."""
    global CONFIG_PATH, CHECKPOINT_PATH, m, H
    if m is not None:
        return m
    CONFIG_PATH = local_blob("config.json", CONFIG_SHA256)
    CHECKPOINT_PATH = local_blob("pytorch_model.bin", CHECKPOINT_SHA256)
    config_dict = json.loads(CONFIG_PATH.read_text())
    config_dict.pop("step", None)
    model = TT.GPT(TT.GPTConfig(**config_dict)).to(
        device=DEV, dtype=torch.float32).eval()
    model.load_state_dict(torch.load(
        CHECKPOINT_PATH, map_location=DEV, weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = model.transformer.h
    if len(blocks) != 18 or model.config.n_embd != D or model.config.n_head != 9:
        raise ValueError("loaded checkpoint architecture differs from bilin18 contract")
    m, H = model, blocks
    return m
