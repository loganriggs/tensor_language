"""Source-closed, local-only bilin18 model boundary for suffix experiments.

This module intentionally does not import the historical ship runner: that runner
loads unrelated row roles and rebuilds fitted objects at import/run time.  The
facade pins the checkpoint and exposes one explicit early-MLP dispatcher.  Loading
the canonical frozen ship and sealing dispatcher aliases belong to the observed
adapter layered above this file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import jacclust.tt_model as TT


MODEL_REPO = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
MODEL_REVISION = "ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240"
DEFAULT_SNAPSHOT = Path(
    "/workspace/.hf_home/hub/"
    "models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/"
    f"snapshots/{MODEL_REVISION}"
)
CONFIG_SHA256 = "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c"
WEIGHTS_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
WEIGHTS_BYTES = 2_067_738_635
TOKENIZER_VOCAB = 50_257
LOGIT_VOCAB = 50_304
EARLY_SITES = (0, 1, 2)

EXPECTED_CONFIG: Mapping[str, Any] = {
    "vocab_size": LOGIT_VOCAB,
    "n_layer": 18,
    "n_head": 9,
    "n_embd": 1152,
    "squared_mlp": False,
    "bilinear": True,
    "expansion_factor": 4,
    "gated": False,
    "squared_attn": True,
    "bilinear_attn": True,
    "step": 9726,
}

EarlyDispatcher = Callable[[int, torch.nn.Module, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CheckpointReceipt:
    revision: str
    snapshot: str
    config_sha256: str
    weights_sha256: str
    weights_bytes: int
    tokenizer_vocab: int
    logit_vocab: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact deployed topology without silently dropping fields."""

    observed = dict(config)
    if observed != dict(EXPECTED_CONFIG):
        changed = {
            key: (EXPECTED_CONFIG.get(key), observed.get(key))
            for key in sorted(set(EXPECTED_CONFIG) | set(observed))
            if EXPECTED_CONFIG.get(key) != observed.get(key)
        }
        raise RuntimeError(f"bilin18 config differs from the pinned contract: {changed}")
    return observed


def validate_snapshot(
    snapshot: Path | str = DEFAULT_SNAPSHOT, *, verify_weights_sha256: bool = True,
) -> CheckpointReceipt:
    """Validate the pinned local snapshot; this function never performs network I/O."""

    root = Path(snapshot)
    if root.name != MODEL_REVISION:
        raise RuntimeError("bilin18 snapshot revision is not pinned")
    config_path = root / "config.json"
    weights_path = root / "pytorch_model.bin"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError("pinned bilin18 snapshot is incomplete")
    config_sha256 = _sha256_file(config_path)
    if config_sha256 != CONFIG_SHA256:
        raise RuntimeError("bilin18 config bytes changed")
    with config_path.open("r", encoding="utf-8") as source:
        validate_config(json.load(source))
    weights_bytes = weights_path.stat().st_size
    if weights_bytes != WEIGHTS_BYTES:
        raise RuntimeError("bilin18 weight file size changed")
    weights_sha256 = (
        _sha256_file(weights_path) if verify_weights_sha256 else WEIGHTS_SHA256
    )
    if weights_sha256 != WEIGHTS_SHA256:
        raise RuntimeError("bilin18 weight bytes changed")
    return CheckpointReceipt(
        revision=MODEL_REVISION,
        snapshot=str(root.resolve()),
        config_sha256=config_sha256,
        weights_sha256=weights_sha256,
        weights_bytes=weights_bytes,
        tokenizer_vocab=TOKENIZER_VOCAB,
        logit_vocab=LOGIT_VOCAB,
    )


def validate_production_model(model: TT.GPT) -> None:
    if type(model) is not TT.GPT:
        raise RuntimeError("observed model is not the exact TT.GPT implementation")
    config = {
        key: getattr(model.config, key)
        for key in EXPECTED_CONFIG
        if key != "step"
    }
    expected = {key: value for key, value in EXPECTED_CONFIG.items() if key != "step"}
    if config != expected:
        raise RuntimeError("loaded bilin18 topology differs from the pinned config")
    if len(model.transformer.h) != 18 or tuple(model.lm_head.weight.shape) != (
        LOGIT_VOCAB, 1152,
    ):
        raise RuntimeError("loaded bilin18 component dimensions changed")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("observed bilin18 must be frozen and in eval mode")


def load_bilin18(
    *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32,
    snapshot: Path | str = DEFAULT_SNAPSHOT, verify_weights_sha256: bool = True,
) -> tuple[TT.GPT, CheckpointReceipt]:
    """Load the exact checkpoint locally, with no Hub client or fallback path."""

    receipt = validate_snapshot(snapshot, verify_weights_sha256=verify_weights_sha256)
    root = Path(snapshot)
    with (root / "config.json").open("r", encoding="utf-8") as source:
        config = validate_config(json.load(source))
    constructor = dict(config)
    constructor.pop("step")
    model = TT.GPT(TT.GPTConfig(**constructor)).to(device=device, dtype=dtype)
    state = torch.load(
        root / "pytorch_model.bin", map_location=device, weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    validate_production_model(model)
    return model, receipt


def validate_tokens(tokens: torch.Tensor, *, production_shape: bool = True) -> None:
    expected = (4, 256)
    if not torch.is_tensor(tokens) or tokens.ndim != 2 or (
        production_shape and tuple(tokens.shape) != expected
    ):
        raise RuntimeError(f"tokens must have shape {expected}")
    if tokens.dtype != torch.long or tokens.numel() == 0:
        raise RuntimeError("tokens must be a nonempty torch.long tensor")
    low, high = int(tokens.min()), int(tokens.max())
    if low < 0 or high >= TOKENIZER_VOCAB:
        raise RuntimeError("token ID is outside the 50,257-entry tokenizer support")


def forward_with_early_dispatch(
    model: TT.GPT,
    tokens: torch.Tensor,
    dispatcher: EarlyDispatcher,
    *,
    require_production: bool = True,
) -> torch.Tensor:
    """Run one exact model forward, delegating MLP0/1/2 to ``dispatcher``.

    The dispatcher receives the literal site, block object, and live normalized
    MLP input.  It must return the deployed write at that site.  This facade owns
    the single softcap and never slices padded output columns.
    """

    if require_production:
        validate_production_model(model)
    validate_tokens(tokens, production_shape=require_production)
    if not callable(dispatcher):
        raise TypeError("early dispatcher must be callable")

    x = model.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, v1 = block.attn(F.rms_norm(x, (x.size(-1),)), v1)
        x = x + attention_write
        z = F.rms_norm(x, (x.size(-1),))
        write = dispatcher(site, block, z) if site in EARLY_SITES else block.mlp(z)
        if not torch.is_tensor(write) or write.shape != z.shape or write.dtype != z.dtype or (
            write.device != z.device
        ) or not bool(torch.isfinite(write.detach()).all()):
            raise RuntimeError(f"MLP{site} dispatcher write is malformed")
        x = x + write

    logits = model.lm_head(F.rms_norm(x, (x.size(-1),)))
    logits = (30.0 * torch.tanh(logits / 30.0)).float()
    expected_vocab = LOGIT_VOCAB if require_production else model.config.vocab_size
    if tuple(logits.shape) != (*tokens.shape, expected_vocab):
        raise RuntimeError("bilin18 logits have the wrong unsliced output shape")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("bilin18 logits are nonfinite")
    return logits
