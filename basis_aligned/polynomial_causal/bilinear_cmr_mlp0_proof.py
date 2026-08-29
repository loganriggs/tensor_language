#!/usr/bin/env python3
"""CPU proof receipt for causal-mechanism reduction on the actual MLP0 weights.

This is an algebra/implementation check, not a natural-text behavioral result.  It
uses synthetic full-support states because gauge invariance and folding are pointwise
identities.  The later scored experiment must use frozen natural-text roles.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bilinear_causal_mechanism_reduction as cmr
import bilin18_observed_model_facade as facade


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "bilinear_cmr_mlp0_proof_results.json"
SEED = 2026082907
ROWS = 256
TOPK = 512


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def publish_create_only(value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def relative_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (actual - expected).square().mean().sqrt()
        / expected.square().mean().sqrt().clamp_min(torch.finfo(expected.dtype).tiny)
    )


def jaccard(first: torch.Tensor, second: torch.Tensor) -> float:
    a, b = set(first.tolist()), set(second.tolist())
    return len(a & b) / len(a | b)


@torch.no_grad()
def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("bilinear CMR proof receipt is create-only and already exists")
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    model, checkpoint = facade.load_bilin18(device="cpu", dtype=torch.float32)
    mlp = model.transformer.h[0].mlp
    left = mlp.Left.weight.detach().clone()
    right = mlp.Right.weight.detach().clone()
    down = mlp.Down.weight.detach().clone()
    bias = mlp.Down_bias.detach().clone()
    del model

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    states = torch.randn(ROWS, left.shape[1], generator=generator)
    states = states / states.square().mean(dim=1, keepdim=True).sqrt()
    products = cmr.product_activations(states, left, right)
    write = products @ down.T + bias

    log_left = torch.empty(left.shape[0]).uniform_(-3.0, 3.0, generator=generator)
    log_right = torch.empty(right.shape[0]).uniform_(-3.0, 3.0, generator=generator)
    scale_left, scale_right = log_left.exp(), log_right.exp()
    left_g, right_g, down_g = cmr.apply_channel_gauge(
        left, right, down, scale_left, scale_right,
    )
    products_g = cmr.product_activations(states, left_g, right_g)
    write_g = products_g @ down_g.T + bias

    raw_variance = products.var(dim=0, unbiased=False)
    raw_variance_g = products_g.var(dim=0, unbiased=False)
    scores = cmr.cmr_logit_scores(products, down)
    scores_g = cmr.cmr_logit_scores(products_g, down_g)
    raw_top = torch.topk(raw_variance, TOPK).indices
    raw_top_g = torch.topk(raw_variance_g, TOPK).indices
    score_top = torch.topk(scores, TOPK).indices
    score_top_g = torch.topk(scores_g, TOPK).indices

    replaced = torch.topk(scores, 64, largest=False).indices.sort().values
    constants = products[:, replaced].mean(dim=0)
    constant_fold = cmr.compile_constant_replacement(down, bias, replaced, constants)
    intervened = products.clone()
    intervened[:, replaced] = constants
    constant_reference = intervened @ down.T + bias
    constant_compiled = (
        products[:, constant_fold.kept] @ constant_fold.down.T + constant_fold.bias
    )

    affine_replaced = replaced[:8]
    affine_keep = torch.tensor([
        index for index in range(products.shape[1])
        if index not in set(affine_replaced.tolist())
    ])
    affine_intercept = products[:, affine_replaced].mean(dim=0)
    affine_coefficients = 1e-3 * torch.randn(
        affine_replaced.numel(), affine_keep.numel(), generator=generator,
    )
    affine_fold = cmr.compile_affine_replacement(
        down, bias, affine_replaced, affine_intercept, affine_coefficients,
    )
    affine_intervened = products.clone()
    affine_intervened[:, affine_replaced] = (
        affine_intercept + products[:, affine_keep] @ affine_coefficients.T
    )
    affine_reference = affine_intervened @ down.T + bias
    affine_compiled = (
        products[:, affine_fold.kept] @ affine_fold.down.T + affine_fold.bias
    )

    active = score_top[:64]
    off_diagonal = float(cmr.off_diagonal_fraction(
        products[:, active], down[:, active],
    ))
    score_relative_error = float(
        (scores_g - scores).abs().max() / scores.abs().max().clamp_min(1e-30)
    )
    measurements = {
        "map_gauge_relative_rms": relative_rms(write_g, write),
        "raw_variance_top512_jaccard_after_exact_gauge": jaccard(raw_top, raw_top_g),
        "cmr_top512_jaccard_after_exact_gauge": jaccard(score_top, score_top_g),
        "cmr_score_max_relative_error_after_exact_gauge": score_relative_error,
        "constant_fold_relative_rms": relative_rms(constant_compiled, constant_reference),
        "affine_fold_relative_rms": relative_rms(affine_compiled, affine_reference),
        "top64_joint_off_diagonal_fraction_synthetic_states": off_diagonal,
    }
    predictions = {
        "A_exact_bilinear_map_survives_gauge": measurements[
            "map_gauge_relative_rms"
        ] <= 2e-6,
        "B_raw_variance_selection_is_gauge_fragile": measurements[
            "raw_variance_top512_jaccard_after_exact_gauge"
        ] < 0.5,
        "C_cmr_selection_is_gauge_invariant": measurements[
            "cmr_top512_jaccard_after_exact_gauge"
        ] == 1.0 and score_relative_error <= 5e-6,
        "D_constant_replacement_folds_exactly": measurements[
            "constant_fold_relative_rms"
        ] <= 2e-6,
        "E_affine_replacement_folds_exactly": measurements[
            "affine_fold_relative_rms"
        ] <= 2e-6,
    }
    result = {
        "status": "pass" if all(predictions.values()) else "proof_failure",
        "scope": (
            "CPU algebra/implementation proof on actual frozen MLP0 weights; "
            "synthetic full-support states; no natural-text behavioral claim"
        ),
        "source": {
            "method": "Causal Mechanism Reduction, arXiv:2602.24266v2",
            "checkpoint": asdict(checkpoint),
            "mlp0_shapes": {
                "left": list(left.shape),
                "right": list(right.shape),
                "down": list(down.shape),
                "bias": list(bias.shape),
            },
            "weight_hashes": {
                "left": tensor_sha256(left),
                "right": tensor_sha256(right),
                "down": tensor_sha256(down),
                "bias": tensor_sha256(bias),
            },
        },
        "protocol": {
            "seed": SEED,
            "synthetic_rows": ROWS,
            "ranking_k": TOPK,
            "gauge_log_scale_interval": [-3.0, 3.0],
            "constant_replaced_channels": 64,
            "affine_replaced_channels": 8,
        },
        "measurements": measurements,
        "predictions": predictions,
        "next_falsifier": (
            "On frozen natural-text rows, compare CMR joint logit-risk ranking against "
            "variance, invariant tensor mass, and random groups; require held-out CE/KL, "
            "interchange agreement, and sequential C512+MLP1+MLP2 composition."
        ),
    }
    publish_create_only(result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
