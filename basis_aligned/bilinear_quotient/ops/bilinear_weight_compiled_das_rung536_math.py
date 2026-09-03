#!/usr/bin/env python3
"""Stage-A algebra and planted recovery for weight-compiled product-space DAS."""

from __future__ import annotations

import json
from pathlib import Path

import torch


OUT = Path(__file__).resolve().parents[1] / "bilinear_weight_compiled_das_rung536_math.json"
SEED = 536
D, H, M, K = 12, 24, 17, 3
N_FIT, N_TEST = 768, 384
STEPS = 1200
LR = 0.04


def orthonormal(matrix):
    return torch.linalg.qr(matrix, mode="reduced").Q


def product_features(x, left, right):
    return (x @ left.T) * (x @ right.T)


def compiled_forms(left, right, down, basis):
    forms, decoders = [], []
    for index in range(basis.shape[1]):
        direction = basis[:, index]
        ordered = left.T @ torch.diag(direction) @ right
        forms.append(0.5 * (ordered + ordered.T))
        decoders.append(down @ direction)
    return torch.stack(forms), torch.stack(decoders, dim=1)


def compiled_output(x, forms, decoders):
    scalars = torch.einsum("nd,kde,ne->nk", x, forms, x)
    return scalars @ decoders.T


def main():
    torch.manual_seed(SEED)
    dtype = torch.float64
    left = torch.randn(H, D, dtype=dtype)
    right = torch.randn(H, D, dtype=dtype)
    down = torch.randn(M, H, dtype=dtype)
    planted = orthonormal(torch.randn(H, K, dtype=dtype))
    planted_projector = planted @ planted.T

    x = torch.randn(N_FIT + N_TEST, D, dtype=dtype)
    donor = torch.randn(N_FIT + N_TEST, D, dtype=dtype)
    delta = product_features(donor, left, right) - product_features(x, left, right)
    target = delta @ planted_projector

    raw = torch.nn.Parameter(torch.randn(H, K, dtype=dtype))
    optimizer = torch.optim.Adam([raw], lr=LR)
    losses = []
    for _step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        basis = orthonormal(raw)
        projector = basis @ basis.T
        prediction = delta[:N_FIT] @ projector
        loss = (prediction - target[:N_FIT]).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    learned = orthonormal(raw.detach())
    projector = learned @ learned.T
    heldout_prediction = delta[N_FIT:] @ projector
    heldout_target = target[N_FIT:]
    relative_error = float(
        torch.linalg.vector_norm(heldout_prediction - heldout_target)
        / torch.linalg.vector_norm(heldout_target))
    overlap = float(torch.trace(planted_projector @ projector) / K)

    forms, decoders = compiled_forms(left, right, down, learned)
    direct = product_features(x, left, right) @ projector @ down.T
    compiled = compiled_output(x, forms, decoders)
    compilation_error = float((direct - compiled).abs().max())

    direct_interchange = delta @ projector @ down.T
    compiled_interchange = (
        compiled_output(donor, forms, decoders) - compiled_output(x, forms, decoders))
    interchange_error = float((direct_interchange - compiled_interchange).abs().max())

    rotation = orthonormal(torch.randn(K, K, dtype=dtype))
    rotated = learned @ rotation
    gauge_error = float((
        product_features(x, left, right) @ (rotated @ rotated.T) @ down.T - direct
    ).abs().max())

    checks = {
        "pred_a_exact_weight_compilation": compilation_error <= 1e-10,
        "pred_b_exact_interchange_compilation": interchange_error <= 1e-10,
        "pred_c_basis_gauge": gauge_error <= 1e-10,
        "pred_d_planted_recovery": overlap >= 0.99 and relative_error <= 0.05,
    }
    result = {
        "status": "cpu_stage_a_complete",
        "rung": 536,
        **checks,
        "all_stage_a_checks_pass": all(checks.values()),
        "dimensions": {"input": D, "product": H, "output": M, "subspace": K},
        "fit_pairs": N_FIT,
        "heldout_pairs": N_TEST,
        "optimizer": {"steps": STEPS, "learning_rate": LR},
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "projector_overlap": overlap,
        "heldout_projected_interchange_relative_error": relative_error,
        "maximum_weight_compilation_error": compilation_error,
        "maximum_interchange_compilation_error": interchange_error,
        "maximum_basis_gauge_error": gauge_error,
        "model_loaded": False,
        "new_model_forwards": 0,
        "real_model_das_authorized": False,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
