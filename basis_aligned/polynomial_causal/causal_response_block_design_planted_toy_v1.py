"""Replay the planted block-sensor known answer without model or corpus access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from causal_response_factorization_v1 import (
    block_d_optimal_anchor_mask,
    infer_document_codes,
    predict_from_codes,
    prospective_anchor_arm_mask,
)


HERE = Path(__file__).resolve().parent
SOURCE_NAMES = (
    "CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_1.md",
    "causal_response_factorization_v1.py",
    "test_causal_response_factorization_v1.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_receipt() -> dict[str, object]:
    generator = torch.Generator().manual_seed(20260830)
    phases, sources, targets, rank, documents = 2, 20, 10, 8, 200
    basis = torch.randn(
        (phases * sources * targets, rank), generator=generator, dtype=torch.float64
    )
    arm_scale = torch.exp(
        torch.linspace(-2.5, 1.0, phases * sources, dtype=torch.float64)
    )
    basis = (
        basis.reshape(phases * sources, targets, rank)
        * arm_scale[:, None, None]
    ).reshape(phases * sources * targets, rank).contiguous()
    basis = basis @ torch.randn((rank, rank), generator=generator, dtype=torch.float64)
    true_codes = torch.randn(
        (documents, rank), generator=generator, dtype=torch.float64
    )
    truth = basis @ true_codes.T
    valid = torch.ones_like(truth, dtype=torch.bool)
    d_mask, d_arms, path = block_d_optimal_anchor_mask(
        basis, shape=(phases, sources, targets), arms=4
    )
    h_mask, h_arms = prospective_anchor_arm_mask(
        phases, sources, targets, arms=4
    )
    noise = 0.02 * truth.std() * torch.randn(
        truth.shape, generator=generator, dtype=torch.float64
    )

    def score(mask: torch.Tensor) -> dict[str, object]:
        observed = truth.clone()
        observed[mask] = truth[mask] + noise[mask]
        codes, supported = infer_document_codes(
            basis, observed, valid, mask, ridge=1e-8, minimum_anchor_ratio=1
        )
        prediction = predict_from_codes(basis, codes)
        nonanchor = (~mask)[:, None].expand_as(truth)
        mse = float(((prediction[nonanchor] - truth[nonanchor]) ** 2).mean())
        singular = torch.linalg.svdvals(basis[mask])
        return {
            "code_mse": float(((codes - true_codes) ** 2).mean()),
            "condition_number": float(singular[0] / singular[-1]),
            "minimum_design_singular_value": float(singular[-1]),
            "mse_nonanchor": mse,
            "nrmse": mse ** 0.5 / float(truth.square().mean().sqrt()),
            "supported_documents": int(supported.sum()),
        }

    d_report = score(d_mask)
    h_report = score(h_mask)
    return {
        "claim_boundary": (
            "CPU planted sensor-design known answer only; no bilin18 model, corpus, "
            "FIT bundle, EVAL role, or protected outcome was opened"
        ),
        "code_dimension": rank,
        "d_optimal": {"arms": list(d_arms), "logdet_path": list(path), **d_report},
        "noise_sd_fraction_of_response_sd": 0.02,
        "nonanchor_mse_ratio_dopt_over_hash": (
            float(d_report["mse_nonanchor"]) / float(h_report["mse_nonanchor"])
        ),
        "outcome_blind_hash": {"arms": list(h_arms), **h_report},
        "physical_arm_budget": 4,
        "schema": "causal_response_block_design_planted_toy_v1",
        "seed": 20260830,
        "shape": [phases, sources, targets, documents],
        "source_sha256": {name: _sha256(HERE / name) for name in SOURCE_NAMES},
    }


if __name__ == "__main__":
    print(json.dumps(build_receipt(), indent=2, sort_keys=True))

