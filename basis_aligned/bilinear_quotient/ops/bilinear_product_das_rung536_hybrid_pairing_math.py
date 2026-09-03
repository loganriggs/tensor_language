#!/usr/bin/env python3
"""Rung 536 CPU known-answer test for exact T/I hybrid-pair targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parents[1] / "bilinear_product_das_rung536_hybrid_pairing_math.json"
SEED = 536042
TOL = 2e-13


def relerr(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-300))


def main() -> None:
    rng = np.random.default_rng(SEED)
    input_dim, product_dim, output_dim, examples = 11, 23, 13, 37
    left = rng.standard_normal((product_dim, input_dim))
    right = rng.standard_normal((product_dim, input_dim))
    down = rng.standard_normal((output_dim, product_dim))
    p_base = rng.standard_normal((examples, input_dim))
    q_base = rng.standard_normal((examples, input_dim))
    p_donor = rng.standard_normal((examples, input_dim))
    q_donor = rng.standard_normal((examples, input_dim))

    def lr(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x @ left.T, x @ right.T

    def g(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        l, r = lr(p + q)
        return l * r

    def branches(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lp, rp = lr(p)
        lq, rq = lr(q)
        return lp * rp, lp * rq + lq * rp, lq * rq

    g_base = g(p_base, q_base)
    t_base, i_base, c_base = branches(p_base, q_base)
    t_donor, _, _ = branches(p_donor, q_base)
    _, i_donor, c_donor = branches(p_base, q_donor)

    token_hybrid_delta = g(p_donor, q_base) - g_base
    context_hybrid_delta = g(p_base, q_donor) - g_base
    token_expected_delta = (t_donor - t_base) + (
        branches(p_donor, q_base)[1] - i_base
    )
    context_expected_delta = (i_donor - i_base) + (c_donor - c_base)

    token_target = g_base + t_donor - t_base
    interaction_target = g_base + i_donor - i_base
    full_closure = t_base + i_base + c_base

    metrics = {
        "full_T_I_C_closure_relative_error": relerr(full_closure, g_base),
        "token_hybrid_delta_identity_relative_error": relerr(
            token_hybrid_delta, token_expected_delta
        ),
        "context_hybrid_delta_identity_relative_error": relerr(
            context_hybrid_delta, context_expected_delta
        ),
        "token_target_output_identity_relative_error": relerr(
            token_target @ down.T,
            g_base @ down.T + (t_donor - t_base) @ down.T,
        ),
        "interaction_target_output_identity_relative_error": relerr(
            interaction_target @ down.T,
            g_base @ down.T + (i_donor - i_base) @ down.T,
        ),
    }
    passed = all(value <= TOL for value in metrics.values())
    assert passed

    result = {
        "rung": 536,
        "stage": "hybrid_pairing_known_answer",
        "status": "passed",
        "seed": SEED,
        "dimensions": {
            "examples": examples,
            "input": input_dim,
            "product": product_dim,
            "output": output_dim,
        },
        "tolerance": TOL,
        "metrics": metrics,
        "interpretation": {
            "token_hybrid_contains": ["token_only_delta", "token_by_context_delta"],
            "context_hybrid_contains": ["token_by_context_delta", "context_only_delta"],
            "DAS_task": "project each hybrid delta onto only its exact requested branch delta",
            "fixed_low_dimensional_projector_guaranteed": False,
        },
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    OUT.write_text(payload)
    print(payload, end="")
    print("result_sha256", hashlib.sha256(payload.encode()).hexdigest())


if __name__ == "__main__":
    main()
