#!/usr/bin/env python3
"""Known-answer toy for an intervention-relative projected causal abstraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
C = np.array([[1.0, -2.0], [0.5, 3.0]])


def low_edit(z: np.ndarray, name: str) -> np.ndarray:
    out = z.copy()
    if name == "add_a":
        out[:, 0] += 0.7
    elif name == "mix_ab":
        out[:, 1] += 0.4 * out[:, 0]
    elif name == "hidden_nuisance_to_a":
        out[:, 0] += 0.6 * out[:, 2]
    else:
        raise KeyError(name)
    return out


def high_edit(u: np.ndarray, name: str) -> np.ndarray:
    out = u.copy()
    if name == "add_a":
        out[:, 0] += 0.7
    elif name == "mix_ab":
        out[:, 1] += 0.4 * out[:, 0]
    else:
        raise KeyError(name)
    return out


def encode(z: np.ndarray) -> np.ndarray:
    return z @ P.T


def abstract_suffix(u: np.ndarray) -> np.ndarray:
    return u @ C.T


def concrete_suffix(z: np.ndarray, epsilon: float) -> np.ndarray:
    nuisance = epsilon * np.tanh(z[:, 2:3])
    return abstract_suffix(encode(z)) + np.concatenate((nuisance, -nuisance), axis=1)


def max_row_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, axis=1).max(initial=0.0))


def fit_linear(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    design = np.concatenate((np.ones((len(train_x), 1)), train_x), axis=1)
    coef = np.linalg.lstsq(design, train_y, rcond=None)[0]
    return np.concatenate((np.ones((len(test_x), 1)), test_x), axis=1) @ coef


def run(seed: int = 260830, epsilon: float = 0.03) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = 20_000, 8_000
    scales = np.array([1.0, 0.35, 8.0])
    train = rng.normal(size=(n_train, 3)) * scales
    test = rng.normal(size=(n_test, 3)) * scales
    allowed = ("add_a", "mix_ab")

    commute = {}
    suffix_error = {}
    for name in allowed:
        low = low_edit(test, name)
        high = high_edit(encode(test), name)
        commute[name] = max_row_norm(encode(low) - high)
        suffix_error[name] = max_row_norm(
            concrete_suffix(low, epsilon) - abstract_suffix(high)
        )

    low_composed = low_edit(low_edit(test, "add_a"), "mix_ab")
    high_composed = high_edit(high_edit(encode(test), "add_a"), "mix_ab")
    composition_commute = max_row_norm(encode(low_composed) - high_composed)
    composition_suffix_error = max_row_norm(
        concrete_suffix(low_composed, epsilon) - abstract_suffix(high_composed)
    )
    certified_suffix_bound = float(np.sqrt(2.0) * epsilon)

    # Same abstract state, different nuisance.  Allowed edits preserve the fiber;
    # the deliberately hidden edit transports nuisance into the abstract state.
    fiber = np.array([[0.25, -0.4, -2.0], [0.25, -0.4, 2.0]])
    allowed_fiber_spread = max(
        max_row_norm(np.diff(encode(low_edit(fiber, name)), axis=0)) for name in allowed
    )
    hidden_fiber_separation = max_row_norm(
        np.diff(encode(low_edit(fiber, "hidden_nuisance_to_a")), axis=0)
    )

    # A variance-optimal rank-2 PCA code keeps the high-variance nuisance and a,
    # discarding b even though b is causally required by the suffix.
    centered = train - train.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pca_basis = vt[:2].T
    target_train = abstract_suffix(encode(train))
    target_test = abstract_suffix(encode(test))
    pca_prediction = fit_linear(train @ pca_basis, target_train, test @ pca_basis)
    causal_prediction = fit_linear(encode(train), target_train, encode(test))
    pca_mse = float(np.mean((pca_prediction - target_test) ** 2))
    causal_mse = float(np.mean((causal_prediction - target_test) ** 2))

    # Gauge test: rotate the microstate and conjugate encoder/interventions back.
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    gauged = test @ q
    gauge_commute = {}
    for name in allowed:
        low_original = low_edit(gauged @ q.T, name)
        low_gauged = low_original @ q
        encoded_gauged = low_gauged @ q.T @ P.T
        high = high_edit((gauged @ q.T) @ P.T, name)
        gauge_commute[name] = max_row_norm(encoded_gauged - high)

    checks = {
        "allowed_edits_commute": max(commute.values()) < 1e-12,
        "allowed_composition_commutes": composition_commute < 1e-12,
        "suffix_error_within_certificate": max(
            *suffix_error.values(), composition_suffix_error,
        ) <= certified_suffix_bound + 1e-12,
        "allowed_edits_preserve_fibers": allowed_fiber_spread < 1e-12,
        "hidden_intervention_breaks_abstraction": hidden_fiber_separation > 1.0,
        "causal_code_beats_matched_rank_pca": causal_mse < 1e-20 and pca_mse > 0.1,
        "gauge_invariant": max(gauge_commute.values()) < 1e-12,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {
        "schema": "toy_projected_causal_abstraction_receipt_v1",
        "seed": seed,
        "epsilon": epsilon,
        "n_train": n_train,
        "n_test": n_test,
        "allowed_commutation_max": commute,
        "composition_commutation_max": composition_commute,
        "allowed_suffix_error_max": suffix_error,
        "composition_suffix_error_max": composition_suffix_error,
        "certified_suffix_error_bound": certified_suffix_bound,
        "allowed_fiber_spread": allowed_fiber_spread,
        "hidden_fiber_separation": hidden_fiber_separation,
        "matched_rank_pca_mse": pca_mse,
        "causal_code_mse": causal_mse,
        "gauged_commutation_max": gauge_commute,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
