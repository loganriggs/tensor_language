#!/usr/bin/env python3
"""RUNG518 -- circuit-defined quotient of MLP0 head-by-source pieces.

pred_a: the exact/live 45-piece intervention and planted-recovery instrument passes
pred_b: 1--16 pairs pass two-half task+circuit discovery and permutation control
pred_c: a frozen pair predicts the other 30 circuit families and documents
pred_d: a confirmed pair passes bidirectional physical replacement
pred_e: a physical component crosses heads while splitting one native head

BQGATE: EXPERIMENT
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
PREREG = POLY / "MLP0_HEAD_RELATION_CIRCUIT_QUOTIENT_RUNG518_PREREGISTRATION.md"
PREREG_SHA256 = "53d3ab942a43a3b9a49f1d64682b8c603871d928f80ef737f18c556b67a1b25e"
GROUPS = ("SELF", "PREVIOUS", "NEAR", "DISTANT_SAME", "DISTANT_OTHER")
N_HEADS = 9
N_ATOMS = N_HEADS * len(GROUPS)
ATOM_NAMES = tuple(f"H{head}.{group}" for head in range(N_HEADS) for group in GROUPS)
PLANTED_PAIRS = ((0, 7), (3, 14), (8, 31), (20, 44))
PLANTED_SEEDS = tuple(range(51800, 51808))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atom_index(head: int, group: int) -> int:
    if not 0 <= head < N_HEADS or not 0 <= group < len(GROUPS):
        raise ValueError("head or relation index is outside the frozen vocabulary")
    return head * len(GROUPS) + group


def atom_parts(atom: int) -> tuple[int, int]:
    if not 0 <= atom < N_ATOMS:
        raise ValueError("atom index is outside the frozen vocabulary")
    return divmod(atom, len(GROUPS))


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    return float((left @ right) / denominator.clamp_min(1e-30))


def _relative_residual(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    return float((actual - predicted).norm() / actual.norm().clamp_min(1e-30))


def pair_metrics(responses: dict, left: int, right: int) -> dict:
    """Fit half0 circuit scale and score both backgrounds without pooling them away."""
    fit_left = responses["half0"]["circuit"][left].reshape(-1).double()
    fit_right = responses["half0"]["circuit"][right].reshape(-1).double()
    beta = float((fit_left @ fit_right) / fit_right.square().sum().clamp_min(1e-30))
    safe_beta = beta if abs(beta) > 1e-30 else 1.0
    row = {"beta_left_from_right": beta, "halves": {}}
    material = True
    holds = .25 <= abs(beta) <= 4
    for half in ("half0", "half1"):
        row["halves"][half] = {}
        for background in range(2):
            entry = {}
            for kind in ("circuit", "task"):
                lvec = responses[half][kind][left, background].double()
                rvec = responses[half][kind][right, background].double()
                predicted_left = beta * rvec
                predicted_right = lvec / safe_beta
                signed_cosine = _cosine(lvec, rvec) * (1 if beta >= 0 else -1)
                entry[kind] = {
                    "signed_cosine": signed_cosine,
                    "left_from_right_relative_residual": _relative_residual(
                        lvec, predicted_left),
                    "right_from_left_relative_residual": _relative_residual(
                        rvec, predicted_right),
                }
                if kind == "circuit":
                    material &= min(float(lvec.square().mean().sqrt()),
                                    float(rvec.square().mean().sqrt())) >= .0005
                    holds &= (signed_cosine >= .85
                              and max(entry[kind]["left_from_right_relative_residual"],
                                      entry[kind]["right_from_left_relative_residual"]) <= .50)
                else:
                    material &= min(float(lvec.norm()), float(rvec.norm())) >= .00025
                    holds &= (signed_cosine >= .70
                              and max(entry[kind]["left_from_right_relative_residual"],
                                      entry[kind]["right_from_left_relative_residual"]) <= .65)
            row["halves"][half][str(background)] = entry
    row["material"] = bool(material)
    row["holds"] = bool(material and holds)
    return row


def discover_pairs(responses: dict) -> list[dict]:
    candidates = []
    for left in range(N_ATOMS):
        for right in range(left + 1, N_ATOMS):
            metrics = pair_metrics(responses, left, right)
            if metrics["holds"]:
                candidates.append({
                    "left": left, "right": right,
                    "left_name": ATOM_NAMES[left], "right_name": ATOM_NAMES[right],
                    **metrics,
                })
    return candidates


def planted_problem(seed: int) -> tuple[dict, set[tuple[int, int]]]:
    generator = torch.Generator().manual_seed(seed)
    responses = {}
    for half in ("half0", "half1"):
        responses[half] = {
            "circuit": .01 * torch.randn(N_ATOMS, 2, 32, generator=generator,
                                          dtype=torch.float64),
            "task": .01 * torch.randn(N_ATOMS, 2, 4, generator=generator,
                                       dtype=torch.float64),
        }
    betas = (0.5, -0.75, 1.5, -2.0)
    for (left, right), beta in zip(PLANTED_PAIRS, betas):
        for half in responses.values():
            for kind in ("circuit", "task"):
                half[kind][left] = beta * half[kind][right]
    return responses, set(PLANTED_PAIRS)


def planted_suite() -> dict:
    cases = []
    all_exact = True
    for seed in PLANTED_SEEDS:
        responses, expected = planted_problem(seed)
        found = {(row["left"], row["right"]) for row in discover_pairs(responses)}
        exact = found == expected
        all_exact &= exact
        cases.append({"seed": seed, "expected": sorted(expected), "found": sorted(found),
                      "exact": exact})
    return {"cases": cases, "all_eight_exact": bool(all_exact)}


def dry_run() -> dict:
    if sha256(PREREG) != PREREG_SHA256:
        raise RuntimeError("rung518 preregistration changed after source freeze")
    planted = planted_suite()
    if not planted["all_eight_exact"]:
        raise RuntimeError("rung518 planted pair detector failed")
    return {
        "status": "dry_run_passed", "rung": 518,
        "model_loaded": False, "model_outcomes_opened": False,
        "heads": N_HEADS, "relations": list(GROUPS), "atoms": N_ATOMS,
        "unordered_pairs": N_ATOMS * (N_ATOMS - 1) // 2,
        "registered_predictions": {
            'pred_a_exact_live_45_piece_instrument': None,
            'pred_b_small_circuit_defined_relation': None,
            'pred_c_heldout_circuit_prediction': None,
            'pred_d_bidirectional_physical_interchange': None,
            'pred_e_native_boundary_changing_unit': None,
        },
        "planted_recovery": planted,
        "preregistration_sha256": sha256(PREREG),
    }


def scientific_main() -> None:
    raise RuntimeError(
        "rung518 scientific path is fail-closed until exact atom construction, "
        "62-circuit collection, controls, confirmation, and physical replacement are implemented")


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(dry_run(), indent=2, sort_keys=True))
        return
    scientific_main()


if __name__ == "__main__":
    main()
