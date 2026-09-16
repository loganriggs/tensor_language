#!/usr/bin/env python3
"""CPU audit of the reusable sparse-interaction graph implementation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from basis_aligned.polynomial_causal import sparse_interaction_graph as graph
from basis_aligned.polynomial_causal.extracted_circuits.equality_l5h5_m4_explicit_interaction_graph_v1 import node as equality
from basis_aligned.polynomial_causal.extracted_circuits.subject_number_l11h3_sparse_graph_v1 import node as subject

HERE = Path(__file__).resolve().parent
OUT = HERE / "SPARSE_INTERACTION_GRAPH_V1_RESULT.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rng = np.random.default_rng(20260917)
    dividends = {mask: rng.normal(size=(9, 2)) for mask in range(16)}
    corners = {mask: sum((value for term, value in dividends.items() if term & mask == term),
                         np.zeros((9, 2))) for mask in range(16)}
    recovered = graph.full_dividends(corners, 4)
    transform_error = max(float(np.max(np.abs(recovered[mask] - dividends[mask]))) for mask in range(16))
    closure_error = max(float(np.max(np.abs(graph.reconstruct(recovered, mask) - corners[mask]))) for mask in range(16))

    minimal = graph.required_corners(subject.MASKS, len(subject.PORTS))
    target_atoms = {mask: rng.normal(size=12) for mask in subject.MASKS}
    baseline = rng.normal(size=12)
    subject_corners = {0: baseline}
    for corner in minimal[1:]:
        subject_corners[corner] = baseline - sum(
            (value for mask, value in target_atoms.items() if mask & corner == mask), np.zeros_like(baseline))
    generic_subject = graph.damage_atoms(subject_corners, subject.MASKS, len(subject.PORTS))
    packaged_subject = subject.decompose({mask: torch.tensor(value) for mask, value in subject_corners.items()})
    subject_error = max(float(np.max(np.abs(generic_subject[mask] - packaged_subject[name].numpy())))
                        for mask, name in zip(subject.MASKS, subject.TERMS))

    torch.manual_seed(20260917)
    baseline_t = torch.randn(2, 3, dtype=torch.float64)
    child = torch.randn_like(baseline_t); remainder = torch.randn_like(baseline_t)
    interaction = torch.randn_like(baseline_t)
    equality_corners = {0: baseline_t, 1: baseline_t + child, 2: baseline_t + remainder,
                        3: baseline_t + child + remainder + interaction}
    generic_equality = graph.full_dividends(equality_corners, 2)
    packaged_equality = equality.decompose(*(equality_corners[mask] for mask in range(4)))
    equality_error = max(float((generic_equality[mask] - packaged_equality[name]).abs().max())
                         for mask, name in ((0, "baseline"), (1, "child_effect"),
                                            (2, "remainder_effect"), (3, "interaction")))

    atoms = {1: np.array([3., 0., 2., 0., 4., 0.]), 2: np.array([0., 2., 0., 3., 0., 4.]),
             3: np.array([.1, -.1, .1, -.1, .1, -.1])}
    target = atoms[1] + atoms[2]
    selected, curve = graph.greedy_select(target, atoms, [0, 1, 2, 3], 2)
    reports = graph.evaluate_frozen(target, atoms, selected,
                                    {"discovery": [0, 1, 2, 3], "fresh_ood": [4, 5]})
    overlap_rejected = False
    try:
        graph.validate_panels({"discovery": [0, 1], "ood": [1, 2]})
    except ValueError:
        overlap_rejected = True

    predicates = {
        "pred_a_exact_vector_mobius_and_closure": transform_error <= 1e-12 and closure_error <= 1e-12,
        "pred_b_minimal_corner_subject_replay": minimal == subject.REQUIRED_CORNERS and subject_error <= 1e-12,
        "pred_c_independent_equality_package_replay": equality_error <= 1e-12,
        "pred_d_frozen_selection_and_ood_audit": selected == (1, 2)
            and max(report["relative_l2"] for report in reports.values()) <= 1e-12 and overlap_rejected,
    }
    result = {
        "schema": "sparse_interaction_graph_v1_result",
        "terminal": "reusable_sparse_interaction_graph_verified" if all(predicates.values()) else "invalid",
        "predictions": predicates,
        "instrument": {"transform_max_abs_error": transform_error, "closure_max_abs_error": closure_error,
                       "subject_package_max_abs_error": subject_error, "equality_package_max_abs_error": equality_error,
                       "subject_required_corners": list(minimal), "subject_full_lattice_corners": 32,
                       "subject_corner_reduction_fraction": 1. - len(minimal) / 32., "overlapping_ood_rejected": overlap_rejected},
        "selection_fixture": {"selected_masks": list(selected), "discovery_curve": list(curve), "panels": reports},
        "implementation_sha256": sha(HERE / "sparse_interaction_graph.py"),
        "test_sha256": sha(HERE / "test_sparse_interaction_graph.py"),
        "runner_sha256": sha(Path(__file__)),
        "replayed_packages": ["subject_number_l11h3_sparse_graph_v1", "equality_l5h5_m4_explicit_interaction_graph_v1"],
        "scope": "CPU verification of reusable exact decomposition, minimal-corner extraction, frozen discovery selection, disjoint OOD evaluation, and parity with two independently exported circuit graphs. No new model-behavior claim.",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(predicates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
