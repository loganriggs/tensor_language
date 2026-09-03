#!/usr/bin/env python3
"""Independent terminal audit of rung 527's discovery-gated result."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

import torch


REPO = Path("/workspace/tensor_language")
BQ = REPO / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
sys.path.insert(0, str(OPS))

import mlp0_centered_context_source_quotient_rung527_math as qm  # noqa: E402


RESULT = BQ / "mlp0_centered_context_source_quotient_rung527_results.json"
RUNNER = OPS / "mlp0_centered_context_source_quotient_rung527_run.py"
MATH = OPS / "mlp0_centered_context_source_quotient_rung527_math.py"
SMOKE = BQ / "mlp0_centered_context_source_quotient_rung527_gpu_smoke_results.json"
OUT = BQ / "mlp0_centered_context_source_quotient_rung527_terminal_audit.json"

EXPECTED = {
    RESULT: "8f8581c2fc0a29ffd45f6383eab9af58d9a239c5715e9467a16843e33f5ee682",
    RUNNER: "15bad0e96087f59239096e3aef41c93af284a5da60573d61caa34f36b83f7e61",
    MATH: "b70524d35eaba7e8aea37e3cfd3a4e042a251b03c09059685cb8b8875cf8e833",
    SMOKE: "c93aa35817e787248b456bb1db12b093ba0e582604d85316ec13ecae16e7c4c8",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _effect_views(collection: dict) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.tensor(collection["sums"], dtype=torch.float64)
    counts = torch.tensor(collection["counts"], dtype=torch.float64)
    if tuple(sums.shape) != (20, 2, 2, 32) or tuple(counts.shape) != (2, 2, 32):
        raise AssertionError("discovery sum/count shape changed")
    if not bool((counts > 0).all()):
        raise AssertionError("discovery contains an unsupported circuit cell")
    means = sums / counts.unsqueeze(0)
    halves = means[:, :, 0] - means[:, :, 1]
    pooled_means = sums.sum(1) / counts.sum(0).unsqueeze(0)
    pooled = pooled_means[:, 0] - pooled_means[:, 1]
    return halves, pooled


def audit_data(result: dict) -> dict:
    if result.get("rung") != 527 or result.get("status") != "complete":
        raise AssertionError("result identity changed")
    if result.get("runner_sha256") != EXPECTED[RUNNER]:
        raise AssertionError("recorded runner hash changed")
    if result.get("smoke_authority", {}).get("sha256") != EXPECTED[SMOKE]:
        raise AssertionError("smoke authority changed")
    if result.get("planted_suite", {}).get("holds") is not True:
        raise AssertionError("planted suite did not pass")

    collection = result["discovery"]["collection"]
    halves, pooled = _effect_views(collection)
    stored_halves = torch.tensor(result["discovery"]["effects_by_half"], dtype=torch.float64)
    stored_pooled = torch.tensor(result["discovery"]["pooled_effects"], dtype=torch.float64)
    if not torch.equal(halves, stored_halves) or not torch.equal(pooled, stored_pooled):
        raise AssertionError("stored effect vectors do not reproduce sums/counts exactly")

    candidates, summary = qm.discover_pairs(halves)
    controls = qm.permutation_control_counts(halves, range(527_300, 527_316))
    control_q95 = float(torch.quantile(
        torch.tensor(controls, dtype=torch.float64), .95, interpolation="higher"))
    if summary != result["discovery"]["summary"]:
        raise AssertionError("discovery summary does not independently reproduce")
    if candidates != result["discovery"]["candidates"]:
        raise AssertionError("discovery candidates do not independently reproduce")
    if controls != result["discovery"]["permutation_control_counts"] \
            or control_q95 != result["discovery"]["permutation_control_q95_higher"]:
        raise AssertionError("permutation controls do not independently reproduce")

    diagnostics = collection["diagnostics"]
    required_instrument = bool(
        diagnostics["calls_exact"]
        and diagnostics["state_replay_max_abs"] == 0.0
        and diagnostics["source_partition_maximum_relative_squared"] <= 1e-12
        and diagnostics["context_closure_relative_squared"] <= 1e-12
        and max(diagnostics["remainder_energy_fraction"]) <= 0.01
        and diagnostics["minimum_term_edit_rms"] > 0
        and diagnostics["zero_term_edits"] == 0
        and diagnostics["supports_positive"])
    pred_a = bool(required_instrument and result["planted_suite"]["holds"])
    pred_b = bool(pred_a and summary["small_relation"] and len(candidates) > control_q95)
    expected_predictions = {
        "pred_a_exact_live_instrument": pred_a,
        "pred_b_small_discovery_equivalence_relation": pred_b,
        "pred_c_heldout_circuits_and_documents": False,
        "pred_d_bidirectional_physical_substitution": False,
        "pred_e_nontrivial_context_term_grouping": False,
    }
    for key, value in expected_predictions.items():
        if result.get(key) is not value:
            raise AssertionError(f"prediction mismatch for {key}")
    if result.get("strong_null") is not True:
        raise AssertionError("registered strong null did not fire")
    if result.get("confirmation_opened") is not False or result.get("confirmation") is not None:
        raise AssertionError("confirmation did not remain sealed")
    if result.get("physical_substitution_opened") is not False \
            or result.get("substitutions") is not None:
        raise AssertionError("physical substitutions did not remain sealed")
    if result["execution_price"] != {
        "model_backwards": 0, "model_forwards": 1302,
        "peak_gpu_memory_bytes": result["execution_price"]["peak_gpu_memory_bytes"],
    }:
        raise AssertionError("execution count changed")

    rms = halves.flatten(1).square().mean(1).sqrt()
    stability = torch.tensor([
        qm.safe_cosine(halves[term, 0], halves[term, 1])
        for term in range(qm.N_TERMS)], dtype=torch.float64)
    incremental = {"material_and_scale": 0, "passes_D0": 0, "passes_D1": 0,
                   "passes_both": 0}
    pair_rows = []
    for left, right in itertools.combinations(range(qm.N_TERMS), 2):
        denominator = halves[right, 0].square().sum().clamp_min(1e-30)
        beta = float(halves[right, 0].dot(halves[left, 0]) / denominator)
        if abs(beta) < 1e-30:
            continue
        row = qm.pair_metrics(halves, left, right, beta)
        d0, d1 = row["windows"]["half0"], row["windows"]["half1"]
        material_scale = row["material"] and row["scale_holds"]
        pass_d0 = bool(
            min(d0["left_from_right_cosine"], d0["right_from_left_cosine"]) >= .90
            and max(d0["left_from_right_relative_residual"],
                    d0["right_from_left_relative_residual"]) <= .35)
        pass_d1 = bool(
            min(d1["left_from_right_cosine"], d1["right_from_left_cosine"]) >= .80
            and max(d1["left_from_right_relative_residual"],
                    d1["right_from_left_relative_residual"]) <= .50)
        incremental["material_and_scale"] += int(material_scale)
        incremental["passes_D0"] += int(material_scale and pass_d0)
        incremental["passes_D1"] += int(material_scale and pass_d1)
        incremental["passes_both"] += int(material_scale and pass_d0 and pass_d1)
        pair_rows.append({
            "left": row["left_name"], "right": row["right_name"], "beta": beta,
            "D0_cosine": d0["left_from_right_cosine"],
            "D0_relative_residual": d0["left_from_right_relative_residual"],
            "D1_cosine": d1["left_from_right_cosine"],
            "D1_relative_residual": d1["left_from_right_relative_residual"],
        })
    pair_rows.sort(key=lambda row: (
        min(row["D0_cosine"], row["D1_cosine"])
        - max(row["D0_relative_residual"], row["D1_relative_residual"])), reverse=True)
    return {
        "status": "audit_passed", "rung": 527,
        "hashes": {str(path.relative_to(REPO)): expected for path, expected in EXPECTED.items()},
        "predictions_recomputed": expected_predictions,
        "candidate_count_recomputed": len(candidates),
        "permutation_counts_recomputed": controls,
        "confirmation_and_substitution_sealed": True,
        "material_term_count": int((rms >= qm.MATERIAL_RMS).sum()),
        "term_rms_range": [float(rms.min()), float(rms.max())],
        "term_cross_half_cosine": {
            "minimum": float(stability.min()), "median": float(stability.median()),
            "maximum": float(stability.max()),
        },
        "incremental_pair_gates": incremental,
        "best_descriptive_pairs": pair_rows[:8],
        "calls_reconciled": True,
    }


def main() -> None:
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise AssertionError(f"frozen terminal hash changed: {path}")
    audit = audit_data(json.loads(RESULT.read_text()))
    if OUT.exists():
        raise FileExistsError(f"audit output already exists: {OUT}")
    temporary = OUT.with_name(OUT.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    os.link(temporary, OUT)
    temporary.unlink()
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
