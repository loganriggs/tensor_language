#!/usr/bin/env python3
"""Independent CPU audit of rung 529 sufficient statistics and frozen gates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


REPO = Path("/workspace/tensor_language")
BQ = REPO / "basis_aligned/bilinear_quotient"
POLY = REPO / "basis_aligned/polynomial_causal"
RESULT = BQ / "equality_shared_private_transition_consensus_rung529_results.json"
BUNDLE = BQ / "equality_shared_private_transition_consensus_rung529_bundle.pt"
OUT = BQ / "equality_shared_private_transition_consensus_rung529_terminal_audit.json"
FROZEN = {
    RESULT: "48fcea16042edff7de071de1a12fc99903c4b7f76aa81b7947665fbe175ed446",
    BUNDLE: "4fd6bc3c153adfebd8ab7656cbd51778f28595f28b8a9c807b19a7ec58b26bed",
    BQ / "ops/equality_shared_private_transition_consensus_rung529_run.py":
        "8d3c00b73efaca5427d32d3fec0cd8629ffe3cddb70389a9d4c878cb38730f85",
    BQ / "ops/equality_shared_private_transition_consensus_rung529_full.py":
        "9dfcc568538e540ae04c55e8e841211f11d5e75078b38b8e268e94f5edce6886",
    BQ / "equality_shared_private_transition_consensus_rung529_gpu_smoke_v2_results.json":
        "03a039a0ea4735f196d9f84457803f88ac95eea46b19ae89de8b8eef5223d213",
    POLY / "EQUALITY_SHARED_PRIVATE_TRANSITION_CONSENSUS_RUNG529_PREREGISTRATION.md":
        "638a140610d800a9745157fbb1498bbb36d152d3a612ad54a82b9b3ac47c20ea",
    BQ / "ops/equality_shared_private_transition_consensus_rung529_math.py":
        "77503f27c2838af78a806f69c5d99b276e232eadca2ab4fed8c889b855e15014",
}
ACTIONS = ("N", "P", "Z7", "Z8")
CELLS = (
    "all_positive", "near_match", "far_match", "one_earlier_match",
    "multiple_earlier_matches", "off_target",
)
TASK_INDICES = tuple(range(1, 5))
SEEDS = tuple(range(529_300, 529_316))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(target: torch.Tensor, observed: torch.Tensor) -> dict[str, float]:
    target = target.double().reshape(-1)
    observed = observed.double().reshape(-1)
    target2 = float(target @ target)
    observed2 = float(observed @ observed)
    cross = float(target @ observed)
    cosine = cross / math.sqrt(max(target2 * observed2, 1e-300))
    residual = math.sqrt(float((target - observed) @ (target - observed)) / max(target2, 1e-300))
    return {"cosine": cosine, "relative_residual": residual}


def views(phase: dict[str, Any], prefix: str) -> dict[str, torch.Tensor]:
    task_sums = phase[f"{prefix}_task_sums"].double()
    circuit_sums = phase[f"{prefix}_circuit_sums"].double()
    task_counts = phase["task_counts"].double()
    circuit_counts = phase["circuit_counts"].double()
    split = phase["bounds"][2] - phase["bounds"][0]
    task_halves = torch.stack([
        task_sums[..., lo:hi, :].sum(-2) / task_counts[lo:hi].sum(0).clamp_min(1)
        for lo, hi in ((0, split), (split, task_counts.shape[0]))
    ], dim=-3)
    task_pooled = task_sums.sum(-2) / task_counts.sum(0).clamp_min(1)
    means = circuit_sums / circuit_counts.clamp_min(1)
    circuit_halves = (means[..., 0, :] - means[..., 1, :]).movedim(-2, -3)
    pooled_sums = circuit_sums.sum(-3)
    pooled_counts = circuit_counts.sum(0).clamp_min(1)
    pooled_means = pooled_sums / pooled_counts
    circuit_pooled = pooled_means[..., 0, :] - pooled_means[..., 1, :]
    return {
        "task_halves": task_halves[..., list(TASK_INDICES)],
        "task_pooled": task_pooled[..., list(TASK_INDICES)],
        "circuit_halves": circuit_halves,
        "circuit_pooled": circuit_pooled,
    }


def audit() -> dict[str, Any]:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite audit: {OUT}")
    hashes = {str(path.relative_to(REPO)): sha256(path) for path in FROZEN}
    if any(hashes[str(path.relative_to(REPO))] != expected for path, expected in FROZEN.items()):
        raise RuntimeError("frozen authority hash changed")
    result = json.loads(RESULT.read_text())
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    phases = bundle["phases"]
    if tuple(phases) != ("discovery", "confirmation"):
        raise RuntimeError("unexpected phase opening")

    discovery = phases["discovery"]
    target = views(discovery, "target")
    consensus = views(discovery, "consensus")
    single = views(discovery, "single")
    wrong = views(discovery, "wrong")
    discovery_checks = {}
    candidates = []
    for i, name in enumerate(discovery["targets"]):
        circuit = [metrics(target["circuit_halves"][i, h], consensus["circuit_halves"][i, h]) for h in range(2)]
        task = [metrics(target["task_halves"][i, h], consensus["task_halves"][i, h]) for h in range(2)]
        continuations = [[
            metrics(target["circuit_halves"][i, h, c], consensus["circuit_halves"][i, h, c])
            for c in range(4)] for h in range(2)]
        single_indices = [j for j, pair in enumerate(discovery["single_pairs"]) if pair[0] == name]
        wrong_indices = [j for j, pair in enumerate(discovery["wrong_pairs"]) if pair[0] == name]
        singles = {
            discovery["single_pairs"][j][1]: [
                metrics(target["circuit_halves"][i, h], single["circuit_halves"][j, h]) for h in range(2)]
            for j in single_indices
        }
        wrongs = {
            discovery["wrong_pairs"][j][1]: [
                metrics(target["circuit_halves"][i, h], wrong["circuit_halves"][j, h]) for h in range(2)]
            for j in wrong_indices
        }
        permutation_cosines = []
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            scrambled = consensus["circuit_halves"][i].clone()
            for continuation in range(4):
                order = torch.randperm(scrambled.shape[-1], generator=generator)
                scrambled[:, continuation] = scrambled[:, continuation, order]
            permutation_cosines.append(metrics(target["circuit_halves"][i, 0], scrambled[0])["cosine"])
        q95 = float(torch.quantile(torch.tensor(permutation_cosines, dtype=torch.float64), .95))
        best_single = min(singles, key=lambda donor: singles[donor][0]["relative_residual"])
        strongest_wrong = max(wrongs, key=lambda control: wrongs[control][0]["cosine"])
        live = bool(
            (target["circuit_halves"][i].square().mean(-1).sqrt() >= 5e-4).all()
            and (consensus["circuit_halves"][i].square().mean(-1).sqrt() >= 5e-4).all()
            and target["task_halves"][i].reshape(2, -1).norm(dim=-1).min() >= 2.5e-4
            and consensus["task_halves"][i].reshape(2, -1).norm(dim=-1).min() >= 2.5e-4)
        gates = {
            "live": live,
            "circuit_d0": circuit[0]["cosine"] >= .90 and circuit[0]["relative_residual"] <= .35,
            "circuit_d1": circuit[1]["cosine"] >= .80 and circuit[1]["relative_residual"] <= .50,
            "task_both": all(row["cosine"] >= .70 and row["relative_residual"] <= .65 for row in task),
            "continuations": all(
                continuations[h][c]["cosine"] >= (.65 if h == 0 else .55)
                for h in range(2) for c in range(4)),
            "d0_beats_every_single_by_005": all(
                circuit[0]["relative_residual"] <= rows[0]["relative_residual"] - .05
                for rows in singles.values()),
            "d1_within_002_of_best_single": circuit[1]["relative_residual"] <= min(
                rows[1]["relative_residual"] for rows in singles.values()) + .02,
            "d0_control_margin": circuit[0]["cosine"] >= max(
                q95, max(rows[0]["cosine"] for rows in wrongs.values())) + .10,
        }
        holds = all(gates.values())
        discovery_checks[name] = {
            "circuit": circuit, "task": task, "continuations": continuations,
            "singles": singles, "wrongs": wrongs, "permutation_q95": q95,
            "best_single": best_single, "strongest_wrong": strongest_wrong,
            "gates": gates, "holds": holds,
        }
        if holds:
            candidates.append({"target": name, "single_donor": best_single, "wrong_control": strongest_wrong})

    confirmation = phases["confirmation"]
    ct = views(confirmation, "target")
    cc = views(confirmation, "consensus")
    cs = views(confirmation, "single")
    cw = views(confirmation, "wrong")
    confirmation_checks = {}
    passers = []
    for i, name in enumerate(confirmation["targets"]):
        windows = {}
        for window, target_c, consensus_c, target_t, consensus_t, single_c, wrong_c in (
            ("half0", ct["circuit_halves"][i, 0], cc["circuit_halves"][i, 0],
             ct["task_halves"][i, 0], cc["task_halves"][i, 0], cs["circuit_halves"][i, 0], cw["circuit_halves"][i, 0]),
            ("half1", ct["circuit_halves"][i, 1], cc["circuit_halves"][i, 1],
             ct["task_halves"][i, 1], cc["task_halves"][i, 1], cs["circuit_halves"][i, 1], cw["circuit_halves"][i, 1]),
            ("pooled", ct["circuit_pooled"][i], cc["circuit_pooled"][i],
             ct["task_pooled"][i], cc["task_pooled"][i], cs["circuit_pooled"][i], cw["circuit_pooled"][i]),
        ):
            circuit = metrics(target_c, consensus_c)
            task = metrics(target_t, consensus_t)
            single_row = metrics(target_c, single_c)
            wrong_row = metrics(target_c, wrong_c)
            continuation_rows = [metrics(target_c[c], consensus_c[c]) for c in range(4)]
            clauses = {
                "absolute_circuit": circuit["cosine"] >= .75 and circuit["relative_residual"] <= .55,
                "task": task["cosine"] >= .70 and task["relative_residual"] <= .65,
                "beats_frozen_single_by_003": circuit["relative_residual"] <= single_row["relative_residual"] - .03,
                "beats_frozen_wrong_by_010_cosine": circuit["cosine"] >= wrong_row["cosine"] + .10,
                "every_continuation_positive": all(row["cosine"] > 0 for row in continuation_rows),
            }
            windows[window] = {
                "circuit": circuit, "task": task, "single": single_row, "wrong": wrong_row,
                "continuations": continuation_rows, "clauses": clauses, "holds": all(clauses.values()),
            }
        holds = all(row["holds"] for row in windows.values())
        confirmation_checks[name] = {"windows": windows, "holds": holds}
        if holds:
            passers.append(name)

    call_counts = {
        "discovery": int(discovery["diagnostics"]["full_model_forwards"]),
        "confirmation": int(confirmation["diagnostics"]["full_model_forwards"]),
    }
    audit_passes = bool(
        candidates == result["discovery"]["candidates"]
        and passers == result["confirmation"]["passing"] == []
        and call_counts == {"discovery": 7688, "confirmation": 1612}
        and sum(call_counts.values()) == result["execution_price"]["full_model_forwards"] == 9300
        and result["execution_price"]["calls_exact"] is True
        and result["pred_a_exact_live_shared_private_instrument"] is True
        and result["pred_b_consensus_beats_every_singleton"] is True
        and result["pred_c_new_document_physical_consensus"] is False
        and result["pred_d_heldout_circuits_and_documents"] is False
        and result["pred_e_sufficient_selectively_removable_shared_state"] is False
        and result["strong_null"] is True
        and "validation" not in phases)
    report = {
        "status": "audit_passed" if audit_passes else "audit_failed",
        "rung": 529,
        "hashes": hashes,
        "recomputed_discovery": discovery_checks,
        "recomputed_candidates": candidates,
        "recomputed_confirmation": confirmation_checks,
        "recomputed_confirmation_passers": passers,
        "call_counts": call_counts,
        "validation_and_selectivity_sealed": "validation" not in phases,
        "audit_passes": audit_passes,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": report["status"], "candidates": candidates, "confirmation_passers": passers,
        "call_counts": call_counts,
        "z7_confirmation_failed_clauses": {
            window: [name for name, held in row["clauses"].items() if not held]
            for window, row in confirmation_checks["Z7"]["windows"].items()
        },
    }, indent=2, sort_keys=True))
    if not audit_passes:
        raise RuntimeError("rung529 terminal audit failed")
    return report


if __name__ == "__main__":
    audit()
