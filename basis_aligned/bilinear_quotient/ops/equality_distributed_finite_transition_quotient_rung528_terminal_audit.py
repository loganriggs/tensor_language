#!/usr/bin/env python3
"""Independent terminal audit for rung 528's discovery stop."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


REPO = Path("/workspace/tensor_language")
BQ = REPO / "basis_aligned/bilinear_quotient"
POLY = REPO / "basis_aligned/polynomial_causal"
RESULT = BQ / "equality_distributed_finite_transition_quotient_rung528_results.json"
BUNDLE = BQ / "equality_distributed_finite_transition_quotient_rung528_bundle.pt"
RUNNER = BQ / "ops/equality_distributed_finite_transition_quotient_rung528_run.py"
PREREG = POLY / "EQUALITY_DISTRIBUTED_FINITE_TRANSITION_QUOTIENT_RUNG528_PREREGISTRATION.md"
SMOKE = BQ / "equality_distributed_finite_transition_quotient_rung528_gpu_smoke_v2_results.json"
OUT = BQ / "equality_distributed_finite_transition_quotient_rung528_terminal_audit.json"
EXPECTED = {
    RESULT: "f931e5fb6f618b002203ce1e870a8ad4442ed3a38a7475809754ab2de91554b6",
    BUNDLE: "c17db82832a76daba23f74e57e75abc258093c6820c79c93a62d8d29b6143d38",
    RUNNER: "69e728bae2b67fcdc30beebbdc0e65981646d6dbfe474743e37d46e22cd89427",
    PREREG: "8e8bdb6af3f0ede2a86a07fa75f86bcefc58e6d8c9214169d5bc8de4f759ad77",
    SMOKE: "436d98a2c5f66fc8fdedf1143d2cd4d145e73134bd076708085614262cd83374",
}
SOURCES = ("N", "P", "Z7", "Z8")
WRONG = ("W7", "W8")
CONTEXT_INDICES = (1, 2, 3, 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(target: torch.Tensor, source: torch.Tensor, beta: float | None = None):
    target = target.double().reshape(-1)
    source = source.double().reshape(-1)
    if beta is None:
        beta = float(torch.dot(source, target) / torch.dot(source, source).clamp_min(1e-300))
    cosine = float(torch.dot(target, source) / (
        torch.linalg.vector_norm(target) * torch.linalg.vector_norm(source)).clamp_min(1e-300))
    residual = float(torch.linalg.vector_norm(target - beta * source)
                     / torch.linalg.vector_norm(target).clamp_min(1e-300))
    return {"beta": beta, "cosine": cosine, "relative_residual": residual}


def views(task_sums, circuit_sums, task_counts, circuit_counts):
    split = task_counts.shape[0] // 2
    tasks = []
    for lo, hi in ((0, split), (split, task_counts.shape[0])):
        tasks.append(task_sums[..., lo:hi, :].sum(-2)
                     / task_counts[lo:hi].sum(0).clamp_min(1))
    task_halves = torch.stack(tasks, dim=-3)[..., list(CONTEXT_INDICES)]
    circuit_means = circuit_sums / circuit_counts.clamp_min(1)
    circuit_halves = (
        circuit_means[..., 0, :] - circuit_means[..., 1, :]).movedim(-2, -3)
    return task_halves, circuit_halves


def audit():
    hashes = {str(path.relative_to(REPO)): sha256(path) for path in EXPECTED}
    if any(hashes[str(path.relative_to(REPO))] != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("frozen rung528 artifact changed")
    result = json.loads(RESULT.read_text())
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=True)
    discovery = bundle["phases"]["discovery"]
    if set(bundle["phases"]) != {"discovery"}:
        raise RuntimeError("a sealed conditional phase appears in bundle")
    unit_task, unit_circuit = views(
        discovery["unit_task_sums"], discovery["unit_circuit_sums"],
        discovery["task_counts"], discovery["circuit_counts"])
    wrong_task, wrong_circuit = views(
        discovery["wrong_task_sums"], discovery["wrong_circuit_sums"],
        discovery["task_counts"], discovery["circuit_counts"])
    recomputed = {}
    for source_index, source in enumerate(SOURCES[1:], 1):
        beta = metrics(unit_circuit[0, 0], unit_circuit[source_index, 0])["beta"]
        circuit = [metrics(unit_circuit[0, half], unit_circuit[source_index, half], beta)
                   for half in range(2)]
        task = [metrics(unit_task[0, half], unit_task[source_index, half], beta)
                for half in range(2)]
        per_continuation = [[
            metrics(unit_circuit[0, half, continuation],
                    unit_circuit[source_index, half, continuation], beta)
            for continuation in range(4)] for half in range(2)]
        material = bool(
            (unit_circuit[[0, source_index]].square().mean(-1).sqrt() >= 5e-4).all()
            and (unit_task[[0, source_index]].reshape(2, 2, -1).norm(dim=-1) >= 2.5e-4).all())
        passes = bool(
            material and .25 <= beta <= 4
            and circuit[0]["cosine"] >= .90 and circuit[0]["relative_residual"] <= .35
            and circuit[1]["cosine"] >= .80 and circuit[1]["relative_residual"] <= .50
            and task[0]["cosine"] >= .70 and task[0]["relative_residual"] <= .65
            and task[1]["cosine"] >= .70 and task[1]["relative_residual"] <= .65
            and all(row[continuation]["cosine"] >= (.65 if half == 0 else .55)
                    for half, row in enumerate(per_continuation) for continuation in range(4)))
        recomputed[source] = {
            "beta": beta,
            "material": material,
            "D0_circuit": circuit[0],
            "D1_circuit": circuit[1],
            "D0_task": task[0],
            "D1_task": task[1],
            "passes_before_controls": passes,
        }
        recorded = result["discovery"]["checks"][source]
        for key in ("beta",):
            if not math.isclose(recomputed[source][key], recorded[key], rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(f"{source} scalar differs")
        for window in ("D0_circuit", "D1_circuit", "D0_task", "D1_task"):
            recorded_window = recorded["circuit"][0 if window == "D0_circuit" else 1] \
                if "circuit" in window else recorded["task"][0 if window == "D0_task" else 1]
            for field in ("cosine", "relative_residual"):
                if not math.isclose(
                    recomputed[source][window][field], recorded_window[field], rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError(f"{source} {window} {field} differs")
    if any(row["passes_before_controls"] for row in recomputed.values()):
        raise RuntimeError("auditor unexpectedly finds a discovery passer")
    diagnostics = result["phase_diagnostics"]["discovery"]
    status = bool(
        result.get("pred_a_exact_live_boundary_instrument") is True
        and result.get("pred_b_at_least_one_discovery_transition_relation") is False
        and result.get("strong_null") is True
        and result["discovery"]["candidates"] == []
        and result["physical_discovery"]["opened"] is False
        and result["confirmation"]["opened"] is False
        and result["validation"]["opened"] is False
        and diagnostics["full_model_forwards"] == 1984
        and diagnostics["forwards_exact"] is True
        and diagnostics["boundary_calls_exact"] is True
        and bundle["raw_tokens_logits_boundaries_or_hidden_states_included"] is False)
    if not status:
        raise RuntimeError("terminal route audit failed")
    return {
        "status": "audit_passed",
        "rung": 528,
        "hashes": hashes,
        "recomputed_pairs": recomputed,
        "material_pair_count": sum(row["material"] for row in recomputed.values()),
        "precontrol_passer_count": sum(row["passes_before_controls"] for row in recomputed.values()),
        "physical_confirmation_validation_sealed": True,
        "calls_reconciled": True,
        "wrong_control_shapes": {
            "task": list(wrong_task.shape), "circuit": list(wrong_circuit.shape)},
    }


if __name__ == "__main__":
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite audit: {OUT}")
    report = audit()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
