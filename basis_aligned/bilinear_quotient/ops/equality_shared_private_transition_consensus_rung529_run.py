#!/usr/bin/env python3
"""RUNG 529 -- physical leave-one-action-out shared/private transition test.

pred_a: the post-MLP12 state construction and all calls are exact and live
pred_b: at least one consensus beats every single donor in discovery
pred_c: at least one frozen candidate predicts new documents
pred_d: at least one candidate predicts unopened circuit families and documents
pred_e: its consensus is sufficient while its private remainder removes equality

Strong null: any of A--E fails.  Later phases remain fail-closed.
Literal price: 7,688 unconditional and at most 23,396 full-model forwards,
zero backwards, zero fitted values, and zero deployed values saved.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch


REPO = Path("/workspace/tensor_language")
BQ = REPO / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
POLY = REPO / "basis_aligned/polynomial_causal"
for path in (OPS, POLY, BQ, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade  # noqa: E402
import equality_distributed_finite_transition_quotient_rung528_run as r528  # noqa: E402
import equality_shared_private_transition_consensus_rung529_math as qm  # noqa: E402


RUNNER = Path(__file__).resolve()
PREREG = POLY / "EQUALITY_SHARED_PRIVATE_TRANSITION_CONSENSUS_RUNG529_PREREGISTRATION.md"
MATH = OPS / "equality_shared_private_transition_consensus_rung529_math.py"
R528_RUNNER = OPS / "equality_distributed_finite_transition_quotient_rung528_run.py"
R528_RESULT = BQ / "equality_distributed_finite_transition_quotient_rung528_results.json"
R528_BUNDLE = BQ / "equality_distributed_finite_transition_quotient_rung528_bundle.pt"
R528_AUDIT = BQ / "equality_distributed_finite_transition_quotient_rung528_terminal_audit.json"
R528_DIAGNOSIS = BQ / "rung528_leave_one_action_out_consensus_diagnosis_results.json"
OUT = BQ / "equality_shared_private_transition_consensus_rung529_results.json"
BUNDLE = BQ / "equality_shared_private_transition_consensus_rung529_bundle.pt"
SMOKE_OUT = BQ / "equality_shared_private_transition_consensus_rung529_gpu_smoke_results.json"

FROZEN_SHA256 = {
    PREREG: "638a140610d800a9745157fbb1498bbb36d152d3a612ad54a82b9b3ac47c20ea",
    MATH: "77503f27c2838af78a806f69c5d99b276e232eadca2ab4fed8c889b855e15014",
    R528_RUNNER: "69e728bae2b67fcdc30beebbdc0e65981646d6dbfe474743e37d46e22cd89427",
    R528_RESULT: "f931e5fb6f618b002203ce1e870a8ad4442ed3a38a7475809754ab2de91554b6",
    R528_BUNDLE: "c17db82832a76daba23f74e57e75abc258093c6820c79c93a62d8d29b6143d38",
    R528_AUDIT: "a3843265eb15a1fe6771c848843dfafe5703d100933331aab114dbf0e2286f71",
    R528_DIAGNOSIS: "207ff8cfdac919ac4a817564450a6339a2b420d6e91a6f545a9723ed6aded67c",
}

CELLS = r528.CELLS
TASK_CONTEXT_INDICES = r528.TASK_CONTEXT_INDICES
MASK_TYPES = r528.MASK_TYPES
CONTINUATIONS = qm.CONTINUATIONS
CONTINUATION_PATCHES = r528.CONTINUATION_PATCHES
BATCH = 4
DISCOVERY = (0, 248, 124)
CONFIRMATION = (248, 496, 372)
VALIDATION = (500, 1000, 750)
PERMUTATION_SEEDS = tuple(range(529_300, 529_316))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite result: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_dependencies() -> tuple[dict[str, str], tuple]:
    observed = {}
    for path, expected in FROZEN_SHA256.items():
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen dependency changed: {path}: {actual} != {expected}")
        observed[str(path.relative_to(REPO))] = actual
    old = json.loads(R528_RESULT.read_text())
    audit = json.loads(R528_AUDIT.read_text())
    diagnosis = json.loads(R528_DIAGNOSIS.read_text())
    if not (
        old.get("pred_a_exact_live_boundary_instrument") is True
        and old.get("pred_b_at_least_one_discovery_transition_relation") is False
        and old.get("strong_null") is True
        and old.get("execution_price", {}).get("calls_exact") is True
    ):
        raise RuntimeError("rung528 terminal null changed")
    if not (
        audit.get("status") == "audit_passed"
        and audit.get("calls_reconciled") is True
        and audit.get("precontrol_passer_count") == 0
        and diagnosis.get("status") == "complete"
        and diagnosis.get("physical_consensus_insertion_opened") is False
    ):
        raise RuntimeError("rung528 audit or consensus diagnosis changed")
    _old_dependencies, population = r528.validate_dependencies()
    return observed, population


def _pairs_for_phase(
    targets: Sequence[str], candidates: Sequence[Mapping[str, str]] | None,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    if candidates is None:
        singles = tuple((target, donor) for target in qm.ACTIONS for donor in qm.ACTIONS if donor != target)
        wrong = tuple(
            (target, control)
            for target in qm.ACTIONS
            for control in (("W7", "W8") if target in ("N", "P") else (("W8",) if target == "Z7" else ("W7",)))
        )
        return singles, wrong
    by_target = {str(row["target"]): row for row in candidates}
    if tuple(by_target) != tuple(targets):
        raise ValueError("candidate order and target order differ")
    return (
        tuple((target, str(by_target[target]["single_donor"])) for target in targets),
        tuple((target, str(by_target[target]["wrong_control"])) for target in targets),
    )


def _allocate(kind_count: int, documents: int, tags: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(kind_count, 4, documents, len(CELLS), dtype=torch.float64),
        torch.zeros(kind_count, 4, 2, len(MASK_TYPES), tags, dtype=torch.float64),
    )


def _empty_phase(bounds, tags, targets, single_pairs, wrong_pairs):
    documents = bounds[1] - bounds[0]
    data: dict[str, Any] = {
        "bounds": tuple(bounds),
        "tags": tuple(tags),
        "targets": tuple(targets),
        "single_pairs": tuple(single_pairs),
        "wrong_pairs": tuple(wrong_pairs),
        "task_counts": torch.zeros(documents, len(CELLS), dtype=torch.float64),
        "circuit_counts": torch.zeros(2, len(MASK_TYPES), len(tags), dtype=torch.float64),
        "diagnostics": {
            "direct_forwards": 0,
            "absent_forwards": 0,
            "action_boundary_forwards": 0,
            "wrong_boundary_forwards": 0,
            "target_continuation_forwards": 0,
            "consensus_forwards": 0,
            "private_forwards": 0,
            "single_forwards": 0,
            "wrong_consensus_forwards": 0,
            "boundary_captures": 0,
            "boundary_overrides": 0,
            "continuation_write_captures": 0,
            "continuation_write_patches": 0,
            "native_replay_logit_max_abs": 0.0,
            "native_replay_boundary_max_abs": 0.0,
            "maximum_target_reconstruction_boundary_abs": 0.0,
            "factor_reconstruction_max": 0.0,
            "minimum_score_edit_rms": math.inf,
            "minimum_transition_rms": math.inf,
            "minimum_consensus_rms": math.inf,
            "minimum_private_rms": math.inf,
            "minimum_single_rms": math.inf,
            "minimum_wrong_consensus_rms": math.inf,
            "minimum_boundary_override_rms": math.inf,
            "minimum_continuation_patch_rms": math.inf,
        },
    }
    for prefix, count in (
        ("target", len(targets)),
        ("consensus", len(targets)),
        ("private", len(targets)),
        ("single", len(single_pairs)),
        ("wrong", len(wrong_pairs)),
    ):
        data[f"{prefix}_task_sums"], data[f"{prefix}_circuit_sums"] = _allocate(count, documents, len(tags))
    return data


def _update_diagnostics(total: dict, diagnostics: dict, audit: dict) -> None:
    r528._update_run_diagnostics(total, diagnostics, audit)
    if audit["boundary_overrides"]:
        total["minimum_boundary_override_rms"] = min(
            total["minimum_boundary_override_rms"], diagnostics["boundary_override_rms"])


def _record_effects(data, prefix, nll, local, local_masks, circuit_matrix):
    if nll.shape[0] == 0:
        return
    data[f"{prefix}_task_sums"][:, :, local:local + BATCH] = r528._task_sums(nll, local_masks)
    flattened = nll.reshape(nll.shape[0] * 4, -1).double()
    data[f"{prefix}_circuit_sums"] += torch.matmul(flattened, circuit_matrix.T).view(
        nll.shape[0], 4, 2, len(MASK_TYPES), len(data["tags"]))


@torch.no_grad()
def collect_phase(
    model, rows, task_masks, circuit_masks, tags, scales, bounds,
    *, candidates: Sequence[Mapping[str, str]] | None,
):
    targets = qm.ACTIONS if candidates is None else tuple(str(row["target"]) for row in candidates)
    single_pairs, wrong_pairs = _pairs_for_phase(targets, candidates)
    required_wrong = tuple(name for name in ("W7", "W8") if any(pair[1] == name for pair in wrong_pairs))
    data = _empty_phase(bounds, tags, targets, single_pairs, wrong_pairs)
    diagnostics = data["diagnostics"]
    device = next(model.parameters()).device
    lo_doc, hi_doc, _split = bounds

    for start in range(lo_doc, hi_doc, BATCH):
        stop = start + BATCH
        local = start - lo_doc
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        local_masks = {cell: task_masks[cell][start:stop] for cell in CELLS}
        circuit_matrix, circuit_counts = r528._circuit_matrix(circuit_masks, tags, start, stop, bounds)
        data["circuit_counts"] += circuit_counts
        data["task_counts"][local:local + BATCH] = torch.stack(
            [local_masks[cell].sum(1).double() for cell in CELLS], dim=-1)

        direct_logits, direct_state, diag, audit = r528.boundary_forward(model, tokens, direct=True)
        diagnostics["direct_forwards"] += 1
        _update_diagnostics(diagnostics, diag, audit)
        absent_logits, absent_state, diag, audit = r528.boundary_forward(
            model, tokens, action="P", absent=True, scales=scales, capture_writes=("a14", "m17"))
        diagnostics["absent_forwards"] += 1
        _update_diagnostics(diagnostics, diag, audit)
        absent_nll = r528._nll(absent_logits, batch_rows)

        action_runs: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}
        deltas: dict[str, torch.Tensor] = {}
        for action in qm.ACTIONS:
            logits, state, diag, audit = r528.boundary_forward(model, tokens, action=action, scales=scales)
            diagnostics["action_boundary_forwards"] += 1
            _update_diagnostics(diagnostics, diag, audit)
            action_runs[action] = (logits, state)
            delta = state["native_boundary"].float() - absent_state["native_boundary"].float()
            deltas[action] = delta
            diagnostics["minimum_transition_rms"] = min(
                diagnostics["minimum_transition_rms"], float(delta.square().mean().sqrt()))
        diagnostics["native_replay_logit_max_abs"] = max(
            diagnostics["native_replay_logit_max_abs"], r528.maximum_abs(direct_logits, action_runs["N"][0]))
        diagnostics["native_replay_boundary_max_abs"] = max(
            diagnostics["native_replay_boundary_max_abs"],
            r528.maximum_abs(direct_state["native_boundary"], action_runs["N"][1]["native_boundary"]))

        wrong_deltas: dict[str, torch.Tensor] = {}
        for action in required_wrong:
            _logits, state, diag, audit = r528.boundary_forward(model, tokens, action=action, scales=scales)
            diagnostics["wrong_boundary_forwards"] += 1
            _update_diagnostics(diagnostics, diag, audit)
            wrong_deltas[action] = state["native_boundary"].float() - absent_state["native_boundary"].float()

        decomposition = qm.leave_one_out_decomposition(deltas)
        single_states = qm.single_donor_states(deltas)
        wrong_states = qm.wrong_sign_consensus_states(
            deltas, {name: wrong_deltas[name] for name in ("W7", "W8")}) if len(required_wrong) == 2 else None
        if wrong_states is None:
            # The math helper validates both frozen wrong states.  Conditional
            # phases may need only one, so construct only the registered terms.
            aligned = qm.aligned_states(deltas)
            wrong_states = {target: {} for target in qm.ACTIONS}
            for target, control in wrong_pairs:
                source = "Z7" if control == "W7" else "Z8"
                donors = tuple(action for action in qm.ACTIONS if action != target)
                terms = [
                    qm.BETAS[action] * (wrong_deltas[control] if action == source else deltas[action])
                    for action in donors
                ]
                wrong_states[target][control] = torch.stack(terms).mean(0) / qm.BETAS[target]

        for target in targets:
            rounded = r528.scaled_boundary(
                absent_state["native_boundary"], decomposition[target]["reconstruction"], 1.0)
            diagnostics["maximum_target_reconstruction_boundary_abs"] = max(
                diagnostics["maximum_target_reconstruction_boundary_abs"],
                r528.maximum_abs(rounded, action_runs[target][1]["native_boundary"]))
            for key, diagnostic_key in (
                ("consensus", "minimum_consensus_rms"), ("private", "minimum_private_rms")):
                diagnostics[diagnostic_key] = min(
                    diagnostics[diagnostic_key],
                    float(decomposition[target][key].float().square().mean().sqrt()))
        for target, donor in single_pairs:
            diagnostics["minimum_single_rms"] = min(
                diagnostics["minimum_single_rms"], float(single_states[target][donor].square().mean().sqrt()))
        for target, control in wrong_pairs:
            diagnostics["minimum_wrong_consensus_rms"] = min(
                diagnostics["minimum_wrong_consensus_rms"],
                float(wrong_states[target][control].square().mean().sqrt()))

        target_nll = []
        for target in targets:
            rows_for_target = []
            for continuation in CONTINUATIONS:
                if continuation == "native":
                    logits = action_runs[target][0]
                else:
                    sites = CONTINUATION_PATCHES[continuation]
                    logits, _state, diag, audit = r528.boundary_forward(
                        model, tokens, action="P", absent=True, scales=scales,
                        boundary_override=r528.scaled_boundary(
                            absent_state["native_boundary"], deltas[target], 1.0),
                        patch_writes={site: absent_state[site] for site in sites})
                    diagnostics["target_continuation_forwards"] += 1
                    _update_diagnostics(diagnostics, diag, audit)
                rows_for_target.append(r528._nll(logits, batch_rows) - absent_nll)
            target_nll.append(torch.stack(rows_for_target))
        _record_effects(data, "target", torch.stack(target_nll), local, local_masks, circuit_matrix)

        def run_edits(prefix: str, states: Sequence[torch.Tensor]) -> None:
            all_rows = []
            for edit in states:
                continuation_rows = []
                for continuation in CONTINUATIONS:
                    sites = CONTINUATION_PATCHES[continuation]
                    logits, _state, diag, audit = r528.boundary_forward(
                        model, tokens, action="P", absent=True, scales=scales,
                        boundary_override=r528.scaled_boundary(absent_state["native_boundary"], edit, 1.0),
                        patch_writes={site: absent_state[site] for site in sites})
                    diagnostics[f"{prefix}_forwards"] += 1
                    _update_diagnostics(diagnostics, diag, audit)
                    continuation_rows.append(r528._nll(logits, batch_rows) - absent_nll)
                all_rows.append(torch.stack(continuation_rows))
            _record_effects(data, prefix.removesuffix("_consensus"), torch.stack(all_rows),
                            local, local_masks, circuit_matrix)

        run_edits("consensus", [decomposition[target]["consensus"] for target in targets])
        run_edits("private", [decomposition[target]["private"] for target in targets])
        run_edits("single", [single_states[target][donor] for target, donor in single_pairs])
        run_edits("wrong_consensus", [wrong_states[target][control] for target, control in wrong_pairs])

    batches = (hi_doc - lo_doc) // BATCH
    other_states = len(targets) * 2 + len(single_pairs) + len(wrong_pairs)
    expected = {
        "direct_forwards": batches,
        "absent_forwards": batches,
        "action_boundary_forwards": 4 * batches,
        "wrong_boundary_forwards": len(required_wrong) * batches,
        "target_continuation_forwards": 3 * len(targets) * batches,
        "consensus_forwards": 4 * len(targets) * batches,
        "private_forwards": 4 * len(targets) * batches,
        "single_forwards": 4 * len(single_pairs) * batches,
        "wrong_consensus_forwards": 4 * len(wrong_pairs) * batches,
    }
    diagnostics["expected_forwards"] = expected
    diagnostics["forwards_exact"] = all(diagnostics[key] == value for key, value in expected.items())
    diagnostics["full_model_forwards"] = sum(expected.values())
    diagnostics["boundary_calls_exact"] = bool(
        diagnostics["boundary_captures"] == diagnostics["full_model_forwards"]
        and diagnostics["boundary_overrides"] == batches * (3 * len(targets) + 4 * other_states)
        and diagnostics["continuation_write_captures"] == 2 * batches
        and diagnostics["continuation_write_patches"] == 4 * batches * (len(targets) + other_states))
    local_split = bounds[2] - bounds[0]
    task_half_support = torch.stack((
        data["task_counts"][:local_split].sum(0), data["task_counts"][local_split:].sum(0)))
    diagnostics["supports_positive"] = bool(
        (task_half_support > 0).all() and (data["circuit_counts"] > 0).all())
    return data


def phase_views(data, prefix: str):
    return r528._effect_views(
        data[f"{prefix}_task_sums"], data[f"{prefix}_circuit_sums"],
        data["task_counts"], data["circuit_counts"])


def instrument_holds(data) -> bool:
    d = data["diagnostics"]
    return bool(
        d["forwards_exact"] and d["boundary_calls_exact"] and d["supports_positive"]
        and d["factor_reconstruction_max"] <= 1e-10
        and d["minimum_score_edit_rms"] > 0 and d["minimum_transition_rms"] > 0
        and d["minimum_consensus_rms"] > 0 and d["minimum_private_rms"] > 0
        and d["minimum_single_rms"] > 0 and d["minimum_wrong_consensus_rms"] > 0
        and d["minimum_boundary_override_rms"] > 0 and d["minimum_continuation_patch_rms"] > 0
        and d["native_replay_logit_max_abs"] == 0.0
        and d["native_replay_boundary_max_abs"] == 0.0
        and d["maximum_target_reconstruction_boundary_abs"] == 0.0)


def discover_candidates(data) -> tuple[list[dict[str, str]], dict[str, Any]]:
    target = phase_views(data, "target")
    consensus = phase_views(data, "consensus")
    single = phase_views(data, "single")
    wrong = phase_views(data, "wrong")
    candidates = []
    checks = {}
    for target_index, target_name in enumerate(data["targets"]):
        single_indices = [i for i, pair in enumerate(data["single_pairs"]) if pair[0] == target_name]
        wrong_indices = [i for i, pair in enumerate(data["wrong_pairs"]) if pair[0] == target_name]
        permutation_cosines = []
        for seed in PERMUTATION_SEEDS:
            generator = torch.Generator().manual_seed(seed)
            scrambled = consensus["circuit_halves"][target_index].clone()
            for continuation in range(4):
                order = torch.randperm(scrambled.shape[-1], generator=generator)
                scrambled[:, continuation] = scrambled[:, continuation, order]
            permutation_cosines.append(qm.compare_effects(
                target["circuit_halves"][target_index], scrambled)[0]["cosine"])
        score = qm.score_discovery_target(
            target["circuit_halves"][target_index], target["task_halves"][target_index],
            consensus["circuit_halves"][target_index], consensus["task_halves"][target_index],
            {data["single_pairs"][i][1]: single["circuit_halves"][i] for i in single_indices},
            {data["wrong_pairs"][i][1]: wrong["circuit_halves"][i] for i in wrong_indices},
            permutation_cosines)
        score["single_pairs"] = [data["single_pairs"][i] for i in single_indices]
        score["wrong_pairs"] = [data["wrong_pairs"][i] for i in wrong_indices]
        checks[target_name] = score
        if score["passes_response_gates"]:
            best_single = min(
                score["single"], key=lambda donor: score["single"][donor][0]["relative_residual"])
            strongest_wrong = max(
                score["wrong"], key=lambda control: score["wrong"][control][0]["cosine"])
            candidates.append({
                "target": target_name,
                "single_donor": best_single,
                "wrong_control": strongest_wrong,
            })
    return candidates, checks


def _comparison(target, observed):
    return qm.compare_effects(target, observed)


def score_repeat(data, *, circuit_cosine: float, circuit_error: float):
    target = phase_views(data, "target")
    consensus = phase_views(data, "consensus")
    single = phase_views(data, "single")
    wrong = phase_views(data, "wrong")
    passers = []
    checks = {}
    for i, target_name in enumerate(data["targets"]):
        windows = {}
        holds = True
        for window, target_circuit, observed_circuit, target_task, observed_task, single_circuit, wrong_circuit in (
            ("half0", target["circuit_halves"][i, 0], consensus["circuit_halves"][i, 0],
             target["task_halves"][i, 0], consensus["task_halves"][i, 0],
             single["circuit_halves"][i, 0], wrong["circuit_halves"][i, 0]),
            ("half1", target["circuit_halves"][i, 1], consensus["circuit_halves"][i, 1],
             target["task_halves"][i, 1], consensus["task_halves"][i, 1],
             single["circuit_halves"][i, 1], wrong["circuit_halves"][i, 1]),
            ("pooled", target["circuit_pooled"][i], consensus["circuit_pooled"][i],
             target["task_pooled"][i], consensus["task_pooled"][i],
             single["circuit_pooled"][i], wrong["circuit_pooled"][i]),
        ):
            circuit = r528._one_comparison(target_circuit, observed_circuit, 1.0, circuit_cosine, circuit_error)
            task = r528._one_comparison(target_task, observed_task, 1.0, .70, .65)
            single_row = qm.r528.relation_metrics(target_circuit, single_circuit, 1.0)
            wrong_row = qm.r528.relation_metrics(target_circuit, wrong_circuit, 1.0)
            continuations = [
                qm.r528.relation_metrics(target_circuit[c], observed_circuit[c], 1.0) for c in range(4)]
            window_holds = bool(
                circuit["holds"] and task["holds"]
                and circuit["relative_residual"] <= single_row["relative_residual"] - .03
                and circuit["cosine"] >= wrong_row["cosine"] + .10
                and all(row["cosine"] > 0 for row in continuations))
            windows[window] = {
                "circuit": circuit,
                "task": task,
                "frozen_single": single_row,
                "frozen_wrong": wrong_row,
                "continuations": continuations,
                "holds": window_holds,
            }
            holds &= window_holds
        checks[target_name] = {"windows": windows, "holds": bool(holds)}
        if holds:
            passers.append(target_name)
    return passers, checks


def score_selectivity(confirmation, validation, candidates):
    all_index = CELLS.index("all_positive")
    off_index = CELLS.index("off_target")
    checks = {}
    passers = []
    by_name = {str(row["target"]): row for row in candidates}
    for target_name in by_name:
        holds = True
        phases = {}
        for phase_name, data in (("confirmation", confirmation), ("validation", validation)):
            index = data["targets"].index(target_name)
            target = phase_views(data, "target")["task_halves_full"]
            private = phase_views(data, "private")["task_halves_full"]
            windows = []
            for half in range(2):
                full_all = float(target[index, half, 0, all_index])
                full_off = float(target[index, half, 0, off_index])
                private_all = float(private[index, half, 0, all_index])
                private_off = float(private[index, half, 0, off_index])
                row_holds = bool(
                    abs(full_all) >= .002
                    and abs(private_all) <= .25 * abs(full_all)
                    and abs(private_off - full_off) <= .001)
                windows.append({
                    "full_all_copy_effect_nat": full_all,
                    "private_all_copy_effect_nat": private_all,
                    "private_retained_fraction": abs(private_all) / max(abs(full_all), 1e-30),
                    "full_off_target_effect_nat": full_off,
                    "private_off_target_effect_nat": private_off,
                    "off_target_absolute_difference_nat": abs(private_off - full_off),
                    "holds": row_holds,
                })
                holds &= row_holds
            phases[phase_name] = windows
        checks[target_name] = {"phases": phases, "holds": bool(holds)}
        if holds:
            passers.append(target_name)
    return passers, checks


def _bundle_phase(data):
    return dict(data)


@torch.no_grad()
def gpu_smoke(output_path: Path = SMOKE_OUT) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite smoke: {output_path}")
    dependencies, population = validate_dependencies()
    rows, _task_masks, _circuit_masks, scales, _discovery_tags, _validation_tags, metadata = population
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    batch_rows = rows[:BATCH]
    tokens = batch_rows[:, :-1].to(next(model.parameters()).device)
    direct_logits, direct_state, _diag, _audit = r528.boundary_forward(model, tokens, direct=True)
    absent_logits, absent_state, _diag, _audit = r528.boundary_forward(
        model, tokens, action="P", absent=True, scales=scales, capture_writes=("a14", "m17"))
    action_runs = {}
    deltas = {}
    for action in qm.ACTIONS:
        logits, state, _diag, _audit = r528.boundary_forward(model, tokens, action=action, scales=scales)
        action_runs[action] = (logits, state)
        deltas[action] = state["native_boundary"].float() - absent_state["native_boundary"].float()
    wrong_deltas = {}
    for action in ("W7", "W8"):
        _logits, state, _diag, _audit = r528.boundary_forward(model, tokens, action=action, scales=scales)
        wrong_deltas[action] = state["native_boundary"].float() - absent_state["native_boundary"].float()
    decomposition = qm.leave_one_out_decomposition(deltas)
    singles = qm.single_donor_states(deltas)
    wrong = qm.wrong_sign_consensus_states(deltas, wrong_deltas)
    all_edits = (
        [decomposition[target][kind] for target in qm.ACTIONS for kind in ("consensus", "private")]
        + [singles[target][donor] for target in qm.ACTIONS for donor in qm.ACTIONS if donor != target]
        + [wrong[target][control] for target in qm.ACTIONS for control in wrong[target]]
    )
    max_reconstruction = 0.0
    for target in qm.ACTIONS:
        reconstructed = r528.scaled_boundary(
            absent_state["native_boundary"], decomposition[target]["reconstruction"], 1.0)
        max_reconstruction = max(max_reconstruction, r528.maximum_abs(
            reconstructed, action_runs[target][1]["native_boundary"]))
    minimum_edit_rms = min(float(edit.square().mean().sqrt()) for edit in all_edits)
    maximum_override_rms = 0.0
    # Exercise every constructed state once without retaining CE or circuit data.
    for edit in all_edits:
        _logits, _state, diag, _audit = r528.boundary_forward(
            model, tokens, action="P", absent=True, scales=scales,
            boundary_override=r528.scaled_boundary(absent_state["native_boundary"], edit, 1.0))
        maximum_override_rms = max(maximum_override_rms, diag["boundary_override_rms"])
    minimum_patch_rms = math.inf
    for continuation in CONTINUATIONS[1:]:
        sites = CONTINUATION_PATCHES[continuation]
        _logits, _state, diag, _audit = r528.boundary_forward(
            model, tokens, action="P", absent=True, scales=scales,
            boundary_override=r528.scaled_boundary(
                absent_state["native_boundary"], decomposition["Z7"]["consensus"], 1.0),
            patch_writes={site: absent_state[site] for site in sites})
        minimum_patch_rms = min(minimum_patch_rms, diag["continuation_patch_rms_max"])
    total_forwards = 1 + 1 + 4 + 2 + len(all_edits) + 3
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and r528.maximum_abs(direct_logits, action_runs["N"][0]) == 0.0
        and r528.maximum_abs(direct_state["native_boundary"], action_runs["N"][1]["native_boundary"]) == 0.0
        and max_reconstruction == 0.0 and minimum_edit_rms > 0
        and maximum_override_rms > 0 and minimum_patch_rms > 0
        and total_forwards == 37)
    result = {
        "status": "complete",
        "rung": 529,
        "claim_level": "managed_gpu_instrument_smoke_only",
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "dependency_sha256": dependencies,
        "runner_sha256": file_sha256(RUNNER),
        "input_identity": metadata,
        "pred_a_smoke_exact_live_all_state_constructions": pred_a,
        "strong_null": not pred_a,
        "scientific_task_or_circuit_effects_retained": False,
        "diagnostics": {
            "all_constructed_edit_states": len(all_edits),
            "maximum_target_reconstruction_boundary_abs": max_reconstruction,
            "minimum_edit_state_rms": minimum_edit_rms,
            "maximum_boundary_override_rms": maximum_override_rms,
            "minimum_continuation_patch_rms": minimum_patch_rms,
            "full_model_forwards": total_forwards,
            "expected_full_model_forwards": 37,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
    }
    atomic_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@torch.no_grad()
def run_full(required_smoke_sha256: str, launcher: Path) -> dict[str, Any]:
    if OUT.exists() or BUNDLE.exists():
        raise FileExistsError("refusing to overwrite rung529 full namespace")
    if file_sha256(SMOKE_OUT) != required_smoke_sha256:
        raise RuntimeError("managed smoke hash changed")
    smoke = json.loads(SMOKE_OUT.read_text())
    if not (
        smoke.get("pred_a_smoke_exact_live_all_state_constructions") is True
        and smoke.get("strong_null") is False
        and smoke.get("scientific_task_or_circuit_effects_retained") is False
    ):
        raise RuntimeError("managed smoke authority is absent or failed")
    dependencies, population = validate_dependencies()
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = population
    wall_started = time.time()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()

    phases = {}
    phases["discovery"] = collect_phase(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, DISCOVERY, candidates=None)
    pred_a = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256 and instrument_holds(phases["discovery"]))
    candidates, discovery_checks = discover_candidates(phases["discovery"])
    pred_b = bool(pred_a and 1 <= len(candidates) <= 4)
    confirmed = []
    confirmation_checks = {}
    validated = []
    validation_checks = {}
    selective = []
    selectivity_checks = {}

    if pred_b:
        phases["confirmation"] = collect_phase(
            model, rows, task_masks, circuit_masks, discovery_tags, scales, CONFIRMATION,
            candidates=candidates)
        confirmed, confirmation_checks = score_repeat(
            phases["confirmation"], circuit_cosine=.75, circuit_error=.55)
    pred_c = bool(pred_b and confirmed)

    confirmed_candidates = [row for row in candidates if row["target"] in confirmed]
    if pred_c:
        phases["validation"] = collect_phase(
            model, rows, task_masks, circuit_masks, validation_tags, scales, VALIDATION,
            candidates=confirmed_candidates)
        validated, validation_checks = score_repeat(
            phases["validation"], circuit_cosine=.70, circuit_error=.60)
    pred_d = bool(pred_c and validated)

    validated_candidates = [row for row in confirmed_candidates if row["target"] in validated]
    if pred_d:
        selective, selectivity_checks = score_selectivity(
            phases["confirmation"], phases["validation"], validated_candidates)
    pred_e = bool(pred_d and selective)

    total_forwards = sum(phase["diagnostics"]["full_model_forwards"] for phase in phases.values())
    expected_forwards = 7688
    if pred_b:
        unique_wrong = len({row["wrong_control"] for row in candidates})
        expected_forwards += 62 * (6 + unique_wrong + 19 * len(candidates))
    if pred_c:
        unique_wrong = len({row["wrong_control"] for row in confirmed_candidates})
        expected_forwards += 125 * (6 + unique_wrong + 19 * len(confirmed_candidates))
    calls_exact = total_forwards == expected_forwards
    pred_a = bool(pred_a and calls_exact and all(instrument_holds(phase) for phase in phases.values()))
    if not pred_a:
        pred_b = pred_c = pred_d = pred_e = False
        candidates = []
        confirmed = validated = selective = []
    strong_null = not all((pred_a, pred_b, pred_c, pred_d, pred_e))

    bundle = {
        "schema": "equality_shared_private_transition_consensus_rung529_sufficient_statistics_v1",
        "phases": {name: _bundle_phase(value) for name, value in phases.items()},
        "raw_tokens_logits_boundaries_or_hidden_states_included": False,
    }
    torch.save(bundle, BUNDLE)
    torch.cuda.synchronize()
    runtime_s = time.time() - wall_started
    result = {
        "status": "complete",
        "rung": 529,
        "claim_level": "heldout_shared_private_causal_state_decomposition_not_internal_minimality_or_compression",
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "dependency_sha256": dependencies,
        "runner_sha256": file_sha256(RUNNER),
        "launcher_sha256": file_sha256(launcher),
        "managed_smoke_sha256": file_sha256(SMOKE_OUT),
        "input_identity": metadata,
        "continuations": list(CONTINUATIONS),
        "frozen_betas": qm.BETAS,
        "discovery": {"candidates": candidates, "checks": discovery_checks},
        "confirmation": {"opened": "confirmation" in phases, "passing": confirmed, "checks": confirmation_checks},
        "validation": {"opened": "validation" in phases, "passing": validated, "checks": validation_checks},
        "selectivity": {"passing": selective, "checks": selectivity_checks},
        "phase_diagnostics": {name: value["diagnostics"] for name, value in phases.items()},
        "pred_a_exact_live_shared_private_instrument": pred_a,
        "pred_b_consensus_beats_every_singleton": pred_b,
        "pred_c_new_document_physical_consensus": pred_c,
        "pred_d_heldout_circuits_and_documents": pred_d,
        "pred_e_sufficient_selectively_removable_shared_state": pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {
            "path": str(BUNDLE), "sha256": file_sha256(BUNDLE), "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "full_model_forwards": total_forwards,
            "expected_full_model_forwards": expected_forwards,
            "maximum_registered_full_model_forwards": 23396,
            "calls_exact": calls_exact,
            "model_backwards": 0,
            "fitted_values": 0,
            "deployed_values_added": 0,
            "deployed_values_removed": 0,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "runtime_s": runtime_s,
        "next_step": (
            "split_validated_consensus_into_exact_attention_and_MLP_interactions"
            if not strong_null else
            "repair_instrument_only" if not pred_a else
            "close_post_MLP12_leave_one_action_out_consensus_route" if not pred_b else
            "close_consensus_as_discovery_specific" if not pred_c else
            "close_consensus_as_known_circuit_specific" if not pred_d else
            "retain_predictive_consensus_without_selective_removal_claim"
        ),
    }
    atomic_json(OUT, result)
    print(json.dumps({
        "status": result["status"], "rung": 529,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "candidates": candidates, "confirmed": confirmed, "validated": validated, "selective": selective,
        "execution_price": result["execution_price"], "runtime_s": runtime_s,
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))
    return result


def dry_run() -> dict[str, Any]:
    dependencies, population = validate_dependencies()
    rows, _task, _circuit, _scales, discovery_tags, validation_tags, _metadata = population
    planted = qm.planted_suite()
    discovery_singles, discovery_wrong = _pairs_for_phase(qm.ACTIONS, None)
    assert tuple(rows.shape) == (1000, 257)
    assert len(discovery_tags) == 32 and len(validation_tags) == 30
    assert len(discovery_singles) == 12 and len(discovery_wrong) == 6
    assert tuple(CONTINUATION_PATCHES) == CONTINUATIONS
    assert 62 * 124 == 7688
    assert 7688 + 62 * (8 + 19 * 4) + 125 * (8 + 19 * 4) == 23396
    assert planted["passes"]
    return {
        "status": "dry_run_passed",
        "rung": 529,
        "model_loaded": False,
        "outcomes_opened": False,
        "dependencies": dependencies,
        "discovery_documents": list(DISCOVERY),
        "confirmation_documents": list(CONFIRMATION),
        "validation_documents": list(VALIDATION),
        "discovery_circuits": len(discovery_tags),
        "validation_circuits": len(validation_tags),
        "all_targets": list(qm.ACTIONS),
        "single_donor_substitutions": len(discovery_singles),
        "wrong_sign_consensuses": len(discovery_wrong),
        "unconditional_discovery_forwards": 7688,
        "maximum_conditional_forwards": 23396,
        "planted_suite_passes": planted["passes"],
    }
