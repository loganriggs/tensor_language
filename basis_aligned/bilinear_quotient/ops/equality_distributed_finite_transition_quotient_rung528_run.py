#!/usr/bin/env python3
"""RUNG 528 -- continuation-defined equality finite-transition quotient.

pred_a: raw post-MLP12 capture/insertion is exact, live, and call-accounted
pred_b: one or more correct-gauge transitions share a discovery response
pred_c: a discovery relation passes physical insertion and new documents
pred_d: a fixed relation predicts 30 unopened circuits and document halves
pred_e: at least one validated relation is task-selective

Strong null: any of A--E fails.  Later phases remain fail-closed.
Literal price: at most 11,485 forwards, zero backwards, zero deployed values.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


REPO = Path("/workspace/tensor_language")
BQ = REPO / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
POLY = REPO / "basis_aligned/polynomial_causal"
for path in (OPS, POLY, BQ, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade  # noqa: E402
import equality_distributed_finite_transition_quotient_rung528_math as qm  # noqa: E402
import equality_score_gauged_downstream_program_rung505 as r505  # noqa: E402
import natural_action_conditioned_later_write_state_atlas_rung506 as r506  # noqa: E402


RUNNER = Path(__file__).resolve()
PREREG = POLY / "EQUALITY_DISTRIBUTED_FINITE_TRANSITION_QUOTIENT_RUNG528_PREREGISTRATION.md"
MATH = OPS / "equality_distributed_finite_transition_quotient_rung528_math.py"
R505_SOURCE = OPS / "equality_score_gauged_downstream_program_rung505.py"
R505_RESULT = BQ / "equality_score_gauged_downstream_program_rung505_results.json"
R506_SOURCE = OPS / "natural_action_conditioned_later_write_state_atlas_rung506.py"
R506_RESULT = BQ / "natural_action_conditioned_later_write_state_atlas_rung506_results.json"
R501_SOURCE = OPS / "equality_score_directed_action_graph_rung501.py"
R501_RESULT = BQ / "equality_score_directed_action_graph_rung501_results.json"
R510_SOURCE = OPS / "mlp10_observable_predictive_state_quotient_rung510.py"
R510_RESULT = BQ / "mlp10_observable_predictive_state_quotient_rung510_results.json"
OUT = BQ / "equality_distributed_finite_transition_quotient_rung528_results.json"
SMOKE_OUT = BQ / "equality_distributed_finite_transition_quotient_rung528_gpu_smoke_results.json"
SMOKE_V2_OUT = BQ / "equality_distributed_finite_transition_quotient_rung528_gpu_smoke_v2_results.json"

FROZEN_SHA256 = {
    PREREG: "8e8bdb6af3f0ede2a86a07fa75f86bcefc58e6d8c9214169d5bc8de4f759ad77",
    MATH: "1363371d5df5a8e8e14682907a172ccb65b39df7c5c5e9ffb3ce0dfe72ae5728",
    R505_SOURCE: "0c5f6679ec40cb02bd6af1e28b0b41ca2ad7967fd4b6c9d73a4f388153f3e4de",
    R505_RESULT: "3720a2feb24fc5ec4554d858a00a576a1fcd44f0e789d2b728e66483d7d8d1a1",
    R506_SOURCE: "9a17e28312a0e7214e5fc587123e3267e2650b382f3a40daf12ad1a380b1d004",
    R506_RESULT: "f86e5f0303ab0616ea14e3141fd09886ca54d326e8d83ea6c8c13a62f66db75e",
    R501_SOURCE: "97f3946f558f3d61fc952a9b6ddc7c334b51ccc0ccfe5f02c6ecced417f1e077",
    R501_RESULT: "b17a9b274e4c61e0b4a3fc68d8ce84ec6f8e76f257c3d898e6a6990492301c4f",
    R510_SOURCE: "7901aa5d9c7c39bf5666e0f081bfe08047f23c73eec08b12508c601def7b967a",
    R510_RESULT: "16d100e7b92152fc70939b000934699882605c30c513c570f6c519b80f943177",
}

D = 1152
TOKENS = 256
BOUNDARY_SITE = 12
CONTINUATION_PATCHES = {
    "native": (),
    "without_A14": ("a14",),
    "without_M17": ("m17",),
    "without_both": ("a14", "m17"),
}
SOURCES = r505.SOURCES
WRONG_SIGNS = r505.WRONG_SIGNS
CELLS = r505.CELLS
TASK_CONTEXT_INDICES = tuple(CELLS.index(name) for name in r505.CONTEXT_CELLS)
MASK_TYPES = ("member", "slice_control")
BATCH = 4
DISCOVERY = (0, 248, 124)
PHYSICAL_DISCOVERY = (124, 248, 186)
CONFIRMATION = (248, 496, 372)
VALIDATION = (500, 1000, 750)
CANDIDATE_SOURCES = ("P", "Z7", "Z8")
PERMUTATION_SEEDS = tuple(range(528_300, 528_316))
BUNDLE = BQ / "equality_distributed_finite_transition_quotient_rung528_bundle.pt"
SMOKE_V2_SHA256 = "436d98a2c5f66fc8fdedf1143d2cd4d145e73134bd076708085614262cd83374"


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
    r506_result = json.loads(R506_RESULT.read_text())
    r510_result = json.loads(R510_RESULT.read_text())
    if not (
        r506_result.get("pred_a_exact_live_conditional_instrument") is True
        and r506_result.get("pred_b_score_actions_recalibrate_new_documents") is True
        and r506_result.get("pred_c_at_least_one_whole_write_edge_confirms") is False
        and r506_result.get("strong_null") is True
        and r506_result.get("next_step") == "split_fixed_writes_into_exact_attention_or_bilinear_terms"
    ):
        raise RuntimeError("rung506 whole-write null route changed")
    if not (
        r510_result.get("pred_a_exact_live_singleton_and_substitution_instrument") is True
        and r510_result.get("pred_b_one_to_sixteen_discovery_equivalence_pairs") is False
        and r510_result.get("strong_null") is True
    ):
        raise RuntimeError("rung510 exact-term null route changed")
    population = r506.validate_inputs()
    return observed, population


def scaled_boundary(absent_boundary: torch.Tensor, donor_delta: torch.Tensor, gamma: float) -> torch.Tensor:
    """Construct a deployed boundary in float32, then round once to model dtype."""
    if absent_boundary.shape != donor_delta.shape:
        raise ValueError("boundary and transition shapes differ")
    if not math.isfinite(gamma):
        raise ValueError("boundary scale is nonfinite")
    result = absent_boundary.float() + float(gamma) * donor_delta.float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("scaled boundary is nonfinite")
    return result.to(absent_boundary.dtype)


@torch.no_grad()
def boundary_forward(
    model,
    tokens: torch.Tensor,
    *,
    action: str = "N",
    absent: bool = False,
    scales: Mapping[str, Mapping[str, float]] | None = None,
    direct: bool = False,
    boundary_override: torch.Tensor | None = None,
    capture_writes: Sequence[str] = (),
    patch_writes: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float], dict[str, int]]:
    """Run the production recurrence while exposing one exact raw residual boundary."""
    if action not in r505.SOURCE_ACTIONS:
        raise ValueError("unregistered score action")
    if direct and (action != "N" or absent or boundary_override is not None or capture_writes or patch_writes):
        raise ValueError("direct native arm cannot carry analytical edits")
    pair = r505.SOURCE_ACTIONS[action]["pair"]
    if (pair is not None or absent) and scales is None:
        raise ValueError("score edit requires frozen scales")
    capture_set = set(capture_writes)
    patch_writes = {} if patch_writes is None else dict(patch_writes)
    allowed_writes = {"a14", "m17"}
    if len(capture_set) != len(capture_writes) or not capture_set <= allowed_writes:
        raise ValueError("capture identity changed")
    if not set(patch_writes) <= allowed_writes:
        raise ValueError("continuation patch identity changed")

    cached: dict[str, torch.Tensor] = {}
    captures: dict[str, torch.Tensor] = {}
    states: dict[str, torch.Tensor] = {}
    diagnostics = {
        "factor_reconstruction_max": 0.0,
        "score_edit_rms": 0.0,
        "boundary_override_rms": 0.0,
        "continuation_patch_rms_max": 0.0,
    }
    audit = {
        "native_attention": 0,
        "replayed_attention": 0,
        "native_mlp": 0,
        "boundary_captures": 0,
        "boundary_overrides": 0,
        "write_captures": 0,
        "write_patches": 0,
    }
    factor_parent = r505.action_parent.factor_parent

    def maybe_patch(key: str, write: torch.Tensor) -> torch.Tensor:
        if key in patch_writes:
            replacement = patch_writes[key]
            if replacement.shape != write.shape or replacement.dtype != write.dtype \
                    or replacement.device != write.device or not bool(torch.isfinite(replacement).all()):
                raise RuntimeError(f"malformed continuation patch at {key}")
            diagnostics["continuation_patch_rms_max"] = max(
                diagnostics["continuation_patch_rms_max"],
                float((replacement.float() - write.float()).square().mean().sqrt()),
            )
            write = replacement
            audit["write_patches"] += 1
        if key in capture_set:
            captures[key] = write.detach().clone()
            audit["write_captures"] += 1
        return write

    def attention(site: int, block, state: torch.Tensor, first_value: torch.Tensor | None):
        if direct or site not in factor_parent.stage1.SITE_HEADS:
            write, next_value = block.attn(state, first_value)
            audit["native_attention"] += 1
        else:
            write, factors, support, error = factor_parent._factor_site(
                state, first_value, block.attn, site, tokens)
            audit["replayed_attention"] += 1
            diagnostics["factor_reconstruction_max"] = max(
                diagnostics["factor_reconstruction_max"], error)
            edit_pair = (0, 3) if absent else pair
            if edit_pair is not None:
                donor, recipient = edit_pair
                if site == factor_parent.TERMS[donor][1]:
                    cached.update(factors[donor])
                if site == factor_parent.TERMS[recipient][1]:
                    if not cached:
                        raise RuntimeError("donor factors unavailable at recipient")
                    target = factors[recipient]
                    replacement = torch.zeros_like(target["factor_term"])
                    if not absent:
                        row = r505.signed_scales(scales, action)
                        if row is None:
                            raise RuntimeError("edited score action lacks scale")
                        replacement = torch.bmm(cached["p"] * row["score_ratio"] * support, target["u"])
                    edit = replacement.to(write.dtype) - target["native_term"]
                    write = write + edit
                    diagnostics["score_edit_rms"] = float(edit.float().square().mean().sqrt())
            next_value = first_value
        return maybe_patch(f"a{site}", write), next_value

    facade.validate_production_model(model)
    facade.validate_tokens(tokens, production_shape=True)
    x = F.rms_norm(model.transformer.wte(tokens), (D,))
    x0 = x
    first_value = None
    prior_writes: list[torch.Tensor] = []
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_state = F.rms_norm(x, (D,))
        attention_write, first_value = attention(site, block, attention_state, first_value)
        x = x + attention_write
        mlp_state = F.rms_norm(x, (D,))
        mlp_write = maybe_patch(f"m{site}", block.mlp(mlp_state))
        audit["native_mlp"] += 1
        prior_writes.append(mlp_write)
        x = x + mlp_write
        if site == BOUNDARY_SITE:
            states["native_boundary"] = x.detach().clone()
            states["embedding_skip"] = x0.detach().clone()
            states["first_value"] = first_value.detach().clone()
            audit["boundary_captures"] += 1
            if boundary_override is not None:
                if boundary_override.shape != x.shape or boundary_override.dtype != x.dtype \
                        or boundary_override.device != x.device or not bool(torch.isfinite(boundary_override).all()):
                    raise RuntimeError("malformed boundary override")
                diagnostics["boundary_override_rms"] = float(
                    (boundary_override.float() - x.float()).square().mean().sqrt())
                x = boundary_override
                audit["boundary_overrides"] += 1
            states["effective_boundary"] = x.detach().clone()

    logits = model.lm_head(F.rms_norm(x, (D,)))
    logits = (30.0 * torch.tanh(logits / 30.0)).float()
    if tuple(logits.shape) != (*tokens.shape, facade.LOGIT_VOCAB) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("rung528 logits malformed")
    expected_attention = {"native_attention": 18, "replayed_attention": 0} if direct else {
        "native_attention": 15, "replayed_attention": 3}
    if any(audit[key] != value for key, value in expected_attention.items()) or audit["native_mlp"] != 18:
        raise RuntimeError(f"forward call audit changed: {audit}")
    if audit["boundary_captures"] != 1 or audit["boundary_overrides"] != int(boundary_override is not None):
        raise RuntimeError(f"boundary audit changed: {audit}")
    if audit["write_captures"] != len(capture_set) or set(captures) != capture_set \
            or audit["write_patches"] != len(patch_writes):
        raise RuntimeError(f"continuation audit changed: {audit}")
    states.update(captures)
    return logits, states, diagnostics, audit


def maximum_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def relative_squared(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = left.double() - right.double()
    return float(difference.square().sum() / left.double().square().sum().clamp_min(1e-300))


def _phase_slices(bounds: tuple[int, int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    lo, hi, split = bounds
    return ((lo, split), (split, hi))


def _nll(logits: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    targets = rows[:, 1:].to(logits.device)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).view(len(rows), TOKENS).float().cpu()


def _task_sums(delta_nll: torch.Tensor, masks: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = torch.stack([masks[cell].double() for cell in CELLS], dim=-1)
    leading = delta_nll.shape[:-2]
    flat = delta_nll.reshape(-1, delta_nll.shape[-2], delta_nll.shape[-1]).double()
    values = torch.einsum("abp,bpc->abc", flat, mask)
    return values.reshape(*leading, delta_nll.shape[-2], len(CELLS))


def _circuit_matrix(circuit_masks, tags, start: int, stop: int, bounds):
    return r506._circuit_mask_matrix(circuit_masks, tags, start, stop, bounds)


def _empty_phase(bounds, tags, *, include_units: bool, include_wrong: bool, substitutions):
    documents = bounds[1] - bounds[0]
    result = {
        "bounds": tuple(bounds),
        "tags": tuple(tags),
        "task_counts": torch.zeros(documents, len(CELLS), dtype=torch.float64),
        "circuit_counts": torch.zeros(2, len(MASK_TYPES), len(tags), dtype=torch.float64),
        "unit_task_sums": None,
        "unit_circuit_sums": None,
        "wrong_task_sums": None,
        "wrong_circuit_sums": None,
        "substitution_task_sums": None,
        "substitution_circuit_sums": None,
        "substitutions": [dict(row) for row in substitutions],
        "diagnostics": {
            "direct_forwards": 0,
            "absent_forwards": 0,
            "action_boundary_forwards": 0,
            "unit_insertion_forwards": 0,
            "wrong_boundary_forwards": 0,
            "wrong_insertion_forwards": 0,
            "substitution_forwards": 0,
            "boundary_captures": 0,
            "boundary_overrides": 0,
            "continuation_write_captures": 0,
            "continuation_write_patches": 0,
            "native_replay_logit_max_abs": 0.0,
            "native_replay_boundary_max_abs": 0.0,
            "maximum_self_insertion_logit_abs": 0.0,
            "maximum_self_insertion_logit_relative_squared": 0.0,
            "maximum_effective_boundary_abs": 0.0,
            "maximum_embedding_skip_abs": 0.0,
            "maximum_first_value_abs": 0.0,
            "factor_reconstruction_max": 0.0,
            "minimum_score_edit_rms": math.inf,
            "minimum_transition_rms": math.inf,
            "minimum_continuation_patch_rms": math.inf,
        },
    }
    if include_units:
        result["unit_task_sums"] = torch.zeros(
            len(SOURCES), len(qm.CONTINUATIONS), documents, len(CELLS), dtype=torch.float64)
        result["unit_circuit_sums"] = torch.zeros(
            len(SOURCES), len(qm.CONTINUATIONS), 2, len(MASK_TYPES), len(tags), dtype=torch.float64)
    if include_wrong:
        result["wrong_task_sums"] = torch.zeros(
            len(WRONG_SIGNS), len(qm.CONTINUATIONS), documents, len(CELLS), dtype=torch.float64)
        result["wrong_circuit_sums"] = torch.zeros(
            len(WRONG_SIGNS), len(qm.CONTINUATIONS), 2, len(MASK_TYPES), len(tags), dtype=torch.float64)
    if substitutions:
        result["substitution_task_sums"] = torch.zeros(
            len(substitutions), 2, len(qm.CONTINUATIONS), documents, len(CELLS), dtype=torch.float64)
        result["substitution_circuit_sums"] = torch.zeros(
            len(substitutions), 2, len(qm.CONTINUATIONS), 2, len(MASK_TYPES), len(tags), dtype=torch.float64)
    return result


def _update_run_diagnostics(total: dict, diagnostics: dict, audit: dict) -> None:
    total["boundary_captures"] += audit["boundary_captures"]
    total["boundary_overrides"] += audit["boundary_overrides"]
    total["continuation_write_captures"] += audit["write_captures"]
    total["continuation_write_patches"] += audit["write_patches"]
    total["factor_reconstruction_max"] = max(
        total["factor_reconstruction_max"], diagnostics["factor_reconstruction_max"])
    if diagnostics["score_edit_rms"] > 0:
        total["minimum_score_edit_rms"] = min(
            total["minimum_score_edit_rms"], diagnostics["score_edit_rms"])
    if audit["write_patches"]:
        total["minimum_continuation_patch_rms"] = min(
            total["minimum_continuation_patch_rms"], diagnostics["continuation_patch_rms_max"])


@torch.no_grad()
def collect_phase(
    model,
    rows,
    task_masks,
    circuit_masks,
    tags,
    scales,
    bounds,
    *,
    include_units: bool,
    include_wrong: bool,
    substitutions=(),
    include_direct: bool,
):
    data = _empty_phase(
        bounds, tags, include_units=include_units, include_wrong=include_wrong,
        substitutions=substitutions)
    diagnostics = data["diagnostics"]
    device = next(model.parameters()).device
    lo_doc, hi_doc, _split = bounds
    for start in range(lo_doc, hi_doc, BATCH):
        stop = start + BATCH
        local = start - lo_doc
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        local_masks = {cell: task_masks[cell][start:stop] for cell in CELLS}
        circuit_matrix, circuit_counts = _circuit_matrix(
            circuit_masks, tags, start, stop, bounds)
        data["circuit_counts"] += circuit_counts
        data["task_counts"][local:local + BATCH] = torch.stack(
            [local_masks[cell].sum(1).double() for cell in CELLS], dim=-1)

        direct_logits = direct_state = None
        if include_direct:
            direct_logits, direct_state, direct_diag, direct_audit = boundary_forward(
                model, tokens, direct=True)
            diagnostics["direct_forwards"] += 1
            _update_run_diagnostics(diagnostics, direct_diag, direct_audit)

        absent_logits, absent_state, absent_diag, absent_audit = boundary_forward(
            model, tokens, action="P", absent=True, scales=scales,
            capture_writes=("a14", "m17"))
        diagnostics["absent_forwards"] += 1
        _update_run_diagnostics(diagnostics, absent_diag, absent_audit)
        absent_nll = _nll(absent_logits, batch_rows)

        action_runs = {}
        deltas = {}
        for action in SOURCES:
            logits, state, run_diag, run_audit = boundary_forward(
                model, tokens, action=action, scales=scales)
            diagnostics["action_boundary_forwards"] += 1
            _update_run_diagnostics(diagnostics, run_diag, run_audit)
            action_runs[action] = (logits, state)
            delta = state["native_boundary"].float() - absent_state["native_boundary"].float()
            deltas[action] = delta
            diagnostics["minimum_transition_rms"] = min(
                diagnostics["minimum_transition_rms"], float(delta.square().mean().sqrt()))

        if include_direct:
            diagnostics["native_replay_logit_max_abs"] = max(
                diagnostics["native_replay_logit_max_abs"],
                maximum_abs(direct_logits, action_runs["N"][0]))
            diagnostics["native_replay_boundary_max_abs"] = max(
                diagnostics["native_replay_boundary_max_abs"],
                maximum_abs(direct_state["native_boundary"], action_runs["N"][1]["native_boundary"]))

        if include_units:
            unit_nll = []
            for action in SOURCES:
                action_rows = []
                for continuation in qm.CONTINUATIONS:
                    sites = CONTINUATION_PATCHES[continuation]
                    replacement = scaled_boundary(absent_state["native_boundary"], deltas[action], 1.0)
                    logits, state, run_diag, run_audit = boundary_forward(
                        model, tokens, action="P", absent=True, scales=scales,
                        boundary_override=replacement,
                        patch_writes={site: absent_state[site] for site in sites})
                    diagnostics["unit_insertion_forwards"] += 1
                    _update_run_diagnostics(diagnostics, run_diag, run_audit)
                    diagnostics["maximum_effective_boundary_abs"] = max(
                        diagnostics["maximum_effective_boundary_abs"],
                        maximum_abs(state["effective_boundary"], action_runs[action][1]["native_boundary"]))
                    if continuation == "native":
                        diagnostics["maximum_self_insertion_logit_abs"] = max(
                            diagnostics["maximum_self_insertion_logit_abs"],
                            maximum_abs(logits, action_runs[action][0]))
                        diagnostics["maximum_self_insertion_logit_relative_squared"] = max(
                            diagnostics["maximum_self_insertion_logit_relative_squared"],
                            relative_squared(action_runs[action][0], logits))
                        diagnostics["maximum_embedding_skip_abs"] = max(
                            diagnostics["maximum_embedding_skip_abs"],
                            maximum_abs(state["embedding_skip"], action_runs[action][1]["embedding_skip"]))
                        diagnostics["maximum_first_value_abs"] = max(
                            diagnostics["maximum_first_value_abs"],
                            maximum_abs(state["first_value"], action_runs[action][1]["first_value"]))
                    action_rows.append(_nll(logits, batch_rows) - absent_nll)
                unit_nll.append(torch.stack(action_rows))
            unit_nll = torch.stack(unit_nll)
            data["unit_task_sums"][:, :, local:local + BATCH] = _task_sums(unit_nll, local_masks)
            flattened = unit_nll.reshape(len(SOURCES) * len(qm.CONTINUATIONS), -1).double()
            data["unit_circuit_sums"] += torch.matmul(flattened, circuit_matrix.T).view(
                len(SOURCES), len(qm.CONTINUATIONS), 2, len(MASK_TYPES), len(tags))

        if include_wrong:
            wrong_nll = []
            for action in WRONG_SIGNS:
                _logits, state, run_diag, run_audit = boundary_forward(
                    model, tokens, action=action, scales=scales)
                diagnostics["wrong_boundary_forwards"] += 1
                _update_run_diagnostics(diagnostics, run_diag, run_audit)
                delta = state["native_boundary"].float() - absent_state["native_boundary"].float()
                diagnostics["minimum_transition_rms"] = min(
                    diagnostics["minimum_transition_rms"], float(delta.square().mean().sqrt()))
                action_rows = []
                for continuation in qm.CONTINUATIONS:
                    sites = CONTINUATION_PATCHES[continuation]
                    logits, _state, run_diag, run_audit = boundary_forward(
                        model, tokens, action="P", absent=True, scales=scales,
                        boundary_override=scaled_boundary(absent_state["native_boundary"], delta, 1.0),
                        patch_writes={site: absent_state[site] for site in sites})
                    diagnostics["wrong_insertion_forwards"] += 1
                    _update_run_diagnostics(diagnostics, run_diag, run_audit)
                    action_rows.append(_nll(logits, batch_rows) - absent_nll)
                wrong_nll.append(torch.stack(action_rows))
            wrong_nll = torch.stack(wrong_nll)
            data["wrong_task_sums"][:, :, local:local + BATCH] = _task_sums(wrong_nll, local_masks)
            flattened = wrong_nll.reshape(len(WRONG_SIGNS) * len(qm.CONTINUATIONS), -1).double()
            data["wrong_circuit_sums"] += torch.matmul(flattened, circuit_matrix.T).view(
                len(WRONG_SIGNS), len(qm.CONTINUATIONS), 2, len(MASK_TYPES), len(tags))

        if substitutions:
            substitution_nll = []
            for candidate in substitutions:
                source = candidate["source"]
                beta = candidate["beta"]
                direction_rows = []
                for donor, gamma in ((source, beta), ("N", 1.0 / beta)):
                    continuation_rows = []
                    for continuation in qm.CONTINUATIONS:
                        sites = CONTINUATION_PATCHES[continuation]
                        logits, _state, run_diag, run_audit = boundary_forward(
                            model, tokens, action="P", absent=True, scales=scales,
                            boundary_override=scaled_boundary(
                                absent_state["native_boundary"], deltas[donor], gamma),
                            patch_writes={site: absent_state[site] for site in sites})
                        diagnostics["substitution_forwards"] += 1
                        _update_run_diagnostics(diagnostics, run_diag, run_audit)
                        continuation_rows.append(_nll(logits, batch_rows) - absent_nll)
                    direction_rows.append(torch.stack(continuation_rows))
                substitution_nll.append(torch.stack(direction_rows))
            substitution_nll = torch.stack(substitution_nll)
            data["substitution_task_sums"][:, :, :, local:local + BATCH] = _task_sums(
                substitution_nll, local_masks)
            flattened = substitution_nll.reshape(
                len(substitutions) * 2 * len(qm.CONTINUATIONS), -1).double()
            data["substitution_circuit_sums"] += torch.matmul(
                flattened, circuit_matrix.T).view(
                    len(substitutions), 2, len(qm.CONTINUATIONS),
                    2, len(MASK_TYPES), len(tags))

    batches = (hi_doc - lo_doc) // BATCH
    expected = {
        "direct_forwards": batches if include_direct else 0,
        "absent_forwards": batches,
        "action_boundary_forwards": batches * len(SOURCES),
        "unit_insertion_forwards": batches * len(SOURCES) * len(qm.CONTINUATIONS) if include_units else 0,
        "wrong_boundary_forwards": batches * len(WRONG_SIGNS) if include_wrong else 0,
        "wrong_insertion_forwards": batches * len(WRONG_SIGNS) * len(qm.CONTINUATIONS) if include_wrong else 0,
        "substitution_forwards": batches * len(substitutions) * 2 * len(qm.CONTINUATIONS),
    }
    diagnostics["expected_forwards"] = expected
    diagnostics["forwards_exact"] = all(diagnostics[key] == value for key, value in expected.items())
    diagnostics["full_model_forwards"] = sum(diagnostics[key] for key in expected)
    expected_captures = diagnostics["full_model_forwards"]
    expected_overrides = diagnostics["unit_insertion_forwards"] + diagnostics["wrong_insertion_forwards"] \
        + diagnostics["substitution_forwards"]
    expected_write_captures = 2 * batches
    expected_write_patches = 4 * batches * (
        (len(SOURCES) if include_units else 0)
        + (len(WRONG_SIGNS) if include_wrong else 0)
        + (2 * len(substitutions)))
    diagnostics["boundary_calls_exact"] = bool(
        diagnostics["boundary_captures"] == expected_captures
        and diagnostics["boundary_overrides"] == expected_overrides
        and diagnostics["continuation_write_captures"] == expected_write_captures
        and diagnostics["continuation_write_patches"] == expected_write_patches)
    local_split = bounds[2] - bounds[0]
    task_half_support = torch.stack((
        data["task_counts"][:local_split].sum(0),
        data["task_counts"][local_split:].sum(0),
    ))
    diagnostics["supports_positive"] = bool(
        (task_half_support > 0).all() and (data["circuit_counts"] > 0).all())
    return data


def _effect_views(task_sums, circuit_sums, task_counts, circuit_counts):
    task_halves = []
    for lo_abs, hi_abs in _phase_slices((0, task_counts.shape[0], task_counts.shape[0] // 2)):
        numerator = task_sums[..., lo_abs:hi_abs, :].sum(-2)
        denominator = task_counts[lo_abs:hi_abs].sum(0).clamp_min(1)
        task_halves.append(numerator / denominator)
    task_halves = torch.stack(task_halves, dim=-3)
    task_pooled = task_sums.sum(-2) / task_counts.sum(0).clamp_min(1)
    means = circuit_sums / circuit_counts.clamp_min(1)
    circuit_halves = (means[..., 0, :] - means[..., 1, :]).movedim(-2, -3)
    pooled_sums = circuit_sums.sum(-3)
    pooled_counts = circuit_counts.sum(0).clamp_min(1)
    pooled_means = pooled_sums / pooled_counts
    circuit_pooled = pooled_means[..., 0, :] - pooled_means[..., 1, :]
    return {
        "task_halves_full": task_halves,
        "task_halves": task_halves[..., list(TASK_CONTEXT_INDICES)],
        "task_pooled_full": task_pooled,
        "task_pooled": task_pooled[..., list(TASK_CONTEXT_INDICES)],
        "circuit_halves": circuit_halves,
        "circuit_pooled": circuit_pooled,
    }


def phase_views(data, prefix: str):
    return _effect_views(
        data[f"{prefix}_task_sums"], data[f"{prefix}_circuit_sums"],
        data["task_counts"], data["circuit_counts"])


def instrument_holds(data, *, require_self_replay: bool) -> bool:
    d = data["diagnostics"]
    return bool(
        d["forwards_exact"] and d["boundary_calls_exact"] and d["supports_positive"]
        and d["factor_reconstruction_max"] <= 1e-10
        and d["minimum_score_edit_rms"] > 0 and d["minimum_transition_rms"] > 0
        and (d["minimum_continuation_patch_rms"] > 0 or d["continuation_write_patches"] == 0)
        and (not require_self_replay or (
            d["native_replay_logit_max_abs"] == 0.0
            and d["native_replay_boundary_max_abs"] == 0.0
            and d["maximum_self_insertion_logit_abs"] == 0.0
            and d["maximum_effective_boundary_abs"] == 0.0
            and d["maximum_embedding_skip_abs"] == 0.0
            and d["maximum_first_value_abs"] == 0.0)))


def _q95(values) -> float:
    return float(torch.quantile(torch.as_tensor(values, dtype=torch.float64), .95))


def discover_candidates(unit_views, wrong_views):
    candidates = []
    checks = {}
    target = SOURCES.index("N")
    wrong_cosines = []
    for wrong_index, wrong in enumerate(WRONG_SIGNS):
        row = qm.score_pair(
            unit_views["circuit_halves"][target], wrong_views["circuit_halves"][wrong_index],
            unit_views["task_halves"][target], wrong_views["task_halves"][wrong_index])
        wrong_cosines.append(row["circuit"][0]["cosine"])
    for source in CANDIDATE_SOURCES:
        source_index = SOURCES.index(source)
        row = qm.score_pair(
            unit_views["circuit_halves"][target], unit_views["circuit_halves"][source_index],
            unit_views["task_halves"][target], unit_views["task_halves"][source_index])
        permutation_cosines = []
        for seed in PERMUTATION_SEEDS:
            generator = torch.Generator().manual_seed(seed)
            scrambled = unit_views["circuit_halves"][source_index].clone()
            for continuation in range(len(qm.CONTINUATIONS)):
                order = torch.randperm(scrambled.shape[-1], generator=generator)
                scrambled[:, continuation] = scrambled[:, continuation, order]
            control = qm.score_pair(
                unit_views["circuit_halves"][target], scrambled,
                unit_views["task_halves"][target], unit_views["task_halves"][source_index])
            permutation_cosines.append(control["circuit"][0]["cosine"])
        control_q95 = _q95(permutation_cosines)
        strongest_control = max(control_q95, max(wrong_cosines))
        margin = row["circuit"][0]["cosine"] - strongest_control
        holds = bool(row["passes_without_controls"] and margin >= .10)
        checks[source] = {
            **row,
            "permutation_cosines": permutation_cosines,
            "permutation_q95": control_q95,
            "wrong_sign_cosines": dict(zip(WRONG_SIGNS, wrong_cosines)),
            "strongest_control": strongest_control,
            "control_margin": margin,
            "holds": holds,
        }
        if holds:
            candidates.append({"source": source, "beta": row["beta"]})
    return candidates, checks


def _one_comparison(target, observed, beta: float, cosine_bar: float, residual_bar: float):
    row = qm.relation_metrics(target, observed, beta)
    row["holds"] = bool(row["cosine"] >= cosine_bar and row["relative_residual"] <= residual_bar)
    return row


def score_physical(discovery_unit, physical_substitution, candidates):
    checks = {}
    passing = []
    n_index = SOURCES.index("N")
    for candidate_index, candidate in enumerate(candidates):
        source_index = SOURCES.index(candidate["source"])
        row = {"directions": {}}
        holds = True
        for direction, target_index in enumerate((n_index, source_index)):
            circuit_target = discovery_unit["circuit_halves"][target_index, 1]
            task_target = discovery_unit["task_halves"][target_index, 1]
            circuit_observed = physical_substitution["circuit_pooled"][candidate_index, direction]
            task_observed = physical_substitution["task_pooled"][candidate_index, direction]
            circuit = _one_comparison(circuit_target, circuit_observed, 1.0, .80, .50)
            task = _one_comparison(task_target, task_observed, 1.0, .70, .65)
            continuations = [
                _one_comparison(
                    circuit_target[continuation], circuit_observed[continuation], 1.0, 0.0, math.inf)
                for continuation in range(len(qm.CONTINUATIONS))
            ]
            direction_holds = bool(
                circuit["holds"] and task["holds"]
                and all(item["cosine"] > 0 for item in continuations))
            row["directions"]["N_from_source" if direction == 0 else "source_from_N"] = {
                "circuit": circuit,
                "task": task,
                "continuations": continuations,
                "holds": direction_holds,
            }
            holds &= direction_holds
        row["holds"] = bool(holds)
        checks[candidate["source"]] = row
        if holds:
            passing.append(candidate)
    return passing, checks


def score_repeat(unit, substitution, candidates, *, cosine_bar: float, residual_bar: float):
    checks = {}
    passing = []
    n_index = SOURCES.index("N")
    for candidate_index, candidate in enumerate(candidates):
        source_index = SOURCES.index(candidate["source"])
        row = {"native_relation": {}, "substitutions": {}}
        holds = True
        for window, target_circuit, source_circuit, target_task, source_task in (
            ("half0", unit["circuit_halves"][n_index, 0], unit["circuit_halves"][source_index, 0],
             unit["task_halves"][n_index, 0], unit["task_halves"][source_index, 0]),
            ("half1", unit["circuit_halves"][n_index, 1], unit["circuit_halves"][source_index, 1],
             unit["task_halves"][n_index, 1], unit["task_halves"][source_index, 1]),
            ("pooled", unit["circuit_pooled"][n_index], unit["circuit_pooled"][source_index],
             unit["task_pooled"][n_index], unit["task_pooled"][source_index]),
        ):
            circuit = _one_comparison(
                target_circuit, source_circuit, candidate["beta"], cosine_bar, residual_bar)
            task = _one_comparison(target_task, source_task, candidate["beta"], .70, .65)
            continuations = [
                _one_comparison(
                    target_circuit[c], source_circuit[c], candidate["beta"], 0.0, math.inf)
                for c in range(len(qm.CONTINUATIONS))
            ]
            window_holds = bool(
                circuit["holds"] and task["holds"]
                and all(item["cosine"] > 0 for item in continuations))
            row["native_relation"][window] = {
                "circuit": circuit, "task": task, "continuations": continuations,
                "holds": window_holds,
            }
            holds &= window_holds
        for direction, target_index in enumerate((n_index, source_index)):
            direction_name = "N_from_source" if direction == 0 else "source_from_N"
            row["substitutions"][direction_name] = {}
            for window, half in (("half0", 0), ("half1", 1), ("pooled", None)):
                target_circuit = unit["circuit_pooled"][target_index] if half is None \
                    else unit["circuit_halves"][target_index, half]
                target_task = unit["task_pooled"][target_index] if half is None \
                    else unit["task_halves"][target_index, half]
                observed_circuit = substitution["circuit_pooled"][candidate_index, direction] if half is None \
                    else substitution["circuit_halves"][candidate_index, direction, half]
                observed_task = substitution["task_pooled"][candidate_index, direction] if half is None \
                    else substitution["task_halves"][candidate_index, direction, half]
                circuit = _one_comparison(
                    target_circuit, observed_circuit, 1.0, cosine_bar, residual_bar)
                task = _one_comparison(target_task, observed_task, 1.0, .70, .65)
                continuations = [
                    _one_comparison(
                        target_circuit[c], observed_circuit[c], 1.0, 0.0, math.inf)
                    for c in range(len(qm.CONTINUATIONS))
                ]
                window_holds = bool(
                    circuit["holds"] and task["holds"]
                    and all(item["cosine"] > 0 for item in continuations))
                row["substitutions"][direction_name][window] = {
                    "circuit": circuit, "task": task, "continuations": continuations,
                    "holds": window_holds,
                }
                holds &= window_holds
        row["holds"] = bool(holds)
        checks[candidate["source"]] = row
        if holds:
            passing.append(candidate)
    return passing, checks


def selective_pairs(confirmation_unit, validation_unit, candidates):
    checks = {}
    passing = []
    native_continuation = qm.CONTINUATIONS.index("native")
    all_index = CELLS.index("all_positive")
    off_index = CELLS.index("off_target")
    for candidate in candidates:
        row = {}
        holds = True
        for phase_name, unit in (("confirmation", confirmation_unit), ("validation", validation_unit)):
            phase = {}
            for source in ("N", candidate["source"]):
                source_index = SOURCES.index(source)
                windows = []
                for half in range(2):
                    all_effect = float(unit["task_halves_full"][source_index, half, native_continuation, all_index])
                    off_effect = float(unit["task_halves_full"][source_index, half, native_continuation, off_index])
                    window_holds = bool(abs(all_effect) >= .002 and abs(all_effect) >= 3 * abs(off_effect))
                    windows.append({
                        "all_copy_effect_nat": all_effect,
                        "off_target_effect_nat": off_effect,
                        "absolute_selectivity_ratio": abs(all_effect) / max(abs(off_effect), 1e-30),
                        "holds": window_holds,
                    })
                    holds &= window_holds
                phase[source] = windows
            row[phase_name] = phase
        row["holds"] = bool(holds)
        checks[candidate["source"]] = row
        if holds:
            passing.append(candidate)
    return passing, checks


@torch.no_grad()
def gpu_smoke(output_path: Path = SMOKE_OUT) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite result: {output_path}")
    dependencies, population = validate_dependencies()
    rows, _task_masks, _circuit_masks, scales, discovery_tags, _validation_tags, metadata = population
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    tokens = rows[:4, :-1].to("cuda")

    direct_logits, direct_state, direct_diag, direct_audit = boundary_forward(
        model, tokens, direct=True)
    absent_logits, absent_state, absent_diag, absent_audit = boundary_forward(
        model, tokens, action="P", absent=True, scales=scales,
        capture_writes=("a14", "m17"))
    action_runs = {}
    injections = {}
    total_patches = 0
    minimum_transition_rms = math.inf
    minimum_continuation_patch_rms = math.inf
    maximum_self_logit_error = 0.0
    maximum_self_logit_relative_squared = 0.0
    maximum_effective_boundary_error = 0.0
    maximum_skip_state_error = 0.0
    maximum_first_value_error = 0.0
    factor_reconstruction_max = absent_diag["factor_reconstruction_max"]
    minimum_score_edit_rms = absent_diag["score_edit_rms"]
    for action in SOURCES:
        logits, state, diagnostics, audit = boundary_forward(
            model, tokens, action=action, scales=scales)
        action_runs[action] = {"logits": logits, "state": state, "diagnostics": diagnostics, "audit": audit}
        factor_reconstruction_max = max(factor_reconstruction_max, diagnostics["factor_reconstruction_max"])
        if diagnostics["score_edit_rms"] > 0:
            minimum_score_edit_rms = min(minimum_score_edit_rms, diagnostics["score_edit_rms"])
        delta = state["native_boundary"].float() - absent_state["native_boundary"].float()
        transition_rms = float(delta.square().mean().sqrt())
        minimum_transition_rms = min(minimum_transition_rms, transition_rms)
        injections[action] = {}
        for continuation, sites in CONTINUATION_PATCHES.items():
            patches = {site: absent_state[site] for site in sites}
            replacement = scaled_boundary(absent_state["native_boundary"], delta, 1.0)
            injected_logits, injected_state, injected_diag, injected_audit = boundary_forward(
                model, tokens, action="P", absent=True, scales=scales,
                boundary_override=replacement, patch_writes=patches)
            injections[action][continuation] = injected_logits
            factor_reconstruction_max = max(
                factor_reconstruction_max, injected_diag["factor_reconstruction_max"])
            total_patches += injected_audit["write_patches"]
            if sites:
                minimum_continuation_patch_rms = min(
                    minimum_continuation_patch_rms,
                    injected_diag["continuation_patch_rms_max"])
            maximum_effective_boundary_error = max(
                maximum_effective_boundary_error,
                maximum_abs(injected_state["effective_boundary"], state["native_boundary"]))
            if continuation == "native":
                maximum_self_logit_error = max(
                    maximum_self_logit_error, maximum_abs(injected_logits, logits))
                maximum_self_logit_relative_squared = max(
                    maximum_self_logit_relative_squared, relative_squared(logits, injected_logits))
                maximum_skip_state_error = max(
                    maximum_skip_state_error,
                    maximum_abs(injected_state["embedding_skip"], state["embedding_skip"]))
                maximum_first_value_error = max(
                    maximum_first_value_error,
                    maximum_abs(injected_state["first_value"], state["first_value"]))

    native_replay_logit_max_abs = maximum_abs(direct_logits, action_runs["N"]["logits"])
    native_replay_boundary_max_abs = maximum_abs(
        direct_state["native_boundary"], action_runs["N"]["state"]["native_boundary"])
    diagnostics = {
        "full_model_forwards": 22,
        "direct_forwards": 1,
        "analytical_forwards": 21,
        "boundary_captures": 22,
        "boundary_overrides": 16,
        "continuation_write_captures": absent_audit["write_captures"],
        "continuation_write_patches": total_patches,
        "native_replay_logit_max_abs": native_replay_logit_max_abs,
        "native_replay_boundary_max_abs": native_replay_boundary_max_abs,
        "maximum_self_insertion_logit_abs": maximum_self_logit_error,
        "maximum_self_insertion_logit_relative_squared": maximum_self_logit_relative_squared,
        "maximum_effective_boundary_abs": maximum_effective_boundary_error,
        "maximum_embedding_skip_abs": maximum_skip_state_error,
        "maximum_first_value_abs": maximum_first_value_error,
        "factor_reconstruction_max": factor_reconstruction_max,
        "minimum_score_edit_rms": minimum_score_edit_rms,
        "minimum_transition_rms": minimum_transition_rms,
        "minimum_continuation_patch_rms": minimum_continuation_patch_rms,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "calls_exact": bool(total_patches == 16 and absent_audit["write_captures"] == 2),
    }
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and native_replay_logit_max_abs == 0.0
        and native_replay_boundary_max_abs == 0.0
        and maximum_self_logit_error == 0.0
        and maximum_effective_boundary_error == 0.0
        and maximum_skip_state_error == 0.0
        and maximum_first_value_error == 0.0
        and factor_reconstruction_max <= 1e-10
        and diagnostics["calls_exact"])
    pred_b = bool(minimum_score_edit_rms > 0 and minimum_transition_rms > 0)
    pred_c = bool(minimum_continuation_patch_rms > 0)
    result = {
        "status": "complete",
        "rung": "528_gpu_smoke",
        "claim_level": "raw_boundary_operational_smoke_no_scientific_outcomes_retained",
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "dependency_sha256": dependencies,
        "input_identity": {
            "documents_exercised": [0, 4],
            "circuit_tags_named_but_not_scored": discovery_tags[:2],
            "population": metadata,
        },
        "diagnostics": diagnostics,
        "planted_suite": qm.planted_suite(),
        "check_a_exact_boundary_self_replay": pred_a,
        "check_b_score_transitions_are_live": pred_b,
        "check_c_continuation_patches_are_live": pred_c,
        "strong_null": not (pred_a and pred_b and pred_c),
        "scientific_task_or_circuit_effects_retained": False,
        "next_step": "implement_and_enqueue_full_rung528" if pred_a and pred_b and pred_c else "repair_instrument_only",
    }
    atomic_json(output_path, result)
    print(json.dumps({
        "status": result["status"],
        "rung": result["rung"],
        "checks": {key: value for key, value in result.items() if key.startswith("check_")},
        "strong_null": result["strong_null"],
        "diagnostics": diagnostics,
        "scientific_outcomes_retained": False,
    }, indent=2, sort_keys=True))
    return result


def _bundle_phase(data):
    return {key: value for key, value in data.items() if key != "diagnostics"} | {
        "diagnostics": data["diagnostics"]}


@torch.no_grad()
def run_full() -> dict[str, object]:
    if OUT.exists() or BUNDLE.exists():
        raise FileExistsError("refusing to overwrite rung528 full namespace")
    if file_sha256(SMOKE_V2_OUT) != SMOKE_V2_SHA256:
        raise RuntimeError("managed v2 smoke hash changed")
    smoke = json.loads(SMOKE_V2_OUT.read_text())
    if not (
        smoke.get("check_a_exact_boundary_self_replay") is True
        and smoke.get("check_b_score_transitions_are_live") is True
        and smoke.get("check_c_continuation_patches_are_live") is True
        and smoke.get("strong_null") is False
        and smoke.get("scientific_task_or_circuit_effects_retained") is False
    ):
        raise RuntimeError("managed v2 smoke authority is absent or failed")
    dependencies, population = validate_dependencies()
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = population
    wall_started = time.time()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    phases = {}
    phases["discovery"] = collect_phase(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, DISCOVERY,
        include_units=True, include_wrong=True, substitutions=(), include_direct=True)
    discovery_unit = phase_views(phases["discovery"], "unit")
    discovery_wrong = phase_views(phases["discovery"], "wrong")
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and instrument_holds(phases["discovery"], require_self_replay=True))
    candidates, discovery_checks = discover_candidates(discovery_unit, discovery_wrong)
    pred_b = bool(pred_a and 1 <= len(candidates) <= 3)

    physical_candidates = []
    physical_checks = {}
    confirmed_candidates = []
    confirmation_checks = {}
    validated_candidates = []
    validation_checks = {}
    selective_candidates = []
    selectivity_checks = {}

    if pred_b:
        phases["physical_discovery"] = collect_phase(
            model, rows, task_masks, circuit_masks, discovery_tags, scales, PHYSICAL_DISCOVERY,
            include_units=False, include_wrong=False, substitutions=candidates, include_direct=False)
        physical_views = phase_views(phases["physical_discovery"], "substitution")
        physical_candidates, physical_checks = score_physical(
            discovery_unit, physical_views, candidates)

    if physical_candidates:
        phases["confirmation"] = collect_phase(
            model, rows, task_masks, circuit_masks, discovery_tags, scales, CONFIRMATION,
            include_units=True, include_wrong=False, substitutions=physical_candidates,
            include_direct=True)
        confirmation_unit = phase_views(phases["confirmation"], "unit")
        confirmation_substitution = phase_views(phases["confirmation"], "substitution")
        confirmed_candidates, confirmation_checks = score_repeat(
            confirmation_unit, confirmation_substitution, physical_candidates,
            cosine_bar=.75, residual_bar=.55)
    pred_c = bool(pred_a and pred_b and confirmed_candidates)

    if pred_c:
        phases["validation"] = collect_phase(
            model, rows, task_masks, circuit_masks, validation_tags, scales, VALIDATION,
            include_units=True, include_wrong=False, substitutions=confirmed_candidates,
            include_direct=True)
        validation_unit = phase_views(phases["validation"], "unit")
        validation_substitution = phase_views(phases["validation"], "substitution")
        validated_candidates, validation_checks = score_repeat(
            validation_unit, validation_substitution, confirmed_candidates,
            cosine_bar=.70, residual_bar=.60)
    pred_d = bool(pred_c and validated_candidates)

    if pred_d:
        confirmed_names = {row["source"] for row in confirmed_candidates}
        confirmed_for_validation = [
            row for row in validated_candidates if row["source"] in confirmed_names]
        selective_candidates, selectivity_checks = selective_pairs(
            confirmation_unit, validation_unit, confirmed_for_validation)
    pred_e = bool(pred_d and selective_candidates)
    strong_null = not all((pred_a, pred_b, pred_c, pred_d, pred_e))

    torch.cuda.synchronize()
    runtime_s = time.time() - wall_started
    total_forwards = sum(
        phase["diagnostics"]["full_model_forwards"] for phase in phases.values())
    expected_forwards = 1984
    if pred_b:
        expected_forwards += 31 * (5 + 8 * len(candidates))
    if physical_candidates:
        expected_forwards += 62 * (22 + 8 * len(physical_candidates))
    if pred_c:
        expected_forwards += 125 * (22 + 8 * len(confirmed_candidates))
    calls_exact = total_forwards == expected_forwards
    pred_a = bool(pred_a and calls_exact and all(
        instrument_holds(phase, require_self_replay=phase["diagnostics"]["direct_forwards"] > 0)
        for phase in phases.values()))
    if not pred_a:
        pred_c = pred_d = pred_e = False
        strong_null = True

    bundle = {
        "schema": "equality_distributed_finite_transition_quotient_rung528_sufficient_statistics_v1",
        "phases": {name: _bundle_phase(value) for name, value in phases.items()},
        "raw_tokens_logits_boundaries_or_hidden_states_included": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete",
        "rung": 528,
        "claim_level": "N_centered_continuation_defined_finite_transition_pairs_not_transitive_quotient_or_compression",
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "dependency_sha256": dependencies,
        "runner_sha256": file_sha256(RUNNER),
        "managed_smoke_v2_sha256": file_sha256(SMOKE_V2_OUT),
        "input_identity": metadata,
        "continuations": list(qm.CONTINUATIONS),
        "candidate_pairs": [f"N:{source}" for source in CANDIDATE_SOURCES],
        "discovery": {
            "candidates": candidates,
            "checks": discovery_checks,
        },
        "physical_discovery": {
            "opened": "physical_discovery" in phases,
            "passing": physical_candidates,
            "checks": physical_checks,
        },
        "confirmation": {
            "opened": "confirmation" in phases,
            "passing": confirmed_candidates,
            "checks": confirmation_checks,
        },
        "validation": {
            "opened": "validation" in phases,
            "passing": validated_candidates,
            "checks": validation_checks,
        },
        "selectivity": {
            "passing": selective_candidates,
            "checks": selectivity_checks,
        },
        "phase_diagnostics": {name: value["diagnostics"] for name, value in phases.items()},
        "pred_a_exact_live_boundary_instrument": pred_a,
        "pred_b_at_least_one_discovery_transition_relation": pred_b,
        "pred_c_physical_substitution_and_new_documents": pred_c,
        "pred_d_heldout_circuits_and_documents": pred_d,
        "pred_e_selective_distributed_state": pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {
            "path": str(BUNDLE),
            "sha256": file_sha256(BUNDLE),
            "bytes": BUNDLE.stat().st_size,
        },
        "execution_price": {
            "full_model_forwards": total_forwards,
            "expected_full_model_forwards": expected_forwards,
            "calls_exact": calls_exact,
            "model_backwards": 0,
            "fitted_positive_scalars": len(candidates),
            "deployed_values_added": 0,
            "deployed_values_removed": 0,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "runtime_s": runtime_s,
        "next_step": (
            "split_validated_transition_internally_while_preserving_continuation_fingerprint"
            if not strong_null else
            "repair_boundary_instrument_only" if not pred_a else
            "close_post_MLP12_distributed_state_equivalence" if not pred_b else
            "close_response_similarity_as_non_interchangeable_or_document_specific" if not pred_c else
            "close_relation_as_discovery_circuit_specific" if not pred_d else
            "retain_broad_state_equivalence_without_equality_circuit_label"
        ),
    }
    atomic_json(OUT, result)
    print(json.dumps({
        "status": result["status"],
        "rung": result["rung"],
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "candidates": candidates,
        "physical_candidates": physical_candidates,
        "confirmed_candidates": confirmed_candidates,
        "validated_candidates": validated_candidates,
        "selective_candidates": selective_candidates,
        "execution_price": result["execution_price"],
        "runtime_s": runtime_s,
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))
    return result


def dry_run() -> dict[str, object]:
    dependencies, population = validate_dependencies()
    rows, _task_masks, _circuit_masks, _scales, discovery_tags, validation_tags, _metadata = population
    planted = qm.planted_suite()
    assert tuple(rows.shape) == (1000, 257)
    assert len(discovery_tags) == 32 and len(validation_tags) == 30
    assert tuple(CONTINUATION_PATCHES) == qm.CONTINUATIONS
    assert sum(len(value) for value in CONTINUATION_PATCHES.values()) == 4
    assert 62 * 32 == 1984
    assert 1984 + 31 * (5 + 8 * 3) + 62 * (22 + 8 * 3) + 125 * (22 + 8 * 3) == 11485
    assert planted["passes"]
    return {
        "status": "dry_run_passed",
        "rung": 528,
        "model_loaded": False,
        "outcomes_opened": False,
        "dependencies": dependencies,
        "discovery_documents": [0, 248, 124],
        "confirmation_documents": [248, 496, 372],
        "validation_documents": [500, 1000, 750],
        "discovery_circuits": len(discovery_tags),
        "validation_circuits": len(validation_tags),
        "unconditional_discovery_forwards": 1984,
        "maximum_conditional_forwards": 11485,
        "registered_a_exact_boundary_self_replay": True,
        "registered_b_shared_discovery_transition": True,
        "registered_c_physical_new_document_prediction": True,
        "registered_d_heldout_circuits_documents": True,
        "registered_e_selective_distributed_state": True,
        "planted_suite_passes": planted["passes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpu_smoke:
        gpu_smoke(SMOKE_V2_OUT if os.environ.get("R528_SMOKE_V2") == "1" else SMOKE_OUT)
        return
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(dry_run(), indent=2, sort_keys=True))
        return
    run_full()


if __name__ == "__main__":
    main()
