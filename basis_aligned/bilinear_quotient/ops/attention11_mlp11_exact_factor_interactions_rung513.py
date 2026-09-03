#!/usr/bin/env python3
"""RUNG513 -- exact finite factor interactions at attention11 and MLP11."""

# BQGATE: EXPERIMENT
# pred_a: exact branch, factor-corner, Mobius, MLP, calibration, and patch instrument is live
# pred_b: at least one fixed branch/term group preserves all three source relations in discovery
# pred_c: a discovery group predicts fresh documents with its scales frozen
# pred_d: a confirmed group passes all six physical term substitutions and removal controls
# pred_e: one causally interchangeable term is reused by at least two MLP10 branch subsets

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops", POLY, ROOT.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import mlp10_branch_first_consumer_quotient_rung512 as r512
from jacclust.tt_model import apply_rotary_emb


r511 = r512.r511
parent = r511.r510.r509.parent
PREREG = POLY / "ATTENTION11_MLP11_EXACT_FACTOR_INTERACTIONS_RUNG513_PREREGISTRATION.md"
R512_RESULT = ROOT / "mlp10_branch_first_consumer_quotient_rung512_results.json"
R512_BUNDLE = ROOT / "mlp10_branch_first_consumer_quotient_rung512_bundle.pt"
R512_SOURCE = ROOT / "ops/mlp10_branch_first_consumer_quotient_rung512.py"
OUT = ROOT / "attention11_mlp11_exact_factor_interactions_rung513_results.json"
BUNDLE = ROOT / "attention11_mlp11_exact_factor_interactions_rung513_bundle.pt"

HASHES = {
    PREREG: "b895d1aefdac4c7deee0477c260a5e1ec087477925e841d0d2b8ebb4a02670aa",
    R512_RESULT: "118d28d4d3b106df6b9d20d165a955ace2bfc07ee35b07e9ea748ecb9d6d877e",
    R512_BUNDLE: "504b7d8e892009cfe2c88462f99db53132108adb6b13b21521b0bb3dbf350113",
    R512_SOURCE: "ed66fc329b6ad6ce0e6e4b843bbef0046a53d9dbb229f0f7c99604e75ef96f9b",
}

DISCOVERY = r511.DISCOVERY
CONFIRMATION = r511.CONFIRMATION
WINDOWS = ("half0", "half1", "pooled")
TASK_CELLS = parent.TASK_CELLS
COPY_CELLS = r512.COPY_CELLS
FACTOR_NAMES = ("Q", "K", "Q2", "K2", "V")
ATTENTION_MASKS = tuple(range(1, 32))
ATTENTION_TERMS = tuple(
    "A11{" + ",".join(FACTOR_NAMES[index] for index in range(5) if mask & (1 << index)) + "}"
    for mask in ATTENTION_MASKS
)
MLP_TERMS = ("M11{L}", "M11{R}", "M11{L,R}")
TERM_NAMES = ATTENTION_TERMS + MLP_TERMS
SELECTED_SUBSETS = tuple(
    r511.SUBSET_NAMES.index(name)
    for name in ("L", "R", "L+R", "L+LR", "R+LR", "L+R+LR")
)
N_LOCAL_NODES = len(parent.SOURCES) * len(SELECTED_SUBSETS)
RELATION_ACTIONS = ((0, 2), (0, 3), (1, 2))  # N-Z7, N-Z8, P-Z7.
RELATION_NAMES = tuple(f"{parent.SOURCES[left]}-{parent.SOURCES[right]}"
                       for left, right in RELATION_ACTIONS)
SOURCE_RELATION_NAMES = tuple(
    f"{parent.SOURCES[left]}::{r511.SUBSET_NAMES[subset]} <-> "
    f"{parent.SOURCES[right]}::{r511.SUBSET_NAMES[subset]}"
    for subset in SELECTED_SUBSETS for left, right in RELATION_ACTIONS
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_node(action: int, selected_subset: int) -> int:
    return action * len(SELECTED_SUBSETS) + selected_subset


def global_node(action: int, selected_subset: int) -> int:
    return action * r511.N_SUBSETS + SELECTED_SUBSETS[selected_subset]


def term_site(term_index: int) -> str:
    return "a11" if term_index < len(ATTENTION_TERMS) else "m11"


@torch.no_grad()
def factor_consumer_call(model, callable_):
    captured = {}
    block = model.transformer.h[11]

    def attention_pre(_module, arguments):
        if len(arguments) != 2:
            raise RuntimeError(f"attention11 input signature changed: {len(arguments)}")
        captured["a11_input"] = arguments[0].detach().clone()
        captured["a11_v1"] = arguments[1].detach().clone()

    def attention_post(_module, _arguments, output):
        captured["a11"] = output[0].detach().clone()

    def mlp_pre(_module, arguments):
        captured["m11_input"] = arguments[0].detach().clone()

    def mlp_post(_module, _arguments, output):
        captured["m11"] = output.detach().clone()

    handles = [
        block.attn.register_forward_pre_hook(attention_pre),
        block.attn.register_forward_hook(attention_post),
        block.mlp.register_forward_pre_hook(mlp_pre),
        block.mlp.register_forward_hook(mlp_post),
    ]
    try:
        result = callable_()
    finally:
        for handle in handles:
            handle.remove()
    expected = {"a11_input", "a11_v1", "a11", "m11_input", "m11"}
    if set(captured) != expected:
        raise RuntimeError(f"factor consumer capture changed: {sorted(captured)}")
    return result, captured


@torch.no_grad()
def attention_factors(attention, capture: dict) -> tuple[torch.Tensor, ...]:
    state, first_value = capture["a11_input"], capture["a11_v1"]
    batch, tokens, _ = state.shape
    shape = (batch, tokens, attention.n_head, attention.head_dim)
    q = attention.c_q(state).view(shape)
    k = attention.c_k(state).view(shape)
    q2 = attention.c_q2(state).view(shape)
    k2 = attention.c_k2(state).view(shape)
    value = attention.c_v(state).view(shape)
    value = (1 - attention.lamb) * value + attention.lamb * first_value.view(shape)
    cos, sin = attention.rotary(q)
    q = apply_rotary_emb(F.rms_norm(q, (attention.head_dim,)), cos, sin)
    k = apply_rotary_emb(F.rms_norm(k, (attention.head_dim,)), cos, sin)
    q2 = apply_rotary_emb(F.rms_norm(q2, (attention.head_dim,)), cos, sin)
    k2 = apply_rotary_emb(F.rms_norm(k2, (attention.head_dim,)), cos, sin)
    return q, k, q2, k2, value


@torch.no_grad()
def attention_corner(attention, factors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    q, k, q2, k2, value = factors
    mixed = attention.squared_attention(q, k, value, q2, k2)
    mixed = mixed.transpose(1, 2).contiguous().view(q.shape[0], q.shape[1], -1)
    return attention.c_proj(mixed)


def mobius_terms(corners: dict[int, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    if set(corners) != set(range(32)):
        raise ValueError("five-factor corner cube changed")
    terms = []
    for mask in ATTENTION_MASKS:
        value = torch.zeros_like(corners[0], dtype=torch.float32)
        subset = mask
        while True:
            sign = -1.0 if ((mask.bit_count() - subset.bit_count()) & 1) else 1.0
            value.add_(corners[subset].float(), alpha=sign)
            if subset == 0:
                break
            subset = (subset - 1) & mask
        terms.append(value)
    return tuple(terms)


@torch.no_grad()
def exact_terms(model, removed: dict, current: dict) -> tuple[tuple[torch.Tensor, ...], dict]:
    attention = model.transformer.h[11].attn
    baseline = attention_factors(attention, removed)
    intact = attention_factors(attention, current)
    corners = {
        mask: attention_corner(attention, tuple(
            intact[index] if mask & (1 << index) else baseline[index]
            for index in range(5)))
        for mask in range(32)
    }
    attention_terms = mobius_terms(corners)
    attention_total = current["a11"].float() - removed["a11"].float()
    attention_sum = sum(attention_terms)
    remainder = attention_total - attention_sum
    attention_ratio = float(remainder.square().mean().sqrt()
                            / attention_total.square().mean().sqrt().clamp_min(1e-30))

    mlp11 = model.transformer.h[11].mlp
    mlp_terms, mlp_diag = r511.deployed_branches(
        mlp11,
        {"z": removed["m11_input"], "deployed_write": removed["m11"]},
        {"z": current["m11_input"], "deployed_write": current["m11"]},
    )
    diagnostics = {
        "removed_attention_corner_replay_max_abs": float(
            (corners[0] - removed["a11"]).float().abs().max()),
        "intact_attention_corner_replay_max_abs": float(
            (corners[31] - current["a11"]).float().abs().max()),
        "attention_mobius_relative_squared": r511._relative_squared(attention_total, attention_sum),
        "attention_numerical_remainder_rms_ratio": attention_ratio,
        "mlp_deployed_branch_sum_relative_squared": mlp_diag[
            "deployed_branch_sum_relative_squared"],
        "mlp_removed_corner_replay_max_abs": mlp_diag["absent_corner_replay_max_abs"],
        "mlp_intact_corner_replay_max_abs": mlp_diag["current_corner_replay_max_abs"],
        "attention_total_rms": float(attention_total.square().mean().sqrt()),
        "mlp_total_rms": mlp_diag["total_rms"],
    }
    return attention_terms + mlp_terms, diagnostics


def _empty_statistics() -> dict:
    return {
        "term_gram": {
            window: torch.zeros(len(TERM_NAMES), N_LOCAL_NODES, N_LOCAL_NODES,
                                dtype=torch.float64)
            for window in WINDOWS
        },
        "total_gram": {
            window: torch.zeros(2, N_LOCAL_NODES, N_LOCAL_NODES, dtype=torch.float64)
            for window in WINDOWS
        },
        "term_total_cross": {
            window: torch.zeros(len(TERM_NAMES), N_LOCAL_NODES, N_LOCAL_NODES,
                                dtype=torch.float64)
            for window in WINDOWS
        },
        "source_gram": {
            window: torch.zeros(N_LOCAL_NODES, N_LOCAL_NODES, dtype=torch.float64)
            for window in WINDOWS
        },
    }


def _update_statistics(statistics: dict, term_vectors: list[list[torch.Tensor]],
                       total_vectors: list[list[torch.Tensor]],
                       source_vectors: list[torch.Tensor], half: str) -> None:
    for term_index in range(len(TERM_NAMES)):
        values = torch.stack([term_vectors[node][term_index] for node in range(N_LOCAL_NODES)]).double()
        gram = values @ values.T
        site_index = 0 if term_site(term_index) == "a11" else 1
        totals = torch.stack([total_vectors[node][site_index]
                              for node in range(N_LOCAL_NODES)]).double()
        cross = values @ totals.T
        for window in (half, "pooled"):
            statistics["term_gram"][window][term_index] += gram
            statistics["term_total_cross"][window][term_index] += cross
    for site_index in range(2):
        values = torch.stack([total_vectors[node][site_index]
                              for node in range(N_LOCAL_NODES)]).double()
        gram = values @ values.T
        for window in (half, "pooled"):
            statistics["total_gram"][window][site_index] += gram
    values = torch.stack(source_vectors).double()
    gram = values @ values.T
    for window in (half, "pooled"):
        statistics["source_gram"][window] += gram


def _empty_diagnostics() -> dict:
    row = r511._empty_diagnostics()
    row.update({
        "factor_consumer_captures": 0,
        "factor_consumer_captures_expected": 0,
        "factor_consumer_captures_exact": False,
        "attention_corner_evaluations": 0,
        "attention_corner_evaluations_expected": 0,
        "attention_corner_evaluations_exact": False,
        "mlp_corner_evaluations": 0,
        "mlp_corner_evaluations_expected": 0,
        "mlp_corner_evaluations_exact": False,
        "removed_attention_corner_replay_max_abs": 0.0,
        "intact_attention_corner_replay_max_abs": 0.0,
        "attention_mobius_relative_squared": 0.0,
        "attention_numerical_remainder_rms_ratio": 0.0,
        "mlp_deployed_branch_sum_relative_squared": 0.0,
        "mlp_removed_corner_replay_max_abs": 0.0,
        "mlp_intact_corner_replay_max_abs": 0.0,
    })
    return row


def _update_exact_diagnostics(total: dict, current: dict) -> None:
    for key in (
        "removed_attention_corner_replay_max_abs",
        "intact_attention_corner_replay_max_abs",
        "attention_mobius_relative_squared",
        "attention_numerical_remainder_rms_ratio",
        "mlp_deployed_branch_sum_relative_squared",
        "mlp_removed_corner_replay_max_abs",
        "mlp_intact_corner_replay_max_abs",
    ):
        total[key] = max(total[key], current[key])


@torch.no_grad()
def collect_terms(model, rows, task_masks, circuit_masks, circuit_tags, scales, bounds):
    lo, hi, split = bounds
    documents = hi - lo
    task = torch.zeros(r511.N_ACTIONS, len(r511.ARMS), documents, len(TASK_CELLS),
                       dtype=torch.float64)
    counts = torch.zeros(documents, len(TASK_CELLS), dtype=torch.float64)
    base_task = torch.zeros_like(counts)
    circuit_sums = torch.zeros(
        r511.N_ACTIONS, len(r511.ARMS), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    statistics = _empty_statistics()
    diagnostics = _empty_diagnostics()
    device = next(model.parameters()).device
    mlp10 = model.transformer.h[parent.TARGET].mlp

    for start in range(lo, hi, parent.BATCH):
        stop = start + parent.BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in TASK_CELLS}
        copy_mask = masks["all_positive"].to(device)
        half = "half0" if start < split else "half1"

        direct_logits, _, direct_diag, _ = parent._forward(
            model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        r511.r510.r509._update_diagnostics(diagnostics, direct_diag)
        absent_logits, absent, absent_diag, _ = r511._captured_forward(
            model, tokens, scales, action="P", absent=True)
        diagnostics["calls"]["analytical"] += 1
        diagnostics["hooks"] += 1
        r511.r510.r509._update_diagnostics(diagnostics, absent_diag)
        base_task[local:local + len(batch_rows)] = parent._task_sums(
            parent._nll(absent_logits, batch_rows).detach().cpu().unsqueeze(0), masks)[0]

        term_vectors = [[None] * len(TERM_NAMES) for _ in range(N_LOCAL_NODES)]
        total_vectors = [[None] * 2 for _ in range(N_LOCAL_NODES)]
        source_vectors = [None] * N_LOCAL_NODES
        action_nll = []
        for action_index, source in enumerate(parent.SOURCES):
            current_result, current_consumer = factor_consumer_call(
                model, lambda source=source: r511._captured_forward(
                    model, tokens, scales, action=source))
            logits, current, current_diag, _ = current_result
            diagnostics["calls"]["analytical"] += 1
            diagnostics["hooks"] += 1
            diagnostics["factor_consumer_captures"] += 1
            r511.r510.r509._update_diagnostics(diagnostics, current_diag)
            parent._score_delta_closure(diagnostics, current, absent)
            if source == "N":
                diagnostics["native_replay_logit_max_abs"] = max(
                    diagnostics["native_replay_logit_max_abs"],
                    float((logits.float() - direct_logits.float()).abs().max()))
                diagnostics["native_replay_relative_squared"] = max(
                    diagnostics["native_replay_relative_squared"],
                    r511._relative_squared(direct_logits, logits))
            branches, branch_diag = r511.deployed_branches(mlp10, absent, current)
            diagnostics["four_corner_replays"] += 1
            r511._update_branch_diagnostics(diagnostics, branch_diag)
            nll_rows = [parent._nll(logits, batch_rows).detach().cpu()]
            for subset_index in range(r511.N_SUBSETS):
                delta10 = r511.subset_output(branches, subset_index)
                replacement = current["deployed_write"].float() - delta10
                removed_result, removed_consumer = factor_consumer_call(
                    model, lambda source=source, replacement=replacement: (
                        parent.score_parent.run_forward(
                            model, tokens, action=source, scales=scales,
                            patch_writes={"m10": replacement.to(current["deployed_write"].dtype)})))
                edited_logits, _captures, patch_diag, patch_audit = removed_result
                diagnostics["calls"]["analytical"] += 1
                diagnostics["factor_consumer_captures"] += 1
                diagnostics["subset_patches"] += patch_audit["patches"]
                edit_rms = patch_diag["patch_rms_max"]
                diagnostics["zero_term_edits"] += int(edit_rms <= 0)
                if edit_rms > 0:
                    diagnostics["minimum_nonzero_term_edit_rms"] = min(
                        diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
                nll_rows.append(parent._nll(edited_logits, batch_rows).detach().cpu())
                if subset_index not in SELECTED_SUBSETS:
                    continue
                selected = SELECTED_SUBSETS.index(subset_index)
                node = local_node(action_index, selected)
                terms, term_diag = exact_terms(model, removed_consumer, current_consumer)
                diagnostics["attention_corner_evaluations"] += 32
                diagnostics["mlp_corner_evaluations"] += 4
                _update_exact_diagnostics(diagnostics, term_diag)
                term_vectors[node] = [term[copy_mask].reshape(-1).float().cpu()
                                      for term in terms]
                total_vectors[node][0] = (
                    current_consumer["a11"].float() - removed_consumer["a11"].float()
                )[copy_mask].reshape(-1).cpu()
                total_vectors[node][1] = (
                    current_consumer["m11"].float() - removed_consumer["m11"].float()
                )[copy_mask].reshape(-1).cpu()
                source_vectors[node] = delta10[copy_mask].reshape(-1).float().cpu()
                del terms
            task[action_index, :, local:local + len(batch_rows)] = parent._task_sums(
                torch.stack(nll_rows), masks)
            action_nll.append(torch.stack(nll_rows))

        if any(value is None for rows_ in term_vectors for value in rows_) \
                or any(value is None for rows_ in total_vectors for value in rows_) \
                or any(value is None for value in source_vectors):
            raise RuntimeError("term response collection incomplete")
        _update_statistics(statistics, term_vectors, total_vectors, source_vectors, half)
        counts[local:local + len(batch_rows)] = torch.stack(
            [masks[cell].sum(1).double() for cell in TASK_CELLS], -1)
        matrix, observed = parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        circuit_counts += observed
        for action_index, nll_stack in enumerate(action_nll):
            circuit_sums[action_index] += torch.matmul(
                nll_stack.view(len(r511.ARMS), -1).double(), matrix.T,
            ).view(len(r511.ARMS), 2, 2, len(circuit_tags))

    batches = documents // parent.BATCH
    diagnostics["calls_expected"] = {
        "direct": batches,
        "analytical": batches * (1 + r511.N_ACTIONS * (1 + r511.N_SUBSETS)),
    }
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["hooks_expected"] = batches * (1 + r511.N_ACTIONS)
    diagnostics["hooks_exact"] = diagnostics["hooks"] == diagnostics["hooks_expected"]
    diagnostics["four_corner_replays_expected"] = batches * r511.N_ACTIONS
    diagnostics["four_corner_replays_exact"] = (
        diagnostics["four_corner_replays"] == diagnostics["four_corner_replays_expected"])
    diagnostics["subset_patches_expected"] = batches * r511.N_ACTIONS * r511.N_SUBSETS
    diagnostics["subset_patches_exact"] = (
        diagnostics["subset_patches"] == diagnostics["subset_patches_expected"])
    diagnostics["patches"] = diagnostics["subset_patches"]
    diagnostics["patches_expected"] = diagnostics["subset_patches_expected"]
    diagnostics["patches_exact"] = diagnostics["subset_patches_exact"]
    diagnostics["factor_consumer_captures_expected"] = batches * r511.N_ACTIONS * (1 + r511.N_SUBSETS)
    diagnostics["factor_consumer_captures_exact"] = (
        diagnostics["factor_consumer_captures"]
        == diagnostics["factor_consumer_captures_expected"])
    diagnostics["attention_corner_evaluations_expected"] = batches * N_LOCAL_NODES * 32
    diagnostics["attention_corner_evaluations_exact"] = (
        diagnostics["attention_corner_evaluations"]
        == diagnostics["attention_corner_evaluations_expected"])
    diagnostics["mlp_corner_evaluations_expected"] = batches * N_LOCAL_NODES * 4
    diagnostics["mlp_corner_evaluations_exact"] = (
        diagnostics["mlp_corner_evaluations"] == diagnostics["mlp_corner_evaluations_expected"])
    return {
        "bounds": bounds, "arms": r511.ARMS, "task": task,
        "task_counts": counts, "base_task": base_task, "source_task": task[:, 0],
        "circuit_tags": tuple(circuit_tags), "circuit_sums": circuit_sums,
        "circuit_counts": circuit_counts, "statistics": statistics,
        "diagnostics": diagnostics,
    }


def _relation_metrics(collection: dict, term_index: int, left: int, right: int,
                      beta: float) -> dict:
    result = {"beta_left_from_right": beta,
              "scale_holds": bool(.25 <= abs(beta) <= 4), "windows": {}}
    site_index = 0 if term_site(term_index) == "a11" else 1
    for window in WINDOWS:
        gram = collection["statistics"]["term_gram"][window][term_index]
        total = collection["statistics"]["total_gram"][window][site_index]
        ll, rr, lr = float(gram[left, left]), float(gram[right, right]), float(gram[left, right])
        total_l, total_r = float(total[left, left]), float(total[right, right])
        cosine = math.copysign(1.0, beta) * lr / max(math.sqrt(max(ll * rr, 0.0)), 1e-30)
        inverse = 1.0 / beta if beta else math.inf
        forward = math.sqrt(max(ll - 2 * beta * lr + beta * beta * rr, 0.0)
                            / max(ll, 1e-30))
        backward = math.sqrt(max(rr - 2 * inverse * lr + inverse * inverse * ll, 0.0)
                             / max(rr, 1e-30)) if math.isfinite(inverse) else math.inf
        fractions = [math.sqrt(max(ll, 0.0) / max(total_l, 1e-30)),
                     math.sqrt(max(rr, 0.0) / max(total_r, 1e-30))]
        result["windows"][window] = {
            "cosine": cosine,
            "left_from_right_relative_residual": forward,
            "right_from_left_relative_residual": backward,
            "term_to_complete_response_rms": fractions,
            "material": bool(min(fractions) >= .10),
        }
    return result


def _source_metrics(collection: dict, left: int, right: int, beta: float) -> dict:
    result = {"beta_left_from_right": beta,
              "scale_holds": bool(.25 <= abs(beta) <= 4), "windows": {}}
    for window in WINDOWS:
        gram = collection["statistics"]["source_gram"][window]
        ll, rr, lr = float(gram[left, left]), float(gram[right, right]), float(gram[left, right])
        inverse = 1.0 / beta if beta else math.inf
        result["windows"][window] = {
            "cosine": math.copysign(1.0, beta) * lr / max(math.sqrt(max(ll * rr, 0.0)), 1e-30),
            "left_from_right_relative_residual": math.sqrt(
                max(ll - 2 * beta * lr + beta * beta * rr, 0.0) / max(ll, 1e-30)),
            "right_from_left_relative_residual": math.sqrt(
                max(rr - 2 * inverse * lr + inverse * inverse * ll, 0.0) / max(rr, 1e-30)
            ) if math.isfinite(inverse) else math.inf,
            "material": bool(ll > 0 and rr > 0),
        }
    return result


def _metrics_hold(metrics: dict, cosine: float, residual: float) -> bool:
    return bool(metrics["scale_holds"] and all(
        metrics["windows"][window]["material"]
        and metrics["windows"][window]["cosine"] >= cosine
        and max(metrics["windows"][window]["left_from_right_relative_residual"],
                metrics["windows"][window]["right_from_left_relative_residual"]) <= residual
        for window in ("half0", "half1")
    ))


def _fit_beta(collection: dict, term_index: int, left: int, right: int) -> float:
    gram = collection["statistics"]["term_gram"]["half0"][term_index]
    return float(gram[left, right] / gram[right, right].clamp_min(1e-30))


def _fit_source_beta(collection: dict, left: int, right: int) -> float:
    gram = collection["statistics"]["source_gram"]["half0"]
    return float(gram[left, right] / gram[right, right].clamp_min(1e-30))


def reproduce_source_relations(collection: dict) -> tuple[list[str], dict]:
    passing, checks = [], {}
    for selected_subset, subset_index in enumerate(SELECTED_SUBSETS):
        subset_name = r511.SUBSET_NAMES[subset_index]
        for left_action, right_action in RELATION_ACTIONS:
            left = local_node(left_action, selected_subset)
            right = local_node(right_action, selected_subset)
            beta = _fit_source_beta(collection, left, right)
            metrics = _source_metrics(collection, left, right, beta)
            name = (f"{parent.SOURCES[left_action]}::{subset_name} <-> "
                    f"{parent.SOURCES[right_action]}::{subset_name}")
            holds = _metrics_hold(metrics, .85, .55)
            checks[name] = {"metrics": metrics, "holds": holds}
            if holds:
                passing.append(name)
    return passing, checks


def discover_groups(collection: dict) -> tuple[list[dict], dict]:
    candidates, checks = [], {}
    for selected_subset, subset_index in enumerate(SELECTED_SUBSETS):
        subset_name = r511.SUBSET_NAMES[subset_index]
        for term_index, term_name in enumerate(TERM_NAMES):
            group = {
                "selected_subset": selected_subset,
                "subset_index": subset_index,
                "subset_name": subset_name,
                "term_index": term_index,
                "term_name": term_name,
                "site": term_site(term_index),
                "relations": {},
            }
            holds = True
            for relation_index, (left_action, right_action) in enumerate(RELATION_ACTIONS):
                left, right = (local_node(left_action, selected_subset),
                               local_node(right_action, selected_subset))
                beta = _fit_beta(collection, term_index, left, right)
                metrics = _relation_metrics(collection, term_index, left, right, beta)
                relation_holds = _metrics_hold(metrics, .85, .55)
                group["relations"][RELATION_NAMES[relation_index]] = {
                    "left_action": left_action, "right_action": right_action,
                    "left_node": left, "right_node": right,
                    "beta_left_from_right": beta,
                    "metrics": metrics, "holds": relation_holds,
                }
                holds &= relation_holds
            group["holds"] = bool(holds)
            key = f"{subset_name} @ {term_name}"
            checks[key] = group
            if holds:
                candidates.append(group)
    return candidates, {
        "fixed_groups": len(SELECTED_SUBSETS) * len(TERM_NAMES),
        "relation_term_tests": len(SELECTED_SUBSETS) * len(TERM_NAMES) * len(RELATION_ACTIONS),
        "candidate_count": len(candidates), "checks": checks,
    }


def confirm_groups(collection: dict, candidates: list[dict]) -> tuple[list[dict], dict]:
    confirmed, checks = [], {}
    for candidate in candidates:
        row = {"relations": {}, "holds": True}
        for relation_name, relation in candidate["relations"].items():
            metrics = _relation_metrics(
                collection, candidate["term_index"], relation["left_node"],
                relation["right_node"], relation["beta_left_from_right"])
            holds = _metrics_hold(metrics, .75, .65)
            row["relations"][relation_name] = {"metrics": metrics, "holds": holds}
            row["holds"] &= holds
        key = f"{candidate['subset_name']} @ {candidate['term_name']}"
        checks[key] = row
        if row["holds"]:
            confirmed.append(candidate)
    return confirmed, checks


def _window_slice(bounds, window: str) -> slice:
    lo, hi, split = bounds
    if window == "half0":
        return slice(0, split - lo)
    if window == "half1":
        return slice(split - lo, hi - lo)
    return slice(0, hi - lo)


def _physical_empty(candidate_count: int, documents: int, circuit_count: int) -> dict:
    return {
        "bounds": None,
        "intact_task": torch.zeros(r511.N_ACTIONS, documents, len(TASK_CELLS), dtype=torch.float64),
        "removal_task": torch.zeros(candidate_count, r511.N_ACTIONS, documents,
                                     len(TASK_CELLS), dtype=torch.float64),
        "substitution_task": torch.zeros(candidate_count, 6, documents,
                                          len(TASK_CELLS), dtype=torch.float64),
        "task_counts": torch.zeros(documents, len(TASK_CELLS), dtype=torch.float64),
        "intact_circuit_sums": torch.zeros(r511.N_ACTIONS, 2, 2, circuit_count,
                                            dtype=torch.float64),
        "removal_circuit_sums": torch.zeros(candidate_count, r511.N_ACTIONS, 2, 2,
                                             circuit_count, dtype=torch.float64),
        "substitution_circuit_sums": torch.zeros(candidate_count, 6, 2, 2,
                                                  circuit_count, dtype=torch.float64),
        "circuit_counts": torch.zeros(2, 2, circuit_count, dtype=torch.float64),
    }


def _patch_write(consumer: dict, term: torch.Tensor, site: str) -> torch.Tensor:
    return consumer[site].float() - term.float()


def _substitution_write(target: dict, target_term: torch.Tensor,
                        donor_term: torch.Tensor, site: str, beta: float) -> torch.Tensor:
    return target[site].float() - target_term.float() + beta * donor_term.float()


@torch.no_grad()
def collect_physical(model, rows, task_masks, circuit_masks, circuit_tags,
                     scales, bounds, candidates):
    lo, hi, _split = bounds
    data = _physical_empty(len(candidates), hi - lo, len(circuit_tags))
    data["bounds"] = bounds
    diagnostics = {
        "calls": 0, "calls_expected": 0, "calls_exact": False,
        "branch_patches": 0, "branch_patches_expected": 0,
        "consumer_patches": 0, "consumer_patches_expected": 0,
        "patches_exact": False, "zero_patch_edits": 0,
        "minimum_patch_rms": math.inf, "maximum_patch_capture_error": 0.0,
        "attention_corner_evaluations": 0, "mlp_corner_evaluations": 0,
    }
    device = next(model.parameters()).device
    mlp10 = model.transformer.h[parent.TARGET].mlp
    needed = {(candidate["selected_subset"], candidate["term_index"])
              for candidate in candidates}

    for start in range(lo, hi, parent.BATCH):
        stop, local = start + parent.BATCH, start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in TASK_CELLS}
        absent_result, _ = factor_consumer_call(
            model, lambda: r511._captured_forward(model, tokens, scales, action="P", absent=True))
        _absent_logits, absent, _absent_diag, _ = absent_result
        diagnostics["calls"] += 1

        action_data = {}
        for action_index, source in enumerate(parent.SOURCES):
            current_result, current_consumer = factor_consumer_call(
                model, lambda source=source: r511._captured_forward(
                    model, tokens, scales, action=source))
            logits, current, _diag, _audit = current_result
            diagnostics["calls"] += 1
            nll = parent._nll(logits, batch_rows).detach().cpu()
            data["intact_task"][action_index, local:local + len(batch_rows)] = parent._task_sums(
                nll.unsqueeze(0), masks)[0]
            branches, _ = r511.deployed_branches(mlp10, absent, current)
            action_data[action_index] = {"current": current_consumer, "terms": {}, "nll": nll}
            for selected_subset, subset_index in enumerate(SELECTED_SUBSETS):
                delta10 = r511.subset_output(branches, subset_index)
                replacement = current["deployed_write"].float() - delta10
                removed_result, removed_consumer = factor_consumer_call(
                    model, lambda source=source, replacement=replacement: parent.score_parent.run_forward(
                        model, tokens, action=source, scales=scales,
                        patch_writes={"m10": replacement.to(current["deployed_write"].dtype)}))
                _removed_logits, _captures, diag, audit = removed_result
                diagnostics["calls"] += 1
                diagnostics["branch_patches"] += audit["patches"]
                patch_rms = diag["patch_rms_max"]
                diagnostics["zero_patch_edits"] += int(patch_rms <= 0)
                diagnostics["minimum_patch_rms"] = min(
                    diagnostics["minimum_patch_rms"], patch_rms if patch_rms > 0 else math.inf)
                relevant_terms = [term_index for ss, term_index in needed if ss == selected_subset]
                if relevant_terms:
                    terms, _ = exact_terms(model, removed_consumer, current_consumer)
                    diagnostics["attention_corner_evaluations"] += 32
                    diagnostics["mlp_corner_evaluations"] += 4
                    for term_index in relevant_terms:
                        action_data[action_index]["terms"][(selected_subset, term_index)] = terms[term_index]

        data["task_counts"][local:local + len(batch_rows)] = torch.stack(
            [masks[cell].sum(1).double() for cell in TASK_CELLS], -1)
        matrix, observed = parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        data["circuit_counts"] += observed
        for action_index in range(r511.N_ACTIONS):
            nll = action_data[action_index]["nll"]
            data["intact_circuit_sums"][action_index] += torch.matmul(
                nll.reshape(1, -1).double(), matrix.T).view(2, 2, len(circuit_tags))

        for candidate_index, candidate in enumerate(candidates):
            key = (candidate["selected_subset"], candidate["term_index"])
            site = candidate["site"]
            for action_index in range(r511.N_ACTIONS):
                target = action_data[action_index]
                replacement = _patch_write(target["current"], target["terms"][key], site)
                logits, captures, diag, audit = parent.score_parent.run_forward(
                    model, tokens, action=parent.SOURCES[action_index], scales=scales,
                    patch_writes={site: replacement.to(target["current"][site].dtype)},
                    capture_keys=(site,))
                diagnostics["calls"] += 1
                diagnostics["consumer_patches"] += audit["patches"]
                diagnostics["maximum_patch_capture_error"] = max(
                    diagnostics["maximum_patch_capture_error"],
                    float((captures[site] - replacement.to(captures[site].dtype)).float().abs().max()))
                patch_rms = diag["patch_rms_max"]
                diagnostics["zero_patch_edits"] += int(patch_rms <= 0)
                diagnostics["minimum_patch_rms"] = min(
                    diagnostics["minimum_patch_rms"], patch_rms if patch_rms > 0 else math.inf)
                nll = parent._nll(logits, batch_rows).detach().cpu()
                data["removal_task"][candidate_index, action_index,
                                     local:local + len(batch_rows)] = parent._task_sums(
                                         nll.unsqueeze(0), masks)[0]
                data["removal_circuit_sums"][candidate_index, action_index] += torch.matmul(
                    nll.reshape(1, -1).double(), matrix.T).view(2, 2, len(circuit_tags))

            direction_index = 0
            for relation_name in RELATION_NAMES:
                relation = candidate["relations"][relation_name]
                for target_action, donor_action, beta in (
                    (relation["left_action"], relation["right_action"],
                     relation["beta_left_from_right"]),
                    (relation["right_action"], relation["left_action"],
                     1.0 / relation["beta_left_from_right"]),
                ):
                    target, donor = action_data[target_action], action_data[donor_action]
                    replacement = _substitution_write(
                        target["current"], target["terms"][key], donor["terms"][key], site, beta)
                    logits, captures, diag, audit = parent.score_parent.run_forward(
                        model, tokens, action=parent.SOURCES[target_action], scales=scales,
                        patch_writes={site: replacement.to(target["current"][site].dtype)},
                        capture_keys=(site,))
                    diagnostics["calls"] += 1
                    diagnostics["consumer_patches"] += audit["patches"]
                    diagnostics["maximum_patch_capture_error"] = max(
                        diagnostics["maximum_patch_capture_error"],
                        float((captures[site] - replacement.to(captures[site].dtype)).float().abs().max()))
                    patch_rms = diag["patch_rms_max"]
                    diagnostics["zero_patch_edits"] += int(patch_rms <= 0)
                    diagnostics["minimum_patch_rms"] = min(
                        diagnostics["minimum_patch_rms"], patch_rms if patch_rms > 0 else math.inf)
                    nll = parent._nll(logits, batch_rows).detach().cpu()
                    data["substitution_task"][candidate_index, direction_index,
                                               local:local + len(batch_rows)] = parent._task_sums(
                                                   nll.unsqueeze(0), masks)[0]
                    data["substitution_circuit_sums"][candidate_index, direction_index] += torch.matmul(
                        nll.reshape(1, -1).double(), matrix.T).view(2, 2, len(circuit_tags))
                    direction_index += 1

    batches = (hi - lo) // parent.BATCH
    diagnostics["calls_expected"] = batches * (1 + r511.N_ACTIONS
                                                 + r511.N_ACTIONS * len(SELECTED_SUBSETS)
                                                 + 10 * len(candidates))
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["branch_patches_expected"] = batches * r511.N_ACTIONS * len(SELECTED_SUBSETS)
    diagnostics["consumer_patches_expected"] = batches * 10 * len(candidates)
    diagnostics["patches_exact"] = bool(
        diagnostics["branch_patches"] == diagnostics["branch_patches_expected"]
        and diagnostics["consumer_patches"] == diagnostics["consumer_patches_expected"])
    data["diagnostics"] = diagnostics
    return data


def mismatch_diagnostics(collection: dict) -> dict:
    output = {}
    for selected_subset, subset_index in enumerate(SELECTED_SUBSETS):
        subset_name = r511.SUBSET_NAMES[subset_index]
        for relation_index, (left_action, right_action) in enumerate(RELATION_ACTIONS):
            left = local_node(left_action, selected_subset)
            right = local_node(right_action, selected_subset)
            beta = _fit_source_beta(collection, left, right)
            relation_name = f"{subset_name}::{RELATION_NAMES[relation_index]}"
            output[relation_name] = {"beta_source": beta, "sites": {}}
            for site_index, site in enumerate(("a11", "m11")):
                terms = range(31) if site == "a11" else range(31, 34)
                site_row = {}
                for window in WINDOWS:
                    total = collection["statistics"]["total_gram"][window][site_index]
                    total_norm_sq = float(total[left, left] - 2 * beta * total[left, right]
                                          + beta * beta * total[right, right])
                    shares = {}
                    for term_index in terms:
                        cross = collection["statistics"]["term_total_cross"][window][term_index]
                        inner = float(cross[left, left] - beta * cross[left, right]
                                      - beta * cross[right, left]
                                      + beta * beta * cross[right, right])
                        shares[TERM_NAMES[term_index]] = inner / max(total_norm_sq, 1e-30)
                    site_row[window] = {
                        "complete_mismatch_norm_squared": max(total_norm_sq, 0.0),
                        "term_signed_inner_product_fractions": shares,
                        "numerical_remainder_signed_fraction": 1.0 - sum(shares.values()),
                    }
                output[relation_name]["sites"][site] = site_row
    return output


def _task_effect(data: dict, candidate_index: int, action: int,
                 direction_index: int | None, window: str) -> torch.Tensor:
    sl = _window_slice(data["bounds"], window)
    if direction_index is None:
        edited = data["removal_task"][candidate_index, action, sl]
    else:
        edited = data["substitution_task"][candidate_index, direction_index, sl]
    numerator = (edited - data["intact_task"][action, sl]).sum(0)
    denominator = data["task_counts"][sl].sum(0).clamp_min(1)
    effect = numerator / denominator
    return effect[[TASK_CELLS.index(cell) for cell in COPY_CELLS]]


def _off_target(data: dict, candidate_index: int, action: int,
                direction_index: int, window: str) -> float:
    sl = _window_slice(data["bounds"], window)
    index = TASK_CELLS.index("off_target")
    numerator = (data["substitution_task"][candidate_index, direction_index, sl, index]
                 - data["intact_task"][action, sl, index]).sum()
    denominator = data["task_counts"][sl, index].sum().clamp_min(1)
    return float(numerator / denominator)


def _circuit_effect(data: dict, candidate_index: int, action: int,
                    direction_index: int | None, window: str) -> torch.Tensor:
    if window == "pooled":
        intact = data["intact_circuit_sums"][action].sum(0)
        counts = data["circuit_counts"].sum(0)
        edited = (data["removal_circuit_sums"][candidate_index, action].sum(0)
                  if direction_index is None
                  else data["substitution_circuit_sums"][candidate_index, direction_index].sum(0))
    else:
        half = 0 if window == "half0" else 1
        intact = data["intact_circuit_sums"][action, half]
        counts = data["circuit_counts"][half]
        edited = (data["removal_circuit_sums"][candidate_index, action, half]
                  if direction_index is None
                  else data["substitution_circuit_sums"][candidate_index, direction_index, half])
    effect = (edited - intact) / counts.clamp_min(1)
    return effect[0] - effect[1]


def score_physical(data: dict, candidates: list[dict]) -> tuple[list[dict], dict]:
    passing, checks = [], {}
    for candidate_index, candidate in enumerate(candidates):
        group = {"relations": {}, "holds": True}
        direction_index = 0
        for relation_name in RELATION_NAMES:
            relation = candidate["relations"][relation_name]
            relation_row = {"directions": {}, "holds": True}
            for side, target_action in (
                ("left_from_right", relation["left_action"]),
                ("right_from_left", relation["right_action"]),
            ):
                side_row = {"windows": {}, "holds": True}
                for window in WINDOWS:
                    removal_task = _task_effect(
                        data, candidate_index, target_action, None, window)
                    substitution_task = _task_effect(
                        data, candidate_index, target_action, direction_index, window)
                    removal_circuit = _circuit_effect(
                        data, candidate_index, target_action, None, window)
                    substitution_circuit = _circuit_effect(
                        data, candidate_index, target_action, direction_index, window)
                    task_norm = float(torch.linalg.vector_norm(removal_task))
                    circuit_rms = float(removal_circuit.square().mean().sqrt())
                    task_ratio = float(torch.linalg.vector_norm(substitution_task)
                                       / torch.linalg.vector_norm(removal_task).clamp_min(1e-30))
                    circuit_ratio = float(torch.linalg.vector_norm(substitution_circuit)
                                          / torch.linalg.vector_norm(removal_circuit).clamp_min(1e-30))
                    off_target = _off_target(
                        data, candidate_index, target_action, direction_index, window)
                    holds = bool(task_norm >= .00025 and circuit_rms >= .0005
                                 and task_ratio <= .50 and circuit_ratio <= .50
                                 and abs(off_target) <= .002)
                    side_row["windows"][window] = {
                        "term_removal_copy_task_norm_nat": task_norm,
                        "term_removal_heldout_circuit_rms_nat": circuit_rms,
                        "substitution_to_removal_copy_task_norm_ratio": task_ratio,
                        "substitution_to_removal_circuit_norm_ratio": circuit_ratio,
                        "off_target_ce_damage_nat": off_target,
                        "holds": holds,
                    }
                    if window in ("half0", "half1"):
                        side_row["holds"] &= holds
                relation_row["directions"][side] = side_row
                relation_row["holds"] &= side_row["holds"]
                direction_index += 1
            group["relations"][relation_name] = relation_row
            group["holds"] &= relation_row["holds"]
        key = f"{candidate['subset_name']} @ {candidate['term_name']}"
        checks[key] = group
        if group["holds"]:
            passing.append(candidate)
    return passing, checks


def collection_instrument(collection: dict) -> bool:
    diagnostics = collection["diagnostics"]
    statistics = collection["statistics"]
    return bool(
        r511._instrument(collection)
        and diagnostics["factor_consumer_captures_exact"]
        and diagnostics["attention_corner_evaluations_exact"]
        and diagnostics["mlp_corner_evaluations_exact"]
        and diagnostics["removed_attention_corner_replay_max_abs"] == 0.0
        and diagnostics["intact_attention_corner_replay_max_abs"] == 0.0
        and diagnostics["attention_numerical_remainder_rms_ratio"] <= .01
        and diagnostics["mlp_deployed_branch_sum_relative_squared"] <= 1e-12
        and all(torch.isfinite(statistics[key][window]).all()
                for key in ("term_gram", "total_gram", "term_total_cross", "source_gram")
                for window in WINDOWS)
    )


def physical_instrument(data: dict) -> bool:
    diagnostics = data["diagnostics"]
    return bool(
        diagnostics["calls_exact"] and diagnostics["patches_exact"]
        and diagnostics["zero_patch_edits"] == 0
        and diagnostics["minimum_patch_rms"] > 0
        and diagnostics["maximum_patch_capture_error"] == 0.0
    )


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(R512_RESULT.read_text())
    if not (result.get("pred_a_exact_live_consumer_instrument") is True
            and result.get("pred_b_consumer_local_discovery_relation") is False
            and result.get("analysis", {}).get("discovery_summary", {}).get("candidate_count") == 0
            and result.get("next_step")
            == "split_attention11_q_k_q2_k2_value_and_mlp11_left_right_product_finitely"):
        raise RuntimeError("rung512 route changed")
    source_relations = sorted({
        key.split(" @ ")[0]
        for key, row in result["analysis"]["discovery_summary"]["checks"].items()
        if row["source_holds"]
    })
    if source_relations != sorted(SOURCE_RELATION_NAMES):
        raise RuntimeError(f"rung512 source relation list changed: {source_relations}")
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = \
        r512.validate_inputs()
    return rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, {
        **metadata,
        "rung512_result_sha256": sha256(R512_RESULT),
        "rung512_bundle_sha256": sha256(R512_BUNDLE),
        "source_relations": source_relations,
        "term_names": list(TERM_NAMES),
        "fixed_groups": 204,
        "relation_term_tests": 612,
    }


def _toy_collection(seed: int = 513) -> dict:
    generator = torch.Generator().manual_seed(seed)
    statistics = _empty_statistics()
    for window in WINDOWS:
        term_values = torch.randn(len(TERM_NAMES), N_LOCAL_NODES, 96, generator=generator)
        total_values = torch.randn(2, N_LOCAL_NODES, 96, generator=generator)
        source_values = torch.randn(N_LOCAL_NODES, 96, generator=generator)
        # Plant one complete four-action equivalence for L at A11{Q}.
        template = torch.randn(96, generator=generator)
        for action in range(4):
            term_values[0, local_node(action, 0)] = (action + 1) * template
        total_values[0, :, :].mul_(8.0)
        for term_index in range(len(TERM_NAMES)):
            statistics["term_gram"][window][term_index] = (
                term_values[term_index].double() @ term_values[term_index].double().T)
            site = 0 if term_index < 31 else 1
            statistics["term_total_cross"][window][term_index] = (
                term_values[term_index].double() @ total_values[site].double().T)
        for site in range(2):
            statistics["total_gram"][window][site] = (
                total_values[site].double() @ total_values[site].double().T)
        statistics["source_gram"][window] = source_values.double() @ source_values.double().T
    return {"statistics": statistics}


def dry_run() -> None:
    baseline = (2., 3., 5., 7., 11.)
    intact = (13., 17., 19., 23., 29.)
    corners = {
        mask: torch.tensor([math.prod(
            intact[index] if mask & (1 << index) else baseline[index]
            for index in range(5))])
        for mask in range(32)
    }
    terms = mobius_terms(corners)
    assert torch.allclose(sum(terms), corners[31] - corners[0], rtol=1e-6, atol=1e-2)
    toy = _toy_collection()
    candidates, summary = discover_groups(toy)
    planted = [row for row in candidates if row["selected_subset"] == 0 and row["term_index"] == 0]
    assert len(planted) == 1
    confirmed, _ = confirm_groups(toy, planted)
    assert len(confirmed) == 1
    target = {"a11": torch.tensor([3., 4.]), "m11": torch.tensor([5., 6.])}
    removed = _patch_write(target, torch.tensor([1., 2.]), "a11")
    replaced = _substitution_write(
        target, torch.tensor([1., 2.]), torch.tensor([2., 1.]), "a11", .5)
    assert torch.equal(removed, torch.tensor([2., 2.]))
    assert torch.equal(replaced, torch.tensor([3., 2.5]))
    assert len(TERM_NAMES) == 34 and summary["fixed_groups"] == 204
    assert summary["relation_term_tests"] == 612
    assert 4216 + 1798 + 620 * 204 == 132494
    print(json.dumps({
        "status": "dry_run_passed", "rung": 513, "model_loaded": False,
        "outcomes_opened": False, "attention_terms": 31, "mlp_terms": 3,
        "fixed_groups": 204, "relation_term_tests": 612,
        "planted_group_recovered": True, "maximum_conditional_forwards": 132494,
    }, indent=2, sort_keys=True))


@torch.no_grad()
def gpu_smoke() -> None:
    rows, task_masks, circuit_masks, scales, discovery_tags, _validation_tags, _metadata = \
        validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    bounds = (500, 504, 502)
    collection = collect_terms(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, bounds)
    smoke_calibration = parent._calibration(
        collection["base_task"], collection["source_task"],
        collection["task_counts"], bounds)
    native_recovery = smoke_calibration["pooled"]["N"]["recovery_vs_native"]
    candidate = {
        "selected_subset": 0, "subset_index": SELECTED_SUBSETS[0],
        "subset_name": r511.SUBSET_NAMES[SELECTED_SUBSETS[0]],
        "term_index": 0, "term_name": TERM_NAMES[0], "site": "a11",
        "relations": {
            relation_name: {
                "left_action": pair[0], "right_action": pair[1],
                "left_node": local_node(pair[0], 0), "right_node": local_node(pair[1], 0),
                "beta_left_from_right": 1.0,
            }
            for relation_name, pair in zip(RELATION_NAMES, RELATION_ACTIONS)
        },
    }
    physical = collect_physical(
        model, rows, task_masks, circuit_masks, discovery_tags, scales,
        bounds, [candidate])
    checks = {
        "weights": checkpoint.weights_sha256 == facade.WEIGHTS_SHA256,
        "collection": collection_instrument(collection),
        "native_calibration_semantics": .9 <= native_recovery <= 1.1,
        "physical": physical_instrument(physical),
        "all_768_attention_corners": collection["diagnostics"]["attention_corner_evaluations"] == 768,
        "all_96_mlp_corners": collection["diagnostics"]["mlp_corner_evaluations"] == 96,
        "all_28_branch_patches": collection["diagnostics"]["subset_patches"] == 28,
        "all_10_consumer_patches": physical["diagnostics"]["consumer_patches"] == 10,
    }
    passed = all(checks.values())
    print(json.dumps({
        "status": "smoke_passed" if passed else "smoke_failed", "rung": 513,
        "scientific_outcomes_retained": False, "checks": checks,
        "collection_diagnostics": collection["diagnostics"],
        "smoke_calibration": smoke_calibration,
        "physical_diagnostics": physical["diagnostics"],
        "full_forwards": sum(collection["diagnostics"]["calls"].values())
                         + physical["diagnostics"]["calls"],
        "backwards": 0,
    }, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(f"rung513 smoke failed: "
                           f"{sorted(name for name, value in checks.items() if not value)}")


def _bundle_collection(collection: dict) -> dict:
    return {key: value for key, value in collection.items() if key != "diagnostics"}


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        dry_run()
        return
    if os.environ.get("BQLIB_GPU_SMOKE") == "1" or "--gpu-smoke" in sys.argv:
        gpu_smoke()
        return
    started = time.time()
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = \
        validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung513 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    collections = {}
    collections["discovery"] = collect_terms(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, DISCOVERY)
    discovery_calibration = parent._calibration(
        collections["discovery"]["base_task"], collections["discovery"]["source_task"],
        collections["discovery"]["task_counts"], DISCOVERY)
    discovery_calibration_ok = parent.state_parent.calibration_holds(discovery_calibration)
    source_relations, source_checks = reproduce_source_relations(collections["discovery"])
    source_reproduced = source_relations == list(SOURCE_RELATION_NAMES)
    candidates, discovery_summary = discover_groups(collections["discovery"])
    mismatch = mismatch_diagnostics(collections["discovery"])

    confirmed, confirmation_checks = [], {}
    confirmation_calibration, confirmation_calibration_ok = {}, False
    if discovery_calibration_ok and source_reproduced and candidates:
        collections["confirmation"] = collect_terms(
            model, rows, task_masks, circuit_masks, validation_tags, scales, CONFIRMATION)
        confirmation_calibration = parent._calibration(
            collections["confirmation"]["base_task"], collections["confirmation"]["source_task"],
            collections["confirmation"]["task_counts"], CONFIRMATION)
        confirmation_calibration_ok = parent.state_parent.calibration_holds(
            confirmation_calibration)
        confirmed, confirmation_checks = confirm_groups(collections["confirmation"], candidates)

    physical, physical_passing, physical_checks = None, [], {}
    if confirmation_calibration_ok and confirmed:
        physical = collect_physical(
            model, rows, task_masks, circuit_masks, validation_tags, scales,
            CONFIRMATION, confirmed)
        physical_passing, physical_checks = score_physical(physical, confirmed)

    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and discovery_calibration_ok and source_reproduced
        and collection_instrument(collections["discovery"])
        and ("confirmation" not in collections or (
            confirmation_calibration_ok and collection_instrument(collections["confirmation"])))
        and (physical is None or physical_instrument(physical))
    )
    pred_b = bool(pred_a and candidates)
    pred_c = bool(pred_b and confirmation_calibration_ok and confirmed)
    pred_d = bool(pred_c and physical_passing)
    reused_terms = sorted({
        candidate["term_name"] for candidate in physical_passing
        if sum(other["term_name"] == candidate["term_name"]
               and other["subset_name"] != candidate["subset_name"]
               for other in physical_passing) > 0
    })
    pred_e = bool(pred_d and reused_terms)
    strong_null = not (pred_a and pred_b and pred_c and pred_d)
    if not pred_a:
        next_step = "repair_exact_factor_or_patch_instrument_only"
    elif not pred_b:
        next_step = "preregister_sparse_multi_term_mismatch_combinations_with_planted_identifiability_control"
    elif not pred_c:
        next_step = "identify_document_dependent_change_in_fixed_factor_relation"
    elif not pred_d:
        next_step = "split_at_first_downstream_reader_rejecting_the_factor_term_substitution"
    elif not pred_e:
        next_step = "validate_branch_specific_factor_circuit_on_fixed_ood_code"
    else:
        next_step = "validate_reusable_factor_vocabulary_on_ood_code_and_joint_composition"

    bundle_payload = {
        "schema": "rung513_exact_factor_interactions_v1",
        "collections": {name: _bundle_collection(collection)
                        for name, collection in collections.items()},
        "candidates": candidates, "confirmed": confirmed,
        "physical": None if physical is None else {
            key: value for key, value in physical.items() if key != "diagnostics"},
        "physical_passing": physical_passing,
        "raw_tokens_logits_hidden_states_or_weights_included": False,
    }
    torch.save(bundle_payload, BUNDLE)
    result = {
        "status": "complete", "rung": 513,
        "claim_level": "factor_interaction_screen_until_heldout_physical_substitution_passes",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "factor_vocabulary": {"attention": list(ATTENTION_TERMS), "mlp": list(MLP_TERMS)},
        "calibration": {"discovery": discovery_calibration,
                        "confirmation": confirmation_calibration},
        "calibration_holds": {"discovery": discovery_calibration_ok,
                              "confirmation": confirmation_calibration_ok},
        "source_relation_reproduction": {
            "expected": list(SOURCE_RELATION_NAMES), "observed": source_relations,
            "holds": source_reproduced, "checks": source_checks,
        },
        "diagnostics": {name: collection["diagnostics"]
                        for name, collection in collections.items()},
        "physical_diagnostics": None if physical is None else physical["diagnostics"],
        "analysis": {
            "discovery_summary": discovery_summary,
            "discovery_candidates": candidates,
            "signed_mismatch_decomposition": mismatch,
            "confirmation_checks": confirmation_checks,
            "confirmed_groups": confirmed,
            "physical_checks": physical_checks,
            "physical_groups": physical_passing,
            "reused_term_names": reused_terms,
        },
        'pred_a_exact_live_factor_interaction_instrument': pred_a,
        'pred_b_shared_factor_term_discovery': pred_b,
        'pred_c_frozen_factor_relation_predicts_fresh_documents': pred_c,
        'pred_d_bidirectional_physical_factor_substitution': pred_d,
        'pred_e_reused_factor_term_across_branch_subsets': pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "full_forwards": sum(sum(collection["diagnostics"]["calls"].values())
                                 for collection in collections.values())
                             + (0 if physical is None else physical["diagnostics"]["calls"]),
            "backwards": 0,
            "local_attention_corner_evaluations": sum(
                collection["diagnostics"]["attention_corner_evaluations"]
                for collection in collections.values())
                + (0 if physical is None else physical["diagnostics"]["attention_corner_evaluations"]),
            "local_mlp_corner_evaluations": sum(
                collection["diagnostics"]["mlp_corner_evaluations"]
                for collection in collections.values())
                + (0 if physical is None else physical["diagnostics"]["mlp_corner_evaluations"]),
            "fixed_groups": 204, "relation_term_tests": 612,
            "discovery_candidates": len(candidates), "confirmed_groups": len(confirmed),
            "physical_groups": len(physical_passing), "maximum_conditional_forwards": 132494,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started, "next_step": next_step,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 513,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e, "strong_null": strong_null,
        "source_relations_reproduced": source_reproduced,
        "discovery_candidates": len(candidates), "confirmed_groups": len(confirmed),
        "physical_groups": len(physical_passing), "reused_terms": reused_terms,
        "execution_price": result["execution_price"], "next_step": next_step,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
