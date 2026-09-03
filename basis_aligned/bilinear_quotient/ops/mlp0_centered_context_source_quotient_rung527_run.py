#!/usr/bin/env python3
"""RUNG 527 -- exact centered MLP0 context-source interaction quotient.

pred_a: the 20-term centered polynomial and every finite edit are exact/live
pred_b: one to eight downstream-equivalent term pairs transfer across D0/D1
pred_c: a frozen pair transfers to 30 unopened circuits and new documents
pred_d: a held-out pair passes bidirectional physical term substitution
pred_e: at least one physical pair merges different source supports/operations

Strong null: any of A--E fails.  Confirmation and substitution are fail-closed.
Literal price: 23,040 diagnostic reference values, zero deployed values.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping

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
import mlp0_branch_circuit_response_rung481 as r481  # noqa: E402
import mlp0_centered_context_anova_factorial as r400  # noqa: E402
import mlp0_centered_context_source_quotient_rung527_math as qm  # noqa: E402
import mlp0_source_relation_factorial_rung517 as r517  # noqa: E402


RUNNER = Path(__file__).resolve()
PREREG = POLY / "MLP0_CENTERED_CONTEXT_SOURCE_QUOTIENT_RUNG527_PREREGISTRATION.md"
MATH = OPS / "mlp0_centered_context_source_quotient_rung527_math.py"
R400_SOURCE = OPS / "mlp0_centered_context_anova_factorial.py"
R400_RESULT = BQ / "mlp0_centered_context_anova_factorial_results.json"
R401_RESULT = BQ / "mlp0_centered_context_anova_exact_residual_results.json"
R517_SOURCE = OPS / "mlp0_source_relation_factorial_rung517.py"
R517_RESULT = BQ / "mlp0_source_relation_factorial_rung517_results.json"
R481_SOURCE = OPS / "mlp0_branch_circuit_response_rung481.py"
R526_RESULT = BQ / "mlp0_circuit_response_operator_quotient_rung526_results.json"
OUT = BQ / "mlp0_centered_context_source_quotient_rung527_results.json"
SMOKE_OUT = BQ / "mlp0_centered_context_source_quotient_rung527_gpu_smoke_results.json"

FROZEN_SHA256 = {
    PREREG: "1f39a1d38f6b11275ab2021f53be17a7d2837623fac25eaa45b8ca599cee93ab",
    MATH: "b70524d35eaba7e8aea37e3cfd3a4e042a251b03c09059685cb8b8875cf8e833",
    R400_SOURCE: "1495ec13abf80bbd3d0bf33db8c0457e1bc5eab7421bcb1b96a780278d808322",
    R400_RESULT: "101c74a89595b51a40a48c2199422ffa4fe06fcd63d50eebb2b3a6af97e1fc58",
    R401_RESULT: "6650b97c9f5b53714d29f999eff6653bdbc9273c9238e4c10ce607d8d5728277",
    R517_SOURCE: "5d9acfa5798e9d391e6507d5d7136ec498e4f1b42893a372c47c07c7be6bae97",
    R517_RESULT: "c8405a36cab0e8b50d91e3f525bf5a5106a95d2c42447ce9b83ab29378fd8307",
    R481_SOURCE: "ef08017a30ceb0c9e4481198fc1d58c5b0bf8cd37707d2223c42db9eb04f1f44",
    R526_RESULT: "4c60406f67359eab77de991983a6ff9bc756e0a6cd7902201dc0f1b0d0b721ea",
}

D = 1152
TOKENS = 256
BATCH = 4
SCORING = slice(64, 256)
D_BOUNDS = (0, 248, 124)
V_BOUNDS = (500, 1000, 750)
SMOKE_BOUNDS = (0, 32, 16)
PERMUTATION_SEEDS = tuple(range(527_300, 527_316))
REFERENCE_VALUES = 20 * D

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


def validate_dependencies() -> dict[str, str]:
    observed = {}
    for path, expected in FROZEN_SHA256.items():
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen dependency changed: {path}: {actual} != {expected}")
        observed[str(path.relative_to(REPO))] = actual
    r401 = json.loads(R401_RESULT.read_text())
    if not (
        r401.get("rung") == 401
        and r401.get("pred_a_exact_residual_identity_and_live_census") is True
        and r401.get("pred_b_all_physical_arms_reproduce_rung400") is True
        and r401.get("pred_c_inherited_context_outcome_holds_without_bar_change") is True
        and r401.get("null_exact_repair_or_context_stability_fails") is False
    ):
        raise RuntimeError("rung 401 exact context authority changed")
    r517_result = json.loads(R517_RESULT.read_text())
    if not (
        r517_result.get("rung") == 517
        and r517_result.get("pred_a_exact_live_instrument") is True
        and r517_result.get("strong_null") is True
    ):
        raise RuntimeError("rung 517 source-partition authority changed")
    r526 = json.loads(R526_RESULT.read_text())
    if not (
        r526.get("rung") == 526
        and r526.get("pred_a_exact_live_leakage_free_instrument") is True
        and r526.get("pred_b_same_circuit_new_document_transfer") is False
        and r526.get("strong_null") is True
        and r526.get("physical_successor_licensed") is False
    ):
        raise RuntimeError("rung 526 did not license the context-only route")
    return observed


def population():
    rows, circuit_masks, discovery_tags, validation_tags, fit_rows, metadata = r481.validate_inputs()
    if tuple(rows.shape) != (1000, 257):
        raise RuntimeError("circuit population rows changed")
    if len(discovery_tags) != 32 or len(validation_tags) != 30:
        raise RuntimeError("62-circuit partition changed")
    return rows, circuit_masks, list(discovery_tags), list(validation_tags), fit_rows, metadata


@torch.no_grad()
def fit_references(model, fit_rows: torch.Tensor, device: torch.device) -> tuple[dict, dict, torch.Tensor, dict]:
    reference = r400._reference_moments(model, fit_rows, device)
    group_reference = r517.group_reference(model, fit_rows, device)
    block0 = model.transformer.h[0]
    left = block0.mlp.Left.weight.detach().float()
    right = block0.mlp.Right.weight.detach().float()
    down = block0.mlp.Down.weight.detach().float()
    mean_state = reference["token_mean"] + reference["context_mean"]
    quadratic_sum = torch.zeros(qm.N_QUADRATIC, D, dtype=torch.float64)
    linear_sum = torch.zeros(qm.N_LINEAR, D, dtype=torch.float64)
    count = 0
    maximum_partition_error = 0.0
    for start in range(0, len(fit_rows), BATCH):
        tokens = fit_rows[start:start + BATCH, :-1].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        attention_state = F.rms_norm(token_base, (D,))
        split = r517.attention0_source_writes(block0, attention_state, tokens)
        deltas = split["group_writes"] - group_reference["group_means"][:, None, None]
        raw_terms = qm.uncentered_terms(
            deltas.float(), mean_state.float(), left, right, down, reference["gain_mean"])
        quadratic_sum += raw_terms[qm.N_LINEAR:].double().sum((1, 2)).cpu()
        linear_sum += raw_terms[:qm.N_LINEAR].double().sum((1, 2)).cpu()
        count += tokens.shape[0] * tokens.shape[1]
        maximum_partition_error = max(
            maximum_partition_error,
            split["diagnostics"]["semantic_plus_remainder_relative_mse"])
    quadratic_means = (quadratic_sum / count).float().to(device)
    complete_native = reference["gain_mean"] * reference["context_self_mean"]
    semantic = quadratic_means.sum(0)
    diagnostics = {
        "fit_positions": count,
        "source_partition_maximum_relative_squared": maximum_partition_error,
        "linear_fit_mean_max_abs": float((linear_sum / count).abs().max()),
        "quadratic_expectation_sum_relative_squared_vs_native_context": float(
            (semantic.double() - complete_native.double()).square().sum()
            / complete_native.double().square().sum().clamp_min(1e-30)),
        "quadratic_expectation_values": int(quadratic_means.numel()),
    }
    return reference, group_reference, quadratic_means, diagnostics


@torch.no_grad()
def context_terms(model, tokens: torch.Tensor, token_base: torch.Tensor,
                  reference: dict, group_reference: dict,
                  quadratic_means: torch.Tensor) -> dict:
    block0 = model.transformer.h[0]
    left = block0.mlp.Left.weight.detach().float()
    right = block0.mlp.Right.weight.detach().float()
    down = block0.mlp.Down.weight.detach().float()
    attention_state = F.rms_norm(token_base, (D,))
    split = r517.attention0_source_writes(block0, attention_state, tokens)
    deltas = split["group_writes"] - group_reference["group_means"][:, None, None]
    raw_terms = qm.uncentered_terms(
        deltas.float(),
        (reference["token_mean"] + reference["context_mean"]).float(),
        left, right, down, reference["gain_mean"])
    terms = qm.center_terms(raw_terms, quadratic_means)
    normalized = F.rms_norm(token_base + split["native_write"], (D,))
    _retained, branches, _g, _gain, _collinearity = r400._components(
        token_base, split["native_write"], normalized,
        reference, left, right, down)
    parent = branches["C"]
    semantic = terms.sum(0)
    remainder = parent - semantic
    return {
        "terms": terms,
        "parent": parent,
        "remainder": remainder,
        "expected_state": normalized,
        "expected_attention": split["native_write"],
        "source_partition_relative_squared":
            split["diagnostics"]["semantic_plus_remainder_relative_mse"],
    }


def native_forward(model, tokens: torch.Tensor, expected: dict, diagnostics: dict) -> torch.Tensor:
    def attention(event):
        diagnostics["native_attention_calls"] += 1
        return event.block.attn(event.state, event.first_value)

    def mlp(event):
        diagnostics["native_mlp_calls"] += 1
        native = event.block.mlp(event.state)
        if event.site == 0:
            diagnostics["state_replay_max_abs"] = max(
                diagnostics["state_replay_max_abs"],
                float((event.state - expected["expected_state"]).abs().max()),
                float((event.attention_write - expected["expected_attention"]).abs().max()),
            )
        return native

    diagnostics["native_forwards"] += 1
    return facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)


def removal_forward(model, tokens: torch.Tensor, expected: dict,
                    removal: torch.Tensor, diagnostics: dict) -> torch.Tensor:
    def attention(event):
        diagnostics["removal_attention_calls"] += 1
        return event.block.attn(event.state, event.first_value)

    def mlp(event):
        native = event.block.mlp(event.state)
        if event.site != 0:
            diagnostics["removal_other_mlp_calls"] += 1
            return native
        diagnostics["removal_site0_calls"] += 1
        diagnostics["state_replay_max_abs"] = max(
            diagnostics["state_replay_max_abs"],
            float((event.state - expected["expected_state"]).abs().max()),
            float((event.attention_write - expected["expected_attention"]).abs().max()),
        )
        deployed = removal.to(native.dtype)
        edit_rms = float(deployed.float().square().mean().sqrt())
        diagnostics["minimum_term_edit_rms"] = min(
            diagnostics["minimum_term_edit_rms"], edit_rms)
        diagnostics["zero_term_edits"] += int(edit_rms <= 0)
        return native - deployed

    diagnostics["removal_forwards"] += 1
    return facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)


def empty_diagnostics() -> dict:
    return {
        "manual_attention_captures": 0,
        "native_forwards": 0,
        "native_attention_calls": 0,
        "native_mlp_calls": 0,
        "removal_forwards": 0,
        "removal_attention_calls": 0,
        "removal_site0_calls": 0,
        "removal_other_mlp_calls": 0,
        "state_replay_max_abs": 0.0,
        "source_partition_maximum_relative_squared": 0.0,
        "context_closure_num": 0.0,
        "context_closure_den": 0.0,
        "context_energy": [0.0, 0.0],
        "remainder_energy": [0.0, 0.0],
        "minimum_term_edit_rms": float("inf"),
        "zero_term_edits": 0,
    }


def batch_selections(circuit_masks, tags, start: int, stop: int, split: int):
    return r481._batch_selections(circuit_masks, tags, start, stop, split)


@torch.no_grad()
def collect_phase(model, rows: torch.Tensor, circuit_masks: dict, tags: list[str],
                  bounds: tuple[int, int, int], reference: dict,
                  group_reference: dict, quadratic_means: torch.Tensor,
                  *, keep_native_nll: bool = False) -> dict:
    lo, hi, split = bounds
    if any(value % BATCH for value in bounds):
        raise ValueError("phase bounds must align to four-document batches")
    sums = torch.zeros(qm.N_TERMS, 2, 2, len(tags), dtype=torch.float64)
    counts = torch.zeros(2, 2, len(tags), dtype=torch.float64)
    native_store = torch.empty(hi - lo, TOKENS, dtype=torch.float32) \
        if keep_native_nll else None
    diagnostics = empty_diagnostics()
    device = next(model.parameters()).device
    block0 = model.transformer.h[0]
    for start in range(lo, hi, BATCH):
        stop = start + BATCH
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        targets = batch_rows[:, 1:].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        exact = context_terms(
            model, tokens, token_base, reference, group_reference, quadratic_means)
        diagnostics["manual_attention_captures"] += 1
        diagnostics["source_partition_maximum_relative_squared"] = max(
            diagnostics["source_partition_maximum_relative_squared"],
            exact["source_partition_relative_squared"])
        difference = exact["terms"].sum(0) + exact["remainder"] - exact["parent"]
        diagnostics["context_closure_num"] += float(difference.double().square().sum())
        diagnostics["context_closure_den"] += float(exact["parent"].double().square().sum())
        half = 0 if start < split else 1
        diagnostics["context_energy"][half] += float(
            exact["parent"][:, SCORING].double().square().sum())
        diagnostics["remainder_energy"][half] += float(
            exact["remainder"][:, SCORING].double().square().sum())

        native_logits = native_forward(model, tokens, exact, diagnostics)
        native_nll = F.cross_entropy(
            native_logits.reshape(-1, native_logits.shape[-1]), targets.reshape(-1),
            reduction="none").view(BATCH, TOKENS)
        if native_store is not None:
            native_store[start - lo:stop - lo] = native_nll.detach().cpu()
        selections = batch_selections(circuit_masks, tags, start, stop, split)
        for half_index, kind, circuit, selected in selections:
            counts[half_index, kind, circuit] += int(selected.sum())
        for term in range(qm.N_TERMS):
            logits = removal_forward(
                model, tokens, exact, exact["terms"][term], diagnostics)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                reduction="none").view(BATCH, TOKENS)
            delta = nll - native_nll
            for half_index, kind, circuit, selected_cpu in selections:
                sums[term, half_index, kind, circuit] += float(
                    delta[selected_cpu.to(device)].sum())
        del native_logits, native_nll, exact

    batches = (hi - lo) // BATCH
    expected_calls = {
        "manual_attention_captures": batches,
        "native_forwards": batches,
        "native_attention_calls": 18 * batches,
        "native_mlp_calls": 18 * batches,
        "removal_forwards": qm.N_TERMS * batches,
        "removal_attention_calls": qm.N_TERMS * 18 * batches,
        "removal_site0_calls": qm.N_TERMS * batches,
        "removal_other_mlp_calls": qm.N_TERMS * 17 * batches,
    }
    observed_calls = {key: diagnostics[key] for key in expected_calls}
    diagnostics["expected_calls"] = expected_calls
    diagnostics["calls_exact"] = observed_calls == expected_calls
    diagnostics["context_closure_relative_squared"] = (
        diagnostics["context_closure_num"]
        / max(diagnostics["context_closure_den"], 1e-30))
    diagnostics["remainder_energy_fraction"] = [
        diagnostics["remainder_energy"][half]
        / max(diagnostics["context_energy"][half], 1e-30)
        for half in range(2)
    ]
    diagnostics["supports_positive"] = bool((counts > 0).all())
    return {
        "bounds": bounds,
        "tags": tags,
        "sums": sums,
        "counts": counts,
        "native_nll": native_store,
        "diagnostics": diagnostics,
    }


def effect_views(collection: dict) -> dict[str, torch.Tensor]:
    sums = collection["sums"].double()
    counts = collection["counts"].double().clamp_min(1)
    half_means = sums / counts.unsqueeze(0)
    halves = half_means[:, :, 0] - half_means[:, :, 1]
    pooled_sums = sums.sum(1)
    pooled_counts = counts.sum(0).clamp_min(1)
    pooled_means = pooled_sums / pooled_counts.unsqueeze(0)
    pooled = pooled_means[:, 0] - pooled_means[:, 1]
    return {"halves": halves, "pooled": pooled}


def instrument_holds(collection: dict, *, require_support: bool) -> bool:
    d = collection["diagnostics"]
    return bool(
        d["calls_exact"]
        and d["state_replay_max_abs"] == 0.0
        and d["source_partition_maximum_relative_squared"] <= 1e-12
        and d["context_closure_relative_squared"] <= 1e-12
        and max(d["remainder_energy_fraction"]) <= 0.01
        and d["minimum_term_edit_rms"] > 0
        and d["zero_term_edits"] == 0
        and (d["supports_positive"] or not require_support)
    )


def quantile95(values: list[int]) -> float:
    return float(torch.quantile(
        torch.tensor(values, dtype=torch.float64), .95, interpolation="higher"))


def serial_collection(collection: dict) -> dict:
    result = {
        "bounds": list(collection["bounds"]),
        "tags": collection["tags"],
        "sums": collection["sums"].tolist(),
        "counts": collection["counts"].tolist(),
        "diagnostics": collection["diagnostics"],
    }
    if collection["native_nll"] is not None:
        result["native_nll_sha256"] = hashlib.sha256(
            collection["native_nll"].contiguous().numpy().tobytes()).hexdigest()
    return result


def smoke_tags(circuit_masks: dict, discovery_tags: list[str]) -> list[str]:
    lo, hi, split = SMOKE_BOUNDS
    eligible = []
    for tag in discovery_tags:
        mask = circuit_masks[tag]
        good = True
        for left, right in ((lo, split), (split, hi)):
            for kind in ("member", "slice_control"):
                if int(mask[kind].view(1000, TOKENS)[left:right].sum()) <= 0:
                    good = False
        if good:
            eligible.append(tag)
    if len(eligible) < 2:
        raise RuntimeError("no two smoke circuits have support in both halves")
    return eligible[:2]


def smoke_authority() -> dict:
    result = json.loads(SMOKE_OUT.read_text())
    if not (
        result.get("rung") == 527
        and result.get("status") == "gpu_smoke_complete"
        and result.get("smoke_a_exact_live") is True
        and result.get("runner_sha256") == file_sha256(RUNNER)
        and result.get("preregistration_sha256") == FROZEN_SHA256[PREREG]
    ):
        raise RuntimeError("rung 527 GPU smoke authority is absent or stale")
    return {"sha256": file_sha256(SMOKE_OUT), "result": result}


@torch.no_grad()
def collect_substitutions(model, rows: torch.Tensor, circuit_masks: dict,
                          tags: list[str], bounds: tuple[int, int, int],
                          reference: dict, group_reference: dict,
                          quadratic_means: torch.Tensor, candidates: list[dict],
                          native_nll: torch.Tensor) -> dict:
    lo, hi, split = bounds
    directions = []
    for edge, candidate in enumerate(candidates):
        directions.extend((
            {"edge": edge, "target": candidate["left"], "donor": candidate["right"],
             "scale": candidate["beta_left_from_right"], "side": "left_from_right"},
            {"edge": edge, "target": candidate["right"], "donor": candidate["left"],
             "scale": 1.0 / candidate["beta_left_from_right"], "side": "right_from_left"},
        ))
    sums = torch.zeros(len(directions), 2, 2, len(tags), dtype=torch.float64)
    counts = torch.zeros(2, 2, len(tags), dtype=torch.float64)
    diagnostics = empty_diagnostics()
    device = next(model.parameters()).device
    block0 = model.transformer.h[0]
    for start in range(lo, hi, BATCH):
        stop = start + BATCH
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        targets = batch_rows[:, 1:].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        exact = context_terms(
            model, tokens, token_base, reference, group_reference, quadratic_means)
        diagnostics["manual_attention_captures"] += 1
        selections = batch_selections(circuit_masks, tags, start, stop, split)
        for half_index, kind, circuit, selected in selections:
            counts[half_index, kind, circuit] += int(selected.sum())
        baseline = native_nll[start - lo:stop - lo].to(device)
        for direction_index, direction in enumerate(directions):
            removal = direction["scale"] * exact["terms"][direction["donor"]]
            logits = removal_forward(model, tokens, exact, removal, diagnostics)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                reduction="none").view(BATCH, TOKENS)
            delta = nll - baseline
            for half_index, kind, circuit, selected_cpu in selections:
                sums[direction_index, half_index, kind, circuit] += float(
                    delta[selected_cpu.to(device)].sum())
    batches = (hi - lo) // BATCH
    expected_calls = {
        "manual_attention_captures": batches,
        "native_forwards": 0,
        "native_attention_calls": 0,
        "native_mlp_calls": 0,
        "removal_forwards": len(directions) * batches,
        "removal_attention_calls": len(directions) * 18 * batches,
        "removal_site0_calls": len(directions) * batches,
        "removal_other_mlp_calls": len(directions) * 17 * batches,
    }
    observed = {key: diagnostics[key] for key in expected_calls}
    diagnostics["expected_calls"] = expected_calls
    diagnostics["calls_exact"] = observed == expected_calls
    diagnostics["supports_positive"] = bool((counts > 0).all())
    return {
        "bounds": bounds, "tags": tags, "directions": directions,
        "sums": sums, "counts": counts, "diagnostics": diagnostics,
    }


def substitution_effect_views(collection: dict) -> dict[str, torch.Tensor]:
    sums = collection["sums"].double()
    counts = collection["counts"].double().clamp_min(1)
    means = sums / counts.unsqueeze(0)
    halves = means[:, :, 0] - means[:, :, 1]
    pooled_sums = sums.sum(1)
    pooled_counts = counts.sum(0).clamp_min(1)
    pooled_means = pooled_sums / pooled_counts.unsqueeze(0)
    pooled = pooled_means[:, 0] - pooled_means[:, 1]
    return {"halves": halves, "pooled": pooled}


def score_substitutions(collection: dict, exact: dict[str, torch.Tensor],
                        candidates: list[dict]) -> tuple[list[dict], dict]:
    observed = substitution_effect_views(collection)
    passing = []
    checks = {}
    for edge, candidate in enumerate(candidates):
        pair = {"directions": {}, "holds": True}
        for local, side in enumerate(("left_from_right", "right_from_left")):
            direction = 2 * edge + local
            target = collection["directions"][direction]["target"]
            row = {"windows": {}, "holds": True}
            for half in range(2):
                actual = exact["halves"][target, half]
                predicted = observed["halves"][direction, half]
                cosine = qm.safe_cosine(actual, predicted)
                residual = qm.relative_residual(actual, predicted)
                holds = cosine >= qm.CONFIRMATION_COSINE \
                    and residual <= qm.CONFIRMATION_RESIDUAL
                row["windows"][f"half{half}"] = {
                    "cosine": cosine, "relative_residual": residual, "holds": holds}
                row["holds"] &= holds
            actual = exact["pooled"][target]
            predicted = observed["pooled"][direction]
            cosine = qm.safe_cosine(actual, predicted)
            residual = qm.relative_residual(actual, predicted)
            holds = cosine >= qm.CONFIRMATION_COSINE \
                and residual <= qm.CONFIRMATION_RESIDUAL
            row["windows"]["pooled"] = {
                "cosine": cosine, "relative_residual": residual, "holds": holds}
            row["holds"] &= holds
            pair["directions"][side] = row
            pair["holds"] &= row["holds"]
        key = f"{candidate['left_name']} <-> {candidate['right_name']}"
        checks[key] = pair
        if pair["holds"]:
            passing.append(candidate)
    return passing, checks


def quotient_groups(passing: list[dict]) -> list[dict]:
    adjacency: dict[int, set[int]] = {}
    edge_scale = {}
    for edge in passing:
        left, right = edge["left"], edge["right"]
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
        edge_scale[(left, right)] = edge["beta_left_from_right"]
    groups = []
    visited = set()
    for root in sorted(adjacency):
        if root in visited:
            continue
        stack = [root]
        nodes = set()
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            stack.extend(adjacency.get(node, ()))
        visited |= nodes
        ordered = sorted(nodes)
        complete = all((left, right) in edge_scale
                       for left, right in itertools.combinations(ordered, 2))
        scale_to_root = {root: 1.0}
        for node in ordered[1:]:
            key = (min(root, node), max(root, node))
            if key in edge_scale:
                beta = edge_scale[key]
                scale_to_root[node] = 1.0 / beta
        errors = []
        if complete:
            for left, right in itertools.combinations(ordered, 2):
                expected = scale_to_root[left] / scale_to_root[right]
                errors.append(abs(edge_scale[(left, right)] / expected - 1.0))
        cycle_error = max(errors, default=0.0 if len(ordered) == 2 else math.inf)
        if complete and cycle_error <= 0.25:
            groups.append({
                "terms": [qm.TERM_NAMES[node] for node in ordered],
                "maximum_scale_cycle_relative_error": cycle_error,
            })
    return groups


def nontrivial_pair(candidate: dict) -> bool:
    left = qm.TERM_SPECS[candidate["left"]]
    right = qm.TERM_SPECS[candidate["right"]]
    return bool(left["operation"] != right["operation"] or left["sources"] != right["sources"])


def run_smoke() -> None:
    started = time.time()
    if SMOKE_OUT.exists():
        raise FileExistsError(f"smoke output already exists: {SMOKE_OUT}")
    dependencies = validate_dependencies()
    rows, circuit_masks, discovery_tags, _validation_tags, fit_rows, metadata = population()
    planted = qm.planted_suite()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device=device, dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    reference, group_reference, quadratic_means, reference_diagnostics = fit_references(
        model, fit_rows, device)
    tags = smoke_tags(circuit_masks, discovery_tags)
    collection = collect_phase(
        model, rows, circuit_masks, tags, SMOKE_BOUNDS,
        reference, group_reference, quadratic_means)
    pred_a = bool(
        planted["holds"]
        and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and instrument_holds(collection, require_support=True))
    result = {
        "status": "gpu_smoke_complete", "rung": 527,
        "claim_level": "instrument_smoke_not_scientific_result",
        "runner_sha256": file_sha256(RUNNER),
        "preregistration_sha256": FROZEN_SHA256[PREREG],
        "dependency_sha256": dependencies,
        "checkpoint": checkpoint.__dict__,
        "input_metadata": metadata,
        "planted_suite": planted,
        "reference_diagnostics": reference_diagnostics,
        "collection": serial_collection(collection),
        "smoke_a_exact_live": pred_a,
        "discovery_science_opened": False,
        "confirmation_opened": False,
        "physical_substitution_opened": False,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime_s": time.time() - started,
    }
    atomic_json(SMOKE_OUT, result)
    print(json.dumps({
        "status": result["status"], "smoke_a": pred_a,
        "remainder_energy_fraction": collection["diagnostics"]["remainder_energy_fraction"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)


def run_full() -> None:
    started = time.time()
    if OUT.exists():
        raise FileExistsError(f"result already exists: {OUT}")
    dependencies = validate_dependencies()
    smoke = smoke_authority()
    rows, circuit_masks, discovery_tags, validation_tags, fit_rows, metadata = population()
    planted = qm.planted_suite()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device=device, dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    reference, group_reference, quadratic_means, reference_diagnostics = fit_references(
        model, fit_rows, device)
    discovery_collection = collect_phase(
        model, rows, circuit_masks, discovery_tags, D_BOUNDS,
        reference, group_reference, quadratic_means)
    discovery_effect = effect_views(discovery_collection)
    candidates, discovery_summary = qm.discover_pairs(discovery_effect["halves"])
    control_counts = qm.permutation_control_counts(
        discovery_effect["halves"], PERMUTATION_SEEDS)
    control_q95 = quantile95(control_counts)
    pred_a = bool(
        planted["holds"]
        and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and instrument_holds(discovery_collection, require_support=True))
    pred_b = bool(
        pred_a
        and discovery_summary["small_relation"]
        and len(candidates) > control_q95)

    confirmation_collection = None
    confirmation_checks = {}
    confirmed = []
    pred_c = False
    substitutions = None
    physical_checks = {}
    physical_pairs = []
    groups = []
    pred_d = pred_e = False
    if pred_b:
        confirmation_collection = collect_phase(
            model, rows, circuit_masks, validation_tags, V_BOUNDS,
            reference, group_reference, quadratic_means, keep_native_nll=True)
        confirmation_effect = effect_views(confirmation_collection)
        confirmed, confirmation_checks = qm.confirmation_pairs(
            confirmation_effect["halves"], candidates, confirmation_effect["pooled"])
        pred_c = bool(
            instrument_holds(confirmation_collection, require_support=True)
            and len(confirmed) >= 1)
        if pred_c:
            substitutions = collect_substitutions(
                model, rows, circuit_masks, validation_tags, V_BOUNDS,
                reference, group_reference, quadratic_means, confirmed,
                confirmation_collection["native_nll"])
            physical_pairs, physical_checks = score_substitutions(
                substitutions, confirmation_effect, confirmed)
            substitution_instrument = bool(
                substitutions["diagnostics"]["calls_exact"]
                and substitutions["diagnostics"]["supports_positive"]
                and substitutions["diagnostics"]["state_replay_max_abs"] == 0.0
                and substitutions["diagnostics"]["minimum_term_edit_rms"] > 0
                and substitutions["diagnostics"]["zero_term_edits"] == 0)
            pred_d = bool(substitution_instrument and physical_pairs)
            pred_e = bool(pred_d and any(nontrivial_pair(row) for row in physical_pairs))
            groups = quotient_groups(physical_pairs)

    strong_null = bool(not (pred_a and pred_b and pred_c and pred_d and pred_e))
    if not pred_a:
        next_step = "repair_named_rung527_instrument_clause_only"
    elif not pred_b and discovery_summary["candidate_count"] == 0:
        next_step = "finite_predictive_state_quotient_or_different_module"
    elif not pred_b:
        next_step = "add_independent_downstream_readouts_without_selecting_best_pairs"
    elif not pred_c:
        next_step = "localize_first_consumer_that_breaks_discovery_relation"
    elif not pred_d:
        next_step = "preserve_noninterchangeability_and_localize_first_consumer"
    elif not pred_e:
        next_step = "retain_only_trivial_portability_without_grouping_claim"
    else:
        next_step = "ood_joint_composition_and_literal_price"

    result = {
        "status": "complete", "rung": 527,
        "claim_level": "exact_context_term_pairs_until_physical_substitution",
        "runner_sha256": file_sha256(RUNNER),
        "preregistration_sha256": FROZEN_SHA256[PREREG],
        "dependency_sha256": dependencies,
        "smoke_authority": smoke,
        "checkpoint": checkpoint.__dict__,
        "input_metadata": metadata,
        "term_specs": list(qm.TERM_SPECS),
        "planted_suite": planted,
        "reference_diagnostics": reference_diagnostics,
        "discovery": {
            "collection": serial_collection(discovery_collection),
            "effects_by_half": discovery_effect["halves"].tolist(),
            "pooled_effects": discovery_effect["pooled"].tolist(),
            "summary": discovery_summary,
            "candidates": candidates,
            "permutation_control_counts": control_counts,
            "permutation_control_q95_higher": control_q95,
        },
        "confirmation": None if confirmation_collection is None else {
            "collection": serial_collection(confirmation_collection),
            "checks": confirmation_checks,
            "confirmed_pairs": confirmed,
        },
        "substitutions": None if substitutions is None else {
            "bounds": list(substitutions["bounds"]),
            "tags": substitutions["tags"],
            "directions": substitutions["directions"],
            "sums": substitutions["sums"].tolist(),
            "counts": substitutions["counts"].tolist(),
            "diagnostics": substitutions["diagnostics"],
            "checks": physical_checks,
            "physical_pairs": physical_pairs,
            "quotient_groups": groups,
        },
        "pred_a_exact_live_instrument": pred_a,
        "pred_b_small_discovery_equivalence_relation": pred_b,
        "pred_c_heldout_circuits_and_documents": pred_c,
        "pred_d_bidirectional_physical_substitution": pred_d,
        "pred_e_nontrivial_context_term_grouping": pred_e,
        "strong_null": strong_null,
        "confirmation_opened": confirmation_collection is not None,
        "physical_substitution_opened": substitutions is not None,
        "literal_price": {
            "diagnostic_reference_values": REFERENCE_VALUES,
            "deployed_values_added": 0,
            "deployed_values_removed": 0,
            "compression_claim": False,
        },
        "execution_price": {
            "model_forwards": (
                discovery_collection["diagnostics"]["native_forwards"]
                + discovery_collection["diagnostics"]["removal_forwards"]
                + (0 if confirmation_collection is None else
                   confirmation_collection["diagnostics"]["native_forwards"]
                   + confirmation_collection["diagnostics"]["removal_forwards"])
                + (0 if substitutions is None else
                   substitutions["diagnostics"]["removal_forwards"])),
            "model_backwards": 0,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "next_step": next_step,
        "runtime_s": time.time() - started,
    }
    atomic_json(OUT, result)
    print(json.dumps({
        "status": result["status"],
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "strong_null": strong_null,
        "discovery_candidates": len(candidates),
        "control_q95": control_q95,
        "confirmed_pairs": len(confirmed),
        "physical_pairs": len(physical_pairs),
        "next_step": next_step,
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-smoke", action="store_true")
    args = parser.parse_args(argv)
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert qm.N_TERMS == 20 and REFERENCE_VALUES == 23_040
        assert D_BOUNDS == (0, 248, 124) and V_BOUNDS == (500, 1000, 750)
        assert len(PERMUTATION_SEEDS) == 16 and qm.MAX_CANDIDATES == 8
        print(json.dumps({
            "status": "dry_run_passed", "rung": 527,
            "model_loaded": False, "gpu_smoke": args.gpu_smoke,
            "discovery_model_forwards": 1302,
            "conditional_confirmation_model_forwards": 2625,
            "maximum_conditional_substitution_forwards": 2000,
            "confirmation_opened": False,
            "physical_substitution_opened": False,
        }, indent=2, sort_keys=True))
        return
    if args.gpu_smoke:
        run_smoke()
    else:
        run_full()


if __name__ == "__main__":
    main()
