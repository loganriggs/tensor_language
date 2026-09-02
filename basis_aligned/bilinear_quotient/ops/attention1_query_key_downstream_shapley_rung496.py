#!/usr/bin/env python3
"""RUNG496 -- exact Q1/K1/Q2/K2/V allocations grouped by downstream use.

The five-factor/Shapley algebra, real normalized-suffix collection, controls,
conditional validation, scorer, and receipt implement the frozen registration.
"""

# BQGATE: EXPERIMENT
# pred_a exact five-factor arms, Mobius/Shapley closure, calls, masks, and liveness
# pred_b one cross-head query/query or key/key side confirms under three allocations
# pred_c the frozen side relation predicts held-out documents and circuit families
# pred_d the shared side is more specific than opposite-side or whole-head similarity
# pred_e candidate only; finite input-side interchange remains separately required

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


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import attention1_downstream_use_quotient_rung495 as parent
import bilin18_observed_model_facade as facade


PREREG = POLY / "ATTENTION1_QUERY_KEY_DOWNSTREAM_SHAPLEY_RUNG496_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/attention1_downstream_use_quotient_rung495.py"
PARENT_RESULT = ROOT / "attention1_downstream_use_quotient_rung495b_results.json"
PARENT_BUNDLE = ROOT / "attention1_downstream_use_quotient_rung495b_bundle.pt"
OUT = ROOT / "attention1_query_key_downstream_shapley_rung496_results.json"
BUNDLE = ROOT / "attention1_query_key_downstream_shapley_rung496_bundle.pt"
HASHES = {
    PREREG: "603d427f83d603b43647eeed3b05147282f92482976cd2c537a7d0ae64ef15a7",
    PARENT_SOURCE: "5385ad0c540f9cbfef153bc6e545d7d5daaf0916129bb6e4e1a99dc355cae74d",
    PARENT_RESULT: "f06b6098380883a51260fa6646a6fa8d8e1ee7e1f9d784b428e95d8e94161eee",
    PARENT_BUNDLE: "295bf8048ff5b8ad5b29acf9d3f25e91c0b18ef1711aca6c5b4b845f6b447f9a",
}
FACTOR_NAMES = ("Q1", "K1", "Q2", "K2", "V")
HEADS = parent.HEADS
HEAD_DIM = parent.HEAD_DIM
D = parent.D
FULL_MASK = (1 << len(FACTOR_NAMES)) - 1
PIECE_NAMES = tuple(
    f"h{head}.{factor}" for head in range(HEADS) for factor in FACTOR_NAMES)
SIDE_INDICES = tuple(
    index for index, name in enumerate(PIECE_NAMES) if not name.endswith(".V"))
VIEWS = ("shapley", "first", "last")
BRANCHES = parent.BRANCHES
MASK_TYPES = parent.MASK_TYPES
TOKENS = parent.TOKENS
BATCH = parent.BATCH
DISCOVERY_RANGE = parent.DISCOVERY_RANGE
DISCOVERY_SPLIT = parent.DISCOVERY_SPLIT
VALIDATION_RANGE = parent.VALIDATION_RANGE
VALIDATION_SPLIT = parent.VALIDATION_SPLIT
POSITION_SHIFTS = parent.POSITION_SHIFTS
CIRCUIT_PERMUTATION_SEEDS = tuple(range(20260902960, 20260902976))
ELIGIBLE_PAIRS = tuple(
    (left, right)
    for left, right in itertools.combinations(SIDE_INDICES, 2)
    if left // len(FACTOR_NAMES) != right // len(FACTOR_NAMES)
    and ((left % len(FACTOR_NAMES) in (0, 2)
          and right % len(FACTOR_NAMES) in (0, 2))
         or (left % len(FACTOR_NAMES) in (1, 3)
             and right % len(FACTOR_NAMES) in (1, 3)))
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _linear(value, weight):
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def attention_factors(attention, state, first_value):
    """Return float32 Q1,K1,Q2,K2,V factors from the real attention state."""
    batch, length, width = state.shape
    if width != D or first_value.shape != (batch, length, HEADS, HEAD_DIM):
        raise RuntimeError("attention1 five-factor interface changed")
    state = state.float()
    first_value = first_value.float()
    q1 = _linear(state, attention.c_q.weight).view(batch, length, HEADS, HEAD_DIM)
    k1 = _linear(state, attention.c_k.weight).view(batch, length, HEADS, HEAD_DIM)
    q2 = _linear(state, attention.c_q2.weight).view(batch, length, HEADS, HEAD_DIM)
    k2 = _linear(state, attention.c_k2.weight).view(batch, length, HEADS, HEAD_DIM)
    raw_value = _linear(state, attention.c_v.weight).view(
        batch, length, HEADS, HEAD_DIM)
    value = (1 - attention.lamb) * raw_value + attention.lamb * first_value
    cos, sin = attention.rotary(q1)
    module = sys.modules[type(attention).__module__]
    q1 = module.apply_rotary_emb(F.rms_norm(q1, (HEAD_DIM,)), cos, sin)
    k1 = module.apply_rotary_emb(F.rms_norm(k1, (HEAD_DIM,)), cos, sin)
    q2 = module.apply_rotary_emb(F.rms_norm(q2, (HEAD_DIM,)), cos, sin)
    k2 = module.apply_rotary_emb(F.rms_norm(k2, (HEAD_DIM,)), cos, sin)
    return q1, k1, q2, k2, value


def _per_head_factor_writes(attention, factors):
    """Evaluate one five-factor arm as [batch,query,head,residual]."""
    q1, k1, q2, k2, value = (factor.float() for factor in factors)
    length = q1.shape[1]
    score1 = torch.einsum("bqhd,bkhd->bhqk", q1, k1) / HEAD_DIM
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HEAD_DIM
    pattern = score1 * score2
    causal = torch.tril(torch.ones(
        length, length, dtype=torch.bool, device=pattern.device))
    pattern = pattern.masked_fill(~causal, 0)
    head_values = torch.einsum("bhqk,bkhu->bhqu", pattern, value)
    output_weight = attention.c_proj.weight.to(
        device=head_values.device, dtype=torch.float32).reshape(D, HEADS, HEAD_DIM)
    return torch.einsum("bhqu,ohu->bqho", head_values, output_weight)


def exact_factor_allocations(attention, normal_factors, absent_factors):
    """Return exact Shapley, factor-first, and factor-last raw-write pieces.

    Output views have shape [batch,query,45,residual] in head-major order.
    The Shapley pieces exactly sum to the full normal-minus-absent write.
    """
    arms = []
    for mask in range(1 << len(FACTOR_NAMES)):
        factors = tuple(
            normal_factors[index] if mask & (1 << index)
            else absent_factors[index]
            for index in range(len(FACTOR_NAMES)))
        arms.append(_per_head_factor_writes(attention, factors))
    arms = torch.stack(arms, dim=0)

    effects = torch.zeros_like(arms)
    for mask in range(1 << len(FACTOR_NAMES)):
        for child in range(1 << len(FACTOR_NAMES)):
            if child & ~mask:
                continue
            sign = -1.0 if ((mask.bit_count() - child.bit_count()) % 2) else 1.0
            effects[mask] += sign * arms[child]

    shapley = [torch.zeros_like(arms[0]) for _ in FACTOR_NAMES]
    for mask in range(1, 1 << len(FACTOR_NAMES)):
        share = effects[mask] / mask.bit_count()
        for factor in range(len(FACTOR_NAMES)):
            if mask & (1 << factor):
                shapley[factor] += share
    first = [effects[1 << factor] for factor in range(len(FACTOR_NAMES))]
    last = [
        arms[FULL_MASK] - arms[FULL_MASK ^ (1 << factor)]
        for factor in range(len(FACTOR_NAMES))]

    def flatten(values):
        stacked = torch.stack(values, dim=3)  # [batch,query,head,factor,residual]
        return stacked.reshape(stacked.shape[0], stacked.shape[1], -1, D)

    normal_write = arms[FULL_MASK].sum(2)
    absent_write = arms[0].sum(2)
    return {
        "shapley": flatten(shapley),
        "first": flatten(first),
        "last": flatten(last),
    }, {
        "normal_write": normal_write,
        "absent_write": absent_write,
        "factor_delta": normal_write - absent_write,
        "head_delta": arms[FULL_MASK] - arms[0],
        "mobius_reconstruction": effects[1:].sum(0).sum(2),
        "shapley_reconstruction": flatten(shapley).sum(2),
    }


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("rung") != 495 \
            or result.get("repair_id") != "495b_float32_factor_arithmetic" \
            or result.get("pred_a_exact_live_instrument") is not True \
            or result.get("pred_b_cross_head_downstream_equivalence") is not False \
            or result.get("validation_documents_and_tags_opened") is not False \
            or result.get("strong_null") is not True \
            or result.get("next_step") != \
            "split_QK_score_sides_into_query_and_key_downstream_use":
        raise RuntimeError("lawful rung495b null did not license rung496")
    rows, fit_rows, circuit_masks, discovery_tags, validation_tags, metadata = \
        parent.validate_inputs()
    return rows, fit_rows, circuit_masks, discovery_tags, validation_tags, {
        **metadata,
        "factor_names": list(FACTOR_NAMES),
        "piece_names": list(PIECE_NAMES),
        "eligible_side_indices": list(SIDE_INDICES),
        "eligible_pairs": len(ELIGIBLE_PAIRS),
    }


def _native_factor_cache(model, tokens, reference):
    cache = parent.factor_parent.parent._native_prefix(model, tokens, reference)
    block1 = model.transformer.h[1]
    after_m0 = cache["before_m0"] + cache["m0"]
    before_a1 = block1.lambdas[0] * after_m0 + block1.lambdas[1] * cache["x0"]
    state1 = F.rms_norm(before_a1, (D,))
    cache["attention1_factors"] = tuple(
        value.detach() for value in attention_factors(
            block1.attn, state1, cache["first_value"]))
    direct32, _ = block1.attn(state1.float(), cache["first_value"].float())
    cache["attention1_direct32"] = direct32.detach()
    return cache


def _absent_gradient_forward(model, tokens, cache, branch):
    capture = {}
    calls = {"attention": 0, "mlp": 0, "site0_removal": 0, "a1_leaf": 0}

    def attention(event):
        calls["attention"] += 1
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 1:
            calls["a1_leaf"] += 1
            capture["factors"] = tuple(
                value.detach() for value in attention_factors(
                    event.block.attn, event.state, event.first_value))
            direct32, _ = event.block.attn(
                event.state.float(), event.first_value.float())
            capture["attention1_direct32"] = direct32.detach()
            write = write.detach().requires_grad_(True)
            capture["attention1_leaf"] = write
        return write, first_value

    def mlp(event):
        calls["mlp"] += 1
        write = event.block.mlp(event.state)
        if event.site == 0:
            calls["site0_removal"] += 1
            capture["mlp0_state_error"] = float(
                (event.state - F.rms_norm(cache["before_m0"], (D,))).abs().max())
            return cache["m0"] - branch
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    expected = {
        "factors", "attention1_direct32", "attention1_leaf", "mlp0_state_error"}
    if set(capture) != expected:
        raise RuntimeError("branch-absent five-factor gradient capture failed")
    return logits, capture, calls


def _empty_collection(tags, piece_indices, shifts):
    return {
        "sums": torch.zeros(
            len(VIEWS), 2, len(BRANCHES), len(MASK_TYPES), len(tags),
            len(shifts), len(piece_indices), dtype=torch.float64),
        "head_sums": torch.zeros(
            2, len(BRANCHES), len(MASK_TYPES), len(tags), len(shifts),
            HEADS, dtype=torch.float64),
        "complete_sums": torch.zeros(
            2, len(BRANCHES), len(MASK_TYPES), len(tags), dtype=torch.float64),
        "counts": torch.zeros(2, len(MASK_TYPES), len(tags), dtype=torch.float64),
        "piece_indices": list(piece_indices), "shifts": list(shifts),
        "tags": list(tags), "views": list(VIEWS),
    }


def collect_phase(model, rows, circuit_masks, tags, reference, start_doc, stop_doc,
                  split, piece_indices=tuple(range(45)), shifts=(0,)):
    piece_indices = tuple(piece_indices)
    shifts = tuple(shifts)
    collection = _empty_collection(tags, piece_indices, shifts)
    calls = {
        "native_prefixes": 0, "absent_forwards": 0, "absent_attention": 0,
        "absent_mlp": 0, "site0_removals": 0, "a1_gradient_leaves": 0,
        "backwards": 0,
    }
    errors = {
        "native_factor_rebuild_relative_squared_max": 0.0,
        "absent_factor_rebuild_relative_squared_max": 0.0,
        "factor_mobius_closure_relative_squared_max": 0.0,
        "factor_shapley_closure_relative_squared_max": 0.0,
        "gradient_contraction_num": 0.0,
        "gradient_contraction_den": 0.0,
        "mlp0_state_max_abs": 0.0,
        "analytical_num": 0.0, "analytical_den": 0.0,
        "deployed_num": 0.0, "deployed_den": 0.0,
        "complete_attention_delta_rms_min": float("inf"),
        "allocation_rms_min": float("inf"),
        "gradient_rms_min": float("inf"),
        "response_abs_max": 0.0,
    }
    device = next(model.parameters()).device
    block1 = model.transformer.h[1]
    all_pieces = piece_indices == tuple(range(45))

    for start in range(start_doc, stop_doc, BATCH):
        stop = min(start + BATCH, stop_doc)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        targets = batch_rows[:, 1:].to(device)
        cache = _native_factor_cache(model, tokens, reference)
        calls["native_prefixes"] += 1
        errors["analytical_num"] += cache["analytical_num"]
        errors["analytical_den"] += cache["analytical_den"]
        errors["deployed_num"] += cache["deployed_num"]
        errors["deployed_den"] += cache["deployed_den"]

        selections = parent.circuit_parent._batch_selections(
            circuit_masks, tags, start, stop, split)
        for half, mask_index, circuit_index, selected in selections:
            collection["counts"][half, mask_index, circuit_index] += int(selected.sum())

        for branch_index, branch_name in enumerate(BRANCHES):
            logits, capture, audit = _absent_gradient_forward(
                model, tokens, cache, cache["branches"][branch_name])
            calls["absent_forwards"] += 1
            calls["absent_attention"] += audit["attention"]
            calls["absent_mlp"] += audit["mlp"]
            calls["site0_removals"] += audit["site0_removal"]
            calls["a1_gradient_leaves"] += audit["a1_leaf"]
            errors["mlp0_state_max_abs"] = max(
                errors["mlp0_state_max_abs"], capture["mlp0_state_error"])

            allocations, detail = exact_factor_allocations(
                block1.attn, cache["attention1_factors"], capture["factors"])
            errors["native_factor_rebuild_relative_squared_max"] = max(
                errors["native_factor_rebuild_relative_squared_max"],
                parent._relative_squared(
                    detail["normal_write"], cache["attention1_direct32"]))
            errors["absent_factor_rebuild_relative_squared_max"] = max(
                errors["absent_factor_rebuild_relative_squared_max"],
                parent._relative_squared(
                    detail["absent_write"], capture["attention1_direct32"]))
            errors["factor_mobius_closure_relative_squared_max"] = max(
                errors["factor_mobius_closure_relative_squared_max"],
                parent._relative_squared(
                    detail["mobius_reconstruction"], detail["factor_delta"]))
            errors["factor_shapley_closure_relative_squared_max"] = max(
                errors["factor_shapley_closure_relative_squared_max"],
                parent._relative_squared(
                    detail["shapley_reconstruction"], detail["factor_delta"]))
            errors["complete_attention_delta_rms_min"] = min(
                errors["complete_attention_delta_rms_min"],
                float(detail["factor_delta"].double().square().mean().sqrt()))
            allocation_stack = torch.stack(
                [allocations[name] for name in VIEWS], dim=0)
            chosen = allocation_stack[:, :, :, piece_indices]
            eligible_local = [
                local for local, global_index in enumerate(piece_indices)
                if global_index in SIDE_INDICES]
            if eligible_local:
                errors["allocation_rms_min"] = min(
                    errors["allocation_rms_min"],
                    float(chosen[:, :, :, eligible_local].double().square()
                          .mean(dim=(1, 2, 4)).sqrt().min()))

            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                reduction="none").view(len(batch_rows), TOKENS)
            leaf = capture["attention1_leaf"]
            for selection_index, (half, mask_index, circuit_index,
                                  selected_cpu) in enumerate(selections):
                selected = selected_cpu.to(device)
                gradient = torch.autograd.grad(
                    nll[selected].sum(), leaf,
                    retain_graph=selection_index + 1 < len(selections),
                    allow_unused=False)[0]
                calls["backwards"] += 1
                errors["gradient_rms_min"] = min(
                    errors["gradient_rms_min"],
                    float(gradient.double().square().mean().sqrt()))
                view_responses, head_responses = [], []
                for shift in shifts:
                    shifted = chosen if shift == 0 else torch.roll(
                        chosen, shift, dims=2)
                    view_responses.append(torch.einsum(
                        "btd,vbtpd->vp", gradient.float(), shifted.float()))
                    head_delta = detail["head_delta"]
                    shifted_head = head_delta if shift == 0 else torch.roll(
                        head_delta, shift, dims=1)
                    head_responses.append(torch.einsum(
                        "btd,bthd->h", gradient.float(), shifted_head.float()))
                responses = torch.stack(view_responses, dim=1)  # [view,shift,piece]
                head_response = torch.stack(head_responses, dim=0)  # [shift,head]
                if not bool(torch.isfinite(responses).all()) \
                        or not bool(torch.isfinite(head_response).all()):
                    raise RuntimeError("nonfinite downstream side response")
                collection["sums"][
                    :, half, branch_index, mask_index, circuit_index] += \
                    responses.detach().double().cpu()
                collection["head_sums"][
                    half, branch_index, mask_index, circuit_index] += \
                    head_response.detach().double().cpu()
                complete = (gradient.float() * detail["factor_delta"].float()).sum()
                collection["complete_sums"][
                    half, branch_index, mask_index, circuit_index] += float(complete)
                errors["response_abs_max"] = max(
                    errors["response_abs_max"], float(responses.abs().max()),
                    float(head_response.abs().max()), float(complete.abs()))
                if all_pieces:
                    mismatch = responses[VIEWS.index("shapley"), shifts.index(0)].sum() \
                        - complete
                    errors["gradient_contraction_num"] += float(
                        mismatch.double().square())
                    errors["gradient_contraction_den"] += float(
                        complete.double().square())
            del logits, nll, leaf, allocations, allocation_stack, chosen

    batches = math.ceil((stop_doc - start_doc) / BATCH)
    expected = {
        "native_prefixes": batches,
        "absent_forwards": len(BRANCHES) * batches,
        "absent_attention": 18 * len(BRANCHES) * batches,
        "absent_mlp": 18 * len(BRANCHES) * batches,
        "site0_removals": len(BRANCHES) * batches,
        "a1_gradient_leaves": len(BRANCHES) * batches,
        "backwards": parent._expected_backwards(
            circuit_masks, tags, start_doc, stop_doc, split),
    }
    eligible_norms = []
    for view in range(len(VIEWS)):
        for local, global_index in enumerate(piece_indices):
            if global_index in SIDE_INDICES:
                eligible_norms.append(float(
                    collection["sums"][view, ..., local].square().sum().sqrt()))
    collection["instrument"] = {
        "calls": calls, "expected_calls": expected, "calls_exact": calls == expected,
        **{key: value for key, value in errors.items()
           if key not in ("analytical_num", "analytical_den", "deployed_num",
                          "deployed_den", "gradient_contraction_num",
                          "gradient_contraction_den")},
        "analytical_branch_identity_relative_squared": errors["analytical_num"]
            / max(errors["analytical_den"], 1e-30),
        "deployed_branch_identity_relative_squared": errors["deployed_num"]
            / max(errors["deployed_den"], 1e-30),
        "gradient_contraction_relative_squared": errors["gradient_contraction_num"]
            / max(errors["gradient_contraction_den"], 1e-30),
        "eligible_view_response_norm_min": min(eligible_norms, default=0.0),
        "documents": stop_doc - start_doc,
    }
    return collection


def _fingerprints(collection):
    counts = collection["counts"][None, :, None, :, :, None, None].clamp_min(1)
    means = collection["sums"] / counts
    raw = means[:, :, :, 0] - means[:, :, :, 1]
    adjusted = raw - raw.mean(2, keepdim=True)

    head_counts = collection["counts"][:, None, :, :, None, None].clamp_min(1)
    head_means = collection["head_sums"] / head_counts
    head_raw = head_means[:, :, 0] - head_means[:, :, 1]
    head_adjusted = head_raw - head_raw.mean(1, keepdim=True)

    complete_counts = collection["counts"][:, None].clamp_min(1)
    complete_means = collection["complete_sums"] / complete_counts
    complete = complete_means[:, :, 0] - complete_means[:, :, 1]
    complete_adjusted = complete - complete.mean(1, keepdim=True)
    return raw, adjusted, head_raw, head_adjusted, complete, complete_adjusted


def _materiality(bank, complete, view, half, piece):
    numerator = torch.linalg.vector_norm(
        bank[view, half, :, :, 0, piece].reshape(-1))
    denominator = torch.linalg.vector_norm(complete[half].reshape(-1)).clamp_min(1e-30)
    return float(numerator / denominator)


def _permutation_controls(left, right):
    return parent._permutation_controls(
        left, right, CIRCUIT_PERMUTATION_SEEDS)


def _pair_metrics(raw, adjusted, complete, complete_adjusted, half, left, right,
                  position_raw=None, position_adjusted=None):
    report = {}
    for view_index, view_name in enumerate(VIEWS):
        report[view_name] = {}
        for kind, bank, full in (
            ("raw", raw, complete),
            ("branch_mean_removed", adjusted, complete_adjusted),
        ):
            left_vector = bank[view_index, half, :, :, 0, left]
            right_vector = bank[view_index, half, :, :, 0, right]
            row = parent._scaled_report(left_vector, right_vector)
            permutation = _permutation_controls(left_vector, right_vector)
            row.update({
                "left_materiality": _materiality(
                    bank, full, view_index, half, left),
                "right_materiality": _materiality(
                    bank, full, view_index, half, right),
                "circuit_permutation_cosines": permutation,
                "circuit_permutation_q95": parent._quantile95(permutation),
            })
            position_bank = position_raw if kind == "raw" else position_adjusted
            if position_bank is not None:
                position_cosines = [
                    parent._cosine(
                        left_vector,
                        position_bank[view_index, half, :, :, shift_index, right])
                    for shift_index in range(1, len(POSITION_SHIFTS) + 1)
                ]
                row["position_shift_cosines"] = position_cosines
                row["position_shift_q95"] = parent._quantile95(position_cosines)
            report[view_name][kind] = row
    return report


def _allocation_holds(report, half):
    primary_cosine = .90 if half == 0 else .80
    primary_residual = .45 if half == 0 else .60
    primary_margin = .10 if half == 0 else .05
    endpoint_cosine = .65 if half == 0 else .55
    for kind in ("raw", "branch_mean_removed"):
        shapley = report["shapley"][kind]
        if not (
            shapley["cosine"] >= primary_cosine
            and shapley["best_scale_residual"] <= primary_residual
            and min(shapley["left_materiality"], shapley["right_materiality"]) >= .05
            and shapley["cosine"] >=
                shapley["circuit_permutation_q95"] + primary_margin
        ):
            return False
        for view in ("first", "last"):
            endpoint = report[view][kind]
            if not (
                endpoint["cosine"] >= endpoint_cosine
                and endpoint["left_to_right_scale"] *
                    shapley["left_to_right_scale"] > 0
                and min(endpoint["left_materiality"],
                        endpoint["right_materiality"]) >= .01
                and endpoint["cosine"] >=
                    endpoint["circuit_permutation_q95"] + .05
            ):
                return False
    return True


def _opposite(global_index):
    factor = global_index % len(FACTOR_NAMES)
    opposite_factor = {0: 1, 1: 0, 2: 3, 3: 2}[factor]
    return global_index - factor + opposite_factor


def _specificity_report(collection, global_pair):
    raw, adjusted, head_raw, head_adjusted, complete, complete_adjusted = \
        _fingerprints(collection)
    del adjusted, head_adjusted, complete, complete_adjusted
    lookup = {global_index: local for local, global_index
              in enumerate(collection["piece_indices"])}
    left_global, right_global = global_pair
    required = (left_global, right_global,
                _opposite(left_global), _opposite(right_global))
    if any(index not in lookup for index in required):
        raise RuntimeError("specificity collection omitted an opposite-side piece")
    left, right, opposite_left, opposite_right = (lookup[index] for index in required)
    rows = []
    for half in range(2):
        candidate = parent._cosine(
            raw[0, half, :, :, 0, left], raw[0, half, :, :, 0, right])
        opposite = parent._cosine(
            raw[0, half, :, :, 0, opposite_left],
            raw[0, half, :, :, 0, opposite_right])
        left_head = left_global // len(FACTOR_NAMES)
        right_head = right_global // len(FACTOR_NAMES)
        complete_head = parent._cosine(
            head_raw[half, :, :, 0, left_head],
            head_raw[half, :, :, 0, right_head])
        rows.append({
            "candidate_shapley_cosine": candidate,
            "opposite_side_cross_head_cosine": opposite,
            "complete_head_change_cosine": complete_head,
            "candidate_minus_opposite": candidate - opposite,
            "candidate_minus_complete_head": candidate - complete_head,
        })
    return {
        "halves": rows,
        "opposite_margin_holds": all(
            row["candidate_minus_opposite"] >= .20 for row in rows),
        "complete_head_margin_holds": all(
            row["candidate_minus_complete_head"] >= .20 for row in rows),
    }


def _preliminary_analysis(collection):
    raw, adjusted, _, _, complete, complete_adjusted = _fingerprints(collection)
    lookup = {global_index: local for local, global_index
              in enumerate(collection["piece_indices"])}
    candidates = []
    for left_global, right_global in ELIGIBLE_PAIRS:
        if left_global not in lookup or right_global not in lookup:
            continue
        left, right = lookup[left_global], lookup[right_global]
        if min(
            _materiality(raw, complete, 0, 0, left),
            _materiality(raw, complete, 0, 0, right),
        ) < .05:
            continue
        report = _pair_metrics(
            raw, adjusted, complete, complete_adjusted, 0, left, right)
        metric = report["shapley"]["raw"]
        candidates.append((
            -metric["cosine"], metric["best_scale_residual"],
            PIECE_NAMES[left_global], PIECE_NAMES[right_global],
            left_global, right_global, report))
    candidates.sort()
    if not candidates:
        return {
            "selected_pair": None, "selected_indices": None,
            "candidate_count": 0, "mutual_nearest": False,
            "preliminary_holds": False, "specificity": None,
        }
    _, _, _, _, left_global, right_global, half0 = candidates[0]
    left, right = lookup[left_global], lookup[right_global]
    half1 = _pair_metrics(
        raw, adjusted, complete, complete_adjusted, 1, left, right)
    scale_stable = {}
    for kind in ("raw", "branch_mean_removed"):
        scale0 = half0["shapley"][kind]["left_to_right_scale"]
        scale1 = half1["shapley"][kind]["left_to_right_scale"]
        scale_stable[kind] = bool(
            scale0 > 0 and .5 <= scale1 / scale0 <= 1.5)
    specificity = _specificity_report(collection, (left_global, right_global))
    return {
        "selected_pair": [PIECE_NAMES[left_global], PIECE_NAMES[right_global]],
        "selected_indices": [left_global, right_global],
        "candidate_count": len(candidates),
        "mutual_nearest": True,
        "half0": half0, "half1": half1,
        "scale_stable": scale_stable,
        "specificity": specificity,
        "preliminary_holds": bool(
            _allocation_holds(half0, 0)
            and _allocation_holds(half1, 1)
            and all(scale_stable.values())),
    }


def _position_and_validation_report(collection, global_pair):
    raw, adjusted, _, _, complete, complete_adjusted = _fingerprints(collection)
    lookup = {global_index: local for local, global_index
              in enumerate(collection["piece_indices"])}
    left, right = lookup[global_pair[0]], lookup[global_pair[1]]
    rows = [
        _pair_metrics(
            raw, adjusted, complete, complete_adjusted, half, left, right,
            position_raw=raw, position_adjusted=adjusted)
        for half in range(2)
    ]
    return {
        "pair": [PIECE_NAMES[index] for index in global_pair],
        "halves": rows,
    }


def _instrument_valid(instrument, require_contraction=True):
    return bool(
        instrument["calls_exact"]
        and instrument["native_factor_rebuild_relative_squared_max"] <= 1e-10
        and instrument["absent_factor_rebuild_relative_squared_max"] <= 1e-10
        and instrument["factor_mobius_closure_relative_squared_max"] <= 1e-10
        and instrument["factor_shapley_closure_relative_squared_max"] <= 1e-10
        and (not require_contraction
             or instrument["gradient_contraction_relative_squared"] <= 1e-9)
        and instrument["mlp0_state_max_abs"] == 0.0
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and instrument["complete_attention_delta_rms_min"] > 0
        and instrument["allocation_rms_min"] > 0
        and instrument["gradient_rms_min"] > 0
        and instrument["response_abs_max"] > 0
        and instrument["eligible_view_response_norm_min"] > 0)


def _b_holds(preliminary, position):
    if not preliminary["preliminary_holds"] or position is None:
        return False
    for half, report in enumerate(position["halves"]):
        margin = .10 if half == 0 else .05
        for kind in ("raw", "branch_mean_removed"):
            base = preliminary[f"half{half}"]["shapley"][kind]
            control = report["shapley"][kind]
            if base["cosine"] < control["position_shift_q95"] + margin:
                return False
    return True


def _c_holds(validation, discovery):
    if validation is None:
        return False
    for half, report in enumerate(validation["halves"]):
        for kind in ("raw", "branch_mean_removed"):
            shapley = report["shapley"][kind]
            discovery_scale = discovery["half0"]["shapley"][kind][
                "left_to_right_scale"]
            if not (
                shapley["cosine"] >= .75
                and shapley["best_scale_residual"] <= .65
                and min(shapley["left_materiality"],
                        shapley["right_materiality"]) >= .05
                and shapley["left_to_right_scale"] * discovery_scale > 0
                and shapley["cosine"] >=
                    shapley["circuit_permutation_q95"] + .05
                and shapley["cosine"] >= shapley["position_shift_q95"] + .05
            ):
                return False
            for view in ("first", "last"):
                endpoint = report[view][kind]
                if not (
                    endpoint["cosine"] >= .50
                    and endpoint["left_to_right_scale"] * discovery_scale > 0
                    and min(endpoint["left_materiality"],
                            endpoint["right_materiality"]) >= .01
                    and endpoint["cosine"] >=
                        endpoint["circuit_permutation_q95"] + .05
                ):
                    return False
    return True


def _d_holds(discovery_specificity, validation_specificity):
    if discovery_specificity is None or validation_specificity is None:
        return False
    return bool(
        (discovery_specificity["opposite_margin_holds"]
         and validation_specificity["opposite_margin_holds"])
        or (discovery_specificity["complete_head_margin_holds"]
            and validation_specificity["complete_head_margin_holds"]))


def _next_step(pred_a, pred_b, pred_c, pred_d):
    if not pred_a:
        return "repair_instrument_only_no_scientific_successor"
    if pred_b and pred_c and pred_d:
        return "preregister_finite_query_key_input_side_interchange"
    if pred_b and pred_c and not pred_d:
        return "test_whole_head_redundancy_without_shared_side_claim"
    return "predictive_state_causal_quotient_across_module_boundaries"


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(PIECE_NAMES) == 45 and len(SIDE_INDICES) == 36
        assert len(ELIGIBLE_PAIRS) == 288
        print(json.dumps({
            "status": "dry_run_passed",
            "rung": 496,
            "model_loaded": False,
            "downstream_use_outcomes_opened": False,
            "five_factor_arms": 32,
            "shapley_pieces": len(PIECE_NAMES),
            "eligible_query_key_sides": len(SIDE_INDICES),
            "eligible_cross_head_pairs": len(ELIGIBLE_PAIRS),
            "discovery_prefixes": 125,
            "discovery_absent_forwards": 500,
            "implemented_now": [
                "frozen_parent_authority",
                "float32_five_factor_reconstruction",
                "mobius_interactions",
                "exact_shapley_allocation",
                "factor_first_and_factor_last_controls",
                "real_normalized_suffix_gradients",
                "three_view_downstream_fingerprints",
                "frozen_pair_selection_and_controls",
                "conditional_validation_and_partner_specificity",
                "registered_scorer_and_receipt",
            ],
        }, indent=2, sort_keys=True))
        return
    started = time.time()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung496 output namespace already exists")
    rows, fit_rows, circuit_masks, discovery_tags, validation_tags, metadata = \
        validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.requires_grad_(False)
    reference = parent.component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    torch.cuda.reset_peak_memory_stats()

    discovery = collect_phase(
        model, rows, circuit_masks, discovery_tags, reference,
        *DISCOVERY_RANGE, DISCOVERY_SPLIT)
    preliminary = _preliminary_analysis(discovery)
    position_collection = position = None
    if preliminary["preliminary_holds"]:
        pair = tuple(preliminary["selected_indices"])
        position_collection = collect_phase(
            model, rows, circuit_masks, discovery_tags, reference,
            *DISCOVERY_RANGE, DISCOVERY_SPLIT, piece_indices=pair,
            shifts=(0, *POSITION_SHIFTS))
        position = _position_and_validation_report(position_collection, pair)

    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and _instrument_valid(discovery["instrument"]))
    if position_collection is not None:
        pred_a &= _instrument_valid(position_collection["instrument"], False)
    pred_b = bool(pred_a and _b_holds(preliminary, position))

    validation_collection = validation = validation_specificity = None
    if pred_b:
        pair = tuple(preliminary["selected_indices"])
        validation_indices = tuple(dict.fromkeys(
            (*pair, _opposite(pair[0]), _opposite(pair[1]))))
        validation_collection = collect_phase(
            model, rows, circuit_masks, validation_tags, reference,
            *VALIDATION_RANGE, VALIDATION_SPLIT,
            piece_indices=validation_indices, shifts=(0, *POSITION_SHIFTS))
        validation = _position_and_validation_report(validation_collection, pair)
        validation_specificity = _specificity_report(validation_collection, pair)
        pred_a &= _instrument_valid(validation_collection["instrument"], False)
    pred_c = bool(pred_a and pred_b and _c_holds(validation, preliminary))
    pred_d = bool(pred_a and pred_b and pred_c and _d_holds(
        preliminary.get("specificity"), validation_specificity))
    pred_e = bool(pred_a and pred_b and pred_c and pred_d)
    strong_null = bool(not pred_a or not pred_b)

    bundle = {
        "schema": "rung496_attention1_query_key_downstream_shapley_v1",
        "discovery": {key: value for key, value in discovery.items()
                      if key != "instrument"},
        "position_controls": None if position_collection is None else
            {key: value for key, value in position_collection.items()
             if key != "instrument"},
        "validation": None if validation_collection is None else
            {key: value for key, value in validation_collection.items()
             if key != "instrument"},
        "validation_opened": validation_collection is not None,
    }
    torch.save(bundle, BUNDLE)
    receipt = {
        "status": "completed", "rung": 496, "claim_level": "screen",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "factor_names": list(FACTOR_NAMES),
        "piece_names": list(PIECE_NAMES),
        "eligible_side_indices": list(SIDE_INDICES),
        "analysis": {
            "preliminary": preliminary,
            "position_controls": position,
            "validation": validation,
            "validation_specificity": validation_specificity,
        },
        "instrument": {
            "discovery": discovery["instrument"],
            "position_controls": None if position_collection is None else
                position_collection["instrument"],
            "validation": None if validation_collection is None else
                validation_collection["instrument"],
        },
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE)},
        'pred_a_exact_live_instrument': bool(pred_a),
        'pred_b_shared_query_or_key_side': bool(pred_b),
        'pred_c_heldout_documents_and_circuits': bool(pred_c),
        'pred_d_shared_side_not_whole_head': bool(pred_d),
        'pred_e_shared_side_candidate_only': bool(pred_e),
        "validation_documents_and_tags_opened": validation_collection is not None,
        "strong_null": strong_null,
        "execution_price": {
            "discovery_prefixes": discovery["instrument"]["calls"]["native_prefixes"],
            "discovery_full_forwards": discovery["instrument"]["calls"]["absent_forwards"],
            "discovery_backwards": discovery["instrument"]["calls"]["backwards"],
            "position_control_full_forwards": 0 if position_collection is None else
                position_collection["instrument"]["calls"]["absent_forwards"],
            "position_control_backwards": 0 if position_collection is None else
                position_collection["instrument"]["calls"]["backwards"],
            "validation_full_forwards": 0 if validation_collection is None else
                validation_collection["instrument"]["calls"]["absent_forwards"],
            "validation_backwards": 0 if validation_collection is None else
                validation_collection["instrument"]["calls"]["backwards"],
            "deployed_parameters_saved": 0,
            "deployed_parameters_added": 0,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "next_step": _next_step(pred_a, pred_b, pred_c, pred_d),
        "runtime_s": time.time() - started,
    }
    parent.dump(receipt, OUT)
    print(json.dumps({
        "status": receipt["status"], "rung": 496,
        "predictions": {key: value for key, value in receipt.items()
                        if key.startswith("pred_")},
        "selected_pair": preliminary.get("selected_pair"),
        "validation_opened": validation_collection is not None,
        "strong_null": strong_null,
        "next_step": receipt["next_step"],
        "runtime_s": receipt["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
