#!/usr/bin/env python3
"""RUNG495 -- exact below-head attention1 pieces grouped by downstream use.

This is a discovery screen.  It constructs the seven finite A/B/V Möbius terms
inside every attention1 head and compares their 62-circuit response signatures
through the real normalized suffix.  Native head identity is provenance, not
the assumed circuit basis.
"""

# BQGATE: EXPERIMENT
# pred_a exact live factor, branch, mask, call, and normalized-suffix gradient instrument
# pred_b one cross-head downstream-use pair survives fixed discovery halves and controls
# pred_c the frozen pair predicts held-out documents and circuit tags
# pred_d at least one native head contains two downstream-distinct material pieces
# pred_e selected pair is only a candidate for a separately registered physical interchange

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
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mlp0_TI_site_graded_merge_intervention_rung493 as branch_parent
import mlp0_attention1_finite_path_factorial_rung484 as factor_parent
import mlp0_branch_circuit_response_rung481 as circuit_parent
import mlp0_centered_context_anova_factorial as component_parent
import bilin18_observed_model_facade as facade


PREREG = POLY / "ATTENTION1_DOWNSTREAM_USE_QUOTIENT_RUNG495_PREREGISTRATION.md"
R494_RESULT = ROOT / "equality_query_scaled_single_index_causal_rung494_results.json"
R493_SOURCE = ROOT / "ops/mlp0_TI_site_graded_merge_intervention_rung493.py"
R493_RESULT = ROOT / "mlp0_TI_site_graded_merge_intervention_rung493_results.json"
R484_SOURCE = ROOT / "ops/mlp0_attention1_finite_path_factorial_rung484.py"
R481_SOURCE = ROOT / "ops/mlp0_branch_circuit_response_rung481.py"
COMPONENT_SOURCE = ROOT / "ops/mlp0_centered_context_anova_factorial.py"
OUT = ROOT / "attention1_downstream_use_quotient_rung495b_results.json"
BUNDLE = ROOT / "attention1_downstream_use_quotient_rung495b_bundle.pt"
HASHES = {
    PREREG: "084ce5c20d1aac72f0a8325b454532ac9d4fc3d56eacc618434dff34f7b67568",
    R494_RESULT: "8b384663af5fe6b9291c4180f1ea6147a40835cc5e64a172a72f73087ddad261",
    R493_SOURCE: "4f77c3898d8237373a7a35439dabc590882eda47490aaed33a797ddec2cfe08b",
    R493_RESULT: "1131a0dc61f94ca2dba92073eed1a21c2f46a46ac18004be146e15a78161339d",
    R484_SOURCE: "42f66fba01361c976660554197fef7aa66cb20d80eb5b6351b01a1f6e3bf9d54",
    R481_SOURCE: "ef08017a30ceb0c9e4481198fc1d58c5b0bf8cd37707d2223c42db9eb04f1f44",
    COMPONENT_SOURCE: "1495ec13abf80bbd3d0bf33db8c0457e1bc5eab7421bcb1b96a780278d808322",
}
BRANCHES = ("T", "C", "I", "S")
FACTORS = ("QK1", "QK2", "OV")
FACTOR_MASKS = tuple(range(1, 8))
HEADS = 9
HEAD_DIM = 128
D = 1152
TOKENS = 256
BATCH = 4
DISCOVERY_RANGE = (0, 500)
DISCOVERY_SPLIT = 250
VALIDATION_RANGE = (500, 1000)
VALIDATION_SPLIT = 750
MASK_TYPES = ("member", "slice_control")
POSITION_SHIFTS = tuple(range(1, 17))
CIRCUIT_PERMUTATION_SEEDS = tuple(range(20260902950, 20260902966))
EXPECTED_DISCOVERY_FORWARDS = (DISCOVERY_RANGE[1] // BATCH) * (1 + len(BRANCHES))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def factor_name(mask: int) -> str:
    return "x".join(name for index, name in enumerate(FACTORS) if mask & (1 << index))


PIECE_NAMES = tuple(
    f"h{head}.{factor_name(mask)}"
    for head in range(HEADS) for mask in FACTOR_MASKS
)
CROSS_HEAD_PAIRS = tuple(
    (left, right)
    for left, right in itertools.combinations(range(len(PIECE_NAMES)), 2)
    if left // len(FACTOR_MASKS) != right // len(FACTOR_MASKS)
)
WITHIN_HEAD_PAIRS = tuple(
    (left, right)
    for left, right in itertools.combinations(range(len(PIECE_NAMES)), 2)
    if left // len(FACTOR_MASKS) == right // len(FACTOR_MASKS)
)


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    r494 = json.loads(R494_RESULT.read_text())
    if r494.get("rung") != 494 \
            or r494.get("pred_a_exact_live_scaled_intervention") is not True \
            or r494.get("pred_b_half_strength_causal_interpolation") is not False \
            or r494.get("pred_c_one_and_half_strength_causal_transfer") is not True \
            or r494.get("pred_d_document_half_stability") is not False \
            or r494.get("strong_null") is not True \
            or r494.get("next_step") != "attention1_exact_QK1_QK2_OV_downstream_use_decomposition":
        raise RuntimeError("rung494 did not license the below-head attention1 route")
    rows, fit_rows, branch_metadata = branch_parent.validate_inputs()
    circuit_rows, circuit_masks, discovery_tags, validation_tags, _, circuit_metadata = \
        circuit_parent.validate_inputs()
    if not torch.equal(rows, circuit_rows):
        raise RuntimeError("branch and 62-circuit row authorities differ")
    if len(discovery_tags) != 32 or len(validation_tags) != 30:
        raise RuntimeError("frozen 32/30 circuit split changed")
    return rows, fit_rows, circuit_masks, discovery_tags, validation_tags, {
        "branch": branch_metadata,
        "circuits": circuit_metadata,
        "piece_names": list(PIECE_NAMES),
        "cross_head_pairs": len(CROSS_HEAD_PAIRS),
        "within_head_pairs": len(WITHIN_HEAD_PAIRS),
    }


def _linear(value, weight):
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def _per_head_attention_writes(attention, parts):
    """Return [batch, query, head, residual] before summing native heads."""
    # The registered exact factor algebra is float32.  The production forward
    # and its differentiated raw-write leaf remain BF16; only the analytical
    # eight-arm/Mobius construction uses the parent rung484 precision path.
    score_a, score_b, value = (part.float() for part in parts)
    length = score_a.shape[-1]
    pattern = score_a * score_b
    causal = torch.tril(torch.ones(
        length, length, dtype=torch.bool, device=pattern.device))
    pattern = pattern.masked_fill(~causal, 0)
    head_values = torch.einsum("bhqk,bkhu->bhqu", pattern, value)
    output_weight = attention.c_proj.weight.to(
        device=head_values.device, dtype=head_values.dtype).reshape(D, HEADS, HEAD_DIM)
    return torch.einsum("bhqu,ohu->bqho", head_values, output_weight)


def exact_factor_pieces(attention, normal_parts, absent_parts):
    """Return 63 finite pieces and exact endpoint/closure diagnostics.

    Arms choose each of QK1/QK2/OV from absent (bit 0) or normal (bit 1).
    The output piece axis is ordered head-major according to PIECE_NAMES.
    """
    arms = []
    for mask in range(8):
        parts = tuple(
            normal_parts[index] if mask & (1 << index) else absent_parts[index]
            for index in range(3)
        )
        arms.append(_per_head_attention_writes(attention, parts))
    arms = torch.stack(arms, dim=0)  # [arm,batch,query,head,residual]
    effects = torch.zeros_like(arms)
    for mask in range(8):
        for child in range(8):
            if child & ~mask:
                continue
            sign = -1.0 if ((mask.bit_count() - child.bit_count()) % 2) else 1.0
            effects[mask] += sign * arms[child]
    pieces = effects[1:].permute(1, 2, 3, 0, 4).contiguous().reshape(
        arms.shape[1], arms.shape[2], HEADS * len(FACTOR_MASKS), D)
    normal_write = arms[7].sum(2)
    absent_write = arms[0].sum(2)
    reconstructed_delta = pieces.sum(2)
    return pieces, {
        "normal_write": normal_write,
        "absent_write": absent_write,
        "factor_delta": normal_write - absent_write,
        "reconstructed_delta": reconstructed_delta,
    }


def _relative_squared(left, right):
    left = torch.as_tensor(left, dtype=torch.float64)
    right = torch.as_tensor(right, dtype=torch.float64)
    return float((left - right).square().sum() / right.square().sum().clamp_min(1e-30))


def _cosine(left, right):
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator.clamp_min(1e-30))


def _scaled_report(left, right):
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    left2 = torch.dot(left, left).clamp_min(1e-30)
    right2 = torch.dot(right, right).clamp_min(1e-30)
    dot = torch.dot(left, right)
    scale = float(dot / left2)
    cosine = float(dot / torch.sqrt(left2 * right2))
    residual = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return {"cosine": cosine, "left_to_right_scale": scale,
            "best_scale_residual": residual}


def _quantile95(values):
    return float(torch.quantile(
        torch.as_tensor(values, dtype=torch.float64), .95, interpolation="higher"))


def _native_factor_cache(model, tokens, reference):
    cache = factor_parent.parent._native_prefix(model, tokens, reference)
    block1 = model.transformer.h[1]
    after_m0 = cache["before_m0"] + cache["m0"]
    before_a1 = block1.lambdas[0] * after_m0 + block1.lambdas[1] * cache["x0"]
    state1 = F.rms_norm(before_a1, (D,))
    parts32 = factor_parent._attention_parts(
        block1.attn, state1.float(), cache["first_value"].float())
    direct32, _ = block1.attn(state1.float(), cache["first_value"].float())
    cache["attention1_parts"] = tuple(value.detach() for value in parts32)
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
            parts32 = factor_parent._attention_parts(
                event.block.attn, event.state.float(), event.first_value.float())
            direct32, _ = event.block.attn(
                event.state.float(), event.first_value.float())
            capture["parts"] = tuple(value.detach() for value in parts32)
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
    if set(capture) != {
            "parts", "attention1_direct32", "attention1_leaf", "mlp0_state_error"}:
        raise RuntimeError("branch-absent attention1 gradient capture failed")
    return logits, capture, calls


def _expected_backwards(circuit_masks, tags, start_doc, stop_doc, split):
    count = 0
    for start in range(start_doc, stop_doc, BATCH):
        stop = min(start + BATCH, stop_doc)
        count += len(circuit_parent._batch_selections(
            circuit_masks, tags, start, stop, split))
    return count * len(BRANCHES)


def _empty_collection(tags, piece_indices, shifts):
    return {
        "sums": torch.zeros(
            2, len(BRANCHES), len(MASK_TYPES), len(tags), len(shifts),
            len(piece_indices), dtype=torch.float64),
        "complete_sums": torch.zeros(
            2, len(BRANCHES), len(MASK_TYPES), len(tags), dtype=torch.float64),
        "counts": torch.zeros(2, len(MASK_TYPES), len(tags), dtype=torch.float64),
        "piece_indices": list(piece_indices), "shifts": list(shifts), "tags": list(tags),
    }


def collect_phase(model, rows, circuit_masks, tags, reference, start_doc, stop_doc,
                  split, piece_indices=tuple(range(63)), shifts=(0,)):
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
        "gradient_contraction_num": 0.0,
        "gradient_contraction_den": 0.0,
        "mlp0_state_max_abs": 0.0,
        "analytical_num": 0.0, "analytical_den": 0.0,
        "deployed_num": 0.0, "deployed_den": 0.0,
        "complete_attention_delta_rms_min": float("inf"),
        "gradient_rms_min": float("inf"),
        "response_abs_max": 0.0,
    }
    device = next(model.parameters()).device
    block1 = model.transformer.h[1]
    all_pieces = piece_indices == tuple(range(63))

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

        selections = circuit_parent._batch_selections(
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

            pieces, detail = exact_factor_pieces(
                block1.attn, cache["attention1_parts"], capture["parts"])
            errors["native_factor_rebuild_relative_squared_max"] = max(
                errors["native_factor_rebuild_relative_squared_max"],
                _relative_squared(
                    detail["normal_write"], cache["attention1_direct32"]))
            errors["absent_factor_rebuild_relative_squared_max"] = max(
                errors["absent_factor_rebuild_relative_squared_max"],
                _relative_squared(
                    detail["absent_write"], capture["attention1_direct32"]))
            errors["factor_mobius_closure_relative_squared_max"] = max(
                errors["factor_mobius_closure_relative_squared_max"],
                _relative_squared(detail["reconstructed_delta"], detail["factor_delta"]))
            errors["complete_attention_delta_rms_min"] = min(
                errors["complete_attention_delta_rms_min"],
                float(detail["factor_delta"].double().square().mean().sqrt()))
            chosen = pieces[:, :, piece_indices]
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
                shift_responses = []
                for shift in shifts:
                    shifted = chosen if shift == 0 else torch.roll(chosen, shift, dims=1)
                    shift_responses.append(torch.einsum(
                        "btd,btpd->p", gradient.float(), shifted.float()))
                responses = torch.stack(shift_responses)
                if not bool(torch.isfinite(responses).all()):
                    raise RuntimeError("nonfinite downstream piece response")
                collection["sums"][
                    half, branch_index, mask_index, circuit_index] += \
                    responses.detach().double().cpu()
                complete = (gradient.float() * detail["factor_delta"].float()).sum()
                collection["complete_sums"][
                    half, branch_index, mask_index, circuit_index] += float(complete)
                errors["response_abs_max"] = max(
                    errors["response_abs_max"], float(responses.abs().max()),
                    float(complete.abs()))
                if all_pieces:
                    mismatch = responses[shifts.index(0)].sum() - complete
                    errors["gradient_contraction_num"] += float(mismatch.double().square())
                    errors["gradient_contraction_den"] += float(complete.double().square())
            del logits, nll, leaf, pieces, chosen

    batches = math.ceil((stop_doc - start_doc) / BATCH)
    expected = {
        "native_prefixes": batches,
        "absent_forwards": len(BRANCHES) * batches,
        "absent_attention": 18 * len(BRANCHES) * batches,
        "absent_mlp": 18 * len(BRANCHES) * batches,
        "site0_removals": len(BRANCHES) * batches,
        "a1_gradient_leaves": len(BRANCHES) * batches,
        "backwards": _expected_backwards(
            circuit_masks, tags, start_doc, stop_doc, split),
    }
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
        "documents": stop_doc - start_doc,
    }
    return collection


def _fingerprints(collection):
    counts = collection["counts"][:, None, :, :, None, None].clamp_min(1)
    means = collection["sums"] / counts
    raw = means[:, :, 0] - means[:, :, 1]  # [half,branch,tag,shift,piece]
    adjusted = raw - raw.mean(1, keepdim=True)
    complete_counts = collection["counts"][:, None].clamp_min(1)
    complete_means = collection["complete_sums"] / complete_counts
    complete = complete_means[:, :, 0] - complete_means[:, :, 1]
    complete_adjusted = complete - complete.mean(1, keepdim=True)
    return raw, adjusted, complete, complete_adjusted


def _materiality(bank, complete, half, piece):
    numerator = torch.linalg.vector_norm(bank[half, :, :, 0, piece].reshape(-1))
    denominator = torch.linalg.vector_norm(complete[half].reshape(-1)).clamp_min(1e-30)
    return float(numerator / denominator)


def _permutation_controls(left, right, seeds):
    controls = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(right.shape[1], generator=generator)
        controls.append(_cosine(left, right[:, permutation]))
    return controls


def _pair_metrics(raw, adjusted, complete, complete_adjusted, half, left, right,
                  position_raw=None, position_adjusted=None):
    report = {}
    for name, bank, full in (
        ("raw", raw, complete), ("branch_mean_removed", adjusted, complete_adjusted),
    ):
        left_vector = bank[half, :, :, 0, left]
        right_vector = bank[half, :, :, 0, right]
        row = _scaled_report(left_vector, right_vector)
        permutation = _permutation_controls(
            left_vector, right_vector, CIRCUIT_PERMUTATION_SEEDS)
        row.update({
            "left_materiality": _materiality(bank, full, half, left),
            "right_materiality": _materiality(bank, full, half, right),
            "circuit_permutation_cosines": permutation,
            "circuit_permutation_q95": _quantile95(permutation),
        })
        position_bank = position_raw if name == "raw" else position_adjusted
        if position_bank is not None:
            position_cosines = [
                _cosine(left_vector, position_bank[half, :, :, shift_index, 1])
                for shift_index in range(1, len(POSITION_SHIFTS) + 1)
            ]
            row["position_shift_cosines"] = position_cosines
            row["position_shift_q95"] = _quantile95(position_cosines)
        report[name] = row
    return report


def _preliminary_analysis(collection):
    raw, adjusted, complete, complete_adjusted = _fingerprints(collection)
    material = [
        _materiality(raw, complete, 0, piece) >= .05
        for piece in range(len(PIECE_NAMES))
    ]
    candidates = []
    for left, right in CROSS_HEAD_PAIRS:
        if not (material[left] and material[right]):
            continue
        report = _pair_metrics(
            raw, adjusted, complete, complete_adjusted, 0, left, right)
        candidates.append((
            report["raw"]["cosine"], -report["raw"]["best_scale_residual"],
            PIECE_NAMES[left], PIECE_NAMES[right], left, right, report))
    candidates.sort(reverse=True)
    selected = None if not candidates else candidates[0]
    if selected is None:
        return {
            "selected_pair": None, "selected_indices": None,
            "candidate_count": 0, "preliminary_holds": False,
            "within_head_split": None,
        }
    _, _, _, _, left, right, half0 = selected
    half1 = _pair_metrics(raw, adjusted, complete, complete_adjusted, 1, left, right)
    mutual = True  # the globally highest material cross-head cosine is mutual by construction
    half0_nonposition = all(
        half0[kind]["cosine"] >= .90
        and half0[kind]["best_scale_residual"] <= .45
        and min(half0[kind]["left_materiality"], half0[kind]["right_materiality"]) >= .05
        and half0[kind]["cosine"] >= half0[kind]["circuit_permutation_q95"] + .10
        for kind in ("raw", "branch_mean_removed"))
    half1_nonposition = all(
        half1[kind]["cosine"] >= .80
        and half1[kind]["best_scale_residual"] <= .55
        and min(half1[kind]["left_materiality"], half1[kind]["right_materiality"]) >= .05
        and half1[kind]["cosine"] >= half1[kind]["circuit_permutation_q95"] + .05
        for kind in ("raw", "branch_mean_removed"))
    scale0 = half0["raw"]["left_to_right_scale"]
    scale1 = half1["raw"]["left_to_right_scale"]
    scale_stable = bool(scale0 > 0 and .5 <= scale1 / scale0 <= 1.5)

    split_candidates = []
    for split_left, split_right in WITHIN_HEAD_PAIRS:
        if not (material[split_left] and material[split_right]):
            continue
        split_report0 = _pair_metrics(
            raw, adjusted, complete, complete_adjusted, 0, split_left, split_right)
        split_candidates.append((
            split_report0["raw"]["cosine"], PIECE_NAMES[split_left],
            PIECE_NAMES[split_right], split_left, split_right, split_report0))
    split_candidates.sort()
    split = None
    if split_candidates:
        _, _, _, split_left, split_right, split0 = split_candidates[0]
        split1 = _pair_metrics(
            raw, adjusted, complete, complete_adjusted, 1, split_left, split_right)
        split_holds = bool(
            split0["raw"]["cosine"] <= .20
            and split0["raw"]["best_scale_residual"] >= .85
            and split1["raw"]["cosine"] <= .30
            and split1["raw"]["best_scale_residual"] >= .80
            and min(split1["raw"]["left_materiality"],
                    split1["raw"]["right_materiality"]) >= .05)
        split = {
            "pair": [PIECE_NAMES[split_left], PIECE_NAMES[split_right]],
            "indices": [split_left, split_right], "half0": split0,
            "half1": split1, "holds": split_holds,
        }
    return {
        "selected_pair": [PIECE_NAMES[left], PIECE_NAMES[right]],
        "selected_indices": [left, right], "candidate_count": len(candidates),
        "mutual_nearest": mutual, "half0": half0, "half1": half1,
        "scale_stable": scale_stable,
        "preliminary_holds": bool(
            mutual and half0_nonposition and half1_nonposition and scale_stable),
        "within_head_split": split,
    }


def _position_and_validation_report(collection, global_pair):
    raw, adjusted, complete, complete_adjusted = _fingerprints(collection)
    # The conditional collection stores only the frozen pair, in left/right order.
    report = []
    for half in range(2):
        row = _pair_metrics(
            raw, adjusted, complete, complete_adjusted, half, 0, 1,
            position_raw=raw, position_adjusted=adjusted)
        report.append(row)
    return {"pair": [PIECE_NAMES[index] for index in global_pair], "halves": report}


def _instrument_valid(instrument, require_contraction=True):
    return bool(
        instrument["calls_exact"]
        and instrument["native_factor_rebuild_relative_squared_max"] <= 1e-10
        and instrument["absent_factor_rebuild_relative_squared_max"] <= 1e-10
        and instrument["factor_mobius_closure_relative_squared_max"] <= 1e-10
        and (not require_contraction
             or instrument["gradient_contraction_relative_squared"] <= 1e-9)
        and instrument["mlp0_state_max_abs"] == 0.0
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and instrument["complete_attention_delta_rms_min"] > 0
        and instrument["gradient_rms_min"] > 0
        and instrument["response_abs_max"] > 0)


def _next_step(pred_a, pred_e):
    if not pred_a:
        return "repair_instrument_only_no_scientific_successor"
    if pred_e:
        return "preregister_physical_cross_head_piece_interchange"
    return "split_QK_score_sides_into_query_and_key_downstream_use"


def _b_holds(preliminary, position):
    if not preliminary["preliminary_holds"] or position is None:
        return False
    for half, row in enumerate(position["halves"]):
        margin = .10 if half == 0 else .05
        for kind in ("raw", "branch_mean_removed"):
            base = preliminary[f"half{half}"][kind]
            pos = row[kind]
            if base["cosine"] < pos["position_shift_q95"] + margin:
                return False
    return True


def _c_holds(validation, discovery_scale):
    if validation is None:
        return False
    for row in validation["halves"]:
        for kind in ("raw", "branch_mean_removed"):
            metric = row[kind]
            if not (
                metric["cosine"] >= .75
                and metric["best_scale_residual"] <= .60
                and min(metric["left_materiality"], metric["right_materiality"]) >= .05
                and metric["left_to_right_scale"] * discovery_scale > 0
                and metric["cosine"] >= metric["circuit_permutation_q95"] + .05
                and metric["cosine"] >= metric["position_shift_q95"] + .05
            ):
                return False
    return True


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(PIECE_NAMES) == 63
        assert len(CROSS_HEAD_PAIRS) == 1764
        assert len(WITHIN_HEAD_PAIRS) == 189
        print(json.dumps({
            "status": "dry_run_passed",
            "rung": 495,
            "model_loaded": False,
            "downstream_use_outcomes_opened": False,
            "validation_documents_or_tags_opened": False,
            "pieces_per_branch": len(PIECE_NAMES),
            "branches": list(BRANCHES),
            "discovery_forward_price": EXPECTED_DISCOVERY_FORWARDS,
            "implemented_now": [
                "frozen_authority_validation",
                "exact_per_head_factor_arms",
                "seven_term_mobius_decomposition",
                "branch_absent_gradient_capture",
                "62_circuit_signature_accumulation",
                "permutation_and_position_controls",
                "registered_scorer_and_receipt",
            ],
            "conditional_position_control_forwards": EXPECTED_DISCOVERY_FORWARDS,
            "conditional_validation_forwards": EXPECTED_DISCOVERY_FORWARDS,
        }, indent=2, sort_keys=True))
        return
    started = time.time()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung495 output namespace already exists")
    rows, fit_rows, circuit_masks, discovery_tags, validation_tags, metadata = \
        validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.requires_grad_(False)
    reference = component_parent._reference_moments(
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
    pred_d = bool(
        preliminary.get("within_head_split") is not None
        and preliminary["within_head_split"]["holds"])

    validation_collection = validation = None
    if pred_b:
        pair = tuple(preliminary["selected_indices"])
        validation_collection = collect_phase(
            model, rows, circuit_masks, validation_tags, reference,
            *VALIDATION_RANGE, VALIDATION_SPLIT, piece_indices=pair,
            shifts=(0, *POSITION_SHIFTS))
        validation = _position_and_validation_report(validation_collection, pair)
        pred_a &= _instrument_valid(validation_collection["instrument"], False)
    discovery_scale = 0.0 if preliminary.get("half0") is None else \
        preliminary["half0"]["raw"]["left_to_right_scale"]
    pred_c = bool(pred_a and pred_b and _c_holds(validation, discovery_scale))
    pred_e = bool(pred_a and pred_b and pred_c)
    strong_null = bool(not pred_a or not pred_b)

    bundle = {
        "schema": "rung495b_attention1_downstream_use_v1",
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
        "status": "completed", "rung": 495, "repair_id": "495b_float32_factor_arithmetic",
        "claim_level": "screen",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(BRANCHES), "factors": list(FACTORS),
        "piece_names": list(PIECE_NAMES),
        "analysis": {
            "preliminary": preliminary,
            "position_controls": position,
            "validation": validation,
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
        'pred_b_cross_head_downstream_equivalence': bool(pred_b),
        'pred_c_heldout_documents_and_circuits': bool(pred_c),
        'pred_d_within_head_split': bool(pred_d),
        'pred_e_downstream_equivalent_candidate_only': bool(pred_e),
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
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "next_step": _next_step(pred_a, pred_e),
        "runtime_s": time.time() - started,
    }
    dump(receipt, OUT)
    print(json.dumps({
        "status": receipt["status"], "rung": 495,
        "predictions": {key: value for key, value in receipt.items()
                        if key.startswith("pred_")},
        "selected_pair": preliminary.get("selected_pair"),
        "validation_opened": validation_collection is not None,
        "strong_null": strong_null, "next_step": receipt["next_step"],
        "runtime_s": receipt["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
