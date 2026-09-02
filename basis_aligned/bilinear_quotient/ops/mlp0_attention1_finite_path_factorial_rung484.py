#!/usr/bin/env python3
"""RUNG484 -- exact finite MLP0 T/I paths through attention1.

For each complete T or I removal, rebuild attention1 from all eight combinations
of native/removed score side A, score side B, and carried value V.  Measure exact
downstream CE effects, not a native-point derivative.  A proper physical subset
must transfer across document halves and distinguish shared versus split T/I
path use on equality-positive positions.
"""

# BQGATE: EXPERIMENT
# pred_a exact finite attention factor replay, branch identity, calls, and closure
# pred_b a proper physical attention path predicts the complete attention route
# pred_c the exact seven-effect path decomposition is stable across halves
# pred_d exactly one shared-versus-split T/I path relation holds
# pred_e selected paths are informative for equality-positive positions
# pred_f the unchanged relation and selected physical paths validate on held-out documents

from __future__ import annotations

import hashlib
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

import bilin18_observed_model_facade as facade
import circuit_induction_tensor as induction
import mlp0_immediate_consumer_quotient_rung483 as parent
import mlp0_centered_context_anova_factorial as component_parent


PREREG = POLY / "MLP0_ATTENTION1_FINITE_PATH_FACTORIAL_RUNG484_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp0_immediate_consumer_quotient_rung483.py"
PARENT_RESULT = ROOT / "mlp0_immediate_consumer_quotient_rung483_results.json"
OUT = ROOT / "mlp0_attention1_finite_path_factorial_rung484_results.json"
HASHES = {
    PREREG: "3e2abfeec269bd59c1a949af3d20915698b6fc81232c958c4fdbc1d2da1a7668",
    PARENT_SOURCE: "9763502b99b8693826a5985c8f25a3ebe7763c3cd176c3aebeeb140833a61f4c",
    PARENT_RESULT: "357fec66133993b2a85e8f8f2c549318c4d888336e41d2a5ad68604ad52225e6",
    ROOT / "ops/mlp0_branch_circuit_response_rung481.py":
        "ef08017a30ceb0c9e4481198fc1d58c5b0bf8cd37707d2223c42db9eb04f1f44",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
}
BRANCHES = ("T", "I")
COMPONENTS = ("A", "B", "V")
FULL_ARM = 7
DISCOVERY_RANGE = (0, 500)
DISCOVERY_SPLIT = 250
VALIDATION_RANGE = (500, 1000)
VALIDATION_SPLIT = 750
BATCH = 4
D = 1152
HEADS = 9
HEAD_DIM = 128
TOKENS = 256
POSITION_SHIFTS = parent.POSITION_SHIFTS
EXPECTED_POSITIVE_QUARTERS = (25344, 24861, 25306, 25541)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cosine(left, right) -> float:
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator.clamp_min(1e-30))


def _scaled_error(predictor, target) -> tuple[float, float]:
    predictor = torch.as_tensor(predictor, dtype=torch.float64).reshape(-1)
    target = torch.as_tensor(target, dtype=torch.float64).reshape(-1)
    predictor2 = torch.dot(predictor, predictor)
    dot = torch.dot(predictor, target)
    alpha = float(dot / predictor2.clamp_min(1e-30))
    residual = target - alpha * predictor
    error = float(torch.linalg.vector_norm(residual)
                  / torch.linalg.vector_norm(target).clamp_min(1e-30))
    return alpha, error


def _mask_name(mask: int) -> str:
    names = [name for index, name in enumerate(COMPONENTS) if mask & (1 << index)]
    return "+".join(names) if names else "EMPTY"


def _mobius(performance):
    """Möbius effects along the final axis of an eight-arm factorial."""
    performance = torch.as_tensor(performance, dtype=torch.float64)
    if performance.shape[-1] != 8:
        raise ValueError("three-component factorial must have eight arms")
    output = torch.zeros_like(performance)
    for mask in range(8):
        members = tuple(index for index in range(3) if mask & (1 << index))
        for child in range(8):
            if child & ~mask:
                continue
            sign = -1.0 if ((mask.bit_count() - child.bit_count()) % 2) else 1.0
            output[..., mask] += sign * performance[..., child]
    return output


def _linear(value, weight):
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def _attention_parts(attention, state, first_value):
    batch, length, width = state.shape
    if width != D or first_value.shape != (batch, length, HEADS, HEAD_DIM):
        raise RuntimeError("attention1 factor interface changed")
    q = _linear(state, attention.c_q.weight).view(batch, length, HEADS, HEAD_DIM)
    k = _linear(state, attention.c_k.weight).view(batch, length, HEADS, HEAD_DIM)
    q2 = _linear(state, attention.c_q2.weight).view(batch, length, HEADS, HEAD_DIM)
    k2 = _linear(state, attention.c_k2.weight).view(batch, length, HEADS, HEAD_DIM)
    raw_value = _linear(state, attention.c_v.weight).view(batch, length, HEADS, HEAD_DIM)
    value = (1 - attention.lamb) * raw_value + attention.lamb * first_value.view_as(raw_value)
    cos, sin = attention.rotary(q)
    module = sys.modules[type(attention).__module__]
    q = module.apply_rotary_emb(F.rms_norm(q, (HEAD_DIM,)), cos, sin)
    k = module.apply_rotary_emb(F.rms_norm(k, (HEAD_DIM,)), cos, sin)
    q2 = module.apply_rotary_emb(F.rms_norm(q2, (HEAD_DIM,)), cos, sin)
    k2 = module.apply_rotary_emb(F.rms_norm(k2, (HEAD_DIM,)), cos, sin)
    score_a = torch.einsum("bqhd,bkhd->bhqk", q, k) / HEAD_DIM
    score_b = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HEAD_DIM
    return score_a, score_b, value


def _attention_write(attention, parts):
    score_a, score_b, value = parts
    length = score_a.shape[-1]
    pattern = score_a * score_b
    causal = torch.tril(torch.ones(
        length, length, dtype=torch.bool, device=pattern.device))
    pattern = pattern.masked_fill(~causal, 0)
    heads = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
    flattened = heads.transpose(1, 2).contiguous().view(
        score_a.shape[0], length, D)
    return _linear(flattened, attention.c_proj.weight)


def _relative_squared(left, right) -> float:
    left = torch.as_tensor(left).double()
    right = torch.as_tensor(right).double()
    return float((left - right).square().sum() / right.square().sum().clamp_min(1e-30))


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(PARENT_RESULT.read_text())
    if receipt.get("rung") != 483 \
            or receipt.get("pred_b_tangent_predicts_physical_T_I") is not False \
            or receipt.get("strong_null") is not True \
            or receipt.get("validation_licensed_and_opened") is not False \
            or receipt.get("next_step") != "task_conditioned_reader_functionals_or_finite_interchange":
        raise RuntimeError("rung483 did not license the finite task-conditioned route")
    rows, fit_rows, metadata = parent.validate_inputs()
    tokens = rows[:, :TOKENS].contiguous()
    positive = induction.induction_fetch_mask(tokens).any(-1).cpu()
    counts = tuple(int(positive[start:stop].sum()) for start, stop in (
        (0, 250), (250, 500), (500, 750), (750, 1000)))
    if counts != EXPECTED_POSITIVE_QUARTERS:
        raise RuntimeError(f"equality-positive support changed: {counts}")
    return rows, positive, fit_rows, {**metadata, "positive_quarter_counts": list(counts)}


@torch.no_grad()
def _native_forward(model, tokens, reference):
    capture = {}
    audit = {"attention": 0, "mlp": 0}

    def attention(event):
        audit["attention"] += 1
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 1:
            parts = _attention_parts(event.block.attn, event.state, event.first_value)
            rebuilt = _attention_write(event.block.attn, parts)
            parts32 = _attention_parts(
                event.block.attn, event.state.float(), event.first_value.float())
            rebuilt32 = _attention_write(event.block.attn, parts32)
            direct32, _ = event.block.attn(
                event.state.float(), event.first_value.float())
            capture["native_parts"] = tuple(value.detach().clone() for value in parts)
            capture["native_attention1"] = write.detach().clone()
            capture["native_factor_bf16_error"] = _relative_squared(rebuilt, write)
            capture["native_factor_float32_error"] = _relative_squared(rebuilt32, direct32)
            capture["first_value1"] = event.first_value.detach().clone()
        return write, first_value

    # The exact rung401 branches are recovered after this native replay by the
    # independently checked parent prefix helper.
    def mlp_capture(event):
        audit["mlp"] += 1
        write = event.block.mlp(event.state)
        if event.site == 0:
            capture["native_mlp0_state"] = event.state.detach().clone()
            capture["native_mlp0"] = write.detach().clone()
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp_capture, require_production=True)
    prefix = parent._native_prefix(model, tokens, reference)
    capture["branches"] = {name: prefix["branches"][name].detach().clone()
                           for name in BRANCHES}
    capture["branch_identity"] = {
        "analytical_num": prefix["analytical_num"],
        "analytical_den": prefix["analytical_den"],
        "deployed_num": prefix["deployed_num"],
        "deployed_den": prefix["deployed_den"],
    }
    capture["prefix_attention1_error"] = _relative_squared(
        prefix["a1"], capture["native_attention1"])
    return logits, capture, audit


@torch.no_grad()
def _arm_forward(model, tokens, capture, branch_name, mask):
    branch = capture["branches"][branch_name]
    audit = {"attention": 0, "mlp": 0, "site0": 0, "site1": 0}
    errors = {
        "mlp0_state_max_abs": 0.0, "first_value_max_abs": 0.0,
        "removed_factor_bf16_relative_squared": 0.0,
        "removed_factor_float32_relative_squared": 0.0,
        "all_native_relative_squared": 0.0,
    }

    def attention(event):
        audit["attention"] += 1
        if event.site != 1:
            return event.block.attn(event.state, event.first_value)
        audit["site1"] += 1
        errors["first_value_max_abs"] = float(
            (event.first_value - capture["first_value1"]).abs().max())
        removed_parts = _attention_parts(event.block.attn, event.state, event.first_value)
        removed_native, _ = event.block.attn(event.state, event.first_value)
        rebuilt_removed = _attention_write(event.block.attn, removed_parts)
        errors["removed_factor_bf16_relative_squared"] = _relative_squared(
            rebuilt_removed, removed_native)
        removed_parts32 = _attention_parts(
            event.block.attn, event.state.float(), event.first_value.float())
        rebuilt_removed32 = _attention_write(event.block.attn, removed_parts32)
        removed_native32, _ = event.block.attn(
            event.state.float(), event.first_value.float())
        errors["removed_factor_float32_relative_squared"] = _relative_squared(
            rebuilt_removed32, removed_native32)
        hybrid = tuple(
            capture["native_parts"][index] if mask & (1 << index) else removed_parts[index]
            for index in range(3))
        write = _attention_write(event.block.attn, hybrid)
        if mask == FULL_ARM:
            errors["all_native_relative_squared"] = _relative_squared(
                write, capture["native_attention1"])
        return write, event.first_value

    def mlp(event):
        audit["mlp"] += 1
        write = event.block.mlp(event.state)
        if event.site != 0:
            return write
        audit["site0"] += 1
        errors["mlp0_state_max_abs"] = float(
            (event.state - capture["native_mlp0_state"]).abs().max())
        return capture["native_mlp0"] - branch

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    return logits, audit, errors


def _per_token_ce(logits, targets):
    return F.cross_entropy(
        logits.float().transpose(1, 2), targets, reduction="none").detach().cpu()


def collect_phase(model, rows, positive, reference, start_doc, stop_doc):
    ce_batches = []
    native_batches = []
    audit = {
        "native_forwards": 0, "arm_forwards": 0,
        "native_attention_calls": 0, "native_mlp_calls": 0,
        "arm_attention_calls": 0, "arm_mlp_calls": 0,
        "arm_site0_calls": 0, "arm_site1_calls": 0,
    }
    errors = {
        "native_factor_bf16_relative_squared_max": 0.0,
        "native_factor_float32_relative_squared_max": 0.0,
        "removed_factor_bf16_relative_squared_max": 0.0,
        "removed_factor_float32_relative_squared_max": 0.0,
        "all_native_relative_squared_max": 0.0,
        "prefix_attention1_relative_squared_max": 0.0,
        "mlp0_state_max_abs": 0.0, "first_value_max_abs": 0.0,
        "analytical_num": 0.0, "analytical_den": 0.0,
        "deployed_num": 0.0, "deployed_den": 0.0,
    }
    device = next(model.parameters()).device
    for start in range(start_doc, stop_doc, BATCH):
        stop = min(start + BATCH, stop_doc)
        tokens = rows[start:stop, :-1].to(device)
        targets = rows[start:stop, 1:].to(device)
        native_logits, capture, native_audit = _native_forward(model, tokens, reference)
        audit["native_forwards"] += 1
        audit["native_attention_calls"] += native_audit["attention"]
        audit["native_mlp_calls"] += native_audit["mlp"]
        native_batches.append(_per_token_ce(native_logits, targets))
        errors["native_factor_bf16_relative_squared_max"] = max(
            errors["native_factor_bf16_relative_squared_max"],
            capture["native_factor_bf16_error"])
        errors["native_factor_float32_relative_squared_max"] = max(
            errors["native_factor_float32_relative_squared_max"],
            capture["native_factor_float32_error"])
        errors["prefix_attention1_relative_squared_max"] = max(
            errors["prefix_attention1_relative_squared_max"],
            capture["prefix_attention1_error"])
        identity = capture["branch_identity"]
        for key in ("analytical_num", "analytical_den", "deployed_num", "deployed_den"):
            errors[key] += identity[key]

        branch_ce = []
        for branch in BRANCHES:
            arms = []
            for mask in range(8):
                logits, arm_audit, arm_errors = _arm_forward(
                    model, tokens, capture, branch, mask)
                audit["arm_forwards"] += 1
                audit["arm_attention_calls"] += arm_audit["attention"]
                audit["arm_mlp_calls"] += arm_audit["mlp"]
                audit["arm_site0_calls"] += arm_audit["site0"]
                audit["arm_site1_calls"] += arm_audit["site1"]
                for key in ("removed_factor_bf16_relative_squared",
                            "removed_factor_float32_relative_squared",
                            "all_native_relative_squared"):
                    errors[f"{key}_max"] = max(
                        errors[f"{key}_max"], arm_errors[key])
                errors["mlp0_state_max_abs"] = max(
                    errors["mlp0_state_max_abs"], arm_errors["mlp0_state_max_abs"])
                errors["first_value_max_abs"] = max(
                    errors["first_value_max_abs"], arm_errors["first_value_max_abs"])
                arms.append(_per_token_ce(logits, targets))
            branch_ce.append(torch.stack(arms, dim=-1))
        ce_batches.append(torch.stack(branch_ce, dim=0))
        del native_logits, capture, branch_ce

    batches = math.ceil((stop_doc - start_doc) / BATCH)
    expected = {
        "native_forwards": batches, "arm_forwards": 16 * batches,
        "native_attention_calls": 18 * batches,
        "native_mlp_calls": 18 * batches,
        "arm_attention_calls": 18 * 16 * batches,
        "arm_mlp_calls": 18 * 16 * batches,
        "arm_site0_calls": 16 * batches,
        "arm_site1_calls": 16 * batches,
    }
    ce = torch.cat(ce_batches, dim=1)
    native_ce = torch.cat(native_batches, dim=0)
    phase_positive = positive[start_doc:stop_doc].clone()
    instrument = {
        "calls": audit, "expected_calls": expected, "calls_exact": audit == expected,
        "native_factor_bf16_relative_squared_max":
            errors["native_factor_bf16_relative_squared_max"],
        "native_factor_float32_relative_squared_max":
            errors["native_factor_float32_relative_squared_max"],
        "removed_factor_bf16_relative_squared_max":
            errors["removed_factor_bf16_relative_squared_max"],
        "removed_factor_float32_relative_squared_max":
            errors["removed_factor_float32_relative_squared_max"],
        "all_native_relative_squared_max": errors["all_native_relative_squared_max"],
        "prefix_attention1_relative_squared_max":
            errors["prefix_attention1_relative_squared_max"],
        "mlp0_state_max_abs": errors["mlp0_state_max_abs"],
        "first_value_max_abs": errors["first_value_max_abs"],
        "analytical_branch_identity_relative_squared": errors["analytical_num"]
            / max(errors["analytical_den"], 1e-30),
        "deployed_branch_identity_relative_squared": errors["deployed_num"]
            / max(errors["deployed_den"], 1e-30),
        "documents": stop_doc - start_doc,
        "positions": int((stop_doc - start_doc) * TOKENS),
        "equality_positive_positions": int(phase_positive.sum()),
    }
    return ce, native_ce, phase_positive, instrument


def _view(value, mask):
    return value[mask]


def _effect_report(predictor, target):
    return {
        "cosine": _cosine(predictor, target),
        "fit_scalar": _scaled_error(predictor, target)[0],
        "best_scalar_adjusted_relative_error": _scaled_error(predictor, target)[1],
        "predictor_rms": float(torch.as_tensor(predictor).double().square().mean().sqrt()),
        "target_rms": float(torch.as_tensor(target).double().square().mean().sqrt()),
    }


def _select_mask(benefits, positive):
    target = _view(benefits[..., FULL_ARM], positive)
    candidates = []
    for mask in range(1, FULL_ARM):
        report = _effect_report(_view(benefits[..., mask], positive), target)
        report.update({"mask": mask, "name": _mask_name(mask),
                       "components": mask.bit_count()})
        report["eligible"] = bool(
            report["cosine"] >= .90
            and report["best_scalar_adjusted_relative_error"] <= .35)
        candidates.append(report)
    eligible = [row for row in candidates if row["eligible"]]
    selected = min(eligible, key=lambda row: (
        row["components"], row["best_scalar_adjusted_relative_error"], row["mask"])) \
        if eligible else None
    return selected, candidates


def _profile(performance, positive):
    effects = _mobius(performance)
    values = torch.stack([
        _view(effects[..., mask], positive).double().mean() for mask in range(1, 8)])
    return values / values.abs().sum().clamp_min(1e-30), effects


def _shuffle_cosines(predictor, target, positive):
    return [
        _cosine(_view(predictor, positive), _view(torch.roll(
            target, shifts=shift, dims=1), positive))
        for shift in POSITION_SHIFTS
    ]


def _shuffle_absolute_means(value, positive):
    return [
        float(_view(torch.roll(value, shifts=shift, dims=1), positive).double().abs().mean())
        for shift in POSITION_SHIFTS
    ]


def analyze_phase(ce, positive, split_index, frozen_masks=None):
    ce = ce.double()
    performance = -ce
    benefits = ce[..., 0, None] - ce
    half_slices = (slice(0, split_index), slice(split_index, ce.shape[1]))
    reports = {branch: {"halves": []} for branch in BRANCHES}
    profiles = torch.zeros(2, 2, 7, dtype=torch.float64)
    descriptive_masks = {branch: [] for branch in BRANCHES}
    for bi, branch in enumerate(BRANCHES):
        for hi, docs in enumerate(half_slices):
            half_benefits = benefits[bi, docs]
            half_performance = performance[bi, docs]
            half_positive = positive[docs]
            selected, candidates = _select_mask(half_benefits, half_positive)
            descriptive_masks[branch].append(None if selected is None else selected["mask"])
            profile, mobius = _profile(half_performance, half_positive)
            profiles[hi, bi] = profile
            reports[branch]["halves"].append({
                "documents": [docs.start or 0, docs.stop],
                "positive_positions": int(half_positive.sum()),
                "descriptive_selected": selected,
                "candidate_reports": candidates,
                "path_profile": profile.tolist(),
                "mobius_route_closure_relative_squared": _relative_squared(
                    mobius[..., 1:].sum(-1),
                    performance[bi, docs, ..., FULL_ARM]
                    - performance[bi, docs, ..., 0]),
            })

    selected_masks = dict(frozen_masks or {
        branch: descriptive_masks[branch][0] for branch in BRANCHES})
    selection_live = all(mask is not None for mask in selected_masks.values())
    pred_b = selection_live
    pred_e = selection_live
    for bi, branch in enumerate(BRANCHES):
        mask = selected_masks[branch]
        reports[branch]["selected_mask"] = mask
        reports[branch]["selected_name"] = None if mask is None else _mask_name(mask)
        if mask is None:
            reports[branch]["heldout_selected_reports"] = None
            reports[branch]["task_selectivity"] = None
            continue
        held_docs = half_slices[1]
        held_positive = positive[held_docs]
        predictor = benefits[bi, held_docs, ..., mask]
        target = benefits[bi, held_docs, ..., FULL_ARM]
        eq_report = _effect_report(_view(predictor, held_positive),
                                   _view(target, held_positive))
        all_report = _effect_report(predictor, target)
        shuffle = _shuffle_cosines(predictor, target, held_positive)
        q95 = float(torch.quantile(torch.tensor(shuffle, dtype=torch.float64),
                                   .95, interpolation="higher"))
        b_holds = bool(
            eq_report["cosine"] >= .80
            and eq_report["best_scalar_adjusted_relative_error"] <= .50
            and all_report["cosine"] >= .80
            and all_report["best_scalar_adjusted_relative_error"] <= .50
            and eq_report["cosine"] >= q95 + .15)
        pred_b &= b_holds
        reports[branch]["heldout_selected_reports"] = {
            "equality_positive": eq_report, "all_positions": all_report,
            "position_shift_cosines": shuffle, "position_shift_q95": q95,
            "holds": b_holds,
        }

        task_halves = []
        signs = []
        for hi, docs in enumerate(half_slices):
            half_positive = positive[docs]
            selected_effect = benefits[bi, docs, ..., mask]
            target_effect = benefits[bi, docs, ..., FULL_ARM]
            eq_values = _view(selected_effect, half_positive)
            signed_mean = float(eq_values.mean())
            signs.append(math.copysign(1.0, signed_mean) if signed_mean else 0.0)
            controls = _shuffle_absolute_means(selected_effect, half_positive)
            control_median = float(torch.tensor(controls).median())
            eq_abs = float(eq_values.abs().mean())
            eq_cos = _cosine(eq_values, _view(target_effect, half_positive))
            all_cos = _cosine(selected_effect, target_effect)
            holds = bool(eq_abs >= 1.25 * control_median and eq_cos >= all_cos + .05)
            pred_e &= holds
            task_halves.append({
                "signed_mean_nat": signed_mean,
                "equality_mean_absolute_nat": eq_abs,
                "position_shift_absolute_means_nat": controls,
                "position_shift_median_absolute_nat": control_median,
                "equality_to_all_cosine_margin": eq_cos - all_cos,
                "holds_without_sign_transport": holds,
            })
        sign_transport = signs[0] != 0 and signs[0] == signs[1]
        pred_e &= sign_transport
        reports[branch]["task_selectivity"] = {
            "halves": task_halves, "signs": signs,
            "sign_transport": sign_transport,
        }

    profile_stability = {}
    pred_c = selection_live
    for bi, branch in enumerate(BRANCHES):
        cosine = _cosine(profiles[0, bi], profiles[1, bi])
        max_share_change = float((profiles[0, bi].abs() - profiles[1, bi].abs()).abs().max())
        same_descriptive = descriptive_masks[branch][0] is not None \
            and descriptive_masks[branch][0] == descriptive_masks[branch][1]
        holds = bool(cosine >= .85 and max_share_change <= .20 and same_descriptive)
        pred_c &= holds
        profile_stability[branch] = {
            "cosine": cosine, "maximum_absolute_share_change": max_share_change,
            "descriptive_masks": descriptive_masks[branch],
            "descriptive_names": [None if mask is None else _mask_name(mask)
                                  for mask in descriptive_masks[branch]],
            "holds": holds,
        }

    cross_branch_profile_cosines = [
        _cosine(profiles[half, 0], profiles[half, 1]) for half in range(2)]
    t_route = benefits[0, ..., FULL_ARM]
    i_route = benefits[1, ..., FULL_ARM]
    fit_positive = positive[half_slices[0]]
    held_positive = positive[half_slices[1]]
    alpha, _ = _scaled_error(_view(t_route[half_slices[0]], fit_positive),
                             _view(i_route[half_slices[0]], fit_positive))
    held_shared_error = float(torch.linalg.vector_norm(
        _view(i_route[half_slices[1]], held_positive)
        - alpha * _view(t_route[half_slices[1]], held_positive))
        / torch.linalg.vector_norm(
            _view(i_route[half_slices[1]], held_positive)).clamp_min(1e-30))
    shared = bool(
        selection_live
        and min(cross_branch_profile_cosines) >= .90
        and selected_masks["T"] == selected_masks["I"]
        and held_shared_error <= .35)

    cross_use = []
    split = bool(selection_live and selected_masks["T"] != selected_masks["I"]
                 and max(cross_branch_profile_cosines) <= .60)
    if selection_live:
        for owner in range(2):
            mask = selected_masks[BRANCHES[owner]]
            for half, docs in enumerate(half_slices):
                half_positive = positive[docs]
                own = _cosine(
                    _view(benefits[owner, docs, ..., mask], half_positive),
                    _view(benefits[owner, docs, ..., FULL_ARM], half_positive))
                other = 1 - owner
                transferred = _cosine(
                    _view(benefits[other, docs, ..., mask], half_positive),
                    _view(benefits[other, docs, ..., FULL_ARM], half_positive))
                holds = own >= transferred + .20
                split &= holds
                cross_use.append({
                    "mask_owner": BRANCHES[owner], "mask": mask,
                    "mask_name": _mask_name(mask), "half": half,
                    "own_cosine": own, "other_branch_cosine": transferred,
                    "margin": own - transferred, "holds": holds,
                })

    relation = "shared" if shared else "split" if split else None
    closure_ok = all(
        row["mobius_route_closure_relative_squared"] <= 1e-8
        for branch in BRANCHES for row in reports[branch]["halves"])
    return {
        "branch_reports": reports,
        "selected_masks": selected_masks,
        "selected_names": {key: None if value is None else _mask_name(value)
                           for key, value in selected_masks.items()},
        "profile_stability": profile_stability,
        "cross_branch_profile_cosines": cross_branch_profile_cosines,
        "shared_scale_fit_half0": alpha,
        "shared_scale_relative_error_half1": held_shared_error,
        "cross_use_reports": cross_use,
        "relation": relation, "shared_holds": shared, "split_holds": split,
        "exactly_one_relation": shared != split,
        "pred_b_physical_path_predicts_route": bool(pred_b),
        "pred_c_path_decomposition_stable": bool(pred_c),
        "pred_e_task_informative": bool(pred_e),
        "mobius_closure_holds": closure_ok,
    }


def _instrument_valid(instrument, analysis, expected_positive):
    return bool(
        instrument["calls_exact"]
        and instrument["native_factor_float32_relative_squared_max"] <= 1e-8
        and instrument["removed_factor_float32_relative_squared_max"] <= 1e-8
        and instrument["native_factor_bf16_relative_squared_max"] <= 1e-5
        and instrument["removed_factor_bf16_relative_squared_max"] <= 1e-5
        and instrument["all_native_relative_squared_max"] <= 1e-5
        and instrument["prefix_attention1_relative_squared_max"] <= 1e-12
        and instrument["mlp0_state_max_abs"] == 0.0
        and instrument["first_value_max_abs"] == 0.0
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and instrument["equality_positive_positions"] == expected_positive
        and analysis["mobius_closure_holds"])


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert BRANCHES == ("T", "I") and COMPONENTS == ("A", "B", "V")
        assert tuple(mask.bit_count() for mask in range(1, FULL_ARM)) == (1, 1, 2, 1, 2, 2)
        assert len(set(POSITION_SHIFTS)) == 16
        print(json.dumps({
            "status": "dry_run_passed", "rung": 484,
            "model_loaded": False, "outcomes_opened": False,
            "validation_outcomes_opened": False, "final_or_sealed_opened": False,
            "discovery_forwards": (500 // BATCH) * 17,
            "conditional_validation_forwards": (500 // BATCH) * 17,
            "proper_subset_masks": list(range(1, FULL_ARM)),
            "registered_predictions": ["pred_a", "pred_b", "pred_c",
                                       "pred_d_shared", "pred_d_split",
                                       "pred_e", "pred_f"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung484 output namespace already exists")
    rows, positive, fit_rows, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    discovery_ce, discovery_native_ce, discovery_positive, discovery_instrument = \
        collect_phase(model, rows, positive, reference, *DISCOVERY_RANGE)
    discovery = analyze_phase(
        discovery_ce, discovery_positive, DISCOVERY_SPLIT - DISCOVERY_RANGE[0])
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and _instrument_valid(
            discovery_instrument, discovery,
            sum(EXPECTED_POSITIVE_QUARTERS[:2])))
    pred_b = discovery["pred_b_physical_path_predicts_route"]
    pred_c = discovery["pred_c_path_decomposition_stable"]
    relation = discovery["relation"]
    pred_e = discovery["pred_e_task_informative"]
    validation_licensed = bool(pred_a and pred_b and pred_c and pred_e
                               and relation is not None)
    validation = validation_instrument = None
    pred_f = False
    if validation_licensed:
        validation_ce, validation_native_ce, validation_positive, validation_instrument = \
            collect_phase(model, rows, positive, reference, *VALIDATION_RANGE)
        validation = analyze_phase(
            validation_ce, validation_positive,
            VALIDATION_SPLIT - VALIDATION_RANGE[0],
            frozen_masks=discovery["selected_masks"])
        validation_descriptive_same = all(
            all(mask == discovery["selected_masks"][branch]
                for mask in validation["profile_stability"][branch]["descriptive_masks"])
            for branch in BRANCHES)
        pred_f = bool(
            _instrument_valid(
                validation_instrument, validation,
                sum(EXPECTED_POSITIVE_QUARTERS[2:]))
            and validation["pred_b_physical_path_predicts_route"]
            and validation["pred_c_path_decomposition_stable"]
            and validation["pred_e_task_informative"]
            and validation["relation"] == relation
            and validation_descriptive_same)
        del validation_ce, validation_native_ce
    strong_null = bool(not pred_a or not pred_b or not pred_c or not pred_e
                       or relation is None)
    result = {
        "status": "complete", "rung": 484,
        "claim_level": "exact_finite_attention1_path_identification_screen",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(BRANCHES), "attention_components": list(COMPONENTS),
        "component_arms": {str(mask): _mask_name(mask) for mask in range(8)},
        "position_shift_offsets": list(POSITION_SHIFTS),
        "discovery": {
            "documents": list(DISCOVERY_RANGE), "split": DISCOVERY_SPLIT,
            "instrument": discovery_instrument, "analysis": discovery,
            "native_ce_summary": {
                "mean": float(discovery_native_ce.mean()),
                "rms": float(discovery_native_ce.double().square().mean().sqrt()),
            },
        },
        "validation": None if validation is None else {
            "documents": list(VALIDATION_RANGE), "split": VALIDATION_SPLIT,
            "instrument": validation_instrument, "analysis": validation,
        },
        "selected_relation": relation,
        'pred_a_exact_lawful_instrument': pred_a,
        'pred_b_physical_attention_path': pred_b,
        'pred_c_stable_path_decomposition': pred_c,
        'pred_d_exactly_one_T_I_path_relation': relation is not None,
        "d_shared_holds": discovery["shared_holds"],
        "d_split_holds": discovery["split_holds"],
        'pred_e_task_informative': pred_e,
        'pred_f_heldout_documents': pred_f,
        "validation_licensed_and_opened": validation_licensed,
        "strong_null": strong_null,
        "final_or_sealed_opened": False,
        "execution_price": {
            "discovery_full_model_forwards": discovery_instrument["calls"]["native_forwards"]
                + discovery_instrument["calls"]["arm_forwards"],
            "validation_full_model_forwards": 0 if validation_instrument is None else
                validation_instrument["calls"]["native_forwards"]
                + validation_instrument["calls"]["arm_forwards"],
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "below_head_selected_attention1_path_grouping_and_finite_interchange"
            if pred_f and relation == "split" else
            "exact_cross_branch_shared_path_interchange"
            if pred_f and relation == "shared" else
            "direct_mlp1_finite_bilinear_path_factorial"
            if not pred_b else
            "context_conditioned_finite_path_mixture"
            if not pred_c else
            "different_behavior_or_per_token_reader_learning"
            if not pred_e else
            "heldout_path_relation_failed_no_claim"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 484,
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "selected_relation": relation,
        "selected_paths": discovery["selected_names"],
        "strong_null": strong_null,
        "validation_opened": validation_licensed,
        "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
