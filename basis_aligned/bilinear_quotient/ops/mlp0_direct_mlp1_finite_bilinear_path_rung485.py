#!/usr/bin/env python3
"""RUNG485 -- exact finite T/I paths through direct MLP1 bilinear sides.

Attention1 is restored native while T or I remains absent from MLP0.  MLP1 is
rebuilt from all four native/absent combinations of its Left and Right product
activations, then the complete suffix is evaluated.  A separate frozen-token
test asks whether T's direct downstream effects repeat by input token identity.
"""

# BQGATE: EXPERIMENT
# pred_a exact direct-state replay, bilinear factorization, calls, and closure
# pred_b one proper MLP1 side predicts each complete direct finite route
# pred_c the signed Left/Right/interaction path profiles transfer across halves
# pred_d exactly one shared-versus-split T/I direct path relation holds
# pred_e input token identity predicts the token-only downstream effect
# pred_f the path relation and frozen token predictor validate on held-out documents

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
import mlp0_attention1_finite_path_factorial_rung484 as parent
import mlp0_immediate_consumer_quotient_rung483 as branch_parent
import mlp0_centered_context_anova_factorial as component_parent


PREREG = POLY / "MLP0_DIRECT_MLP1_FINITE_BILINEAR_PATH_RUNG485_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp0_attention1_finite_path_factorial_rung484.py"
PARENT_RESULT = ROOT / "mlp0_attention1_finite_path_factorial_rung484_results.json"
OUT = ROOT / "mlp0_direct_mlp1_finite_bilinear_path_rung485_results.json"
HASHES = {
    PREREG: "08386dc78f3233abd2936ec9efcfaebd0524466ca4d33fe9d2070381b45fd563",
    PARENT_SOURCE: "42f66fba01361c976660554197fef7aa66cb20d80eb5b6351b01a1f6e3bf9d54",
    PARENT_RESULT: "7f24d3fb2280bf3d1e8fb49667db28c5bd58c9ce41906fdcdc0c35288a71f90b",
    ROOT / "ops/mlp0_immediate_consumer_quotient_rung483.py":
        "9763502b99b8693826a5985c8f25a3ebe7763c3cd176c3aebeeb140833a61f4c",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
}
BRANCHES = ("T", "I")
SIDES = ("L", "R")
FULL_ARM = 3
DISCOVERY_RANGE = (0, 500)
DISCOVERY_SPLIT = 250
VALIDATION_RANGE = (500, 1000)
VALIDATION_SPLIT = 750
BATCH = 4
TOKENS = 256
VOCAB = facade.TOKENIZER_VOCAB
POSITION_SHIFTS = parent.POSITION_SHIFTS
EXPECTED_DISCOVERY_TOKENS = 698
EXPECTED_VALIDATION_SUPPORTED = 656


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _side_name(mask: int) -> str:
    names = [name for index, name in enumerate(SIDES) if mask & (1 << index)]
    return "+".join(names) if names else "EMPTY"


def _linear(value, weight):
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def _mlp_parts(mlp, state):
    return _linear(state, mlp.Left.weight), _linear(state, mlp.Right.weight)


def _mlp_write(mlp, parts):
    left, right = parts
    write = _linear(left * right, mlp.Down.weight)
    return write + mlp.Down_bias.to(device=write.device, dtype=write.dtype)


def _mobius(performance):
    performance = torch.as_tensor(performance, dtype=torch.float64)
    if performance.shape[-1] != 4:
        raise ValueError("two-side factorial must have four arms")
    output = torch.zeros_like(performance)
    for mask in range(4):
        for child in range(4):
            if child & ~mask:
                continue
            sign = -1.0 if ((mask.bit_count() - child.bit_count()) % 2) else 1.0
            output[..., mask] += sign * performance[..., child]
    return output


def _pearson(left, right) -> float:
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    return parent._cosine(left - left.mean(), right - right.mean())


def _select_side(benefits):
    target = benefits[..., FULL_ARM]
    candidates = []
    for mask in (1, 2):
        report = parent._effect_report(benefits[..., mask], target)
        report.update({"mask": mask, "name": _side_name(mask)})
        report["eligible"] = bool(
            report["cosine"] >= .90
            and report["best_scalar_adjusted_relative_error"] <= .35)
        candidates.append(report)
    eligible = [row for row in candidates if row["eligible"]]
    selected = min(eligible, key=lambda row: (
        row["best_scalar_adjusted_relative_error"], row["mask"])) if eligible else None
    return selected, candidates


def _token_counts(token_ids):
    return torch.bincount(token_ids.reshape(-1), minlength=VOCAB).to(torch.int64)


def _aggregate_error_key(key):
    """Preserve quantities already named as maxima; otherwise record batch max."""
    return key if key.endswith("_max_abs") else f"{key}_max"


def _token_means(values, token_ids):
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    token_ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(-1)
    sums = torch.zeros(VOCAB, dtype=torch.float64)
    counts = torch.zeros(VOCAB, dtype=torch.float64)
    sums.scatter_add_(0, token_ids, values)
    counts.scatter_add_(0, token_ids, torch.ones_like(values))
    return sums / counts.clamp_min(1), counts


def _input_token_sets(rows):
    tokens = rows[:, :TOKENS].cpu()
    counts = [_token_counts(tokens[start:stop]) for start, stop in (
        (0, 250), (250, 500), (500, 750), (750, 1000))]
    discovery = torch.where((counts[0] >= 8) & (counts[1] >= 8))[0]
    validation = discovery[(counts[2][discovery] >= 4) & (counts[3][discovery] >= 4)]
    if len(discovery) != EXPECTED_DISCOVERY_TOKENS \
            or len(validation) != EXPECTED_VALIDATION_SUPPORTED:
        raise RuntimeError("frozen token-support census changed")
    return tokens, discovery, validation, counts


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(PARENT_RESULT.read_text())
    if receipt.get("rung") != 484 \
            or receipt.get("pred_a_exact_lawful_instrument") is not True \
            or receipt.get("pred_b_physical_attention_path") is not False \
            or receipt.get("strong_null") is not True \
            or receipt.get("validation_licensed_and_opened") is not False \
            or receipt.get("next_step") != "direct_mlp1_finite_bilinear_path_factorial":
        raise RuntimeError("rung484 did not license the direct-MLP1 route")
    rows, positive, fit_rows, metadata = parent.validate_inputs()
    token_ids, discovery_tokens, validation_tokens, counts = _input_token_sets(rows)
    return rows, positive, fit_rows, token_ids, discovery_tokens, validation_tokens, {
        **metadata,
        "discovery_frequent_token_count": len(discovery_tokens),
        "discovery_frequent_positions": [
            int(counts[0][discovery_tokens].sum()),
            int(counts[1][discovery_tokens].sum()),
        ],
        "validation_supported_token_count": len(validation_tokens),
    }


@torch.no_grad()
def _native_forward(model, tokens, reference):
    capture = {}
    audit = {"attention": 0, "mlp": 0}

    def attention(event):
        audit["attention"] += 1
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 1:
            capture["native_attention1"] = write.detach().clone()
        return write, first_value

    def mlp(event):
        audit["mlp"] += 1
        write = event.block.mlp(event.state)
        if event.site == 0:
            capture["native_mlp0_state"] = event.state.detach().clone()
            capture["native_mlp0"] = write.detach().clone()
        elif event.site == 1:
            parts = _mlp_parts(event.block.mlp, event.state)
            rebuilt = _mlp_write(event.block.mlp, parts)
            parts32 = _mlp_parts(event.block.mlp, event.state.float())
            rebuilt32 = _mlp_write(event.block.mlp, parts32)
            direct32 = event.block.mlp(event.state.float())
            capture["native_mlp1_state"] = event.state.detach().clone()
            capture["native_mlp1"] = write.detach().clone()
            capture["native_parts"] = tuple(value.detach().clone() for value in parts)
            capture["native_factor_bf16_error"] = parent._relative_squared(rebuilt, write)
            capture["native_factor_float32_error"] = parent._relative_squared(
                rebuilt32, direct32)
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    prefix = branch_parent._native_prefix(model, tokens, reference)
    capture["branches"] = {name: prefix["branches"][name].detach().clone()
                           for name in BRANCHES}
    capture["branch_identity"] = {
        "analytical_num": prefix["analytical_num"],
        "analytical_den": prefix["analytical_den"],
        "deployed_num": prefix["deployed_num"],
        "deployed_den": prefix["deployed_den"],
    }
    capture["prefix_attention1_error"] = parent._relative_squared(
        prefix["a1"], capture["native_attention1"])
    capture["prefix_mlp1_error"] = parent._relative_squared(
        prefix["m1"], capture["native_mlp1"])
    return logits, capture, audit


@torch.no_grad()
def _arm_forward(model, tokens, capture, branch_name, mask):
    branch = capture["branches"][branch_name]
    audit = {
        "attention": 0, "mlp": 0, "site0_removal": 0,
        "attention1_restore": 0, "mlp1_injection": 0,
    }
    errors = {
        "mlp0_state_max_abs": 0.0,
        "attention1_write_max_abs": 0.0,
        "absent_factor_bf16_relative_squared": 0.0,
        "absent_factor_float32_relative_squared": 0.0,
        "all_native_relative_squared": 0.0,
    }

    def attention(event):
        audit["attention"] += 1
        if event.site != 1:
            return event.block.attn(event.state, event.first_value)
        audit["attention1_restore"] += 1
        return capture["native_attention1"], event.first_value

    def mlp(event):
        audit["mlp"] += 1
        if event.site == 0:
            audit["site0_removal"] += 1
            errors["mlp0_state_max_abs"] = float(
                (event.state - capture["native_mlp0_state"]).abs().max())
            return capture["native_mlp0"] - branch
        if event.site != 1:
            return event.block.mlp(event.state)
        audit["mlp1_injection"] += 1
        errors["attention1_write_max_abs"] = float(
            (event.attention_write - capture["native_attention1"]).abs().max())
        absent_parts = _mlp_parts(event.block.mlp, event.state)
        absent_native = event.block.mlp(event.state)
        rebuilt_absent = _mlp_write(event.block.mlp, absent_parts)
        errors["absent_factor_bf16_relative_squared"] = parent._relative_squared(
            rebuilt_absent, absent_native)
        absent_parts32 = _mlp_parts(event.block.mlp, event.state.float())
        rebuilt_absent32 = _mlp_write(event.block.mlp, absent_parts32)
        absent_native32 = event.block.mlp(event.state.float())
        errors["absent_factor_float32_relative_squared"] = parent._relative_squared(
            rebuilt_absent32, absent_native32)
        hybrid = tuple(
            capture["native_parts"][index] if mask & (1 << index) else absent_parts[index]
            for index in range(2))
        write = _mlp_write(event.block.mlp, hybrid)
        if mask == FULL_ARM:
            errors["all_native_relative_squared"] = parent._relative_squared(
                write, capture["native_mlp1"])
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    return logits, audit, errors


def collect_phase(model, rows, reference, start_doc, stop_doc):
    ce_batches = []
    native_batches = []
    audit = {
        "native_forwards": 0, "arm_forwards": 0,
        "native_attention_calls": 0, "native_mlp_calls": 0,
        "arm_attention_calls": 0, "arm_mlp_calls": 0,
        "site0_removal_calls": 0, "attention1_restore_calls": 0,
        "mlp1_injection_calls": 0,
    }
    errors = {
        "native_factor_bf16_relative_squared_max": 0.0,
        "native_factor_float32_relative_squared_max": 0.0,
        "absent_factor_bf16_relative_squared_max": 0.0,
        "absent_factor_float32_relative_squared_max": 0.0,
        "all_native_relative_squared_max": 0.0,
        "prefix_attention1_relative_squared_max": 0.0,
        "prefix_mlp1_relative_squared_max": 0.0,
        "mlp0_state_max_abs": 0.0,
        "attention1_write_max_abs": 0.0,
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
        native_batches.append(parent._per_token_ce(native_logits, targets))
        for key in ("native_factor_bf16_error", "native_factor_float32_error"):
            errors[f"{key.replace('_error', '_relative_squared')}_max"] = max(
                errors[f"{key.replace('_error', '_relative_squared')}_max"], capture[key])
        errors["prefix_attention1_relative_squared_max"] = max(
            errors["prefix_attention1_relative_squared_max"],
            capture["prefix_attention1_error"])
        errors["prefix_mlp1_relative_squared_max"] = max(
            errors["prefix_mlp1_relative_squared_max"], capture["prefix_mlp1_error"])
        for key in ("analytical_num", "analytical_den", "deployed_num", "deployed_den"):
            errors[key] += capture["branch_identity"][key]

        branch_ce = []
        for branch in BRANCHES:
            arms = []
            for mask in range(4):
                logits, arm_audit, arm_errors = _arm_forward(
                    model, tokens, capture, branch, mask)
                audit["arm_forwards"] += 1
                audit["arm_attention_calls"] += arm_audit["attention"]
                audit["arm_mlp_calls"] += arm_audit["mlp"]
                audit["site0_removal_calls"] += arm_audit["site0_removal"]
                audit["attention1_restore_calls"] += arm_audit["attention1_restore"]
                audit["mlp1_injection_calls"] += arm_audit["mlp1_injection"]
                for key in arm_errors:
                    aggregate_key = _aggregate_error_key(key)
                    errors[aggregate_key] = max(errors[aggregate_key], arm_errors[key])
                arms.append(parent._per_token_ce(logits, targets))
            branch_ce.append(torch.stack(arms, dim=-1))
        ce_batches.append(torch.stack(branch_ce, dim=0))
        del native_logits, capture, branch_ce

    batches = math.ceil((stop_doc - start_doc) / BATCH)
    expected = {
        "native_forwards": batches, "arm_forwards": 8 * batches,
        "native_attention_calls": 18 * batches, "native_mlp_calls": 18 * batches,
        "arm_attention_calls": 18 * 8 * batches,
        "arm_mlp_calls": 18 * 8 * batches,
        "site0_removal_calls": 8 * batches,
        "attention1_restore_calls": 8 * batches,
        "mlp1_injection_calls": 8 * batches,
    }
    ce = torch.cat(ce_batches, dim=1)
    native_ce = torch.cat(native_batches, dim=0)
    instrument = {
        "calls": audit, "expected_calls": expected, "calls_exact": audit == expected,
        **{key: value for key, value in errors.items()
           if key not in ("analytical_num", "analytical_den", "deployed_num", "deployed_den")},
        "analytical_branch_identity_relative_squared": errors["analytical_num"]
            / max(errors["analytical_den"], 1e-30),
        "deployed_branch_identity_relative_squared": errors["deployed_num"]
            / max(errors["deployed_den"], 1e-30),
        "documents": stop_doc - start_doc,
        "positions": (stop_doc - start_doc) * TOKENS,
    }
    return ce, native_ce, instrument


def _profile(performance, position_mask=None):
    effects = _mobius(performance)
    if position_mask is None:
        values = torch.stack([effects[..., mask].mean() for mask in range(1, 4)])
    else:
        values = torch.stack(
            [effects[..., mask][position_mask].mean() for mask in range(1, 4)])
    return values / values.abs().sum().clamp_min(1e-30), effects


def _frequent_token_report(route, token_ids, frequent):
    halves = (slice(0, 250), slice(250, 500))
    means, counts = [], []
    for docs in halves:
        mean, count = _token_means(route[docs], token_ids[docs])
        means.append(mean[frequent])
        counts.append(count[frequent])
    weight = torch.minimum(counts[0], counts[1]).sqrt()
    unweighted = parent._cosine(means[0], means[1])
    weighted = parent._cosine(weight * means[0], weight * means[1])
    dense_predictor = torch.zeros(VOCAB, dtype=torch.float64)
    dense_predictor[frequent] = means[0]
    held_tokens = token_ids[halves[1]]
    supported = torch.isin(held_tokens, frequent)
    prediction = dense_predictor[held_tokens][supported]
    target = route[halves[1]][supported]
    fit_tokens = token_ids[halves[0]]
    fit_supported = torch.isin(fit_tokens, frequent)
    global_mean = float(route[halves[0]][fit_supported].mean())
    global_prediction = torch.full_like(target, global_mean)
    token_rmse = float((target - prediction).square().mean().sqrt())
    global_rmse = float((target - global_prediction).square().mean().sqrt())
    improvement = 1.0 - token_rmse / max(global_rmse, 1e-30)
    pearson = _pearson(prediction, target)
    return {
        "token_ids": frequent.tolist(),
        "fit_token_means_nat": means[0].tolist(),
        "half1_token_means_nat": means[1].tolist(),
        "half_counts": [counts[0].to(torch.int64).tolist(),
                        counts[1].to(torch.int64).tolist()],
        "unweighted_profile_cosine": unweighted,
        "minimum_count_weighted_profile_cosine": weighted,
        "fit_global_mean_nat": global_mean,
        "heldout_position_count": int(supported.sum()),
        "token_mean_rmse_nat": token_rmse,
        "global_mean_rmse_nat": global_rmse,
        "rmse_improvement_over_global": improvement,
        "per_position_pearson": pearson,
    }


def analyze_phase(ce, token_ids, positive, split_index, frozen_sides=None):
    ce = ce.double()
    performance = -ce
    benefits = ce[..., 0, None] - ce
    halves = (slice(0, split_index), slice(split_index, ce.shape[1]))
    reports = {branch: {"halves": []} for branch in BRANCHES}
    profiles = torch.zeros(2, 2, 3, dtype=torch.float64)
    descriptive = {branch: [] for branch in BRANCHES}
    for bi, branch in enumerate(BRANCHES):
        for hi, docs in enumerate(halves):
            selected, candidates = _select_side(benefits[bi, docs])
            descriptive[branch].append(None if selected is None else selected["mask"])
            profile, mobius = _profile(performance[bi, docs])
            equality = positive[docs]
            equality_profile, _ = _profile(performance[bi, docs], equality)
            equality_candidates = []
            for mask in (1, 2):
                report = parent._effect_report(
                    benefits[bi, docs, ..., mask][equality],
                    benefits[bi, docs, ..., FULL_ARM][equality])
                report.update({"mask": mask, "name": _side_name(mask)})
                equality_candidates.append(report)
            profiles[hi, bi] = profile
            reports[branch]["halves"].append({
                "descriptive_selected": selected,
                "candidate_reports": candidates,
                "path_profile": profile.tolist(),
                "equality_positive_positions": int(equality.sum()),
                "equality_candidate_reports": equality_candidates,
                "equality_path_profile": equality_profile.tolist(),
                "mobius_route_closure_relative_squared": parent._relative_squared(
                    mobius[..., 1:].sum(-1),
                    performance[bi, docs, ..., FULL_ARM]
                    - performance[bi, docs, ..., 0]),
                "arm_benefit_rms_nat": [
                    float(benefits[bi, docs, ..., mask].square().mean().sqrt())
                    for mask in range(4)],
            })
    selected_sides = dict(frozen_sides or {
        branch: descriptive[branch][0] for branch in BRANCHES})
    live = all(mask is not None for mask in selected_sides.values())
    pred_b = live
    for bi, branch in enumerate(BRANCHES):
        mask = selected_sides[branch]
        reports[branch]["selected_side"] = mask
        reports[branch]["selected_name"] = None if mask is None else _side_name(mask)
        if mask is None:
            reports[branch]["heldout_selected"] = None
            continue
        predictor = benefits[bi, halves[1], ..., mask]
        target = benefits[bi, halves[1], ..., FULL_ARM]
        held = parent._effect_report(predictor, target)
        shifts = parent._shuffle_cosines(
            predictor, target, torch.ones_like(predictor, dtype=torch.bool))
        q95 = float(torch.quantile(torch.tensor(shifts, dtype=torch.float64),
                                   .95, interpolation="higher"))
        holds = bool(
            held["cosine"] >= .80
            and held["best_scalar_adjusted_relative_error"] <= .50
            and held["cosine"] >= q95 + .15)
        pred_b &= holds
        reports[branch]["heldout_selected"] = {
            **held, "position_shift_cosines": shifts,
            "position_shift_q95": q95, "holds": holds,
            "equality_positive": parent._effect_report(
                predictor[positive[halves[1]]],
                target[positive[halves[1]]]),
        }

    stability = {}
    pred_c = live
    for bi, branch in enumerate(BRANCHES):
        cosine = parent._cosine(profiles[0, bi], profiles[1, bi])
        share_change = float((profiles[0, bi].abs() - profiles[1, bi].abs()).abs().max())
        same_side = descriptive[branch][0] is not None \
            and descriptive[branch][0] == descriptive[branch][1]
        holds = bool(cosine >= .85 and share_change <= .20 and same_side)
        pred_c &= holds
        stability[branch] = {
            "cosine": cosine, "maximum_absolute_share_change": share_change,
            "descriptive_sides": descriptive[branch],
            "descriptive_names": [None if mask is None else _side_name(mask)
                                  for mask in descriptive[branch]],
            "holds": holds,
        }

    cross_profiles = [parent._cosine(profiles[half, 0], profiles[half, 1])
                      for half in range(2)]
    equality_cross_profiles = [parent._cosine(
        torch.tensor(reports["T"]["halves"][half]["equality_path_profile"]),
        torch.tensor(reports["I"]["halves"][half]["equality_path_profile"]))
        for half in range(2)]
    t_route = benefits[0, ..., FULL_ARM]
    i_route = benefits[1, ..., FULL_ARM]
    alpha, _ = parent._scaled_error(t_route[halves[0]], i_route[halves[0]])
    held_shared_error = float(torch.linalg.vector_norm(
        i_route[halves[1]] - alpha * t_route[halves[1]])
        / torch.linalg.vector_norm(i_route[halves[1]]).clamp_min(1e-30))
    shared = bool(live and min(cross_profiles) >= .90
                  and selected_sides["T"] == selected_sides["I"]
                  and held_shared_error <= .35)
    split = bool(live and max(cross_profiles) <= .60
                 and selected_sides["T"] != selected_sides["I"])
    cross_use = []
    if live:
        for owner in range(2):
            mask = selected_sides[BRANCHES[owner]]
            for half, docs in enumerate(halves):
                own = parent._cosine(
                    benefits[owner, docs, ..., mask],
                    benefits[owner, docs, ..., FULL_ARM])
                other = 1 - owner
                transferred = parent._cosine(
                    benefits[other, docs, ..., mask],
                    benefits[other, docs, ..., FULL_ARM])
                holds = own >= transferred + .20
                split &= holds
                cross_use.append({
                    "owner": BRANCHES[owner], "side": _side_name(mask),
                    "half": half, "own_cosine": own,
                    "other_branch_cosine": transferred,
                    "margin": own - transferred, "holds": holds,
                })
    relation = "shared" if shared else "split" if split else None
    closure = all(row["mobius_route_closure_relative_squared"] <= 1e-8
                  for branch in BRANCHES for row in reports[branch]["halves"])
    route_rms_live = all(
        min(reports[branch]["halves"][half]["arm_benefit_rms_nat"][1:]) > 0
        for branch in BRANCHES for half in range(2))
    return {
        "branch_reports": reports,
        "selected_sides": selected_sides,
        "selected_names": {key: None if value is None else _side_name(value)
                           for key, value in selected_sides.items()},
        "profile_stability": stability,
        "cross_branch_profile_cosines": cross_profiles,
        "equality_cross_branch_profile_cosines": equality_cross_profiles,
        "shared_scale_fit_half0": alpha,
        "shared_scale_relative_error_half1": held_shared_error,
        "cross_use_reports": cross_use,
        "shared_holds": shared, "split_holds": split,
        "relation": relation,
        "pred_b_physical_side_predicts_route": bool(pred_b),
        "pred_c_path_profile_stable": bool(pred_c),
        "mobius_closure_holds": closure,
        "all_branch_side_routes_live": route_rms_live,
        "complete_route_effects": benefits[..., FULL_ARM],
    }


def _discovery_token_analysis(analysis, token_ids, frequent):
    reports = {}
    for bi, branch in enumerate(BRANCHES):
        reports[branch] = _frequent_token_report(
            analysis["complete_route_effects"][bi], token_ids, frequent)
    t = reports["T"]
    pred_e = bool(
        t["unweighted_profile_cosine"] >= .70
        and t["minimum_count_weighted_profile_cosine"] >= .70
        and t["rmse_improvement_over_global"] >= .10
        and t["per_position_pearson"] >= .30)
    reports["T_minus_I"] = {
        key: reports["T"][key] - reports["I"][key]
        for key in ("unweighted_profile_cosine",
                    "minimum_count_weighted_profile_cosine",
                    "rmse_improvement_over_global", "per_position_pearson")
    }
    return reports, pred_e


def _frozen_token_validation(validation_analysis, validation_token_ids,
                             discovery_token_report, supported_tokens):
    fit_ids = torch.tensor(discovery_token_report["token_ids"], dtype=torch.long)
    fit_means = torch.tensor(
        discovery_token_report["fit_token_means_nat"], dtype=torch.float64)
    dense = torch.zeros(VOCAB, dtype=torch.float64)
    dense[fit_ids] = fit_means
    global_mean = discovery_token_report["fit_global_mean_nat"]
    route = validation_analysis["complete_route_effects"][BRANCHES.index("T")]
    reports = []
    holds = True
    for half, docs in enumerate((slice(0, 250), slice(250, 500))):
        tokens = validation_token_ids[docs]
        supported = torch.isin(tokens, supported_tokens)
        target = route[docs][supported]
        predictor = dense[tokens][supported]
        baseline = torch.full_like(target, global_mean)
        token_rmse = float((target - predictor).square().mean().sqrt())
        global_rmse = float((target - baseline).square().mean().sqrt())
        improvement = 1.0 - token_rmse / max(global_rmse, 1e-30)
        pearson = _pearson(predictor, target)
        passed = improvement >= .05 and pearson > 0
        holds &= passed
        reports.append({
            "half": half, "positions": int(supported.sum()),
            "token_mean_rmse_nat": token_rmse,
            "global_mean_rmse_nat": global_rmse,
            "rmse_improvement_over_global": improvement,
            "per_position_pearson": pearson, "holds": passed,
        })
    return reports, bool(holds)


def _instrument_valid(instrument, analysis):
    return bool(
        instrument["calls_exact"]
        and instrument["native_factor_float32_relative_squared_max"] <= 1e-8
        and instrument["absent_factor_float32_relative_squared_max"] <= 1e-8
        and instrument["native_factor_bf16_relative_squared_max"] <= 1e-5
        and instrument["absent_factor_bf16_relative_squared_max"] <= 1e-5
        and instrument["all_native_relative_squared_max"] <= 1e-5
        and instrument["prefix_attention1_relative_squared_max"] <= 1e-12
        and instrument["prefix_mlp1_relative_squared_max"] <= 1e-12
        and instrument["mlp0_state_max_abs"] == 0.0
        and instrument["attention1_write_max_abs"] == 0.0
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and analysis["mobius_closure_holds"]
        and analysis["all_branch_side_routes_live"])


def _serial_analysis(analysis):
    return {key: value for key, value in analysis.items()
            if key != "complete_route_effects"}


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert BRANCHES == ("T", "I") and SIDES == ("L", "R")
        assert len(set(POSITION_SHIFTS)) == 16
        print(json.dumps({
            "status": "dry_run_passed", "rung": 485,
            "model_loaded": False, "outcomes_opened": False,
            "validation_outcomes_opened": False,
            "final_or_sealed_opened": False,
            "discovery_forwards": (500 // BATCH) * 9,
            "conditional_validation_forwards": (500 // BATCH) * 9,
            "proper_side_masks": [1, 2],
            "registered_predictions": ["pred_a", "pred_b", "pred_c",
                                       "pred_d_shared", "pred_d_split",
                                       "pred_e", "pred_f"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung485 output namespace already exists")
    (rows, positive, fit_rows, input_tokens, discovery_tokens,
     validation_tokens, metadata) = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    discovery_ce, discovery_native_ce, discovery_instrument = collect_phase(
        model, rows, reference, *DISCOVERY_RANGE)
    discovery_analysis = analyze_phase(
        discovery_ce, input_tokens[0:500], positive[0:500], DISCOVERY_SPLIT)
    token_analysis, pred_e = _discovery_token_analysis(
        discovery_analysis, input_tokens[0:500], discovery_tokens)
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and _instrument_valid(discovery_instrument, discovery_analysis))
    pred_b = discovery_analysis["pred_b_physical_side_predicts_route"]
    pred_c = discovery_analysis["pred_c_path_profile_stable"]
    relation = discovery_analysis["relation"]
    validation_licensed = bool(pred_a and pred_b and pred_c and pred_e
                               and relation is not None)
    validation_analysis = validation_instrument = token_validation = None
    pred_f = False
    if validation_licensed:
        validation_ce, validation_native_ce, validation_instrument = collect_phase(
            model, rows, reference, *VALIDATION_RANGE)
        validation_analysis = analyze_phase(
            validation_ce, input_tokens[500:1000], positive[500:1000],
            VALIDATION_SPLIT - VALIDATION_RANGE[0],
            frozen_sides=discovery_analysis["selected_sides"])
        token_validation, tokens_hold = _frozen_token_validation(
            validation_analysis, input_tokens[500:1000],
            token_analysis["T"], validation_tokens)
        selected_stable = all(
            all(mask == discovery_analysis["selected_sides"][branch]
                for mask in validation_analysis["profile_stability"][branch]["descriptive_sides"])
            for branch in BRANCHES)
        pred_f = bool(
            _instrument_valid(validation_instrument, validation_analysis)
            and validation_analysis["pred_b_physical_side_predicts_route"]
            and validation_analysis["pred_c_path_profile_stable"]
            and validation_analysis["relation"] == relation
            and selected_stable and tokens_hold)
        del validation_ce, validation_native_ce
    strong_null = bool(not pred_a or not pred_b or not pred_c or not pred_e
                       or relation is None)
    result = {
        "status": "complete", "rung": 485,
        "claim_level": "exact_direct_mlp1_path_and_token_effect_identification_screen",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(BRANCHES), "mlp1_sides": list(SIDES),
        "arms": {str(mask): _side_name(mask) for mask in range(4)},
        "position_shift_offsets": list(POSITION_SHIFTS),
        "discovery": {
            "documents": list(DISCOVERY_RANGE), "split": DISCOVERY_SPLIT,
            "instrument": discovery_instrument,
            "analysis": _serial_analysis(discovery_analysis),
            "token_analysis": token_analysis,
            "native_ce_summary": {
                "mean": float(discovery_native_ce.mean()),
                "rms": float(discovery_native_ce.double().square().mean().sqrt()),
            },
        },
        "validation": None if validation_analysis is None else {
            "documents": list(VALIDATION_RANGE), "split": VALIDATION_SPLIT,
            "instrument": validation_instrument,
            "analysis": _serial_analysis(validation_analysis),
            "frozen_T_token_prediction": token_validation,
        },
        "selected_relation": relation,
        'pred_a_exact_lawful_instrument': pred_a,
        'pred_b_physical_mlp1_side': pred_b,
        'pred_c_stable_bilinear_path': pred_c,
        'pred_d_exactly_one_T_I_relation': relation is not None,
        "d_shared_holds": discovery_analysis["shared_holds"],
        "d_split_holds": discovery_analysis["split_holds"],
        'pred_e_token_identity_predicts_T_effect': pred_e,
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
            "separate_T_I_below_product_state_analysis"
            if pred_f and relation == "split" else
            "exact_cross_branch_shared_side_interchange"
            if pred_f and relation == "shared" else
            "token_effect_clustering_then_physical_interchange"
            if pred_f else
            "coupled_token_by_context_finite_response_tensor"
            if not pred_b else
            "context_conditioned_direct_path"
            if not pred_c else
            "live_state_conditioned_reader_for_MLP0_T"
            if not pred_e else
            "heldout_direct_path_failed_no_claim"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 485,
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "selected_sides": discovery_analysis["selected_names"],
        "selected_relation": relation,
        "T_token_effect": {
            key: token_analysis["T"][key] for key in (
                "unweighted_profile_cosine", "minimum_count_weighted_profile_cosine",
                "rmse_improvement_over_global", "per_position_pearson")},
        "strong_null": strong_null,
        "validation_opened": validation_licensed,
        "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
