#!/usr/bin/env python3
"""RUNG486 -- exact block-1 carrier cube and named bigram response test."""

# BQGATE: EXPERIMENT
# pred_a exact native/absent replay, carrier corners, calls, and closure
# pred_b complete seven-term carrier profiles transfer across halves
# pred_c previous-current pairs predict T better than current token alone
# pred_d exactly one shared-versus-split T/I response law holds
# pred_e the frozen response law validates on held-out documents

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import mlp0_direct_mlp1_finite_bilinear_path_rung485 as parent
import mlp0_immediate_consumer_quotient_rung483 as branch_parent
import mlp0_centered_context_anova_factorial as component_parent


PREREG = POLY / "MLP0_COUPLED_BLOCK1_BIGRAM_RESPONSE_RUNG486_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp0_direct_mlp1_finite_bilinear_path_rung485.py"
PARENT_RESULT = ROOT / "mlp0_direct_mlp1_finite_bilinear_path_rung485_results.json"
OUT = ROOT / "mlp0_coupled_block1_bigram_response_rung486_results.json"
HASHES = {
    PREREG: "24c741722dfbcad2c1e3f132c47f047178cbe363551505475cf5f2e5375d8090",
    PARENT_SOURCE: "2449de7bb291b4059c02795307d8b6ea917c348b31007c3c333530626dfd4e2f",
    PARENT_RESULT: "a1ecf427958b442542014e00b033c24eec4e7c6e1ed5dd43b18d4d8f95ace278",
    ROOT / "ops/mlp0_immediate_consumer_quotient_rung483.py":
        "9763502b99b8693826a5985c8f25a3ebe7763c3cd176c3aebeeb140833a61f4c",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
}
BRANCHES = ("T", "C", "I")
CARRIERS = ("D", "A", "M")
FULL_ARM = 7
DISCOVERY_RANGE = (0, 500)
VALIDATION_RANGE = (500, 1000)
SPLIT = 250
BATCH = 4
TOKENS = 256
VOCAB = facade.TOKENIZER_VOCAB
PAIR_THRESHOLD = 8
VALIDATION_PAIR_THRESHOLD = 4
EXPECTED_PAIRS = 287
EXPECTED_PAIR_POSITIONS = (7859, 8292)
EXPECTED_VALIDATION_PAIRS = 269
EXPECTED_VALIDATION_POSITIONS = (8016, 7626)
POSITION_SHIFTS = parent.POSITION_SHIFTS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _arm_name(mask):
    names = [name for index, name in enumerate(CARRIERS)
             if mask & (1 << index)]
    return "+".join(names) if names else "EMPTY"


def _mobius(performance):
    performance = torch.as_tensor(performance, dtype=torch.float64)
    if performance.shape[-1] != 8:
        raise ValueError("three-carrier cube must have eight arms")
    output = torch.zeros_like(performance)
    for mask in range(8):
        for child in range(8):
            if child & ~mask:
                continue
            sign = -1.0 if ((mask.bit_count() - child.bit_count()) % 2) else 1.0
            output[..., mask] += sign * performance[..., child]
    return output


def _cosine(left, right):
    return parent.parent._cosine(
        torch.as_tensor(left, dtype=torch.float64).reshape(-1),
        torch.as_tensor(right, dtype=torch.float64).reshape(-1))


def _pearson(left, right):
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    return _cosine(left - left.mean(), right - right.mean())


def _relative_error(prediction, target):
    return float(torch.linalg.vector_norm(target - prediction)
                 / torch.linalg.vector_norm(target).clamp_min(1e-30))


def _pair_ids(tokens):
    tokens = torch.as_tensor(tokens, dtype=torch.long)
    return tokens[:, 1:] + VOCAB * tokens[:, :-1]


def _counts(ids):
    unique, count = torch.unique(ids.reshape(-1), return_counts=True)
    return dict(zip(unique.tolist(), count.tolist()))


def _pair_census(input_tokens):
    ids = _pair_ids(input_tokens)
    dictionaries = [_counts(ids[start:stop]) for start, stop in (
        (0, 250), (250, 500), (500, 750), (750, 1000))]
    discovery = sorted(
        set(key for key, value in dictionaries[0].items()
            if value >= PAIR_THRESHOLD)
        & set(key for key, value in dictionaries[1].items()
              if value >= PAIR_THRESHOLD))
    validation = [key for key in discovery
                  if dictionaries[2].get(key, 0) >= VALIDATION_PAIR_THRESHOLD
                  and dictionaries[3].get(key, 0) >= VALIDATION_PAIR_THRESHOLD]
    discovery_positions = tuple(
        sum(dictionaries[half][key] for key in discovery) for half in (0, 1))
    validation_positions = tuple(
        sum(dictionaries[half][key] for key in validation) for half in (2, 3))
    if (len(discovery), discovery_positions, len(validation), validation_positions) != (
            EXPECTED_PAIRS, EXPECTED_PAIR_POSITIONS,
            EXPECTED_VALIDATION_PAIRS, EXPECTED_VALIDATION_POSITIONS):
        raise RuntimeError("frozen previous-current pair census changed")
    return (ids, torch.tensor(discovery, dtype=torch.long),
            torch.tensor(validation, dtype=torch.long), dictionaries)


def _membership(ids, groups):
    indices = torch.searchsorted(groups, ids)
    clipped = indices.clamp_max(len(groups) - 1)
    return (indices < len(groups)) & (groups[clipped] == ids), clipped


def _group_means(values, ids, groups):
    values = torch.as_tensor(values, dtype=torch.float64)
    ids = torch.as_tensor(ids, dtype=torch.long)
    supported, index = _membership(ids, groups)
    flat_values = values[supported]
    flat_index = index[supported]
    if flat_values.ndim == 1:
        flat_values = flat_values[:, None]
    sums = torch.zeros(len(groups), flat_values.shape[-1], dtype=torch.float64)
    sums.index_add_(0, flat_index, flat_values)
    counts = torch.bincount(flat_index, minlength=len(groups)).double()
    means = sums / counts[:, None].clamp_min(1)
    return means, counts, supported, index


def _lookup(means, supported, index):
    return means[index[supported]]


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(PARENT_RESULT.read_text())
    if receipt.get("rung") != 485 \
            or receipt.get("pred_a_exact_lawful_instrument") is not True \
            or any(receipt.get(key) is not False for key in (
                "pred_b_physical_mlp1_side", "pred_c_stable_bilinear_path",
                "pred_d_exactly_one_T_I_relation",
                "pred_e_token_identity_predicts_T_effect")) \
            or receipt.get("validation_licensed_and_opened") is not False \
            or receipt.get("strong_null") is not True \
            or receipt.get("next_step") != "coupled_token_by_context_finite_response_tensor":
        raise RuntimeError("rung485 did not license rung486")
    rows, positive, fit_rows, metadata = parent.parent.validate_inputs()
    input_tokens = rows[:, :TOKENS].cpu()
    pair_ids, discovery, validation, dictionaries = _pair_census(input_tokens)
    return rows, positive, fit_rows, input_tokens, pair_ids, discovery, validation, {
        **metadata,
        "discovery_pair_count": len(discovery),
        "discovery_pair_positions": list(EXPECTED_PAIR_POSITIONS),
        "validation_pair_count": len(validation),
        "validation_pair_positions": list(EXPECTED_VALIDATION_POSITIONS),
        "pair_ids_sha256": hashlib.sha256(discovery.numpy().tobytes()).hexdigest(),
        "validation_pair_ids_sha256": hashlib.sha256(validation.numpy().tobytes()).hexdigest(),
        "quarter_pair_dictionary_sizes": [len(value) for value in dictionaries],
    }


@torch.no_grad()
def _native_forward(model, tokens, reference):
    capture = {}
    calls = {"attention": 0, "mlp": 0}

    def attention(event):
        calls["attention"] += 1
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 1:
            capture["A"] = write.detach().clone()
        return write, first_value

    def mlp(event):
        calls["mlp"] += 1
        write = event.block.mlp(event.state)
        if event.site == 0:
            capture["D"] = write.detach().clone()
            capture["mlp0_state"] = event.state.detach().clone()
        elif event.site == 1:
            capture["M"] = write.detach().clone()
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    prefix = branch_parent._native_prefix(model, tokens, reference)
    capture["branches"] = {name: prefix["branches"][name].detach().clone()
                           for name in BRANCHES}
    capture["identity"] = {key: prefix[key] for key in (
        "analytical_num", "analytical_den", "deployed_num", "deployed_den")}
    capture["prefix_errors"] = {
        "D": parent.parent._relative_squared(prefix["m0"], capture["D"]),
        "A": parent.parent._relative_squared(prefix["a1"], capture["A"]),
        "M": parent.parent._relative_squared(prefix["m1"], capture["M"]),
    }
    return logits, capture, calls


@torch.no_grad()
def _absent_forward(model, tokens, native, branch):
    capture = {}
    calls = {"attention": 0, "mlp": 0, "site0_removal": 0}

    def attention(event):
        calls["attention"] += 1
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 1:
            capture["A"] = write.detach().clone()
        return write, first_value

    def mlp(event):
        calls["mlp"] += 1
        write = event.block.mlp(event.state)
        if event.site == 0:
            calls["site0_removal"] += 1
            capture["mlp0_state_error"] = float(
                (event.state - native["mlp0_state"]).abs().max())
            write = native["D"] - branch
            capture["D"] = write.detach().clone()
        elif event.site == 1:
            capture["M"] = write.detach().clone()
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    return logits, capture, calls


@torch.no_grad()
def _cube_forward(model, tokens, native, absent, mask):
    calls = {"attention": 0, "mlp": 0, "D": 0, "A": 0, "M": 0}

    def attention(event):
        calls["attention"] += 1
        if event.site != 1:
            return event.block.attn(event.state, event.first_value)
        calls["A"] += 1
        source = native if mask & 2 else absent
        return source["A"], event.first_value

    def mlp(event):
        calls["mlp"] += 1
        if event.site == 0:
            calls["D"] += 1
            source = native if mask & 1 else absent
            return source["D"]
        if event.site == 1:
            calls["M"] += 1
            source = native if mask & 4 else absent
            return source["M"]
        return event.block.mlp(event.state)

    return facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True), calls


def collect_phase(model, rows, reference, start_doc, stop_doc):
    ce_batches = []
    native_ce_batches = []
    calls = {
        "native_forwards": 0, "absent_forwards": 0, "cube_forwards": 0,
        "native_attention": 0, "native_mlp": 0,
        "absent_attention": 0, "absent_mlp": 0, "site0_removal": 0,
        "cube_attention": 0, "cube_mlp": 0,
        "D_injections": 0, "A_injections": 0, "M_injections": 0,
    }
    errors = {
        "native_prefix_D_relative_squared_max": 0.0,
        "native_prefix_A_relative_squared_max": 0.0,
        "native_prefix_M_relative_squared_max": 0.0,
        "absent_corner_logits_relative_squared_max": 0.0,
        "native_corner_logits_relative_squared_max": 0.0,
        "mlp0_state_max_abs": 0.0,
        "analytical_num": 0.0, "analytical_den": 0.0,
        "deployed_num": 0.0, "deployed_den": 0.0,
    }
    device = next(model.parameters()).device
    for start in range(start_doc, stop_doc, BATCH):
        stop = min(start + BATCH, stop_doc)
        tokens = rows[start:stop, :-1].to(device)
        targets = rows[start:stop, 1:].to(device)
        native_logits, native, native_calls = _native_forward(
            model, tokens, reference)
        calls["native_forwards"] += 1
        calls["native_attention"] += native_calls["attention"]
        calls["native_mlp"] += native_calls["mlp"]
        native_ce_batches.append(parent.parent._per_token_ce(native_logits, targets))
        for name, error in native["prefix_errors"].items():
            key = f"native_prefix_{name}_relative_squared_max"
            errors[key] = max(errors[key], error)
        for key in ("analytical_num", "analytical_den", "deployed_num", "deployed_den"):
            errors[key] += native["identity"][key]

        branch_ce = []
        for branch_name in BRANCHES:
            absent_logits, absent, absent_calls = _absent_forward(
                model, tokens, native, native["branches"][branch_name])
            calls["absent_forwards"] += 1
            calls["absent_attention"] += absent_calls["attention"]
            calls["absent_mlp"] += absent_calls["mlp"]
            calls["site0_removal"] += absent_calls["site0_removal"]
            errors["mlp0_state_max_abs"] = max(
                errors["mlp0_state_max_abs"], absent["mlp0_state_error"])
            arms = []
            for mask in range(8):
                logits, cube_calls = _cube_forward(
                    model, tokens, native, absent, mask)
                calls["cube_forwards"] += 1
                calls["cube_attention"] += cube_calls["attention"]
                calls["cube_mlp"] += cube_calls["mlp"]
                for name in CARRIERS:
                    calls[f"{name}_injections"] += cube_calls[name]
                if mask == 0:
                    errors["absent_corner_logits_relative_squared_max"] = max(
                        errors["absent_corner_logits_relative_squared_max"],
                        parent.parent._relative_squared(logits, absent_logits))
                elif mask == FULL_ARM:
                    errors["native_corner_logits_relative_squared_max"] = max(
                        errors["native_corner_logits_relative_squared_max"],
                        parent.parent._relative_squared(logits, native_logits))
                arms.append(parent.parent._per_token_ce(logits, targets))
            branch_ce.append(torch.stack(arms, dim=-1))
        ce_batches.append(torch.stack(branch_ce, dim=0))

    batches = math.ceil((stop_doc - start_doc) / BATCH)
    expected = {
        "native_forwards": batches,
        "absent_forwards": len(BRANCHES) * batches,
        "cube_forwards": 8 * len(BRANCHES) * batches,
        "native_attention": 18 * batches,
        "native_mlp": 18 * batches,
        "absent_attention": 18 * len(BRANCHES) * batches,
        "absent_mlp": 18 * len(BRANCHES) * batches,
        "site0_removal": len(BRANCHES) * batches,
        "cube_attention": 18 * 8 * len(BRANCHES) * batches,
        "cube_mlp": 18 * 8 * len(BRANCHES) * batches,
        "D_injections": 8 * len(BRANCHES) * batches,
        "A_injections": 8 * len(BRANCHES) * batches,
        "M_injections": 8 * len(BRANCHES) * batches,
    }
    instrument = {
        "calls": calls, "expected_calls": expected, "calls_exact": calls == expected,
        **{key: value for key, value in errors.items()
           if key not in ("analytical_num", "analytical_den",
                          "deployed_num", "deployed_den")},
        "analytical_branch_identity_relative_squared": errors["analytical_num"]
            / max(errors["analytical_den"], 1e-30),
        "deployed_branch_identity_relative_squared": errors["deployed_num"]
            / max(errors["deployed_den"], 1e-30),
        "documents": stop_doc - start_doc,
        "positions": (stop_doc - start_doc) * TOKENS,
    }
    return (torch.cat(ce_batches, dim=1),
            torch.cat(native_ce_batches, dim=0), instrument)


def _profile(effects, docs):
    values = torch.stack(
        [effects[docs, ..., mask].mean() for mask in range(1, 8)])
    return values / values.abs().sum().clamp_min(1e-30)


def analyze_carriers(ce, split_index=SPLIT):
    performance = -torch.as_tensor(ce, dtype=torch.float64)
    effects = _mobius(performance)
    benefits = ce.double()[..., 0, None] - ce.double()
    halves = (slice(0, split_index), slice(split_index, ce.shape[1]))
    reports = {}
    pred_b = True
    for branch_index, branch in enumerate(BRANCHES):
        profiles = [_profile(effects[branch_index], docs) for docs in halves]
        cosine = _cosine(*profiles)
        share_change = float((profiles[0].abs() - profiles[1].abs()).abs().max())
        holds = cosine >= .90 and share_change <= .15
        pred_b &= holds
        reports[branch] = {
            "profiles": [value.tolist() for value in profiles],
            "cross_half_cosine": cosine,
            "maximum_absolute_share_change": share_change,
            "holds": holds,
            "complete_route_rms_nat": [
                float(benefits[branch_index, docs, ..., FULL_ARM].square().mean().sqrt())
                for docs in halves],
            "term_rms_nat": [[
                float(effects[branch_index, docs, ..., mask].square().mean().sqrt())
                for mask in range(1, 8)] for docs in halves],
            "mobius_closure_relative_squared": [parent.parent._relative_squared(
                effects[branch_index, docs, ..., 1:].sum(-1),
                performance[branch_index, docs, ..., FULL_ARM]
                - performance[branch_index, docs, ..., 0]) for docs in halves],
        }
    live = all(
        min(reports[branch]["complete_route_rms_nat"]
            + reports[branch]["term_rms_nat"][0]
            + reports[branch]["term_rms_nat"][1]) > 0
        for branch in BRANCHES)
    closure = all(max(reports[branch]["mobius_closure_relative_squared"]) <= 1e-8
                  for branch in BRANCHES)
    return {
        "branch_reports": reports,
        "pred_b_carrier_profiles_stable": bool(pred_b),
        "all_routes_and_terms_live": live,
        "mobius_closure_holds": closure,
        "effects": effects,
        "complete_routes": benefits[..., FULL_ARM],
    }


def _rmse(prediction, target):
    return float((torch.as_tensor(prediction).double()
                  - torch.as_tensor(target).double()).square().mean().sqrt())


def _prediction_metrics(prediction, target, baseline):
    prediction = torch.as_tensor(prediction, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    baseline = torch.as_tensor(baseline, dtype=torch.float64)
    prediction_rmse = _rmse(prediction, target)
    baseline_rmse = _rmse(baseline, target)
    return {
        "prediction_rmse_nat": prediction_rmse,
        "baseline_rmse_nat": baseline_rmse,
        "rmse_improvement_over_baseline": 1.0 - prediction_rmse / max(baseline_rmse, 1e-30),
        "cosine": _cosine(prediction, target),
        "pearson": _pearson(prediction, target),
    }


def _fit_named_predictors(route, carrier, tokens, pair_ids, groups, fit_docs):
    pair_fit = pair_ids[fit_docs]
    token_fit = tokens[fit_docs, 1:]
    previous_fit = tokens[fit_docs, :-1]
    route_fit = route[fit_docs, 1:]
    carrier_fit = carrier[fit_docs, 1:]
    pair_route, counts, supported, index = _group_means(
        route_fit, pair_fit, groups)
    pair_carrier, _, _, _ = _group_means(carrier_fit, pair_fit, groups)
    current_groups = torch.unique(token_fit[supported]).sort().values
    previous_groups = torch.unique(previous_fit[supported]).sort().values
    current_route, _, _, _ = _group_means(
        route_fit[supported], token_fit[supported], current_groups)
    current_carrier, _, _, _ = _group_means(
        carrier_fit[supported], token_fit[supported], current_groups)
    previous_route, _, _, _ = _group_means(
        route_fit[supported], previous_fit[supported], previous_groups)
    previous_carrier, _, _, _ = _group_means(
        carrier_fit[supported], previous_fit[supported], previous_groups)
    return {
        "groups": groups,
        "pair_route": pair_route, "pair_carrier": pair_carrier,
        "current_groups": current_groups, "previous_groups": previous_groups,
        "current_route": current_route, "current_carrier": current_carrier,
        "previous_route": previous_route, "previous_carrier": previous_carrier,
        "global_route": float(route_fit[supported].mean()),
        "global_carrier": carrier_fit[supported].mean(dim=0),
        "counts": counts,
    }


def _evaluate_predictors(fitted, route, carrier, tokens, pair_ids, docs,
                         include_shifts=True):
    ids = pair_ids[docs]
    current = tokens[docs, 1:]
    previous = tokens[docs, :-1]
    route_target = route[docs, 1:]
    carrier_target = carrier[docs, 1:]
    supported, index = _membership(ids, fitted["groups"])
    pair_route = _lookup(fitted["pair_route"], supported, index).squeeze(-1)
    pair_carrier = _lookup(fitted["pair_carrier"], supported, index)
    current_supported, current_index = _membership(
        current[supported], fitted["current_groups"])
    previous_supported, previous_index = _membership(
        previous[supported], fitted["previous_groups"])
    if not bool(current_supported.all() and previous_supported.all()):
        raise RuntimeError("held-out pair contains a token absent from its fitted pair")
    current_route = fitted["current_route"][current_index].squeeze(-1)
    current_carrier = fitted["current_carrier"][current_index]
    previous_route = fitted["previous_route"][previous_index].squeeze(-1)
    previous_carrier = fitted["previous_carrier"][previous_index]
    route_selected = route_target[supported]
    carrier_selected = carrier_target[supported]
    output = {
        "positions": int(supported.sum()),
        "complete_route_pair_vs_current": _prediction_metrics(
            pair_route, route_selected, current_route),
        "complete_route_pair_vs_previous": _prediction_metrics(
            pair_route, route_selected, previous_route),
        "complete_route_pair_vs_global": _prediction_metrics(
            pair_route, route_selected,
            torch.full_like(route_selected, fitted["global_route"])),
        "carrier_pair_vs_current": _prediction_metrics(
            pair_carrier, carrier_selected, current_carrier),
        "carrier_pair_vs_previous": _prediction_metrics(
            pair_carrier, carrier_selected, previous_carrier),
    }
    if include_shifts:
        dense_prediction = torch.zeros_like(route_target)
        dense_prediction[supported] = pair_route
        shifts = parent.parent._shuffle_cosines(
            dense_prediction, route_target, supported)
        output["position_shift_cosines"] = shifts
        output["position_shift_q95"] = float(torch.quantile(
            torch.tensor(shifts, dtype=torch.float64), .95,
            interpolation="higher"))
    return output, {
        "supported": supported, "index": index,
        "pair_carrier": pair_carrier, "carrier_target": carrier_selected,
    }


def _mean_tensor_by_half(carrier, pair_ids, groups):
    output = []
    counts = []
    for docs in (slice(0, 250), slice(250, 500)):
        means, count, _, _ = _group_means(
            carrier[docs, 1:], pair_ids[docs], groups)
        output.append(means)
        counts.append(count)
    return output, counts


def analyze_named_context(carrier_analysis, tokens, pair_ids, groups):
    route = carrier_analysis["complete_routes"]
    effects = carrier_analysis["effects"][..., 1:]
    fit_docs, held_docs = slice(0, 250), slice(250, 500)
    reports = {}
    fitted = {}
    evaluation_aux = {}
    means_by_branch = {}
    counts_by_branch = {}
    for branch_index, branch in enumerate(BRANCHES):
        fitted[branch] = _fit_named_predictors(
            route[branch_index], effects[branch_index], tokens, pair_ids,
            groups, fit_docs)
        report, auxiliary = _evaluate_predictors(
            fitted[branch], route[branch_index], effects[branch_index],
            tokens, pair_ids, held_docs)
        means, counts = _mean_tensor_by_half(
            effects[branch_index], pair_ids, groups)
        weight = torch.minimum(*counts).sqrt()[:, None]
        report["pair_carrier_profile_cross_half_cosine_unweighted"] = _cosine(
            means[0], means[1])
        report["pair_carrier_profile_cross_half_cosine_weighted"] = _cosine(
            weight * means[0], weight * means[1])
        reports[branch] = report
        evaluation_aux[branch] = auxiliary
        means_by_branch[branch] = means
        counts_by_branch[branch] = counts

    t = reports["T"]
    route_report = t["complete_route_pair_vs_current"]
    carrier_report = t["carrier_pair_vs_current"]
    pred_c = bool(
        route_report["rmse_improvement_over_baseline"] >= .10
        and route_report["pearson"] >= .30
        and route_report["cosine"] >= t["position_shift_q95"] + .15
        and carrier_report["rmse_improvement_over_baseline"] >= .10
        and carrier_report["cosine"] >= .50)

    cross_cosines = []
    for half in range(2):
        weight = torch.minimum(
            counts_by_branch["T"][half], counts_by_branch["I"][half]).sqrt()[:, None]
        cross_cosines.append(_cosine(
            weight * means_by_branch["T"][half],
            weight * means_by_branch["I"][half]))
    weight0 = torch.minimum(
        counts_by_branch["T"][0], counts_by_branch["I"][0]).sqrt()[:, None]
    weight1 = torch.minimum(
        counts_by_branch["T"][1], counts_by_branch["I"][1]).sqrt()[:, None]
    t0 = (weight0 * means_by_branch["T"][0]).reshape(-1)
    i0 = (weight0 * means_by_branch["I"][0]).reshape(-1)
    alpha = float(torch.dot(t0, i0) / torch.dot(t0, t0).clamp_min(1e-30))
    shared_error = _relative_error(
        alpha * weight1 * means_by_branch["T"][1],
        weight1 * means_by_branch["I"][1])
    shared = min(cross_cosines) >= .90 and shared_error <= .35

    split_reports = []
    split = max(cross_cosines) <= .60
    for branch in ("T", "I"):
        other = "I" if branch == "T" else "T"
        own_aux = evaluation_aux[branch]
        other_means = fitted[other]["pair_carrier"]
        cross_prediction = _lookup(
            other_means, own_aux["supported"], own_aux["index"])
        own_rmse = _rmse(own_aux["pair_carrier"], own_aux["carrier_target"])
        cross_rmse = _rmse(cross_prediction, own_aux["carrier_target"])
        improvement = (cross_rmse - own_rmse) / max(cross_rmse, 1e-30)
        holds = improvement >= .10
        split &= holds
        split_reports.append({
            "target_branch": branch, "own_predictor_rmse_nat": own_rmse,
            "cross_branch_predictor_rmse_nat": cross_rmse,
            "relative_improvement": improvement, "holds": holds,
        })
    relation = "shared" if shared else "split" if split else None
    serial_fitted = {branch: {
        "pair_ids": value["groups"].tolist(),
        "pair_route_means_nat": value["pair_route"].squeeze(-1).tolist(),
        "pair_carrier_means_nat": value["pair_carrier"].tolist(),
        "current_token_ids": value["current_groups"].tolist(),
        "current_route_means_nat": value["current_route"].squeeze(-1).tolist(),
        "current_carrier_means_nat": value["current_carrier"].tolist(),
        "previous_token_ids": value["previous_groups"].tolist(),
        "previous_route_means_nat": value["previous_route"].squeeze(-1).tolist(),
        "previous_carrier_means_nat": value["previous_carrier"].tolist(),
        "global_route_mean_nat": value["global_route"],
        "global_carrier_mean_nat": value["global_carrier"].tolist(),
        "pair_counts": value["counts"].to(torch.int64).tolist(),
    } for branch, value in fitted.items()}
    return {
        "branch_prediction_reports": reports,
        "cross_branch_weighted_pair_carrier_cosines": cross_cosines,
        "shared_scale_fit_half0": alpha,
        "shared_scale_relative_error_half1": shared_error,
        "split_cross_use_reports": split_reports,
        "shared_holds": bool(shared), "split_holds": bool(split),
        "relation": relation,
        "pred_c_bigram_predicts_T_response": pred_c,
        "frozen_predictors": serial_fitted,
    }


def _restore_fitted(serial, device="cpu"):
    output = {}
    for branch, value in serial.items():
        output[branch] = {
            "groups": torch.tensor(value["pair_ids"], dtype=torch.long, device=device),
            "pair_route": torch.tensor(value["pair_route_means_nat"], dtype=torch.float64,
                                        device=device)[:, None],
            "pair_carrier": torch.tensor(value["pair_carrier_means_nat"], dtype=torch.float64,
                                          device=device),
            "current_groups": torch.tensor(value["current_token_ids"], dtype=torch.long,
                                             device=device),
            "current_route": torch.tensor(value["current_route_means_nat"], dtype=torch.float64,
                                           device=device)[:, None],
            "current_carrier": torch.tensor(value["current_carrier_means_nat"], dtype=torch.float64,
                                             device=device),
            "previous_groups": torch.tensor(value["previous_token_ids"], dtype=torch.long,
                                              device=device),
            "previous_route": torch.tensor(value["previous_route_means_nat"], dtype=torch.float64,
                                            device=device)[:, None],
            "previous_carrier": torch.tensor(value["previous_carrier_means_nat"], dtype=torch.float64,
                                              device=device),
            "global_route": value["global_route_mean_nat"],
            "global_carrier": torch.tensor(value["global_carrier_mean_nat"], dtype=torch.float64,
                                            device=device),
        }
    return output


def analyze_validation(carrier_analysis, tokens, pair_ids, validation_groups,
                       discovery_context, frozen_relation):
    route = carrier_analysis["complete_routes"]
    effects = carrier_analysis["effects"][..., 1:]
    fitted = _restore_fitted(discovery_context["frozen_predictors"])
    quarter_reports = {branch: [] for branch in BRANCHES}
    pred_e = carrier_analysis["pred_b_carrier_profiles_stable"]
    for branch_index, branch in enumerate(BRANCHES):
        # Restrict the unchanged discovery predictor to the pre-frozen validation subset.
        keep = torch.isin(fitted[branch]["groups"], validation_groups)
        local = {key: value for key, value in fitted[branch].items()}
        local["groups"] = fitted[branch]["groups"][keep]
        for key in ("pair_route", "pair_carrier"):
            local[key] = fitted[branch][key][keep]
        for quarter, docs in enumerate((slice(0, 250), slice(250, 500))):
            report, _ = _evaluate_predictors(
                local, route[branch_index], effects[branch_index],
                tokens, pair_ids, docs, include_shifts=False)
            primary = report["complete_route_pair_vs_current"]
            carrier = report["carrier_pair_vs_current"]
            holds = bool(
                primary["rmse_improvement_over_baseline"] >= .05
                and primary["pearson"] > 0
                and carrier["cosine"] >= .35)
            report.update({"quarter": quarter, "holds": holds})
            quarter_reports[branch].append(report)
            if branch == "T":
                pred_e &= holds

    # Recompute the descriptive relation on the two validation quarters, while
    # retaining discovery-frozen predictors and (for shared) its fitted scale.
    means = {}
    counts = {}
    for branch_index, branch in enumerate(("T", "I")):
        means[branch], counts[branch] = _mean_tensor_by_half(
            effects[BRANCHES.index(branch)], pair_ids, validation_groups)
    cosines = []
    for half in range(2):
        weight = torch.minimum(counts["T"][half], counts["I"][half]).sqrt()[:, None]
        cosines.append(_cosine(weight * means["T"][half], weight * means["I"][half]))
    if frozen_relation == "shared":
        alpha = discovery_context["shared_scale_fit_half0"]
        shared_errors = []
        for half in range(2):
            weight = torch.minimum(counts["T"][half], counts["I"][half]).sqrt()[:, None]
            shared_errors.append(_relative_error(
                alpha * weight * means["T"][half], weight * means["I"][half]))
        relation_holds = min(cosines) >= .90 and max(shared_errors) <= .35
        relation_details = {"cosines": cosines, "shared_errors": shared_errors}
    else:
        relation_holds = max(cosines) <= .60
        cross_use = []
        for target_branch in ("T", "I"):
            other_branch = "I" if target_branch == "T" else "T"
            target_index = BRANCHES.index(target_branch)
            own_fit = fitted[target_branch]
            other_fit = fitted[other_branch]
            for quarter, docs in enumerate((slice(0, 250), slice(250, 500))):
                ids = pair_ids[docs]
                supported, index = _membership(ids, validation_groups)
                validation_index = torch.searchsorted(
                    own_fit["groups"], ids[supported])
                other_index = torch.searchsorted(
                    other_fit["groups"], ids[supported])
                target = effects[target_index, docs, 1:][supported]
                own_prediction = own_fit["pair_carrier"][validation_index]
                cross_prediction = other_fit["pair_carrier"][other_index]
                own_rmse = _rmse(own_prediction, target)
                cross_rmse = _rmse(cross_prediction, target)
                improvement = (cross_rmse - own_rmse) / max(cross_rmse, 1e-30)
                holds = improvement >= .10
                relation_holds &= holds
                cross_use.append({
                    "target_branch": target_branch, "quarter": quarter,
                    "own_predictor_rmse_nat": own_rmse,
                    "cross_branch_predictor_rmse_nat": cross_rmse,
                    "relative_improvement": improvement, "holds": holds,
                })
        relation_details = {"cosines": cosines, "cross_use_reports": cross_use}
    pred_e &= relation_holds
    return {
        "quarter_reports": quarter_reports,
        "relation": frozen_relation if relation_holds else None,
        "relation_details": relation_details,
        "relation_holds": relation_holds,
        "pred_e_heldout_documents": bool(pred_e),
    }


def _instrument_valid(instrument, analysis):
    return bool(
        instrument["calls_exact"]
        and instrument["native_prefix_D_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_A_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_M_relative_squared_max"] <= 1e-12
        and instrument["absent_corner_logits_relative_squared_max"] <= 1e-12
        and instrument["native_corner_logits_relative_squared_max"] <= 1e-12
        and instrument["mlp0_state_max_abs"] == 0.0
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and analysis["mobius_closure_holds"]
        and analysis["all_routes_and_terms_live"])


def _serial_carrier_analysis(analysis):
    return {key: value for key, value in analysis.items()
            if key not in ("effects", "complete_routes")}


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert BRANCHES == ("T", "C", "I") and CARRIERS == ("D", "A", "M")
        assert len(POSITION_SHIFTS) == 16
        print(json.dumps({
            "status": "dry_run_passed", "rung": 486,
            "model_loaded": False, "outcomes_opened": False,
            "validation_outcomes_opened": False,
            "final_or_sealed_opened": False,
            "discovery_forwards": (500 // BATCH) * 28,
            "conditional_validation_forwards": (500 // BATCH) * 28,
            "branches": list(BRANCHES), "carriers": list(CARRIERS),
            "registered_predictions": ["pred_a", "pred_b", "pred_c",
                                       "pred_d_shared", "pred_d_split", "pred_e"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung486 output namespace already exists")
    (rows, positive, fit_rows, input_tokens, pair_ids, discovery_pairs,
     validation_pairs, metadata) = validate_inputs()
    del positive
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    discovery_ce, discovery_native_ce, discovery_instrument = collect_phase(
        model, rows, reference, *DISCOVERY_RANGE)
    discovery_carriers = analyze_carriers(discovery_ce)
    discovery_context = analyze_named_context(
        discovery_carriers, input_tokens[0:500], pair_ids[0:500], discovery_pairs)
    pred_a = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
                  and _instrument_valid(discovery_instrument, discovery_carriers))
    pred_b = discovery_carriers["pred_b_carrier_profiles_stable"]
    pred_c = discovery_context["pred_c_bigram_predicts_T_response"]
    relation = discovery_context["relation"]
    validation_licensed = bool(pred_a and pred_b and pred_c and relation is not None)
    validation_carriers = validation_context = validation_instrument = None
    pred_e = False
    if validation_licensed:
        validation_ce, validation_native_ce, validation_instrument = collect_phase(
            model, rows, reference, *VALIDATION_RANGE)
        validation_carriers = analyze_carriers(validation_ce)
        validation_context = analyze_validation(
            validation_carriers, input_tokens[500:1000], pair_ids[500:1000],
            validation_pairs, discovery_context, relation)
        pred_e = bool(_instrument_valid(validation_instrument, validation_carriers)
                      and validation_context["pred_e_heldout_documents"])
        del validation_ce, validation_native_ce
    strong_null = bool(not pred_a or not pred_b or not pred_c or relation is None)
    result = {
        "status": "complete", "rung": 486,
        "claim_level": "exact_coupled_carrier_and_named_context_identification_screen",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(BRANCHES), "carriers": list(CARRIERS),
        "arms": {str(mask): _arm_name(mask) for mask in range(8)},
        "discovery": {
            "documents": list(DISCOVERY_RANGE), "split": SPLIT,
            "instrument": discovery_instrument,
            "carrier_analysis": _serial_carrier_analysis(discovery_carriers),
            "context_analysis": discovery_context,
            "native_ce_mean": float(discovery_native_ce.mean()),
        },
        "validation": None if validation_carriers is None else {
            "documents": list(VALIDATION_RANGE), "split": 750,
            "instrument": validation_instrument,
            "carrier_analysis": _serial_carrier_analysis(validation_carriers),
            "context_analysis": validation_context,
        },
        "selected_relation": relation,
        'pred_a_exact_lawful_instrument': pred_a,
        'pred_b_stable_complete_carrier_decomposition': pred_b,
        'pred_c_bigram_predicts_T_response': pred_c,
        'pred_d_exactly_one_T_I_relation': relation is not None,
        "d_shared_holds": discovery_context["shared_holds"],
        "d_split_holds": discovery_context["split_holds"],
        'pred_e_heldout_documents': pred_e,
        "validation_licensed_and_opened": validation_licensed,
        "strong_null": strong_null,
        "final_or_sealed_opened": False,
        "execution_price": {
            "discovery_full_model_forwards": sum(
                discovery_instrument["calls"][key] for key in
                ("native_forwards", "absent_forwards", "cube_forwards")),
            "validation_full_model_forwards": 0 if validation_instrument is None else sum(
                validation_instrument["calls"][key] for key in
                ("native_forwards", "absent_forwards", "cube_forwards")),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "exact_cross_branch_carrier_interchange"
            if pred_e and relation == "shared" else
            "branch_specific_context_reader_extraction"
            if pred_e and relation == "split" else
            "continuous_live_attention0_state_finite_reader"
            if not pred_c else
            "retain_T_I_separate_context_conditioned_readers"
            if relation is None else
            "heldout_context_law_failed_no_claim"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    t = discovery_context["branch_prediction_reports"]["T"]
    print(json.dumps({
        "status": "complete", "rung": 486,
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "T_pair_route": t["complete_route_pair_vs_current"],
        "T_pair_carrier": t["carrier_pair_vs_current"],
        "selected_relation": relation, "strong_null": strong_null,
        "validation_opened": validation_licensed,
        "next_step": result["next_step"], "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
