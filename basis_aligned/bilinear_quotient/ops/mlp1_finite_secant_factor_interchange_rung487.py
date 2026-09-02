#!/usr/bin/env python3
"""RUNG487 -- exact MLP1 finite-secant factor interchange across T/C/I."""

# BQGATE: EXPERIMENT
# pred_a exact polarization, replay, calls, and physical own-secant identity
# pred_b own finite responses transfer across document halves
# pred_c at least one context-factor or direction-factor interchange edge
# pred_d the factor-sharing graph is stable across discovery halves
# pred_e the frozen graph validates on held-out documents

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
import mlp0_coupled_block1_bigram_response_rung486 as parent
import mlp0_attention1_finite_path_factorial_rung484 as factorial_parent
import mlp0_immediate_consumer_quotient_rung483 as branch_parent
import mlp0_centered_context_anova_factorial as component_parent


PREREG = POLY / "MLP1_FINITE_SECANT_FACTOR_INTERCHANGE_RUNG487_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp0_coupled_block1_bigram_response_rung486.py"
PARENT_RESULT = ROOT / "mlp0_coupled_block1_bigram_response_rung486_results.json"
OUT = ROOT / "mlp1_finite_secant_factor_interchange_rung487_results.json"
HASHES = {
    PREREG: "59a3ef38ee640dd31df43668e1bbcb0eeb9d3bfd31f0cdcf393a142a1dbf142f",
    PARENT_SOURCE: "4cf42487272688bfb03430e5aa5a27b78421df5b3138a6b71cd4c9a6061a607f",
    PARENT_RESULT: "f36ed7bed41d5908fd5f1da977a6ec72c21414e91829dfdda6fe39cfaf3ec941",
    ROOT / "ops/mlp0_attention1_finite_path_factorial_rung484.py":
        "42f66fba01361c976660554197fef7aa66cb20d80eb5b6351b01a1f6e3bf9d54",
    ROOT / "ops/mlp0_immediate_consumer_quotient_rung483.py":
        "9763502b99b8693826a5985c8f25a3ebe7763c3cd176c3aebeeb140833a61f4c",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
}
BRANCHES = ("T", "C", "I")
UNORDERED_PAIRS = (("T", "C"), ("T", "I"), ("C", "I"))
ORDERED_PAIRS = tuple((target, donor) for pair in UNORDERED_PAIRS
                      for target, donor in (pair, pair[::-1]))
MODES = ("own", "context", "direction", "both")
POSITION_SHIFTS = parent.POSITION_SHIFTS
DISCOVERY_RANGE = (0, 500)
VALIDATION_RANGE = (500, 1000)
SPLIT = 250
BATCH = 4
TOKENS = 256


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_name(pair):
    return f"{pair[0]}<-{pair[1]}"


def _cosine(left, right):
    return parent._cosine(left, right)


def _effect_report(predictor, target):
    return factorial_parent._effect_report(predictor, target)


def _relative_squared(left, right):
    return factorial_parent._relative_squared(left, right)


def _linear(value, weight):
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def _mlp_write(mlp, state):
    left = _linear(state, mlp.Left.weight)
    right = _linear(state, mlp.Right.weight)
    write = _linear(left * right, mlp.Down.weight)
    return write + mlp.Down_bias.to(device=write.device, dtype=write.dtype)


def _secant(mlp, delta, midpoint):
    left_delta = _linear(delta, mlp.Left.weight)
    right_delta = _linear(delta, mlp.Right.weight)
    left_midpoint = _linear(midpoint, mlp.Left.weight)
    right_midpoint = _linear(midpoint, mlp.Right.weight)
    return _linear(
        left_delta * right_midpoint + left_midpoint * right_delta,
        mlp.Down.weight)


def _secants_for_pair(mlp, states, target, donor):
    native = states["native"].float()
    target_absent = states[target].float()
    donor_absent = states[donor].float()
    delta_target = native - target_absent
    delta_donor = native - donor_absent
    midpoint_target = (native + target_absent) / 2
    midpoint_donor = (native + donor_absent) / 2
    return {
        "own": _secant(mlp, delta_target, midpoint_target),
        "context": _secant(mlp, delta_target, midpoint_donor),
        "direction": _secant(mlp, delta_donor, midpoint_target),
        "both": _secant(mlp, delta_donor, midpoint_donor),
    }, (delta_target, delta_donor, midpoint_target, midpoint_donor)


def validate_inputs():
    for path, expected in HASHES.items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(PARENT_RESULT.read_text())
    if receipt.get("rung") != 486 \
            or receipt.get("pred_a_exact_lawful_instrument") is not True \
            or receipt.get("pred_b_stable_complete_carrier_decomposition") is not True \
            or receipt.get("pred_c_bigram_predicts_T_response") is not False \
            or receipt.get("pred_d_exactly_one_T_I_relation") is not False \
            or receipt.get("pred_e_heldout_documents") is not False \
            or receipt.get("validation_licensed_and_opened") is not False \
            or receipt.get("strong_null") is not True \
            or receipt.get("next_step") != "continuous_live_attention0_state_finite_reader":
        raise RuntimeError("rung486 did not license rung487")
    rows, positive, fit_rows, metadata = factorial_parent.validate_inputs()
    return rows, positive, fit_rows, metadata


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
            capture["z"] = event.state.detach().clone()
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    prefix = branch_parent._native_prefix(model, tokens, reference)
    capture["branches"] = {name: prefix["branches"][name].detach().clone()
                           for name in BRANCHES}
    capture["identity"] = {key: prefix[key] for key in (
        "analytical_num", "analytical_den", "deployed_num", "deployed_den")}
    capture["prefix_errors"] = {
        "D": _relative_squared(prefix["m0"], capture["D"]),
        "A": _relative_squared(prefix["a1"], capture["A"]),
        "M": _relative_squared(prefix["m1"], capture["M"]),
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
            capture["z"] = event.state.detach().clone()
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    return logits, capture, calls


@torch.no_grad()
def _physical_forward(model, tokens, absent, secant):
    calls = {"attention": 0, "mlp": 0, "D": 0, "A": 0, "M": 0}

    def attention(event):
        calls["attention"] += 1
        if event.site != 1:
            return event.block.attn(event.state, event.first_value)
        calls["A"] += 1
        return absent["A"], event.first_value

    def mlp(event):
        calls["mlp"] += 1
        if event.site == 0:
            calls["D"] += 1
            return absent["D"]
        if event.site == 1:
            calls["M"] += 1
            return (absent["M"].float() + secant.float()).to(absent["M"].dtype)
        return event.block.mlp(event.state)

    return facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True), calls


def _accumulate_cosine(stats, half, pair_index, mode_index, control_index,
                       prediction, target):
    prediction = prediction.double()
    target = target.double()
    stats[half, pair_index, mode_index, control_index, 0] += float(
        (prediction * target).sum())
    stats[half, pair_index, mode_index, control_index, 1] += float(
        prediction.square().sum())
    stats[half, pair_index, mode_index, control_index, 2] += float(
        target.square().sum())


def _cosines_from_stats(stats):
    denominator = (stats[..., 1] * stats[..., 2]).sqrt().clamp_min(1e-30)
    return stats[..., 0] / denominator


def collect_phase(model, rows, reference, start_doc, stop_doc):
    arm_batches = []
    absent_batches = []
    native_batches = []
    write_stats = torch.zeros(2, len(ORDERED_PAIRS), 2,
                              1 + len(POSITION_SHIFTS), 3, dtype=torch.float64)
    calls = {
        "native_forwards": 0, "absent_forwards": 0, "physical_forwards": 0,
        "native_attention": 0, "native_mlp": 0,
        "absent_attention": 0, "absent_mlp": 0, "site0_removal": 0,
        "physical_attention": 0, "physical_mlp": 0,
        "D_injections": 0, "A_injections": 0, "M_injections": 0,
    }
    errors = {
        "native_prefix_D_relative_squared_max": 0.0,
        "native_prefix_A_relative_squared_max": 0.0,
        "native_prefix_M_relative_squared_max": 0.0,
        "mlp0_state_max_abs": 0.0,
        "polarization_float32_relative_squared_max": 0.0,
        "polarization_bf16_relative_squared_max": 0.0,
        "own_native_write_relative_squared_max": 0.0,
        "analytical_num": 0.0, "analytical_den": 0.0,
        "deployed_num": 0.0, "deployed_den": 0.0,
    }
    device = next(model.parameters()).device
    mlp1 = model.transformer.h[1].mlp
    for start in range(start_doc, stop_doc, BATCH):
        stop = min(start + BATCH, stop_doc)
        tokens = rows[start:stop, :-1].to(device)
        targets = rows[start:stop, 1:].to(device)
        native_logits, native, native_calls = _native_forward(
            model, tokens, reference)
        calls["native_forwards"] += 1
        calls["native_attention"] += native_calls["attention"]
        calls["native_mlp"] += native_calls["mlp"]
        native_batches.append(factorial_parent._per_token_ce(
            native_logits, targets))
        for name, error in native["prefix_errors"].items():
            key = f"native_prefix_{name}_relative_squared_max"
            errors[key] = max(errors[key], error)
        for key in ("analytical_num", "analytical_den", "deployed_num", "deployed_den"):
            errors[key] += native["identity"][key]

        absent = {}
        absent_ce = {}
        for branch in BRANCHES:
            logits, capture, audit = _absent_forward(
                model, tokens, native, native["branches"][branch])
            absent[branch] = capture
            absent_ce[branch] = factorial_parent._per_token_ce(logits, targets)
            calls["absent_forwards"] += 1
            calls["absent_attention"] += audit["attention"]
            calls["absent_mlp"] += audit["mlp"]
            calls["site0_removal"] += audit["site0_removal"]
            errors["mlp0_state_max_abs"] = max(
                errors["mlp0_state_max_abs"], capture["mlp0_state_error"])
            delta32 = native["z"].float() - capture["z"].float()
            midpoint32 = (native["z"].float() + capture["z"].float()) / 2
            secant32 = _secant(mlp1, delta32, midpoint32)
            direct32 = _mlp_write(mlp1, native["z"].float()) \
                - _mlp_write(mlp1, capture["z"].float())
            errors["polarization_float32_relative_squared_max"] = max(
                errors["polarization_float32_relative_squared_max"],
                _relative_squared(secant32, direct32))
            secant = secant32.to(native["M"].dtype)
            deployed = native["M"] - capture["M"]
            errors["polarization_bf16_relative_squared_max"] = max(
                errors["polarization_bf16_relative_squared_max"],
                _relative_squared(secant, deployed))
            own_write = (capture["M"].float() + secant.float()).to(native["M"].dtype)
            errors["own_native_write_relative_squared_max"] = max(
                errors["own_native_write_relative_squared_max"],
                _relative_squared(own_write, native["M"]))
        absent_batches.append(torch.stack([absent_ce[name] for name in BRANCHES]))

        state_map = {"native": native["z"],
                     **{name: absent[name]["z"] for name in BRANCHES}}
        pair_arms = []
        for pair_index, (target, donor) in enumerate(ORDERED_PAIRS):
            secants, factors = _secants_for_pair(
                mlp1, state_map, target, donor)
            delta_target, delta_donor, midpoint_target, midpoint_donor = factors
            arms = []
            for mode in MODES:
                logits, audit = _physical_forward(
                    model, tokens, absent[target], secants[mode])
                calls["physical_forwards"] += 1
                calls["physical_attention"] += audit["attention"]
                calls["physical_mlp"] += audit["mlp"]
                for name in ("D", "A", "M"):
                    calls[f"{name}_injections"] += audit[name]
                arms.append(factorial_parent._per_token_ce(logits, targets))
            pair_arms.append(torch.stack(arms, dim=-1))

            # Same-position and factor-shifted MLP1-write controls. Split the
            # one batch crossing the reporting boundary before aggregation.
            own = secants["own"]
            for local_start, local_stop, half in (
                    (0, max(0, min(stop, start_doc + SPLIT) - start), 0),
                    (max(0, start_doc + SPLIT - start), stop - start, 1)):
                if local_stop <= local_start:
                    continue
                sl = slice(local_start, local_stop)
                for mode_index, mode in enumerate(("context", "direction")):
                    _accumulate_cosine(
                        write_stats, half, pair_index, mode_index, 0,
                        secants[mode][sl], own[sl])
                    for control_index, shift in enumerate(POSITION_SHIFTS, start=1):
                        if mode == "context":
                            shifted = _secant(
                                mlp1, delta_target[sl],
                                torch.roll(midpoint_donor[sl], shift, dims=1))
                        else:
                            shifted = _secant(
                                mlp1, torch.roll(delta_donor[sl], shift, dims=1),
                                midpoint_target[sl])
                        _accumulate_cosine(
                            write_stats, half, pair_index, mode_index,
                            control_index, shifted, own[sl])
        arm_batches.append(torch.stack(pair_arms))

    batches = math.ceil((stop_doc - start_doc) / BATCH)
    expected = {
        "native_forwards": batches,
        "absent_forwards": 3 * batches,
        "physical_forwards": 4 * len(ORDERED_PAIRS) * batches,
        "native_attention": 18 * batches, "native_mlp": 18 * batches,
        "absent_attention": 18 * 3 * batches,
        "absent_mlp": 18 * 3 * batches, "site0_removal": 3 * batches,
        "physical_attention": 18 * 4 * len(ORDERED_PAIRS) * batches,
        "physical_mlp": 18 * 4 * len(ORDERED_PAIRS) * batches,
        "D_injections": 4 * len(ORDERED_PAIRS) * batches,
        "A_injections": 4 * len(ORDERED_PAIRS) * batches,
        "M_injections": 4 * len(ORDERED_PAIRS) * batches,
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
    }
    return {
        "arms": torch.cat(arm_batches, dim=1),
        "absent": torch.cat(absent_batches, dim=1),
        "native": torch.cat(native_batches, dim=0),
        "write_cosines": _cosines_from_stats(write_stats),
        "instrument": instrument,
    }


def _scaled_error(predictor, target):
    predictor = torch.as_tensor(predictor, dtype=torch.float64).reshape(-1)
    target = torch.as_tensor(target, dtype=torch.float64).reshape(-1)
    alpha = float(torch.dot(predictor, target)
                  / torch.dot(predictor, predictor).clamp_min(1e-30))
    return alpha, float(torch.linalg.vector_norm(target - alpha * predictor)
                        / torch.linalg.vector_norm(target).clamp_min(1e-30))


def analyze_phase(collected, positive, frozen_edges=None):
    arms = collected["arms"].double()
    absent = collected["absent"].double()
    benefits = torch.empty_like(arms)
    for pair_index, (target, _) in enumerate(ORDERED_PAIRS):
        benefits[pair_index] = absent[BRANCHES.index(target), ..., None] \
            - arms[pair_index]
    halves = (slice(0, SPLIT), slice(SPLIT, 2 * SPLIT))
    pair_reports = {}
    descriptive_edges = []
    for unordered in UNORDERED_PAIRS:
        ordered_indices = [ORDERED_PAIRS.index(unordered),
                           ORDERED_PAIRS.index(unordered[::-1])]
        report = {"directions": {}}
        context_holds = True
        direction_holds = True
        for pair_index in ordered_indices:
            pair = ORDERED_PAIRS[pair_index]
            direction_report = {"halves": []}
            for half, docs in enumerate(halves):
                own = benefits[pair_index, docs, ..., MODES.index("own")]
                context = benefits[pair_index, docs, ..., MODES.index("context")]
                direction = benefits[pair_index, docs, ..., MODES.index("direction")]
                context_effect = _effect_report(context, own)
                direction_effect = _effect_report(direction, own)
                write_context = float(collected["write_cosines"][
                    half, pair_index, 0, 0])
                write_direction = float(collected["write_cosines"][
                    half, pair_index, 1, 0])
                context_controls = collected["write_cosines"][
                    half, pair_index, 0, 1:].tolist()
                direction_controls = collected["write_cosines"][
                    half, pair_index, 1, 1:].tolist()
                context_q95 = float(torch.quantile(
                    torch.tensor(context_controls), .95, interpolation="higher"))
                direction_q95 = float(torch.quantile(
                    torch.tensor(direction_controls), .95, interpolation="higher"))
                context_cell = bool(
                    context_effect["cosine"] >= .80
                    and context_effect["best_scalar_adjusted_relative_error"] <= .50
                    and context_effect["cosine"] >= direction_effect["cosine"] + .15
                    and write_context >= context_q95 + .15)
                direction_cell = bool(
                    direction_effect["cosine"] >= .80
                    and direction_effect["best_scalar_adjusted_relative_error"] <= .50
                    and direction_effect["cosine"] >= context_effect["cosine"] + .15
                    and write_direction >= direction_q95 + .15)
                context_holds &= context_cell
                direction_holds &= direction_cell
                equality = positive[docs]
                direction_report["halves"].append({
                    "half": half,
                    "own_rms_nat": float(own.square().mean().sqrt()),
                    "context_effect": context_effect,
                    "direction_effect": direction_effect,
                    "both_effect": _effect_report(
                        benefits[pair_index, docs, ..., MODES.index("both")], own),
                    "context_equality_effect": _effect_report(
                        context[equality], own[equality]),
                    "direction_equality_effect": _effect_report(
                        direction[equality], own[equality]),
                    "context_write_cosine": write_context,
                    "context_shift_cosines": context_controls,
                    "context_shift_q95": context_q95,
                    "direction_write_cosine": write_direction,
                    "direction_shift_cosines": direction_controls,
                    "direction_shift_q95": direction_q95,
                    "context_cell_holds": context_cell,
                    "direction_cell_holds": direction_cell,
                })
            report["directions"][_pair_name(pair)] = direction_report
        edge_type = "context" if context_holds else "direction" if direction_holds else None
        if edge_type is not None:
            descriptive_edges.append({"pair": list(unordered), "type": edge_type})
        report.update({
            "context_edge_holds": bool(context_holds),
            "direction_edge_holds": bool(direction_holds),
            "edge_type": edge_type,
        })
        pair_reports["-".join(unordered)] = report

    own_reports = {}
    pred_b = True
    for branch in BRANCHES:
        pair_index = next(index for index, pair in enumerate(ORDERED_PAIRS)
                          if pair[0] == branch)
        own0 = benefits[pair_index, halves[0], ..., MODES.index("own")]
        own1 = benefits[pair_index, halves[1], ..., MODES.index("own")]
        rms = [float(own0.square().mean().sqrt()),
               float(own1.square().mean().sqrt())]
        mean_abs = [float(own0.abs().mean()), float(own1.abs().mean())]
        rms_ratio = rms[1] / max(rms[0], 1e-30)
        mean_abs_ratio = mean_abs[1] / max(mean_abs[0], 1e-30)
        holds = bool(min(rms) >= .10 and .80 <= rms_ratio <= 1.25
                     and .80 <= mean_abs_ratio <= 1.25)
        pred_b &= holds
        own_reports[branch] = {
            "rms_nat": rms, "half1_over_half0_rms": rms_ratio,
            "mean_absolute_effect_nat": mean_abs,
            "half1_over_half0_mean_absolute_effect": mean_abs_ratio,
            "signed_mean_effect_nat": [float(own0.mean()), float(own1.mean())],
            "equality_positive_mean_effect_nat": [
                float(own0[positive[halves[0]]].mean()),
                float(own1[positive[halves[1]]].mean())],
            "holds": holds,
        }
    edges = descriptive_edges if frozen_edges is None else frozen_edges
    edge_keys = {(tuple(edge["pair"]), edge["type"]) for edge in descriptive_edges}
    graph_holds = bool(edges) and all(
        (tuple(edge["pair"]), edge["type"]) in edge_keys for edge in edges)
    all_live = bool((benefits.square().mean(dim=(1, 2)) > 0).all())
    return {
        "own_response_reports": own_reports,
        "pair_reports": pair_reports,
        "descriptive_edges": descriptive_edges,
        "frozen_edges": edges,
        "pred_b_own_responses_stable": bool(pred_b),
        "pred_c_at_least_one_factor_edge": bool(descriptive_edges),
        "pred_d_factor_graph_stable": graph_holds,
        "all_physical_effects_live": all_live,
    }


def _instrument_valid(instrument, analysis):
    return bool(
        instrument["calls_exact"]
        and instrument["native_prefix_D_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_A_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_M_relative_squared_max"] <= 1e-12
        and instrument["mlp0_state_max_abs"] == 0.0
        and instrument["polarization_float32_relative_squared_max"] <= 1e-8
        and instrument["polarization_bf16_relative_squared_max"] <= 1e-5
        and instrument["own_native_write_relative_squared_max"] <= 1e-5
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and analysis["all_physical_effects_live"])


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(UNORDERED_PAIRS) == 3 and len(ORDERED_PAIRS) == 6
        assert len(MODES) == 4 and len(POSITION_SHIFTS) == 16
        print(json.dumps({
            "status": "dry_run_passed", "rung": 487,
            "model_loaded": False, "outcomes_opened": False,
            "validation_outcomes_opened": False,
            "final_or_sealed_opened": False,
            "discovery_forwards": (500 // BATCH) * 28,
            "conditional_validation_forwards": (500 // BATCH) * 28,
            "branches": list(BRANCHES),
            "ordered_pairs": [list(pair) for pair in ORDERED_PAIRS],
            "modes": list(MODES),
            "registered_predictions": ["pred_a", "pred_b", "pred_c_context",
                                       "pred_c_direction", "pred_d", "pred_e"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung487 output namespace already exists")
    rows, positive, fit_rows, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    discovery = collect_phase(model, rows, reference, *DISCOVERY_RANGE)
    discovery_analysis = analyze_phase(discovery, positive[0:500])
    pred_a = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
                  and _instrument_valid(discovery["instrument"], discovery_analysis))
    pred_b = discovery_analysis["pred_b_own_responses_stable"]
    pred_c = discovery_analysis["pred_c_at_least_one_factor_edge"]
    pred_d = discovery_analysis["pred_d_factor_graph_stable"]
    validation_licensed = bool(pred_a and pred_b and pred_d)
    validation = validation_analysis = None
    pred_e = False
    if validation_licensed:
        validation = collect_phase(model, rows, reference, *VALIDATION_RANGE)
        validation_analysis = analyze_phase(
            validation, positive[500:1000],
            frozen_edges=discovery_analysis["frozen_edges"])
        pred_e = bool(_instrument_valid(validation["instrument"], validation_analysis)
                      and validation_analysis["pred_b_own_responses_stable"]
                      and validation_analysis["pred_d_factor_graph_stable"])
    strong_null = bool(not pred_a or not pred_b or not pred_d)
    result = {
        "status": "complete", "rung": 487,
        "claim_level": "exact_finite_secant_factor_interchange_screen",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(BRANCHES),
        "ordered_pairs": [list(pair) for pair in ORDERED_PAIRS],
        "modes": list(MODES),
        "position_shift_offsets": list(POSITION_SHIFTS),
        "discovery": {
            "documents": list(DISCOVERY_RANGE), "split": SPLIT,
            "instrument": discovery["instrument"],
            "analysis": discovery_analysis,
            "native_ce_mean": float(discovery["native"].mean()),
        },
        "validation": None if validation is None else {
            "documents": list(VALIDATION_RANGE), "split": 750,
            "instrument": validation["instrument"],
            "analysis": validation_analysis,
            "native_ce_mean": float(validation["native"].mean()),
        },
        "selected_edges": discovery_analysis["frozen_edges"],
        'pred_a_exact_lawful_instrument': pred_a,
        'pred_b_stable_own_finite_responses': pred_b,
        'pred_c_at_least_one_factor_interchange_edge': pred_c,
        'pred_d_stable_factor_sharing_graph': pred_d,
        'pred_e_heldout_documents': pred_e,
        "validation_licensed_and_opened": validation_licensed,
        "strong_null": strong_null,
        "final_or_sealed_opened": False,
        "execution_price": {
            "discovery_full_model_forwards": sum(
                discovery["instrument"]["calls"][key] for key in
                ("native_forwards", "absent_forwards", "physical_forwards")),
            "validation_full_model_forwards": 0 if validation is None else sum(
                validation["instrument"]["calls"][key] for key in
                ("native_forwards", "absent_forwards", "physical_forwards")),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "cross_document_finite_secant_factor_interchange"
            if pred_e else
            "within_branch_integrated_secant_response_reader"
            if not pred_d else
            "heldout_factor_graph_failed_no_claim"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 487,
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "selected_edges": result["selected_edges"],
        "strong_null": strong_null,
        "validation_opened": validation_licensed,
        "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
