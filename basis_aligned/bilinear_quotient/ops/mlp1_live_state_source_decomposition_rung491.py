#!/usr/bin/env python3
"""RUNG491 -- exact named-source decomposition of MLP1's native-state reader."""

# BQGATE: EXPERIMENT
# pred_a exact named-state and bilinear-source decomposition
# pred_b full native-state response remains valid for T and I
# pred_c at least one named source jointly modulates T and I
# pred_d the shared necessary-source set is stable and non-numerical
# pred_e the frozen shared source set validates on held-out intervention outcomes

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
import mlp1_finite_secant_factor_interchange_rung487 as base
import mlp1_branch_resolved_response_rung490 as parent


PREREG = POLY / "MLP1_LIVE_STATE_SOURCE_DECOMPOSITION_RUNG491_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp1_branch_resolved_response_rung490.py"
PARENT_RESULT = ROOT / "mlp1_branch_resolved_response_rung490_results.json"
OUT = ROOT / "mlp1_live_state_source_decomposition_rung491_results.json"
HASHES = {
    PREREG: "b847482cc98f867f696a5497e656fa9311ba812931cd84e41401867c17e0229e",
    PARENT_SOURCE: "e92cae03e1b88f6be34f6e77b8c77e63d063c53683379ebf38cfb7f4cba2d436",
    PARENT_RESULT: "cd4052b06d8b90ebfbcfa45721b701b526c7d2e8bbfa9f383e9b14cbbc779e8b",
}
BRANCHES = ("T", "C", "I")
SOURCES = ("E", "A0", "M0_OTHER", "M0_T", "M0_C", "M0_I", "M0_S", "A1",
           "NUMERICAL")
SEMANTIC_SOURCES = SOURCES[:-1]
MODES = ("own", "full") \
    + tuple(f"single_{source}" for source in SOURCES) \
    + tuple(f"leave_{source}" for source in SOURCES)
DISCOVERY_RANGE = (0, 500)
VALIDATION_RANGE = (500, 1000)
SPLIT = 250
BATCH = 4
D = 1152
POSITION_SHIFTS = base.POSITION_SHIFTS
U = 2.0 ** -8
POLARIZATION_BF16_BAR = 8 * U * U
OWN_WRITE_BF16_BAR = 4 * U * U


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(PARENT_RESULT.read_text())
    if receipt.get("rung") != 490 \
            or receipt.get("pred_a_exact_lawful_instrument") is not True \
            or receipt.get("pred_b_T_I_native_pass_C_fails") is not True \
            or receipt.get("pred_c_material_T_I_versus_C_contrast") is not True \
            or receipt.get("pred_d_finite_correction_order_T_I_C") is not True \
            or receipt.get("strong_null") is not False \
            or receipt.get("next_step") \
            != "branchwise_integrated_response_shared_T_I_native_term_separate_corrections":
        raise RuntimeError("rung490 did not license named live-state sources")
    return parent.validate_inputs()


def _native_with_sources(model, tokens, reference):
    capture = {}
    calls = {"attention": 0, "mlp": 0}

    def attention(event):
        calls["attention"] += 1
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 0:
            capture["a0"] = write.detach().clone()
        elif event.site == 1:
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
    prefix = base.branch_parent._native_prefix(model, tokens, reference)
    capture["branches"] = {
        name: prefix["branches"][name].detach().clone() for name in BRANCHES
    }
    capture["identity"] = {key: prefix[key] for key in (
        "analytical_num", "analytical_den", "deployed_num", "deployed_den")}
    capture["prefix_errors"] = {
        "D": base._relative_squared(prefix["m0"], capture["D"]),
        "A": base._relative_squared(prefix["a1"], capture["A"]),
        "M": base._relative_squared(prefix["m1"], capture["M"]),
    }

    block0, block1 = model.transformer.h[0], model.transformer.h[1]
    x0 = prefix["x0"].float()
    before_a0 = (block0.lambdas[0] + block0.lambdas[1]).float() * x0
    a0 = prefix["before_m0"].float() - before_a0
    all_branches = {
        name: prefix["branches"][name].float()
        for name in base.branch_parent.BRANCHES
    }
    branch_sum = sum(all_branches.values(), start=torch.zeros_like(prefix["m0"].float()))
    other = prefix["m0"].float() - branch_sum
    lambda10 = block1.lambdas[0].float()
    lambda11 = block1.lambdas[1].float()
    raw_sources = {
        "E": (lambda10 * (block0.lambdas[0] + block0.lambdas[1]).float()
              + lambda11) * x0,
        "A0": lambda10 * a0,
        "M0_OTHER": lambda10 * other,
        **{f"M0_{name}": lambda10 * value for name, value in all_branches.items()},
        "A1": prefix["a1"].float(),
    }
    after_m0 = prefix["before_m0"] + prefix["m0"]
    before_a1 = block1.lambdas[0] * after_m0 + block1.lambdas[1] * prefix["x0"]
    unnormalized = (before_a1 + prefix["a1"]).float()
    z = capture["z"].float()
    gain = (z * unnormalized).sum(-1, keepdim=True) \
        / unnormalized.square().sum(-1, keepdim=True).clamp_min(1e-30)
    state_sources = {name: gain * raw_sources[name] for name in SEMANTIC_SOURCES}
    state_sources["NUMERICAL"] = z - sum(
        state_sources.values(), start=torch.zeros_like(z))
    capture["state_sources"] = state_sources
    capture["state_source_relative_squared"] = base._relative_squared(
        sum(state_sources.values(), start=torch.zeros_like(z)), z)
    capture["numerical_state_energy_share"] = float(
        state_sources["NUMERICAL"].double().square().sum()
        / z.double().square().sum().clamp_min(1e-30))
    capture["prefix_z_relative_squared"] = base._relative_squared(
        F.rms_norm((before_a1 + prefix["a1"]), (D,)), capture["z"])
    return logits, capture, calls


def _accumulate(stats, half, target_index, source_index, control_index,
                prediction, target):
    prediction = prediction.double()
    target = target.double()
    stats[half, target_index, source_index, control_index, 0] += float(
        (prediction * target).sum())
    stats[half, target_index, source_index, control_index, 1] += float(
        prediction.square().sum())
    stats[half, target_index, source_index, control_index, 2] += float(
        target.square().sum())


def _cosines(stats):
    return stats[..., 0] / (stats[..., 1] * stats[..., 2]).sqrt().clamp_min(1e-30)


def collect_phase(model, rows, reference, start_doc, stop_doc):
    arm_batches = []
    absent_batches = []
    native_batches = []
    source_write_stats = torch.zeros(
        2, len(BRANCHES), len(SOURCES), 1 + len(POSITION_SHIFTS), 3,
        dtype=torch.float64)
    # Keep the same target/source/control/stat layout as source_write_stats.
    # FULL has one synthetic source, hence the singleton third axis.
    full_write_stats = torch.zeros(
        2, len(BRANCHES), 1, 1 + len(POSITION_SHIFTS), 3,
        dtype=torch.float64)
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
        "prefix_z_relative_squared_max": 0.0,
        "mlp0_state_max_abs": 0.0,
        "state_source_relative_squared_max": 0.0,
        "numerical_state_energy_share_max": 0.0,
        "bilinear_source_sum_relative_squared_max": 0.0,
        "full_plus_curvature_bf16_relative_squared_max": 0.0,
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
        native_logits, native, native_calls = _native_with_sources(
            model, tokens, reference)
        calls["native_forwards"] += 1
        calls["native_attention"] += native_calls["attention"]
        calls["native_mlp"] += native_calls["mlp"]
        native_batches.append(base.factorial_parent._per_token_ce(native_logits, targets))
        for name, error in native["prefix_errors"].items():
            errors[f"native_prefix_{name}_relative_squared_max"] = max(
                errors[f"native_prefix_{name}_relative_squared_max"], error)
        errors["prefix_z_relative_squared_max"] = max(
            errors["prefix_z_relative_squared_max"], native["prefix_z_relative_squared"])
        errors["state_source_relative_squared_max"] = max(
            errors["state_source_relative_squared_max"],
            native["state_source_relative_squared"])
        errors["numerical_state_energy_share_max"] = max(
            errors["numerical_state_energy_share_max"],
            native["numerical_state_energy_share"])
        for key in ("analytical_num", "analytical_den", "deployed_num", "deployed_den"):
            errors[key] += native["identity"][key]

        absent = {}
        absent_ce = {}
        for branch in BRANCHES:
            logits, capture, audit = base._absent_forward(
                model, tokens, native, native["branches"][branch])
            absent[branch] = capture
            absent_ce[branch] = base.factorial_parent._per_token_ce(logits, targets)
            calls["absent_forwards"] += 1
            calls["absent_attention"] += audit["attention"]
            calls["absent_mlp"] += audit["mlp"]
            calls["site0_removal"] += audit["site0_removal"]
            errors["mlp0_state_max_abs"] = max(
                errors["mlp0_state_max_abs"], capture["mlp0_state_error"])
        absent_batches.append(torch.stack([absent_ce[name] for name in BRANCHES]))

        target_arms = []
        for target_index, branch in enumerate(BRANCHES):
            delta = native["z"].float() - absent[branch]["z"].float()
            source_terms = {
                source: base._secant(mlp1, delta, native["state_sources"][source])
                for source in SOURCES
            }
            full = sum(source_terms.values(), start=torch.zeros_like(delta))
            direct_full = base._secant(mlp1, delta, native["z"].float())
            curvature = -0.5 * base._secant(mlp1, delta, delta)
            own = direct_full + curvature
            errors["bilinear_source_sum_relative_squared_max"] = max(
                errors["bilinear_source_sum_relative_squared_max"],
                base._relative_squared(full, direct_full))
            deployed = native["M"] - absent[branch]["M"]
            combined = (full + curvature).to(deployed.dtype)
            errors["full_plus_curvature_bf16_relative_squared_max"] = max(
                errors["full_plus_curvature_bf16_relative_squared_max"],
                base._relative_squared(combined, deployed))
            own_write = (absent[branch]["M"].float() + own).to(native["M"].dtype)
            errors["own_native_write_relative_squared_max"] = max(
                errors["own_native_write_relative_squared_max"],
                base._relative_squared(own_write, native["M"]))
            writes = {"own": own, "full": full}
            writes.update({f"single_{source}": term
                           for source, term in source_terms.items()})
            writes.update({f"leave_{source}": full - term
                           for source, term in source_terms.items()})
            arms = []
            for mode in MODES:
                logits, audit = base._physical_forward(
                    model, tokens, absent[branch], writes[mode])
                calls["physical_forwards"] += 1
                calls["physical_attention"] += audit["attention"]
                calls["physical_mlp"] += audit["mlp"]
                for name in ("D", "A", "M"):
                    calls[f"{name}_injections"] += audit[name]
                arms.append(base.factorial_parent._per_token_ce(logits, targets))
            target_arms.append(torch.stack(arms, dim=-1))

            boundary = start_doc + SPLIT
            for local_start, local_stop, half in (
                    (0, max(0, min(stop, boundary) - start), 0),
                    (max(0, boundary - start), stop - start, 1)):
                if local_stop <= local_start:
                    continue
                sl = slice(local_start, local_stop)
                _accumulate(full_write_stats, half, target_index, 0, 0,
                            full[sl], own[sl])
                for control_index, shift in enumerate(POSITION_SHIFTS, start=1):
                    shifted_full = base._secant(
                        mlp1, delta[sl],
                        torch.roll(native["z"].float()[sl], shift, dims=1))
                    _accumulate(full_write_stats, half, target_index, 0, control_index,
                                shifted_full, own[sl])
                for source_index, source in enumerate(SOURCES):
                    term = source_terms[source]
                    _accumulate(source_write_stats, half, target_index, source_index, 0,
                                term[sl], full[sl])
                    for control_index, shift in enumerate(POSITION_SHIFTS, start=1):
                        shifted = base._secant(
                            mlp1, delta[sl],
                            torch.roll(native["state_sources"][source][sl], shift, dims=1))
                        _accumulate(source_write_stats, half, target_index, source_index,
                                    control_index, shifted, full[sl])
        arm_batches.append(torch.stack(target_arms))

    batches = math.ceil((stop_doc - start_doc) / BATCH)
    physical_per_batch = len(BRANCHES) * len(MODES)
    expected = {
        "native_forwards": batches,
        "absent_forwards": len(BRANCHES) * batches,
        "physical_forwards": physical_per_batch * batches,
        "native_attention": 18 * batches, "native_mlp": 18 * batches,
        "absent_attention": 18 * len(BRANCHES) * batches,
        "absent_mlp": 18 * len(BRANCHES) * batches,
        "site0_removal": len(BRANCHES) * batches,
        "physical_attention": 18 * physical_per_batch * batches,
        "physical_mlp": 18 * physical_per_batch * batches,
        "D_injections": physical_per_batch * batches,
        "A_injections": physical_per_batch * batches,
        "M_injections": physical_per_batch * batches,
    }
    instrument = {
        "calls": calls, "expected_calls": expected, "calls_exact": calls == expected,
        **{key: value for key, value in errors.items()
           if key not in ("analytical_num", "analytical_den", "deployed_num", "deployed_den")},
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
        "source_write_stats": source_write_stats,
        "source_write_cosines": _cosines(source_write_stats),
        "full_write_cosines": _cosines(full_write_stats)[:, :, 0],
        "instrument": instrument,
    }


def analyze_phase(collected, frozen_sources=None):
    arms = collected["arms"].double()
    absent = collected["absent"].double()
    benefits = absent[..., None] - arms
    halves = (slice(0, SPLIT), slice(SPLIT, 2 * SPLIT))
    reports = {}
    shared_sets = []
    pred_b = True
    numerical_control = True
    for half, docs in enumerate(halves):
        target_reports = {}
        necessary_sets = {}
        for target_index, branch in enumerate(BRANCHES):
            own = benefits[target_index, docs, ..., MODES.index("own")]
            full = benefits[target_index, docs, ..., MODES.index("full")]
            full_report = base._effect_report(full, own)
            full_write = float(collected["full_write_cosines"][half, target_index, 0])
            full_shift_q95 = float(torch.quantile(
                collected["full_write_cosines"][half, target_index, 1:],
                .95, interpolation="higher"))
            full_holds = bool(
                full_report["cosine"] >= .90
                and full_report["best_scalar_adjusted_relative_error"] <= .45
                and full_write >= full_shift_q95 + .15)
            if branch in ("T", "I"):
                pred_b &= full_holds
            source_reports = {}
            necessary = []
            full_rms = float(full.square().mean().sqrt())
            for source_index, source in enumerate(SOURCES):
                singleton = benefits[
                    target_index, docs, ..., MODES.index(f"single_{source}")]
                leave = benefits[
                    target_index, docs, ..., MODES.index(f"leave_{source}")]
                singleton_report = base._effect_report(singleton, full)
                leave_report = base._effect_report(leave, own)
                cosine_drop = full_report["cosine"] - leave_report["cosine"]
                error_increase = leave_report["best_scalar_adjusted_relative_error"] \
                    - full_report["best_scalar_adjusted_relative_error"]
                singleton_rms_ratio = float(singleton.square().mean().sqrt()) \
                    / max(full_rms, 1e-30)
                stats = collected["source_write_stats"][half, target_index, source_index]
                write_cosine = float(collected["source_write_cosines"][
                    half, target_index, source_index, 0])
                shift_q95 = float(torch.quantile(
                    collected["source_write_cosines"][half, target_index, source_index, 1:],
                    .95, interpolation="higher"))
                write_rms_ratio = math.sqrt(float(stats[0, 1]) / max(float(stats[0, 2]), 1e-30))
                necessary_holds = bool(
                    source != "NUMERICAL"
                    and cosine_drop >= .03
                    and error_increase >= .05
                    and singleton_rms_ratio >= .10
                    and write_cosine >= shift_q95 + .15)
                if necessary_holds:
                    necessary.append(source)
                singleton_sufficient = bool(
                    singleton_report["cosine"] >= .80
                    and singleton_report["best_scalar_adjusted_relative_error"] <= .50)
                source_reports[source] = {
                    "singleton_effect": singleton_report,
                    "leave_out_effect_vs_own": leave_report,
                    "full_minus_leave_cosine": cosine_drop,
                    "leave_minus_full_error": error_increase,
                    "singleton_effect_rms_over_full": singleton_rms_ratio,
                    "write_cosine_with_full": write_cosine,
                    "shift_q95": shift_q95,
                    "write_rms_over_full": write_rms_ratio,
                    "necessary_holds": necessary_holds,
                    "singleton_sufficient": singleton_sufficient,
                }
            necessary_sets[branch] = necessary
            target_reports[branch] = {
                "full_effect_vs_own": full_report,
                "full_write_cosine": full_write,
                "full_shift_q95": full_shift_q95,
                "full_holds": full_holds,
                "necessary_sources": necessary,
                "sources": source_reports,
            }
        shared = sorted(set(necessary_sets["T"]) & set(necessary_sets["I"]))
        numerical_ratios = [
            target_reports[branch]["sources"]["NUMERICAL"]["write_rms_over_full"]
            for branch in ("T", "I")
        ]
        numerical_ok = bool(
            "NUMERICAL" not in necessary_sets["T"]
            and "NUMERICAL" not in necessary_sets["I"]
            and max(numerical_ratios) < .02)
        numerical_control &= numerical_ok
        reports[f"half{half}"] = {
            "targets": target_reports,
            "shared_T_I_necessary_sources": shared,
            "numerical_write_ratios_T_I": numerical_ratios,
            "numerical_control_holds": numerical_ok,
        }
        shared_sets.append(shared)
    pred_c = all(bool(sources) for sources in shared_sets)
    stable = bool(pred_c and shared_sets[0] == shared_sets[1] and numerical_control)
    frozen_holds = frozen_sources is None or all(sources == frozen_sources for sources in shared_sets)
    semantic_indices = [MODES.index(mode) for mode in MODES
                        if "NUMERICAL" not in mode]
    semantic_live = bool((benefits[..., semantic_indices].square().mean(dim=(1, 2)) > 0).all())
    return {
        "half_reports": reports,
        "shared_source_sets": shared_sets,
        "selected_shared_sources": shared_sets[0] if stable else [],
        "pred_b_full_native_response_T_I": bool(pred_b),
        "pred_c_shared_named_source": bool(pred_c),
        "pred_d_stable_set_and_numerical_control": stable,
        "frozen_sources": frozen_sources,
        "frozen_sources_hold": bool(frozen_holds),
        "semantic_physical_arms_live": semantic_live,
    }


def instrument_valid(instrument, analysis):
    return bool(
        instrument["calls_exact"]
        and instrument["native_prefix_D_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_A_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_M_relative_squared_max"] <= 1e-12
        and instrument["prefix_z_relative_squared_max"] <= 1e-12
        and instrument["mlp0_state_max_abs"] == 0.0
        and instrument["state_source_relative_squared_max"] <= 1e-12
        and instrument["bilinear_source_sum_relative_squared_max"] <= 1e-8
        and instrument["full_plus_curvature_bf16_relative_squared_max"]
        <= POLARIZATION_BF16_BAR
        and instrument["own_native_write_relative_squared_max"] <= OWN_WRITE_BF16_BAR
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and analysis["semantic_physical_arms_live"])


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(SOURCES) == 9 and len(MODES) == 20
        assert len(BRANCHES) * len(MODES) == 60
        probe = torch.zeros(
            2, len(BRANCHES), 1, 1 + len(POSITION_SHIFTS), 3,
            dtype=torch.float64)
        unit = torch.ones(2, 3)
        _accumulate(probe, 0, 0, 0, 0, unit, unit)
        probe_cosines = _cosines(probe)
        assert probe_cosines.shape == (
            2, len(BRANCHES), 1, 1 + len(POSITION_SHIFTS))
        assert float(probe_cosines[0, 0, 0, 0]) == 1.0
        print(json.dumps({
            "status": "dry_run_passed", "rung": 491,
            "model_loaded": False, "outcomes_opened": False,
            "validation_outcomes_opened": False, "final_or_sealed_opened": False,
            "discovery_forwards": (500 // BATCH) * (1 + 3 + 60),
            "conditional_validation_forwards": (500 // BATCH) * (1 + 3 + 60),
            "sources": list(SOURCES),
            "registered_predictions": ["pred_a", "pred_b", "pred_c", "pred_d", "pred_e"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung491 output namespace already exists")
    rows, fit_rows, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = base.component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    discovery = collect_phase(model, rows, reference, *DISCOVERY_RANGE)
    discovery_analysis = analyze_phase(discovery)
    pred_a = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
                  and instrument_valid(discovery["instrument"], discovery_analysis))
    pred_b = discovery_analysis["pred_b_full_native_response_T_I"]
    pred_c = discovery_analysis["pred_c_shared_named_source"]
    pred_d = discovery_analysis["pred_d_stable_set_and_numerical_control"]
    validation_licensed = bool(pred_a and pred_b and pred_c and pred_d)
    validation = validation_analysis = None
    pred_e = False
    if validation_licensed:
        frozen_sources = discovery_analysis["selected_shared_sources"]
        validation = collect_phase(model, rows, reference, *VALIDATION_RANGE)
        validation_analysis = analyze_phase(validation, frozen_sources=frozen_sources)
        pred_e = bool(
            instrument_valid(validation["instrument"], validation_analysis)
            and validation_analysis["pred_b_full_native_response_T_I"]
            and validation_analysis["pred_c_shared_named_source"]
            and validation_analysis["pred_d_stable_set_and_numerical_control"]
            and validation_analysis["frozen_sources_hold"])
    strong_null = bool(not pred_a or not pred_b or not pred_c or not pred_d)
    result = {
        "status": "complete", "rung": 491,
        "claim_level": "named_MLP1_live_state_source_decomposition",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(BRANCHES), "sources": list(SOURCES), "modes": list(MODES),
        "position_shift_offsets": list(POSITION_SHIFTS),
        "discovery": {
            "documents": list(DISCOVERY_RANGE), "split": SPLIT,
            "instrument": discovery["instrument"], "analysis": discovery_analysis,
            "native_ce_mean": float(discovery["native"].mean()),
        },
        "validation": None if validation is None else {
            "documents": list(VALIDATION_RANGE), "split": 750,
            "instrument": validation["instrument"], "analysis": validation_analysis,
            "native_ce_mean": float(validation["native"].mean()),
        },
        "selected_shared_sources": discovery_analysis["selected_shared_sources"],
        'pred_a_exact_lawful_instrument': pred_a,
        'pred_b_full_native_response_T_I': pred_b,
        'pred_c_shared_named_source_T_I': pred_c,
        'pred_d_stable_source_set_and_numerical_control': pred_d,
        'pred_e_heldout_intervention_outcomes': pred_e,
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
            "selective_named_source_removal_and_composition"
            if pred_e else "two_dimensional_integrated_native_curvature_response"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 491,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "selected_shared_sources": result["selected_shared_sources"],
        "validation_opened": validation_licensed,
        "strong_null": strong_null,
        "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
