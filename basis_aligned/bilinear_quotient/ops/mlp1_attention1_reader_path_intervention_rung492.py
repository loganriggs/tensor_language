#!/usr/bin/env python3
"""RUNG492 -- selective attention1-to-MLP1 reader-path intervention."""

# BQGATE: EXPERIMENT
# pred_a exact selective-reader identity and lawful physical interventions
# pred_b actual attention1 knockout materially changes T and I branch effects
# pred_c selective A1-to-MLP1 edit captures a positional part of that change
# pred_d the complete supported branch set is stable and collateral is narrower
# pred_e the frozen path and branch set validate on prospective intervention outcomes

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
import mlp1_live_state_source_decomposition_rung491 as parent


PREREG = POLY / "MLP1_ATTENTION1_READER_PATH_INTERVENTION_RUNG492_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp1_live_state_source_decomposition_rung491.py"
PARENT_RESULT = ROOT / "mlp1_live_state_source_decomposition_rung491_results.json"
OUT = ROOT / "mlp1_attention1_reader_path_intervention_rung492_results.json"
HASHES = {
    PREREG: "75b468bd108bedee7f117fd7271612c4e7ca4fdf07fd7f6d999dd929eb33015c",
    PARENT_SOURCE: "329dbe8bbf1947b67676e5180fb3f0d6032150731b233e5767c4d0a7f0be554b",
    PARENT_RESULT: "f2df82c0c2d98489999d8860056fb66c3bfab1a5b3bf3f99befbc26d70ee9cc7",
}
BRANCHES = ("T", "C", "I", "S")
POSITION_SHIFTS = parent.POSITION_SHIFTS
CONDITIONS = ("same",) + tuple(f"shift_{value}" for value in POSITION_SHIFTS)
DISCOVERY_RANGE = (0, 500)
VALIDATION_RANGE = (500, 1000)
SPLIT = 250
BATCH = 4
U = 2.0 ** -8
SELECTIVE_BF16_BAR = 16 * U * U
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
    if receipt.get("rung") != 491 \
            or receipt.get("pred_a_exact_lawful_instrument") is not True \
            or receipt.get("pred_b_full_native_response_T_I") is not True \
            or receipt.get("pred_c_shared_named_source_T_I") is not True \
            or receipt.get("pred_d_stable_source_set_and_numerical_control") is not True \
            or receipt.get("pred_e_heldout_intervention_outcomes") is not True \
            or receipt.get("validation_licensed_and_opened") is not True \
            or receipt.get("strong_null") is not False \
            or receipt.get("selected_shared_sources") != ["A1"] \
            or receipt.get("next_step") != "selective_named_source_removal_and_composition":
        raise RuntimeError("rung491 did not license the selective A1 reader-path test")
    return parent.validate_inputs()


@torch.no_grad()
def _native_all(model, tokens, reference):
    logits, capture, calls = parent._native_with_sources(model, tokens, reference)
    prefix = parent.base.branch_parent._native_prefix(model, tokens, reference)
    capture["branches"] = {
        name: prefix["branches"][name].detach().clone() for name in BRANCHES
    }
    capture["S_prefix_replay_relative_squared"] = parent.base._relative_squared(
        capture["branches"]["S"], prefix["branches"]["S"])
    return logits, capture, calls


@torch.no_grad()
def _selective_forward(model, tokens, trajectory, mlp1_write):
    audit = {
        "attention": 0, "mlp": 0, "D": 0, "A": 0, "M": 0,
        "A_identity_max_abs": 0.0,
    }

    def attention(event):
        audit["attention"] += 1
        if event.site != 1:
            return event.block.attn(event.state, event.first_value)
        audit["A"] += 1
        write = trajectory["A"]
        audit["A_identity_max_abs"] = max(
            audit["A_identity_max_abs"],
            float((write - trajectory["A"]).abs().max()))
        return write, event.first_value

    def mlp(event):
        audit["mlp"] += 1
        if event.site == 0:
            audit["D"] += 1
            return trajectory["D"]
        if event.site == 1:
            audit["M"] += 1
            return mlp1_write
        return event.block.mlp(event.state)

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    return logits, audit


@torch.no_grad()
def _attention1_knockout_forward(model, tokens, site0_write):
    audit = {
        "attention": 0, "mlp": 0, "D": 0, "A1_zero": 0,
        "A1_max_abs": 0.0,
    }

    def attention(event):
        audit["attention"] += 1
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site != 1:
            return write, first_value
        audit["A1_zero"] += 1
        zero = torch.zeros_like(write)
        audit["A1_max_abs"] = max(audit["A1_max_abs"], float(zero.abs().max()))
        return zero, first_value

    def mlp(event):
        audit["mlp"] += 1
        if event.site == 0:
            audit["D"] += 1
            return site0_write
        return event.block.mlp(event.state)

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    return logits, audit


def _update_max(target, key, value):
    target[key] = max(target[key], float(value))


@torch.no_grad()
def collect_phase(model, rows, reference, start_doc, stop_doc):
    normal_native_batches = []
    normal_absent_batches = []
    selective_native_batches = []
    selective_absent_batches = []
    knockout_native_batches = []
    knockout_absent_batches = []
    calls = {
        "normal_native_forwards": 0, "normal_absent_forwards": 0,
        "selective_native_forwards": 0, "selective_absent_forwards": 0,
        "knockout_native_forwards": 0, "knockout_absent_forwards": 0,
        "normal_native_attention": 0, "normal_native_mlp": 0,
        "normal_absent_attention": 0, "normal_absent_mlp": 0,
        "normal_site0_removal": 0,
        "selective_attention": 0, "selective_mlp": 0,
        "selective_D": 0, "selective_A": 0, "selective_M": 0,
        "knockout_attention": 0, "knockout_mlp": 0,
        "knockout_D": 0, "knockout_A1_zero": 0,
        "standalone_MLP1_evaluations": 0,
    }
    errors = {
        "native_prefix_D_relative_squared_max": 0.0,
        "native_prefix_A_relative_squared_max": 0.0,
        "native_prefix_M_relative_squared_max": 0.0,
        "prefix_z_relative_squared_max": 0.0,
        "mlp0_state_max_abs": 0.0,
        "S_prefix_replay_relative_squared_max": 0.0,
        "state_source_relative_squared_max": 0.0,
        "selective_reader_identity_relative_squared_max": 0.0,
        "selective_reader_deployed_relative_squared_max": 0.0,
        "normal_own_write_relative_squared_max": 0.0,
        "selective_A_identity_max_abs": 0.0,
        "knockout_A1_max_abs": 0.0,
        "native_A1_source_rms_min": float("inf"),
        "analytical_num": 0.0, "analytical_den": 0.0,
        "deployed_num": 0.0, "deployed_den": 0.0,
    }
    device = next(model.parameters()).device
    mlp1 = model.transformer.h[1].mlp

    for start in range(start_doc, stop_doc, BATCH):
        stop = min(start + BATCH, stop_doc)
        tokens = rows[start:stop, :-1].to(device)
        targets = rows[start:stop, 1:].to(device)
        native_logits, native, native_calls = _native_all(
            model, tokens, reference)
        calls["normal_native_forwards"] += 1
        calls["normal_native_attention"] += native_calls["attention"]
        calls["normal_native_mlp"] += native_calls["mlp"]
        normal_native_batches.append(
            parent.base.factorial_parent._per_token_ce(native_logits, targets))
        for name, value in native["prefix_errors"].items():
            _update_max(errors, f"native_prefix_{name}_relative_squared_max", value)
        _update_max(errors, "prefix_z_relative_squared_max",
                    native["prefix_z_relative_squared"])
        _update_max(errors, "S_prefix_replay_relative_squared_max",
                    native["S_prefix_replay_relative_squared"])
        _update_max(errors, "state_source_relative_squared_max",
                    native["state_source_relative_squared"])
        a1_source = native["state_sources"]["A1"].float()
        errors["native_A1_source_rms_min"] = min(
            errors["native_A1_source_rms_min"],
            float(a1_source.double().square().mean().sqrt()))
        for key in ("analytical_num", "analytical_den", "deployed_num", "deployed_den"):
            errors[key] += native["identity"][key]

        absent = {}
        normal_absent_ce = []
        for branch in BRANCHES:
            logits, capture, audit = parent.base._absent_forward(
                model, tokens, native, native["branches"][branch])
            absent[branch] = capture
            normal_absent_ce.append(
                parent.base.factorial_parent._per_token_ce(logits, targets))
            calls["normal_absent_forwards"] += 1
            calls["normal_absent_attention"] += audit["attention"]
            calls["normal_absent_mlp"] += audit["mlp"]
            calls["normal_site0_removal"] += audit["site0_removal"]
            _update_max(errors, "mlp0_state_max_abs", capture["mlp0_state_error"])
        normal_absent_batches.append(torch.stack(normal_absent_ce))

        native_state = native["z"].float()
        native_direct = parent.base._mlp_write(mlp1, native_state)
        absent_states = {
            branch: absent[branch]["z"].float() for branch in BRANCHES
        }
        absent_direct = {
            branch: parent.base._mlp_write(mlp1, absent_states[branch])
            for branch in BRANCHES
        }
        calls["standalone_MLP1_evaluations"] += 1 + len(BRANCHES)
        for branch in BRANCHES:
            normal_own = (absent[branch]["M"].float()
                          + native_direct - absent_direct[branch]).to(native["M"].dtype)
            _update_max(errors, "normal_own_write_relative_squared_max",
                        parent.base._relative_squared(normal_own, native["M"]))

        condition_sources = (a1_source,) + tuple(
            torch.roll(a1_source, shift, dims=1) for shift in POSITION_SHIFTS)
        selective_native_ce = []
        selective_absent_ce = [[] for _ in BRANCHES]
        for condition_source in condition_sources:
            native_edit_fp = parent.base._mlp_write(
                mlp1, native_state - condition_source)
            calls["standalone_MLP1_evaluations"] += 1
            native_edit = native_edit_fp.to(native["M"].dtype)
            logits, audit = _selective_forward(
                model, tokens, native, native_edit)
            selective_native_ce.append(
                parent.base.factorial_parent._per_token_ce(logits, targets))
            calls["selective_native_forwards"] += 1
            for key in ("attention", "mlp", "D", "A", "M"):
                calls[f"selective_{key}"] += audit[key]
            _update_max(errors, "selective_A_identity_max_abs",
                        audit["A_identity_max_abs"])

            for branch_index, branch in enumerate(BRANCHES):
                branch_state = absent_states[branch]
                absent_edit_fp = parent.base._mlp_write(
                    mlp1, branch_state - condition_source)
                calls["standalone_MLP1_evaluations"] += 1
                absent_edit = absent_edit_fp.to(absent[branch]["M"].dtype)
                logits, audit = _selective_forward(
                    model, tokens, absent[branch], absent_edit)
                selective_absent_ce[branch_index].append(
                    parent.base.factorial_parent._per_token_ce(logits, targets))
                calls["selective_absent_forwards"] += 1
                for key in ("attention", "mlp", "D", "A", "M"):
                    calls[f"selective_{key}"] += audit[key]
                _update_max(errors, "selective_A_identity_max_abs",
                            audit["A_identity_max_abs"])

                delta = native_state - branch_state
                expected = native_direct - absent_direct[branch] \
                    - parent.base._secant(mlp1, delta, condition_source)
                actual = native_edit_fp - absent_edit_fp
                _update_max(errors, "selective_reader_identity_relative_squared_max",
                            parent.base._relative_squared(actual, expected))
                deployed_actual = (absent_edit.float() + expected).to(native_edit.dtype)
                _update_max(errors, "selective_reader_deployed_relative_squared_max",
                            parent.base._relative_squared(deployed_actual, native_edit))
        selective_native_batches.append(torch.stack(selective_native_ce))
        selective_absent_batches.append(torch.stack([
            torch.stack(values) for values in selective_absent_ce]))

        knockout_logits, knockout_audit = _attention1_knockout_forward(
            model, tokens, native["D"])
        knockout_native_batches.append(
            parent.base.factorial_parent._per_token_ce(knockout_logits, targets))
        calls["knockout_native_forwards"] += 1
        for key in ("attention", "mlp", "D", "A1_zero"):
            calls[f"knockout_{key}"] += knockout_audit[key]
        _update_max(errors, "knockout_A1_max_abs", knockout_audit["A1_max_abs"])
        knockout_absent_ce = []
        for branch in BRANCHES:
            logits, audit = _attention1_knockout_forward(
                model, tokens, absent[branch]["D"])
            knockout_absent_ce.append(
                parent.base.factorial_parent._per_token_ce(logits, targets))
            calls["knockout_absent_forwards"] += 1
            for key in ("attention", "mlp", "D", "A1_zero"):
                calls[f"knockout_{key}"] += audit[key]
            _update_max(errors, "knockout_A1_max_abs", audit["A1_max_abs"])
        knockout_absent_batches.append(torch.stack(knockout_absent_ce))

    batches = math.ceil((stop_doc - start_doc) / BATCH)
    selective_conditions = len(CONDITIONS)
    expected = {
        "normal_native_forwards": batches,
        "normal_absent_forwards": len(BRANCHES) * batches,
        "selective_native_forwards": selective_conditions * batches,
        "selective_absent_forwards": len(BRANCHES) * selective_conditions * batches,
        "knockout_native_forwards": batches,
        "knockout_absent_forwards": len(BRANCHES) * batches,
        "normal_native_attention": 18 * batches,
        "normal_native_mlp": 18 * batches,
        "normal_absent_attention": 18 * len(BRANCHES) * batches,
        "normal_absent_mlp": 18 * len(BRANCHES) * batches,
        "normal_site0_removal": len(BRANCHES) * batches,
        "selective_attention": 18 * selective_conditions * (1 + len(BRANCHES)) * batches,
        "selective_mlp": 18 * selective_conditions * (1 + len(BRANCHES)) * batches,
        "selective_D": selective_conditions * (1 + len(BRANCHES)) * batches,
        "selective_A": selective_conditions * (1 + len(BRANCHES)) * batches,
        "selective_M": selective_conditions * (1 + len(BRANCHES)) * batches,
        "knockout_attention": 18 * (1 + len(BRANCHES)) * batches,
        "knockout_mlp": 18 * (1 + len(BRANCHES)) * batches,
        "knockout_D": (1 + len(BRANCHES)) * batches,
        "knockout_A1_zero": (1 + len(BRANCHES)) * batches,
        "standalone_MLP1_evaluations": (
            (1 + len(BRANCHES)) * (1 + selective_conditions) * batches),
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
        "normal_native": torch.cat(normal_native_batches, dim=0),
        "normal_absent": torch.cat(normal_absent_batches, dim=1),
        "selective_native": torch.cat(selective_native_batches, dim=1),
        "selective_absent": torch.cat(selective_absent_batches, dim=2),
        "knockout_native": torch.cat(knockout_native_batches, dim=0),
        "knockout_absent": torch.cat(knockout_absent_batches, dim=1),
        "instrument": instrument,
    }


def _rms(value):
    return float(value.double().square().mean().sqrt())


def analyze_phase(collected, frozen_branch_set=None):
    native = collected["normal_native"].double()
    absent = collected["normal_absent"].double()
    selective_native = collected["selective_native"].double()
    selective_absent = collected["selective_absent"].double()
    knockout_native = collected["knockout_native"].double()
    knockout_absent = collected["knockout_absent"].double()
    half_slices = (slice(0, SPLIT), slice(SPLIT, 2 * SPLIT))
    reports = {}
    supported_sets = []
    pred_b = pred_c = True
    collateral_holds = True
    all_arms_live = True

    for half_index, docs in enumerate(half_slices):
        native_reader_damage = selective_native[0, docs] - native[docs]
        native_knockout_damage = knockout_native[docs] - native[docs]
        reader_collateral_rms = _rms(native_reader_damage)
        knockout_collateral_rms = _rms(native_knockout_damage)
        collateral = bool(reader_collateral_rms < knockout_collateral_rms)
        collateral_holds &= collateral
        branch_reports = {}
        supported = []
        for branch_index, branch in enumerate(BRANCHES):
            normal_effect = absent[branch_index, docs] - native[docs]
            selective_effect = selective_absent[branch_index, :, docs] \
                - selective_native[:, docs]
            knockout_effect = knockout_absent[branch_index, docs] \
                - knockout_native[docs]
            reader_modulations = normal_effect.unsqueeze(0) - selective_effect
            knockout_modulation = normal_effect - knockout_effect
            normal_rms = _rms(normal_effect)
            knockout_rms = _rms(knockout_modulation)
            reader_rms = _rms(reader_modulations[0])
            knockout_over_normal = knockout_rms / max(normal_rms, 1e-30)
            covariance = float(
                (knockout_modulation * normal_effect).double().mean())
            dependence_holds = bool(knockout_over_normal >= .10 and covariance > 0)
            effect_report = parent.base._effect_report(
                reader_modulations[0], knockout_modulation)
            shifted_cosines = torch.tensor([
                parent.base._cosine(value, knockout_modulation)
                for value in reader_modulations[1:]
            ], dtype=torch.float64)
            shifted_q95 = float(torch.quantile(
                shifted_cosines, .95, interpolation="higher"))
            reader_over_knockout = reader_rms / max(knockout_rms, 1e-30)
            positional_margin = effect_report["cosine"] - shifted_q95
            reader_path_holds = bool(
                effect_report["cosine"] >= .60
                and effect_report["best_scalar_adjusted_relative_error"] <= .80
                and reader_over_knockout >= .25
                and positional_margin >= .10)
            branch_supported = bool(dependence_holds and reader_path_holds)
            if branch_supported:
                supported.append(branch)
            if branch in ("T", "I"):
                pred_b &= dependence_holds
                pred_c &= reader_path_holds
            branch_live = bool(
                normal_rms > 0 and knockout_rms > 0
                and bool((reader_modulations.square().mean(dim=(1, 2)) > 0).all()))
            all_arms_live &= branch_live
            branch_reports[branch] = {
                "normal_effect_rms": normal_rms,
                "knockout_modulation_rms": knockout_rms,
                "knockout_modulation_rms_over_normal": knockout_over_normal,
                "knockout_modulation_dot_normal_mean": covariance,
                "actual_A1_dependence_holds": dependence_holds,
                "reader_modulation_rms": reader_rms,
                "reader_modulation_rms_over_knockout": reader_over_knockout,
                "reader_vs_knockout_modulation": effect_report,
                "shifted_reader_cosines": shifted_cosines.tolist(),
                "shifted_reader_cosine_q95": shifted_q95,
                "same_minus_shift_q95": positional_margin,
                "reader_path_holds": reader_path_holds,
                "branch_supported": branch_supported,
                "all_interventions_live": branch_live,
            }
        reports[f"half{half_index}"] = {
            "branches": branch_reports,
            "supported_branches": supported,
            "selective_native_CE_change_rms": reader_collateral_rms,
            "full_A1_knockout_native_CE_change_rms": knockout_collateral_rms,
            "selective_collateral_lower": collateral,
        }
        supported_sets.append(supported)

    stable_set = bool(
        supported_sets[0] == supported_sets[1]
        and all(branch in supported_sets[0] for branch in ("T", "I")))
    frozen_holds = frozen_branch_set is None or all(
        value == frozen_branch_set for value in supported_sets)
    return {
        "half_reports": reports,
        "supported_branch_sets": supported_sets,
        "selected_supported_branches": supported_sets[0] if stable_set else [],
        "pred_b_actual_A1_dependence_T_I": bool(pred_b),
        "pred_c_selective_reader_captures_T_I": bool(pred_c),
        "stable_supported_branch_set": stable_set,
        "selective_collateral_lower_all_halves": bool(collateral_holds),
        "frozen_branch_set": frozen_branch_set,
        "frozen_branch_set_holds": bool(frozen_holds),
        "all_physical_interventions_live": bool(all_arms_live),
    }


def instrument_valid(instrument, analysis):
    return bool(
        instrument["calls_exact"]
        and instrument["native_prefix_D_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_A_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_M_relative_squared_max"] <= 1e-12
        and instrument["prefix_z_relative_squared_max"] <= 1e-12
        and instrument["mlp0_state_max_abs"] == 0.0
        and instrument["S_prefix_replay_relative_squared_max"] <= 1e-12
        and instrument["state_source_relative_squared_max"] <= 1e-12
        and instrument["selective_reader_identity_relative_squared_max"] <= 1e-8
        and instrument["selective_reader_deployed_relative_squared_max"]
            <= SELECTIVE_BF16_BAR
        and instrument["normal_own_write_relative_squared_max"] <= OWN_WRITE_BF16_BAR
        and instrument["selective_A_identity_max_abs"] == 0.0
        and instrument["knockout_A1_max_abs"] == 0.0
        and instrument["native_A1_source_rms_min"] > 0
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and analysis["all_physical_interventions_live"])


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(BRANCHES) == 4 and len(CONDITIONS) == 17
        per_batch = (1 + len(BRANCHES)) * (2 + len(CONDITIONS))
        assert per_batch == 95
        print(json.dumps({
            "status": "dry_run_passed", "rung": 492,
            "model_loaded": False, "outcomes_opened": False,
            "validation_outcomes_opened": False, "final_or_sealed_opened": False,
            "discovery_forwards": (500 // BATCH) * per_batch,
            "conditional_validation_forwards": (500 // BATCH) * per_batch,
            "branches": list(BRANCHES), "conditions": list(CONDITIONS),
            "registered_predictions": ["pred_a", "pred_b", "pred_c", "pred_d", "pred_e"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung492 output namespace already exists")
    rows, fit_rows, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = parent.base.component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    discovery = collect_phase(model, rows, reference, *DISCOVERY_RANGE)
    discovery_analysis = analyze_phase(discovery)
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and instrument_valid(discovery["instrument"], discovery_analysis))
    pred_b = discovery_analysis["pred_b_actual_A1_dependence_T_I"]
    pred_c = discovery_analysis["pred_c_selective_reader_captures_T_I"]
    pred_d = bool(
        discovery_analysis["stable_supported_branch_set"]
        and discovery_analysis["selective_collateral_lower_all_halves"])
    validation_licensed = bool(pred_a and pred_b and pred_c and pred_d)
    validation = validation_analysis = None
    pred_e = False
    if validation_licensed:
        frozen_branch_set = discovery_analysis["selected_supported_branches"]
        validation = collect_phase(model, rows, reference, *VALIDATION_RANGE)
        validation_analysis = analyze_phase(
            validation, frozen_branch_set=frozen_branch_set)
        pred_e = bool(
            instrument_valid(validation["instrument"], validation_analysis)
            and validation_analysis["pred_b_actual_A1_dependence_T_I"]
            and validation_analysis["pred_c_selective_reader_captures_T_I"]
            and validation_analysis["stable_supported_branch_set"]
            and validation_analysis["selective_collateral_lower_all_halves"]
            and validation_analysis["frozen_branch_set_holds"])
    strong_null = bool(not (pred_a and pred_b and pred_c and pred_d))
    result = {
        "status": "complete", "rung": 492,
        "claim_level": "selective_attention1_to_MLP1_reader_path_intervention",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(BRANCHES), "conditions": list(CONDITIONS),
        "position_shift_offsets": list(POSITION_SHIFTS),
        "discovery": {
            "documents": list(DISCOVERY_RANGE), "split": SPLIT,
            "instrument": discovery["instrument"], "analysis": discovery_analysis,
        },
        "validation": None if validation is None else {
            "documents": list(VALIDATION_RANGE), "split": 750,
            "instrument": validation["instrument"], "analysis": validation_analysis,
        },
        "selected_supported_branches": discovery_analysis[
            "selected_supported_branches"],
        'pred_a_exact_lawful_intervention': pred_a,
        'pred_b_actual_A1_dependence_T_I': pred_b,
        'pred_c_selective_reader_captures_T_I': pred_c,
        'pred_d_stable_branch_group_and_narrower_collateral': pred_d,
        'pred_e_prospective_intervention_type_validation': pred_e,
        "validation_licensed_and_opened": validation_licensed,
        "strong_null": strong_null,
        "final_or_sealed_opened": False,
        "execution_price": {
            "discovery_full_model_forwards": sum(
                value for key, value in discovery["instrument"]["calls"].items()
                if key.endswith("_forwards")),
            "validation_full_model_forwards": 0 if validation is None else sum(
                value for key, value in validation["instrument"]["calls"].items()
                if key.endswith("_forwards")),
            "discovery_standalone_MLP1_evaluations": discovery["instrument"]["calls"][
                "standalone_MLP1_evaluations"],
            "validation_standalone_MLP1_evaluations": 0 if validation is None else
                validation["instrument"]["calls"]["standalone_MLP1_evaluations"],
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "new_corpus_A1_reader_path_and_semantic_preservation"
            if pred_e else "retain_local_A1_output_attribution_and_test_T_I_filtration"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 492,
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "selected_supported_branches": result["selected_supported_branches"],
        "validation_opened": validation_licensed,
        "strong_null": strong_null,
        "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
