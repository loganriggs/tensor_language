#!/usr/bin/env python3
"""RUNG493 -- physical site-graded merge of exact MLP0 branch responses."""

# BQGATE: EXPERIMENT
# pred_a exact lawful live merge instrument
# pred_b attention1 merge removes a positional part of the T/I CE-effect contrast
# pred_c the T/I distinction is more causally live at attention1 than MLP1
# pred_d the depth gradient is T/I-specific among all branch pairs
# pred_e the frozen relation validates on prospective intervention outcomes

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

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import mlp1_attention1_reader_path_intervention_rung492 as parent


PREREG = POLY / "MLP0_TI_SITE_GRADED_MERGE_INTERVENTION_RUNG493_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp1_attention1_reader_path_intervention_rung492.py"
PARENT_RESULT = ROOT / "mlp1_attention1_reader_path_intervention_rung492_results.json"
OUT = ROOT / "mlp0_TI_site_graded_merge_intervention_rung493_results.json"
HASHES = {
    PREREG: "e0e59a8d61c88efc2f7140a364169e468164b708d546c4f15c7f30fb54676e00",
    PARENT_SOURCE: "68e5087d2085c55ca93c5d749ee0abea366733dbc8bf6624167b831cda687ed6",
    PARENT_RESULT: "27084ccf6dcea3c1f92014c11b4910b060f0db0d5bffb35170c4c845e531f1fd",
}
BRANCHES = parent.BRANCHES
PAIRS = tuple(itertools.combinations(BRANCHES, 2))
PAIR_NAMES = tuple(f"{left}x{right}" for left, right in PAIRS)
MODES = ("A_RECOMPUTE", "A_DIRECT", "M_ONLY")
SHIFT_MODES = ("A_RECOMPUTE", "M_ONLY")
POSITION_SHIFTS = parent.POSITION_SHIFTS
DISCOVERY_RANGE = (0, 500)
VALIDATION_RANGE = (500, 1000)
HALF = 250
BATCH = 4


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
    if receipt.get("rung") != 492 \
            or receipt.get("pred_a_exact_lawful_intervention") is not True \
            or receipt.get("pred_b_actual_A1_dependence_T_I") is not True \
            or receipt.get("pred_c_selective_reader_captures_T_I") is not False \
            or receipt.get("pred_d_stable_branch_group_and_narrower_collateral") is not False \
            or receipt.get("pred_e_prospective_intervention_type_validation") is not False \
            or receipt.get("validation_licensed_and_opened") is not False \
            or receipt.get("strong_null") is not True \
            or receipt.get("selected_supported_branches") != [] \
            or receipt.get("next_step") != "retain_local_A1_output_attribution_and_test_T_I_filtration":
        raise RuntimeError("rung492 did not license the site-graded T/I test")
    return parent.validate_inputs()


def _update_max(target, key, value):
    target[key] = max(target[key], float(value))


def _rms(value):
    return float(torch.as_tensor(value, dtype=torch.float64).square().mean().sqrt())


@torch.no_grad()
def _merge_forward(model, tokens, trajectory, mode, edited_write):
    audit = {
        "attention": 0, "mlp": 0, "D": 0, "A": 0, "M": 0,
        "edited_site": 0, "edited_write_max_abs_error": 0.0,
    }
    actual_edited_write = []

    def attention(event):
        audit["attention"] += 1
        if event.site != 1:
            return event.block.attn(event.state, event.first_value)
        audit["A"] += 1
        if mode in ("A_RECOMPUTE", "A_DIRECT"):
            audit["edited_site"] += 1
            returned = edited_write
            audit["edited_write_max_abs_error"] = max(
                audit["edited_write_max_abs_error"],
                float((returned - edited_write).abs().max()))
            actual_edited_write.append(returned.detach().clone())
            return returned, event.first_value
        return trajectory["A"], event.first_value

    def mlp(event):
        audit["mlp"] += 1
        if event.site == 0:
            audit["D"] += 1
            return trajectory["D"]
        if event.site != 1:
            return event.block.mlp(event.state)
        audit["M"] += 1
        if mode == "A_RECOMPUTE":
            return event.block.mlp(event.state)
        if mode == "A_DIRECT":
            return trajectory["M"]
        if mode == "M_ONLY":
            audit["edited_site"] += 1
            returned = edited_write
            audit["edited_write_max_abs_error"] = max(
                audit["edited_write_max_abs_error"],
                float((returned - edited_write).abs().max()))
            actual_edited_write.append(returned.detach().clone())
            return returned
        raise RuntimeError(f"unknown merge mode: {mode}")

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    if len(actual_edited_write) != 1:
        raise RuntimeError("merge forward did not edit exactly one site")
    return logits, audit, actual_edited_write[0]


def _accumulate_write_gram(gram, phase_start, batch_start, batch_stop, responses):
    """Accumulate [site, branch, branch] Gram matrices into two phase-local halves."""
    for half_index in range(2):
        lo = phase_start + half_index * HALF
        hi = lo + HALF
        selected = (torch.arange(batch_start, batch_stop) >= lo) \
            & (torch.arange(batch_start, batch_stop) < hi)
        if not bool(selected.any()):
            continue
        for site_index, site in enumerate(("A", "M")):
            device_mask = selected.to(responses[BRANCHES[0]][site].device)
            block = torch.stack([responses[name][site][device_mask] for name in BRANCHES])
            flat = block.detach().float().cpu().double().reshape(len(BRANCHES), -1)
            gram[half_index, site_index] += flat @ flat.T


@torch.no_grad()
def collect_phase(model, rows, reference, start_doc, stop_doc):
    native_batches = []
    absent_batches = []
    same_batches = []
    shift_batches = []
    write_gram = torch.zeros(2, 2, len(BRANCHES), len(BRANCHES), dtype=torch.float64)
    calls = {
        "normal_native_forwards": 0, "normal_absent_forwards": 0,
        "merge_forwards": 0, "A_RECOMPUTE_forwards": 0,
        "A_DIRECT_forwards": 0, "M_ONLY_forwards": 0,
        "normal_native_attention": 0, "normal_native_mlp": 0,
        "normal_absent_attention": 0, "normal_absent_mlp": 0,
        "normal_site0_removal": 0,
        "merge_attention": 0, "merge_mlp": 0, "merge_D": 0,
        "merge_A": 0, "merge_M": 0, "merge_edited_site": 0,
    }
    errors = {
        "native_prefix_D_relative_squared_max": 0.0,
        "native_prefix_A_relative_squared_max": 0.0,
        "native_prefix_M_relative_squared_max": 0.0,
        "prefix_z_relative_squared_max": 0.0,
        "mlp0_state_max_abs": 0.0,
        "S_prefix_replay_relative_squared_max": 0.0,
        "state_source_relative_squared_max": 0.0,
        "same_site_pair_max_abs": 0.0,
        "edited_write_max_abs_error": 0.0,
        "merge_adjustment_rms_min": float("inf"),
        "analytical_num": 0.0, "analytical_den": 0.0,
        "deployed_num": 0.0, "deployed_den": 0.0,
    }
    device = next(model.parameters()).device

    for start in range(start_doc, stop_doc, BATCH):
        stop = min(start + BATCH, stop_doc)
        tokens = rows[start:stop, :-1].to(device)
        targets = rows[start:stop, 1:].to(device)
        native_logits, native, native_calls = parent._native_all(model, tokens, reference)
        native_batches.append(parent.parent.base.factorial_parent._per_token_ce(
            native_logits, targets))
        calls["normal_native_forwards"] += 1
        calls["normal_native_attention"] += native_calls["attention"]
        calls["normal_native_mlp"] += native_calls["mlp"]
        for name, value in native["prefix_errors"].items():
            _update_max(errors, f"native_prefix_{name}_relative_squared_max", value)
        _update_max(errors, "prefix_z_relative_squared_max",
                    native["prefix_z_relative_squared"])
        _update_max(errors, "S_prefix_replay_relative_squared_max",
                    native["S_prefix_replay_relative_squared"])
        _update_max(errors, "state_source_relative_squared_max",
                    native["state_source_relative_squared"])
        for key in ("analytical_num", "analytical_den", "deployed_num", "deployed_den"):
            errors[key] += native["identity"][key]

        absent = {}
        absent_ce = []
        for branch in BRANCHES:
            logits, capture, audit = parent.parent.base._absent_forward(
                model, tokens, native, native["branches"][branch])
            absent[branch] = capture
            absent_ce.append(parent.parent.base.factorial_parent._per_token_ce(
                logits, targets))
            calls["normal_absent_forwards"] += 1
            calls["normal_absent_attention"] += audit["attention"]
            calls["normal_absent_mlp"] += audit["mlp"]
            calls["normal_site0_removal"] += audit["site0_removal"]
            _update_max(errors, "mlp0_state_max_abs", capture["mlp0_state_error"])
        absent_batches.append(torch.stack(absent_ce))

        responses = {
            branch: {
                "A": native["A"].float() - absent[branch]["A"].float(),
                "M": native["M"].float() - absent[branch]["M"].float(),
            } for branch in BRANCHES
        }
        _accumulate_write_gram(write_gram, start_doc, start, stop, responses)

        pair_outputs = []
        for left, right in PAIRS:
            mode_outputs = []
            common = {
                site: ((absent[left][site].float() + absent[right][site].float()) / 2)
                    .to(absent[left][site].dtype)
                for site in ("A", "M")
            }
            for site in ("A", "M"):
                for branch in (left, right):
                    adjustment = common[site].float() - absent[branch][site].float()
                    errors["merge_adjustment_rms_min"] = min(
                        errors["merge_adjustment_rms_min"], _rms(adjustment))
            for mode in MODES:
                site = "A" if mode.startswith("A_") else "M"
                side_outputs = []
                actual_site_writes = []
                for branch in (left, right):
                    logits, audit, actual_site_write = _merge_forward(
                        model, tokens, absent[branch], mode, common[site])
                    actual_site_writes.append(actual_site_write)
                    side_outputs.append(
                        parent.parent.base.factorial_parent._per_token_ce(logits, targets))
                    calls["merge_forwards"] += 1
                    calls[f"{mode}_forwards"] += 1
                    for key in ("attention", "mlp", "D", "A", "M", "edited_site"):
                        calls[f"merge_{key}"] += audit[key]
                    _update_max(errors, "edited_write_max_abs_error",
                                audit["edited_write_max_abs_error"])
                _update_max(errors, "same_site_pair_max_abs",
                            (actual_site_writes[0] - actual_site_writes[1]).abs().max())
                mode_outputs.append(torch.stack(side_outputs))
            pair_outputs.append(torch.stack(mode_outputs))
        same_batches.append(torch.stack(pair_outputs))

        ti_left, ti_right = ("T", "I")
        shifted_mode_outputs = []
        for mode in SHIFT_MODES:
            site = "A" if mode.startswith("A_") else "M"
            common = ((absent[ti_left][site].float() + absent[ti_right][site].float()) / 2) \
                .to(absent[ti_left][site].dtype)
            side_outputs = []
            for branch in (ti_left, ti_right):
                adjustment = common.float() - absent[branch][site].float()
                shifted_outputs = []
                for shift in POSITION_SHIFTS:
                    edited = (absent[branch][site].float()
                              + torch.roll(adjustment, shift, dims=1)) \
                        .to(absent[branch][site].dtype)
                    logits, audit, _actual_site_write = _merge_forward(
                        model, tokens, absent[branch], mode, edited)
                    shifted_outputs.append(
                        parent.parent.base.factorial_parent._per_token_ce(logits, targets))
                    calls["merge_forwards"] += 1
                    calls[f"{mode}_forwards"] += 1
                    for key in ("attention", "mlp", "D", "A", "M", "edited_site"):
                        calls[f"merge_{key}"] += audit[key]
                    _update_max(errors, "edited_write_max_abs_error",
                                audit["edited_write_max_abs_error"])
                side_outputs.append(torch.stack(shifted_outputs))
            shifted_mode_outputs.append(torch.stack(side_outputs))
        shift_batches.append(torch.stack(shifted_mode_outputs))

    batches = math.ceil((stop_doc - start_doc) / BATCH)
    same_forwards = len(PAIRS) * len(MODES) * 2 * batches
    shift_forwards = len(SHIFT_MODES) * 2 * len(POSITION_SHIFTS) * batches
    merge_forwards = same_forwards + shift_forwards
    mode_forwards = {
        "A_RECOMPUTE": (len(PAIRS) * 2 + 2 * len(POSITION_SHIFTS)) * batches,
        "A_DIRECT": len(PAIRS) * 2 * batches,
        "M_ONLY": (len(PAIRS) * 2 + 2 * len(POSITION_SHIFTS)) * batches,
    }
    expected = {
        "normal_native_forwards": batches,
        "normal_absent_forwards": len(BRANCHES) * batches,
        "merge_forwards": merge_forwards,
        **{f"{mode}_forwards": count for mode, count in mode_forwards.items()},
        "normal_native_attention": 18 * batches,
        "normal_native_mlp": 18 * batches,
        "normal_absent_attention": 18 * len(BRANCHES) * batches,
        "normal_absent_mlp": 18 * len(BRANCHES) * batches,
        "normal_site0_removal": len(BRANCHES) * batches,
        "merge_attention": 18 * merge_forwards,
        "merge_mlp": 18 * merge_forwards,
        "merge_D": merge_forwards,
        "merge_A": merge_forwards,
        "merge_M": merge_forwards,
        "merge_edited_site": merge_forwards,
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
        "normal_native": torch.cat(native_batches, dim=0),
        "normal_absent": torch.cat(absent_batches, dim=1),
        "same": torch.cat(same_batches, dim=3),
        "shifts": torch.cat(shift_batches, dim=3),
        "write_gram": write_gram,
        "instrument": instrument,
    }


def _contrast_report(normal, merged):
    normal = torch.as_tensor(normal, dtype=torch.float64).reshape(-1)
    merged = torch.as_tensor(merged, dtype=torch.float64).reshape(-1)
    removed = normal - merged
    normal2 = torch.dot(normal, normal).clamp_min(1e-30)
    removed2 = torch.dot(removed, removed).clamp_min(1e-30)
    aligned = float(torch.dot(removed, normal) / normal2)
    cosine = float(torch.dot(removed, normal) / torch.sqrt(removed2 * normal2))
    return {
        "normal_contrast_rms": _rms(normal),
        "merged_contrast_rms": _rms(merged),
        "removal_rms": _rms(removed),
        "aligned_removed_fraction": aligned,
        "removal_cosine": cosine,
        "residual_ratio": _rms(merged) / max(_rms(normal), 1e-30),
    }


def _common_share(gram, left, right):
    i, j = BRANCHES.index(left), BRANCHES.index(right)
    numerator = gram[i, i] + gram[j, j] + 2 * gram[i, j]
    denominator = 2 * (gram[i, i] + gram[j, j])
    return float(numerator / max(float(denominator), 1e-30))


def analyze_phase(collected, frozen_top_pair=None):
    absent = collected["normal_absent"].double()
    same = collected["same"].double()
    shifts = collected["shifts"].double()
    reports = {}
    all_live = True
    pred_b = pred_c = pred_d = True
    top_pairs = []

    for half_index in range(2):
        docs = slice(half_index * HALF, (half_index + 1) * HALF)
        pair_reports = {}
        write_gradients = {}
        physical_gaps = {}
        for pair_index, (left, right) in enumerate(PAIRS):
            left_index, right_index = BRANCHES.index(left), BRANCHES.index(right)
            normal = absent[left_index, docs] - absent[right_index, docs]
            mode_reports = {}
            for mode_index, mode in enumerate(MODES):
                merged = same[pair_index, mode_index, 0, docs] \
                    - same[pair_index, mode_index, 1, docs]
                report = _contrast_report(normal, merged)
                all_live &= report["removal_rms"] > 0
                if (left, right) == ("T", "I") and mode in SHIFT_MODES:
                    shift_mode_index = SHIFT_MODES.index(mode)
                    shifted_fractions = []
                    for shift_index in range(len(POSITION_SHIFTS)):
                        shifted = shifts[shift_mode_index, 0, shift_index, docs] \
                            - shifts[shift_mode_index, 1, shift_index, docs]
                        shifted_report = _contrast_report(normal, shifted)
                        shifted_fractions.append(
                            shifted_report["aligned_removed_fraction"])
                        all_live &= shifted_report["removal_rms"] > 0
                    shifted_q95 = float(torch.quantile(
                        torch.tensor(shifted_fractions, dtype=torch.float64),
                        .95, interpolation="higher"))
                    report["shifted_aligned_removed_fractions"] = shifted_fractions
                    report["shifted_aligned_removed_fraction_q95"] = shifted_q95
                    report["same_minus_shift_q95"] = (
                        report["aligned_removed_fraction"] - shifted_q95)
                mode_reports[mode] = report

            gram = collected["write_gram"][half_index]
            attention_share = _common_share(gram[0], left, right)
            mlp_share = _common_share(gram[1], left, right)
            write_gradient = mlp_share - attention_share
            physical_gap = (mode_reports["A_RECOMPUTE"]["aligned_removed_fraction"]
                            - mode_reports["M_ONLY"]["aligned_removed_fraction"])
            write_gradients[f"{left}x{right}"] = write_gradient
            physical_gaps[f"{left}x{right}"] = physical_gap
            pair_reports[f"{left}x{right}"] = {
                "attention1_common_share": attention_share,
                "mlp1_common_share": mlp_share,
                "write_common_share_increase": write_gradient,
                "physical_A_minus_M_aligned_fraction": physical_gap,
                "modes": mode_reports,
            }

        ti = pair_reports["TxI"]
        a = ti["modes"]["A_RECOMPUTE"]
        m = ti["modes"]["M_ONLY"]
        b_holds = bool(
            a["removal_cosine"] >= .50
            and a["aligned_removed_fraction"] >= .20
            and a["residual_ratio"] <= .95
            and a["same_minus_shift_q95"] >= .10)
        c_holds = bool(
            ti["physical_A_minus_M_aligned_fraction"] >= .10
            and a["removal_rms"] >= 1.25 * m["removal_rms"])
        other_names = [name for name in PAIR_NAMES if name != "TxI"]
        top_write = max(write_gradients, key=write_gradients.get)
        top_physical = max(physical_gaps, key=physical_gaps.get)
        top_pair = "TxI" if top_write == top_physical == "TxI" else None
        top_pairs.append(top_pair)
        d_holds = bool(
            ti["write_common_share_increase"] >= .15
            and ti["write_common_share_increase"]
                >= max(write_gradients[name] for name in other_names) + .05
            and ti["physical_A_minus_M_aligned_fraction"]
                >= max(physical_gaps[name] for name in other_names) + .05
            and top_pair == "TxI")
        pred_b &= b_holds
        pred_c &= c_holds
        pred_d &= d_holds
        reports[f"half{half_index}"] = {
            "pairs": pair_reports,
            "write_gradient_order": sorted(
                write_gradients, key=write_gradients.get, reverse=True),
            "physical_gap_order": sorted(
                physical_gaps, key=physical_gaps.get, reverse=True),
            "top_pair_both_comparisons": top_pair,
            "pred_b_attention1_T_I_merge": b_holds,
            "pred_c_progressive_T_I_merge": c_holds,
            "pred_d_T_I_specific": d_holds,
        }

    stable_top = bool(top_pairs == ["TxI", "TxI"])
    frozen_holds = frozen_top_pair is None or all(
        value == frozen_top_pair for value in top_pairs)
    return {
        "half_reports": reports,
        "top_pairs": top_pairs,
        "stable_top_pair": stable_top,
        "frozen_top_pair": frozen_top_pair,
        "frozen_top_pair_holds": frozen_holds,
        "pred_b_attention1_merge_removes_T_I_contrast": bool(pred_b),
        "pred_c_progressive_T_I_merge": bool(pred_c),
        "pred_d_T_I_specific_depth_gradient": bool(pred_d and stable_top),
        "all_physical_interventions_live": bool(all_live),
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
        and instrument["same_site_pair_max_abs"] == 0.0
        and instrument["edited_write_max_abs_error"] == 0.0
        and instrument["merge_adjustment_rms_min"] > 0
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and analysis["all_physical_interventions_live"])


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert BRANCHES == ("T", "C", "I", "S")
        assert len(PAIRS) == 6 and len(MODES) == 3 and len(SHIFT_MODES) == 2
        per_batch = 1 + len(BRANCHES) \
            + len(PAIRS) * len(MODES) * 2 \
            + len(SHIFT_MODES) * 2 * len(POSITION_SHIFTS)
        assert per_batch == 105
        print(json.dumps({
            "status": "dry_run_passed", "rung": 493,
            "model_loaded": False, "outcomes_opened": False,
            "validation_outcomes_opened": False, "final_or_sealed_opened": False,
            "discovery_forwards": (500 // BATCH) * per_batch,
            "conditional_validation_forwards": (500 // BATCH) * per_batch,
            "branches": list(BRANCHES), "pairs": list(PAIR_NAMES),
            "modes": list(MODES), "shift_modes": list(SHIFT_MODES),
            "registered_predictions": ["pred_a", "pred_b", "pred_c", "pred_d", "pred_e"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung493 output namespace already exists")
    rows, fit_rows, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = parent.parent.base.component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    discovery = collect_phase(model, rows, reference, *DISCOVERY_RANGE)
    discovery_analysis = analyze_phase(discovery)
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and instrument_valid(discovery["instrument"], discovery_analysis))
    pred_b = discovery_analysis["pred_b_attention1_merge_removes_T_I_contrast"]
    pred_c = discovery_analysis["pred_c_progressive_T_I_merge"]
    pred_d = discovery_analysis["pred_d_T_I_specific_depth_gradient"]
    validation_licensed = bool(pred_a and pred_b and pred_c and pred_d)
    validation = validation_analysis = None
    pred_e = False
    if validation_licensed:
        validation = collect_phase(model, rows, reference, *VALIDATION_RANGE)
        validation_analysis = analyze_phase(validation, frozen_top_pair="TxI")
        pred_e = bool(
            instrument_valid(validation["instrument"], validation_analysis)
            and validation_analysis["pred_b_attention1_merge_removes_T_I_contrast"]
            and validation_analysis["pred_c_progressive_T_I_merge"]
            and validation_analysis["pred_d_T_I_specific_depth_gradient"]
            and validation_analysis["frozen_top_pair_holds"])
    strong_null = bool(not (pred_a and pred_b and pred_c and pred_d))
    result = {
        "status": "complete", "rung": 493,
        "claim_level": "site_graded_physical_T_I_response_merge",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(BRANCHES), "pairs": list(PAIR_NAMES),
        "modes": list(MODES), "shift_modes": list(SHIFT_MODES),
        "position_shift_offsets": list(POSITION_SHIFTS),
        "discovery": {
            "documents": list(DISCOVERY_RANGE), "halves": [[0, 250], [250, 500]],
            "instrument": discovery["instrument"], "analysis": discovery_analysis,
        },
        "validation": None if validation is None else {
            "documents": list(VALIDATION_RANGE), "halves": [[500, 750], [750, 1000]],
            "instrument": validation["instrument"], "analysis": validation_analysis,
        },
        'pred_a_exact_lawful_live_merge_instrument': pred_a,
        'pred_b_attention1_merge_removes_T_I_contrast': pred_b,
        'pred_c_progressive_T_I_merge': pred_c,
        'pred_d_T_I_specific_depth_gradient': pred_d,
        'pred_e_prospective_intervention_outcome_validation': pred_e,
        "validation_licensed_and_opened": validation_licensed,
        "strong_null": strong_null,
        "final_or_sealed_opened": False,
        "execution_price": {
            "discovery_full_model_forwards": (
                discovery["instrument"]["calls"]["normal_native_forwards"]
                + discovery["instrument"]["calls"]["normal_absent_forwards"]
                + discovery["instrument"]["calls"]["merge_forwards"]),
            "validation_full_model_forwards": 0 if validation is None else (
                validation["instrument"]["calls"]["normal_native_forwards"]
                + validation["instrument"]["calls"]["normal_absent_forwards"]
                + validation["instrument"]["calls"]["merge_forwards"]),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "new_corpus_site_graded_TI_merge_and_semantic_preservation"
            if pred_e else "attention1_exact_QK1_QK2_OV_downstream_use_or_scalar_composition"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 493,
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "validation_opened": validation_licensed,
        "strong_null": strong_null,
        "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
