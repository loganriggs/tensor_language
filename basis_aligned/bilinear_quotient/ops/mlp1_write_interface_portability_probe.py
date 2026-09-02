#!/usr/bin/env python3
"""MLP1-write interface portability probe -- is the chokepoint a portable interface?

# BQGATE: EXPERIMENT
# pred_a_exact_lawful_live_transplant_instrument
# pred_b_cross_document_write_portability
# pred_c_branch_specific_transplant

Parallel-lane probe (Claude). Transplants a donor document's branch MLP1
write-adjustment into a recipient document (fixed XOR-1 pairing) and asks whether
it recovers the recipient branch's effect the way the recipient's own adjustment
does, against 16 position-shift controls and a crossed-branch donor control.
Imports the frozen rung493 module as a hash-pinned library; modifies no
registered file. Preregistration:
polynomial_causal/MLP1_WRITE_INTERFACE_PORTABILITY_PREREGISTRATION.md
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import mlp0_TI_site_graded_merge_intervention_rung493 as r493

PREREG = POLY / "MLP1_WRITE_INTERFACE_PORTABILITY_PREREGISTRATION.md"
R493_SOURCE = ROOT / "ops/mlp0_TI_site_graded_merge_intervention_rung493.py"
R493_RESULT = ROOT / "mlp0_TI_site_graded_merge_intervention_rung493_results.json"
OUT = ROOT / "mlp1_write_interface_portability_probe_results.json"
BUNDLE = ROOT / "mlp1_write_interface_portability_probe_per_token.pt"
HASHES = {
    PREREG: "c6283c3653224442342208a6afb57756b72e0ee4714ac68756e8b5964dbe0d09",
    R493_SOURCE: "4f77c3898d8237373a7a35439dabc590882eda47490aaed33a797ddec2cfe08b",
    R493_RESULT: "1131a0dc61f94ca2dba92073eed1a21c2f46a46ac18004be146e15a78161339d",
}

TI = ("T", "I")
ARMS = ("OWN", "DONOR", "CROSSED")
DONOR_PERM = (1, 0, 3, 2)
POSITION_SHIFTS = r493.POSITION_SHIFTS
BATCH = r493.BATCH
DOC_RANGE = (0, 500)
HALF = 250
BATCHES = (DOC_RANGE[1] - DOC_RANGE[0]) // BATCH
FORWARDS_PER_BATCH = 1 + len(TI) * (1 + len(ARMS) + len(POSITION_SHIFTS))
EXPECTED_NATIVE = BATCHES
EXPECTED_ABSENT = BATCHES * len(TI)
EXPECTED_MERGE = BATCHES * len(TI) * (len(ARMS) + len(POSITION_SHIFTS))
EXPECTED_FORWARDS = EXPECTED_NATIVE + EXPECTED_ABSENT + EXPECTED_MERGE


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or r493.sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(R493_RESULT.read_text())
    required = {
        "rung": 493,
        "pred_a_exact_lawful_live_merge_instrument": True,
        "pred_b_attention1_merge_removes_T_I_contrast": False,
        "pred_c_progressive_T_I_merge": False,
        "pred_d_T_I_specific_depth_gradient": False,
        "pred_e_prospective_intervention_outcome_validation": False,
        "validation_licensed_and_opened": False,
        "strong_null": True,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise RuntimeError("rung493 receipt does not license the portability probe")
    rows, fit_rows, metadata = r493.validate_inputs()
    return rows, fit_rows, {
        **metadata,
        "probe_branches": list(TI), "probe_arms": list(ARMS),
        "donor_permutation": list(DONOR_PERM),
        "position_shift_offsets": list(POSITION_SHIFTS),
    }


@torch.no_grad()
def collect(model, rows, reference):
    ce = r493.parent.parent.base.factorial_parent._per_token_ce
    start_doc, stop_doc = DOC_RANGE
    native_ce, absent_ce, arm_ce, shift_ce = [], [], [], []
    calls = {"native_forwards": 0, "absent_forwards": 0, "merge_forwards": 0,
             "merge_edited_site": 0}
    errors = {
        "native_prefix_D_relative_squared_max": 0.0,
        "native_prefix_A_relative_squared_max": 0.0,
        "native_prefix_M_relative_squared_max": 0.0,
        "prefix_z_relative_squared_max": 0.0,
        "mlp0_state_max_abs": 0.0,
        "S_prefix_replay_relative_squared_max": 0.0,
        "state_source_relative_squared_max": 0.0,
        "edited_write_max_abs_error": 0.0,
        "adjustment_rms_min": float("inf"),
        "donor_minus_own_adjustment_rms_min": float("inf"),
        "analytical_num": 0.0, "analytical_den": 0.0,
        "deployed_num": 0.0, "deployed_den": 0.0,
    }
    device = next(model.parameters()).device
    perm = torch.tensor(DONOR_PERM)
    for start in range(start_doc, stop_doc, BATCH):
        stop = start + BATCH
        if stop > stop_doc:
            raise RuntimeError("document count must be a multiple of the batch size")
        tokens = rows[start:stop, :-1].to(device)
        targets = rows[start:stop, 1:].to(device)
        native_logits, native, _native_calls = r493.parent._native_all(
            model, tokens, reference)
        calls["native_forwards"] += 1
        native_ce.append(ce(native_logits, targets).double().cpu())
        for name, value in native["prefix_errors"].items():
            r493._update_max(errors, f"native_prefix_{name}_relative_squared_max", value)
        r493._update_max(errors, "prefix_z_relative_squared_max",
                         native["prefix_z_relative_squared"])
        r493._update_max(errors, "S_prefix_replay_relative_squared_max",
                         native["S_prefix_replay_relative_squared"])
        r493._update_max(errors, "state_source_relative_squared_max",
                         native["state_source_relative_squared"])
        for key in ("analytical_num", "analytical_den", "deployed_num", "deployed_den"):
            errors[key] += native["identity"][key]

        absent, adjustments = {}, {}
        batch_absent = []
        for branch in TI:
            logits, capture, _audit = r493.parent.parent.base._absent_forward(
                model, tokens, native, native["branches"][branch])
            calls["absent_forwards"] += 1
            absent[branch] = capture
            batch_absent.append(ce(logits, targets).double().cpu())
            r493._update_max(errors, "mlp0_state_max_abs", capture["mlp0_state_error"])
            adjustment = native["M"].float() - capture["M"].float()
            adjustments[branch] = adjustment
            errors["adjustment_rms_min"] = min(
                errors["adjustment_rms_min"], r493._rms(adjustment))
        absent_ce.append(torch.stack(batch_absent))

        batch_arms, batch_shifts = [], []
        for branch in TI:
            other = TI[1 - TI.index(branch)]
            own_adj = adjustments[branch]
            donor_adj = adjustments[branch][perm]
            crossed_adj = adjustments[other][perm]
            errors["donor_minus_own_adjustment_rms_min"] = min(
                errors["donor_minus_own_adjustment_rms_min"],
                r493._rms(donor_adj - own_adj))
            base_write = absent[branch]["M"].float()
            edited_writes = {
                "OWN": native["M"],
                "DONOR": (base_write + donor_adj).to(native["M"].dtype),
                "CROSSED": (base_write + crossed_adj).to(native["M"].dtype),
            }
            arm_rows = []
            for arm in ARMS:
                logits, audit, _actual = r493._merge_forward(
                    model, tokens, absent[branch], "M_ONLY", edited_writes[arm])
                calls["merge_forwards"] += 1
                calls["merge_edited_site"] += audit["edited_site"]
                r493._update_max(errors, "edited_write_max_abs_error",
                                 audit["edited_write_max_abs_error"])
                arm_rows.append(ce(logits, targets).double().cpu())
            batch_arms.append(torch.stack(arm_rows))
            shift_rows = []
            for shift in POSITION_SHIFTS:
                edited = (base_write + torch.roll(own_adj, shift, dims=1)) \
                    .to(native["M"].dtype)
                logits, audit, _actual = r493._merge_forward(
                    model, tokens, absent[branch], "M_ONLY", edited)
                calls["merge_forwards"] += 1
                calls["merge_edited_site"] += audit["edited_site"]
                r493._update_max(errors, "edited_write_max_abs_error",
                                 audit["edited_write_max_abs_error"])
                shift_rows.append(ce(logits, targets).double().cpu())
            batch_shifts.append(torch.stack(shift_rows))
        arm_ce.append(torch.stack(batch_arms))
        shift_ce.append(torch.stack(batch_shifts))

    expected = {
        "native_forwards": EXPECTED_NATIVE, "absent_forwards": EXPECTED_ABSENT,
        "merge_forwards": EXPECTED_MERGE, "merge_edited_site": EXPECTED_MERGE,
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
        "native": torch.cat(native_ce, dim=0),
        "absent": torch.cat(absent_ce, dim=1),
        "arms": torch.cat(arm_ce, dim=2),
        "shifts": torch.cat(shift_ce, dim=2),
        "instrument": instrument,
    }


def _aligned_stats(recovery, x):
    xx = float(torch.dot(x, x))
    fraction = float(torch.dot(recovery, x)) / max(xx, 1e-30)
    cosine = float(torch.dot(recovery, x)
                   / (torch.linalg.vector_norm(recovery)
                      * torch.linalg.vector_norm(x)).clamp_min(1e-30))
    return fraction, cosine


def analyze(collected):
    native = collected["native"]
    documents = native.shape[0]
    halves = ((0, HALF), (HALF, documents))
    reports = {}
    b_flags, c_flags, own_live = [], [], []
    for branch_index, branch in enumerate(TI):
        reports[branch] = {}
        absent = collected["absent"][branch_index]
        for half_index, (lo, hi) in enumerate(halves):
            x = (absent[lo:hi] - native[lo:hi]).reshape(-1)
            row = {}
            for arm_index, arm in enumerate(ARMS):
                arm_ce = collected["arms"][branch_index, arm_index, lo:hi]
                recovery = (absent[lo:hi] - arm_ce).reshape(-1)
                fraction, cosine = _aligned_stats(recovery, x)
                row[arm] = {
                    "aligned_recovered_fraction": fraction,
                    "recovery_cosine": cosine,
                    "recovery_rms_over_effect_rms": float(
                        recovery.square().mean().sqrt()
                        / x.square().mean().sqrt().clamp_min(1e-30)),
                }
            shift_fractions = []
            for shift_index in range(len(POSITION_SHIFTS)):
                shift = collected["shifts"][branch_index, shift_index, lo:hi]
                recovery = (absent[lo:hi] - shift).reshape(-1)
                fraction, _cosine = _aligned_stats(recovery, x)
                shift_fractions.append(fraction)
            q95 = float(np.quantile(shift_fractions, .95))
            row["shift_fractions"] = shift_fractions
            row["shift_fraction_q95"] = q95
            own = row["OWN"]["aligned_recovered_fraction"]
            donor = row["DONOR"]["aligned_recovered_fraction"]
            crossed = row["CROSSED"]["aligned_recovered_fraction"]
            own_live.append(bool(own > 0))
            b_flags.append(bool(
                donor >= .25 * own
                and donor >= q95 + .05
                and row["DONOR"]["recovery_cosine"] >= .30))
            c_flags.append(bool(donor >= crossed + .05))
            reports[branch][f"half{half_index}"] = row
    return {
        "reports": reports,
        "own_reference_live": bool(all(own_live)),
        "pred_b_flags": b_flags, "pred_c_flags": c_flags,
        "pred_b_cross_document_write_portability": bool(all(b_flags)),
        "pred_c_branch_specific_transplant": bool(all(c_flags)),
    }


def _synthetic_collected():
    generator = torch.Generator().manual_seed(0)
    documents, length = 12, 16
    native = torch.rand(documents, length, dtype=torch.float64, generator=generator)
    absent = native + .5 + .1 * torch.rand(
        len(TI), documents, length, dtype=torch.float64, generator=generator)
    arms = torch.stack([
        torch.stack([native.expand(documents, length) + .1,
                     absent[index] - .2, absent[index] - .1])
        for index in range(len(TI))])
    shifts = absent.unsqueeze(1).repeat(1, len(POSITION_SHIFTS), 1, 1) \
        - .05 * torch.rand(len(TI), len(POSITION_SHIFTS), documents, length,
                           dtype=torch.float64, generator=generator)
    return {"native": native, "absent": absent, "arms": arms, "shifts": shifts}


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert FORWARDS_PER_BATCH == 41
        assert EXPECTED_FORWARDS == 5125
        assert EXPECTED_MERGE == 4750
        assert len(POSITION_SHIFTS) == 16
        global HALF
        real_half = HALF
        HALF = 6
        try:
            analysis = analyze(_synthetic_collected())
        finally:
            HALF = real_half
        assert len(analysis["pred_b_flags"]) == 4
        assert len(analysis["pred_c_flags"]) == 4
        for path, expected in HASHES.items():
            if not path.is_file() or r493.sha256(path) != expected:
                raise RuntimeError(f"frozen hash mismatch: {path}")
        print(json.dumps({
            "status": "dry_run_passed", "rung": "mlp1_write_interface_portability",
            "model_loaded": False, "outcomes_opened": False,
            "validation_or_sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "synthetic_analysis_exercised": True,
        }, indent=2, sort_keys=True))
        return
    rows, fit_rows, metadata = validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("portability probe output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = r493.parent.parent.base.component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    collected = collect(model, rows, reference)
    analysis = analyze(collected)
    instrument = collected["instrument"]
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and instrument["calls_exact"]
        and instrument["native_prefix_D_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_A_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_M_relative_squared_max"] <= 1e-12
        and instrument["prefix_z_relative_squared_max"] <= 1e-12
        and instrument["mlp0_state_max_abs"] == 0.0
        and instrument["S_prefix_replay_relative_squared_max"] <= 1e-12
        and instrument["state_source_relative_squared_max"] <= 1e-12
        and instrument["edited_write_max_abs_error"] == 0.0
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and instrument["adjustment_rms_min"] >= 1e-4
        and instrument["donor_minus_own_adjustment_rms_min"] >= 1e-4
        and analysis["own_reference_live"])
    pred_b = analysis["pred_b_cross_document_write_portability"]
    pred_c = analysis["pred_c_branch_specific_transplant"]
    strong_null = bool(not pred_a or not pred_b or not pred_c)
    torch.save({
        "schema": "mlp1_write_interface_portability_v1",
        "native": collected["native"].float(),
        "absent": collected["absent"].float(),
        "arms": collected["arms"].float(),
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": "mlp1_write_interface_portability",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "chokepoint_interface_portability_test",
        "source_hashes": {str(path): r493.sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(TI), "arms": list(ARMS),
        "donor_permutation": list(DONOR_PERM),
        "position_shift_offsets": list(POSITION_SHIFTS),
        "documents": list(DOC_RANGE), "halves": [[0, HALF], [HALF, DOC_RANGE[1]]],
        "analysis": analysis,
        "instrument": instrument,
        "bundle": {"path": str(BUNDLE), "sha256": r493.sha256(BUNDLE),
                   "raw_tokens_logits_or_hidden_states_included": False},
        'pred_a_exact_lawful_live_transplant_instrument': pred_a,
        'pred_b_cross_document_write_portability': pred_b,
        'pred_c_branch_specific_transplant': pred_c,
        "validation_or_sealed_opened": False,
        "strong_null": strong_null,
        "execution_price": {
            "full_model_forwards": EXPECTED_FORWARDS,
            "measured_forwards": sum(instrument["calls"].values())
                - instrument["calls"]["merge_edited_site"],
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "register_validation_500_1000_and_unrelated_circuit_preservation"
            if not strong_null else
            "chokepoint_is_a_site_not_an_interface_close_interface_language"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"], "rung": result["rung"],
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "strong_null": strong_null,
        "instrument": {key: value for key, value in instrument.items()
                       if key != "calls"},
        "analysis": {key: value for key, value in analysis.items()
                     if key != "reports"},
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
