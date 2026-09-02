#!/usr/bin/env python3
"""Token-keyed transplant test -- is the MLP1 write-adjustment keyed by token identity?

# BQGATE: EXPERIMENT
# pred_a_exact_lawful_live_masked_instrument
# pred_b_token_keyed_restoration
# pred_c_off_key_toxicity_reproduced

Parallel-lane probe (Claude). Mechanism follow-up to the portability probe
(section 2618): whole-write donor transplants were actively harmful; here the
transplant is restricted to positions where donor and recipient share the SAME
token at the SAME position (T is the token-only branch, so its write-adjustment
may be token-keyed). Control: same edit budget on sampled mismatched positions.
Imports the frozen rung493 module as a hash-pinned library. Preregistration:
polynomial_causal/MLP1_WRITE_TOKEN_KEYED_TRANSPLANT_PREREGISTRATION.md
"""

from __future__ import annotations

import json
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

PREREG = POLY / "MLP1_WRITE_TOKEN_KEYED_TRANSPLANT_PREREGISTRATION.md"
R493_SOURCE = ROOT / "ops/mlp0_TI_site_graded_merge_intervention_rung493.py"
R493_RESULT = ROOT / "mlp0_TI_site_graded_merge_intervention_rung493_results.json"
PROBE_RESULT = ROOT / "mlp1_write_interface_portability_probe_results.json"
OUT = ROOT / "mlp1_write_token_keyed_transplant_results.json"
BUNDLE = ROOT / "mlp1_write_token_keyed_transplant_per_token.pt"
HASHES = {
    PREREG: "4d8481b70d593ca798a5b03d9a8aa86be567881daf503c67c717bd3161d6d572",
    R493_SOURCE: "4f77c3898d8237373a7a35439dabc590882eda47490aaed33a797ddec2cfe08b",
    R493_RESULT: "1131a0dc61f94ca2dba92073eed1a21c2f46a46ac18004be146e15a78161339d",
    PROBE_RESULT: "1c50849582f09858e258fa0f5d276439d8b83442888fb7a8c681b3967318954f",
}

TI = ("T", "I")
ARMS = ("OWN_MATCH", "DONOR_MATCH", "DONOR_MISMATCH")
DONOR_PERM = (1, 0, 3, 2)
SAMPLE_SEED = 20260902
BATCH = r493.BATCH
DOC_RANGE = (0, 500)
HALF = 250
BATCHES = (DOC_RANGE[1] - DOC_RANGE[0]) // BATCH
EXPECTED_NATIVE = BATCHES
EXPECTED_ABSENT = BATCHES * len(TI)
EXPECTED_MERGE = BATCHES * len(TI) * len(ARMS)
EXPECTED_FORWARDS = EXPECTED_NATIVE + EXPECTED_ABSENT + EXPECTED_MERGE
MIN_MATCHED_PER_CELL = 500


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or r493.sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    probe = json.loads(PROBE_RESULT.read_text())
    required = {
        "pred_a_exact_lawful_live_transplant_instrument": True,
        "pred_b_cross_document_write_portability": False,
        "pred_c_branch_specific_transplant": False,
        "strong_null": True,
    }
    if any(probe.get(key) != value for key, value in required.items()):
        raise RuntimeError("portability probe receipt does not license this test")
    rows, fit_rows, metadata = r493.validate_inputs()
    return rows, fit_rows, {
        **metadata,
        "probe_branches": list(TI), "probe_arms": list(ARMS),
        "donor_permutation": list(DONOR_PERM),
        "mismatch_sample_seed": SAMPLE_SEED,
    }


def build_masks(tokens, generator):
    """Match mask (same token, same position, pos>=1) and same-count mismatch sample."""
    perm = torch.tensor(DONOR_PERM, device=tokens.device)
    match = tokens.eq(tokens[perm])
    match[:, 0] = False
    mismatch = torch.zeros_like(match)
    for doc in range(tokens.shape[0]):
        count = int(match[doc].sum())
        candidates = torch.nonzero(~match[doc], as_tuple=False).reshape(-1)
        candidates = candidates[candidates >= 1]
        if count > candidates.numel():
            raise RuntimeError("mismatch sample cannot meet the match-count budget")
        order = torch.randperm(candidates.numel(), generator=generator)
        mismatch[doc, candidates[order[:count]].to(mismatch.device)] = True
    return match, mismatch


@torch.no_grad()
def collect(model, rows, reference):
    ce = r493.parent.parent.base.factorial_parent._per_token_ce
    start_doc, stop_doc = DOC_RANGE
    generator = torch.Generator().manual_seed(SAMPLE_SEED)
    native_ce, absent_ce, arm_ce = [], [], []
    match_masks, mismatch_masks = [], []
    calls = {"native_forwards": 0, "absent_forwards": 0, "merge_forwards": 0}
    errors = {
        "native_prefix_D_relative_squared_max": 0.0,
        "native_prefix_A_relative_squared_max": 0.0,
        "native_prefix_M_relative_squared_max": 0.0,
        "prefix_z_relative_squared_max": 0.0,
        "mlp0_state_max_abs": 0.0,
        "S_prefix_replay_relative_squared_max": 0.0,
        "state_source_relative_squared_max": 0.0,
        "edited_write_max_abs_error": 0.0,
        "edit_rms_min": float("inf"),
        "analytical_num": 0.0, "analytical_den": 0.0,
        "deployed_num": 0.0, "deployed_den": 0.0,
    }
    device = next(model.parameters()).device
    perm = torch.tensor(DONOR_PERM)
    for start in range(start_doc, stop_doc, BATCH):
        stop = start + BATCH
        tokens = rows[start:stop, :-1].to(device)
        targets = rows[start:stop, 1:].to(device)
        match, mismatch = build_masks(tokens.cpu(), generator)
        match_masks.append(match)
        mismatch_masks.append(mismatch)
        match_d = match.to(device)
        mismatch_d = mismatch.to(device)
        native_logits, native, _calls = r493.parent._native_all(
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
            adjustments[branch] = native["M"].float() - capture["M"].float()
        absent_ce.append(torch.stack(batch_absent))

        batch_arms = []
        for branch in TI:
            base = absent[branch]["M"].float()
            donor_adj = adjustments[branch][perm]
            native_write = native["M"].float()
            edited_bank = {
                "OWN_MATCH": torch.where(match_d.unsqueeze(-1), native_write, base),
                "DONOR_MATCH": torch.where(
                    match_d.unsqueeze(-1), base + donor_adj, base),
                "DONOR_MISMATCH": torch.where(
                    mismatch_d.unsqueeze(-1), base + donor_adj, base),
            }
            arm_rows = []
            for arm in ARMS:
                edited = edited_bank[arm].to(native["M"].dtype)
                errors["edit_rms_min"] = min(
                    errors["edit_rms_min"], r493._rms(edited.float() - base))
                logits, audit, _actual = r493._merge_forward(
                    model, tokens, absent[branch], "M_ONLY", edited)
                calls["merge_forwards"] += 1
                r493._update_max(errors, "edited_write_max_abs_error",
                                 audit["edited_write_max_abs_error"])
                arm_rows.append(ce(logits, targets).double().cpu())
            batch_arms.append(torch.stack(arm_rows))
        arm_ce.append(torch.stack(batch_arms))

    expected = {"native_forwards": EXPECTED_NATIVE,
                "absent_forwards": EXPECTED_ABSENT,
                "merge_forwards": EXPECTED_MERGE}
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
        "match": torch.cat(match_masks, dim=0),
        "mismatch": torch.cat(mismatch_masks, dim=0),
        "instrument": instrument,
    }


def _masked_stats(recovery, x, mask):
    r = recovery[mask].reshape(-1)
    v = x[mask].reshape(-1)
    xx = float(torch.dot(v, v))
    fraction = float(torch.dot(r, v)) / max(xx, 1e-30)
    cosine = float(torch.dot(r, v)
                   / (torch.linalg.vector_norm(r)
                      * torch.linalg.vector_norm(v)).clamp_min(1e-30))
    return fraction, cosine, int(mask.sum())


def analyze(collected):
    native = collected["native"]
    documents = native.shape[0]
    halves = ((0, HALF), (HALF, documents))
    arm_masks = {"OWN_MATCH": "match", "DONOR_MATCH": "match",
                 "DONOR_MISMATCH": "mismatch"}
    reports = {}
    b_flags, c_flags, own_live, counts_ok = [], [], [], []
    for branch_index, branch in enumerate(TI):
        reports[branch] = {}
        absent = collected["absent"][branch_index]
        for half_index, (lo, hi) in enumerate(halves):
            x = absent[lo:hi] - native[lo:hi]
            row = {}
            for arm_index, arm in enumerate(ARMS):
                mask = collected[arm_masks[arm]][lo:hi]
                recovery = absent[lo:hi] - collected["arms"][branch_index, arm_index, lo:hi]
                fraction, cosine, count = _masked_stats(recovery, x, mask)
                row[arm] = {
                    "aligned_recovered_fraction": fraction,
                    "recovery_cosine": cosine,
                    "scored_positions": count,
                }
            own = row["OWN_MATCH"]["aligned_recovered_fraction"]
            donor = row["DONOR_MATCH"]["aligned_recovered_fraction"]
            off_key = row["DONOR_MISMATCH"]["aligned_recovered_fraction"]
            own_live.append(bool(own > 0))
            counts_ok.append(bool(
                row["DONOR_MATCH"]["scored_positions"] >= MIN_MATCHED_PER_CELL))
            b_flags.append(bool(
                donor >= .25 * own and donor >= off_key + .25 and donor >= 0))
            c_flags.append(bool(off_key <= 0))
            reports[branch][f"half{half_index}"] = row
    return {
        "reports": reports,
        "own_reference_live": bool(all(own_live)),
        "matched_counts_ok": bool(all(counts_ok)),
        "pred_b_flags": b_flags, "pred_c_flags": c_flags,
        "pred_b_token_keyed_restoration": bool(all(b_flags)),
        "pred_c_off_key_toxicity_reproduced": bool(all(c_flags)),
    }


def _synthetic_collected():
    generator = torch.Generator().manual_seed(1)
    documents, length = 12, 32
    native = torch.rand(documents, length, dtype=torch.float64, generator=generator)
    absent = native + .5 + .1 * torch.rand(
        len(TI), documents, length, dtype=torch.float64, generator=generator)
    arms = torch.stack([
        torch.stack([absent[index] - .3, absent[index] - .2, absent[index] + .1])
        for index in range(len(TI))])
    match = torch.rand(documents, length, generator=generator) < .3
    match[:, 0] = False
    mismatch = ~match
    mismatch[:, 0] = False
    return {"native": native, "absent": absent, "arms": arms,
            "match": match, "mismatch": mismatch}


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert EXPECTED_FORWARDS == 1125
        assert EXPECTED_MERGE == 750
        global HALF, MIN_MATCHED_PER_CELL
        real_half, real_min = HALF, MIN_MATCHED_PER_CELL
        HALF, MIN_MATCHED_PER_CELL = 6, 1
        try:
            analysis = analyze(_synthetic_collected())
        finally:
            HALF, MIN_MATCHED_PER_CELL = real_half, real_min
        assert len(analysis["pred_b_flags"]) == 4
        assert len(analysis["pred_c_flags"]) == 4
        tokens = torch.tensor([[9, 1, 2, 3], [9, 1, 5, 6], [7, 7, 7, 7], [7, 6, 7, 6]])
        match, mismatch = build_masks(
            tokens, torch.Generator().manual_seed(SAMPLE_SEED))
        assert match.tolist()[0] == [False, True, False, False]
        assert int(match.sum()) == int(mismatch.sum())
        assert not (match & mismatch).any()
        for path, expected in HASHES.items():
            if not path.is_file() or r493.sha256(path) != expected:
                raise RuntimeError(f"frozen hash mismatch: {path}")
        print(json.dumps({
            "status": "dry_run_passed", "rung": "mlp1_write_token_keyed_transplant",
            "model_loaded": False, "outcomes_opened": False,
            "validation_or_sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "synthetic_analysis_and_mask_logic_exercised": True,
        }, indent=2, sort_keys=True))
        return
    rows, fit_rows, metadata = validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("token-keyed transplant output namespace already exists")
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
        and instrument["edit_rms_min"] > 0
        and analysis["own_reference_live"]
        and analysis["matched_counts_ok"])
    pred_b = analysis["pred_b_token_keyed_restoration"]
    pred_c = analysis["pred_c_off_key_toxicity_reproduced"]
    strong_null = bool(not pred_a or not pred_b or not pred_c)
    torch.save({
        "schema": "mlp1_write_token_keyed_transplant_v1",
        "native": collected["native"].float(),
        "absent": collected["absent"].float(),
        "arms": collected["arms"].float(),
        "match": collected["match"], "mismatch": collected["mismatch"],
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": "mlp1_write_token_keyed_transplant",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "chokepoint_content_mechanism_test",
        "source_hashes": {str(path): r493.sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(TI), "arms": list(ARMS),
        "donor_permutation": list(DONOR_PERM),
        "mismatch_sample_seed": SAMPLE_SEED,
        "documents": list(DOC_RANGE), "halves": [[0, HALF], [HALF, DOC_RANGE[1]]],
        "analysis": analysis,
        "instrument": instrument,
        "bundle": {"path": str(BUNDLE), "sha256": r493.sha256(BUNDLE),
                   "raw_tokens_logits_or_hidden_states_included": False},
        'pred_a_exact_lawful_live_masked_instrument': pred_a,
        'pred_b_token_keyed_restoration': pred_b,
        'pred_c_off_key_toxicity_reproduced': pred_c,
        "validation_or_sealed_opened": False,
        "strong_null": strong_null,
        "execution_price": {
            "full_model_forwards": EXPECTED_FORWARDS,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "register_cross_corpus_token_matched_validation"
            if not strong_null else
            "write_is_context_bound_below_token_grain_route_to_content_decomposition"),
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
