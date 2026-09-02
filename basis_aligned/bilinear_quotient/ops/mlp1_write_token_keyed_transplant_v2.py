#!/usr/bin/env python3
"""Token-keyed transplant v2 -- overlap-maximized donors, four-action format.

# BQGATE: EXPERIMENT
# pred_a_exact_lawful_live_supplied_instrument
# pred_b_token_keyed_restoration_v2
# pred_c_off_key_toxicity_reproduced_v2
# pred_d_keyed_restoration_composes

Parallel-lane probe (Claude). V1 (section 2620) failed its own supply floors
(XOR-1 pairing, ~1.9 matches/doc). V2 fixes supply by measurement: greedy
best-overlap donor selection within each document half (measured 1,776/1,764
matched positions per half, min 3/doc). Same open question: is the MLP1
write-adjustment token-keyed? Adds a COMPOSE arm per the four-action receipt
format. Imports the frozen rung493 module as a hash-pinned library.
Preregistration:
polynomial_causal/MLP1_WRITE_TOKEN_KEYED_TRANSPLANT_V2_PREREGISTRATION.md
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

PREREG = POLY / "MLP1_WRITE_TOKEN_KEYED_TRANSPLANT_V2_PREREGISTRATION.md"
R493_SOURCE = ROOT / "ops/mlp0_TI_site_graded_merge_intervention_rung493.py"
R493_RESULT = ROOT / "mlp0_TI_site_graded_merge_intervention_rung493_results.json"
V1_RESULT = ROOT / "mlp1_write_token_keyed_transplant_results.json"
OUT = ROOT / "mlp1_write_token_keyed_transplant_v2_results.json"
BUNDLE = ROOT / "mlp1_write_token_keyed_transplant_v2_per_token.pt"
HASHES = {
    PREREG: "94fa5cfb4e802aeed79e101aa796aac25e50e632920fc6aca017b6413bd6e401",
    R493_SOURCE: "4f77c3898d8237373a7a35439dabc590882eda47490aaed33a797ddec2cfe08b",
    R493_RESULT: "1131a0dc61f94ca2dba92073eed1a21c2f46a46ac18004be146e15a78161339d",
    V1_RESULT: "50bb220572d22e6f36b7d9c01ead544c9b8ad6640f800c7d4408144fc197989e",
}

TI = ("T", "I")
ARMS = ("OWN_MATCH", "DONOR_MATCH", "DONOR_MISMATCH", "COMPOSE")
SAMPLE_SEED = 20260903
BATCH = r493.BATCH
DOC_RANGE = (0, 500)
HALF = 250
BATCHES = (DOC_RANGE[1] - DOC_RANGE[0]) // BATCH
EXPECTED_NATIVE = BATCHES
EXPECTED_ABSENT = BATCHES * len(TI)
EXPECTED_BOTH_ABSENT = BATCHES
EXPECTED_MERGE = BATCHES * len(TI) * 3
EXPECTED_FORWARDS = (EXPECTED_NATIVE + EXPECTED_ABSENT
                     + EXPECTED_BOTH_ABSENT + EXPECTED_MERGE)
MIN_MATCHED_PER_HALF = 1500
MIN_MATCHED_PER_DOC = 2


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or r493.sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    v1 = json.loads(V1_RESULT.read_text())
    if v1.get("pred_a_exact_lawful_live_masked_instrument") is not False \
            or v1.get("strong_null") is not True:
        raise RuntimeError("v1 receipt does not license the v2 redesign")
    rows, fit_rows, metadata = r493.validate_inputs()
    return rows, fit_rows, {
        **metadata,
        "v2_arms": list(ARMS), "mismatch_sample_seed": SAMPLE_SEED,
        "donor_rule": "greedy_max_same_position_overlap_within_half_ties_lowest",
    }


def build_donor_map(tokens_half):
    """Deterministic best-overlap donor per document within one half."""
    count = tokens_half.shape[0]
    donors = torch.zeros(count, dtype=torch.long)
    for doc in range(count):
        overlap = (tokens_half == tokens_half[doc].unsqueeze(0))[:, 1:].sum(1)
        overlap[doc] = -1
        donors[doc] = int(torch.argmax(overlap))  # argmax returns lowest tie index
    return donors


def build_masks(tokens_half, donors, generator):
    match = torch.zeros_like(tokens_half, dtype=torch.bool)
    mismatch = torch.zeros_like(match)
    for doc in range(tokens_half.shape[0]):
        m = tokens_half[doc].eq(tokens_half[donors[doc]])
        m[0] = False
        match[doc] = m
        need = int(m.sum())
        candidates = torch.nonzero(~m, as_tuple=False).reshape(-1)
        candidates = candidates[candidates >= 1]
        if need > candidates.numel():
            raise RuntimeError("mismatch sample cannot meet the match-count budget")
        order = torch.randperm(candidates.numel(), generator=generator)
        mismatch[doc, candidates[order[:need]]] = True
    return match, mismatch


@torch.no_grad()
def collect(model, rows, reference):
    ce = r493.parent.parent.base.factorial_parent._per_token_ce
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(SAMPLE_SEED)
    calls = {"native_forwards": 0, "absent_forwards": 0,
             "both_absent_forwards": 0, "merge_forwards": 0}
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
    native_ce = torch.zeros(DOC_RANGE[1], rows.shape[1] - 1, dtype=torch.float64)
    absent_ce = torch.zeros(len(TI), DOC_RANGE[1], rows.shape[1] - 1, dtype=torch.float64)
    arm_ce = torch.zeros(len(TI), len(ARMS), DOC_RANGE[1], rows.shape[1] - 1,
                         dtype=torch.float64)
    match_all = torch.zeros(DOC_RANGE[1], rows.shape[1] - 1, dtype=torch.bool)
    mismatch_all = torch.zeros_like(match_all)
    donor_all = torch.zeros(DOC_RANGE[1], dtype=torch.long)
    match_per_doc = torch.zeros(DOC_RANGE[1], dtype=torch.long)

    for half_start in range(DOC_RANGE[0], DOC_RANGE[1], HALF):
        half_stop = half_start + HALF
        tokens_half = rows[half_start:half_stop, :-1]
        donors = build_donor_map(tokens_half)
        match, mismatch = build_masks(tokens_half, donors, generator)
        donor_all[half_start:half_stop] = donors + half_start
        match_all[half_start:half_stop] = match
        mismatch_all[half_start:half_stop] = mismatch
        match_per_doc[half_start:half_stop] = match.sum(1)

        # Pass 1: baselines; cache CPU bf16 trajectories + float32 M writes.
        cache = {}
        for start in range(half_start, half_stop, BATCH):
            stop = start + BATCH
            tokens = rows[start:stop, :-1].to(device)
            targets = rows[start:stop, 1:].to(device)
            native_logits, native, _c = r493.parent._native_all(
                model, tokens, reference)
            calls["native_forwards"] += 1
            native_ce[start:stop] = ce(native_logits, targets).double().cpu()
            for name, value in native["prefix_errors"].items():
                r493._update_max(errors, f"native_prefix_{name}_relative_squared_max", value)
            r493._update_max(errors, "prefix_z_relative_squared_max",
                             native["prefix_z_relative_squared"])
            r493._update_max(errors, "S_prefix_replay_relative_squared_max",
                             native["S_prefix_replay_relative_squared"])
            r493._update_max(errors, "state_source_relative_squared_max",
                             native["state_source_relative_squared"])
            for key in ("analytical_num", "analytical_den",
                        "deployed_num", "deployed_den"):
                errors[key] += native["identity"][key]
            entry = {"native_M": native["M"].float().cpu()}
            for branch_index, branch in enumerate(TI):
                logits, capture, _a = r493.parent.parent.base._absent_forward(
                    model, tokens, native, native["branches"][branch])
                calls["absent_forwards"] += 1
                absent_ce[branch_index, start:stop] = ce(logits, targets).double().cpu()
                r493._update_max(errors, "mlp0_state_max_abs", capture["mlp0_state_error"])
                entry[branch] = {key: capture[key].cpu() for key in ("D", "A", "M")}
            both_branch = native["branches"]["T"] + native["branches"]["I"]
            logits, capture, _a = r493.parent.parent.base._absent_forward(
                model, tokens, native, both_branch)
            calls["both_absent_forwards"] += 1
            r493._update_max(errors, "mlp0_state_max_abs", capture["mlp0_state_error"])
            entry["BOTH"] = {key: capture[key].cpu() for key in ("D", "A", "M")}
            cache[start] = entry

        # Pass 2: edited forwards with cached donor adjustments.
        for start in range(half_start, half_stop, BATCH):
            stop = start + BATCH
            tokens = rows[start:stop, :-1].to(device)
            targets = rows[start:stop, 1:].to(device)
            entry = cache[start]
            match_d = match_all[start:stop].to(device)
            mismatch_d = mismatch_all[start:stop].to(device)
            native_M = entry["native_M"].to(device)
            for branch_index, branch in enumerate(TI):
                trajectory = {key: value.to(device)
                              for key, value in entry[branch].items()}
                both = {key: value.to(device) for key, value in entry["BOTH"].items()}
                base = trajectory["M"].float()
                both_base = both["M"].float()
                donor_adj = torch.stack([
                    cache[(int(donor_all[doc]) // BATCH) * BATCH]["native_M"][
                        int(donor_all[doc]) % BATCH].to(device)
                    - cache[(int(donor_all[doc]) // BATCH) * BATCH][branch]["M"][
                        int(donor_all[doc]) % BATCH].to(device).float()
                    for doc in range(start, stop)])
                edited_bank = {
                    "OWN_MATCH": (torch.where(match_d.unsqueeze(-1), native_M.float(), base),
                                  trajectory),
                    "DONOR_MATCH": (torch.where(match_d.unsqueeze(-1),
                                                base + donor_adj, base), trajectory),
                    "DONOR_MISMATCH": (torch.where(mismatch_d.unsqueeze(-1),
                                                   base + donor_adj, base), trajectory),
                    "COMPOSE": (torch.where(match_d.unsqueeze(-1),
                                            both_base + donor_adj, both_base), both),
                }
                for arm_index, arm in enumerate(ARMS):
                    edited_f, traj = edited_bank[arm]
                    reference_base = both_base if arm == "COMPOSE" else base
                    edited = edited_f.to(trajectory["M"].dtype)
                    errors["edit_rms_min"] = min(
                        errors["edit_rms_min"],
                        r493._rms(edited.float() - reference_base))
                    logits, audit, _actual = r493._merge_forward(
                        model, tokens, traj, "M_ONLY", edited)
                    calls["merge_forwards"] += 1
                    r493._update_max(errors, "edited_write_max_abs_error",
                                     audit["edited_write_max_abs_error"])
                    arm_ce[branch_index, arm_index, start:stop] = \
                        ce(logits, targets).double().cpu()
        del cache

    expected = {"native_forwards": EXPECTED_NATIVE,
                "absent_forwards": EXPECTED_ABSENT,
                "both_absent_forwards": EXPECTED_BOTH_ABSENT,
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
        "matched_positions_half0": int(match_all[:HALF].sum()),
        "matched_positions_half1": int(match_all[HALF:].sum()),
        "min_matches_per_doc": int(match_per_doc.min()),
        "documents": DOC_RANGE[1] - DOC_RANGE[0],
    }
    return {
        "native": native_ce, "absent": absent_ce, "arms": arm_ce,
        "match": match_all, "mismatch": mismatch_all, "donors": donor_all,
        "instrument": instrument,
    }


def _masked_stats(recovery, x, mask):
    r = recovery[mask].reshape(-1)
    v = x[mask].reshape(-1)
    fraction = float(torch.dot(r, v)) / max(float(torch.dot(v, v)), 1e-30)
    cosine = float(torch.dot(r, v)
                   / (torch.linalg.vector_norm(r)
                      * torch.linalg.vector_norm(v)).clamp_min(1e-30))
    return fraction, cosine, int(mask.sum())


def analyze(collected):
    native = collected["native"]
    documents = native.shape[0]
    halves = ((0, HALF), (HALF, documents))
    arm_masks = {"OWN_MATCH": "match", "DONOR_MATCH": "match",
                 "DONOR_MISMATCH": "mismatch", "COMPOSE": "match"}
    reports = {}
    b_flags, c_flags, d_flags, own_live = [], [], [], []
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
                row[arm] = {"aligned_recovered_fraction": fraction,
                            "recovery_cosine": cosine,
                            "scored_positions": count}
            own = row["OWN_MATCH"]["aligned_recovered_fraction"]
            donor = row["DONOR_MATCH"]["aligned_recovered_fraction"]
            off_key = row["DONOR_MISMATCH"]["aligned_recovered_fraction"]
            compose = row["COMPOSE"]["aligned_recovered_fraction"]
            own_live.append(bool(own > 0))
            b_flags.append(bool(
                donor >= .25 * own and donor >= off_key + .25 and donor >= 0))
            c_flags.append(bool(off_key <= 0))
            d_flags.append(bool(
                compose >= .5 * donor and (compose >= 0) == (donor >= 0)))
            reports[branch][f"half{half_index}"] = row
    pred_b = bool(all(b_flags))
    return {
        "reports": reports,
        "own_reference_live": bool(all(own_live)),
        "pred_b_flags": b_flags, "pred_c_flags": c_flags,
        "pred_d_flags": d_flags,
        "pred_b_token_keyed_restoration_v2": pred_b,
        "pred_c_off_key_toxicity_reproduced_v2": bool(all(c_flags)),
        "pred_d_keyed_restoration_composes": bool(pred_b and all(d_flags)),
        "pred_d_scored": pred_b,
    }


def _synthetic_collected():
    generator = torch.Generator().manual_seed(2)
    documents, length = 12, 32
    native = torch.rand(documents, length, dtype=torch.float64, generator=generator)
    absent = native + .5 + .1 * torch.rand(
        len(TI), documents, length, dtype=torch.float64, generator=generator)
    arms = torch.stack([
        torch.stack([absent[i] - .3, absent[i] - .2, absent[i] + .1, absent[i] - .12])
        for i in range(len(TI))])
    match = torch.rand(documents, length, generator=generator) < .3
    match[:, 0] = False
    mismatch = ~match
    mismatch[:, 0] = False
    return {"native": native, "absent": absent, "arms": arms,
            "match": match, "mismatch": mismatch}


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert EXPECTED_FORWARDS == 1250
        assert EXPECTED_MERGE == 750
        global HALF
        real_half = HALF
        HALF = 6
        try:
            analysis = analyze(_synthetic_collected())
        finally:
            HALF = real_half
        assert len(analysis["pred_b_flags"]) == 4
        assert len(analysis["pred_d_flags"]) == 4
        tokens = torch.tensor([[9, 1, 2, 3, 4, 5], [9, 1, 5, 6, 7, 8], [9, 1, 2, 8, 9, 9], [7, 6, 7, 6, 1, 2]])
        donors = build_donor_map(tokens)
        assert donors.tolist() == [2, 0, 0, 1]
        match, mismatch = build_masks(
            tokens, donors, torch.Generator().manual_seed(SAMPLE_SEED))
        assert match.tolist()[0] == [False, True, True, False, False, False]
        assert int(match.sum()) == int(mismatch.sum())
        assert not (match & mismatch).any()
        for path, expected in HASHES.items():
            if not path.is_file() or r493.sha256(path) != expected:
                raise RuntimeError(f"frozen hash mismatch: {path}")
        print(json.dumps({
            "status": "dry_run_passed", "rung": "mlp1_write_token_keyed_transplant_v2",
            "model_loaded": False, "outcomes_opened": False,
            "validation_or_sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "synthetic_analysis_donor_and_mask_logic_exercised": True,
        }, indent=2, sort_keys=True))
        return
    rows, fit_rows, metadata = validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("v2 output namespace already exists")
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
        and instrument["matched_positions_half0"] >= MIN_MATCHED_PER_HALF
        and instrument["matched_positions_half1"] >= MIN_MATCHED_PER_HALF
        and instrument["min_matches_per_doc"] >= MIN_MATCHED_PER_DOC
        and analysis["own_reference_live"])
    pred_b = analysis["pred_b_token_keyed_restoration_v2"]
    pred_c = analysis["pred_c_off_key_toxicity_reproduced_v2"]
    pred_d = analysis["pred_d_keyed_restoration_composes"]
    strong_null = bool(not pred_a or not pred_b or not pred_c)
    torch.save({
        "schema": "mlp1_write_token_keyed_transplant_v2",
        "native": collected["native"].float(),
        "absent": collected["absent"].float(),
        "arms": collected["arms"].float(),
        "match": collected["match"], "mismatch": collected["mismatch"],
        "donors": collected["donors"],
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": "mlp1_write_token_keyed_transplant_v2",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "chokepoint_content_mechanism_test_v2",
        "source_hashes": {str(path): r493.sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(TI), "arms": list(ARMS),
        "documents": list(DOC_RANGE), "halves": [[0, HALF], [HALF, DOC_RANGE[1]]],
        "analysis": analysis,
        "instrument": instrument,
        "bundle": {"path": str(BUNDLE), "sha256": r493.sha256(BUNDLE),
                   "raw_tokens_logits_or_hidden_states_included": False},
        'pred_a_exact_lawful_live_supplied_instrument': pred_a,
        'pred_b_token_keyed_restoration_v2': pred_b,
        'pred_c_off_key_toxicity_reproduced_v2': pred_c,
        'pred_d_keyed_restoration_composes': pred_d,
        "pred_d_scored": analysis["pred_d_scored"],
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
