#!/usr/bin/env python3
"""Locality-density envelope extension: six points, determinism-anchored.

# BQGATE: EXPERIMENT
# pred_a_exact_lawful_live_envelope_instrument
# pred_b_six_point_strict_monotone_with_span
# pred_c_determinism_anchor_and_interleaving

Parallel-lane probe (Claude). Extends the section-2636 locality law with
densities 1/128 and 1/2, re-measuring all four prior points in-run with the
SAME seed and nested-mask construction, so the common densities double as a
bit-level determinism anchor against the hash-pinned v1b receipt. Imports the
frozen rung493 module hash-pinned. Preregistration:
polynomial_causal/MLP1_WRITE_LOCALITY_DENSITY_ENVELOPE_EXTENSION_PREREGISTRATION.md
"""

from __future__ import annotations

import json
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
import mlp0_TI_site_graded_merge_intervention_rung493 as r493

PREREG = POLY / "MLP1_WRITE_LOCALITY_DENSITY_ENVELOPE_EXTENSION_PREREGISTRATION.md"
R493_SOURCE = ROOT / "ops/mlp0_TI_site_graded_merge_intervention_rung493.py"
R493_RESULT = ROOT / "mlp0_TI_site_graded_merge_intervention_rung493_results.json"
V1B_RESULT = ROOT / "mlp1_write_locality_density_profile_v1b_results.json"
OUT = ROOT / "mlp1_write_locality_density_envelope_ext_results.json"
BUNDLE = ROOT / "mlp1_write_locality_density_envelope_ext_per_token.pt"
HASHES = {
    PREREG: "62075ff8ebbd3b9caf47ca8a6d60859a8330941e6615e1ea286cf68e859de593",
    R493_SOURCE: "4f77c3898d8237373a7a35439dabc590882eda47490aaed33a797ddec2cfe08b",
    R493_RESULT: "1131a0dc61f94ca2dba92073eed1a21c2f46a46ac18004be146e15a78161339d",
    V1B_RESULT: "301b9f1d6c247ab04ed48e78ca05c0d414289fcc977d6d659d1a0f011bc23de8",
}

TI = ("T", "I")
DENSITIES = (1 / 128, 1 / 64, 1 / 16, 1 / 4, 1 / 2, 1.0)
DENSITY_KEYS = ("d128", "d64", "d16", "d4", "d2", "d1")
COMMON = {"d64": 0, "d16": 1, "d4": 2, "d1": 3}  # index into v1b profile order
MASK_SEED = 20260904
BATCH = r493.BATCH
DOC_RANGE = (0, 500)
HALF = 250
BATCHES = (DOC_RANGE[1] - DOC_RANGE[0]) // BATCH
EXPECTED_FORWARDS = BATCHES * (1 + len(TI) + len(TI) * len(DENSITIES))
# Derived floors (prereg): E[scored/half @1/128] = 250*255/128 ~= 498, -3sigma
# (sigma ~= sqrt(63750*(1/128)) ~= 22) -> 420. E[non-empty docs/half] =
# 250*(1-(1-1/128)^255) ~= 216, -3sigma (sigma ~= 5.8) -> 195.
MIN_SCORED_PER_CELL_LOWEST = 420
MIN_NONEMPTY_DOCS_PER_HALF_LOWEST = 195
DETERMINISM_TOL = 1e-9


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or r493.sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    v1b = json.loads(V1B_RESULT.read_text())
    required = {
        "pred_a_exact_lawful_live_density_instrument": True,
        "pred_b_profile_rises_monotonically_and_materially": True,
        "pred_c_nonlocality_material_and_full_anchor_reproduces": True,
        "strong_null": False,
    }
    if any(v1b.get(key) != value for key, value in required.items()):
        raise RuntimeError("v1b receipt does not license the envelope extension")
    anchors = {
        branch: {half: v1b["analysis"]["reports"][branch][half]["profile"]
                 for half in ("half0", "half1")}
        for branch in TI
    }
    rows, fit_rows, metadata = r493.validate_inputs()
    return rows, fit_rows, anchors, {
        **metadata, "densities": list(DENSITIES), "mask_seed": MASK_SEED,
    }


def build_nested_masks(doc_count, length, generator):
    uniform = torch.rand(doc_count, length, generator=generator)
    uniform[:, 0] = 2.0
    return {key: uniform < density
            for key, density in zip(DENSITY_KEYS, DENSITIES)}


@torch.no_grad()
def collect(model, rows, reference):
    ce = r493.parent.parent.base.factorial_parent._per_token_ce
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(MASK_SEED)
    docs = DOC_RANGE[1] - DOC_RANGE[0]
    length = rows.shape[1] - 1
    masks_all = build_nested_masks(docs, length, generator)
    nesting_ok = all(
        bool((~masks_all[DENSITY_KEYS[i]] | masks_all[DENSITY_KEYS[i + 1]]).all())
        for i in range(len(DENSITY_KEYS) - 1))
    lowest = masks_all["d128"]
    nonempty_per_half = min(
        int((lowest[:HALF].sum(1) > 0).sum()),
        int((lowest[HALF:].sum(1) > 0).sum()))
    native_ce = torch.zeros(docs, length, dtype=torch.float64)
    absent_ce = torch.zeros(len(TI), docs, length, dtype=torch.float64)
    arm_ce = torch.zeros(len(TI), len(DENSITIES), docs, length, dtype=torch.float64)
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
    for start in range(DOC_RANGE[0], DOC_RANGE[1], BATCH):
        stop = start + BATCH
        tokens = rows[start:stop, :-1].to(device)
        targets = rows[start:stop, 1:].to(device)
        batch_masks = {key: masks_all[key][start:stop].to(device)
                       for key in DENSITY_KEYS}
        native_logits, native, _c = r493.parent._native_all(model, tokens, reference)
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
        native_M = native["M"].float()
        for branch_index, branch in enumerate(TI):
            logits, capture, _a = r493.parent.parent.base._absent_forward(
                model, tokens, native, native["branches"][branch])
            calls["absent_forwards"] += 1
            absent_ce[branch_index, start:stop] = ce(logits, targets).double().cpu()
            r493._update_max(errors, "mlp0_state_max_abs", capture["mlp0_state_error"])
            base = capture["M"].float()
            for density_index, key in enumerate(DENSITY_KEYS):
                edited = torch.where(
                    batch_masks[key].unsqueeze(-1), native_M, base
                ).to(capture["M"].dtype)
                errors["edit_rms_min"] = min(
                    errors["edit_rms_min"], r493._rms(edited.float() - base))
                arm_logits, audit, _actual = r493._merge_forward(
                    model, tokens, capture, "M_ONLY", edited)
                calls["merge_forwards"] += 1
                r493._update_max(errors, "edited_write_max_abs_error",
                                 audit["edited_write_max_abs_error"])
                arm_ce[branch_index, density_index, start:stop] = \
                    ce(arm_logits, targets).double().cpu()
                del arm_logits
    expected = {"native_forwards": BATCHES,
                "absent_forwards": BATCHES * len(TI),
                "merge_forwards": BATCHES * len(TI) * len(DENSITIES)}
    instrument = {
        "calls": calls, "expected_calls": expected, "calls_exact": calls == expected,
        **{key: value for key, value in errors.items()
           if key not in ("analytical_num", "analytical_den",
                          "deployed_num", "deployed_den")},
        "analytical_branch_identity_relative_squared": errors["analytical_num"]
            / max(errors["analytical_den"], 1e-30),
        "deployed_branch_identity_relative_squared": errors["deployed_num"]
            / max(errors["deployed_den"], 1e-30),
        "nesting_exact": nesting_ok,
        "min_nonempty_docs_per_half_lowest": nonempty_per_half,
        "documents": docs,
    }
    return {"native": native_ce, "absent": absent_ce, "arms": arm_ce,
            "masks": masks_all, "instrument": instrument}


def _masked_stats(recovery, x, mask):
    r = recovery[mask].reshape(-1)
    v = x[mask].reshape(-1)
    fraction = float(torch.dot(r, v)) / max(float(torch.dot(v, v)), 1e-30)
    cosine = float(torch.dot(r, v)
                   / (torch.linalg.vector_norm(r)
                      * torch.linalg.vector_norm(v)).clamp_min(1e-30))
    return fraction, cosine, int(mask.sum())


def analyze(collected, anchors):
    native = collected["native"]
    docs = native.shape[0]
    halves = ((0, HALF), (HALF, docs))
    reports = {}
    b_flags, c_flags, counts_ok = [], [], []
    for branch_index, branch in enumerate(TI):
        reports[branch] = {}
        absent = collected["absent"][branch_index]
        for half_index, (lo, hi) in enumerate(halves):
            x = absent[lo:hi] - native[lo:hi]
            profile = []
            row = {}
            for density_index, key in enumerate(DENSITY_KEYS):
                mask = collected["masks"][key][lo:hi]
                recovery = (absent[lo:hi]
                            - collected["arms"][branch_index, density_index, lo:hi])
                fraction, cosine, count = _masked_stats(recovery, x, mask)
                profile.append(fraction)
                row[key] = {"aligned_recovered_fraction": fraction,
                            "recovery_cosine": cosine,
                            "scored_positions": count}
            counts_ok.append(bool(
                row["d128"]["scored_positions"] >= MIN_SCORED_PER_CELL_LOWEST))
            monotone = all(profile[i] < profile[i + 1]
                           for i in range(len(profile) - 1))
            span = profile[-1] - profile[0]
            b_flags.append(bool(monotone and span >= .30))
            anchor = anchors[branch][f"half{half_index}"]
            anchor_dev = max(
                abs(profile[{"d64": 1, "d16": 2, "d4": 3, "d1": 5}[key]]
                    - anchor[v1b_index])
                for key, v1b_index in COMMON.items())
            interleave = (profile[0] < profile[1]
                          and profile[3] < profile[4] < profile[5])
            c_flags.append(bool(anchor_dev <= DETERMINISM_TOL and interleave))
            row["profile"] = profile
            row["span"] = span
            row["monotone"] = monotone
            row["max_common_density_deviation_from_v1b"] = anchor_dev
            reports[branch][f"half{half_index}"] = row
    return {
        "reports": reports,
        "scored_counts_ok": bool(all(counts_ok)),
        "pred_b_flags": b_flags, "pred_c_flags": c_flags,
        "pred_b_six_point_strict_monotone_with_span": bool(all(b_flags)),
        "pred_c_determinism_anchor_and_interleaving": bool(all(c_flags)),
    }


def _synthetic(anchors):
    docs, length = 12, 96
    generator = torch.Generator().manual_seed(4)
    native = torch.rand(docs, length, dtype=torch.float64, generator=generator)
    absent = (native + .5).expand(len(TI), docs, length).clone()
    arms = torch.zeros(len(TI), len(DENSITIES), docs, length, dtype=torch.float64)
    fracs = (.15, anchors["T"]["half0"][0], anchors["T"]["half0"][1],
             anchors["T"]["half0"][2], .7, anchors["T"]["half0"][3])
    for bi, br in enumerate(TI):
        for di in range(len(DENSITIES)):
            f = fracs[di] if br == "T" else min(.99, fracs[di] + .001)
            arms[bi, di] = absent[bi] - f * (absent[bi] - native)
    masks = build_nested_masks(docs, length, torch.Generator().manual_seed(MASK_SEED))
    return {"native": native, "absent": absent, "arms": arms, "masks": masks}


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert EXPECTED_FORWARDS == 1875
        assert len(DENSITIES) == 6
        masks = build_nested_masks(8, 256, torch.Generator().manual_seed(MASK_SEED))
        for i in range(5):
            assert bool((~masks[DENSITY_KEYS[i]] | masks[DENSITY_KEYS[i + 1]]).all())
        anchors = {br: {"half0": [.3, .4, .5, .95], "half1": [.3, .4, .5, .95]}
                   for br in TI}
        global HALF, MIN_SCORED_PER_CELL_LOWEST, DETERMINISM_TOL
        real = (HALF, MIN_SCORED_PER_CELL_LOWEST, DETERMINISM_TOL)
        HALF, MIN_SCORED_PER_CELL_LOWEST, DETERMINISM_TOL = 6, 1, 1e-6
        try:
            analysis = analyze(_synthetic(anchors), anchors)
        finally:
            HALF, MIN_SCORED_PER_CELL_LOWEST, DETERMINISM_TOL = real
        assert len(analysis["pred_b_flags"]) == 4
        assert analysis["pred_b_six_point_strict_monotone_with_span"] is True
        for path, expected in HASHES.items():
            if not path.is_file() or r493.sha256(path) != expected:
                raise RuntimeError(f"frozen hash mismatch: {path}")
        print(json.dumps({
            "status": "dry_run_passed",
            "rung": "mlp1_write_locality_density_envelope_extension",
            "model_loaded": False, "outcomes_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "synthetic_analysis_and_nesting_exercised": True,
        }, indent=2, sort_keys=True))
        return
    rows, fit_rows, anchors, metadata = validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("envelope extension output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = r493.parent.parent.base.component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    collected = collect(model, rows, reference)
    analysis = analyze(collected, anchors)
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
        and instrument["nesting_exact"]
        and instrument["min_nonempty_docs_per_half_lowest"]
            >= MIN_NONEMPTY_DOCS_PER_HALF_LOWEST
        and analysis["scored_counts_ok"])
    pred_b = analysis["pred_b_six_point_strict_monotone_with_span"]
    pred_c = analysis["pred_c_determinism_anchor_and_interleaving"]
    strong_null = bool(not pred_a or not pred_b or not pred_c)
    torch.save({
        "schema": "mlp1_write_locality_density_envelope_ext_v1",
        "native": collected["native"].float(),
        "absent": collected["absent"].float(),
        "arms": collected["arms"].float(),
        "masks": collected["masks"],
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete",
        "rung": "mlp1_write_locality_density_envelope_extension",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "locality_envelope_certificate_measurement",
        "source_hashes": {str(path): r493.sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(TI), "densities": list(DENSITIES),
        "mask_seed": MASK_SEED,
        "documents": list(DOC_RANGE), "halves": [[0, HALF], [HALF, DOC_RANGE[1]]],
        "analysis": analysis, "instrument": instrument,
        "bundle": {"path": str(BUNDLE), "sha256": r493.sha256(BUNDLE),
                   "raw_tokens_logits_or_hidden_states_included": False},
        'pred_a_exact_lawful_live_envelope_instrument': pred_a,
        'pred_b_six_point_strict_monotone_with_span': pred_b,
        'pred_c_determinism_anchor_and_interleaving': pred_c,
        "validation_or_sealed_opened": False,
        "strong_null": strong_null,
        "execution_price": {
            "full_model_forwards": EXPECTED_FORWARDS,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "instrument_breach_repair_only" if not pred_a else
            ("six_point_envelope_feeds_extraction_certificate" if not strong_null
             else ("determinism_alarm_repair_only" if not pred_c else
                   "envelope_not_monotone_restate_certificate_on_monotone_hull")),
        ),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"], "rung": result["rung"],
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "strong_null": strong_null,
        "analysis": {key: value for key, value in analysis.items()
                     if key != "reports"},
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
