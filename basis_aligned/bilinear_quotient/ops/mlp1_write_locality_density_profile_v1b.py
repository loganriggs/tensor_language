#!/usr/bin/env python3
"""Locality-density profile of the MLP1 write restoration (v1b floors repair).

# BQGATE: EXPERIMENT
# pred_a_exact_lawful_live_density_instrument
# pred_b_profile_rises_monotonically_and_materially
# pred_c_nonlocality_material_and_full_anchor_reproduces

Parallel-lane probe (Claude). Turns the 2618/2620 sleeper (sparse-mask own
restoration recovers only .15-.53 of the branch effect AT the edited positions
vs .956 for the full write) into a measured locality law: restore the native
MLP1 write on NESTED seeded random masks of four densities and score recovery
on the edited positions. Imports the frozen rung493 module hash-pinned.
Preregistration:
polynomial_causal/MLP1_WRITE_LOCALITY_DENSITY_PROFILE_PREREGISTRATION.md
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

PREREG = POLY / "MLP1_WRITE_LOCALITY_DENSITY_PROFILE_PREREGISTRATION.md"
R493_SOURCE = ROOT / "ops/mlp0_TI_site_graded_merge_intervention_rung493.py"
R493_RESULT = ROOT / "mlp0_TI_site_graded_merge_intervention_rung493_results.json"
PROBE_RESULT = ROOT / "mlp1_write_interface_portability_probe_results.json"
V2B_RESULT = ROOT / "mlp1_write_token_keyed_transplant_v2b_results.json"
V1_RESULT = ROOT / "mlp1_write_locality_density_profile_results.json"
OUT = ROOT / "mlp1_write_locality_density_profile_v1b_results.json"
BUNDLE = ROOT / "mlp1_write_locality_density_profile_v1b_per_token.pt"
HASHES = {
    PREREG: "ff4de50b5c88323fec38dacddc09e9440e03bec02aeb05e248708a7c3eda19a9",
    R493_SOURCE: "4f77c3898d8237373a7a35439dabc590882eda47490aaed33a797ddec2cfe08b",
    R493_RESULT: "1131a0dc61f94ca2dba92073eed1a21c2f46a46ac18004be146e15a78161339d",
    PROBE_RESULT: "1c50849582f09858e258fa0f5d276439d8b83442888fb7a8c681b3967318954f",
    V2B_RESULT: "b25bb2eed99e813c0ba18552a4e807566c108c1456c3e859dcde7440d71ea6d4",
    V1_RESULT: "f251a04575d26315d791eefae034f566902428e651597417c62bbe5ee9cd3d3f",
}

TI = ("T", "I")
DENSITIES = (1 / 64, 1 / 16, 1 / 4, 1.0)
DENSITY_KEYS = ("d64", "d16", "d4", "d1")
MASK_SEED = 20260904
BATCH = r493.BATCH
DOC_RANGE = (0, 500)
HALF = 250
BATCHES = (DOC_RANGE[1] - DOC_RANGE[0]) // BATCH
EXPECTED_FORWARDS = BATCHES * (1 + len(TI) + len(TI) * len(DENSITIES))
MIN_SCORED_PER_CELL_LOWEST = 900
MIN_NONEMPTY_DOCS_LOWEST = 480
FULL_ANCHOR_WINDOW = (0.92, 1.00)


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or r493.sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    probe = json.loads(PROBE_RESULT.read_text())
    if probe.get("pred_a_exact_lawful_live_transplant_instrument") is not True:
        raise RuntimeError("portability probe receipt not lawful")
    v2b = json.loads(V2B_RESULT.read_text())
    if v2b.get("pred_a_exact_lawful_live_supplied_instrument") is not True:
        raise RuntimeError("v2b receipt not lawful")
    v1 = json.loads(V1_RESULT.read_text())
    if v1.get("pred_a_exact_lawful_live_density_instrument") is not False \
            or v1.get("strong_null") is not True:
        raise RuntimeError("v1 receipt does not license the v1b floors repair")
    rows, fit_rows, metadata = r493.validate_inputs()
    return rows, fit_rows, {
        **metadata, "densities": list(DENSITIES), "mask_seed": MASK_SEED,
    }


def build_nested_masks(doc_count, length, generator):
    """Nested Bernoulli-thinned masks: d64 subset d16 subset d4 subset all pos>=1."""
    uniform = torch.rand(doc_count, length, generator=generator)
    uniform[:, 0] = 2.0  # position 0 excluded at every density
    masks = {}
    for key, density in zip(DENSITY_KEYS, DENSITIES):
        masks[key] = uniform < density
    return masks


@torch.no_grad()
def collect(model, rows, reference):
    ce = r493.parent.parent.base.factorial_parent._per_token_ce
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(MASK_SEED)
    docs = DOC_RANGE[1] - DOC_RANGE[0]
    length = rows.shape[1] - 1
    masks_all = build_nested_masks(docs, length, generator)
    nesting_ok = bool(
        (~masks_all["d64"] | masks_all["d16"]).all()
        and (~masks_all["d16"] | masks_all["d4"]).all()
        and (~masks_all["d4"] | masks_all["d1"]).all())
    min_per_doc = int(masks_all["d64"].sum(1).min())
    nonempty_docs = int((masks_all["d64"].sum(1) > 0).sum())
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
        "min_lowest_density_positions_per_doc": min_per_doc,
        "nonempty_docs_lowest_density": nonempty_docs,
        "documents": docs,
    }
    return {"native": native_ce, "absent": absent_ce, "arms": arm_ce,
            "masks": {key: masks_all[key] for key in DENSITY_KEYS},
            "instrument": instrument}


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
            counts_ok.append(bool(row["d64"]["scored_positions"]
                                  >= MIN_SCORED_PER_CELL_LOWEST))
            monotone = all(profile[i] < profile[i + 1] for i in range(3))
            span = profile[3] - profile[0]
            b_flags.append(bool(monotone and span >= .30))
            c_flags.append(bool(
                profile[0] <= .60
                and FULL_ANCHOR_WINDOW[0] <= profile[3] <= FULL_ANCHOR_WINDOW[1]))
            row["profile"] = profile
            row["span"] = span
            row["monotone"] = monotone
            row["propagation_share_lowest"] = (
                1 - profile[0] / profile[3] if profile[3] > 0 else None)
            reports[branch][f"half{half_index}"] = row
    return {
        "reports": reports,
        "scored_counts_ok": bool(all(counts_ok)),
        "pred_b_flags": b_flags, "pred_c_flags": c_flags,
        "pred_b_profile_rises_monotonically_and_materially": bool(all(b_flags)),
        "pred_c_nonlocality_material_and_full_anchor_reproduces": bool(all(c_flags)),
    }


def _synthetic_collected():
    generator = torch.Generator().manual_seed(3)
    docs, length = 12, 64
    native = torch.rand(docs, length, dtype=torch.float64, generator=generator)
    absent = native + .5
    masks = build_nested_masks(docs, length, torch.Generator().manual_seed(MASK_SEED))
    arms = torch.zeros(len(TI), len(DENSITIES), docs, length, dtype=torch.float64)
    for bi in range(len(TI)):
        for di, frac in enumerate((.3, .5, .7, .95)):
            arms[bi, di] = absent - frac * (absent - native)
    return {"native": native, "absent": absent.expand(len(TI), docs, length).clone(),
            "arms": arms, "masks": masks}


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert EXPECTED_FORWARDS == 1375
        assert len(DENSITIES) == 4
        masks = build_nested_masks(8, 256, torch.Generator().manual_seed(MASK_SEED))
        assert bool((~masks["d64"] | masks["d16"]).all())
        assert bool((~masks["d16"] | masks["d4"]).all())
        assert not masks["d1"][:, 0].any()
        global HALF, MIN_SCORED_PER_CELL_LOWEST
        real = (HALF, MIN_SCORED_PER_CELL_LOWEST)
        HALF, MIN_SCORED_PER_CELL_LOWEST = 6, 1
        try:
            analysis = analyze(_synthetic_collected())
        finally:
            HALF, MIN_SCORED_PER_CELL_LOWEST = real
        assert len(analysis["pred_b_flags"]) == 4
        assert analysis["pred_b_profile_rises_monotonically_and_materially"] is True
        for path, expected in HASHES.items():
            if not path.is_file() or r493.sha256(path) != expected:
                raise RuntimeError(f"frozen hash mismatch: {path}")
        print(json.dumps({
            "status": "dry_run_passed", "rung": "mlp1_write_locality_density_profile_v1b",
            "model_loaded": False, "outcomes_opened": False,
            "validation_or_sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "synthetic_profile_and_nesting_exercised": True,
        }, indent=2, sort_keys=True))
        return
    rows, fit_rows, metadata = validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("v1b output namespace already exists")
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
        and instrument["nesting_exact"]
        and instrument["nonempty_docs_lowest_density"] >= MIN_NONEMPTY_DOCS_LOWEST
        and analysis["scored_counts_ok"])
    pred_b = analysis["pred_b_profile_rises_monotonically_and_materially"]
    pred_c = analysis["pred_c_nonlocality_material_and_full_anchor_reproduces"]
    strong_null = bool(not pred_a or not pred_b or not pred_c)
    torch.save({
        "schema": "mlp1_write_locality_density_profile_v1",
        "native": collected["native"].float(),
        "absent": collected["absent"].float(),
        "arms": collected["arms"].float(),
        "masks": collected["masks"],
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": "mlp1_write_locality_density_profile_v1b",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "write_site_locality_law_measurement",
        "source_hashes": {str(path): r493.sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(TI), "densities": list(DENSITIES),
        "mask_seed": MASK_SEED,
        "documents": list(DOC_RANGE), "halves": [[0, HALF], [HALF, DOC_RANGE[1]]],
        "analysis": analysis, "instrument": instrument,
        "bundle": {"path": str(BUNDLE), "sha256": r493.sha256(BUNDLE),
                   "raw_tokens_logits_or_hidden_states_included": False},
        'pred_a_exact_lawful_live_density_instrument': pred_a,
        'pred_b_profile_rises_monotonically_and_materially': pred_b,
        'pred_c_nonlocality_material_and_full_anchor_reproduces': pred_c,
        "validation_or_sealed_opened": False,
        "strong_null": strong_null,
        "execution_price": {
            "full_model_forwards": EXPECTED_FORWARDS,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "instrument_breach_repair_only" if not pred_a else
            ("locality_law_measured_feed_extraction_pricing" if not strong_null else
             "profile_flat_or_nonmonotone_check_reversal_with_independent_control")),
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
