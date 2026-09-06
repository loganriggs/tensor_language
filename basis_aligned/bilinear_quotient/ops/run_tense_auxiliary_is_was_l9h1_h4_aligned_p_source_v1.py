#!/usr/bin/env python3
"""Alignment-preserving is/was H1/H4 P selectivity and source test."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_instrument_and_coverage pred_b_alignment_preserving_P_selectivity pred_c_changed_carrier_source_explains_residual pred_d_changed_source_specificity pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_candidate_tense_auxiliary_is_was_aligned_p_v1 as rows_builder
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_l9h1_h4_source_term_factorial_v1 as source_instrument


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_l9h1_h4_aligned_p_source_v1.json"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_l9h1_h4_aligned_p_source_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.l9h1_h4_aligned_p_source_v1"
PATHS = {
    "source_factorial": ROOT / "circuits/followups/tense_auxiliary_is_was_l9h1_h4_source_term_factorial_v1_result.json",
    "head_reuse": ROOT / "circuits/followups/tense_auxiliary_is_was_l9h1_h4_cross_task_reader_reuse_v1_result.json",
    "source_instrument": ROOT / "ops/run_aspectual_anchor_l9h1_h4_source_term_factorial_v1.py",
    "aligned_p_builder": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_aligned_p_v1.py",
    "q_is_scale_authority": ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json",
}
EXPECTED_PRIOR_SHA256 = "2b42c7337ae771387cd5f100ba69c6dbe5cbb45953cebb9cb370307143bdac19"
EXPECTED = {
    "source_factorial": "4c266158213edcda9f0c86b19064cabe6d673815167b69d9eff381ddadda9cf5",
    "head_reuse": "ca3139b6eba33f3d06c6d79c5b772f8ecf568e16918d2e2211c282847d577070",
    "source_instrument": "8e890efd3520cfbece1d71f3ffb58397c732d8fc9c9446c74af9ac0380f2ca01",
    "aligned_p_builder": "15f0db2b22d2fa5c674cfa8370f7ab0c341f2ca10cc7311e2b383c6ba735e2ba",
    "q_is_scale_authority": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
}
EXPECTED_ROWS_SHA256 = "9b42b6f255a767822954bea19a610089e88fcb39f8ed2b8008a5189d2fa7c3bb"


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_error(first, second):
    if len(first.answer_foil) != len(second.answer_foil):
        raise ExperimentError("pair coverage changed")
    return max(abs(float(a) - float(b)) for left, right in zip(first.answer_foil, second.answer_foil) for a, b in zip(left, right))


def normalized_effects(base_output, patched_output, scale):
    values = []
    for base_pair, patched_pair in zip(base_output.answer_foil, patched_output.answer_foil):
        base_margin = float(base_pair[0]) - float(base_pair[1])
        patched_margin = float(patched_pair[0]) - float(patched_pair[1])
        effect = abs(patched_margin - base_margin) / scale
        if not math.isfinite(effect):
            raise ExperimentError("nonfinite normalized P effect")
        values.append(effect)
    return values


def prediction_record(a, b, c, d, e):
    return {
        "pred_a_authority_capability_exact_instrument_and_coverage": a,
        "pred_b_alignment_preserving_P_selectivity": b,
        "pred_c_changed_carrier_source_explains_residual": c,
        "pred_d_changed_source_specificity": d,
        "pred_e_exact_zero_fit_price": e,
    }


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    prior = json.loads(PRIOR.read_text())
    source_result = json.loads(PATHS["source_factorial"].read_text())
    reuse_result = json.loads(PATHS["head_reuse"].read_text())
    rows = rows_builder.build_rows()
    expected_authorities = {"source_factorial_sha256": EXPECTED["source_factorial"], "head_reuse_sha256": EXPECTED["head_reuse"], "source_instrument_sha256": EXPECTED["source_instrument"], "aligned_p_builder_sha256": EXPECTED["aligned_p_builder"], "aligned_p_rows_sha256": EXPECTED_ROWS_SHA256, "q_is_scale_authority_sha256": EXPECTED["q_is_scale_authority"]}
    ok = prior.get("candidate_id") == CANDIDATE_ID and prior.get("authorities") == expected_authorities and source_result.get("terminal") == "screen" and reuse_result.get("terminal") == "null" and rows_builder.validate_rows(rows) == EXPECTED_ROWS_SHA256 and len(rows) == 16
    if not ok:
        raise ExperimentError("candidate, prior results, or aligned P rows changed")
    return rows, json.loads(PATHS["q_is_scale_authority"].read_text())


def main():
    rows, scale_result = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "rows": 16, "intervention_arms": 6, "model_forwards": 10, "example_evaluations": 160, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = source_instrument.SourceBackend.load("cuda")
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    forward_calls = 0
    base_native = backend.native(base_batch, capture=True)
    forward_calls += 1
    donor_native = backend.native(donor_batch, capture=True)
    forward_calls += 1
    base_manual, base_manual_capture = backend.manual_forward(base_batch)
    forward_calls += 1
    donor_manual, donor_manual_capture = backend.manual_forward(donor_batch)
    forward_calls += 1
    manual_error = max(pair_error(base_native, base_manual), pair_error(donor_native, donor_manual))
    reconstruction_error = max(float(base_manual_capture["reconstruction_max_abs"]), float(donor_manual_capture["reconstruction_max_abs"]))
    capability_cells = []
    for direction, parity in (("present_to_past", 0), ("past_to_present", 1)):
        indices = [index for index, row in enumerate(rows) if row["group_number"] % 2 == parity]
        for side, output in (("base", base_native), ("donor", donor_native)):
            accuracy = sum(float(output.answer_foil[index][0]) > float(output.answer_foil[index][1]) for index in indices) / len(indices)
            capability_cells.append({"direction": direction, "side": side, "count": len(indices), "accuracy": accuracy, "threshold": 0.85, "passed": accuracy >= 0.85})
    identity, _ = backend.manual_forward(base_batch, donor_batch=base_batch, donor_capture=base_manual_capture, arm="full_pair")
    forward_calls += 1
    full_manual, _ = backend.manual_forward(base_batch, donor_batch=donor_batch, donor_capture=donor_manual_capture, arm="full_pair")
    forward_calls += 1
    full_hook = backend.patched_heads(base_batch, layer=9, heads=source_instrument.HEADS, donor_cache=donor_native.captured)
    forward_calls += 1
    source_outputs = {}
    for label, arm in (("changed_noun", "cue_joint"), ("determiner", "last_joint"), ("self", "self_joint")):
        source_outputs[label], _ = backend.manual_forward(base_batch, donor_batch=donor_batch, donor_capture=donor_manual_capture, arm=arm)
        forward_calls += 1
    identity_error = pair_error(base_native, identity)
    route_error = pair_error(full_manual, full_hook)
    scale = float(scale_result["score"]["families"]["target_scale"])
    effects = {"full_pair": normalized_effects(base_native, full_manual, scale)}
    effects.update({label: normalized_effects(base_native, output, scale) for label, output in source_outputs.items()})
    summaries = {label: {"count": len(values), "mean_normalized_absolute_effect": statistics.fmean(values), "max_normalized_absolute_effect": max(values)} for label, values in effects.items()}
    full_mean = summaries["full_pair"]["mean_normalized_absolute_effect"]
    changed_mean = summaries["changed_noun"]["mean_normalized_absolute_effect"]
    changed_retained = changed_mean / full_mean if full_mean > 0.0 else math.nan
    pred_a = all(cell["passed"] for cell in capability_cells) and manual_error <= 1e-4 and reconstruction_error <= 1e-4 and identity_error <= 1e-5 and route_error <= 1e-5 and all(len(values) == 16 for values in effects.values())
    pred_b = full_mean <= 0.20
    pred_c = full_mean < 0.02 or (math.isfinite(changed_retained) and changed_retained >= 0.50)
    pred_d = all(summaries[label]["mean_normalized_absolute_effect"] <= changed_mean for label in ("determiner", "self"))
    price = {"model_forwards": forward_calls, "example_evaluations": forward_calls * len(rows), "rows": len(rows), "intervention_arms": 6, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 10, "example_evaluations": 160, "rows": 16, "intervention_arms": 6, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = prediction_record(pred_a, pred_b, pred_c, pred_d, pred_e)
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_e else "invalid")
    reason = {"screen": "aligned_P_selectivity_and_changed_carrier_source_supported", "null": "aligned_P_selectivity_or_source_prediction_misses", "invalid": "authority_alignment_capability_instrument_finiteness_coverage_or_price_invalid"}[terminal]
    result = {
        "schema": "tense_auxiliary_is_was_l9h1_h4_aligned_p_source_result_v1", "candidate_id": CANDIDATE_ID,
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "authority_sha256": EXPECTED, "rows_sha256": EXPECTED_ROWS_SHA256, "capability_cells": capability_cells,
        "instrument": {"manual_native_max_abs_logit_error": manual_error, "source_reconstruction_max_abs_error": reconstruction_error, "identity_max_abs_logit_error": identity_error, "manual_full_vs_trusted_hook_max_abs_logit_error": route_error},
        "summaries": summaries, "changed_noun_retained_fraction_of_full": changed_retained, "effects": effects,
        "predictions": predictions,
        "price": price,
        "terminal": terminal, "reason": reason, "serial_seconds": time.perf_counter() - started,
        "next_action": "promote H1/H4 as shared contextual readers within alignment-preserving scope and trace upstream contextualization" if terminal == "screen" else "retain the H1/H4 P nuisance boundary without changing the aligned paraphrase",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability_cells", "instrument", "summaries", "changed_noun_retained_fraction_of_full", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
