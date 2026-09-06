#!/usr/bin/env python3
"""Prospective no-fit test of the shared local-L9-value edge on will/had."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_factorization pred_b_prospective_writer_and_bank_recurrence pred_c_local_value_edge_reuses pred_d_interaction_is_secondary pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_das_subspace as das
import circuit_fast_screen_candidate_temporal_auxiliary as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_v1 as inherited


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_local_l9_value_reuse_v1.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_local_l9_value_reuse_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.local_l9_value_reuse_v1"
PATHS = {
    "builder": ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py",
    "capability_screen": ROOT / "circuits/fast_screens/temporal_auxiliary_will_vs_had_v1_result.json",
    "factor_instrument": ROOT / "ops/run_tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_v1.py",
    "typed_shared_path": ROOT / "circuits/followups/aspectual_tense_typed_shared_contextual_path_v2_result.json",
}
EXPECTED_PRIOR_SHA256 = "71b2e3fb939b37e007ec91a89f72ed14272def49cae521a94497c3b9a59b08f4"
EXPECTED = {
    "builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
    "capability_screen": "71a833c52c39b98be6f576d11749522e2725c540a29d61a2352e315978684ec9",
    "factor_instrument": "6826d33fadd2af133000cb3c826b4d89c535f576c3f057e667e68dece98e7d39",
    "typed_shared_path": "18b877a114f718f9817dee4402130099cef0a109d7fcb4f5dd9f843636e5759f",
}
EXPECTED_ROWS_SHA256 = "b74ff65cf6b59142bf38172dae7f70a25e5f471bbeaa038f9b3971caedf913da"
FACTORS = inherited.FACTORS


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    if {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("authority hash changed")
    all_rows = candidate.build_rows()
    if candidate.validate_rows(all_rows) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("temporal row authority changed")
    rows = [row for row in all_rows if row["transform_id"] in {"A1", "A2"}]
    capability = json.loads(PATHS["capability_screen"].read_text())
    typed = json.loads(PATHS["typed_shared_path"].read_text())
    if (
        len(rows) != 64
        or capability.get("terminal") != "screen"
        or capability.get("selected_site_id") != "resid:18"
        or capability.get("head_stage") != "skipped_no_passing_attention_module"
        or typed.get("terminal") != "screen"
        or len(inherited.subsets()) != 8
    ):
        raise ExperimentError("population or prospective boundary changed")
    return rows


def main():
    rows = validate_static()
    dryrun = {
        "candidate_id": CANDIDATE_ID,
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "rows": 64,
        "family_batches": 2,
        "factor_arms": 8,
        "intervention_records": 512,
        "model_forwards": 24,
        "example_evaluations": 768,
        "fitted_scalars": 0,
        "grid_evaluations": 0,
        "root_evaluations": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = inherited.Backend.load("cuda")
    records = []
    capability_cells = []
    summaries = {inherited.arm_id(subset): {} for subset in inherited.subsets()}
    writer_summary = {}
    values = {}
    instrument = {
        "layer0_v1_invariance_max_abs_error": 0.0,
        "base_effective_value_recombination_max_abs_error": 0.0,
        "hybrid_effective_value_recombination_max_abs_error": 0.0,
        "bank_factor_closure_max_abs_error": 0.0,
        "empty_writer_hook_max_abs_logit_error": 0.0,
        "bilinear_tensor_reconstruction_max_abs_error": 0.0,
        "attention_source_reconstruction_max_abs_error": 0.0,
        "observed_lambda9": None,
    }
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        if len(family_rows) != 32:
            raise ExperimentError("family batch size changed")
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, base_mlp_capture = backend.capture_bilinear(base_batch)
        donor_output, donor_mlp_capture = backend.capture_bilinear(donor_batch)
        empty_output, base_attention, base_raw, empty_tensor_error = backend.capture_writer_raw(
            base_batch, donor_batch, base_mlp_capture, donor_mlp_capture, ()
        )
        writer_output, hybrid_attention, hybrid_raw, writer_tensor_error = backend.capture_writer_raw(
            base_batch,
            donor_batch,
            base_mlp_capture,
            donor_mlp_capture,
            inherited.path.FACTORS,
        )
        forward_calls += 4
        evaluations += 4 * len(family_rows)
        native = {}
        for side, output in (("base", base_output), ("donor", donor_output)):
            for row, pair in zip(family_rows, output.answer_foil):
                native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                    str(row["row_id"]), family, side, *producer._finite_pair(pair)
                )
        for direction in ("future_to_anterior", "anterior_to_future"):
            cell_rows = [row for row in family_rows if row["direction_id"] == direction]
            for side in ("base", "donor"):
                accuracy = sum(
                    native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows
                ) / len(cell_rows)
                capability_cells.append(
                    {
                        "family": family,
                        "direction": direction,
                        "side": side,
                        "count": len(cell_rows),
                        "accuracy": accuracy,
                        "threshold": 0.85,
                        "passed": accuracy >= 0.85,
                    }
                )
        tensors, checks = inherited.factor_tensors(
            backend,
            base_batch,
            donor_batch,
            base_attention,
            hybrid_attention,
            base_raw,
            hybrid_raw,
        )
        for key, value in checks.items():
            if key == "observed_lambda9":
                instrument[key] = value
            else:
                instrument[key] = max(float(instrument[key]), value)
        instrument["empty_writer_hook_max_abs_logit_error"] = max(
            float(instrument["empty_writer_hook_max_abs_logit_error"]),
            max(
                abs(a - b)
                for pair_a, pair_b in zip(base_output.answer_foil, empty_output.answer_foil)
                for a, b in zip(pair_a, pair_b)
            ),
        )
        instrument["bilinear_tensor_reconstruction_max_abs_error"] = max(
            float(instrument["bilinear_tensor_reconstruction_max_abs_error"]),
            empty_tensor_error,
            writer_tensor_error,
        )
        instrument["attention_source_reconstruction_max_abs_error"] = max(
            float(instrument["attention_source_reconstruction_max_abs_error"]),
            float(base_attention["reconstruction_max_abs"]),
            float(hybrid_attention["reconstruction_max_abs"]),
        )
        writer_values = []
        for row, pair in zip(family_rows, writer_output.answer_foil):
            answer, foil = producer._finite_pair(pair)
            writer_values.append(
                kernel.signed_pairwise_donor_recovery(
                    -native[(str(row["row_id"]), "base")].margin,
                    native[(str(row["row_id"]), "donor")].margin,
                    -(answer - foil),
                )
            )
        writer_summary[family] = inherited.summarize(writer_values)
        for subset in inherited.subsets():
            output = inherited.intervene(backend, base_batch, tensors, subset)
            forward_calls += 1
            evaluations += len(family_rows)
            arm_values = []
            for row, pair in zip(family_rows, output.answer_foil):
                answer, foil = producer._finite_pair(pair)
                recovery = kernel.signed_pairwise_donor_recovery(
                    -native[(str(row["row_id"]), "base")].margin,
                    native[(str(row["row_id"]), "donor")].margin,
                    -(answer - foil),
                )
                arm_values.append(recovery)
                records.append(
                    {
                        "arm": inherited.arm_id(subset),
                        "family": family,
                        "row_id": str(row["row_id"]),
                        "recovery": recovery,
                    }
                )
            summaries[inherited.arm_id(subset)][family] = inherited.summarize(arm_values)
            values[(subset, family)] = summaries[inherited.arm_id(subset)][family]["mean_recovery"]
    shapley = {
        family: inherited.factorial_shapley(
            {subset: values[(subset, family)] for subset in inherited.subsets()}
        )
        for family in ("A1", "A2")
    }
    local = (FACTORS[1],)
    additive = (FACTORS[0], FACTORS[1])
    local_retained = {
        family: values[(local, family)] / values[(FACTORS, family)]
        if abs(values[(FACTORS, family)]) > 1e-9
        else math.nan
        for family in ("A1", "A2")
    }
    additive_retained = {
        family: values[(additive, family)] / values[(FACTORS, family)]
        if abs(values[(FACTORS, family)]) > 1e-9
        else math.nan
        for family in ("A1", "A2")
    }
    writer_mean = statistics.fmean(
        writer_summary[family]["mean_recovery"] for family in ("A1", "A2")
    )
    full_mean = statistics.fmean(values[(FACTORS, family)] for family in ("A1", "A2"))
    pred_a = (
        all(cell["passed"] for cell in capability_cells)
        and instrument["bilinear_tensor_reconstruction_max_abs_error"] <= 2e-3
        and max(
            value
            for key, value in instrument.items()
            if key.endswith("error") and key != "bilinear_tensor_reconstruction_max_abs_error"
        )
        <= 1e-4
        and len(records) == 512
    )
    pred_b = (
        writer_mean >= 0.10
        and full_mean >= 0.05
        and all(
            writer_summary[family]["mean_recovery"] > 0.0
            and writer_summary[family]["direction_fraction"] >= 0.75
            and values[(FACTORS, family)] > 0.0
            and summaries[inherited.arm_id(FACTORS)][family]["direction_fraction"] >= 0.75
            for family in ("A1", "A2")
        )
    )
    pred_c = all(
        math.isfinite(local_retained[family])
        and shapley[family][FACTORS[1]] > 0.0
        and shapley[family][FACTORS[1]] > shapley[family][FACTORS[0]]
        and shapley[family][FACTORS[1]] > shapley[family][FACTORS[2]]
        and local_retained[family] >= 0.60
        and summaries[FACTORS[1]][family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")
    )
    pred_d = all(
        math.isfinite(additive_retained[family])
        and additive_retained[family] >= 0.80
        and summaries[inherited.arm_id(additive)][family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")
    )
    price = {
        "model_forwards": forward_calls,
        "example_evaluations": evaluations,
        "rows": len(rows),
        "family_batches": 2,
        "factor_arms": 8,
        "intervention_records": len(records),
        "fitted_scalars": 0,
        "grid_evaluations": 0,
        "root_evaluations": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }
    pred_e = price == {
        "model_forwards": 24,
        "example_evaluations": 768,
        "rows": 64,
        "family_batches": 2,
        "factor_arms": 8,
        "intervention_records": 512,
        "fitted_scalars": 0,
        "grid_evaluations": 0,
        "root_evaluations": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }
    predictions = {
        "pred_a_authority_capability_exact_factorization": pred_a,
        "pred_b_prospective_writer_and_bank_recurrence": pred_b,
        "pred_c_local_value_edge_reuses": pred_c,
        "pred_d_interaction_is_secondary": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_e else "invalid")
    result = {
        "schema": "temporal_auxiliary_will_had_local_l9_value_reuse_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "authority_sha256": EXPECTED,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "scope_boundary": "Prospective internal-edge reuse only; no full will/had circuit identification is claimed.",
        "capability_cells": capability_cells,
        "instrument": instrument,
        "writer_summary": writer_summary,
        "summaries": summaries,
        "factor_shapley": shapley,
        "local_value_retained_fraction": local_retained,
        "routing_plus_local_retained_fraction": additive_retained,
        "predictions": predictions,
        "price": price,
        "terminal": terminal,
        "reason": {
            "screen": "local_l9_value_edge_predicts_new_temporal_auxiliary_construction",
            "null": "prospective_writer_bank_or_local_value_reuse_prediction_misses",
            "invalid": "authority_capability_capture_closure_coverage_or_price_invalid",
        }[terminal],
        "serial_seconds": time.perf_counter() - started,
        "next_action": (
            "add capability-qualified will/had internal edge to typed graph and test selective manipulation"
            if terminal == "screen"
            else "bound typed local-L9-value reuse to aspectual/copular tasks and localize the later will/had route"
        ),
    }
    atomic_create_json(OUT, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "candidate_id",
                    "scope_boundary",
                    "instrument",
                    "writer_summary",
                    "summaries",
                    "factor_shapley",
                    "local_value_retained_fraction",
                    "routing_plus_local_retained_fraction",
                    "predictions",
                    "price",
                    "terminal",
                    "reason",
                    "next_action",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
