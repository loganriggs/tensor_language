#!/usr/bin/env python3
"""Exact semantic source partition for the will/had L9H1/H4 reader pair."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_partition pred_b_complete_h1h4_recurrence pred_c_literal_temporal_cue_dominates pred_d_subject_onset_and_self_secondary pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import attention_source_group_eval as source
import circuit_das_subspace as das
import circuit_fast_screen_candidate_temporal_auxiliary as candidate
from circuit_fast_screen_managed_runner import atomic_create_json
import run_tense_auxiliary_is_was_mlp4_to_l9h1_h4_path_mediation_v1 as capture_backend


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_l9h1h4_source_groups_v1.json"
HEAD_SCREEN = ROOT / "circuits/followups/temporal_auxiliary_will_had_attn9_h1h4_complement_factorial_v1_result.json"
FAST_SCREEN = ROOT / "circuits/fast_screens/temporal_auxiliary_will_vs_had_v1_result.json"
EVALUATOR = ROOT / "ops/attention_source_group_eval.py"
BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
BACKEND = ROOT / "ops/run_tense_auxiliary_is_was_mlp4_to_l9h1_h4_path_mediation_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9h1h4_source_groups_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.l9h1h4_source_groups_v1"
EXPECTED = {
    "prior": "c3a7da92306cf4fb8f8adb3ca7ddabbf6c977d61cb2acc9f99b2960819f25dc4",
    "head_screen": "a885356476efe339f5d97c73f08989b2616beee13740edf81f48f4ae77ace570",
    "fast_screen": "71a833c52c39b98be6f576d11749522e2725c540a29d61a2352e315978684ec9",
    "evaluator": "6ecf5f40b92f94cb32bccf1a703e527a3d468281936e63d4c7e91e8af66b4348",
    "builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
    "backend": "ba2d36ca8a2dc18de92f4ea43d25f7769d5bcb2cfa110290c355d6aff8717501",
}
EXPECTED_ROWS_SHA256 = "b74ff65cf6b59142bf38172dae7f70a25e5f471bbeaa038f9b3971caedf913da"
ARMS = ("full_pair", "all_sources") + source.GROUP_ORDER
EXPECTED_FULL = {"A1": 0.3440319120336044, "A2": 0.36416007305921677}


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_error(first, second):
    return max(
        abs(float(a) - float(b))
        for pair_a, pair_b in zip(first.answer_foil, second)
        for a, b in zip(pair_a, pair_b)
    )


def validate_static():
    paths = {
        "prior": PRIOR,
        "head_screen": HEAD_SCREEN,
        "fast_screen": FAST_SCREEN,
        "evaluator": EVALUATOR,
        "builder": BUILDER,
        "backend": BACKEND,
    }
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    rows_all = candidate.build_rows()
    if candidate.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("row authority changed")
    rows = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    head_screen = json.loads(HEAD_SCREEN.read_text())
    fast_screen = json.loads(FAST_SCREEN.read_text())
    if (
        len(rows) != 64
        or head_screen.get("terminal") != "screen"
        or not all(head_screen.get("predictions", {}).values())
        or fast_screen.get("terminal") != "screen"
        or len(ARMS) != 7
    ):
        raise ExperimentError("population or parent screen changed")
    for family in ("A1", "A2"):
        row = next(item for item in rows if item["transform_id"] == family)
        groups = source.aligned_source_partition(
            row["base_ids"], row["donor_ids"], row["base_semantic_position"]
        )
        if tuple(groups) != source.GROUP_ORDER:
            raise ExperimentError("source partition order changed")
    return rows, fast_screen


def main():
    rows, fast_screen = validate_static()
    dryrun = {
        "candidate_id": CANDIDATE_ID,
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "rows": 64,
        "family_batches": 2,
        "arms": list(ARMS),
        "model_forwards": 18,
        "example_evaluations": 576,
        "intervention_records": 448,
        "fitted_scalars": 0,
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
    backend = capture_backend.Backend.load("cuda")
    frozen_native = {
        (str(item["row_id"]), str(item["side"])): (
            float(item["answer_logit"]), float(item["foil_logit"])
        )
        for item in fast_screen["run"]["native_logits"]
    }
    arm_records = {arm: [] for arm in ARMS}
    capability_cells = []
    manual_error = 0.0
    reconstruction_error = 0.0
    all_source_closure_error = 0.0
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, base_capture = backend.manual_forward(base_batch)
        donor_output, donor_capture = backend.manual_forward(donor_batch)
        forward_calls += 2
        evaluations += 2 * len(family_rows)
        for side, output in (("base", base_output), ("donor", donor_output)):
            frozen = [frozen_native[(row_id, side)] for row_id in base_batch.row_ids]
            manual_error = max(manual_error, pair_error(output, frozen))
        reconstruction_error = max(
            reconstruction_error,
            float(base_capture["reconstruction_max_abs"]),
            float(donor_capture["reconstruction_max_abs"]),
        )
        outputs = {
            "full_pair": source.intervene_complete_heads(
                backend, base_batch, donor_batch, donor_capture
            ),
            "all_sources": source.intervene_source_groups(
                backend,
                base_batch,
                donor_batch,
                base_capture,
                donor_capture,
                source.GROUP_ORDER,
            ),
        }
        for group in source.GROUP_ORDER:
            outputs[group] = source.intervene_source_groups(
                backend,
                base_batch,
                donor_batch,
                base_capture,
                donor_capture,
                (group,),
            )
        forward_calls += len(ARMS)
        evaluations += len(ARMS) * len(family_rows)
        all_source_closure_error = max(
            all_source_closure_error,
            max(
                abs(float(a) - float(b))
                for pair_a, pair_b in zip(
                    outputs["all_sources"].answer_foil,
                    outputs["full_pair"].answer_foil,
                )
                for a, b in zip(pair_a, pair_b)
            ),
        )
        for arm, output in outputs.items():
            arm_records[arm].extend(
                source.recovery_records(
                    family_rows, base_output, donor_output, output, arm=arm
                )
            )
        for direction in ("future_to_anterior", "anterior_to_future"):
            indices = [
                index
                for index, row in enumerate(family_rows)
                if row["direction_id"] == direction
            ]
            for side, output in (("base", base_output), ("donor", donor_output)):
                accuracy = sum(
                    float(output.answer_foil[index][0]) > float(output.answer_foil[index][1])
                    for index in indices
                ) / len(indices)
                capability_cells.append(
                    {
                        "family": family,
                        "direction": direction,
                        "side": side,
                        "count": len(indices),
                        "accuracy": accuracy,
                        "threshold": 0.85,
                        "passed": accuracy >= 0.85,
                    }
                )
    summaries = {
        arm: source.summarize_by_family(records) for arm, records in arm_records.items()
    }
    full = summaries["full_pair"]
    fractions = {
        group: {
            family: summaries[group][family]["mean_recovery"] / full[family]["mean_recovery"]
            for family in ("A1", "A2")
        }
        for group in source.GROUP_ORDER
    }
    pred_a = (
        all(cell["passed"] for cell in capability_cells)
        and manual_error <= 1e-4
        and reconstruction_error <= 1e-4
        and all_source_closure_error <= 1e-4
        and all(len(records) == 64 for records in arm_records.values())
        and all(
            math.isfinite(float(record["recovery"]))
            for records in arm_records.values()
            for record in records
        )
    )
    pred_b = all(
        abs(full[family]["mean_recovery"] - EXPECTED_FULL[family]) <= 0.03
        and full[family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")
    )
    pred_c = all(
        summaries["cue"][family]["mean_recovery"] > 0.0
        and fractions["cue"][family] >= 0.50
        and summaries["cue"][family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")
    )
    pred_d = all(
        abs(fractions[group][family]) <= 0.35
        for group in ("subject_onset", "self")
        for family in ("A1", "A2")
    )
    price = {
        "model_forwards": forward_calls,
        "example_evaluations": evaluations,
        "rows": len(rows),
        "arms": len(ARMS),
        "intervention_records": sum(len(records) for records in arm_records.values()),
        "fitted_scalars": 0,
        "grid_evaluations": 0,
        "root_evaluations": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }
    pred_e = price == {
        "model_forwards": 18,
        "example_evaluations": 576,
        "rows": 64,
        "arms": 7,
        "intervention_records": 448,
        "fitted_scalars": 0,
        "grid_evaluations": 0,
        "root_evaluations": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }
    predictions = {
        "pred_a_authority_capability_exact_partition": pred_a,
        "pred_b_complete_h1h4_recurrence": pred_b,
        "pred_c_literal_temporal_cue_dominates": pred_c,
        "pred_d_subject_onset_and_self_secondary": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else (
        "null" if pred_a and pred_b and pred_e else "invalid"
    )
    result = {
        "schema": "temporal_auxiliary_will_had_l9h1h4_source_groups_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": EXPECTED["prior"],
        "authority_sha256": EXPECTED,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "source_groups": list(source.GROUP_ORDER),
        "capability_cells": capability_cells,
        "instrument": {
            "manual_selected_logit_max_abs_error": manual_error,
            "selected_head_source_reconstruction_max_abs_error": reconstruction_error,
            "all_sources_vs_full_pair_selected_logit_max_abs_error": all_source_closure_error,
        },
        "summaries": summaries,
        "group_fraction_of_full": fractions,
        "arm_records": arm_records,
        "predictions": predictions,
        "price": price,
        "terminal": terminal,
        "reason": {
            "screen": "literal_temporal_cue_is_dominant_will_had_h1h4_source",
            "null": "cue_dominance_or_secondary_source_prediction_misses",
            "invalid": "authority_capability_replay_partition_closure_recurrence_coverage_or_price_invalid",
        }[terminal],
        "scope_boundary": "P/C selectivity is inherited from the exact parent head-group screen; upstream cue writers remain unlocalized.",
        "serial_seconds": time.perf_counter() - started,
        "next_action": (
            "test whether the cue-source H1/H4 edge composes with attention11"
            if terminal == "screen"
            else "retain the complete source partition and follow the empirically dominant non-cue group"
        ),
    }
    atomic_create_json(OUT, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "candidate_id",
                    "instrument",
                    "summaries",
                    "group_fraction_of_full",
                    "predictions",
                    "price",
                    "terminal",
                    "reason",
                    "scope_boundary",
                    "next_action",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
