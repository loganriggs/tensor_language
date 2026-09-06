#!/usr/bin/env python3
"""Exact residual-depth sweep for the dominant will/had subject-onset source bank."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_instrument pred_b_subject_onset_becomes_sufficient pred_c_early_onset_by_boundary5 pred_d_literal_cue_control_reported pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_das_subspace as das
import circuit_fast_screen_candidate_temporal_auxiliary as candidate
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_subject_onset_state_depth_v1.json"
SOURCE = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9h1h4_source_groups_v1_result.json"
FAST_SCREEN = ROOT / "circuits/fast_screens/temporal_auxiliary_will_vs_had_v1_result.json"
BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
SOURCE_EVAL = ROOT / "ops/attention_source_group_eval.py"
ONSET_EVAL = ROOT / "ops/residual_source_onset_eval.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_subject_onset_state_depth_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.subject_onset_state_depth_v1"
EXPECTED = {
    "prior": "42def715c67c630974954a1ec6c24ac1e00c58b8d2fc47c2f784808cddd0901b",
    "source": "c738d33cae91ee70f911ea5ea610d0156bc618b41b6574051d33ef6cd21e9f8d",
    "fast_screen": "71a833c52c39b98be6f576d11749522e2725c540a29d61a2352e315978684ec9",
    "builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
    "source_eval": "6ecf5f40b92f94cb32bccf1a703e527a3d468281936e63d4c7e91e8af66b4348",
    "onset_eval": "8f0235743c0450797ade21ac663d4bf735f1784931c2c60ef8f69d4f7cd113a7",
}
EXPECTED_ROWS_SHA256 = "b74ff65cf6b59142bf38172dae7f70a25e5f471bbeaa038f9b3971caedf913da"
GROUPS = ("subject_onset", "cue")
BOUNDARIES = tuple(range(10))
MODEL_FORWARDS = 44
EXAMPLE_EVALUATIONS = 1408
INTERVENTION_RECORDS = 1280
IDENTITY_TOLERANCE = 1.0e-4
INERT_TOLERANCE = 1.0e-7


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def pair_error(output, expected):
    return max(
        abs(float(actual) - float(reference))
        for actual_pair, reference_pair in zip(output.answer_foil, expected)
        for actual, reference in zip(actual_pair, reference_pair)
    )


def validate_static():
    paths = {
        "prior": PRIOR,
        "source": SOURCE,
        "fast_screen": FAST_SCREEN,
        "builder": BUILDER,
        "source_eval": SOURCE_EVAL,
        "onset_eval": ONSET_EVAL,
    }
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    prior = json.loads(PRIOR.read_text())
    source = json.loads(SOURCE.read_text())
    fast_screen = json.loads(FAST_SCREEN.read_text())
    rows_all = candidate.build_rows()
    if candidate.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("row authority changed")
    rows = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    capability = [
        cell for cell in fast_screen["run"]["capability_cells"]
        if cell["family"] in {"A1", "A2"}
    ]
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or source.get("terminal") not in {"screen", "null"}
        or not source.get("predictions", {}).get("pred_a_authority_capability_exact_partition")
        or fast_screen.get("terminal") != "screen"
        or len(rows) != 64
        or len(capability) != 4
        or not all(cell["passed"] for cell in capability)
    ):
        raise ExperimentError("population, capability, or parent evidence changed")
    return rows, fast_screen, capability


def dryrun_receipt():
    return {
        "candidate_id": CANDIDATE_ID,
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "rows": 64,
        "groups": list(GROUPS),
        "boundaries": list(BOUNDARIES),
        "model_forwards": MODEL_FORWARDS,
        "example_evaluations": EXAMPLE_EVALUATIONS,
        "intervention_records": INTERVENTION_RECORDS,
        "fitted_scalars": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }


def main():
    rows, fast_screen, capability_cells = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = utc_now()
    started = time.perf_counter()
    backend = onset.ResidualGroupBackend.load("cuda")
    frozen_native = {
        (str(item["row_id"]), str(item["side"])): (
            float(item["answer_logit"]), float(item["foil_logit"])
        )
        for item in fast_screen["run"]["native_logits"]
        if item["family"] in {"A1", "A2"}
    }
    batches = []
    manual_error = 0.0
    boundary0_state_max_abs = 0.0
    forward_calls = 0
    evaluations = 0

    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, base_states = backend.forward_states(base_batch, maximum_boundary=9)
        donor_output, donor_states = backend.forward_states(donor_batch, maximum_boundary=9)
        forward_calls += 2
        evaluations += 2 * len(family_rows)
        manual_error = max(
            manual_error,
            pair_error(base_output, [frozen_native[(row_id, "base")] for row_id in base_batch.row_ids]),
            pair_error(donor_output, [frozen_native[(row_id, "donor")] for row_id in donor_batch.row_ids]),
        )
        positions = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        for index, row_positions in enumerate(positions):
            for position in row_positions:
                difference = (
                    base_states[0][index, position].float()
                    - donor_states[0][index, position].float()
                ).abs().max()
                boundary0_state_max_abs = max(boundary0_state_max_abs, float(difference))
        batches.append(
            (family_rows, base_batch, donor_batch, base_output, donor_output, donor_states)
        )

    records = []
    boundary0_logit_max_abs = 0.0
    for group in GROUPS:
        for boundary in BOUNDARIES:
            for family_rows, base_batch, donor_batch, base_output, donor_output, donor_states in batches:
                patched_output, _ = backend.forward_states(
                    base_batch,
                    maximum_boundary=9,
                    donor_batch=donor_batch,
                    donor_states=donor_states,
                    boundary=boundary,
                    group_name=group,
                )
                forward_calls += 1
                evaluations += len(family_rows)
                if group == "subject_onset" and boundary == 0:
                    boundary0_logit_max_abs = max(
                        boundary0_logit_max_abs,
                        pair_error(patched_output, base_output.answer_foil),
                    )
                records.extend(
                    onset.recovery_records(
                        family_rows,
                        base_output,
                        donor_output,
                        patched_output,
                        group=group,
                        boundary=boundary,
                    )
                )

    curves = {
        group: onset.curve(records, group, BOUNDARIES)
        for group in GROUPS
    }
    subject_onset_boundary = onset.earliest_passing(curves["subject_onset"])
    pred_a = bool(
        manual_error <= IDENTITY_TOLERANCE
        and boundary0_state_max_abs <= INERT_TOLERANCE
        and boundary0_logit_max_abs <= INERT_TOLERANCE
        and all(cell["passed"] for cell in capability_cells)
    )
    pred_b = subject_onset_boundary is not None and 1 <= subject_onset_boundary <= 9
    pred_c = subject_onset_boundary is not None and subject_onset_boundary <= 5
    pred_d = len(curves["cue"]) == len(BOUNDARIES)
    unique_records = {
        (record["group"], record["boundary"], record["row_id"])
        for record in records
    }
    pred_e = bool(
        forward_calls == MODEL_FORWARDS
        and evaluations == EXAMPLE_EVALUATIONS
        and len(records) == INTERVENTION_RECORDS
        and len(unique_records) == INTERVENTION_RECORDS
        and all(math.isfinite(float(record["recovery"])) for record in records)
    )
    terminal = "screen" if pred_a and pred_b and pred_c and pred_e else (
        "null" if pred_a and pred_e else "invalid"
    )
    onset_regime = (
        "early_by_boundary5" if pred_c else
        "late_boundary6_to9" if pred_b else
        "absent_through_boundary9"
    )
    reason = {
        "screen": "subject_onset_source_state_is_sufficient_by_boundary5",
        "null": f"subject_onset_{onset_regime}",
        "invalid": "authority_capability_instrument_coverage_or_price_invalid",
    }[terminal]
    result = {
        "schema": "temporal_auxiliary_will_had_subject_onset_state_depth_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_capability_exact_instrument": pred_a,
            "pred_b_subject_onset_becomes_sufficient": pred_b,
            "pred_c_early_onset_by_boundary5": pred_c,
            "pred_d_literal_cue_control_reported": pred_d,
            "pred_e_exact_zero_fit_price": pred_e,
        },
        "score": {
            "manual_scored_logit_max_abs": manual_error,
            "boundary0_subject_onset_state_max_abs": boundary0_state_max_abs,
            "boundary0_subject_onset_scored_logit_max_abs": boundary0_logit_max_abs,
            "subject_onset_boundary": subject_onset_boundary,
            "onset_regime": onset_regime,
            "curves": curves,
            "capability_cells": capability_cells,
            "forward_calls": forward_calls,
            "example_evaluations": evaluations,
            "intervention_record_count": len(records),
            "fitted_scalars": 0,
            "grid_evaluations": 0,
            "root_evaluations": 0,
            "transformer_backwards": 0,
            "model_updates": 0,
        },
        "intervention_records": records,
        "terminal": terminal,
        "reason": reason,
        "next_action": (
            "factor the crossing block interval into attention and MLP writers at the exact subject-onset bank"
            if terminal == "screen"
            else "retain the exact onset classification and localize the later or distributed writer without changing the source bank"
        ),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "reason": reason,
        "predictions": result["predictions"],
        "subject_onset_boundary": subject_onset_boundary,
        "onset_regime": onset_regime,
        "subject_onset_curve": [point["mean_target_recovery"] for point in curves["subject_onset"]],
        "cue_curve": [point["mean_target_recovery"] for point in curves["cue"]],
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
