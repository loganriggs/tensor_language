#!/usr/bin/env python3
"""Exact block-8 component cube at the will/had subject-onset source bank."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_cube pred_b_boundary9_effect_recurrence pred_c_attention8_is_dominant_writer pred_d_attention8_is_material pred_e_exact_zero_fit_price
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
import run_aspectual_tense_h1h4_deep_resid9_block8_factorial_v1 as block_cube


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_subject_onset_block8_component_cube_v1.json"
DEPTH_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_subject_onset_state_depth_v1_result.json"
FAST_SCREEN = ROOT / "circuits/fast_screens/temporal_auxiliary_will_vs_had_v1_result.json"
BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
SOURCE_EVAL = ROOT / "ops/attention_source_group_eval.py"
ONSET_EVAL = ROOT / "ops/residual_source_onset_eval.py"
CAPTURE = ROOT / "ops/run_aspectual_tense_h1h4_deep_resid9_block8_factorial_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_subject_onset_block8_component_cube_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.subject_onset_block8_component_cube_v1"
EXPECTED = {
    "prior": "803cab93d6bf97cb52a05beb357944a10c115aec88ce9ce8ecb9187e83ca4550",
    "depth_result": "b1f5ca50c669019684b5308b10dcd083de67897327bc6eecbcd92c978fc9cdae",
    "fast_screen": "71a833c52c39b98be6f576d11749522e2725c540a29d61a2352e315978684ec9",
    "builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
    "source_eval": "6ecf5f40b92f94cb32bccf1a703e527a3d468281936e63d4c7e91e8af66b4348",
    "onset_eval": "8f0235743c0450797ade21ac663d4bf735f1784931c2c60ef8f69d4f7cd113a7",
    "capture": "a4424a1d4eeea8b34b1b5cf5adefd81f05d7d3479808463389bfceb6460c7474",
}
EXPECTED_ROWS_SHA256 = "b74ff65cf6b59142bf38172dae7f70a25e5f471bbeaa038f9b3971caedf913da"
BRANCHES = block_cube.BRANCHES
SUBSETS = block_cube.subsets()
FULL_TARGET = {"A1": 0.6445113916817302, "A2": 0.5051666348472565}
MODEL_FORWARDS = 20
EXAMPLE_EVALUATIONS = 640
INTERVENTION_RECORDS = 512
EXACT_TOLERANCE = 2.0e-4


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def arm_id(subset):
    return block_cube.arm_id(subset)


def pair_error(output, expected):
    return max(
        abs(float(actual) - float(reference))
        for actual_pair, reference_pair in zip(output.answer_foil, expected)
        for actual, reference in zip(actual_pair, reference_pair)
    )


def validate_static():
    paths = {
        "prior": PRIOR,
        "depth_result": DEPTH_RESULT,
        "fast_screen": FAST_SCREEN,
        "builder": BUILDER,
        "source_eval": SOURCE_EVAL,
        "onset_eval": ONSET_EVAL,
        "capture": CAPTURE,
    }
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    depth = json.loads(DEPTH_RESULT.read_text())
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
        depth.get("reason") != "subject_onset_late_boundary6_to9"
        or depth["score"].get("subject_onset_boundary") != 9
        or fast_screen.get("terminal") != "screen"
        or len(rows) != 64
        or len(capability) != 4
        or not all(cell["passed"] for cell in capability)
        or len(SUBSETS) != 8
    ):
        raise ExperimentError("population, capability, or parent localization changed")
    return rows, fast_screen, capability


def dryrun_receipt():
    return {
        "candidate_id": CANDIDATE_ID,
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "rows": 64,
        "components": list(BRANCHES),
        "component_subsets": len(SUBSETS),
        "model_forwards": MODEL_FORWARDS,
        "example_evaluations": EXAMPLE_EVALUATIONS,
        "intervention_records": INTERVENTION_RECORDS,
        "fitted_scalars": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }


def component_states(raw):
    return {
        BRANCHES[0]: raw["z8"].float(),
        BRANCHES[1]: raw["attention8"].float(),
        BRANCHES[2]: raw["mlp8"].float(),
    }


def assembled_state(base_components, donor_components, subset):
    selected = set(subset)
    return sum(
        (
            donor_components[branch] if branch in selected else base_components[branch]
            for branch in BRANCHES
        ),
        next(iter(base_components.values())).new_zeros(next(iter(base_components.values())).shape),
    )


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
    recombination_error = 0.0
    full_state_closure_error = 0.0
    forward_calls = 0
    evaluations = 0

    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        captures = {}
        outputs = {}
        for side, batch in (("base", base_batch), ("donor", donor_batch)):
            output, _capture, _logits, raw = block_cube.capture_components(backend, batch)
            outputs[side] = output
            captures[side] = component_states(raw)
            expected = [frozen_native[(row_id, side)] for row_id in batch.row_ids]
            manual_error = max(manual_error, pair_error(output, expected))
            x9 = sum(captures[side].values(), raw["z8"].float().new_zeros(raw["z8"].shape))
            block9 = backend.model.transformer.h[9]
            deep9 = raw["z9"].float() - block9.lambdas[1].detach().float() * raw["x0"].float()
            recombination_error = max(
                recombination_error,
                float((block9.lambdas[0].detach().float() * x9 - deep9).abs().max()),
            )
            forward_calls += 1
            evaluations += len(family_rows)
        full_hybrid = assembled_state(captures["base"], captures["donor"], BRANCHES)
        donor_x9 = sum(
            captures["donor"].values(),
            next(iter(captures["donor"].values())).new_zeros(next(iter(captures["donor"].values())).shape),
        )
        full_state_closure_error = max(
            full_state_closure_error,
            float((full_hybrid - donor_x9).abs().max()),
        )
        batches.append((family_rows, base_batch, donor_batch, outputs, captures))

    records = []
    empty_logit_closure = 0.0
    for subset in SUBSETS:
        arm = arm_id(subset)
        for family_rows, base_batch, donor_batch, outputs, captures in batches:
            hybrid = assembled_state(captures["base"], captures["donor"], subset)
            patched_output, _ = backend.forward_states(
                base_batch,
                maximum_boundary=9,
                donor_batch=donor_batch,
                donor_states=tuple(hybrid for _ in range(10)),
                boundary=9,
                group_name="subject_onset",
            )
            forward_calls += 1
            evaluations += len(family_rows)
            if not subset:
                empty_logit_closure = max(
                    empty_logit_closure,
                    pair_error(patched_output, outputs["base"].answer_foil),
                )
            records.extend(
                onset.recovery_records(
                    family_rows,
                    outputs["base"],
                    outputs["donor"],
                    patched_output,
                    group=arm,
                    boundary=9,
                )
            )

    summaries = {
        arm_id(subset): {
            family: onset.summarize([
                record for record in records
                if record["group"] == arm_id(subset) and record["family"] == family
            ])
            for family in ("A1", "A2")
        }
        for subset in SUBSETS
    }
    shapley = {}
    shapley_error = 0.0
    for family in ("A1", "A2"):
        values = {
            subset: summaries[arm_id(subset)][family]["mean_recovery"]
            for subset in SUBSETS
        }
        accounting = block_cube.factorial_accounting(values)
        shapley[family] = accounting["shapley"]
        shapley_error = max(shapley_error, float(accounting["efficiency_error"]))

    full_arm = arm_id(BRANCHES)
    omit_attention = arm_id(tuple(branch for branch in BRANCHES if branch != "attention8_update"))
    attention_omission_loss = {
        family: summaries[full_arm][family]["mean_recovery"]
        - summaries[omit_attention][family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    pred_a = bool(
        all(cell["passed"] for cell in capability_cells)
        and max(manual_error, recombination_error, full_state_closure_error, empty_logit_closure, shapley_error) <= EXACT_TOLERANCE
        and len(records) == INTERVENTION_RECORDS
        and all(math.isfinite(float(record["recovery"])) for record in records)
    )
    pred_b = all(
        abs(summaries[full_arm][family]["mean_recovery"] - FULL_TARGET[family]) <= 0.03
        and summaries[full_arm][family]["direction_fraction"] == 1.0
        for family in ("A1", "A2")
    )
    pred_c = all(
        shapley[family]["attention8_update"] > 0.0
        and shapley[family]["attention8_update"] > shapley[family]["block8_entry_z8"]
        and shapley[family]["attention8_update"] > shapley[family]["mlp8_update"]
        for family in ("A1", "A2")
    )
    pred_d = all(attention_omission_loss[family] >= 0.10 for family in ("A1", "A2"))
    pred_e = bool(
        forward_calls == MODEL_FORWARDS
        and evaluations == EXAMPLE_EVALUATIONS
        and len(records) == INTERVENTION_RECORDS
        and len({(record["group"], record["row_id"]) for record in records}) == INTERVENTION_RECORDS
    )
    predictions = {
        "pred_a_authority_capability_exact_cube": pred_a,
        "pred_b_boundary9_effect_recurrence": pred_b,
        "pred_c_attention8_is_dominant_writer": pred_c,
        "pred_d_attention8_is_material": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else (
        "null" if pred_a and pred_b and pred_e else "invalid"
    )
    reason = {
        "screen": "block8_attention_is_dominant_material_subject_onset_writer",
        "null": "registered_block8_attention_writer_prediction_missed",
        "invalid": "authority_capability_closure_recurrence_coverage_or_price_invalid",
    }[terminal]
    result = {
        "schema": "temporal_auxiliary_will_had_subject_onset_block8_component_cube_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "dryrun": dryrun,
        "instrument": {
            "manual_scored_logit_max_abs": manual_error,
            "block8_component_recombination_max_abs": recombination_error,
            "full_state_closure_max_abs": full_state_closure_error,
            "empty_scored_logit_closure_max_abs": empty_logit_closure,
            "shapley_efficiency_max_abs": shapley_error,
        },
        "summaries": summaries,
        "component_shapley": shapley,
        "attention_omission_loss": attention_omission_loss,
        "predictions": predictions,
        "price": {
            "model_forwards": forward_calls,
            "example_evaluations": evaluations,
            "intervention_records": len(records),
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
            "partition block8 attention by head and exact source group at the subject-onset destinations"
            if terminal == "screen"
            else "retain the exact cube and follow the observed component winner without promoting attention"
        ),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        key: result[key]
        for key in (
            "candidate_id", "instrument", "component_shapley", "attention_omission_loss",
            "predictions", "price", "terminal", "reason", "next_action"
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
