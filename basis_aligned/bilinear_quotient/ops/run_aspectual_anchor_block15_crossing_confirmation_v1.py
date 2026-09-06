#!/usr/bin/env python3
# BQGATE: frozen A-E disjoint block15 confirmation predictions; CUDA is managed-queue only.
"""Exact block15 crossing on the untouched confirmation half."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_battery_integration_contract as battery
import circuit_candidate_aspectual_lexical_holdout_v5 as holdout
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as parent
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_block15_crossing_confirmation_v1.json"
PARENT = ROOT / "circuits/followups/aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_block15_crossing_confirmation_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.block15_crossing_confirmation_v1"
EXPECTED_PRIOR_SHA256 = "224f9982609c750cf9547e0d75af3707117665230b9ef12ce39f138707d6264a"
EXPECTED_PARENT_SHA256 = "f534448e1b6e27195928d0e748147f43703225666fa102ece9d9a59d2f70c7ab"
EXPECTED_PARENT_RUNNER_SHA256 = "38c6ed4d8e2f7c66d7c3a48bbcafb1d0848927a6c502caa4f322b2e1b0867c4d"
EXPECTED_BUILDER_SHA256 = "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859"
EXPECTED_ROWS_SHA256 = "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
EXPECTED_CONFIRMATION_SHA256 = "ad198e745d3c2b900e097219aae918f9ec506271f159bdcdf9852db56e12e55b"
BOUNDARY = 15
MODEL_FORWARDS_MAX = 26
EXAMPLE_EVALUATIONS_MAX = 240


class ExperimentError(RuntimeError):
    pass


def sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    for file_path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256,
        PARENT: EXPECTED_PARENT_SHA256,
        PARENT_RUNNER: EXPECTED_PARENT_RUNNER_SHA256,
        BUILDER: EXPECTED_BUILDER_SHA256,
    }.items():
        if sha256(file_path) != digest:
            raise ExperimentError(f"authority hash changed: {file_path.name}")
    prior = json.loads(PRIOR.read_text())
    result = json.loads(PARENT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if result.get("terminal") != "screen":
        raise ExperimentError("selection result changed")
    if result["score"]["selection_increments"]["block15"] != 0.02125431588988444:
        raise ExperimentError("selection-ranked block15 increment changed")
    rows_all = holdout.build_rows()
    if holdout.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    confirmation = tuple(selected[16:])
    if parent.ids_sha256(confirmation) != EXPECTED_CONFIRMATION_SHA256:
        raise ExperimentError("confirmation split changed")
    parent_rows = parent_runner.candidate.build_rows(parent_runner.candidate.TASK_ID)
    parent_spec = parent_runner.build_spec(parent_rows)
    spec = replace(
        parent_spec,
        experiment_id="aspectual-anchor-block15-crossing-confirmation-v1",
        authority_sha256=EXPECTED_ROWS_SHA256,
        expected_fit_rows=len(rows_all),
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT", forward_calls=MODEL_FORWARDS_MAX,
            example_evaluations=EXAMPLE_EVALUATIONS_MAX,
            backward_calls=0, model_updates=0, evidence_bytes=32768,
        ),
    )
    enriched_all = screen.validate_fit_authority(spec, rows_all)
    confirmation = tuple(enriched_all[str(row["row_id"])] for row in confirmation)
    if len(confirmation) != 16 or len(parent.subsets()) != 8:
        raise ExperimentError("population or factorial changed")
    return confirmation, spec, result


def main() -> None:
    rows, spec, parent_result = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_block15_crossing_confirmation_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "row_count": len(rows),
        "boundary": BOUNDARY,
        "factors": list(parent.BASE_FACTORS),
        "factorial_arms": len(parent.subsets()),
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0,
        "model_updates": 0,
        "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = utc_now()
    started = time.perf_counter()
    backend = parent.SuffixBackend.load("cuda")
    native = {}
    arm_values = {
        subset: {"A1": [], "A2": []} for subset in parent.subsets()
    }
    ceiling_values = {"A1": [], "A2": []}
    writer_values = {"A1": [], "A2": []}
    raw_records = []
    manual_base_max_abs = 0.0
    writer_tensor_error_max_abs = 0.0
    boundary_tensor_error_max_abs = 0.0
    full_to_ceiling_max_abs = 0.0
    forward_calls = 0
    evaluations = 0

    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            base_native, base_bilinear = backend.capture_bilinear(base_batch)
            donor_native, donor_bilinear = backend.capture_bilinear(donor_batch)
            base_manual, base_suffix = backend.capture_suffix(base_batch)
            writer_output, hybrid_suffix, writer_error = backend.capture_writer_suffix(
                base_batch, donor_batch, base_bilinear, donor_bilinear
            )
            forward_calls += 4
            evaluations += 4 * len(chunk)
            writer_tensor_error_max_abs = max(writer_tensor_error_max_abs, writer_error)
            for reference, manual in zip(base_native.answer_foil, base_manual.answer_foil):
                manual_base_max_abs = max(
                    manual_base_max_abs,
                    abs(reference[0] - manual[0]), abs(reference[1] - manual[1]),
                )
            for side, output in (("base", base_native), ("donor", donor_native)):
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                        str(row["row_id"]), family, side, answer, foil
                    )
            for row, pair in zip(chunk, writer_output.answer_foil):
                answer, foil, value = parent.recovery(row, pair, native)
                writer_values[family].append(value)
                raw_records.append({
                    "arm_id": "writer_two_term", "family": family,
                    "row_id": str(row["row_id"]), "answer_logit": answer,
                    "foil_logit": foil, "recovery": value,
                })
            lambda0 = backend.model.transformer.h[BOUNDARY].lambdas[0]
            for i, query in enumerate(base_batch.semantic_positions):
                reconstructed = (
                    lambda0.float() * (
                        hybrid_suffix[f"resid{BOUNDARY}"][i, query].float()
                        - base_suffix[f"resid{BOUNDARY}"][i, query].float()
                    )
                    + hybrid_suffix[f"attention{BOUNDARY}"][i, query].float()
                    - base_suffix[f"attention{BOUNDARY}"][i, query].float()
                    + hybrid_suffix[f"mlp{BOUNDARY}"][i, query].float()
                    - base_suffix[f"mlp{BOUNDARY}"][i, query].float()
                )
                direct = (
                    hybrid_suffix[f"resid{BOUNDARY + 1}"][i, query].float()
                    - base_suffix[f"resid{BOUNDARY + 1}"][i, query].float()
                )
                boundary_tensor_error_max_abs = max(
                    boundary_tensor_error_max_abs, float((reconstructed - direct).abs().max())
                )
            outputs = {
                parent.arm_id(subset): backend.crossing(
                    base_batch, base_suffix, hybrid_suffix, BOUNDARY, subset
                ) for subset in parent.subsets()
            }
            outputs["direct_query_ceiling"] = backend.direct_query(
                base_batch, base_suffix, hybrid_suffix, BOUNDARY + 1
            )
            forward_calls += len(parent.subsets()) + 1
            evaluations += (len(parent.subsets()) + 1) * len(chunk)
            full_name = parent.arm_id(parent.BASE_FACTORS)
            for full_pair, direct_pair in zip(
                outputs[full_name].answer_foil, outputs["direct_query_ceiling"].answer_foil
            ):
                full_to_ceiling_max_abs = max(
                    full_to_ceiling_max_abs,
                    abs(full_pair[0] - direct_pair[0]), abs(full_pair[1] - direct_pair[1]),
                )
            for name, output in outputs.items():
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil, value = parent.recovery(row, pair, native)
                    if name == "direct_query_ceiling":
                        ceiling_values[family].append(value)
                    else:
                        subset = next(
                            item for item in parent.subsets() if parent.arm_id(item) == name
                        )
                        arm_values[subset][family].append(value)
                    raw_records.append({
                        "arm_id": name, "family": family, "row_id": str(row["row_id"]),
                        "answer_logit": answer, "foil_logit": foil, "recovery": value,
                    })

    summaries = {}
    targets = {}
    for subset in parent.subsets():
        families = {
            family: parent.summarize(arm_values[subset][family])
            for family in ("A1", "A2")
        }
        target = statistics.fmean(
            families[family]["mean_recovery"] for family in ("A1", "A2")
        )
        summaries[parent.arm_id(subset)] = {
            "factors": list(subset), "families": families,
            "mean_target_recovery": target,
        }
        targets[subset] = target
    shapley = {}
    for factor in parent.BASE_FACTORS:
        total = 0.0
        for subset in parent.subsets():
            if factor in subset:
                continue
            extended = tuple(
                item for item in parent.BASE_FACTORS if item in set(subset) | {factor}
            )
            weight = (
                math.factorial(len(subset))
                * math.factorial(len(parent.BASE_FACTORS) - len(subset) - 1)
                / math.factorial(len(parent.BASE_FACTORS))
            )
            total += weight * (targets[extended] - targets[subset])
        shapley[factor] = total
    full_name = parent.arm_id(parent.BASE_FACTORS)
    removal_damage = {
        factor: {
            family: summaries[full_name]["families"][family]["mean_recovery"]
            - summaries[parent.arm_id(tuple(item for item in parent.BASE_FACTORS if item != factor))]["families"][family]["mean_recovery"]
            for family in ("A1", "A2")
        }
        for factor in parent.BASE_FACTORS
    }
    writer_summary = {
        family: parent.summarize(writer_values[family]) for family in ("A1", "A2")
    }
    bound_writer = parent_result["score"]["writer"]["confirmation"]
    current_capability = all(
        native[(str(row["row_id"]), side)].margin > 0.0
        for row in rows for side in ("base", "donor")
    )
    pred_a = (
        current_capability and manual_base_max_abs <= 1.0e-4
        and writer_tensor_error_max_abs <= 2.0e-3
        and boundary_tensor_error_max_abs <= 0.04
        and full_to_ceiling_max_abs <= 0.125
    )
    pred_b = all(
        abs(writer_summary[family]["mean_recovery"] - bound_writer[family]["mean_recovery"]) <= 1.0e-6
        and writer_summary[family]["mean_recovery"] > 0.0
        and writer_summary[family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")
    )
    pred_c = (
        targets[parent.BASE_FACTORS] > targets[("carried",)]
        and shapley["attention"] + shapley["mlp"] >= 0.005
    )
    pred_d = any(
        all(removal_damage[factor][family] > 0.0 for family in ("A1", "A2"))
        for factor in ("attention", "mlp")
    )
    pred_e = (
        len(raw_records) == 160 and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = (
        "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e))
        else "null" if pred_a and pred_b and pred_e
        else "invalid"
    )
    reason = {
        "screen": "block15_secondary_suffix_amplifier_confirmed",
        "null": "block15_active_component_confirmation_failed",
        "invalid": "authority_split_capability_instrument_writer_or_coverage_invalid",
    }[terminal]
    ceiling_summary = {
        family: parent.summarize(ceiling_values[family]) for family in ("A1", "A2")
    }
    result = {
        "schema": "aspectual_anchor_block15_crossing_confirmation_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_split_capability_and_instrument": pred_a,
            "pred_b_writer_recurrence": pred_b,
            "pred_c_active_block15_confirmation": pred_c,
            "pred_d_new_factor_necessity": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "boundary": BOUNDARY, "factorial_arms": summaries,
            "factorial_shapley_target_recovery": shapley,
            "full_removal_damage": removal_damage,
            "direct_query_ceiling": {
                "families": ceiling_summary,
                "mean_target_recovery": statistics.fmean(
                    ceiling_summary[family]["mean_recovery"] for family in ("A1", "A2")
                ),
            },
            "writer": writer_summary,
            "manual_base_scored_logit_max_abs": manual_base_max_abs,
            "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "boundary_tensor_reconstruction_max_abs": boundary_tensor_error_max_abs,
            "full_to_direct_ceiling_scored_logit_max_abs": full_to_ceiling_max_abs,
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "raw_record_count": len(raw_records), "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records,
        "terminal": terminal, "reason": reason,
        "next_action": (
            "compile confirmed block11 and block15 suffix crossings into program v3"
            if terminal == "screen"
            else "retain block15 as selection-only"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason,
        "predictions": result["predictions"], "shapley": shapley,
        "full": targets[parent.BASE_FACTORS], "carried": targets[("carried",)],
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
