#!/usr/bin/env python3
"""Factor the fresh temporal writer's post-block9 subject-state bypass."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_bypass_cube pred_b_residual_bypass_recurrence pred_c_carried_entry_dominates_bypass pred_d_block9_subject_updates_are_nonessential pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as destination_source
import attention_source_group_eval as source_score
import block_component_state_eval as component
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_das_subspace as das
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_writer_block9_subject_bypass_cube_v1.json"
MEDIATION_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_reader_mediation_v1_result.json"
WRITER_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8h1_fresh_cue_path_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
COMPONENT_EVAL = ROOT / "ops/block_component_state_eval.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block9_subject_bypass_cube_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_writer_block9_subject_bypass_cube_v1"
EXPECTED = {
    "prior": "0f083e8f0187681a1555a344f7cad76a961fbdd27d3e33fd575636a0ddc2c99b",
    "mediation_result": "505ce01c9b88e7f088554db9fd8d0fa2eafeabe43067aa61cdd9f2681f420f68",
    "writer_result": "2da5c4b424b620bbfe24cc98049a0520429102b7d37de45d49a48ef887181641",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "component_eval": "c44c5c392475fa4ead02c6402e44e6f14620caa055737d968111da80621d0379",
}
EXPECTED_ROWS_SHA256 = "c35d7f2159c47620b6f36f12c4f6c41c1630afae61a7f101ab9ff74e4e7e5e0a"
SUBSETS = component.subsets()
FULL_TARGET = {"A1": 0.0952183357896019, "A2": 0.0579133270139523}
MODEL_FORWARDS = 20
EXAMPLE_EVALUATIONS = 640
INTERVENTION_RECORDS = 512


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {
        "prior": PRIOR,
        "mediation_result": MEDIATION_RESULT,
        "writer_result": WRITER_RESULT,
        "builder": BUILDER,
        "component_eval": COMPONENT_EVAL,
    }
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    mediated = json.loads(MEDIATION_RESULT.read_text())
    writer = json.loads(WRITER_RESULT.read_text())
    rows_all = candidate.build_rows()
    if candidate.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("fresh row authority changed")
    rows = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    if (
        mediated.get("terminal") != "null"
        or not mediated["predictions"].get("pred_a_authority_capability_exact_mediation_instrument")
        or not mediated["predictions"].get("pred_b_fresh_writer_effect_recurrence")
        or writer.get("terminal") != "screen"
        or len(rows) != 64
        or len(SUBSETS) != 8
    ):
        raise ExperimentError("population or partial-mediation authority changed")
    return rows, writer["capability_cells"]


def dryrun_receipt():
    return {
        "candidate_id": CANDIDATE_ID,
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "rows": 64,
        "components": list(component.COMPONENTS),
        "component_subsets": len(SUBSETS),
        "model_forwards": MODEL_FORWARDS,
        "example_evaluations": EXAMPLE_EVALUATIONS,
        "intervention_records": INTERVENTION_RECORDS,
        "fitted_scalars": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }


def capture_base_source_and_block(backend, batch):
    source_capture = {}
    attention = backend.model.transformer.h[8].attn
    heads = backend.model.config.n_head
    head_dim = backend.model.config.n_embd // heads

    def capture_inputs(_module, arguments):
        current = arguments[0]
        v1 = arguments[1] if len(arguments) > 1 else None
        pattern, value, reconstructed = destination_source._attention_terms(
            backend, attention, current, v1
        )
        source_capture.update(
            pattern=pattern.detach().clone(),
            value=value.detach().clone(),
            reconstructed=reconstructed.detach().clone(),
        )

    def capture_heads(_module, arguments):
        flattened = arguments[0]
        source_capture["head_output"] = flattened.detach().clone().view(
            len(batch.row_ids), flattened.shape[1], heads, head_dim
        )

    handles = [
        attention.register_forward_pre_hook(capture_inputs),
        attention.c_proj.register_forward_pre_hook(capture_heads),
    ]
    try:
        output, states, components, block_error = component.capture(
            backend,
            batch,
            9,
            lambda: backend.forward_states(batch, maximum_boundary=10),
        )
    finally:
        for handle in handles:
            handle.remove()
    required = {"pattern", "value", "reconstructed", "head_output"}
    if not required.issubset(source_capture):
        raise ExperimentError("joint base source/component capture incomplete")
    source_capture["reconstruction_max_abs"] = float(
        (source_capture["reconstructed"].float() - source_capture["head_output"].float())
        .abs().max()
    )
    return output, states, source_capture, components, block_error


def capture_writer_block(
    backend, base_batch, donor_batch, writer_base, writer_donor, destinations
):
    writer_hook = mediation.fixed_source_delta_hook(
        backend,
        base_batch,
        donor_batch,
        writer_base,
        writer_donor,
        destinations,
        ("cue",),
        selected_heads=(1,),
    )
    handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
    try:
        return component.capture(
            backend,
            base_batch,
            9,
            lambda: backend.forward_states(base_batch, maximum_boundary=10),
        )
    finally:
        handle.remove()


def main():
    rows, capability_cells = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = utc_now()
    started = time.perf_counter()
    backend = onset.ResidualGroupBackend.load("cuda")
    if backend.model.config.n_head != 9:
        raise ExperimentError("frozen head inventory changed")
    batches = []
    reconstruction_error = 0.0
    source_reconstruction_error = 0.0
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, base_states, writer_base, base_components, base_error = (
            capture_base_source_and_block(backend, base_batch)
        )
        donor_output, writer_donor = destination_source.capture_layer_attention(
            backend, donor_batch, layer=8
        )
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_output, writer_states, writer_components, writer_error = capture_writer_block(
            backend, base_batch, donor_batch, writer_base, writer_donor, destinations
        )
        forward_calls += 3
        evaluations += 3 * len(family_rows)
        reconstruction_error = max(reconstruction_error, base_error, writer_error)
        source_reconstruction_error = max(
            source_reconstruction_error,
            float(writer_base["reconstruction_max_abs"]),
            float(writer_donor["reconstruction_max_abs"]),
        )
        batches.append({
            "rows": family_rows,
            "base_batch": base_batch,
            "donor_batch": donor_batch,
            "base_output": base_output,
            "donor_output": donor_output,
            "writer_output": writer_output,
            "base_states": base_states,
            "writer_states": writer_states,
            "base_components": base_components,
            "writer_components": writer_components,
        })

    records = []
    empty_closure_error = 0.0
    full_state_closure_error = 0.0
    for subset in SUBSETS:
        arm = component.arm_id(subset)
        for item in batches:
            if not subset:
                output = item["base_output"]
                empty_closure_error = max(
                    empty_closure_error, 0.0
                )
            else:
                hybrid = component.assemble(
                    item["base_components"], item["writer_components"], subset
                )
                if subset == component.COMPONENTS:
                    full_state_closure_error = max(
                        full_state_closure_error,
                        float((hybrid.float() - item["writer_states"][10].float()).abs().max()),
                    )
                output, _states = backend.forward_states(
                    item["base_batch"],
                    maximum_boundary=10,
                    donor_batch=item["donor_batch"],
                    donor_states=tuple(hybrid for _ in range(11)),
                    boundary=10,
                    group_name="subject_onset",
                )
                forward_calls += 1
                evaluations += len(item["rows"])
            records.extend(source_score.recovery_records(
                item["rows"], item["base_output"], item["donor_output"], output, arm=arm
            ))

    summaries = {
        component.arm_id(subset): source_score.summarize_by_family([
            record for record in records if record["arm"] == component.arm_id(subset)
        ])
        for subset in SUBSETS
    }
    full_arm = component.arm_id(component.COMPONENTS)
    shapley = {}
    shapley_error = 0.0
    for family in ("A1", "A2"):
        accounting = component.factorial_accounting({
            subset: summaries[component.arm_id(subset)][family]["mean_recovery"]
            for subset in SUBSETS
        })
        shapley[family] = accounting["shapley"]
        shapley_error = max(shapley_error, float(accounting["efficiency_error"]))
    entry_arm = component.arm_id(("entry",))
    entry_retained = {
        family: summaries[entry_arm][family]["mean_recovery"]
        / summaries[full_arm][family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    leave_one_retained = {
        family: {
            "omit_attention": summaries[component.arm_id(("entry", "mlp"))][family]["mean_recovery"]
            / summaries[full_arm][family]["mean_recovery"],
            "omit_mlp": summaries[component.arm_id(("entry", "attention"))][family]["mean_recovery"]
            / summaries[full_arm][family]["mean_recovery"],
        }
        for family in ("A1", "A2")
    }
    pred_a = bool(
        all(cell["passed"] for cell in capability_cells)
        and max(
            reconstruction_error,
            source_reconstruction_error,
            empty_closure_error,
            full_state_closure_error,
            shapley_error,
        ) <= 1.0e-4
        and len(records) == INTERVENTION_RECORDS
        and all(math.isfinite(float(record["recovery"])) for record in records)
    )
    pred_b = all(
        abs(summaries[full_arm][family]["mean_recovery"] - FULL_TARGET[family]) <= 0.03
        and summaries[full_arm][family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")
    )
    pred_c = all(
        shapley[family]["entry"] > 0.0
        and shapley[family]["entry"] > shapley[family]["attention"]
        and shapley[family]["entry"] > shapley[family]["mlp"]
        and entry_retained[family] >= 0.75
        for family in ("A1", "A2")
    )
    pred_d = all(
        retained >= 0.75
        for family in ("A1", "A2")
        for retained in leave_one_retained[family].values()
    )
    pred_e = bool(
        forward_calls == MODEL_FORWARDS
        and evaluations == EXAMPLE_EVALUATIONS
        and len(records) == INTERVENTION_RECORDS
        and len({(record["arm"], record["row_id"]) for record in records})
        == INTERVENTION_RECORDS
    )
    predictions = {
        "pred_a_authority_capability_exact_bypass_cube": pred_a,
        "pred_b_residual_bypass_recurrence": pred_b,
        "pred_c_carried_entry_dominates_bypass": pred_c,
        "pred_d_block9_subject_updates_are_nonessential": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else (
        "null" if pred_a and pred_b and pred_e else "invalid"
    )
    reason = {
        "screen": "fresh_writer_bypass_is_carried_subject_state_across_block9",
        "null": "registered_carried_subject_bypass_prediction_missed",
        "invalid": "authority_capability_component_closure_recurrence_coverage_or_price_invalid",
    }[terminal]
    result = {
        "schema": "temporal_auxiliary_will_had_fresh_writer_block9_subject_bypass_cube_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "dryrun": dryrun,
        "instrument": {
            "block_component_reconstruction_max_abs": reconstruction_error,
            "writer_source_reconstruction_max_abs": source_reconstruction_error,
            "empty_base_closure_max_abs": empty_closure_error,
            "full_writer_subject_state_closure_max_abs": full_state_closure_error,
            "shapley_efficiency_max_abs": shapley_error,
        },
        "summaries": summaries,
        "component_shapley": shapley,
        "entry_retained_fraction": entry_retained,
        "leave_one_update_retained_fraction": leave_one_retained,
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
        "records": records,
        "terminal": terminal,
        "reason": reason,
        "next_action": (
            "localize the first downstream consumer of the carried subject-state bypass after block9"
            if terminal == "screen"
            else "retain the partial reader path and follow the observed bypass component"
        ),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        key: result[key]
        for key in (
            "candidate_id", "instrument", "component_shapley", "entry_retained_fraction",
            "leave_one_update_retained_fraction", "predictions", "price", "terminal",
            "reason", "next_action"
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
