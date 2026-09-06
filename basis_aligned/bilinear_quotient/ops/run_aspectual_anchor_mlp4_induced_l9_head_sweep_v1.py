#!/usr/bin/env python3
# BQGATE: frozen A-E all-head mediation predictions; CUDA is managed-queue only.
"""Singleton and leave-one-out L9 heads for the two-term MLP4 signal."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_fast_screen_candidate_aspectual as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_block4_contextual_source_writer_factorial_v1 as block4
import run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1 as path
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp4_induced_l9_head_sweep_v1.json"
PARENT = ROOT / "circuits/followups/aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp4_induced_l9_head_sweep_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp4_induced_l9_head_sweep_v1"
EXPECTED_PRIOR_SHA256 = "d25d9ca7c01b0ad68f7b834941ab652c4a2dfe3bcc8e6e3bda548b6f802349b6"
EXPECTED_PARENT_SHA256 = "649cc961fd4203a9d7489344bbf169754081a288b5d575bcefcab2caf41da9ab"
EXPECTED_PARENT_RUNNER_SHA256 = "5449d45505267f2ebc92c9285b7a984fce773f834c2071e1d7e6d1a9f1949372"
EXPECTED_AUTHORITY_SHA256 = "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3"
HEADS = tuple(range(9))
WRITER_FACTORS = ("left_change", "right_change")
ARMS = (
    "writer_two_term",
    "all_heads",
    "h1_h4",
    *(f"h{head}" for head in HEADS),
    *(f"all_except_h{head}" for head in HEADS),
)
MODEL_FORWARDS_MAX = 50
EXAMPLE_EVALUATIONS_MAX = 1600


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
    }.items():
        if sha256(file_path) != digest:
            raise ExperimentError(f"authority hash changed: {file_path.name}")
    prior = json.loads(PRIOR.read_text())
    parent = json.loads(PARENT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if parent.get("terminal") != "null":
        raise ExperimentError("parent terminal changed")
    if parent["score"]["bank_to_writer_retained_fraction"] >= 0.40:
        raise ExperimentError("parent null basis changed")
    rows = candidate.build_rows(candidate.TASK_ID)
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    spec = parent_runner.build_spec(rows)
    enriched_all = screen.validate_fit_authority(spec, rows)
    enriched = tuple(enriched_all[str(row["row_id"])] for row in selected)
    if len(enriched) != 64 or len(ARMS) != 21:
        raise ExperimentError("population or arm inventory changed")
    return enriched, spec


class HeadSweepBackend(path.PathBackend):
    def mediate_heads(
        self,
        base_batch: producer.ModelBatch,
        hybrid_attention,
        selected_heads: tuple[int, ...] | None,
    ):
        if selected_heads is not None and (
            len(selected_heads) != len(set(selected_heads))
            or any(head not in HEADS for head in selected_heads)
        ):
            raise ExperimentError("head subset changed")
        head_dim = self.model.config.n_embd // self.model.config.n_head

        def patch_heads(_module, arguments):
            flattened = arguments[0]
            head_output = flattened.view(
                len(base_batch.row_ids),
                flattened.shape[1],
                self.model.config.n_head,
                head_dim,
            ).clone()
            for i, q in enumerate(base_batch.semantic_positions):
                if selected_heads is None:
                    head_output[i, q] = hybrid_attention["head_output"][i, q]
                else:
                    for head in selected_heads:
                        head_output[i, q, head] = hybrid_attention["head_output"][i, q, head]
            return (head_output.reshape_as(flattened),) + tuple(arguments[1:])

        handle = self.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(
            patch_heads
        )
        try:
            output, _ = self.manual_forward(base_batch)
        finally:
            handle.remove()
        return output


def summarize(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("recovery missing or nonfinite")
    return {
        "count": len(values),
        "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def main() -> None:
    rows, spec = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_mlp4_induced_l9_head_sweep_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "row_count": len(rows),
        "heads": list(HEADS),
        "arms": list(ARMS),
        "direct_closure_controls": 1,
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
    backend = HeadSweepBackend.load("cuda")
    native = {}
    arm_values = {arm: {"A1": [], "A2": []} for arm in ARMS}
    logits = {}
    raw_records = []
    empty_max_abs = 0.0
    tensor_error_max_abs = 0.0
    attention_reconstruction_max_abs = 0.0
    all_heads_direct_max_abs = 0.0
    forward_calls = 0
    evaluations = 0

    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            base_output, base_capture = backend.capture_bilinear(base_batch)
            donor_output, donor_capture = backend.capture_bilinear(donor_batch)
            forward_calls += 2
            evaluations += 2 * len(chunk)
            for side, output in (("base", base_output), ("donor", donor_output)):
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                        str(row["row_id"]), family, side, answer, foil
                    )

            empty_output, base_attention, tensor_error = backend.capture_writer(
                base_batch, donor_batch, base_capture, donor_capture, ()
            )
            writer_output, hybrid_attention, tensor_error_2 = backend.capture_writer(
                base_batch, donor_batch, base_capture, donor_capture, WRITER_FACTORS
            )
            forward_calls += 2
            evaluations += 2 * len(chunk)
            tensor_error_max_abs = max(tensor_error_max_abs, tensor_error, tensor_error_2)
            attention_reconstruction_max_abs = max(
                attention_reconstruction_max_abs,
                float(base_attention["reconstruction_max_abs"]),
                float(hybrid_attention["reconstruction_max_abs"]),
            )
            for native_pair, empty_pair in zip(base_output.answer_foil, empty_output.answer_foil):
                empty_max_abs = max(
                    empty_max_abs,
                    abs(native_pair[0] - empty_pair[0]),
                    abs(native_pair[1] - empty_pair[1]),
                )

            head_sets = {
                "all_heads": HEADS,
                "h1_h4": (1, 4),
                **{f"h{head}": (head,) for head in HEADS},
                **{
                    f"all_except_h{head}": tuple(other for other in HEADS if other != head)
                    for head in HEADS
                },
            }
            outputs = {"writer_two_term": writer_output}
            for arm, selected_heads in head_sets.items():
                outputs[arm] = backend.mediate_heads(
                    base_batch, hybrid_attention, selected_heads
                )
            direct_output = backend.mediate_heads(base_batch, hybrid_attention, None)
            forward_calls += len(head_sets) + 1
            evaluations += (len(head_sets) + 1) * len(chunk)
            for selected, direct in zip(
                outputs["all_heads"].answer_foil, direct_output.answer_foil
            ):
                all_heads_direct_max_abs = max(
                    all_heads_direct_max_abs,
                    abs(selected[0] - direct[0]),
                    abs(selected[1] - direct[1]),
                )
            for arm, output in outputs.items():
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    row_id = str(row["row_id"])
                    recovery = kernel.signed_pairwise_donor_recovery(
                        -native[(row_id, "base")].margin,
                        native[(row_id, "donor")].margin,
                        -(answer - foil),
                    )
                    arm_values[arm][family].append(recovery)
                    logits[(arm, row_id)] = (answer, foil)
                    raw_records.append({
                        "arm_id": arm,
                        "family": family,
                        "row_id": row_id,
                        "answer_logit": answer,
                        "foil_logit": foil,
                        "recovery": recovery,
                    })

    native_capability = True
    for family in ("A1", "A2"):
        for direction in ("past_to_present", "present_to_past"):
            cell_rows = [
                row for row in rows
                if row["transform_id"] == family and row["direction_id"] == direction
            ]
            for side in ("base", "donor"):
                accuracy = sum(
                    native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows
                ) / len(cell_rows)
                native_capability = native_capability and accuracy >= 0.85

    summaries = {}
    targets = {}
    for arm in ARMS:
        families = {family: summarize(arm_values[arm][family]) for family in ("A1", "A2")}
        target = statistics.fmean(
            families[family]["mean_recovery"] for family in ("A1", "A2")
        )
        summaries[arm] = {"families": families, "mean_target_recovery": target}
        targets[arm] = target

    attributions = {}
    all_target = targets["all_heads"]
    for head in HEADS:
        singleton = targets[f"h{head}"]
        necessity = all_target - targets[f"all_except_h{head}"]
        attributions[f"h{head}"] = {
            "singleton_recovery": singleton,
            "full_minus_leave_one_out_recovery": necessity,
            "endpoint_average": 0.5 * (singleton + necessity),
            "family_singleton": {
                family: summaries[f"h{head}"]["families"][family]["mean_recovery"]
                for family in ("A1", "A2")
            },
            "family_full_minus_leave_one_out": {
                family: summaries["all_heads"]["families"][family]["mean_recovery"]
                - summaries[f"all_except_h{head}"]["families"][family]["mean_recovery"]
                for family in ("A1", "A2")
            },
        }
    ranking = sorted(
        (f"h{head}" for head in HEADS),
        key=lambda name: (-attributions[name]["endpoint_average"], name),
    )
    additional_heads = [
        name for name in ranking
        if name not in {"h1", "h4"}
        and attributions[name]["endpoint_average"] >= 0.03
        and all(value > 0.0 for value in attributions[name]["family_singleton"].values())
        and all(
            value > 0.0
            for value in attributions[name]["family_full_minus_leave_one_out"].values()
        )
    ]
    writer_target = targets["writer_two_term"]
    all_retained = all_target / writer_target

    pred_a = (
        native_capability
        and empty_max_abs <= 1.0e-4
        and tensor_error_max_abs <= 2.0e-3
        and attention_reconstruction_max_abs <= 1.0e-4
        and all_heads_direct_max_abs <= 0.125
    )
    pred_b = (
        abs(writer_target - 0.33379277118533013) <= 0.02
        and abs(targets["h1_h4"] - 0.13009089135863688) <= 0.02
        and all(
            summaries[arm]["families"][family]["mean_recovery"] > 0.0
            and summaries[arm]["families"][family]["direction_fraction"] >= 0.80
            for arm in ("writer_two_term", "h1_h4")
            for family in ("A1", "A2")
        )
    )
    pred_c = all_retained >= 0.50 and all(
        summaries["all_heads"]["families"][family]["mean_recovery"] > 0.0
        and summaries["all_heads"]["families"][family]["direction_fraction"] >= 0.80
        for family in ("A1", "A2")
    )
    pred_d = bool(additional_heads)
    expected_records = len(ARMS) * len(rows)
    pred_e = (
        len(raw_records) == expected_records
        and len(logits) == expected_records
        and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = (
        "screen" if pred_a and pred_b and pred_c and pred_d and pred_e
        else ("null" if pred_a and pred_b and pred_e else "invalid")
    )
    reason = {
        "screen": "additional_l9_heads_carry_mlp4_induced_signal",
        "null": "no_additional_direct_l9_head_route",
        "invalid": "head_sweep_instrument_recurrence_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_mlp4_induced_l9_head_sweep_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_exact_instrument": pred_a,
            "pred_b_writer_and_h1h4_recurrence": pred_b,
            "pred_c_all_head_mediation": pred_c,
            "pred_d_additional_head": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "empty_hook_scored_logit_max_abs": empty_max_abs,
            "bilinear_tensor_reconstruction_max_abs": tensor_error_max_abs,
            "attention_source_reconstruction_max_abs": attention_reconstruction_max_abs,
            "all_heads_to_direct_preprojection_scored_logit_max_abs": all_heads_direct_max_abs,
            "arms": summaries,
            "all_heads_to_writer_retained_fraction": all_retained,
            "head_endpoint_attributions": attributions,
            "head_endpoint_ranking": ranking,
            "licensed_additional_heads": additional_heads,
            "forward_calls": forward_calls,
            "example_evaluations": evaluations,
            "raw_record_count": len(raw_records),
            "model_backwards": 0,
            "model_updates": 0,
            "fit_parameters": 0,
        },
        "intervention_logits": raw_records,
        "terminal": terminal,
        "reason": reason,
        "next_action": (
            "factor the source positions read by the licensed additional L9 heads"
            if terminal == "screen"
            else "search downstream routes outside direct L9 final-query heads"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "reason": reason,
        "predictions": result["predictions"],
        "all_heads_to_writer": all_retained,
        "ranking": ranking,
        "additional_heads": additional_heads,
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
