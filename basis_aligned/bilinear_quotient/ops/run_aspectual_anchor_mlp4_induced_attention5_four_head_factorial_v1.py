#!/usr/bin/env python3
# BQGATE: frozen A-E attention5 four-head factorial; CUDA is managed-queue only.
"""Exact H7/H1/H6/H8 factorial for the distributed attention5 transporter."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
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
import run_aspectual_anchor_mlp4_induced_attention5_head_sweep_v1 as head_sweep
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp4_induced_attention5_four_head_factorial_v1.json"
PARENT = ROOT / "circuits/followups/aspectual_anchor_mlp4_induced_attention5_head_sweep_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp4_induced_attention5_head_sweep_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp4_induced_attention5_four_head_factorial_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp4_induced_attention5_four_head_factorial_v1"
EXPECTED_PRIOR_SHA256 = "3529235bf882954e779a9ac13e39eeeab66c10fe16147bc9f8a45eb32cdab2d1"
EXPECTED_PARENT_SHA256 = "5dc3c6d7ae7eb4bb966d55bdf2ccffea9389e5b3fe075e2e66ec0686e2972eaa"
EXPECTED_PARENT_RUNNER_SHA256 = "8d81665f3533233fb331535dd2b80d7a43bbbc2f05b2541d586229b156f6b4c2"
EXPECTED_AUTHORITY_SHA256 = "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3"
SELECTED_HEADS = (7, 1, 6, 8)
ALL_HEADS = tuple(range(9))
WRITER_FACTORS = ("left_change", "right_change")
MODEL_FORWARDS_MAX = 44
EXAMPLE_EVALUATIONS_MAX = 1408


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def subsets():
    return tuple(
        subset
        for width in range(len(SELECTED_HEADS) + 1)
        for subset in itertools.combinations(SELECTED_HEADS, width)
    )


def arm_id(subset: tuple[int, ...]) -> str:
    return "empty" if not subset else "+".join(f"h{head}" for head in subset)


def validate_static():
    for path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256,
        PARENT: EXPECTED_PARENT_SHA256,
        PARENT_RUNNER: EXPECTED_PARENT_RUNNER_SHA256,
    }.items():
        if sha256(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    parent = json.loads(PARENT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if parent.get("terminal") != "null":
        raise ExperimentError("parent terminal changed")
    if parent["score"]["head_endpoint_ranking"][:4] != [
        "h7", "h1", "h6", "h8"
    ]:
        raise ExperimentError("sealed head ranking changed")
    if parent["score"]["licensed_attention5_heads"]:
        raise ExperimentError("parent unexpectedly licensed a singleton")
    rows = candidate.build_rows(candidate.TASK_ID)
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    spec = parent_runner.build_spec(rows)
    enriched_all = screen.validate_fit_authority(spec, rows)
    enriched = tuple(enriched_all[str(row["row_id"])] for row in selected)
    if len(enriched) != 64 or len(subsets()) != 16:
        raise ExperimentError("population or factorial changed")
    return enriched, spec


class FactorialBackend(head_sweep.Attention5Backend):
    def mediate_heads(
        self,
        base_batch: producer.ModelBatch,
        hybrid_attention,
        selected_heads: tuple[int, ...],
    ):
        if (
            len(selected_heads) != len(set(selected_heads))
            or any(head not in ALL_HEADS for head in selected_heads)
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
            for i, query in enumerate(base_batch.semantic_positions):
                for head in selected_heads:
                    head_output[i, query, head] = hybrid_attention["head_output"][
                        i, query, head
                    ]
            return (head_output.reshape_as(flattened),) + tuple(arguments[1:])

        handle = self.model.transformer.h[5].attn.c_proj.register_forward_pre_hook(
            patch_heads
        )
        try:
            return self.native(base_batch, capture=False)
        finally:
            handle.remove()


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
        "schema": "aspectual_anchor_mlp4_induced_attention5_four_head_factorial_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "row_count": len(rows),
        "selected_heads": list(SELECTED_HEADS),
        "factorial_arm_count": len(subsets()),
        "all_head_ceiling_arms": 1,
        "writer_arms": 1,
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
    backend = FactorialBackend.load("cuda")
    native = {}
    factorial_values = {
        subset: {"A1": [], "A2": []} for subset in subsets()
    }
    all_head_values = {"A1": [], "A2": []}
    writer_values = {"A1": [], "A2": []}
    logits = {}
    raw_records = []
    empty_capture_max_abs = 0.0
    tensor_error_max_abs = 0.0
    forward_calls = 0
    evaluations = 0

    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            base_output, base_capture = backend.capture_bilinear(base_batch)
            donor_output, donor_capture = backend.capture_bilinear(donor_batch)
            empty_output, _, tensor_error = backend.capture_writer(
                base_batch, donor_batch, base_capture, donor_capture, ()
            )
            writer_output, hybrid_attention, tensor_error_2 = backend.capture_writer(
                base_batch, donor_batch, base_capture, donor_capture, WRITER_FACTORS
            )
            forward_calls += 4
            evaluations += 4 * len(chunk)
            tensor_error_max_abs = max(
                tensor_error_max_abs, tensor_error, tensor_error_2
            )
            for reference, empty in zip(base_output.answer_foil, empty_output.answer_foil):
                empty_capture_max_abs = max(
                    empty_capture_max_abs,
                    abs(reference[0] - empty[0]),
                    abs(reference[1] - empty[1]),
                )
            for side, output in (("base", base_output), ("donor", donor_output)):
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                        str(row["row_id"]), family, side, answer, foil
                    )

            outputs = {
                arm_id(subset): backend.mediate_heads(
                    base_batch, hybrid_attention, subset
                )
                for subset in subsets()
            }
            outputs["all_nine_heads"] = backend.mediate_heads(
                base_batch, hybrid_attention, ALL_HEADS
            )
            outputs["writer_two_term"] = writer_output
            forward_calls += len(subsets()) + 1
            evaluations += (len(subsets()) + 1) * len(chunk)
            for arm, output in outputs.items():
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    row_id = str(row["row_id"])
                    recovery = kernel.signed_pairwise_donor_recovery(
                        -native[(row_id, "base")].margin,
                        native[(row_id, "donor")].margin,
                        -(answer - foil),
                    )
                    if arm == "all_nine_heads":
                        all_head_values[family].append(recovery)
                    elif arm == "writer_two_term":
                        writer_values[family].append(recovery)
                    else:
                        subset = next(item for item in subsets() if arm_id(item) == arm)
                        factorial_values[subset][family].append(recovery)
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
                    native[(str(row["row_id"]), side)].margin > 0.0
                    for row in cell_rows
                ) / len(cell_rows)
                native_capability = native_capability and accuracy >= 0.85

    summaries = {}
    values = {}
    for subset in subsets():
        families = {
            family: summarize(factorial_values[subset][family])
            for family in ("A1", "A2")
        }
        target = statistics.fmean(
            families[family]["mean_recovery"] for family in ("A1", "A2")
        )
        summaries[arm_id(subset)] = {
            "heads": list(subset),
            "families": families,
            "mean_target_recovery": target,
        }
        values[subset] = target
    all_head_summary = {
        family: summarize(all_head_values[family]) for family in ("A1", "A2")
    }
    writer_summary = {
        family: summarize(writer_values[family]) for family in ("A1", "A2")
    }
    all_head_target = statistics.fmean(
        all_head_summary[family]["mean_recovery"] for family in ("A1", "A2")
    )
    writer_target = statistics.fmean(
        writer_summary[family]["mean_recovery"] for family in ("A1", "A2")
    )
    shapley = {}
    count = len(SELECTED_HEADS)
    for head in SELECTED_HEADS:
        total = 0.0
        for subset in subsets():
            if head in subset:
                continue
            extended = tuple(
                selected for selected in SELECTED_HEADS
                if selected in set(subset) | {head}
            )
            weight = (
                math.factorial(len(subset))
                * math.factorial(count - len(subset) - 1)
                / math.factorial(count)
            )
            total += weight * (values[extended] - values[subset])
        shapley[f"h{head}"] = total
    ranking = sorted(
        (f"h{head}" for head in SELECTED_HEADS),
        key=lambda name: (-shapley[name], name),
    )
    winner = int(ranking[0][1:])
    without_winner = tuple(head for head in SELECTED_HEADS if head != winner)
    family_drops = {
        family: summaries[arm_id(SELECTED_HEADS)]["families"][family]["mean_recovery"]
        - summaries[arm_id(without_winner)]["families"][family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    retained_fraction = values[SELECTED_HEADS] / all_head_target

    pred_a = (
        native_capability
        and empty_capture_max_abs <= 1.0e-4
        and tensor_error_max_abs <= 2.0e-3
        and abs(all_head_target - 0.05531263467112856) <= 0.01
    )
    pred_b = abs(writer_target - 0.33379277118533013) <= 0.02 and all(
        writer_summary[family]["mean_recovery"] > 0.0
        and writer_summary[family]["direction_fraction"] >= 0.80
        for family in ("A1", "A2")
    )
    pred_c = retained_fraction >= 0.75 and all(
        summaries[arm_id(SELECTED_HEADS)]["families"][family]["mean_recovery"] > 0.0
        and summaries[arm_id(SELECTED_HEADS)]["families"][family]["direction_fraction"] >= 0.80
        for family in ("A1", "A2")
    )
    pred_d = (
        sum(value > 0.0 for value in shapley.values()) >= 3
        and all(drop > 0.0 for drop in family_drops.values())
    )
    expected_records = (len(subsets()) + 2) * len(rows)
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
        "screen": "attention5_four_head_transporter",
        "null": "four_head_compression_or_factorial_support_failed",
        "invalid": "factorial_instrument_recurrence_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_mlp4_induced_attention5_four_head_factorial_result_v1",
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
            "pred_b_writer_recurrence": pred_b,
            "pred_c_four_head_compression": pred_c,
            "pred_d_factorial_support": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "empty_capture_scored_logit_max_abs": empty_capture_max_abs,
            "bilinear_tensor_reconstruction_max_abs": tensor_error_max_abs,
            "factorial_arms": summaries,
            "all_nine_heads": {
                "families": all_head_summary,
                "mean_target_recovery": all_head_target,
            },
            "writer_two_term": {
                "families": writer_summary,
                "mean_target_recovery": writer_target,
            },
            "four_head_retained_fraction": retained_fraction,
            "factorial_shapley_target_recovery": shapley,
            "factor_ranking": ranking,
            "dominant_head": f"h{winner}",
            "dominant_head_full_removal_family_drops": family_drops,
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
            "factor source terms for the licensed attention5 four-head transporter"
            if terminal == "screen"
            else "retain whole attention5 as the onset transporter"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "reason": reason,
        "predictions": result["predictions"],
        "retained_fraction": retained_fraction,
        "shapley": shapley,
        "ranking": ranking,
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
