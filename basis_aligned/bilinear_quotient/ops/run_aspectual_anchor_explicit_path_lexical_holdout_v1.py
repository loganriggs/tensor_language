#!/usr/bin/env python3
# BQGATE: frozen A-F prospective path predictions; CUDA is managed-queue only.
"""Prospective lexical/recombination transfer of the explicit aspectual path."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_battery_integration_contract as battery
import circuit_candidate_aspectual_lexical_holdout_v2 as holdout
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_mlp4_to_attention5_four_head_source_identity_v1 as discovery
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_explicit_path_lexical_holdout_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v2.py"
DISCOVERY = ROOT / "circuits/followups/aspectual_anchor_mlp4_to_attention5_four_head_source_identity_v1_result.json"
DISCOVERY_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp4_to_attention5_four_head_source_identity_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_explicit_path_lexical_holdout_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.explicit_path_lexical_holdout_v1"
EXPECTED_PRIOR_SHA256 = "4d38531edcf97eed13d2724362a7d17eb2d2a0fbeed00208f92dd3e6028a014e"
EXPECTED_BUILDER_SHA256 = "d4f37373ab52be5faf98fb1576179d659bbded8b4a5b75c7b7d7ec1fb567116a"
EXPECTED_HOLDOUT_SHA256 = "1418bfb6e0eb69a788cd11bd1da7b77585bd65cb5868fb7382f946a2072a1a25"
EXPECTED_DISCOVERY_SHA256 = "ec368faeb25c7309e77ef5a844c336e0511f92e19e2f714183b55b0e75c9d905"
EXPECTED_DISCOVERY_RUNNER_SHA256 = "2ad65455c45ce8104b77075030a9930e2cd3939c6ca5cdbfafda0b48ed205b7e"
ARMS = (
    "writer_two_term",
    "attention5_complete_four",
    "attention5_bank_four",
    "attention5_all_nine",
)
MODEL_FORWARDS_MAX = 20
EXAMPLE_EVALUATIONS_MAX = 320


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    for path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256,
        BUILDER: EXPECTED_BUILDER_SHA256,
        DISCOVERY: EXPECTED_DISCOVERY_SHA256,
        DISCOVERY_RUNNER: EXPECTED_DISCOVERY_RUNNER_SHA256,
    }.items():
        if sha256(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    discovery_result = json.loads(DISCOVERY.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if discovery_result.get("terminal") != "screen":
        raise ExperimentError("discovery path is not a screen")
    rows = holdout.build_rows()
    if holdout.validate_rows(rows) != EXPECTED_HOLDOUT_SHA256:
        raise ExperimentError("holdout authority changed")
    parent_rows = parent_runner.candidate.build_rows(parent_runner.candidate.TASK_ID)
    parent_spec = parent_runner.build_spec(parent_rows)
    spec = replace(
        parent_spec,
        experiment_id="aspectual-anchor-explicit-path-lexical-holdout-v1",
        authority_sha256=EXPECTED_HOLDOUT_SHA256,
        expected_fit_rows=len(rows),
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT",
            forward_calls=MODEL_FORWARDS_MAX,
            example_evaluations=EXAMPLE_EVALUATIONS_MAX,
            backward_calls=0,
            model_updates=0,
            evidence_bytes=32768,
        ),
    )
    enriched = tuple(screen.validate_fit_authority(spec, rows).values())
    if len(enriched) != 64 or len(ARMS) != 4:
        raise ExperimentError("holdout population or arms changed")
    return enriched, spec


class HoldoutBackend(discovery.SourceIdentityBackend):
    def validate_target_alignment(
        self, base_batch: producer.ModelBatch, donor_batch: producer.ModelBatch
    ) -> None:
        for base_ids, donor_ids, base_query, donor_query in zip(
            base_batch.token_rows,
            donor_batch.token_rows,
            base_batch.semantic_positions,
            donor_batch.semantic_positions,
        ):
            differences = [
                position
                for position, (base_id, donor_id) in enumerate(zip(base_ids, donor_ids))
                if base_id != donor_id
            ]
            if (
                len(base_ids) != len(donor_ids)
                or len(differences) != 1
                or base_query != donor_query
                or base_query != differences[0] + 4
            ):
                raise ExperimentError("holdout cue-to-subject token alignment failed")

    def intervene_all_heads(
        self,
        base_batch: producer.ModelBatch,
        hybrid_capture,
    ):
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
                head_output[i, query] = hybrid_capture["head_output"][i, query]
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
        "schema": "aspectual_anchor_explicit_path_lexical_holdout_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "builder_sha256": EXPECTED_BUILDER_SHA256,
        "holdout_authority_sha256": EXPECTED_HOLDOUT_SHA256,
        "discovery_result_sha256": EXPECTED_DISCOVERY_SHA256,
        "row_count": len(rows),
        "target_row_count": sum(row["transform_id"] in {"A1", "A2"} for row in rows),
        "arms": list(ARMS),
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
    backend = HoldoutBackend.load("cuda")
    native = {}
    arm_values = {arm: {"A1": [], "A2": []} for arm in ARMS}
    logits = {}
    raw_records = []
    bank_to_complete_max_abs = 0.0
    tensor_error_max_abs = 0.0
    forward_calls = 0
    evaluations = 0

    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            backend.validate_target_alignment(base_batch, donor_batch)
            base_output, base_bilinear = backend.capture_bilinear(base_batch)
            donor_output, donor_bilinear = backend.capture_bilinear(donor_batch)
            _, base_attention = backend.capture_attention5(base_batch)
            writer_output, hybrid_attention, tensor_error = backend.capture_writer_attention5(
                base_batch, donor_batch, base_bilinear, donor_bilinear
            )
            complete_output = backend.intervene_sources(
                base_batch, donor_batch, base_attention, hybrid_attention,
                "complete_four_heads",
            )
            bank_output = backend.intervene_sources(
                base_batch, donor_batch, base_attention, hybrid_attention,
                "last_period_determiner",
            )
            all_output = backend.intervene_all_heads(base_batch, hybrid_attention)
            forward_calls += 7
            evaluations += 7 * len(chunk)
            tensor_error_max_abs = max(tensor_error_max_abs, tensor_error)
            for side, output in (("base", base_output), ("donor", donor_output)):
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                        str(row["row_id"]), family, side, answer, foil
                    )
            outputs = {
                "writer_two_term": writer_output,
                "attention5_complete_four": complete_output,
                "attention5_bank_four": bank_output,
                "attention5_all_nine": all_output,
            }
            for complete, bank in zip(
                complete_output.answer_foil, bank_output.answer_foil
            ):
                bank_to_complete_max_abs = max(
                    bank_to_complete_max_abs,
                    abs(complete[0] - bank[0]),
                    abs(complete[1] - bank[1]),
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
                        "arm_id": arm, "family": family, "row_id": row_id,
                        "answer_logit": answer, "foil_logit": foil, "recovery": recovery,
                    })

    for family in ("P", "C"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for side in ("base", "donor"):
            for chunk in producer._chunks(family_rows, spec.batch_size):
                batch = producer._batch(spec, chunk, side)
                output = backend.native(batch, capture=False)
                forward_calls += 1
                evaluations += len(chunk)
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                        str(row["row_id"]), family, side, answer, foil
                    )

    cells, capability = producer._capability(spec, rows, native)
    pred_a = bool(cells and all(cell.passed for cell in cells))
    summaries = {}
    targets = {}
    for arm in ARMS:
        families = {family: summarize(arm_values[arm][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        summaries[arm] = {"families": families, "mean_target_recovery": target}
        targets[arm] = target
    writer = targets["writer_two_term"]
    complete = targets["attention5_complete_four"]
    bank = targets["attention5_bank_four"]
    all_nine = targets["attention5_all_nine"]
    pred_b = writer >= 0.20 and all(summaries["writer_two_term"]["families"][family]["mean_recovery"] > 0.0 and summaries["writer_two_term"]["families"][family]["direction_fraction"] >= 0.80 for family in ("A1", "A2"))
    pred_c = complete / writer >= 0.08 and all(summaries["attention5_complete_four"]["families"][family]["mean_recovery"] > 0.0 and summaries["attention5_complete_four"]["families"][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = bank / complete >= 0.95 and bank_to_complete_max_abs <= 0.125
    pred_e = complete / all_nine >= 0.70
    expected_records = len(ARMS) * 32
    pred_f = len(raw_records) == expected_records and len(logits) == expected_records and tensor_error_max_abs <= 2.0e-3 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    terminal = "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e, pred_f)) else ("null" if pred_a and pred_f else "invalid")
    reason = {"screen": "explicit_path_lexical_transfer", "null": "prospective_path_transfer_failed", "invalid": "holdout_capability_alignment_instrument_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_explicit_path_lexical_holdout_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "builder_sha256": EXPECTED_BUILDER_SHA256,
        "holdout_authority_sha256": EXPECTED_HOLDOUT_SHA256,
        "discovery_result_sha256": EXPECTED_DISCOVERY_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_native_capability": pred_a,
            "pred_b_writer_transfer": pred_b,
            "pred_c_four_head_path_transfer": pred_c,
            "pred_d_source_identity_transfer": pred_d,
            "pred_e_four_head_compression_transfer": pred_e,
            "pred_f_exact_coverage": pred_f,
        },
        "score": {
            "capability": managed.literal_json(asdict(capability)),
            "capability_cells": managed.literal_json([asdict(cell) for cell in cells]),
            "arms": summaries,
            "four_head_to_writer_fraction": complete / writer,
            "bank_to_complete_fraction": bank / complete,
            "four_head_to_all_nine_fraction": complete / all_nine,
            "bank_to_complete_scored_logit_max_abs": bank_to_complete_max_abs,
            "bilinear_tensor_reconstruction_max_abs": tensor_error_max_abs,
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "raw_record_count": len(raw_records), "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records, "terminal": terminal, "reason": reason,
        "next_action": "compile the prospectively validated explicit path" if terminal == "screen" else "retain discovery-only path scope",
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": result["predictions"], "writer": writer, "four_head": complete, "four_to_writer": complete / writer, "four_to_all": complete / all_nine, "bank_to_four": bank / complete, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
