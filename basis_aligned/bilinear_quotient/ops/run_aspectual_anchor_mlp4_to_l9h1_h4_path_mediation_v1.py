#!/usr/bin/env python3
# BQGATE: frozen A-E writer-to-reader mediation predictions; CUDA is managed-queue only.
"""Exact mediation of the two-term MLP4 write through L9H1/H4 source terms."""

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
import run_aspectual_anchor_l9h1_h4_source_term_factorial_v1 as source_terms
import run_aspectual_anchor_mlp4_bilinear_response_factorial_v1 as mlp4
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1.json"
WRITER = ROOT / "circuits/followups/aspectual_anchor_mlp4_bilinear_response_factorial_v2_result.json"
WRITER_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp4_bilinear_response_factorial_v2.py"
READER = ROOT / "circuits/followups/aspectual_anchor_l9h1_h4_downstream_source_bank_v2_result.json"
READER_RUNNER = ROOT / "ops/run_aspectual_anchor_l9h1_h4_downstream_source_bank_v2.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp4_to_l9h1_h4_path_mediation_v1"
EXPECTED_PRIOR_SHA256 = "4597be029bfbb87613dfc563d4a637e39eb51842b95457cb2bfcb4c1e1bf616d"
EXPECTED_WRITER_SHA256 = "359483cfb4807e9293e1f25f877db8d7303bc76333d83a6d237cf72a9c7e77e4"
EXPECTED_WRITER_RUNNER_SHA256 = "d3d0fcde5a8fb40f1dafcb560727e721bdaa07bb3ec9e974020e8d0338412591"
EXPECTED_READER_SHA256 = "6d694f92d35970f4eb5eba25ca3d9aff15cdbd1949db158a8be18e827e0423a7"
EXPECTED_READER_RUNNER_SHA256 = "5f2e1abc0d5c2cef168427e5d61d7de7b297a519867ca76c6708b92d3f94a4f1"
EXPECTED_AUTHORITY_SHA256 = "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3"
WRITER_FACTORS = ("left_change", "right_change")
HEADS = (1, 4)
BANK = ("last", "period", "determiner")
ARMS = (
    "writer_two_term",
    "h1h4_complete",
    "h1h4_all_sources",
    "h1h4_last_period_determiner",
    "h1h4_cue_self",
)
MODEL_FORWARDS_MAX = 20
EXAMPLE_EVALUATIONS_MAX = 640


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    expected = {
        PRIOR: EXPECTED_PRIOR_SHA256,
        WRITER: EXPECTED_WRITER_SHA256,
        WRITER_RUNNER: EXPECTED_WRITER_RUNNER_SHA256,
        READER: EXPECTED_READER_SHA256,
        READER_RUNNER: EXPECTED_READER_RUNNER_SHA256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    writer = json.loads(WRITER.read_text())
    reader = json.loads(READER.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if writer.get("terminal") != "screen" or reader.get("terminal") != "screen":
        raise ExperimentError("writer or reader authority is not a screen")
    if writer["score"]["selected_two_factor_subprogram"] != list(WRITER_FACTORS):
        raise ExperimentError("writer factor set changed")
    if reader["score"]["last_period_determiner_retained_fraction"] < 0.80:
        raise ExperimentError("reader source bank authority changed")
    rows = candidate.build_rows(candidate.TASK_ID)
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    spec = parent_runner.build_spec(rows)
    enriched_all = screen.validate_fit_authority(spec, rows)
    enriched = tuple(enriched_all[str(row["row_id"])] for row in selected)
    if len(enriched) != 64 or len(ARMS) != 5:
        raise ExperimentError("population or arm inventory changed")
    return enriched, spec


class PathBackend(mlp4.BilinearBackend, source_terms.SourceBackend):
    def capture_writer(
        self,
        base_batch: producer.ModelBatch,
        donor_batch: producer.ModelBatch,
        base_capture,
        donor_capture,
        factors: tuple[str, ...],
    ):
        if factors not in ((), WRITER_FACTORS):
            raise ExperimentError("writer factor set changed")
        projected, tensor_error = self.projected_terms(base_capture, donor_capture)
        positions = block4.source_positions(base_batch, donor_batch)

        def patch_mlp4(_module, _arguments, output):
            changed = output.clone()
            for i, bank in enumerate(positions):
                for position in bank:
                    delta = self.torch.zeros_like(
                        changed[i, position], dtype=self.torch.float32
                    )
                    for factor in factors:
                        delta += projected[factor][i, position]
                    changed[i, position] = (
                        changed[i, position].float() + delta
                    ).to(changed.dtype)
            return changed

        handle = self.model.transformer.h[4].mlp.register_forward_hook(patch_mlp4)
        try:
            output, attention_capture = self.manual_forward(base_batch)
        finally:
            handle.remove()
        return output, attention_capture, tensor_error

    def mediate(
        self,
        base_batch: producer.ModelBatch,
        donor_batch: producer.ModelBatch,
        base_attention,
        hybrid_attention,
        source_names: tuple[str, ...] | None,
    ):
        allowed = (None, BANK, ("cue", "self"), ("all",))
        if source_names not in allowed:
            raise ExperimentError("reader arm changed")
        head_dim = self.model.config.n_embd // self.model.config.n_head

        def patch_heads(_module, arguments):
            flattened = arguments[0]
            head_output = flattened.view(
                len(base_batch.row_ids),
                flattened.shape[1],
                self.model.config.n_head,
                head_dim,
            ).clone()
            for i, (base_ids, donor_ids, q) in enumerate(zip(
                base_batch.token_rows,
                donor_batch.token_rows,
                base_batch.semantic_positions,
            )):
                differences = [
                    position
                    for position, (base_id, donor_id) in enumerate(zip(base_ids, donor_ids))
                    if base_id != donor_id
                ]
                if len(base_ids) != len(donor_ids) or len(differences) != 1:
                    raise ExperimentError("cue alignment changed")
                cue = differences[0]
                positions = {
                    "cue": cue,
                    "last": cue + 1,
                    "period": cue + 2,
                    "determiner": cue + 3,
                    "self": q,
                }
                if any(not 0 <= position < len(base_ids) for position in positions.values()):
                    raise ExperimentError("source position is out of range")
                if source_names is None:
                    for head in HEADS:
                        head_output[i, q, head] = hybrid_attention["head_output"][i, q, head]
                    continue
                selected_positions = (
                    tuple(range(q + 1))
                    if source_names == ("all",)
                    else tuple(positions[name] for name in source_names)
                )
                for position in selected_positions:
                    for head in HEADS:
                        base_term = (
                            base_attention["pattern"][i, head, q, position]
                            * base_attention["value"][i, position, head]
                        )
                        hybrid_term = (
                            hybrid_attention["pattern"][i, head, q, position]
                            * hybrid_attention["value"][i, position, head]
                        )
                        head_output[i, q, head] += hybrid_term - base_term
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
        raise ExperimentError("recovery is missing or nonfinite")
    return {
        "count": len(values),
        "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def main() -> None:
    rows, spec = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "writer_result_sha256": EXPECTED_WRITER_SHA256,
        "reader_result_sha256": EXPECTED_READER_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "row_count": len(rows),
        "arms": list(ARMS),
        "writer_factors": list(WRITER_FACTORS),
        "heads": list(HEADS),
        "source_bank": list(BANK),
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
    backend = PathBackend.load("cuda")
    native = {}
    arm_values = {arm: {"A1": [], "A2": []} for arm in ARMS}
    logits = {}
    raw_records = []
    manual_empty_max_abs = 0.0
    all_to_complete_max_abs = 0.0
    tensor_error_max_abs = 0.0
    attention_reconstruction_max_abs = 0.0
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
            for reference, manual in zip(base_output.answer_foil, empty_output.answer_foil):
                manual_empty_max_abs = max(
                    manual_empty_max_abs,
                    abs(reference[0] - manual[0]),
                    abs(reference[1] - manual[1]),
                )

            outputs = {
                "writer_two_term": writer_output,
                "h1h4_complete": backend.mediate(
                    base_batch, donor_batch, base_attention, hybrid_attention, None
                ),
                "h1h4_all_sources": backend.mediate(
                    base_batch, donor_batch, base_attention, hybrid_attention, ("all",)
                ),
                "h1h4_last_period_determiner": backend.mediate(
                    base_batch, donor_batch, base_attention, hybrid_attention, BANK
                ),
                "h1h4_cue_self": backend.mediate(
                    base_batch, donor_batch, base_attention, hybrid_attention, ("cue", "self")
                ),
            }
            forward_calls += 4
            evaluations += 4 * len(chunk)
            for complete, all_sources in zip(
                outputs["h1h4_complete"].answer_foil,
                outputs["h1h4_all_sources"].answer_foil,
            ):
                all_to_complete_max_abs = max(
                    all_to_complete_max_abs,
                    abs(complete[0] - all_sources[0]),
                    abs(complete[1] - all_sources[1]),
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
    writer_recovery = targets["writer_two_term"]
    all_recovery = targets["h1h4_all_sources"]
    bank_recovery = targets["h1h4_last_period_determiner"]
    cue_self_recovery = targets["h1h4_cue_self"]
    bank_writer_retained = bank_recovery / writer_recovery
    bank_all_retained = bank_recovery / all_recovery
    cue_self_absolute_fraction = abs(cue_self_recovery) / abs(all_recovery)

    pred_a = (
        native_capability
        and manual_empty_max_abs <= 1.0e-4
        and tensor_error_max_abs <= 2.0e-3
        and attention_reconstruction_max_abs <= 1.0e-4
        and all_to_complete_max_abs <= 0.125
    )
    pred_b = abs(writer_recovery - 0.33379287409141367) <= 0.02 and all(
        summaries["writer_two_term"]["families"][family]["mean_recovery"] > 0.0
        and summaries["writer_two_term"]["families"][family]["direction_fraction"] >= 0.80
        for family in ("A1", "A2")
    )
    pred_c = bank_writer_retained >= 0.40 and all(
        summaries["h1h4_last_period_determiner"]["families"][family]["mean_recovery"] > 0.0
        and summaries["h1h4_last_period_determiner"]["families"][family]["direction_fraction"] >= 0.80
        for family in ("A1", "A2")
    )
    pred_d = bank_all_retained >= 0.80 and cue_self_absolute_fraction <= 0.25
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
        "screen": "two_term_mlp4_to_contextual_l9h1_h4_path",
        "null": "writer_reader_mediation_or_specificity_failed",
        "invalid": "path_instrument_writer_recurrence_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "writer_result_sha256": EXPECTED_WRITER_SHA256,
        "reader_result_sha256": EXPECTED_READER_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_exact_path_instrument": pred_a,
            "pred_b_writer_recurrence": pred_b,
            "pred_c_bank_mediation": pred_c,
            "pred_d_reader_specificity": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "manual_empty_hook_scored_logit_max_abs": manual_empty_max_abs,
            "bilinear_tensor_reconstruction_max_abs": tensor_error_max_abs,
            "attention_source_reconstruction_max_abs": attention_reconstruction_max_abs,
            "all_sources_to_complete_h1h4_scored_logit_max_abs": all_to_complete_max_abs,
            "arms": summaries,
            "bank_to_writer_retained_fraction": bank_writer_retained,
            "bank_to_all_h1h4_retained_fraction": bank_all_retained,
            "cue_self_absolute_all_h1h4_fraction": cue_self_absolute_fraction,
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
            "compile the licensed writer-to-reader path into the aspectual tensor program"
            if terminal == "screen"
            else "retain separately localized writer and reader without a mediation edge"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "reason": reason,
        "predictions": result["predictions"],
        "bank_to_writer": bank_writer_retained,
        "bank_to_all": bank_all_retained,
        "cue_self_to_all_abs": cue_self_absolute_fraction,
        "closure_max_abs": all_to_complete_max_abs,
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
