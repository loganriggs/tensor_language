#!/usr/bin/env python3
# BQGATE: frozen A-E split missing-block compression predictions; CUDA is managed-queue only.
"""Select and confirm omitted new-write blocks for the sparse suffix recurrence."""

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
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_mlp11_15_bilinear_compression_split_v2 as mlp_parent
import run_aspectual_anchor_sparse_suffix_recurrence_confirmation_v1 as sparse_parent
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_sparse_suffix_missing_block_compression_split_v1.json"
SPARSE_RESULT = ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_recurrence_confirmation_v1_result.json"
SPARSE_RUNNER = ROOT / "ops/run_aspectual_anchor_sparse_suffix_recurrence_confirmation_v1.py"
PROGRAM = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v5_result.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_missing_block_compression_split_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.sparse_suffix_missing_block_compression_split_v1"
EXPECTED_PRIOR_SHA256 = "2818b4116295a987d8ff4c5a5ce487bb730053230458146e5688f4deebb2649f"
EXPECTED_SPARSE_RESULT_SHA256 = "db666e5e006d5ecb3300806399c682441ea92b990cb1155eaa76479899326ef2"
EXPECTED_SPARSE_RUNNER_SHA256 = "45801d9ec0d887c722148dfb39a5017d1345bb846b1837db1f90330b4c0532cc"
EXPECTED_PROGRAM_SHA256 = "7f851ffe62cd37305a558d89db305fd75d1f7276aacbbc81d7c914f1afdb5d08"
EXPECTED_BUILDER_SHA256 = "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859"
EXPECTED_ROWS_SHA256 = "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
EXPECTED_SELECTION_SHA256 = "d150ff72d1423058a01aa2140563315c041b1be98a59066e8dc4a98688775fe8"
EXPECTED_CONFIRMATION_SHA256 = "ad198e745d3c2b900e097219aae918f9ec506271f159bdcdf9852db56e12e55b"
CANDIDATE_BLOCKS = (10, 12, 13, 14, 16, 17)
SELECTED_WIDTH = 2
MODEL_FORWARDS_MAX = 50
EXAMPLE_EVALUATIONS_MAX = 400


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def selection_arms() -> tuple[str, ...]:
    return (
        "no_omitted", "all_omitted",
        *(f"only_block{boundary}" for boundary in CANDIDATE_BLOCKS),
        *(f"all_except_block{boundary}" for boundary in CANDIDATE_BLOCKS),
    )


def validate_static():
    for path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256, SPARSE_RESULT: EXPECTED_SPARSE_RESULT_SHA256,
        SPARSE_RUNNER: EXPECTED_SPARSE_RUNNER_SHA256, PROGRAM: EXPECTED_PROGRAM_SHA256,
        BUILDER: EXPECTED_BUILDER_SHA256,
    }.items():
        if sha256(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    sparse = json.loads(SPARSE_RESULT.read_text())
    program = json.loads(PROGRAM.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID or sparse.get("terminal") != "null"
        or program.get("terminal") != "release" or not all(program["predictions"].values())
        or sparse["predictions"]["pred_a_authority_capability_and_exact_dense_recurrence"] is not True
        or sparse["predictions"]["pred_e_exact_coverage"] is not True
    ):
        raise ExperimentError("upstream authority changed")
    rows_all = holdout.build_rows()
    if holdout.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("row authority changed")
    target = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    selection, confirmation = tuple(target[:16]), tuple(target[16:])
    if suffix.ids_sha256(selection) != EXPECTED_SELECTION_SHA256:
        raise ExperimentError("selection split changed")
    if suffix.ids_sha256(confirmation) != EXPECTED_CONFIRMATION_SHA256:
        raise ExperimentError("confirmation split changed")
    parent_rows = parent_runner.candidate.build_rows(parent_runner.candidate.TASK_ID)
    parent_spec = parent_runner.build_spec(parent_rows)
    spec = replace(
        parent_spec,
        experiment_id="aspectual-anchor-sparse-suffix-missing-block-compression-split-v1",
        authority_sha256=EXPECTED_ROWS_SHA256, expected_fit_rows=len(rows_all),
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT", forward_calls=MODEL_FORWARDS_MAX,
            example_evaluations=EXAMPLE_EVALUATIONS_MAX,
            backward_calls=0, model_updates=0, evidence_bytes=98304,
        ),
    )
    enriched = screen.validate_fit_authority(spec, rows_all)
    selection = tuple(enriched[str(row["row_id"])] for row in selection)
    confirmation = tuple(enriched[str(row["row_id"])] for row in confirmation)
    if len(selection) != len(confirmation) or len(selection) != 16 or len(selection_arms()) != 14:
        raise ExperimentError("population or arm inventory changed")
    return selection, confirmation, spec


class MissingBlockBackend(sparse_parent.SparseSuffixBackend):
    def selected_recurrence_delta(
        self, batch, role_banks, base_capture, hybrid_capture, terms, selected_blocks
    ):
        if len(selected_blocks) != len(set(selected_blocks)) or any(
            boundary not in CANDIDATE_BLOCKS for boundary in selected_blocks
        ):
            raise ExperimentError("omitted-block subset changed")
        deltas = [
            hybrid_capture["resid10"][i, query].float()
            - base_capture["resid10"][i, query].float()
            for i, query in enumerate(batch.semantic_positions)
        ]
        for boundary in range(10, 18):
            lambda0 = self.model.transformer.h[boundary].lambdas[0].float()
            projected_attention = projected_mlp = None
            if boundary in (11, 15):
                projected_attention = self.projected_source_delta(
                    batch, role_banks, *terms[boundary], boundary,
                    mlp_parent.SOURCE_BANK_BY_BOUNDARY[boundary],
                )
                projected_mlp, _error = self.projected_mlp_terms(
                    base_capture, hybrid_capture, boundary
                )
            next_deltas = []
            for i, query in enumerate(batch.semantic_positions):
                delta = lambda0 * deltas[i]
                if boundary in (11, 15):
                    delta = delta + projected_attention[i, query]
                    for factor in sparse_parent.SELECTED_MLP[boundary]:
                        delta = delta + projected_mlp[factor][i, query]
                elif boundary in selected_blocks:
                    delta = (
                        delta
                        + hybrid_capture[f"attention{boundary}"][i, query].float()
                        - base_capture[f"attention{boundary}"][i, query].float()
                        + hybrid_capture[f"mlp{boundary}"][i, query].float()
                        - base_capture[f"mlp{boundary}"][i, query].float()
                    )
                next_deltas.append(delta)
            deltas = next_deltas
        return deltas

    def selected_recurrence_readout(
        self, batch, role_banks, base_capture, hybrid_capture, terms, selected_blocks
    ):
        deltas = self.selected_recurrence_delta(
            batch, role_banks, base_capture, hybrid_capture, terms, selected_blocks
        )
        state = base_capture["resid18"].clone()
        for i, query in enumerate(batch.semantic_positions):
            state[i, query] = (state[i, query].float() + deltas[i]).to(state.dtype)
        return self.final_readout(batch, state)


def summarize(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("recovery is missing or nonfinite")
    return {
        "count": len(values), "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def main() -> None:
    selection, confirmation, spec = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_sparse_suffix_missing_block_compression_split_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "selection_row_ids_sha256": EXPECTED_SELECTION_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "selection_rows": len(selection), "confirmation_rows": len(confirmation),
        "candidate_blocks": list(CANDIDATE_BLOCKS),
        "selection_arms": len(selection_arms()), "selected_width": SELECTED_WIDTH,
        "confirmation_arms": 3, "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = MissingBlockBackend.load("cuda")
    native, captures = {}, {"selection": [], "confirmation": []}
    writer_values = {phase: {family: [] for family in ("A1", "A2")} for phase in captures}
    selection_values = {
        arm: {family: [] for family in ("A1", "A2")} for arm in selection_arms()
    }
    confirmation_values = {
        arm: {family: [] for family in ("A1", "A2")}
        for arm in ("no_omitted", "selected_two", "all_omitted")
    }
    raw_records = []
    manual_base_max_abs = writer_tensor_error_max_abs = all_omitted_writer_logit_max_abs = 0.0
    forward_calls = evaluations = 0

    for phase, rows in (("selection", selection), ("confirmation", confirmation)):
        for family in ("A1", "A2"):
            family_rows = [row for row in rows if row["transform_id"] == family]
            for chunk in producer._chunks(family_rows, spec.batch_size):
                base_batch = producer._batch(spec, chunk, "base")
                donor_batch = producer._batch(spec, chunk, "donor")
                role_banks = backend.role_positions(base_batch, donor_batch)
                base_native, base_bilinear = backend.capture_bilinear(base_batch)
                donor_native, donor_bilinear = backend.capture_bilinear(donor_batch)
                base_manual, base_capture = backend.capture_suffix_heads(base_batch)
                writer_output, hybrid_capture, writer_error = backend.capture_writer_suffix_heads(
                    base_batch, donor_batch, base_bilinear, donor_bilinear
                )
                forward_calls += 4
                evaluations += 4 * len(chunk)
                writer_tensor_error_max_abs = max(writer_tensor_error_max_abs, writer_error)
                for reference, manual in zip(base_native.answer_foil, base_manual.answer_foil):
                    manual_base_max_abs = max(
                        manual_base_max_abs, abs(reference[0] - manual[0]), abs(reference[1] - manual[1])
                    )
                for side, output in (("base", base_native), ("donor", donor_native)):
                    for row, pair in zip(chunk, output.answer_foil):
                        answer, foil = producer._finite_pair(pair)
                        native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                            str(row["row_id"]), family, side, answer, foil
                        )
                for row, pair in zip(chunk, writer_output.answer_foil):
                    answer, foil, value = suffix.recovery(row, pair, native)
                    writer_values[phase][family].append(value)
                    raw_records.append({
                        "phase": phase, "arm_id": "writer_two_term", "family": family,
                        "row_id": str(row["row_id"]), "answer_logit": answer,
                        "foil_logit": foil, "recovery": value,
                    })
                terms = {}
                for boundary in (11, 15):
                    bp, bv, _be = backend.attention_terms(base_batch, base_capture, boundary)
                    hp, hv, _he = backend.attention_terms(base_batch, hybrid_capture, boundary)
                    terms[boundary] = ((bp, bv), (hp, hv))
                captures[phase].append((family, chunk, base_batch, role_banks, base_capture, hybrid_capture, writer_output, terms))

    for family, chunk, batch, role_banks, base_capture, hybrid_capture, writer_output, terms in captures["selection"]:
        block_sets = {
            "no_omitted": (), "all_omitted": CANDIDATE_BLOCKS,
            **{f"only_block{boundary}": (boundary,) for boundary in CANDIDATE_BLOCKS},
            **{f"all_except_block{boundary}": tuple(other for other in CANDIDATE_BLOCKS if other != boundary) for boundary in CANDIDATE_BLOCKS},
        }
        for arm, blocks in block_sets.items():
            output = backend.selected_recurrence_readout(
                batch, role_banks, base_capture, hybrid_capture, terms, blocks
            )
            forward_calls += 1
            evaluations += len(chunk)
            for row, pair, writer_pair in zip(chunk, output.answer_foil, writer_output.answer_foil):
                answer, foil, value = suffix.recovery(row, pair, native)
                selection_values[arm][family].append(value)
                raw_records.append({
                    "phase": "selection", "arm_id": arm, "family": family,
                    "row_id": str(row["row_id"]), "answer_logit": answer,
                    "foil_logit": foil, "recovery": value,
                })
                if arm == "all_omitted":
                    all_omitted_writer_logit_max_abs = max(
                        all_omitted_writer_logit_max_abs,
                        abs(answer - writer_pair[0]), abs(foil - writer_pair[1]),
                    )

    arm_summaries, targets = {}, {}
    for arm in selection_arms():
        families = {family: summarize(selection_values[arm][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        arm_summaries[arm] = {"families": families, "mean_target_recovery": target}
        targets[arm] = target
    attributions = {}
    for boundary in CANDIDATE_BLOCKS:
        singleton = targets[f"only_block{boundary}"] - targets["no_omitted"]
        necessity = targets["all_omitted"] - targets[f"all_except_block{boundary}"]
        attributions[str(boundary)] = {
            "singleton_increment": singleton,
            "full_minus_leave_one_out_increment": necessity,
            "selection_score": 0.5 * (singleton + necessity),
        }
    ranking = sorted(
        CANDIDATE_BLOCKS,
        key=lambda boundary: (-attributions[str(boundary)]["selection_score"], CANDIDATE_BLOCKS.index(boundary)),
    )
    selected_blocks = tuple(boundary for boundary in CANDIDATE_BLOCKS if boundary in ranking[:SELECTED_WIDTH])
    selection_summary = {
        "arms": arm_summaries, "attributions": attributions,
        "ranking": list(ranking), "selected_blocks": list(selected_blocks),
        "all_minus_none_increment": targets["all_omitted"] - targets["no_omitted"],
    }

    for family, chunk, batch, role_banks, base_capture, hybrid_capture, writer_output, terms in captures["confirmation"]:
        for arm, blocks in {
            "no_omitted": (), "selected_two": selected_blocks, "all_omitted": CANDIDATE_BLOCKS,
        }.items():
            output = backend.selected_recurrence_readout(
                batch, role_banks, base_capture, hybrid_capture, terms, blocks
            )
            forward_calls += 1
            evaluations += len(chunk)
            for row, pair, writer_pair in zip(chunk, output.answer_foil, writer_output.answer_foil):
                answer, foil, value = suffix.recovery(row, pair, native)
                confirmation_values[arm][family].append(value)
                raw_records.append({
                    "phase": "confirmation", "arm_id": arm, "family": family,
                    "row_id": str(row["row_id"]), "answer_logit": answer,
                    "foil_logit": foil, "recovery": value,
                })
                if arm == "all_omitted":
                    all_omitted_writer_logit_max_abs = max(
                        all_omitted_writer_logit_max_abs,
                        abs(answer - writer_pair[0]), abs(foil - writer_pair[1]),
                    )

    confirmation_summary, confirmation_targets = {}, {}
    for arm in ("no_omitted", "selected_two", "all_omitted"):
        families = {family: summarize(confirmation_values[arm][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        confirmation_summary[arm] = {"families": families, "mean_target_recovery": target}
        confirmation_targets[arm] = target
    increment_fraction = (
        (confirmation_targets["selected_two"] - confirmation_targets["no_omitted"])
        / (confirmation_targets["all_omitted"] - confirmation_targets["no_omitted"])
    )
    total_fraction = confirmation_targets["selected_two"] / confirmation_targets["all_omitted"]
    family_increments = {
        family: confirmation_summary["selected_two"]["families"][family]["mean_recovery"]
        - confirmation_summary["no_omitted"]["families"][family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    writer_summary = {
        phase: {family: summarize(writer_values[phase][family]) for family in ("A1", "A2")}
        for phase in ("selection", "confirmation")
    }
    capability = all(
        native[(str(row["row_id"]), side)].margin > 0.0
        for row in selection + confirmation for side in ("base", "donor")
    )
    pred_a = (
        capability and manual_base_max_abs <= 1.0e-4
        and writer_tensor_error_max_abs <= 2.0e-3
        and all_omitted_writer_logit_max_abs <= 0.125
    )
    pred_b = selection_summary["all_minus_none_increment"] > 0.0 and len(set(selected_blocks)) == SELECTED_WIDTH
    pred_c = increment_fraction >= 0.70 and all(value > 0.0 for value in family_increments.values())
    pred_d = (
        total_fraction >= 0.85
        and all(
            confirmation_summary["selected_two"]["families"][family]["direction_fraction"] >= 0.75
            for family in ("A1", "A2")
        )
    )
    pred_e = (
        len(raw_records) == 304
        and len({(record["phase"], record["arm_id"], record["row_id"]) for record in raw_records}) == 304
        and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e)) else (
        "null" if pred_a and pred_b and pred_e else "invalid"
    )
    reason = {
        "screen": "two_missing_suffix_blocks_close_sparse_recurrence_gap",
        "null": "two_missing_blocks_do_not_close_enough_gap",
        "invalid": "authority_split_capability_dense_control_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_sparse_suffix_missing_block_compression_split_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "sparse_null_sha256": EXPECTED_SPARSE_RESULT_SHA256,
        "selection_row_ids_sha256": EXPECTED_SELECTION_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_split_capability_and_dense_control": pred_a,
            "pred_b_positive_selection_missing_signal": pred_b,
            "pred_c_disjoint_two_block_increment_compression": pred_c,
            "pred_d_disjoint_total_program_sufficiency": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "selection": selection_summary,
            "confirmation": {
                "arms": confirmation_summary, "selected_blocks": list(selected_blocks),
                "selected_increment_fraction": increment_fraction,
                "selected_total_to_all_fraction": total_fraction,
                "selected_family_increments": family_increments,
            },
            "writer": writer_summary,
            "manual_base_scored_logit_max_abs": manual_base_max_abs,
            "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "all_omitted_to_writer_logit_max_abs": all_omitted_writer_logit_max_abs,
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "raw_record_count": len(raw_records), "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records, "terminal": terminal, "reason": reason,
        "next_action": "factor selected missing blocks into attention/MLP terms" if terminal == "screen" else "retain dense suffix recurrence",
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason,
        "predictions": result["predictions"], "ranking": list(ranking),
        "selected_blocks": list(selected_blocks), "increment_fraction": increment_fraction,
        "total_fraction": total_fraction, "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
