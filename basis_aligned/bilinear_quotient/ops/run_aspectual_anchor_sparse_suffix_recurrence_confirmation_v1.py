#!/usr/bin/env python3
# BQGATE: frozen A-E sparse suffix recurrence predictions; CUDA is managed-queue only.
"""Confirm lambda recurrence plus only source/MLP-resolved block11 and block15 writes."""

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
import run_aspectual_anchor_attention11h3_15h5_source_compression_split_v1 as source_parent
import run_aspectual_anchor_mlp11_15_bilinear_compression_split_v1 as mlp_engine
import run_aspectual_anchor_mlp11_15_bilinear_compression_split_v2 as mlp_parent
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_sparse_suffix_recurrence_confirmation_v1.json"
PROGRAM = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v5_result.json"
MLP_RESULT = ROOT / "circuits/followups/aspectual_anchor_mlp11_15_bilinear_compression_split_v2_result.json"
SOURCE_RELEASE = ROOT / "circuits/followups/aspectual_anchor_attention11h3_15h5_source_compression_release_v1_result.json"
MLP_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp11_15_bilinear_compression_split_v2.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_recurrence_confirmation_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.sparse_suffix_recurrence_confirmation_v1"
EXPECTED_PRIOR_SHA256 = "cd097946fa338641126f198db4c162ca84e3ed32439ee8599453ef72f1f64ba9"
EXPECTED_PROGRAM_SHA256 = "7f851ffe62cd37305a558d89db305fd75d1f7276aacbbc81d7c914f1afdb5d08"
EXPECTED_MLP_RESULT_SHA256 = "8cb9fee89b29892916f0ae9ac331c60bd6b1eeb4d1ea375ad16553176a52633a"
EXPECTED_SOURCE_RELEASE_SHA256 = "e80f06ef21344139d33d7bc0793a20f564bf360ad2f7ea2d76edd52ab3421df5"
EXPECTED_MLP_RUNNER_SHA256 = "037c79096ac6e643cb533f8298b8acd0799e5cedae1efa65fccef8b2d5ba6b3b"
EXPECTED_ROWS_SHA256 = "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
EXPECTED_CONFIRMATION_SHA256 = "ad198e745d3c2b900e097219aae918f9ec506271f159bdcdf9852db56e12e55b"
ARMS = ("carried_only", "sparse_without11", "sparse_without15", "sparse_11_15", "dense_all")
SELECTED_MLP = {
    11: ("left_change", "right_change"),
    15: ("left_change", "bilinear_interaction"),
}
MODEL_FORWARDS_MAX = 18
EXAMPLE_EVALUATIONS_MAX = 144


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    for path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256, PROGRAM: EXPECTED_PROGRAM_SHA256,
        MLP_RESULT: EXPECTED_MLP_RESULT_SHA256, SOURCE_RELEASE: EXPECTED_SOURCE_RELEASE_SHA256,
        MLP_RUNNER: EXPECTED_MLP_RUNNER_SHA256,
    }.items():
        if sha256(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    program = json.loads(PROGRAM.read_text())
    mlp = json.loads(MLP_RESULT.read_text())
    source = json.loads(SOURCE_RELEASE.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID or program.get("terminal") != "release"
        or mlp.get("terminal") != "screen" or source.get("terminal") != "release"
        or not all(program["predictions"].values()) or not all(mlp["predictions"].values())
    ):
        raise ExperimentError("upstream terminal changed")
    for boundary in (11, 15):
        if tuple(mlp["score"]["confirmation"][str(boundary)]["selected_factors"]) != SELECTED_MLP[boundary]:
            raise ExperimentError("corrected MLP factor set changed")
        if tuple(source["released_banks"][str(boundary)]["source_roles"]) != mlp_parent.SOURCE_BANK_BY_BOUNDARY[boundary]:
            raise ExperimentError("released source bank changed")
    rows_all = holdout.build_rows()
    if holdout.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("row authority changed")
    target = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    rows = tuple(target[16:])
    if suffix.ids_sha256(rows) != EXPECTED_CONFIRMATION_SHA256:
        raise ExperimentError("confirmation population changed")
    parent_rows = parent_runner.candidate.build_rows(parent_runner.candidate.TASK_ID)
    parent_spec = parent_runner.build_spec(parent_rows)
    spec = replace(
        parent_spec,
        experiment_id="aspectual-anchor-sparse-suffix-recurrence-confirmation-v1",
        authority_sha256=EXPECTED_ROWS_SHA256, expected_fit_rows=len(rows_all),
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT", forward_calls=MODEL_FORWARDS_MAX,
            example_evaluations=EXAMPLE_EVALUATIONS_MAX,
            backward_calls=0, model_updates=0, evidence_bytes=49152,
        ),
    )
    enriched = screen.validate_fit_authority(spec, rows_all)
    rows = tuple(enriched[str(row["row_id"])] for row in rows)
    if len(rows) != 16 or len(ARMS) != 5:
        raise ExperimentError("population or arm inventory changed")
    return rows, spec


class SparseSuffixBackend(mlp_parent.CorrectedSuffixMlpBackend):
    def final_readout(self, batch, state):
        torch, F, model = self.torch, self.F, self.model
        lengths = tuple(len(row) for row in batch.token_rows)
        with torch.no_grad():
            logits = 30.0 * torch.tanh(
                model.lm_head(F.rms_norm(state, (model.config.n_embd,))) / 30.0
            )
            values = tuple(
                (
                    float(logits[i, length - 1, batch.answer_ids[i]].float()),
                    float(logits[i, length - 1, batch.foil_ids[i]].float()),
                ) for i, length in enumerate(lengths)
            )
        return producer.BatchOutput(values, {})

    def recurrence_delta(
        self, batch, role_banks, base_capture, hybrid_capture, terms, arm
    ):
        if arm not in ARMS:
            raise ExperimentError("sparse suffix arm changed")
        deltas = [
            hybrid_capture["resid10"][i, query].float()
            - base_capture["resid10"][i, query].float()
            for i, query in enumerate(batch.semantic_positions)
        ]
        for boundary in range(10, 18):
            lambda0 = self.model.transformer.h[boundary].lambdas[0].float()
            next_deltas = []
            projected_attention = projected_mlp = None
            include_sparse = (
                boundary in (11, 15) and arm in ("sparse_11_15", f"sparse_without{26 - boundary}")
            )
            if include_sparse:
                projected_attention = self.projected_source_delta(
                    batch, role_banks, *terms[boundary], boundary,
                    mlp_parent.SOURCE_BANK_BY_BOUNDARY[boundary],
                )
                projected_mlp, _error = self.projected_mlp_terms(
                    base_capture, hybrid_capture, boundary
                )
            for i, query in enumerate(batch.semantic_positions):
                delta = lambda0 * deltas[i]
                if arm == "dense_all":
                    delta = (
                        delta
                        + hybrid_capture[f"attention{boundary}"][i, query].float()
                        - base_capture[f"attention{boundary}"][i, query].float()
                        + hybrid_capture[f"mlp{boundary}"][i, query].float()
                        - base_capture[f"mlp{boundary}"][i, query].float()
                    )
                elif include_sparse:
                    delta = delta + projected_attention[i, query]
                    for factor in SELECTED_MLP[boundary]:
                        delta = delta + projected_mlp[factor][i, query]
                next_deltas.append(delta)
            deltas = next_deltas
        return deltas

    def recurrence_readout(
        self, batch, role_banks, base_capture, hybrid_capture, terms, arm
    ):
        deltas = self.recurrence_delta(
            batch, role_banks, base_capture, hybrid_capture, terms, arm
        )
        state = base_capture["resid18"].clone()
        for i, query in enumerate(batch.semantic_positions):
            state[i, query] = (state[i, query].float() + deltas[i]).to(state.dtype)
        return self.final_readout(batch, state), deltas


def summarize(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("recovery is missing or nonfinite")
    return {
        "count": len(values), "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def main() -> None:
    rows, spec = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_sparse_suffix_recurrence_confirmation_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "row_count": len(rows), "arms": list(ARMS),
        "selected_mlp_factors": {str(k): list(v) for k, v in SELECTED_MLP.items()},
        "selected_source_banks": {str(k): list(v) for k, v in mlp_parent.SOURCE_BANK_BY_BOUNDARY.items()},
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = SparseSuffixBackend.load("cuda")
    native, captures = {}, []
    writer_values = {family: [] for family in ("A1", "A2")}
    arm_values = {arm: {family: [] for family in ("A1", "A2")} for arm in ARMS}
    raw_records = []
    manual_base_max_abs = writer_tensor_error_max_abs = 0.0
    dense_tensor_max_abs = dense_writer_logit_max_abs = 0.0
    forward_calls = evaluations = 0

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
                writer_values[family].append(value)
                raw_records.append({
                    "arm_id": "writer_two_term", "family": family, "row_id": str(row["row_id"]),
                    "answer_logit": answer, "foil_logit": foil, "recovery": value,
                })
            terms = {}
            for boundary in (11, 15):
                bp, bv, _be = backend.attention_terms(base_batch, base_capture, boundary)
                hp, hv, _he = backend.attention_terms(base_batch, hybrid_capture, boundary)
                terms[boundary] = ((bp, bv), (hp, hv))
            captures.append((family, chunk, base_batch, role_banks, base_capture, hybrid_capture, writer_output, terms))

    for family, chunk, batch, role_banks, base_capture, hybrid_capture, writer_output, terms in captures:
        for arm in ARMS:
            output, deltas = backend.recurrence_readout(
                batch, role_banks, base_capture, hybrid_capture, terms, arm
            )
            forward_calls += 1
            evaluations += len(chunk)
            if arm == "dense_all":
                for i, query in enumerate(batch.semantic_positions):
                    direct = hybrid_capture["resid18"][i, query].float() - base_capture["resid18"][i, query].float()
                    dense_tensor_max_abs = max(dense_tensor_max_abs, float((deltas[i] - direct).abs().max()))
                for pair, writer_pair in zip(output.answer_foil, writer_output.answer_foil):
                    dense_writer_logit_max_abs = max(
                        dense_writer_logit_max_abs,
                        abs(pair[0] - writer_pair[0]), abs(pair[1] - writer_pair[1]),
                    )
            for row, pair in zip(chunk, output.answer_foil):
                answer, foil, value = suffix.recovery(row, pair, native)
                arm_values[arm][family].append(value)
                raw_records.append({
                    "arm_id": arm, "family": family, "row_id": str(row["row_id"]),
                    "answer_logit": answer, "foil_logit": foil, "recovery": value,
                })

    summaries, targets = {}, {}
    for arm in ARMS:
        families = {family: summarize(arm_values[arm][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        summaries[arm] = {"families": families, "mean_target_recovery": target}
        targets[arm] = target
    writer_summary = {family: summarize(writer_values[family]) for family in ("A1", "A2")}
    sparse_fraction = targets["sparse_11_15"] / targets["dense_all"]
    block11_damage = {
        family: summaries["sparse_11_15"]["families"][family]["mean_recovery"]
        - summaries["sparse_without11"]["families"][family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    block15_damage = {
        family: summaries["sparse_11_15"]["families"][family]["mean_recovery"]
        - summaries["sparse_without15"]["families"][family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    capability = all(
        native[(str(row["row_id"]), side)].margin > 0.0 for row in rows for side in ("base", "donor")
    )
    pred_a = (
        capability and manual_base_max_abs <= 1.0e-4
        and writer_tensor_error_max_abs <= 2.0e-3
        and dense_tensor_max_abs <= 0.04 and dense_writer_logit_max_abs <= 0.125
    )
    pred_b = (
        sparse_fraction >= 0.85
        and all(
            summaries["sparse_11_15"]["families"][family]["mean_recovery"] > 0.0
            and summaries["sparse_11_15"]["families"][family]["direction_fraction"] >= 0.75
            for family in ("A1", "A2")
        )
    )
    pred_c = all(value > 0.0 for value in block11_damage.values())
    pred_d = all(value > 0.0 for value in block15_damage.values())
    pred_e = (
        len(raw_records) == 96
        and len({(record["arm_id"], record["row_id"]) for record in raw_records}) == 96
        and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e)) else (
        "null" if pred_a and pred_e else "invalid"
    )
    reason = {
        "screen": "sparse_block11_15_suffix_recurrence_confirmed",
        "null": "sparse_sufficiency_or_necessity_failed",
        "invalid": "authority_capability_dense_closure_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_sparse_suffix_recurrence_confirmation_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "program_v5_release_sha256": EXPECTED_PROGRAM_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_capability_and_exact_dense_recurrence": pred_a,
            "pred_b_sparse_suffix_sufficiency": pred_b,
            "pred_c_block11_sparse_necessity": pred_c,
            "pred_d_block15_sparse_necessity": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "arms": summaries, "writer": writer_summary,
            "sparse_to_dense_fraction": sparse_fraction,
            "block11_removal_damage": block11_damage,
            "block15_removal_damage": block15_damage,
            "manual_base_scored_logit_max_abs": manual_base_max_abs,
            "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "dense_recurrence_tensor_max_abs": dense_tensor_max_abs,
            "dense_to_writer_logit_max_abs": dense_writer_logit_max_abs,
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "raw_record_count": len(raw_records), "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records, "terminal": terminal, "reason": reason,
        "next_action": "compile sparse suffix recurrence into transparent program v6" if terminal == "screen" else "retain native intervening suffix blocks",
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason,
        "predictions": result["predictions"], "arms": {arm: targets[arm] for arm in ARMS},
        "sparse_to_dense_fraction": sparse_fraction,
        "block11_removal_damage": block11_damage, "block15_removal_damage": block15_damage,
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
