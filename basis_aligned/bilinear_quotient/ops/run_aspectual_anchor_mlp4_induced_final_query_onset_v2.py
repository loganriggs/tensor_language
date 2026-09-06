#!/usr/bin/env python3
# BQGATE: frozen A-E final-query onset predictions; CUDA is managed-queue only.
"""Trace the two-term MLP4 source write into final-query residual states."""

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
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp4_induced_final_query_onset_v2.json"
PARENT = ROOT / "circuits/followups/aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2_result.json"
PARENT_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp4_induced_final_query_onset_v2_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp4_induced_final_query_onset_v2"
EXPECTED_PRIOR_SHA256 = "0a45242795100da34393308e4460784993cf89d8e46480285a089d4a61d3ce51"
EXPECTED_PARENT_SHA256 = "fb62926d6b52bbef750d0306d38a16539ca8393b94ed23448bcd29a9ee912221"
EXPECTED_PARENT_RUNNER_SHA256 = "54db17a05a70deb41e762c77eb342fdc522a3849981f66507615e21ab06743ea"
EXPECTED_AUTHORITY_SHA256 = "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3"
WRITER_FACTORS = ("left_change", "right_change")
BOUNDARIES = (5, 6, 7, 8, 9)
MODEL_FORWARDS_MAX = 18
EXAMPLE_EVALUATIONS_MAX = 576


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
    if parent.get("terminal") != "screen" or parent["score"]["dominant_factor"] != "attention9":
        raise ExperimentError("parent crossing decision changed")
    rows = candidate.build_rows(candidate.TASK_ID)
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    spec = parent_runner.build_spec(rows)
    enriched_all = screen.validate_fit_authority(spec, rows)
    enriched = tuple(enriched_all[str(row["row_id"])] for row in selected)
    if len(enriched) != 64 or BOUNDARIES != (5, 6, 7, 8, 9):
        raise ExperimentError("population or boundary inventory changed")
    return enriched, spec


class OnsetBackend(path.PathBackend):
    def capture_states(self, batch: producer.ModelBatch):
        torch, F, model = self.torch, self.F, self.model
        tokens, lengths = self._tensor_batch(batch)
        capture = {}
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x0 = x
            v1 = None
            for layer, transformer_block in enumerate(model.transformer.h):
                if layer in BOUNDARIES:
                    capture[f"resid{layer}"] = x.detach().clone()
                    capture[f"v1_{layer}"] = v1.detach().clone()
                live = transformer_block.lambdas[0] * x + transformer_block.lambdas[1] * x0
                attention, v1 = transformer_block.attn(
                    F.rms_norm(live, (model.config.n_embd,)), v1
                )
                normalized = F.rms_norm(live + attention, (model.config.n_embd,))
                if layer == 4:
                    capture["left"] = transformer_block.mlp.Left(normalized).detach().clone()
                    capture["right"] = transformer_block.mlp.Right(normalized).detach().clone()
                x = live + attention + transformer_block.mlp(normalized)
            capture["x0"] = x0.detach().clone()
            logits = 30.0 * torch.tanh(
                model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0
            )
            values = tuple(
                (
                    float(logits[i, length - 1, batch.answer_ids[i]].float()),
                    float(logits[i, length - 1, batch.foil_ids[i]].float()),
                )
                for i, length in enumerate(lengths)
            )
        expected = {"left", "right", "x0"} | {
            name for boundary in BOUNDARIES for name in (f"resid{boundary}", f"v1_{boundary}")
        }
        if set(capture) != expected:
            raise ExperimentError("state capture incomplete")
        return producer.BatchOutput(values, {}), capture

    def capture_writer_states(self, base_batch, donor_batch, base_capture, donor_capture):
        projected, tensor_error = self.projected_terms(base_capture, donor_capture)
        positions = block4.source_positions(base_batch, donor_batch)

        def patch_mlp4(_module, _arguments, output):
            changed = output.clone()
            for i, bank in enumerate(positions):
                for position in bank:
                    delta = sum(
                        (projected[factor][i, position] for factor in WRITER_FACTORS),
                        self.torch.zeros_like(changed[i, position], dtype=self.torch.float32),
                    )
                    changed[i, position] = (changed[i, position].float() + delta).to(changed.dtype)
            return changed

        handle = self.model.transformer.h[4].mlp.register_forward_hook(patch_mlp4)
        try:
            output, capture = self.capture_states(base_batch)
        finally:
            handle.remove()
        return output, capture, tensor_error

    def intervene_boundary(self, batch, base_capture, hybrid_capture, boundary):
        if boundary not in BOUNDARIES:
            raise ExperimentError("boundary changed")
        torch, F, model = self.torch, self.F, self.model
        state = base_capture[f"resid{boundary}"].clone()
        for i, q in enumerate(batch.semantic_positions):
            state[i, q] = hybrid_capture[f"resid{boundary}"][i, q]
        x = state
        x0 = base_capture["x0"]
        v1 = base_capture[f"v1_{boundary}"]
        lengths = tuple(len(row) for row in batch.token_rows)
        with torch.no_grad():
            for layer in range(boundary, 18):
                transformer_block = model.transformer.h[layer]
                live = transformer_block.lambdas[0] * x + transformer_block.lambdas[1] * x0
                attention, v1 = transformer_block.attn(
                    F.rms_norm(live, (model.config.n_embd,)), v1
                )
                x = live + attention
                x = x + transformer_block.mlp(F.rms_norm(x, (model.config.n_embd,)))
            logits = 30.0 * torch.tanh(
                model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0
            )
            values = tuple(
                (
                    float(logits[i, length - 1, batch.answer_ids[i]].float()),
                    float(logits[i, length - 1, batch.foil_ids[i]].float()),
                )
                for i, length in enumerate(lengths)
            )
        return producer.BatchOutput(values, {})


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
        "schema": "aspectual_anchor_mlp4_induced_final_query_onset_dryrun_v2",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "row_count": len(rows),
        "boundaries": list(BOUNDARIES),
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
    backend = OnsetBackend.load("cuda")
    native = {}
    boundary_values = {boundary: {"A1": [], "A2": []} for boundary in BOUNDARIES}
    writer_values = {"A1": [], "A2": []}
    logits = {}
    raw_records = []
    manual_native_max_abs = 0.0
    writer_tensor_error_max_abs = 0.0
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            base_output, base_capture = backend.capture_states(base_batch)
            donor_output, donor_capture = backend.capture_states(donor_batch)
            native_base = backend.native(base_batch, capture=False)
            writer_output, hybrid_capture, tensor_error = backend.capture_writer_states(
                base_batch, donor_batch, base_capture, donor_capture
            )
            forward_calls += 4
            evaluations += 4 * len(chunk)
            writer_tensor_error_max_abs = max(writer_tensor_error_max_abs, tensor_error)
            for manual, reference in zip(base_output.answer_foil, native_base.answer_foil):
                manual_native_max_abs = max(
                    manual_native_max_abs,
                    abs(manual[0] - reference[0]),
                    abs(manual[1] - reference[1]),
                )
            for side, output in (("base", native_base), ("donor", donor_output)):
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                        str(row["row_id"]), family, side, answer, foil
                    )
            for row, pair in zip(chunk, writer_output.answer_foil):
                answer, foil = producer._finite_pair(pair)
                row_id = str(row["row_id"])
                writer_values[family].append(kernel.signed_pairwise_donor_recovery(
                    -native[(row_id, "base")].margin,
                    native[(row_id, "donor")].margin,
                    -(answer - foil),
                ))
            for boundary in BOUNDARIES:
                output = backend.intervene_boundary(
                    base_batch, base_capture, hybrid_capture, boundary
                )
                forward_calls += 1
                evaluations += len(chunk)
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    row_id = str(row["row_id"])
                    recovery = kernel.signed_pairwise_donor_recovery(
                        -native[(row_id, "base")].margin,
                        native[(row_id, "donor")].margin,
                        -(answer - foil),
                    )
                    boundary_values[boundary][family].append(recovery)
                    logits[(boundary, row_id)] = (answer, foil)
                    raw_records.append({
                        "boundary": boundary, "family": family, "row_id": row_id,
                        "answer_logit": answer, "foil_logit": foil, "recovery": recovery,
                    })

    native_capability = True
    for family in ("A1", "A2"):
        for direction in ("past_to_present", "present_to_past"):
            cell_rows = [row for row in rows if row["transform_id"] == family and row["direction_id"] == direction]
            for side in ("base", "donor"):
                accuracy = sum(native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows) / len(cell_rows)
                native_capability = native_capability and accuracy >= 0.85
    summaries = {}
    targets = {}
    for boundary in BOUNDARIES:
        families = {family: summarize(boundary_values[boundary][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        summaries[str(boundary)] = {"families": families, "mean_target_recovery": target}
        targets[boundary] = target
    writer_summary = {family: summarize(writer_values[family]) for family in ("A1", "A2")}
    writer_target = statistics.fmean(writer_summary[family]["mean_recovery"] for family in ("A1", "A2"))
    passing = [boundary for boundary in BOUNDARIES if all(summaries[str(boundary)]["families"][family]["mean_recovery"] >= 0.05 and summaries[str(boundary)]["families"][family]["direction_fraction"] >= 0.80 for family in ("A1", "A2"))]
    first_passing = passing[0] if passing else None
    pred_a = native_capability and manual_native_max_abs <= 1.0e-4 and writer_tensor_error_max_abs <= 2.0e-3 and abs(writer_target - 0.33379277118533013) <= 0.02
    pred_b = abs(targets[5]) <= 0.01
    pred_c = first_passing == 9
    pred_d = targets[9] >= 0.08 and all(summaries["9"]["families"][family]["mean_recovery"] > 0.0 for family in ("A1", "A2"))
    expected_records = len(BOUNDARIES) * len(rows)
    pred_e = len(raw_records) == expected_records and len(logits) == expected_records and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    terminal = "screen" if pred_a and pred_b and pred_c and pred_d and pred_e else ("null" if pred_a and pred_e else "invalid")
    reason = {"screen": "block8_onset_of_carried_final_query_branch", "null": "frozen_final_query_onset_prediction_failed", "invalid": "onset_instrument_recurrence_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_mlp4_induced_final_query_onset_result_v2",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256, "dryrun": dryrun,
        "predictions": {
            "pred_a_exact_instrument": pred_a,
            "pred_b_source_only_start": pred_b,
            "pred_c_block8_onset": pred_c,
            "pred_d_resid9_strength": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "manual_native_scored_logit_max_abs": manual_native_max_abs,
            "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "writer_two_term": {"families": writer_summary, "mean_target_recovery": writer_target},
            "boundary_curve": summaries, "passing_boundaries": passing,
            "first_passing_boundary": first_passing,
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "raw_record_count": len(raw_records), "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records, "terminal": terminal, "reason": reason,
        "next_action": "factor block8 into carried/attention/MLP terms" if terminal == "screen" else "follow the observed frozen onset boundary",
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": result["predictions"], "curve": {str(boundary): targets[boundary] for boundary in BOUNDARIES}, "first_passing": first_passing, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
