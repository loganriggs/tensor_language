#!/usr/bin/env python3
# BQGATE: frozen A-E prospective block6-8 crossing predictions; CUDA is managed-queue only.
"""Shared-capture exact block6/7/8 crossing factorials on the lexical holdout."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import itertools
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
import run_aspectual_anchor_block4_contextual_source_writer_factorial_v1 as block4
import run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1 as path
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_blocks6_8_crossing_factorials_lexical_holdout_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
UPSTREAM = ROOT / "circuits/followups/aspectual_anchor_explicit_path_lexical_holdout_v2_result.json"
DOWNSTREAM = ROOT / "circuits/followups/aspectual_anchor_attention9_h1h4_lexical_holdout_v1_result.json"
PROGRAM_RELEASE = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v1_result.json"
BACKEND_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_blocks6_8_crossing_factorials_lexical_holdout_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.blocks6_8_crossing_factorials_lexical_holdout_v1"
EXPECTED_PRIOR_SHA256 = "2f97602eb826a639d3b479a7a70c1621566f30e034b90bcbf64145b2920224e2"
EXPECTED_BUILDER_SHA256 = "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859"
EXPECTED_ROWS_SHA256 = "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
EXPECTED_UPSTREAM_SHA256 = "fd1b4ae15e1d327001c8b172bcbecb0f15609d6da01bec8c8dddbf8de107549e"
EXPECTED_DOWNSTREAM_SHA256 = "e07c5b210839a70ae1152fed907c4078a3439b833e8a2169346995abc16b2292"
EXPECTED_PROGRAM_RELEASE_SHA256 = "a2751011ac5fa02fcec433f2f83090f0911bdd5be1c84aeff0bf3ab8e3875cf1"
EXPECTED_BACKEND_RUNNER_SHA256 = "5449d45505267f2ebc92c9285b7a984fce773f834c2071e1d7e6d1a9f1949372"
BOUNDARIES = (6, 7, 8)
WRITER_FACTORS = ("left_change", "right_change")
MODEL_FORWARDS_MAX = 70
EXAMPLE_EVALUATIONS_MAX = 1152


class ExperimentError(RuntimeError):
    pass


def sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def factors(boundary: int) -> tuple[str, str, str]:
    return (f"carried{boundary}", f"attention{boundary}", f"mlp{boundary}")


def subsets(boundary: int):
    items = factors(boundary)
    return tuple(
        subset for width in range(len(items) + 1)
        for subset in itertools.combinations(items, width)
    )


def arm_id(subset: tuple[str, ...]) -> str:
    return "empty" if not subset else "+".join(subset)


def validate_static():
    expected = {
        PRIOR: EXPECTED_PRIOR_SHA256,
        BUILDER: EXPECTED_BUILDER_SHA256,
        UPSTREAM: EXPECTED_UPSTREAM_SHA256,
        DOWNSTREAM: EXPECTED_DOWNSTREAM_SHA256,
        PROGRAM_RELEASE: EXPECTED_PROGRAM_RELEASE_SHA256,
        BACKEND_RUNNER: EXPECTED_BACKEND_RUNNER_SHA256,
    }
    for file_path, digest in expected.items():
        if sha256(file_path) != digest:
            raise ExperimentError(f"authority hash changed: {file_path.name}")
    prior = json.loads(PRIOR.read_text())
    upstream = json.loads(UPSTREAM.read_text())
    downstream = json.loads(DOWNSTREAM.read_text())
    release = json.loads(PROGRAM_RELEASE.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if upstream.get("terminal") != "screen" or downstream.get("terminal") != "screen":
        raise ExperimentError("prospective endpoint authority changed")
    if release.get("terminal") != "release" or not all(release["predictions"].values()):
        raise ExperimentError("typed program release changed")
    rows = holdout.build_rows()
    if holdout.validate_rows(rows) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("holdout row authority changed")
    parent_rows = parent_runner.candidate.build_rows(parent_runner.candidate.TASK_ID)
    parent_spec = parent_runner.build_spec(parent_rows)
    spec = replace(
        parent_spec,
        experiment_id="aspectual-anchor-blocks6-8-crossing-factorials-lexical-holdout-v1",
        authority_sha256=EXPECTED_ROWS_SHA256,
        expected_fit_rows=len(rows),
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT",
            forward_calls=MODEL_FORWARDS_MAX,
            example_evaluations=EXAMPLE_EVALUATIONS_MAX,
            backward_calls=0,
            model_updates=0,
            evidence_bytes=131072,
        ),
    )
    enriched_all = screen.validate_fit_authority(spec, rows)
    selected = tuple(
        enriched_all[str(row["row_id"])]
        for row in rows if row["transform_id"] in {"A1", "A2"}
    )
    if len(rows) != 64 or len(selected) != 32:
        raise ExperimentError("holdout population changed")
    if any(len(subsets(boundary)) != 8 for boundary in BOUNDARIES):
        raise ExperimentError("factorial inventory changed")
    if not all(cell["passed"] for cell in upstream["score"]["capability_cells"]):
        raise ExperimentError("bound native capability changed")
    return selected, spec, upstream


class MultiCrossingBackend(path.PathBackend):
    def capture_crossings(self, batch: producer.ModelBatch):
        torch, F, model = self.torch, self.F, self.model
        tokens, lengths = self._tensor_batch(batch)
        capture = {}
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x0 = x
            capture["x0"] = x0.detach().clone()
            v1 = None
            for layer, block in enumerate(model.transformer.h):
                if layer in BOUNDARIES:
                    capture[f"resid{layer}"] = x.detach().clone()
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                x = live + attention
                mlp_output = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                x = x + mlp_output
                if layer in BOUNDARIES:
                    capture[f"attention{layer}"] = attention.detach().clone()
                    capture[f"mlp{layer}"] = mlp_output.detach().clone()
                    capture[f"resid{layer + 1}"] = x.detach().clone()
                    capture[f"v1_after{layer}"] = v1.detach().clone()
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
        expected = {"x0"}
        for boundary in BOUNDARIES:
            expected.update({
                f"resid{boundary}", f"attention{boundary}", f"mlp{boundary}",
                f"resid{boundary + 1}", f"v1_after{boundary}",
            })
        if set(capture) != expected:
            raise ExperimentError("multi-boundary capture incomplete")
        return producer.BatchOutput(values, {}), capture

    def capture_writer_crossings(
        self, base_batch, donor_batch, base_capture, donor_capture
    ):
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
                    changed[i, position] = (
                        changed[i, position].float() + delta
                    ).to(changed.dtype)
            return changed

        handle = self.model.transformer.h[4].mlp.register_forward_hook(patch_mlp4)
        try:
            output, capture = self.capture_crossings(base_batch)
        finally:
            handle.remove()
        return output, capture, tensor_error

    def suffix_from_boundary(self, batch, state, x0, v1, boundary: int):
        torch, F, model = self.torch, self.F, self.model
        lengths = tuple(len(row) for row in batch.token_rows)
        x = state
        with torch.no_grad():
            for layer in range(boundary + 1, 18):
                block = model.transformer.h[layer]
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                x = live + attention
                x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,)))
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

    def intervene_crossing(self, batch, base_capture, hybrid_capture, boundary, subset):
        if subset is not None and subset not in subsets(boundary):
            raise ExperimentError("crossing arm changed")
        state = base_capture[f"resid{boundary + 1}"].clone()
        lambda0 = self.model.transformer.h[boundary].lambdas[0]
        for i, query in enumerate(batch.semantic_positions):
            if subset is None:
                state[i, query] = hybrid_capture[f"resid{boundary + 1}"][i, query]
                continue
            delta = self.torch.zeros_like(state[i, query], dtype=self.torch.float32)
            if f"carried{boundary}" in subset:
                delta += lambda0.float() * (
                    hybrid_capture[f"resid{boundary}"][i, query].float()
                    - base_capture[f"resid{boundary}"][i, query].float()
                )
            for kind in ("attention", "mlp"):
                factor = f"{kind}{boundary}"
                if factor in subset:
                    delta += (
                        hybrid_capture[factor][i, query].float()
                        - base_capture[factor][i, query].float()
                    )
            state[i, query] = (state[i, query].float() + delta).to(state.dtype)
        return self.suffix_from_boundary(
            batch, state, base_capture["x0"],
            base_capture[f"v1_after{boundary}"], boundary,
        )


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
    rows, spec, upstream = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_blocks6_8_crossing_factorials_lexical_holdout_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "row_count": len(rows),
        "boundaries": list(BOUNDARIES),
        "factors": {str(boundary): list(factors(boundary)) for boundary in BOUNDARIES},
        "factorial_arms_per_boundary": 8,
        "direct_ceiling_arms": len(BOUNDARIES),
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
    backend = MultiCrossingBackend.load("cuda")
    native = {}
    arm_values = {
        boundary: {
            subset: {"A1": [], "A2": []} for subset in subsets(boundary)
        } for boundary in BOUNDARIES
    }
    ceiling_values = {
        boundary: {"A1": [], "A2": []} for boundary in BOUNDARIES
    }
    writer_values = {"A1": [], "A2": []}
    raw_records = []
    manual_base_max_abs = 0.0
    writer_tensor_error_max_abs = 0.0
    boundary_tensor_error_max_abs = {boundary: 0.0 for boundary in BOUNDARIES}
    full_to_ceiling_max_abs = {boundary: 0.0 for boundary in BOUNDARIES}
    forward_calls = 0
    evaluations = 0

    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            base_native, base_bilinear = backend.capture_bilinear(base_batch)
            donor_native, donor_bilinear = backend.capture_bilinear(donor_batch)
            base_manual, base_crossings = backend.capture_crossings(base_batch)
            writer_output, hybrid_crossings, writer_error = backend.capture_writer_crossings(
                base_batch, donor_batch, base_bilinear, donor_bilinear
            )
            forward_calls += 4
            evaluations += 4 * len(chunk)
            writer_tensor_error_max_abs = max(writer_tensor_error_max_abs, writer_error)
            for reference, manual in zip(base_native.answer_foil, base_manual.answer_foil):
                manual_base_max_abs = max(
                    manual_base_max_abs,
                    abs(reference[0] - manual[0]),
                    abs(reference[1] - manual[1]),
                )
            for side, output in (("base", base_native), ("donor", donor_native)):
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                        str(row["row_id"]), family, side, answer, foil
                    )
            for row, pair in zip(chunk, writer_output.answer_foil):
                answer, foil = producer._finite_pair(pair)
                row_id = str(row["row_id"])
                recovery = kernel.signed_pairwise_donor_recovery(
                    -native[(row_id, "base")].margin,
                    native[(row_id, "donor")].margin,
                    -(answer - foil),
                )
                writer_values[family].append(recovery)
                raw_records.append({
                    "boundary": "writer", "arm_id": "writer_two_term",
                    "family": family, "row_id": row_id,
                    "answer_logit": answer, "foil_logit": foil, "recovery": recovery,
                })

            for boundary in BOUNDARIES:
                lambda0 = backend.model.transformer.h[boundary].lambdas[0]
                for i, query in enumerate(base_batch.semantic_positions):
                    reconstructed = (
                        lambda0.float() * (
                            hybrid_crossings[f"resid{boundary}"][i, query].float()
                            - base_crossings[f"resid{boundary}"][i, query].float()
                        )
                        + hybrid_crossings[f"attention{boundary}"][i, query].float()
                        - base_crossings[f"attention{boundary}"][i, query].float()
                        + hybrid_crossings[f"mlp{boundary}"][i, query].float()
                        - base_crossings[f"mlp{boundary}"][i, query].float()
                    )
                    direct = (
                        hybrid_crossings[f"resid{boundary + 1}"][i, query].float()
                        - base_crossings[f"resid{boundary + 1}"][i, query].float()
                    )
                    boundary_tensor_error_max_abs[boundary] = max(
                        boundary_tensor_error_max_abs[boundary],
                        float((reconstructed - direct).abs().max()),
                    )
                outputs = {
                    arm_id(subset): backend.intervene_crossing(
                        base_batch, base_crossings, hybrid_crossings, boundary, subset
                    ) for subset in subsets(boundary)
                }
                outputs["direct_query_ceiling"] = backend.intervene_crossing(
                    base_batch, base_crossings, hybrid_crossings, boundary, None
                )
                forward_calls += len(subsets(boundary)) + 1
                evaluations += (len(subsets(boundary)) + 1) * len(chunk)
                full_name = arm_id(factors(boundary))
                for full_pair, direct_pair in zip(
                    outputs[full_name].answer_foil,
                    outputs["direct_query_ceiling"].answer_foil,
                ):
                    full_to_ceiling_max_abs[boundary] = max(
                        full_to_ceiling_max_abs[boundary],
                        abs(full_pair[0] - direct_pair[0]),
                        abs(full_pair[1] - direct_pair[1]),
                    )
                for name, output in outputs.items():
                    for row, pair in zip(chunk, output.answer_foil):
                        answer, foil = producer._finite_pair(pair)
                        row_id = str(row["row_id"])
                        recovery = kernel.signed_pairwise_donor_recovery(
                            -native[(row_id, "base")].margin,
                            native[(row_id, "donor")].margin,
                            -(answer - foil),
                        )
                        if name == "direct_query_ceiling":
                            ceiling_values[boundary][family].append(recovery)
                        else:
                            subset = next(
                                item for item in subsets(boundary) if arm_id(item) == name
                            )
                            arm_values[boundary][subset][family].append(recovery)
                        raw_records.append({
                            "boundary": boundary, "arm_id": name,
                            "family": family, "row_id": row_id,
                            "answer_logit": answer, "foil_logit": foil,
                            "recovery": recovery,
                        })

    current_capability = True
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
                current_capability = current_capability and accuracy >= 0.85

    writer_summary = {
        family: summarize(writer_values[family]) for family in ("A1", "A2")
    }
    writer_target = statistics.fmean(
        writer_summary[family]["mean_recovery"] for family in ("A1", "A2")
    )
    boundary_results = {}
    full_targets = []
    active_predictions = []
    for boundary in BOUNDARIES:
        summaries = {}
        values = {}
        for subset in subsets(boundary):
            families = {
                family: summarize(arm_values[boundary][subset][family])
                for family in ("A1", "A2")
            }
            target = statistics.fmean(
                families[family]["mean_recovery"] for family in ("A1", "A2")
            )
            summaries[arm_id(subset)] = {
                "factors": list(subset), "families": families,
                "mean_target_recovery": target,
            }
            values[subset] = target
        shapley = {}
        all_factors = factors(boundary)
        for factor in all_factors:
            total = 0.0
            for subset in subsets(boundary):
                if factor in subset:
                    continue
                extended = tuple(
                    item for item in all_factors if item in set(subset) | {factor}
                )
                weight = (
                    math.factorial(len(subset))
                    * math.factorial(len(all_factors) - len(subset) - 1)
                    / math.factorial(len(all_factors))
                )
                total += weight * (values[extended] - values[subset])
            shapley[factor] = total
        full_name = arm_id(all_factors)
        full_targets.append(values[all_factors])
        removal_damage = {
            factor: {
                family: summaries[full_name]["families"][family]["mean_recovery"]
                - summaries[arm_id(tuple(item for item in all_factors if item != factor))]["families"][family]["mean_recovery"]
                for family in ("A1", "A2")
            }
            for factor in all_factors
        }
        new_factors = (f"attention{boundary}", f"mlp{boundary}")
        active_predictions.append(
            shapley[f"carried{boundary}"] > 0.0
            and sum(shapley[factor] for factor in new_factors) >= 0.003
            and any(
                all(removal_damage[factor][family] > 0.0 for family in ("A1", "A2"))
                for factor in new_factors
            )
        )
        ceiling_summary = {
            family: summarize(ceiling_values[boundary][family])
            for family in ("A1", "A2")
        }
        boundary_results[str(boundary)] = {
            "factorial_arms": summaries,
            "direct_query_ceiling": {
                "families": ceiling_summary,
                "mean_target_recovery": statistics.fmean(
                    ceiling_summary[family]["mean_recovery"] for family in ("A1", "A2")
                ),
            },
            "factorial_shapley_target_recovery": shapley,
            "new_factor_full_removal_damage": {
                factor: removal_damage[factor] for factor in new_factors
            },
            "dominant_factor": max(all_factors, key=lambda factor: shapley[factor]),
            "full_crossing_to_writer_fraction": values[all_factors] / writer_target,
            "tensor_reconstruction_max_abs": boundary_tensor_error_max_abs[boundary],
            "full_to_direct_ceiling_scored_logit_max_abs": full_to_ceiling_max_abs[boundary],
        }

    bound_capability = all(
        cell["passed"] for cell in upstream["score"]["capability_cells"]
    )
    pred_a = (
        bound_capability and current_capability
        and manual_base_max_abs <= 1.0e-4
        and writer_tensor_error_max_abs <= 2.0e-3
        and all(value <= 0.04 for value in boundary_tensor_error_max_abs.values())
        and all(value <= 0.125 for value in full_to_ceiling_max_abs.values())
    )
    pred_b = abs(writer_target - 0.2835613798233539) <= 0.01 and all(
        writer_summary[family]["mean_recovery"] > 0.0
        and writer_summary[family]["direction_fraction"] >= 0.80
        for family in ("A1", "A2")
    )
    open_attention5 = upstream["score"]["arms"]["attention5_all_nine"]["mean_target_recovery"]
    pred_c = (
        open_attention5 < full_targets[0] < full_targets[1] < full_targets[2]
        and all(
            boundary_results[str(boundary)]["factorial_arms"][arm_id(factors(boundary))]["families"][family]["direction_fraction"] >= 0.75
            for boundary in BOUNDARIES for family in ("A1", "A2")
        )
    )
    pred_d = all(active_predictions)
    pred_e = (
        len(raw_records) == 896
        and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = (
        "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e))
        else "null" if pred_a and pred_b and pred_e
        else "invalid"
    )
    reason = {
        "screen": "prospective_blocks6_8_active_monotone_strengthening",
        "null": "prospective_intermediate_strengthening_prediction_failed",
        "invalid": "authority_capability_instrument_writer_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_blocks6_8_crossing_factorials_lexical_holdout_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "upstream_result_sha256": EXPECTED_UPSTREAM_SHA256,
        "downstream_result_sha256": EXPECTED_DOWNSTREAM_SHA256,
        "program_release_sha256": EXPECTED_PROGRAM_RELEASE_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_capability_and_exact_instrument": pred_a,
            "pred_b_writer_recurrence": pred_b,
            "pred_c_monotone_crossing_recurrence": pred_c,
            "pred_d_active_strengthening_components": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "manual_base_scored_logit_max_abs": manual_base_max_abs,
            "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "writer_two_term": {
                "families": writer_summary,
                "mean_target_recovery": writer_target,
            },
            "open_attention5_all_nine_mean_recovery": open_attention5,
            "boundaries": boundary_results,
            "full_crossing_curve": {
                f"resid{boundary + 1}": value
                for boundary, value in zip(BOUNDARIES, full_targets)
            },
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
            "compile prospectively localized block6-8 strengthening into typed path v2"
            if terminal == "screen"
            else "retain measured intermediate curve without component promotion"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "reason": reason,
        "predictions": result["predictions"],
        "writer": writer_target,
        "open_attention5": open_attention5,
        "full_crossing_curve": result["score"]["full_crossing_curve"],
        "shapley": {
            boundary: boundary_results[str(boundary)]["factorial_shapley_target_recovery"]
            for boundary in BOUNDARIES
        },
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
