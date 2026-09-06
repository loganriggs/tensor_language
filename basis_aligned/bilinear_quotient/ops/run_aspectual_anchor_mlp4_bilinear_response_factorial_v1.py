#!/usr/bin/env python3
# BQGATE: frozen A-E bilinear-response predictions; CUDA is managed-queue only.
"""Exact left/right/interaction factorization of the aspectual MLP4 write."""

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
import run_aspectual_anchor_block4_contextual_source_writer_factorial_v1 as block4
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp4_bilinear_response_factorial_v1.json"
PARENT = ROOT / "circuits/followups/aspectual_anchor_block4_contextual_source_writer_factorial_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_aspectual_anchor_block4_contextual_source_writer_factorial_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp4_bilinear_response_factorial_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp4_bilinear_response_factorial_v1"
EXPECTED_PRIOR_SHA256 = "f7a8178c34a798f5ed9082d1d85cc6f4f6f8e3a2ca065e8c40f1362b12b2bcc9"
EXPECTED_PARENT_SHA256 = "c20f69b83ef1cf07d5e986c6ff1e5dc53fb30bb4beb5e5f809fca6a2509fc85a"
EXPECTED_PARENT_RUNNER_SHA256 = "3246f56399abe00a2d178a6a5a5f81bed64e0f22487c06b5c0ce7f76cf1b5691"
EXPECTED_AUTHORITY_SHA256 = "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3"
FACTORS = ("left_change", "right_change", "bilinear_interaction")
MODEL_FORWARDS_MAX = 28
EXAMPLE_EVALUATIONS_MAX = 896


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def subsets():
    return tuple(
        subset for width in range(len(FACTORS) + 1)
        for subset in itertools.combinations(FACTORS, width)
    )


def arm_id(subset: tuple[str, ...]) -> str:
    return "empty" if not subset else "+".join(subset)


def validate_static():
    if sha256(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior-art hash changed")
    if sha256(PARENT) != EXPECTED_PARENT_SHA256:
        raise ExperimentError("parent result hash changed")
    if sha256(PARENT_RUNNER) != EXPECTED_PARENT_RUNNER_SHA256:
        raise ExperimentError("parent runner hash changed")
    prior = json.loads(PRIOR.read_text())
    parent = json.loads(PARENT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if parent.get("terminal") != "screen" or parent["score"]["block4_writer_winner"] != "mlp4":
        raise ExperimentError("parent MLP4 decision changed")
    rows = candidate.build_rows(candidate.TASK_ID)
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    spec = parent_runner.build_spec(rows)
    enriched_all = screen.validate_fit_authority(spec, rows)
    enriched = tuple(enriched_all[str(row["row_id"])] for row in selected)
    if len(enriched) != 64 or len(subsets()) != 8:
        raise ExperimentError("population or factorial changed")
    return enriched, spec


class BilinearBackend(block4.ComponentBackend):
    def capture_bilinear(self, batch: producer.ModelBatch):
        output, capture = self.capture_components(batch)
        F = self.F
        module = self.model.transformer.h[4].mlp
        live = (
            self.model.transformer.h[4].lambdas[0] * capture["resid4"]
            + self.model.transformer.h[4].lambdas[1] * capture["x0"]
        )
        normalized = F.rms_norm(
            live + capture["attention4"], (self.model.config.n_embd,)
        )
        capture["left"] = module.Left(normalized).detach().clone()
        capture["right"] = module.Right(normalized).detach().clone()
        return output, capture

    def projected_terms(self, base_capture, donor_capture):
        F = self.F
        weight = self.model.transformer.h[4].mlp.Down.weight.float()
        left_base = base_capture["left"].float()
        right_base = base_capture["right"].float()
        delta_left = donor_capture["left"].float() - left_base
        delta_right = donor_capture["right"].float() - right_base
        hidden = {
            "left_change": delta_left * right_base,
            "right_change": left_base * delta_right,
            "bilinear_interaction": delta_left * delta_right,
        }
        projected = {
            name: F.linear(value, weight, None) for name, value in hidden.items()
        }
        direct_fp32 = F.linear(
            donor_capture["left"].float() * donor_capture["right"].float()
            - left_base * right_base,
            weight,
            None,
        )
        reconstruction = sum(projected.values())
        error = float((reconstruction - direct_fp32).abs().max())
        return projected, error

    def intervene_factors(
        self,
        base_batch: producer.ModelBatch,
        donor_batch: producer.ModelBatch,
        base_capture,
        donor_capture,
        subset: tuple[str, ...],
    ):
        if subset not in subsets():
            raise ExperimentError("factorial arm changed")
        projected, error = self.projected_terms(base_capture, donor_capture)
        state = base_capture["resid5"].clone()
        for i, bank in enumerate(block4.source_positions(base_batch, donor_batch)):
            for position in bank:
                delta = self.torch.zeros_like(state[i, position], dtype=self.torch.float32)
                for factor in subset:
                    delta += projected[factor][i, position]
                state[i, position] = (state[i, position].float() + delta).to(state.dtype)
        output = self.suffix_from_resid5(
            base_batch, state, base_capture["x0"], base_capture["v1_after4"]
        )
        return output, error


def summary(values: list[float]) -> dict[str, object]:
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
        "schema": "aspectual_anchor_mlp4_bilinear_response_factorial_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "row_count": len(rows),
        "factors": list(FACTORS),
        "factorial_arm_count": len(subsets()),
        "direct_ceiling_arms": 1,
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
    backend = BilinearBackend.load("cuda")
    native = {}
    batch_pairs = []
    manual_logit_max_abs = 0.0
    tensor_reconstruction_max_abs = 0.0
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            captures = {}
            for side, batch in (("base", base_batch), ("donor", donor_batch)):
                reference = backend.native(batch, capture=False)
                manual, capture = backend.capture_bilinear(batch)
                forward_calls += 2
                evaluations += 2 * len(chunk)
                for reference_pair, manual_pair in zip(reference.answer_foil, manual.answer_foil):
                    manual_logit_max_abs = max(
                        manual_logit_max_abs,
                        abs(reference_pair[0] - manual_pair[0]),
                        abs(reference_pair[1] - manual_pair[1]),
                    )
                for row_id, pair in zip(batch.row_ids, reference.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(row_id, side)] = producer.NativeLogitEvidence(
                        row_id, family, side, answer, foil  # type: ignore[arg-type]
                    )
                captures[side] = capture
            _, error = backend.projected_terms(captures["base"], captures["donor"])
            tensor_reconstruction_max_abs = max(tensor_reconstruction_max_abs, error)
            batch_pairs.append((family, tuple(chunk), base_batch, donor_batch, captures["base"], captures["donor"]))

    native_capability = True
    for family in ("A1", "A2"):
        for direction in ("past_to_present", "present_to_past"):
            cell_rows = [row for row in rows if row["transform_id"] == family and row["direction_id"] == direction]
            for side in ("base", "donor"):
                accuracy = sum(native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows) / len(cell_rows)
                native_capability = native_capability and accuracy >= 0.85

    arm_values = {subset: {"A1": [], "A2": []} for subset in subsets()}
    ceiling_values = {"A1": [], "A2": []}
    logits = {}
    raw_records = []
    for subset in subsets():
        name = arm_id(subset)
        for family, chunk, base_batch, donor_batch, base_capture, donor_capture in batch_pairs:
            output, _ = backend.intervene_factors(
                base_batch, donor_batch, base_capture, donor_capture, subset
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
                arm_values[subset][family].append(recovery)
                logits[(name, row_id)] = (answer, foil)
                raw_records.append({"arm_id": name, "family": family, "row_id": row_id, "answer_logit": answer, "foil_logit": foil, "recovery": recovery})
    for family, chunk, base_batch, donor_batch, base_capture, donor_capture in batch_pairs:
        output = backend.intervene(
            base_batch, donor_batch, base_capture, donor_capture, ("mlp4",)
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
            ceiling_values[family].append(recovery)
            logits[("direct_mlp4_ceiling", row_id)] = (answer, foil)
            raw_records.append({"arm_id": "direct_mlp4_ceiling", "family": family, "row_id": row_id, "answer_logit": answer, "foil_logit": foil, "recovery": recovery})

    summaries = {}
    values = {}
    for subset in subsets():
        families = {family: summary(arm_values[subset][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        summaries[arm_id(subset)] = {"factors": list(subset), "families": families, "mean_target_recovery": target}
        values[subset] = target
    ceiling_summary = {family: summary(ceiling_values[family]) for family in ("A1", "A2")}
    ceiling_target = statistics.fmean(ceiling_summary[family]["mean_recovery"] for family in ("A1", "A2"))
    full = FACTORS
    full_name = arm_id(full)
    closure_max_abs = max(
        abs(value - reference)
        for row in rows
        for value, reference in zip(
            logits[(full_name, str(row["row_id"]))],
            logits[("direct_mlp4_ceiling", str(row["row_id"]))],
        )
    )
    shapley = {}
    n = len(FACTORS)
    all_subsets = subsets()
    for factor in FACTORS:
        total = 0.0
        for subset in all_subsets:
            if factor in subset:
                continue
            extended = tuple(item for item in FACTORS if item in set(subset) | {factor})
            weight = math.factorial(len(subset)) * math.factorial(n - len(subset) - 1) / math.factorial(n)
            total += weight * (values[extended] - values[subset])
        shapley[factor] = total
    ranked = sorted(FACTORS, key=lambda factor: (-shapley[factor], factor))
    top_two = tuple(factor for factor in FACTORS if factor in set(ranked[:2]))
    retained_fraction = values[top_two] / values[full]
    winner = ranked[0]
    without_winner = tuple(factor for factor in FACTORS if factor != winner)
    family_drops = {
        family: summaries[full_name]["families"][family]["mean_recovery"]
        - summaries[arm_id(without_winner)]["families"][family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    pred_a = native_capability and manual_logit_max_abs <= 1.0e-4 and tensor_reconstruction_max_abs <= 1.0e-3 and closure_max_abs <= 0.125
    pred_b = abs(values[full] - 0.3189104741714948) <= 0.02 and all(
        summaries[full_name]["families"][family]["mean_recovery"] > 0.0
        and summaries[full_name]["families"][family]["direction_fraction"] >= 0.80
        for family in ("A1", "A2")
    )
    pred_c = retained_fraction >= 0.80 and all(
        summaries[arm_id(top_two)]["families"][family]["direction_fraction"] >= 0.80
        for family in ("A1", "A2")
    )
    pred_d = shapley[winner] >= 0.10 and all(drop > 0.0 for drop in family_drops.values())
    expected_records = (len(all_subsets) + 1) * len(rows)
    pred_e = len(raw_records) == expected_records and len(logits) == expected_records and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    terminal = "screen" if pred_a and pred_b and pred_c and pred_d and pred_e else ("null" if pred_a and pred_e else "invalid")
    reason = {"screen": "mlp4_bilinear_two_term_subprogram", "null": "bilinear_response_compression_failed", "invalid": "bilinear_closure_capability_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_mlp4_bilinear_response_factorial_result_v1",
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
            "pred_a_exact_bilinear_closure": pred_a,
            "pred_b_parent_mlp4_recurrence": pred_b,
            "pred_c_two_term_compression": pred_c,
            "pred_d_dominant_factor": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "manual_scored_logit_max_abs": manual_logit_max_abs,
            "fp32_projected_tensor_reconstruction_max_abs": tensor_reconstruction_max_abs,
            "full_factor_to_direct_ceiling_scored_logit_max_abs": closure_max_abs,
            "factorial_shapley_target_recovery": shapley,
            "factor_ranking": ranked,
            "dominant_factor": winner,
            "dominant_factor_full_removal_family_drops": family_drops,
            "selected_two_factor_subprogram": list(top_two),
            "two_factor_retained_fraction": retained_fraction,
            "factorial_arms": summaries,
            "direct_mlp4_ceiling": {"families": ceiling_summary, "mean_target_recovery": ceiling_target},
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
            "test the fixed two-factor MLP4 response inside the complete source-to-head path"
            if terminal == "screen" else "retain whole MLP4 as the contextual writer"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": result["predictions"], "shapley": shapley, "top_two": list(top_two), "retained_fraction": retained_fraction, "closure_max_abs": closure_max_abs, "tensor_error": tensor_reconstruction_max_abs, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
