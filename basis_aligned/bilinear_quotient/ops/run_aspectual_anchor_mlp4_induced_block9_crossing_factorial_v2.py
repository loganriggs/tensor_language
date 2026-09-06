#!/usr/bin/env python3
# BQGATE: frozen A-E block9 crossing predictions; CUDA is managed-queue only.
"""Exact carried/attention/MLP block9 crossing of the two-term MLP4 signal."""

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
import run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1 as path
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2.json"
PARENT = ROOT / "circuits/followups/aspectual_anchor_mlp4_induced_l9_head_sweep_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp4_induced_l9_head_sweep_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp4_induced_block9_crossing_factorial_v2"
EXPECTED_PRIOR_SHA256 = "2be16da2e0350749c6837edf43057fbcd49f776a3ba6f8099964fc022d03e5b2"
EXPECTED_PARENT_SHA256 = "0d092efa06ad697b419253262bf05db2ed2e5e2cbb0f0d3a5ff23aac9021e2b5"
EXPECTED_PARENT_RUNNER_SHA256 = "38a945b0b08d588d49e107abd19d8b09ec744c289c27859fe56a84eccbf6a126"
EXPECTED_AUTHORITY_SHA256 = "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3"
WRITER_FACTORS = ("left_change", "right_change")
FACTORS = ("carried9", "attention9", "mlp9")
MODEL_FORWARDS_MAX = 28
EXAMPLE_EVALUATIONS_MAX = 896


class ExperimentError(RuntimeError):
    pass


def sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def subsets():
    return tuple(
        subset
        for width in range(len(FACTORS) + 1)
        for subset in itertools.combinations(FACTORS, width)
    )


def arm_id(subset: tuple[str, ...]) -> str:
    return "empty" if not subset else "+".join(subset)


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
    if parent.get("terminal") != "null" or parent["score"]["licensed_additional_heads"]:
        raise ExperimentError("parent null changed")
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


class CrossingBackend(path.PathBackend):
    def capture_block9(self, batch: producer.ModelBatch):
        torch, F, model = self.torch, self.F, self.model
        tokens, lengths = self._tensor_batch(batch)
        capture = {}
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x0 = x
            v1 = None
            for layer, block in enumerate(model.transformer.h):
                if layer == 9:
                    capture["resid9"] = x.detach().clone()
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                x = live + attention
                mlp_output = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                x = x + mlp_output
                if layer == 9:
                    capture.update({
                        "attention9": attention.detach().clone(),
                        "mlp9": mlp_output.detach().clone(),
                        "resid10": x.detach().clone(),
                        "x0": x0.detach().clone(),
                        "v1_after9": v1.detach().clone(),
                    })
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
        expected = {"resid9", "attention9", "mlp9", "resid10", "x0", "v1_after9"}
        if set(capture) != expected:
            raise ExperimentError("block9 capture incomplete")
        return producer.BatchOutput(values, {}), capture

    def capture_writer_block9(
        self,
        base_batch: producer.ModelBatch,
        donor_batch: producer.ModelBatch,
        base_capture,
        donor_capture,
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
            output, capture = self.capture_block9(base_batch)
        finally:
            handle.remove()
        return output, capture, tensor_error

    def suffix_from_resid10(self, batch: producer.ModelBatch, state, x0, v1):
        torch, F, model = self.torch, self.F, self.model
        lengths = tuple(len(row) for row in batch.token_rows)
        x = state
        with torch.no_grad():
            for layer in range(10, 18):
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

    def intervene_crossing(self, batch, base_capture, hybrid_capture, subset):
        if subset is not None and subset not in subsets():
            raise ExperimentError("crossing arm changed")
        state = base_capture["resid10"].clone()
        lambda0 = self.model.transformer.h[9].lambdas[0]
        for i, q in enumerate(batch.semantic_positions):
            if subset is None:
                state[i, q] = hybrid_capture["resid10"][i, q]
                continue
            delta = self.torch.zeros_like(state[i, q], dtype=self.torch.float32)
            if "carried9" in subset:
                delta += lambda0.float() * (
                    hybrid_capture["resid9"][i, q].float()
                    - base_capture["resid9"][i, q].float()
                )
            if "attention9" in subset:
                delta += (
                    hybrid_capture["attention9"][i, q].float()
                    - base_capture["attention9"][i, q].float()
                )
            if "mlp9" in subset:
                delta += (
                    hybrid_capture["mlp9"][i, q].float()
                    - base_capture["mlp9"][i, q].float()
                )
            state[i, q] = (state[i, q].float() + delta).to(state.dtype)
        return self.suffix_from_resid10(
            batch, state, base_capture["x0"], base_capture["v1_after9"]
        )


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
        "schema": "aspectual_anchor_mlp4_induced_block9_crossing_factorial_dryrun_v2",
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
    backend = CrossingBackend.load("cuda")
    native = {}
    arm_values = {subset: {"A1": [], "A2": []} for subset in subsets()}
    ceiling_values = {"A1": [], "A2": []}
    writer_values = {"A1": [], "A2": []}
    logits = {}
    raw_records = []
    manual_base_max_abs = 0.0
    tensor_reconstruction_max_abs = 0.0
    full_to_ceiling_max_abs = 0.0
    writer_tensor_error_max_abs = 0.0
    forward_calls = 0
    evaluations = 0

    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            base_native, base_bilinear = backend.capture_bilinear(base_batch)
            donor_native, donor_bilinear = backend.capture_bilinear(donor_batch)
            base_manual, base_crossing = backend.capture_block9(base_batch)
            writer_output, hybrid_crossing, writer_tensor_error = backend.capture_writer_block9(
                base_batch, donor_batch, base_bilinear, donor_bilinear
            )
            forward_calls += 4
            evaluations += 4 * len(chunk)
            writer_tensor_error_max_abs = max(
                writer_tensor_error_max_abs, writer_tensor_error
            )
            for reference, manual in zip(base_native.answer_foil, base_manual.answer_foil):
                manual_base_max_abs = max(
                    manual_base_max_abs,
                    abs(reference[0] - manual[0]),
                    abs(reference[1] - manual[1]),
                )
            lambda0 = backend.model.transformer.h[9].lambdas[0]
            for i, q in enumerate(base_batch.semantic_positions):
                reconstructed = (
                    lambda0.float() * (
                        hybrid_crossing["resid9"][i, q].float()
                        - base_crossing["resid9"][i, q].float()
                    )
                    + hybrid_crossing["attention9"][i, q].float()
                    - base_crossing["attention9"][i, q].float()
                    + hybrid_crossing["mlp9"][i, q].float()
                    - base_crossing["mlp9"][i, q].float()
                )
                direct = (
                    hybrid_crossing["resid10"][i, q].float()
                    - base_crossing["resid10"][i, q].float()
                )
                tensor_reconstruction_max_abs = max(
                    tensor_reconstruction_max_abs,
                    float((reconstructed - direct).abs().max()),
                )
            for side, output in (("base", base_native), ("donor", donor_native)):
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                        str(row["row_id"]), family, side, answer, foil
                    )
            outputs = {}
            for subset in subsets():
                outputs[arm_id(subset)] = backend.intervene_crossing(
                    base_batch, base_crossing, hybrid_crossing, subset
                )
            outputs["direct_resid10_ceiling"] = backend.intervene_crossing(
                base_batch, base_crossing, hybrid_crossing, None
            )
            forward_calls += len(subsets()) + 1
            evaluations += (len(subsets()) + 1) * len(chunk)
            for full_pair, direct_pair in zip(
                outputs[arm_id(FACTORS)].answer_foil,
                outputs["direct_resid10_ceiling"].answer_foil,
            ):
                full_to_ceiling_max_abs = max(
                    full_to_ceiling_max_abs,
                    abs(full_pair[0] - direct_pair[0]),
                    abs(full_pair[1] - direct_pair[1]),
                )
            outputs["writer_two_term"] = writer_output
            for arm, output in outputs.items():
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    row_id = str(row["row_id"])
                    recovery = kernel.signed_pairwise_donor_recovery(
                        -native[(row_id, "base")].margin,
                        native[(row_id, "donor")].margin,
                        -(answer - foil),
                    )
                    if arm == "direct_resid10_ceiling":
                        ceiling_values[family].append(recovery)
                    elif arm == "writer_two_term":
                        writer_values[family].append(recovery)
                    else:
                        subset = next(item for item in subsets() if arm_id(item) == arm)
                        arm_values[subset][family].append(recovery)
                    logits[(arm, row_id)] = (answer, foil)
                    raw_records.append({
                        "arm_id": arm, "family": family, "row_id": row_id,
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
    values = {}
    for subset in subsets():
        families = {family: summarize(arm_values[subset][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        summaries[arm_id(subset)] = {"factors": list(subset), "families": families, "mean_target_recovery": target}
        values[subset] = target
    ceiling_summary = {family: summarize(ceiling_values[family]) for family in ("A1", "A2")}
    writer_summary = {family: summarize(writer_values[family]) for family in ("A1", "A2")}
    ceiling_target = statistics.fmean(ceiling_summary[family]["mean_recovery"] for family in ("A1", "A2"))
    writer_target = statistics.fmean(writer_summary[family]["mean_recovery"] for family in ("A1", "A2"))
    shapley = {}
    for factor in FACTORS:
        total = 0.0
        for subset in subsets():
            if factor in subset:
                continue
            extended = tuple(item for item in FACTORS if item in set(subset) | {factor})
            weight = math.factorial(len(subset)) * math.factorial(len(FACTORS) - len(subset) - 1) / math.factorial(len(FACTORS))
            total += weight * (values[extended] - values[subset])
        shapley[factor] = total
    winner = max(FACTORS, key=lambda factor: shapley[factor])
    without_winner = tuple(factor for factor in FACTORS if factor != winner)
    family_drops = {
        family: summaries[arm_id(FACTORS)]["families"][family]["mean_recovery"]
        - summaries[arm_id(without_winner)]["families"][family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    full_retained = values[FACTORS] / writer_target
    pred_a = native_capability and manual_base_max_abs <= 1.0e-4 and writer_tensor_error_max_abs <= 2.0e-3 and tensor_reconstruction_max_abs <= 0.04 and full_to_ceiling_max_abs <= 0.125
    pred_b = abs(writer_target - 0.33379277118533013) <= 0.02 and all(writer_summary[family]["mean_recovery"] > 0.0 and writer_summary[family]["direction_fraction"] >= 0.80 for family in ("A1", "A2"))
    pred_c = full_retained >= 0.65 and all(summaries[arm_id(FACTORS)]["families"][family]["mean_recovery"] > 0.0 and summaries[arm_id(FACTORS)]["families"][family]["direction_fraction"] >= 0.80 for family in ("A1", "A2"))
    pred_d = shapley[winner] >= 0.10 and all(drop > 0.0 for drop in family_drops.values())
    expected_records = (len(subsets()) + 2) * len(rows)
    pred_e = len(raw_records) == expected_records and len(logits) == expected_records and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    terminal = "screen" if pred_a and pred_b and pred_c and pred_d and pred_e else ("null" if pred_a and pred_b and pred_e else "invalid")
    reason = {"screen": "block9_route_into_resid10_carrier", "null": "final_query_block9_crossing_insufficient", "invalid": "crossing_instrument_recurrence_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_mlp4_induced_block9_crossing_factorial_result_v2",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_exact_crossing_instrument": pred_a,
            "pred_b_writer_recurrence": pred_b,
            "pred_c_resid10_final_query_sufficiency": pred_c,
            "pred_d_dominant_crossing_factor": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "manual_base_scored_logit_max_abs": manual_base_max_abs,
            "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "block9_tensor_reconstruction_max_abs": tensor_reconstruction_max_abs,
            "full_to_direct_ceiling_scored_logit_max_abs": full_to_ceiling_max_abs,
            "factorial_arms": summaries,
            "direct_resid10_ceiling": {"families": ceiling_summary, "mean_target_recovery": ceiling_target},
            "writer_two_term": {"families": writer_summary, "mean_target_recovery": writer_target},
            "full_crossing_to_writer_retained_fraction": full_retained,
            "factorial_shapley_target_recovery": shapley,
            "dominant_factor": winner,
            "dominant_factor_full_removal_family_drops": family_drops,
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "raw_record_count": len(raw_records), "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records,
        "terminal": terminal, "reason": reason,
        "next_action": "compile the dominant block9 crossing into the circuit" if terminal == "screen" else "test indirect source-position routes beyond the final-query crossing",
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": result["predictions"], "full_to_writer": full_retained, "shapley": shapley, "winner": winner, "tensor_error": tensor_reconstruction_max_abs, "closure": full_to_ceiling_max_abs, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
