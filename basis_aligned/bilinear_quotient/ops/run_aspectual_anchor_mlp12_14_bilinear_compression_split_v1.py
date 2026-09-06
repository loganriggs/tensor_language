#!/usr/bin/env python3
"""Split exact bilinear-response compression of suffix MLP12 and MLP14."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_exact_instruments pred_b_shapley_closure_and_selection pred_c_bilinear_increment_compression pred_d_bilinear_total_sufficiency pred_e_exact_coverage
from __future__ import annotations

from itertools import combinations
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_fast_screen_producer as producer
import run_aspectual_anchor_mlp11_15_bilinear_compression_split_v1 as mlp_engine
import run_aspectual_anchor_mlp11_15_bilinear_compression_split_v2 as mlp_parent
import run_aspectual_anchor_sparse_suffix_missing_block_compression_split_v1 as block_parent
import run_aspectual_anchor_suffix_block12_14_component_factorial_split_v1 as component_parent
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp12_14_bilinear_compression_split_v1.json"
COMPONENT_RESULT = ROOT / "circuits/followups/aspectual_anchor_suffix_block12_14_component_factorial_split_v1_result.json"
COMPONENT_RUNNER = ROOT / "ops/run_aspectual_anchor_suffix_block12_14_component_factorial_split_v1.py"
BILINEAR_ENGINE = ROOT / "ops/run_aspectual_anchor_mlp11_15_bilinear_compression_split_v2.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp12_14_bilinear_compression_split_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp12_14_bilinear_compression_split_v1"
EXPECTED_PRIOR_SHA256 = "7f9e6a8ae4aaa814a8c1ac98649e81ced388fd3ed5eea9509b4028b773f14d2b"
EXPECTED_COMPONENT_RESULT_SHA256 = "f064d136d8d094da01e6aa15e3084242aa214d9b98ec2679388342f5f78049d7"
EXPECTED_COMPONENT_RUNNER_SHA256 = "fe654f784e00d68dce083617666f540a059f8bd9b17489567ecf51f1f23d1172"
EXPECTED_BILINEAR_ENGINE_SHA256 = "037c79096ac6e643cb533f8298b8acd0799e5cedae1efa65fccef8b2d5ba6b3b"
FACTORS = ("left_change", "right_change", "bilinear_interaction")
BOUNDARIES = (12, 14)
MODEL_FORWARDS_MAX = 54
EXAMPLE_EVALUATIONS_MAX = 432


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subsets() -> tuple[tuple[str, ...], ...]:
    return tuple(combo for width in range(len(FACTORS) + 1) for combo in combinations(FACTORS, width))


def factor_label(selected: tuple[str, ...]) -> str:
    return "none" if not selected else "+".join(selected)


def shapley(targets: dict[tuple[str, ...], float]) -> dict[str, float]:
    values = {}
    for factor in FACTORS:
        others = tuple(other for other in FACTORS if other != factor)
        contribution = 0.0
        for width in range(len(others) + 1):
            weight = math.factorial(width) * math.factorial(len(FACTORS) - width - 1) / math.factorial(len(FACTORS))
            for combo in combinations(others, width):
                base = tuple(item for item in FACTORS if item in combo)
                added = tuple(item for item in FACTORS if item in set(combo) | {factor})
                contribution += weight * (targets[added] - targets[base])
        values[factor] = contribution
    return values


def validate_static():
    for path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256, COMPONENT_RESULT: EXPECTED_COMPONENT_RESULT_SHA256,
        COMPONENT_RUNNER: EXPECTED_COMPONENT_RUNNER_SHA256, BILINEAR_ENGINE: EXPECTED_BILINEAR_ENGINE_SHA256,
    }.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    parent = json.loads(COMPONENT_RESULT.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen"
        or parent["score"]["confirmation"]["selected_components"] != ["mlp12", "mlp14"]
        or len(subsets()) != 8
    ):
        raise ExperimentError("experiment authority changed")
    selection, confirmation, spec, _reference = component_parent.validate_static()
    reference = {}
    for record in parent["intervention_logits"]:
        if (
            record.get("phase") == "selection" and record.get("arm_id") == "mlp12+mlp14"
        ) or (
            record.get("phase") == "confirmation" and record.get("arm_id") == "selected_one_per_block"
        ):
            reference[(record["phase"], record["family"], record["row_id"])] = (record["answer_logit"], record["foil_logit"])
    if len(reference) != 32:
        raise ExperimentError("parent MLP reference changed")
    return selection, confirmation, spec, reference


class BilinearSuffixBackend(component_parent.ComponentBackend):
    def bilinear_recurrence_readout(
        self, batch, role_banks, base_capture, hybrid_capture, attention_terms,
        mlp_terms, factors_by_boundary: dict[int, tuple[str, ...]],
    ):
        if set(factors_by_boundary) != set(BOUNDARIES) or any(
            selected not in subsets() for selected in factors_by_boundary.values()
        ):
            raise ExperimentError("bilinear factor set changed")
        deltas = [
            hybrid_capture["resid10"][i, query].float() - base_capture["resid10"][i, query].float()
            for i, query in enumerate(batch.semantic_positions)
        ]
        for boundary in range(10, 18):
            lambda0 = self.model.transformer.h[boundary].lambdas[0].float()
            projected_attention = projected_mlp = None
            if boundary in (11, 15):
                projected_attention = self.projected_source_delta(
                    batch, role_banks, *attention_terms[boundary], boundary,
                    mlp_parent.SOURCE_BANK_BY_BOUNDARY[boundary],
                )
                projected_mlp, _error = self.projected_mlp_terms(base_capture, hybrid_capture, boundary)
            next_deltas = []
            for i, query in enumerate(batch.semantic_positions):
                delta = lambda0 * deltas[i]
                if boundary in (11, 15):
                    delta = delta + projected_attention[i, query]
                    for factor in block_parent.sparse_parent.SELECTED_MLP[boundary]:
                        delta = delta + projected_mlp[factor][i, query]
                elif boundary in BOUNDARIES:
                    for factor in factors_by_boundary[boundary]:
                        delta = delta + mlp_terms[boundary][factor][i, query]
                next_deltas.append(delta)
            deltas = next_deltas
        state = base_capture["resid18"].clone()
        for i, query in enumerate(batch.semantic_positions):
            state[i, query] = (state[i, query].float() + deltas[i]).to(state.dtype)
        return self.final_readout(batch, state)


def main() -> None:
    selection, confirmation, spec, reference = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_mlp12_14_bilinear_compression_split_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "selection_rows": len(selection), "confirmation_rows": len(confirmation),
        "boundaries": list(BOUNDARIES), "factors": list(FACTORS),
        "selection_arms": 16, "confirmation_arms": 3,
        "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = component_parent.utc_now(), time.perf_counter()
    backend = BilinearSuffixBackend.load("cuda")
    native, captures = {}, {"selection": [], "confirmation": []}
    writer_values = {phase: {family: [] for family in ("A1", "A2")} for phase in captures}
    raw_records = []
    manual_base_max_abs = mlp_tensor_error_max_abs = full_reference_max_abs = 0.0
    writer_tensor_error_max_abs = 0.0
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
                writer_output, hybrid_capture, writer_error = backend.capture_writer_suffix_heads(base_batch, donor_batch, base_bilinear, donor_bilinear)
                forward_calls += 4
                evaluations += 4 * len(chunk)
                writer_tensor_error_max_abs = max(writer_tensor_error_max_abs, writer_error)
                for native_pair, manual_pair in zip(base_native.answer_foil, base_manual.answer_foil):
                    manual_base_max_abs = max(manual_base_max_abs, abs(native_pair[0] - manual_pair[0]), abs(native_pair[1] - manual_pair[1]))
                for side, output in (("base", base_native), ("donor", donor_native)):
                    for row, pair in zip(chunk, output.answer_foil):
                        answer, foil = producer._finite_pair(pair)
                        native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(str(row["row_id"]), family, side, answer, foil)
                for row, pair in zip(chunk, writer_output.answer_foil):
                    answer, foil, value = suffix.recovery(row, pair, native)
                    writer_values[phase][family].append(value)
                    raw_records.append({"phase": phase, "arm_id": "writer_two_term", "family": family, "row_id": str(row["row_id"]), "answer_logit": answer, "foil_logit": foil, "recovery": value})
                attention_terms = {}
                for boundary in (11, 15):
                    bp, bv, _be = backend.attention_terms(base_batch, base_capture, boundary)
                    hp, hv, _he = backend.attention_terms(base_batch, hybrid_capture, boundary)
                    attention_terms[boundary] = ((bp, bv), (hp, hv))
                mlp_terms = {}
                for boundary in BOUNDARIES:
                    projected, error = backend.projected_mlp_terms(base_capture, hybrid_capture, boundary)
                    mlp_terms[boundary] = projected
                    mlp_tensor_error_max_abs = max(mlp_tensor_error_max_abs, error)
                captures[phase].append((family, chunk, base_batch, role_banks, base_capture, hybrid_capture, attention_terms, mlp_terms))

    selection_values = {boundary: {factor_label(selected): {family: [] for family in ("A1", "A2")} for selected in subsets()} for boundary in BOUNDARIES}
    for family, chunk, batch, role_banks, base_capture, hybrid_capture, attention_terms, mlp_terms in captures["selection"]:
        for boundary in BOUNDARIES:
            other = 26 - boundary
            for selected in subsets():
                label = f"boundary{boundary}:{factor_label(selected)}"
                factor_map = {boundary: selected, other: FACTORS}
                output = backend.bilinear_recurrence_readout(batch, role_banks, base_capture, hybrid_capture, attention_terms, mlp_terms, factor_map)
                forward_calls += 1
                evaluations += len(chunk)
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil, value = suffix.recovery(row, pair, native)
                    selection_values[boundary][factor_label(selected)][family].append(value)
                    raw_records.append({"phase": "selection", "arm_id": label, "family": family, "row_id": str(row["row_id"]), "answer_logit": answer, "foil_logit": foil, "recovery": value})
                    if selected == FACTORS:
                        expected = reference[("selection", family, str(row["row_id"]))]
                        full_reference_max_abs = max(full_reference_max_abs, abs(answer - expected[0]), abs(foil - expected[1]))

    selection_summary, shapley_values, selected_by_boundary, closure = {}, {}, {}, {}
    for boundary in BOUNDARIES:
        targets, arms = {}, {}
        for selected in subsets():
            label = factor_label(selected)
            families = {family: component_parent.summarize(selection_values[boundary][label][family]) for family in ("A1", "A2")}
            target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
            arms[label] = {"factors": list(selected), "families": families, "mean_target_recovery": target}
            targets[selected] = target
        values = shapley(targets)
        chosen = tuple(factor for factor in FACTORS if factor in sorted(FACTORS, key=lambda factor: (-values[factor], FACTORS.index(factor)))[:2])
        shapley_values[str(boundary)] = values
        selected_by_boundary[str(boundary)] = list(chosen)
        closure[str(boundary)] = abs(sum(values.values()) - (targets[FACTORS] - targets[()]))
        selection_summary[str(boundary)] = arms

    confirmation_sets = {
        "no_mlp12_14": {12: (), 14: ()},
        "selected_two_each": {boundary: tuple(selected_by_boundary[str(boundary)]) for boundary in BOUNDARIES},
        "full_mlp12_14": {12: FACTORS, 14: FACTORS},
    }
    confirmation_values = {label: {family: [] for family in ("A1", "A2")} for label in confirmation_sets}
    for family, chunk, batch, role_banks, base_capture, hybrid_capture, attention_terms, mlp_terms in captures["confirmation"]:
        for label, factor_map in confirmation_sets.items():
            output = backend.bilinear_recurrence_readout(batch, role_banks, base_capture, hybrid_capture, attention_terms, mlp_terms, factor_map)
            forward_calls += 1
            evaluations += len(chunk)
            for row, pair in zip(chunk, output.answer_foil):
                answer, foil, value = suffix.recovery(row, pair, native)
                confirmation_values[label][family].append(value)
                raw_records.append({"phase": "confirmation", "arm_id": label, "family": family, "row_id": str(row["row_id"]), "answer_logit": answer, "foil_logit": foil, "recovery": value})
                if label == "full_mlp12_14":
                    expected = reference[("confirmation", family, str(row["row_id"]))]
                    full_reference_max_abs = max(full_reference_max_abs, abs(answer - expected[0]), abs(foil - expected[1]))

    confirmation_summary, targets = {}, {}
    for label in confirmation_sets:
        families = {family: component_parent.summarize(confirmation_values[label][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        confirmation_summary[label] = {"factors_by_boundary": {str(k): list(v) for k, v in confirmation_sets[label].items()}, "families": families, "mean_target_recovery": target}
        targets[label] = target
    increment_fraction = (targets["selected_two_each"] - targets["no_mlp12_14"]) / (targets["full_mlp12_14"] - targets["no_mlp12_14"])
    total_fraction = targets["selected_two_each"] / targets["full_mlp12_14"]
    family_increments = {family: confirmation_summary["selected_two_each"]["families"][family]["mean_recovery"] - confirmation_summary["no_mlp12_14"]["families"][family]["mean_recovery"] for family in ("A1", "A2")}
    capability = all(native[(str(row["row_id"]), side)].margin > 0.0 for row in selection + confirmation for side in ("base", "donor"))
    pred_a = capability and manual_base_max_abs <= 1.0e-4 and writer_tensor_error_max_abs <= 2.0e-3 and mlp_tensor_error_max_abs <= 5.0e-3 and full_reference_max_abs <= 0.125
    pred_b = all(closure[str(boundary)] <= 1.0e-10 and len(selected_by_boundary[str(boundary)]) == 2 for boundary in BOUNDARIES)
    pred_c = increment_fraction >= 0.85 and all(value > 0.0 for value in family_increments.values())
    pred_d = total_fraction >= 0.95 and all(confirmation_summary["selected_two_each"]["families"][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_e = len(raw_records) == 336 and len({(record["phase"], record["arm_id"], record["row_id"]) for record in raw_records}) == 336 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    terminal = "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e)) else ("null" if pred_a and pred_b and pred_e else "invalid")
    reason = {"screen": "two_bilinear_terms_per_suffix_mlp_are_sufficient", "null": "two_bilinear_terms_per_suffix_mlp_are_insufficient", "invalid": "authority_instrument_control_shapley_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_mlp12_14_bilinear_compression_split_result_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": component_parent.utc_now(), "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "component_parent_sha256": EXPECTED_COMPONENT_RESULT_SHA256,
        "evidence_class": "conditional_post_selection_bilinear_resolution", "dryrun": dryrun,
        "predictions": {"pred_a_authority_capability_and_exact_instruments": pred_a, "pred_b_shapley_closure_and_selection": pred_b, "pred_c_bilinear_increment_compression": pred_c, "pred_d_bilinear_total_sufficiency": pred_d, "pred_e_exact_coverage": pred_e},
        "score": {
            "selection": {"arms": selection_summary, "shapley": shapley_values, "shapley_closure_error": closure, "selected_factors_by_boundary": selected_by_boundary},
            "confirmation": {"arms": confirmation_summary, "selected_factors_by_boundary": selected_by_boundary, "selected_increment_fraction": increment_fraction, "selected_total_to_full_fraction": total_fraction, "selected_family_increments": family_increments},
            "manual_base_scored_logit_max_abs": manual_base_max_abs, "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "mlp12_14_bilinear_tensor_reconstruction_max_abs": mlp_tensor_error_max_abs, "full_terms_to_parent_logit_max_abs": full_reference_max_abs,
            "forward_calls": forward_calls, "example_evaluations": evaluations, "raw_record_count": len(raw_records), "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records, "terminal": terminal, "reason": reason,
        "next_action": "compile selected MLP12/14 bilinear-response terms into transparent program v6" if terminal == "screen" else "retain all three response terms at the insufficient MLP boundary",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": result["predictions"], "shapley": shapley_values, "selected_factors_by_boundary": selected_by_boundary, "increment_fraction": increment_fraction, "total_fraction": total_fraction, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
