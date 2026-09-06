#!/usr/bin/env python3
"""Exact attention/MLP lattice for post-selected suffix blocks 12 and 14."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_exact_control pred_b_shapley_closure_and_selection pred_c_component_increment_compression pred_d_component_total_sufficiency pred_e_exact_coverage
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_battery_integration_contract as battery
import circuit_fast_screen_producer as producer
import run_aspectual_anchor_mlp11_15_bilinear_compression_split_v2 as mlp_parent
import run_aspectual_anchor_sparse_suffix_missing_block_compression_split_v1 as block_parent
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_suffix_block12_14_component_factorial_split_v1.json"
BLOCK_RESULT = ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_missing_block_compression_split_v2_result.json"
BLOCK_RUNNER = ROOT / "ops/run_aspectual_anchor_sparse_suffix_missing_block_compression_split_v2.py"
SPARSE_RESULT = ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_recurrence_confirmation_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_suffix_block12_14_component_factorial_split_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.suffix_block12_14_component_factorial_split_v1"
EXPECTED_PRIOR_SHA256 = "4664b1b1b397002e7d3d31005322827e8da8e0c89d83a3536bd81cb5ad380841"
EXPECTED_BLOCK_RESULT_SHA256 = "8f2f795564e7071b2ec467165a679ddc22e24a9e0b02e656628df957fd59ad69"
EXPECTED_BLOCK_RUNNER_SHA256 = "f94eae689fc15689baa3f42683ad71d18405f7c9caf86bfc792b8297e733d8fc"
EXPECTED_SPARSE_RESULT_SHA256 = "db666e5e006d5ecb3300806399c682441ea92b990cb1155eaa76479899326ef2"
FACTORS = ("attention12", "mlp12", "attention14", "mlp14")
MODEL_FORWARDS_MAX = 54
EXAMPLE_EVALUATIONS_MAX = 432


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def subsets() -> tuple[tuple[str, ...], ...]:
    return tuple(combo for width in range(len(FACTORS) + 1) for combo in combinations(FACTORS, width))


def arm_id(selected: tuple[str, ...]) -> str:
    return "none" if not selected else "+".join(selected)


def summarize(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("recovery is missing or nonfinite")
    return {
        "count": len(values), "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def validate_static():
    expected = {
        PRIOR: EXPECTED_PRIOR_SHA256, BLOCK_RESULT: EXPECTED_BLOCK_RESULT_SHA256,
        BLOCK_RUNNER: EXPECTED_BLOCK_RUNNER_SHA256, SPARSE_RESULT: EXPECTED_SPARSE_RESULT_SHA256,
    }
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    block = json.loads(BLOCK_RESULT.read_text())
    sparse = json.loads(SPARSE_RESULT.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID or block.get("terminal") != "screen"
        or block.get("evidence_class") != "post_outcome_repair_replication"
        or block["score"]["confirmation"]["selected_blocks"] != [12, 14]
        or sparse.get("terminal") != "null" or len(subsets()) != 16
    ):
        raise ExperimentError("experiment authority changed")
    selection, confirmation, spec = block_parent.validate_static()
    spec = replace(
        spec, experiment_id="aspectual-anchor-suffix-block12-14-component-factorial-split-v1",
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT", forward_calls=MODEL_FORWARDS_MAX,
            example_evaluations=EXAMPLE_EVALUATIONS_MAX,
            backward_calls=0, model_updates=0, evidence_bytes=114688,
        ),
    )
    reference = {
        (record["family"], record["row_id"]): (record["answer_logit"], record["foil_logit"])
        for record in block["intervention_logits"]
        if record.get("phase") == "confirmation" and record.get("arm_id") == "selected_two"
    }
    if len(reference) != 16:
        raise ExperimentError("v2 confirmation reference changed")
    return selection, confirmation, spec, reference


class ComponentBackend(block_parent.MissingBlockBackend):
    def component_recurrence_readout(
        self, batch, role_banks, base_capture, hybrid_capture, terms, selected: tuple[str, ...]
    ):
        if any(factor not in FACTORS for factor in selected) or len(selected) != len(set(selected)):
            raise ExperimentError("component subset changed")
        selected_set = set(selected)
        deltas = [
            hybrid_capture["resid10"][i, query].float() - base_capture["resid10"][i, query].float()
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
                projected_mlp, _error = self.projected_mlp_terms(base_capture, hybrid_capture, boundary)
            next_deltas = []
            for i, query in enumerate(batch.semantic_positions):
                delta = lambda0 * deltas[i]
                if boundary in (11, 15):
                    delta = delta + projected_attention[i, query]
                    for factor in block_parent.sparse_parent.SELECTED_MLP[boundary]:
                        delta = delta + projected_mlp[factor][i, query]
                elif boundary in (12, 14):
                    if f"attention{boundary}" in selected_set:
                        delta = delta + hybrid_capture[f"attention{boundary}"][i, query].float() - base_capture[f"attention{boundary}"][i, query].float()
                    if f"mlp{boundary}" in selected_set:
                        delta = delta + hybrid_capture[f"mlp{boundary}"][i, query].float() - base_capture[f"mlp{boundary}"][i, query].float()
                next_deltas.append(delta)
            deltas = next_deltas
        state = base_capture["resid18"].clone()
        for i, query in enumerate(batch.semantic_positions):
            state[i, query] = (state[i, query].float() + deltas[i]).to(state.dtype)
        return self.final_readout(batch, state)


def main() -> None:
    selection, confirmation, spec, reference = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_suffix_block12_14_component_factorial_split_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "selection_rows": len(selection), "confirmation_rows": len(confirmation),
        "factors": list(FACTORS), "selection_arms": len(subsets()), "confirmation_arms": 3,
        "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = ComponentBackend.load("cuda")
    native, captures = {}, {"selection": [], "confirmation": []}
    writer_values = {phase: {family: [] for family in ("A1", "A2")} for phase in captures}
    selection_values = {arm_id(arm): {family: [] for family in ("A1", "A2")} for arm in subsets()}
    raw_records = []
    manual_base_max_abs = writer_tensor_error_max_abs = full_reference_max_abs = 0.0
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
                terms = {}
                for boundary in (11, 15):
                    bp, bv, _be = backend.attention_terms(base_batch, base_capture, boundary)
                    hp, hv, _he = backend.attention_terms(base_batch, hybrid_capture, boundary)
                    terms[boundary] = ((bp, bv), (hp, hv))
                captures[phase].append((family, chunk, base_batch, role_banks, base_capture, hybrid_capture, terms))

    for family, chunk, batch, role_banks, base_capture, hybrid_capture, terms in captures["selection"]:
        for selected in subsets():
            label = arm_id(selected)
            output = backend.component_recurrence_readout(batch, role_banks, base_capture, hybrid_capture, terms, selected)
            forward_calls += 1
            evaluations += len(chunk)
            for row, pair in zip(chunk, output.answer_foil):
                answer, foil, value = suffix.recovery(row, pair, native)
                selection_values[label][family].append(value)
                raw_records.append({"phase": "selection", "arm_id": label, "family": family, "row_id": str(row["row_id"]), "answer_logit": answer, "foil_logit": foil, "recovery": value})

    selection_summary, targets = {}, {}
    for selected in subsets():
        label = arm_id(selected)
        families = {family: summarize(selection_values[label][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        selection_summary[label] = {"factors": list(selected), "families": families, "mean_target_recovery": target}
        targets[frozenset(selected)] = target
    shapley = {}
    n = len(FACTORS)
    for factor in FACTORS:
        others = [item for item in FACTORS if item != factor]
        value = 0.0
        for width in range(n):
            weight = math.factorial(width) * math.factorial(n - width - 1) / math.factorial(n)
            for combo in combinations(others, width):
                subset = frozenset(combo)
                value += weight * (targets[subset | {factor}] - targets[subset])
        shapley[factor] = value
    shapley_closure_error = abs(sum(shapley.values()) - (targets[frozenset(FACTORS)] - targets[frozenset()]))
    selected_components = tuple(
        max((f"attention{boundary}", f"mlp{boundary}"), key=lambda factor: (shapley[factor], factor.startswith("attention")))
        for boundary in (12, 14)
    )

    confirmation_values = {label: {family: [] for family in ("A1", "A2")} for label in ("no_12_14", "selected_one_per_block", "full_12_14")}
    confirmation_sets = {"no_12_14": (), "selected_one_per_block": selected_components, "full_12_14": FACTORS}
    for family, chunk, batch, role_banks, base_capture, hybrid_capture, terms in captures["confirmation"]:
        for label, selected in confirmation_sets.items():
            output = backend.component_recurrence_readout(batch, role_banks, base_capture, hybrid_capture, terms, selected)
            forward_calls += 1
            evaluations += len(chunk)
            for row, pair in zip(chunk, output.answer_foil):
                answer, foil, value = suffix.recovery(row, pair, native)
                confirmation_values[label][family].append(value)
                raw_records.append({"phase": "confirmation", "arm_id": label, "family": family, "row_id": str(row["row_id"]), "answer_logit": answer, "foil_logit": foil, "recovery": value})
                if label == "full_12_14":
                    expected = reference[(family, str(row["row_id"]))]
                    full_reference_max_abs = max(full_reference_max_abs, abs(answer - expected[0]), abs(foil - expected[1]))

    confirmation_summary, confirmation_targets = {}, {}
    for label in confirmation_sets:
        families = {family: summarize(confirmation_values[label][family]) for family in ("A1", "A2")}
        target = statistics.fmean(families[family]["mean_recovery"] for family in ("A1", "A2"))
        confirmation_summary[label] = {"factors": list(confirmation_sets[label]), "families": families, "mean_target_recovery": target}
        confirmation_targets[label] = target
    increment_fraction = (
        (confirmation_targets["selected_one_per_block"] - confirmation_targets["no_12_14"])
        / (confirmation_targets["full_12_14"] - confirmation_targets["no_12_14"])
    )
    total_fraction = confirmation_targets["selected_one_per_block"] / confirmation_targets["full_12_14"]
    family_increments = {
        family: confirmation_summary["selected_one_per_block"]["families"][family]["mean_recovery"] - confirmation_summary["no_12_14"]["families"][family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    capability = all(native[(str(row["row_id"]), side)].margin > 0.0 for row in selection + confirmation for side in ("base", "donor"))
    pred_a = capability and manual_base_max_abs <= 1.0e-4 and writer_tensor_error_max_abs <= 2.0e-3 and full_reference_max_abs <= 0.125
    pred_b = shapley_closure_error <= 1.0e-10 and len(selected_components) == 2 and {int(factor[-2:]) for factor in selected_components} == {12, 14}
    pred_c = increment_fraction >= 0.80 and all(value > 0.0 for value in family_increments.values())
    pred_d = total_fraction >= 0.90 and all(confirmation_summary["selected_one_per_block"]["families"][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_e = len(raw_records) == 336 and len({(record["phase"], record["arm_id"], record["row_id"]) for record in raw_records}) == 336 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    terminal = "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e)) else ("null" if pred_a and pred_b and pred_e else "invalid")
    reason = {"screen": "one_component_per_selected_block_is_sufficient", "null": "one_component_per_selected_block_is_insufficient", "invalid": "authority_control_shapley_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_suffix_block12_14_component_factorial_split_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "block_selection_sha256": EXPECTED_BLOCK_RESULT_SHA256,
        "evidence_class": "conditional_post_selection_component_resolution",
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_capability_and_exact_control": pred_a,
            "pred_b_shapley_closure_and_selection": pred_b,
            "pred_c_component_increment_compression": pred_c,
            "pred_d_component_total_sufficiency": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "selection": {"arms": selection_summary, "shapley": shapley, "shapley_closure_error": shapley_closure_error, "selected_components": list(selected_components)},
            "confirmation": {"arms": confirmation_summary, "selected_components": list(selected_components), "selected_increment_fraction": increment_fraction, "selected_total_to_full_fraction": total_fraction, "selected_family_increments": family_increments},
            "manual_base_scored_logit_max_abs": manual_base_max_abs,
            "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "full_12_14_to_v2_selected_two_logit_max_abs": full_reference_max_abs,
            "forward_calls": forward_calls, "example_evaluations": evaluations, "raw_record_count": len(raw_records),
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records, "terminal": terminal, "reason": reason,
        "next_action": "resolve selected block components into heads/sources or bilinear MLP terms" if terminal == "screen" else "retain full attention+MLP writes at the insufficient selected boundary",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": result["predictions"], "shapley": shapley, "selected_components": list(selected_components), "increment_fraction": increment_fraction, "total_fraction": total_fraction, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
