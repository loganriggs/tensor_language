#!/usr/bin/env python3
"""Two-way operational interchange of explicit v8 variables across constructions."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_capability_and_controls pred_b_target_program_effect pred_c_whole_variable_transfer pred_d_groupwise_operational_equivalence pred_e_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import aspectual_anchor_transparent_path_program_v8 as program
import circuit_fast_screen_producer as producer
import run_aspectual_anchor_mlp11_15_bilinear_compression_split_v2 as mlp_parent
import run_aspectual_anchor_mlp12_14_bilinear_compression_split_v1 as empirical
import run_aspectual_anchor_program_v7_fresh_construction_transfer_v1 as fresh_runner
import run_aspectual_anchor_program_v8_rank1_carrier_mediation_v1 as mediation_parent
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_program_v8_cross_construction_variable_interchange_v1.json"
PROGRAM = ROOT / "ops/aspectual_anchor_transparent_path_program_v8.py"
LEXICAL_BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
FRESH_BUILDER = ROOT / "ops/circuit_candidate_aspectual_fresh_construction_v2.py"
MEDIATION_RUNNER = ROOT / "ops/run_aspectual_anchor_program_v8_rank1_carrier_mediation_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_program_v8_cross_construction_variable_interchange_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.program_v8_cross_construction_variable_interchange_v1"
EXPECTED_PRIOR_SHA256 = "a8affe0e56d29f0c0f95fdd18a4b2131f38c71b1715e1df389d027d09254c395"
EXPECTED = {
    PROGRAM: "87eb67f3a96904534c8d3ddca5e1df59fa14efd88d1174e1cc2805435346bb57",
    LEXICAL_BUILDER: "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859",
    FRESH_BUILDER: "848332a12c22bf523573e015b6f8f0a38b5865db8b77434dcbe6a176d98370ac",
    MEDIATION_RUNNER: "79b184230fa5ccd594c082c634bffd13c61a12e49e47d38dac15fe95aa0298ce",
}
ARMS = ("target_full", "source_full", "swap_initial", "swap_attention", "swap_mlp")
MODEL_FORWARDS_MAX = 72
EXAMPLE_EVALUATIONS_MAX = 576


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("missing/nonfinite recovery")
    return {"count": len(values), "mean_recovery": statistics.fmean(values), "mean_absolute_recovery": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def match_key(row):
    return (row["transform_id"], row["reporter"], row["object_name"], row["direction_id"])


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    mediation_parent.validate_static()
    lexical_selection, lexical_confirmation, lexical_spec, _ = suffix.validate_static()
    fresh_rows_all, fresh_spec = fresh_runner.validate_static()
    lexical_rows = tuple(lexical_selection) + tuple(lexical_confirmation)
    fresh_rows = tuple(row for row in fresh_rows_all if row["transform_id"] in ("A1", "A2"))
    lexical_keys, fresh_keys = {match_key(row) for row in lexical_rows}, {match_key(row) for row in fresh_rows}
    if prior.get("candidate_id") != CANDIDATE_ID or len(lexical_rows) != len(fresh_rows) or len(lexical_rows) != 32 or lexical_keys != fresh_keys or len(lexical_keys) != 32:
        raise ExperimentError("candidate, population, or cross-construction bijection changed")
    return lexical_rows, lexical_spec, fresh_rows, fresh_spec


def main() -> None:
    lexical_rows, lexical_spec, fresh_rows, fresh_spec = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_program_v8_cross_construction_variable_interchange_dryrun_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "target_rows": 64, "arms": list(ARMS),
        "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = empirical.component_parent.utc_now(), time.perf_counter()
    backend = empirical.BilinearSuffixBackend.load("cuda")
    native, components = {}, {"lexical": {}, "fresh": {}}
    manual_base_max_abs = writer_tensor_error_max_abs = mlp_tensor_error_max_abs = 0.0
    forward_calls = evaluations = capture_chunks = 0
    lambdas = {boundary: backend.model.transformer.h[boundary].lambdas[0].float() for boundary in program.SUFFIX_BOUNDARIES}
    weights = {boundary: backend.model.transformer.h[boundary].mlp.Down.weight.float() for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}

    for panel, rows, spec in (("lexical", lexical_rows, lexical_spec), ("fresh", fresh_rows, fresh_spec)):
        for family in ("A1", "A2"):
            family_rows = [row for row in rows if row["transform_id"] == family]
            for chunk in producer._chunks(family_rows, spec.batch_size):
                capture_chunks += 1
                base_batch = producer._batch(spec, chunk, "base")
                donor_batch = producer._batch(spec, chunk, "donor")
                role_banks = backend.role_positions(base_batch, donor_batch)
                base_native, base_bilinear = backend.capture_bilinear(base_batch)
                donor_native, donor_bilinear = backend.capture_bilinear(donor_batch)
                base_manual, base_capture = backend.capture_suffix_heads(base_batch)
                _writer, hybrid_capture, writer_error = backend.capture_writer_suffix_heads(base_batch, donor_batch, base_bilinear, donor_bilinear)
                forward_calls += 4
                evaluations += 4 * len(chunk)
                writer_tensor_error_max_abs = max(writer_tensor_error_max_abs, writer_error)
                for native_pair, manual_pair in zip(base_native.answer_foil, base_manual.answer_foil):
                    manual_base_max_abs = max(manual_base_max_abs, abs(native_pair[0] - manual_pair[0]), abs(native_pair[1] - manual_pair[1]))
                for side, output in (("base", base_native), ("donor", donor_native)):
                    for row, pair in zip(chunk, output.answer_foil):
                        answer, foil = producer._finite_pair(pair)
                        native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(str(row["row_id"]), family, side, answer, foil)
                attention = {}
                for boundary in program.SUFFIX_SOURCE_BOUNDARIES:
                    bp, bv, _ = backend.attention_terms(base_batch, base_capture, boundary)
                    hp, hv, _ = backend.attention_terms(base_batch, hybrid_capture, boundary)
                    attention[boundary] = backend.projected_source_delta(base_batch, role_banks, (bp, bv), (hp, hv), boundary, mlp_parent.SOURCE_BANK_BY_BOUNDARY[boundary])
                mlp_states = {boundary: (*backend.mlp_states(base_capture, boundary), *backend.mlp_states(hybrid_capture, boundary)) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
                for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY:
                    _terms, error = backend.projected_mlp_terms(base_capture, hybrid_capture, boundary)
                    mlp_tensor_error_max_abs = max(mlp_tensor_error_max_abs, error)
                for i, (row, query) in enumerate(zip(chunk, base_batch.semantic_positions)):
                    components[panel][match_key(row)] = {
                        "row": row,
                        "base_resid18": base_capture["resid18"][i, query].float().detach(),
                        "initial": (hybrid_capture["resid10"][i, query].float() - base_capture["resid10"][i, query].float()).detach(),
                        "attention": {boundary: attention[boundary][i, query].detach() for boundary in program.SUFFIX_SOURCE_BOUNDARIES},
                        "mlp": {boundary: tuple(value[i, query].float().detach() for value in mlp_states[boundary]) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY},
                        "answer_id": base_batch.answer_ids[i], "foil_id": base_batch.foil_ids[i],
                    }

    values = {f"{panel}_{family}": {arm: [] for arm in ARMS} for panel in ("lexical", "fresh") for family in ("A1", "A2")}
    records = []
    for target_panel, source_panel in (("lexical", "fresh"), ("fresh", "lexical")):
        for key, target in components[target_panel].items():
            source = components[source_panel][key]
            family = key[0]
            panel = f"{target_panel}_{family}"
            groups = {
                "target_full": (target, target, target),
                "source_full": (source, source, source),
                "swap_initial": (source, target, target),
                "swap_attention": (target, source, target),
                "swap_mlp": (target, target, source),
            }
            for arm, (initial_owner, attention_owner, mlp_owner) in groups.items():
                delta = program.compiled_sparse_suffix_delta(
                    initial_owner["initial"], lambda0_by_boundary=lambdas,
                    source_attention_delta_by_boundary=attention_owner["attention"],
                    mlp_states_by_boundary=mlp_owner["mlp"], down_weight_by_boundary=weights,
                )
                answer_tensor, foil_tensor = program.exact_scored_pair(target["base_resid18"] + delta, backend.model.lm_head, answer_id=target["answer_id"], foil_id=target["foil_id"])
                answer, foil, recovery = suffix.recovery(target["row"], (float(answer_tensor), float(foil_tensor)), native)
                values[panel][arm].append(recovery)
                records.append({"target_panel": target_panel, "source_panel": source_panel, "family": family, "arm_id": arm, "target_row_id": str(target["row"]["row_id"]), "source_row_id": str(source["row"]["row_id"]), "answer_logit": answer, "foil_logit": foil, "recovery": recovery})
    forward_calls += len(ARMS) * capture_chunks
    evaluations += len(ARMS) * 64
    summaries = {panel: {arm: summarize(arm_values) for arm, arm_values in arms.items()} for panel, arms in values.items()}
    ratios = {panel: {arm: summaries[panel][arm]["mean_recovery"] / summaries[panel]["target_full"]["mean_recovery"] for arm in ARMS[1:]} for panel in summaries}
    capability = all(native[(str(row["row_id"]), side)].margin > 0.0 for row in lexical_rows + fresh_rows for side in ("base", "donor"))
    pred_a = capability and manual_base_max_abs <= 1.0e-4 and writer_tensor_error_max_abs <= 2.0e-3 and mlp_tensor_error_max_abs <= 5.0e-3
    pred_b = all(summaries[panel]["target_full"]["mean_recovery"] > 0.0 and summaries[panel]["target_full"]["direction_fraction"] >= 0.75 for panel in summaries)
    pred_c = all(ratios[panel]["source_full"] >= 0.50 and summaries[panel]["source_full"]["direction_fraction"] >= 0.75 for panel in summaries)
    pred_d = all(ratios[panel][arm] >= 0.75 and summaries[panel][arm]["direction_fraction"] >= 0.75 for panel in summaries for arm in ("swap_initial", "swap_attention", "swap_mlp"))
    pred_e = len(records) == 320 and len({(record["target_panel"], record["arm_id"], record["target_row_id"]) for record in records}) == 320 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    predictions = {"pred_a_authority_pairing_capability_and_controls": pred_a, "pred_b_target_program_effect": pred_b, "pred_c_whole_variable_transfer": pred_c, "pred_d_groupwise_operational_equivalence": pred_d, "pred_e_exact_coverage_and_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e else "invalid")
    reason = {"screen": "explicit_v8_variable_groups_are_cross_construction_operationally_equivalent", "null": "whole_or_groupwise_variables_are_construction_specific", "invalid": "authority_pairing_capability_control_target_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_program_v8_cross_construction_variable_interchange_result_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": empirical.component_parent.utc_now(), "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "program_sha256": EXPECTED[PROGRAM], "predictions": predictions,
        "score": {"panels": summaries, "recovery_fraction_vs_target_full": ratios, "manual_base_scored_logit_max_abs": manual_base_max_abs, "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs, "mlp_bilinear_tensor_reconstruction_max_abs": mlp_tensor_error_max_abs, "forward_calls": forward_calls, "example_evaluations": evaluations, "record_count": len(records), "model_backwards": 0, "model_updates": 0, "fit_parameters": 0},
        "intervention_records": records, "terminal": terminal, "reason": reason,
        "next_action": "define cross-construction quotient variables in program v11" if terminal == "screen" else "retain the first failing variable group as construction-specific and test its downstream reader",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "ratios": ratios, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
