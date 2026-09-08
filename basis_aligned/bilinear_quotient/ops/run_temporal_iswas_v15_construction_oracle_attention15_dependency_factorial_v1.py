#!/usr/bin/env python3
"""No-refit construction-oracle by complete-attention-15 dependency factorial."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_parent_replay_zero_arm_factorial_closure_finiteness_and_price pred_b_each_oracle_transfers_own_v15_target_with_live_attention15 pred_c_live_attention_retains_most_fixed_background_effect pred_d_attention15_dependency_interaction_is_small pred_e_live_attention_does_not_worsen_controls pred_f_v16_new_intervention_retains_fixed_effect
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from aligned_full_sequence_patch_contract import derive_full_sequence_alignment_contract
import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16 as v16
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1 as factor_parent
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent
import two_by_two_dependency_contract as dependency


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
WEIGHTS = ROOT / "circuits/followups/temporal_iswas_construction_oracle_weight_convergence_v1_result.json"
DEPENDENCY = ROOT / "ops/two_by_two_dependency_contract.py"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py"
V15_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py"
V16_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16.py"
V16_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v2_audit_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_construction_oracle_attention15_dependency_factorial_v1"
EXPECTED = {
    "prior": "eadf0e364af600eea215df5f3cc080e8773fba83f9f9c6950d3112878d2f0e68",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "weights": "358d3ae1100a639bb930f4a6fa5f05a4bce631659b49b6e91b37fb7fbffe41a7",
    "dependency": "bf6cf52bfa55d4d071c9e10f3efe7ab44df795fd17fa0cdc0a286fa1d75bd73a",
    "parent_runner": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "v15_builder": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
    "v16_builder": "5b1cb38cc62b5682c03505a5090e962d4773015efb44f8fad145515f5066b549",
    "v16_capability": "a1c2baf0bd9548e189ccc4ba4d11c4905f85af1ff7013d408c143f6f2a6434e3",
}
V15_ROWS_SHA256 = "3f1d28abb658040493284b307cc27ba76f422dddb08ee9c53686c557d49f283c"
V16_ROWS_SHA256 = "4c5dfaee126c04ac7ea6ef5f53d6ad62a24806fa7133a0c773471a90e4d2e468"
EXPERTS = {"A1_oracle": "A1", "A2_oracle": "A2"}
PRICE_MAX = {"native_capture_forwards": 20, "differentiable_transformer_forwards": 64,
             "transformer_backward_forwards": 0, "model_updates": 0,
             "example_evaluations": 5000, "fit_parameters": 0}


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def bases_for_evaluation(torch, device, artifact, expert):
    panel = EXPERTS[expert]
    return {evaluation: {
        site: torch.tensor(values, device=device)
        for site, values in artifact["projectors"][panel][str(1 - evaluation)].items()
    } for evaluation in (0, 1)}


def execute_arm(backend, contexts, counters, bases_by_evaluation, complete15):
    outputs = {}
    for parity, context in sorted(contexts.items()):
        outputs[parity] = parent.manual_forward(
            backend, context["base_batch"], counters, context=context,
            raw_by_site=bases_by_evaluation.get(parity, {}), complete15=complete15)
    return outputs


def arm_report(backend, contexts, outputs):
    return {
        "by_evaluation_parity": {
            str(parity): parent.output_metrics(backend, contexts[parity], output)
            for parity, output in outputs.items()
        },
        "combined": parent.combine_reports(
            backend, [(contexts[parity], output) for parity, output in sorted(outputs.items())]),
    }


def margin_response(context, output, panel):
    logits = output["logits"]
    margin = logits[context["index"], context["answer"]] - logits[context["index"], context["foil"]]
    selected = context["panel_indices"][panel]
    return (margin[selected] - context["base_margin"][selected]).detach().cpu().tolist()


def target_vector(context, panel):
    selected = context["panel_indices"][panel]
    return context["target_margin"][selected].detach().cpu().tolist()


def replay(observed, expected):
    return factor_parent.replay_comparison(
        parent.registered_report(observed), parent.registered_report(expected))


def replay_pass(value):
    return bool(value["numeric_schema_match"] and value["categorical_match"]
                and value["numeric_max_abs_error"] <= 1e-4)


def build_factorials(contexts, shared, expert_outputs):
    result = {}
    for expert in EXPERTS:
        result[expert] = {}
        for parity, context in sorted(contexts.items()):
            result[expert][str(parity)] = {}
            panels = [panel for panel in ("A1", "A2", "P", "C")
                      if len(context["panel_indices"][panel])]
            for panel in panels:
                cells = {
                    "00": margin_response(context, shared["00"][parity], panel),
                    "01": margin_response(context, shared["01"][parity], panel),
                    "10": margin_response(context, expert_outputs[expert]["10"][parity], panel),
                    "11": margin_response(context, expert_outputs[expert]["11"][parity], panel),
                }
                factorial = dependency.decompose_dependency_factorial(
                    cells, base_zero_tolerance=1e-4, closure_tolerance=1e-12)
                factorial["upstream_live_target_metrics"] = dependency.vector_metrics(
                    factorial["components"]["upstream_live_attention"], target_vector(context, panel))
                factorial["arm11_target_metrics"] = dependency.vector_metrics(
                    cells["11"], target_vector(context, panel))
                factorial["arm11_mean_absolute_response"] = statistics.fmean(abs(value) for value in cells["11"])
                result[expert][str(parity)][panel] = factorial
    return result


def own_factor(factorials, expert, parity):
    return factorials[expert][str(parity)][EXPERTS[expert]]


def control_metric(reports, expert, cell, parity, panel, key):
    return reports[expert][cell]["by_evaluation_parity"][str(parity)]["controls"][panel][key]


def main():
    paths = {"prior": PRIOR, "oracles": ORACLES, "weights": WEIGHTS,
             "dependency": DEPENDENCY, "parent_runner": PARENT_RUNNER,
             "v15_builder": V15_BUILDER, "v16_builder": V16_BUILDER,
             "v16_capability": V16_CAPABILITY}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise ExperimentError(f"missing dependencies: {missing}")
    observed = {name: sha(path) for name, path in paths.items()}
    prior, oracles, weights = (json.loads(path.read_text()) for path in (PRIOR, ORACLES, WEIGHTS))
    rows15, rows16_all = v15.build_rows(), v16.build_rows()
    rows16 = [row for row in rows16_all if row["transform_id"] in ("A1", "A2", "P")]
    alignment15 = derive_full_sequence_alignment_contract(
        rows15, required_panels=("A1", "A2", "P", "C"))
    alignment16 = derive_full_sequence_alignment_contract(
        rows16, required_panels=("A1", "A2", "P"))
    authority_ok = bool(
        observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
        and v15.validate_rows(rows15) == V15_ROWS_SHA256
        and v16.validate_rows(rows16_all) == V16_ROWS_SHA256
        and alignment15["panel_counts"] == {panel: 16 for panel in ("A1", "A2", "P", "C")}
        and alignment16["panel_counts"] == {panel: 16 for panel in ("A1", "A2", "P")}
        and oracles.get("terminal") == "construction_conditioned_coordinate"
        and weights.get("terminal") == "reader_equivalent_distinct_writes")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "experts": list(EXPERTS),
              "cells": list(dependency.CELLS), "v15_rows": len(rows15),
              "v16_rows": len(rows16), "price_max": PRICE_MAX,
              "v16_scope": "OOD_TEXT_REUSE_NEW_INTERVENTION", "v16_C_excluded": True}
    if not authority_ok:
        raise ExperimentError(f"authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    counters = {key: 0 for key in PRICE_MAX}
    native = backend.native

    def counted_native(batch, *, capture):
        counters["native_capture_forwards"] += 1
        counters["example_evaluations"] += len(batch.row_ids)
        return native(batch, capture=capture)

    backend.native = counted_native
    version_rows = {"v15": rows15, "v16": rows16}
    version_reports, version_factorials, instrument = {}, {}, {
        "manual_native_max_abs_error": 0.0, "base_zero_max_abs_error": 0.0,
        "factorial_closure_max_abs_error": 0.0, "parent_replays": {},
    }
    for version, rows in version_rows.items():
        bank = parent.capture_bank(backend, rows, counters, factors=(version == "v15"))
        contexts = {parity: parent.attach_references(
            backend, parent.subset_context(bank, parent.row_indices(rows, parity=parity)), counters)
            for parity in (0, 1)}
        instrument["manual_native_max_abs_error"] = max(
            instrument["manual_native_max_abs_error"],
            *(context["manual_native_max_abs_error"] for context in contexts.values()))
        shared = {
            "00": execute_arm(backend, contexts, counters, {}, False),
            "01": execute_arm(backend, contexts, counters, {}, True),
        }
        expert_outputs, reports = {}, {}
        for expert in EXPERTS:
            bases = bases_for_evaluation(backend.torch, backend.device, oracles, expert)
            expert_outputs[expert] = {
                "10": execute_arm(backend, contexts, counters, bases, False),
                "11": execute_arm(backend, contexts, counters, bases, True),
            }
            reports[expert] = {
                "00": arm_report(backend, contexts, shared["00"]),
                "01": arm_report(backend, contexts, shared["01"]),
                "10": arm_report(backend, contexts, expert_outputs[expert]["10"]),
                "11": arm_report(backend, contexts, expert_outputs[expert]["11"]),
            }
            expected = oracles["reports"][version][expert]["combined"]
            instrument["parent_replays"][f"{version}:{expert}"] = replay(
                reports[expert]["11"]["combined"], expected)
        factorials = build_factorials(contexts, shared, expert_outputs)
        version_reports[version], version_factorials[version] = reports, factorials
        instrument["base_zero_max_abs_error"] = max(
            instrument["base_zero_max_abs_error"],
            *(item["base_zero_max_abs_error"] for expert in factorials.values()
              for parity in expert.values() for item in parity.values()))
        instrument["factorial_closure_max_abs_error"] = max(
            instrument["factorial_closure_max_abs_error"],
            *(item["factorial_closure_max_abs_error"] for expert in factorials.values()
              for parity in expert.values() for item in parity.values()))
        instrument[f"{version}_attention_reconstruction_max_abs_error"] = bank[
            "attention_reconstruction_max_abs_error"]
        instrument[f"{version}_factor_closure_max_abs_error"] = bank[
            "factor_closure_max_abs_error"]

    counters["fit_parameters"] = 0
    price_ok = all(counters[key] <= PRICE_MAX[key] for key in PRICE_MAX)
    pred_a = bool(
        authority_ok and instrument["manual_native_max_abs_error"] <= 1e-4
        and instrument["base_zero_max_abs_error"] <= 1e-4
        and instrument["factorial_closure_max_abs_error"] <= 1e-12
        and all(replay_pass(value) for value in instrument["parent_replays"].values())
        and instrument["v15_attention_reconstruction_max_abs_error"] <= 1e-4
        and instrument["v15_factor_closure_max_abs_error"] <= 1e-4
        and finite({"reports": version_reports, "factorials": version_factorials,
                    "instrument": instrument}) and price_ok)
    pred_b = all(
        own_factor(version_factorials["v15"], expert, parity)["upstream_live_target_metrics"]["signed_projection"] >= .65
        and own_factor(version_factorials["v15"], expert, parity)["upstream_live_target_metrics"]["direction_fraction"] >= .875
        for expert in EXPERTS for parity in (0, 1))
    pred_c = all(
        own_factor(version_factorials["v15"], expert, parity)["upstream_live_target_metrics"]["signed_projection"]
        >= .75 * own_factor(version_factorials["v15"], expert, parity)["arm11_target_metrics"]["signed_projection"]
        for expert in EXPERTS for parity in (0, 1))
    pred_d = all(
        own_factor(version_factorials["v15"], expert, parity)["summaries"]["mobius_interaction"]["mean_absolute"]
        <= .25 * own_factor(version_factorials["v15"], expert, parity)["arm11_mean_absolute_response"]
        for expert in EXPERTS for parity in (0, 1))
    pred_e = all(
        control_metric(version_reports["v15"], expert, "10", parity, panel, "top1_flip_count")
        <= control_metric(version_reports["v15"], expert, "11", parity, panel, "top1_flip_count")
        and control_metric(version_reports["v15"], expert, "10", parity, panel, "mean_kl")
        <= control_metric(version_reports["v15"], expert, "11", parity, panel, "mean_kl")
        for expert in EXPERTS for parity in (0, 1) for panel in ("P", "C"))
    pred_f = all(
        own_factor(version_factorials["v16"], expert, parity)["upstream_live_target_metrics"]["signed_projection"]
        >= .75 * own_factor(version_factorials["v16"], expert, parity)["arm11_target_metrics"]["signed_projection"]
        and own_factor(version_factorials["v16"], expert, parity)["upstream_live_target_metrics"]["direction_fraction"] >= .875
        and control_metric(version_reports["v16"], expert, "10", parity, "P", "top1_flip_count")
        <= control_metric(version_reports["v16"], expert, "11", parity, "P", "top1_flip_count")
        for expert in EXPERTS for parity in (0, 1))
    predictions = {
        "pred_a_authority_alignment_parent_replay_zero_arm_factorial_closure_finiteness_and_price": pred_a,
        "pred_b_each_oracle_transfers_own_v15_target_with_live_attention15": pred_b,
        "pred_c_live_attention_retains_most_fixed_background_effect": pred_c,
        "pred_d_attention15_dependency_interaction_is_small": pred_d,
        "pred_e_live_attention_does_not_worsen_controls": pred_e,
        "pred_f_v16_new_intervention_retains_fixed_effect": pred_f,
    }
    if not pred_a:
        terminal = "invalid"
    elif not pred_b or not pred_c:
        terminal = "fixed_attention15_background_dependent"
    elif not pred_e:
        terminal = "live_attention_route_nonselective"
    else:
        terminal = "live_attention_route_candidate"
    result = {
        "schema": "temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "v15_rows_sha256": V15_ROWS_SHA256,
        "v16_rows_sha256": V16_ROWS_SHA256,
        "alignment_contracts": {"v15": alignment15, "v16_without_c": alignment16},
        "dryrun": dryrun, "reports": version_reports, "factorials": version_factorials,
        "instrument": instrument, "predictions": predictions, "terminal": terminal,
        "v16_scope": "OOD_TEXT_REUSE_NEW_INTERVENTION",
        "price": {**counters, "maxima": PRICE_MAX},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "instrument": instrument,
        "own_v15": {expert: {str(parity): own_factor(
            version_factorials["v15"], expert, parity) for parity in (0, 1)} for expert in EXPERTS},
        "own_v16": {expert: {str(parity): own_factor(
            version_factorials["v16"], expert, parity) for parity in (0, 1)} for expert in EXPERTS},
        "predictions": predictions, "terminal": terminal, "price": result["price"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
