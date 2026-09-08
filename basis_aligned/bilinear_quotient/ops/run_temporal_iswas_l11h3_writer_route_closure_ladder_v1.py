#!/usr/bin/env python3
"""Distinguish source-value spread, induced L11H3 routing, and writer bypass."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_capture_source_replay_finiteness_and_exact_price pred_b_source_local_mediator_reproduces_the_registered_asymmetric_result pred_c_full_prefix_value_closes_the_selected_iswas_writer_effect pred_d_full_observed_l11h3_head_closes_the_selected_iswas_writer_effect pred_e_temporal_positive_control_and_task_selectivity_hold
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_l11h3_source_writer_formula_mediation_v2 as mediation


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_l11h3_writer_route_closure_ladder_v1.json"
MEDIATION_RESULT = ROOT / "circuits/followups/temporal_iswas_l11h3_source_writer_formula_mediation_v2_result.json"
GREEDY_RESULT = ROOT / "circuits/followups/temporal_iswas_l11h3_source_writer_weight_ordered_greedy_v1_result.json"
BINDING = ROOT / "circuits/prior_art/temporal_iswas_l11h3_source_writer_formula_mediation_v3_authority_binding.json"
MEDIATION_RUNNER = ROOT / "ops/run_temporal_iswas_l11h3_source_writer_formula_mediation_v2.py"
OUT = ROOT / "circuits/followups/temporal_iswas_l11h3_writer_route_closure_ladder_v1_result.json"
EXPECTED = {
    "mediation_result": "383c162f2dcc505248122cd0d98d5b2bb7afedc019299f9b4be537cc791a18c4",
    "greedy_result": "6a2ae6598ae95269f7d9e24e13196155342f690d331c78efed5115a8cf84bda2",
    "binding": "c7ad05cb4ba94f4a491cbfaddb1b13936ee8fa7a356d405e2a143b0033fec285",
    "mediation_runner": "88a37dbd000505713c7504bb4c28a262df438349dcbefcbdc5b3ee29ed9aa1f8",
}
PRICE = {"checkpoint_loads": 1, "model_forwards": 18,
         "sequence_evaluations": 2304, "scored_token_positions": 4608,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
ARMS = ("source_value", "full_prefix_value", "full_head")
PREDICTION_KEYS = (
    "pred_a_authority_pairing_capture_source_replay_finiteness_and_exact_price",
    "pred_b_source_local_mediator_reproduces_the_registered_asymmetric_result",
    "pred_c_full_prefix_value_closes_the_selected_iswas_writer_effect",
    "pred_d_full_observed_l11h3_head_closes_the_selected_iswas_writer_effect",
    "pred_e_temporal_positive_control_and_task_selectivity_hold",
)
BARS = {
    "replay": 1e-5, "recovery": .75, "cosine": .95, "residual": .50,
    "direction": .90, "template_recovery": .50, "template_direction": .75,
    "source_iswas_max": .30, "collateral": .01, "causal_zero": 1e-5,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix_rows(queries):
    return [list(range(query + 1)) for query in queries]


def finite_reports(reports):
    return all(np.isfinite(value) for row in reports for value in row.values()
               if isinstance(value, (int, float)) and not isinstance(value, bool))


def closure(reports, role, arm, *, validation_only=True):
    cells = (("original", "HOLDOUT"), ("ood", "FIT"), ("ood", "HOLDOUT"))
    if not validation_only:
        cells = (("original", "FIT"),) + cells
    for population, phase in cells:
        pooled = next(row for row in reports if row["population"] == population
            and row["role"] == role and row["arm"] == arm and row["phase"] == phase
            and row["template_id"] == "ALL")
        if not (pooled["signed_recovery"] >= BARS["recovery"]
                and pooled["cosine"] >= BARS["cosine"]
                and pooled["relative_residual"] <= BARS["residual"]
                and pooled["direction_agreement"] >= BARS["direction"]):
            return False
        templates = (mediation.parent.loc.original.TEMPLATES if population == "original"
                     else mediation.parent.loc.ood.TEMPLATES)
        for template in templates:
            row = next(row for row in reports if row["population"] == population
                and row["role"] == role and row["arm"] == arm and row["phase"] == phase
                and row["template_id"] == template)
            if not (row["signed_recovery"] >= BARS["template_recovery"]
                    and row["direction_agreement"] >= BARS["template_direction"]):
                return False
    return True


def run_population(backend, authority, population, selected_heads, prior_reports):
    loc, parent, source = mediation.parent.loc, mediation.parent, mediation.source
    rows = authority.build_rows()
    endpoints, lookup = loc.parent.endpoint_bank(rows)
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    torch, F = backend.torch, backend.F
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints],
                          dtype=torch.long, device=backend.device)
    pairs = {role: loc.parent.pair_indices(endpoints, lookup, role) for role in parent.LOCKED}
    queries = {role: [endpoint[f"{role}_position"] - 1
                      for _row, _cell, endpoint in endpoints] for role in parent.LOCKED}
    regions = {role: [loc.source_regions(endpoint["ids"],
                endpoints[pairs[role][index]][2]["ids"], queries[role][index])
                for index, (_row, _cell, endpoint) in enumerate(endpoints)]
               for role in parent.LOCKED}
    with torch.no_grad():
        native_logits, head_captures, native_l11 = mediation.forward_with_l11_capture(
            backend, tokens)
    native = {role: loc.margins(native_logits, endpoints, role) for role in parent.LOCKED}
    reports, replay_errors, route_tensors, iswas_other = [], [], {}, []
    for role, source_region in parent.LOCKED.items():
        source_rows = [item[source_region] for item in regions[role]]
        all_prefix_rows = prefix_rows(queries[role])
        other = "iswas" if role == "temporal" else "temporal"
        with torch.no_grad():
            writer_logits, _unused, writer_l11 = mediation.forward_with_l11_capture(
                backend, tokens, head_captures=head_captures,
                selected_labels=selected_heads[role], pairs=pairs[role],
                position_rows=source_rows)
            source_delta = mediation.recipient_native_source_formula(
                native_l11, writer_l11["v"], source_rows,
                backend.model.transformer.h[11].attn, torch, F)
            prefix_delta = mediation.recipient_native_source_formula(
                native_l11, writer_l11["v"], all_prefix_rows,
                backend.model.transformer.h[11].attn, torch, F)
            width = native_l11["head"].shape[-1] // 9
            head_delta = (writer_l11["head"][..., 3 * width:4 * width]
                          - native_l11["head"][..., 3 * width:4 * width])
            deltas = {"source_value": source_delta,
                      "full_prefix_value": prefix_delta, "full_head": head_delta}
            logits = {arm: source._forward(backend, tokens, direct_delta=delta)[0]
                      for arm, delta in deltas.items()}
        route_tensors[role] = {
            "source_delta_norm": float(source_delta.float().norm()),
            "full_prefix_delta_norm": float(prefix_delta.float().norm()),
            "full_head_delta_norm": float(head_delta.float().norm()),
            "source_to_full_prefix_relative_l2": float(
                (source_delta.float() - prefix_delta.float()).norm()
                / max(float(prefix_delta.float().norm()), 1e-30)),
            "full_prefix_to_full_head_relative_l2": float(
                (prefix_delta.float() - head_delta.float()).norm()
                / max(float(head_delta.float().norm()), 1e-30)),
        }
        writer_effect = loc.margins(writer_logits, endpoints, role) - native[role]
        gold = native[role][pairs[role]] - native[role]
        for arm in ARMS:
            effect = loc.margins(logits[arm], endpoints, role) - native[role]
            effect_other = loc.margins(logits[arm], endpoints, other) - native[other]
            if role == "iswas":
                iswas_other.extend(effect_other.tolist())
            for phase in ("FIT", "HOLDOUT"):
                for template in ("ALL",) + authority.TEMPLATES:
                    chosen = np.asarray([row["phase"] == phase and
                        (template == "ALL" or row["template_id"] == template)
                        for row, _cell, _endpoint in endpoints])
                    metric = mediation.accounting.effect_metrics(
                        effect[chosen], writer_effect[chosen], effect_other[chosen])
                    metric_gold = mediation.accounting.effect_metrics(
                        effect[chosen], gold[chosen], effect_other[chosen])
                    report = {"population": population, "role": role, "phase": phase,
                        "template_id": template, "arm": arm,
                        "selected_heads": list(selected_heads[role]), **metric,
                        "command_gold_non_target_norm_ratio":
                            metric_gold["non_target_to_target_gold_norm"]}
                    reports.append(report)
                    if arm == "source_value":
                        old = next(row for row in prior_reports if row["population"] == population
                            and row["role"] == role and row["phase"] == phase
                            and row["template_id"] == template)
                        fields = ("signed_recovery", "cosine", "relative_residual",
                                  "direction_agreement", "non_target_to_target_gold_norm")
                        replay_errors.extend(abs(report[field] - old[field]) for field in fields)
                        replay_errors.append(abs(report["command_gold_non_target_norm_ratio"]
                            - old["formula_command_gold_non_target_norm_ratio"]))
    return {"reports": reports, "route_tensors": route_tensors,
            "source_report_replay_max_abs_error": max(replay_errors),
            "causal_zero": float(max(abs(value) for value in iswas_other)),
            "capture_shapes": {name: list(value.shape) for name, value in native_l11.items()},
            "writer_capture_shapes": {name: list(value.shape) for name, value in head_captures.items()},
            "coverage_ok": all(set(item[mediation.parent.LOCKED[role]]) <= set(item["full_prefix"])
                               for role in mediation.parent.LOCKED for item in regions[role])}


def main():
    paths = {"mediation_result": MEDIATION_RESULT, "greedy_result": GREEDY_RESULT,
             "binding": BINDING, "mediation_runner": MEDIATION_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    prior = json.loads(PRIOR.read_text())
    dry = {"candidate_id": prior["candidate_id"], "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "authority_ok": observed == EXPECTED, "arms": list(ARMS), "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    mediation_result = json.loads(MEDIATION_RESULT.read_text())
    greedy = json.loads(GREEDY_RESULT.read_text())
    selected_heads = {role: tuple(greedy["selected_prefixes"][role]["heads"])
                      for role in mediation.parent.LOCKED}
    authority_ok = bool(observed == EXPECTED and prior["price"] == PRICE
        and prior["locked_selected_prefixes"] == greedy["selected_prefixes"]
        and mediation_result["selected_prefixes"] == greedy["selected_prefixes"]
        and mediation_result["terminal"] == "writer_formula_partial_mediation"
        and mediation_result["predictions"] == {
            mediation.PREDICTION_KEYS[0]: True,
            mediation.PREDICTION_KEYS[1]: True,
            mediation.PREDICTION_KEYS[2]: False,
            mediation.PREDICTION_KEYS[3]: False,
            mediation.PREDICTION_KEYS[4]: True,
        }
        and {name: sha(path) for name, path in {
            "prior_v1": mediation.PRIOR_V1, "prior_v2": mediation.PRIOR_V2,
            "extractor_result": mediation.EXTRACTOR_RESULT,
            "greedy_runner": mediation.GREEDY_RUNNER, "source_runner": mediation.SOURCE_RUNNER,
            "parent_runner": mediation.PARENT_RUNNER,
            "localization_runner": mediation.LOCALIZATION_RUNNER,
            "original_builder": mediation.ORIGINAL_BUILDER, "ood_builder": mediation.OOD_BUILDER,
            "accounting": mediation.ACCOUNTING, "producer": mediation.PRODUCER}.items()}
            == mediation.EXPECTED)
    if not authority_ok:
        raise RuntimeError("route-ladder authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = mediation.parent.loc.producer.Bilin18TorchBackend.load("cuda")
    prior_reports = [row for panel in mediation_result["panels"].values()
                     for row in panel["reports"]]
    panels = {
        "original": run_population(backend, mediation.parent.loc.original, "original",
                                   selected_heads, prior_reports),
        "ood": run_population(backend, mediation.parent.loc.ood, "ood",
                              selected_heads, prior_reports),
    }
    reports = [row for panel in panels.values() for row in panel["reports"]]
    source_iswas = [row for row in reports if row["role"] == "iswas"
        and row["arm"] == "source_value" and row["template_id"] == "ALL"
        and (row["population"], row["phase"]) in {
            ("original", "HOLDOUT"), ("ood", "FIT"), ("ood", "HOLDOUT")}]
    A = bool(authority_ok and all(panel["coverage_ok"] for panel in panels.values())
        and all(set(panel["capture_shapes"]) == {"q", "k", "q2", "k2", "v", "head"}
                for panel in panels.values())
        and all(set(panel["writer_capture_shapes"]) == {*mediation.parent.LABELS, "L11H3:v"}
                for panel in panels.values()) and finite_reports(reports)
        and PRICE == prior["price"])
    B = bool(max(panel["source_report_replay_max_abs_error"] for panel in panels.values())
        <= BARS["replay"]
        and closure(reports, "temporal", "source_value")
        and all(row["signed_recovery"] <= BARS["source_iswas_max"] for row in source_iswas))
    C = closure(reports, "iswas", "full_prefix_value")
    D = closure(reports, "iswas", "full_head")
    E = bool(closure(reports, "temporal", "source_value")
        and closure(reports, "temporal", "full_head")
        and all(row["command_gold_non_target_norm_ratio"] <= BARS["collateral"]
                for row in reports)
        and all(panel["causal_zero"] <= BARS["causal_zero"] for panel in panels.values()))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid" if not (A and B) else
        "unstable_route_ladder" if not E else
        "full_prefix_value_spreading" if C else
        "writer_induced_l11h3_routing_or_head_operation" if D else
        "l11h3_bypass_dominant")
    result = {"schema": "temporal_iswas_l11h3_writer_route_closure_ladder_result_v1",
        "candidate_id": prior["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": {**observed, "prior": sha(PRIOR)},
        "selected_prefixes": greedy["selected_prefixes"], "panels": panels,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    pooled = [row for row in reports if row["template_id"] == "ALL"]
    print(json.dumps({"predictions": predictions, "terminal": terminal,
        "source_report_replay_max_abs_error": max(panel[
            "source_report_replay_max_abs_error"] for panel in panels.values()),
        "route_tensors": {name: panel["route_tensors"] for name, panel in panels.items()},
        "validation_pooled": [row for row in pooled if (row["population"], row["phase"])
            in {("original", "HOLDOUT"), ("ood", "FIT"), ("ood", "HOLDOUT")}],
        "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
