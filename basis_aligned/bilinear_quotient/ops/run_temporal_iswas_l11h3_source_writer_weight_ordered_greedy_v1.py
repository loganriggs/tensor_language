#!/usr/bin/env python3
"""Select and validate frozen weight-ordered upstream-writer prefixes."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_reference_replay_capture_coverage_finiteness_and_exact_price pred_b_original_fit_selects_a_qualified_prefix_for_each_role pred_c_selected_prefixes_validate_without_reselection pred_d_greedy_prunes_at_least_one_overcomplete_top5_union pred_e_selected_prefixes_are_task_selective_and_causal
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import dual_command_head_module_factorial_contract as accounting
import run_temporal_iswas_l11h3_source_tensor_upstream_head_factorial_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_l11h3_source_writer_weight_ordered_greedy_v1.json"
FACTORIAL = ROOT / "circuits/followups/temporal_iswas_l11h3_source_tensor_upstream_head_factorial_v1_result.json"
ATLAS = ROOT / "circuits/followups/temporal_iswas_l11h3_source_tensor_upstream_weight_atlas_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_l11h3_source_tensor_upstream_head_factorial_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_l11h3_source_writer_weight_ordered_greedy_v1_result.json"
EXPECTED = {
    "prior": "7281c046d67e696aca745062c829dacd55bc0df70b642b06c736bed0071be9b2",
    "factorial": "d00fdf1f256b94199547d9260b6896c56b0af95229240b204cd0eb2798ded0d1",
    "atlas": "f1b1a057c0ad48f3f514970430551b51d42cfa75a549e6491ea2d724664740d6",
    "parent_runner": "74326a9c7aa1a8d2e7a9bf1b3dcf50c70cbbc67384eb8c5c4d2614a6a46396ca",
}
ORDERS = parent.TOP5
PREFIX_ARMS = tuple(f"P{index}" for index in range(1, 6))
PRICE = {"checkpoint_loads": 1, "model_forwards": 26, "sequence_evaluations": 3328,
         "scored_token_positions": 6656, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_pairing_reference_replay_capture_coverage_finiteness_and_exact_price",
    "pred_b_original_fit_selects_a_qualified_prefix_for_each_role",
    "pred_c_selected_prefixes_validate_without_reselection",
    "pred_d_greedy_prunes_at_least_one_overcomplete_top5_union",
    "pred_e_selected_prefixes_are_task_selective_and_causal",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_qualified(row):
    return bool(.50 <= row["signed_recovery"] <= 1.50 and row["cosine"] >= .90
                and row["relative_residual"] <= .75
                and row["direction_agreement"] >= .90)


def validation_qualified(row, *, pooled):
    if pooled:
        return bool(.50 <= row["signed_recovery"] <= 1.75 and row["cosine"] >= .90
                    and row["relative_residual"] <= .90
                    and row["direction_agreement"] >= .90)
    return bool(row["signed_recovery"] >= 0 and row["direction_agreement"] >= .75)


def run_population(backend, authority, population, old_factorial):
    loc = parent.loc
    rows = authority.build_rows()
    endpoints, lookup = loc.parent.endpoint_bank(rows)
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    torch = backend.torch
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
        native_logits, captures = parent._forward(backend, tokens, capture=True)
    native = {role: loc.margins(native_logits, endpoints, role) for role in parent.LOCKED}
    effects, collateral, references, reference_collateral = {}, {}, {}, {}
    for role, source in parent.LOCKED.items():
        source_rows = [item[source] for item in regions[role]]
        other = "iswas" if role == "temporal" else "temporal"
        with torch.no_grad():
            logits, _ = parent._forward(backend, tokens, donor_v=captures["L11H3:v"],
                                        pairs=pairs[role], position_rows=source_rows)
        references[role] = loc.margins(logits, endpoints, role) - native[role]
        reference_collateral[role] = loc.margins(logits, endpoints, other) - native[other]
        for index, arm in enumerate(PREFIX_ARMS, 1):
            with torch.no_grad():
                logits, _ = parent._forward(backend, tokens, captures=captures,
                    selected_labels=ORDERS[role][:index], pairs=pairs[role],
                    position_rows=source_rows)
            effects[(role, arm)] = loc.margins(logits, endpoints, role) - native[role]
            collateral[(role, arm)] = loc.margins(logits, endpoints, other) - native[other]

    reports, replay_errors = [], []
    old_reports = old_factorial["panels"][population]["reports"]
    replay_fields = ("signed_recovery", "cosine", "relative_residual",
                     "direction_agreement", "non_target_to_target_gold_norm")
    for phase in ("FIT", "HOLDOUT"):
        for template in ("ALL",) + authority.TEMPLATES:
            selected = np.asarray([row["phase"] == phase and
                (template == "ALL" or row["template_id"] == template)
                for row, _cell, _endpoint in endpoints])
            for role in parent.LOCKED:
                gold = native[role][pairs[role]] - native[role]
                reference = accounting.effect_metrics(references[role][selected], gold[selected],
                                                       reference_collateral[role][selected])
                reports.append({"population": population, "role": role, "phase": phase,
                    "template_id": template, "arm": "L11H3:v_reference", **reference})
                old = next(row for row in old_reports if row["role"] == role
                    and row["phase"] == phase and row["template_id"] == template
                    and row["arm"] == "L11H3:v_reference")
                replay_errors.extend(abs(reference[field] - old[field]) for field in replay_fields)
                for arm in PREFIX_ARMS:
                    report = accounting.effect_metrics(effects[(role, arm)][selected],
                                                        references[role][selected],
                                                        collateral[(role, arm)][selected])
                    gold_report = accounting.effect_metrics(effects[(role, arm)][selected],
                        gold[selected], collateral[(role, arm)][selected])
                    reports.append({"population": population, "role": role, "phase": phase,
                        "template_id": template, "arm": arm, **report,
                        "command_gold_non_target_norm_ratio":
                            gold_report["non_target_to_target_gold_norm"]})
    causal_zero = max(abs(value) for arm in PREFIX_ARMS
                      for value in collateral[("iswas", arm)])
    return {"reports": reports, "reference_replay_max_abs_error": max(replay_errors),
            "causal_zero": float(causal_zero),
            "capture_shapes": {name: list(value.shape) for name, value in captures.items()},
            "coverage_ok": all(set(item[parent.LOCKED[role]]) <= set(item["full_prefix"])
                               for role in parent.LOCKED for item in regions[role])}


def main():
    paths = {"prior": PRIOR, "factorial": FACTORIAL, "atlas": ATLAS,
             "parent_runner": PARENT_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    prior, factorial, atlas = (json.loads(path.read_text())
                               for path in (PRIOR, FACTORIAL, ATLAS))
    authority_ok = bool(observed == EXPECTED
        and factorial.get("terminal") == "shared_upstream_head_writer"
        and all(factorial.get("predictions", {}).values())
        and atlas.get("frozen_top5") == {
            role: [f"{label}:attn_out" for label in labels] for role, labels in ORDERS.items()})
    dry = {"candidate_id": prior["candidate_id"], "dryrun": True,
           "authority_ok": authority_ok, "gpu_accessed": False, "model_loaded": False,
           "queue_touched": False, "orders": ORDERS, "arms": PREFIX_ARMS, "price": PRICE}
    if not authority_ok:
        raise RuntimeError(f"greedy-prefix authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = parent.loc.producer.Bilin18TorchBackend.load("cuda")
    panels = {
        "original": run_population(backend, parent.loc.original, "original", factorial),
        "ood": run_population(backend, parent.loc.ood, "ood", factorial),
    }
    reports = [row for panel in panels.values() for row in panel["reports"]]

    def report(population, role, phase, template, arm):
        return next(row for row in reports if row["population"] == population
            and row["role"] == role and row["phase"] == phase
            and row["template_id"] == template and row["arm"] == arm)

    selected = {}
    for role in parent.LOCKED:
        eligible = [arm for arm in PREFIX_ARMS
                    if selection_qualified(report("original", role, "FIT", "ALL", arm))]
        selected[role] = eligible[0] if eligible else None
    A = bool(authority_ok and max(panel["reference_replay_max_abs_error"]
        for panel in panels.values()) <= 1e-5 and all(panel["coverage_ok"] for panel in panels.values())
        and all(set(panel["capture_shapes"]) == {*parent.LABELS, "L11H3:v"}
                for panel in panels.values()) and parent.finite(panels)
        and PRICE == prior["frozen_design"]["price"])
    B = bool(all(selected.values()))
    validation_cells = [("original", "HOLDOUT"), ("ood", "FIT"), ("ood", "HOLDOUT")]
    C = bool(B and all(validation_qualified(
        report(population, role, phase, "ALL", selected[role]), pooled=True)
        for role in parent.LOCKED for population, phase in validation_cells)
        and all(validation_qualified(
            report(population, role, phase, template, selected[role]), pooled=False)
            for role in parent.LOCKED for population, phase in validation_cells
            for template in (parent.loc.original.TEMPLATES if population == "original"
                             else parent.loc.ood.TEMPLATES)))
    D = bool(C and any(int(arm[1:]) < 5 for arm in selected.values()))
    E = bool(B and all(panel["causal_zero"] <= 1e-5 for panel in panels.values())
        and all(report(population, role, phase, "ALL", selected[role])[
            "command_gold_non_target_norm_ratio"] <= .01
            for population in panels for role in parent.LOCKED for phase in ("FIT", "HOLDOUT")))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    if not A:
        terminal = "invalid"
    elif not B:
        terminal = "greedy_selection_null"
    elif not C:
        terminal = "greedy_prefix_validation_failure"
    elif D and E:
        terminal = "compact_source_writer_prefixes"
    else:
        terminal = "validated_unpruned_source_writer_prefixes"
    result = {
        "schema": "temporal_iswas_l11h3_source_writer_weight_ordered_greedy_result_v1",
        "candidate_id": prior["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "orders": {role: list(labels) for role, labels in ORDERS.items()},
        "selected_prefixes": {role: None if arm is None else {
            "arm": arm, "length": int(arm[1:]), "heads": list(ORDERS[role][:int(arm[1:])])}
            for role, arm in selected.items()},
        "panels": panels, "predictions": predictions, "terminal": terminal, "price": PRICE,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"selected_prefixes": result["selected_prefixes"],
        "selected_pooled": {role: {f"{population}_{phase}": report(
            population, role, phase, "ALL", selected[role])
            for population, phase in (("original", "FIT"),) + tuple(validation_cells)}
            for role in parent.LOCKED if selected[role]},
        "reference_replay_max_abs_error": max(panel["reference_replay_max_abs_error"]
            for panel in panels.values()), "predictions": predictions,
        "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
