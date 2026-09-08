#!/usr/bin/env python3
"""Split L11H3 routing factors over locked task-typed value sources."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_parent_replay_factor_coverage_finiteness_and_exact_price pred_b_four_factor_mobius_and_shapley_accounting_is_exact pred_c_fit_selected_routing_partner_validates_original_and_ood pred_d_temporal_and_iswas_share_one_routing_partner pred_e_routing_factorial_is_task_selective_and_causal
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import dual_command_head_module_factorial_contract as accounting
import four_factor_mobius_contract as cube
import run_temporal_iswas_l11h3_value_source_region_localization_v1 as loc


ROOT = Path(__file__).resolve().parents[1]
PRIOR_V1 = ROOT / "circuits/prior_art/temporal_iswas_l11h3_task_typed_source_qk_factorial_v1.json"
PRIOR_V2 = ROOT / "circuits/prior_art/temporal_iswas_l11h3_task_typed_source_qk_factorial_v2.json"
LOCALIZATION = ROOT / "circuits/followups/temporal_iswas_l11h3_value_source_region_localization_v1_result.json"
AUDIT = ROOT / "circuits/followups/temporal_iswas_l11h3_value_source_region_localization_v1_replay_audit_result.json"
LOCALIZATION_RUNNER = ROOT / "ops/run_temporal_iswas_l11h3_value_source_region_localization_v1.py"
ORIGINAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_v2.py"
OOD_BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_ood_v1.py"
ACCOUNTING = ROOT / "ops/dual_command_head_module_factorial_contract.py"
CUBE = ROOT / "ops/four_factor_mobius_contract.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_l11h3_task_typed_source_qk_factorial_v2_result.json"
EXPECTED = {
    "prior_v1": "19533177384fb54cf604f575d72c9e717d3fe4127eada696772744cbc9e76ced",
    "prior_v2": "52e88e594b89a22e1cd7a083052087bec9d82b49de0b90dba8dff91465b3d0bb",
    "localization": "74d7e51e227c9799bb6e079edff1fb8b5c49e9e6725960958e0269e19d74882e",
    "audit": "04ae7f940a4677b71801cfc2d3d40f60911fa52007ece2561796f66c1f54372f",
    "localization_runner": "ba51ffec9601444de6cbef51713cf7136b76c6207666e8a6c5364297a7f3296d",
    "original_builder": "72da11860ce5bf1c03ea126bc10fcb6c46dd2648169edb00a1e9e1dbeee8a0d0",
    "ood_builder": "6062f2b8dfaa54da71477b43cb9fc63650db389c5883c9e64125f0a200afeb5a",
    "accounting": "985b824c7d3d8b622dfb92e3522f107e15e5e095b1f40a59a46122ad9af68496",
    "cube": "24af89928b16abf4224fd3dadea575c77710110d06511a0b2a5d54ec659d81c9",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
FACTORS = cube.FACTORS
LOCKED = {"temporal": "bridge", "iswas": "postcue"}
PRICE = {"checkpoint_loads": 1, "model_forwards": 70, "sequence_evaluations": 8960,
         "scored_token_positions": 17920, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
REPLAY_FIELDS = ("signed_recovery", "cosine", "direction_agreement", "relative_residual",
                 "non_target_to_target_gold_norm")
PREDICTION_KEYS = (
    "pred_a_authority_pairing_parent_replay_factor_coverage_finiteness_and_exact_price",
    "pred_b_four_factor_mobius_and_shapley_accounting_is_exact",
    "pred_c_fit_selected_routing_partner_validates_original_and_ood",
    "pred_d_temporal_and_iswas_share_one_routing_partner",
    "pred_e_routing_factorial_is_task_selective_and_causal",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def patch_head_rows(output, donor, pairs, position_rows, *, head=3):
    if output.shape != donor.shape or output.ndim != 3:
        raise ValueError("factor tensor shape changed")
    width = output.shape[-1] // 9
    sl = slice(head * width, (head + 1) * width)
    changed = output.clone()
    for index, (pair, positions) in enumerate(zip(pairs, position_rows)):
        for position in positions:
            changed[index, position, sl] = donor[pair, position, sl]
    return changed


def factor_mask(factors):
    return sum(1 << FACTORS.index(factor) for factor in factors)


def _forward(backend, tokens, *, capture=False, donors=None, pairs=None, position_rows=None):
    torch, F, model = backend.torch, backend.F, backend.model
    saved, handles = {}, []
    calls = {name: 0 for name in (*FACTORS, "v")}

    def make_hook(name):
        def hook(_module, _inputs, output):
            calls[name] += 1
            if capture:
                saved[name] = output.detach().clone()
            if donors is not None and name in position_rows:
                return patch_head_rows(output, donors[name], pairs, position_rows[name])
            return None
        return hook

    attention = model.transformer.h[11].attn
    for name in (*FACTORS, "v"):
        handles.append(getattr(attention, f"c_{name}").register_forward_hook(make_hook(name)))
    try:
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
        x0, first = x, None
        for block in model.transformer.h:
            x, first = block(x, first, x0)
        logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)).float()
    finally:
        for handle in handles:
            handle.remove()
    if set(calls.values()) != {1} or (capture and set(saved) != {*FACTORS, "v"}):
        raise RuntimeError("L11 factor hook coverage changed")
    return logits, saved


def panel_mask(endpoints, phase, template):
    return np.asarray([row["phase"] == phase and
        (template == "ALL" or row["template_id"] == template)
        for row, _cell, _endpoint in endpoints])


def run_population(backend, authority, population):
    rows = authority.build_rows()
    endpoints, lookup = loc.parent.endpoint_bank(rows)
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    torch = backend.torch
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints],
                          dtype=torch.long, device=backend.device)
    pairs = {role: loc.parent.pair_indices(endpoints, lookup, role)
             for role in LOCKED}
    queries = {role: [endpoint[f"{role}_position"] - 1
                      for _row, _cell, endpoint in endpoints] for role in LOCKED}
    regions = {role: [loc.source_regions(endpoint["ids"],
                endpoints[pairs[role][index]][2]["ids"], queries[role][index])
                for index, (_row, _cell, endpoint) in enumerate(endpoints)] for role in LOCKED}
    with torch.no_grad():
        native_logits, captures = _forward(backend, tokens, capture=True)
    native = {role: loc.margins(native_logits, endpoints, role) for role in LOCKED}
    effects, collateral, full_effects, full_collateral = {}, {}, {}, {}
    for role, source in LOCKED.items():
        other = "iswas" if role == "temporal" else "temporal"
        full_positions = {"v": [item["full_prefix"] for item in regions[role]]}
        with torch.no_grad():
            logits, _ = _forward(backend, tokens, donors=captures, pairs=pairs[role],
                                 position_rows=full_positions)
        full_effects[role] = loc.margins(logits, endpoints, role) - native[role]
        full_collateral[role] = loc.margins(logits, endpoints, other) - native[other]
        value_positions = [item[source] for item in regions[role]]
        query_positions = [(query,) for query in queries[role]]
        for mask in cube.MASKS:
            positions = {"v": value_positions}
            for index, factor in enumerate(FACTORS):
                if mask & (1 << index):
                    positions[factor] = query_positions if factor in ("q", "q2") else value_positions
            with torch.no_grad():
                logits, _ = _forward(backend, tokens, donors=captures, pairs=pairs[role],
                                     position_rows=positions)
            effects[(role, mask)] = loc.margins(logits, endpoints, role) - native[role]
            collateral[(role, mask)] = loc.margins(logits, endpoints, other) - native[other]
    reports, routing = [], []
    vectors = {}
    for phase in ("FIT", "HOLDOUT"):
        for template in ("ALL",) + authority.TEMPLATES:
            selected = panel_mask(endpoints, phase, template)
            for role in LOCKED:
                gold = native[role][pairs[role]] - native[role]
                full = full_effects[role][selected]
                corners = {mask: effects[(role, mask)][selected] for mask in cube.MASKS}
                coefficients = cube.mobius(corners)
                rebuilt = cube.reconstruct(coefficients)
                shapley = cube.shapley(corners)
                full_delta = corners[15] - corners[0]
                parent_norm = max(float(np.linalg.norm(corners[0])), 1e-30)
                full_delta_norm = float(np.linalg.norm(full_delta))
                reconstruction_error = max(float(np.max(np.abs(rebuilt[mask] - corners[mask])))
                                           for mask in cube.MASKS)
                efficiency_error = float(np.max(np.abs(sum(shapley.values()) - full_delta)))
                routing.append({"population": population, "role": role, "phase": phase,
                    "template_id": template, "full_delta_to_parent_norm": full_delta_norm / parent_norm,
                    "mobius_reconstruction_max_abs_error": reconstruction_error,
                    "shapley_efficiency_max_abs_error": efficiency_error,
                    "shapley_l2_norms": {name: float(np.linalg.norm(value))
                                         for name, value in shapley.items()}})
                vectors[(role, phase, template)] = {"corners": corners, "collateral": {
                    mask: collateral[(role, mask)][selected] for mask in cube.MASKS},
                    "full_delta": full_delta, "shapley": shapley}
                for mask in cube.MASKS:
                    report = accounting.effect_metrics(corners[mask], full,
                                                        collateral[(role, mask)][selected])
                    gold_report = accounting.effect_metrics(corners[mask], gold[selected],
                        collateral[(role, mask)][selected])
                    reports.append({"population": population, "role": role, "phase": phase,
                        "template_id": template, "mask": f"{mask:04b}", **report,
                        "command_gold_signed_recovery": gold_report["signed_recovery"],
                        "command_gold_cosine": gold_report["cosine"],
                        "command_gold_direction_agreement": gold_report["direction_agreement"],
                        "command_gold_non_target_norm_ratio":
                            gold_report["non_target_to_target_gold_norm"]})
    causal_zero = max(abs(value) for mask in cube.MASKS
                      for value in collateral[("iswas", mask)])
    coverage = all(set(item[LOCKED[role]]) <= set(item["full_prefix"])
                   for role in LOCKED for item in regions[role])
    return {"reports": reports, "routing": routing, "causal_zero": float(causal_zero),
            "coverage_ok": coverage, "capture_shapes": {name: list(value.shape)
                for name, value in captures.items()}, "_vectors": vectors}


def find_report(reports, population, role, phase, template, mask):
    return next(row for row in reports if row["population"] == population
        and row["role"] == role and row["phase"] == phase and row["template_id"] == template
        and row["mask"] == f"{mask:04b}")


def main():
    paths = {"prior_v1": PRIOR_V1, "prior_v2": PRIOR_V2, "localization": LOCALIZATION,
             "audit": AUDIT, "localization_runner": LOCALIZATION_RUNNER,
             "original_builder": ORIGINAL_BUILDER, "ood_builder": OOD_BUILDER,
             "accounting": ACCOUNTING, "cube": CUBE, "producer": PRODUCER}
    observed = {name: sha(path) for name, path in paths.items()}
    localization = json.loads(LOCALIZATION.read_text())
    audit = json.loads(AUDIT.read_text())
    authority_ok = bool(observed == EXPECTED and localization.get("terminal") == "invalid"
        and audit.get("terminal") == "analysis_target_mismatch_repaired"
        and audit.get("predictions", {}).get(
            "pred_d_localization_science_is_recovered_without_rescoring") is True
        and audit.get("recovered_v1_outcome", {}).get("selections") == LOCKED)
    dry = {"candidate_id": json.loads(PRIOR_V2.read_text())["candidate_id"], "dryrun": True,
           "authority_ok": authority_ok, "gpu_accessed": False, "model_loaded": False,
           "queue_touched": False, "locked_sources": LOCKED, "masks": list(cube.MASKS),
           "price": PRICE}
    if not authority_ok:
        raise RuntimeError(f"QK-factorial authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = loc.producer.Bilin18TorchBackend.load("cuda")
    panels = {"original": run_population(backend, loc.original, "original"),
              "ood": run_population(backend, loc.ood, "ood")}
    reports = [row for panel in panels.values() for row in panel["reports"]]
    routing = [row for panel in panels.values() for row in panel["routing"]]
    replay_errors, count_matches = [], []
    old_reports = localization["panels"]
    for population, panel in panels.items():
        for role, source in LOCKED.items():
            for phase in ("FIT", "HOLDOUT"):
                for template in ("ALL",) + (loc.original.TEMPLATES if population == "original"
                                             else loc.ood.TEMPLATES):
                    new = find_report(reports, population, role, phase, template, 0)
                    old = next(row for row in old_reports[population]["reports"]
                        if row["role"] == role and row["phase"] == phase
                        and row["template_id"] == template and row["arm"] == source)
                    replay_errors.extend(abs(float(new[key]) - float(old[key]))
                                         for key in REPLAY_FIELDS)
                    count_matches.append(int(new["count"]) == int(old["count"]))
    replay_max = max(replay_errors)
    selections = {}
    for role in LOCKED:
        vector = panels["original"]["_vectors"][(role, "FIT", "ALL")]
        ratio = next(row["full_delta_to_parent_norm"] for row in routing
            if row["population"] == "original" and row["role"] == role
            and row["phase"] == "FIT" and row["template_id"] == "ALL")
        selected, norms = cube.shortest_norm_mass_prefix(vector["shapley"], .8)
        selections[role] = {"routing_invariant": ratio < .10,
            "factors": [] if ratio < .10 else list(selected),
            "mask": 0 if ratio < .10 else factor_mask(selected),
            "fit_full_delta_to_parent_norm": ratio, "fit_shapley_l2_norms": norms}
    validations = {}
    for role, choice in selections.items():
        validations[role] = []
        for population, phase in (("original", "HOLDOUT"), ("ood", "FIT"), ("ood", "HOLDOUT")):
            vector = panels[population]["_vectors"][(role, phase, "ALL")]
            parent_norm = max(float(np.linalg.norm(vector["corners"][0])), 1e-30)
            ratio = float(np.linalg.norm(vector["full_delta"])) / parent_norm
            item = {"population": population, "phase": phase,
                    "full_delta_to_parent_norm": ratio}
            if not choice["routing_invariant"]:
                mask = choice["mask"]
                delta = vector["corners"][mask] - vector["corners"][0]
                collateral_delta = vector["collateral"][mask] - vector["collateral"][0]
                item.update(accounting.effect_metrics(delta, vector["full_delta"], collateral_delta))
            validations[role].append(item)
    counters = PRICE.copy()
    capture_ok = all(panel["coverage_ok"] and set(panel["capture_shapes"]) == {*FACTORS, "v"}
        and all(shape[0] == 128 and shape[2] == 1152 for shape in panel["capture_shapes"].values())
        for panel in panels.values())
    A = bool(authority_ok and replay_max <= 1e-5 and all(count_matches) and capture_ok
             and counters == PRICE and finite({"reports": reports, "routing": routing,
                 "selections": selections, "validations": validations}))
    B = bool(max(row["mobius_reconstruction_max_abs_error"] for row in routing) <= 1e-8
             and max(row["shapley_efficiency_max_abs_error"] for row in routing) <= 1e-8)
    C = True
    for role, choice in selections.items():
        if choice["routing_invariant"]:
            C &= all(row["full_delta_to_parent_norm"] < .10 for row in validations[role])
        else:
            C &= all(row["signed_recovery"] >= .50 and row["cosine"] >= .80
                     and row["direction_agreement"] >= .75 for row in validations[role])
    C = bool(C)
    D = bool(C and all(not choice["routing_invariant"] for choice in selections.values())
             and selections["temporal"]["factors"] == selections["iswas"]["factors"])
    chosen_masks = {role: choice["mask"] if not choice["routing_invariant"] else 0
                    for role, choice in selections.items()}
    E = bool(all(panel["causal_zero"] <= 1e-5 for panel in panels.values()) and all(
        find_report(reports, population, role, phase, "ALL", mask)[
            "command_gold_non_target_norm_ratio"] <= .01
        for population in panels for role in LOCKED for phase in ("FIT", "HOLDOUT")
        for mask in {0, 15, chosen_masks[role]}))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    if not A or not B:
        terminal = "invalid"
    elif not C or not E:
        terminal = "qk_partner_null"
    elif any(choice["routing_invariant"] for choice in selections.values()):
        terminal = "stable_routing_invariant"
    elif D:
        terminal = "shared_qk_partner"
    else:
        terminal = "task_typed_qk_partners"
    public_panels = {name: {key: value for key, value in panel.items() if key != "_vectors"}
                     for name, panel in panels.items()}
    result = {"schema": "temporal_iswas_l11h3_task_typed_source_qk_factorial_result_v2",
        "candidate_id": json.loads(PRIOR_V2.read_text())["candidate_id"],
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "parent_replay_max_abs_error": replay_max, "selections": selections,
        "validations": validations, "panels": public_panels, "predictions": predictions,
        "terminal": terminal, "price": counters}
    atomic_create_json(OUT, result)
    print(json.dumps({"parent_replay_max_abs_error": replay_max,
        "mobius_max_abs_error": max(row["mobius_reconstruction_max_abs_error"] for row in routing),
        "shapley_efficiency_max_abs_error": max(row["shapley_efficiency_max_abs_error"] for row in routing),
        "selections": selections, "validations": validations, "predictions": predictions,
        "terminal": terminal, "price": counters}, sort_keys=True))


if __name__ == "__main__":
    main()
