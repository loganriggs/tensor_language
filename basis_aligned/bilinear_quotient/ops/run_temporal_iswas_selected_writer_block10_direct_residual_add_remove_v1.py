#!/usr/bin/env python3
"""Test exact final-residual addition/removal for the compact is-was writer."""

# BQGATE: EXPERIMENT pred_a_authority_condition_pairing_capture_self_patch_finiteness_and_exact_price pred_b_direct_add_matches_dynamic_identity_carriage pred_c_direct_remove_matches_dynamic_module_response pred_d_frozen_identity_route_predicts_ood_writer_behavior pred_e_identity_carriage_is_selectively_manipulable
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from circuit_fast_screen_managed_runner import atomic_create_json
import final_residual_intervention as final_route
import residual_identity_route as residual_route
import run_temporal_iswas_selected_writer_residual_skip_module_response_factorial_v1 as factorial


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve()
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_selected_writer_block10_direct_residual_add_remove_v1_conditional.json"
BINDING = ROOT / "circuits/prior_art/temporal_iswas_selected_writer_block10_direct_residual_add_remove_v1_authority_binding.json"
FACTORIAL_RESULT = ROOT / "circuits/followups/temporal_iswas_selected_writer_residual_skip_module_response_factorial_v1_result.json"
FACTORIAL_RUNNER = ROOT / "ops/run_temporal_iswas_selected_writer_residual_skip_module_response_factorial_v1.py"
GREEDY_RESULT = factorial.GREEDY_RESULT
OUT = ROOT / "circuits/followups/temporal_iswas_selected_writer_block10_direct_residual_add_remove_v1_result.json"
EXPECTED = {
    "prior": "9fb12faf4f89594ac9594f3b8747725b0f88b7e52fd8e4c54ddd0eb5daf5c625",
    "factorial_runner": "9e3c9773bb96330f2b2166f5fe2c1087d4ec10b8199f7f4219c527e48ef89c0e",
    "greedy_result": "6a2ae6598ae95269f7d9e24e13196155342f690d331c78efed5115a8cf84bda2",
    "residual_route": "7565b3b1f6bab1dbe83ece43c97b9c9f8a783b0277416488b22b4b44770a480b",
    "final_route": "7486abbc88548dbe23fbbb27bc8ac2499fa9a37a43f11a290b660e5389bceec6",
}
FILES = {
    "prior": PRIOR,
    "factorial_runner": FACTORIAL_RUNNER,
    "greedy_result": GREEDY_RESULT,
    "residual_route": ROOT / "ops/residual_identity_route.py",
    "final_route": ROOT / "ops/final_residual_intervention.py",
}
ARMS = ("writer", "R1M0_dynamic", "R0M1_dynamic", "native_x18_self_patch",
        "direct_add", "direct_remove")
PRICE = {"checkpoint_loads": 1, "model_forwards": 14,
         "sequence_evaluations": 1792, "scored_token_positions": 3584,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_condition_pairing_capture_self_patch_finiteness_and_exact_price",
    "pred_b_direct_add_matches_dynamic_identity_carriage",
    "pred_c_direct_remove_matches_dynamic_module_response",
    "pred_d_frozen_identity_route_predicts_ood_writer_behavior",
    "pred_e_identity_carriage_is_selectively_manipulable",
)
BARS = {"authority_replay": 1e-5, "self_patch_logit": 1e-5,
        "direct_logit": 1e-5, "state_relative_l2": .05,
        "recovery": .5, "cosine": .9, "direction": .9,
        "removal_fraction": .5, "removal_direction": .9,
        "collateral": .01}
VALIDATION_CELLS = (("original", "HOLDOUT"), ("ood", "FIT"), ("ood", "HOLDOUT"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def maxdiff(left, right):
    return float((left - right).abs().max())


def state_relative_l2(observed, predicted, position_rows):
    observed_rows = [observed[index, list(positions)].reshape(-1)
                     for index, positions in enumerate(position_rows)]
    predicted_rows = [predicted[index, list(positions)].reshape(-1)
                      for index, positions in enumerate(position_rows)]
    if not observed_rows or any(row.numel() == 0 for row in observed_rows):
        raise residual_route.ResidualIdentityRouteError("registered state positions are empty")
    return residual_route.relative_l2_error(
        torch.cat(observed_rows), torch.cat(predicted_rows))


def binding_state(observed):
    if not BINDING.exists() or not FACTORIAL_RESULT.exists():
        return None, False
    binding = json.loads(BINDING.read_text())
    result = json.loads(FACTORIAL_RESULT.read_text())
    runner_sha = sha(RUNNER)
    ok = bool(observed == EXPECTED
        and binding.get("status") == "bound"
        and binding.get("prior_sha256") == observed["prior"]
        and binding.get("runner_sha256") == runner_sha
        and binding.get("factorial_result_sha256") == sha(FACTORIAL_RESULT)
        and binding.get("factorial_predictions") == result.get("predictions")
        and binding.get("factorial_terminal") == result.get("terminal")
        and result.get("predictions", {}).get(factorial.PREDICTION_KEYS[0]) is True
        and result.get("predictions", {}).get(factorial.PREDICTION_KEYS[1]) is True
        and result.get("predictions", {}).get(factorial.PREDICTION_KEYS[2]) is True
        and result.get("terminal") in {"identity_skip_dominant", "redundant_dual_carriers"})
    return {"binding": binding, "factorial": result}, ok


def capture_native_or_writer(backend, tokens, *, head_captures=None,
                             selected_heads=(), pairs=None, source_rows=None):
    call = lambda: factorial.capture_state_and_modules(
        backend, tokens, head_captures=head_captures, selected_labels=selected_heads,
        pairs=pairs, position_rows=source_rows)
    result, final_state, final_calls = final_route.execute(backend.model, call)
    return (*result, final_state, final_calls)


def run_population(backend, authority, population, selected_heads):
    loc, parent = factorial.atlas.mediation.parent.loc, factorial.atlas.mediation.parent
    rows = authority.build_rows()
    endpoints, lookup = loc.parent.endpoint_bank(rows)
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    torch = backend.torch
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints],
                          dtype=torch.long, device=backend.device)
    role, other = "iswas", "temporal"
    pairs = loc.parent.pair_indices(endpoints, lookup, role)
    other_pairs = loc.parent.pair_indices(endpoints, lookup, other)
    queries = [endpoint[f"{role}_position"] - 1 for _row, _cell, endpoint in endpoints]
    regions = [loc.source_regions(endpoint["ids"], endpoints[pairs[index]][2]["ids"],
                                  queries[index])
               for index, (_row, _cell, endpoint) in enumerate(endpoints)]
    source_rows = [item[parent.LOCKED[role]] for item in regions]
    position_rows = [list(range(query + 1)) for query in queries]

    with torch.no_grad():
        native = capture_native_or_writer(backend, tokens)
        (native_logits, head_captures, _native_l11, native_x10, native_modules,
         native_x10_calls, native_module_calls, native_x18, native_x18_calls) = native
        writer = capture_native_or_writer(
            backend, tokens, head_captures=head_captures,
            selected_heads=selected_heads, pairs=pairs, source_rows=source_rows)
        (writer_logits, _unused, _writer_l11, writer_x10, writer_modules,
         writer_x10_calls, writer_module_calls, writer_x18, writer_x18_calls) = writer

        dynamics, dynamic_states, coverage = {}, {}, {}
        for name, entry, modules in (
                ("R1M0_dynamic", writer_x10, native_modules),
                ("R0M1_dynamic", native_x10, writer_modules)):
            result, dynamic_states[name], final_calls = final_route.execute(
                backend.model, lambda entry=entry, modules=modules:
                    factorial.run_factorial_arm(
                        backend, tokens, entry, modules, position_rows))
            dynamics[name], state_calls, module_calls = result
            coverage[name] = {"entry_calls": state_calls,
                              "module_calls": module_calls, "final_calls": final_calls}

        lambdas = [backend.model.transformer.h[layer].lambdas[0].detach().float()
                   for layer in range(10, 18)]
        gain = residual_route.skip_gain(value.item() for value in lambdas)
        add_state = residual_route.direct_add(native_x18, native_x10, writer_x10, gain)
        remove_state = residual_route.direct_remove(writer_x18, native_x10, writer_x10, gain)

        (self_result, self_raw, self_calls) = final_route.execute(
            backend.model, lambda: parent._forward(backend, tokens),
            replacement=native_x18, position_rows=position_rows)
        self_logits = self_result[0]
        (add_result, add_raw, add_calls) = final_route.execute(
            backend.model, lambda: parent._forward(backend, tokens),
            replacement=add_state, position_rows=position_rows)
        add_logits = add_result[0]
        (remove_result, remove_raw, remove_calls) = final_route.execute(
            backend.model, lambda: factorial.atlas.mediation.forward_with_l11_capture(
                backend, tokens, head_captures=head_captures,
                selected_labels=selected_heads, pairs=pairs,
                position_rows=source_rows),
            replacement=remove_state, position_rows=position_rows)
        remove_logits = remove_result[0]

    logits = {"writer": writer_logits, "R1M0_dynamic": dynamics["R1M0_dynamic"],
              "R0M1_dynamic": dynamics["R0M1_dynamic"],
              "native_x18_self_patch": self_logits, "direct_add": add_logits,
              "direct_remove": remove_logits}
    raw_final = {"native_x18_self_patch": self_raw, "direct_add": add_raw,
                 "direct_remove": remove_raw}
    native_margins = {name: loc.margins(native_logits, endpoints, name)
                      for name in parent.LOCKED}
    effects = {name: loc.margins(value, endpoints, role) - native_margins[role]
               for name, value in logits.items()}
    other_effects = {name: loc.margins(value, endpoints, other) - native_margins[other]
                     for name, value in logits.items()}
    writer_effect = effects["writer"]
    writer_other = other_effects["writer"]
    gold = native_margins[role][pairs] - native_margins[role]
    other_gold = native_margins[other][other_pairs] - native_margins[other]
    reports = []
    for phase in ("FIT", "HOLDOUT"):
        chosen = np.asarray([row["phase"] == phase for row, _cell, _endpoint in endpoints])
        temporal_norm = max(float(np.linalg.norm(other_gold[chosen])), 1e-30)
        for arm in ARMS:
            metric = factorial.atlas.mediation.accounting.effect_metrics(
                effects[arm][chosen], writer_effect[chosen], other_effects[arm][chosen])
            report = {"population": population, "phase": phase, "arm": arm, **metric,
                      "temporal_command_gold_collateral":
                          float(np.linalg.norm(other_effects[arm][chosen]) / temporal_norm)}
            if arm == "direct_remove":
                removed = writer_effect[chosen] - effects[arm][chosen]
                removal_other = writer_other[chosen] - other_effects[arm][chosen]
                removal = factorial.atlas.mediation.accounting.effect_metrics(
                    removed, writer_effect[chosen], removal_other)
                report.update({f"identity_removal_{key}": value
                               for key, value in removal.items()})
            reports.append(report)

    instrumentation = {
        "skip_gain": gain,
        "native_writer_capture_calls": {
            "x10": [native_x10_calls, writer_x10_calls],
            "x18": [native_x18_calls, writer_x18_calls],
            "native_modules": native_module_calls, "writer_modules": writer_module_calls},
        "dynamic_coverage": coverage,
        "final_patch_calls": {"self": self_calls, "add": add_calls, "remove": remove_calls},
        "native_self_patch_max_abs_logit_error": maxdiff(self_logits, native_logits),
        "native_self_patch_raw_state_relative_l2":
            state_relative_l2(self_raw, native_x18, position_rows),
        "direct_add_vs_R1M0_max_abs_logit_error":
            maxdiff(add_logits, dynamics["R1M0_dynamic"]),
        "direct_remove_vs_R0M1_max_abs_logit_error":
            maxdiff(remove_logits, dynamics["R0M1_dynamic"]),
        "direct_add_vs_R1M0_state_relative_l2":
            state_relative_l2(dynamic_states["R1M0_dynamic"], add_state, position_rows),
        "direct_remove_vs_R0M1_state_relative_l2":
            state_relative_l2(dynamic_states["R0M1_dynamic"], remove_state, position_rows),
        "raw_final_replay": {
            "self": state_relative_l2(self_raw, native_x18, position_rows),
            "add": state_relative_l2(add_raw, native_x18, position_rows),
            "remove": state_relative_l2(remove_raw, writer_x18, position_rows)},
        "pairing_ok": len(pairs) == len(endpoints) and len(other_pairs) == len(endpoints),
        "positions_ok": all(set(source_rows[index]) <= set(position_rows[index])
                            for index in range(len(position_rows))),
    }
    return {"reports": reports, "instrumentation": instrumentation,
            "row_count": len(endpoints), "selected_heads": list(selected_heads),
            "capture_shapes": {"x10": list(native_x10.shape), "x18": list(native_x18.shape)}}


def report_index(panels):
    return {(row["population"], row["phase"], row["arm"]): row
            for panel in panels.values() for row in panel["reports"]}


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior = json.loads(PRIOR.read_text())
    bound, authority_ok = binding_state(observed)
    dry = {"candidate_id": prior["candidate_id"], "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "status": "bound" if authority_ok else "awaiting_binding",
           "authority_ok": authority_ok, "static_authority_ok": observed == EXPECTED,
           "arms": list(ARMS), "populations": ["original", "ood"], "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if not authority_ok:
        raise RuntimeError("direct residual route is not bound to an eligible factorial result")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    if prior["price"] != PRICE:
        raise RuntimeError("direct-route price changed")

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = factorial.atlas.mediation.parent.loc.producer.Bilin18TorchBackend.load("cuda")
    greedy = json.loads(GREEDY_RESULT.read_text())
    selected_heads = tuple(greedy["selected_prefixes"]["iswas"]["heads"])
    if selected_heads != ("L07H07", "L09H04"):
        raise RuntimeError("selected compact writer changed")
    loc = factorial.atlas.mediation.parent.loc
    panels = {
        "original": run_population(backend, loc.original, "original", selected_heads),
        "ood": run_population(backend, loc.ood, "ood", selected_heads),
    }
    by = report_index(panels)
    instruments = [panel["instrumentation"] for panel in panels.values()]
    A = bool(authority_ok
        and bound["factorial"].get("writer_replay_max_abs_error", math.inf)
            <= BARS["authority_replay"]
        and sum(panel["row_count"] * 7 for panel in panels.values())
        == PRICE["sequence_evaluations"] and all(item["pairing_ok"] and item["positions_ok"]
        and item["native_self_patch_max_abs_logit_error"] <= BARS["self_patch_logit"]
        and item["native_self_patch_raw_state_relative_l2"] <= BARS["state_relative_l2"]
        and item["native_writer_capture_calls"]["x10"] == [1, 1]
        and item["native_writer_capture_calls"]["x18"] == [1, 1]
        and set(item["native_writer_capture_calls"]["native_modules"].values()) == {1}
        and set(item["native_writer_capture_calls"]["writer_modules"].values()) == {1}
        and set(item["final_patch_calls"].values()) == {1}
        and all(cell["entry_calls"] == 1 and cell["final_calls"] == 1
                and set(cell["module_calls"].values()) == {1}
                for cell in item["dynamic_coverage"].values()) for item in instruments)
        and finite(panels) and PRICE == prior["price"])
    B = all(item["direct_add_vs_R1M0_max_abs_logit_error"] <= BARS["direct_logit"]
            and item["direct_add_vs_R1M0_state_relative_l2"] <= BARS["state_relative_l2"]
            for item in instruments)
    C = all(item["direct_remove_vs_R0M1_max_abs_logit_error"] <= BARS["direct_logit"]
            and item["direct_remove_vs_R0M1_state_relative_l2"] <= BARS["state_relative_l2"]
            for item in instruments)
    D = all(by[(population, phase, "direct_add")]["signed_recovery"] >= BARS["recovery"]
            and by[(population, phase, "direct_add")]["cosine"] >= BARS["cosine"]
            and by[(population, phase, "direct_add")]["direction_agreement"] >= BARS["direction"]
            for population, phase in VALIDATION_CELLS)
    E = bool(all(by[(population, phase, "direct_remove")]["identity_removal_signed_recovery"]
                 >= BARS["removal_fraction"]
                 and by[(population, phase, "direct_remove")]["identity_removal_direction_agreement"]
                 >= BARS["removal_direction"] for population, phase in VALIDATION_CELLS)
        and all(row["temporal_command_gold_collateral"] <= BARS["collateral"]
                for panel in panels.values() for row in panel["reports"]))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid" if not A else "arithmetic_or_hook_failure" if not (B and C)
                else "ood_instability" if not D else "nonselective_or_redundant" if not E
                else "direct_residual_identified")
    result = {"schema": "temporal_iswas_selected_writer_block10_direct_residual_add_remove_result_v1",
        "candidate_id": prior["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": {**observed, "binding": sha(BINDING),
                             "factorial_result": sha(FACTORIAL_RESULT)},
        "factorial_authority": bound["factorial"], "selected_writer_heads": list(selected_heads),
        "panels": panels, "predictions": predictions, "terminal": terminal,
        "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({"predictions": predictions, "terminal": terminal,
                      "instrumentation": {name: panel["instrumentation"]
                                          for name, panel in panels.items()},
                      "validation": [row for panel in panels.values() for row in panel["reports"]
                          if (row["population"], row["phase"]) in VALIDATION_CELLS],
                      "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
