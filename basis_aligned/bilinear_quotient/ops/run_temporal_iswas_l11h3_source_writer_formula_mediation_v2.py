#!/usr/bin/env python3
"""Mediate selected upstream writers through the exact L11H3 source formula."""

# BQGATE: EXPERIMENT pred_a_bound_authority_pairing_writer_replay_capture_coverage_finiteness_and_exact_price pred_b_exact_recipient_native_formula_equals_the_value_mediator pred_c_selected_writer_effects_are_mediated_by_the_l11h3_source_formula pred_d_formula_mediation_is_stable_within_templates pred_e_formula_mediation_is_task_selective_and_causal
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import dual_command_head_module_factorial_contract as accounting
import run_temporal_iswas_l11h3_native_routing_source_term_extraction_v1 as source
import run_temporal_iswas_l11h3_source_tensor_upstream_head_factorial_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR_V1 = ROOT / "circuits/prior_art/temporal_iswas_l11h3_source_writer_formula_mediation_v1_conditional.json"
PRIOR_V2 = ROOT / "circuits/prior_art/temporal_iswas_l11h3_source_writer_formula_mediation_v2_price_amendment.json"
BINDING = ROOT / "circuits/prior_art/temporal_iswas_l11h3_source_writer_formula_mediation_v3_authority_binding.json"
GREEDY = ROOT / "circuits/followups/temporal_iswas_l11h3_source_writer_weight_ordered_greedy_v1_result.json"
EXTRACTOR_RESULT = ROOT / "circuits/followups/temporal_iswas_l11h3_native_routing_source_term_extraction_v1_result.json"
GREEDY_RUNNER = ROOT / "ops/run_temporal_iswas_l11h3_source_writer_weight_ordered_greedy_v1.py"
SOURCE_RUNNER = ROOT / "ops/run_temporal_iswas_l11h3_native_routing_source_term_extraction_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_l11h3_source_writer_formula_mediation_v2_result.json"
EXPECTED = {
    "prior_v1": "fd8a531168d098b99189e7aff21596c11eff5d88aeacb9ca321f28bd417b456a",
    "prior_v2": "5c32755d9ffc7cd7323ba50121c7990ba68b34f3f90e1ddb7b3fbda3ec01a80b",
    "extractor_result": "d55e448e5c950f9a6e316cc43e8d1ca7c95fd625b31021b2572a8e20ba4f245a",
    "greedy_runner": "6660183126fd155e9ab86511e798a564879ef86818760dde256177916d1d13b5",
    "source_runner": "7c68faa314fe38ca0d1579b3c5ff3f6e12439923e49bec767bfb65212beb48dc",
}
PRICE = {"checkpoint_loads": 1, "model_forwards": 18, "sequence_evaluations": 2304,
         "scored_token_positions": 4608, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_bound_authority_pairing_writer_replay_capture_coverage_finiteness_and_exact_price",
    "pred_b_exact_recipient_native_formula_equals_the_value_mediator",
    "pred_c_selected_writer_effects_are_mediated_by_the_l11h3_source_formula",
    "pred_d_formula_mediation_is_stable_within_templates",
    "pred_e_formula_mediation_is_task_selective_and_causal",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward_with_l11_capture(backend, tokens, *, head_captures=None,
                             selected_labels=(), pairs=None, position_rows=None):
    """Use the parent writer intervention while independently observing L11 factors."""
    model = backend.model
    saved, handles = {}, []
    factors = ("q", "k", "q2", "k2", "v")
    calls = {name: 0 for name in factors}
    projection_calls = 0

    def factor_hook(name):
        def hook(_module, _inputs, output):
            calls[name] += 1
            saved[name] = output.detach().clone()
        return hook

    def projection_pre(_module, arguments):
        nonlocal projection_calls
        projection_calls += 1
        saved["head"] = arguments[0].detach().clone()

    attention = model.transformer.h[11].attn
    for name in factors:
        handles.append(getattr(attention, f"c_{name}").register_forward_hook(factor_hook(name)))
    handles.append(attention.c_proj.register_forward_pre_hook(projection_pre))
    try:
        logits, writer_captures = parent._forward(backend, tokens,
            capture=head_captures is None, captures=head_captures,
            selected_labels=selected_labels, pairs=pairs, position_rows=position_rows)
    finally:
        for handle in handles:
            handle.remove()
    if set(calls.values()) != {1} or projection_calls != 1 or set(saved) != {*factors, "head"}:
        raise RuntimeError("combined writer/L11 capture coverage changed")
    return logits, writer_captures, saved


def recipient_native_source_formula(native, induced_v, source_rows, attention, torch, F,
                                    head=3):
    """Contract an arbitrary induced source-local value change through native routing."""
    raw = {name: native[name].view(native[name].shape[0], native[name].shape[1], 9, 128)
           for name in ("q", "k", "q2", "k2")}
    native_v = native["v"].view(native["v"].shape[0], native["v"].shape[1], 9, 128)
    changed_v = induced_v.view(induced_v.shape[0], induced_v.shape[1], 9, 128)
    cos, sin = attention.rotary(raw["q"])
    apply_rotary = sys.modules[type(attention).__module__].apply_rotary_emb
    q = apply_rotary(F.rms_norm(raw["q"], (128,)), cos, sin)
    k = apply_rotary(F.rms_norm(raw["k"], (128,)), cos, sin)
    q2 = apply_rotary(F.rms_norm(raw["q2"], (128,)), cos, sin)
    k2 = apply_rotary(F.rms_norm(raw["k2"], (128,)), cos, sin)
    pattern = (torch.einsum("btd,bsd->bts", q[:, :, head], k[:, :, head]) / 128
               * torch.einsum("btd,bsd->bts", q2[:, :, head], k2[:, :, head]) / 128)
    causal = torch.tril(torch.ones(pattern.shape[1:], dtype=torch.bool, device=pattern.device))
    pattern = pattern.masked_fill(~causal, 0)
    delta_v = changed_v[:, :, head] - native_v[:, :, head]
    mask = torch.zeros(delta_v.shape[:2], dtype=torch.bool, device=delta_v.device)
    for index, positions in enumerate(source_rows):
        mask[index, list(positions)] = True
    delta_v = (1 - attention.lamb) * delta_v * mask.unsqueeze(-1)
    return torch.einsum("bts,bsd->btd", pattern, delta_v)


def run_population(backend, authority, population, greedy, selected_heads):
    loc = parent.loc
    rows = authority.build_rows()
    endpoints, lookup = loc.parent.endpoint_bank(rows)
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    torch, F = backend.torch, backend.F
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints],
                          dtype=torch.long, device=backend.device)
    pairs = {role: loc.parent.pair_indices(endpoints, lookup, role) for role in parent.LOCKED}
    identity = np.arange(len(endpoints), dtype=np.int64)
    queries = {role: [endpoint[f"{role}_position"] - 1
                      for _row, _cell, endpoint in endpoints] for role in parent.LOCKED}
    regions = {role: [loc.source_regions(endpoint["ids"],
                endpoints[pairs[role][index]][2]["ids"], queries[role][index])
                for index, (_row, _cell, endpoint) in enumerate(endpoints)]
               for role in parent.LOCKED}
    with torch.no_grad():
        native_logits, head_captures, native_l11 = forward_with_l11_capture(backend, tokens)
    native = {role: loc.margins(native_logits, endpoints, role) for role in parent.LOCKED}
    reports, algebra, replay_errors = [], {}, []
    formula_collateral = {}
    for role, source_region in parent.LOCKED.items():
        source_rows = [item[source_region] for item in regions[role]]
        other = "iswas" if role == "temporal" else "temporal"
        with torch.no_grad():
            writer_logits, _unused, writer_l11 = forward_with_l11_capture(backend, tokens,
                head_captures=head_captures, selected_labels=selected_heads[role],
                pairs=pairs[role], position_rows=source_rows)
            reference_logits, _ = source._forward(backend, tokens, donor_v=native_l11["v"],
                pairs=pairs[role], value_positions=source_rows)
            value_logits, value_capture = source._forward(backend, tokens, capture=True,
                donor_v=writer_l11["v"], pairs=identity, value_positions=source_rows)
            formula = recipient_native_source_formula(native_l11, writer_l11["v"], source_rows,
                backend.model.transformer.h[11].attn, torch, F)
            formula_logits, _ = source._forward(backend, tokens, direct_delta=formula)
        width = native_l11["head"].shape[-1] // 9
        sl = slice(3 * width, 4 * width)
        observed = value_capture["head"][..., sl] - native_l11["head"][..., sl]
        difference = formula.float() - observed.float()
        algebra[role] = {
            "formula_head_max_abs_error": float(difference.abs().max()),
            "formula_head_relative_l2_error": float(difference.norm()
                / max(float(observed.float().norm()), 1e-30)),
            "formula_value_selected_logit_max_abs_error": float(np.max(np.abs(
                source.selected_logits(formula_logits, endpoints)
                - source.selected_logits(value_logits, endpoints)))),
        }
        writer_effect = loc.margins(writer_logits, endpoints, role) - native[role]
        reference_effect = loc.margins(reference_logits, endpoints, role) - native[role]
        formula_effect = loc.margins(formula_logits, endpoints, role) - native[role]
        writer_other = loc.margins(writer_logits, endpoints, other) - native[other]
        formula_other = loc.margins(formula_logits, endpoints, other) - native[other]
        formula_collateral[role] = formula_other
        old_arm = greedy["selected_prefixes"][role]["arm"]
        old_reports = greedy["panels"][population]["reports"]
        for phase in ("FIT", "HOLDOUT"):
            for template in ("ALL",) + authority.TEMPLATES:
                chosen = np.asarray([row["phase"] == phase and
                    (template == "ALL" or row["template_id"] == template)
                    for row, _cell, _endpoint in endpoints])
                writer_report = accounting.effect_metrics(writer_effect[chosen],
                    reference_effect[chosen], writer_other[chosen])
                gold = native[role][pairs[role]] - native[role]
                writer_gold = accounting.effect_metrics(writer_effect[chosen], gold[chosen],
                                                          writer_other[chosen])
                old = next(row for row in old_reports if row["role"] == role
                    and row["phase"] == phase and row["template_id"] == template
                    and row["arm"] == old_arm)
                for field in ("signed_recovery", "cosine", "relative_residual",
                              "direction_agreement", "non_target_to_target_gold_norm"):
                    replay_errors.append(abs(writer_report[field] - old[field]))
                replay_errors.append(abs(writer_gold["non_target_to_target_gold_norm"]
                                         - old["command_gold_non_target_norm_ratio"]))
                mediation = accounting.effect_metrics(formula_effect[chosen],
                                                        writer_effect[chosen],
                                                        formula_other[chosen])
                formula_gold = accounting.effect_metrics(formula_effect[chosen], gold[chosen],
                                                           formula_other[chosen])
                reports.append({"population": population, "role": role, "phase": phase,
                    "template_id": template, "selected_arm": old_arm,
                    "selected_heads": list(selected_heads[role]), **mediation,
                    "formula_command_gold_non_target_norm_ratio":
                        formula_gold["non_target_to_target_gold_norm"]})
    causal_zero = max(abs(value) for value in formula_collateral["iswas"])
    return {"reports": reports, "algebra": algebra,
            "writer_replay_max_abs_error": max(replay_errors),
            "causal_zero": float(causal_zero),
            "capture_shapes": {name: list(value.shape) for name, value in native_l11.items()},
            "writer_capture_shapes": {name: list(value.shape) for name, value in head_captures.items()},
            "coverage_ok": all(set(item[parent.LOCKED[role]]) <= set(item["full_prefix"])
                               for role in parent.LOCKED for item in regions[role])}


def main():
    base_paths = {"prior_v1": PRIOR_V1, "prior_v2": PRIOR_V2,
                  "extractor_result": EXTRACTOR_RESULT,
                  "greedy_runner": GREEDY_RUNNER, "source_runner": SOURCE_RUNNER}
    observed = {name: sha(path) for name, path in base_paths.items()}
    awaiting_binding = not BINDING.exists() or not GREEDY.exists()
    dry = {"candidate_id": "cross_task.temporal_iswas.l11h3_source_writer_formula_mediation_v2",
           "dryrun": True, "awaiting_binding": awaiting_binding, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        if not awaiting_binding:
            binding = json.loads(BINDING.read_text())
            dry["authority_ok"] = bool(observed == EXPECTED
                and sha(GREEDY) == binding.get("greedy_result_sha256"))
            dry["selected_prefixes"] = binding.get("selected_prefixes")
        print(json.dumps(dry, sort_keys=True))
        return
    if awaiting_binding:
        raise RuntimeError("mediation is frozen but not authority-bound to a greedy result")
    binding, greedy = json.loads(BINDING.read_text()), json.loads(GREEDY.read_text())
    selected_heads = {role: tuple(greedy["selected_prefixes"][role]["heads"])
                      for role in parent.LOCKED}
    authority_ok = bool(observed == EXPECTED
        and sha(GREEDY) == binding.get("greedy_result_sha256")
        and binding.get("selected_prefixes") == greedy.get("selected_prefixes")
        and greedy.get("predictions", {}).get(
            "pred_c_selected_prefixes_validate_without_reselection") is True
        and all(greedy.get("selected_prefixes", {}).get(role) for role in parent.LOCKED))
    if not authority_ok:
        raise RuntimeError("mediation authority or selected-prefix binding changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = parent.loc.producer.Bilin18TorchBackend.load("cuda")
    panels = {
        "original": run_population(backend, parent.loc.original, "original", greedy, selected_heads),
        "ood": run_population(backend, parent.loc.ood, "ood", greedy, selected_heads),
    }
    reports = [row for panel in panels.values() for row in panel["reports"]]
    algebra = [row for panel in panels.values() for row in panel["algebra"].values()]
    validation_cells = (("original", "HOLDOUT"), ("ood", "FIT"), ("ood", "HOLDOUT"))

    def report(population, role, phase, template):
        return next(row for row in reports if row["population"] == population
            and row["role"] == role and row["phase"] == phase
            and row["template_id"] == template)

    A = bool(authority_ok and max(panel["writer_replay_max_abs_error"]
        for panel in panels.values()) <= 1e-5 and all(panel["coverage_ok"] for panel in panels.values())
        and all(set(panel["capture_shapes"]) == {"q", "k", "q2", "k2", "v", "head"}
                and set(panel["writer_capture_shapes"]) == {*parent.LABELS, "L11H3:v"}
                for panel in panels.values()) and source.finite(panels)
        and PRICE == json.loads(PRIOR_V2.read_text())["amended_price"])
    B = bool(all(row["formula_head_max_abs_error"] <= 1e-4
        and row["formula_head_relative_l2_error"] <= 1e-4
        and row["formula_value_selected_logit_max_abs_error"] <= 1e-4 for row in algebra))
    C = bool(all(report(population, role, phase, "ALL")["signed_recovery"] >= .75
        and report(population, role, phase, "ALL")["cosine"] >= .95
        and report(population, role, phase, "ALL")["relative_residual"] <= .50
        and report(population, role, phase, "ALL")["direction_agreement"] >= .90
        for population, phase in validation_cells for role in parent.LOCKED))
    D = bool(all(report(population, role, phase, template)["signed_recovery"] >= .50
        and report(population, role, phase, template)["direction_agreement"] >= .75
        for population, phase in validation_cells for role in parent.LOCKED
        for template in (parent.loc.original.TEMPLATES if population == "original"
                         else parent.loc.ood.TEMPLATES)))
    E = bool(all(panel["causal_zero"] <= 1e-5 for panel in panels.values())
        and all(report(population, role, phase, "ALL")[
            "formula_command_gold_non_target_norm_ratio"] <= .01
            for population, phase in validation_cells for role in parent.LOCKED))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("exact_writer_formula_mediation" if all(predictions.values()) else
                "invalid" if not A else "writer_formula_partial_mediation")
    result = {"schema": "temporal_iswas_l11h3_source_writer_formula_mediation_result_v2",
        "candidate_id": binding["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": {**observed, "binding": sha(BINDING), "greedy_result": sha(GREEDY)},
        "selected_prefixes": greedy["selected_prefixes"], "panels": panels,
        "predictions": predictions, "terminal": terminal, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({"selected_prefixes": result["selected_prefixes"],
        "algebra_max_abs": max(row["formula_head_max_abs_error"] for row in algebra),
        "writer_replay_max_abs_error": max(panel["writer_replay_max_abs_error"]
            for panel in panels.values()),
        "validation_pooled": {role: {f"{population}_{phase}": report(
            population, role, phase, "ALL") for population, phase in validation_cells}
            for role in parent.LOCKED}, "predictions": predictions,
        "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
