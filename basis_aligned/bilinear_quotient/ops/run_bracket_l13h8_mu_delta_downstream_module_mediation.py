#!/usr/bin/env python3
"""Twenty-forward exact complete-module mediation screen."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_downstream_module_mediation as authority
import run_bracket_l13h8_mu_delta_mlp15_mediation as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_mu_delta_downstream_module_mediation_v1_result.json"


def module_factor_forward(model, tokens, finals, sources, torch, F, facade, *,
                          replacement_terms=None, restore_module=None,
                          restore_write=None, capture_modules=False):
    """Replay exact L13H8 and optionally restore one complete final-position module write."""
    captured = {}
    arange = torch.arange(tokens.size(0), device=tokens.device)

    def maybe_capture_or_restore(name, write):
        if capture_modules and name in authority.MODULES:
            captured[name] = write[arange, finals].detach().clone()
        if restore_module == name:
            write = write.clone()
            write[arange, finals] = restore_write.to(write.dtype)
        return write

    def attention(event):
        if event.site == authority.PATCH_LAYER:
            write, factors = base.parent.shared.replay_head(
                event.state, event.first_value, event.block.attn, finals, torch, F)
            captured["factors"] = {key: value.detach().clone() for key, value in factors.items()}
            if replacement_terms is not None:
                native_terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
                write = write.clone()
                write[arange, finals] += (replacement_terms - native_terms).to(write.dtype)
            next_first_value = event.first_value
        else:
            write, next_first_value = event.block.attn(event.state, event.first_value)
        write = maybe_capture_or_restore(f"attention{event.site}", write)
        return write, next_first_value

    def mlp(event):
        write = event.block.mlp(event.state)
        return maybe_capture_or_restore(f"mlp{event.site}", write)

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=False).float()
    assert "factors" in captured
    if capture_modules:
        assert all(module in captured for module in authority.MODULES)
    return logits, captured


def evaluate(model, torch, F, facade):
    rows, device = authority.ROWS, next(model.parameters()).device
    tokens, finals, sources = base.parent.pad_rows(rows, torch, device)
    native = base.parent.shared.native_logits(model, tokens, torch, F)
    replay, captured = module_factor_forward(
        model, tokens, finals, sources, torch, F, facade, capture_modules=True)
    replay_error = float((native - replay).abs().max())
    arange = torch.arange(len(rows), device=device)
    factors = captured["factors"]
    terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
    by_group = defaultdict(dict)
    for index, row in enumerate(rows):
        by_group[row["group_id"]][row["delimiter_index"]] = index
    mu = torch.empty_like(terms)
    for index, row in enumerate(rows):
        mu[index] = sum(terms[by_group[row["group_id"]][j]] for j in range(3)) / 3.0
    replacements = {"mu": terms - mu, "delta": mu}
    removed, restored = {}, {}
    for factor, replacement in replacements.items():
        removed[factor] = module_factor_forward(
            model, tokens, finals, sources, torch, F, facade,
            replacement_terms=replacement)[0]
        for module in authority.MODULES:
            restored[(factor, module)] = module_factor_forward(
                model, tokens, finals, sources, torch, F, facade,
                replacement_terms=replacement, restore_module=module,
                restore_write=captured[module])[0]
    records = []
    for index, row in enumerate(rows):
        q, answer = int(finals[index]), row["answer_id"]
        native_vector = base.centered_closer(replay[index, q], torch)
        native_ce = float(F.cross_entropy(
            replay[index, q].unsqueeze(0), torch.tensor([answer], device=device)))
        for factor in ("mu", "delta"):
            removed_vector = base.centered_closer(removed[factor][index, q], torch)
            removed_ce = float(F.cross_entropy(
                removed[factor][index, q].unsqueeze(0), torch.tensor([answer], device=device)))
            for module in authority.MODULES:
                restored_logits = restored[(factor, module)][index, q]
                metrics = base.vector_metrics(
                    native_vector, removed_vector, base.centered_closer(restored_logits, torch))
                restored_ce = float(F.cross_entropy(
                    restored_logits.unsqueeze(0), torch.tensor([answer], device=device)))
                records.append({
                    "row_id": row["row_id"], "group_id": row["group_id"],
                    "family_id": row["family_id"], "role": row["role"],
                    "delimiter_index": row["delimiter_index"], "factor": factor,
                    "module": module,
                    "native_centered_correct_closer": float(native_vector[row["delimiter_index"]]),
                    **metrics, "signed_correct_answer_ce_change": removed_ce - native_ce,
                    "signed_correct_answer_ce_rescue": removed_ce - restored_ce,
                })
    return records, replay_error


def score(records, replay_error):
    bars = authority.compile_plan()["bars"]
    cells, families = defaultdict(list), defaultdict(dict)
    for row in records:
        cells[(row["family_id"], row["factor"], row["module"])].append(row)
        families[row["family_id"]][row["row_id"]] = row["native_centered_correct_closer"]
    reports = {}
    for (family, factor, module), rows in sorted(cells.items()):
        reports[f"{family}|{factor}|{module}"] = {
            "n": len(rows),
            "median_total_effect_norm": statistics.median(r["total_effect_norm"] for r in rows),
            "median_projection_recovery": statistics.median(r["projection_recovery"] for r in rows),
            "median_rescue_cosine": statistics.median(r["rescue_cosine"] for r in rows),
            "positive_projection_fraction": sum(r["projection_recovery"] > 0 for r in rows) / len(rows),
            "median_signed_correct_answer_ce_change": statistics.median(
                r["signed_correct_answer_ce_change"] for r in rows),
            "median_signed_correct_answer_ce_rescue": statistics.median(
                r["signed_correct_answer_ce_rescue"] for r in rows),
        }
    native = {family: sum(value > 0 for value in values.values()) / len(values)
              for family, values in families.items()}
    live = all(reports[f"{family}|{factor}|mlp13"]["median_total_effect_norm"] >=
               bars["median_live_centered_effect_norm_each_factor_family_min"]
               for family in authority.FAMILIES for factor in ("mu", "delta"))
    instrument = {"native_capability": min(native.values()) >=
                  bars["native_positive_fraction_each_family_min"],
                  "native_replay": replay_error <= bars["native_replay_max_absolute_logit_error_max"],
                  "live_factor_effects": live}

    def passes(family, factor, module):
        report = reports[f"{family}|{factor}|{module}"]
        return (report["median_projection_recovery"] >=
                bars["median_projection_recovery_each_factor_target_family_min"]
                and report["median_rescue_cosine"] >= bars["median_rescue_cosine_each_factor_target_family_min"]
                and report["positive_projection_fraction"] >=
                bars["positive_projection_fraction_each_factor_target_family_min"])

    qualifying = {factor: [module for module in authority.MODULES
                           if all(passes(family, factor, module)
                                  for family in authority.TARGET_FAMILIES)]
                  for factor in ("mu", "delta")}
    instrument_live = all(instrument.values())
    mediated = instrument_live and all(qualifying.values())
    return {"instrument_checks": instrument, "instrument_live": instrument_live,
            "native_positive_fraction_by_family": native,
            "family_factor_module_reports": reports,
            "qualifying_modules_by_factor_on_target_constructions": qualifying,
            "stability_rewrites_reported_separately": list(authority.STABILITY_FAMILIES),
            "module_mediation_held": mediated,
            "predictions": {"pred_a": instrument_live, "pred_b": mediated,
                            "pred_c": instrument_live and not mediated}}


def main():
    plan = authority.compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists(): raise RuntimeError(f"refusing to overwrite {OUT}")
    base.parent.shared.candidate = authority
    torch, F, facade = base.parent.shared._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        records, replay_error = evaluate(model, torch, F, facade)
    screen = score(records, replay_error)
    terminal = "invalid" if not screen["instrument_live"] else (
        "screen" if screen["module_mediation_held"] else "null")
    result = {"schema": "bracket_l13h8_mu_delta_downstream_module_mediation_result_v1",
              "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "native_replay_max_absolute_logit_error": replay_error,
              "raw": records, "screen": screen, "evaluated_splits": ["FRESH_BASIC"],
              "forbidden_splits_opened": [], "model_forwards": 20, "terminal": terminal}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "model_forwards": 20}, indent=2))


if __name__ == "__main__":
    main()
