#!/usr/bin/env python3
"""Exact eight-forward residual-route versus write-bank factorial."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_residual_write_bank_factorial as authority
import run_bracket_l13h8_mu_delta_mlp15_mediation as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_mu_delta_residual_write_bank_factorial_v1_result.json"


def bank_factor_forward(model, tokens, finals, sources, torch, F, facade, *,
                        replacement_terms=None, install_bank=None, capture_bank=False):
    """Replay exact L13H8 and capture or install the complete final-position write bank."""
    captured = {}
    arange = torch.arange(tokens.size(0), device=tokens.device)

    def bank_write(name, write):
        if capture_bank and name in authority.WRITE_BANK:
            captured[name] = write[arange, finals].detach().clone()
        if name in (install_bank or {}):
            write = write.clone()
            write[arange, finals] = install_bank[name].to(write.dtype)
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
        return bank_write(f"attention{event.site}", write), next_first_value

    def mlp(event):
        return bank_write(f"mlp{event.site}", event.block.mlp(event.state))

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=False).float()
    assert "factors" in captured
    if capture_bank:
        assert all(name in captured for name in authority.WRITE_BANK)
    return logits, captured


def projection(component, total, eps=1e-8):
    return float((component * total).sum()) / (float((total * total).sum()) + eps)


def evaluate(model, torch, F, facade):
    rows, device = authority.ROWS, next(model.parameters()).device
    tokens, finals, sources = base.parent.pad_rows(rows, torch, device)
    native_model = base.parent.shared.native_logits(model, tokens, torch, F)
    nn, native_capture = bank_factor_forward(
        model, tokens, finals, sources, torch, F, facade, capture_bank=True)
    replay_error = float((native_model - nn).abs().max())
    arange = torch.arange(len(rows), device=device)
    factors = native_capture["factors"]
    terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
    by_group = defaultdict(dict)
    for index, row in enumerate(rows):
        by_group[row["group_id"]][row["delimiter_index"]] = index
    mu = torch.empty_like(terms)
    for index, row in enumerate(rows):
        mu[index] = sum(terms[by_group[row["group_id"]][j]] for j in range(3)) / 3.0
    replacements = {"mu": terms - mu, "delta": mu}
    corners = {}
    native_bank = {name: native_capture[name] for name in authority.WRITE_BANK}
    for factor, replacement in replacements.items():
        rr, removed_capture = bank_factor_forward(
            model, tokens, finals, sources, torch, F, facade,
            replacement_terms=replacement, capture_bank=True)
        removed_bank = {name: removed_capture[name] for name in authority.WRITE_BANK}
        rn = bank_factor_forward(
            model, tokens, finals, sources, torch, F, facade,
            replacement_terms=replacement, install_bank=native_bank)[0]
        nr = bank_factor_forward(
            model, tokens, finals, sources, torch, F, facade,
            install_bank=removed_bank)[0]
        corners[factor] = (rr, rn, nr)
    records = []
    for index, row in enumerate(rows):
        q, answer = int(finals[index]), row["answer_id"]
        c_nn = base.centered_closer(nn[index, q], torch)
        l_nn = float(F.cross_entropy(nn[index, q].unsqueeze(0),
                                     torch.tensor([answer], device=device)))
        for factor in ("mu", "delta"):
            rr, rn, nr = (corner[index, q] for corner in corners[factor])
            c_rr, c_rn, c_nr = (base.centered_closer(value, torch) for value in (rr, rn, nr))
            total = c_nn - c_rr
            residual = c_nn - c_rn
            write = c_nn - c_nr
            interaction = c_nn - c_rn - c_nr + c_rr
            l_rr, l_rn, l_nr = (float(F.cross_entropy(
                value.unsqueeze(0), torch.tensor([answer], device=device))) for value in (rr, rn, nr))
            ce_total, ce_residual, ce_write = l_rr - l_nn, l_rn - l_nn, l_nr - l_nn
            ce_interaction = l_rr - l_rn - l_nr + l_nn
            records.append({
                "row_id": row["row_id"], "group_id": row["group_id"],
                "family_id": row["family_id"], "role": row["role"],
                "delimiter_index": row["delimiter_index"], "factor": factor,
                "native_centered_correct_closer": float(c_nn[row["delimiter_index"]]),
                "total_effect_norm": float(total.norm()),
                "residual_path_projection": projection(residual, total),
                "write_bank_projection": projection(write, total),
                "interaction_projection": projection(interaction, total),
                "vector_identity_max_absolute_error": float(
                    (residual + write - interaction - total).abs().max()),
                "signed_correct_answer_ce_total_damage": ce_total,
                "signed_correct_answer_ce_residual_path_damage": ce_residual,
                "signed_correct_answer_ce_write_bank_damage": ce_write,
                "signed_correct_answer_ce_loss_interaction": ce_interaction,
                "ce_identity_absolute_error": abs(
                    ce_total - ce_residual - ce_write - ce_interaction),
            })
    return records, replay_error


def score(records, replay_error):
    bars = authority.compile_plan()["bars"]
    cells, families = defaultdict(list), defaultdict(dict)
    for row in records:
        cells[(row["family_id"], row["factor"])].append(row)
        families[row["family_id"]][row["row_id"]] = row["native_centered_correct_closer"]
    reports = {}
    for (family, factor), rows in sorted(cells.items()):
        reports[f"{family}|{factor}"] = {
            "n": len(rows),
            "median_total_effect_norm": statistics.median(r["total_effect_norm"] for r in rows),
            "median_residual_path_projection": statistics.median(r["residual_path_projection"] for r in rows),
            "median_write_bank_projection": statistics.median(r["write_bank_projection"] for r in rows),
            "median_interaction_projection": statistics.median(r["interaction_projection"] for r in rows),
            "median_absolute_interaction_projection": statistics.median(abs(r["interaction_projection"]) for r in rows),
            "median_ce_total_damage": statistics.median(r["signed_correct_answer_ce_total_damage"] for r in rows),
            "median_ce_residual_path_damage": statistics.median(r["signed_correct_answer_ce_residual_path_damage"] for r in rows),
            "median_ce_write_bank_damage": statistics.median(r["signed_correct_answer_ce_write_bank_damage"] for r in rows),
            "median_ce_loss_interaction": statistics.median(r["signed_correct_answer_ce_loss_interaction"] for r in rows),
        }
    native = {family: sum(value > 0 for value in values.values()) / len(values)
              for family, values in families.items()}
    checks = {
        "native_capability": min(native.values()) >= bars["native_positive_fraction_each_family_min"],
        "native_replay": replay_error <= bars["native_replay_max_absolute_logit_error_max"],
        "live_factor_effects": all(reports[f"{family}|{factor}"]["median_total_effect_norm"] >=
                                   bars["median_live_centered_effect_norm_each_factor_family_min"]
                                   for family in authority.FAMILIES for factor in ("mu", "delta")),
        "vector_identity": max(r["vector_identity_max_absolute_error"] for r in records) <=
                           bars["factorial_vector_identity_max_absolute_error"],
        "ce_identity": max(r["ce_identity_absolute_error"] for r in records) <=
                       bars["factorial_ce_identity_max_absolute_error"],
    }
    instrument_live = all(checks.values())
    targets = [reports[f"{family}|{factor}"] for family in authority.TARGET_FAMILIES
               for factor in ("mu", "delta")]
    residual_dominated = instrument_live and all(
        report["median_residual_path_projection"] >=
        bars["median_residual_projection_each_factor_target_family_min"] for report in targets)
    downstream_material = instrument_live and any(
        max(report["median_write_bank_projection"], report["median_absolute_interaction_projection"])
        >= bars["median_write_projection_or_absolute_interaction_projection_each_factor_target_family_min"]
        for report in targets)
    return {"instrument_checks": checks, "instrument_live": instrument_live,
            "native_positive_fraction_by_family": native, "family_factor_reports": reports,
            "stability_rewrites_reported_separately": list(authority.STABILITY_FAMILIES),
            "residual_dominated": residual_dominated,
            "downstream_write_or_interaction_material": downstream_material,
            "predictions": {"pred_a": instrument_live, "pred_b": residual_dominated,
                            "pred_c": downstream_material}}


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
    terminal = "invalid" if not screen["instrument_live"] else "screen"
    result = {"schema": "bracket_l13h8_mu_delta_residual_write_bank_factorial_result_v1",
              "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "native_replay_max_absolute_logit_error": replay_error,
              "raw": records, "screen": screen, "evaluated_splits": ["FRESH_BASIC"],
              "forbidden_splits_opened": [], "model_forwards": 8, "terminal": terminal}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "model_forwards": 8}, indent=2))


if __name__ == "__main__":
    main()
