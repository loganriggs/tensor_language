#!/usr/bin/env python3
"""Exact six-forward BASIC screen of L13H8 mu/delta mediation by final-position MLP15."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_mlp15_mediation as authority
import run_bracket_l13h8_semantic_open_shared_contrast as parent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_mu_delta_mlp15_mediation_v1_result.json"
CLOSERS = parent.CLOSERS


def factor_mlp15_forward(model, tokens, finals, sources, torch, F, facade, *,
                         replacement_terms=None, restore_mlp15=None, capture_mlp15=False):
    """Replay exact L13H8, optionally replace its opener term and/or final MLP15 write."""
    captured = {}
    arange = torch.arange(tokens.size(0), device=tokens.device)

    def attention(event):
        if event.site != authority.PATCH_LAYER:
            return event.block.attn(event.state, event.first_value)
        write, factors = parent.shared.replay_head(
            event.state, event.first_value, event.block.attn, finals, torch, F)
        captured["factors"] = {key: value.detach().clone() for key, value in factors.items()}
        if replacement_terms is not None:
            native_terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
            write = write.clone()
            write[arange, finals] += (replacement_terms - native_terms).to(write.dtype)
        return write, event.first_value

    def mlp(event):
        write = event.block.mlp(event.state)
        if event.site == authority.MEDIATOR_LAYER:
            if capture_mlp15:
                captured["mlp15_final"] = write[arange, finals].detach().clone()
            if restore_mlp15 is not None:
                write = write.clone()
                write[arange, finals] = restore_mlp15.to(write.dtype)
        return write

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=False).float()
    assert "factors" in captured
    if capture_mlp15:
        assert "mlp15_final" in captured
    return logits, captured


def centered_closer(logits, torch):
    selected = logits[..., list(CLOSERS)]
    return selected - selected.mean(dim=-1, keepdim=True)


def vector_metrics(native, removed, restored, eps=1e-8):
    d, r = native - removed, restored - removed
    dot = float((r * d).sum())
    d2 = float((d * d).sum())
    r2 = float((r * r).sum())
    return {"total_effect_norm": d2 ** 0.5,
            "rescue_norm": r2 ** 0.5,
            "projection_recovery": dot / (d2 + eps),
            "rescue_cosine": dot / ((d2 * r2) ** 0.5 + eps)}


def evaluate(model, torch, F, facade):
    rows, device = authority.ROWS, next(model.parameters()).device
    tokens, finals, sources = parent.pad_rows(rows, torch, device)
    native = parent.shared.native_logits(model, tokens, torch, F)
    replay, captured = factor_mlp15_forward(
        model, tokens, finals, sources, torch, F, facade, capture_mlp15=True)
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
    delta = terms - mu
    remove_mu, _ = factor_mlp15_forward(
        model, tokens, finals, sources, torch, F, facade, replacement_terms=delta)
    remove_delta, _ = factor_mlp15_forward(
        model, tokens, finals, sources, torch, F, facade, replacement_terms=mu)
    restore_mu, _ = factor_mlp15_forward(
        model, tokens, finals, sources, torch, F, facade, replacement_terms=delta,
        restore_mlp15=captured["mlp15_final"])
    restore_delta, _ = factor_mlp15_forward(
        model, tokens, finals, sources, torch, F, facade, replacement_terms=mu,
        restore_mlp15=captured["mlp15_final"])
    outputs = {"mu": (remove_mu, restore_mu), "delta": (remove_delta, restore_delta)}
    records = []
    for index, row in enumerate(rows):
        q, answer = int(finals[index]), row["answer_id"]
        native_vector = centered_closer(replay[index, q], torch)
        native_ce = float(F.cross_entropy(replay[index, q].unsqueeze(0),
                                          torch.tensor([answer], device=device)))
        for factor, (removed, restored) in outputs.items():
            metrics = vector_metrics(native_vector, centered_closer(removed[index, q], torch),
                                     centered_closer(restored[index, q], torch))
            removed_ce = float(F.cross_entropy(removed[index, q].unsqueeze(0),
                                               torch.tensor([answer], device=device)))
            restored_ce = float(F.cross_entropy(restored[index, q].unsqueeze(0),
                                                torch.tensor([answer], device=device)))
            records.append({"row_id": row["row_id"], "group_id": row["group_id"],
                            "family_id": row["family_id"], "role": row["role"],
                            "delimiter_index": row["delimiter_index"], "factor": factor,
                            "native_centered_correct_closer": float(native_vector[row["delimiter_index"]]),
                            **metrics,
                            "signed_correct_answer_ce_change": removed_ce - native_ce,
                            "signed_correct_answer_ce_rescue": removed_ce - restored_ce})
    return records, replay_error


def score(records, replay_error):
    bars = authority.compile_plan()["bars"]
    cells = defaultdict(list)
    families = defaultdict(list)
    for row in records:
        cells[(row["family_id"], row["factor"])].append(row)
        families[row["family_id"]].append(row)
    reports = {}
    for (family, factor), rows in sorted(cells.items()):
        reports[f"{family}|{factor}"] = {
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
    native_by_family = {}
    for family, rows in families.items():
        unique = {row["row_id"]: row["native_centered_correct_closer"] for row in rows}
        native_by_family[family] = sum(value > 0 for value in unique.values()) / len(unique)
    instrument = {
        "native_capability": min(native_by_family.values()) >= bars["native_positive_fraction_each_family_min"],
        "native_replay": replay_error <= bars["native_replay_max_absolute_logit_error_max"],
        "live_factor_effects": all(report["median_total_effect_norm"] >=
                                   bars["median_live_centered_effect_norm_each_factor_family_min"]
                                   for report in reports.values()),
    }
    mediation = all(
        report["median_projection_recovery"] >= bars["median_projection_recovery_each_factor_family_min"]
        and report["median_rescue_cosine"] >= bars["median_rescue_cosine_each_factor_family_min"]
        and report["positive_projection_fraction"] >= bars["positive_projection_fraction_each_factor_family_min"]
        for report in reports.values())
    live = all(instrument.values())
    target_failure = any(
        reports[f"{family}|{factor}"]["median_projection_recovery"] < bars["median_projection_recovery_each_factor_family_min"]
        or reports[f"{family}|{factor}"]["median_rescue_cosine"] < bars["median_rescue_cosine_each_factor_family_min"]
        or reports[f"{family}|{factor}"]["positive_projection_fraction"] < bars["positive_projection_fraction_each_factor_family_min"]
        for family in authority.TARGET_FAMILIES for factor in ("mu", "delta"))
    return {"instrument_checks": instrument, "instrument_live": live,
            "native_positive_fraction_by_family": native_by_family,
            "family_factor_reports": reports,
            "mlp15_mediation_held": live and mediation,
            "predictions": {"pred_a": live, "pred_b": live and mediation,
                            "pred_c": live and target_failure}}


def main():
    plan = authority.compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists(): raise RuntimeError(f"refusing to overwrite {OUT}")
    parent.shared.candidate = authority
    torch, F, facade = parent.shared._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                           verify_weights_sha256=True)
    with torch.no_grad():
        records, replay_error = evaluate(model, torch, F, facade)
    screen = score(records, replay_error)
    terminal = "invalid" if not screen["instrument_live"] else (
        "screen" if screen["mlp15_mediation_held"] else "null")
    result = {"schema": "bracket_l13h8_mu_delta_mlp15_mediation_result_v1",
              "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "native_replay_max_absolute_logit_error": replay_error,
              "raw": records, "screen": screen, "evaluated_splits": ["FRESH_BASIC"],
              "forbidden_splits_opened": [], "model_forwards": 6, "terminal": terminal}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "model_forwards": 6}, indent=2))


if __name__ == "__main__":
    main()
