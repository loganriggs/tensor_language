#!/usr/bin/env python3
"""Exact ten-forward causal restoration of three fixed R549 attention heads."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_r549_attention_mediation as authority
import run_bracket_l13h8_mu_delta_mlp15_mediation as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_mu_delta_r549_attention_mediation_v1_result.json"


def replay_attention_with_head(state, first_value, attention, finals, head, torch, F):
    """Return the exact whole attention write and one head's post-OV residual contribution."""
    batch, length, width = state.shape
    heads, head_d = 9, width // 9
    q = base.parent.shared._linear(state, attention.c_q.weight, F).view(batch, length, heads, head_d)
    k = base.parent.shared._linear(state, attention.c_k.weight, F).view(batch, length, heads, head_d)
    q2 = base.parent.shared._linear(state, attention.c_q2.weight, F).view(batch, length, heads, head_d)
    k2 = base.parent.shared._linear(state, attention.c_k2.weight, F).view(batch, length, heads, head_d)
    raw = base.parent.shared._linear(state, attention.c_v.weight, F).view(batch, length, heads, head_d)
    value = (1 - attention.lamb) * raw + attention.lamb * first_value.view_as(raw)
    cos, sin = attention.rotary(q)
    rotary = sys.modules[type(attention).__module__].apply_rotary_emb
    q = rotary(F.rms_norm(q, (head_d,)), cos, sin)
    k = rotary(F.rms_norm(k, (head_d,)), cos, sin)
    q2 = rotary(F.rms_norm(q2, (head_d,)), cos, sin)
    k2 = rotary(F.rms_norm(k2, (head_d,)), cos, sin)
    pattern = torch.einsum("bqhd,bkhd->bhqk", q, k) / head_d
    pattern *= torch.einsum("bqhd,bkhd->bhqk", q2, k2) / head_d
    pattern = pattern.masked_fill(
        ~torch.tril(torch.ones(length, length, dtype=torch.bool, device=state.device)), 0)
    all_heads = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
    write = base.parent.shared._linear(
        all_heads.transpose(1, 2).contiguous().view(batch, length, width),
        attention.c_proj.weight, F)
    arange = torch.arange(batch, device=state.device)
    p = pattern[arange, head, finals]
    weight = attention.c_proj.weight[:, head * head_d:(head + 1) * head_d]
    u = base.parent.shared._linear(value[:, :, head].float(), weight.float(), F)
    contribution = torch.einsum("bk,bkd->bd", p.float(), u)
    return write, contribution


def attention_factor_forward(model, tokens, finals, sources, torch, F, facade, *,
                             replacement_terms=None, restore_site=None,
                             restore_contribution=None, capture_heads=False):
    """Replay L13H8; optionally restore one exact downstream final-position head write."""
    captured = {}
    arange = torch.arange(tokens.size(0), device=tokens.device)
    head_by_layer = {layer: head for layer, head in authority.HEADS}

    def attention(event):
        if event.site == authority.PATCH_LAYER:
            write, factors = base.parent.shared.replay_head(
                event.state, event.first_value, event.block.attn, finals, torch, F)
            captured["factors"] = {key: value.detach().clone() for key, value in factors.items()}
            if replacement_terms is not None:
                native_terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
                write = write.clone()
                write[arange, finals] += (replacement_terms - native_terms).to(write.dtype)
            return write, event.first_value
        if event.site not in head_by_layer:
            return event.block.attn(event.state, event.first_value)
        site = (event.site, head_by_layer[event.site])
        write, contribution = replay_attention_with_head(
            event.state, event.first_value, event.block.attn, finals, site[1], torch, F)
        if capture_heads:
            captured[site] = contribution.detach().clone()
        if restore_site == site:
            write = write.clone()
            write[arange, finals] += (restore_contribution - contribution).to(write.dtype)
        return write, event.first_value

    logits = facade.forward_with_dispatch(
        model, tokens, attention, lambda event: event.block.mlp(event.state),
        require_production=False).float()
    assert "factors" in captured
    if capture_heads:
        assert all(site in captured for site in authority.HEADS)
    return logits, captured


def evaluate(model, torch, F, facade):
    rows, device = authority.ROWS, next(model.parameters()).device
    tokens, finals, sources = base.parent.pad_rows(rows, torch, device)
    native = base.parent.shared.native_logits(model, tokens, torch, F)
    replay, captured = attention_factor_forward(
        model, tokens, finals, sources, torch, F, facade, capture_heads=True)
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
    removed = {}
    for factor, replacement in replacements.items():
        removed[factor] = attention_factor_forward(
            model, tokens, finals, sources, torch, F, facade,
            replacement_terms=replacement)[0]
    restored = {}
    for factor, replacement in replacements.items():
        for site in authority.HEADS:
            restored[(factor, site)] = attention_factor_forward(
                model, tokens, finals, sources, torch, F, facade,
                replacement_terms=replacement, restore_site=site,
                restore_contribution=captured[site])[0]
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
            for site in authority.HEADS:
                restored_logits = restored[(factor, site)][index, q]
                metrics = base.vector_metrics(
                    native_vector, removed_vector, base.centered_closer(restored_logits, torch))
                restored_ce = float(F.cross_entropy(
                    restored_logits.unsqueeze(0), torch.tensor([answer], device=device)))
                records.append({
                    "row_id": row["row_id"], "group_id": row["group_id"],
                    "family_id": row["family_id"], "role": row["role"],
                    "delimiter_index": row["delimiter_index"], "factor": factor,
                    "head": f"L{site[0]}H{site[1]}",
                    "native_centered_correct_closer": float(native_vector[row["delimiter_index"]]),
                    **metrics, "signed_correct_answer_ce_change": removed_ce - native_ce,
                    "signed_correct_answer_ce_rescue": removed_ce - restored_ce,
                })
    return records, replay_error


def score(records, replay_error):
    bars = authority.compile_plan()["bars"]
    cells, families = defaultdict(list), defaultdict(dict)
    for row in records:
        cells[(row["family_id"], row["factor"], row["head"])].append(row)
        families[row["family_id"]][row["row_id"]] = row["native_centered_correct_closer"]
    reports = {}
    for (family, factor, head), rows in sorted(cells.items()):
        reports[f"{family}|{factor}|{head}"] = {
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
    # Removal effects do not depend on restored head; use any head report per family/factor.
    live = all(reports[f"{family}|{factor}|L14H1"]["median_total_effect_norm"] >=
               bars["median_live_centered_effect_norm_each_factor_family_min"]
               for family in authority.FAMILIES for factor in ("mu", "delta"))
    instrument = {"native_capability": min(native.values()) >=
                  bars["native_positive_fraction_each_family_min"],
                  "native_replay": replay_error <= bars["native_replay_max_absolute_logit_error_max"],
                  "live_factor_effects": live}

    def passes(family, factor, head):
        report = reports[f"{family}|{factor}|{head}"]
        return (report["median_projection_recovery"] >=
                bars["median_projection_recovery_each_factor_family_min"]
                and report["median_rescue_cosine"] >= bars["median_rescue_cosine_each_factor_family_min"]
                and report["positive_projection_fraction"] >=
                bars["positive_projection_fraction_each_factor_family_min"])

    qualifying = {factor: [head for head in ("L14H1", "L15H3", "L16H1")
                           if all(passes(family, factor, head)
                                  for family in authority.TARGET_FAMILIES)]
                  for factor in ("mu", "delta")}
    instrument_live = all(instrument.values())
    mediated = instrument_live and all(qualifying.values())
    return {"instrument_checks": instrument, "instrument_live": instrument_live,
            "native_positive_fraction_by_family": native,
            "family_factor_head_reports": reports,
            "qualifying_heads_by_factor_on_target_constructions": qualifying,
            "stability_rewrites_reported_separately": list(authority.STABILITY_FAMILIES),
            "attention_mediation_held": mediated,
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
        "screen" if screen["attention_mediation_held"] else "null")
    result = {"schema": "bracket_l13h8_mu_delta_r549_attention_mediation_result_v1",
              "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "native_replay_max_absolute_logit_error": replay_error,
              "raw": records, "screen": screen, "evaluated_splits": ["FRESH_BASIC"],
              "forbidden_splits_opened": [], "model_forwards": 10, "terminal": terminal}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "model_forwards": 10}, indent=2))


if __name__ == "__main__":
    main()
