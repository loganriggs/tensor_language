#!/usr/bin/env python3
"""Six-forward all-three-head joint mediation follow-up."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_r549_attention_joint_mediation as authority
import run_bracket_l13h8_mu_delta_r549_attention_mediation as parent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_mu_delta_r549_attention_joint_mediation_v1_result.json"
INDIVIDUAL = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_mu_delta_r549_attention_mediation_v1_result.json"


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_individual_sums():
    if sha256_file(INDIVIDUAL) != authority.INDIVIDUAL_RESULT_SHA256:
        raise RuntimeError("frozen individual mediation result SHA256 mismatch")
    result = json.loads(INDIVIDUAL.read_text())
    sums = defaultdict(lambda: {"projection_recovery": 0.0, "ce_rescue": 0.0})
    counts = defaultdict(int)
    for row in result["raw"]:
        key = (row["row_id"], row["factor"])
        sums[key]["projection_recovery"] += row["projection_recovery"]
        sums[key]["ce_rescue"] += row["signed_correct_answer_ce_rescue"]
        counts[key] += 1
    if not sums or any(count != 3 for count in counts.values()):
        raise RuntimeError("frozen individual result does not contain exactly three heads per row/factor")
    return dict(sums)


def joint_attention_factor_forward(model, tokens, finals, sources, torch, F, facade, *,
                                   replacement_terms=None, restore_contributions=None,
                                   capture_heads=False):
    """Successor-local plural restoration; the completed parent remains byte-frozen."""
    captured = {}
    arange = torch.arange(tokens.size(0), device=tokens.device)
    head_by_layer = {layer: head for layer, head in authority.HEADS}

    def attention(event):
        if event.site == authority.PATCH_LAYER:
            write, factors = parent.base.parent.shared.replay_head(
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
        write, contribution = parent.replay_attention_with_head(
            event.state, event.first_value, event.block.attn, finals, site[1], torch, F)
        if capture_heads:
            captured[site] = contribution.detach().clone()
        if site in (restore_contributions or {}):
            write = write.clone()
            write[arange, finals] += (restore_contributions[site] - contribution).to(write.dtype)
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
    tokens, finals, sources = parent.base.parent.pad_rows(rows, torch, device)
    native = parent.base.parent.shared.native_logits(model, tokens, torch, F)
    replay, captured = joint_attention_factor_forward(
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
    removed, restored = {}, {}
    joint_native = {site: captured[site] for site in authority.HEADS}
    for factor, replacement in replacements.items():
        removed[factor] = joint_attention_factor_forward(
            model, tokens, finals, sources, torch, F, facade,
            replacement_terms=replacement)[0]
        restored[factor] = joint_attention_factor_forward(
            model, tokens, finals, sources, torch, F, facade,
            replacement_terms=replacement, restore_contributions=joint_native)[0]
    records = []
    for index, row in enumerate(rows):
        q, answer = int(finals[index]), row["answer_id"]
        native_vector = parent.base.centered_closer(replay[index, q], torch)
        native_ce = float(F.cross_entropy(
            replay[index, q].unsqueeze(0), torch.tensor([answer], device=device)))
        for factor in ("mu", "delta"):
            removed_vector = parent.base.centered_closer(removed[factor][index, q], torch)
            restored_vector = parent.base.centered_closer(restored[factor][index, q], torch)
            metrics = parent.base.vector_metrics(native_vector, removed_vector, restored_vector)
            removed_ce = float(F.cross_entropy(
                removed[factor][index, q].unsqueeze(0), torch.tensor([answer], device=device)))
            restored_ce = float(F.cross_entropy(
                restored[factor][index, q].unsqueeze(0), torch.tensor([answer], device=device)))
            ce_rescue = removed_ce - restored_ce
            records.append({
                "row_id": row["row_id"], "group_id": row["group_id"],
                "family_id": row["family_id"], "role": row["role"],
                "delimiter_index": row["delimiter_index"], "factor": factor,
                "native_centered_correct_closer": float(native_vector[row["delimiter_index"]]),
                **metrics, "signed_correct_answer_ce_change": removed_ce - native_ce,
                "signed_correct_answer_ce_rescue": ce_rescue,
            })
    return records, replay_error


def append_frozen_individual_comparison(records, individual_sums):
    """Append the SHA-bound comparison only after current joint outputs exist."""
    for row in records:
        frozen = individual_sums[(row["row_id"], row["factor"])]
        row["sum_frozen_individual_projection_recovery"] = frozen["projection_recovery"]
        row["joint_minus_sum_individual_projection_recovery"] = (
            row["projection_recovery"] - frozen["projection_recovery"])
        row["sum_frozen_individual_ce_rescue"] = frozen["ce_rescue"]
        row["joint_minus_sum_individual_ce_rescue"] = (
            row["signed_correct_answer_ce_rescue"] - frozen["ce_rescue"])


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
            "median_joint_projection_recovery": statistics.median(r["projection_recovery"] for r in rows),
            "median_joint_rescue_cosine": statistics.median(r["rescue_cosine"] for r in rows),
            "positive_joint_projection_fraction": sum(r["projection_recovery"] > 0 for r in rows) / len(rows),
            "median_signed_correct_answer_ce_change": statistics.median(
                r["signed_correct_answer_ce_change"] for r in rows),
            "median_signed_correct_answer_ce_rescue": statistics.median(
                r["signed_correct_answer_ce_rescue"] for r in rows),
            "median_joint_minus_sum_individual_projection_recovery": statistics.median(
                r["joint_minus_sum_individual_projection_recovery"] for r in rows),
            "median_joint_minus_sum_individual_ce_rescue": statistics.median(
                r["joint_minus_sum_individual_ce_rescue"] for r in rows),
        }
    native = {family: sum(value > 0 for value in values.values()) / len(values)
              for family, values in families.items()}
    instrument = {
        "native_capability": min(native.values()) >= bars["native_positive_fraction_each_family_min"],
        "native_replay": replay_error <= bars["native_replay_max_absolute_logit_error_max"],
        "live_factor_effects": all(
            reports[f"{family}|{factor}"]["median_total_effect_norm"] >=
            bars["median_live_centered_effect_norm_each_factor_family_min"]
            for family in authority.FAMILIES for factor in ("mu", "delta")),
    }
    def passes(family, factor):
        report = reports[f"{family}|{factor}"]
        return (report["median_joint_projection_recovery"] >=
                bars["median_joint_projection_recovery_each_factor_target_family_min"]
                and report["median_joint_rescue_cosine"] >=
                bars["median_joint_rescue_cosine_each_factor_target_family_min"]
                and report["positive_joint_projection_fraction"] >=
                bars["positive_joint_projection_fraction_each_factor_target_family_min"])
    instrument_live = all(instrument.values())
    joint = instrument_live and all(passes(family, factor)
                                    for family in authority.TARGET_FAMILIES
                                    for factor in ("mu", "delta"))
    return {"instrument_checks": instrument, "instrument_live": instrument_live,
            "native_positive_fraction_by_family": native,
            "family_factor_reports": reports,
            "stability_rewrites_reported_separately": list(authority.STABILITY_FAMILIES),
            "joint_attention_mediation_held": joint,
            "predictions": {"pred_a": instrument_live, "pred_b": joint,
                            "pred_c": instrument_live and not joint}}


def main():
    plan = authority.compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists(): raise RuntimeError(f"refusing to overwrite {OUT}")
    parent.base.parent.shared.candidate = authority
    torch, F, facade = parent.base.parent.shared._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        records, replay_error = evaluate(model, torch, F, facade)
    # Post-measurement descriptive comparison to the already-frozen, SHA-bound parent.
    append_frozen_individual_comparison(records, frozen_individual_sums())
    screen = score(records, replay_error)
    terminal = "invalid" if not screen["instrument_live"] else (
        "screen" if screen["joint_attention_mediation_held"] else "null")
    result = {"schema": "bracket_l13h8_mu_delta_r549_attention_joint_mediation_result_v1",
              "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "individual_result_sha256": authority.INDIVIDUAL_RESULT_SHA256,
              "native_replay_max_absolute_logit_error": replay_error,
              "raw": records, "screen": screen, "evaluated_splits": ["FRESH_BASIC"],
              "forbidden_splits_opened": [], "model_forwards": 6, "terminal": terminal}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "model_forwards": 6}, indent=2))


if __name__ == "__main__":
    main()
