#!/usr/bin/env python3
"""Three-forward exact readout decomposition of the L13H8 mu/delta residual path."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_direct_readout_fold as authority
import run_bracket_l13h8_mu_delta_residual_write_bank_factorial as parent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_mu_delta_direct_readout_fold_v1_result.json"


def softcap(raw, torch):
    return 30.0 * torch.tanh(raw / 30.0)


def projection(component, total, eps=1e-8):
    return float((component * total).sum()) / (float((total * total).sum()) + eps)


def capture_lm_head_input(model, forward):
    """Run one model forward and capture the exact normalized lm_head input."""
    captured = []

    def hook(_module, args):
        if len(args) != 1:
            raise RuntimeError("lm_head pre-hook expected exactly one positional input")
        captured.append(args[0].detach().clone())

    handle = model.lm_head.register_forward_pre_hook(hook)
    try:
        output = forward()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one lm_head call, observed {len(captured)}")
    return output, captured[0]


def solve_rms_scales(z_native, z_removed, removed_factor, torch, eps=1e-12):
    """Solve r_N*z_N-r_R*z_R=f and report its relative residual."""
    design = torch.stack((z_native, -z_removed), dim=-1).float()
    target = removed_factor.float()
    scales = torch.linalg.lstsq(design, target.unsqueeze(-1)).solution[:2, 0]
    reconstructed = design @ scales
    residual = reconstructed - target
    relative = float(residual.norm() / (target.norm() + eps))
    return scales[0], scales[1], float(residual.abs().max()), relative


def readout_components(weight, bias, z_native, z_removed, removed_factor,
                       r_native, r_removed, torch, F):
    """Contract the exact residual factor and RMS-scale correction through W_U."""
    raw_native = F.linear(z_native.float(), weight.float(),
                          None if bias is None else bias.float())
    raw_removed = F.linear(z_removed.float(), weight.float(),
                           None if bias is None else bias.float())
    direct = F.linear((removed_factor.float() / r_native).unsqueeze(0),
                      weight.float(), None).squeeze(0)
    norm = F.linear(((r_removed / r_native - 1.0) * z_removed.float()).unsqueeze(0),
                    weight.float(), None).squeeze(0)
    raw_difference = raw_native - raw_removed
    final_difference = softcap(raw_native, torch) - softcap(raw_removed, torch)
    softcap_correction = final_difference - raw_difference
    return {
        "raw_native": raw_native,
        "raw_removed": raw_removed,
        "direct": direct,
        "normalization": norm,
        "softcap": softcap_correction,
        "raw_difference": raw_difference,
        "final_difference": final_difference,
    }


def propagated_factor(model, factor, torch):
    """Carry a block-13 attention-write difference through later residual lambdas."""
    answer = factor.float()
    for site in range(authority.PATCH_LAYER + 1, len(model.transformer.h)):
        answer = model.transformer.h[site].lambdas[0].float() * answer
    return answer


def evaluate(model, torch, F, facade):
    rows, device = authority.ROWS, next(model.parameters()).device
    tokens, finals, sources = parent.base.parent.pad_rows(rows, torch, device)
    arange = torch.arange(len(rows), device=device)

    native_pair, z_native_all = capture_lm_head_input(
        model,
        lambda: parent.bank_factor_forward(
            model, tokens, finals, sources, torch, F, facade, capture_bank=True),
    )
    native_logits, native_capture = native_pair
    native_bank = {name: native_capture[name] for name in authority.WRITE_BANK}
    factors = native_capture["factors"]
    terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
    by_group = defaultdict(dict)
    for index, row in enumerate(rows):
        by_group[row["group_id"]][row["delimiter_index"]] = index
    mu = torch.empty_like(terms)
    for index, row in enumerate(rows):
        mu[index] = sum(terms[by_group[row["group_id"]][j]] for j in range(3)) / 3.0
    removed_factors = {"mu": mu, "delta": terms - mu}
    replacements = {"mu": terms - mu, "delta": mu}

    removed = {}
    for factor in ("mu", "delta"):
        pair, z_all = capture_lm_head_input(
            model,
            lambda factor=factor: parent.bank_factor_forward(
                model, tokens, finals, sources, torch, F, facade,
                replacement_terms=replacements[factor], install_bank=native_bank),
        )
        removed[factor] = (pair[0], z_all)

    z_native = z_native_all[arange, finals]
    weight, bias = model.lm_head.weight.detach(), model.lm_head.bias
    records = []
    for index, row in enumerate(rows):
        q, answer = int(finals[index]), row["answer_id"]
        native_exact = native_logits[index, q]
        native_ce = float(F.cross_entropy(
            native_exact.unsqueeze(0), torch.tensor([answer], device=device)))
        for factor in ("mu", "delta"):
            removed_logits, z_removed_all = removed[factor]
            removed_exact = removed_logits[index, q]
            z_r = z_removed_all[index, q]
            carried = propagated_factor(model, removed_factors[factor][index], torch)
            r_n, r_r, fit_max, fit_relative = solve_rms_scales(
                z_native[index], z_r, carried, torch)
            components = readout_components(
                weight, bias, z_native[index], z_r, carried, r_n, r_r, torch, F)
            raw_identity = float((components["direct"] + components["normalization"] -
                                  components["raw_difference"]).abs().max())
            final_identity = float((components["direct"] + components["normalization"] +
                                    components["softcap"] -
                                    components["final_difference"]).abs().max())
            native_replay = float((softcap(components["raw_native"], torch) -
                                   native_exact).abs().max())
            removed_replay = float((softcap(components["raw_removed"], torch) -
                                    removed_exact).abs().max())
            selected = list(parent.base.CLOSERS)
            centered = {}
            for name in ("direct", "normalization", "softcap", "final_difference"):
                vector = components[name][selected]
                centered[name] = vector - vector.mean()
            total = centered["final_difference"]
            removed_ce = float(F.cross_entropy(
                removed_exact.unsqueeze(0), torch.tensor([answer], device=device)))
            records.append({
                "row_id": row["row_id"], "group_id": row["group_id"],
                "family_id": row["family_id"], "role": row["role"],
                "delimiter_index": row["delimiter_index"], "factor": factor,
                "native_centered_correct_closer": float(
                    parent.base.centered_closer(native_exact, torch)[row["delimiter_index"]]),
                "r_native": float(r_n), "r_removed": float(r_r),
                "rms_scale_fit_max_absolute_residual": fit_max,
                "rms_scale_fit_relative_residual": fit_relative,
                "raw_logit_identity_max_absolute_error": raw_identity,
                "final_logit_identity_max_absolute_error": final_identity,
                "native_softcap_output_replay_max_absolute_error": native_replay,
                "removed_softcap_output_replay_max_absolute_error": removed_replay,
                "centered_final_effect_norm": float(total.norm()),
                "centered_direct_folded_factor_norm": float(centered["direct"].norm()),
                "centered_normalization_scale_correction_norm": float(centered["normalization"].norm()),
                "centered_softcap_correction_norm": float(centered["softcap"].norm()),
                "direct_folded_factor_projection": projection(centered["direct"], total),
                "normalization_scale_correction_projection": projection(centered["normalization"], total),
                "softcap_correction_projection": projection(centered["softcap"], total),
                "native_correct_answer_ce": native_ce,
                "removed_correct_answer_ce": removed_ce,
                "signed_correct_answer_ce_damage": removed_ce - native_ce,
            })
    return records


def score(records):
    bars = authority.compile_plan()["bars"]
    cells, native = defaultdict(list), defaultdict(dict)
    for row in records:
        cells[(row["family_id"], row["factor"])].append(row)
        native[row["family_id"]][row["row_id"]] = row["native_centered_correct_closer"]
    reports = {}
    for (family, factor), rows in sorted(cells.items()):
        reports[f"{family}|{factor}"] = {
            "n": len(rows),
            "median_centered_final_effect_norm": statistics.median(
                r["centered_final_effect_norm"] for r in rows),
            "median_direct_folded_factor_projection": statistics.median(
                r["direct_folded_factor_projection"] for r in rows),
            "median_normalization_scale_correction_projection": statistics.median(
                r["normalization_scale_correction_projection"] for r in rows),
            "median_softcap_correction_projection": statistics.median(
                r["softcap_correction_projection"] for r in rows),
            "median_absolute_normalization_scale_correction_projection": statistics.median(
                abs(r["normalization_scale_correction_projection"]) for r in rows),
            "median_absolute_softcap_correction_projection": statistics.median(
                abs(r["softcap_correction_projection"]) for r in rows),
            "median_signed_correct_answer_ce_damage": statistics.median(
                r["signed_correct_answer_ce_damage"] for r in rows),
        }
    native_fraction = {
        family: sum(value > 0 for value in values.values()) / len(values)
        for family, values in native.items()
    }
    checks = {
        "native_capability": min(native_fraction.values()) >=
                             bars["native_positive_fraction_each_family_min"],
        "live_factor_effects": all(
            report["median_centered_final_effect_norm"] >=
            bars["median_live_centered_effect_norm_each_factor_family_min"]
            for report in reports.values()),
        "rms_scale_fit": max(r["rms_scale_fit_relative_residual"] for r in records) <=
                         bars["rms_scale_fit_relative_residual_max"],
        "raw_logit_identity": max(r["raw_logit_identity_max_absolute_error"] for r in records) <=
                              bars["raw_logit_identity_max_absolute_error"],
        "softcap_output_replay": max(
            max(r["native_softcap_output_replay_max_absolute_error"],
                r["removed_softcap_output_replay_max_absolute_error"]) for r in records) <=
            bars["softcap_output_replay_max_absolute_error"],
        "final_logit_identity": max(r["final_logit_identity_max_absolute_error"] for r in records) <=
                                bars["final_logit_identity_max_absolute_error"],
    }
    instrument_live = all(checks.values())
    targets = [reports[f"{family}|{factor}"] for family in authority.TARGET_FAMILIES
               for factor in ("mu", "delta")]
    direct_dominated = instrument_live and all(
        report["median_direct_folded_factor_projection"] >=
        bars["median_direct_projection_each_factor_target_family_min"]
        for report in targets)
    alternative_material = instrument_live and any(
        max(report["median_absolute_normalization_scale_correction_projection"],
            report["median_absolute_softcap_correction_projection"]) >=
        bars["median_absolute_normalization_or_softcap_projection_target_family_min"]
        for report in targets)
    return {
        "instrument_checks": checks,
        "instrument_live": instrument_live,
        "native_positive_fraction_by_family": native_fraction,
        "family_factor_reports": reports,
        "stability_rewrites_reported_separately": list(authority.STABILITY_FAMILIES),
        "direct_folded_factor_dominated": direct_dominated,
        "normalization_or_softcap_material": alternative_material,
        "predictions": {
            "pred_a": instrument_live,
            "pred_b": direct_dominated,
            "pred_c": instrument_live and (not direct_dominated) and alternative_material,
        },
    }


def main():
    plan = authority.compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    parent.base.parent.shared.candidate = authority
    torch, F, facade = parent.base.parent.shared._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        records = evaluate(model, torch, F, facade)
    screen = score(records)
    terminal = "invalid" if not screen["instrument_live"] else "screen"
    result = {
        "schema": "bracket_l13h8_mu_delta_direct_readout_fold_result_v1",
        "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "raw": records,
        "screen": screen,
        "evaluated_splits": ["FRESH_BASIC"],
        "forbidden_splits_opened": [],
        "model_forwards": 3,
        "terminal": terminal,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal,
                      "model_forwards": 3}, indent=2))


if __name__ == "__main__":
    main()
