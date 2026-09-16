#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_native_capability pred_c_fresh_response_replay pred_d_fresh_causal_installation pred_e_fresh_causal_removal pred_f_controls_and_random_null
"""Fresh causal confirmation of the frozen head17.2 QK2/value corner."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]

import torch

import circuit_fast_screen_managed_runner as managed
import run_setting2_regional_attention17h2_factor_interaction_fold_v1 as fold
from regional_cue_row_check_v1 import validate


RUNNER = Path(__file__).resolve()
PREREG = P / "SETTING2_REGIONAL_ATTENTION17H2_FACTOR_CORNER_FRESH_V1_PREREGISTRATION.md"
ROWS = P / "SETTING2_REGIONAL_ATTENTION17H2_FACTOR_CORNER_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_FACTOR_CORNER_FRESH_V1_BINDING.json"
DISCOVERY = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_factor_interaction_fold_v1_result.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_factor_corner_fresh_v1_result.json"
PRICE = {
    "physical_model_executions": 12,
    "full_model_sequences": 96,
    "factor_corners": 3,
    "equal_norm_random_nulls": 8,
    "additional_suffix_evaluations": 20,
    "additional_suffix_sequences": 960,
    "fits": 0,
    "backwards": 0,
    "parameter_updates": 0,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(actual, expected):
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


def cosine(actual, expected):
    return float((actual * expected).sum() / (actual.norm() * expected.norm()).clamp_min(1e-30))


def rms(x):
    return float(x.square().mean().sqrt())


def load_bound():
    binding = json.loads(BINDING.read_text())
    files = {
        "preregistration": PREREG,
        "rows": ROWS,
        "row_check": P / "regional_cue_row_check_v1.py",
        "discovery_result": DISCOVERY,
        "factor_fold_runner": Path(fold.__file__).resolve(),
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    discovery = json.loads(DISCOVERY.read_text())
    if discovery["terminal"] != "head17_2_sparse_factor_candidate" or not all(discovery["predictions"].values()) or discovery["selected_support_terms"] != ["qk2", "value", "qk2_x_value"]:
        raise ValueError("frozen discovery support changed")
    rows_doc = json.loads(ROWS.read_text())
    rows = rows_doc["rows"]
    checks = validate(rows)
    buckets = {}
    for index, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(index)
    executions = 2 * sum(math.ceil(len(indices) / 8) for indices in buckets.values())
    if len(rows) != 48 or rows_doc["prior_context_overlap"] != 0 or executions != PRICE["physical_model_executions"]:
        raise ValueError("row or execution authority changed")
    return binding, rows, checks, buckets


def plan():
    _, rows, checks, _ = load_bound()
    return {
        "schema": "setting2_regional_attention17h2_factor_corner_fresh_v1_plan",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "rows": len(rows),
        "frozen_corner": {"qk1": "native", "qk2": "edited", "value": "edited"},
        "price": PRICE,
        "row_checks": checks,
        "binding_sha256": sha(BINDING),
    }


@torch.no_grad()
def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast

    model = load_model_fast().cuda().eval()
    _, rows, checks, buckets = load_bound()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    lambdas = torch.stack([block.lambdas.detach().double().cpu() for block in model.transformer.h])
    embed_c, write_c = fold.helper.coefficients(lambdas)
    unembed = model.state_dict()["lm_head.weight"].double().cpu()
    native_pre = torch.empty(48, 1152, dtype=torch.float64)
    edited_pre = torch.empty_like(native_pre)
    full_delta = torch.empty_like(native_pre)
    selected_delta = torch.empty_like(native_pre)
    native_logits = torch.empty(48, 50304, dtype=torch.float64)
    edited_logits = torch.empty_like(native_logits)
    numerators = {name: 0.0 for name in ("carry", "attention9", "attention17", "head2", "corners", "selected_expansion")}
    denominators = {name: 0.0 for name in numerators}
    executions = 0
    for _, indices in sorted(buckets.items()):
        for offset in range(0, len(indices), 8):
            selected_rows = indices[offset:offset + 8]
            tokens = torch.tensor([rows[index]["ids"] for index in selected_rows], device=device)
            native = fold.run_to_head17(model, tokens, False, embed_c, write_c)
            edited = fold.run_to_head17(model, tokens, True, embed_c, write_c)
            executions += 2
            for run in (native, edited):
                numerators["carry"] += run["carry_num"]
                denominators["carry"] += run["carry_den"]
                numerators["attention9"] += run["attention9_num"]
                denominators["attention9"] += run["attention9_den"]
                numerators["attention17"] += run["attention17_num"]
                denominators["attention17"] += run["attention17_den"]
                numerators["head2"] += run["head2_num"]
                denominators["head2"] += run["head2_den"]
            output_weight = model.transformer.h[17].attn.c_proj.weight[:, 2 * 128:3 * 128]
            corner_native = fold.head_write(native["score1"], native["score2"], native["value"], output_weight)[:, -1].double()
            corner_selected = fold.head_write(native["score1"], edited["score2"], edited["value"], output_weight)[:, -1].double()
            corner_full = fold.head_write(edited["score1"], edited["score2"], edited["value"], output_weight)[:, -1].double()
            delta_score2 = edited["score2"] - native["score2"]
            delta_value = edited["value"] - native["value"]
            expanded_selected = (
                fold.head_write(native["score1"], delta_score2, native["value"], output_weight)
                + fold.head_write(native["score1"], native["score2"], delta_value, output_weight)
                + fold.head_write(native["score1"], delta_score2, delta_value, output_weight)
            )[:, -1].double()
            numerators["corners"] += float((corner_native - native["head2"].double()).square().sum() + (corner_full - edited["head2"].double()).square().sum())
            denominators["corners"] += float(native["head2"].double().square().sum() + edited["head2"].double().square().sum())
            numerators["selected_expansion"] += float((expanded_selected - (corner_selected - corner_native)).square().sum())
            denominators["selected_expansion"] += float((corner_selected - corner_native).square().sum())
            native_pre[selected_rows] = native["pre"].double().cpu()
            edited_pre[selected_rows] = edited["pre"].double().cpu()
            full_delta[selected_rows] = (corner_full - corner_native).cpu()
            selected_delta[selected_rows] = (corner_selected - corner_native).cpu()
            native_logits[selected_rows] = native["logits"].double().cpu()
            edited_logits[selected_rows] = edited["logits"].double().cpu()

    readers = torch.stack([unembed[row["uk_id"]] - unembed[row["us_id"]] for row in rows])
    full_response = (full_delta * readers).sum(-1)
    selected_response = (selected_delta * readers).sum(-1)
    paired_full_response = full_response[::2] - full_response[1::2]
    paired_selected_response = selected_response[::2] - selected_response[1::2]
    response_error = rel(paired_selected_response, paired_full_response)
    response_cosine = cosine(paired_selected_response, paired_full_response)

    suffix_inputs = (native_pre + full_delta, native_pre + selected_delta, edited_pre - full_delta, edited_pre - selected_delta)
    suffix_logits = [fold.suffix_logits(model, tensor.to(device=device, dtype=model_dtype)).double().cpu() for tensor in suffix_inputs]
    values = torch.stack([
        fold.score_readouts(native_logits, rows),
        fold.score_readouts(suffix_logits[0], rows),
        fold.score_readouts(suffix_logits[1], rows),
        fold.score_readouts(edited_logits, rows),
        fold.score_readouts(suffix_logits[2], rows),
        fold.score_readouts(suffix_logits[3], rows),
    ])
    install_effect = values[1:3] - values[0]
    removal_effect = values[4:6] - values[3]
    native_pair = values[0, ::2, 0] - values[0, 1::2, 0]
    full_install_all = install_effect[0, ::2, 0] - install_effect[0, 1::2, 0]
    selected_install_all = install_effect[1, ::2, 0] - install_effect[1, 1::2, 0]
    full_remove_all = removal_effect[0, ::2, 0] - removal_effect[0, 1::2, 0]
    selected_remove_all = removal_effect[1, ::2, 0] - removal_effect[1, 1::2, 0]

    generator = torch.Generator(device="cpu").manual_seed(202609160338)
    output_weight = model.transformer.h[17].attn.c_proj.weight[:, 2 * 128:3 * 128].double().cpu()
    random_errors = []
    for _ in range(PRICE["equal_norm_random_nulls"]):
        coefficients = torch.randn(48, 128, generator=generator, dtype=torch.float64)
        random_write = coefficients @ output_weight.T
        random_write *= (selected_delta.norm(dim=-1) / random_write.norm(dim=-1).clamp_min(1e-30))[:, None]
        random_install_logits = fold.suffix_logits(model, (native_pre + random_write).to(device=device, dtype=model_dtype)).double().cpu()
        random_remove_logits = fold.suffix_logits(model, (edited_pre - random_write).to(device=device, dtype=model_dtype)).double().cpu()
        random_install = fold.score_readouts(random_install_logits, rows) - values[0]
        random_remove = fold.score_readouts(random_remove_logits, rows) - values[3]
        ri = random_install[::2, 0] - random_install[1::2, 0]
        rr = random_remove[::2, 0] - random_remove[1::2, 0]
        random_errors.append({"installation_relative_l2": rel(ri, full_install_all), "removal_relative_l2": rel(rr, full_remove_all)})
    selected_install_error = rel(selected_install_all, full_install_all)
    selected_remove_error = rel(selected_remove_all, full_remove_all)
    random_pass = selected_install_error + .10 <= min(item["installation_relative_l2"] for item in random_errors) and selected_remove_error + .10 <= min(item["removal_relative_l2"] for item in random_errors)

    capability = response_pass = installation_pass = removal_pass = control_pass = True
    family_reports = {}
    for family in sorted({row["family"] for row in rows}):
        row_ids = [index for index, row in enumerate(rows) if row["family"] == family]
        pair_ids = [index // 2 for index in row_ids[::2]]
        native = native_pair[pair_ids]
        response_family_error = rel(paired_selected_response[pair_ids], paired_full_response[pair_ids])
        fi, si = full_install_all[pair_ids], selected_install_all[pair_ids]
        fr, sr = full_remove_all[pair_ids], selected_remove_all[pair_ids]
        install = {"relative_l2": rel(si, fi), "cosine": cosine(si, fi), "sign_agreement": float(((si * fi) > 0).double().mean()), "full_target_rms": rms(fi), "selected_target_rms": rms(si)}
        removal = {"relative_l2": rel(sr, fr), "cosine": cosine(sr, fr), "sign_agreement": float(((sr * fr) > 0).double().mean()), "full_target_rms": rms(fr), "selected_target_rms": rms(sr)}
        install_controls = {fold.READOUTS[index][0]: {"full_rms": rms(install_effect[0, row_ids, index]), "selected_rms": rms(install_effect[1, row_ids, index])} for index in range(1, len(fold.READOUTS))}
        removal_controls = {fold.READOUTS[index][0]: {"full_rms": rms(removal_effect[0, row_ids, index]), "selected_rms": rms(removal_effect[1, row_ids, index])} for index in range(1, len(fold.READOUTS))}
        family_capability = int((native > 0).sum()) >= 9 and rms(native) >= .01
        family_response = response_family_error <= .35
        family_install = install["relative_l2"] <= .35 and install["cosine"] >= .90 and install["sign_agreement"] >= .80
        family_remove = removal["relative_l2"] <= .35 and removal["cosine"] >= .90 and removal["sign_agreement"] >= .80
        family_controls = all(value["selected_rms"] <= 1.25 * value["full_rms"] + 1e-6 for value in (*install_controls.values(), *removal_controls.values()))
        capability = capability and family_capability
        response_pass = response_pass and family_response
        installation_pass = installation_pass and family_install
        removal_pass = removal_pass and family_remove
        control_pass = control_pass and family_controls
        family_reports[str(family)] = {
            "native_positive_pairs": int((native > 0).sum()),
            "native_paired_target_rms": rms(native),
            "response_relative_l2": response_family_error,
            "installation": install,
            "removal": removal,
            "installation_controls": install_controls,
            "removal_controls": removal_controls,
            "passes_capability": family_capability,
            "passes_response": family_response,
            "passes_installation": family_install,
            "passes_removal": family_remove,
            "passes_controls": family_controls,
        }

    errors = {name + "_relative_error": math.sqrt(numerators[name] / max(denominators[name], 1e-30)) for name in numerators}
    instrument = executions == PRICE["physical_model_executions"] and max(errors.values()) <= 2e-6 and bool(torch.isfinite(values).all())
    predictions = {
        "pred_a_exact_instrument": bool(instrument),
        "pred_b_native_capability": bool(instrument and capability),
        "pred_c_fresh_response_replay": bool(instrument and capability and response_error <= .25 and response_cosine >= .90 and response_pass),
        "pred_d_fresh_causal_installation": bool(instrument and capability and installation_pass),
        "pred_e_fresh_causal_removal": bool(instrument and capability and removal_pass),
        "pred_f_controls_and_random_null": bool(instrument and capability and control_pass and random_pass),
    }
    terminal = "fresh_head17_2_factor_corner" if all(predictions.values()) else "valid_head17_2_factor_corner_null" if instrument and capability else "invalid"
    result = {
        "schema": "setting2_regional_attention17h2_factor_corner_fresh_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "instrument_errors": errors,
        "frozen_corner": {"qk1": "native", "qk2": "edited", "value": "edited"},
        "response_relative_l2": response_error,
        "response_cosine": response_cosine,
        "selected_aggregate_installation_relative_l2": selected_install_error,
        "selected_aggregate_removal_relative_l2": selected_remove_error,
        "family_reports": family_reports,
        "equal_norm_random_null_errors": random_errors,
        "price": PRICE,
        "row_checks": checks,
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Fresh controlled transfer of the frozen head17.2 native-QK1/edited-QK2-and-value corner under response replay, causal installation/removal, unrelated-reader, and same-head equal-norm random controls; live factor ports remain external.",
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "errors": errors, "response_error": response_error, "response_cosine": response_cosine, "installation_error": selected_install_error, "removal_error": selected_remove_error, "families": family_reports, "random_nulls": random_errors}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
