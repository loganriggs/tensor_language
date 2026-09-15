#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 55forwards440seq; successor behavioral interaction scalar;1fit;0updates;noquantization.
"""Fit one suppressive interaction scalar and confirm it on prefixed length-six rows."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np


RUNNER = Path(__file__).resolve()
OPS = RUNNER.parent
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
FIT_ROWS = POLY / "SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_ROWS.json"
HOLD_ROWS = POLY / "SUCCESSOR_POINTER_PREFIXED_LENGTH6_V1_ROWS.json"
PREREG = POLY / "SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V1_PREREGISTRATION.md"
INTERACTION = POLY / "SUCCESSOR_POINTER_CROSS_TYPE_INTERACTION_V1_RESULT.json"
CONFIRMATION = POLY / "SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_RESULT.json"
BINDING = POLY / "SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V1_BINDING.json"
OUT = POLY / "SUCCESSOR_POINTER_BEHAVIORAL_INTERACTION_SCALAR_V1_RESULT.json"
REQUESTED_DRY = bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
CONDITIONS = ("coherent", "early_swap_control", "late_swap_incoherent")
FAMILIES = ("month", "digit")
PREFIXES = ("sequence", "pattern", "continue", "items")
MODULES = tuple(name for layer in range(18) for name in (f"attn{layer}", f"mlp{layer}"))
GROUPS = {
    "A": tuple(f"attn{x}" for x in range(8, 13)),
    "M": tuple(f"mlp{x}" for x in range(8, 13)),
    "AM": tuple(name for x in range(8, 13) for name in (f"attn{x}", f"mlp{x}")),
}
BARS = {
    "maximum_self_logit_absolute_error": 1e-5,
    "maximum_full_ceiling_logit_absolute_error": 1e-4,
    "minimum_native_accuracy": 0.75,
    "minimum_family_mean_late_backward_target": 1.0,
    "minimum_parent_projection": 0.50,
    "minimum_parent_cosine": 0.70,
    "maximum_control_rms_fraction": 0.50,
    "minimum_prediction_cosine": 0.90,
    "maximum_prediction_relative_l2": 0.35,
    "minimum_prediction_sign": 0.80,
    "minimum_prediction_norm_ratio": 0.65,
    "maximum_prediction_norm_ratio": 1.35,
}
PRICE = {"forwards": 55, "sequences": 440, "fits": 1, "backwards": 0, "updates": 0}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {
        "fit_rows": FIT_ROWS,
        "holdout_rows": HOLD_ROWS,
        "preregistration": PREREG,
        "interaction_result": INTERACTION,
        "confirmation_result": CONFIRMATION,
    }
    for name, expected in binding["files"].items():
        if digest(paths[name]) != expected:
            raise RuntimeError(f"bound {name} changed")
    fit, hold = json.loads(FIT_ROWS.read_text()), json.loads(HOLD_ROWS.read_text())
    if binding["fit_row_manifest_sha256"] != fit["row_manifest_sha256"]:
        raise RuntimeError("fit row manifest changed")
    if binding["holdout_row_manifest_sha256"] != hold["row_manifest_sha256"]:
        raise RuntimeError("holdout row manifest changed")
    if binding["bars"] != BARS or binding["price"] != PRICE:
        raise RuntimeError("bars or price changed")
    if binding["groups"] != {name: list(members) for name, members in GROUPS.items()}:
        raise RuntimeError("groups changed")
    return binding, fit, hold


def aligned_rows(frozen, prefix_id=None):
    selected = [row for row in frozen["rows"] if prefix_id is None or row.get("prefix_id") == prefix_id]
    by_key = {}
    for row in selected:
        key = (row["family"], int(row["final_index"]))
        by_key.setdefault(key, {})[row["condition"]] = row
    keys = sorted(by_key)
    if len(keys) != 8 or any(set(by_key[key]) != set(CONDITIONS) for key in keys):
        raise RuntimeError("incomplete aligned eight-group factorial")
    return keys, {condition: [by_key[key][condition] for key in keys] for condition in CONDITIONS}


def plan():
    _, fit, hold = load_bound()
    aligned_rows(fit)
    for prefix in PREFIXES:
        aligned_rows(hold, prefix)
    return {
        "schema": "successor_pointer_behavioral_interaction_scalar_v1_plan",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "fit_groups": 8,
        "holdout_groups": 32,
        "coefficient_count": 1,
        "law": "beta_times_additive_component_effect",
        "bars": BARS,
        "price": PRICE,
        "predicates": [
            "pred_a_exact_instrument_and_live_parent",
            "pred_b_suppressive_scalar_selected",
            "pred_c_prefixed_scalar_transfer",
            "pred_d_scalar_null",
        ],
    }


def margin(logits, rows, direction):
    return np.asarray([
        values[int(row["recipient_answer_id"])] - values[int(row["donors"][direction]["answer_id"])]
        for values, row in zip(logits, rows)
    ], dtype=np.float64)


def comparison(predicted, actual):
    pn, an = float(np.linalg.norm(predicted)), float(np.linalg.norm(actual))
    return {
        "count": len(actual),
        "cosine": float(np.dot(predicted, actual) / (pn * an)) if pn and an else 0.0,
        "relative_l2_error": float(np.linalg.norm(predicted - actual) / an) if an else float("inf"),
        "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))),
        "predicted_to_actual_norm_ratio": pn / an if an else float("inf"),
    }


def prediction_passes(metrics):
    return bool(
        metrics["cosine"] >= BARS["minimum_prediction_cosine"]
        and metrics["relative_l2_error"] <= BARS["maximum_prediction_relative_l2"]
        and metrics["sign_agreement"] >= BARS["minimum_prediction_sign"]
        and BARS["minimum_prediction_norm_ratio"] <= metrics["predicted_to_actual_norm_ratio"] <= BARS["maximum_prediction_norm_ratio"]
    )


def main():
    binding, fit_frozen, hold_frozen = load_bound()
    fit_keys, fit_rows = aligned_rows(fit_frozen)
    hold_panels = {prefix: aligned_rows(hold_frozen, prefix) for prefix in PREFIXES}
    if REQUESTED_DRY:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(600)
    sys.path.insert(0, str(ROOT / "basis_aligned/qk_mdl/algo_tasks/successor"))
    import torch
    import successor_lib as sl
    from circuit_fast_screen_managed_runner import atomic_create_json

    torch.set_num_threads(2)
    model, cfg = sl.load_model()
    device = next(model.parameters()).device
    counts = {"forwards": 0, "sequences": 0}

    def counted_run(tokens, **kwargs):
        counts["forwards"] += 1
        counts["sequences"] += len(tokens)
        return sl.run(model, cfg, tokens, **kwargs)

    def run_panel(keys, rows):
        query = int(rows["coherent"][0]["query_position"])
        if any(int(row["query_position"]) != query for condition in CONDITIONS for row in rows[condition]):
            raise RuntimeError("query mismatch within panel")
        tensors = {
            condition: torch.tensor([row["token_ids"] for row in rows[condition]], dtype=torch.long, device=device)
            for condition in CONDITIONS
        }
        native, caches = {}, {}
        for condition in CONDITIONS:
            values, cache = counted_run(tensors[condition], collect=True)
            native[condition] = values[:, query].float().cpu().numpy()
            caches[condition] = cache

        def patches(source, destination, selected=None):
            head, mlp = {}, {}
            chosen = set(MODULES if selected is None else selected)
            for layer in range(18):
                if f"attn{layer}" in chosen:
                    hybrid = caches[destination][("h", layer)].clone()
                    hybrid[:, query] = caches[source][("h", layer)][:, query]
                    for head_index in range(cfg["n_head"]):
                        head[(layer, head_index)] = hybrid[:, :, head_index]
                if f"mlp{layer}" in chosen:
                    hybrid = caches[destination][("m", layer)].clone()
                    hybrid[:, query] = caches[source][("m", layer)][:, query]
                    mlp[layer] = hybrid
            return head, mlp

        self_head = {(layer, h): caches["late_swap_incoherent"][("h", layer)][:, :, h] for layer in range(18) for h in range(cfg["n_head"])}
        self_mlp = {layer: caches["late_swap_incoherent"][("m", layer)] for layer in range(18)}
        self_values, _ = counted_run(tensors["late_swap_incoherent"], patch_head=self_head, patch_mlp=self_mlp)
        full_head, full_mlp = patches("coherent", "late_swap_incoherent")
        ceiling_values, _ = counted_run(tensors["late_swap_incoherent"], patch_head=full_head, patch_mlp=full_mlp)
        arms = {name: {} for name in GROUPS}
        for name, members in GROUPS.items():
            for destination in ("late_swap_incoherent", "early_swap_control"):
                head, mlp = patches("coherent", destination, members)
                values, _ = counted_run(tensors[destination], patch_head=head or None, patch_mlp=mlp or None)
                arms[name][destination] = values[:, query].float().cpu().numpy()
        return {
            "keys": keys,
            "rows": rows,
            "native": native,
            "arms": arms,
            "self_error": float(np.max(np.abs(self_values[:, query].float().cpu().numpy() - native["late_swap_incoherent"]))),
            "ceiling_error": float(np.max(np.abs(ceiling_values[:, query].float().cpu().numpy() - native["coherent"]))),
        }

    def effects(panel, destination, direction):
        rows, native, arms = panel["rows"], panel["native"], panel["arms"]
        baseline = margin(native[destination], rows[destination], direction)
        return {name: margin(arms[name][destination], rows[destination], direction) - baseline for name in GROUPS}

    def summarize_parent(panel):
        keys, rows, native = panel["keys"], panel["rows"], panel["native"]
        target = margin(native["coherent"], rows["coherent"], "backward") - margin(native["late_swap_incoherent"], rows["late_swap_incoherent"], "backward")
        late = effects(panel, "late_swap_incoherent", "backward")
        late_forward = effects(panel, "late_swap_incoherent", "forward")
        early = effects(panel, "early_swap_control", "backward")
        capability = {
            family: {
                condition: float(np.mean([
                    int(np.argmax(native[condition][i])) == int(rows[condition][i]["recipient_answer_id"])
                    for i, key in enumerate(keys) if key[0] == family
                ]))
                for condition in CONDITIONS
            }
            for family in FAMILIES
        }
        reports = {}
        for family in FAMILIES:
            indices = np.asarray([i for i, key in enumerate(keys) if key[0] == family])
            t, exact = target[indices], late["AM"][indices]
            target_norm2 = float(np.dot(t, t)); en, tn = float(np.linalg.norm(exact)), float(np.linalg.norm(t))
            target_rms = float(np.sqrt(np.mean(t ** 2)))
            report = {
                "n": len(indices),
                "mean_target": float(np.mean(t)),
                "projection": float(np.dot(exact, t) / target_norm2) if target_norm2 else 0.0,
                "cosine": float(np.dot(exact, t) / (en * tn)) if en and tn else 0.0,
                "late_forward_control_fraction": float(np.sqrt(np.mean(late_forward["AM"][indices] ** 2)) / target_rms),
                "early_backward_control_fraction": float(np.sqrt(np.mean(early["AM"][indices] ** 2)) / target_rms),
            }
            report["passes"] = bool(
                report["mean_target"] >= BARS["minimum_family_mean_late_backward_target"]
                and report["projection"] >= BARS["minimum_parent_projection"]
                and report["cosine"] >= BARS["minimum_parent_cosine"]
                and report["late_forward_control_fraction"] <= BARS["maximum_control_rms_fraction"]
                and report["early_backward_control_fraction"] <= BARS["maximum_control_rms_fraction"]
            )
            reports[family] = report
        instrument = bool(
            panel["self_error"] <= BARS["maximum_self_logit_absolute_error"]
            and panel["ceiling_error"] <= BARS["maximum_full_ceiling_logit_absolute_error"]
            and all(value >= BARS["minimum_native_accuracy"] for cells in capability.values() for value in cells.values())
            and all(report["passes"] for report in reports.values())
        )
        return {
            "instrument": instrument,
            "self_error": panel["self_error"],
            "ceiling_error": panel["ceiling_error"],
            "capability": capability,
            "families": reports,
            "target": target,
            "late": late,
            "late_forward": late_forward,
            "early": early,
        }

    def scalar_report(summary, beta, cell_keys):
        x = summary["late"]["A"] + summary["late"]["M"]
        y = summary["late"]["AM"]
        predicted = beta * x
        reports = {}
        for name, indices in cell_keys.items():
            ix = np.asarray(indices)
            metrics = comparison(predicted[ix], y[ix])
            target_rms = float(np.sqrt(np.mean(summary["target"][ix] ** 2)))
            pred_late_forward = beta * (summary["late_forward"]["A"][ix] + summary["late_forward"]["M"][ix])
            pred_early = beta * (summary["early"]["A"][ix] + summary["early"]["M"][ix])
            metrics["late_forward_control_fraction"] = float(np.sqrt(np.mean(pred_late_forward ** 2)) / target_rms)
            metrics["early_backward_control_fraction"] = float(np.sqrt(np.mean(pred_early ** 2)) / target_rms)
            metrics["passes"] = bool(
                prediction_passes(metrics)
                and metrics["late_forward_control_fraction"] <= BARS["maximum_control_rms_fraction"]
                and metrics["early_backward_control_fraction"] <= BARS["maximum_control_rms_fraction"]
            )
            reports[name] = metrics
        return {
            "reports": reports,
            "passes": all(report["passes"] for report in reports.values()),
            "exact_additive": x.tolist(),
            "exact_joint": y.tolist(),
            "exact_interaction": (y - x).tolist(),
            "prediction": predicted.tolist(),
        }

    started = time.perf_counter()
    fit_panel = run_panel(fit_keys, fit_rows)
    fit_summary = summarize_parent(fit_panel)
    fit_x = fit_summary["late"]["A"] + fit_summary["late"]["M"]
    fit_y = fit_summary["late"]["AM"]
    beta = float(np.dot(fit_x, fit_y) / np.dot(fit_x, fit_x))
    fit_cells = {"overall": list(range(8))}
    fit_cells.update({family: [i for i, key in enumerate(fit_keys) if key[0] == family] for family in FAMILIES})
    fit_scalar = scalar_report(fit_summary, beta, fit_cells)
    fit_pass = bool(fit_summary["instrument"] and 0.0 < beta < 1.0 and fit_scalar["passes"])

    hold_results = {}
    all_hold_predicted, all_hold_exact = [], []
    hold_instrument = True
    hold_pass = True
    for prefix in PREFIXES:
        keys, rows = hold_panels[prefix]
        panel = run_panel(keys, rows)
        summary = summarize_parent(panel)
        cells = {f"{prefix}|{family}": [i for i, key in enumerate(keys) if key[0] == family] for family in FAMILIES}
        scalar = scalar_report(summary, beta, cells)
        hold_results[prefix] = {
            "instrument": summary["instrument"],
            "self_error": summary["self_error"],
            "ceiling_error": summary["ceiling_error"],
            "capability": summary["capability"],
            "parent_families": summary["families"],
            "scalar": scalar,
        }
        hold_instrument &= summary["instrument"]
        hold_pass &= scalar["passes"]
        all_hold_predicted.extend(scalar["prediction"])
        all_hold_exact.extend(scalar["exact_joint"])

    all_hold_metrics = comparison(np.asarray(all_hold_predicted), np.asarray(all_hold_exact))
    observed_price = {**PRICE, "observed_forwards": counts["forwards"], "observed_sequences": counts["sequences"]}
    price_exact = counts == {"forwards": PRICE["forwards"], "sequences": PRICE["sequences"]}
    pred_a = bool(fit_summary["instrument"] and hold_instrument and price_exact)
    pred_b = bool(pred_a and fit_pass)
    pred_c = bool(pred_b and hold_pass and prediction_passes(all_hold_metrics))
    pred_d = bool(pred_a and not pred_c)
    terminal = "invalid_or_dead_parent" if not pred_a else ("behavioral_interaction_scalar_transfer" if pred_c else "behavioral_interaction_scalar_null")
    result = {
        "schema": "successor_pointer_behavioral_interaction_scalar_v1_result",
        "terminal": terminal,
        "predictions": {
            "pred_a_exact_instrument_and_live_parent": pred_a,
            "pred_b_suppressive_scalar_selected": pred_b,
            "pred_c_prefixed_scalar_transfer": pred_c,
            "pred_d_scalar_null": pred_d,
        },
        "coefficient": beta,
        "interaction_gain": beta - 1.0,
        "fit": {
            "instrument": fit_summary["instrument"],
            "parent_families": fit_summary["families"],
            "capability": fit_summary["capability"],
            "self_error": fit_summary["self_error"],
            "ceiling_error": fit_summary["ceiling_error"],
            "scalar": fit_scalar,
        },
        "holdout": hold_results,
        "holdout_overall": all_hold_metrics,
        "bars": BARS,
        "price": observed_price,
        "fit_row_manifest_sha256": fit_frozen["row_manifest_sha256"],
        "holdout_row_manifest_sha256": hold_frozen["row_manifest_sha256"],
        "binding_sha256": digest(BINDING),
        "runner_sha256": digest(RUNNER),
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "claim_boundary": "one scalar behavioral interaction adapter conditional on live attention and MLP component effects; component generators and full residual interaction remain external; no quantization",
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_create_json(OUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
