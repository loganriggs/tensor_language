#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_oriented_decomposition pred_b_fit_candidate_and_active_nulls pred_c_select_replication pred_d_oriented_interaction_compression
"""Split the numeric cached-payload downstream MLP cross response by orientation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time


RUNNER = Path(__file__).resolve()
OPS = RUNNER.parent
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
PREREG = POLY / "NUMERIC_DOWNSTREAM_ORIENTED_BILINEAR_V1_PREREGISTRATION.md"
BINDING = POLY / "NUMERIC_DOWNSTREAM_ORIENTED_BILINEAR_V1_BINDING.json"
OUT = POLY / "NUMERIC_DOWNSTREAM_ORIENTED_BILINEAR_V1_RESULT.json"
ROWS = ROOT / "basis_aligned/bilinear_quotient/numbered_list_cached_value_downstream_use_rows_rung582.json"
RECEIPT = ROOT / "basis_aligned/bilinear_quotient/numbered_list_cached_value_downstream_use_rows_rung582_receipt.json"
R590 = ROOT / "basis_aligned/bilinear_quotient/numbered_list_cached_value_downstream_use_rung590_results.json"
COMPONENTS = (
    "left_delta_right_background",
    "left_background_right_delta",
    "contrast_self",
    "joint_response",
)
SITES = (8, 10, 12, 14)
NULLS = ("different_group_same_cell", "same_source_other_action")
PRICE = {"maximum_forwards": 632, "backwards": 0, "fits": 0}
PREDICTION_REGISTRY = {
    "pred_a_exact_oriented_decomposition": None,
    "pred_b_fit_candidate_and_active_nulls": None,
    "pred_c_select_replication": None,
    "pred_d_oriented_interaction_compression": None,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bound() -> None:
    binding = json.loads(BINDING.read_text())
    paths = {"rows": ROWS, "receipt": RECEIPT, "r590_result": R590, "preregistration": PREREG}
    if binding["files"] != {name: digest(path) for name, path in paths.items()}:
        raise RuntimeError("oriented-decomposition authority binding changed")
    if binding["components"] != list(COMPONENTS) or binding["sites"] != list(SITES) or binding["price"] != PRICE:
        raise RuntimeError("oriented-decomposition design binding changed")
    old = json.loads(R590.read_text())
    if old.get("decision") != "downstream_use_decomposition_null" or old.get("evaluated_splits") != ["FIT"]:
        raise RuntimeError("R590 no longer certifies unopened SELECT authority")


def plan() -> dict:
    load_bound()
    return {
        "schema": "numeric_downstream_oriented_bilinear_v1_plan",
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "sites": list(SITES), "components": list(COMPONENTS), "nulls": list(NULLS),
        "selection_split": "FIT", "conditionally_opened_split": "SELECT",
        "forbidden_splits": ["FINAL_TEST", "OOD"], "price": PRICE,
        "predicates": list(PREDICTION_REGISTRY),
    }


def main() -> None:
    load_bound()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan(), indent=2, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    sys.path.insert(0, str(OPS))
    import torch
    import torch.nn.functional as F
    import numbered_list_cached_value_downstream_use_rung584 as engine
    from circuit_fast_screen_managed_runner import atomic_create_json

    torch.set_num_threads(2)
    rows = engine.load_authority()
    if any(row["split"] in {"FINAL_TEST", "OOD"} for row in []):
        raise RuntimeError("forbidden split dispatch")

    def oriented_response(mlp, state_without, state_with):
        delta = state_with - state_without
        l0 = F.linear(state_without, mlp.Left.weight.to(state_without.dtype))
        r0 = F.linear(state_without, mlp.Right.weight.to(state_without.dtype))
        ld = F.linear(delta, mlp.Left.weight.to(delta.dtype))
        rd = F.linear(delta, mlp.Right.weight.to(delta.dtype))
        left = F.linear(ld * r0, mlp.Down.weight.to(delta.dtype))
        right = F.linear(l0 * rd, mlp.Down.weight.to(delta.dtype))
        self_term = F.linear(ld * rd, mlp.Down.weight.to(delta.dtype))
        joint = left + right + self_term
        direct = F.linear(
            F.linear(state_with, mlp.Left.weight.to(state_with.dtype))
            * F.linear(state_with, mlp.Right.weight.to(state_with.dtype)) - l0 * r0,
            mlp.Down.weight.to(state_with.dtype),
        )
        error = float((joint.float() - direct.float()).square().sum()) / max(float(direct.float().square().sum()), 1e-30)
        return {COMPONENTS[0]: left, COMPONENTS[1]: right, COMPONENTS[2]: self_term,
                COMPONENTS[3]: joint, "direct_response": direct, "relative_squared_error": error}

    engine.COMPONENTS = COMPONENTS
    engine.SELECTION = tuple((site, component) for site in SITES for component in COMPONENTS)
    engine.torch_bilinear_response = oriented_response
    model, checkpoint = engine.facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    started = time.time(); calls = 0
    fit_cache, fit_capture, count, fit_exactness = engine.capture_split(model, rows, "FIT")
    calls += count
    if max(float(fit_exactness[k]) for k in (
        "native_end_to_end_smoke_relative_squared_error", "native_replay_relative_squared_error",
        "head_source_sum_relative_squared_error", "value_split_relative_squared_error",
        "cached_bus_relative_squared_error", "projected_term_relative_squared_error",
        "bilinear_response_relative_squared_error")) > 1e-10:
        raise RuntimeError("FIT exactness gate failed before intervention interpretation")
    fit_reports = {}; fit_raw = {}; provisional = None
    for site in SITES:
        for component in COMPONENTS:
            name = f"mlp{site}_{component}"
            fit_raw[name], count = engine.evaluate_component(model, rows, "FIT", fit_cache, site, component)
            calls += count
            fit_reports[name] = engine.score_candidate(fit_raw[name], cell_prefix=f"oriented:FIT:{name}", authority_rows=rows)
            if provisional is None and fit_reports[name]["passed_without_nulls"]:
                provisional = name
    selected = None; fit_null_reports = {}
    if provisional is not None:
        site = int(provisional[3:].split("_", 1)[0]); component = provisional[3:].split("_", 1)[1]
        for null_name, donor_map in engine.r582.deterministic_null_maps(rows, "FIT").items():
            raw, count = engine.evaluate_component(model, rows, "FIT", fit_cache, site, component, donor_map, null_name)
            calls += count
            fit_null_reports[null_name] = engine.score_null(
                fit_raw[provisional], raw, cell_prefix=f"oriented:FIT:{provisional}:{null_name}",
                real_report=fit_reports[provisional], null_name=null_name, authority_rows=rows)
        if set(fit_null_reports) == set(NULLS) and all(x["passed"] for x in fit_null_reports.values()):
            selected = provisional
    select_exactness = None; select_reports = {}; select_null_reports = {}; opened = ["FIT"]
    if selected is not None:
        opened.append("SELECT")
        select_cache, _capture, count, select_exactness = engine.capture_split(model, rows, "SELECT"); calls += count
        if max(float(select_exactness[k]) for k in (
            "native_end_to_end_smoke_relative_squared_error", "native_replay_relative_squared_error",
            "head_source_sum_relative_squared_error", "value_split_relative_squared_error",
            "cached_bus_relative_squared_error", "projected_term_relative_squared_error",
            "bilinear_response_relative_squared_error")) > 1e-10:
            raise RuntimeError("SELECT exactness gate failed before intervention interpretation")
        site = int(selected[3:].split("_", 1)[0]); selected_component = selected[3:].split("_", 1)[1]
        select_raw = {}
        for component in COMPONENTS:
            name = f"mlp{site}_{component}"
            select_raw[name], count = engine.evaluate_component(model, rows, "SELECT", select_cache, site, component); calls += count
            select_reports[name] = engine.score_candidate(
                select_raw[name], cell_prefix=f"oriented:SELECT:{name}",
                frozen_scales=fit_reports[selected]["fit_scales"], authority_rows=rows)
        for null_name, donor_map in engine.r582.deterministic_null_maps(rows, "SELECT").items():
            raw, count = engine.evaluate_component(model, rows, "SELECT", select_cache, site, selected_component, donor_map, null_name); calls += count
            select_null_reports[null_name] = engine.score_null(
                select_raw[selected], raw, cell_prefix=f"oriented:SELECT:{selected}:{null_name}",
                real_report=select_reports[selected], null_name=null_name, authority_rows=rows)
    pred_a = bool(max(float(fit_exactness[k]) for k in fit_exactness) <= 1e-10 and (select_exactness is None or max(float(select_exactness[k]) for k in select_exactness) <= 1e-10))
    pred_b = bool(selected is not None)
    pred_c = bool(selected and select_reports[selected]["passed_without_nulls"] and all(x["passed"] for x in select_null_reports.values()))
    pred_d = bool(pred_c and selected.split("_", 2)[2] in COMPONENTS[:2])
    predictions = dict(zip(PREDICTION_REGISTRY, (pred_a, pred_b, pred_c, pred_d), strict=True))
    terminal = "oriented_interaction_compression" if pred_d else "reference_component_only" if pred_c else "fit_null" if not pred_b else "select_null"
    result = {"schema": "numeric_downstream_oriented_bilinear_v1_result", "terminal": terminal,
              "predictions": predictions, "provisional": provisional, "selected": selected,
              "opened_splits": opened, "forbidden_splits_opened": [], "fit_exactness": fit_exactness,
              "select_exactness": select_exactness, "fit_reports": fit_reports,
              "fit_null_reports": fit_null_reports, "select_reports": select_reports,
              "select_null_reports": select_null_reports, "price": {**PRICE, "observed_forwards": calls},
              "model_backwards": 0, "model_weights_updated": False, "fits": 0,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "elapsed_seconds": time.time() - started, "runner_sha256": digest(RUNNER)}
    atomic_create_json(OUT, result); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
