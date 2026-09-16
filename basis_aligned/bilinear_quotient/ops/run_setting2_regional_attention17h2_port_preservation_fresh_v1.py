#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_fresh_response_replay pred_c_fresh_bidirectional_causality pred_d_fresh_preservation pred_e_beats_target_only pred_f_frozen_support_specificity
"""Fresh confirmation of the frozen five-edge head17.2 source program."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random

import run_setting2_regional_attention17h2_port_preservation_search_v1 as search


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
PREREG = P / "SETTING2_REGIONAL_ATTENTION17H2_PORT_PRESERVATION_FRESH_V1_PREREGISTRATION.md"
ROWS = P / "SETTING2_REGIONAL_ATTENTION17H2_FACTOR_CORNER_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_PORT_PRESERVATION_FRESH_V1_BINDING.json"
PARENT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_port_preservation_search_v3_result.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_port_preservation_fresh_v1_result.json"
FROZEN = (0, 1, 2, 4, 13)
TARGET_ONLY = (0, 12, 15)
_pool = [support for support in itertools.combinations(range(len(search.NAMES)), 5) if support != FROZEN]
NULLS = tuple(random.Random(202609160417).sample(_pool, 16))
SUPPORTS = (FROZEN, TARGET_ONLY) + NULLS
PRICE = {
    "physical_model_executions": 12,
    "full_model_sequences": 96,
    "source_terms": 16,
    "support_candidates": 18,
    "candidate_suffix_sequences": 1728,
    "complete_corner_suffix_sequences": 96,
    "token_logits_per_sequence": 12,
    "support_nulls": 16,
    "fits": 0,
    "backwards": 0,
    "parameter_updates": 0,
}
PREDICATES = {
    "pred_a_exact_instrument": None,
    "pred_b_fresh_response_replay": None,
    "pred_c_fresh_bidirectional_causality": None,
    "pred_d_fresh_preservation": None,
    "pred_e_beats_target_only": None,
    "pred_f_frozen_support_specificity": None,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authority():
    binding = json.loads(BINDING.read_text())
    files = {
        "preregistration": PREREG,
        "rows": ROWS,
        "row_check": P / "regional_cue_row_check_v1.py",
        "parent_result": PARENT,
        "search_runner": Path(search.__file__).resolve(),
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    parent = json.loads(PARENT.read_text())
    frozen_names = [search.NAMES[index] for index in FROZEN]
    if parent["terminal"] != "head17_2_preserving_port_candidate" or parent["selected"]["support"] != frozen_names or not all(parent["predictions"].values()):
        raise ValueError("discovery support authority changed")
    return binding, parent


def load_bound_fresh():
    binding, parent = authority()
    rows_document = json.loads(ROWS.read_text())
    rows = rows_document["rows"]
    checks = search.validate(rows)
    buckets = {}
    for index, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(index)
    executions = 2 * sum(math.ceil(len(indices) / 8) for indices in buckets.values())
    if len(rows) != 48 or rows_document["prior_context_overlap"] != 0 or executions != PRICE["physical_model_executions"] or len(SUPPORTS) != PRICE["support_candidates"]:
        raise ValueError("row, support, or execution price changed")
    adapted_parent = {"selected_support": [search.NAMES[index] for index in TARGET_ONLY]}
    return binding, adapted_parent, rows, checks, buckets


original_atomic_create_json = search.managed.atomic_create_json


def frozen_atomic_create_json(path, result):
    authority()
    by_support = {tuple(report["support"]): report for report in result["top_candidates"]}
    frozen_names = tuple(search.NAMES[index] for index in FROZEN)
    target_names = tuple(search.NAMES[index] for index in TARGET_ONLY)
    null_names = [tuple(search.NAMES[index] for index in support) for support in NULLS]
    if set(by_support) != {frozen_names, target_names, *null_names}:
        raise RuntimeError("evaluated support set changed")
    frozen = by_support[frozen_names]
    target_only = by_support[target_names]
    null_reports = [{"support": list(names), "minimax_normalized_gate_score": by_support[names]["minimax_normalized_gate_score"]} for names in null_names]
    null_median = sorted(report["minimax_normalized_gate_score"] for report in null_reports)[len(null_reports) // 2 - 1]
    instrument = result["predictions"]["pred_a_exact_instrument"]
    improves = frozen["maximum_control_gate_ratio"] <= .90 * target_only["maximum_control_gate_ratio"] and frozen["passes_response"] and frozen["passes_causality"]
    specificity = frozen["minimax_normalized_gate_score"] + .10 <= null_median
    predictions = {
        "pred_a_exact_instrument": bool(instrument),
        "pred_b_fresh_response_replay": bool(instrument and frozen["passes_response"]),
        "pred_c_fresh_bidirectional_causality": bool(instrument and frozen["passes_causality"]),
        "pred_d_fresh_preservation": bool(instrument and frozen["passes_controls"]),
        "pred_e_beats_target_only": bool(instrument and improves),
        "pred_f_frozen_support_specificity": bool(instrument and specificity),
    }
    result["schema"] = "setting2_regional_attention17h2_port_preservation_fresh_v1_result"
    result["terminal"] = "fresh_head17_2_five_edge_port_program" if all(predictions.values()) else "valid_fresh_five_edge_port_null" if instrument else "invalid"
    result["predictions"] = predictions
    result["selected"] = frozen
    result["target_only_baseline"] = target_only
    result["support_null_reports"] = null_reports
    result["support_null_median_minimax_score"] = null_median
    result["selection_rule"] = "frozen discovery support; no fresh-panel selection"
    result["scope"] = "Fresh-for-source-support confirmation of the frozen five unit-gain propagated module edges for the head17.2 native-QK1/edited-QK2-and-value corner; module-write differences remain live ports."
    original_atomic_create_json(path, result)


def configure():
    search.RUNNER = RUNNER
    search.PREREG = PREREG
    search.ROWS = ROWS
    search.BINDING = BINDING
    search.PARENT = PARENT
    search.OUT = OUT
    search.SUPPORTS = SUPPORTS
    search.PRICE = PRICE
    search.load_bound = load_bound_fresh
    search.managed.atomic_create_json = frozen_atomic_create_json


def main():
    authority()
    configure()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        planned = search.plan()
        planned["schema"] = "setting2_regional_attention17h2_port_preservation_fresh_v1_plan"
        planned["frozen_support"] = [search.NAMES[index] for index in FROZEN]
        planned["selection"] = "none on fresh panel"
        print(json.dumps(planned, sort_keys=True))
        return
    search.main()


if __name__ == "__main__":
    main()
