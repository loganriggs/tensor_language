#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_response_replay pred_c_bidirectional_causality pred_d_preservation pred_e_improves_target_only pred_f_support_specificity pred_g_improves_width3
"""Extend the preservation-aware head17.2 port search through width four."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import run_setting2_regional_attention17h2_port_preservation_search_v1 as v1


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
PREREG = P / "SETTING2_REGIONAL_ATTENTION17H2_PORT_PRESERVATION_SEARCH_V2_PREREGISTRATION.md"
ROWS = P / "ODD_FRAMING_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_PORT_PRESERVATION_SEARCH_V2_BINDING.json"
PARENT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_port_preservation_search_v1_result.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_port_preservation_search_v2_result.json"
SUPPORTS = tuple(support for width in range(5) for support in itertools.combinations(range(len(v1.NAMES)), width))
PRICE = {
    "physical_model_executions": 12,
    "full_model_sequences": 96,
    "source_terms": 16,
    "support_candidates": 2517,
    "candidate_suffix_sequences": 241632,
    "complete_corner_suffix_sequences": 96,
    "token_logits_per_sequence": 12,
    "support_nulls": 16,
    "fits": 0,
    "backwards": 0,
    "parameter_updates": 0,
}
PREDICATES = {
    "pred_a_exact_instrument": None,
    "pred_b_response_replay": None,
    "pred_c_bidirectional_causality": None,
    "pred_d_preservation": None,
    "pred_e_improves_target_only": None,
    "pred_f_support_specificity": None,
    "pred_g_improves_width3": None,
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
        "v1_runner": Path(v1.__file__).resolve(),
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    parent = json.loads(PARENT.read_text())
    if parent["terminal"] != "valid_preservation_search_null" or parent["selected"]["support"] != ["attn9", "attn10", "attn11"]:
        raise ValueError("width-three preservation null changed")
    return binding, parent


def load_bound_v2():
    binding, parent = authority()
    rows = json.loads(ROWS.read_text())["rows"]
    checks = v1.validate(rows)
    buckets = {}
    for index, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(index)
    executions = 2 * sum(math.ceil(len(indices) / 8) for indices in buckets.values())
    if len(rows) != 48 or executions != PRICE["physical_model_executions"] or len(SUPPORTS) != PRICE["support_candidates"]:
        raise ValueError("row, support, or execution price changed")
    adapted_parent = {"selected_support": parent["target_only_baseline"]["support"]}
    return binding, adapted_parent, rows, checks, buckets


original_atomic_create_json = v1.managed.atomic_create_json


def extended_atomic_create_json(path, result):
    _, parent = authority()
    width3_score = parent["selected"]["minimax_normalized_gate_score"]
    improves_width3 = result["selected"]["minimax_normalized_gate_score"] <= .90 * width3_score
    result["predictions"]["pred_g_improves_width3"] = bool(result["predictions"]["pred_a_exact_instrument"] and improves_width3)
    result["terminal"] = "head17_2_preserving_port_candidate" if all(result["predictions"].values()) else "valid_preservation_search_null" if result["predictions"]["pred_a_exact_instrument"] else "invalid"
    result["schema"] = "setting2_regional_attention17h2_port_preservation_search_v2_result"
    result["width3_baseline"] = {
        "support": parent["selected"]["support"],
        "minimax_normalized_gate_score": width3_score,
        "required_ten_percent_improvement_ceiling": .90 * width3_score,
    }
    result["scope"] += " V2 extends the identical frozen search from width three through width four."
    original_atomic_create_json(path, result)


def configure():
    v1.RUNNER = RUNNER
    v1.PREREG = PREREG
    v1.ROWS = ROWS
    v1.BINDING = BINDING
    v1.PARENT = PARENT
    v1.OUT = OUT
    v1.SUPPORTS = SUPPORTS
    v1.PRICE = PRICE
    v1.load_bound = load_bound_v2
    v1.managed.atomic_create_json = extended_atomic_create_json


def main():
    authority()
    configure()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        planned = v1.plan()
        planned["schema"] = "setting2_regional_attention17h2_port_preservation_search_v2_plan"
        planned["maximum_support_width"] = 4
        print(json.dumps(planned, sort_keys=True))
        return
    v1.main()


if __name__ == "__main__":
    main()
