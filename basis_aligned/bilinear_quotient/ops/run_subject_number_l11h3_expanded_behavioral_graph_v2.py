#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_expanded_instrument pred_b_nine_term_composition_ood pred_c_ten_term_composition_ood
"""Prospective nine/ten-term extension of the exact V1 expanded graph."""
from __future__ import annotations

import hashlib, json, os
from pathlib import Path

import run_subject_number_l11h3_expanded_behavioral_graph_v1 as experiment

RUNNER = Path(__file__).resolve(); ROOT = RUNNER.parents[3]; POLY = ROOT / "basis_aligned/polynomial_causal"
FAILED_RUNNER = Path(experiment.__file__)
FAILED_RESULT = POLY / "SUBJECT_NUMBER_L11H3_EXPANDED_BEHAVIORAL_GRAPH_V1_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_L11H3_EXPANDED_BEHAVIORAL_GRAPH_V2_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_L11H3_EXPANDED_BEHAVIORAL_GRAPH_V2_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_L11H3_EXPANDED_BEHAVIORAL_GRAPH_V2_RESULT.json"
EXPECTED_FAILED_RUNNER_SHA256 = "594e694c835ed61b97e6f0b3bf1edf54b43b3203b70e5081443f7626ea8ee2af"
EXPECTED_FAILED_RESULT_SHA256 = "a62012f30bbe3f273291d7f2f5ea044952af7fdc07b7411b3bcea91a4bd9bd08"
STEPS = 10
PRICE = dict(experiment.PRICE, greedy_steps=STEPS)
PREDICTION_REGISTRY = {"pred_a_exact_expanded_instrument": None,
    "pred_b_nine_term_composition_ood": None, "pred_c_ten_term_composition_ood": None}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def configure():
    if sha(FAILED_RUNNER) != EXPECTED_FAILED_RUNNER_SHA256 or sha(FAILED_RESULT) != EXPECTED_FAILED_RESULT_SHA256:
        raise ValueError("V1 audit source changed")
    failed = json.loads(FAILED_RESULT.read_text())
    if failed["terminal"] != "valid_expanded_behavioral_decomposition" \
            or failed["predictions"]["pred_c_eight_term_composition_ood"] \
            or failed["graph"]["selected_masks"] != [8, 1, 4, 2, 24, 12, 16, 9]:
        raise ValueError("V1 failure or selection changed")
    experiment.RUNNER = RUNNER; experiment.PREREG = PREREG; experiment.BINDING = BINDING; experiment.OUT = OUT
    experiment.STEPS = STEPS; experiment.PRICE = PRICE; experiment.PREDICTION_REGISTRY = PREDICTION_REGISTRY
    return failed

def passes(report):
    bars = experiment.BARS
    return all(p["relative_l2"] <= bars["maximum_composition_relative_l2"]
               and p["cosine"] >= bars["minimum_composition_cosine"]
               and p["aligned_recovery"] > bars["minimum_aligned_recovery"] for p in report.values())

def main():
    failed = configure()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        planned = experiment.plan(); planned["schema"] = "subject_number_l11h3_expanded_behavioral_graph_v2_plan"
        planned["correction"] = "extend_discovery_greedy_budget_to_10"
        planned["failed_v1_terminal"] = failed["terminal"]
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    original_write = experiment.managed.atomic_create_json
    def corrected_write(path, result):
        if result["graph"]["selected_masks"][:8] != failed["graph"]["selected_masks"]:
            raise ValueError("first eight masks did not reproduce")
        pred_a = result["instrument"]["finite"] and result["predictions"]["pred_a_exact_expanded_instrument"]
        pred_b = bool(pred_a and passes(result["graph"]["prefix_panels"]["9"]))
        pred_c = bool(pred_a and passes(result["graph"]["prefix_panels"]["10"]))
        result["schema"] = "subject_number_l11h3_expanded_behavioral_graph_v2_result"
        result["predictions"] = dict(zip(PREDICTION_REGISTRY, (bool(pred_a), pred_b, pred_c)))
        result["terminal"] = "expanded_sparse_behavioral_graph" if pred_b or pred_c else "valid_expanded_behavioral_decomposition"
        result["correction"] = {"failed_v1_terminal": failed["terminal"],
            "failed_v1_lexical_ood_relative_l2": failed["graph"]["prefix_panels"]["8"]["lexical_ood"]["relative_l2"],
            "change": "greedy budget extended from 8 to 10; all else unchanged"}
        result["scope"] = "Prospective nine/ten-term extension of exact V1; first eight masks reproduced; no threshold or panel changes."
        original_write(path, result)
    experiment.managed.atomic_create_json = corrected_write
    try: experiment.main()
    finally: experiment.managed.atomic_create_json = original_write

if __name__ == "__main__": main()
