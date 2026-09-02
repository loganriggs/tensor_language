#!/usr/bin/env python3
"""RUNG488 -- BF16-calibrated rerun of exact MLP1 secant-factor interchange."""

# BQGATE: EXPERIMENT
# pred_a exact instrument under BF16 unit-roundoff-derived bars
# pred_b stable material own finite responses
# pred_c exactly the frozen T-I context-factor interchange edge
# pred_d the same graph in both discovery halves
# pred_e the frozen graph validates on held-out documents

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import mlp1_finite_secant_factor_interchange_rung487 as base
from receipt import dump


PREREG = POLY / "MLP1_FINITE_SECANT_FACTOR_INTERCHANGE_RUNG488_PREREGISTRATION.md"
BASE_SOURCE = ROOT / "ops/mlp1_finite_secant_factor_interchange_rung487.py"
BASE_RESULT = ROOT / "mlp1_finite_secant_factor_interchange_rung487_results.json"
OUT = ROOT / "mlp1_finite_secant_factor_interchange_rung488_results.json"
HASHES = {
    PREREG: "d55251fac45bba5c3a6d7ac71c23fca9ab3f8e361d3e156262e3885cae05fcdd",
    BASE_SOURCE: "0339a0e24189eb0eff4ef73940cbe617c3ff6ede75e1e18c4477875893343ffc",
    BASE_RESULT: "eb3f575d41d3514be6fe21ab285d854af30c6b38029d146397fa93163a6df593",
}
U = 2.0 ** -8
POLARIZATION_BF16_BAR = 8 * U * U
OWN_WRITE_BF16_BAR = 4 * U * U
FROZEN_EDGES = [{"pair": ["T", "I"], "type": "context"}]


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or base.sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    parent = json.loads(BASE_RESULT.read_text())
    instrument = parent["discovery"]["instrument"]
    if parent.get("rung") != 487 \
            or parent.get("pred_a_exact_lawful_instrument") is not False \
            or parent.get("pred_b_stable_own_finite_responses") is not True \
            or parent.get("pred_c_at_least_one_factor_interchange_edge") is not True \
            or parent.get("pred_d_stable_factor_sharing_graph") is not True \
            or parent.get("pred_e_heldout_documents") is not False \
            or parent.get("validation_licensed_and_opened") is not False \
            or parent.get("selected_edges") != FROZEN_EDGES \
            or instrument["polarization_float32_relative_squared_max"] > 1e-8 \
            or instrument["polarization_bf16_relative_squared_max"] <= 1e-5 \
            or instrument["own_native_write_relative_squared_max"] <= 1e-5:
        raise RuntimeError("rung487 does not match the registered precision-repair parent")
    return base.validate_inputs()


def instrument_valid(instrument, analysis):
    return bool(
        instrument["calls_exact"]
        and instrument["native_prefix_D_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_A_relative_squared_max"] <= 1e-12
        and instrument["native_prefix_M_relative_squared_max"] <= 1e-12
        and instrument["mlp0_state_max_abs"] == 0.0
        and instrument["polarization_float32_relative_squared_max"] <= 1e-8
        and instrument["polarization_bf16_relative_squared_max"] <= POLARIZATION_BF16_BAR
        and instrument["own_native_write_relative_squared_max"] <= OWN_WRITE_BF16_BAR
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and analysis["all_physical_effects_live"])


def exact_frozen_graph(analysis):
    return analysis["descriptive_edges"] == FROZEN_EDGES \
        and analysis["pred_d_factor_graph_stable"] is True


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert POLARIZATION_BF16_BAR == 0.0001220703125
        assert OWN_WRITE_BF16_BAR == 0.00006103515625
        assert FROZEN_EDGES == [{"pair": ["T", "I"], "type": "context"}]
        print(json.dumps({
            "status": "dry_run_passed", "rung": 488,
            "model_loaded": False, "outcomes_opened": False,
            "validation_outcomes_opened": False,
            "final_or_sealed_opened": False,
            "discovery_forwards": (500 // base.BATCH) * 28,
            "conditional_validation_forwards": (500 // base.BATCH) * 28,
            "polarization_bf16_bar": POLARIZATION_BF16_BAR,
            "own_write_bf16_bar": OWN_WRITE_BF16_BAR,
            "frozen_graph": FROZEN_EDGES,
            "registered_predictions": ["pred_a", "pred_b", "pred_c", "pred_d", "pred_e"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung488 output namespace already exists")
    rows, positive, fit_rows, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = base.component_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))

    discovery = base.collect_phase(model, rows, reference, *base.DISCOVERY_RANGE)
    discovery_analysis = base.analyze_phase(discovery, positive[:500])
    pred_a = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
                  and instrument_valid(discovery["instrument"], discovery_analysis))
    pred_b = discovery_analysis["pred_b_own_responses_stable"]
    pred_c = discovery_analysis["descriptive_edges"] == FROZEN_EDGES
    pred_d = exact_frozen_graph(discovery_analysis)
    validation_licensed = bool(pred_a and pred_b and pred_c and pred_d)

    validation = validation_analysis = None
    pred_e = False
    if validation_licensed:
        validation = base.collect_phase(model, rows, reference, *base.VALIDATION_RANGE)
        validation_analysis = base.analyze_phase(
            validation, positive[500:1000], frozen_edges=FROZEN_EDGES)
        pred_e = bool(instrument_valid(validation["instrument"], validation_analysis)
                      and validation_analysis["pred_b_own_responses_stable"]
                      and exact_frozen_graph(validation_analysis))

    strong_null = bool(not pred_a or not pred_b or not pred_d or not pred_e)
    result = {
        "status": "complete", "rung": 488,
        "claim_level": "heldout_exact_finite_secant_factor_interchange",
        "source_hashes": {str(path): base.sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(base.BRANCHES),
        "ordered_pairs": [list(pair) for pair in base.ORDERED_PAIRS],
        "modes": list(base.MODES),
        "position_shift_offsets": list(base.POSITION_SHIFTS),
        "precision_bars": {
            "bf16_unit_roundoff": U,
            "polarization_relative_squared": POLARIZATION_BF16_BAR,
            "own_write_relative_squared": OWN_WRITE_BF16_BAR,
        },
        "discovery": {
            "documents": list(base.DISCOVERY_RANGE), "split": base.SPLIT,
            "instrument": discovery["instrument"], "analysis": discovery_analysis,
            "native_ce_mean": float(discovery["native"].mean()),
        },
        "validation": None if validation is None else {
            "documents": list(base.VALIDATION_RANGE), "split": 750,
            "instrument": validation["instrument"], "analysis": validation_analysis,
            "native_ce_mean": float(validation["native"].mean()),
        },
        "selected_edges": FROZEN_EDGES if pred_d else [],
        'pred_a_exact_lawful_instrument': pred_a,
        'pred_b_stable_own_finite_responses': pred_b,
        'pred_c_exact_frozen_factor_interchange_edge': pred_c,
        'pred_d_stable_factor_sharing_graph': pred_d,
        'pred_e_heldout_documents': pred_e,
        "validation_licensed_and_opened": validation_licensed,
        "strong_null": strong_null,
        "final_or_sealed_opened": False,
        "execution_price": {
            "discovery_full_model_forwards": sum(
                discovery["instrument"]["calls"][key] for key in
                ("native_forwards", "absent_forwards", "physical_forwards")),
            "validation_full_model_forwards": 0 if validation is None else sum(
                validation["instrument"]["calls"][key] for key in
                ("native_forwards", "absent_forwards", "physical_forwards")),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "cross_document_T_I_shared_live_state_extraction_and_selective_swap"
            if pred_e else "within_branch_integrated_secant_response_reader"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 488,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "selected_edges": result["selected_edges"],
        "strong_null": strong_null,
        "validation_opened": validation_licensed,
        "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
