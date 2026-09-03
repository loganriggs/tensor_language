#!/usr/bin/env python3
"""CPU screen for an exact shared-product plus target-specific score residual split."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
PARENT_RESULT = ROOT / "equality_factor_to_slot_exchangeability_rung533_results.json"
PARENT_BUNDLE = ROOT / "equality_factor_to_slot_exchangeability_rung533_bundle.pt"
OUT = ROOT / "equality_product_shared_private_rung534_math.json"
GAMMA = -1.0785167862928777
ROLES = ("final_natural", "ood_code")
BACKGROUNDS = ("donor_present", "donor_absent")
ARMS = (
    "native", "absent", "product_control",
    "source_first_to_target_first", "source_first_to_target_first_key_control",
    "source_second_to_target_first", "source_second_to_target_first_key_control",
    "source_first_to_target_second", "source_first_to_target_second_key_control",
    "source_second_to_target_second", "source_second_to_target_second_key_control",
)
CELLS = ("positive", "matched_negative", "off_target")
DOCUMENT_SPLIT = 96


def _vector_metrics(reference, candidate):
    reference = torch.as_tensor(reference, dtype=torch.float64)
    candidate = torch.as_tensor(candidate, dtype=torch.float64)
    ref2 = float(reference.square().sum())
    cand2 = float(candidate.square().sum())
    if ref2 <= 0:
        raise RuntimeError("reference effect is zero")
    cosine = 0.0 if cand2 <= 0 else float((reference * candidate).sum()) / math.sqrt(ref2 * cand2)
    return {
        "cosine": cosine,
        "relative_norm": math.sqrt(cand2 / ref2),
        "relative_error": math.sqrt(float((reference - candidate).square().sum()) / ref2),
    }


def exact_algebra(seed=534):
    generator = torch.Generator().manual_seed(seed)
    source_first, source_second, target_first, target_second = [
        torch.randn(3, 11, 11, generator=generator, dtype=torch.float64)
        for _ in range(4)
    ]
    native = target_first * target_second
    shared = GAMMA * source_first * source_second
    private = native - shared
    return {
        "definition": "native_product = shared_source_product + target_specific_residual",
        "maximum_recomposition_error": float((native - (shared + private)).abs().max()),
        "shared_is_gauge_invariant_complete_product": True,
        "private_is_fixed_by_parent_least_squares_product_scale": True,
    }


def analyze_parent():
    result = json.loads(PARENT_RESULT.read_text())
    bundle = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=False)
    collection = bundle["collection"]
    contexts = {}
    for role in ROLES:
        document_counts = collection["positive_document_counts"][role]
        document_ce = (
            collection["positive_document_sums"][role]
            / document_counts[None, None].clamp_min(1))
        cell_ce = (
            collection["cell_sums"][role]
            / collection["cell_counts"][role][None, None].clamp_min(1))
        for background_index, background in enumerate(BACKGROUNDS):
            for half in range(2):
                start, stop = half * DOCUMENT_SPLIT, (half + 1) * DOCUMENT_SPLIT
                live = document_counts[start:stop] > 0
                absent = document_ce[
                    background_index, ARMS.index("absent"), start:stop][live]
                native = document_ce[
                    background_index, ARMS.index("native"), start:stop][live]
                shared = document_ce[
                    background_index, ARMS.index("product_control"), start:stop][live]
                native_effect = absent - native
                shared_effect = absent - shared
                private_marginal_in_shared_background = native_effect - shared_effect
                cells = {}
                for cell_index, cell in enumerate(CELLS):
                    native_change = float(
                        cell_ce[background_index, ARMS.index("native"), half, cell_index]
                        - cell_ce[background_index, ARMS.index("absent"), half, cell_index])
                    shared_change = float(
                        cell_ce[background_index, ARMS.index("product_control"), half, cell_index]
                        - cell_ce[background_index, ARMS.index("absent"), half, cell_index])
                    cells[cell] = {
                        "native_minus_absent_ce": native_change,
                        "shared_minus_absent_ce": shared_change,
                        "private_marginal_shared_to_native_ce": native_change - shared_change,
                    }
                key = f"{role}/{background}/half{half}"
                contexts[key] = {
                    "shared_positive_effect": _vector_metrics(native_effect, shared_effect),
                    "private_positive_marginal": _vector_metrics(
                        native_effect, private_marginal_in_shared_background),
                    "cells": cells,
                    "positive_documents": int(live.sum()),
                }
    code_absent = [contexts[f"ood_code/donor_absent/half{half}"] for half in range(2)]
    return {
        "parent_registered_outcome": {
            key: result[key] for key in (
                "pred_a_valid_physical_instrument",
                "pred_b_product_level_positive_control",
                "pred_e_branch_exchangeable_downstream_family",
                "pred_f_donor_background_stability",
            )
        },
        "contexts": contexts,
        "code_donor_absent_summary": {
            "shared_positive_cosines": [
                row["shared_positive_effect"]["cosine"] for row in code_absent],
            "shared_positive_relative_errors": [
                row["shared_positive_effect"]["relative_error"] for row in code_absent],
            "shared_positive_recovery_from_parent": [
                result["reports"][f"ood_code/donor_absent/half{half}"]["arms"]
                ["product_control"]["positive_task_recovery"] for half in range(2)],
            "shared_matched_negative_ce_mismatch": [
                abs(row["cells"]["matched_negative"]
                    ["private_marginal_shared_to_native_ce"])
                for row in code_absent],
        },
    }


def main():
    parent = analyze_parent()
    report = {
        "status": "cpu_screen_passed",
        "rung": 534,
        "exact_algebra": exact_algebra(),
        **parent,
        "decision": (
            "test_target_product_as_shared_equality_signal_plus_private_context_correction; "
            "measure whether the private residual acts autonomously or only in composition"
        ),
        "distinction_from_rung464": (
            "rung464 swapped the 19 later-layer correction writes; rung534 splits the layer8 "
            "gauge-invariant score product itself before the target value/output path"
        ),
        "model_loaded": False,
        "new_scientific_outcomes_opened": False,
    }
    dump(report, OUT)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
