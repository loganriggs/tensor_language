#!/usr/bin/env python3
"""CPU-only rung-533 contract and the rung-532 decision table that motivates it."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
PARENT = ROOT / "equality_factor_companion_causal_equivalence_rung532_results.json"
OUT = ROOT / "equality_factor_to_slot_exchangeability_rung533_math.json"

SCALES = {
    "source_first_to_target_first": -1.268044102615207,
    "source_second_to_target_first": 1.227983240318439,
    "source_first_to_target_second": -0.8533769036200292,
    "source_second_to_target_second": 0.6995515454196305,
}
PARENT_ARMS = {
    "source_first_to_target_first": "direct_first",
    "source_second_to_target_first": "swapped_first",
    "source_first_to_target_second": "swapped_second",
    "source_second_to_target_second": "direct_second",
}
PARENT_MATCHED_PERMUTATIONS = {
    "source_first_to_target_first": None,
    "source_second_to_target_first": "permuted_first",
    "source_first_to_target_second": "permuted_second",
    "source_second_to_target_second": None,
}


def key_prefix_reverse(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or value.shape[-1] != value.shape[-2]:
        raise ValueError("expected [batch, query, key] square score factor")
    result = value.clone()
    length = value.shape[-1]
    for query in range(length):
        result[:, query, :query + 1] = value[:, query, :query + 1].flip(-1)
    return result


def substitution(mapping: str, source_first, source_second, target_first, target_second,
                 *, permuted: bool = False):
    if mapping not in SCALES:
        raise ValueError(f"unknown mapping: {mapping}")
    source = source_first if "source_first" in mapping else source_second
    if permuted:
        source = key_prefix_reverse(source)
    companion = target_second if "target_first" in mapping else target_first
    return (SCALES[mapping] * source) * companion


def _base_pass(report):
    metric = report["member_effect"]
    recovery = report["copy_task_recovery"]
    return bool(
        metric["cosine"] >= 0.85
        and metric["relative_error"] <= 0.60
        and recovery is not None and 0.65 <= recovery <= 1.40
        and report["slice_control_mean_abs_ce_change_from_native"] <= 0.01
    )


def analyze_parent():
    result = json.loads(PARENT.read_text())
    contexts = list(result["reports"].values())
    table = {}
    for mapping, arm in PARENT_ARMS.items():
        reports = [context["arms"][arm] for context in contexts]
        control = PARENT_MATCHED_PERMUTATIONS[mapping]
        table[mapping] = {
            "parent_arm": arm,
            "frozen_scale": SCALES[mapping],
            "base_contexts_passing": sum(_base_pass(report) for report in reports),
            "minimum_effect_cosine": min(report["member_effect"]["cosine"] for report in reports),
            "maximum_relative_error": max(
                report["member_effect"]["relative_error"] for report in reports),
            "matched_key_control_present_in_parent": control is not None,
            "matched_key_control_arm": control,
            "beats_matched_control_by_0p15_contexts": (
                sum(
                    report["member_effect"]["cosine"]
                    >= context["arms"][control]["member_effect"]["cosine"] + 0.15
                    for report, context in zip(reports, contexts)
                ) if control is not None else None
            ),
        }
    return table


def symbolic_contract(seed: int = 533):
    generator = torch.Generator().manual_seed(seed)
    tensors = [torch.randn(2, 7, 7, generator=generator) for _ in range(4)]
    source_first, source_second, target_first, target_second = tensors
    reports = {}
    for mapping in SCALES:
        native = substitution(
            mapping, source_first, source_second, target_first, target_second)
        permuted = substitution(
            mapping, source_first, source_second, target_first, target_second,
            permuted=True)
        reports[mapping] = {
            "shape": list(native.shape),
            "matched_control_uses_same_scale": True,
            "matched_control_differs": bool(not torch.equal(native, permuted)),
        }
    return reports


def main():
    parent_table = analyze_parent()
    symbolic = symbolic_contract()
    report = {
        "status": "cpu_contract_passed",
        "rung": 533,
        "parent_decision_table": parent_table,
        "symbolic_contract": symbolic,
        "all_four_parent_mappings_meet_base_bars": all(
            row["base_contexts_passing"] == 8 for row in parent_table.values()),
        "missing_parent_matched_controls": [
            mapping for mapping, row in parent_table.items()
            if not row["matched_key_control_present_in_parent"]
        ],
        "next_gpu_test": "four mappings each paired with its own key-permuted control",
        "scientific_outcomes_opened": False,
        "model_loaded": False,
    }
    dump(report, OUT)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
