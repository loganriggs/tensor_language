#!/usr/bin/env python3
"""CPU-only terminal audit of R544 from its saved row-level measurements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "pending_opener_four_closer_site_gate_rung544_results.json"
ROWS = ROOT / "pending_opener_unique_rows_rung543_v2.json"
OUT = ROOT / "pending_opener_four_closer_site_gate_rung544_audit.json"
TARGETS = ("direct_type_substitution", "completed_then_reopened_order")
SPLITS = ("FIT", "SELECT")
TOKEN_LABELS = {8: ")", 60: "]", 92: "}", 1: '"'}
SUPPORTED_POSTHOC = {8, 60, 1}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result = json.loads(RESULT.read_text())
    rows = json.loads(ROWS.read_text())
    assert result["pred_a_exact_instrument"] is True
    assert result["model_forwards"] == 450 and result["model_backwards"] == 0
    assert result["forbidden_splits_opened"] == []
    assert result["input_sha256"][str(ROWS)] == sha256(ROWS)
    assert rows["row_count"] == 1200 and rows["group_count"] == 240

    failed_pairs: dict[str, dict[str, list[str]]] = {}
    failed_values = set()
    non_curly_cells = []
    curly_cells = []
    for split in SPLITS:
        failed_pairs[split] = {}
        for family in TARGETS:
            cells = result["capability"][split][family]["ordered_pairs"]
            failures = sorted(pair for pair, cell in cells.items() if not cell["passed"])
            expected = sorted(pair for pair in cells if "92" in pair.split("->"))
            assert failures == expected
            failed_pairs[split][family] = failures
            for pair, cell in cells.items():
                values = {int(value) for value in pair.split("->")}
                (curly_cells if 92 in values else non_curly_cells).append(cell)
                if not cell["passed"]:
                    failed_values.update(values & {92})

    assert failed_values == {92}
    assert all(cell["base_correct_fraction"] == 1.0 and cell["donor_correct_fraction"] == 1.0
               for cell in non_curly_cells)
    assert all(result["site_reports"]["attn13h8"][split][family]["passed"]
               for split in SPLITS for family in TARGETS)
    assert all(result["site_reports"]["attn13h8"][split][family]["causally_live"]
               for split in SPLITS
               for family in ("surface_paraphrase", "distance_shift", "nonopener_punctuation_substitution"))

    posthoc_rows = [
        row for split in SPLITS for family in TARGETS
        for row in result["raw_capability"][split][family]
        if row["base_answer_id"] in SUPPORTED_POSTHOC and row["donor_answer_id"] in SUPPORTED_POSTHOC
    ]
    posthoc_three_value = {
        "values": [TOKEN_LABELS[value] for value in sorted(SUPPORTED_POSTHOC)],
        "n_rows": len(posthoc_rows),
        "base_correct_fraction": float(np.mean([row["base_margin"] > 0 for row in posthoc_rows])),
        "donor_correct_fraction": float(np.mean([row["donor_margin"] > 0 for row in posthoc_rows])),
        "status": "diagnostic_only_selected_after_R544; requires fresh preregistered confirmation",
    }
    assert posthoc_three_value["base_correct_fraction"] == 1.0
    assert posthoc_three_value["donor_correct_fraction"] == 1.0

    audit = {
        "rung": 544,
        "status": "terminal_audit_complete",
        "result_sha256": sha256(RESULT),
        "exact_budget_and_split_authority": True,
        "four_value_capability_held": False,
        "failed_native_value_ids": [92],
        "failed_native_values": [TOKEN_LABELS[92]],
        "failed_ordered_pairs": failed_pairs,
        "all_non_curly_native_cells_perfect": True,
        "attn13h8_target_and_control_full_state_gate_held": True,
        "resid8_full_state_gate_held": False,
        "posthoc_three_value_diagnostic": posthoc_three_value,
        "decision": (
            "Reject the four-value extension before any projector fit. The model does not natively predict the "
            "curly closer on these prompts, although complete attn13h8 swaps causally move every requested "
            "closer contrast and all controls are live. Build fresh three-value confirmation rows before treating "
            "parenthesis/square/quote as a supported variable domain."
        ),
        "final_test_or_ood_opened": False,
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
