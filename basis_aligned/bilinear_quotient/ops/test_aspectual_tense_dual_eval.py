#!/usr/bin/env python3
"""Fast CPU contract tests for aspectual_tense_dual_eval."""

from types import SimpleNamespace

import aspectual_tense_dual_eval as evaluator


def main():
    assert evaluator.verify_contract()
    rows = []
    for group in range(2):
        for family in evaluator.FAMILIES:
            rows.append({"family": family, "direction_id": "present_to_past" if group == 0 else "past_to_present", "group_number": group})
    outputs = {
        "base": SimpleNamespace(answer_foil=[(1.0, 0.0)] * len(rows)),
        "donor": SimpleNamespace(answer_foil=[(1.0, 0.0)] * len(rows)),
    }
    cells = evaluator.capability_cells("has_had", rows, outputs)
    assert len(cells) == 8 and all(cell["total"] == 2 and cell["passed"] for cell in cells)
    records = []
    for bank in evaluator.BANKS:
        for family in evaluator.FAMILIES:
            key = evaluator.METRIC_BY_FAMILY[family]
            records.extend({"bank": bank, "family": family, key: 0.0 if family == "C" else 1.0} for _ in range(2))
    summaries = evaluator.summarize_program_records(records)
    assert all(evaluator.program_bars_pass(summaries[bank]) for bank in evaluator.BANKS)
    try:
        evaluator.metric_summary([{"recovery": float("nan")}], "recovery")
        raise AssertionError("nonfinite metric accepted")
    except evaluator.DualEvalError:
        pass
    print("aspectual_tense_dual_eval: PASS")


if __name__ == "__main__":
    main()
