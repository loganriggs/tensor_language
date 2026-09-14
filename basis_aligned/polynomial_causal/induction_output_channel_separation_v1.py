#!/usr/bin/env python3
"""Post-result audit of answer-channel versus full-vocabulary preservation.

This reuses the thresholds frozen for INDUCTION_LIVE_CLAMP_V1.  It does not
rescore logits, choose a new factor, or identify a downstream consumer.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROWS = HERE / "INDUCTION_LIVE_CLAMP_V1_ROWS.jsonl"
PARENT = HERE / "INDUCTION_LIVE_CLAMP_V1_RESULT.json"
OUT = HERE / "INDUCTION_OUTPUT_CHANNEL_SEPARATION_V1_RESULT.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parent = json.loads(PARENT.read_text())
    assert digest(ROWS) == parent["row_sha256"]
    rows = [json.loads(line) for line in ROWS.open()]
    rows = [r for r in rows if r["cell"].startswith("selector_payload_joint_answer_preserved|")]
    assert len(rows) == 288

    cells = {}
    for cell in sorted({r["cell"] for r in rows}):
        items = [r for r in rows if r["cell"] == cell]
        report = parent["reports"][cell]
        assert len(items) == 72 == len({r["group_id"] for r in items})
        mean_ce = float(np.mean([r["ce_damage"] for r in items]))
        correct_fraction = float(np.mean([r["correct"] for r in items]))
        median_vocab = float(np.median([r["vocab_rms"] for r in items]))
        assert abs(mean_ce - report["mean_ce_damage"]) < 1e-12
        assert abs(correct_fraction - report["correct_fraction"]) < 1e-12
        assert abs(median_vocab - report["median_vocab_rms"]) < 1e-12

        # These are the parent experiment's necessary bars, reused unchanged.
        answer_channel_preserved = mean_ce <= 0.10 and correct_fraction >= 0.75
        full_distribution_preserved = median_vocab <= report["vocabulary_bar"]
        cells[cell] = {
            "groups": 72,
            "mean_ce_damage": mean_ce,
            "correct_fraction": correct_fraction,
            "median_vocab_rms": median_vocab,
            "frozen_vocab_bar": report["vocabulary_bar"],
            "answer_channel_preserved": bool(answer_channel_preserved),
            "full_distribution_preserved": bool(full_distribution_preserved),
            "answer_preserved_distribution_failed": bool(
                answer_channel_preserved and not full_distribution_preserved
            ),
        }

    separated = sum(v["answer_preserved_distribution_failed"] for v in cells.values())
    answer_pass = sum(v["answer_channel_preserved"] for v in cells.values())
    distribution_pass = sum(v["full_distribution_preserved"] for v in cells.values())
    out = {
        "experiment": "induction_output_channel_separation_v1",
        "analysis_status": "post_result_exploratory_audit",
        "rows": len(rows),
        "groups": 72,
        "cells": cells,
        "summary": {
            "answer_channel_pass_cells": answer_pass,
            "full_distribution_pass_cells": distribution_pass,
            "answer_preserved_distribution_failed_cells": separated,
            "all_cells_correct_fraction_at_least_0_75": all(
                v["correct_fraction"] >= 0.75 for v in cells.values()
            ),
        },
        "interpretation": (
            "Three of four answer-preserving joint cells retain the frozen CE and "
            "answer-choice channel while all four fail the frozen vocabulary bar. "
            "The remaining cell retains answer choice but misses the CE bar. This is "
            "output-level evidence that the four-site operation has contextual uses "
            "outside the copied-answer channel; it does not identify the consumer."
        ),
        "claim_boundary": (
            "No new threshold, factor, module, fit, intervention, held-out split, or "
            "consumer identity. A prospective native capture is required to locate "
            "or manipulate the contextual consumer."
        ),
        "source_sha256": digest(Path(__file__)),
        "rows_sha256": digest(ROWS),
        "parent_result_sha256": digest(PARENT),
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
