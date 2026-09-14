#!/usr/bin/env python3
"""Decompose the V2 post-clamp context change into baseline and pointer terms."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "SUCCESSOR_POINTER_PREFIX_INTERACTION_V2_RESULT.json"
OUT = HERE / "SUCCESSOR_POINTER_PREFIX_INTERACTION_V2_ANALYSIS.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text())
    if source["terminal"] != "mixed_interaction":
        raise RuntimeError("unexpected source terminal")
    by_key = {
        (row["family"], row["direction"], row["final_index"], row["condition"]): row
        for row in source["records"]
    }
    reports = defaultdict(dict)
    max_identity_error = 0.0
    for family in source["plan"]["families"]:
        indices = sorted({row["final_index"] for row in source["records"]
                          if row["family"] == family})
        for direction in source["plan"]["directions"]:
            direction_report = {}
            for condition in ("early_swap_control", "late_swap_incoherent"):
                rows = []
                for index in indices:
                    coherent = by_key[(family, direction, index, "coherent")]
                    changed = by_key[(family, direction, index, condition)]
                    baseline = changed["self_donor_margin"] - coherent["self_donor_margin"]
                    pointer = changed["pointer_effect"] - coherent["pointer_effect"]
                    post = changed["donor_margin"] - coherent["donor_margin"]
                    error = abs(post - baseline - pointer)
                    max_identity_error = max(max_identity_error, error)
                    rows.append({
                        "final_index": index,
                        "baseline_margin_change": baseline,
                        "pointer_effect_change": pointer,
                        "post_clamp_margin_change": post,
                        "identity_absolute_error": error,
                        "donor_answer_win_before": coherent["donor_answer_win"],
                        "donor_answer_win_after": changed["donor_answer_win"],
                    })
                denominator = statistics.fmean(
                    abs(row["baseline_margin_change"]) + abs(row["pointer_effect_change"])
                    for row in rows
                )
                flips = [row for row in rows if
                         row["donor_answer_win_before"] != row["donor_answer_win_after"]]
                direction_report[condition] = {
                    "n": len(rows),
                    "mean_baseline_margin_change": statistics.fmean(
                        row["baseline_margin_change"] for row in rows),
                    "mean_pointer_effect_change": statistics.fmean(
                        row["pointer_effect_change"] for row in rows),
                    "mean_post_clamp_margin_change": statistics.fmean(
                        row["post_clamp_margin_change"] for row in rows),
                    "baseline_share_of_absolute_change_components": (
                        statistics.fmean(abs(row["baseline_margin_change"]) for row in rows)
                        / denominator if denominator else None
                    ),
                    "baseline_larger_than_pointer_fraction": statistics.fmean(
                        abs(row["baseline_margin_change"]) > abs(row["pointer_effect_change"])
                        for row in rows
                    ),
                    "donor_answer_win_flip_count": len(flips),
                    "flip_rows": flips,
                    "rows": rows,
                }
            reports[family][direction] = direction_report
    out = {
        "schema": "successor_pointer_prefix_interaction_v2_analysis",
        "source": str(SOURCE.relative_to(HERE.parents[1])),
        "source_sha256": digest(SOURCE),
        "algebra": "post-clamp context change = self baseline-margin change + pointer-effect change",
        "maximum_identity_absolute_error": max_identity_error,
        "reports": reports,
        "interpretation_rule": (
            "A donor-win change without a positive pointer-effect interaction is attributed to "
            "the changed baseline decision state, not a multiplicative gate on the pointer channel."
        ),
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(OUT),
        "maximum_identity_absolute_error": max_identity_error,
        "late_swap": {
            family: {
                direction: reports[family][direction]["late_swap_incoherent"]
                for direction in source["plan"]["directions"]
            } for family in source["plan"]["families"]
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
