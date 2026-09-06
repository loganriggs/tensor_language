#!/usr/bin/env python3
"""Hash-bound block5 specialization of the exact block-crossing factorial."""

from __future__ import annotations

import hashlib
from pathlib import Path


SCRIPT_FILE = Path(globals()["__file__"]).resolve()
ROOT = SCRIPT_FILE.parents[1]
TEMPLATE = ROOT / "ops/run_aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2.py"
EXPECTED_TEMPLATE_SHA256 = "54db17a05a70deb41e762c77eb342fdc522a3849981f66507615e21ab06743ea"
STATIC_PREDICATES = {
    "pred_a_exact_crossing_instrument": "native/manual and algebraic closure",
    "pred_b_writer_recurrence": "fixed two-term writer recurrence",
    "pred_c_onset_recurrence": "resid6 onset recurrence",
    "pred_d_attention5_transport": "attention5 dominance and necessity",
    "pred_e_exact_coverage": "all frozen arms and rows",
}


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one template occurrence: {old!r}")
    return source.replace(old, new)


def main() -> None:
    payload = TEMPLATE.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError(
            f"block9 template changed: expected={EXPECTED_TEMPLATE_SHA256} observed={observed}"
        )
    source = payload.decode("utf-8")
    exact_replacements = (
        (
            "aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2.json",
            "aspectual_anchor_mlp4_induced_block5_crossing_factorial_v1.json",
        ),
        (
            "aspectual_anchor_mlp4_induced_l9_head_sweep_v1_result.json",
            "aspectual_anchor_mlp4_induced_final_query_onset_v2_result.json",
        ),
        (
            "run_aspectual_anchor_mlp4_induced_l9_head_sweep_v1.py",
            "run_aspectual_anchor_mlp4_induced_final_query_onset_v2.py",
        ),
        (
            "aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2_result.json",
            "aspectual_anchor_mlp4_induced_block5_crossing_factorial_v1_result.json",
        ),
        (
            "aspectual_anchor.has_vs_had.mlp4_induced_block9_crossing_factorial_v2",
            "aspectual_anchor.has_vs_had.mlp4_induced_block5_crossing_factorial_v1",
        ),
        (
            "2be16da2e0350749c6837edf43057fbcd49f776a3ba6f8099964fc022d03e5b2",
            "a821343df9a38a77df8ccaa49700b5d9b9d3e0bcfe258f7c4105ff8d7851af6a",
        ),
        (
            "0d092efa06ad697b419253262bf05db2ed2e5e2cbb0f0d3a5ff23aac9021e2b5",
            "c37d3269c4a3b7fc1163e905471804c1ed9d865f50fef0eca220f6d5e4343fd2",
        ),
        (
            "38a945b0b08d588d49e107abd19d8b09ec744c289c27859fe56a84eccbf6a126",
            "436e36dbf2ccc909ede5caae1330a4c2eb8c8da27d64590d8f1aed53891fe30e",
        ),
        (
            "if parent.get(\"terminal\") != \"null\" or parent[\"score\"][\"licensed_additional_heads\"]:\n        raise ExperimentError(\"parent null changed\")",
            "if parent.get(\"terminal\") != \"null\" or parent[\"score\"][\"first_passing_boundary\"] != 6:\n        raise ExperimentError(\"parent onset changed\")",
        ),
        (
            "pred_c = full_retained >= 0.65 and all(summaries[arm_id(FACTORS)][\"families\"][family][\"mean_recovery\"] > 0.0 and summaries[arm_id(FACTORS)][\"families\"][family][\"direction_fraction\"] >= 0.80 for family in (\"A1\", \"A2\"))",
            "pred_c = abs(values[FACTORS] - 0.05531263467112856) <= 0.01 and all(summaries[arm_id(FACTORS)][\"families\"][family][\"mean_recovery\"] >= 0.05 and summaries[arm_id(FACTORS)][\"families\"][family][\"direction_fraction\"] >= 0.80 for family in (\"A1\", \"A2\"))",
        ),
        (
            "pred_d = shapley[winner] >= 0.10 and all(drop > 0.0 for drop in family_drops.values())",
            "pred_d = winner == \"attention5\" and shapley[winner] >= 0.04 and all(drop > 0.0 for drop in family_drops.values())",
        ),
        (
            "\"pred_c_resid10_final_query_sufficiency\": pred_c",
            "\"pred_c_onset_recurrence\": pred_c",
        ),
        (
            "\"pred_d_dominant_crossing_factor\": pred_d",
            "\"pred_d_attention5_transport\": pred_d",
        ),
    )
    for old, new in exact_replacements:
        source = replace_once(source, old, new)
    for old, new in (
        ("carried9", "carried5"),
        ("resid10", "resid6"),
        ("resid9", "resid5"),
        ("attention9", "attention5"),
        ("mlp9", "mlp5"),
        ("block9", "block5"),
        ("layer == 9", "layer == 5"),
        ("transformer.h[9]", "transformer.h[5]"),
        ("range(10, 18)", "range(6, 18)"),
        ("factorial_dryrun_v2", "factorial_dryrun_v1"),
        ("factorial_result_v2", "factorial_result_v1"),
        ("final_query_block5_crossing_insufficient", "block5_onset_factorization_failed"),
        ("compile the dominant block5 crossing into the circuit", "factor the attention5 source terms"),
    ):
        if old not in source:
            raise RuntimeError(f"missing template token: {old!r}")
        source = source.replace(old, new)
    namespace = {
        "__name__": "__main__",
        "__file__": str(SCRIPT_FILE),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(source, str(SCRIPT_FILE), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
