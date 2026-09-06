#!/usr/bin/env python3
"""Hash-bound prospective holdout specialization of the exact block9 factorial."""

from __future__ import annotations

import hashlib
from pathlib import Path


SCRIPT_FILE = Path(globals()["__file__"]).resolve()
ROOT = SCRIPT_FILE.parents[1]
TEMPLATE = ROOT / "ops/run_aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2.py"
EXPECTED_TEMPLATE_SHA256 = "54db17a05a70deb41e762c77eb342fdc522a3849981f66507615e21ab06743ea"
STATIC_PREDICATES = {
    "pred_a_exact_crossing_instrument": "bound authority/capability and exact closure",
    "pred_b_writer_recurrence": "prospective writer recurrence",
    "pred_c_resid10_final_query_sufficiency": "prospective resid10 crossing",
    "pred_d_attention9_dominance": "prospective attention9 dominance and necessity",
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
    replacements = (
        (
            "from datetime import datetime, timezone",
            "from dataclasses import replace\nfrom datetime import datetime, timezone",
        ),
        (
            "import circuit_fast_screen_candidate_aspectual as candidate",
            "import circuit_battery_integration_contract as battery\nimport circuit_candidate_aspectual_lexical_holdout_v5 as candidate",
        ),
        (
            "aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2.json",
            "aspectual_anchor_block9_crossing_factorial_lexical_holdout_v1.json",
        ),
        (
            "aspectual_anchor_mlp4_induced_l9_head_sweep_v1_result.json",
            "aspectual_anchor_blocks6_8_crossing_factorials_lexical_holdout_v1_result.json",
        ),
        (
            "run_aspectual_anchor_mlp4_induced_l9_head_sweep_v1.py",
            "run_aspectual_anchor_blocks6_8_crossing_factorials_lexical_holdout_v1.py",
        ),
        (
            "aspectual_anchor_mlp4_induced_block9_crossing_factorial_v2_result.json",
            "aspectual_anchor_block9_crossing_factorial_lexical_holdout_v1_result.json",
        ),
        (
            "aspectual_anchor.has_vs_had.mlp4_induced_block9_crossing_factorial_v2",
            "aspectual_anchor.has_vs_had.block9_crossing_factorial_lexical_holdout_v1",
        ),
        (
            "2be16da2e0350749c6837edf43057fbcd49f776a3ba6f8099964fc022d03e5b2",
            "87711cda73b8fdbf13351169ac48faa156ff4eaa9c18c8ac9fb1a652ab99c2af",
        ),
        (
            "0d092efa06ad697b419253262bf05db2ed2e5e2cbb0f0d3a5ff23aac9021e2b5",
            "0c440a3f4e22941e9d2be40d02df0f1262c08ca646f0d7939206bf5a6cfaf932",
        ),
        (
            "38a945b0b08d588d49e107abd19d8b09ec744c289c27859fe56a84eccbf6a126",
            "5dd175fb2222870ba156502f6188cca6138b58878b202b4895a73c93c8ccd7e3",
        ),
        (
            "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3",
            "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493",
        ),
        ("EXAMPLE_EVALUATIONS_MAX = 896", "EXAMPLE_EVALUATIONS_MAX = 448"),
        (
            "abs(writer_target - 0.33379277118533013) <= 0.02",
            "abs(writer_target - 0.2835613798233539) <= 0.01",
        ),
        (
            "pred_d = shapley[winner] >= 0.10 and all(drop > 0.0 for drop in family_drops.values())",
            "pred_d = winner == \"attention9\" and shapley[winner] >= 0.08 and all(drop > 0.0 for drop in family_drops.values())",
        ),
        (
            '"pred_d_' + 'dominant_crossing_factor": pred_d',
            '"pred_d_' + 'attention9_dominance": pred_d',
        ),
        (
            "aspectual_anchor_mlp4_induced_block9_crossing_factorial_dryrun_v2",
            "aspectual_anchor_block9_crossing_factorial_lexical_holdout_dryrun_v1",
        ),
        (
            "aspectual_anchor_mlp4_induced_block9_crossing_factorial_result_v2",
            "aspectual_anchor_block9_crossing_factorial_lexical_holdout_result_v1",
        ),
        (
            '"block9_route_into_resid10_carrier"',
            '"prospective_block9_route_into_resid10_carrier"',
        ),
        (
            '"compile the dominant block9 crossing into the circuit"',
            '"compile the prospectively validated full path into typed program v2"',
        ),
    )
    for old, new in replacements:
        source = replace_once(source, old, new)

    start_marker = "def validate_static():\n"
    end_marker = "\n\nclass CrossingBackend"
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise RuntimeError("template validation markers changed")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    validation = '''def validate_static():
    for file_path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256,
        PARENT: EXPECTED_PARENT_SHA256,
        PARENT_RUNNER: EXPECTED_PARENT_RUNNER_SHA256,
    }.items():
        if sha256(file_path) != digest:
            raise ExperimentError(f"authority hash changed: {file_path.name}")
    prior = json.loads(PRIOR.read_text())
    parent = json.loads(PARENT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if parent.get("terminal") != "screen" or parent["score"]["full_crossing_curve"]["resid9"] != 0.0876184706243302:
        raise ExperimentError("prospective resid9 parent changed")
    rows_all = candidate.build_rows()
    if candidate.validate_rows(rows_all) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("row authority changed")
    selected = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    parent_rows = parent_runner.candidate.build_rows(parent_runner.candidate.TASK_ID)
    parent_spec = parent_runner.build_spec(parent_rows)
    spec = replace(
        parent_spec,
        experiment_id="aspectual-anchor-block9-crossing-factorial-lexical-holdout-v1",
        authority_sha256=EXPECTED_AUTHORITY_SHA256,
        expected_fit_rows=len(rows_all),
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT", forward_calls=MODEL_FORWARDS_MAX,
            example_evaluations=EXAMPLE_EVALUATIONS_MAX,
            backward_calls=0, model_updates=0, evidence_bytes=65536,
        ),
    )
    enriched_all = screen.validate_fit_authority(spec, rows_all)
    enriched = tuple(enriched_all[str(row["row_id"])] for row in selected)
    if len(rows_all) != 64 or len(enriched) != 32 or len(subsets()) != 8:
        raise ExperimentError("population or factorial changed")
    return enriched, spec
'''
    source = source[:start] + validation + source[end:]
    namespace = {
        "__name__": "__main__",
        "__file__": str(SCRIPT_FILE),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(source, str(SCRIPT_FILE), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
