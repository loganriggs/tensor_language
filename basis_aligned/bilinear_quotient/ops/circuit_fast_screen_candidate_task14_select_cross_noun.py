#!/usr/bin/env python3
# BQLANE: cpu
"""Alternative SELECT donor profile for the fixed Task14 carrier sites.

Targets and donors differ in noun group and syntax while preserving target
subject-number direction and attractor plurality. Site selection remains FIT-only.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import circuit_fast_screen_candidate_task14_select_cross_syntax as matched


ROOT = Path(__file__).resolve().parent.parent
TASK_ID = matched.TASK_ID
PHASE = matched.PHASE
PARTITION = matched.PARTITION
VALIDATION_SCOPE = "unseen_nouns_templates_and_cross_noun_donors_after_fit_site_selection"
SCHEMA = "task14_select_cross_noun_cross_syntax_authority_v1"
SITE_IDS = matched.SITE_IDS
BATCH_SIZE = matched.BATCH_SIZE
MIN_NATIVE_CELL_ACCURACY = matched.MIN_NATIVE_CELL_ACCURACY
MIN_CELL_DIRECTION_FRACTION = matched.MIN_CELL_DIRECTION_FRACTION
MIN_CELL_MEAN_RECOVERY = matched.MIN_CELL_MEAN_RECOVERY
MIN_DONOR_DENOMINATOR = matched.MIN_DONOR_DENOMINATOR
EXPECTED_SOURCE_SHA256 = dict(matched.EXPECTED_SOURCE_SHA256)
EXPECTED_SOURCE_SHA256.update({
    "matched_select_result": "dbe254e8a62d2180d6d1901e6b917d56d9dd8cd8bf37cec27c4579c9d9427eb2",
})


class SelectCrossNounAuthorityError(ValueError):
    pass


def _panels() -> dict[str, dict[str, dict]]:
    result_path = ROOT / (
        "circuits/fast_screens/"
        "task14_subject_verb_agreement_select_cross_syntax_v1_result.json"
    )
    if hashlib.sha256(result_path.read_bytes()).hexdigest() != \
            EXPECTED_SOURCE_SHA256["matched_select_result"]:
        raise SelectCrossNounAuthorityError("matched SELECT result changed")
    output: dict[str, dict[str, dict]] = {}
    for row in matched._source_rows():
        output.setdefault(str(row["group_id"]), {})[str(row["transform_id"])] = row
    if len(output) != 32 or any(set(panel) != {"A1", "A2", "P", "C"}
                                for panel in output.values()):
        raise SelectCrossNounAuthorityError("SELECT panels changed")
    return output


def _cross_noun_pairing(panels: Mapping[str, Mapping[str, object]]) -> dict[str, str]:
    strata: dict[tuple[str, bool], list[tuple[int, str]]] = defaultdict(list)
    for group_id, panel in panels.items():
        a1 = panel["A1"]
        key = (str(a1["base_subject_number"]), bool(a1["base_attractor_plural"]))
        strata[key].append((int(a1["group_number"]), group_id))
    if set(len(groups) for groups in strata.values()) != {8} or len(strata) != 4:
        raise SelectCrossNounAuthorityError("subject/attractor strata changed")
    pairing = {}
    for groups in strata.values():
        ordered = [group_id for _number, group_id in sorted(groups)]
        for index, group_id in enumerate(ordered):
            pairing[group_id] = ordered[(index + 1) % len(ordered)]
    return pairing


def _build_rows_unvalidated() -> list[dict]:
    panels = _panels()
    pairing = _cross_noun_pairing(panels)
    output = []
    for target_group_id, target_panel in sorted(
        panels.items(), key=lambda item: int(item[1]["A1"]["group_number"]),
    ):
        donor_group_id = pairing[target_group_id]
        donor_panel = panels[donor_group_id]
        for target_family, donor_family in (("A1", "A2"), ("A2", "A1")):
            target = matched._endpoint(target_panel[target_family], "base")
            donor = matched._endpoint(donor_panel[donor_family], "donor")
            target_syntax = "pp" if target_family == "A1" else "relative"
            donor_syntax = "pp" if donor_family == "A1" else "relative"
            cell_id = (
                f"{target_syntax}_{target['subject_number']}_to_"
                f"{donor_syntax}_{donor['subject_number']}"
            )
            identity = [SCHEMA, target_group_id, donor_group_id, target_family, donor_family]
            output.append({
                "schema": SCHEMA, "task_id": TASK_ID, "split": PHASE,
                "partition": PARTITION, "validation_scope": VALIDATION_SCOPE,
                "row_id": matched.canonical_sha256(identity),
                "group_id": target_group_id,
                "target_group_id": target_group_id, "donor_group_id": donor_group_id,
                "cell_id": cell_id,
                "target_family": target_family, "donor_family": donor_family,
                "target_endpoint_id": target["endpoint_id"],
                "donor_endpoint_id": donor["endpoint_id"],
                "base_ids": target["ids"], "donor_ids": donor["ids"],
                "base_text": target["text"], "donor_text": donor["text"],
                "base_answer_id": target["answer_id"], "base_foil_id": target["foil_id"],
                "donor_answer_id": donor["answer_id"], "donor_foil_id": donor["foil_id"],
                "base_semantic_position": target["position"],
                "donor_semantic_position": donor["position"],
                "base_subject_number": target["subject_number"],
                "donor_subject_number": donor["subject_number"],
                "base_head_pair": target["head_pair"],
                "donor_head_pair": donor["head_pair"],
                "base_attractor_plural": target["attractor_plural"],
                "donor_attractor_plural": donor["attractor_plural"],
                "expected_effect": "toward_opposite_number_cross_syntax_cross_noun_donor",
            })
    return output


def build_rows() -> list[dict]:
    rows = _build_rows_unvalidated()
    validate_rows(rows)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if len(materialized) != 64 or len({row.get("row_id") for row in materialized}) != 64:
        raise SelectCrossNounAuthorityError("authority must contain 64 unique rows")
    cells: dict[str, int] = {}
    for row in materialized:
        if row.get("schema") != SCHEMA or row.get("split") != PHASE \
                or row.get("partition") != PARTITION \
                or row.get("validation_scope") != VALIDATION_SCOPE:
            raise SelectCrossNounAuthorityError("row identity changed")
        if row.get("target_group_id") == row.get("donor_group_id") \
                or row.get("base_head_pair") == row.get("donor_head_pair"):
            raise SelectCrossNounAuthorityError("donor is not cross-noun")
        if row.get("base_attractor_plural") != row.get("donor_attractor_plural"):
            raise SelectCrossNounAuthorityError("attractor plurality is not matched")
        if row.get("base_subject_number") == row.get("donor_subject_number") \
                or row.get("base_answer_id") != row.get("donor_foil_id") \
                or row.get("base_foil_id") != row.get("donor_answer_id"):
            raise SelectCrossNounAuthorityError("subject-number orientation is not reversed")
        if {row.get("target_family"), row.get("donor_family")} != {"A1", "A2"}:
            raise SelectCrossNounAuthorityError("syntax is not crossed")
        cells[str(row["cell_id"])] = cells.get(str(row["cell_id"]), 0) + 1
    if len(cells) != 4 or set(cells.values()) != {16}:
        raise SelectCrossNounAuthorityError(f"direction-cell balance changed: {cells}")
    if matched.canonical_sha256(materialized) != \
            matched.canonical_sha256(_build_rows_unvalidated()):
        raise SelectCrossNounAuthorityError("rows differ from exact regenerated authority")
    return matched.canonical_sha256(materialized)


def compile_plan(rows: Sequence[Mapping[str, object]] | None = None) -> dict:
    materialized = build_rows() if rows is None else [dict(row) for row in rows]
    digest = validate_rows(materialized)
    calls = []
    for side in ("base", "donor"):
        for start in range(0, len(materialized), BATCH_SIZE):
            calls.append({"kind": "native", "side": side,
                          "row_ids": [row["row_id"] for row in materialized[start:start+BATCH_SIZE]]})
    for site_id in SITE_IDS:
        for start in range(0, len(materialized), BATCH_SIZE):
            calls.append({"kind": "exact_single_position_interchange", "site_id": site_id,
                          "row_ids": [row["row_id"] for row in materialized[start:start+BATCH_SIZE]]})
    plan = {
        "schema": "task14_targeted_cross_syntax_plan_v1", "phase": PHASE,
        "partition": PARTITION, "validation_scope": VALIDATION_SCOPE,
        "authority_sha256": digest, "source_sha256": EXPECTED_SOURCE_SHA256,
        "site_ids": list(SITE_IDS), "row_count": 64, "batch_size": BATCH_SIZE,
        "calls": calls,
        "price": {"forward_calls": 8, "example_evaluations": 256,
                  "backward_calls": 0, "model_updates": 0,
                  "raw_numeric_evidence_bytes": 2048},
        "score": {"minimum_native_cell_accuracy": MIN_NATIVE_CELL_ACCURACY,
                  "minimum_cell_direction_fraction": MIN_CELL_DIRECTION_FRACTION,
                  "minimum_cell_mean_recovery": MIN_CELL_MEAN_RECOVERY},
        "correction": (
            "This profile changes the matched-noun SELECT donor rule to a frozen cyclic "
            "cross-noun donor within subject-number and attractor-plurality strata."
        ),
    }
    plan["compiled_sha256"] = matched.canonical_sha256(plan)
    return plan


if __name__ == "__main__":
    import json
    print(json.dumps(compile_plan(), sort_keys=True))
