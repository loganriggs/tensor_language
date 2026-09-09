#!/usr/bin/env python3
"""Held-out rows for the four v25-native-capable structural is/was families."""
from __future__ import annotations

from collections import Counter

import circuit_candidate_tense_auxiliary_is_was_structural_ood_v25 as v25


SCHEMA, SPLIT, GROUPS = v25.SCHEMA, v25.SPLIT, 16
SEED = 20261426
TASK_ID, TASK_SPEC, READOUT = v25.TASK_ID, v25.TASK_SPEC, v25.READOUT
canonical_sha256, CandidateBankError = v25.canonical_sha256, v25.CandidateBankError
_REPORTERS = (
    "clinical conservator", "diagnostic radiographer", "chief adjudicator", "resident dramaturg",
    "botanical perfumer", "acoustic phonetician", "civic mediator", "wine sommelier",
    "collection registrar", "document custodian", "calibration technician", "fiscal analyst",
    "compliance inspector", "study coordinator", "grant evaluator", "data specialist",
)
TARGET_STRUCTURES = ("fronted_era", "subordinate_clause_time",
                     "reported_source_time", "postnominal_time")


def _answer(present):
    return READOUT[0] if present else READOUT[1]


def _panel(group_number):
    reporter = _REPORTERS[group_number]
    alternate = _REPORTERS[(group_number + 5) % GROUPS]
    variant = group_number % 2
    a1_structure = (0, 3)[variant]
    a2_structure = (0, 2)[variant]
    forward = (group_number // 2) % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    sentence_types = (
        "present_progressive" if base_present else "past_progressive",
        "present_progressive" if donor_present else "past_progressive",
    )
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'structural_holdout_v26', SEED, group_number])[:24]}"
    common = dict(
        seed=SEED, task_id=TASK_ID, group_number=group_number, group_id=group_id,
        reporter=reporter, alternate_reporter=alternate, adjective="structural_holdout_v26",
        object_name=TARGET_STRUCTURES[variant] + "__" + TARGET_STRUCTURES[variant + 2],
        spec=TASK_SPEC, vocabulary=READOUT,
    )
    a1_base, a1_suffix = v25._a1(a1_structure, reporter, base_present)
    a1_donor, _ = v25._a1(a1_structure, reporter, donor_present)
    a2_base, a2_suffix = v25._a2(a2_structure, reporter, base_present)
    a2_donor, _ = v25._a2(a2_structure, reporter, donor_present)
    p_donor, _ = v25._p(a1_structure, reporter, base_present)
    base_location, donor_location = (("harbor", "canyon") if forward
                                     else ("canyon", "harbor"))
    c_base, c_suffix = v25._c(a2_structure, reporter, base_present, base_location)
    c_donor, _ = v25._c(a2_structure, reporter, base_present, donor_location)
    a1_name = TARGET_STRUCTURES[variant] + "_holdout_v26"
    a2_name = TARGET_STRUCTURES[variant + 2] + "_holdout_v26"
    rows = [
        v25.base.builder._row(
            **common, transform_id="A1", construction_id=a1_name, direction_id=direction,
            matched_suffix=a1_suffix, base_text=a1_base, donor_text=a1_donor,
            base_answer=_answer(base_present), donor_answer=_answer(donor_present),
            sentence_types=sentence_types),
        v25.base.builder._row(
            **common, transform_id="A2", construction_id=a2_name, direction_id=direction,
            matched_suffix=a2_suffix, base_text=a2_base, donor_text=a2_donor,
            base_answer=_answer(base_present), donor_answer=_answer(donor_present),
            sentence_types=sentence_types),
        v25.base.builder._row(
            **common, transform_id="P", construction_id=a1_name + "_paraphrase",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=a1_suffix, base_text=a1_base, donor_text=p_donor,
            base_answer=_answer(base_present), donor_answer=_answer(base_present),
            sentence_types=(sentence_types[0], sentence_types[0])),
        v25.base.builder._row(
            **common, transform_id="C", construction_id=a2_name + "_location_control",
            direction_id="harbor_to_canyon" if forward else "canyon_to_harbor",
            matched_suffix=c_suffix, base_text=c_base, donor_text=c_donor,
            base_answer=_answer(base_present), donor_answer=_answer(base_present),
            sentence_types=(sentence_types[0], sentence_types[0])),
    ]
    if any(row["base_semantic_position"] != row["donor_semantic_position"] for row in rows):
        raise CandidateBankError("v26 requires pairwise position-compatible rows")
    return rows


def _build():
    return [row for group_number in range(GROUPS) for row in _panel(group_number)]


def _validate(rows):
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed v26 authority")
    try:
        digest = v25.base.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except v25.base.battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    old_modules = v25._history_modules() + (v25,)
    old_rows = [row for module in old_modules
                for row in (module._build() if hasattr(module, "_build") else module.build_rows())]
    old_ids = {str(row["row_id"]) for row in old_rows}
    old_reporters = {str(row.get("reporter")) for row in old_rows}
    old_text = {row[key] for row in old_rows for key in ("base_text", "donor_text")}
    counts = Counter(row["construction_id"] for row in materialized)
    if (len(materialized) != 64 or len({row["row_id"] for row in materialized}) != 64
            or set(_REPORTERS) & old_reporters or {row["row_id"] for row in materialized} & old_ids
            or any(row[key] in old_text for row in materialized for key in ("base_text", "donor_text"))
            or len(counts) != 8 or set(counts.values()) != {8}
            or any(not all(row["construction_checks"].values()) for row in materialized)
            or any(row["base_semantic_position"] != row["donor_semantic_position"] for row in materialized)):
        raise CandidateBankError("v26 count, history, structure, or alignment check failed")
    return digest


def build_rows(task_id=TASK_ID):
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    rows = _build()
    _validate(rows)
    return rows


def validate_rows(rows, *, task_id=TASK_ID):
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    return _validate(rows)


def authority_sha256(task_id=TASK_ID):
    return validate_rows(build_rows(task_id), task_id=task_id)


if __name__ == "__main__":
    rows = build_rows()
    print("authority:", authority_sha256())
    for row in rows[:8]:
        print(row["family"], row["construction_id"], row["base_text"], "->", row["donor_text"])
