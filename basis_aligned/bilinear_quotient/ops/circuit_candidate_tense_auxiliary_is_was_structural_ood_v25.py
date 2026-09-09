#!/usr/bin/env python3
"""History-disjoint, eight-structure v25 authority for is/was OOD tests."""
from __future__ import annotations

import importlib
from typing import Any, Mapping, Sequence

import circuit_candidate_aspectual_different_readout_is_was_v1 as base


SCHEMA, SPLIT, GROUPS = base.SCHEMA, base.SPLIT, 16
SEED = 20261325
TASK_ID, READOUT = base.TASK_ID, base.READOUT
canonical_sha256, CandidateBankError = base.canonical_sha256, base.CandidateBankError
TASK_SPEC = base.battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_structure_diverse_is_was_panels",
    answer_role="score_jointly_tokenized_is_versus_was",
    transforms=(
        base.battery.TransformSpec("A1", "four_structure_temporal_swap_set_one", True, "toward_donor"),
        base.battery.TransformSpec("A2", "four_structure_temporal_swap_set_two", True, "toward_donor"),
        base.battery.TransformSpec("P", "structure_matched_same_tense_paraphrase", False, "invariant"),
        base.battery.TransformSpec("C", "structure_matched_unrelated_location_swap", False, "registered_active"),
    ),
)
_REPORTERS = (
    "field conservator", "medical radiographer", "senior adjudicator", "stage dramaturg",
    "artisan perfumer", "forensic phonetician", "court mediator", "head sommelier",
    "museum registrar", "archive custodian", "instrument technician", "policy analyst",
    "safety inspector", "research coordinator", "project evaluator", "records specialist",
)
STRUCTURES = (
    "fronted_era", "post_subject_time", "relative_clause_time", "subordinate_clause_time",
    "reported_source_time", "long_coordinated_prefix", "postnominal_time", "embedded_predicate_time",
)


def _answer(present):
    return READOUT[0] if present else READOUT[1]


def _a1(structure, reporter, present):
    if structure == 0:
        return f"In the {'modern' if present else 'ancient'} era, the {reporter}", f" the {reporter}"
    if structure == 1:
        return (f"The {reporter} working in the studio {'now' if present else 'then'}, according to the record",
                ", according to the record")
    if structure == 2:
        return (f"The instrument that the {reporter} examined {'today' if present else 'yesterday'}, according to the report",
                ", according to the report")
    return (f"Although the protocol {'has' if present else 'had'} changed, "
            f"{'today' if present else 'then'} the {reporter}", f" the {reporter}")


def _p(structure, reporter, present):
    if structure == 0:
        return f"In the {'current' if present else 'former'} era, the {reporter}", f" the {reporter}"
    if structure == 1:
        return (f"The {reporter} working in the studio {'today' if present else 'earlier'}, according to the record",
                ", according to the record")
    if structure == 2:
        return (f"The instrument that the {reporter} examined {'recently' if present else 'previously'}, according to the report",
                ", according to the report")
    return (f"Although the procedure {'has' if present else 'had'} changed, "
            f"{'today' if present else 'then'} the {reporter}", f" the {reporter}")


def _a2(structure, reporter, present):
    if structure == 0:
        period = "today's" if present else "the old"
        return (f"According to {period} report, the {reporter}",
                f" the {reporter}")
    if structure == 1:
        return ("After reviewing the evidence and consulting the team, "
                f"{'today' if present else 'then'} the {reporter}", f" the {reporter}")
    if structure == 2:
        return (f"The central question for the {reporter} {'today' if present else 'yesterday'}, by all accounts",
                ", by all accounts")
    return (f"What the {reporter} "
            f"{'describes as current' if present else 'described as former'}, in the official summary",
            ", in the official summary")


def _c(structure, reporter, present, location):
    if structure == 0:
        period = "today's" if present else "the old"
        return (f"According to {period} report from the {location}, the {reporter}",
                f" the {reporter}")
    if structure == 1:
        return (f"After reviewing the evidence near the {location} and consulting the team, "
                f"{'today' if present else 'then'} the {reporter}", f" the {reporter}")
    if structure == 2:
        return (f"The central question for the {reporter} near the {location} "
                f"{'today' if present else 'yesterday'}, by all accounts", ", by all accounts")
    return (f"What the {reporter} at the {location} "
            f"{'describes as current' if present else 'described as former'}, in the official summary",
            ", in the official summary")


def _panel(group_number: int) -> list[dict[str, Any]]:
    reporter = _REPORTERS[group_number]
    alternate = _REPORTERS[(group_number + 5) % GROUPS]
    structure = group_number % 4
    forward = (group_number // 4) % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'structural_ood_v25', SEED, group_number])[:24]}"
    sentence_types = (
        "present_progressive" if base_present else "past_progressive",
        "present_progressive" if donor_present else "past_progressive",
    )
    common = dict(
        seed=SEED, task_id=TASK_ID, group_number=group_number, group_id=group_id,
        reporter=reporter, alternate_reporter=alternate, adjective="structural_ood_v25",
        object_name=STRUCTURES[structure] + "__" + STRUCTURES[structure + 4],
        spec=TASK_SPEC, vocabulary=READOUT,
    )
    a1_base, a1_suffix = _a1(structure, reporter, base_present)
    a1_donor, _ = _a1(structure, reporter, donor_present)
    a2_base, a2_suffix = _a2(structure, reporter, base_present)
    a2_donor, _ = _a2(structure, reporter, donor_present)
    p_donor, _ = _p(structure, reporter, base_present)
    base_location, donor_location = (("harbor", "canyon") if forward
                                     else ("canyon", "harbor"))
    c_base, c_suffix = _c(structure, reporter, base_present, base_location)
    c_donor, _ = _c(structure, reporter, base_present, donor_location)
    rows = [
        base.builder._row(
            **common, transform_id="A1", construction_id=STRUCTURES[structure] + "_v25",
            direction_id=direction, matched_suffix=a1_suffix, base_text=a1_base,
            donor_text=a1_donor, base_answer=_answer(base_present),
            donor_answer=_answer(donor_present), sentence_types=sentence_types),
        base.builder._row(
            **common, transform_id="A2", construction_id=STRUCTURES[structure + 4] + "_v25",
            direction_id=direction, matched_suffix=a2_suffix, base_text=a2_base,
            donor_text=a2_donor, base_answer=_answer(base_present),
            donor_answer=_answer(donor_present), sentence_types=sentence_types),
        base.builder._row(
            **common, transform_id="P", construction_id=STRUCTURES[structure] + "_paraphrase_v25",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=a1_suffix, base_text=a1_base, donor_text=p_donor,
            base_answer=_answer(base_present), donor_answer=_answer(base_present),
            sentence_types=(sentence_types[0], sentence_types[0])),
        base.builder._row(
            **common, transform_id="C", construction_id=STRUCTURES[structure + 4] + "_location_control_v25",
            direction_id="harbor_to_canyon" if forward else "canyon_to_harbor",
            matched_suffix=c_suffix, base_text=c_base, donor_text=c_donor,
            base_answer=_answer(base_present), donor_answer=_answer(base_present),
            sentence_types=(sentence_types[0], sentence_types[0])),
    ]
    if any(row["base_semantic_position"] != row["donor_semantic_position"] for row in rows):
        raise CandidateBankError("v25 requires pairwise position-compatible rows")
    return rows


def _build():
    return [row for group_number in range(GROUPS) for row in _panel(group_number)]


def _history_modules():
    modules = [importlib.import_module(f"circuit_candidate_aspectual_different_readout_is_was_v{i}")
               for i in range(1, 4)]
    modules.extend(importlib.import_module(
        f"circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v{i}") for i in range(4, 25))
    return tuple(modules)


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed v25 authority")
    try:
        digest = base.battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except base.battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    old_banks = tuple(module._build() if hasattr(module, "_build") else module.build_rows()
                      for module in _history_modules())
    old_ids = {str(row["row_id"]) for bank in old_banks for row in bank}
    old_reporters = {str(row.get("reporter")) for bank in old_banks for row in bank}
    old_text = {row[key] for bank in old_banks for row in bank
                for key in ("base_text", "donor_text")}
    ids = {str(row["row_id"]) for row in materialized}
    construction_ids = {str(row["construction_id"]) for row in materialized}
    lengths = [len(row["base_ids"]) for row in materialized]
    if (len(materialized) != 64 or len(ids) != 64 or ids & old_ids
            or set(_REPORTERS) & old_reporters
            or any(row[key] in old_text for row in materialized for key in ("base_text", "donor_text"))
            or any(not all(row["construction_checks"].values()) for row in materialized)
            or any(row["base_semantic_position"] != row["donor_semantic_position"] for row in materialized)
            or len(construction_ids) != 16 or min(lengths) < 8 or max(lengths) < 15):
        raise CandidateBankError("v25 count, history, structure, alignment, or length check failed")
    return digest


def build_rows(task_id: str = TASK_ID):
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    rows = _build()
    _validate(rows)
    return rows


def validate_rows(rows, *, task_id: str = TASK_ID):
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    return _validate(rows)


def authority_sha256(task_id: str = TASK_ID):
    return validate_rows(build_rows(task_id), task_id=task_id)


if __name__ == "__main__":
    rows = build_rows()
    print("authority:", authority_sha256())
    for row in rows[:16]:
        print(row["family"], row["construction_id"], row["base_text"], "->", row["donor_text"])
