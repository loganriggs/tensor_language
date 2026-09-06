"""Declarative factory for sealed aspectual archive/explanatory lexicon panels."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_candidate_aspectual_fresh_construction_v2 as base
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_canonical_control_v2 as canonical


def build(*, seed: int, tag: str, agents: Sequence[str], periods: Sequence[str]) -> list[dict[str, Any]]:
    if len(agents) != 16 or len(periods) != 16 or len(set(agents)) != 16 or len(set(periods)) != 16:
        raise base.CandidateBankError("lexicon factory requires 16 unique agents and periods")
    rows = []
    for group_number, (agent, period) in enumerate(zip(agents, periods)):
        forward = group_number % 2 == 0
        base_present, donor_present = (True, False) if forward else (False, True)
        direction = "present_to_past" if forward else "past_to_present"
        group_id = f"FIT:{base.canonical_sha256([base.SCHEMA, base.TASK_ID, tag, seed, group_number])[:24]}"
        sentence_types = ("present_perfect" if base_present else "past_perfect", "present_perfect" if donor_present else "past_perfect")
        common_no_vocab = dict(seed=seed, task_id=base.TASK_ID, group_number=group_number, group_id=group_id, reporter=agent, alternate_reporter=agents[(group_number + 5) % 16], adjective=tag, object_name=period, spec=base.TASK_SPEC)
        common = dict(common_no_vocab, vocabulary=base.ASPECT)
        suffix_text = f" the {agent}"
        answer = lambda present: base.ASPECT[0] if present else base.ASPECT[1]
        archive = lambda chosen_period, present: f"According to the archive, {'since' if present else 'by'} last {chosen_period} the {agent}"
        explanatory = lambda present: f"As the archive explains, {'since' if present else 'by'} last {period} the {agent}"
        rows.extend([
            builder._row(**common, transform_id="A1", construction_id="archive_evidential_prefix_anchor", direction_id=direction, matched_suffix=suffix_text, base_text=archive(period, base_present), donor_text=archive(period, donor_present), base_answer=answer(base_present), donor_answer=answer(donor_present), sentence_types=sentence_types),
            builder._row(**common, transform_id="A2", construction_id="explanatory_subordinate_prefix_anchor", direction_id=direction, matched_suffix=suffix_text, base_text=explanatory(base_present), donor_text=explanatory(donor_present), base_answer=answer(base_present), donor_answer=answer(donor_present), sentence_types=sentence_types),
            builder._row(**common, transform_id="P", construction_id="archive_evidential_prefix_same_answer", direction_id="primary_to_alternative" if forward else "alternative_to_primary", matched_suffix=suffix_text, base_text=archive(period, base_present), donor_text=archive(periods[(group_number + 7) % 16], base_present), base_answer=answer(base_present), donor_answer=answer(base_present), sentence_types=(sentence_types[0], sentence_types[0])),
            builder._row(**common_no_vocab, transform_id="C", **canonical.row_kwargs(group_number, forward)),
        ])
    return rows


def validate(rows: Sequence[Mapping[str, object]], *, seed: int, tag: str, agents: Sequence[str], periods: Sequence[str], disjoint_row_sets: Sequence[set[str]] = ()) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != build(seed=seed, tag=tag, agents=agents, periods=periods):
        raise base.CandidateBankError("rows differ from declarative lexicon authority")
    try:
        digest = battery.validate_rows(base.TASK_SPEC, materialized, required_phases=(base.SPLIT,))
    except battery.BatteryContractError as error:
        raise base.CandidateBankError(str(error)) from error
    row_ids = {str(row["row_id"]) for row in materialized}
    if len(materialized) != 64 or len(row_ids) != 64:
        raise base.CandidateBankError("lexicon authority count or uniqueness changed")
    if any(not all(row["construction_checks"].values()) for row in materialized):
        raise base.CandidateBankError("a construction check is false")
    if any(row_ids & prior_ids for prior_ids in disjoint_row_sets):
        raise base.CandidateBankError("lexicon authority overlaps a prior row-ID set")
    return digest
