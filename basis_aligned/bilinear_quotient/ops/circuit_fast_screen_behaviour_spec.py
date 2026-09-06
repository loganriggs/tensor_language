#!/usr/bin/env python3
"""Declare a fast-screen behaviour in ~25 lines instead of ~190.

Every candidate this lane has authored since `sentence_terminal` shares one skeleton: two target
constructions differing in a single cue, an answer-preserving edit, the canonical same-answer
control, and a matched final token per row. The behaviour-specific content is about 25 lines --
two answer tokens, two frame builders, a P variant, and some labels. The other ~165 lines are
copied, and copying them by textual substitution has cost roughly two name defects per behaviour
(`_ALTERNATE_AGENTS` vs `_AGENTS_ALT`, tables removed while still referenced, builders renamed in
one place and not another).

An earlier attempt at this failed because it did meta-programming on module source. This does
not: it is an ordinary dataclass plus one shared panel/validate implementation, which is the same
factoring Codex applied to the numeric-sequence candidates.

Usage -- the whole of a new behaviour:

    SPEC = BehaviourSpec(
        task_id="my_behaviour.a_vs_b",
        vocabulary=(" a", " b"),
        generator_role="generate_linked_my_behaviour_fit_panels",
        answer_role="score_jointly_tokenized_a_versus_b",
        a1=Family("bare_frame", "bare_frame_swap", lambda i, pos: f"..."),
        a2=Family("report_frame", "report_frame_swap", lambda i, pos: f"..."),
        p_donor=lambda i, pos: f"...",          # answer-preserving variant of A1
        a1_suffix=lambda i: " shared tail",     # must end BOTH sides of the A1/P rows
        a2_suffix=lambda i: " shared tail",
        directions=("forward_to_reverse", "reverse_to_forward"),
        kinds=("positive", "negative"),
    )
    TASK_ID, TASK_SPEC = SPEC.task_id, SPEC.battery_spec()
    build_rows, validate_rows, authority_sha256 = SPEC.api()

Design invariants the shared implementation enforces for you, both of which cost this lane
repeated defects (see ops/README.md): P never varies the final input token, because `p_donor`
must produce a string ending in the same `a1_suffix`; and A2 carries its own suffix rather than
inheriting A1's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_canonical_control_v2 as canonical
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_candidates as lex

SCHEMA = builder.SCHEMA
SPLIT = builder.SPLIT
DEFAULT_GROUPS = builder.DEFAULT_GROUPS
DEFAULT_SEED = builder.DEFAULT_SEED
CandidateBankError = builder.CandidateBankError
canonical_sha256 = builder.canonical_sha256


@dataclass(frozen=True)
class Family:
    """One target construction: its cell label, its transform role, and its text builder."""
    construction_id: str
    generator_role: str
    text: Callable[[int, bool], str]


@dataclass(frozen=True)
class BehaviourSpec:
    task_id: str
    vocabulary: tuple[str, str]
    generator_role: str
    answer_role: str
    a1: Family
    a2: Family
    p_donor: Callable[[int, bool], str]
    a1_suffix: Callable[[int], str]
    a2_suffix: Callable[[int], str]
    directions: tuple[str, str]
    kinds: tuple[str, str]
    p_generator_role: str = "answer_preserving_rewrite"

    def answer(self, positive: bool) -> str:
        return self.vocabulary[0] if positive else self.vocabulary[1]

    def battery_spec(self) -> battery.BatteryTaskSpec:
        return battery.BatteryTaskSpec(
            task_id=self.task_id,
            generator_role=self.generator_role,
            answer_role=self.answer_role,
            transforms=(
                battery.TransformSpec("A1", self.a1.generator_role, True, "toward_donor"),
                battery.TransformSpec("A2", self.a2.generator_role, True, "toward_donor"),
                battery.TransformSpec("P", self.p_generator_role, False, "invariant"),
                canonical.TRANSFORM,
            ),
        )

    # ---- row construction -------------------------------------------------

    def _panel(self, seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
        spec = self.battery_spec()
        group_id = f"FIT:{canonical_sha256([SCHEMA, self.task_id, seed, group_number])[:24]}"
        forward = group_number % 2 == 0
        base_pos, donor_pos = (True, False) if forward else (False, True)
        direction = self.directions[0] if forward else self.directions[1]
        kind = lambda pos: self.kinds[0] if pos else self.kinds[1]
        common = dict(seed=seed, task_id=self.task_id, group_number=group_number,
                      group_id=group_id, reporter=lex._REPORTERS[case_index][0],
                      alternate_reporter=lex._REPORTERS[case_index][1],
                      adjective=lex._ADJECTIVES[case_index], object_name=lex._OBJECTS[case_index],
                      spec=spec)
        vocab = dict(common, vocabulary=self.vocabulary)
        return [
            builder._row(**vocab, transform_id="A1", construction_id=self.a1.construction_id,
                         direction_id=direction, matched_suffix=self.a1_suffix(case_index),
                         base_text=self.a1.text(case_index, base_pos),
                         donor_text=self.a1.text(case_index, donor_pos),
                         base_answer=self.answer(base_pos), donor_answer=self.answer(donor_pos),
                         sentence_types=(kind(base_pos), kind(donor_pos))),
            builder._row(**vocab, transform_id="A2", construction_id=self.a2.construction_id,
                         direction_id=direction, matched_suffix=self.a2_suffix(case_index),
                         base_text=self.a2.text(case_index, base_pos),
                         donor_text=self.a2.text(case_index, donor_pos),
                         base_answer=self.answer(base_pos), donor_answer=self.answer(donor_pos),
                         sentence_types=(kind(base_pos), kind(donor_pos))),
            builder._row(**vocab, transform_id="P",
                         construction_id=f"{self.a1.construction_id}_{kind(base_pos)}",
                         direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                         matched_suffix=self.a1_suffix(case_index),
                         base_text=self.a1.text(case_index, base_pos),
                         donor_text=self.p_donor(case_index, base_pos),
                         base_answer=self.answer(base_pos), donor_answer=self.answer(base_pos),
                         sentence_types=(kind(base_pos), kind(base_pos))),
            builder._row(**common, transform_id="C",
                         **canonical.row_kwargs(case_index, forward)),
        ]

    def build(self, groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
        order = lex._permutation(seed)
        return [row for n in range(groups) for row in self._panel(seed, n, order[n])]

    def validate(self, rows: Sequence[Mapping[str, object]], groups: int = DEFAULT_GROUPS,
                 seed: int = DEFAULT_SEED) -> str:
        if type(groups) is not int or not 2 <= groups <= DEFAULT_GROUPS or groups % 2:
            raise CandidateBankError("groups must be an even integer from 2 through 32")
        materialized = [dict(row) for row in rows]
        if materialized != self.build(groups, seed):
            raise CandidateBankError("rows differ from the deterministic semantic authority")
        try:
            digest = battery.validate_rows(self.battery_spec(), materialized,
                                           required_phases=(SPLIT,))
        except battery.BatteryContractError as error:
            raise CandidateBankError(str(error)) from error
        if len({str(r["row_id"]) for r in materialized}) != len(materialized):
            raise CandidateBankError("row IDs are not unique")
        cells: dict[tuple[str, str], int] = {}
        for row in materialized:
            if not all(row["construction_checks"].values()):
                raise CandidateBankError("a stored construction check is false")
            key = (str(row["transform_id"]), str(row["direction_id"]))
            cells[key] = cells.get(key, 0) + 1
        half = groups // 2
        for transform in ("A1", "A2"):
            for label in self.directions:
                if cells.get((transform, label)) != half:
                    raise CandidateBankError(f"{transform} ordered directions are unbalanced")
        for label in ("base_to_donor", "donor_to_base"):
            if cells.get(("C", label)) != half:
                raise CandidateBankError("C ordered directions are unbalanced")
        return digest

    def api(self):
        """The three functions the managed runner imports from a candidate module."""
        def build_rows(task_id: str = self.task_id, groups: int = DEFAULT_GROUPS,
                       seed: int = DEFAULT_SEED):
            rows = self.build(groups, seed)
            self.validate(rows, groups, seed)
            return rows

        def validate_rows(rows, *, task_id: str = self.task_id, groups: int = DEFAULT_GROUPS,
                          seed: int = DEFAULT_SEED):
            return self.validate(rows, groups, seed)

        def authority_sha256(task_id: str = self.task_id, groups: int = DEFAULT_GROUPS,
                             seed: int = DEFAULT_SEED):
            return self.validate(build_rows(task_id, groups, seed), groups, seed)

        return build_rows, validate_rows, authority_sha256
