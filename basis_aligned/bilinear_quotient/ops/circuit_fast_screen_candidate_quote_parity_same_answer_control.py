#!/usr/bin/env python3
"""quote_parity with a SAME-ANSWER control in place of the inch-mark one.

The v1 screen is recorded `native_behavior_incapable` with zero sites screened. A1, A2 and P
pass every ordered cell at native answer-vs-foil margins of +4.44 / +5.07 / +4.38; only the
control fails, at 0.50. Its control asks for an inch-mark double quote after a bare numeral --
`"The board is 12 inches long; in customary notation it measures 12"` -- which GPT-2 small does
not produce, so it scored correct exactly on the period side. Unpassable by construction, and
it stopped the run before a single site was screened. Same class as the sentence-terminal
instruction-copy control.

A1/A2/P are held byte-identical to the v1 authority; only the control changes (standing lesson
2). The replacement keeps the answer FIXED at the period and varies the content, because an
answer-changing control at the patched position is carried by any site that carries the
prediction (measured: C = 1.000 at resid:18 for sentence_terminal).

Known limitation, stated up front: the same-answer control's measured ceiling is ~0.07 against
a 0.35 bar, so the C clause is NOT at risk here. This screen's honest content is target
recovery plus P invariance; C/A1 at the selected site is the discriminating statistic to
report, not the pass itself.

Implementation. v1's row builder reads its module-level TASK_ID and TASK_SPEC, so this module
rebinds both while building rather than duplicating 120 lines of construction checks. That is
only safe if the borrowed path is faithful, so `reproduces_v1_authority()` rebuilds v1's OWN
rows through this module and asserts the digest matches (standing lesson 7). Run it before
trusting any row emitted here.
"""
from __future__ import annotations

import contextlib
from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_quote_parity as v1

canonical_sha256 = v1.canonical_sha256
QuoteParityCandidateError = v1.QuoteParityCandidateError
SCHEMA = v1.SCHEMA
SPLIT = v1.SPLIT
DEFAULT_GROUPS = v1.DEFAULT_GROUPS
DEFAULT_SEED = v1.DEFAULT_SEED

TASK_ID = "quote_parity.pending_close_same_answer_control"
SAME_ANSWER_CONTROL = battery.TransformSpec(
    "C", "completed_count_same_answer_rewrite", False, "registered_active")
V1_CONTROL = battery.TransformSpec(
    "C", "sentence_end_vs_inch_unit_mark", True, "registered_active")


def _spec(task_id: str, control: battery.TransformSpec) -> battery.BatteryTaskSpec:
    a1, a2, p, _ = v1.TASK_SPEC.transforms
    return battery.BatteryTaskSpec(
        task_id=task_id,
        generator_role=v1.TASK_SPEC.generator_role,
        answer_role=v1.TASK_SPEC.answer_role,
        transforms=(a1, a2, p, control),
    )


TASK_SPEC = _spec(TASK_ID, SAME_ANSWER_CONTROL)
V1_SPEC = _spec(v1.TASK_ID, V1_CONTROL)


@contextlib.contextmanager
def _bound(task_id: str, spec: battery.BatteryTaskSpec):
    """Borrow v1's row builder under this candidate's identity."""
    old_id, old_spec = v1.TASK_ID, v1.TASK_SPEC
    v1.TASK_ID, v1.TASK_SPEC = task_id, spec
    try:
        yield
    finally:
        v1.TASK_ID, v1.TASK_SPEC = old_id, old_spec


def _count_text(number: int) -> str:
    return f"The inventory has {number} items; its final count is exactly {number}"


def _total_text(number: int) -> str:
    return f"The ledger lists {number} entries; the recorded total stands at {number}"


def _panel(seed: int, group_number: int, case_index: int, *, kind: str,
           task_id: str) -> list[dict[str, Any]]:
    writer, alternate = v1._WRITERS[case_index]
    adjective, object_name = v1._ADJECTIVES[case_index], v1._OBJECTS[case_index]
    suffix = v1._suffix(adjective, object_name)
    group_id = f"FIT:{canonical_sha256([SCHEMA, task_id, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_state, donor_state = ("outside", "pending") if forward else ("pending", "outside")
    direction = f"{base_state}_to_{donor_state}"
    p_base_writer, p_donor_writer = (writer, alternate) if forward else (alternate, writer)
    common = dict(seed=seed, group_number=group_number, group_id=group_id, writer=writer,
                  alternate_writer=alternate, adjective=adjective, object_name=object_name)

    if kind == "v1":
        c_base_answer, c_donor_answer = (".", '"') if forward else ('"', ".")
        control = v1._row(
            **common, transform_id="C", construction_id="sentence_end_vs_inch_unit_mark",
            direction_id="period_to_quote" if forward else "quote_to_period",
            base_text=v1._control_text(c_base_answer, 12 + group_number),
            donor_text=v1._control_text(c_donor_answer, 12 + group_number),
            base_answer=c_base_answer, donor_answer=c_donor_answer)
    else:
        number = 12 + group_number
        first, second = ((_count_text(number), _total_text(number)) if forward
                         else (_total_text(number), _count_text(number)))
        control = v1._row(
            **common, transform_id="C",
            construction_id="completed_count_same_answer_rewrite",
            direction_id="base_to_donor" if forward else "donor_to_base",
            base_text=first, donor_text=second, base_answer=".", donor_answer=".")

    return [
        v1._row(**common, transform_id="A1", construction_id="single_span",
                direction_id=direction,
                base_text=v1._description_text(writer, suffix, base_state, "single_span"),
                donor_text=v1._description_text(writer, suffix, donor_state, "single_span"),
                base_answer=v1._answer(base_state), donor_answer=v1._answer(donor_state)),
        v1._row(**common, transform_id="A2", construction_id="balanced_prefix",
                direction_id=direction,
                base_text=v1._description_text(writer, suffix, base_state, "balanced_prefix"),
                donor_text=v1._description_text(writer, suffix, donor_state, "balanced_prefix"),
                base_answer=v1._answer(base_state), donor_answer=v1._answer(donor_state)),
        v1._row(**common, transform_id="P", construction_id=f"single_span_{base_state}",
                direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                base_text=v1._description_text(p_base_writer, suffix, base_state, "single_span"),
                donor_text=v1._description_text(p_donor_writer, suffix, base_state, "single_span"),
                base_answer=v1._answer(base_state), donor_answer=v1._answer(base_state)),
        control,
    ]


def _build(groups: int, seed: int, *, kind: str, task_id: str,
           spec: battery.BatteryTaskSpec) -> list[dict[str, Any]]:
    order = v1._permutation(seed)
    with _bound(task_id, spec):
        return [row
                for group_number in range(groups)
                for row in _panel(seed, group_number, order[group_number],
                                  kind=kind, task_id=task_id)]


def _validate(rows, groups: int, seed: int, *, kind: str, task_id: str,
              spec: battery.BatteryTaskSpec) -> str:
    if type(groups) is not int or not 2 <= groups <= DEFAULT_GROUPS or groups % 2:
        raise QuoteParityCandidateError("groups must be an even integer from 2 through 32")
    materialized = [dict(row) for row in rows]
    if materialized != _build(groups, seed, kind=kind, task_id=task_id, spec=spec):
        raise QuoteParityCandidateError("rows differ from the deterministic semantic authority")
    try:
        digest = battery.validate_rows(spec, materialized, required_phases=(SPLIT,))
    except battery.BatteryContractError as error:
        raise QuoteParityCandidateError(str(error)) from error
    row_ids = [str(row["row_id"]) for row in materialized]
    if len(row_ids) != len(set(row_ids)):
        raise QuoteParityCandidateError("row IDs are not unique")
    cells: dict[tuple[str, str], int] = {}
    for row in materialized:
        if not all(row["construction_checks"].values()):
            raise QuoteParityCandidateError("a stored construction check is false")
        key = (str(row["transform_id"]), str(row["direction_id"]))
        cells[key] = cells.get(key, 0) + 1
    half = groups // 2
    for transform in ("A1", "A2"):
        if cells.get((transform, "outside_to_pending")) != half \
                or cells.get((transform, "pending_to_outside")) != half:
            raise QuoteParityCandidateError(f"{transform} ordered directions are unbalanced")
    forward_id, reverse_id = (("period_to_quote", "quote_to_period") if kind == "v1"
                              else ("base_to_donor", "donor_to_base"))
    if cells.get(("C", forward_id)) != half or cells.get(("C", reverse_id)) != half:
        raise QuoteParityCandidateError("C ordered directions are unbalanced")
    return digest


def build_rows(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
               seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    rows = _build(groups, seed, kind="same_answer", task_id=TASK_ID, spec=TASK_SPEC)
    _validate(rows, groups, seed, kind="same_answer", task_id=TASK_ID, spec=TASK_SPEC)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID,
                  groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED) -> str:
    return _validate(rows, groups, seed, kind="same_answer", task_id=TASK_ID, spec=TASK_SPEC)


def authority_sha256(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
                     seed: int = DEFAULT_SEED) -> str:
    return validate_rows(build_rows(task_id, groups, seed), groups=groups, seed=seed)


def reproduces_v1_authority() -> tuple[bool, str, str]:
    """Control: rebuild v1's OWN rows through this module and compare digests."""
    rows = _build(DEFAULT_GROUPS, DEFAULT_SEED, kind="v1", task_id=v1.TASK_ID, spec=V1_SPEC)
    mine = _validate(rows, DEFAULT_GROUPS, DEFAULT_SEED, kind="v1",
                     task_id=v1.TASK_ID, spec=V1_SPEC)
    return mine == v1.authority_sha256(), mine, v1.authority_sha256()


if __name__ == "__main__":
    same, mine, theirs = reproduces_v1_authority()
    print(f"control (v1 rows through this module): {'MATCH' if same else 'DRIFT'}")
    print(f"  this module  {mine}\n  v1 authority {theirs}")
    print(f"same-answer authority: {authority_sha256()}")
