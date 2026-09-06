"""Second prospective lexicon authority for frozen upstream aspectual gain."""

from __future__ import annotations

from typing import Mapping, Sequence

import circuit_candidate_aspectual_fresh_construction_v2 as v2
import circuit_candidate_aspectual_fresh_lexicon_v3 as v3
import circuit_candidate_aspectual_lexicon_factory as factory


TASK_ID = v2.TASK_ID
TASK_SPEC = v2.TASK_SPEC
SPLIT = v2.SPLIT
SEED = 20260914
TAG = "fresh_lexicon_v4"
CandidateBankError = v2.CandidateBankError
_AGENTS = (
    "ranger", "weaver", "barber", "banker", "author", "butcher", "carpenter", "plumber",
    "scholar", "trader", "farrier", "potter", "surveyor", "librarian", "captain", "gardener",
)
_PERIODS = (
    "summer", "autumn", "spring", "rehearsal", "contract", "lecture", "inquiry", "repair",
    "evacuation", "inspection", "negotiation", "training", "residency", "internship", "excavation", "competition",
)


def _prior_ids() -> tuple[set[str], set[str]]:
    return ({str(row["row_id"]) for row in v2.build_rows()}, {str(row["row_id"]) for row in v3.build_rows()})


def build_rows(task_id: str = TASK_ID) -> list[dict]:
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    rows = factory.build(seed=SEED, tag=TAG, agents=_AGENTS, periods=_PERIODS)
    factory.validate(rows, seed=SEED, tag=TAG, agents=_AGENTS, periods=_PERIODS, disjoint_row_sets=_prior_ids())
    if set(_AGENTS) & (set(v2._AGENTS) | set(v3._AGENTS)) or set(_PERIODS) & (set(v2._PERIODS) | set(v3._PERIODS)):
        raise CandidateBankError("v4 lexicon overlaps prior prospective lexicon")
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID) -> str:
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    return factory.validate(rows, seed=SEED, tag=TAG, agents=_AGENTS, periods=_PERIODS, disjoint_row_sets=_prior_ids())


def authority_sha256(task_id: str = TASK_ID) -> str:
    return validate_rows(build_rows(task_id), task_id=task_id)


if __name__ == "__main__":
    rows = build_rows()
    print("authority:", authority_sha256())
    for family in ("A1", "A2", "P", "C"):
        row = next(item for item in rows if item["family"] == family)
        print(f"{family}: {row['base_text']!r} -> {row['base_answer']!r}")
        print(f"    {row['donor_text']!r} -> {row['donor_answer']!r}")
