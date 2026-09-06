"""Third prospective lexicon authority for frozen upstream aspectual gain."""

from __future__ import annotations

from typing import Mapping, Sequence

import circuit_candidate_aspectual_fresh_construction_v2 as v2
import circuit_candidate_aspectual_fresh_lexicon_v3 as v3
import circuit_candidate_aspectual_fresh_lexicon_v4 as v4
import circuit_candidate_aspectual_lexicon_factory as factory


TASK_ID = v2.TASK_ID
TASK_SPEC = v2.TASK_SPEC
SPLIT = v2.SPLIT
SEED = 20260915
TAG = "fresh_lexicon_v5"
CandidateBankError = v2.CandidateBankError
_AGENTS = (
    "forester", "miller", "porter", "chemist", "journalist", "musician", "professor", "engineer",
    "accountant", "director", "manager", "researcher", "secretary", "surgeon", "technician", "translator",
)
_PERIODS = (
    "holiday", "weekend", "month", "year", "century", "session", "meeting", "operation",
    "performance", "review", "assessment", "exercise", "rescue", "transfer", "installation", "commission",
)


def _prior_rows():
    return (v2.build_rows(), v3.build_rows(), v4.build_rows())


def _validate_lexicon_disjointness():
    prior_agents = set(v2._AGENTS) | set(v3._AGENTS) | set(v4._AGENTS)
    prior_periods = set(v2._PERIODS) | set(v3._PERIODS) | set(v4._PERIODS)
    if set(_AGENTS) & prior_agents or set(_PERIODS) & prior_periods:
        raise CandidateBankError("v5 lexicon overlaps a prior prospective lexicon")


def build_rows(task_id: str = TASK_ID) -> list[dict]:
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    _validate_lexicon_disjointness()
    rows = factory.build(seed=SEED, tag=TAG, agents=_AGENTS, periods=_PERIODS)
    factory.validate(rows, seed=SEED, tag=TAG, agents=_AGENTS, periods=_PERIODS, disjoint_row_sets=tuple({str(row["row_id"]) for row in prior} for prior in _prior_rows()))
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID) -> str:
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    _validate_lexicon_disjointness()
    return factory.validate(rows, seed=SEED, tag=TAG, agents=_AGENTS, periods=_PERIODS, disjoint_row_sets=tuple({str(row["row_id"]) for row in prior} for prior in _prior_rows()))


def authority_sha256(task_id: str = TASK_ID) -> str:
    return validate_rows(build_rows(task_id), task_id=task_id)


if __name__ == "__main__":
    rows = build_rows()
    print("authority:", authority_sha256())
    for family in ("A1", "A2", "P", "C"):
        row = next(item for item in rows if item["family"] == family)
        print(f"{family}: {row['base_text']!r} -> {row['base_answer']!r}")
        print(f"    {row['donor_text']!r} -> {row['donor_answer']!r}")
