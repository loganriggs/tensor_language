from collections import Counter
import pytest

import circuit_candidate_numbered_list_cached_term_head_source_factorial as candidate


def test_exact_select_membership_and_positions():
    rows = candidate.build_rows()
    assert candidate.validate_rows(rows) == "3b8a4873a6702eabe042e5d71ccf6e8e2c30d0213c0a893a1999400827f2ce97"
    assert Counter(row["family"] for row in rows) == Counter({family: 32 for family in candidate.FAMILIES})
    assert Counter(row["endpoint"] for row in rows) == Counter({"base": 64, "donor": 64})
    assert len({row["group_id"] for row in rows}) == 32
    assert all(row["query_position"] == len(row["ids"]) - 1 for row in rows)
    assert all(0 <= row["source_position"] < row["query_position"] for row in rows)


def test_plan_is_small_select_only_factorial():
    plan = candidate.compile_plan()
    assert plan["conditions"] == list(candidate.CONDITIONS)
    assert plan["price"] == {"forward_calls": 45, "example_evaluations": 640,
                             "backward_calls": 0, "model_updates": 0,
                             "raw_numeric_evidence_bytes": 13312}
    assert plan["opened_splits"] == ["SELECT"]
    assert plan["closed_splits"] == ["FINAL_TEST", "OOD"]
    assert plan["execution_policy"] == {"compile_mode": "cpu_only",
        "science_execution": "managed_queue_only", "enqueue_after_preregistration": True,
        "create_only": True}


def test_membership_and_semantic_mutations_fail_closed():
    rows = candidate.build_rows()
    with pytest.raises(candidate.CandidateError):
        candidate.validate_rows(rows[:-1])
    changed = [dict(row) for row in rows]
    changed[0]["source_position"] = changed[0]["query_position"]
    with pytest.raises(candidate.CandidateError, match="source position"):
        candidate.validate_rows(changed)
