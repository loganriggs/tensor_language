"""Unit tests for the circuit battery task bank (CPU, no model). `pytest -q ops/test_circuit_battery_tasks.py`"""
import circuit_battery_tasks as B


def _rows(tid, n=6):
    return B.build_rows(tid, per_cell=n)


def test_every_task_fills_every_cell():
    for tid in B.TASKS:
        rows = _rows(tid)
        fams = B.TASKS[tid].families
        cells = {(r["family"], r["split"]) for r in rows}
        assert cells == {(f, s) for f in fams for s in B.SPLITS}, tid
        assert len(rows) == 6 * len(fams) * len(B.SPLITS), tid


def test_construction_checks_all_true():
    for tid in B.TASKS:
        for r in _rows(tid):
            assert all(r["construction_checks"].values()), (tid, r["row_id"])


def test_answers_are_single_tokens_and_in_vocab():
    for tid in B.TASKS:
        vocab = set(B.candidate_strings(tid))
        for r in _rows(tid):
            assert len(B.ENC.encode(r["base_answer"])) == 1
            assert len(B.ENC.encode(r["donor_answer"])) == 1
            assert r["base_answer"] in vocab and r["donor_answer"] in vocab


def test_family_semantics():
    """A1/A2/C must change the answer; P must preserve it and change the prompt."""
    for tid in B.TASKS:
        for r in _rows(tid):
            if r["family"] == "P":
                assert r["base_answer"] == r["donor_answer"] and not r["answer_changes"], (tid, r["row_id"])
                assert r["base_text"] != r["donor_text"]
            else:
                assert r["base_answer"] != r["donor_answer"] and r["answer_changes"], (tid, r["row_id"])


def test_answer_is_not_already_visible_for_computed_families():
    """A1/A2 answers must be COMPUTED, not copyable from the prompt; C answers must be copyable."""
    for tid in B.TASKS:
        for r in _rows(tid):
            words = r["base_text"].translate(str.maketrans("\n.:+=()[]{}", "           ")).split()
            visible = r["base_answer"].strip() in words
            if r["family"] in ("A1", "A2") and not B.TASKS[tid].answer_visible_in_prompt:
                assert not visible, (tid, r["family"], r["base_text"], r["base_answer"])
            if r["family"] == "C":
                assert visible, (tid, r["family"], r["base_text"], r["base_answer"])


def test_splits_use_disjoint_value_pools():
    """No FIT prompt may reappear in SELECT or TEST."""
    for tid in B.TASKS:
        by = {s: {r["base_text"] for r in _rows(tid) if r["split"] == s} for s in B.SPLITS}
        assert not (by["FIT"] & by["SELECT"]), tid
        assert not (by["FIT"] & by["TEST"]), tid
        assert not (by["SELECT"] & by["TEST"]), tid


def test_deterministic():
    for tid in B.TASKS:
        assert [r["row_id"] for r in _rows(tid)] == [r["row_id"] for r in _rows(tid)], tid


def test_interchange_alignment_available():
    """At least half of each task's A1 rows must have base/donor of equal token length."""
    for tid in B.TASKS:
        a1 = [r for r in _rows(tid, 12) if r["family"] == "A1"]
        aligned = [r for r in a1 if len(r["base_ids"]) == len(r["donor_ids"])]
        assert len(aligned) >= 0.5 * len(a1), (tid, len(aligned), len(a1))
