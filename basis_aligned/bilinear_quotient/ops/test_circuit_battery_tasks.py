"""Unit tests for the circuit battery task bank (CPU, no model). `pytest -q ops/test_circuit_battery_tasks.py`"""
import os
import pathlib
import sys

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


def test_deterministic_ACROSS_PROCESSES():
    """The original bank seeded with Python's hash(), which is salted per process, so the
    "frozen" rows differed between interpreters -- invisible to a same-process determinism test
    and caught only by Codex's audit. This test spawns real subprocesses."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import circuit_battery_tasks as B; "
            "print(','.join(r['row_id'][:8] for r in B.build_rows('numbered_list.index_successor', per_cell=3)))"
            % str(pathlib.Path(B.__file__).parent))
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(k)}).stdout.strip()
            for k in (0, 1, 2)}
    assert len(outs) == 1, outs


def test_groups_are_complete_and_share_one_situation():
    """Each group is ONE generated situation transformed into every family (Codex SS2810 point 3)."""
    for tid in B.TASKS:
        fams = set(B.TASKS[tid].families)
        groups = {}
        for r in _rows(tid):
            groups.setdefault((r["split"], r["group_id"]), set()).add(r["family"])
        assert groups, tid
        for key, got in groups.items():
            assert got == fams, (tid, key, got)


def test_joint_tokenization_boundary():
    """encode(prompt) must be an exact prefix of encode(prompt + answer), with a one-token suffix
    at the real continuation boundary (Codex SS2810 point 4)."""
    for tid in B.TASKS:
        for r in _rows(tid):
            assert B.ENC.encode(r["base_text"] + r["base_answer"]) == r["base_ids"] + [r["base_answer_id"]]
            assert B.ENC.encode(r["donor_text"] + r["donor_answer"]) == r["donor_ids"] + [r["donor_answer_id"]]


def test_heldout_splits_are_value_disjoint_where_claimed():
    """TEST/OOD must never share an ANSWER VALUE with FIT/SELECT within a family, except for the
    tasks whose vocabulary is too small to allow it -- which are listed explicitly, not hidden."""
    small_vocab = {"bracket.close_innermost", "arithmetic.small_addition", "numeric_run.last_plus_one"}
    for tid in B.TASKS:
        pol = B.split_policy(tid, per_cell=6)
        assert pol["prompts_disjoint_all_pairs"], tid
        if tid not in small_vocab:
            assert pol["all_families_heldout_value_disjoint"], (tid, pol["per_family"])


def test_interchange_alignment_available():
    """At least half of each task's A1 rows must have base/donor of equal token length."""
    for tid in B.TASKS:
        a1 = [r for r in _rows(tid, 12) if r["family"] == "A1"]
        aligned = [r for r in a1 if len(r["base_ids"]) == len(r["donor_ids"])]
        assert len(aligned) >= 0.5 * len(a1), (tid, len(aligned), len(a1))
