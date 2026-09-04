"""Unit tests for the circuit battery task bank (CPU, no model). `pytest -q ops/test_circuit_battery_tasks.py`"""
import os
import pathlib
import sys

import circuit_battery_tasks as B


FROZEN_ROW_HASHES = {
    "alphabet_run.successor": "ea79ea55d441c4c0",
    "arithmetic.small_addition": "1438810a5cddacf3",
    "bracket.close_innermost": "e0391883dfc3f275",
    "counting_words.comma_list": "4d5b6eeacc764b84",
    "counting_words.successor": "f0a1cdac4d7a6690",
    "induction.copy_successor": "0ca0e5f87cd7fa43",
    "keyed_line.counter_successor": "07a608189d4cc150",
    "letter_list.index_successor": "651fa396a2cd9bc9",
    "letter_list.lowercase": "edc1c19ab99bfeca",
    "month.successor": "35c1431eebed9ae5",
    "numbered_list.index_successor": "a3a79c39e3bbb19e",
    "numeric_run.last_plus_one": "e3852ae3d10ea85e",
    "numeric_sequence.continuation": "a7663a934cc85a41",
    "numeric_sequence.countdown": "d126fc10cb945fe1",
    "paren_list.index_successor": "574ff7fec799a5eb",
    "percent_run.step_continuation": "dd62accb84c4c355",
    "roman_list.index_successor": "45e857c3aa4db6eb",
    "variable_lookup.assignment": "887379fc25fdd917",
    "verbatim_repeat.copy": "142ec67d23224168",
    "weekday.successor": "721e5a4e2a51d070",
    "year_run.successor": "1564aad2bdc18053",
}


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
            words = r["base_text"].translate(str.maketrans("\n.:+=()[]{},%", "             ")).split()
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
    # Tasks whose answer vocabulary cannot support value-disjoint held-out splits, named
    # explicitly rather than exempted by a blanket rule:
    #   bracket   - three bracket types in total
    #   addition  - the small addend is drawn from a shared 2..7 range
    #   numeric_run - A2 perturbs the final term within +-6, which crosses the split partition
    #   percent_run - the answer is start + 3*step with step in {5, 10}, so two splits' starts can
    #                 land on the same answer
    small_vocab = {"bracket.close_innermost", "arithmetic.small_addition", "numeric_run.last_plus_one",
                   "percent_run.step_continuation"}
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


def test_bank_extensions_do_not_perturb_existing_tasks():
    """Adding a behaviour must never change another behaviour's rows.

    Seeds are per (task, split, group), so this holds by construction -- but the whole value of the
    protocol is that §2817's numbers stay reproducible when the bank grows, and "by construction" is
    what the process-salted hash() bug of §2817 also looked like. These hashes were frozen when the
    bank went from 16 to 21 tasks, with all 16 verified bit-identical against the previous bank.
    """
    import hashlib
    for tid, expect in FROZEN_ROW_HASHES.items():
        got = hashlib.sha256(",".join(r["row_id"] for r in B.build_rows(tid, per_cell=8)).encode()).hexdigest()[:16]
        assert got == expect, (tid, got, expect)


def test_frozen_hashes_cover_every_task():
    assert set(FROZEN_ROW_HASHES) == set(B.TASKS), set(B.TASKS) ^ set(FROZEN_ROW_HASHES)
