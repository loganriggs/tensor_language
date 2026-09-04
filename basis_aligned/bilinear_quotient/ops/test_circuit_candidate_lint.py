"""Tests for ops/circuit_candidate_lint.py.

The pooled-grouping test exists because the first version grouped per capability cell only and reported
"ok" on a candidate whose introduction order I had already shown by hand to determine the answer: when
every row of a cell carries the same answer, no within-cell feature can vary with it. A lint that gives
false comfort is worse than no lint.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuit_candidate_lint as L


def _row(answer, foil, text="A was a woman, and B was a man. A carried it.",
         actor="A", cell="A1/active/toward_donor"):
    return {"base_answer_id": answer, "base_foil_id": foil, "base_text": text,
            "base_antecedent": actor, "woman_label": "A", "man_label": "B",
            "capability_cell_id": cell}


def test_endpoint_merge_flags_period_quote():
    """'.' + '\"' merges to token 526, which is neither answer nor foil."""
    out = L.lint_endpoint_merge([_row(13, 1)])
    assert any("526" in f for f in out), out


def test_endpoint_merge_silent_on_pronouns():
    """' he' + ' she' does not merge, so there is nothing to flag."""
    assert L.lint_endpoint_merge([_row(339, 673)]) == []


def test_order_predicts_is_caught_when_pooled_even_if_invisible_per_cell():
    """The regression: one answer per cell hides the confound unless rows are also pooled."""
    rows = [_row(673, 339, "A was a woman, and B was a man. A carried it.", "A", "cell1"),
            _row(339, 673, "A was a woman, and B was a man. B carried it.", "B", "cell2")]
    out = L.lint_order_predicts_answer(rows)
    assert any("POOLED" in f for f in out), out
    assert any("PERFECTLY predicts" in f for f in out), out


def test_order_predicts_is_quiet_when_order_is_counterbalanced():
    """Swapping the introduction order for half the rows must clear the flag."""
    rows = [_row(673, 339, "A was a woman, and B was a man. A carried it.", "A", "c"),
            _row(339, 673, "A was a woman, and B was a man. B carried it.", "B", "c"),
            _row(673, 339, "B was a man, and A was a woman. A carried it.", "A", "c"),
            _row(339, 673, "B was a man, and A was a woman. B carried it.", "B", "c")]
    out = L.lint_order_predicts_answer(rows)
    assert all("PERFECTLY predicts" not in f for f in out), out


def test_order_predicts_skips_when_fields_absent():
    out = L.lint_order_predicts_answer([{"base_answer_id": 1, "base_foil_id": 13, "base_text": "x"}])
    assert out and out[0].startswith("skipped:")
