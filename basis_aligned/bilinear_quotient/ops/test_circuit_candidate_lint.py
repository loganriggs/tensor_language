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


def _frow(answer, **extra):
    r = {"base_answer_id": answer, "base_foil_id": 1, "base_text": "x", "capability_cell_id": "c"}
    r.update(extra)
    return r


def test_feature_predicts_finds_a_prompt_derived_predictor():
    rows = [_frow(318, base_subject_number="singular"), _frow(389, base_subject_number="plural")]
    out = L.lint_feature_predicts_answer(rows)
    assert any("base_subject_number" in line for line in out), out


def test_feature_predicts_ignores_design_bookkeeping():
    """Regression: direction_id/capability_cell_id encode the swap direction by construction.

    Reporting them buried the one finding that mattered under two lines of noise.
    """
    rows = [_frow(318, direction_id="singular_to_plural", transform_id="A1"),
            _frow(389, direction_id="plural_to_singular", transform_id="A2")]
    out = L.lint_feature_predicts_answer(rows)
    assert out and out[0].startswith("ok:"), out


def test_feature_predicts_ignores_id_suffixed_fields():
    rows = [_frow(318, template_id="t1"), _frow(389, template_id="t2")]
    assert L.lint_feature_predicts_answer(out_rows := rows)[0].startswith("ok:")


def test_feature_predicts_survives_unhashable_fields():
    """Some candidates carry list-valued fields (token positions); they must not crash the lint."""
    rows = [_frow(318, base_head_positions=[1, 2], n="s"), _frow(389, base_head_positions=[3], n="p")]
    out = L.lint_feature_predicts_answer(rows)
    assert any("'n'" in line for line in out), out
