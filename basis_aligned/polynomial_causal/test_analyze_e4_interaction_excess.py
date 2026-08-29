from __future__ import annotations

import pytest

import analyze_e4_interaction_excess as analysis


def _toy_ledger():
    documents = ["d0", "d1"]
    candidates = {}
    effects = {
        "L5H5": 1.0, "L7H3": 2.0, "L8H3": 3.0, "L8H4": 4.0,
        "registered_four_head_set": 15.0,
    }
    for candidate, effect in effects.items():
        candidates[candidate] = {}
        for document in documents:
            candidates[candidate][document] = {}
            for cell in analysis.CELLS:
                scale = {"positive": 1.0, "matched_negative": 0.2, "off_target": 0.5}[cell]
                candidates[candidate][document][cell] = {
                    "n": 1, "native_nll_sum": 2.0,
                    "ablated_nll_sum": 2.0 + scale * effect,
                    "support_sha256": f"{document}:{cell}",
                }
    return {"ordered_document_ids": documents, "candidates": candidates}


def test_known_answer_joint_minus_singleton_sum():
    result = analysis.bootstrap_interaction_excess(_toy_ledger(), draws=20, seed=3)
    assert result["point"] == pytest.approx([5.0, 1.0, 2.5, 4.0])
    assert result["joint_to_singleton_sum_ratio"] == pytest.approx(1.5)
    assert result["simultaneous_q05_lower"] == pytest.approx(result["point"])
    assert result["simultaneous_q95_upper"] == pytest.approx(result["point"])


def test_rejects_nonshared_native_baseline():
    ledger = _toy_ledger()
    ledger["candidates"]["L5H5"]["d0"]["positive"]["native_nll_sum"] = 3.0
    with pytest.raises(RuntimeError, match="shared support"):
        analysis.bootstrap_interaction_excess(ledger, draws=5)

