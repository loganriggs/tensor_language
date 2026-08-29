import copy
import json

import pytest

import analyze_e4_distribution_nonadditivity as analysis
import analyze_e4_interaction_excess as ce_analysis


def _ledger():
    return json.loads(ce_analysis.LEDGER.read_text())


def test_kl_nonadditivity_matches_receipt_backed_point_values():
    result = analysis.analyze(_ledger(), draws=64, seed=9)
    assert result["documents"] == 192
    assert result["cells"]["positive"]["joint_kl"] == pytest.approx(
        0.388348347722717
    )
    assert result["cells"]["positive"]["singleton_kl_sum"] == pytest.approx(
        0.059864089512422974
    )
    assert result["cells"]["positive"]["joint_minus_singleton_sum"] == pytest.approx(
        0.32848425821029403
    )


def test_shared_support_or_baseline_drift_fails_closed():
    ledger = _ledger()
    document = ledger["ordered_document_ids"][0]
    broken = copy.deepcopy(ledger)
    broken["candidates"]["L5H5"][document]["positive"]["n"] += 1
    with pytest.raises(RuntimeError, match="shared support/baseline"):
        analysis.analyze(broken, draws=8)
