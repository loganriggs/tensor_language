import pytest

import head_response_mediation_scorer as scorer


def test_score_cells_reports_exact_component_metrics():
    report = scorer.score_cells({
        "00": [0.0, 0.0], "01": [1.0, 2.0],
        "10": [3.0, 6.0], "11": [4.0, 8.0],
    }, [4.0, 8.0])
    assert report["head_rescue"] == [1.0, 2.0]
    assert report["head_reset_loss"] == [1.0, 2.0]
    assert report["metrics"]["head_rescue"]["signed_projection"] == pytest.approx(.25)
    assert report["closure_max_abs_error"] == 0.0


def test_head_ranking_is_descending_with_lexical_tie_break():
    reports = {
        "L15H2": {"metrics": {"head_reset_loss": {"signed_projection": .3}}},
        "L15H0": {"metrics": {"head_reset_loss": {"signed_projection": .3}}},
        "L15H1": {"metrics": {"head_reset_loss": {"signed_projection": .2}}},
    }
    assert scorer.deterministic_head_ranking(reports, "head_reset_loss") == [
        "L15H0", "L15H2", "L15H1"]


def test_singletons_compose_against_module_vector():
    singletons = {
        "h0": {"head_reset_loss": [1.0, 0.0]},
        "h1": {"head_reset_loss": [0.0, 2.0]},
    }
    report = scorer.singleton_module_composition(
        singletons, {"head_reset_loss": [1.0, 2.0]})
    assert report["summed_singleton_reset_loss"] == [1.0, 2.0]
    assert report["metrics"]["cosine"] == pytest.approx(1.0)
    assert report["metrics"]["relative_l2_error"] == pytest.approx(0.0)


def test_bad_schemas_fail_closed():
    with pytest.raises(scorer.MediationScoreError):
        scorer.score_cells({"00": [0.0], "01": [0.0], "10": [0.0], "11": [0.0]}, [1.0, 2.0])
    with pytest.raises(scorer.MediationScoreError):
        scorer.deterministic_head_ranking({"h": {}}, "head_reset_loss")
    with pytest.raises(scorer.MediationScoreError):
        scorer.singleton_module_composition({}, {"head_reset_loss": [1.0]})
