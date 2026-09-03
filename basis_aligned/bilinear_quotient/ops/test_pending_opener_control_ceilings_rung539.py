import os

os.environ["BQLIB_NO_MODEL"] = "1"

import pending_opener_control_ceilings_rung539 as rung


def test_price_and_scope_are_frozen():
    assert rung.EXPECTED_PAIRS == 128
    assert rung.EXPECTED_FORWARDS == 48
    assert rung.SITE == "resid8"
    assert rung.SPLITS == ("FIT", "SELECT")


def test_absolute_ceiling_summary_preserves_sign_and_materiality():
    values = [1.0, -1.0, 0.5, -0.5] * 4
    report = rung.summarize(values, [0.2] * len(values), 1)
    assert report["mean_signed_endpoint_change"] == 0.0
    assert report["mean_absolute_endpoint_change"] == 0.75
    assert report["bootstrap95_lower_mean_absolute"] > 0.05
    assert report["mean_full_vocabulary_logit_rms"] == 0.2
