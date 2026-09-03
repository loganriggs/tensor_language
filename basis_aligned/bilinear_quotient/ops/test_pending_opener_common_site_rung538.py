import os

os.environ["BQLIB_NO_MODEL"] = "1"

import pending_opener_common_site_rung538 as rung


def test_site_order_is_causal_and_complete():
    assert len(rung.SITE_ORDER) == 15
    assert rung.SITE_ORDER[:4] == ("resid8", "mlp_product8", "resid9", "mlp_product9")
    assert rung.SITE_ORDER[10:13] == ("resid13", "attn13h8", "mlp_product13")
    assert rung.EXPECTED_FORWARDS == 496


def test_summary_uses_all_three_frozen_bars():
    held = rung.summarize([2.0] * 8 + [-0.1] * 2, 1)
    assert held["mean_donorward_movement"] > 0
    assert held["positive_movement_fraction"] == 0.8
    assert held["bootstrap95_lower_mean"] > 0
    assert held["passed"] is True
    assert rung.summarize([0.1] * 6 + [-0.1] * 4, 1)["passed"] is False


def test_common_site_requires_every_family_split_and_direction():
    raw = {
        site: {
            split: {family: {"base_to_donor": [1.0] * 8, "donor_to_base": [1.0] * 8}
                    for family in rung.FAMILIES}
            for split in rung.SPLITS
        } for site in rung.SITE_ORDER
    }
    reports, passing = rung.score(raw)
    assert passing == list(rung.SITE_ORDER)
    raw["resid8"]["SELECT"]["closed_then_reopened_type"]["donor_to_base"] = [-1.0] * 8
    reports, passing = rung.score(raw)
    assert reports["resid8"]["common_live"] is False
    assert passing[0] == "mlp_product8"
