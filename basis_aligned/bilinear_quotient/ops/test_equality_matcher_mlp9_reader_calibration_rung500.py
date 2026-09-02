import sys
from pathlib import Path


OPS = Path(__file__).resolve().parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import equality_matcher_mlp9_reader_calibration_rung500 as rung


def test_exact_synthetic_reader_report():
    stats = rung._empty_stats()
    index = (0, 0, 0, 0, 0)
    stats["ref2"][index] = 9
    stats["hyb2"][index] = 36
    stats["cross"][index] = 18
    stats["write2"][index] = 100
    stats["tokens"][index] = 4
    report = rung._report(stats, index)
    assert report["cosine"] == 1
    assert report["positive_fit_scale"] == .5
    assert report["scaled_residual"] == 0


def test_price_and_observation_are_frozen():
    assert rung.BOUNDS == (500, 1000, 750)
    assert rung.CELLS == ("copy_positive", "noncopy_equality", "all_noncopy")
    assert rung.HYBRIDS == ("score_donor", "payload_donor", "whole_donor")
    assert 125 + 2375 == 2500
