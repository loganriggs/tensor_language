import sys
from pathlib import Path


OPS = Path(__file__).resolve().parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import equality_matcher_copy_task_calibration_rung499 as rung


def test_support_frozen_tags_and_data_boundary():
    assert rung.BOUNDS == (500, 1000, 750)
    assert len(rung.SELECTED_TAGS) == 9
    assert len(set(rung.SELECTED_TAGS)) == 9


def test_pattern_control_win_uses_frozen_disjunction():
    positive = {"cosine": .71, "scaled_residual": .50}
    assert rung._wins(positive, {"cosine": .60, "scaled_residual": .49})
    assert rung._wins(positive, {"cosine": .70, "scaled_residual": .70})
    assert not rung._wins(positive, {"cosine": .65, "scaled_residual": .60})
