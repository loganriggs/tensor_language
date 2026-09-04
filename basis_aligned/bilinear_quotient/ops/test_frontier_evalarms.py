"""Guard for ops/frontier_evalarms.py: the eval-only classifier must not mislabel a refitting rung."""
import sys
from pathlib import Path

OPS = Path(__file__).parent
sys.path.insert(0, str(OPS))
import frontier_evalarms as FE


def test_eval_only_rungs_are_detected():
    for name in ("frontier_error_decomposition", "frontier_mlp_side_error_share",
                 "frontier_attn5_error_share"):
        p = OPS / f"{name}.py"
        if not p.is_file():
            continue
        ok, last_fit, knob = FE.check_eval_only(p)
        assert ok, (name, last_fit, knob)
        assert min(knob) > last_fit


def test_refitting_rungs_are_NOT_mislabelled():
    """The collapse family reads its knob inside fit_attnd; treating it as eval-only would be silent nonsense."""
    for name, knob in (("frontier_a5_constant_collapse", "COLLAPSE"),
                       ("frontier_tail_refit_split", "TAILMODE")):
        p = OPS / f"{name}.py"
        if not p.is_file():
            continue
        ok, _lf, _k = FE.check_eval_only(p, knob=knob)
        assert not ok, name


def test_savings_matches_the_logged_measurement():
    # three eval-only rungs this hour: 4 + 4 + 3 arms at 94.1 s per pipeline run
    total = sum(FE.savings(n, 94.1) for n in (4, 4, 3))
    assert 680 <= total <= 720, total
