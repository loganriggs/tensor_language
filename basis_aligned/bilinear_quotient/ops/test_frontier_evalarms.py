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


def test_factorial_arms_covers_the_grid_and_names_are_unique():
    arms = FE.factorial_arms({"tail_scale": [0.25, 0.5], "cp_scale": [None, 0.5]})
    names = [n for n, _ in arms]
    assert names[0] == "baseline"
    assert len(arms) == 1 + 4, arms            # 2 x 2, none of which is entirely empty
    assert len(set(names)) == len(names), names
    specs = [s for _, s in arms[1:]]
    assert {"tail_scale": 0.25} in specs                       # cp_scale None is dropped
    assert {"tail_scale": 0.5, "cp_scale": 0.5} in specs


def test_factorial_drops_the_all_none_cell():
    arms = FE.factorial_arms({"a": [None, 1.0]}, baseline=False)
    assert len(arms) == 1 and arms[0][1] == {"a": 1.0}, arms


def test_subset_arms_is_the_two_to_the_k_pattern():
    arms = FE.subset_arms({"T": {"tail_scale": 0.25}, "C": {"cp_scale": 0.5},
                           "F": {"a_scale": 0.5}})
    names = [n for n, _ in arms]
    assert names == ["baseline", "C", "F", "T", "CF", "CT", "FT", "CFT"], names
    assert dict(arms)["CFT"] == {"tail_scale": 0.25, "cp_scale": 0.5, "a_scale": 0.5}


def test_arm_names_are_deterministic_across_calls():
    a = FE.factorial_arms({"x": [0.25], "y": [1.5]})
    b = FE.factorial_arms({"y": [1.5], "x": [0.25]})
    assert a == b, (a, b)


def test_the_logged_economics_hold():
    per = lambda n: (90 + n * 3.5) / n
    assert round(per(8), 1) == 14.8 and round(per(40), 1) == 5.8
