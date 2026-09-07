import pytest

from multi_environment_projector_contract import (
    MultiEnvironmentContractError, score_checkpoint, select_checkpoint,
)


def checkpoint(step, a1, a2, p=(.01, .02, .03, 0), c=(.02, .03, .04, 0)):
    return {
        "step": step,
        "targets": {
            "A1": {"signed_projection": a1[0], "direction_fraction": a1[1]},
            "A2": {"signed_projection": a2[0], "direction_fraction": a2[1]},
        },
        "controls": {
            "P": {"median_kl": p[0], "mean_kl": p[1], "max_kl": p[2], "top1_flip_count": p[3]},
            "C": {"median_kl": c[0], "mean_kl": c[1], "max_kl": c[2], "top1_flip_count": c[3]},
        },
    }


ARGS = dict(target_panels=("A1", "A2"), control_panels=("P", "C"),
            projection_min=.75, direction_min=.875)


def test_worst_construction_not_average_licenses_target():
    report = score_checkpoint(checkpoint(4, (.95, 1), (.60, 1)), **ARGS)
    assert not report["feasible"]
    assert report["worst_target_violation"] == pytest.approx(.15)
    assert report["minimum_target_projection"] == .60


def test_feasible_target_beats_zero_effect_control_solution():
    zero = checkpoint(0, (0, 0), (0, 0), p=(0, 0, 0, 0), c=(0, 0, 0, 0))
    causal = checkpoint(4, (.80, 1), (.76, .90), p=(.02, .02, .03, 0), c=(.01, .02, .04, 0))
    chosen, report = select_checkpoint([zero, causal], **ARGS)
    assert chosen is causal
    assert report["feasible"]


def test_controls_choose_only_among_target_feasible_checkpoints():
    early = checkpoint(4, (.80, 1), (.76, 1), p=(.02, .03, .05, 0), c=(.03, .03, .04, 0))
    later = checkpoint(8, (.78, 1), (.77, 1), p=(.01, .02, .06, 0), c=(.01, .02, .05, 1))
    chosen, report = select_checkpoint([early, later], **ARGS)
    assert chosen is later
    assert report["control_objective"] == pytest.approx(.015)


def test_worst_panel_not_pooled_average_sets_control_objective():
    report = score_checkpoint(
        checkpoint(4, (.8, 1), (.8, 1), p=(.001, .001, .002, 0), c=(.1, .2, .3, 2)), **ARGS)
    assert report["worst_control_median_kl"] == .1
    assert report["worst_control_mean_kl"] == .2
    assert report["control_objective"] == pytest.approx(.15)
    assert report["total_control_flips"] == 2


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(step=-1),
    lambda value: value["targets"].pop("A2"),
    lambda value: value["controls"]["P"].update(mean_kl=float("nan")),
    lambda value: value["controls"]["C"].update(top1_flip_count=-1),
])
def test_malformed_checkpoint_fails_closed(mutation):
    value = checkpoint(0, (.8, 1), (.8, 1))
    mutation(value)
    with pytest.raises(MultiEnvironmentContractError):
        score_checkpoint(value, **ARGS)
