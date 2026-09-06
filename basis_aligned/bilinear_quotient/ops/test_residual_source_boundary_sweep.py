import pytest

import residual_source_onset_eval as onset


class Output:
    def __init__(self, values):
        self.answer_foil = values


class Backend:
    def forward_states(self, batch, **kwargs):
        boundary = kwargs["boundary"]
        return Output(((0.0, float(boundary)),)), ()


def item():
    return {
        "rows": ({"transform_id": "A1", "direction_id": "forward", "row_id": "r"},),
        "base_batch": "base",
        "donor_batch": "donor",
        "base_output": Output(((0.0, 0.0),)),
        "donor_output": Output(((4.0, 0.0),)),
        "donor_states": (0, 1, 2),
    }


def test_sweep_scores_precomputed_states_and_price():
    result = onset.sweep_precomputed_states(
        Backend(), (item(),), boundaries=(1, 2), group_name="cue", maximum_boundary=2,
        recovery_bar=0.2, direction_bar=1.0,
    )
    assert [point["mean_target_recovery"] for point in result["curve"]] == [0.25, 0.5]
    assert [point["passed"] for point in result["curve"]] == [True, True]
    assert result["forward_calls"] == 2
    assert result["example_evaluations"] == 2
    assert result["base_scored_logit_max_abs_by_boundary"] == {"1": 1.0, "2": 2.0}


def test_sweep_rejects_unordered_or_duplicate_boundaries():
    with pytest.raises(onset.ResidualOnsetError):
        onset.sweep_precomputed_states(
            Backend(), (item(),), boundaries=(2, 1), group_name="cue", maximum_boundary=2,
            recovery_bar=0.0, direction_bar=0.0,
        )
    with pytest.raises(onset.ResidualOnsetError):
        onset.sweep_precomputed_states(
            Backend(), (item(),), boundaries=(1, 1), group_name="cue", maximum_boundary=2,
            recovery_bar=0.0, direction_bar=0.0,
        )


def test_all_positions_requires_alignment_and_covers_through_query():
    base = type("Batch", (), {"semantic_positions": (2, 4)})()
    donor = type("Batch", (), {"semantic_positions": (2, 4)})()
    assert onset.positions_for_group(base, donor, "all_positions") == ((0, 1, 2), (0, 1, 2, 3, 4))
    shifted = type("Batch", (), {"semantic_positions": (2, 5)})()
    with pytest.raises(onset.ResidualOnsetError):
        onset.positions_for_group(base, shifted, "all_positions")
