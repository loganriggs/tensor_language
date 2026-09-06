import residual_source_onset_eval as onset


class Batch:
    row_ids = ("a", "b")
    token_rows = (
        (10, 20, 30, 40, 50, 60),
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    )
    semantic_positions = (5, 9)


def test_subject_onset_positions_use_exact_two_token_group():
    donor = Batch()
    donor.token_rows = (
        (11, 20, 30, 40, 50, 60),
        (1, 2, 3, 4, 99, 6, 7, 8, 9, 10),
    )
    assert onset.positions_for_group(Batch(), donor, "subject_onset") == ((1, 2), (5, 6))


def test_curve_and_earliest_passing_contract():
    records = []
    for boundary, value in ((0, 0.0), (1, 0.4), (2, 0.6)):
        for family in ("A1", "A2"):
            records.extend(
                {"group": "subject_onset", "boundary": boundary, "family": family, "recovery": value}
                for _ in range(4)
            )
    points = onset.curve(records, "subject_onset", range(3))
    assert onset.earliest_passing(points) == 2
    assert points[2]["mean_target_recovery"] == 0.6
