import pytest

import attention_source_group_eval as source


def test_direct_frame_partition_is_complete_and_disjoint():
    groups = source.aligned_source_partition(
        (10, 20, 30, 40, 50, 60), (11, 20, 30, 40, 50, 60), 5
    )
    assert groups == {
        "prefix": (),
        "cue": (0,),
        "subject_onset": (1, 2),
        "intervening_suffix": (3, 4),
        "self": (5,),
    }
    flattened = [position for name in source.GROUP_ORDER for position in groups[name]]
    assert flattened == list(range(6))


def test_report_frame_partition_is_complete_and_disjoint():
    groups = source.aligned_source_partition(
        tuple(range(10)), (0, 1, 2, 3, 99, 5, 6, 7, 8, 9), 9
    )
    assert groups["prefix"] == (0, 1, 2, 3)
    assert groups["cue"] == (4,)
    assert groups["subject_onset"] == (5, 6)
    assert groups["intervening_suffix"] == (7, 8)
    assert groups["self"] == (9,)


def test_partition_rejects_unaligned_or_short_rows():
    with pytest.raises(source.SourceGroupError):
        source.aligned_source_partition((1, 2), (3, 4), 1)
    with pytest.raises(source.SourceGroupError):
        source.aligned_source_partition((1, 2, 3), (9, 2, 3), 2)


def test_group_selection_rejects_duplicates_and_unknowns():
    assert source.validate_group_names(("cue", "self")) == ("cue", "self")
    with pytest.raises(source.SourceGroupError):
        source.validate_group_names(("cue", "cue"))
    with pytest.raises(source.SourceGroupError):
        source.validate_group_names(("unknown",))


def test_summary_contract():
    records = [
        {"family": "A1", "recovery": 0.5},
        {"family": "A1", "recovery": -0.25},
        {"family": "A2", "recovery": 1.0},
    ]
    assert source.summarize_by_family(records)["A1"] == {
        "count": 2,
        "mean_recovery": 0.125,
        "mean_absolute_recovery": 0.375,
        "direction_fraction": 0.5,
    }
