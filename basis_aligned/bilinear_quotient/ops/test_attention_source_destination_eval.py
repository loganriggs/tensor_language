import pytest

import attention_source_destination_eval as source


def test_cue_partition_is_exact_at_each_subject_destination():
    base = (10, 20, 30, 40, 50)
    donor = (11, 20, 30, 40, 50)
    assert source.cue_partition(base, donor, 1) == {
        "prefix": (), "cue": (0,), "local": (1,)
    }
    assert source.cue_partition(base, donor, 2) == {
        "prefix": (), "cue": (0,), "local": (1, 2)
    }


def test_partition_rejects_noncausal_destination():
    with pytest.raises(source.AttentionSourceDestinationError):
        source.cue_partition((1, 2, 3), (9, 2, 3), 0)


def test_partition_handles_report_prefix():
    groups = source.cue_partition(
        (1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 9, 6, 7), 6
    )
    assert groups["prefix"] == (0, 1, 2, 3)
    assert groups["cue"] == (4,)
    assert groups["local"] == (5, 6)
