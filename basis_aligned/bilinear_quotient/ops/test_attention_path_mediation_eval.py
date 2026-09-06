import pytest

import attention_path_mediation_eval as mediation


class Batch:
    row_ids = ("a", "b")
    semantic_positions = (5, 8)


def test_reader_positions_accept_exact_causal_sets():
    assert mediation.validate_reader_positions(Batch(), ((1, 2), (4, 5))) == (
        (1, 2), (4, 5)
    )


def test_reader_positions_reject_duplicates_and_post_query():
    with pytest.raises(mediation.AttentionPathMediationError):
        mediation.validate_reader_positions(Batch(), ((1, 1), (4, 5)))
    with pytest.raises(mediation.AttentionPathMediationError):
        mediation.validate_reader_positions(Batch(), ((1, 2), (4, 9)))
