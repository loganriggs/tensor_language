import torch

import equality_product_circuit_response_graph_rung477b as subject


def test_crossing_batch_is_split_between_halves():
    mask = torch.zeros(subject.DOCUMENTS * subject.TOKENS, dtype=torch.bool)
    shaped = mask.view(subject.DOCUMENTS, subject.TOKENS)
    shaped[248:252, 3] = True
    first = subject._half_batch_mask(mask, 248, 252, 0, 250)
    second = subject._half_batch_mask(mask, 248, 252, 250, 500)
    assert first.sum() == 2
    assert second.sum() == 2
    assert torch.equal(first | second, shaped[248:252])
    assert not bool((first & second).any())


def test_noncrossing_batch_has_one_active_half():
    mask = torch.ones(subject.DOCUMENTS * subject.TOKENS, dtype=torch.bool)
    assert subject._half_batch_mask(mask, 40, 44, 0, 250).all()
    assert not subject._half_batch_mask(mask, 40, 44, 250, 500).any()


def test_frozen_execution_shape():
    assert subject.BATCH == 4
    assert subject.EXPECTED_FORWARDS == 625
    assert subject.HALVES == ((0, 250), (250, 500))
