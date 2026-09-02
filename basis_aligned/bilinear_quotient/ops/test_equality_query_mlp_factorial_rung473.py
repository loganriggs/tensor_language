import torch

import equality_query_mlp_factorial_rung473 as subject


def test_mobius_terms_reconstruct_union():
    mains = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    pair_interactions = torch.tensor([[.1, .2], [-.3, .4], [.5, -.6]])
    pairs = torch.stack([
        mains[left] + mains[right] + pair_interactions[pi]
        for pi, (left, right) in enumerate(subject.PAIRS)
    ])
    triple = torch.tensor([.7, -.8])
    union = mains.sum(0) + pair_interactions.sum(0) + triple
    found_pairs, found_triple, reconstructed = subject.mobius_terms(mains, pairs, union)
    assert torch.allclose(found_pairs, pair_interactions)
    assert torch.allclose(found_triple, triple)
    assert torch.allclose(reconstructed, union)


def test_pair_names_match_pair_indices():
    assert subject.PAIR_NAMES == ("m8+m9", "m8+m12", "m9+m12")


def test_expected_call_formulas():
    assert subject.FORWARDS_PER_BATCH == 19
    assert subject.EXPECTED_FORWARDS == subject.EXPECTED_BATCHES * 19
    assert subject.EXPECTED_PATCH_CALLS_PER_BATCH == 30
