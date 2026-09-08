import torch

import tied_embedding_semantic_router_contract as contract


def test_feature_is_pair_direction_and_pair_order_invariant():
    torch.manual_seed(0)
    weight = torch.randn(10, 6)
    left = contract.feature(torch, weight, ((1, 2), (3, 4)))
    right = contract.feature(torch, weight, ((4, 3), (2, 1)))
    assert torch.allclose(left, right)


def test_empty_signature_is_explicit_zero():
    value = contract.feature(torch, torch.randn(5, 3), ())
    assert torch.equal(value, torch.zeros(6))


def test_cosine_centroids_classify_synthetic_semantic_pairs():
    values = torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9], [-1., 0.], [-.9, -.1]])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    centroids = contract.fit(torch, values, labels)
    predicted, _ = contract.predict(torch, values, centroids)
    assert torch.equal(predicted, labels)
