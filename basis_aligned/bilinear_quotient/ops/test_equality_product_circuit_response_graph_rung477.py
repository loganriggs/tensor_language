import torch

import equality_product_circuit_response_graph_rung477 as subject


def test_response_profiles_make_member_minus_control_and_center():
    sums = torch.zeros(2, 2, 2, 3, 4, 3, dtype=torch.float64)
    counts = torch.ones(2, 2, 3, dtype=torch.float64)
    counts[:, 0] = 2
    counts[:, 1] = 4
    sums[:, :, 0] = torch.tensor([2.0, 4.0, 8.0])
    sums[:, :, 1] = torch.tensor([0.0, 4.0, 4.0])
    means, contrast, normalized, norms = subject.response_profiles(sums, counts)
    assert means.shape == sums.shape
    assert torch.allclose(contrast[0, 0, 0, 0], torch.tensor([1., 1., 3.], dtype=torch.float64))
    assert torch.allclose(normalized.mean(-1), torch.zeros_like(normalized.mean(-1)), atol=1e-12)
    assert bool((norms > 0).all())


def test_pair_graph_finds_mutual_identical_terms_on_cpu():
    old = subject.HIDDEN
    try:
        subject.HIDDEN = 4
        base = torch.eye(4, dtype=torch.float64)
        normalized = torch.stack([base, base, base]).view(1, 1, 3, 4, 4).expand(2, 2, -1, -1, -1).clone()
        eligible = torch.ones(3, 4, dtype=torch.bool)
        graph = subject.pair_graph(normalized, eligible, 0, 1, device="cpu")
    finally:
        subject.HIDDEN = old
    assert graph["left_indices"].tolist() == [0, 1, 2, 3]
    assert graph["right_indices"].tolist() == [0, 1, 2, 3]
    assert float(graph["view_cosines"].min()) == 1.0


def test_frozen_family_split_and_forward_formula():
    assert set(subject.DISCOVERY_ROOTS).isdisjoint(subject.VALIDATION_ROOTS)
    assert subject.EXPECTED_FORWARDS == 625
    assert len(subject.PERMUTATION_SEEDS) == 16
