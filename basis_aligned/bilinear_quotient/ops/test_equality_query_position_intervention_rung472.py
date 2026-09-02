import torch

import equality_query_position_intervention_rung472 as subject


def test_position_masks_partition_prefix():
    query, nonquery, prefix, active = subject.position_masks(
        3, 10, [(0, 7), (2, 4)], torch.device("cpu"),
    )
    assert torch.equal(query | nonquery, prefix)
    assert not bool((query & nonquery).any())
    assert query[0, 7] and nonquery[0, :7].all() and prefix[0, :8].all()
    assert not bool(prefix[1].any()) and torch.equal(active, torch.tensor([1, 0, 1]).bool())


def test_metrics_report_query_relationship():
    target = torch.tensor([0., 1., 2., 3.])
    proposed = target.clone()
    metrics = subject._metrics(target, proposed)
    assert metrics["pearson"] > .999
    assert metrics["normalized_l2_error"] < 1e-12


def test_query_interaction_definition():
    singles = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    interaction = torch.tensor([.5, -.25])
    union = singles.sum(0) + interaction
    assert torch.allclose(union - singles.sum(0), interaction)
