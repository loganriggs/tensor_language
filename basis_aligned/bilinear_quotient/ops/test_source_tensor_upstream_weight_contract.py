import torch
import pytest

import source_tensor_upstream_weight_contract as contract


def test_covariance_is_trace_one_and_axis_enrichment_is_128():
    rows = torch.zeros(3, 128)
    rows[:, 7] = torch.tensor([1.0, 2.0, 3.0])
    covariance = contract.trace_one_covariance(torch, rows)
    assert torch.isclose(torch.trace(covariance), torch.tensor(1.0))
    composed = torch.zeros(128, 2)
    composed[7] = 1.0
    assert contract.enrichment(torch, composed, covariance) == pytest.approx(128.0)


def test_isotropic_covariance_has_unit_enrichment():
    covariance = torch.eye(128) / 128
    composed = torch.randn(128, 9, generator=torch.Generator().manual_seed(3))
    assert contract.enrichment(torch, composed, covariance) == pytest.approx(1.0, abs=1e-6)


def test_top_labels_and_jaccard_are_deterministic():
    rows = [{"label": "b", "score": 2.0}, {"label": "a", "score": 2.0},
            {"label": "c", "score": 1.0}]
    assert contract.top_labels(rows, "score", 2) == ["a", "b"]
    assert contract.jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_rejects_zero_bank_and_zero_map():
    with pytest.raises(ValueError, match="zero"):
        contract.trace_one_covariance(torch, torch.zeros(2, 128))
    with pytest.raises(ValueError, match="zero"):
        contract.enrichment(torch, torch.zeros(128, 2), torch.eye(128) / 128)
