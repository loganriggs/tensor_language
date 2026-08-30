import numpy as np
import pytest

from causal_response_tensor_contract import (
    DocumentResponse,
    identifiability_counterexamples,
    summarize_document,
    validate_response_records,
)


def test_document_statistics_recover_signed_contrast_and_legacy_ratio() -> None:
    response = summarize_document(
        9,
        np.asarray([2.0, -1.0, 3.0, 1.0]),
        np.asarray([1, 1, 0, 0]),
        np.asarray([0, 0, 1, 1]),
    )
    assert response.signed_contrast == pytest.approx(-1.5)
    assert response.absolute_concentration == pytest.approx(0.75)


def test_contract_rejects_overlap_and_duplicate_documents() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        summarize_document(
            1,
            np.ones(3),
            np.asarray([1, 0, 0]),
            np.asarray([1, 0, 1]),
        )
    record = DocumentResponse(3, 1.0, 1.0, 1, 1.0, 1.0, 1)
    with pytest.raises(ValueError, match="unique"):
        validate_response_records([record, record])


def test_contract_requires_document_level_replication() -> None:
    record = DocumentResponse(3, 1.0, 1.0, 1, 1.0, 1.0, 1)
    with pytest.raises(ValueError, match="at least two"):
        validate_response_records([record])


def test_ratio_only_nonidentifiability_counterexamples() -> None:
    result = identifiability_counterexamples()
    same = result["same_ratio_different_contrast"]
    assert same["ratio_first"] == same["ratio_scaled"]
    assert same["signed_contrast_first"] != same["signed_contrast_scaled"]
    nonadd = result["ratio_nonadditivity"]
    assert nonadd["pooled_ratio"] != nonadd["sum_of_ratios"]
    signs = result["absolute_value_sign_loss"]
    assert signs["member_abs_mean_cancelling"] == signs["member_abs_mean_aligned"]
    assert signs["member_signed_mean_cancelling"] != signs["member_signed_mean_aligned"]
