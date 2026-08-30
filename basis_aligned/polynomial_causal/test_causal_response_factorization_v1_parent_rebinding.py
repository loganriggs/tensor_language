import inspect

import pytest

import causal_response_factorization_v1_parent_binding as parent
import causal_response_factorization_v1_parent_rebinding as rebinding


def test_rebinding_repeats_the_published_validation_except_the_comparator():
    """Every check of the published binding must survive in the rebinding."""

    published = inspect.getsource(parent.fit_parent_binding_without_tensor_load)
    ours = inspect.getsource(rebinding.fit_parent_binding_by_content_identity)
    for message in (
        "FIT authority semantics changed",
        "FIT authority identity does not replay",
        "FIT authority output namespace differs from the bound parent",
        "FIT manifest semantics changed",
        "FIT manifest identity does not replay",
        "FIT success receipt payload does not replay",
        "FIT terminal or receipt changed during parent validation",
        "changed during terminal replay",
    ):
        assert message in published and message in ours
    assert "_same_receipt_bound_artifact" not in ours
    assert "_same_content_identity" in ours
    assert rebinding.CONTENT_IDENTITY_KEYS == ("path", "sha256", "bytes")


def test_content_identity_ignores_physical_identity_but_not_bytes():
    expected = {
        "path": "p", "present": True, "sha256": "a" * 64, "bytes": 3,
        "device": 1, "inode": 2, "mtime_ns": 3, "ctime_ns": 4,
    }
    observed = {**expected, "device": 9, "inode": 8, "mtime_ns": 7, "ctime_ns": 6}
    assert rebinding._same_content_identity(expected, observed) is True
    assert rebinding._same_content_identity(expected, {**observed, "bytes": 4}) is False
    assert rebinding._same_content_identity(expected, {**observed, "sha256": "b" * 64}) is False
    assert rebinding._same_content_identity(expected, {**observed, "path": "q"}) is False
    missing = {key: value for key, value in observed.items() if key != "inode"}
    assert rebinding._same_content_identity(expected, missing) is False


def test_production_rebinding_replays_and_reports_physical_deviation():
    """On this box the published binding cannot replay; the rebinding must, honestly."""

    with pytest.raises(RuntimeError, match="artifact changed"):
        parent.fit_parent_binding_without_tensor_load()
    binding = rebinding.fit_parent_binding_by_content_identity()
    assert binding["schema"] == "causal_response_factorization_v1_fit_parent_binding"
    assert binding["tensor_values_deserialized"] is False
    assert binding["authorized_for_eval"] is False
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    assert binding["binding_sha256"] == parent._logical_sha256(body)
    deviation = rebinding.physical_identity_deviation()
    assert set(deviation["artifacts"]) == {"authority", "bundle", "manifest"}
    for record in deviation["artifacts"].values():
        assert record["content_identity_matches"] is True
        assert record["sha256"] == record["sha256"]
        assert set(record["recorded"]) == set(record["observed"]) == {
            "device", "inode", "mtime_ns", "ctime_ns",
        }
    assert binding["bundle_sha256"] == deviation["artifacts"]["bundle"]["sha256"]
