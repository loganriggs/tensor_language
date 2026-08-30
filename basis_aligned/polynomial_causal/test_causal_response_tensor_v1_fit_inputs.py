import inspect
import copy

import pytest
import torch

import causal_response_tensor_v1_fit_inputs as inputs


def test_production_reconstruction_has_no_caller_control_and_exact_role_specs():
    assert list(inspect.signature(inputs.reconstruct_production_fit_inputs).parameters) == []
    capability = inputs.reconstruct_production_fit_inputs()
    owned = capability.take_once()
    assert owned.rows.shape == (1_000, 257)
    assert owned.rows.dtype == torch.int64
    assert owned.row_document_ids.shape == (1_000,)
    assert owned.fit_row_indices.shape == (496,)
    assert owned.model_rows_sha256 == inputs.tensor_sha256(owned.rows)
    assert torch.unique(
        owned.row_document_ids[owned.fit_row_indices]
    ).numel() == 343
    assert len(owned.specs) == 49
    assert owned.spec_order_sha256 == inputs.PRODUCTION_SPEC_ORDER_SHA256
    assert [spec.component for spec in owned.specs] == sorted(
        [spec.component for spec in owned.specs],
        key=lambda component: inputs.PRODUCTION_COMPONENT_ORDER.index(component),
    )
    assert list(owned.support_hashes) == [spec.tag for spec in owned.specs]


def test_input_capability_is_one_use_and_owns_returned_storage():
    capability = inputs.reconstruct_production_fit_inputs()
    owned = capability.take_once()
    original_hash = inputs.tensor_sha256(owned.specs[0].member_mask)
    owned.rows.zero_()
    owned.fit_row_indices.zero_()
    owned.specs[0].member_mask.zero_()
    assert owned.support_hashes[owned.specs[0].tag]["member_mask_sha256"] == original_hash
    with pytest.raises(RuntimeError, match="cannot be copied"):
        copy.copy(capability)
    with pytest.raises(RuntimeError, match="already spent"):
        capability.take_once()


def test_input_capability_cannot_be_constructed_externally():
    with pytest.raises(RuntimeError, match="cannot be constructed"):
        inputs.FitInputCapability()


def test_parent_hash_tampering_fails_before_deserialization(monkeypatch, tmp_path):
    changed = tmp_path / "split.json"
    changed.write_bytes(inputs.SPLIT.read_bytes() + b"\n")
    monkeypatch.setattr(inputs, "SPLIT", changed)
    monkeypatch.setattr(
        inputs.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tensor parent deserialized after split hash drift")
        ),
    )
    # Tensor parents are read first in the production order, so target the census
    # path as well to prove the hash guard precedes its torch.load.
    changed_census = tmp_path / "census.pt"
    changed_census.write_bytes(b"not the frozen census")
    monkeypatch.setattr(inputs, "CENSUS", changed_census)
    with pytest.raises(RuntimeError, match="parent hash changed"):
        inputs.reconstruct_production_fit_inputs()
