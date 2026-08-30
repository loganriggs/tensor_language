import pytest
import torch

import causal_response_tensor_v1_fit_inputs as inputs


def test_private_production_reconstruction_is_guarded_and_exact():
    calls = []
    owned = inputs._reconstruct_production_fit_inputs_after_authority(
        lambda: calls.append("guarded")
    )
    assert calls == ["guarded", "guarded"]
    assert owned.rows.shape == (1_000, 257)
    assert owned.rows.dtype == torch.int64
    assert owned.row_document_ids.shape == (1_000,)
    assert owned.fit_row_indices.shape == (496,)
    assert owned.model_rows_sha256 == inputs.tensor_sha256(owned.rows)
    assert torch.unique(
        owned.row_document_ids[owned.fit_row_indices]
    ).numel() == 343
    assert owned.fit_document_ids_sha256 == (
        "0f514805a7615e5ef3fe862eb8bf37bebfe8c57b8b7e781fbb25907c729b808d"
    )
    assert len(owned.specs) == 49
    assert owned.spec_order_sha256 == inputs.PRODUCTION_SPEC_ORDER_SHA256
    assert [spec.component for spec in owned.specs] == sorted(
        [spec.component for spec in owned.specs],
        key=lambda component: inputs.PRODUCTION_COMPONENT_ORDER.index(component),
    )
    assert list(owned.support_hashes) == [spec.tag for spec in owned.specs]


def test_no_forgeable_input_capability_or_public_reconstructor_exists():
    assert not hasattr(inputs, "FitInputCapability")
    assert not hasattr(inputs, "reconstruct_production_fit_inputs")


def test_authority_guard_runs_before_parent_access(monkeypatch):
    monkeypatch.setattr(
        inputs, "_load_torch_parent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("parent accessed before authority")
        ),
    )
    with pytest.raises(RuntimeError, match="authority absent"):
        inputs._reconstruct_production_fit_inputs_after_authority(
            lambda: (_ for _ in ()).throw(RuntimeError("authority absent"))
        )


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
        inputs._reconstruct_production_fit_inputs_after_authority(lambda: None)
