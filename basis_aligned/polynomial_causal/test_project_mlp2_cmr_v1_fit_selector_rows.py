from __future__ import annotations

import inspect

import pytest
import torch

import project_mlp2_cmr_v1_fit_selector_rows as projection


def real_role() -> dict[str, torch.Tensor]:
    return {
        name: value.clone()
        for name, value in torch.load(
            projection.COMBINED, map_location="cpu", weights_only=True,
        )["FIT_SELECTOR"].items()
    }


def test_real_fit_selector_role_has_exact_semantics() -> None:
    summary = projection.validate_role(real_role())
    assert summary["eligible_positions"] == 31_505
    assert summary["support_documents"] == 191
    assert summary["all_false_ordinals"] == [82]
    assert summary["tensor_hashes"] == projection.EXPECTED_TENSOR_HASHES


def test_role_validation_rejects_mask_or_identity_drift() -> None:
    role = real_role()
    role["eligible_mask"][0, 0] = True
    with pytest.raises(RuntimeError, match="identity changed"):
        projection.validate_role(role)
    role = real_role()
    role["rows"][0, 0] = (role["rows"][0, 0] + 1) % projection.EOT
    with pytest.raises(RuntimeError, match="identity changed"):
        projection.validate_role(role)


def test_parent_receipt_manifest_joins_and_hashes_replay() -> None:
    hashes, captured = projection.parent_snapshot()
    assert hashes["combined"] == projection.COMBINED_SHA256
    assert hashes["combined_manifest"] == projection.COMBINED_MANIFEST_SHA256
    assert hashes["combined_receipt"] == projection.COMBINED_RECEIPT_SHA256
    assert set(captured) == {"combined", "combined_manifest", "combined_receipt"}


def test_projection_is_model_free_role_only_and_receipt_last() -> None:
    source = inspect.getsource(projection.project)
    assert "load_bilin18" not in source
    assert 'combined[\n        "FIT_SELECTOR"\n    ]' in source
    assert '"VALIDATION"' in source and '"REPLICATION"' in source
    assert "del combined, captured" in source
    assert source.rfind("write_create_only(RECEIPT") > source.rfind("final_guard(")
    receipt_tail = source[source.rfind("write_create_only(RECEIPT"):]
    assert receipt_tail.count("write_") == 1


def test_create_only_writer_fsyncs_and_never_overwrites(tmp_path) -> None:
    path = tmp_path / "x.json"
    projection.write_create_only(path, b"{}")
    with pytest.raises(FileExistsError):
        projection.write_create_only(path, b"no")
    assert path.read_bytes() == b"{}"
