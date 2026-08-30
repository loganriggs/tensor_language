from __future__ import annotations

import hashlib

import pytest
import torch

import induction_equality_tensor_final_ood_v2_retry1 as subject


class ScalarState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.tensor(1.5, dtype=torch.bfloat16))
        self.vector = torch.nn.Parameter(torch.tensor([2.0, 3.0], dtype=torch.bfloat16))


def _reference_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_scalar_bfloat16_state_hash_is_exact_and_stable():
    model = ScalarState()
    assert subject.model_state_sha256(model) == _reference_hash(model)
    before = subject.model_state_sha256(model)
    with torch.no_grad():
        model.scalar.add_(1)
    assert subject.model_state_sha256(model) != before


def test_retry_uses_fresh_namespace_and_keeps_spent_v2_failure():
    assert subject.FAILURE.name.endswith("v2_retry1_failure.json")
    assert subject.v2.FAILURE == subject.FAILURE
    assert subject.v2.OUTPUTS == (
        subject.AUTHORITY, subject.LEDGER, subject.RESULT,
        subject.MANIFEST, subject.RECEIPT, subject.FAILURE,
    )
    spent = subject.HERE / "induction_equality_tensor_final_ood_v2_failure.json"
    assert spent.is_file()
    assert spent not in subject.v2.OUTPUTS
    lineage = subject._lineage_snapshot()
    assert lineage[str(spent.resolve())] == subject.v2.file_sha256(spent)
    assert all(
        lineage[str(path.resolve())] is None for path in subject.ORIGINAL_ABSENT
    )


def test_retry_source_closure_includes_wrapper_test_and_amendment():
    sources = set(subject.v2.SOURCE_PATHS)
    assert subject.Path(subject.__file__).resolve() in sources
    assert subject.HERE / "test_induction_equality_tensor_final_ood_v2_retry1.py" in sources
    assert subject.AMENDMENT in sources
    assert set(subject.ORIGINAL_REQUIRED) <= sources


def test_lineage_snapshot_rejects_missing_failure_and_reused_receipt(tmp_path, monkeypatch):
    audit = tmp_path / "audit.json"
    authority = tmp_path / "authority.json"
    failure = tmp_path / "failure.json"
    receipt = tmp_path / "receipt.json"
    audit.write_text("{}")
    authority.write_text("{}")
    monkeypatch.setattr(subject, "ORIGINAL_REQUIRED", (audit, authority, failure))
    monkeypatch.setattr(subject, "ORIGINAL_ABSENT", (receipt,))
    with pytest.raises(RuntimeError, match="lineage artifact missing"):
        subject._lineage_snapshot()
    failure.write_text("{}")
    receipt.write_text("{}")
    with pytest.raises(RuntimeError, match="namespace was reused"):
        subject._lineage_snapshot()
