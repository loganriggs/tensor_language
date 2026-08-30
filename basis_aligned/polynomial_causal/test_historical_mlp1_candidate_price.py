from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

import audit_historical_mlp1_candidate_price as subject


def test_frozen_historical_candidate_price_is_complete_and_not_correction_only():
    result = subject.audit()
    storage = result["literal_storage_reals"]
    runtime = result["runtime_multiplies_per_token"]
    assert storage["token_table"] == 50_257 * 1_152
    assert storage["ridge"] == 2_304 * 1_152 + 2_304 + 1_152
    assert storage["rank64_predictor"] == 80_192
    assert storage["rank64_basis"] == 1_152 * 64
    assert storage["rank64_correction"] == 153_920
    assert storage["complete_candidate"] == 60_707_648
    assert storage["complete_candidate_over_native"] > 3.8
    assert storage["correction_only_over_native"] < 0.01
    assert runtime["complete_candidate_over_native"] < 0.19
    assert result["semantics"]["depends_on_live_mlp0_write"] is True
    assert result["decision"]["run_as_current_standalone_simplification"] is False
    assert result["parents"]["compiler_v21_programs"]["selected_true_site1"] == (
        "B_l6_r64"
    )
    assert len(result["parents"]["external_rank64_basis"]["site1_tensor_sha256"]) == 64
    assert result["source_binding"] is None


def test_stable_torch_rejects_a_to_b_to_a_swap(tmp_path: Path, monkeypatch):
    path = tmp_path / "parent.pt"
    torch.save({"value": torch.tensor([1.0])}, path)
    original = path.read_bytes()
    alternate_path = tmp_path / "alternate.pt"
    torch.save({"value": torch.tensor([2.0])}, alternate_path)
    alternate = alternate_path.read_bytes()
    assert hashlib.sha256(original).digest() != hashlib.sha256(alternate).digest()

    real_read_bytes = Path.read_bytes

    def swapped_read_bytes(candidate: Path) -> bytes:
        if candidate == path:
            return alternate
        return real_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", swapped_read_bytes)
    with pytest.raises(RuntimeError, match="tensor parent raced read"):
        subject.stable_torch(path)
