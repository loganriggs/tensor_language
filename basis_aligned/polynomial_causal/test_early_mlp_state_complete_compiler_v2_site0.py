from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import early_mlp_state_complete_compiler_v2_site0 as site0


def test_site0_binds_every_prelabel_authority_and_closes_runner_test() -> None:
    for path, expected in site0.PINS.items():
        assert site0.file_sha256(path) == expected
    names = {path.name for path in site0.SOURCE_CLOSURE}
    assert "early_mlp_state_complete_compiler_v2_site0.py" in names
    assert "test_early_mlp_state_complete_compiler_v2_site0.py" in names
    protocol = json.loads(site0.SOLVER_PROTOCOL.read_text())
    assert protocol["status"] == "frozen_before_any_compiler_label_or_gradient_capture"


def test_frozen_parameter_guard_restores_flags() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    list(module.parameters())[0].requires_grad_(False)
    before = [parameter.requires_grad for parameter in module.parameters()]
    with site0.FrozenParameters(module):
        assert not any(parameter.requires_grad for parameter in module.parameters())
    assert [parameter.requires_grad for parameter in module.parameters()] == before


def test_copy_and_valid_masks_match_registered_positions() -> None:
    idx = torch.arange(256).view(1, -1)
    targets = torch.arange(1, 257).view(1, -1)
    targets[0, 100] = idx[0, 90]
    valid = site0._valid_mask(targets)
    copy = site0._copy_mask(idx, targets)
    assert not bool(valid[:, :64].any())
    assert bool(valid[:, 64:].all())
    assert copy[0, 100]
    assert not bool(copy[:, :64].any())


def test_artifact_validator_requires_selected_program_and_site1_scope(
    monkeypatch, tmp_path,
) -> None:
    artifact = tmp_path / "program.pt"
    receipt = tmp_path / "receipt.json"
    torch.save({"status": "frozen_before_any_site1_capture",
                "selection": {"selected": "missing"}, "candidates": {}}, artifact)
    receipt.write_text(json.dumps({
        "artifact_sha256": site0.file_sha256(artifact),
        "authorized_for_training": True, "training_license_sites": [1],
    }))
    monkeypatch.setattr(site0, "ARTIFACT", artifact)
    monkeypatch.setattr(site0, "RECEIPT", receipt)
    with pytest.raises(RuntimeError, match="selected program"):
        site0.validate_artifact()


def test_site0_outputs_are_isolated_and_no_final_path_is_referenced() -> None:
    assert all("site0" in path.name for path in site0.OUTPUTS)
    source = site0.Path(site0.__file__).read_text()
    assert 'rows["compiler_final"]' not in source
    assert "compiler_final" not in [path.name for path in site0.OUTPUTS]
