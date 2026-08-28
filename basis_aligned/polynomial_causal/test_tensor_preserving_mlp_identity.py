from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tensor_preserving_mlp_identity import MLPNativePoison


def test_mlp_native_poison_replaces_and_restores_objects() -> None:
    blocks = nn.ModuleList([nn.Module() for _ in range(3)])
    originals = []
    for block in blocks:
        block.mlp = nn.Linear(4, 4)
        originals.append(block.mlp)
    model = SimpleNamespace(transformer=SimpleNamespace(h=blocks))
    poison = MLPNativePoison(model)
    with poison.scope():
        for site, block in enumerate(blocks):
            assert block.mlp is poison.installed[site]
            with pytest.raises(RuntimeError, match=f"MLP{site}"):
                block.mlp(torch.randn(2, 4))
    assert poison.restored and poison.inert
    assert all(block.mlp is originals[site] for site, block in enumerate(blocks))
    assert poison.calls == {0: 1, 1: 1, 2: 1}


def test_identity_protocol_requires_bias_poison_and_create_only_result() -> None:
    import tensor_preserving_mlp_identity as identity

    source = identity.Path(identity.__file__).read_text()
    prereg = identity.PREREG.read_text()
    assert "Down_bias" in prereg
    assert "literal_native_mlp_calls_in_program_arm" in source
    assert "storage_disjoint" in source
    assert "286_675_200" in source
    assert "os.O_EXCL" in source
