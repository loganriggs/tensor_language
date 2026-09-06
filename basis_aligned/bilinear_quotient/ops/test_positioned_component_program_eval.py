from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

import circuit_fast_screen_producer as producer
import positioned_component_program_eval as positioned


def test_position_banks_allow_different_lengths_but_require_pairing() -> None:
    recipient, donor = positioned.validate_position_banks(
        (4, 7), (8, 3), ((1, 3), (4, 5)), ((5, 6), (0, 2))
    )
    assert recipient == ((1, 3), (4, 5))
    assert donor == ((5, 6), (0, 2))
    with pytest.raises(positioned.PositionedComponentError):
        positioned.validate_position_banks((4,), (8,), ((1, 3),), ((5,),))


def test_full_module_replacement_changes_only_paired_positions() -> None:
    recipient = torch.arange(2 * 5 * 6).view(2, 5, 6).float()
    donor = 1000.0 + torch.arange(2 * 7 * 6).view(2, 7, 6).float()
    changed = positioned._replace_tensor_positions(
        recipient, donor, ((1, 3), (0, 4)), ((2, 6), (3, 5))
    )
    assert torch.equal(changed[0, 1], donor[0, 2])
    assert torch.equal(changed[0, 3], donor[0, 6])
    assert torch.equal(changed[1, 0], donor[1, 3])
    assert torch.equal(changed[1, 4], donor[1, 5])
    assert torch.equal(changed[0, 0], recipient[0, 0])
    assert torch.equal(changed[1, 2], recipient[1, 2])


def test_head_replacement_changes_only_declared_slices() -> None:
    recipient = torch.zeros((1, 4, 12))
    donor = torch.arange(1 * 5 * 12).view(1, 5, 12).float()
    changed = positioned._replace_head_positions(
        recipient, donor, ((2,),), ((3,),), (0, 2), 3
    )
    assert torch.equal(changed[0, 2, 0:4], donor[0, 3, 0:4])
    assert torch.equal(changed[0, 2, 8:12], donor[0, 3, 8:12])
    assert torch.equal(changed[0, 2, 4:8], recipient[0, 2, 4:8])
    assert torch.equal(changed[0, 1], recipient[0, 1])


def test_component_validation_rejects_overlapping_module_declarations() -> None:
    with pytest.raises(positioned.PositionedComponentError):
        positioned.validate_components(
            (
                positioned.Component("attention_heads", 9, (1,)),
                positioned.Component("attention_heads", 9, (4,)),
            ),
            layers=18, heads=9,
        )
    assert positioned.validate_components(
        (
            positioned.Component("mlp", 4),
            positioned.Component("attention_heads", 9, (1, 4)),
        ),
        layers=18, heads=9,
    )


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_proj = torch.nn.Identity()

    def forward(self, value):
        return self.c_proj(value)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()
        self.mlp = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.mlp.weight.copy_(2.0 * torch.eye(4))


class _HookBackend:
    def __init__(self):
        self.model = SimpleNamespace(
            transformer=SimpleNamespace(h=torch.nn.ModuleList([_Block()])),
            config=SimpleNamespace(n_head=2),
        )

    def native(self, batch, *, capture):
        maximum = max(map(len, batch.token_rows))
        rows = [list(row) + [0] * (maximum - len(row)) for row in batch.token_rows]
        scalar = torch.tensor(rows, dtype=torch.float32)
        value = torch.stack(tuple(scalar + offset for offset in range(4)), dim=-1)
        block = self.model.transformer.h[0]
        attention = block.attn(value)
        live = value + attention
        output = live + block.mlp(live)
        pairs = tuple((float(output[i, position].sum().detach()), 0.0)
                      for i, position in enumerate(batch.semantic_positions))
        return producer.BatchOutput(pairs, {})


def _batch(side, tokens, position=2):
    return producer.ModelBatch(
        row_ids=("r0",), side=side, token_rows=(tuple(tokens),),
        answer_ids=(0,), foil_ids=(1,), semantic_positions=(position,),
    )


def test_capture_and_joint_hook_patch_execute_end_to_end() -> None:
    backend = _HookBackend()
    recipient = _batch("base", (1, 2, 3, 4), position=1)
    donor = _batch("donor", (7, 8, 9, 10), position=2)
    components = (
        positioned.Component("attention_heads", 0, (1,)),
        positioned.Component("mlp", 0),
    )
    _native, cache = positioned.capture_full_components(backend, donor, components)
    output = positioned.patch_positioned_components(
        backend, recipient, donor, components, cache, ((1,),), ((2,),)
    )
    # At the recipient query, head 1 comes from the donor's position 2 and the
    # MLP output is then exactly replaced by that donor position's MLP output.
    assert output.answer_foil == ((210.0, 0.0),)
    assert backend.native(recipient, capture=False).answer_foil == ((84.0, 0.0),)
    assert not backend.model.transformer.h[0].attn.c_proj._forward_pre_hooks
    assert not backend.model.transformer.h[0].mlp._forward_hooks
