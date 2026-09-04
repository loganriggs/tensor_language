"""CPU-only tests for the Task 14 head-11.3 projector adapter."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


OPS = Path(__file__).parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))
PATH = OPS / "task14_head11_3_projector_adapter.py"
SPEC = importlib.util.spec_from_file_location("task14_head11_3_projector_adapter", PATH)
ADAPTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADAPTER
SPEC.loader.exec_module(ADAPTER)


def _values(batch: int = 2, sequence: int = 4) -> torch.Tensor:
    count = batch * sequence * ADAPTER.MODEL_WIDTH
    return torch.arange(count, dtype=torch.float64).reshape(
        batch, sequence, ADAPTER.MODEL_WIDTH
    )


def _cache(row_ids, donors):
    return {
        (row_id, ADAPTER.SITE_ID): donor
        for row_id, donor in zip(row_ids, donors)
    }


def test_zero_space_is_exact_identity_and_does_not_alias_input() -> None:
    recipient = torch.randn(3, ADAPTER.HEAD_WIDTH, dtype=torch.float64)
    donor = torch.randn_like(recipient)
    empty = torch.empty(ADAPTER.HEAD_WIDTH, 0, dtype=torch.float64)
    patched = ADAPTER.projected_head_interchange(recipient, donor, empty)
    assert torch.equal(patched, recipient)
    assert patched.data_ptr() != recipient.data_ptr()


def test_full_space_is_exact_full_donor_interchange() -> None:
    recipient = torch.randn(2, ADAPTER.HEAD_WIDTH, dtype=torch.float64)
    donor = torch.randn_like(recipient)
    full = torch.eye(ADAPTER.HEAD_WIDTH, dtype=torch.float64)
    patched = ADAPTER.projected_head_interchange(recipient, donor, full)
    assert torch.equal(patched, donor)


def test_action_is_invariant_to_orthogonal_basis_rotation() -> None:
    generator = torch.Generator().manual_seed(141103)
    recipient = torch.randn(7, ADAPTER.HEAD_WIDTH, generator=generator, dtype=torch.float64)
    donor = torch.randn(7, ADAPTER.HEAD_WIDTH, generator=generator, dtype=torch.float64)
    frame = torch.linalg.qr(
        torch.randn(ADAPTER.HEAD_WIDTH, 8, generator=generator, dtype=torch.float64),
        mode="reduced",
    )[0]
    rotation = torch.linalg.qr(
        torch.randn(8, 8, generator=generator, dtype=torch.float64), mode="reduced"
    )[0]
    original = ADAPTER.projected_head_interchange(recipient, donor, frame)
    rotated = ADAPTER.projected_head_interchange(recipient, donor, frame @ rotation)
    assert torch.allclose(original, rotated, atol=3e-12, rtol=0)


def test_pre_hook_changes_only_head_3_at_declared_positions_before_c_proj() -> None:
    value = _values()
    original = value.clone()
    row_ids = ("row-a", "row-b")
    positions = (1, 3)
    donors = (
        torch.linspace(-4.0, 4.0, ADAPTER.HEAD_WIDTH, dtype=torch.float64),
        torch.linspace(10.0, 20.0, ADAPTER.HEAD_WIDTH, dtype=torch.float64),
    )
    adapter = ADAPTER.Head11_3ProjectedInterchange(
        torch.eye(ADAPTER.HEAD_WIDTH, dtype=torch.float64)
    )
    sentinel = object()
    arguments = adapter.pre_output_projection_hook(
        (value, sentinel),
        row_ids=row_ids,
        semantic_positions=positions,
        donor_cache=_cache(row_ids, donors),
    )
    patched, preserved = arguments
    assert preserved is sentinel
    assert torch.equal(value, original)

    expected = original.clone()
    for index, (position, donor) in enumerate(zip(positions, donors)):
        expected[index, position, ADAPTER.HEAD_START:ADAPTER.HEAD_STOP] = donor
    assert torch.equal(patched, expected)

    # A forward-pre-hook must alter c_proj's input, not its residual-space
    # output.  An arbitrary linear projection therefore sees exactly the
    # patched concatenated heads.
    projection = torch.randn(
        ADAPTER.MODEL_WIDTH, 19,
        generator=torch.Generator().manual_seed(141104), dtype=torch.float64,
    )
    assert torch.equal(patched @ projection, expected @ projection)
    assert not torch.equal(patched @ projection, original @ projection)


def test_partial_frame_keeps_orthogonal_head_coordinates_from_recipient() -> None:
    frame = torch.eye(ADAPTER.HEAD_WIDTH, dtype=torch.float64)[:, :5]
    adapter = ADAPTER.Head11_3ProjectedInterchange(frame)
    value = torch.zeros(1, 2, ADAPTER.MODEL_WIDTH, dtype=torch.float64)
    donor = torch.arange(ADAPTER.HEAD_WIDTH, dtype=torch.float64) + 1
    patched = adapter.patch_c_proj_input(
        value,
        row_ids=("row",),
        semantic_positions=(1,),
        donor_cache=_cache(("row",), (donor,)),
    )
    head = patched[0, 1, ADAPTER.HEAD_START:ADAPTER.HEAD_STOP]
    assert torch.equal(head[:5], donor[:5])
    assert torch.equal(head[5:], torch.zeros_like(head[5:]))
    assert torch.equal(patched[0, 0], value[0, 0])
    assert torch.equal(patched[0, 1, :ADAPTER.HEAD_START], value[0, 1, :ADAPTER.HEAD_START])
    assert torch.equal(patched[0, 1, ADAPTER.HEAD_STOP:], value[0, 1, ADAPTER.HEAD_STOP:])


def test_dryrun_cannot_import_model_loader_or_claim_fit(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "fastload" or name.startswith("fastload."):
            raise AssertionError("dry-run attempted to import the model loader")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    plan = ADAPTER.compile_dryrun(8)
    assert plan == {
        "schema": "task14_head11_3_projector_adapter_dryrun_v1",
        "site_id": "attn:11:head:03",
        "layer": 11,
        "head": 3,
        "ambient_dimension": 128,
        "rank": 8,
        "equation": "o_base + ((o_donor - o_base) @ U) @ U.T",
        "hook": "model.transformer.h[11].attn.c_proj.forward_pre_hook",
        "projector_basis_gauge_invariant": True,
        "model_loaded": False,
        "scientific_data_read": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "rank_selected": False,
        "frame_fitted": False,
    }


def test_invalid_frames_positions_and_cache_fail_closed() -> None:
    with pytest.raises(ADAPTER.ProjectedHeadInterchangeError, match="orthonormal"):
        ADAPTER.Head11_3ProjectedInterchange(
            torch.ones(ADAPTER.HEAD_WIDTH, 2, dtype=torch.float64)
        )
    adapter = ADAPTER.Head11_3ProjectedInterchange(
        torch.eye(ADAPTER.HEAD_WIDTH, dtype=torch.float64)[:, :1]
    )
    value = torch.zeros(1, 2, ADAPTER.MODEL_WIDTH, dtype=torch.float64)
    with pytest.raises(ADAPTER.ProjectedHeadInterchangeError, match="outside"):
        adapter.patch_c_proj_input(
            value, row_ids=("row",), semantic_positions=(2,), donor_cache={}
        )
    with pytest.raises(ADAPTER.ProjectedHeadInterchangeError, match="donor cache"):
        adapter.patch_c_proj_input(
            value, row_ids=("row",), semantic_positions=(1,), donor_cache={}
        )

