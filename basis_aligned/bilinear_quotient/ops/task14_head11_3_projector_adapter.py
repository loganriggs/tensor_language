"""CPU-safe projected interchange adapter for Task 14 head 11.3.

The scientific operation is applied to the 128 values emitted by head 3 at
attention layer 11, immediately before the concatenated heads enter ``c_proj``::

    o_patched = o_base + ((o_donor - o_base) @ U) @ U.T

``U`` has shape ``[128, rank]`` and orthonormal columns.  The operation depends
only on the projector ``U U.T`` and is therefore invariant to replacing ``U``
by ``U R`` for any orthogonal within-subspace change of basis ``R``.

This module imports no checkpoint, model loader, dataset, CUDA helper, or queue
code.  It reuses the existing DAS projector algebra and only supplies the exact
head-slice/semantic-position scatter needed by the Task 14 fast-screen runner.
It does not choose a rank, fit a frame, or open any scientific data partition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

import das_shared_private_lib as daslib


SITE_ID = "attn:11:head:03"
LAYER = 11
HEAD = 3
MODEL_WIDTH = 1152
HEAD_COUNT = 9
HEAD_WIDTH = MODEL_WIDTH // HEAD_COUNT
HEAD_START = HEAD * HEAD_WIDTH
HEAD_STOP = (HEAD + 1) * HEAD_WIDTH
DEFAULT_ORTHONORMALITY_ATOL = 1e-5


class ProjectedHeadInterchangeError(ValueError):
    """The frame or pre-output-projection intervention contract is invalid."""


def validate_head_frame(
    frame: torch.Tensor, *, atol: float = DEFAULT_ORTHONORMALITY_ATOL
) -> None:
    """Validate an orthonormal head-space frame, including the empty frame.

    ``das_shared_private_lib`` deliberately requires positive rank.  Here an
    empty ``[128, 0]`` frame is useful as the exact no-intervention endpoint.
    """

    if not isinstance(frame, torch.Tensor) or frame.ndim != 2:
        raise ProjectedHeadInterchangeError("frame must be a rank-2 torch tensor")
    if tuple(frame.shape[:1]) != (HEAD_WIDTH,) or frame.shape[1] > HEAD_WIDTH:
        raise ProjectedHeadInterchangeError(
            f"frame must have shape ({HEAD_WIDTH}, rank), 0 <= rank <= {HEAD_WIDTH}"
        )
    if not frame.is_floating_point():
        raise ProjectedHeadInterchangeError("frame must have a floating dtype")
    if not math.isfinite(atol) or atol <= 0:
        raise ProjectedHeadInterchangeError("atol must be finite and positive")
    if not bool(torch.isfinite(frame).all().detach().cpu()):
        raise ProjectedHeadInterchangeError("frame contains a non-finite value")
    if frame.shape[1] == 0:
        return
    try:
        daslib.validate_orthonormal_frame(frame, atol=atol, name="head11_3_frame")
    except ValueError as error:
        raise ProjectedHeadInterchangeError(str(error)) from error


def projected_head_interchange(
    recipient: torch.Tensor,
    donor: torch.Tensor,
    frame: torch.Tensor,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Interchange only the subspace identified by ``frame`` in head space."""

    if recipient.shape != donor.shape:
        raise ProjectedHeadInterchangeError(
            "recipient and donor head values must have identical shapes"
        )
    if recipient.ndim == 0 or recipient.shape[-1] != HEAD_WIDTH:
        raise ProjectedHeadInterchangeError(
            f"head values must have final dimension {HEAD_WIDTH}"
        )
    if not recipient.is_floating_point() or not donor.is_floating_point():
        raise ProjectedHeadInterchangeError("head values must have floating dtypes")
    if recipient.device != donor.device:
        raise ProjectedHeadInterchangeError(
            "recipient and donor head values must occupy the same device"
        )
    if validate:
        validate_head_frame(frame)
    if frame.shape[1] == 0:
        return recipient.clone()
    # A square orthonormal frame spans the whole head space.  Return the donor
    # directly so the full-space endpoint exactly reproduces the already-audited
    # whole-head interchange rather than differing by floating-point roundoff.
    if frame.shape[1] == HEAD_WIDTH:
        return donor.to(dtype=recipient.dtype).clone()
    local_frame = frame.to(device=recipient.device)
    calculation_dtype = torch.promote_types(
        torch.promote_types(recipient.dtype, donor.dtype), local_frame.dtype
    )
    recipient_work = recipient.to(dtype=calculation_dtype)
    donor_work = donor.to(dtype=calculation_dtype)
    local_frame = local_frame.to(dtype=calculation_dtype)
    # The frame was validated before conversion.  Avoid rejecting a legal
    # frame merely because a low-precision model dtype rounds U^T U.
    patched = daslib.projection_interchange(
        recipient_work, donor_work, local_frame, validate=False
    )
    return patched.to(dtype=recipient.dtype)


@dataclass(frozen=True)
class Head11_3ProjectedInterchange:
    """Scatter projected donor values into the exact pre-``c_proj`` head slice."""

    frame: torch.Tensor
    orthonormality_atol: float = DEFAULT_ORTHONORMALITY_ATOL

    def __post_init__(self) -> None:
        validate_head_frame(self.frame, atol=self.orthonormality_atol)

    @property
    def rank(self) -> int:
        return int(self.frame.shape[1])

    def patch_c_proj_input(
        self,
        value: torch.Tensor,
        *,
        row_ids: Sequence[str],
        semantic_positions: Sequence[int],
        donor_cache: Mapping[tuple[str, str], object],
    ) -> torch.Tensor:
        """Patch declared row/position pairs in concatenated pre-projection heads.

        ``value`` is the first positional argument of attention layer 11's
        ``c_proj``.  It has shape ``[batch, sequence, 1152]``.  Donor cache
        entries are the corresponding detached ``[128]`` head-3 values.
        """

        if not isinstance(value, torch.Tensor) or value.ndim != 3 \
                or value.shape[-1] != MODEL_WIDTH:
            raise ProjectedHeadInterchangeError(
                f"c_proj input must have shape [batch, sequence, {MODEL_WIDTH}]"
            )
        if len(row_ids) != value.shape[0] or len(semantic_positions) != value.shape[0]:
            raise ProjectedHeadInterchangeError(
                "row IDs and semantic positions must match the batch dimension"
            )
        if len(set(row_ids)) != len(row_ids):
            raise ProjectedHeadInterchangeError("row IDs must be unique within a batch")

        changed = value.clone()
        for batch_index, (row_id, position) in enumerate(
            zip(row_ids, semantic_positions)
        ):
            if not isinstance(row_id, str) or not row_id:
                raise ProjectedHeadInterchangeError("row IDs must be nonempty strings")
            if type(position) is not int or not 0 <= position < value.shape[1]:
                raise ProjectedHeadInterchangeError(
                    f"semantic position is outside the c_proj input for {row_id}"
                )
            donor = donor_cache.get((row_id, SITE_ID))
            if not isinstance(donor, torch.Tensor) or tuple(donor.shape) != (HEAD_WIDTH,):
                raise ProjectedHeadInterchangeError(
                    f"donor cache lacks exact {HEAD_WIDTH}-value slice {row_id}/{SITE_ID}"
                )
            recipient = value[batch_index, position, HEAD_START:HEAD_STOP]
            donor = donor.to(device=value.device, dtype=value.dtype)
            changed[batch_index, position, HEAD_START:HEAD_STOP] = (
                projected_head_interchange(
                    recipient, donor, self.frame, validate=False
                )
            )
        return changed

    def pre_output_projection_hook(
        self,
        arguments: tuple[object, ...],
        *,
        row_ids: Sequence[str],
        semantic_positions: Sequence[int],
        donor_cache: Mapping[tuple[str, str], object],
    ) -> tuple[object, ...]:
        """Return the tuple expected from a PyTorch ``forward_pre_hook``.

        Register this only on ``model.transformer.h[11].attn.c_proj``.  Keeping
        the method at the argument-tuple boundary makes it impossible to confuse
        this 128-dimensional head-value frame with the 1,152-dimensional
        post-``c_proj`` residual write.
        """

        if not isinstance(arguments, tuple) or not arguments:
            raise ProjectedHeadInterchangeError(
                "c_proj forward-pre-hook arguments must be a nonempty tuple"
            )
        value = arguments[0]
        if not isinstance(value, torch.Tensor):
            raise ProjectedHeadInterchangeError(
                "the first c_proj argument must be a torch tensor"
            )
        patched = self.patch_c_proj_input(
            value,
            row_ids=row_ids,
            semantic_positions=semantic_positions,
            donor_cache=donor_cache,
        )
        return (patched,) + arguments[1:]


def compile_dryrun(rank: int) -> dict[str, object]:
    """Describe the adapter without loading a model or reading scientific data."""

    if type(rank) is not int or not 0 <= rank <= HEAD_WIDTH:
        raise ProjectedHeadInterchangeError(
            f"rank must be an integer in [0, {HEAD_WIDTH}]"
        )
    return {
        "schema": "task14_head11_3_projector_adapter_dryrun_v1",
        "site_id": SITE_ID,
        "layer": LAYER,
        "head": HEAD,
        "ambient_dimension": HEAD_WIDTH,
        "rank": rank,
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


__all__ = [
    "DEFAULT_ORTHONORMALITY_ATOL",
    "HEAD",
    "HEAD_COUNT",
    "HEAD_START",
    "HEAD_STOP",
    "HEAD_WIDTH",
    "LAYER",
    "MODEL_WIDTH",
    "SITE_ID",
    "Head11_3ProjectedInterchange",
    "ProjectedHeadInterchangeError",
    "compile_dryrun",
    "projected_head_interchange",
    "validate_head_frame",
]
