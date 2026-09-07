"""Gauge-safe linear algebra for pulling canonical causal-response modes into state space."""
from __future__ import annotations


def pullback(torch, commands, readers, state_basis, source_scale, target_scale, *, rank):
    """Return canonical response scores and physical state covectors.

    ``commands`` and ``readers`` are row-factor matrices in the same orthonormal state gauge,
    so H = commands @ readers.T. Scales are positive per-row diagonal task-balancing factors.
    """
    if commands.ndim != 2 or readers.ndim != 2 or commands.shape[1] != readers.shape[1]:
        raise ValueError("command and reader factors must be matrices with one shared state width")
    if state_basis.ndim != 2 or state_basis.shape[1] != commands.shape[1]:
        raise ValueError("state basis must have the shared state width as columns")
    if source_scale.shape != (commands.shape[0],) or target_scale.shape != (readers.shape[0],):
        raise ValueError("row scale shapes do not match factors")
    if rank < 1 or rank > commands.shape[1]:
        raise ValueError("rank is outside the shared state width")
    cb = commands * source_scale[:, None]
    rb = readers * target_scale[:, None]
    u, singular, vh = torch.linalg.svd(cb @ rb.T, full_matrices=False)
    root = singular[:rank].sqrt()
    source_scores = u[:, :rank] * root
    reader_scores = vh[:rank].T * root
    source_coordinates = torch.linalg.pinv(cb) @ source_scores
    reader_coordinates = torch.linalg.pinv(rb) @ reader_scores
    return {
        "singular_values": singular,
        "source_scores": source_scores,
        "reader_scores": reader_scores,
        "rank_response": source_scores @ reader_scores.T,
        "source_coordinates": source_coordinates,
        "reader_coordinates": reader_coordinates,
        "physical_source_covectors": state_basis @ source_coordinates,
        "physical_reader_covectors": state_basis @ reader_coordinates,
        "source_score_replay": cb @ source_coordinates,
        "reader_score_replay": rb @ reader_coordinates,
    }
