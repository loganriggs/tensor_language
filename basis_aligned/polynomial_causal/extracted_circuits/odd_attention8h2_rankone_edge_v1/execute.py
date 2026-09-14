"""Standalone conditional executor for the inherited-city head8.2 write."""
from pathlib import Path

import torch
import torch.nn.functional as F


def load_program(path=None):
    path = Path(path) if path else Path(__file__).with_name("program.pt")
    program = torch.load(path, map_location="cpu", weights_only=True)
    required = {"value_weight", "output_weight", "mixture", "head", "source_layer", "writer_layer"}
    if not required.issubset(program):
        raise ValueError("Incomplete rank-one edge program")
    return program


def writer(program, recipient_embedding, donor_embedding):
    """Compute the shared residual writer from normalized city-token embeddings."""
    delta_value = program["mixture"] * F.linear(
        donor_embedding - recipient_embedding, program["value_weight"]
    )
    return F.linear(delta_value, program["output_weight"])


def execute(program, routing, recipient_embedding, donor_embedding):
    """Return destination-by-residual writes from caller-supplied routing scalars."""
    direction = writer(program, recipient_embedding, donor_embedding)
    return routing[..., None] * direction[:, None, :]

