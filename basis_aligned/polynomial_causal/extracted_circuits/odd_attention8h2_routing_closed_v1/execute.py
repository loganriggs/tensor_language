"""Standalone state-conditional executor for the inherited-city head8.2 write."""
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
POLY = HERE.parents[1]
sys.path.insert(0, str(POLY))
from attention8h2_routing_closure_v1 import routing


def load_program(path=None):
    path = Path(path) if path else HERE / "program.pt"
    program = torch.load(path, map_location="cpu", weights_only=True)
    required = {"q1_weight", "k1_weight", "q2_weight", "k2_weight", "value_weight", "output_weight", "mixture"}
    if not required.issubset(program):
        raise ValueError("Incomplete routing-closed program")
    return program


def execute(program, current, city_index, recipient_embedding, donor_embedding):
    city_routing = routing(program, current, city_index)
    delta_value = program["mixture"] * F.linear(
        donor_embedding - recipient_embedding, program["value_weight"]
    )
    writer = F.linear(delta_value, program["output_weight"])
    return city_routing[..., None] * writer[:, None, :]

