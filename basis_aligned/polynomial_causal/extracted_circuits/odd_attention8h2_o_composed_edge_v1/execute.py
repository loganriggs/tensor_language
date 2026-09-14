"""Standalone conditional head8.2 inherited-city to head9.8-O executor."""
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
POLY = HERE.parents[1]
sys.path.insert(0, str(POLY))
from even_value_shared_graph_v1 import SharedGraph
from odd_attention8h2_o_composed_edge_v1 import composed_o_delta


def load_program(path=None):
    path = Path(path) if path else HERE / "program.pt"
    program = torch.load(path, map_location="cpu", weights_only=True)
    if not {"edge", "o_graph"}.issubset(program):
        raise ValueError("Incomplete composed edge program")
    return program


def execute(program, block8_current, city_index, recipient_embedding,
            donor_embedding, block9_residual, initial, first_values, reentry,
            framing_mask):
    graph = SharedGraph(program["o_graph"])
    return composed_o_delta(
        program["edge"], graph, block8_current, city_index,
        recipient_embedding, donor_embedding, block9_residual, initial,
        first_values, reentry, framing_mask,
    )

