"""Compose the routing-closed head8.2 edge with the exact head9.8 O graph."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from attention8h2_routing_closure_v1 import routing as head8_city_routing
from even_value_shared_graph_v1 import SharedGraph
from odd_source_positions_v1 import select_sources
from odd_source_swap_interaction_v1 import source_factors
from odd_value_source_split_v1 import value_parts


def head8_write(program, block8_current, city_index, recipient_embedding, donor_embedding):
    route = head8_city_routing(program, block8_current, city_index)
    delta_value = program["mixture"] * F.linear(donor_embedding - recipient_embedding, program["value_weight"])
    writer = F.linear(delta_value, program["output_weight"])
    return route[..., None] * writer[:, None, :]


def composed_o_delta(edge_program, o_graph, block8_current, city_index,
                     recipient_embedding, donor_embedding, block9_residual,
                     initial, first_values, reentry, framing_mask):
    write = head8_write(edge_program, block8_current, city_index, recipient_embedding, donor_embedding)
    mask = framing_mask.to(write.device)
    changed_residual = block9_residual + write * mask[None, :, None]
    original_current = F.rms_norm(reentry[0] * block9_residual + reentry[1] * initial, (block9_residual.size(-1),))
    changed_current = F.rms_norm(reentry[0] * changed_residual + reentry[1] * initial, (block9_residual.size(-1),))
    odd_routing, _ = source_factors(o_graph, original_current, original_current, original_current, first_values)
    original_value, _ = value_parts(o_graph, original_current, first_values)
    changed_value, _ = value_parts(o_graph, changed_current, first_values)
    return select_sources(odd_routing * (changed_value - original_value), mask[None]) @ o_graph.p["output"].double().T


def main():
    out = ROOT / "ODD_ATTENTION8H2_O_COMPOSED_EDGE_V1_CPU_CONTROL.json"
    if out.exists():
        raise FileExistsError(out)
    torch.manual_seed(14091848)
    edge = torch.load(ROOT / "ODD_ATTENTION8H2_ROUTING_CLOSURE_V1_PROGRAM.pt", weights_only=True)
    graph = SharedGraph(torch.load(ROOT / "EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt", weights_only=True))
    tokens = 20
    block8 = F.rms_norm(torch.randn(1, tokens, 1152), (1152,))
    block9 = torch.randn_like(block8)
    initial = torch.randn_like(block8)
    first = torch.randn(1, tokens, 9, 128)
    er, ed = F.rms_norm(torch.randn(1, 1152), (1152,)), F.rms_norm(torch.randn(1, 1152), (1152,))
    mask = torch.arange(tokens) >= 4
    reentry = torch.tensor([.9, .1])
    result_tensor = composed_o_delta(edge, graph, block8, 3, er, ed, block9, initial, first, reentry, mask)
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(), "pred_a": tuple(result_tensor.shape) == (1, tokens, 1152) and bool(torch.isfinite(result_tensor).all()),
        "output_shape": list(result_tensor.shape), "output_norm": float(result_tensor.norm()),
        "edge_static_scalars": sum(value.numel() for value in edge.values() if isinstance(value, torch.Tensor)),
        "o_graph_static_scalars": sum(value.numel() for value in graph.p.values() if isinstance(value, torch.Tensor)),
        "scope": "Synthetic execution/shape control for the composed head8.2 inherited-city to head9.8-O delta. No native replay or behavioral claim.",
    }
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
