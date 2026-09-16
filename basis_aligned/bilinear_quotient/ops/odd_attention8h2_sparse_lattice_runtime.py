"""Shared native measurement runtime for the inherited-city three-port lattice."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from attention8h2_city_key_value_v1 import factorial_channels
from attention8h2_city_value_source_v1 import value_arms
from odd_contextual_positions_v1 import contextual_masks
from odd_semantic_positions_v1 import semantic_masks
from odd_source_positions_v1 import select_sources
from odd_source_swap_interaction_v1 import source_factors
from odd_value_source_split_v1 import value_parts
from run_odd_attention8h2_city_key_value_v1 import head_factor_parts


READOUTS = [
    ("target", None), ("cat_dog", (3797, 3290)),
    ("red_blue", (2266, 4171)), ("monday_tuesday", (3321, 3431)),
    ("apple_orange", (17180, 10912)),
]


@torch.no_grad()
def measure_lattice(model, graph, discovery, natural):
    """Measure all eight routing/current/inherited behavioral corners."""
    rows = list(discovery) + list(natural)
    discovery_semantic = semantic_masks(discovery)
    discovery_context = contextual_masks(discovery)
    state = {"corner": 0}
    attention_inputs, preprojection = [], []
    source_replays, factorial_errors, value_errors = [], [], []
    full_channel_errors, outside = [], []
    write_norms = {corner: [] for corner in range(1, 8)}
    attention8 = model.transformer.h[8].attn
    output_weight = attention8.c_proj.weight[:, 2 * 128:3 * 128]

    def masks(index, device):
        if index < len(discovery):
            city = discovery_context[index]["city"].to(device)
            destination = discovery_semantic[index]["framing"].to(device)
        else:
            row = rows[index]
            city = torch.zeros(len(row["ids"]), dtype=torch.bool, device=device)
            city[row["city_position"]] = True
            destination = torch.zeros_like(city)
            destination[row["destination_positions"]] = True
        return city, destination

    def attention8_pre(module, args):
        if state["corner"] == 0:
            attention_inputs.append((args[0].detach().cpu(), args[1].detach().cpu()))

    def projection_pre(module, args):
        if state["corner"] == 0:
            preprojection.append(args[0].detach().cpu())

    def block9_pre(module, args):
        x, _, x0 = args
        corner = state["corner"]
        if corner == 0:
            return None
        index, donor = state["index"], state["index"] ^ 1
        city, destination = masks(index, x.device)
        recipient_current, recipient_first = [value.to(x.device) for value in attention_inputs[index]]
        donor_current, donor_first = [value.to(x.device) for value in attention_inputs[donor]]
        donor_key = recipient_current.clone()
        donor_key[:, city] = donor_current[:, city]
        routing_recipient, _ = head_factor_parts(
            attention8, recipient_current, recipient_current, recipient_current, recipient_first
        )
        routing_donor, _ = head_factor_parts(
            attention8, recipient_current, donor_key, recipient_current, recipient_first
        )
        batch, tokens, _ = recipient_current.shape
        current_recipient = attention8.c_v(recipient_current).view(
            batch, tokens, attention8.n_head, attention8.head_dim
        )[:, :, 2]
        current_donor = attention8.c_v(donor_current).view(
            batch, tokens, attention8.n_head, attention8.head_dim
        )[:, :, 2]
        first_recipient = recipient_first.view(
            batch, tokens, attention8.n_head, attention8.head_dim
        )[:, :, 2]
        first_donor = donor_first.view(
            batch, tokens, attention8.n_head, attention8.head_dim
        )[:, :, 2]
        value_corners = value_arms(
            attention8.lamb, current_recipient, current_donor, first_recipient, first_donor
        )
        factors = factorial_channels(
            routing_recipient, routing_donor,
            value_corners["recipient"][:, None], value_corners["donor"][:, None],
        )
        original = factors["recipient"]
        source_replays.append(float(
            (original.sum(-2) - preprojection[index].to(x.device)[..., 2 * 128:3 * 128]).norm()
            / original.sum(-2).norm().clamp_min(1e-8)
        ))
        factorial_errors.append(float(
            (factors["additive"] + factors["mixed"] - factors["donor"]).norm()
            / factors["donor"].norm().clamp_min(1e-8)
        ))
        value_errors.append(float(
            ((value_corners["current"] - value_corners["recipient"])
             + (value_corners["first"] - value_corners["recipient"])
             - (value_corners["donor"] - value_corners["recipient"])).norm()
            / (value_corners["donor"] - value_corners["recipient"]).norm().clamp_min(1e-8)
        ))
        routing = routing_donor if corner & 1 else routing_recipient
        current = current_donor if corner & 2 else current_recipient
        first = first_donor if corner & 4 else first_recipient
        value = (1 - attention8.lamb) * current + attention8.lamb * first
        channels = routing[..., None] * value[:, None]
        if corner == 7:
            full_channel_errors.append(float(
                (channels - factors["donor"]).norm() / factors["donor"].norm().clamp_min(1e-8)
            ))
        write_delta = F.linear(select_sources(channels - original, city[None]), output_weight)
        write_norms[corner].append(float(write_delta[:, destination].norm()))
        hybrid = x.clone()
        hybrid[:, destination] += write_delta[:, destination]
        outside.append(float((hybrid[:, ~destination] - x[:, ~destination]).abs().max()))
        mixed = module.lambdas[0] * hybrid + module.lambdas[1] * x0
        state["hybrid_current"] = F.rms_norm(mixed, (mixed.size(-1),))
        return None

    handles = [
        attention8.register_forward_pre_hook(attention8_pre),
        attention8.c_proj.register_forward_pre_hook(projection_pre),
        model.transformer.h[9].register_forward_pre_hook(block9_pre),
    ]

    def attention9_out(module, args, result):
        if state["corner"] == 0:
            return result
        current, first = args
        _, destination = masks(state["index"], current.device)
        routing, _ = source_factors(graph, current, current, current, first)
        original, _ = value_parts(graph, current, first)
        changed, _ = value_parts(graph, state["hybrid_current"], first)
        delta = select_sources(
            routing * (changed - original), destination[None]
        ) @ graph.p["output"].double().T
        return result[0] + delta.to(result[0].dtype), result[1]

    handles.append(model.transformer.h[9].attn.register_forward_hook(attention9_out))
    values = torch.zeros(8, len(rows), len(READOUTS), dtype=torch.float64)
    count = 0

    def forward(index, corner):
        nonlocal count
        row = rows[index]
        ids = torch.tensor([row["ids"]], device="cuda")
        x = F.rms_norm(model.transformer.wte(ids), (1152,))
        x0, first = x, None
        for block in model.transformer.h:
            x, first = block(x, first, x0)
        scores = (30 * torch.tanh(model.lm_head(F.rms_norm(x[:, -1], (1152,))) / 30))[0]
        pairs = [(row["uk_id"], row["us_id"])] + [pair for _, pair in READOUTS[1:]]
        for readout, (left, right) in enumerate(pairs):
            values[corner, index, readout] = (scores[left] - scores[right]).cpu()
        count += 1

    try:
        for index in range(len(rows)):
            forward(index, 0)
        assert len(attention_inputs) == len(preprojection) == len(rows)
        for corner in range(1, 8):
            state["corner"] = corner
            for index in range(len(rows)):
                state["index"] = index
                forward(index, corner)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "values": values,
        "source_replays": source_replays,
        "factorial_errors": factorial_errors,
        "value_errors": value_errors,
        "full_channel_errors": full_channel_errors,
        "outside": outside,
        "write_norms": write_norms,
        "body_forwards": count,
    }
