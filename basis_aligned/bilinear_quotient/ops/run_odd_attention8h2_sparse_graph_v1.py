#!/usr/bin/env python3
# BQGATE:1152bodyforwards;144prefixes;300seconds;no fitting.
"""pred_a instrument; pred_b prospective OOD; pred_c composition.
pred_d reusable sparse graph; pred_e unrelated-token selectivity.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]
import numpy as np
import torch
import torch.nn.functional as F

import sparse_interaction_graph as sparse
from attention8h2_city_key_value_v1 import factorial_channels
from attention8h2_city_value_source_v1 import value_arms
from odd_contextual_positions_v1 import contextual_masks
from odd_semantic_positions_v1 import semantic_masks
from odd_source_positions_v1 import select_sources
from odd_source_swap_interaction_v1 import source_factors
from odd_value_source_split_v1 import value_parts
from regional_cue_row_check_v1 import validate
from run_even_value_factorial_native_v1 import setup, cpu_control
from run_odd_attention8h2_city_key_value_v1 import head_factor_parts
from sparse_path_stability_atlas_v1 import digest


STEM = "ODD_ATTENTION8H2_SPARSE_GRAPH_V1"
PORTS = ("routing", "current_value", "inherited_value")
READOUTS = [
    ("target", None), ("cat_dog", (3797, 3290)),
    ("red_blue", (2266, 4171)), ("monday_tuesday", (3321, 3431)),
    ("apple_orange", (17180, 10912)),
]


def row_hash(row):
    copied = dict(row)
    claimed = copied.pop("row_sha256")
    payload = json.dumps(copied, sort_keys=True, separators=(",", ":")).encode()
    return claimed, hashlib.sha256(payload).hexdigest()


def mask_name(mask):
    return "+".join(PORTS[index] for index in range(3) if mask & (1 << index))


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(path) == value for path, value in binding.items())
    discovery = json.loads((P / "ODD_ATTENTION8H2_CHAIN_FRESH_V1_ROWS.json").read_text())["rows"]
    natural_doc = json.loads((P / (STEM + "_ROWS.json")).read_text())
    natural = natural_doc["rows"]
    assert len(discovery) == 48 and len(natural) == 96
    validate(discovery)
    assert all(row_hash(row)[0] == row_hash(row)[1] for row in natural)
    assert all(
        natural[index]["cue"] == "British"
        and natural[index + 1]["cue"] == "American"
        and natural[index]["context_id"] == natural[index + 1]["context_id"]
        and natural[index]["endpoint"] == natural[index + 1]["endpoint"]
        and sum(a != b for a, b in zip(natural[index]["ids"], natural[index + 1]["ids"])) == 1
        for index in range(0, len(natural), 2)
    )
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        control = json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())
        assert control["pred_a"] and control["rows"] == 96
        print("1152bodyforwards;144prefixes;8corners;3greedysteps;CPUcontrol", cpu_control())
        return

    output = P / (STEM + "_RESULT.json")
    artifact = P / (STEM + "_ARTIFACT.pt")
    assert not output.exists() and not artifact.exists()
    started = time.perf_counter()
    signal.alarm(300)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast

    model = load_model_fast().cuda().eval()
    graph, _, _, _ = setup("cuda")
    rows = discovery + natural
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
        index = state["index"]
        donor = index ^ 1
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
    assert count == 1152 and bool(torch.isfinite(values).all())

    effects = values - values[0:1]
    target_corners = {corner: effects[corner, :, 0].numpy() for corner in range(8)}
    dividends = sparse.full_dividends(target_corners, len(PORTS))
    target = target_corners[7]
    atoms = {mask: dividends[mask] for mask in range(1, 8)}
    discovery_indices = tuple(range(48))
    ood_indices = tuple(range(48, len(rows)))
    selected, selection_curve = sparse.greedy_select(
        target, atoms, discovery_indices, steps=3
    )
    reports = sparse.evaluate_frozen(
        target, atoms, selected,
        {"discovery": discovery_indices, "fresh_natural_ood": ood_indices},
    )
    prediction = sparse.compose(atoms, selected)
    closure_error = float(
        np.linalg.norm(sparse.reconstruct(dividends, 7) - target)
        / max(np.linalg.norm(target), 1e-30)
    )
    sign_reversals = int(np.sum(
        (prediction[list(ood_indices)] * target[list(ood_indices)] < 0)
        & (np.abs(target[list(ood_indices)]) >= 1e-5)
    ))
    composition_cells = {}
    for cue in ("British", "American"):
        indices = [48 + index for index, row in enumerate(natural) if row["cue"] == cue]
        composition_cells["cue_" + cue.lower()] = sparse.metrics(target, prediction, indices)
    for endpoint in range(6):
        indices = [48 + index for index, row in enumerate(natural) if row["endpoint"] == endpoint]
        composition_cells[f"endpoint_{endpoint}"] = sparse.metrics(target, prediction, indices)

    predicted_readouts = []
    for readout in range(len(READOUTS)):
        corners = {corner: effects[corner, :, readout].numpy() for corner in range(8)}
        terms = sparse.full_dividends(corners, len(PORTS))
        predicted_readouts.append(sparse.compose(terms, selected))
    predicted_readouts = np.stack(predicted_readouts, axis=-1)
    target_rms = float(np.sqrt(np.mean(predicted_readouts[list(ood_indices), 0] ** 2)))
    control_ratios = {
        READOUTS[readout][0]: float(
            np.sqrt(np.mean(predicted_readouts[list(ood_indices), readout] ** 2))
            / max(target_rms, 1e-30)
        )
        for readout in range(1, len(READOUTS))
    }
    required = sparse.required_corners(selected, len(PORTS))
    selected_rms = {
        str(mask): float(np.sqrt(np.mean(atoms[mask][list(ood_indices)] ** 2)))
        for mask in selected
    }
    instrument = (
        max(source_replays) <= 1e-5
        and max(factorial_errors) <= 1e-5
        and max(value_errors) <= 1e-5
        and max(full_channel_errors) <= 1e-5
        and closure_error <= 1e-5
        and max(outside) == 0
        and all(min(norms) >= 1e-8 for norms in write_norms.values())
        and count == 1152
        and bool(torch.isfinite(values).all())
    )
    pred_b = (
        reports["fresh_natural_ood"]["relative_l2"] <= .35
        and reports["fresh_natural_ood"]["cosine"] >= .9
        and sign_reversals == 0
    )
    pred_c = all(cell["relative_l2"] <= .35 for cell in composition_cells.values())
    pred_d = (
        reports["discovery"]["relative_l2"] <= .25
        and len(required) < 8
        and min(selected_rms.values()) >= 1e-8
    )
    pred_e = max(control_ratios.values()) <= .5
    torch.save({
        "values": values, "effects": effects, "selected_masks": selected,
        "ports": PORTS, "required_corners": required,
    }, artifact)
    result = {
        "terminal": "valid_odd_attention8h2_sparse_graph" if all(
            (instrument, pred_b, pred_c, pred_d, pred_e)
        ) else "valid_odd_attention8h2_sparse_graph_near_miss",
        "pred_a": instrument, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e,
        "ports": list(PORTS),
        "selected_masks": list(selected),
        "selected_terms": [mask_name(mask) for mask in selected],
        "required_corners": list(required),
        "selection_curve": list(selection_curve),
        "metrics": reports,
        "composition_cells": composition_cells,
        "material_ood_sign_reversals": sign_reversals,
        "selected_atom_ood_rms": selected_rms,
        "target_ood_rms": target_rms,
        "control_ratios": control_ratios,
        "max_head_source_replay_error": max(source_replays),
        "max_factorial_error": max(factorial_errors),
        "max_value_sum_error": max(value_errors),
        "max_full_corner_channel_error": max(full_channel_errors),
        "mobius_closure_error": closure_error,
        "max_outside_hybrid_delta": max(outside),
        "body_forwards": count,
        "fit_parameters": 0,
        "model_updates": 0,
        "seconds": time.perf_counter() - started,
        "artifact_sha256": digest(artifact),
        "source_shas": binding,
        "scope": (
            "Prospective coefficient-free three-atom selection on authored inherited-city rows and "
            "frozen evaluation on outcome-blind skip39000 FineWeb contexts. Conditional head8.2 to "
            "head9.8-O intervention with native suffix; not yet standalone-package or equal-norm-removal certification."
        ),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "source_shas"}, indent=2))
    signal.alarm(0)


if __name__ == "__main__":
    main()
