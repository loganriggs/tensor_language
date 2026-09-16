#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_instrument_decoder_and_native_capability pred_b_coordinate_removal_live pred_c_equal_distance_specificity_and_preservation
"""Causally remove the frozen number coordinate from fresh subject embeddings."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

import circuit_fast_screen_candidate_subject_number_embedding_removal_fresh as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json"
DECODER_FRESH = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_embedding_decoder_fresh_v1_result.json"
PREREG = POLY / "SUBJECT_NUMBER_EMBEDDING_COORDINATE_REMOVAL_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_EMBEDDING_COORDINATE_REMOVAL_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_EMBEDDING_COORDINATE_REMOVAL_V1_RESULT.json"
NULLS = 32
SEED = 20260917
BARS = {
    "maximum_control_fraction": .5,
    "maximum_manual_replay_error": 1e-5,
    "maximum_random_geometry_error": 1e-5,
    "maximum_target_decoder_residual": 1e-5,
    "maximum_target_geometry_error": 1e-5,
    "minimum_cell_positive_fraction": .75,
    "minimum_native_accuracy": .75,
    "minimum_random_median_ratio": 2.,
    "minimum_target_damage_fraction": .05,
    "minimum_target_damage_rms": .02,
}
PRICE = {"forwards": 35, "sequences": 2240, "fits": 0,
         "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {
    "pred_a_instrument_decoder_and_native_capability": None,
    "pred_b_coordinate_removal_live": None,
    "pred_c_equal_distance_specificity_and_preservation": None,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"authority": Path(authority.__file__), "decoder": DECODER,
             "decoder_fresh_result": DECODER_FRESH, "preregistration": PREREG}
    if binding["files"] != {name: sha(path) for name, path in paths.items()} \
            or binding["authority_sha256"] != authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"] != BARS or binding["nulls"] != NULLS \
            or binding["null_seed"] != SEED or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, fresh = (json.loads(path.read_text()) for path in (DECODER, DECODER_FRESH))
    if decoder["terminal"] != "embedding_number_decoder_frozen" \
            or fresh["terminal"] != "embedding_number_decoder_fresh_held":
        raise ValueError("decoder parent changed")
    rows = authority.build_rows()
    if authority.canonical(rows) != binding["authority_sha256"]:
        raise ValueError("authority changed")
    return binding, decoder, fresh, rows


def plan():
    binding, decoder, _, rows = load_bound()
    return {
        "schema": "subject_number_embedding_coordinate_removal_v1_plan",
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "rows": len(rows), "sequence_length": len(rows[0]["token_ids"]),
        "fresh_pairs": len(authority.PAIRS), "random_equal_distance_axes": NULLS,
        "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"],
        "authority_sha256": authority.canonical(rows),
        "bars": BARS, "price": PRICE, "binding_sha256": sha(BINDING),
    }


def cell_reports(values, rows):
    groups = {"overall": list(range(len(rows)))}
    for field, levels in (("number", ("singular", "plural")),
                          ("template_id", ("near", "behind")),
                          ("stratum", ("irregular", "regular"))):
        for level in levels:
            groups[f"{field}|{level}"] = [i for i, row in enumerate(rows)
                                               if row[field] == level]
    return {name: {"count": len(ids),
                   "positive_fraction": float(np.mean(values[ids] > 0)),
                   "rms": float(np.sqrt(np.mean(values[ids] ** 2)))}
            for name, ids in groups.items()}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(600)
    binding, decoder, fresh, rows = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies()
    torch.set_num_threads(2)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    started = time.perf_counter()
    device = next(model.parameters()).device
    tokens = torch.tensor([row["token_ids"] for row in rows],
                          dtype=torch.long, device=device)
    positions = torch.tensor([row["subject_position"] for row in rows],
                             dtype=torch.long, device=device)
    batch = torch.arange(len(rows), device=device)
    selected_ids = torch.tensor([
        [row["native_answer_id"], row["opposite_answer_id"],
         row["control_token_ids"]["can"], row["control_token_ids"]["will"]]
        for row in rows], dtype=torch.long, device=device)
    counts = {"forwards": 0, "sequences": 0, "fits": 0,
              "backwards": 0, "parameter_updates": 0}

    def select(logits):
        return logits[batch[:, None], positions[:, None], selected_ids].detach().float().cpu().numpy()

    def facade_forward():
        counts["forwards"] += 1; counts["sequences"] += len(rows)
        with torch.no_grad():
            logits = facade.forward_with_dispatch(
                model, tokens,
                lambda event: event.block.attn(event.state, event.first_value),
                lambda event: event.block.mlp(event.state),
                require_production=False)
        return select(logits)

    def forward_from_input(initial):
        counts["forwards"] += 1; counts["sequences"] += len(rows)
        with torch.no_grad():
            x = initial; x0 = initial; first_value = None
            for block in model.transformer.h:
                x = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, first_value = block.attn(F.rms_norm(x, (x.shape[-1],)),
                                                     first_value)
                x = x + attention
                x = x + block.mlp(F.rms_norm(x, (x.shape[-1],)))
            logits = model.lm_head(F.rms_norm(x, (x.shape[-1],)))
            logits = (30. * torch.tanh(logits / 30.)).float()
        return select(logits)

    with torch.no_grad():
        base_input = F.rms_norm(model.transformer.wte(tokens),
                                (model.config.n_embd,)).float()
    subject = base_input[batch, positions].double()
    decoder_axis = torch.tensor(decoder["frozen_decoder"]["axis"],
                                dtype=torch.float64, device=device)
    decoder_threshold = float(decoder["frozen_decoder"]["threshold"])
    axis_norm = decoder_axis.norm()
    unit_axis = decoder_axis / axis_norm
    projection = subject @ unit_axis
    target_projection = decoder_threshold / float(axis_norm)
    orthogonal = subject - projection[:, None] * unit_axis
    radius_squared = subject.square().sum(1)
    orthogonal_squared = orthogonal.square().sum(1)
    scale = torch.sqrt((radius_squared - target_projection ** 2) / orthogonal_squared)
    target_subject = target_projection * unit_axis + scale[:, None] * orthogonal
    target_input = base_input.clone()
    target_input[batch, positions] = target_subject.float()
    target_delta = target_subject - subject
    target_distance = target_delta.norm(dim=1)
    target_decoder_residual = (target_subject @ decoder_axis - decoder_threshold).abs()
    target_norm_error = ((target_subject.norm(dim=1) - subject.norm(dim=1)).abs()
                         / subject.norm(dim=1))
    decoder_scores = subject @ decoder_axis - decoder_threshold
    labels = torch.tensor([1. if row["number"] == "plural" else -1. for row in rows],
                          dtype=torch.float64, device=device)
    decoder_margins = decoder_scores * labels

    reference_values = facade_forward()
    base_values = forward_from_input(base_input)
    target_values = forward_from_input(target_input)
    manual_replay_error = float(np.max(np.abs(reference_values - base_values)))

    rng = np.random.default_rng(SEED)
    random_damage = []
    random_geometry_error = 0.
    for _ in range(NULLS):
        random_axis = torch.tensor(rng.standard_normal(model.config.n_embd),
                                   dtype=torch.float64, device=device)
        tangent_direction = random_axis[None, :] \
            - ((subject @ random_axis) / radius_squared)[:, None] * subject
        tangent_direction /= tangent_direction.norm(dim=1, keepdim=True)
        cosine = 1. - target_distance.square() / (2. * radius_squared)
        sine = torch.sqrt(torch.clamp(1. - cosine.square(), min=0.))
        random_subject = (cosine[:, None] * subject
                          + sine[:, None] * subject.norm(dim=1)[:, None]
                          * tangent_direction)
        distance_error = ((random_subject - subject).norm(dim=1) - target_distance).abs() \
            / torch.clamp(target_distance, min=1e-30)
        norm_error = (random_subject.norm(dim=1) - subject.norm(dim=1)).abs() \
            / subject.norm(dim=1)
        random_geometry_error = max(random_geometry_error,
                                    float(distance_error.max()), float(norm_error.max()))
        random_input = base_input.clone()
        random_input[batch, positions] = random_subject.float()
        values = forward_from_input(random_input)
        base_margin = base_values[:, 0] - base_values[:, 1]
        random_margin = values[:, 0] - values[:, 1]
        random_damage.append(base_margin - random_margin)

    base_margin = base_values[:, 0] - base_values[:, 1]
    target_margin = target_values[:, 0] - target_values[:, 1]
    target_damage = base_margin - target_margin
    base_control = base_values[:, 2] - base_values[:, 3]
    target_control = target_values[:, 2] - target_values[:, 3]
    control_change = target_control - base_control
    native_correct = base_margin > 0
    capability = {}
    for field, levels in (("number", ("singular", "plural")),
                          ("template_id", ("near", "behind")),
                          ("stratum", ("irregular", "regular"))):
        for level in levels:
            ids = [i for i, row in enumerate(rows) if row[field] == level]
            capability[f"{field}|{level}"] = {
                "count": len(ids), "accuracy": float(np.mean(native_correct[ids]))}

    target_rms = float(np.sqrt(np.mean(target_damage ** 2)))
    native_rms = float(np.sqrt(np.mean(base_margin ** 2)))
    control_rms = float(np.sqrt(np.mean(control_change ** 2)))
    random_rms = np.asarray([np.sqrt(np.mean(values ** 2)) for values in random_damage])
    random_median = float(np.median(random_rms))
    target_cells = cell_reports(target_damage, rows)
    decoder_pass = bool(torch.all(decoder_margins > 0))
    checkpoint_match = (checkpoint.weights_sha256
                        == decoder["checkpoint_weights_sha256"]
                        == fresh["checkpoint_weights_sha256"])
    instrument = bool(
        checkpoint_match and decoder_pass
        and manual_replay_error <= BARS["maximum_manual_replay_error"]
        and float(target_decoder_residual.max()) <= BARS["maximum_target_decoder_residual"]
        and float(target_norm_error.max()) <= BARS["maximum_target_geometry_error"]
        and random_geometry_error <= BARS["maximum_random_geometry_error"]
        and all(value["accuracy"] >= BARS["minimum_native_accuracy"]
                for value in capability.values())
        and counts == PRICE)
    pred_b = bool(
        instrument and target_rms >= BARS["minimum_target_damage_rms"]
        and target_rms / max(native_rms, 1e-30) >= BARS["minimum_target_damage_fraction"]
        and all(value["positive_fraction"] >= BARS["minimum_cell_positive_fraction"]
                for value in target_cells.values()))
    pred_c = bool(
        pred_b and target_rms > float(random_rms.max())
        and target_rms / max(random_median, 1e-30) >= BARS["minimum_random_median_ratio"]
        and control_rms / max(target_rms, 1e-30) <= BARS["maximum_control_fraction"])
    predictions = dict(zip(PREDICTION_REGISTRY, (instrument, pred_b, pred_c)))
    terminal = ("embedding_number_coordinate_selective_removal" if pred_c else
                "invalid" if not instrument else "embedding_number_coordinate_removal_null")
    result = {
        "schema": "subject_number_embedding_coordinate_removal_v1_result",
        "terminal": terminal, "predictions": predictions,
        "instrument": {
            "checkpoint_match": checkpoint_match,
            "decoder_accuracy": float(torch.mean((decoder_margins > 0).double())),
            "decoder_minimum_signed_margin": float(decoder_margins.min()),
            "manual_replay_max_logit_error": manual_replay_error,
            "target_decoder_max_absolute_residual": float(target_decoder_residual.max()),
            "target_max_relative_norm_error": float(target_norm_error.max()),
            "random_max_relative_geometry_error": random_geometry_error,
            "native_capability": capability, "counts": counts,
        },
        "target_removal": {
            "damage_rms": target_rms,
            "native_margin_rms": native_rms,
            "damage_to_native_margin_rms": target_rms / max(native_rms, 1e-30),
            "cells": target_cells,
            "control_change_rms": control_rms,
            "control_to_target_rms": control_rms / max(target_rms, 1e-30),
            "minimum_edit_distance": float(target_distance.min()),
            "maximum_edit_distance": float(target_distance.max()),
        },
        "equal_distance_random_controls": {
            "count": NULLS, "seed": SEED,
            "damage_rms": random_rms.tolist(),
            "median_damage_rms": random_median,
            "maximum_damage_rms": float(random_rms.max()),
            "target_to_median_ratio": target_rms / max(random_median, 1e-30),
            "target_exceeds_count": int(np.sum(target_rms > random_rms)),
        },
        "records": [{
            "row_id": row["row_id"], "stratum": row["stratum"],
            "template": row["template_id"], "number": row["number"],
            "decoder_score": float(decoder_scores[i]),
            "native_margin": float(base_margin[i]),
            "target_damage": float(target_damage[i]),
            "target_control_change": float(control_change[i]),
            "target_edit_distance": float(target_distance[i]),
        } for i, row in enumerate(rows)],
        "bars": BARS, "price": PRICE,
        "authority_sha256": authority.canonical(rows),
        "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_seconds": time.perf_counter() - started,
        "scope": ("Prospective same-norm removal of the frozen embedding-number coordinate "
                  "on disjoint vocabulary/text, compared with 32 same-position equal-distance "
                  "random tangent edits; no label enters the target intervention."),
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in
                      ("terminal", "predictions", "instrument", "target_removal",
                       "equal_distance_random_controls")}, indent=2, sort_keys=True))
    assert instrument


if __name__ == "__main__":
    main()
