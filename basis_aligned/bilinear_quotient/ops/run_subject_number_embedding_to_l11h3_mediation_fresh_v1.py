#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_fresh_instrument_and_coordinate_removal pred_b_small_full_head_mediation_transfers pred_c_rank1_retains_full_head_path pred_d_rank1_beats_matched_random
"""Fresh confirmation of the small embedding-number to L11H3 mediation path."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

import circuit_fast_screen_candidate_subject_number_embedding_mediation_fresh as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json"
RANK1 = POLY / "SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
DISCOVERY = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_MEDIATION_DISCOVERY_V1_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_MEDIATION_FRESH_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_MEDIATION_FRESH_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_MEDIATION_FRESH_V1_RESULT.json"
LAYER, HEAD, HEAD_WIDTH = 11, 3, 128
NULLS, SEED = 16, 20260919
BARS = {
    "maximum_axis_damage_recovery": .14,
    "maximum_axis_to_full_relative_l2": .2,
    "maximum_full_damage_recovery": .15,
    "maximum_rank1_norm_ratio": 1.25,
    "maximum_state_error": 1e-5,
    "minimum_axis_damage_recovery": .04,
    "minimum_axis_full_cosine": .97,
    "minimum_cell_positive_fraction": .75,
    "minimum_damage_rms": .5,
    "minimum_full_damage_cosine": .75,
    "minimum_full_damage_recovery": .05,
    "minimum_full_positive_fraction": .7,
    "minimum_native_accuracy": .75,
    "minimum_random_median_ratio": 10.,
    "minimum_rank1_norm_ratio": .75,
}
PRICE = {"forwards": 20, "sequences": 1280, "fits": 0,
         "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {
    "pred_a_fresh_instrument_and_coordinate_removal": None,
    "pred_b_small_full_head_mediation_transfers": None,
    "pred_c_rank1_retains_full_head_path": None,
    "pred_d_rank1_beats_matched_random": None,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"authority": Path(authority.__file__), "decoder": DECODER,
             "rank1": RANK1, "discovery": DISCOVERY, "preregistration": PREREG}
    if binding["files"] != {name: sha(path) for name, path in paths.items()} \
            or binding["authority_sha256"] != authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"] != BARS or binding["layer"] != LAYER \
            or binding["head"] != HEAD or binding["nulls"] != NULLS \
            or binding["null_seed"] != SEED or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, rank, discovery = (json.loads(path.read_text())
                                for path in (DECODER, RANK1, DISCOVERY))
    if decoder["terminal"] != "embedding_number_decoder_frozen" \
            or rank["terminal"] != "rank1_frozen_weights_only" \
            or discovery["terminal"] != "embedding_to_l11h3_mediation_null":
        raise ValueError("parent status changed")
    rows = authority.build_rows()
    if authority.canonical(rows) != binding["authority_sha256"]:
        raise ValueError("authority changed")
    return binding, decoder, rank, discovery, rows


def plan():
    _, decoder, rank, discovery, rows = load_bound()
    return {
        "schema": "subject_number_embedding_to_l11h3_mediation_fresh_v1_plan",
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "fresh_rows": len(rows), "fresh_pairs": len(authority.PAIRS),
        "layer": LAYER, "head": HEAD, "random_axis_rescues": NULLS,
        "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"],
        "rank1": rank["rank"], "discovery_terminal": discovery["terminal"],
        "authority_sha256": authority.canonical(rows),
        "bars": BARS, "price": PRICE, "binding_sha256": sha(BINDING),
    }


def metrics(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    pn, an = np.linalg.norm(predicted), np.linalg.norm(actual)
    return {
        "cosine": float(predicted @ actual / max(pn * an, 1e-30)),
        "relative_l2_error": float(np.linalg.norm(predicted - actual) / max(an, 1e-30)),
        "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))),
        "positive_fraction": float(np.mean(predicted > 0)),
        "predicted_to_actual_norm_ratio": float(pn / max(an, 1e-30)),
        "least_squares_recovery": float(predicted @ actual / max(actual @ actual, 1e-30)),
        "effect_rms": float(np.sqrt(np.mean(predicted ** 2))),
    }


def grouped(values, rows):
    groups = {"overall": list(range(len(rows)))}
    for field, levels in (("number", ("singular", "plural")),
                          ("template_id", ("under", "above")),
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
    binding, decoder, rank, discovery, rows = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies()
    torch.set_num_threads(2)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    started = time.perf_counter(); device = next(model.parameters()).device
    tokens = torch.tensor([row["token_ids"] for row in rows],
                          dtype=torch.long, device=device)
    positions = torch.tensor([row["subject_position"] for row in rows],
                             dtype=torch.long, device=device)
    batch = torch.arange(len(rows), device=device)
    selected_ids = torch.tensor([[row["native_answer_id"], row["opposite_answer_id"]]
                                 for row in rows], dtype=torch.long, device=device)
    with torch.no_grad():
        base_input = F.rms_norm(model.transformer.wte(tokens),
                                (model.config.n_embd,)).float()
    subject = base_input[batch, positions].double()
    decoder_axis = torch.tensor(decoder["frozen_decoder"]["axis"],
                                dtype=torch.float64, device=device)
    threshold = float(decoder["frozen_decoder"]["threshold"])
    unit_decoder = decoder_axis / decoder_axis.norm()
    projection = subject @ unit_decoder
    target_projection = threshold / float(decoder_axis.norm())
    orthogonal = subject - projection[:, None] * unit_decoder
    radius_squared = subject.square().sum(1)
    scale = torch.sqrt((radius_squared - target_projection ** 2)
                       / orthogonal.square().sum(1))
    removed_subject = target_projection * unit_decoder + scale[:, None] * orthogonal
    removed_input = base_input.clone()
    removed_input[batch, positions] = removed_subject.float()
    decoder_scores = subject @ decoder_axis - threshold
    labels = torch.tensor([1. if row["number"] == "plural" else -1. for row in rows],
                          dtype=torch.float64, device=device)
    decoder_margins = decoder_scores * labels
    target_decoder_error = float((removed_subject @ decoder_axis - threshold).abs().max())
    target_norm_error = float(((removed_subject.norm(dim=1) - subject.norm(dim=1)).abs()
                               / subject.norm(dim=1)).max())
    writer_axis = torch.tensor(rank["axis"], dtype=torch.float32, device=device)
    writer_axis /= writer_axis.norm()
    random_generator = np.random.default_rng(SEED)
    random_axes = []
    for _ in range(NULLS):
        vector = torch.tensor(random_generator.standard_normal(model.config.n_embd),
                              dtype=torch.float32, device=device)
        vector -= (vector @ writer_axis) * writer_axis
        random_axes.append(vector / vector.norm())

    counts = {"forwards": 0, "sequences": 0, "fits": 0,
              "backwards": 0, "parameter_updates": 0}
    base_head = None; removed_head = None; capture_error = 0.
    rescue_norms = {}; head_slice = slice(HEAD * HEAD_WIDTH, (HEAD + 1) * HEAD_WIDTH)

    def run(initial, rescue=None):
        nonlocal base_head, removed_head, capture_error
        counts["forwards"] += 1; counts["sequences"] += len(rows)
        captured = {}
        with torch.no_grad():
            x = initial; x0 = initial; first_value = None
            for layer, block in enumerate(model.transformer.h):
                x = block.lambdas[0] * x + block.lambdas[1] * x0
                handle = None
                if layer == LAYER:
                    def capture(_module, arguments):
                        captured["head"] = arguments[0][batch, positions, head_slice].detach().clone()
                    handle = block.attn.c_proj.register_forward_pre_hook(capture)
                try:
                    attention, first_value = block.attn(F.rms_norm(x, (x.shape[-1],)),
                                                         first_value)
                finally:
                    if handle is not None: handle.remove()
                if layer == LAYER:
                    current = captured["head"]
                    if rescue is None and initial.data_ptr() == base_input.data_ptr():
                        base_head = current
                    else:
                        if base_head is None: raise RuntimeError("base head missing")
                        if removed_head is None: removed_head = current
                        else: capture_error = max(capture_error, float((current - removed_head).abs().max()))
                        full_delta = F.linear(base_head - current,
                                              block.attn.c_proj.weight[:, head_slice])
                        axis_delta = (full_delta @ writer_axis)[:, None] * writer_axis
                        if rescue == "full": delta = full_delta
                        elif rescue == "axis": delta = axis_delta
                        elif isinstance(rescue, int):
                            delta = axis_delta.norm(dim=1)[:, None] * random_axes[rescue]
                        else: delta = torch.zeros_like(full_delta)
                        rescue_norms.setdefault(str(rescue), []).append(float(delta.norm(dim=1).mean()))
                        attention = attention.clone(); attention[batch, positions] += delta.to(attention.dtype)
                x = x + attention
                x = x + block.mlp(F.rms_norm(x, (x.shape[-1],)))
            logits = model.lm_head(F.rms_norm(x, (x.shape[-1],)))
            logits = (30. * torch.tanh(logits / 30.)).float()
            return logits[batch[:, None], positions[:, None], selected_ids].detach().cpu().numpy()

    base_values = run(base_input); removed_values = run(removed_input)
    full_values = run(removed_input, "full"); axis_values = run(removed_input, "axis")
    random_values = [run(removed_input, index) for index in range(NULLS)]
    base_margin = base_values[:, 0] - base_values[:, 1]
    removed_margin = removed_values[:, 0] - removed_values[:, 1]
    damage = base_margin - removed_margin
    full_rescue = (full_values[:, 0] - full_values[:, 1]) - removed_margin
    axis_rescue = (axis_values[:, 0] - axis_values[:, 1]) - removed_margin
    random_rescues = np.asarray([(values[:, 0] - values[:, 1]) - removed_margin
                                 for values in random_values])
    full_report = metrics(full_rescue, damage)
    axis_damage_report = metrics(axis_rescue, damage)
    axis_full_report = metrics(axis_rescue, full_rescue)
    random_full_reports = [metrics(values, full_rescue) for values in random_rescues]
    random_rms = np.asarray([report["effect_rms"] for report in random_full_reports])
    random_errors = np.asarray([report["relative_l2_error"] for report in random_full_reports])
    damage_cells = grouped(damage, rows)
    native_correct = base_margin > 0
    capability = {}
    for field, levels in (("number", ("singular", "plural")),
                          ("template_id", ("under", "above")),
                          ("stratum", ("irregular", "regular"))):
        for level in levels:
            ids = [i for i, row in enumerate(rows) if row[field] == level]
            capability[f"{field}|{level}"] = {"count": len(ids),
                "accuracy": float(np.mean(native_correct[ids]))}
    finite = bool(np.isfinite(np.asarray([*damage, *full_rescue, *axis_rescue,
                                          *random_rescues.ravel()])).all())
    instrument = bool(
        finite and bool(torch.all(decoder_margins > 0))
        and target_decoder_error <= BARS["maximum_state_error"]
        and target_norm_error <= BARS["maximum_state_error"]
        and capture_error <= BARS["maximum_state_error"]
        and all(cell["accuracy"] >= BARS["minimum_native_accuracy"]
                for cell in capability.values())
        and counts == PRICE and checkpoint.weights_sha256 == decoder["checkpoint_weights_sha256"])
    pred_a = bool(
        instrument and damage_cells["overall"]["rms"] >= BARS["minimum_damage_rms"]
        and all(cell["positive_fraction"] >= BARS["minimum_cell_positive_fraction"]
                for cell in damage_cells.values()))
    pred_b = bool(
        pred_a and full_report["cosine"] >= BARS["minimum_full_damage_cosine"]
        and full_report["positive_fraction"] >= BARS["minimum_full_positive_fraction"]
        and BARS["minimum_full_damage_recovery"] <= full_report["least_squares_recovery"]
        <= BARS["maximum_full_damage_recovery"])
    pred_c = bool(
        pred_b and axis_full_report["cosine"] >= BARS["minimum_axis_full_cosine"]
        and axis_full_report["relative_l2_error"] <= BARS["maximum_axis_to_full_relative_l2"]
        and BARS["minimum_rank1_norm_ratio"] <= axis_full_report["predicted_to_actual_norm_ratio"]
        <= BARS["maximum_rank1_norm_ratio"]
        and BARS["minimum_axis_damage_recovery"] <= axis_damage_report["least_squares_recovery"]
        <= BARS["maximum_axis_damage_recovery"])
    pred_d = bool(
        pred_c and axis_full_report["relative_l2_error"] < float(random_errors.min())
        and axis_full_report["effect_rms"]
        >= BARS["minimum_random_median_ratio"] * float(np.median(random_rms)))
    predictions = dict(zip(PREDICTION_REGISTRY, (pred_a, pred_b, pred_c, pred_d)))
    terminal = ("embedding_to_l11h3_small_mediation_fresh_held" if pred_d else
                "invalid" if not instrument else "embedding_to_l11h3_small_mediation_fresh_null")
    result = {
        "schema": "subject_number_embedding_to_l11h3_mediation_fresh_v1_result",
        "terminal": terminal, "predictions": predictions,
        "instrument": {"finite": finite, "decoder_accuracy": float(torch.mean((decoder_margins > 0).double())),
                       "decoder_minimum_signed_margin": float(decoder_margins.min()),
                       "target_decoder_max_absolute_residual": target_decoder_error,
                       "target_max_relative_norm_error": target_norm_error,
                       "repeated_removed_head_capture_max_absolute_error": capture_error,
                       "native_capability": capability, "counts": counts},
        "embedding_coordinate_removal": {"cells": damage_cells},
        "full_head_rescue_to_damage": full_report,
        "rank1_rescue_to_damage": axis_damage_report,
        "rank1_rescue_to_full_head": axis_full_report,
        "matched_random_rescues_to_full_head": {
            "count": NULLS, "seed": SEED,
            "median_effect_rms": float(np.median(random_rms)),
            "minimum_relative_l2_error": float(random_errors.min()),
            "reports": random_full_reports},
        "mean_rescue_write_norms": {key: float(np.mean(values)) for key, values in rescue_norms.items()},
        "records": [{"row_id": row["row_id"], "number": row["number"],
                     "stratum": row["stratum"], "template": row["template_id"],
                     "decoder_score": float(decoder_scores[i]), "damage": float(damage[i]),
                     "full_head_rescue": float(full_rescue[i]),
                     "rank1_rescue": float(axis_rescue[i])} for i, row in enumerate(rows)],
        "bars": BARS, "price": PRICE, "authority_sha256": authority.canonical(rows),
        "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_seconds": time.perf_counter() - started,
        "scope": ("Prospective fourth-vocabulary confirmation of a small donor-restored "
                  "L11H3 mediation path from same-norm embedding-number coordinate removal; "
                  "full head and native base remain explicit donor ports."),
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in
                      ("terminal", "predictions", "instrument",
                       "embedding_coordinate_removal", "full_head_rescue_to_damage",
                       "rank1_rescue_to_damage", "rank1_rescue_to_full_head",
                       "matched_random_rescues_to_full_head")}, indent=2, sort_keys=True))
    assert instrument


if __name__ == "__main__":
    main()
