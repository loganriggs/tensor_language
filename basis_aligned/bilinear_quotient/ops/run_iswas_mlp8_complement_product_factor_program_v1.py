#!/usr/bin/env python3
"""Exact bilinear factor program for the MLP8 complement writer."""

# BQGATE: EXPERIMENT pred_a_authority_factor_projection_full_replay_finiteness_and_price pred_b_full_complement_factor_program_is_material pred_c_left_right_two_term_program_is_sufficient pred_d_bilinear_interaction_is_secondary pred_e_zero_fit_literal_bilinear_weights
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
import numpy as np

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_complement_product_factor_program_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_postcue_to_attn9_h1h4_value_weight_compiler_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_postcue_to_attn9_h1h4_value_weight_compiler_v1.py"
CONFIRMATION = ROOT / "circuits/followups/iswas_mlp8_complement_attn9_h1h4_fresh_v10_confirmation_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10.py"
WEIGHT_INSTRUMENT = ROOT / "ops/run_iswas_shared_q8_mlp8_postcue_weight_modes_v1.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_complement_product_factor_program_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_complement_product_factor_program_v1"
EXPECTED = {"prior": "95c86a4f31638575fec1da05ed6281b37bf3e54169d4a8093d28f65b5d812347",
    "parent": "393d730e958ace78ef5a4026ed2a35acebb4bf801f6c37daa8cca4746547aadc",
    "parent_runner": "7a65c8ffd57eb93d82b0bf5104745db94e2703666616bbbf6c5f29dc64dc42ff",
    "confirmation": "eb4470b3ed1495a5ad11c0a1e685d0c6ca4e9942532a872f165bc16f07eb69f5",
    "capability": "77e3c5b6e47dc9416133643f422c130427f90fed0746d26f211f54b681718b3d",
    "builder": "13e7954cde01b6b7d826915fc2ae02d4b9e16975150cf73aa5e0a1f906c1b757",
    "weight_instrument": "d82035e45f8818a7de73024500592f1677f887e8be5d8a2e617c28afc6c1238c",
    "iswas": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1"}
FACTORS = ("left_change", "right_change", "bilinear_interaction")
MAX_FORWARDS, MAX_EVALUATIONS = 10, 270


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
def subsets(): return tuple(subset for width in range(4) for subset in itertools.combinations(FACTORS, width))
def arm_name(subset): return "empty" if not subset else "+".join(subset)
def cosine(x, y):
    denominator = float(x.norm()*y.norm())
    return float((x*y).sum())/denominator if denominator else 0.0


def main():
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER,
        "confirmation": CONFIRMATION, "capability": CAPABILITY, "builder": BUILDER,
        "weight_instrument": WEIGHT_INSTRUMENT, "iswas": weight.ISWAS,
        "subspace": weight.SUBSPACE, "family_runner": weight.FAMILY_RUNNER,
        "overlap_runner": weight.OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED: raise RuntimeError("product-factor authority changed")
    prior, parent, confirmation, capability, iswas, subspace = [json.loads(path.read_text())
        for path in (PRIOR, PARENT, CONFIRMATION, CAPABILITY, weight.ISWAS, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows() if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    positions = [weight.postcue_positions(row) for row in rows]
    if prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen" or confirmation.get("terminal") != "screen" or len(rows) != 27:
        raise RuntimeError("parent terminal or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "arms": [arm_name(subset) for subset in subsets()], "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS, "fit_updates": 0,
        "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend, torch = producer.Bilin18TorchBackend.load("cuda"), None; torch = backend.torch
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float()) for layer in range(12, 18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, family[8], gain); s = torch.linalg.qr(modes, mode="reduced").Q
    axis_values = np.asarray(iswas["basis"]["values_column_major"], dtype=np.float32)
    if hashlib.sha256(axis_values.tobytes()).hexdigest() != iswas["basis"]["sha256"]: raise RuntimeError("is/was axis changed")
    axis = torch.as_tensor(axis_values, device=backend.device).reshape(1152, 1); axis /= torch.linalg.vector_norm(axis)
    shared_axis = s@(s.T@axis)
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_output, base_capture = weight.capture_mlp8(backend, base_batch)
    donor_output, donor_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(s.T@down, full_matrices=False)
    dl = donor_capture["left"].float()-base_capture["left"].float()
    dr = donor_capture["right"].float()-base_capture["right"].float()
    raw_factors = {"left_change": dl*base_capture["right"].float(),
        "right_change": base_capture["left"].float()*dr, "bilinear_interaction": dl*dr}
    projected = {name: value-weight.project(value, vh, vh.shape[0]) for name, value in raw_factors.items()}
    full_delta = donor_capture["hidden"].float()-base_capture["hidden"].float()
    complement = full_delta-weight.project(full_delta, vh, vh.shape[0])
    factor_error = float((sum(raw_factors.values())-full_delta).abs().max())
    projection_error = float((sum(projected.values())-complement).abs().max())
    outputs = {}
    for subset in subsets():
        delta = sum((projected[name] for name in subset), torch.zeros_like(complement))
        outputs[arm_name(subset)] = weight.run_hidden_patch(backend, base_batch, base_capture["hidden"], delta, positions)
    forwards, evaluations = 10, 10*len(rows)
    state = lambda output: torch.stack([torch.as_tensor(output.captured[(row["row_id"], "resid:18")]) for row in rows]).to(backend.device).float()
    base18, donor18 = state(base_output), state(donor_output)
    states = {name: state(output) for name, output in outputs.items()}
    target = ((donor18-base18)@axis)@shared_axis.T
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda value: das.head_logits(backend, value)[index, answers]-das.head_logits(backend, value)[index, foils]
    base_margin, target_effect = margin(base18), margin(base18+target)-margin(base18)
    full_name, two_name, interaction_name = arm_name(FACTORS), arm_name(FACTORS[:2]), "bilinear_interaction"
    full_effect, full_coord = margin(states[full_name])-base_margin, (states[full_name]-base18)@s
    target_coord = target@s
    metrics = {panel: {} for panel in ("A1", "A2")}
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device) for panel in ("A1", "A2")}
    for panel, mask in masks.items():
        for name, value in states.items():
            effect, coord = (margin(value)-base_margin)[mask], ((value-base18)@s)[mask]
            metrics[panel][name] = {"absolute_behavior_fraction_of_full": float(effect.abs().mean()/full_effect[mask].abs().mean()),
                "signed_behavior_fraction_of_full": float(effect.mean()/full_effect[mask].mean()),
                "behavior_cosine_to_full": cosine(effect, full_effect[mask]),
                "q8_norm_fraction_of_full": float(coord.norm()/full_coord[mask].norm()),
                "q8_cosine_to_full": cosine(coord.reshape(-1), full_coord[mask].reshape(-1))}
        metrics[panel][full_name].update({"behavior_cosine_to_shared_target": cosine(full_effect[mask], target_effect[mask]),
            "q8_cosine_to_shared_target": cosine(full_coord[mask].reshape(-1), target_coord[mask].reshape(-1))})
    empty_error = float((states["empty"]-base18).abs().max())
    replay_error = max(abs(metrics[p][full_name][key]-confirmation["metrics"][p]["live_complement"][parent_key])
        for p in ("A1", "A2") for key, parent_key in (("behavior_cosine_to_shared_target", "behavior_cosine_to_target"),
        ("q8_cosine_to_shared_target", "q8_cosine_to_target")))
    finite = all(math.isfinite(value) for p in metrics.values() for row in p.values() for value in row.values())
    pred_a = bool(orientation_error <= 1e-6 and factor_error <= .02 and projection_error <= .02
        and empty_error <= 1e-4 and replay_error <= .001 and finite and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = all(metrics[p][full_name]["behavior_cosine_to_shared_target"] >= .95
        and metrics[p][full_name]["q8_cosine_to_shared_target"] >= .80 for p in ("A1", "A2"))
    pred_c = all(metrics[p][two_name][key] >= threshold for p in ("A1", "A2")
        for key, threshold in (("absolute_behavior_fraction_of_full", .90), ("q8_norm_fraction_of_full", .90),
            ("behavior_cosine_to_full", .95), ("q8_cosine_to_full", .95)))
    pred_d = all(metrics[p][interaction_name][key] <= .15 for p in ("A1", "A2")
        for key in ("absolute_behavior_fraction_of_full", "q8_norm_fraction_of_full"))
    pred_e = True
    predictions = {"pred_a_authority_factor_projection_full_replay_finiteness_and_price": pred_a,
        "pred_b_full_complement_factor_program_is_material": pred_b,
        "pred_c_left_right_two_term_program_is_sufficient": pred_c,
        "pred_d_bilinear_interaction_is_secondary": pred_d,
        "pred_e_zero_fit_literal_bilinear_weights": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "iswas_mlp8_complement_product_factor_program_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter()-started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "hidden_factor_reconstruction_max_abs": factor_error,
            "complement_factor_projection_closure_max_abs": projection_error,
            "empty_base_resid18_max_abs": empty_error, "parent_target_metric_replay_max_abs": replay_error, "rows": len(rows)},
        "metrics": metrics, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "metrics",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
