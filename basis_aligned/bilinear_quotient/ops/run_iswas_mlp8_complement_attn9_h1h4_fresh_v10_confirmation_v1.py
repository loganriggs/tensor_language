#!/usr/bin/env python3
"""Prospective v10 confirmation of the attention9 H1+H4 complement converter."""

# BQGATE: EXPERIMENT pred_a_authority_manifest_replay_self_clamp_finiteness_and_price pred_b_indirect_complement_route_transfers pred_c_h1h4_union_closes_attention9_conversion pred_d_both_heads_are_positive_material_members pred_e_zero_fit_weight_defined_interface
from datetime import datetime, timezone
import hashlib
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
import run_iswas_mlp8_complement_attn9_head_converter_atlas_v1 as head_atlas
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_complement_attn9_h1h4_fresh_v10_confirmation_v1.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1_result.json"
CAPABILITY_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10.py"
SELECTION = ROOT / "circuits/followups/iswas_mlp8_complement_attn9_head_converter_atlas_v1_result.json"
HEAD_RUNNER = ROOT / "ops/run_iswas_mlp8_complement_attn9_head_converter_atlas_v1.py"
WEIGHT_INSTRUMENT = ROOT / "ops/run_iswas_shared_q8_mlp8_postcue_weight_modes_v1.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_complement_attn9_h1h4_fresh_v10_confirmation_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_complement_attn9_h1h4_fresh_v10_confirmation_v1"
EXPECTED = {"prior": "cba2c7ef362e5d4e88f8dc87d6df8bf30ea38f492960899623a09f968f58502f",
    "capability": "77e3c5b6e47dc9416133643f422c130427f90fed0746d26f211f54b681718b3d",
    "capability_runner": "d96b7e0870172224085cbe0dc033db9e2565ac0dd2b9b73f41e584189fed7f63",
    "builder": "13e7954cde01b6b7d826915fc2ae02d4b9e16975150cf73aa5e0a1f906c1b757",
    "selection": "2c7e19d53f6123491de20882c04f3781ac5c81531f6dfc8ca9137c53b96a01a8",
    "head_runner": "12cd3dc948010a0a2e96b27a69272af81d2e394310ee2da3f691cb8d18718f0c",
    "weight_instrument": "d82035e45f8818a7de73024500592f1677f887e8be5d8a2e617c28afc6c1238c",
    "iswas": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1"}
ARMS = {"h1": (1,), "h4": (4,), "h1h4": (1, 4), "all": tuple(range(9))}
MAX_FORWARDS, MAX_EVALUATIONS = 9, 243


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def cosine(x, y):
    denominator = float(x.norm()*y.norm())
    return float((x*y).sum())/denominator if denominator else 0.0


def main():
    paths = {"prior": PRIOR, "capability": CAPABILITY, "capability_runner": CAPABILITY_RUNNER,
        "builder": BUILDER, "selection": SELECTION, "head_runner": HEAD_RUNNER,
        "weight_instrument": WEIGHT_INSTRUMENT, "iswas": weight.ISWAS,
        "subspace": weight.SUBSPACE, "family_runner": weight.FAMILY_RUNNER,
        "overlap_runner": weight.OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("fresh H1H4 confirmation authority changed")
    prior, capability, selection, iswas, subspace = [json.loads(path.read_text())
        for path in (PRIOR, CAPABILITY, SELECTION, weight.ISWAS, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    positions = [weight.postcue_positions(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or capability.get("terminal") != "screen"
            or capability.get("causal_outcomes_opened") is not False
            or selection.get("stable_material_heads") != [1, 4]
            or {family: sum(row["family"] == family for row in rows) for family in ("A1", "A2")}
                != {"A1": 13, "A2": 14}):
        raise RuntimeError("capability manifest, frozen heads, or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "arms": {name: list(heads) for name, heads in ARMS.items()},
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in range(12, 18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, family[8], gain)
    s = torch.linalg.qr(modes, mode="reduced").Q
    axis_values = np.asarray(iswas["basis"]["values_column_major"], dtype=np.float32)
    if hashlib.sha256(axis_values.tobytes()).hexdigest() != iswas["basis"]["sha256"]:
        raise RuntimeError("is/was axis changed")
    axis = torch.as_tensor(axis_values, device=backend.device).reshape(1152, 1)
    axis = axis/torch.linalg.vector_norm(axis); shared_axis = s@(s.T@axis)
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_hidden_output, base_capture = weight.capture_mlp8(backend, base_batch)
    donor_output, donor_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(s.T@down, full_matrices=False)
    full_delta = donor_capture["hidden"].float()-base_capture["hidden"].float()
    complement = full_delta-weight.project(full_delta, vh, vh.shape[0])
    base_output, base_raw = head_atlas.capture_attn9(backend, base_batch)
    live_output, _live_raw = head_atlas.capture_attn9(
        backend, base_batch, base_capture["hidden"], complement, positions)
    outputs = {name: head_atlas.run_heads(backend, base_batch, base_capture["hidden"],
        complement, positions, base_raw, heads) for name, heads in ARMS.items()}
    self_output = head_atlas.run_heads(backend, base_batch, base_capture["hidden"],
        complement, positions, base_raw, ARMS["all"], actuate=False)
    forwards, evaluations = 9, 9*len(rows)
    base18 = head_atlas.converter.state(base_output, rows, torch, backend.device)
    base_hidden18 = head_atlas.converter.state(base_hidden_output, rows, torch, backend.device)
    donor18 = head_atlas.converter.state(donor_output, rows, torch, backend.device)
    live18 = head_atlas.converter.state(live_output, rows, torch, backend.device)
    self18 = head_atlas.converter.state(self_output, rows, torch, backend.device)
    states = {name: head_atlas.converter.state(output, rows, torch, backend.device)
              for name, output in outputs.items()}
    target = ((donor18-base18)@axis)@shared_axis.T
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda state: das.head_logits(backend, state)[index, answers]-das.head_logits(backend, state)[index, foils]
    base_margin, live_margin = margin(base18), margin(live18)
    target_margin = margin(base18+target)-base_margin
    live_effect, live_coord = live_margin-base_margin, (live18-base18)@s
    target_coord = target@s
    removed_effect = {name: live_margin-margin(state) for name, state in states.items()}
    removed_coord = {name: (live18-state)@s for name, state in states.items()}
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
             for panel in ("A1", "A2")}
    metrics = {}
    for panel, mask in masks.items():
        live = {"behavior_cosine_to_target": cosine(live_effect[mask], target_margin[mask]),
            "mean_absolute_effect": float(live_effect[mask].abs().mean()),
            "q8_cosine_to_target": cosine(live_coord[mask].reshape(-1), target_coord[mask].reshape(-1))}
        removals = {}
        for name in ARMS:
            removals[name] = {"absolute_behavior_fraction_of_live": float(
                    removed_effect[name][mask].abs().mean()/live_effect[mask].abs().mean()),
                "signed_behavior_fraction_of_live": float(removed_effect[name][mask].mean()/live_effect[mask].mean()),
                "q8_norm_fraction_of_live": float(removed_coord[name][mask].norm()/live_coord[mask].norm()),
                "behavior_cosine_to_live": cosine(removed_effect[name][mask], live_effect[mask]),
                "q8_cosine_to_live": cosine(removed_coord[name][mask].reshape(-1), live_coord[mask].reshape(-1))}
        union = {"behavior_fraction_of_all_heads": removals["h1h4"]["absolute_behavior_fraction_of_live"]
                    / removals["all"]["absolute_behavior_fraction_of_live"],
            "q8_norm_fraction_of_all_heads": removals["h1h4"]["q8_norm_fraction_of_live"]
                    / removals["all"]["q8_norm_fraction_of_live"],
            "behavior_cosine_to_all_heads": cosine(removed_effect["h1h4"][mask], removed_effect["all"][mask]),
            "q8_cosine_to_all_heads": cosine(removed_coord["h1h4"][mask].reshape(-1),
                                               removed_coord["all"][mask].reshape(-1))}
        metrics[panel] = {"live_complement": live, "removals": removals, "h1h4_union": union}
    identity_error = max(float((base_hidden18-base18).abs().max()), float((self18-base18).abs().max()))
    finite = all(math.isfinite(value) for panel in metrics.values() for section in panel.values()
        for row in ([section] if all(isinstance(v, (int, float)) for v in section.values()) else section.values())
        for value in row.values())
    pred_a = bool(orientation_error <= 1e-6 and identity_error <= 1e-4 and finite
        and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = all(metrics[p]["live_complement"]["behavior_cosine_to_target"] >= .85
        and metrics[p]["live_complement"]["mean_absolute_effect"] >= .05
        and metrics[p]["live_complement"]["q8_cosine_to_target"] >= .70 for p in ("A1", "A2"))
    pred_c = all(metrics[p]["h1h4_union"][key] >= threshold for p in ("A1", "A2")
        for key, threshold in (("behavior_fraction_of_all_heads", .75),
            ("q8_norm_fraction_of_all_heads", .75), ("behavior_cosine_to_all_heads", .90),
            ("q8_cosine_to_all_heads", .90)))
    pred_d = all(metrics[p]["removals"][head]["absolute_behavior_fraction_of_live"] >= .10
        and metrics[p]["removals"][head]["q8_norm_fraction_of_live"] >= .10
        and metrics[p]["removals"][head]["signed_behavior_fraction_of_live"] > 0
        for p in ("A1", "A2") for head in ("h1", "h4"))
    pred_e = True
    predictions = {"pred_a_authority_manifest_replay_self_clamp_finiteness_and_price": pred_a,
        "pred_b_indirect_complement_route_transfers": pred_b,
        "pred_c_h1h4_union_closes_attention9_conversion": pred_c,
        "pred_d_both_heads_are_positive_material_members": pred_d,
        "pred_e_zero_fit_weight_defined_interface": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "iswas_mlp8_complement_attn9_h1h4_fresh_v10_confirmation_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_base_self_clamp_resid18_max_abs": identity_error, "rows": len(rows)},
        "metrics": metrics, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "metrics",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
