#!/usr/bin/env python3
"""Prospective v10 confirmation of the frozen five-head Q8 converter."""

# BQGATE: EXPERIMENT pred_a_authority_replay_self_clamp_finiteness_and_price pred_b_three_attention_route_is_material_and_coherent pred_c_five_head_program_is_sufficient pred_d_each_module_contributes_to_the_five_head_union pred_e_zero_fit_prospective_confirmation
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1 as auxiliary
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_complement_five_head_fresh_v10_confirmation_v1.json"
SELECTION = ROOT / "circuits/followups/iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1_result.json"
SELECTION_RUNNER = ROOT / "ops/run_iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10.py"
CORE_CONFIRMATION = ROOT / "circuits/followups/iswas_mlp8_complement_attn9_h1h4_fresh_v10_confirmation_v1_result.json"
OUT = ROOT / "circuits/followups/iswas_mlp8_complement_five_head_fresh_v10_confirmation_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_complement_five_head_fresh_v10_confirmation_v1"
EXPECTED = {
    "prior": "d0493ca95432714ca5a4b53a3f967dc7a416d4ca9e33bc69e5955f963dd94bde",
    "selection": "08975d2a9682a8f880e19230208efd56d7e410e274e6a3f4291b44248dfecd94",
    "selection_runner": "0aaf807785321fea563c304684d670f83f9f559ef2eab501274333930084dda1",
    "capability": "77e3c5b6e47dc9416133643f422c130427f90fed0746d26f211f54b681718b3d",
    "builder": "13e7954cde01b6b7d826915fc2ae02d4b9e16975150cf73aa5e0a1f906c1b757",
    "core_confirmation": "eb4470b3ed1495a5ad11c0a1e685d0c6ca4e9942532a872f165bc16f07eb69f5",
}
SELECTED = {9: (1, 4), 11: (1, 3), 15: (5,)}
ALL = tuple(range(9))
MAX_FORWARDS, MAX_EVALUATIONS = 17, 459


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def cosine(x, y) -> float:
    denominator = float(x.norm() * y.norm())
    return float((x * y).sum()) / denominator if denominator else 0.0


def without_head(layer, head):
    return {key: tuple(value for value in heads if not (key == layer and value == head))
            for key, heads in SELECTED.items()}


def without_module(layer):
    return {key: heads for key, heads in SELECTED.items() if key != layer}


def main() -> None:
    paths = {"prior": PRIOR, "selection": SELECTION, "selection_runner": SELECTION_RUNNER,
             "capability": CAPABILITY, "builder": BUILDER, "core_confirmation": CORE_CONFIRMATION}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("five-head confirmation authority changed")
    inherited_paths = {
        "prior": converter.PRIOR, "weight_v2": converter.WEIGHT_V2,
        "weight_v2_runner": converter.WEIGHT_V2_RUNNER,
        "weight_instrument": converter.WEIGHT_INSTRUMENT, "source": converter.SOURCE,
        "capability": weight.CAPABILITY, "iswas": weight.ISWAS,
        "subspace": weight.SUBSPACE, "builder": weight.BUILDER,
        "family_runner": weight.FAMILY_RUNNER, "overlap_runner": weight.OVERLAP_RUNNER,
    }
    if {name: sha(path) for name, path in inherited_paths.items()} != converter.EXPECTED:
        raise RuntimeError("inherited converter authority changed")
    prior, selection, capability, core_confirmation, subspace = [json.loads(path.read_text())
        for path in (PRIOR, SELECTION, CAPABILITY, CORE_CONFIRMATION, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    positions = [weight.postcue_positions(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or capability.get("terminal") != "screen"
            or capability.get("causal_outcomes_opened") is not False
            or selection.get("stable_material_heads") != ["attn11_head1", "attn11_head3", "attn15_head5"]
            or core_confirmation.get("terminal") != "screen"
            or {p: sum(row["family"] == p for row in rows) for p in ("A1", "A2")} != {"A1": 13, "A2": 14}):
        raise RuntimeError("selection, capability, or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "selected_heads": SELECTED, "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS, "fit_updates": 0,
        "transformer_backwards": 0, "model_updates": 0}
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
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_hidden_output, base_capture = weight.capture_mlp8(backend, base_batch)
    _donor_output, donor_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(s.T @ down, full_matrices=False)
    full_delta = donor_capture["hidden"].float() - base_capture["hidden"].float()
    complement = full_delta - weight.project(full_delta, vh, vh.shape[0])
    base_output, base_raws = auxiliary.capture_raws(backend, base_batch)
    live_output, _live_raws = auxiliary.capture_raws(
        backend, base_batch, base_capture["hidden"], complement, positions)
    arms = {
        "core_selected": SELECTED | {11: (), 15: ()},
        "core_all": {9: ALL},
        "five_selected": SELECTED,
        "three_all": {9: ALL, 11: ALL, 15: ALL},
        **{f"drop_l{layer}h{head}": without_head(layer, head)
           for layer, heads in SELECTED.items() for head in heads},
        **{f"drop_layer{layer}": without_module(layer) for layer in SELECTED},
    }
    outputs = {name: auxiliary.run_heads(backend, base_batch, base_capture["hidden"], complement,
        positions, base_raws, selected) for name, selected in arms.items()}
    self_output = auxiliary.run_heads(backend, base_batch, base_capture["hidden"], complement,
        positions, base_raws, {9: ALL, 11: ALL, 15: ALL}, actuate=False)
    forwards, evaluations = 17, 17 * len(rows)
    state = lambda output: converter.state(output, rows, torch, backend.device)
    base18, base_hidden18, live18, self18 = map(state, (base_output, base_hidden_output, live_output, self_output))
    states = {name: state(value) for name, value in outputs.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda value: das.head_logits(backend, value)[index, answers] - das.head_logits(backend, value)[index, foils]
    base_margin, live_margin = margin(base18), margin(live18)
    removed_e = {name: live_margin - margin(value) for name, value in states.items()}
    removed_c = {name: (live18 - value) @ s for name, value in states.items()}
    live_e, live_c = live_margin - base_margin, (live18 - base18) @ s
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
             for panel in ("A1", "A2")}
    metrics = {panel: {} for panel in masks}
    for panel, mask in masks.items():
        full_e, full_c = removed_e["three_all"][mask], removed_c["three_all"][mask]
        for name in arms:
            e, c = removed_e[name][mask], removed_c[name][mask]
            metrics[panel][name] = {
                "absolute_behavior_fraction_of_live": float(e.abs().mean() / live_e[mask].abs().mean()),
                "q8_norm_fraction_of_live": float(c.norm() / live_c[mask].norm()),
                "behavior_cosine_to_live": cosine(e, live_e[mask]),
                "q8_cosine_to_live": cosine(c.reshape(-1), live_c[mask].reshape(-1)),
                "behavior_fraction_of_three_all": float(e.abs().mean() / full_e.abs().mean()),
                "q8_norm_fraction_of_three_all": float(c.norm() / full_c.norm()),
                "behavior_cosine_to_three_all": cosine(e, full_e),
                "q8_cosine_to_three_all": cosine(c.reshape(-1), full_c.reshape(-1)),
                "behavior_relative_rmse_to_three_all": float((e - full_e).norm() / full_e.norm()),
                "q8_relative_rmse_to_three_all": float((c - full_c).norm() / full_c.norm()),
            }
    core_replay = max(abs(metrics[p]["core_selected"][key]
        - core_confirmation["metrics"][p]["removals"]["h1h4"][parent_key])
        for p in ("A1", "A2")
        for key, parent_key in (("absolute_behavior_fraction_of_live", "absolute_behavior_fraction_of_live"),
                                ("q8_norm_fraction_of_live", "q8_norm_fraction_of_live")))
    identity_error = max(float((base_hidden18 - base18).abs().max()), float((self18 - base18).abs().max()))
    finite = all(math.isfinite(value) for panel in metrics.values()
                 for arm in panel.values() for value in arm.values())
    pred_a = bool(orientation_error <= 1e-6 and identity_error <= 1e-4 and core_replay <= .001
                  and finite and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = all(metrics[p]["three_all"][key] >= threshold for p in ("A1", "A2")
        for key, threshold in (("absolute_behavior_fraction_of_live", .90),
            ("q8_norm_fraction_of_live", .90), ("behavior_cosine_to_live", .98),
            ("q8_cosine_to_live", .98)))
    pred_c = all(metrics[p]["five_selected"][key] >= threshold for p in ("A1", "A2")
        for key, threshold in (("behavior_fraction_of_three_all", .80),
            ("q8_norm_fraction_of_three_all", .75), ("behavior_cosine_to_three_all", .98),
            ("q8_cosine_to_three_all", .95)))
    pred_d = all(any(metrics[p][f"drop_layer{layer}"][key]
        > metrics[p]["five_selected"][key] + 1e-4
        for key in ("behavior_relative_rmse_to_three_all", "q8_relative_rmse_to_three_all"))
        for p in ("A1", "A2") for layer in SELECTED)
    pred_e = True
    predictions = {
        "pred_a_authority_replay_self_clamp_finiteness_and_price": pred_a,
        "pred_b_three_attention_route_is_material_and_coherent": pred_b,
        "pred_c_five_head_program_is_sufficient": pred_c,
        "pred_d_each_module_contributes_to_the_five_head_union": pred_d,
        "pred_e_zero_fit_prospective_confirmation": pred_e,
    }
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "iswas_mlp8_complement_five_head_fresh_v10_confirmation_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": {**EXPECTED, **{f"inherited_{name}": value for name, value in converter.EXPECTED.items()}},
        "dryrun": dryrun, "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_base_self_clamp_resid18_max_abs": identity_error,
            "prior_core_h1h4_metric_replay_max_abs": core_replay, "rows": len(rows)},
        "selected_heads": SELECTED, "metrics": metrics, "predictions": predictions,
        "terminal": terminal, "price": {"model_forwards": forwards,
            "example_evaluations": evaluations, "fit_updates": 0,
            "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "selected_heads",
        "metrics", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
