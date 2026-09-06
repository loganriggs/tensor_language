#!/usr/bin/env python3
"""All-subset source atlas for auxiliary L11/L15 value transport."""

# BQGATE: EXPERIMENT pred_a_authority_partition_value_replay_finiteness_and_price pred_b_at_least_one_small_source_program_is_sufficient pred_c_source_program_is_panel_stable pred_d_both_auxiliary_layers_have_material_value_reads pred_e_zero_fit_exact_subset_inventory
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import attention_source_destination_eval as attention_eval
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_auxiliary_three_head_factor_program_v1 as factor
import run_iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1 as auxiliary
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_auxiliary_value_source_subset_atlas_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_three_head_factor_program_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_auxiliary_three_head_factor_program_v1.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10.py"
ATTENTION_LIBRARY = ROOT / "ops/attention_source_destination_eval.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_value_source_subset_atlas_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_auxiliary_value_source_subset_atlas_v1"
EXPECTED = {
    "prior": "a5cdd6cdb87a107456a05f81ad5ad0ac3bd858f175a0fcc120a6507d3fe4ee63",
    "parent": "cbf754a5edb2bac0cbaf56916f901c95d48296827952a641e63860e4f3e03c73",
    "parent_runner": "b8e1dc98014394c75a90c9e721f4347be5bbd3ac73d65cd56096070c1ecad304",
    "capability": "77e3c5b6e47dc9416133643f422c130427f90fed0746d26f211f54b681718b3d",
    "builder": "13e7954cde01b6b7d826915fc2ae02d4b9e16975150cf73aa5e0a1f906c1b757",
    "attention_library": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
}
GROUPS = ("prefix_before_cue", "cue", "post_cue_before_subject", "subject_determiner", "self")
LAYERS, SELECTED, CORE = (11, 15), {11: (1, 3), 15: (5,)}, (1, 4)
MAX_FORWARDS, MAX_EVALUATIONS = 40, 1080


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def subset_name(subset) -> str:
    return "+".join(subset) if subset else "empty"


def subsets():
    for width in range(len(GROUPS) + 1):
        yield from itertools.combinations(GROUPS, width)


def cosine(x, y) -> float:
    denominator = float(x.norm() * y.norm())
    return float((x * y).sum()) / denominator if denominator else 0.0


def source_partition(row):
    differences = [i for i, pair in enumerate(zip(row["base_ids"], row["donor_ids"])) if pair[0] != pair[1]]
    if len(differences) != 1:
        raise RuntimeError("source atlas requires one aligned cue")
    cue, query = differences[0], int(row["base_semantic_position"])
    groups = {"prefix_before_cue": tuple(range(cue)), "cue": (cue,),
        "post_cue_before_subject": tuple(range(cue + 1, query - 1)),
        "subject_determiner": (query - 1,), "self": (query,)}
    flat = tuple(position for name in GROUPS for position in groups[name])
    if tuple(sorted(flat)) != tuple(range(query + 1)) or len(flat) != len(set(flat)):
        raise RuntimeError("source groups do not partition causal positions")
    return groups


def capture_dual(backend, batch, *, call, capture_raw9=False):
    captures = {layer: {} for layer in LAYERS}
    raw9, handles = {}, []
    if capture_raw9:
        def capture9(_module, arguments):
            raw9["value"] = arguments[0].detach().clone()
        handles.append(backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(capture9))
    for layer in LAYERS:
        attention = backend.model.transformer.h[layer].attn
        def capture_inputs(_module, arguments, layer=layer, attention=attention):
            current, v1 = arguments[0], arguments[1] if len(arguments) > 1 else None
            pattern, value, reconstructed = attention_eval._attention_terms(backend, attention, current, v1)
            captures[layer].update(pattern=pattern.detach().clone(), value=value.detach().clone(),
                                   reconstructed=reconstructed.detach().clone())
        def capture_output(_module, arguments, layer=layer):
            raw = arguments[0]
            heads = int(backend.model.config.n_head)
            captures[layer]["head_output"] = raw.detach().clone().view(
                len(batch.row_ids), raw.shape[1], heads, raw.shape[2] // heads)
        handles.append(attention.register_forward_pre_hook(capture_inputs))
        handles.append(attention.c_proj.register_forward_pre_hook(capture_output))
    try:
        output = call()
    finally:
        for handle in handles: handle.remove()
    for layer in LAYERS:
        captures[layer]["reconstruction_max_abs"] = float(
            (captures[layer]["reconstructed"].float() - captures[layer]["head_output"].float()).abs().max())
    return output, captures, raw9.get("value")


def main() -> None:
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER,
             "capability": CAPABILITY, "builder": BUILDER, "attention_library": ATTENTION_LIBRARY}
    if {key: sha(value) for key, value in paths.items()} != EXPECTED:
        raise RuntimeError("auxiliary source authority changed")
    inherited_paths = {
        "prior": converter.PRIOR, "weight_v2": converter.WEIGHT_V2,
        "weight_v2_runner": converter.WEIGHT_V2_RUNNER,
        "weight_instrument": converter.WEIGHT_INSTRUMENT, "source": converter.SOURCE,
        "capability": weight.CAPABILITY, "iswas": weight.ISWAS,
        "subspace": weight.SUBSPACE, "builder": weight.BUILDER,
        "family_runner": weight.FAMILY_RUNNER, "overlap_runner": weight.OVERLAP_RUNNER,
    }
    if {key: sha(value) for key, value in inherited_paths.items()} != converter.EXPECTED:
        raise RuntimeError("inherited converter authority changed")
    prior, parent, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, PARENT, CAPABILITY, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    source_positions = [weight.postcue_positions(row) for row in rows]
    partitions = [source_partition(row) for row in rows]
    if prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen" or len(rows) != 27:
        raise RuntimeError("parent decision or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "groups": list(GROUPS),
        "subset_arms": [subset_name(value) for value in subsets()],
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
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_hidden_output, base_hidden_capture = weight.capture_mlp8(backend, base_batch)
    _donor_output, donor_hidden_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(s.T @ down, full_matrices=False)
    full_hidden_delta = donor_hidden_capture["hidden"].float() - base_hidden_capture["hidden"].float()
    complement = full_hidden_delta - weight.project(full_hidden_delta, vh, vh.shape[0])
    base_output, base_captures, base_raw9 = capture_dual(
        backend, base_batch, capture_raw9=True, call=lambda: backend.native(base_batch, capture=True))
    core_output, changed_captures, _unused = capture_dual(
        backend, base_batch, capture_raw9=False,
        call=lambda: auxiliary.run_heads(backend, base_batch, base_hidden_capture["hidden"],
            complement, source_positions, {9: base_raw9}, {9: CORE}))
    zero = {layer: torch.zeros_like(base_captures[layer]["head_output"], dtype=torch.float32)
            for layer in LAYERS}
    cells = {layer: {} for layer in LAYERS}
    for layer in LAYERS:
        dp = changed_captures[layer]["pattern"].float() - base_captures[layer]["pattern"].float()
        dv = changed_captures[layer]["value"].float() - base_captures[layer]["value"].float()
        base_pattern = base_captures[layer]["pattern"].float()
        for group in GROUPS:
            cell = torch.zeros_like(zero[layer])
            for i, source in enumerate(partitions):
                selected = list(source[group])
                raw = torch.einsum("hqs,shd->qhd", base_pattern[i, :, :, selected], dv[i, selected])
                cell[i, :, list(SELECTED[layer])] = raw[:, list(SELECTED[layer])]
            cells[layer][group] = cell
    all_value = {layer: sum(cells[layer].values(), zero[layer]) for layer in LAYERS}
    full_factors = {}
    for layer in LAYERS:
        base_pattern, changed_pattern = base_captures[layer]["pattern"].float(), changed_captures[layer]["pattern"].float()
        base_value, changed_value = base_captures[layer]["value"].float(), changed_captures[layer]["value"].float()
        dp, dv = changed_pattern - base_pattern, changed_value - base_value
        raw = (torch.einsum("bhqk,bkhd->bqhd", dp, base_value)
            + torch.einsum("bhqk,bkhd->bqhd", base_pattern, dv)
            + torch.einsum("bhqk,bkhd->bqhd", dp, dv))
        full_factors[layer] = torch.zeros_like(zero[layer])
        full_factors[layer][:, :, list(SELECTED[layer])] = raw[:, :, list(SELECTED[layer])]
    arm_deltas = {subset_name(subset): {layer: sum((cells[layer][group] for group in subset), zero[layer])
                                        for layer in LAYERS} for subset in subsets()}
    arm_deltas.update({"layer11_all": {11: all_value[11], 15: zero[15]},
                       "layer15_all": {11: zero[11], 15: all_value[15]},
                       "complete_all_factors": full_factors})
    outputs = {name: factor.run_factor_arm(backend, base_batch, base_hidden_capture["hidden"], complement,
        source_positions, base_raw9, base_captures, deltas) for name, deltas in arm_deltas.items()}
    self_output = factor.run_factor_arm(backend, base_batch, base_hidden_capture["hidden"], complement,
        source_positions, base_raw9, base_captures, zero, actuate=False)
    forwards, evaluations = 40, 40 * len(rows)
    all_name = subset_name(GROUPS)
    value_closure = 0.0
    for layer in LAYERS:
        raw_value = torch.einsum("bhqk,bkhd->bqhd", base_captures[layer]["pattern"].float(),
            changed_captures[layer]["value"].float() - base_captures[layer]["value"].float())
        expected_value = torch.zeros_like(all_value[layer])
        expected_value[:, :, list(SELECTED[layer])] = raw_value[:, :, list(SELECTED[layer])]
        value_closure = max(value_closure, float((all_value[layer] - expected_value).abs().max()))
    reconstruction = max(capture["reconstruction_max_abs"] for capture in
                         (*base_captures.values(), *changed_captures.values()))
    state = lambda output: converter.state(output, rows, torch, backend.device)
    base18, base_hidden18, core18, self18 = map(state, (base_output, base_hidden_output, core_output, self_output))
    states = {name: state(value) for name, value in outputs.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda value: das.head_logits(backend, value)[index, answers] - das.head_logits(backend, value)[index, foils]
    empty_margin, value_margin = margin(states["empty"]), margin(states[all_name])
    value_e, value_c = value_margin - empty_margin, (states[all_name] - states["empty"]) @ s
    complete_e = margin(states["complete_all_factors"]) - empty_margin
    complete_c = (states["complete_all_factors"] - states["empty"]) @ s
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
             for panel in ("A1", "A2")}
    metrics = {panel: {} for panel in masks}
    parent_replay = 0.0
    for panel, mask in masks.items():
        for arm, value in states.items():
            e, c = (margin(value) - empty_margin)[mask], ((value - states["empty"]) @ s)[mask]
            metrics[panel][arm] = {
                "absolute_behavior_fraction_of_all_value": float(e.abs().mean() / value_e[mask].abs().mean()),
                "behavior_cosine_to_all_value": cosine(e, value_e[mask]),
                "q8_norm_fraction_of_all_value": float(c.norm() / value_c[mask].norm()),
                "q8_cosine_to_all_value": cosine(c.reshape(-1), value_c[mask].reshape(-1)),
            }
        observed = {"absolute_behavior_fraction_of_complete": float(value_e[mask].abs().mean() / complete_e[mask].abs().mean()),
            "q8_norm_fraction_of_complete": float(value_c[mask].norm() / complete_c[mask].norm())}
        for key, value in observed.items():
            parent_replay = max(parent_replay, abs(value - parent["metrics"][panel]["base_pattern_on_value_change"][key]))
    sufficient = {}
    for panel in ("A1", "A2"):
        candidates = []
        for subset in subsets():
            arm = subset_name(subset); row = metrics[panel][arm]
            if len(subset) <= 2 and row["absolute_behavior_fraction_of_all_value"] >= .75 \
                    and row["q8_norm_fraction_of_all_value"] >= .75 \
                    and row["behavior_cosine_to_all_value"] >= .95 \
                    and row["q8_cosine_to_all_value"] >= .95:
                candidates.append((len(subset), arm))
        sufficient[panel] = min(candidates)[1] if candidates else None
    identity_error = max(float((base_hidden18 - base18).abs().max()), float((self18 - base18).abs().max()))
    finite = all(math.isfinite(value) for panel in metrics.values() for arm in panel.values() for value in arm.values())
    pred_a = bool(orientation_error <= 1e-6 and reconstruction <= .001 and value_closure <= .001
                  and parent_replay <= .001 and identity_error <= .001 and finite
                  and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = all(sufficient.values())
    pred_c = bool(pred_b and sufficient["A1"] == sufficient["A2"])
    pred_d = all(any(metrics[p][f"layer{layer}_all"][key] >= .10
        for key in ("absolute_behavior_fraction_of_all_value", "q8_norm_fraction_of_all_value"))
        for p in ("A1", "A2") for layer in LAYERS)
    pred_e = len([name for name in arm_deltas if name not in ("layer11_all", "layer15_all", "complete_all_factors")]) == 32
    predictions = {"pred_a_authority_partition_value_replay_finiteness_and_price": pred_a,
        "pred_b_at_least_one_small_source_program_is_sufficient": pred_b,
        "pred_c_source_program_is_panel_stable": pred_c,
        "pred_d_both_auxiliary_layers_have_material_value_reads": pred_d,
        "pred_e_zero_fit_exact_subset_inventory": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "iswas_mlp8_auxiliary_value_source_subset_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_attention_reconstruction_max_abs": reconstruction,
            "value_source_closure_max_abs": value_closure,
            "parent_all_value_metric_replay_max_abs": parent_replay,
            "native_self_clamp_max_abs": identity_error, "rows": len(rows)},
        "selected_small_source_program": sufficient, "metrics": metrics,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
        "selected_small_source_program", "metrics", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
