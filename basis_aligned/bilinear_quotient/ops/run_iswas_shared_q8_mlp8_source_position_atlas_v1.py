#!/usr/bin/env python3
"""Fresh-v6 source-position atlas for the natural MLP8 shared-Q8 writer."""

# BQGATE: EXPERIMENT pred_a_exact_authority_partition_identity_coverage_and_price pred_b_complete_mlp8_transfers_to_fresh_v6 pred_c_at_least_one_proper_position_group_is_material pred_d_causal_prefix_is_zero pred_e_small_source_union_beats_complete_mlp8_coordinate_error
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_shared_q8_mlp8_source_position_atlas_v1.json"
GREEDY = ROOT / "circuits/followups/iswas_shared_q8_greedy_module_program_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v6_capability_v1_result.json"
ISWAS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6.py"
OVERLAP_RUNNER = ROOT / "ops/run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1.py"
OUT = ROOT / "circuits/followups/iswas_shared_q8_mlp8_source_position_atlas_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_shared_q8_mlp8_source_position_atlas_v1"
EXPECTED = {
    "prior": "637c45d3a9f5e24e082af4e65cc4af62fe5f9ddea408907e42c4e4793246c78b",
    "greedy": "85c6117dd22a50a76c1d478279d3180c67b4d5403581c64b5204f4ce659e6c53",
    "capability": "86ec66fa81346e61382c951e46899236ee1b7b7ec32c16948936fd9de6f77940",
    "iswas": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "builder": "b8541360334bd2793a02fae525a94dda05ce600fd4de5b6c3d953063d4c6b0ae",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1",
}
GROUPS = ("prefix", "cue", "post_cue", "subject_determiner", "self")
ARMS = ("base",)+GROUPS+("cue_suffix", "subject_suffix", "complete_mlp8", "target_shared_resid18")
MAX_FORWARDS, MAX_EVALUATIONS = 12, 768


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def partition(row):
    differences = [index for index, pair in enumerate(zip(row["base_ids"], row["donor_ids"]))
                   if pair[0] != pair[1]]
    if len(row["base_ids"]) != len(row["donor_ids"]) or len(differences) != 1:
        raise RuntimeError("source atlas requires one aligned cue")
    cue, query = differences[0], int(row["base_semantic_position"])
    groups = {"prefix": tuple(range(cue)), "cue": (cue,),
        "post_cue": tuple(range(cue+1, query-1)), "subject_determiner": (query-1,),
        "self": (query,)}
    flat = tuple(position for name in GROUPS for position in groups[name])
    if tuple(sorted(flat)) != tuple(range(query+1)) or len(flat) != len(set(flat)):
        raise RuntimeError("position groups do not partition causal prefix")
    return groups


def capture_mlp8(backend, batch):
    cache = {}
    def capture(_module, _arguments, output):
        cache["mlp8"] = output.detach().clone()
    handle = backend.model.transformer.h[8].mlp.register_forward_hook(capture)
    try:
        output = backend.native(batch, capture=True)
    finally:
        handle.remove()
    return output, cache["mlp8"]


def run_patch(backend, batch, values, positions):
    def patch(_module, _arguments, output):
        changed = output.clone()
        for index, row_positions in enumerate(positions):
            if row_positions:
                changed[index, list(row_positions)] = values[index, list(row_positions)].to(changed)
        return changed
    handle = backend.model.transformer.h[8].mlp.register_forward_hook(patch)
    try:
        return backend.native(batch, capture=True)
    finally:
        handle.remove()


def cosine(x, y):
    denominator = float(x.norm()*y.norm())
    # A zero intervention has zero alignment with the nonzero preregistered
    # target.  Emit the finite mathematical value needed by the result
    # contract rather than treating an intentionally null arm as invalid.
    return float((x*y).sum())/denominator if denominator else 0.0


def main():
    paths = {"prior": PRIOR, "greedy": GREEDY, "capability": CAPABILITY, "iswas": ISWAS,
        "subspace": SUBSPACE, "builder": BUILDER, "overlap_runner": OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("MLP8 source-position authority changed")
    prior, greedy, capability, iswas, subspace = [json.loads(path.read_text())
        for path in (PRIOR, GREEDY, CAPABILITY, ISWAS, SUBSPACE)]
    capable = {}
    for record in capability["native_records"]:
        capable.setdefault(record["row_id"], {})[record["side"]] = bool(record["correct"])
    allowed = {row_id for row_id, sides in capable.items() if sides == {"base": True, "donor": True}}
    rows = [row for row in candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    partitions = [partition(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or greedy.get("terminal") != "null"
            or capability.get("terminal") != "screen" or iswas.get("terminal") != "screen"
            or not rows):
        raise RuntimeError("authority terminal or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "arms": list(ARMS),
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
    q = family[8]
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in range(12, 18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, q, gain)
    s = torch.linalg.qr(modes, mode="reduced").Q
    axis_values = np.asarray(iswas["basis"]["values_column_major"], dtype=np.float32)
    if hashlib.sha256(axis_values.tobytes()).hexdigest() != iswas["basis"]["sha256"]:
        raise RuntimeError("is/was axis changed")
    axis = torch.as_tensor(axis_values, device=backend.device).reshape(1152, 1)
    axis = axis/torch.linalg.vector_norm(axis); shared_axis = s@(s.T@axis)
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_output, base_mlp8 = capture_mlp8(backend, base_batch)
    donor_output, donor_mlp8 = capture_mlp8(backend, donor_batch)
    forwards, evaluations = 2, 2*len(rows)
    base18 = torch.stack([torch.as_tensor(base_output.captured[(row["row_id"], "resid:18")])
                          for row in rows]).to(backend.device).float()
    donor18 = torch.stack([torch.as_tensor(donor_output.captured[(row["row_id"], "resid:18")])
                           for row in rows]).to(backend.device).float()
    target = ((donor18-base18)@axis)@shared_axis.T
    arm_positions = {name: [parts[name] for parts in partitions] for name in GROUPS}
    arm_positions["cue_suffix"] = [tuple(position for name in ("cue", "post_cue", "subject_determiner", "self")
        for position in parts[name]) for parts in partitions]
    arm_positions["subject_suffix"] = [parts["subject_determiner"]+parts["self"] for parts in partitions]
    arm_positions["complete_mlp8"] = [tuple(range(int(row["base_semantic_position"])+1)) for row in rows]
    arm_states = {"base": base18, "target_shared_resid18": base18+target}
    self_output = run_patch(backend, base_batch, base_mlp8, arm_positions["complete_mlp8"])
    forwards += 1; evaluations += len(rows)
    self18 = torch.stack([torch.as_tensor(self_output.captured[(row["row_id"], "resid:18")])
                          for row in rows]).to(backend.device).float()
    self_error = float((self18-base18).abs().max())
    for arm, positions in arm_positions.items():
        output = run_patch(backend, base_batch, donor_mlp8, positions)
        arm_states[arm] = torch.stack([torch.as_tensor(output.captured[(row["row_id"], "resid:18")])
                                       for row in rows]).to(backend.device).float()
        forwards += 1; evaluations += len(rows)
    logits = {arm: das.head_logits(backend, state) for arm, state in arm_states.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margins = {arm: value[index, answers]-value[index, foils] for arm, value in logits.items()}
    target_effect = margins["target_shared_resid18"]-margins["base"]
    target_coordinates = target@s
    metrics = {}
    for arm in arm_positions:
        effect = margins[arm]-margins["base"]
        coordinates = (arm_states[arm]-base18)@s
        coordinate_rmse = float(torch.sqrt(((coordinates-target_coordinates)**2).mean()))
        coordinate_rms = float(torch.sqrt(target_coordinates.square().mean()))
        behavior_rmse = float(torch.sqrt(((effect-target_effect)**2).mean()))
        behavior_rms = float(torch.sqrt(target_effect.square().mean()))
        metrics[arm] = {"coordinate_cosine": cosine(coordinates.reshape(-1), target_coordinates.reshape(-1)),
            "coordinate_relative_rmse": coordinate_rmse/coordinate_rms,
            "behavior_cosine": cosine(effect, target_effect),
            "behavior_relative_rmse": behavior_rmse/behavior_rms,
            "mean_effect": float(effect.mean()), "mean_absolute_effect": float(effect.abs().mean())}
    complete_abs = metrics["complete_mlp8"]["mean_absolute_effect"]
    for arm in GROUPS:
        metrics[arm]["absolute_behavior_fraction_of_complete"] = (
            metrics[arm]["mean_absolute_effect"]/complete_abs)
    prefix_raw = max(float((donor_mlp8[index, list(partitions[index]["prefix"])]
        - base_mlp8[index, list(partitions[index]["prefix"])]).abs().max())
        if partitions[index]["prefix"] else 0.0 for index in range(len(rows)))
    prefix_effect = metrics["prefix"]["mean_absolute_effect"]
    pred_a = bool(orientation_error <= 1e-6 and self_error <= 1e-4
        and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS
        and all(math.isfinite(value) for row in metrics.values() for value in row.values()))
    complete = metrics["complete_mlp8"]
    pred_b = bool(complete["behavior_cosine"] >= .90 and complete["behavior_relative_rmse"] <= .50
        and complete["coordinate_cosine"] >= .70 and complete["coordinate_relative_rmse"] <= .90)
    material = [arm for arm in ("cue", "post_cue", "subject_determiner", "self")
        if metrics[arm]["absolute_behavior_fraction_of_complete"] >= .25
        and metrics[arm]["coordinate_cosine"] >= .50]
    pred_c = bool(material)
    pred_d = prefix_raw <= 1e-6 and prefix_effect <= 1e-6
    best_union = min(("cue_suffix", "subject_suffix"),
        key=lambda arm: metrics[arm]["coordinate_relative_rmse"])
    pred_e = metrics[best_union]["coordinate_relative_rmse"] <= complete["coordinate_relative_rmse"]
    predictions = {"pred_a_exact_authority_partition_identity_coverage_and_price": pred_a,
        "pred_b_complete_mlp8_transfers_to_fresh_v6": pred_b,
        "pred_c_at_least_one_proper_position_group_is_material": pred_c,
        "pred_d_causal_prefix_is_zero": pred_d,
        "pred_e_small_source_union_beats_complete_mlp8_coordinate_error": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "iswas_shared_q8_mlp8_source_position_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "complete_base_self_patch_resid18_max_abs": self_error,
            "prefix_raw_mlp8_delta_max_abs": prefix_raw, "rows": len(rows)},
        "metrics": metrics, "material_proper_groups": material,
        "best_proper_union": best_union, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "metrics",
        "material_proper_groups", "best_proper_union", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__":
    main()
