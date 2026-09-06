#!/usr/bin/env python3
"""Exact MLP8 Down/product modes for the fresh-v6 post-cue Q8 writer."""

# BQGATE: EXPERIMENT pred_a_authority_exact_factor_weight_closure_finiteness_and_price pred_b_rank8_weight_modes_preserve_the_postcue_writer pred_c_rank8_complement_is_secondary pred_d_left_right_terms_explain_the_weight_mode_write pred_e_weight_interface_is_compressive_and_zero_fit
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
PRIOR = ROOT / "circuits/prior_art/iswas_shared_q8_mlp8_postcue_weight_modes_v1.json"
SOURCE = ROOT / "circuits/followups/iswas_shared_q8_mlp8_source_position_atlas_v1_result.json"
SOURCE_RUNNER = ROOT / "ops/run_iswas_shared_q8_mlp8_source_position_atlas_v1.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v6_capability_v1_result.json"
ISWAS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6.py"
FAMILY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1.py"
OVERLAP_RUNNER = ROOT / "ops/run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1.py"
OUT = ROOT / "circuits/followups/iswas_shared_q8_mlp8_postcue_weight_modes_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_shared_q8_mlp8_postcue_weight_modes_v1"
RESULT_SCHEMA = "iswas_shared_q8_mlp8_postcue_weight_modes_result_v1"
DIRECT_Q8_TOLERANCE = 2e-4
EXPECTED = {
    "prior": "6a4b31d6b7f704683bef94bda53521c2639edcb6092794f7d3666776a1b9d719",
    "source": "ffa8596eea052e72eba3a5823dfdfcd124ee28ccc91ca48671efff567f23b14a",
    "source_runner": "dfc84abe3d319b8a92f0007005b9bc206eaab5d3986f3c9ab1d8a80d8e8ab477",
    "capability": "86ec66fa81346e61382c951e46899236ee1b7b7ec32c16948936fd9de6f77940",
    "iswas": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "builder": "b8541360334bd2793a02fae525a94dda05ce600fd4de5b6c3d953063d4c6b0ae",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1",
}
ARMS = ("rank1", "rank2", "rank4", "rank8", "rank8_complement", "complete_postcue",
        "left_rank8", "right_rank8", "interaction_rank8", "left_right_rank8")
MAX_FORWARDS, MAX_EVALUATIONS = 13, 377


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def postcue_positions(row):
    differences = [i for i, pair in enumerate(zip(row["base_ids"], row["donor_ids"]))
                   if pair[0] != pair[1]]
    if len(row["base_ids"]) != len(row["donor_ids"]) or len(differences) != 1:
        raise RuntimeError("post-cue modes require one aligned cue")
    cue, query = differences[0], int(row["base_semantic_position"])
    positions = tuple(range(cue + 1, query - 1))
    if not positions:
        raise RuntimeError("post-cue span is empty")
    return positions


def capture_mlp8(backend, batch):
    cache = {}
    module = backend.model.transformer.h[8].mlp
    handles = [
        module.Left.register_forward_hook(lambda _m, _a, out: cache.__setitem__("left", out.detach().clone())),
        module.Right.register_forward_hook(lambda _m, _a, out: cache.__setitem__("right", out.detach().clone())),
        module.Down.register_forward_pre_hook(lambda _m, args: cache.__setitem__("hidden", args[0].detach().clone())),
    ]
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()
    return output, cache


def run_hidden_patch(backend, batch, base_hidden, delta, positions):
    def patch(_module, arguments):
        changed = arguments[0].clone()
        for i, selected in enumerate(positions):
            changed[i, list(selected)] = (base_hidden[i, list(selected)].float()
                + delta[i, list(selected)].float()).to(changed)
        return (changed,) + tuple(arguments[1:])
    handle = backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(patch)
    try:
        return backend.native(batch, capture=True)
    finally:
        handle.remove()


def cosine(x, y):
    denominator = float(x.norm() * y.norm())
    return float((x*y).sum()) / denominator if denominator else 0.0


def project(hidden, vh, rank):
    basis = vh[:rank]
    return (hidden.float() @ basis.T) @ basis


def main():
    paths = {"prior": PRIOR, "source": SOURCE, "source_runner": SOURCE_RUNNER,
        "capability": CAPABILITY, "iswas": ISWAS, "subspace": SUBSPACE,
        "builder": BUILDER, "family_runner": FAMILY_RUNNER, "overlap_runner": OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("post-cue weight-mode authority changed")
    prior, source, capability, iswas, subspace = [json.loads(path.read_text())
        for path in (PRIOR, SOURCE, CAPABILITY, ISWAS, SUBSPACE)]
    capable = {}
    for record in capability["native_records"]:
        capable.setdefault(record["row_id"], {})[record["side"]] = bool(record["correct"])
    allowed = {row_id for row_id, sides in capable.items() if sides == {"base": True, "donor": True}}
    rows = [row for row in candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    positions = [postcue_positions(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or source.get("terminal") != "screen"
            or source.get("material_proper_groups") != ["post_cue"]
            or capability.get("terminal") != "screen" or iswas.get("terminal") != "screen"
            or len(rows) != 29):
        raise RuntimeError("authority terminal, population, or source decision changed")
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
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in range(12, 18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, family[8], gain)
    s = torch.linalg.qr(modes, mode="reduced").Q
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    a = s.T @ down
    u, singular, vh = torch.linalg.svd(a, full_matrices=False)
    numerical_rank = int((singular > singular.max() * 1e-6).sum())
    projector_error = float(((vh @ vh.T) - torch.eye(vh.shape[0], device=vh.device)).abs().max())
    svd_error = float((a - (u * singular.unsqueeze(0)) @ vh).abs().max())
    axis_values = np.asarray(iswas["basis"]["values_column_major"], dtype=np.float32)
    if hashlib.sha256(axis_values.tobytes()).hexdigest() != iswas["basis"]["sha256"]:
        raise RuntimeError("is/was axis changed")
    axis = torch.as_tensor(axis_values, device=backend.device).reshape(1152, 1)
    axis = axis / torch.linalg.vector_norm(axis)
    shared_axis = s @ (s.T @ axis)
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, base_capture = capture_mlp8(backend, base_batch)
    donor_output, donor_capture = capture_mlp8(backend, donor_batch)
    forwards, evaluations = 2, 2 * len(rows)
    base18 = torch.stack([torch.as_tensor(base_output.captured[(row["row_id"], "resid:18")])
                          for row in rows]).to(backend.device).float()
    donor18 = torch.stack([torch.as_tensor(donor_output.captured[(row["row_id"], "resid:18")])
                           for row in rows]).to(backend.device).float()
    target = ((donor18-base18) @ axis) @ shared_axis.T
    base_hidden, donor_hidden = base_capture["hidden"], donor_capture["hidden"]
    full_delta = donor_hidden.float() - base_hidden.float()
    dl = donor_capture["left"].float() - base_capture["left"].float()
    dr = donor_capture["right"].float() - base_capture["right"].float()
    factors = {"left": dl * base_capture["right"].float(),
        "right": base_capture["left"].float() * dr, "interaction": dl * dr}
    factor_error = float((sum(factors.values()) - full_delta).abs().max())
    projected8 = project(full_delta, vh, vh.shape[0])
    direct_q8_error = float((((full_delta-projected8) @ down.T) @ s).abs().max())
    arm_delta = {f"rank{rank}": project(full_delta, vh, min(rank, vh.shape[0]))
                 for rank in (1, 2, 4, 8)}
    arm_delta.update({"rank8_complement": full_delta-projected8,
        "complete_postcue": full_delta,
        "left_rank8": project(factors["left"], vh, vh.shape[0]),
        "right_rank8": project(factors["right"], vh, vh.shape[0]),
        "interaction_rank8": project(factors["interaction"], vh, vh.shape[0]),
        "left_right_rank8": project(factors["left"]+factors["right"], vh, vh.shape[0])})
    self_output = run_hidden_patch(backend, base_batch, base_hidden, torch.zeros_like(full_delta), positions)
    forwards += 1; evaluations += len(rows)
    self18 = torch.stack([torch.as_tensor(self_output.captured[(row["row_id"], "resid:18")])
                          for row in rows]).to(backend.device).float()
    self_error = float((self18-base18).abs().max())
    arm_states = {"base": base18, "target": base18+target}
    for arm in ARMS:
        output = run_hidden_patch(backend, base_batch, base_hidden, arm_delta[arm], positions)
        arm_states[arm] = torch.stack([torch.as_tensor(output.captured[(row["row_id"], "resid:18")])
                                       for row in rows]).to(backend.device).float()
        forwards += 1; evaluations += len(rows)
    logits = {arm: das.head_logits(backend, state) for arm, state in arm_states.items()}
    row_index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margins = {arm: value[row_index, answers]-value[row_index, foils] for arm, value in logits.items()}
    target_effect = margins["target"]-margins["base"]
    target_coordinates = target @ s
    metrics = {}
    for arm in ARMS:
        effect = margins[arm]-margins["base"]
        coordinates = (arm_states[arm]-base18) @ s
        metrics[arm] = {"behavior_cosine": cosine(effect, target_effect),
            "behavior_relative_rmse": float(torch.sqrt(((effect-target_effect)**2).mean())
                / torch.sqrt(target_effect.square().mean())),
            "coordinate_cosine": cosine(coordinates.reshape(-1), target_coordinates.reshape(-1)),
            "coordinate_relative_rmse": float(torch.sqrt(((coordinates-target_coordinates)**2).mean())
                / torch.sqrt(target_coordinates.square().mean())),
            "coordinate_rms": float(torch.sqrt(coordinates.square().mean())),
            "mean_absolute_effect": float(effect.abs().mean()), "mean_effect": float(effect.mean())}
    complete_abs = metrics["complete_postcue"]["mean_absolute_effect"]
    rank8_abs = metrics["rank8"]["mean_absolute_effect"]
    complete_coord = metrics["complete_postcue"]["coordinate_rms"]
    ratios = {"rank8_behavior_fraction_of_complete": rank8_abs/complete_abs,
        "complement_behavior_fraction_of_complete": metrics["rank8_complement"]["mean_absolute_effect"]/complete_abs,
        "complement_coordinate_rms_fraction_of_complete": metrics["rank8_complement"]["coordinate_rms"]/complete_coord,
        "left_right_behavior_fraction_of_rank8": metrics["left_right_rank8"]["mean_absolute_effect"]/rank8_abs,
        "interaction_behavior_fraction_of_rank8": metrics["interaction_rank8"]["mean_absolute_effect"]/rank8_abs}
    finite = all(math.isfinite(value) for row in metrics.values() for value in row.values())
    pred_a = bool(orientation_error <= 1e-6 and self_error <= 1e-4 and factor_error <= .02
        and projector_error <= 2e-5 and svd_error <= 2e-5
        and direct_q8_error <= DIRECT_Q8_TOLERANCE
        and finite and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = bool(metrics["rank8"]["behavior_cosine"] >= .90
        and metrics["rank8"]["behavior_relative_rmse"] <= .55
        and ratios["rank8_behavior_fraction_of_complete"] >= .80)
    pred_c = bool(ratios["complement_behavior_fraction_of_complete"] <= .20
        and ratios["complement_coordinate_rms_fraction_of_complete"] <= .20)
    pred_d = bool(ratios["left_right_behavior_fraction_of_rank8"] >= .80
        and ratios["interaction_behavior_fraction_of_rank8"] <= .25)
    pred_e = bool(numerical_rank <= 8 and numerical_rank < down.shape[1])
    predictions = {"pred_a_authority_exact_factor_weight_closure_finiteness_and_price": pred_a,
        "pred_b_rank8_weight_modes_preserve_the_postcue_writer": pred_b,
        "pred_c_rank8_complement_is_secondary": pred_c,
        "pred_d_left_right_terms_explain_the_weight_mode_write": pred_d,
        "pred_e_weight_interface_is_compressive_and_zero_fit": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": RESULT_SCHEMA,
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "weight_interface": {"hidden_width": down.shape[1],
            "numerical_rank": numerical_rank, "singular_values": [float(x) for x in singular]},
        "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "base_self_patch_resid18_max_abs": self_error,
            "hidden_factor_reconstruction_max_abs": factor_error,
            "svd_projector_max_abs": projector_error, "svd_reconstruction_max_abs": svd_error,
            "direct_q8_write_closure_max_abs": direct_q8_error, "rows": len(rows)},
        "metrics": metrics, "ratios": ratios, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "weight_interface", "instrument",
        "metrics", "ratios", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
