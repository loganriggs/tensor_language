#!/usr/bin/env python3
"""Finite shared-state causal matrix for temporal and is/was commands/readouts."""

# BQGATE: EXPERIMENT pred_a_exact_authority_state_construction_coverage_and_price pred_b_one_q8_realization_predicts_all_four_quadrants pred_c_finite_matrix_is_effectively_eight_state_but_not_scalar pred_d_cross_task_quadrants_are_observable pred_e_coordinate_pairing_beats_shuffled_control
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as attention_eval
import circuit_candidate_aspectual_different_readout_is_was_v2 as v2
import circuit_candidate_aspectual_different_readout_is_was_v3 as v3
import circuit_candidate_temporal_auxiliary_fresh_cues_v11 as temporal
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
import residual_source_onset_eval as onset
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_auxiliary_will_had_h3_rank8_analytic_final_reader_v1 as analytic
import run_temporal_auxiliary_will_had_h3_rank8_v10_source_factor_atlas_v1 as atlas
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_auxiliary_will_had_l9_triple_to_h3_rank8_weight_writer_v1 as upstream
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_q8_finite_causal_hankel_v1.json"
SHARED_CAUSAL = ROOT / "circuits/followups/temporal_q8_iswas_cdas_shared_specific_causal_v2_result.json"
TEMPORAL_CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v11_capability_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
ISWAS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
V2_CAPABILITY = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v2_capability_result.json"
V3_CAPABILITY = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v3_capability_result.json"
TEMPORAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v11.py"
V2_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v2.py"
V3_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v3.py"
ATLAS_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank8_v10_source_factor_atlas_v1.py"
ANALYTIC_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank8_analytic_final_reader_v1.py"
OVERLAP_RUNNER = ROOT / "ops/run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_q8_finite_causal_hankel_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas_q8_finite_causal_hankel_v1"
EXPECTED = {
    "prior": "2fe29d94b375f1cf9c0c575d1faf5ebd7b45b30e383011b7996f93a71ab05af7",
    "shared_causal": "bd302cb0d104db5afe43906885dff52f851a03e638c6ff30de9d87224ce235bc",
    "temporal_capability": "0330dc5a4f85bc68c4da6f98af2f4208335e65c644ddedd5d8cc487368091026",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "iswas": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "v2_capability": "f76fbcd6174cc9e8e3f77352ee5461815156cdae421ecc43a0e0c3576b63af7e",
    "v3_capability": "744d2fd3c8200ca00005357961df3d435a7a13dbdbd7c3a51a487daee76acec3",
    "temporal_builder": "f75b17669a5fc5299d21f5b44e91530c03c71d75181683c7b6728cb95c862450",
    "v2_builder": "4ed62d06e2ffe5c471efc70eeaa35c7524a66923f19c64c803faef7200d4a62f",
    "v3_builder": "ac240f13478168948814f59c0425270dd345ba03053332b72b34857e9a022638",
    "atlas_runner": "211f847b8e0799a5ee9b889f64183bdc7e67df0862463748c3443f3417efcfda",
    "analytic_runner": "50df4d2393606957516bcc8752cf4f575713ed8f6fc843c3864f585565a7947d",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1",
}
TASKS = ("temporal", "iswas")
MAX_FORWARDS, MAX_EVALUATIONS, CELLS, SEED = 10, 96, 1024, 20260906


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def cosine(x, y):
    denominator = float(x.norm()*y.norm())
    return float((x*y).sum())/denominator if denominator else float("nan")


def selected_margin(backend, states, answer_ids, foil_ids):
    torch = backend.torch
    normalized = backend.F.rms_norm(states, (states.shape[-1],))
    weights = backend.model.lm_head.weight.float()
    answer = weights[answer_ids]
    foil = weights[foil_ids]
    answer_pre = (normalized*answer).sum(-1)
    foil_pre = (normalized*foil).sum(-1)
    return 30.0*torch.tanh(answer_pre/30.0)-30.0*torch.tanh(foil_pre/30.0)


def select_iswas_rows():
    by2 = {family: [row for row in v2.build_rows() if row["family"] == family]
           for family in ("A1", "A2")}
    by3 = {family: [row for row in v3.build_rows() if row["family"] == family]
           for family in ("A1", "A2")}
    return by2["A1"][:4]+by2["A2"][:4]+by3["A1"][:4]+by3["A2"][:4]


def main():
    paths = {"prior": PRIOR, "shared_causal": SHARED_CAUSAL,
        "temporal_capability": TEMPORAL_CAPABILITY, "subspace": SUBSPACE, "iswas": ISWAS,
        "v2_capability": V2_CAPABILITY, "v3_capability": V3_CAPABILITY,
        "temporal_builder": TEMPORAL_BUILDER, "v2_builder": V2_BUILDER,
        "v3_builder": V3_BUILDER, "atlas_runner": ATLAS_RUNNER,
        "analytic_runner": ANALYTIC_RUNNER, "overlap_runner": OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("finite causal Hankel authority changed")
    prior, shared, temporal_cap, subspace, iswas, cap2, cap3 = [json.loads(path.read_text())
        for path in (PRIOR, SHARED_CAUSAL, TEMPORAL_CAPABILITY, SUBSPACE, ISWAS,
                     V2_CAPABILITY, V3_CAPABILITY)]
    if (prior.get("candidate_id") != CANDIDATE_ID or shared.get("terminal") != "screen"
            or temporal_cap.get("terminal") != "manifest" or iswas.get("terminal") != "screen"
            or any(cap.get("terminal") != "screen" for cap in (cap2, cap3))):
        raise RuntimeError("authority terminal changed")
    allowed = {row_id for ids in temporal_cap["jointly_capable_row_ids"].values() for row_id in ids}
    temporal_rows = []
    all_temporal = temporal.build_rows()
    for panel in ("A1", "A2"):
        temporal_rows.extend([row for row in all_temporal
            if row["transform_id"] == panel and row["row_id"] in allowed][:8])
    iswas_rows = select_iswas_rows()
    if len(temporal_rows) != 16 or len(iswas_rows) != 16:
        raise RuntimeError("frozen row population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "temporal_commands": 16,
        "iswas_commands": 16, "downstream_contexts": 32, "matrix_cells": CELLS,
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
    raw_modes, orientation_error, _wrong = overlap.residual_modes(backend, q, gain)
    s = torch.linalg.qr(raw_modes, mode="reduced").Q
    orth_error = float((s.T@s-torch.eye(8, device=s.device)).abs().max())
    writes, contexts, answers, foils, metadata = [], [], [], [], []
    reconstruction = capture_identity = q8_closure = 0.0
    forwards = evaluations = 0
    for panel in ("A1", "A2"):
        rows = [row for row in temporal_rows if row["transform_id"] == panel]
        base_batch = das._batch(backend, rows, side="base")
        donor_batch = das._batch(backend, rows, side="donor")
        base_output, base8 = attention_eval.capture_layer_attention(
            backend, base_batch, 8, call=lambda: backend.native(base_batch, capture=True))
        _donor_output, donor8 = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        base11_output, base_h3 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, base8, donor8, destinations, ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            _writer_output, writer_h3 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        finally:
            handle.remove()
        forwards += 4; evaluations += 4*len(rows)
        reconstruction = max(reconstruction, *(float(item["reconstruction_max_abs"])
            for item in (base8, donor8, base_h3, writer_h3)))
        capture_identity = max(capture_identity, upstream.pair_error(base_output, base11_output))
        for index, (row, query) in enumerate(zip(rows, base_batch.semantic_positions)):
            query = int(query)
            groups = atlas.source_partition(base_batch.token_rows[index], donor_batch.token_rows[index],
                                            query, destinations[index])
            exact = (writer_h3["head_output"][index, query, 3].float()
                     - base_h3["head_output"][index, query, 3].float())@q
            complete = exact.new_zeros(8); suffix = exact.new_zeros(8)
            for group in atlas.GROUPS:
                for factor_name in atlas.FACTORS:
                    coordinate = atlas.factor_head(base_h3, writer_h3, index, query,
                                                   groups[group], factor_name)@q
                    complete += coordinate
                    if group in ("subject_onset", "post_subject", "self"):
                        suffix += coordinate
            q8_closure = max(q8_closure, float((exact-complete).abs().max()))
            writes.append(raw_modes@suffix)
            contexts.append(torch.as_tensor(base_output.captured[(row["row_id"], "resid:18")],
                                            device=backend.device).float())
            answers.append(int(row["donor_answer_id"])); foils.append(int(row["donor_foil_id"]))
            metadata.append({"task": "temporal", "row_id": row["row_id"],
                             "family": panel, "direction": row["direction_id"]})
    base_batch = das._batch(backend, iswas_rows, side="base")
    donor_batch = das._batch(backend, iswas_rows, side="donor")
    base_output = backend.native(base_batch, capture=True)
    donor_output = backend.native(donor_batch, capture=True)
    forwards += 2; evaluations += 2*len(iswas_rows)
    base18 = torch.stack([torch.as_tensor(base_output.captured[(row["row_id"], "resid:18")])
                          for row in iswas_rows]).to(backend.device).float()
    donor18 = torch.stack([torch.as_tensor(donor_output.captured[(row["row_id"], "resid:18")])
                           for row in iswas_rows]).to(backend.device).float()
    axis_values = np.asarray(iswas["basis"]["values_column_major"], dtype=np.float32)
    if hashlib.sha256(axis_values.tobytes()).hexdigest() != iswas["basis"]["sha256"]:
        raise RuntimeError("is/was axis changed")
    axis = torch.as_tensor(axis_values, device=backend.device).reshape(1152, 1)
    axis = axis/torch.linalg.vector_norm(axis)
    shared_axis = s@(s.T@axis)
    iswas_writes = ((donor18-base18)@axis)@shared_axis.T
    for index, row in enumerate(iswas_rows):
        writes.append(iswas_writes[index]); contexts.append(base18[index])
        answers.append(int(row["donor_answer_id"])); foils.append(int(row["donor_foil_id"]))
        metadata.append({"task": "iswas", "row_id": row["row_id"],
                         "family": row["family"], "direction": row["direction_id"]})
    writes = torch.stack(writes); contexts = torch.stack(contexts)
    answer_ids = torch.as_tensor(answers, device=backend.device, dtype=torch.long)
    foil_ids = torch.as_tensor(foils, device=backend.device, dtype=torch.long)
    coordinates = writes@s
    coordinate_replay = float((writes-coordinates@s.T).abs().max())
    readers = torch.stack([analytic.analytic_reader(
        backend, contexts[index], answers[index], foils[index], s.T)[0] for index in range(32)])
    predicted = coordinates@readers.T
    base_margin = selected_margin(backend, contexts, answer_ids, foil_ids)
    full_margin = das.head_logits(backend, contexts)
    index = torch.arange(32, device=backend.device)
    selected_head_error = float((base_margin-(full_margin[index, answer_ids]
        - full_margin[index, foil_ids])).abs().max())
    exact_rows = []
    for write in writes:
        exact_rows.append(selected_margin(backend, contexts+write, answer_ids, foil_ids)-base_margin)
    exact = torch.stack(exact_rows)
    quadrants = {}
    task_slices = {"temporal": slice(0, 16), "iswas": slice(16, 32)}
    for source_task in TASKS:
        for target_task in TASKS:
            block_exact = exact[task_slices[source_task], task_slices[target_task]].reshape(-1)
            block_pred = predicted[task_slices[source_task], task_slices[target_task]].reshape(-1)
            rmse = float(torch.sqrt(((block_exact-block_pred)**2).mean()))
            rms = float(torch.sqrt((block_exact**2).mean()))
            quadrants[f"{source_task}_to_{target_task}"] = {"cosine": cosine(block_exact, block_pred),
                "rmse": rmse, "exact_rms": rms, "relative_rmse": rmse/rms}
    singular_exact = torch.linalg.svdvals(exact)
    singular_pred = torch.linalg.svdvals(predicted)
    singular_commands = torch.linalg.svdvals(coordinates)
    singular_readers = torch.linalg.svdvals(readers)
    rank8_energy = float((singular_exact[:8]**2).sum()/(singular_exact**2).sum())
    predicted_rank = sum(float(value/singular_pred[0]) > 1e-5 for value in singular_pred)
    command_rank = sum(float(value/singular_commands[0]) > 1e-3 for value in singular_commands)
    reader_rank = sum(float(value/singular_readers[0]) > 1e-3 for value in singular_readers)
    within_rms = [quadrants["temporal_to_temporal"]["exact_rms"],
                  quadrants["iswas_to_iswas"]["exact_rms"]]
    cross_floor = .10*math.sqrt(within_rms[0]*within_rms[1])
    generator = np.random.default_rng(SEED)
    permutation = torch.as_tensor(generator.permutation(32), device=backend.device)
    shuffled = coordinates[permutation]@readers.T
    global_cosine = cosine(exact.reshape(-1), predicted.reshape(-1))
    shuffled_cosine = cosine(exact.reshape(-1), shuffled.reshape(-1))
    records = [{"source_index": source, "target_index": target,
        "source_task": metadata[source]["task"], "target_task": metadata[target]["task"],
        "exact_effect": float(exact[source, target]), "predicted_effect": float(predicted[source, target])}
        for source in range(32) for target in range(32)]
    pred_a = bool(orientation_error <= 1e-6 and orth_error <= 1e-5
        and reconstruction <= 5e-4 and capture_identity <= 1e-4 and q8_closure <= 5e-4
        and coordinate_replay <= 1e-3 and selected_head_error <= 1e-5
        and exact.shape == (32, 32) and predicted.shape == (32, 32)
        and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS and len(records) == CELLS
        and torch.isfinite(exact).all() and torch.isfinite(predicted).all())
    pred_b = all(row["cosine"] >= .90 and row["relative_rmse"] <= .30 for row in quadrants.values())
    pred_c = bool(rank8_energy >= .90 and predicted_rank <= 8 and command_rank >= 4 and reader_rank >= 4)
    pred_d = all(quadrants[name]["exact_rms"] >= cross_floor
        for name in ("temporal_to_iswas", "iswas_to_temporal"))
    pred_e = global_cosine-shuffled_cosine >= .20
    predictions = {"pred_a_exact_authority_state_construction_coverage_and_price": pred_a,
        "pred_b_one_q8_realization_predicts_all_four_quadrants": pred_b,
        "pred_c_finite_matrix_is_effectively_eight_state_but_not_scalar": pred_c,
        "pred_d_cross_task_quadrants_are_observable": pred_d,
        "pred_e_coordinate_pairing_beats_shuffled_control": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "temporal_iswas_q8_finite_causal_hankel_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"attention_reconstruction_max_abs": reconstruction,
            "capture_identity_max_abs": capture_identity, "q8_factor_closure_max_abs": q8_closure,
            "f_linear_orientation_max_abs": orientation_error, "state_orthonormality_max_abs": orth_error,
            "coordinate_write_replay_max_abs": coordinate_replay,
            "selected_margin_vs_full_head_max_abs": selected_head_error},
        "row_metadata": metadata, "quadrants": quadrants,
        "rank_analysis": {"rank8_exact_energy": rank8_energy, "predicted_numerical_rank": predicted_rank,
            "command_effective_rank": command_rank, "reader_effective_rank": reader_rank,
            "exact_singular_values": [float(value) for value in singular_exact],
            "predicted_singular_values": [float(value) for value in singular_pred],
            "command_singular_values": [float(value) for value in singular_commands],
            "reader_singular_values": [float(value) for value in singular_readers]},
        "cross_observability": {"cross_rms_floor": cross_floor},
        "shuffle_control": {"seed": SEED, "global_cosine": global_cosine,
            "shuffled_cosine": shuffled_cosine, "cosine_advantage": global_cosine-shuffled_cosine},
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "matrix_cells": len(records), "fit_updates": 0, "transformer_backwards": 0,
            "model_updates": 0}, "records": records}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "quadrants",
        "rank_analysis", "cross_observability", "shuffle_control", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__":
    main()
