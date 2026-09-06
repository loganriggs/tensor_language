#!/usr/bin/env python3
"""Fit a causal-response H3 rank-four basis on v1/v2 and actuate it on v8."""

# BQGATE: EXPERIMENT pred_a_authority_seal_exact_instrument_and_price pred_b_rank4_captures_training_response_object pred_c_rank4_is_fresh_behaviorally_sufficient pred_d_rank4_orthogonal_complement_is_selective pred_e_rank4_restores_downstream_transport
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as attention_eval
import attention_source_group_eval as scoring
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fit_v1
import circuit_candidate_temporal_auxiliary_fresh_cues_v2 as fit_v2
import circuit_candidate_temporal_auxiliary_fresh_cues_v8 as test_v8
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset
import run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1 as instrument

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_response_svd_rank4_v2.json"
RANK2_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v8_capability_v1_result.json"
V1 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
V2 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v2.py"
V8 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v8.py"
INSTRUMENT = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_response_svd_rank4_v2_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_response_svd_rank4_v2"
EXPECTED = {
    "prior": "0ebf4879e5172b982e42a89c638efcbc9de767a61d4de1f237d02287b61171f5",
    "rank2_result": "8a2bfe5ba7ab5626db132509e637fd5df97bd306c7788c03bd180eb7628d8562",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "capability": "fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff",
    "v1": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "v2": "adbfaf91ed2889cc42da85255edf9f5074f1002e9ad93dc1d4ff706de66d1144",
    "v8": "13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf",
    "instrument": "936a7920b164e57473f7b7204352584b5a438dd0745afc27bfe7f0dd80354a66",
}
ARMS = ("base_identity", "writer_live", "h3_full", "h3_frozen_rank2",
        "h3_response_rank4", "h3_response_rank4_orthogonal")
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 40, 1200, 360


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def capture_response_vectors(backend, rows):
    vectors, forwards, evaluations, reconstruction = [], 0, 0, 0.0
    for panel in ("A1", "A2"):
        panel_rows = [row for row in rows if row["transform_id"] == panel]
        base_batch = das._batch(backend, panel_rows, side="base")
        donor_batch = das._batch(backend, panel_rows, side="donor")
        _base_output, base8 = attention_eval.capture_layer_attention(backend, base_batch, 8)
        _donor_output, donor8 = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        _base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, base8, donor8, destinations, ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            _writer_output, writer11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        finally:
            handle.remove()
        forwards += 4; evaluations += 4 * len(panel_rows)
        reconstruction = max(reconstruction, *(float(x["reconstruction_max_abs"])
            for x in (base8, donor8, base11, writer11)))
        for index, query in enumerate(base_batch.semantic_positions):
            for position in range(int(query) + 1):
                vectors.append((writer11["head_output"][index, position, 3].float()
                                - base11["head_output"][index, position, 3].float()))
    return backend.torch.stack(vectors), forwards, evaluations, reconstruction


def main():
    paths = {"prior": PRIOR, "rank2_result": RANK2_RESULT, "subspace": SUBSPACE,
             "capability": CAPABILITY, "v1": V1, "v2": V2, "v8": V8,
             "instrument": INSTRUMENT}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("response-SVD authority changed")
    prior, rank2_result, subspace, capability = [json.loads(path.read_text()) for path in
        (PRIOR, RANK2_RESULT, SUBSPACE, CAPABILITY)]
    if (prior.get("candidate_id") != CANDIDATE_ID
            or rank2_result.get("terminal") != "representation_only"
            or capability.get("terminal") != "manifest"):
        raise RuntimeError("authority terminal changed")
    fit_banks = [[row for row in builder.build_rows() if row["transform_id"] in {"A1", "A2"}]
                 for builder in (fit_v1, fit_v2)]
    fit_rows = [row for bank in fit_banks for row in bank]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    test_rows = [row for row in test_v8.build_rows()
                 if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "fit_rows": len(fit_rows),
        "test_rows": len(test_rows), "rank": 4, "arms": list(ARMS),
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "records": RECORDS, "fit_updates": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fit_captures = [capture_response_vectors(backend, bank) for bank in fit_banks]
    response = backend.torch.cat([item[0] for item in fit_captures])
    forwards = sum(item[1] for item in fit_captures)
    evaluations = sum(item[2] for item in fit_captures)
    reconstruction = max(item[3] for item in fit_captures)
    _u, singular, vh = backend.torch.linalg.svd(response, full_matrices=False)
    q4 = backend.torch.linalg.qr(vh[:4].T.contiguous(), mode="reduced").Q
    q2 = backend.torch.linalg.qr(backend.torch.tensor(
        subspace["axis_artifacts"]["two_task_dim_union_rank2"], device=backend.device).float(),
        mode="reduced").Q
    training_energy = float(singular[:4].square().sum() / singular.square().sum())
    q4_orth_error = float((q4.T @ q4 - backend.torch.eye(4, device=q4.device)).abs().max())
    q2_orth_error = float((q2.T @ q2 - backend.torch.eye(2, device=q2.device)).abs().max())
    records, downstream = [], {}
    identity = algebra = 0.0
    for panel in ("A1", "A2"):
        panel_rows = [row for row in test_rows if row["transform_id"] == panel]
        base_batch = das._batch(backend, panel_rows, side="base")
        donor_batch = das._batch(backend, panel_rows, side="donor")
        base_output, base8 = attention_eval.capture_layer_attention(backend, base_batch, 8)
        donor_output, donor8 = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, base8, donor8, destinations, ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            writer_output, writer11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        finally:
            handle.remove()
        base15_output, base15 = attention_eval.capture_layer_attention(backend, base_batch, 15)
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            writer15_output, writer15 = attention_eval.capture_layer_attention(backend, base_batch, 15)
        finally:
            handle.remove()
        forwards += 6; evaluations += 6 * len(panel_rows)
        identity = max(identity, instrument.pair_error(base_output, base11_output),
                       instrument.pair_error(base_output, base15_output),
                       instrument.pair_error(writer_output, writer15_output))
        reconstruction = max(reconstruction, *(float(x["reconstruction_max_abs"])
            for x in (base8, donor8, base11, writer11, base15, writer15)))
        outputs = {"base_identity": base_output, "writer_live": writer_output}
        captures = {"base_identity": base15, "writer_live": writer15}
        for arm, q, mode in (("h3_full", q4, "full"),
                             ("h3_frozen_rank2", q2, "rank2"),
                             ("h3_response_rank4", q4, "rank2"),
                             ("h3_response_rank4_orthogonal", q4, "orthogonal")):
            outputs[arm], captures[arm], error = instrument.run_mode(
                backend, base_batch, base11, writer11, q, mode)
            algebra = max(algebra, error)
            reconstruction = max(reconstruction, float(captures[arm]["reconstruction_max_abs"]))
            forwards += 1; evaluations += len(panel_rows)
        for arm in ARMS:
            records.extend(dict(record, panel=panel) for record in scoring.recovery_records(
                panel_rows, base_output, donor_output, outputs[arm], arm=arm))
        downstream[panel] = {arm: instrument.l15_pair_norms(
            backend, captures[arm], base15, base_batch) for arm in
            ("h3_full", "h3_frozen_rank2", "h3_response_rank4", "h3_response_rank4_orthogonal")}

    summaries = {panel: {arm: scoring.summarize([r for r in records
        if r["panel"] == panel and r["arm"] == arm]) for arm in ARMS} for panel in ("A1", "A2")}
    behavior_fraction = {panel: {arm: summaries[panel][arm]["mean_recovery"]
        / summaries[panel]["h3_full"]["mean_recovery"] for arm in
        ("h3_frozen_rank2", "h3_response_rank4", "h3_response_rank4_orthogonal")}
        for panel in ("A1", "A2")}
    downstream_means = {panel: {arm: sum(values)/len(values)
        for arm, values in downstream[panel].items()} for panel in ("A1", "A2")}
    downstream_fraction = {panel: {arm: downstream_means[panel][arm]
        / downstream_means[panel]["h3_full"] for arm in
        ("h3_frozen_rank2", "h3_response_rank4", "h3_response_rank4_orthogonal")}
        for panel in ("A1", "A2")}
    pred_a = bool(q4_orth_error <= 1e-5 and q2_orth_error <= 1e-5
        and reconstruction <= 5e-4 and identity <= 1e-4 and algebra <= 1e-6
        and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS and len(records) == RECORDS
        and all(math.isfinite(r["recovery"]) for r in records))
    pred_b = training_energy >= 0.85
    pred_c = all(behavior_fraction[p]["h3_response_rank4"] >= 0.90
        and behavior_fraction[p]["h3_response_rank4"] - behavior_fraction[p]["h3_frozen_rank2"] >= 0.10
        for p in ("A1", "A2"))
    pred_d = all(abs(behavior_fraction[p]["h3_response_rank4_orthogonal"]) <= 0.15
                 for p in ("A1", "A2"))
    pred_e = all(downstream_fraction[p]["h3_response_rank4"] >= 0.85
        and downstream_fraction[p]["h3_response_rank4"]
            - downstream_fraction[p]["h3_frozen_rank2"] >= 0.10 for p in ("A1", "A2"))
    predictions = {
        "pred_a_authority_seal_exact_instrument_and_price": pred_a,
        "pred_b_rank4_captures_training_response_object": pred_b,
        "pred_c_rank4_is_fresh_behaviorally_sufficient": pred_c,
        "pred_d_rank4_orthogonal_complement_is_selective": pred_d,
        "pred_e_rank4_restores_downstream_transport": pred_e,
    }
    full_material = all(abs(summaries[p]["h3_full"]["mean_recovery"]) >= 0.30
        * abs(summaries[p]["writer_live"]["mean_recovery"]) for p in ("A1", "A2"))
    terminal = ("invalid" if not pred_a else "null" if not full_material else "identification"
        if all(predictions.values()) else "wrong_object" if not pred_b or not all(
            behavior_fraction[p]["h3_response_rank4"] > behavior_fraction[p]["h3_frozen_rank2"]
            for p in ("A1", "A2")) else "insufficient_rank")
    result = {"schema": "temporal_auxiliary_h3_response_svd_rank4_result_v2",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter()-started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "post_invalid_correction": "thin_qr_gauge_only_scientific_design_unchanged",
        "fit": {"response_vectors": int(response.shape[0]), "dimension": int(response.shape[1]),
            "rank": 4, "training_energy_fraction": training_energy,
            "leading_singular_values": [float(x) for x in singular[:12]],
            "basis": q4.detach().cpu().tolist()},
        "instrument": {"rank4_orthonormality_max_abs": q4_orth_error,
            "rank2_orthonormality_max_abs": q2_orth_error,
            "attention_reconstruction_max_abs": reconstruction, "identity_max_abs": identity,
            "projection_closure_max_abs": algebra},
        "summaries": summaries, "behavior_fraction_of_full_h3": behavior_fraction,
        "l15_h5_h1_response_norm_means": downstream_means,
        "downstream_fraction_of_full_h3": downstream_fraction,
        "full_h3_material": full_material, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fit_updates": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("candidate_id", "fit", "instrument",
        "behavior_fraction_of_full_h3", "downstream_fraction_of_full_h3", "full_h3_material",
        "predictions", "terminal", "price") if k != "fit"}, sort_keys=True))


if __name__ == "__main__":
    main()
