#!/usr/bin/env python3
"""Augment the frozen H3 rank-two core with exact downstream weight-read modes."""

# BQGATE: EXPERIMENT pred_a_authority_exact_instrument_and_price pred_b_weight_modes_add_static_reader_energy pred_c_weight_rank4_improves_fresh_behavior pred_d_weight_rank4_complement_is_selective pred_e_weight_rank4_improves_downstream_transport
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
import circuit_candidate_temporal_auxiliary_fresh_cues_v8 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset
import run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1 as instrument

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_rank2_plus_weight_read_rank4_v1.json"
RANK2_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1_result.json"
RESPONSE_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_response_svd_rank4_v2_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v8_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v8.py"
INSTRUMENT = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank2_plus_weight_read_rank4_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_rank2_plus_weight_read_rank4_v1"
EXPECTED = {
    "prior": "204a4cec2f4259653933fc4f05ffcba17804ec3b3b28cf8ba14985f4662c9c40",
    "rank2_result": "8a2bfe5ba7ab5626db132509e637fd5df97bd306c7788c03bd180eb7628d8562",
    "response_result": "f7b7a2781a93cb5858fa33f59604c94ca3915a74c55106708b9b90fbe64c4221",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "capability": "fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff",
    "builder": "13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf",
    "instrument": "936a7920b164e57473f7b7204352584b5a438dd0745afc27bfe7f0dd80354a66",
}
ARMS = ("base_identity", "writer_live", "h3_full", "h3_frozen_rank2",
        "h3_weight_rank4", "h3_weight_rank4_orthogonal")
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 24, 1200, 360


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def build_weight_basis(backend, subspace):
    torch, model = backend.torch, backend.model
    q2 = torch.linalg.qr(torch.tensor(
        subspace["axis_artifacts"]["two_task_dim_union_rank2"],
        device=backend.device).float(), mode="reduced").Q
    width = int(model.transformer.h[11].attn.head_dim)
    output = model.transformer.h[11].attn.c_proj.weight.detach().float()[:, 3 * width:4 * width]
    maps = []
    for head_index in (5, 1):
        attention = model.transformer.h[15].attn
        for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v"):
            section = getattr(attention, name).weight.detach().float()[
                head_index * width:(head_index + 1) * width]
            maps.append((section / torch.linalg.matrix_norm(section)) @ output)
    reader = torch.cat(maps, dim=0)
    identity = torch.eye(width, device=backend.device)
    perpendicular = reader @ (identity - q2 @ q2.T)
    _u, singular, vh = torch.linalg.svd(perpendicular, full_matrices=False)
    residual_modes = vh[:2].T
    q4 = torch.linalg.qr(torch.cat((q2, residual_modes), dim=1), mode="reduced").Q

    def energy_fraction(q):
        return float(torch.linalg.matrix_norm(reader @ q @ q.T).square()
                     / torch.linalg.matrix_norm(reader).square())

    return q2, q4, singular, energy_fraction(q2), energy_fraction(q4)


def main():
    paths = {"prior": PRIOR, "rank2_result": RANK2_RESULT,
             "response_result": RESPONSE_RESULT, "subspace": SUBSPACE,
             "capability": CAPABILITY, "builder": BUILDER, "instrument": INSTRUMENT}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("weight-rank4 authority changed")
    prior, rank2_result, response_result, subspace, capability = [
        json.loads(path.read_text()) for path in
        (PRIOR, RANK2_RESULT, RESPONSE_RESULT, SUBSPACE, CAPABILITY)]
    if (prior.get("candidate_id") != CANDIDATE_ID
            or rank2_result.get("terminal") != "representation_only"
            or response_result.get("terminal") != "wrong_object"
            or capability.get("terminal") != "manifest"):
        raise RuntimeError("authority terminal changed")
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "rank": 4, "basis_fit_examples": 0, "basis_fit_labels": 0,
        "reader_heads": ["L15H5", "L15H1"], "reader_factors": ["q", "k", "q2", "k2", "v"],
        "arms": list(ARMS), "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS, "records": RECORDS,
        "fit_updates": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    q2, q4, singular, energy2, energy4 = build_weight_basis(backend, subspace)
    torch = backend.torch
    q2_error = float((q2.T @ q2 - torch.eye(2, device=q2.device)).abs().max())
    q4_error = float((q4.T @ q4 - torch.eye(4, device=q4.device)).abs().max())
    core_error = float((q4 @ q4.T @ q2 - q2).abs().max())
    records, downstream, forwards, evaluations = [], {}, 0, 0
    reconstruction = identity_error = algebra = 0.0
    for panel in ("A1", "A2"):
        panel_rows = [row for row in rows if row["transform_id"] == panel]
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
        identity_error = max(identity_error, instrument.pair_error(base_output, base11_output),
            instrument.pair_error(base_output, base15_output),
            instrument.pair_error(writer_output, writer15_output))
        reconstruction = max(reconstruction, *(float(x["reconstruction_max_abs"])
            for x in (base8, donor8, base11, writer11, base15, writer15)))
        outputs = {"base_identity": base_output, "writer_live": writer_output}
        captures = {"base_identity": base15, "writer_live": writer15}
        for arm, q, mode in (("h3_full", q4, "full"),
                             ("h3_frozen_rank2", q2, "rank2"),
                             ("h3_weight_rank4", q4, "rank2"),
                             ("h3_weight_rank4_orthogonal", q4, "orthogonal")):
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
            ("h3_full", "h3_frozen_rank2", "h3_weight_rank4", "h3_weight_rank4_orthogonal")}

    summaries = {panel: {arm: scoring.summarize([r for r in records
        if r["panel"] == panel and r["arm"] == arm]) for arm in ARMS} for panel in ("A1", "A2")}
    measured = ("h3_frozen_rank2", "h3_weight_rank4", "h3_weight_rank4_orthogonal")
    behavior_fraction = {panel: {arm: summaries[panel][arm]["mean_recovery"]
        / summaries[panel]["h3_full"]["mean_recovery"] for arm in measured}
        for panel in ("A1", "A2")}
    downstream_means = {panel: {arm: sum(values)/len(values)
        for arm, values in downstream[panel].items()} for panel in ("A1", "A2")}
    downstream_fraction = {panel: {arm: downstream_means[panel][arm]
        / downstream_means[panel]["h3_full"] for arm in measured}
        for panel in ("A1", "A2")}
    pred_a = bool(q2_error <= 1e-5 and q4_error <= 1e-5 and core_error <= 1e-5
        and reconstruction <= 5e-4 and identity_error <= 1e-4 and algebra <= 1e-6
        and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS and len(records) == RECORDS
        and all(math.isfinite(r["recovery"]) for r in records))
    pred_b = energy4 - energy2 >= 0.10
    pred_c = all(behavior_fraction[p]["h3_weight_rank4"] >= 0.85
        and behavior_fraction[p]["h3_weight_rank4"]
            - behavior_fraction[p]["h3_frozen_rank2"] >= 0.08 for p in ("A1", "A2"))
    pred_d = all(abs(behavior_fraction[p]["h3_weight_rank4_orthogonal"]) <= 0.18
                 for p in ("A1", "A2"))
    pred_e = all(downstream_fraction[p]["h3_weight_rank4"] >= 0.85
        and downstream_fraction[p]["h3_weight_rank4"]
            - downstream_fraction[p]["h3_frozen_rank2"] >= 0.08 for p in ("A1", "A2"))
    predictions = {
        "pred_a_authority_exact_instrument_and_price": pred_a,
        "pred_b_weight_modes_add_static_reader_energy": pred_b,
        "pred_c_weight_rank4_improves_fresh_behavior": pred_c,
        "pred_d_weight_rank4_complement_is_selective": pred_d,
        "pred_e_weight_rank4_improves_downstream_transport": pred_e,
    }
    terminal = ("invalid" if not pred_a or not pred_b else "identification"
        if all(predictions.values()) else "static_metric_incomplete")
    result = {"schema": "temporal_auxiliary_h3_rank2_plus_weight_read_rank4_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "weight_basis": {
            "rank2_static_reader_energy_fraction": energy2,
            "rank4_static_reader_energy_fraction": energy4,
            "increment": energy4 - energy2,
            "orthogonal_reader_singular_values": [float(x) for x in singular[:8]],
            "basis": q4.detach().cpu().tolist()},
        "instrument": {"rank2_orthonormality_max_abs": q2_error,
            "rank4_orthonormality_max_abs": q4_error, "core_containment_max_abs": core_error,
            "attention_reconstruction_max_abs": reconstruction,
            "identity_max_abs": identity_error, "projection_closure_max_abs": algebra},
        "summaries": summaries, "behavior_fraction_of_full_h3": behavior_fraction,
        "l15_h5_h1_response_norm_means": downstream_means,
        "downstream_fraction_of_full_h3": downstream_fraction,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fit_updates": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("candidate_id", "weight_basis", "instrument",
        "behavior_fraction_of_full_h3", "downstream_fraction_of_full_h3", "predictions",
        "terminal", "price") if k != "weight_basis"}, sort_keys=True))


if __name__ == "__main__":
    main()
