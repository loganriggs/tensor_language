#!/usr/bin/env python3
"""Confirm the v8-selected checkpoint-only H3 rank-seven basis on sealed v9."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_instrument_and_price pred_b_full_h3_is_material pred_c_rank7_is_behaviorally_sufficient pred_d_rank7_complement_is_selective pred_e_rank7_transports_to_l15_pair
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
import circuit_candidate_temporal_auxiliary_fresh_cues_v9 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset
import run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1 as instrument
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_weight_rank7_v9_confirmation_v1.json"
SCREEN = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v9_capability_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v9.py"
FAMILY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1.py"
INSTRUMENT = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_weight_rank7_v9_confirmation_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_weight_rank7_v9_confirmation_v1"
EXPECTED = {
    "prior": "4e9e050913e1a59cbd99b3b8f8e4b872378763defb36e7be8b0bb18693a05df5",
    "screen": "adc318db1b08fd47c034cf4cd15b7234b16582b7ab134275f8c36265219254fc",
    "capability": "828d8b15d9bcf048de32d74384e2f4bc62972f20289515f8eb6c576302262392",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "builder": "9b771713c5803082c95a3566bc41120587e60f99c4c8bacc291602516bbe01a5",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "instrument": "936a7920b164e57473f7b7204352584b5a438dd0745afc27bfe7f0dd80354a66",
}
ARMS = ("base_identity", "writer_live", "h3_full", "h3_weight_rank7",
        "h3_weight_rank7_orthogonal")
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 20, 800, 320


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "screen": SCREEN, "capability": CAPABILITY,
             "subspace": SUBSPACE, "builder": BUILDER,
             "family_runner": FAMILY_RUNNER, "instrument": INSTRUMENT}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("rank7 confirmation authority changed")
    prior, screen, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, SCREEN, CAPABILITY, SUBSPACE)]
    if (prior.get("candidate_id") != CANDIDATE_ID or screen.get("selected_rank") != 7
            or screen.get("terminal") != "screen" or capability.get("terminal") != "manifest"):
        raise RuntimeError("authority terminal changed")
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rank": 7, "rows": len(rows),
        "basis_fit_examples": 0, "basis_fit_labels": 0, "arms": list(ARMS),
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "records": RECORDS, "fit_updates": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    family, singular, energy = family_builder.build_family(backend, subspace)
    q = family[7]
    torch = backend.torch
    orth_error = float((q.T @ q - torch.eye(7, device=q.device)).abs().max())
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
        forwards += 6; evaluations += 6*len(panel_rows)
        identity_error = max(identity_error, instrument.pair_error(base_output, base11_output),
            instrument.pair_error(base_output, base15_output), instrument.pair_error(writer_output, writer15_output))
        reconstruction = max(reconstruction, *(float(x["reconstruction_max_abs"])
            for x in (base8, donor8, base11, writer11, base15, writer15)))
        outputs = {"base_identity": base_output, "writer_live": writer_output}
        captures = {"base_identity": base15, "writer_live": writer15}
        for arm, mode in (("h3_full", "full"), ("h3_weight_rank7", "rank2"),
                          ("h3_weight_rank7_orthogonal", "orthogonal")):
            outputs[arm], captures[arm], error = instrument.run_mode(
                backend, base_batch, base11, writer11, q, mode)
            algebra = max(algebra, error); forwards += 1; evaluations += len(panel_rows)
            reconstruction = max(reconstruction, float(captures[arm]["reconstruction_max_abs"]))
        for arm in ARMS:
            records.extend(dict(record, panel=panel) for record in scoring.recovery_records(
                panel_rows, base_output, donor_output, outputs[arm], arm=arm))
        downstream[panel] = {arm: instrument.l15_pair_norms(backend, captures[arm], base15, base_batch)
            for arm in ("h3_full", "h3_weight_rank7", "h3_weight_rank7_orthogonal")}
    summaries = {panel: {arm: scoring.summarize([r for r in records
        if r["panel"] == panel and r["arm"] == arm]) for arm in ARMS} for panel in ("A1", "A2")}
    behavior = {panel: {arm: summaries[panel][arm]["mean_recovery"]
        / summaries[panel]["h3_full"]["mean_recovery"]
        for arm in ("h3_weight_rank7", "h3_weight_rank7_orthogonal")}
        for panel in ("A1", "A2")}
    writer_fraction = {panel: summaries[panel]["h3_full"]["mean_recovery"]
        / summaries[panel]["writer_live"]["mean_recovery"] for panel in ("A1", "A2")}
    downstream_means = {panel: {arm: sum(values)/len(values) for arm, values in by_arm.items()}
                        for panel, by_arm in downstream.items()}
    transport = {panel: downstream_means[panel]["h3_weight_rank7"]
        / downstream_means[panel]["h3_full"] for panel in ("A1", "A2")}
    pred_a = bool(orth_error <= 1e-5 and reconstruction <= 5e-4 and identity_error <= 1e-4
        and algebra <= 1e-6 and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS
        and len(records) == RECORDS and all(math.isfinite(r["recovery"]) for r in records))
    pred_b = all(summaries[p]["h3_full"]["direction_fraction"] >= .90
                 and writer_fraction[p] >= .30 for p in ("A1", "A2"))
    pred_c = all(behavior[p]["h3_weight_rank7"] >= .90 for p in ("A1", "A2"))
    pred_d = all(abs(behavior[p]["h3_weight_rank7_orthogonal"]) <= .12 for p in ("A1", "A2"))
    pred_e = all(transport[p] >= .90 for p in ("A1", "A2"))
    predictions = {"pred_a_authority_capability_exact_instrument_and_price": pred_a,
        "pred_b_full_h3_is_material": pred_b, "pred_c_rank7_is_behaviorally_sufficient": pred_c,
        "pred_d_rank7_complement_is_selective": pred_d, "pred_e_rank7_transports_to_l15_pair": pred_e}
    terminal = "invalid" if not pred_a else "paired_confirmation" if all(predictions.values()) else "transfer_failure"
    result = {"schema": "temporal_auxiliary_h3_weight_rank7_v9_confirmation_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter()-started,
        "authority_sha256": EXPECTED, "dryrun": dryrun, "basis": {"rank": 7,
            "sha256": tensor_sha(q), "static_reader_energy_fraction": energy[7],
            "leading_orthogonal_reader_singular_values": [float(x) for x in singular[:8]]},
        "instrument": {"orthonormality_max_abs": orth_error,
            "attention_reconstruction_max_abs": reconstruction, "identity_max_abs": identity_error,
            "projection_closure_max_abs": algebra}, "summaries": summaries,
        "h3_full_fraction_of_writer": writer_fraction, "behavior_fraction_of_full_h3": behavior,
        "l15_h5_h1_response_norm_means": downstream_means,
        "rank7_fraction_of_full_l15_h5_h1_response": transport,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fit_updates": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("candidate_id", "basis", "instrument",
        "h3_full_fraction_of_writer", "behavior_fraction_of_full_h3",
        "rank7_fraction_of_full_l15_h5_h1_response", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
