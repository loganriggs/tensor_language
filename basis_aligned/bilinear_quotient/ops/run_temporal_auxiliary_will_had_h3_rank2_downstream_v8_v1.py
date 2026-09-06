#!/usr/bin/env python3
"""Actuate the frozen H3 rank-two union and measure its L15H5/H1 transport."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_instrument_and_price pred_b_full_h3_response_is_material pred_c_rank2_is_behaviorally_sufficient pred_d_orthogonal_complement_is_secondary pred_e_rank2_transports_to_weight_predicted_l15_pair
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

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v8_capability_v1_result.json"
CORE = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_triple_v8_necessity_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
WEIGHTS = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_multicue_weight_interface_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v8.py"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
ONSET = ROOT / "ops/residual_source_onset_eval.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_rank2_downstream_v8_v1"
EXPECTED = {
    "prior": "6569107776cbf794e1c8cc2edc0a2ae06d62b406374320918b12e0a6b598d379",
    "capability": "fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff",
    "core": "dd64838525519a3addd679d1dbbcd6c7e18465b16a1faf06bea23940ef6d51f9",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "weights": "5bf804ee1e61f918edceb0dc9e31ac68fa157de384a943391c4ca4eeb672246a",
    "builder": "13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf",
    "attention": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
    "mediation": "05d07578129db0cb8bf5e46d08e6cd6e9c2bb9c3883675586c83c3efd24b8fda",
    "onset": "c276450cc9ec7c2b0a05e2be0e88bac3df9af7003e370b99b66552083c4f4b45",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
ARMS = ("base_identity", "writer_live", "h3_full", "h3_rank2", "h3_orthogonal")
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 20, 1200, 300


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def pair_error(first, second):
    return max(abs(float(a) - float(b)) for left, right in zip(first.answer_foil, second.answer_foil)
               for a, b in zip(left, right))


def run_mode(backend, batch, base11, writer11, q, mode):
    head_count = backend.model.config.n_head
    head_dim = backend.model.config.n_embd // head_count
    algebra_error = 0.0

    def patch(_module, arguments):
        nonlocal algebra_error
        flattened = arguments[0]
        changed = flattened.clone().view(len(batch.row_ids), flattened.shape[1], head_count, head_dim)
        for index, query in enumerate(batch.semantic_positions):
            for position in range(int(query) + 1):
                delta = (writer11["head_output"][index, position, 3].float()
                         - base11["head_output"][index, position, 3].float())
                projected = (delta @ q) @ q.T
                orthogonal = delta - projected
                component = delta if mode == "full" else projected if mode == "rank2" else orthogonal
                algebra_error = max(algebra_error, float((delta - projected - orthogonal).abs().max()))
                changed[index, position, 3] += component.to(changed)
        return (changed.reshape_as(flattened),) + tuple(arguments[1:])

    handle = backend.model.transformer.h[11].attn.c_proj.register_forward_pre_hook(patch)
    try:
        output, capture15 = attention_eval.capture_layer_attention(backend, batch, 15)
    finally:
        handle.remove()
    return output, capture15, algebra_error


def l15_pair_norms(backend, capture, base, batch):
    values = []
    for index, query in enumerate(batch.semantic_positions):
        pieces = [(capture["head_output"][index, query, head].float()
                   - base["head_output"][index, query, head].float()) for head in (5, 1)]
        values.append(float(backend.torch.linalg.vector_norm(backend.torch.cat(pieces))))
    return values


def main():
    paths = {"prior": PRIOR, "capability": CAPABILITY, "core": CORE,
        "subspace": SUBSPACE, "weights": WEIGHTS, "builder": BUILDER,
        "attention": ATTENTION, "mediation": MEDIATION, "onset": ONSET, "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("rank-two downstream authority changed")
    prior, capability, core, subspace, weights = [json.loads(path.read_text()) for path in
        (PRIOR, CAPABILITY, CORE, SUBSPACE, WEIGHTS)]
    if (prior.get("candidate_id") != CANDIDATE_ID or capability.get("terminal") != "manifest"
            or core.get("terminal") != "paired_confirmation" or weights.get("terminal") != "partial"):
        raise RuntimeError("authority terminal changed")
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "arms": list(ARMS),
        "rows": len(rows), "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS, "records": RECORDS,
        "fit_updates": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    q = backend.torch.linalg.qr(backend.torch.tensor(
        subspace["axis_artifacts"]["two_task_dim_union_rank2"], device=backend.device).float(),
        mode="reduced").Q
    orth_error = float((q.T @ q - backend.torch.eye(2, device=q.device)).abs().max())
    records, downstream, forwards, evaluations = [], {}, 0, 0
    reconstruction = identity = algebra = 0.0
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
        identity = max(identity, pair_error(base_output, base11_output), pair_error(base_output, base15_output),
                       pair_error(writer_output, writer15_output))
        reconstruction = max(reconstruction, *(float(x["reconstruction_max_abs"])
            for x in (base8, donor8, base11, writer11, base15, writer15)))
        outputs = {"base_identity": base_output, "writer_live": writer_output}
        captures = {"base_identity": base15, "writer_live": writer15}
        for arm, mode in (("h3_full", "full"), ("h3_rank2", "rank2"),
                          ("h3_orthogonal", "orthogonal")):
            outputs[arm], captures[arm], error = run_mode(
                backend, base_batch, base11, writer11, q, mode)
            algebra = max(algebra, error)
            reconstruction = max(reconstruction, float(captures[arm]["reconstruction_max_abs"]))
            forwards += 1; evaluations += len(panel_rows)
        for arm in ARMS:
            records.extend(dict(record, panel=panel) for record in scoring.recovery_records(
                panel_rows, base_output, donor_output, outputs[arm], arm=arm))
        downstream[panel] = {arm: l15_pair_norms(backend, captures[arm], base15, base_batch)
                             for arm in ("writer_live", "h3_full", "h3_rank2", "h3_orthogonal")}

    summaries = {panel: {arm: scoring.summarize([r for r in records
        if r["panel"] == panel and r["arm"] == arm]) for arm in ARMS} for panel in ("A1", "A2")}
    behavior_fraction = {panel: {arm: summaries[panel][arm]["mean_recovery"]
        / summaries[panel]["h3_full"]["mean_recovery"] for arm in ("h3_rank2", "h3_orthogonal")}
        for panel in ("A1", "A2")}
    writer_fraction = {panel: summaries[panel]["h3_full"]["mean_recovery"]
        / summaries[panel]["writer_live"]["mean_recovery"] for panel in ("A1", "A2")}
    downstream_means = {panel: {arm: sum(values) / len(values)
        for arm, values in downstream[panel].items()} for panel in ("A1", "A2")}
    downstream_fraction = {panel: downstream_means[panel]["h3_rank2"]
        / downstream_means[panel]["h3_full"] for panel in ("A1", "A2")}
    pred_a = bool(orth_error <= 1e-5 and reconstruction <= 5e-4 and identity <= 1e-4
        and algebra <= 1e-6 and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS
        and len(records) == RECORDS and all(math.isfinite(r["recovery"]) for r in records))
    pred_b = all(summaries[p]["h3_full"]["direction_fraction"] >= 0.90
                 and writer_fraction[p] >= 0.30 for p in ("A1", "A2"))
    pred_c = all(behavior_fraction[p]["h3_rank2"] >= 0.80
                 and summaries[p]["h3_rank2"]["direction_fraction"] >= 0.90 for p in ("A1", "A2"))
    pred_d = all(abs(behavior_fraction[p]["h3_orthogonal"]) <= 0.30 for p in ("A1", "A2"))
    pred_e = all(downstream_fraction[p] >= 0.75 for p in ("A1", "A2"))
    predictions = {
        "pred_a_authority_capability_exact_instrument_and_price": pred_a,
        "pred_b_full_h3_response_is_material": pred_b,
        "pred_c_rank2_is_behaviorally_sufficient": pred_c,
        "pred_d_orthogonal_complement_is_secondary": pred_d,
        "pred_e_rank2_transports_to_weight_predicted_l15_pair": pred_e,
    }
    terminal = ("invalid" if not pred_a else "identification" if all(predictions.values())
                else "representation_only" if pred_b else "null")
    result = {"schema": "temporal_auxiliary_h3_rank2_downstream_v8_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter()-started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"q_orthonormality_max_abs": orth_error,
            "attention_reconstruction_max_abs": reconstruction, "identity_max_abs": identity,
            "projection_closure_max_abs": algebra},
        "summaries": summaries, "h3_full_fraction_of_writer": writer_fraction,
        "behavior_fraction_of_full_h3": behavior_fraction,
        "l15_h5_h1_response_norm_means": downstream_means,
        "rank2_fraction_of_full_l15_h5_h1_response": downstream_fraction,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fit_updates": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("candidate_id", "instrument", "summaries",
        "h3_full_fraction_of_writer", "behavior_fraction_of_full_h3",
        "l15_h5_h1_response_norm_means", "rank2_fraction_of_full_l15_h5_h1_response",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
