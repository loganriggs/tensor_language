#!/usr/bin/env python3
"""Remove L15 head responses during confirmed H3 rank-eight actuation."""

# BQGATE: EXPERIMENT pred_a_authority_exact_composed_instrument_and_price pred_b_complete_l15_response_is_material pred_c_h5h1_concentrate_l15_mediation pred_d_other_seven_are_secondary pred_e_rank8_replays
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
import circuit_candidate_temporal_auxiliary_fresh_cues_v10 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset
import run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1 as instrument
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_rank8_to_l15_pair_necessity_v1.json"
RANK8 = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_weight_rank8_v10_confirmation_v1_result.json"
L15_PAIR = ROOT / "circuits/followups/temporal_auxiliary_will_had_original_writer_attn15_h5h1_joint_v1_result.json"
SCREEN = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v10_capability_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v10.py"
FAMILY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1.py"
INSTRUMENT = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_to_l15_pair_necessity_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_rank8_to_l15_pair_necessity_v1"
EXPECTED = {
    "prior": "222882238ea5b155b4eec6cec41301c6431811b7405cd6194a803452a78f9b37",
    "rank8": "0d19330da62f37f404570455ea4aeb198a7da787d244278935b6798dffc6e7db",
    "l15_pair": "73433f9c265e035e22b227edfc99165a8c34be9508053c081ca2f31f971eefb8",
    "screen": "adc318db1b08fd47c034cf4cd15b7234b16582b7ab134275f8c36265219254fc",
    "capability": "9923322703c72d50b2a1f06138ef35269db48e0a8a4ccb365f82df3519b113ad",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "builder": "e945e0b4679fa74d6cea23594ba553d9b2ffd3ac653c353d09d1873f2a3e4494",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "instrument": "936a7920b164e57473f7b7204352584b5a438dd0745afc27bfe7f0dd80354a66",
}
PAIR, OTHER = (5, 1), (0, 2, 3, 4, 6, 7, 8)
ARMS = ("base_identity", "h3_rank8", "h3_rank8_l15_all_clamped",
        "h3_rank8_l15_h5h1_clamped", "h3_rank8_l15_other7_clamped",
        "base_l15_h5h1_self_clamp")
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 24, 900, 378


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def h3_hook(backend, batch, base11, writer11, q):
    head_count = int(backend.model.config.n_head)
    width = int(backend.model.config.n_embd // head_count)

    def patch(_module, arguments):
        flattened = arguments[0]
        changed = flattened.clone().view(len(batch.row_ids), flattened.shape[1], head_count, width)
        for index, query in enumerate(batch.semantic_positions):
            delta = (writer11["head_output"][index, :int(query)+1, 3].float()
                     - base11["head_output"][index, :int(query)+1, 3].float())
            changed[index, :int(query)+1, 3] += ((delta @ q) @ q.T).to(changed)
        return (changed.reshape_as(flattened),) + tuple(arguments[1:])

    return patch


def l15_clamp_hook(backend, batch, base15, heads):
    selected = tuple(int(head) for head in heads)
    head_count = int(backend.model.config.n_head)
    width = int(backend.model.config.n_embd // head_count)
    if len(selected) != len(set(selected)) or any(not 0 <= head < head_count for head in selected):
        raise RuntimeError("invalid L15 clamp head set")

    def patch(_module, arguments):
        flattened = arguments[0]
        changed = flattened.clone().view(len(batch.row_ids), flattened.shape[1], head_count, width)
        for index, query in enumerate(batch.semantic_positions):
            for head in selected:
                changed[index, :int(query)+1, head] = base15["head_output"][index, :int(query)+1, head].to(changed)
        return (changed.reshape_as(flattened),) + tuple(arguments[1:])

    return patch


def run_composed(backend, batch, base11, writer11, q, base15, heads, *, install_h3=True):
    handles = []
    if install_h3:
        handles.append(backend.model.transformer.h[11].attn.c_proj.register_forward_pre_hook(
            h3_hook(backend, batch, base11, writer11, q)))
    handles.append(backend.model.transformer.h[15].attn.c_proj.register_forward_pre_hook(
        l15_clamp_hook(backend, batch, base15, heads)))
    try:
        return backend.native(batch, capture=False)
    finally:
        for handle in handles:
            handle.remove()


def main():
    paths = {"prior": PRIOR, "rank8": RANK8, "l15_pair": L15_PAIR, "screen": SCREEN,
             "capability": CAPABILITY, "subspace": SUBSPACE, "builder": BUILDER,
             "family_runner": FAMILY_RUNNER, "instrument": INSTRUMENT}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("rank8-to-L15 necessity authority changed")
    prior, rank8_result, l15_pair, screen, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, RANK8, L15_PAIR, SCREEN, CAPABILITY, SUBSPACE)]
    if (prior.get("candidate_id") != CANDIDATE_ID or rank8_result.get("terminal") != "paired_confirmation"
            or l15_pair.get("terminal") != "screen" or screen.get("terminal") != "screen"
            or capability.get("terminal") != "manifest"):
        raise RuntimeError("authority terminal changed")
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "rank": 8,
        "l15_pair": list(PAIR), "l15_other": list(OTHER), "arms": list(ARMS),
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "records": RECORDS, "fit_updates": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    q = family[8]
    records, forwards, evaluations = [], 0, 0
    reconstruction = identity_error = algebra = replay_error = 0.0
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
        rank8_output, rank8_15, error = instrument.run_mode(
            backend, base_batch, base11, writer11, q, "rank2")
        algebra = max(algebra, error)
        replay = run_composed(backend, base_batch, base11, writer11, q, base15, ())
        all_clamped = run_composed(backend, base_batch, base11, writer11, q, base15, range(9))
        pair_clamped = run_composed(backend, base_batch, base11, writer11, q, base15, PAIR)
        other_clamped = run_composed(backend, base_batch, base11, writer11, q, base15, OTHER)
        self_clamp = run_composed(backend, base_batch, base11, writer11, q, base15, PAIR,
                                  install_h3=False)
        forwards += 11; evaluations += 11*len(panel_rows)
        replay_error = max(replay_error, instrument.pair_error(rank8_output, replay))
        identity_error = max(identity_error, instrument.pair_error(base_output, base11_output),
                             instrument.pair_error(base_output, base15_output),
                             instrument.pair_error(base_output, self_clamp))
        reconstruction = max(reconstruction, *(float(x["reconstruction_max_abs"])
            for x in (base8, donor8, base11, writer11, base15, rank8_15)))
        outputs = {"base_identity": base_output, "h3_rank8": rank8_output,
            "h3_rank8_l15_all_clamped": all_clamped,
            "h3_rank8_l15_h5h1_clamped": pair_clamped,
            "h3_rank8_l15_other7_clamped": other_clamped,
            "base_l15_h5h1_self_clamp": self_clamp}
        for arm in ARMS:
            records.extend(dict(record, panel=panel) for record in scoring.recovery_records(
                panel_rows, base_output, donor_output, outputs[arm], arm=arm))
    summaries = {panel: {arm: scoring.summarize([r for r in records
        if r["panel"] == panel and r["arm"] == arm]) for arm in ARMS} for panel in ("A1", "A2")}
    def removed(panel, arm):
        target = summaries[panel]["h3_rank8"]["mean_recovery"]
        return (target - summaries[panel][arm]["mean_recovery"]) / target
    removal = {panel: {"all_l15": removed(panel, "h3_rank8_l15_all_clamped"),
        "h5h1": removed(panel, "h3_rank8_l15_h5h1_clamped"),
        "other7": removed(panel, "h3_rank8_l15_other7_clamped")} for panel in ("A1", "A2")}
    concentration = {panel: {name: removal[panel][name]/removal[panel]["all_l15"]
        for name in ("h5h1", "other7")} for panel in ("A1", "A2")}
    receipt_replay = max(abs(summaries[p]["h3_rank8"]["mean_recovery"]
        - rank8_result["summaries"][p]["h3_weight_rank8"]["mean_recovery"])
        for p in ("A1", "A2"))
    pred_a = bool(reconstruction <= 5e-4 and identity_error <= 1e-4 and algebra <= 1e-6
        and replay_error <= 1e-4 and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS
        and len(records) == RECORDS and all(math.isfinite(r["recovery"]) for r in records))
    pred_b = all(removal[p]["all_l15"] >= .10 for p in ("A1", "A2"))
    pred_c = all(concentration[p]["h5h1"] >= .80 for p in ("A1", "A2"))
    pred_d = all(abs(concentration[p]["other7"]) <= .20 for p in ("A1", "A2"))
    pred_e = receipt_replay <= 1e-6
    predictions = {"pred_a_authority_exact_composed_instrument_and_price": pred_a,
        "pred_b_complete_l15_response_is_material": pred_b,
        "pred_c_h5h1_concentrate_l15_mediation": pred_c,
        "pred_d_other_seven_are_secondary": pred_d, "pred_e_rank8_replays": pred_e}
    terminal = ("invalid" if not pred_a or not pred_e else "identification" if all(predictions.values())
        else "representational_reader" if not pred_b else "distributed_l15_reader")
    result = {"schema": "temporal_auxiliary_h3_rank8_to_l15_pair_necessity_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter()-started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"attention_reconstruction_max_abs": reconstruction,
            "identity_max_abs": identity_error, "projection_closure_max_abs": algebra,
            "composed_rank8_replay_max_abs": replay_error, "receipt_mean_replay_max_abs": receipt_replay},
        "summaries": summaries, "removal_fraction_of_rank8_behavior": removal,
        "fraction_of_complete_l15_removal": concentration, "predictions": predictions,
        "terminal": terminal, "price": {"model_forwards": forwards,
            "example_evaluations": evaluations, "records": len(records),
            "fit_updates": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("candidate_id", "instrument",
        "removal_fraction_of_rank8_behavior", "fraction_of_complete_l15_removal",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
