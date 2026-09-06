#!/usr/bin/env python3
"""Compile tensor-derived H3 rank eight into a direct final-residual actuator."""

# BQGATE: EXPERIMENT pred_a_authority_exact_identity_coverage_and_price pred_b_weight_route_matches_dynamic_all_module_clamp pred_c_direct_route_retains_rank8_behavior pred_d_skip_gain_is_nonzero_and_frozen
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
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset
import run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1 as instrument
import run_temporal_auxiliary_will_had_h3_rank8_downstream_module_removal_atlas_v2 as atlas
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_rank8_weight_direct_residual_route_v1.json"
RANK8 = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_weight_rank8_v10_confirmation_v1_result.json"
ATLAS_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_downstream_module_removal_atlas_v2_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v10_capability_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v10.py"
FAMILY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1.py"
ATLAS_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank8_downstream_module_removal_atlas_v2.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_weight_direct_residual_route_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_rank8_weight_direct_residual_route_v1"
EXPECTED = {
    "prior": "827020bab7eef0f1b44cea087acae62bca6b92905d0028791cde220653db351d",
    "rank8": "0d19330da62f37f404570455ea4aeb198a7da787d244278935b6798dffc6e7db",
    "atlas_result": "66fdced03582fd9890382043dfbf3950e438d2085b378a3533750939bb849d91",
    "capability": "9923322703c72d50b2a1f06138ef35269db48e0a8a4ccb365f82df3519b113ad",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "builder": "e945e0b4679fa74d6cea23594ba553d9b2ffd3ac653c353d09d1873f2a3e4494",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "atlas_runner": "6b9c5e9397534495cb7e41cdcff6d4d3cd88a0557f26bede435308433be85683",
}
ARMS = ("base_identity", "h3_rank8", "dynamic_all_module_clamp", "weight_direct_resid18")
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 18, 600, 252


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def all_clamped_capture(backend, batch, base11, writer11, q, base_modules):
    handles = [backend.model.transformer.h[11].attn.c_proj.register_forward_pre_hook(
        atlas.h3_hook(backend, batch, base11, writer11, q))]
    for site in atlas.SITES:
        kind = site.split(":")[0]
        module = atlas.site_module(backend, site)
        hook = atlas.clamp_hook(batch, base_modules[site], kind)
        handles.append(module.register_forward_pre_hook(hook) if kind == "attn"
                       else module.register_forward_hook(hook))
    try:
        return backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()


def direct_cache(backend, batch, base_output, base11, writer11, q, gain):
    weight = backend.model.transformer.h[11].attn.c_proj.weight
    head_count = int(backend.model.config.n_head)
    head_width = int(backend.model.config.n_embd // head_count)
    cache, writes = {}, []
    for index, (row_id, query) in enumerate(zip(batch.row_ids, batch.semantic_positions)):
        delta = (writer11["head_output"][index, int(query), 3].float()
                 - base11["head_output"][index, int(query), 3].float())
        projected = (delta @ q) @ q.T
        flattened = backend.torch.zeros(head_count * head_width, device=weight.device, dtype=weight.dtype)
        flattened[3*head_width:4*head_width] = projected.to(flattened)
        write = (flattened @ weight) * backend.torch.as_tensor(gain, device=weight.device, dtype=weight.dtype)
        base18 = base_output.captured[(row_id, "resid:18")]
        cache[(row_id, "resid:18")] = base18 + write.to(base18)
        writes.append(write.detach().clone())
    return cache, writes


def main():
    paths = {"prior": PRIOR, "rank8": RANK8, "atlas_result": ATLAS_RESULT,
             "capability": CAPABILITY, "subspace": SUBSPACE, "builder": BUILDER,
             "family_runner": FAMILY_RUNNER, "atlas_runner": ATLAS_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("direct residual route authority changed")
    prior, rank8_result, atlas_result, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, RANK8, ATLAS_RESULT, CAPABILITY, SUBSPACE)]
    if (prior.get("candidate_id") != CANDIDATE_ID
            or rank8_result.get("terminal") != "paired_confirmation"
            or atlas_result.get("terminal") != "residual_route"
            or capability.get("terminal") != "manifest"):
        raise RuntimeError("authority terminal changed")
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "arms": list(ARMS),
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
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in range(12, 18))
    records, forwards, evaluations = [], 0, 0
    reconstruction = identity_error = route_logit_error = route_state_error = replay_error = 0.0
    ratios = {}
    for panel in ("A1", "A2"):
        panel_rows = [row for row in rows if row["transform_id"] == panel]
        base_batch = das._batch(backend, panel_rows, side="base")
        donor_batch = das._batch(backend, panel_rows, side="donor")
        base_output, base8 = attention_eval.capture_layer_attention(
            backend, base_batch, 8, call=lambda: backend.native(base_batch, capture=True))
        donor_output, donor8 = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, base8, donor8, destinations, ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            _writer_output, writer11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        finally:
            handle.remove()
        _capture_output, base_modules = atlas.capture_native_modules(backend, base_batch)
        rank8_output, rank8_capture, closure = instrument.run_mode(
            backend, base_batch, base11, writer11, q, "rank2")
        dynamic_output = all_clamped_capture(backend, base_batch, base11, writer11, q, base_modules)
        cache18, _writes = direct_cache(backend, base_batch, base_output, base11, writer11, q, gain)
        direct_output = backend.patched(base_batch,
            site=kernel.SiteRef(site_id="resid:18", evidence_kind="residual"), donor_cache=cache18)
        self_output = backend.patched(base_batch,
            site=kernel.SiteRef(site_id="resid:18", evidence_kind="residual"),
            donor_cache=base_output.captured)
        forwards += 9
        evaluations += 9 * len(panel_rows)
        identity_error = max(identity_error, instrument.pair_error(base_output, base11_output),
                             instrument.pair_error(base_output, self_output))
        route_logit_error = max(route_logit_error, instrument.pair_error(dynamic_output, direct_output))
        for row_id in base_batch.row_ids:
            route_state_error = max(route_state_error, float((
                dynamic_output.captured[(row_id, "resid:18")] - cache18[(row_id, "resid:18")]
            ).float().abs().max()))
        reconstruction = max(reconstruction, *(float(value["reconstruction_max_abs"])
            for value in (base8, donor8, base11, writer11, rank8_capture)))
        outputs = {"base_identity": base_output, "h3_rank8": rank8_output,
                   "dynamic_all_module_clamp": dynamic_output,
                   "weight_direct_resid18": direct_output}
        for arm in ARMS:
            records.extend(dict(record, panel=panel) for record in scoring.recovery_records(
                panel_rows, base_output, donor_output, outputs[arm], arm=arm))
        panel_summaries = {arm: scoring.summarize([record for record in records
            if record["panel"] == panel and record["arm"] == arm]) for arm in ARMS}
        ratios[panel] = (panel_summaries["weight_direct_resid18"]["mean_recovery"]
                         / panel_summaries["h3_rank8"]["mean_recovery"])
        replay_error = max(replay_error, abs(panel_summaries["h3_rank8"]["mean_recovery"]
            - rank8_result["summaries"][panel]["h3_weight_rank8"]["mean_recovery"]))
    summaries = {panel: {arm: scoring.summarize([record for record in records
        if record["panel"] == panel and record["arm"] == arm]) for arm in ARMS}
        for panel in ("A1", "A2")}
    pred_a = bool(reconstruction <= 5e-4 and identity_error <= 1e-4 and replay_error <= 1e-6
        and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS and len(records) == RECORDS
        and all(math.isfinite(record["recovery"]) for record in records))
    pred_b = route_logit_error <= 1e-4
    pred_c = all(.95 <= ratios[panel] <= 1.05 for panel in ("A1", "A2"))
    pred_d = bool(math.isfinite(gain) and gain != 0.0)
    predictions = {"pred_a_authority_exact_identity_coverage_and_price": pred_a,
        "pred_b_weight_route_matches_dynamic_all_module_clamp": pred_b,
        "pred_c_direct_route_retains_rank8_behavior": pred_c,
        "pred_d_skip_gain_is_nonzero_and_frozen": pred_d}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "temporal_auxiliary_h3_rank8_weight_direct_residual_route_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "frozen_skip_gain": gain,
        "instrument": {"attention_reconstruction_max_abs": reconstruction,
            "native_resid18_self_patch_max_abs": identity_error,
            "rank8_receipt_mean_replay_max_abs": replay_error,
            "weight_vs_dynamic_route_logit_max_abs": route_logit_error,
            "weight_vs_dynamic_route_resid18_max_abs": route_state_error},
        "summaries": summaries, "direct_fraction_of_rank8_behavior": ratios,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "records": len(records), "fit_updates": 0, "model_updates": 0},
        "records": records}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "frozen_skip_gain",
        "instrument", "direct_fraction_of_rank8_behavior", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__":
    main()
