#!/usr/bin/env python3
"""Complete downstream module-removal atlas under tensor-derived H3 rank eight."""

# BQGATE: EXPERIMENT pred_a_authority_exact_replay_self_clamp_and_price pred_b_at_least_one_stable_material_module pred_c_mlp17_is_stable_largest_absolute_module pred_d_mlp17_is_opposing
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
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_rank8_downstream_module_removal_atlas_v2.json"
RANK8 = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_weight_rank8_v10_confirmation_v1_result.json"
L15_NULL = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_to_l15_pair_necessity_v1_result.json"
OLD_ATLAS = ROOT / "circuits/followups/temporal_auxiliary_will_had_original_writer_downstream_module_atlas_v1_result.json"
SCREEN = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v10_capability_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v10.py"
FAMILY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1.py"
INSTRUMENT = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_downstream_module_removal_atlas_v2_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_rank8_downstream_module_removal_atlas_v2"
EXPECTED = {
    "prior": "eb0ca1c415060d32e0fa560ece373c8b83e6741a8081cb663b9cd7f83a45644a",
    "rank8": "0d19330da62f37f404570455ea4aeb198a7da787d244278935b6798dffc6e7db",
    "l15_null": "cfe7d6341fb0763a8812310a763ec9520e8a28f10605950853ded1400685533e",
    "old_atlas": "f8517f43f41444b966b95ff0d8da9449f25bd95d32b726c1082066605cfd076a",
    "screen": "adc318db1b08fd47c034cf4cd15b7234b16582b7ab134275f8c36265219254fc",
    "capability": "9923322703c72d50b2a1f06138ef35269db48e0a8a4ccb365f82df3519b113ad",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "builder": "e945e0b4679fa74d6cea23594ba553d9b2ffd3ac653c353d09d1873f2a3e4494",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "instrument": "936a7920b164e57473f7b7204352584b5a438dd0745afc27bfe7f0dd80354a66",
}
SITES = ("mlp:11",) + tuple(
    f"{kind}:{layer:02d}" for layer in range(12, 18) for kind in ("attn", "mlp"))
ARMS = ("base_identity", "h3_rank8") + tuple(f"remove_{site}" for site in SITES) + ("remove_all",)
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 46, 1500, 1008


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def site_module(backend, site):
    kind, layer_text = site.split(":")
    block = backend.model.transformer.h[int(layer_text)]
    return block.attn.c_proj if kind == "attn" else block.mlp


def capture_native_modules(backend, batch):
    cache, handles = {}, []
    for site in SITES:
        kind = site.split(":")[0]
        if kind == "attn":
            def capture(_module, arguments, site=site):
                cache[site] = arguments[0].detach().clone()
            handles.append(site_module(backend, site).register_forward_pre_hook(capture))
        else:
            def capture(_module, _arguments, output, site=site):
                cache[site] = output.detach().clone()
            handles.append(site_module(backend, site).register_forward_hook(capture))
    try:
        output = backend.native(batch, capture=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(cache) != set(SITES):
        raise RuntimeError("incomplete native module capture")
    return output, cache


def clamp_hook(batch, base_value, kind):
    if kind == "attn":
        def patch(_module, arguments):
            changed = arguments[0].clone()
            for index, query in enumerate(batch.semantic_positions):
                changed[index, :int(query)+1] = base_value[index, :int(query)+1].to(changed)
            return (changed,) + tuple(arguments[1:])
    else:
        def patch(_module, _arguments, output):
            changed = output.clone()
            for index, query in enumerate(batch.semantic_positions):
                changed[index, :int(query)+1] = base_value[index, :int(query)+1].to(changed)
            return changed
    return patch


def h3_hook(backend, batch, base11, writer11, q):
    head_count = int(backend.model.config.n_head)
    head_width = int(backend.model.config.n_embd // head_count)

    def patch(_module, arguments):
        flattened = arguments[0]
        changed = flattened.clone().view(len(batch.row_ids), flattened.shape[1], head_count, head_width)
        for index, query in enumerate(batch.semantic_positions):
            delta = (writer11["head_output"][index, :int(query)+1, 3].float()
                     - base11["head_output"][index, :int(query)+1, 3].float())
            changed[index, :int(query)+1, 3] += ((delta @ q) @ q.T).to(changed)
        return (changed.reshape_as(flattened),) + tuple(arguments[1:])

    return patch


def run_clamped(backend, batch, base11, writer11, q, base_modules, sites, *, install_h3=True):
    handles = []
    if install_h3:
        handles.append(backend.model.transformer.h[11].attn.c_proj.register_forward_pre_hook(
            h3_hook(backend, batch, base11, writer11, q)))
    for site in sites:
        kind = site.split(":")[0]
        module = site_module(backend, site)
        hook = clamp_hook(batch, base_modules[site], kind)
        handles.append(module.register_forward_pre_hook(hook) if kind == "attn"
                       else module.register_forward_hook(hook))
    try:
        return backend.native(batch, capture=False)
    finally:
        for handle in handles:
            handle.remove()


def main():
    paths = {"prior": PRIOR, "rank8": RANK8, "l15_null": L15_NULL,
             "old_atlas": OLD_ATLAS, "screen": SCREEN, "capability": CAPABILITY,
             "subspace": SUBSPACE, "builder": BUILDER, "family_runner": FAMILY_RUNNER,
             "instrument": INSTRUMENT}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("rank8 downstream atlas authority changed")
    prior, rank8_result, l15_null, old_atlas, screen, capability, subspace = [
        json.loads(path.read_text()) for path in
        (PRIOR, RANK8, L15_NULL, OLD_ATLAS, SCREEN, CAPABILITY, SUBSPACE)]
    if (prior.get("candidate_id") != CANDIDATE_ID
            or rank8_result.get("terminal") != "paired_confirmation"
            or l15_null.get("terminal") != "representational_reader"
            or old_atlas.get("terminal") != "screen" or screen.get("terminal") != "screen"
            or capability.get("terminal") != "manifest" or len(SITES) != 13):
        raise RuntimeError("authority terminal or site inventory changed")
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "sites": list(SITES), "arms": list(ARMS), "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS, "records": RECORDS,
        "fit_updates": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    q = family[8]
    records, forwards, evaluations = [], 0, 0
    reconstruction = identity_error = algebra_error = replay_error = 0.0
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
            _writer_output, writer11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        finally:
            handle.remove()
        base_capture_output, base_modules = capture_native_modules(backend, base_batch)
        rank8_output, rank8_capture, closure = instrument.run_mode(
            backend, base_batch, base11, writer11, q, "rank2")
        algebra_error = max(algebra_error, closure)
        outputs = {"base_identity": base_output, "h3_rank8": rank8_output}
        for site in SITES:
            outputs[f"remove_{site}"] = run_clamped(
                backend, base_batch, base11, writer11, q, base_modules, (site,))
        outputs["remove_all"] = run_clamped(
            backend, base_batch, base11, writer11, q, base_modules, SITES)
        self_clamp = run_clamped(
            backend, base_batch, base11, writer11, q, base_modules, SITES, install_h3=False)
        forwards += 21
        evaluations += 21 * len(panel_rows)
        identity_error = max(identity_error, instrument.pair_error(base_output, base11_output),
                             instrument.pair_error(base_output, base_capture_output),
                             instrument.pair_error(base_output, self_clamp))
        replay_error = max(replay_error, abs(scoring.summarize(scoring.recovery_records(
            panel_rows, base_output, donor_output, rank8_output, arm="replay"))["mean_recovery"]
            - rank8_result["summaries"][panel]["h3_weight_rank8"]["mean_recovery"]))
        reconstruction = max(reconstruction, *(float(value["reconstruction_max_abs"])
            for value in (base8, donor8, base11, writer11, rank8_capture)))
        for arm in ARMS:
            records.extend(dict(record, panel=panel) for record in scoring.recovery_records(
                panel_rows, base_output, donor_output, outputs[arm], arm=arm))
    summaries = {panel: {arm: scoring.summarize([record for record in records
        if record["panel"] == panel and record["arm"] == arm]) for arm in ARMS}
        for panel in ("A1", "A2")}
    def removed(panel, arm):
        rank8_mean = summaries[panel]["h3_rank8"]["mean_recovery"]
        return (rank8_mean - summaries[panel][arm]["mean_recovery"]) / rank8_mean
    removal = {panel: {site: removed(panel, f"remove_{site}") for site in SITES}
               | {"all": removed(panel, "remove_all")} for panel in ("A1", "A2")}
    rankings = {panel: sorted(({"site": site, "signed_removal_fraction": removal[panel][site],
        "absolute_removal_fraction": abs(removal[panel][site])} for site in SITES),
        key=lambda row: row["absolute_removal_fraction"], reverse=True) for panel in ("A1", "A2")}
    material = [site for site in SITES if all(abs(removal[panel][site]) >= .10
                                               for panel in ("A1", "A2"))]
    pred_a = bool(reconstruction <= 5e-4 and identity_error <= 1e-4 and algebra_error <= 1e-6
        and replay_error <= 1e-6 and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS
        and len(records) == RECORDS and all(math.isfinite(record["recovery"]) for record in records))
    pred_b = bool(material)
    pred_c = all(rankings[panel][0]["site"] == "mlp:17" for panel in ("A1", "A2"))
    pred_d = all(removal[panel]["mlp:17"] < 0 for panel in ("A1", "A2"))
    predictions = {"pred_a_authority_exact_replay_self_clamp_and_price": pred_a,
        "pred_b_at_least_one_stable_material_module": pred_b,
        "pred_c_mlp17_is_stable_largest_absolute_module": pred_c,
        "pred_d_mlp17_is_opposing": pred_d}
    joint_material = all(abs(removal[panel]["all"]) >= .20 for panel in ("A1", "A2"))
    terminal = ("invalid" if not pred_a else "screen" if pred_b else
                "distributed_downstream" if joint_material else "residual_route")
    result = {"schema": "temporal_auxiliary_h3_rank8_downstream_module_removal_atlas_result_v2",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"attention_reconstruction_max_abs": reconstruction,
            "native_all_site_self_clamp_max_abs": identity_error,
            "projection_closure_max_abs": algebra_error,
            "rank8_receipt_mean_replay_max_abs": replay_error},
        "summaries": summaries, "removal_fraction_of_rank8_behavior": removal,
        "absolute_rankings": rankings, "stable_material_sites": material,
        "joint_removal_material": joint_material, "predictions": predictions,
        "terminal": terminal, "price": {"model_forwards": forwards,
            "example_evaluations": evaluations, "records": len(records),
            "fit_updates": 0, "model_updates": 0}, "records": records}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
        "removal_fraction_of_rank8_behavior", "absolute_rankings", "stable_material_sites",
        "joint_removal_material", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
