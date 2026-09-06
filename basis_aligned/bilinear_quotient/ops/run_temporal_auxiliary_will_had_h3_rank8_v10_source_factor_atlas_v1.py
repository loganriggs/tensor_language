#!/usr/bin/env python3
"""Exact source-region by attention-factor atlas for the confirmed v10 H3 Q8 response."""

# BQGATE: EXPERIMENT pred_a_authority_exact_partition_factor_closure_coverage_and_price pred_b_all_source_value_is_the_dominant_operation pred_c_causal_suffix_sources_close_the_response pred_d_pre_subject_value_change_is_zero pred_e_subject_is_material_but_not_complete
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
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_auxiliary_will_had_l9_triple_to_h3_rank8_weight_writer_v1 as upstream

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_rank8_v10_source_factor_atlas_v1.json"
SUBJECT_NULL = ROOT / "circuits/followups/temporal_auxiliary_will_had_l8_subject_state_to_h3_rank8_weight_writer_v1_result.json"
OLD_SOURCE = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block11h3_source_response_v1_result.json"
DIRECT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_weight_direct_residual_route_v2_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v10_capability_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v10.py"
FAMILY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1.py"
UPSTREAM_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_l9_triple_to_h3_rank8_weight_writer_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_v10_source_factor_atlas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_rank8_v10_source_factor_atlas_v1"
EXPECTED = {
    "prior": "382891ba46ddaff4fb45f3605b8acaacc9da62cae0a89bcb67d33955ce2c1f7a",
    "subject_null": "2c829eff3b00de55442260885cf86ca2b4432c64bcaf7b97bfb8ff739c2bb5b4",
    "old_source": "1fd089aeb63e2ec7e1771d54170461dc18b3bd105e2b7fc08413d8456a515cf1",
    "direct": "571a3d0d22fe159adbc0e37825873b4dda25dda99c46b7a47a9cc6a260de471f",
    "capability": "9923322703c72d50b2a1f06138ef35269db48e0a8a4ccb365f82df3519b113ad",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "builder": "e945e0b4679fa74d6cea23594ba553d9b2ffd3ac653c353d09d1873f2a3e4494",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "upstream_runner": "9dd491e50cc2b46ad3fa4071ef1c53c0335a70dcb94709ec89889328c35ec4b2",
}
GROUPS = ("prefix", "cue", "pre_subject", "subject_onset", "post_subject", "self")
FACTORS = ("pattern_on_base_value", "base_pattern_on_value_change", "pattern_value_interaction")
CELL_ARMS = tuple(f"{group}::{factor}" for group in GROUPS for factor in FACTORS)
GROUP_ARMS = tuple(f"group::{group}" for group in GROUPS)
FACTOR_ARMS = tuple(f"factor::{factor}" for factor in FACTORS)
SPECIAL_ARMS = ("causal_suffix", "complete")
ARMS = ("base_identity",) + CELL_ARMS + GROUP_ARMS + FACTOR_ARMS + SPECIAL_ARMS
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 72, 2400, 1890


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def source_partition(base_ids, donor_ids, query, subject_positions):
    differences = [index for index, pair in enumerate(zip(base_ids, donor_ids)) if pair[0] != pair[1]]
    if len(base_ids) != len(donor_ids) or len(differences) != 1:
        raise RuntimeError("source partition requires one aligned cue difference")
    cue, query = differences[0], int(query)
    subject = tuple(sorted(int(position) for position in subject_positions))
    if len(subject) != 2 or subject[0] <= cue or subject[-1] >= query:
        raise RuntimeError("invalid cue/subject/query ordering")
    groups = {"prefix": tuple(range(cue)), "cue": (cue,),
        "pre_subject": tuple(range(cue+1, subject[0])), "subject_onset": subject,
        "post_subject": tuple(range(subject[-1]+1, query)), "self": (query,)}
    flattened = tuple(position for group in GROUPS for position in groups[group])
    if tuple(sorted(flattened)) != tuple(range(query+1)) or len(flattened) != len(set(flattened)):
        raise RuntimeError("source groups do not exactly partition causal positions")
    return groups


def factor_head(base_h3, writer_h3, index, query, sources, factor):
    if not sources:
        return base_h3["head_output"][index, query, 3].float().new_zeros(128)
    p0 = base_h3["pattern"][index, 3, query, list(sources)].float()
    p1 = writer_h3["pattern"][index, 3, query, list(sources)].float()
    v0 = base_h3["value"][index, list(sources), 3].float()
    v1 = writer_h3["value"][index, list(sources), 3].float()
    if factor == "pattern_on_base_value":
        return ((p1-p0)[:, None]*v0).sum(0)
    if factor == "base_pattern_on_value_change":
        return (p0[:, None]*(v1-v0)).sum(0)
    if factor == "pattern_value_interaction":
        return ((p1-p0)[:, None]*(v1-v0)).sum(0)
    raise RuntimeError("unknown attention factor")


def mean_norm(backend, coordinates):
    return sum(float(backend.torch.linalg.vector_norm(row)) for row in coordinates)/len(coordinates)


def main():
    paths = {"prior": PRIOR, "subject_null": SUBJECT_NULL, "old_source": OLD_SOURCE,
        "direct": DIRECT, "capability": CAPABILITY, "subspace": SUBSPACE,
        "builder": BUILDER, "family_runner": FAMILY_RUNNER, "upstream_runner": UPSTREAM_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("v10 source-factor atlas authority changed")
    prior, subject_null, old_source, direct, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, SUBJECT_NULL, OLD_SOURCE, DIRECT, CAPABILITY, SUBSPACE)]
    if (prior.get("candidate_id") != CANDIDATE_ID or subject_null.get("terminal") != "null"
            or old_source.get("terminal") != "screen" or direct.get("terminal") != "screen"
            or capability.get("terminal") != "manifest"):
        raise RuntimeError("authority terminal changed")
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "rank": 8,
        "groups": list(GROUPS), "factors": list(FACTORS), "arms": list(ARMS),
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "records": RECORDS, "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}
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
    reconstruction = identity = head_closure = q8_closure = replay_error = 0.0
    coordinate_norms, summaries, fractions = {}, {}, {}
    for panel in ("A1", "A2"):
        panel_rows = [row for row in rows if row["transform_id"] == panel]
        base_batch = das._batch(backend, panel_rows, side="base")
        donor_batch = das._batch(backend, panel_rows, side="donor")
        base_output, base8 = attention_eval.capture_layer_attention(
            backend, base_batch, 8, call=lambda: backend.native(base_batch, capture=True))
        donor_output, donor8 = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        base11_output, base_h3 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, base8, donor8, destinations, ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            _writer_output, writer_h3 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        finally:
            handle.remove()
        forwards += 4; evaluations += 4*len(panel_rows)
        reconstruction = max(reconstruction, *(float(value["reconstruction_max_abs"])
            for value in (base8, donor8, base_h3, writer_h3)))
        identity = max(identity, upstream.pair_error(base_output, base11_output))
        partitions = [source_partition(base_ids, donor_ids, query, subjects)
            for base_ids, donor_ids, query, subjects in zip(base_batch.token_rows,
                donor_batch.token_rows, base_batch.semantic_positions, destinations)]
        cells = {arm: [] for arm in CELL_ARMS}
        complete_exact = []
        for index, query in enumerate(base_batch.semantic_positions):
            query = int(query)
            exact_head = (writer_h3["head_output"][index, query, 3].float()
                          - base_h3["head_output"][index, query, 3].float())
            exact_q8 = exact_head@q
            complete_exact.append(exact_q8)
            reconstructed_head = exact_head.new_zeros(128)
            for group in GROUPS:
                for factor in FACTORS:
                    term = factor_head(base_h3, writer_h3, index, query,
                        partitions[index][group], factor)
                    cells[f"{group}::{factor}"].append(term@q)
                    reconstructed_head += term
            head_closure = max(head_closure, float((exact_head-reconstructed_head).abs().max()))
            q8_closure = max(q8_closure, float((exact_q8-reconstructed_head@q).abs().max()))
        coordinates = {arm: backend.torch.stack(values) for arm, values in cells.items()}
        for group in GROUPS:
            coordinates[f"group::{group}"] = sum(
                coordinates[f"{group}::{factor}"] for factor in FACTORS)
        for factor in FACTORS:
            coordinates[f"factor::{factor}"] = sum(
                coordinates[f"{group}::{factor}"] for group in GROUPS)
        coordinates["causal_suffix"] = sum(
            coordinates[f"{group}::{factor}"]
            for group in ("subject_onset", "post_subject", "self") for factor in FACTORS)
        coordinates["complete"] = sum(coordinates[arm] for arm in CELL_ARMS)
        exact_tensor = backend.torch.stack(complete_exact)
        q8_closure = max(q8_closure, float((exact_tensor-coordinates["complete"]).abs().max()))
        outputs = {"base_identity": base_output}
        for arm in ARMS[1:]:
            cache = upstream.endpoint_cache(backend, base_batch, base_output, coordinates[arm], q, gain)
            outputs[arm] = backend.patched(base_batch,
                site=kernel.SiteRef(site_id="resid:18", evidence_kind="residual"), donor_cache=cache)
            forwards += 1; evaluations += len(panel_rows)
        for arm in ARMS:
            records.extend(dict(record, panel=panel) for record in scoring.recovery_records(
                panel_rows, base_output, donor_output, outputs[arm], arm=arm))
        summaries[panel] = {arm: scoring.summarize([record for record in records
            if record["panel"] == panel and record["arm"] == arm]) for arm in ARMS}
        complete_norm = mean_norm(backend, coordinates["complete"])
        coordinate_norms[panel] = {arm: mean_norm(backend, coordinates[arm]) for arm in ARMS[1:]}
        fractions[panel] = {arm: {"q8_norm_fraction": coordinate_norms[panel][arm]/complete_norm,
            "behavior_fraction": summaries[panel][arm]["mean_recovery"]
                / summaries[panel]["complete"]["mean_recovery"]} for arm in ARMS[1:]}
        replay_error = max(replay_error, abs(summaries[panel]["complete"]["mean_recovery"]
            - direct["summaries"][panel]["weight_direct_resid18"]["mean_recovery"]))
    factor_ranking = {panel: sorted(({"factor": factor,
        "mean_absolute_recovery": summaries[panel][f"factor::{factor}"]["mean_absolute_recovery"],
        "behavior_fraction": fractions[panel][f"factor::{factor}"]["behavior_fraction"]}
        for factor in FACTORS), key=lambda row: row["mean_absolute_recovery"], reverse=True)
        for panel in ("A1", "A2")}
    presubject_value_arms = tuple(f"{group}::base_pattern_on_value_change"
        for group in ("prefix", "cue", "pre_subject"))
    pred_a = bool(reconstruction <= 5e-4 and identity <= 1e-4 and head_closure <= 5e-4
        and q8_closure <= 5e-4 and replay_error <= 1e-6 and forwards <= MAX_FORWARDS
        and evaluations <= MAX_EVALUATIONS and len(records) == RECORDS
        and all(math.isfinite(record["recovery"]) for record in records))
    pred_b = all(factor_ranking[panel][0]["factor"] == "base_pattern_on_value_change"
        and fractions[panel]["factor::base_pattern_on_value_change"]["behavior_fraction"] >= .80
        for panel in ("A1", "A2"))
    pred_c = all(fractions[panel]["causal_suffix"]["q8_norm_fraction"] >= .90
        and fractions[panel]["causal_suffix"]["behavior_fraction"] >= .90 for panel in ("A1", "A2"))
    pred_d = all(coordinate_norms[panel][arm] <= 1e-6
        and summaries[panel][arm]["mean_absolute_recovery"] <= 1e-6
        for panel in ("A1", "A2") for arm in presubject_value_arms)
    pred_e = all(.40 <= fractions[panel]["group::subject_onset"]["q8_norm_fraction"] <= .80
        for panel in ("A1", "A2"))
    predictions = {"pred_a_authority_exact_partition_factor_closure_coverage_and_price": pred_a,
        "pred_b_all_source_value_is_the_dominant_operation": pred_b,
        "pred_c_causal_suffix_sources_close_the_response": pred_c,
        "pred_d_pre_subject_value_change_is_zero": pred_d,
        "pred_e_subject_is_material_but_not_complete": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "temporal_auxiliary_h3_rank8_v10_source_factor_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"attention_reconstruction_max_abs": reconstruction,
            "capture_identity_max_abs": identity, "head_factor_closure_max_abs": head_closure,
            "q8_factor_closure_max_abs": q8_closure,
            "complete_direct_route_replay_max_abs": replay_error},
        "summaries": summaries, "mean_q8_coordinate_norm": coordinate_norms,
        "fractions_of_complete": fractions, "factor_rankings": factor_ranking,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "records": len(records), "fit_updates": 0, "transformer_backwards": 0,
            "model_updates": 0}, "records": records}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
        "fractions_of_complete", "factor_rankings", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__":
    main()
