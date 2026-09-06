#!/usr/bin/env python3
"""Zero-fit sealed-v11 transfer of the Q8 causal-suffix attention operation."""

# BQGATE: EXPERIMENT pred_a_exact_authority_closure_coverage_and_price pred_b_frozen_q8_retains_full_h3_response pred_c_frozen_causal_suffix_closes_q8 pred_d_frozen_factor_signature_recurs pred_e_pre_subject_value_is_causally_zero
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
import circuit_candidate_temporal_auxiliary_fresh_cues_v11 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
import residual_source_onset_eval as onset
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_auxiliary_will_had_h3_rank8_v10_source_factor_atlas_v1 as atlas
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_auxiliary_will_had_l9_triple_to_h3_rank8_weight_writer_v1 as upstream

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_rank8_v11_causal_suffix_transfer_v1.json"
V10_ATLAS = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_v10_source_factor_atlas_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v11_capability_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v11.py"
ATLAS_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank8_v10_source_factor_atlas_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_v11_causal_suffix_transfer_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_rank8_v11_causal_suffix_transfer_v1"
EXPECTED = {
    "prior": "9ab5fab11e467d1d3bcf9eb9239e7aec7f5ec0a6113ccb72c581b05fd0b4b3c0",
    "v10_atlas": "7abe69894990d4c1a093afd312f40f65fa53fecee89e2bdd39344722ed4f25d8",
    "capability": "0330dc5a4f85bc68c4da6f98af2f4208335e65c644ddedd5d8cc487368091026",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "builder": "f75b17669a5fc5299d21f5b44e91530c03c71d75181683c7b6728cb95c862450",
    "atlas_runner": "211f847b8e0799a5ee9b889f64183bdc7e67df0862463748c3443f3417efcfda",
}
FACTORS = atlas.FACTORS
ARMS = ("base_identity", "h3_full", "q8_complete", "causal_suffix",
        "factor::base_pattern_on_value_change", "factor::pattern_on_base_value",
        "factor::pattern_value_interaction")
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 20, 640, 441


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def full_h3_query_output(backend, batch, base11, writer11):
    head_count = int(backend.model.config.n_head)
    head_width = int(backend.model.config.n_embd // head_count)

    def patch(_module, arguments):
        flattened = arguments[0]
        changed = flattened.clone().view(len(batch.row_ids), flattened.shape[1], head_count, head_width)
        for index, query in enumerate(batch.semantic_positions):
            query = int(query)
            changed[index, query, 3] += (writer11["head_output"][index, query, 3]
                - base11["head_output"][index, query, 3]).to(changed)
        return (changed.reshape_as(flattened),) + tuple(arguments[1:])

    handle = backend.model.transformer.h[11].attn.c_proj.register_forward_pre_hook(patch)
    try:
        return backend.native(batch, capture=False)
    finally:
        handle.remove()


def main():
    paths = {"prior": PRIOR, "v10_atlas": V10_ATLAS, "capability": CAPABILITY,
             "subspace": SUBSPACE, "builder": BUILDER, "atlas_runner": ATLAS_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("v11 causal-suffix transfer authority changed")
    prior, v10, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, V10_ATLAS, CAPABILITY, SUBSPACE)]
    if (prior.get("candidate_id") != CANDIDATE_ID or v10.get("terminal") != "screen"
            or capability.get("terminal") != "manifest"):
        raise RuntimeError("authority terminal changed")
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "rank": 8,
        "arms": list(ARMS), "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS, "records": RECORDS,
        "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}
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
    reconstruction = identity = head_closure = q8_closure = 0.0
    summaries, fractions, norm_fractions, presubject_value_norms = {}, {}, {}, {}
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
        h3_full = full_h3_query_output(backend, base_batch, base_h3, writer_h3)
        forwards += 5; evaluations += 5*len(panel_rows)
        reconstruction = max(reconstruction, *(float(value["reconstruction_max_abs"])
            for value in (base8, donor8, base_h3, writer_h3)))
        identity = max(identity, upstream.pair_error(base_output, base11_output))
        aggregate = {factor: [] for factor in FACTORS}
        suffix, complete, full_norms, projected_norms = [], [], [], []
        panel_pre = []
        for index, query in enumerate(base_batch.semantic_positions):
            query = int(query)
            groups = atlas.source_partition(base_batch.token_rows[index], donor_batch.token_rows[index],
                                            query, destinations[index])
            exact_head = (writer_h3["head_output"][index, query, 3].float()
                          - base_h3["head_output"][index, query, 3].float())
            exact_q8 = exact_head @ q
            reconstructed = exact_head.new_zeros(128)
            factor_rows, suffix_row = {}, exact_q8.new_zeros(8)
            for factor in FACTORS:
                factor_rows[factor] = exact_q8.new_zeros(8)
            for group in atlas.GROUPS:
                for factor in FACTORS:
                    term = atlas.factor_head(base_h3, writer_h3, index, query, groups[group], factor)
                    reconstructed += term
                    coordinate = term @ q
                    factor_rows[factor] += coordinate
                    if group in ("subject_onset", "post_subject", "self"):
                        suffix_row += coordinate
                    if group in ("prefix", "cue", "pre_subject") and factor == "base_pattern_on_value_change":
                        panel_pre.append(float(backend.torch.linalg.vector_norm(coordinate)))
            head_closure = max(head_closure, float((exact_head-reconstructed).abs().max()))
            q8_closure = max(q8_closure, float((exact_q8-reconstructed@q).abs().max()))
            for factor in FACTORS:
                aggregate[factor].append(factor_rows[factor])
            suffix.append(suffix_row); complete.append(reconstructed@q)
            full_norms.append(float(backend.torch.linalg.vector_norm(exact_head)))
            projected_norms.append(float(backend.torch.linalg.vector_norm(exact_q8)))
        coordinates = {f"factor::{factor}": backend.torch.stack(values)
                       for factor, values in aggregate.items()}
        coordinates["causal_suffix"] = backend.torch.stack(suffix)
        coordinates["q8_complete"] = backend.torch.stack(complete)
        outputs = {"base_identity": base_output, "h3_full": h3_full}
        for arm in ARMS[2:]:
            cache = upstream.endpoint_cache(backend, base_batch, base_output, coordinates[arm], q, gain)
            outputs[arm] = backend.patched(base_batch, site=upstream.kernel.SiteRef(
                site_id="resid:18", evidence_kind="residual"), donor_cache=cache)
            forwards += 1; evaluations += len(panel_rows)
        panel_records = []
        for arm in ARMS:
            panel_records.extend(dict(record, panel=panel) for record in scoring.recovery_records(
                panel_rows, base_output, donor_output, outputs[arm], arm=arm))
        records.extend(panel_records)
        summaries[panel] = {arm: scoring.summarize([record for record in panel_records
            if record["arm"] == arm]) for arm in ARMS}
        q8_behavior = summaries[panel]["q8_complete"]["mean_recovery"]
        fractions[panel] = {arm: summaries[panel][arm]["mean_recovery"] / q8_behavior
                            for arm in ARMS[3:]}
        fractions[panel]["q8_of_h3_full"] = q8_behavior / summaries[panel]["h3_full"]["mean_recovery"]
        norm_fractions[panel] = {
            "causal_suffix_of_complete_q8": atlas.mean_norm(backend, coordinates["causal_suffix"])
                / atlas.mean_norm(backend, coordinates["q8_complete"]),
            "q8_projection_of_full_h3_diagnostic": sum(projected_norms)/sum(full_norms),
        }
        presubject_value_norms[panel] = max(panel_pre, default=0.0)
    pred_a = bool(reconstruction <= 5e-4 and identity <= 1e-4 and head_closure <= 5e-4
        and q8_closure <= 5e-4 and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS
        and len(records) == RECORDS and all(math.isfinite(record["recovery"]) for record in records))
    pred_b = all(fractions[p]["q8_of_h3_full"] >= .80
        and summaries[p]["q8_complete"]["direction_fraction"] >= .90 for p in ("A1", "A2"))
    pred_c = all(norm_fractions[p]["causal_suffix_of_complete_q8"] >= .90
        and fractions[p]["causal_suffix"] >= .90 for p in ("A1", "A2"))
    value, pattern, interaction = (f"factor::{factor}" for factor in FACTORS)
    pred_d = all(abs(fractions[p][value]) >= max(abs(fractions[p][pattern]), abs(fractions[p][interaction]))
        and fractions[p][value] >= .80 and fractions[p][pattern] <= 0
        and abs(fractions[p][interaction]) <= .15 for p in ("A1", "A2"))
    pred_e = all(value <= 1e-6 for value in presubject_value_norms.values())
    predictions = {"pred_a_exact_authority_closure_coverage_and_price": pred_a,
        "pred_b_frozen_q8_retains_full_h3_response": pred_b,
        "pred_c_frozen_causal_suffix_closes_q8": pred_c,
        "pred_d_frozen_factor_signature_recurs": pred_d,
        "pred_e_pre_subject_value_is_causally_zero": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "temporal_auxiliary_h3_rank8_v11_causal_suffix_transfer_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"attention_reconstruction_max_abs": reconstruction,
            "capture_identity_max_abs": identity, "head_factor_closure_max_abs": head_closure,
            "q8_factor_closure_max_abs": q8_closure}, "summaries": summaries,
        "behavior_fractions": fractions, "norm_fractions": norm_fractions,
        "max_pre_subject_value_coordinate_norm": presubject_value_norms,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "records": len(records), "fit_updates": 0, "transformer_backwards": 0,
            "model_updates": 0}, "records": records}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
        "behavior_fractions", "norm_fractions", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
