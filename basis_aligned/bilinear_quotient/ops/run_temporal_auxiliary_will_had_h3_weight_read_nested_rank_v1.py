#!/usr/bin/env python3
"""Screen nested downstream-weight H3 bases for the smallest causal rank."""

# BQGATE: EXPERIMENT pred_a_authority_exact_nested_instrument_and_price pred_b_rank4_replays pred_c_small_rank_reaches_pareto_gate pred_d_monotone_static_reader_energy
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
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1.json"
RANK4 = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank2_plus_weight_read_rank4_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v8_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v8.py"
BASE_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank2_plus_weight_read_rank4_v1.py"
INSTRUMENT = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_weight_read_nested_rank_v1"
EXPECTED = {
    "prior": "cc75d98834faf2c409408b0c8088a0de1ca6998873d42f9ace209cd09af6d8aa",
    "rank4": "c7ee48746a626b8e2c0a3b7a789da1f443daee1022ea161ce150fa85fff2467b",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "capability": "fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff",
    "builder": "13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf",
    "base_runner": "aabb157fd964d7f9ab6cd1d9e0414fe62b360a2de0ea4a42a95f38b0912ac55d",
    "instrument": "936a7920b164e57473f7b7204352584b5a438dd0745afc27bfe7f0dd80354a66",
}
RANKS = tuple(range(2, 9))
ARMS = ("base_identity", "writer_live", "h3_full") + tuple(
    name for rank in RANKS for name in (f"h3_rank{rank}", f"h3_rank{rank}_orthogonal"))
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 48, 1600, 1020


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def build_family(backend, subspace):
    torch, model = backend.torch, backend.model
    q2 = torch.linalg.qr(torch.tensor(subspace["axis_artifacts"]["two_task_dim_union_rank2"],
                                      device=backend.device).float(), mode="reduced").Q
    width = int(model.transformer.h[11].attn.head_dim)
    output = model.transformer.h[11].attn.c_proj.weight.detach().float()[:, 3*width:4*width]
    maps = []
    for head_index in (5, 1):
        attention = model.transformer.h[15].attn
        for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v"):
            section = getattr(attention, name).weight.detach().float()[
                head_index*width:(head_index+1)*width]
            maps.append((section / torch.linalg.matrix_norm(section)) @ output)
    reader = torch.cat(maps)
    perpendicular = reader @ (torch.eye(width, device=backend.device) - q2 @ q2.T)
    _u, singular, vh = torch.linalg.svd(perpendicular, full_matrices=False)
    family = {2: q2}
    for rank in RANKS[1:]:
        family[rank] = torch.linalg.qr(
            torch.cat((q2, vh[:rank-2].T), dim=1), mode="reduced").Q
    denominator = torch.linalg.matrix_norm(reader).square()
    energy = {rank: float(torch.linalg.matrix_norm(reader @ q @ q.T).square() / denominator)
              for rank, q in family.items()}
    return family, singular, energy


def main():
    paths = {"prior": PRIOR, "rank4": RANK4, "subspace": SUBSPACE,
             "capability": CAPABILITY, "builder": BUILDER,
             "base_runner": BASE_RUNNER, "instrument": INSTRUMENT}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("nested-rank authority changed")
    prior, rank4_result, subspace, capability = [json.loads(path.read_text())
        for path in (PRIOR, RANK4, SUBSPACE, CAPABILITY)]
    if (prior.get("candidate_id") != CANDIDATE_ID
            or rank4_result.get("terminal") != "static_metric_incomplete"
            or capability.get("terminal") != "manifest"):
        raise RuntimeError("authority terminal changed")
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "ranks": list(RANKS),
        "arms": list(ARMS), "rows": len(rows), "basis_fit_examples": 0,
        "basis_fit_labels": 0, "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS, "records": RECORDS,
        "fit_updates": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    family, singular, energy = build_family(backend, subspace)
    torch = backend.torch
    orth_error = max(float((q.T @ q - torch.eye(rank, device=q.device)).abs().max())
                     for rank, q in family.items())
    nesting_error = max(float((family[rank+1] @ family[rank+1].T @ family[rank]
                               - family[rank]).abs().max()) for rank in RANKS[:-1])
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
        outputs["h3_full"], captures["h3_full"], error = instrument.run_mode(
            backend, base_batch, base11, writer11, family[2], "full")
        algebra = max(algebra, error); forwards += 1; evaluations += len(panel_rows)
        for rank, q in family.items():
            for suffix, mode in (("", "rank2"), ("_orthogonal", "orthogonal")):
                arm = f"h3_rank{rank}{suffix}"
                outputs[arm], captures[arm], error = instrument.run_mode(
                    backend, base_batch, base11, writer11, q, mode)
                algebra = max(algebra, error); forwards += 1; evaluations += len(panel_rows)
        reconstruction = max(reconstruction, *(float(c["reconstruction_max_abs"])
                                                for c in captures.values()))
        for arm in ARMS:
            records.extend(dict(record, panel=panel) for record in scoring.recovery_records(
                panel_rows, base_output, donor_output, outputs[arm], arm=arm))
        downstream[panel] = {arm: instrument.l15_pair_norms(backend, captures[arm], base15, base_batch)
                             for arm in ARMS if arm.startswith("h3_")}
    summaries = {panel: {arm: scoring.summarize([r for r in records
        if r["panel"] == panel and r["arm"] == arm]) for arm in ARMS} for panel in ("A1", "A2")}
    behavior = {panel: {rank: {
        "projected": summaries[panel][f"h3_rank{rank}"]["mean_recovery"]
            / summaries[panel]["h3_full"]["mean_recovery"],
        "orthogonal": summaries[panel][f"h3_rank{rank}_orthogonal"]["mean_recovery"]
            / summaries[panel]["h3_full"]["mean_recovery"]} for rank in RANKS}
        for panel in ("A1", "A2")}
    downstream_means = {panel: {arm: sum(values)/len(values) for arm, values in values_by_arm.items()}
                        for panel, values_by_arm in downstream.items()}
    transport = {panel: {rank: downstream_means[panel][f"h3_rank{rank}"]
        / downstream_means[panel]["h3_full"] for rank in RANKS} for panel in ("A1", "A2")}
    passing = [rank for rank in RANKS if all(behavior[p][rank]["projected"] >= .90
        and abs(behavior[p][rank]["orthogonal"]) <= .12 and transport[p][rank] >= .90
        for p in ("A1", "A2"))]
    rank4_reference = rank4_result["behavior_fraction_of_full_h3"]
    rank4_transport_reference = rank4_result["downstream_fraction_of_full_h3"]
    replay_error = max([abs(behavior[p][4]["projected"] - rank4_reference[p]["h3_weight_rank4"])
        for p in ("A1", "A2")] + [abs(transport[p][4] - rank4_transport_reference[p]["h3_weight_rank4"])
        for p in ("A1", "A2")])
    pred_a = bool(orth_error <= 1e-5 and nesting_error <= 1e-5 and reconstruction <= 5e-4
        and identity_error <= 1e-4 and algebra <= 1e-6 and forwards <= MAX_FORWARDS
        and evaluations <= MAX_EVALUATIONS and len(records) == RECORDS
        and all(math.isfinite(r["recovery"]) for r in records))
    pred_b = replay_error <= 1e-6
    pred_c = bool(passing and min(passing) <= 6)
    pred_d = all(energy[b] + 1e-7 >= energy[a] for a, b in zip(RANKS, RANKS[1:]))
    predictions = {"pred_a_authority_exact_nested_instrument_and_price": pred_a,
        "pred_b_rank4_replays": pred_b, "pred_c_small_rank_reaches_pareto_gate": pred_c,
        "pred_d_monotone_static_reader_energy": pred_d}
    terminal = "invalid" if not pred_a or not pred_b or not pred_d else "screen" if passing else "insufficient_static_rank"
    result = {"schema": "temporal_auxiliary_h3_weight_read_nested_rank_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter()-started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "weight_family": {"static_reader_energy_fraction": energy,
            "orthogonal_reader_singular_values": [float(x) for x in singular[:12]]},
        "instrument": {"orthonormality_max_abs": orth_error, "nesting_max_abs": nesting_error,
            "attention_reconstruction_max_abs": reconstruction, "identity_max_abs": identity_error,
            "projection_closure_max_abs": algebra, "rank4_replay_max_abs": replay_error},
        "behavior_fraction_of_full_h3": behavior, "downstream_fraction_of_full_h3": transport,
        "passing_ranks": passing, "selected_rank": min(passing) if passing else None,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fit_updates": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("candidate_id", "weight_family", "instrument",
        "behavior_fraction_of_full_h3", "downstream_fraction_of_full_h3", "passing_ranks",
        "selected_rank", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
