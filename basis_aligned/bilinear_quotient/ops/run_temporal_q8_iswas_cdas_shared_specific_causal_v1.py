#!/usr/bin/env python3
"""Causal decomposition of an is/was cDAS write by the temporal Q8 weight projector."""

# BQGATE: EXPERIMENT pred_a_exact_authority_replay_decomposition_coverage_and_price pred_b_temporal_q8_shared_component_is_causally_material pred_c_temporal_q8_shared_component_is_selective pred_d_iswas_specific_component_remains_material pred_e_shared_and_specific_compose_without_signed_reversal
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_candidate_aspectual_different_readout_is_was_v2 as v2
import circuit_candidate_aspectual_different_readout_is_was_v3 as v3
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_q8_iswas_cdas_shared_specific_causal_v1.json"
OVERLAP_RESULT = ROOT / "circuits/followups/temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1_result.json"
ISWAS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
V2_CAPABILITY = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v2_capability_result.json"
V3_CAPABILITY = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v3_capability_result.json"
V2_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v2.py"
V3_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v3.py"
OVERLAP_RUNNER = ROOT / "ops/run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1.py"
OUT = ROOT / "circuits/followups/temporal_q8_iswas_cdas_shared_specific_causal_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_q8_iswas_cdas_shared_specific_causal_v1"
EXPECTED = {
    "prior": "bb833d6e2106a9656e61768a41398dc05c59d3b6ff7f0289b0f51749e0f0e32b",
    "overlap_result": "883861b7392a8b1214491bef2fab80bfd670dfca98d87f91cb73fcb22bf624e6",
    "iswas": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "v2_capability": "f76fbcd6174cc9e8e3f77352ee5461815156cdae421ecc43a0e0c3576b63af7e",
    "v3_capability": "744d2fd3c8200ca00005357961df3d435a7a13dbdbd7c3a51a487daee76acec3",
    "v2_builder": "4ed62d06e2ffe5c471efc70eeaa35c7524a66923f19c64c803faef7200d4a62f",
    "v3_builder": "ac240f13478168948814f59c0425270dd345ba03053332b72b34857e9a022638",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1",
}
ARMS = ("base", "original_cdas", "temporal_q8_shared", "iswas_specific", "shared_plus_specific")
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 2, 208, 520
TARGET_SCALE = 1.9129114151000977


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def group(rows):
    return {family: [row for row in rows if row["family"] == family]
            for family in ("A1", "A2", "P", "C")}


def summarize_family(backend, rows, states, family):
    torch = backend.torch
    index = torch.arange(len(rows), device=backend.device)
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margins = {arm: das.head_logits(backend, value)[index, answer]
               - das.head_logits(backend, value)[index, foil] for arm, value in states.items()}
    records, report = [], {}
    if family in ("A1", "A2"):
        denominator = margins["donor"] - margins["base"]
        keep = denominator.abs() > 1e-6
        for arm in ARMS:
            recovery = (margins[arm]-margins["base"])[keep]/denominator[keep]
            report[arm] = {"mean_recovery": float(recovery.mean()),
                "mean_absolute_recovery": float(recovery.abs().mean()),
                "direction_fraction": float((recovery > 0).float().mean()), "rows": int(keep.sum())}
            kept_indices = torch.nonzero(keep).reshape(-1).tolist()
            records.extend({"row_id": rows[row_index]["row_id"], "family": family, "arm": arm,
                "recovery": float(value)} for row_index, value in zip(kept_indices, recovery.tolist()))
    else:
        for arm in ARMS:
            effects = (margins[arm]-margins["base"]).abs()/TARGET_SCALE
            report[arm] = {"same_answer_effect": float(effects.mean()), "rows": len(rows)}
            records.extend({"row_id": row["row_id"], "family": family, "arm": arm,
                "same_answer_effect": float(value)} for row, value in zip(rows, effects.tolist()))
    return report, records


def main():
    paths = {"prior": PRIOR, "overlap_result": OVERLAP_RESULT, "iswas": ISWAS,
        "subspace": SUBSPACE, "v2_capability": V2_CAPABILITY, "v3_capability": V3_CAPABILITY,
        "v2_builder": V2_BUILDER, "v3_builder": V3_BUILDER, "overlap_runner": OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("shared-specific causal authority changed")
    prior, overlap_result, iswas, subspace, cap2, cap3 = [json.loads(path.read_text())
        for path in (PRIOR, OVERLAP_RESULT, ISWAS, SUBSPACE, V2_CAPABILITY, V3_CAPABILITY)]
    rows2, rows3 = v2.build_rows(), v3.build_rows()
    if (prior.get("candidate_id") != CANDIDATE_ID or overlap_result.get("terminal") != "screen"
            or iswas.get("terminal") != "screen" or any(cap.get("terminal") != "screen" for cap in (cap2, cap3))
            or v2.validate_rows(rows2) != "fef8174cd57a1382d87ae47ddb06ddebd8059e96e4f71472525a066a3048f911"
            or v3.validate_rows(rows3) != "35dc18a4e95764bc2126fc18d672d2f7666e4bc1c84a7bcc06d1299b820d7ad2"):
        raise RuntimeError("authority terminals or rows changed")
    by2, by3 = group(rows2), group(rows3)
    groups = [("v2_A1_heldout", by2["A1"][8:], "A1"), ("v2_A2", by2["A2"], "A2"),
        ("v2_P_heldout", by2["P"][8:], "P"), ("v2_C_heldout", by2["C"][8:], "C")]
    groups += [(f"v3_{family}", by3[family], family) for family in ("A1", "A2", "P", "C")]
    rows, spans = [], {}
    for name, family_rows, family in groups:
        start = len(rows); rows.extend(family_rows); spans[name] = (slice(start, len(rows)), family)
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "arms": list(ARMS),
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "records": RECORDS, "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    captures, native, head_identity = {}, {}, 0.0
    for side in ("base", "donor"):
        batch = das._batch(backend, rows, side=side)
        output = backend.native(batch, capture=True)
        captures[side] = torch.stack([torch.as_tensor(output.captured[(row["row_id"], "resid:18")])
                                      for row in rows]).to(backend.device).float()
        native[side] = output
        logits = das.head_logits(backend, captures[side])
        for index, row in enumerate(rows):
            answer_id = row[f"{side}_answer_id"]; foil_id = row[f"{side}_foil_id"]
            head_identity = max(head_identity,
                abs(float(logits[index, answer_id])-float(output.answer_foil[index][0])),
                abs(float(logits[index, foil_id])-float(output.answer_foil[index][1])))
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    q = family[8]
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in range(12, 18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, q, gain)
    s = torch.linalg.qr(modes, mode="reduced").Q
    axis_values = np.asarray(iswas["basis"]["values_column_major"], dtype=np.float32)
    if hashlib.sha256(axis_values.tobytes()).hexdigest() != iswas["basis"]["sha256"]:
        raise RuntimeError("is/was basis changed")
    a = torch.as_tensor(axis_values, device=backend.device).reshape(1152, 1)
    a = a/torch.linalg.vector_norm(a)
    shared_axis = s@(s.T@a)
    specific_axis = a-shared_axis
    delta = captures["donor"]-captures["base"]
    coefficient = delta@a
    states = {"base": captures["base"], "donor": captures["donor"],
        "original_cdas": captures["base"]+coefficient@a.T,
        "temporal_q8_shared": captures["base"]+coefficient@shared_axis.T,
        "iswas_specific": captures["base"]+coefficient@specific_axis.T,
        "shared_plus_specific": captures["base"]+coefficient@(shared_axis+specific_axis).T}
    write_closure = float((states["original_cdas"]-states["shared_plus_specific"]).abs().max())
    logits_original = das.head_logits(backend, states["original_cdas"])
    logits_composed = das.head_logits(backend, states["shared_plus_specific"])
    logit_closure = float((logits_original-logits_composed).abs().max())
    reports, records = {}, []
    for name, (span, family_name) in spans.items():
        local_states = {arm: value[span] for arm, value in states.items()}
        report, local_records = summarize_family(backend, rows[span], local_states, family_name)
        reports[name] = report; records.extend(dict(record, group=name) for record in local_records)
    released = iswas["score"]["families"]
    replay_error = 0.0
    for name, _span in spans.items():
        reference = released[name]
        measured = reports[name]["original_cdas"]
        for key in set(reference)&set(measured)&{"mean_recovery", "mean_absolute_recovery", "same_answer_effect"}:
            replay_error = max(replay_error, abs(float(reference[key])-float(measured[key])))
    a_groups = ("v2_A1_heldout", "v2_A2", "v3_A1", "v3_A2")
    control_groups = ("v2_P_heldout", "v2_C_heldout", "v3_P", "v3_C")
    pred_a = bool(head_identity <= 1e-3 and orientation_error <= 1e-6
        and write_closure <= 1e-6 and logit_closure <= 1e-5 and replay_error <= 1e-4
        and len(rows) == 104 and len(records) == RECORDS)
    pred_b = all(reports[name]["temporal_q8_shared"]["mean_recovery"] > 0
        and reports[name]["temporal_q8_shared"]["mean_absolute_recovery"] >= .10
        and reports[name]["temporal_q8_shared"]["direction_fraction"] >= .75 for name in a_groups)
    pred_c = all(reports[name]["temporal_q8_shared"]["same_answer_effect"] <= .20
                 for name in control_groups)
    pred_d = all(reports[name]["iswas_specific"]["mean_absolute_recovery"] >= .25 for name in a_groups)
    pred_e = all(reports[name][arm]["mean_recovery"] > 0
        for name in a_groups for arm in ("temporal_q8_shared", "iswas_specific"))
    predictions = {"pred_a_exact_authority_replay_decomposition_coverage_and_price": pred_a,
        "pred_b_temporal_q8_shared_component_is_causally_material": pred_b,
        "pred_c_temporal_q8_shared_component_is_selective": pred_c,
        "pred_d_iswas_specific_component_remains_material": pred_d,
        "pred_e_shared_and_specific_compose_without_signed_reversal": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "temporal_q8_iswas_cdas_shared_specific_causal_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "component_norms": {"shared": float(torch.linalg.vector_norm(shared_axis)),
            "specific": float(torch.linalg.vector_norm(specific_axis))},
        "instrument": {"native_head_max_abs": head_identity,
            "f_linear_orientation_max_abs": orientation_error, "write_sum_max_abs": write_closure,
            "logit_sum_max_abs": logit_closure, "released_cdas_metric_replay_max_abs": replay_error},
        "reports": reports, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": 2, "example_evaluations": 2*len(rows),
            "records": len(records), "fit_updates": 0, "transformer_backwards": 0,
            "model_updates": 0}, "records": records}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "component_norms", "instrument",
        "reports", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
