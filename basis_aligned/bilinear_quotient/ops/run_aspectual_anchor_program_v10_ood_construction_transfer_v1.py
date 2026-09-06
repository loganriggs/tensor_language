#!/usr/bin/env python3
"""Sealed OOD transfer test of donor-free aspectual actuator program v10."""

# BQGATE: EXPERIMENT pred_a_authority_native_capability_and_exact_head pred_b_frozen_donor_free_program pred_c_ood_A_transfer pred_d_ood_P_generalization pred_e_independent_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import aspectual_anchor_transparent_path_program_v10 as program
import circuit_candidate_aspectual_ood_construction_v3 as ood
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_program_v10_ood_construction_transfer_v1.json"
PROGRAM = ROOT / "ops/aspectual_anchor_transparent_path_program_v10.py"
RELEASE = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v10_result.json"
RANK1 = ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_ood_construction_v3.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_program_v10_ood_construction_transfer_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.program_v10_ood_construction_transfer_v1"
EXPECTED_PRIOR_SHA256 = "17d78f31c503184f5e2fd880d23c034f7f3b6d040407f37dbc0ed5846c7ae734"
EXPECTED = {
    PROGRAM: "b5c81e7fab8b7d6503944e5ac40b1a542f4af9470d5cdc5667bbf103ac48a2f5",
    RELEASE: "9677b97b66465ed49ee86b726954d4d74a8ae4ad285b255387f93e3067112661",
    RANK1: "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
    BUILDER: "5f91eaa016b37b39933388a7d12c8154a9f5598bf18c1a24bcb70b53f47cc392",
}
MODEL_FORWARDS_MAX = 17
EXAMPLE_EVALUATIONS_MAX = 136
HEAD_GRID_EVALUATIONS = 16_448


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    release = json.loads(RELEASE.read_text())
    rank1 = json.loads(RANK1.read_text())
    rows = ood.build_rows()
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or release.get("terminal") != "release"
        or rank1.get("terminal") != "screen"
        or release.get("program_sha256") != EXPECTED[PROGRAM]
        or ood.validate_rows(rows) != "ce9b03af12be0ff782fd10d968ba9f46ae100c76642b36ff20eca54c04a59bb0"
        or len(rows) != 64
    ):
        raise ExperimentError("candidate, terminal, program, row hash, or population changed")
    return rows, rank1


def summarize(records, key):
    values = [record[key] for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError(f"missing/nonfinite values for {key}")
    return {
        "count": len(values),
        f"mean_{key}": statistics.fmean(values),
        f"mean_absolute_{key}": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
        "endpoint_fraction": sum(record["grid_index"] in (0, program.ACTUATOR_GRID_POINTS - 1) for record in records) / len(records),
    }


def main() -> None:
    rows, rank1 = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_program_v10_ood_construction_transfer_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "rows": 64,
        "families": ["A1", "A2", "P", "C"],
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "selected_head_grid_evaluations": HEAD_GRID_EVALUATIONS,
        "model_backwards": 0,
        "model_updates": 0,
        "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    q = torch.tensor(rank1["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    if q.shape != (1152,) or abs(float(q.norm()) - 1.0) > 1.0e-4 or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != rank1["basis"]["sha256"]:
        raise ExperimentError("basis reconstruction failed")
    families = {family: [row for row in rows if row["family"] == family] for family in ("A1", "A2", "P", "C")}
    head_ok, head_error = das.verify_head(backend, families["A1"][:8], "resid:18")
    forward_calls, evaluations = 1, 8
    target_scale = float(rank1["score"]["families"]["target_scale"])
    records = []
    capability = True

    for family in ("A1", "A2", "P", "C"):
        for chunk in producer._chunks(families[family], 8):
            base, donor, _ = das.capture_site(backend, chunk, "resid:18")
            forward_calls += 2
            evaluations += 2 * len(chunk)
            for i, row in enumerate(chunk):
                if family in ("A1", "A2"):
                    source = base[i]
                    target_id, foil_id = row["donor_answer_id"], row["donor_foil_id"]
                    base_native = program.exact_selected_margin(base[i], backend.model.lm_head, target_id=row["base_answer_id"], foil_id=row["base_foil_id"])
                    donor_target = program.exact_selected_margin(donor[i], backend.model.lm_head, target_id=target_id, foil_id=foil_id)
                    capability = capability and float(base_native) > 0.0 and float(donor_target) > 0.0
                    source_side = "base"
                elif family == "P":
                    source = donor[i]
                    target_id, foil_id = row["base_foil_id"], row["base_answer_id"]
                    original = program.exact_selected_margin(source, backend.model.lm_head, target_id=foil_id, foil_id=target_id)
                    capability = capability and float(original) > 0.0
                    donor_target = None
                    source_side = "noun_shifted_donor_prompt"
                else:
                    source = base[i]
                    target_id, foil_id = row["base_foil_id"], row["base_answer_id"]
                    original = program.exact_selected_margin(source, backend.model.lm_head, target_id=foil_id, foil_id=target_id)
                    capability = capability and float(original) > 0.0
                    donor_target = None
                    source_side = "base"
                with torch.no_grad():
                    actuation = program.donor_free_margin_reflection(source, q, backend.model.lm_head, target_id=target_id, foil_id=foil_id)
                base_margin = float(actuation["base_target_margin"])
                patched_margin = float(actuation["patched_target_margin"])
                reflection_denominator = -2.0 * base_margin
                reflection = (patched_margin - base_margin) / reflection_denominator if abs(reflection_denominator) > 1.0e-6 else float("nan")
                record = {
                    "family": family,
                    "row_id": str(row["row_id"]),
                    "construction_id": row["construction_id"],
                    "source_side": source_side,
                    "alpha": float(actuation["alpha"]),
                    "grid_index": int(actuation["grid_index"]),
                    "base_target_margin": base_margin,
                    "patched_target_margin": patched_margin,
                    "margin_reflection_fraction": reflection,
                    "confirmation_donor_activation_used_by_program": False,
                }
                if family in ("A1", "A2"):
                    denominator = float(donor_target) - base_margin
                    record["donor_reference_margin"] = float(donor_target)
                    record["recovery"] = (patched_margin - base_margin) / denominator if abs(denominator) > 1.0e-6 else float("nan")
                elif family == "C":
                    record["normalized_unrelated_effect"] = abs(patched_margin - base_margin) / target_scale
                records.append(record)

    by_family = {family: [record for record in records if record["family"] == family] for family in families}
    summaries = {
        "A1": summarize(by_family["A1"], "recovery"),
        "A2": summarize(by_family["A2"], "recovery"),
        "P": summarize(by_family["P"], "margin_reflection_fraction"),
        "C": summarize(by_family["C"], "normalized_unrelated_effect"),
    }
    pred_a = capability and head_ok and head_error <= 1.0e-3
    pred_b = all(record["confirmation_donor_activation_used_by_program"] is False for record in records) and program.ACTUATOR_BUDGET == 7833.8336181640625 and program.ACTUATOR_GRID_POINTS == 257
    pred_c = all(summaries[family]["mean_recovery"] >= 0.50 and summaries[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = summaries["P"]["mean_margin_reflection_fraction"] >= 0.50 and summaries["P"]["direction_fraction"] >= 0.75
    pred_e = summaries["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(records) == 64 and len({record["row_id"] for record in records}) == 64 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX and len(records) * program.ACTUATOR_GRID_POINTS == HEAD_GRID_EVALUATIONS
    predictions = {
        "pred_a_authority_native_capability_and_exact_head": pred_a,
        "pred_b_frozen_donor_free_program": pred_b,
        "pred_c_ood_A_transfer": pred_c,
        "pred_d_ood_P_generalization": pred_d,
        "pred_e_independent_C_selectivity": pred_e,
        "pred_f_exact_coverage_and_price": pred_f,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "program_v10_transfers_selectively_to_sealed_ood_syntax", "null": "program_v10_fails_ood_transfer_or_independent_selectivity", "invalid": "authority_capability_head_program_identity_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_program_v10_ood_construction_transfer_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "program_sha256": EXPECTED[PROGRAM],
        "basis_sha256": rank1["basis"]["sha256"],
        "head_control": {"passed": head_ok, "max_abs_difference": head_error},
        "score": {"families": summaries, "target_scale": target_scale, "forward_calls": forward_calls, "example_evaluations": evaluations, "selected_head_grid_evaluations": len(records) * program.ACTUATOR_GRID_POINTS, "record_count": len(records), "model_backwards": 0, "model_updates": 0, "fit_parameters": 0},
        "predictions": predictions,
        "intervention_records": records,
        "terminal": terminal,
        "reason": reason,
        "next_action": "fold OOD construction scope into program v11" if terminal == "screen" else "retain v10's licensed constructions and test earlier operational interchange",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "families": summaries, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
