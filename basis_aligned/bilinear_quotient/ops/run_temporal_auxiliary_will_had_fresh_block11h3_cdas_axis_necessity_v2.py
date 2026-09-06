#!/usr/bin/env python3
"""Deterministic executable H3-axis artifact and donor-side necessity test."""

# BQGATE: EXPERIMENT pred_a_fit_and_projector_reproduce pred_b_sufficiency_reproduction pred_c_bidirectional_necessity pred_d_selective_neutralization pred_e_executable_axis_artifact pred_f_managed_price
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_unit_greedy as g
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_auxiliary_will_had_fresh_block11h3_single_component_cdas_v1 as parent
import single_component_das_eval as single

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_block11h3_cdas_axis_necessity_v2.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_block11h3_single_component_cdas_v1_result.json"
JOINT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_joint_reader_mediation_v1_result.json"
LIB = ROOT / "ops/single_component_das_eval.py"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_block11h3_cdas_axis_necessity_v2_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_block11h3_cdas_axis_necessity_v2"
EXPECTED = {
    "prior": "ad4a895dc7da86f27909a99e372c8752c02188f56522eaf879cbcb5f57d7b2c3",
    "parent": "5fba2b88bd7a6ec9883ee624e05495600e31258defcc355a22ab219ba9734b78",
    "joint": "7132fab362c1f137ed650bf8144a2b08a175d161c21db7171c2e1b52f2ffa173",
    "lib": "363569c1b1cf20e4f31a4569d2467b4c86a6563405a49766af968037e12028b8",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
}
MODEL_FORWARDS, EXAMPLE_EVALUATIONS = 1838, 29600
BACKWARD_FORWARDS, MODEL_UPDATES = 1800, 900


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "parent": PARENT_RESULT, "joint": JOINT,
             "lib": LIB, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    prior, old, joint = [json.loads(path.read_text()) for path in (PRIOR, PARENT_RESULT, JOINT)]
    rows = candidate.build_rows()
    if (prior.get("candidate_id") != CANDIDATE_ID or old.get("terminal") != "screen"
            or joint.get("terminal") != "screen" or not all(old["predictions"].values())
            or len(rows) != 128):
        raise ExperimentError("population or parent screen changed")
    return rows, old


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "rank": 1,
            "independent_fits": 2, "model_forwards": MODEL_FORWARDS,
            "example_evaluations": EXAMPLE_EVALUATIONS,
            "transformer_backward_forwards": BACKWARD_FORWARDS,
            "model_updates": MODEL_UPDATES, "stored_axis_coordinates": 128}


def donor_neutralization(backend, prep, q):
    out = g.forward_units(
        backend, prep.donor_batch, units=parent.UNIT,
        donor_cache=prep.base_cache, base_cache=prep.donor_cache, q=q)
    patched = [float(answer - foil) for answer, foil in out.tolist()]
    recoveries = [(donor - value) / (donor - base)
                  for base, donor, value in zip(prep.base_axis, prep.donor_axis, patched)]
    return {"mean_recovery_toward_base": sum(recoveries) / len(recoveries),
            "direction_fraction_toward_base": sum(value > 0 for value in recoveries) / len(recoveries),
            "mean_absolute_recovery": sum(abs(value) for value in recoveries) / len(recoveries)}


def donor_control_effect(backend, prep, q, scale):
    out = g.forward_units(
        backend, prep.donor_batch, units=parent.UNIT,
        donor_cache=prep.base_cache, base_cache=prep.donor_cache, q=q)
    patched = [float(answer - foil) for answer, foil in out.tolist()]
    return sum(abs(value - donor) / scale for value, donor in zip(patched, prep.donor_axis)) / len(patched)


def main():
    rows, old = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = parent.producer.Bilin18TorchBackend.load("cuda")
    family_rows = {family: [row for row in rows if row["transform_id"] == family]
                   for family in ("A1", "A2", "P", "C")}
    preps = {"fit": g.prepare(backend, family_rows["A1"][0::2]),
             "heldout": g.prepare(backend, family_rows["A1"][1::2]),
             "a2": g.prepare(backend, family_rows["A2"]),
             "p": g.prepare(backend, family_rows["P"]),
             "c": g.prepare(backend, family_rows["C"])}
    fits = [single.fit(backend, preps["fit"], parent.UNIT, rank=1, steps=150, lr=0.03,
                       random_seeds=(1, 2), complement_weight=1.0) for _ in range(2)]
    q1, q2 = fits[0].q, fits[1].q
    projector_error = float((q1 @ q1.T - q2 @ q2.T).abs().max())
    fit_cosine = float((q1[:, 0] @ q2[:, 0]).abs())
    replay = {key: parent.report_direction(backend, preps[key], q1)
              for key in ("heldout", "a2")}
    necessity = {key: donor_neutralization(backend, preps[key], q1)
                 for key in ("heldout", "a2")}
    scale = g.target_scale(preps["fit"])
    controls = {key: donor_control_effect(backend, preps[key], q1, scale) for key in ("p", "c")}
    vector = q1[:, 0].detach().cpu().float().contiguous()
    axis_bytes = vector.numpy().astype("<f4", copy=False).tobytes()
    axis = {"coordinates": [float(value) for value in vector], "dtype": "float32_le",
            "sha256": hashlib.sha256(axis_bytes).hexdigest(), "norm": float(vector.norm()),
            "coordinate_count": int(vector.numel())}
    old_fit = old["fit"]["best_objective"]["joint"]
    pred_a = bool(all(abs(fit.best_objective["joint"] - old_fit) <= 1e-6 for fit in fits)
                  and projector_error <= 1e-6 and fit_cosine >= 0.999999)
    pred_b = all(abs(replay[key][metric] - old["reports"]["cdas"][key][metric]) <= 0.01
                 for key in ("heldout", "a2")
                 for metric in ("subspace_fraction", "complement_fraction"))
    pred_c = all(necessity[key]["mean_recovery_toward_base"] >= 0.25
                 and necessity[key]["direction_fraction_toward_base"] >= 0.75
                 for key in ("heldout", "a2"))
    pred_d = controls["p"] <= 0.20 and controls["c"] <= 0.35
    pred_e = bool(axis["coordinate_count"] == 128 and abs(axis["norm"] - 1.0) <= 1e-5
                  and all(math.isfinite(value) for value in axis["coordinates"])
                  and hashlib.sha256(axis_bytes).hexdigest() == axis["sha256"])
    pred_f = bool(MODEL_FORWARDS <= 1880 and EXAMPLE_EVALUATIONS <= 31500
                  and BACKWARD_FORWARDS <= 1800 and MODEL_UPDATES == 900)
    predictions = {"pred_a_fit_and_projector_reproduce": pred_a,
                   "pred_b_sufficiency_reproduction": pred_b,
                   "pred_c_bidirectional_necessity": pred_c,
                   "pred_d_selective_neutralization": pred_d,
                   "pred_e_executable_axis_artifact": pred_e,
                   "pred_f_managed_price": pred_f}
    terminal = "screen" if all(predictions.values()) else (
        "invalid" if not pred_a or not pred_e or not pred_f else
        "sufficiency_only" if pred_b and pred_d else "null")
    fit_reports = []
    for fit in fits:
        report = asdict(fit)
        report.pop("q")
        fit_reports.append(report)
    result = {"schema": "temporal_auxiliary_fresh_block11h3_cdas_axis_necessity_result_v2",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "fit_replays": fit_reports,
              "determinism": {"projector_max_abs_error": projector_error,
                              "absolute_cosine": fit_cosine},
              "axis_artifact": axis, "sufficiency_replay": replay,
              "donor_side_necessity": necessity, "donor_control_effects": controls,
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": MODEL_FORWARDS,
                        "example_evaluations": EXAMPLE_EVALUATIONS,
                        "transformer_backward_forwards": BACKWARD_FORWARDS,
                        "model_updates": MODEL_UPDATES}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "determinism",
          "sufficiency_replay", "donor_side_necessity", "donor_control_effects",
          "predictions", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
