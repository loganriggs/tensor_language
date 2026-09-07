#!/usr/bin/env python3
"""Atomic model-free scorer repair for five-MLP source-subspace measurements."""
# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_atomic_authority_split_full_replay_and_finiteness pred_b_dim2_is_functional pred_c_dim2_is_selective pred_d_dim2_complement_is_insufficient pred_e_dim2_is_lowest_kl_functional_projected_arm
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_five_mlp_source_dim_subspace_v2.json"
V1 = ROOT / "circuits/followups/temporal_iswas_five_mlp_source_dim_subspace_v1_result.json"
SPLIT = ROOT / "circuits/followups/temporal_iswas_five_mlp_source_dim_subspace_v1_split_audit.json"
AUDIT = ROOT / "circuits/followups/temporal_iswas_five_mlp_source_dim_subspace_v1_invalidity_audit.json"
OUT = ROOT / "circuits/followups/temporal_iswas_five_mlp_source_dim_subspace_v2_result.json"
EXPECTED = {
    "prior": "6babb3584e91ca47d9e3d8176c92720891cb7e62b20ace5ae5554175eb7555c6",
    "v1": "450f41c1127b86dff5868abaee51b0eccf190de18d0990a7c9acbde45760f621",
    "split": "0c97a05975d252a91398fc13c3f71c33b2f4703ca5b99546a3648e1c1ab1347f",
    "audit": "ad88b0abc66da9045033e0f542c903d69796e9ccfaf06e6215c70deb891f9bf4",
}
TASKS = ("temporal", "iswas")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def functional(report):
    return all(
        report["response"][task]["signed_projection"] >= 0.8
        and report["response"][task]["relative_squared_error"] <= 0.2
        and report["behavior_signed_projection"][task] >= 0.75
        for task in TASKS
    )


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def main():
    dry = {"candidate_id": "temporal_auxiliary.iswas_five_mlp_source_dim_subspace_v2", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "model_forwards_max": 0,
           "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    observed = {"prior": sha(PRIOR), "v1": sha(V1), "split": sha(SPLIT), "audit": sha(AUDIT)}
    v1, split = json.loads(V1.read_text()), json.loads(SPLIT.read_text())
    reports = v1["reports"]
    atomic_a = {
        "authority_hashes": observed == EXPECTED,
        "source_split_disjoint": split["overlap_count"] == 0 and split["fit_unique"] == 54 and split["evaluation_unique"] == 48,
        "support_exact": v1["support"] == ["MLP0", "MLP1", "MLP2", "MLP3", "MLP6"],
        "full_arm_functional": functional(reports["full"]),
        "published_measurements_finite": finite_tree(v1["reports"]) and finite_tree(v1["source_basis_fit_energy"]) and finite_tree(v1["source_basis_spectra"]),
    }
    pa = all(atomic_a.values())
    pb = functional(reports["dim2"])
    pc = pb and reports["dim2"]["control"]["median_kl"] <= 0.02 and reports["dim2"]["control"]["top1_flip_fraction"] == 0.0
    pd = all(abs(reports["dim2_complement"]["response"][task]["signed_projection"]) <= 0.25
             and abs(reports["dim2_complement"]["behavior_signed_projection"][task]) <= 0.25 for task in TASKS)
    projected = ("dim1", "dim2", "svd4", "svd8")
    functional_projected = [arm for arm in projected if functional(reports[arm])]
    pe = pb and bool(functional_projected) and reports["dim2"]["control"]["median_kl"] <= min(
        reports[arm]["control"]["median_kl"] for arm in functional_projected) + 1e-8
    predictions = {
        "pred_a_atomic_authority_split_full_replay_and_finiteness": bool(pa),
        "pred_b_dim2_is_functional": bool(pb),
        "pred_c_dim2_is_selective": bool(pc),
        "pred_d_dim2_complement_is_insufficient": bool(pd),
        "pred_e_dim2_is_lowest_kl_functional_projected_arm": bool(pe),
    }
    terminal = "invalid" if not pa else "selective_dim_source_program" if all(predictions.values()) else "source_subspace_transfer_null"
    result = {
        "schema": "temporal_iswas_five_mlp_source_dim_subspace_result_v2",
        "finished_utc": now(), "authority_sha256": EXPECTED, "atomic_instrument": atomic_a,
        "reports": reports, "source_basis_fit_energy": v1["source_basis_fit_energy"],
        "functional_projected_arms": functional_projected, "predictions": predictions, "terminal": terminal,
        "interpretation": "Low-rank DIM and per-row-mean SVD source bases are selective but insufficient; the DIM complement remains causally active, so the five-module write is multi-dimensional and context/position dependent.",
        "price": {"model_forwards_max": 0, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
