#!/usr/bin/env python3
"""Model-free scorer correction for the shared-eight pairwise Möbius pilot."""
# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_and_v1_measurements_valid pred_b_singleton_nonadditivity_reproduces pred_c_nontrivial_pair_interaction_both_tasks pred_d_degree_two_is_executable_both_tasks pred_e_pair_interactions_are_sparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank46_shared8_pairwise_mobius_v2.json"
V1 = ROOT / "circuits/followups/temporal_iswas_rank46_shared8_pairwise_mobius_v1_result.json"
AUDIT = ROOT / "circuits/followups/temporal_iswas_rank46_shared8_pairwise_mobius_v1_invalidity_audit.json"
OUT = ROOT / "circuits/followups/temporal_iswas_rank46_shared8_pairwise_mobius_v2_result.json"
EXPECTED = {
    "prior": "75fed0535b4b8d4d676ee1c0ceb6339278b7ba08c71f66c12e75859d17854746",
    "v1": "4b9a9cdb06b3eab1b842f037d80822814187bf821ee1077c96880d603717f19c",
    "audit": "d5f150b68fe29775f7003dbf85962cb35b7743028bd071c5eb0f11efee3c0b81",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    dry = {
        "candidate_id": "temporal_auxiliary.iswas_rank46_shared8_pairwise_mobius_v2",
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "model_forwards_max": 0,
        "fit_updates": 0,
        "model_updates": 0,
        "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    observed = {"prior": sha(PRIOR), "v1": sha(V1), "audit": sha(AUDIT)}
    v1 = json.loads(V1.read_text())
    tasks = ("temporal", "iswas")
    pa = (
        observed == EXPECTED
        and v1["terminal"] == "sparse_degree_two_interaction_map"
        and v1["subset_count"] == 38
        and all(math.isfinite(float(v1["pairwise_sum"][task][metric])) for task in tasks
                for metric in ("signed_projection", "relative_squared_error", "norm_ratio"))
    )
    pb = all(
        v1["singleton_sum"][task]["signed_projection"] >= 1.5
        and v1["singleton_sum"][task]["relative_squared_error"] >= 1.0
        for task in tasks
    )
    pc = min(v1["max_pair_norm_ratio"].values()) >= 0.05
    pd = all(
        0.8 <= v1["pairwise_sum"][task]["signed_projection"] <= 1.2
        and v1["pairwise_sum"][task]["relative_squared_error"] <= 0.2
        for task in tasks
    )
    pe = v1["top8_pair_mass_fraction"] >= 0.50
    predictions = {
        "pred_a_authority_and_v1_measurements_valid": bool(pa),
        "pred_b_singleton_nonadditivity_reproduces": bool(pb),
        "pred_c_nontrivial_pair_interaction_both_tasks": bool(pc),
        "pred_d_degree_two_is_executable_both_tasks": bool(pd),
        "pred_e_pair_interactions_are_sparse": bool(pe),
    }
    terminal = "invalid" if not pa else "sparse_degree_two_interaction_map" if all(predictions.values()) else "higher_order_or_ordered_interaction_boundary"
    result = {
        "schema": "temporal_iswas_rank46_shared8_pairwise_mobius_result_v2",
        "finished_utc": now(),
        "authority_sha256": EXPECTED,
        "source_result": str(V1.relative_to(ROOT)),
        "singleton_sum": v1["singleton_sum"],
        "pairwise_sum": v1["pairwise_sum"],
        "pairwise_rse_improvement": v1["pairwise_rse_improvement"],
        "max_pair_norm_ratio": v1["max_pair_norm_ratio"],
        "top8_pair_mass_fraction": v1["top8_pair_mass_fraction"],
        "top_pairs": v1["top_pairs"],
        "predictions": predictions,
        "terminal": terminal,
        "interpretation": "Pair interactions are large and norm-concentrated, but their complete degree-two sum is not executable. Ordered degree-three-or-higher recurrence or normalization-mediated interactions dominate.",
        "price": {"model_forwards_max": 0, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
