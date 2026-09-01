"""RUNG 302B -- REPAIR THE INERT DELIMITER-STATE SHUFFLE CONTROL.

Rung 302 used ``np.roll(labels, 3)`` as its state null.  On the balanced,
alternating two-state population an odd roll is exactly the complementary
partition, so the two centroid sets merely swap names and shuffled R2 is
bit-identical to real R2.  That control tests nothing.

Rerun the otherwise unchanged natural-prefix/Hankel experiment, but replace
the inert roll with 64 seeded balanced random permutations.  Reject any draw
whose partition or complement agrees with more than 75% of the true labels.
For every draw report both in-sample centroid R2 and the original disjoint-half
nearest-centroid accuracy.

Frozen predictions
------------------
pred_a_shuffle_is_live:
    All 64 draws are retained and no partition/complement agreement exceeds
    .75 in any report.
pred_b_real_state_exceeds_shuffle_r2:
    In every task/corpus native arm, real state R2 exceeds the shuffled R2
    95th percentile by at least .05.
pred_c_real_classifier_exceeds_shuffle:
    In every task/corpus native arm, real heldout accuracy exceeds mean
    shuffled heldout accuracy by at least .10.

Null: any native real R2 fails to exceed shuffled median, or any real heldout
accuracy fails to exceed shuffled mean.  This repair cannot rescue rung 302's
failed rank, causal-head, suffix-NLL, or literal-price bars; it only determines
whether the reported state association was above a live label null.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "delimiter_predictive_state_control_repair_results.json"
N_SHUFFLES = 64
SEED = 30220260901


def _make_report(base, matrix: np.ndarray, labels: np.ndarray,
                 suffixes: list[tuple[int, ...]]) -> dict[str, object]:
    centered = matrix - matrix.mean(0, keepdims=True)
    centroids = np.stack([centered[labels == state].mean(0) for state in (0, 1)])
    predicted = centroids[labels]
    denominator = max(np.square(centered).sum(), 1e-20)
    state_r2 = 1.0 - np.square(centered - predicted).sum() / denominator
    separation = float(np.linalg.norm(centroids[1] - centroids[0]))
    fit = np.arange(len(labels)) % 4 < 2
    test = ~fit
    fit_centroids = np.stack([centered[np.logical_and(fit, labels == state)].mean(0) for state in (0, 1)])
    distance = np.square(centered[test, None, :] - fit_centroids[None, :, :]).sum(2)
    heldout_accuracy = float((distance.argmin(1) == labels[test]).mean())

    rng = np.random.default_rng(SEED)
    shuffled_r2: list[float] = []
    shuffled_accuracy: list[float] = []
    agreements: list[float] = []
    attempts = 0
    while len(shuffled_r2) < N_SHUFFLES:
        attempts += 1
        assert attempts < 10000
        shuffled = rng.permutation(labels)
        agreement = float((shuffled == labels).mean())
        partition_agreement = max(agreement, 1.0 - agreement)
        if partition_agreement > 0.75:
            continue
        shuffle_centroids = np.stack([centered[shuffled == state].mean(0) for state in (0, 1)])
        shuffle_predicted = shuffle_centroids[shuffled]
        shuffled_r2.append(float(1.0 - np.square(centered - shuffle_predicted).sum() / denominator))
        shuffle_fit_centroids = np.stack([
            centered[np.logical_and(fit, shuffled == state)].mean(0) for state in (0, 1)
        ])
        shuffle_distance = np.square(centered[test, None, :] - shuffle_fit_centroids[None, :, :]).sum(2)
        shuffled_accuracy.append(float((shuffle_distance.argmin(1) == shuffled[test]).mean()))
        agreements.append(partition_agreement)

    nested = {}
    for maximum in (1, 2, 3):
        columns = [index for index, suffix in enumerate(suffixes) if len(suffix) <= maximum]
        nested[str(maximum)] = {
            "n_suffixes": len(columns),
            "interaction_rank90": base._rank90(base._interaction(matrix[:, columns])),
        }
    r2 = np.asarray(shuffled_r2)
    accuracy = np.asarray(shuffled_accuracy)
    return {
        "state_r2": float(state_r2),
        "shuffled_state_r2": float(r2.mean()),
        "state_separation": separation,
        "heldout_accuracy": heldout_accuracy,
        "nested_hankel": nested,
        "centroids": centroids.tolist(),
        "live_shuffle_control": {
            "seed": SEED,
            "draws": N_SHUFFLES,
            "attempts": attempts,
            "max_partition_or_complement_agreement": max(agreements),
            "state_r2_mean": float(r2.mean()),
            "state_r2_median": float(np.median(r2)),
            "state_r2_p95": float(np.quantile(r2, 0.95)),
            "state_r2_min": float(r2.min()),
            "state_r2_max": float(r2.max()),
            "heldout_accuracy_mean": float(accuracy.mean()),
            "heldout_accuracy_p95": float(np.quantile(accuracy, 0.95)),
            "heldout_accuracy_min": float(accuracy.min()),
            "heldout_accuracy_max": float(accuracy.max()),
        },
    }


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n96_skip1200.pt").exists()
        assert N_SHUFFLES >= 32 and 0 <= SEED
        print("DELIMITER STATE CONTROL REPAIR | dry run: live shuffle and bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    import delimiter_predictive_state_hankel as base

    base.OUT = OUT
    base._state_report = lambda matrix, labels, suffixes: _make_report(base, matrix, labels, suffixes)
    base.main()
    result = json.loads(OUT.read_text())
    native = [result["reports"][corpus][task]["native"]
              for corpus in ("fineweb", "wikitext") for task in ("quote", "parenthesis")]
    pred_a = all(row["live_shuffle_control"]["draws"] == N_SHUFFLES
                 and row["live_shuffle_control"]["max_partition_or_complement_agreement"] <= 0.75
                 for row in native)
    pred_b = all(row["state_r2"] >= row["live_shuffle_control"]["state_r2_p95"] + 0.05
                 for row in native)
    pred_c = all(row["heldout_accuracy"] >= row["live_shuffle_control"]["heldout_accuracy_mean"] + 0.10
                 for row in native)
    null = any(row["state_r2"] <= row["live_shuffle_control"]["state_r2_median"]
               or row["heldout_accuracy"] <= row["live_shuffle_control"]["heldout_accuracy_mean"]
               for row in native)
    result.update({
        "status": "delimiter_predictive_state_control_repair_complete",
        "rung": "302B",
        "claim_level": "corrected_label_null_for_rung302_only",
        "parent_predicates": {
            "small_predictive_state": result.pop("pred_a_small_predictive_state"),
            "state_transfers": result.pop("pred_b_state_transfers"),
            "delimiter_head_carries_state": result.pop("pred_c_delimiter_head_carries_state"),
            "no_small_causal_state_null": result.pop("null_no_small_causal_state"),
        },
        'pred_a_shuffle_is_live': bool(pred_a),
        'pred_b_real_state_exceeds_shuffle_r2': bool(pred_b),
        'pred_c_real_classifier_exceeds_shuffle': bool(pred_c),
        "null_no_state_above_live_shuffle": bool(null),
    })
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"repair_predicates": [pred_a, pred_b, pred_c], "repair_null": null}, indent=2))
    print("DELIMITER STATE CONTROL REPAIR DONE")


if __name__ == "__main__":
    main()
