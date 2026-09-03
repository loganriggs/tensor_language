#!/usr/bin/env python3
"""CPU-only post-result structural audit of R556."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
RESULT = ROOT / "pending_opener_target_invariance_das_rung556_results.json"
BUNDLE = ROOT / "pending_opener_target_invariance_das_rung556_projectors.pt"
OUT = ROOT / "pending_opener_target_invariance_das_rung559_audit.json"
EXPECTED_RESULT_SHA256 = "d091844838515753c4b131dd7e79722f1814de2e30de1950c5100e7233a84bdd"
EXPECTED_BUNDLE_SHA256 = "94d3ef675f2089ea02d08fad321da8e98db058c94f808da8a0a898b7e8ff3a4a"
TARGETS = ("direct_three_value_type_substitution", "completed_then_reopened_three_value_order")
CONTROLS = (
    "pending_type_preserved_surface_rewrite",
    "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
)
RANKS = (1, 2, 4, 8, 16)
SEEDS = (0, 1, 2)
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 556


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_lower(values: list[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[choices].mean(1), .025))


def close(left: float, right: float, tolerance: float = 1e-12) -> None:
    if abs(left - right) > tolerance:
        raise RuntimeError(f"summary mismatch: {left} != {right}")


def main() -> None:
    if sha256(RESULT) != EXPECTED_RESULT_SHA256 or sha256(BUNDLE) != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError("R556 result or projector bundle changed")
    result = json.loads(RESULT.read_text())
    reports = result["fits"]
    stable_by_rank = {}
    fit_index = 0
    target_cells = control_cells = 0
    for rank in RANKS:
        passing_seeds = 0
        for seed in SEEDS:
            report = reports[str(rank)][str(seed)]
            passed = True
            cell_seed = BOOTSTRAP_SEED + 20 * fit_index
            for family in TARGETS:
                for direction in ("base_to_donor", "donor_to_base"):
                    cell = report["targets"][family][direction]
                    values = np.asarray(cell["values"], dtype=np.float64)
                    assert cell["n"] == len(values) == 36
                    close(cell["mean"], float(values.mean()))
                    close(cell["median"], float(np.median(values)))
                    close(cell["positive_fraction"], float(np.mean(values > 0)))
                    close(cell["bootstrap95_lower_mean"], bootstrap_lower(values.tolist(), cell_seed))
                    computed = bool(float(np.median(values)) >= .5
                                    and cell["bootstrap95_lower_mean"] > 0
                                    and float(np.mean(values > 0)) >= .75)
                    assert cell["passed"] is computed
                    passed &= computed
                    cell_seed += 1
                    target_cells += 1
            for family in CONTROLS:
                for direction in ("base_to_donor", "donor_to_base"):
                    cell = report["controls"][family][direction]
                    endpoint = np.asarray(cell["endpoint_values"], dtype=np.float64)
                    rms = np.asarray(cell["full_vocabulary_logit_rms_values"], dtype=np.float64)
                    assert cell["n"] == len(endpoint) == len(rms) == 36
                    close(cell["mean_absolute_closer_margin_change"], float(np.mean(np.abs(endpoint))))
                    close(cell["mean_full_vocabulary_logit_rms"], float(np.mean(rms)))
                    assert np.isfinite(cell["fraction_of_complete_head_margin_change"])
                    assert np.isfinite(cell["fraction_of_complete_head_full_vocabulary_rms"])
                    computed = bool(
                        cell["mean_absolute_closer_margin_change"] <= .10
                        and cell["fraction_of_complete_head_margin_change"] <= .25
                        and cell["fraction_of_complete_head_full_vocabulary_rms"] <= .25
                    )
                    assert cell["passed"] is computed
                    passed &= computed
                    control_cells += 1
            assert report["passed"] is bool(passed)
            passing_seeds += int(passed)
            fit_index += 1
        stable = passing_seeds >= 2
        assert reports[str(rank)]["seed_stable"] is stable
        random = result["random_controls"][str(rank)]
        close(random["mean_across_seeds"], float(np.mean(random["mean_target_recovery_by_seed"])))
        random_passed = random["mean_across_seeds"] < .10
        assert random["passed"] is random_passed
        eligible = bool(stable and random_passed)
        assert reports[str(rank)]["rank_eligible"] is eligible
        stable_by_rank[str(rank)] = {
            "passing_seeds": passing_seeds,
            "seed_stable": stable,
            "random_control_passed": random_passed,
            "eligible": eligible,
        }

    eligible = [rank for rank in RANKS if stable_by_rank[str(rank)]["eligible"]]
    selected = min(eligible) if eligible else None
    assert result["selected_rank"] == selected
    assert result["pred_b_stable_selective_projector_exists"] is (selected is not None)
    assert result["pred_c_random_subspaces_below_bar"] is all(
        result["random_controls"][str(rank)]["passed"] for rank in RANKS
    )
    assert result["strong_null"] is (selected is None)
    assert result["pred_a_exact_instrument"] is True
    assert result["native_model_forwards"] == 68
    assert result["gradient_suffix_evaluations"] == 3600
    assert result["no_gradient_suffix_evaluations"] == 675
    assert result["model_forwards"] == 4343 and result["model_backwards"] == 3600
    assert result["model_weights_updated"] is False
    assert result["evaluated_splits"] == ["FIT", "SELECT"]
    assert result["forbidden_splits_opened"] == []
    assert result["bundle_sha256"] == EXPECTED_BUNDLE_SHA256
    audit = {
        "rung": 559,
        "audited_rung": 556,
        "status": "terminal_structural_audit_complete",
        "result_sha256": EXPECTED_RESULT_SHA256,
        "bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "target_cells_recomputed": target_cells,
        "control_cells_recomputed": control_cells,
        "stable_by_rank": stable_by_rank,
        "selected_rank": selected,
        "strong_null_recomputed": selected is None,
        "execution_budget_exact": True,
        "split_opening_exact": True,
        "model_forwards": 0,
        "model_backwards": 0,
        "limitation": (
            "R556 did not save row-level complete-head denominators, so the audit reapplies rather than independently "
            "reconstructs the two saved normalized control ratios."
        ),
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
