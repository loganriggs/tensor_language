"""Deterministic analyzer over the published validation table (Amendment 16, last step).

Reads ONLY the receipt-bound validation table. Forms the preregistered Pareto frontier
over `(persistent values, per-document values, calibrated validation MSE, worst-owner
NRMSE)` per design and budget, keeps the pooled and block-relative views separately,
and records two facts the scorer's raw numbers hide:

1. a calibration panel that anchors an ENTIRE owner block leaves zero scored cells for
   that block, so its pooled/worst numbers are computed on an easier population than a
   panel that keeps every block - such panels are marked `complete_owner_coverage: false`
   and are ineligible for the block-balanced frontier;
2. `worst_owner_pair_nrmse` in the scorer silently skips empty pairs (max over NaN); it
   is recomputed here over scored pairs only and the count of scored pairs is retained.

Selects no winner. Publishes create-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
TABLE = HERE / "causal_response_factorization_v1_validation_table.json"
RECEIPT = HERE / "causal_response_factorization_v1_validation_terminal" / "receipt.json"
OUTPUT = HERE / "causal_response_factorization_v1_validation_analysis.json"
SCHEMA = "causal_response_factorization_v1_validation_analysis"
FRONTIER_COORDINATES = (
    "persistent_values", "per_document_values",
    "calibrated_pooled_mse_median", "worst_owner_pair_nrmse_median",
)
UNCONDITIONAL_FAILURE_NRMSE = 0.95


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def logical_sha256(value: object) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode())


def _median(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.median(finite) if finite else None


def _range(values: list[float | None]) -> dict[str, float | None]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return {"min": min(finite) if finite else None, "max": max(finite) if finite else None}


def _pair_summary(report: Mapping[str, Any], owner_count: int) -> dict[str, Any]:
    """Recompute the worst pair over SCORED pairs and retain the coverage census."""

    pairs = report["owner_pairs"]
    scored = {key: value for key, value in pairs.items() if value["cells"] > 0}
    worst = max(
        (value["nrmse_by_training_rms"] for value in scored.values()
         if value["nrmse_by_training_rms"] is not None), default=None,
    )
    worst_key = None
    if worst is not None:
        worst_key = next(
            key for key, value in scored.items() if value["nrmse_by_training_rms"] == worst
        )
    return {
        "pooled_mse": report["pooled"]["mse"],
        "pooled_nrmse": report["pooled"]["nrmse_by_training_rms"],
        "pooled_signed_correlation": report["pooled"]["signed_correlation"],
        "pooled_cells": report["pooled"]["cells"],
        "owner_pairs_scored": len(scored),
        "owner_pairs_total": owner_count * owner_count,
        "complete_owner_coverage": len(scored) == owner_count * owner_count,
        "worst_owner_pair_nrmse_over_scored_pairs": worst,
        "worst_owner_pair": worst_key,
        "scorer_worst_owner_pair_nrmse": report["worst_owner_pair_nrmse"],
    }


def _dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    keys = FRONTIER_COORDINATES
    return all(left[key] <= right[key] for key in keys) and any(
        left[key] < right[key] for key in keys
    )


def _frontier(points: Mapping[str, Mapping[str, float]]) -> list[str]:
    ids = sorted(points)
    return [
        left for left in ids
        if not any(_dominates(points[right], points[left]) for right in ids if right != left)
    ]


def analyze(table: Mapping[str, Any]) -> dict[str, Any]:
    owners = list(table["owner_components"])
    owner_count = len(owners)
    designs = list(table["designs"])
    budgets = [str(value) for value in table["calibration_arm_budgets"]]
    by_pair: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in table["candidates"]:
        by_pair.setdefault((row["global_rank"], row["private_rank_each_owner"]), []).append(row)

    rank_pairs: list[dict[str, Any]] = []
    frontier_points: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        design: {budget: {} for budget in budgets} for design in designs
    }
    unconditional_points: dict[str, dict[str, float]] = {}
    for (global_rank, private_rank), rows in sorted(by_pair.items()):
        pair_id = f"g{global_rank}_p{private_rank}"
        seeds = [row["seed"] for row in rows]
        unconditional = [_pair_summary(row["unconditional"], owner_count) for row in rows]
        entry: dict[str, Any] = {
            "rank_pair": [global_rank, private_rank], "seeds": seeds,
            "persistent_values": rows[0]["persistent_values"],
            "per_document_values": rows[0]["per_document_values"],
            "unconditional": {
                "pooled_nrmse_median": _median([u["pooled_nrmse"] for u in unconditional]),
                "pooled_nrmse_range": _range([u["pooled_nrmse"] for u in unconditional]),
                "pooled_mse_median": _median([u["pooled_mse"] for u in unconditional]),
                "signed_correlation_median": _median(
                    [u["pooled_signed_correlation"] for u in unconditional]
                ),
                "worst_owner_pair_nrmse_median": _median(
                    [u["worst_owner_pair_nrmse_over_scored_pairs"] for u in unconditional]
                ),
                "worst_owner_pairs": [u["worst_owner_pair"] for u in unconditional],
                "complete_owner_coverage": all(u["complete_owner_coverage"] for u in unconditional),
                "per_seed": unconditional,
            },
            "calibrated": {},
        }
        if entry["unconditional"]["complete_owner_coverage"]:
            unconditional_points[pair_id] = {
                "persistent_values": entry["persistent_values"],
                "per_document_values": entry["per_document_values"],
                "calibrated_pooled_mse_median": entry["unconditional"]["pooled_mse_median"],
                "worst_owner_pair_nrmse_median": entry["unconditional"]["worst_owner_pair_nrmse_median"],
            }
        for design in designs:
            entry["calibrated"][design] = {}
            for budget in budgets:
                panels = [row["calibrated"][design][budget] for row in rows]
                scored = [panel for panel in panels if panel["status"] == "scored"]
                summaries = [_pair_summary(panel["calibrated"], owner_count) for panel in scored]
                support = [panel["supported_document_fraction"] for panel in scored]
                eligible_seeds = [
                    panel["support_gate_passes"] and summary["complete_owner_coverage"]
                    for panel, summary in zip(scored, summaries)
                ]
                cell = {
                    "seeds_scored": len(scored),
                    "seeds_failed": len(panels) - len(scored),
                    "support_fraction_median": _median(support),
                    "support_gate_passes_all_seeds": bool(scored) and all(
                        panel["support_gate_passes"] for panel in scored
                    ),
                    "owner_pairs_scored_per_seed": [s["owner_pairs_scored"] for s in summaries],
                    "complete_owner_coverage_all_seeds": bool(summaries) and all(
                        s["complete_owner_coverage"] for s in summaries
                    ),
                    "pooled_nrmse_median": _median([s["pooled_nrmse"] for s in summaries]),
                    "pooled_nrmse_range": _range([s["pooled_nrmse"] for s in summaries]),
                    "pooled_mse_median": _median([s["pooled_mse"] for s in summaries]),
                    "signed_correlation_median": _median(
                        [s["pooled_signed_correlation"] for s in summaries]
                    ),
                    "worst_owner_pair_nrmse_median": _median(
                        [s["worst_owner_pair_nrmse_over_scored_pairs"] for s in summaries]
                    ),
                    "worst_owner_pairs": [s["worst_owner_pair"] for s in summaries],
                    "eligible_for_block_balanced_frontier": (
                        len(scored) == len(panels) and bool(eligible_seeds) and all(eligible_seeds)
                    ),
                    "per_seed": summaries,
                }
                entry["calibrated"][design][budget] = cell
                if cell["eligible_for_block_balanced_frontier"]:
                    frontier_points[design][budget][pair_id] = {
                        "persistent_values": entry["persistent_values"],
                        "per_document_values": entry["per_document_values"],
                        "calibrated_pooled_mse_median": cell["pooled_mse_median"],
                        "worst_owner_pair_nrmse_median": cell["worst_owner_pair_nrmse_median"],
                    }
        rank_pairs.append(entry)

    frontiers = {
        design: {
            budget: {
                "eligible_rank_pairs": sorted(points),
                "nondominated_rank_pairs": _frontier(points),
                "coordinates": list(FRONTIER_COORDINATES),
            }
            for budget, points in per_budget.items()
        }
        for design, per_budget in frontier_points.items()
    }
    unconditional_frontier = {
        "eligible_rank_pairs": sorted(unconditional_points),
        "nondominated_rank_pairs": _frontier(unconditional_points),
        "coordinates": list(FRONTIER_COORDINATES),
        "note": "unconditional arm: mse is the mean-training-code prediction error",
    }
    unconditional_nrmse = [
        entry["unconditional"]["pooled_nrmse_median"] for entry in rank_pairs
    ]
    has_independent_only = any(entry["rank_pair"][0] == 0 for entry in rank_pairs)
    fixed_population = {}
    for budget in budgets:
        blind = {
            "g{}_p{}".format(*entry["rank_pair"]): entry["calibrated"]["sha256_outcome_blind_blocks"][budget]
            for entry in rank_pairs
        } if "sha256_outcome_blind_blocks" in designs else {}
        fixed_population[budget] = {
            pair_id: {
                "pooled_nrmse_median": cell["pooled_nrmse_median"],
                "worst_owner_pair_nrmse_median": cell["worst_owner_pair_nrmse_median"],
                "complete_owner_coverage_all_seeds": cell["complete_owner_coverage_all_seeds"],
            }
            for pair_id, cell in blind.items()
        }
    return {
        "schema": SCHEMA,
        "status": "complete_deterministic_analysis_no_selection",
        "owner_components": owners,
        "designs": designs,
        "calibration_arm_budgets": budgets,
        "training_response_rms": table["training_response_rms"],
        "rank_pairs": rank_pairs,
        "block_balanced_frontiers": frontiers,
        "unconditional_frontier": unconditional_frontier,
        "fixed_population_view": {
            "design": "sha256_outcome_blind_blocks",
            "note": (
                "the outcome-blind design anchors the same arms for every candidate, so "
                "its scored population is identical across candidates at a budget"
            ),
            "by_budget": fixed_population,
        },
        "prospective_failure_pattern": {
            "unconditional_failure_nrmse_bar": UNCONDITIONAL_FAILURE_NRMSE,
            "unconditional_pooled_nrmse_median_by_pair": dict(zip(
                ["g{}_p{}".format(*entry["rank_pair"]) for entry in rank_pairs],
                unconditional_nrmse,
            )),
            "unconditional_fails_broadly": all(
                value is not None and value >= UNCONDITIONAL_FAILURE_NRMSE
                for value in unconditional_nrmse
            ),
            "hierarchy_support": (
                "untestable: the frozen library contains no independent-only "
                "(global rank 0) candidate" if not has_independent_only else "tested"
            ),
        },
        "candidate_selected": False,
        "eval_values_read": False,
    }


def publish_create_only(value: Mapping[str, Any], output: Path) -> bytes:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if output.exists():
        raise RuntimeError("validation analysis output is already spent")
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(raw); sink.flush(); os.fsync(sink.fileno())
        if json.loads(temporary.read_bytes()) != json.loads(raw):
            raise RuntimeError("staged validation analysis does not replay")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return raw


def main() -> None:
    table_raw = TABLE.read_bytes()
    receipt = json.loads(RECEIPT.read_bytes())
    table_digest = sha256(table_raw)
    if receipt.get("kind") != "receipt" or receipt["payload"].get("table_sha256") != table_digest:
        raise RuntimeError("validation table is not the receipt-bound artifact")
    table = json.loads(table_raw)
    if table.get("candidate_selected") is not False:
        raise RuntimeError("validation table already carries a selection")
    body = {
        **analyze(table),
        "table_artifact_sha256": table_digest,
        "table_logical_sha256": table["table_sha256"],
        "receipt_artifact_sha256": sha256(RECEIPT.read_bytes()),
        "analysis_source_sha256": sha256(Path(__file__).read_bytes()),
    }
    value = {**body, "analysis_sha256": logical_sha256(body)}
    publish_create_only(value, OUTPUT)
    print(sha256(OUTPUT.read_bytes()))


if __name__ == "__main__":
    main()
