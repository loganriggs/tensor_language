#!/usr/bin/env python3
"""Deterministic FIT-only summary of the completed causal-response factor grid."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
GRID = HERE / "causal_response_factorization_v1_grid_results" / "terminal.json"
OUTPUT = HERE / "causal_response_factorization_v1_training_analysis.json"
OWNER_LABELS = ("a8", "a16", "m16", "a3", "m14", "m13")
OWNER_GROUP_SIZES = (16, 13, 6, 5, 5, 4)
SEEDS = (2026083001, 2026083002, 2026083003)
RANK_PAIRS = (
    *((rank, 0) for rank in (1, 2, 4, 8, 16, 32)),
    *((0, rank) for rank in (1, 2, 4, 8)),
    (1, 1), (2, 1), (4, 1), (4, 2), (8, 2), (8, 4), (16, 4),
)
STEPS = 2_000
LEARNING_RATE = 0.03
NUMERICAL_FAILURE_MESSAGES = {
    "accelerated shared/private optimizer became nonfinite",
    "accelerated canonical replay ended nonfinite",
}
MINIMUM_IMPROVEMENT = 1e-4
RESULT_CELL_KEYS = {
    "source_closure_sha256", "input_binding_sha256", "global_rank",
    "private_rank_each_owner", "seed", "steps", "learning_rate", "optimizer_device",
    "persistent_values", "per_document_values", "amortized_total_values",
    "strict_dense_matched_rank", "amortized_total_dense_rank_noncontrolling",
    "prediction_multiply_adds_per_document", "calibration_cells_training_stage",
    "registered_validation_calibration_arm_budgets",
    "registered_validation_calibration_costs", "initial_mse", "final_mse",
    "improvement_fraction", "healthy", "minimum_improvement", "elapsed_seconds",
    "validation_values_read", "eval_values_read", "training_response_rms",
    "normalized_training_mse", "phase_mse", "source_owner_mse", "target_owner_mse",
    "owner_pair_nrmse", "worst_owner_pair_nrmse", "kind", "artifact",
    "artifact_sha256", "bytes",
}
FAILURE_CELL_KEYS = {
    "schema", "status", "source_closure_sha256", "input_binding_sha256",
    "global_rank", "private_rank_each_owner", "seed", "steps", "learning_rate",
    "optimizer_device", "elapsed_seconds", "error_type", "error_message",
    "validation_values_read", "eval_values_read", "kind", "artifact",
    "artifact_sha256", "bytes",
}
ANALYSIS_SOURCE_PATHS = (
    Path("basis_aligned/polynomial_causal/causal_response_factorization_v1_training_analysis.py"),
    Path("basis_aligned/polynomial_causal/test_causal_response_factorization_v1_training_analysis.py"),
    Path("basis_aligned/polynomial_causal/causal_response_factorization_v1_grid_runner.py"),
    Path("basis_aligned/polynomial_causal/test_causal_response_factorization_v1_grid_runner.py"),
    Path("basis_aligned/polynomial_causal/causal_response_factorization_v1_grid_independent_audit.json"),
    Path("basis_aligned/polynomial_causal/causal_response_factorization_v1_candidate_price_audit.py"),
    Path("basis_aligned/polynomial_causal/causal_response_factorization_v1_candidate_price_audit.json"),
    Path("basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_PREREGISTRATION.md"),
    *(Path(f"basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_{index}.md") for index in range(1, 14)),
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def logical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return sha256(raw)


def structured_price(global_rank: int, private_rank: int) -> tuple[int, int]:
    persistent = global_rank * 100 + private_rank * sum(
        2 + owner_sources + 49 for owner_sources in OWNER_GROUP_SIZES
    )
    code = global_rank + len(OWNER_GROUP_SIZES) * private_rank
    return persistent, code


def rank_id(row: dict[str, object]) -> str:
    return f"g{int(row['global_rank']):02d}_p{int(row['private_rank_each_owner']):02d}"


def dominates(
    left: dict[str, object], right: dict[str, object], errors: tuple[str, ...]
) -> bool:
    keys = ("persistent_values", "per_document_values", *errors)
    le = all(float(left[key]) <= float(right[key]) for key in keys)
    strict = any(float(left[key]) < float(right[key]) for key in keys)
    return le and strict


def frontier(rows: list[dict[str, object]], errors: tuple[str, ...]) -> list[str]:
    """Return rank-pair points, never individual optimizer seeds."""
    return sorted(
        rank_id(row)
        for row in rows
        if not any(
            other is not row and dominates(other, row, errors)
            for other in rows
        )
    )


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def _cell_stem(global_rank: int, private_rank: int, seed: int) -> str:
    return f"g{global_rank:02d}_p{private_rank:02d}_s{seed}"


def _finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite")
    return float(value)


def analysis_source_closure() -> dict[str, object]:
    root = HERE.parents[1]
    hashes = {str(path): sha256((root / path).read_bytes()) for path in ANALYSIS_SOURCE_PATHS}
    body: dict[str, object] = {"paths": hashes}
    return {**body, "sha256": logical_sha256(body)}


def _validate_input_binding(value: object) -> dict[str, object]:
    keys = {
        "artifact_binding", "response_sha256", "valid_sha256", "document_ids_sha256",
        "original_document_indices_sha256", "source_groups_sha256", "shape",
        "owner_components", "phases", "source_tags", "target_tags",
        "validation_values_read", "eval_values_read", "sha256",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError("grid input-binding schema changed")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value["sha256"] != logical_sha256(body):
        raise RuntimeError("grid input-binding hash does not replay")
    if value["shape"] != [2, 49, 49, 229] or value["owner_components"] != list(OWNER_LABELS) or value["phases"] != ["full", "residual"]:
        raise RuntimeError("grid input topology changed")
    if len(value["source_tags"]) != 49 or value["source_tags"] != value["target_tags"]:
        raise RuntimeError("grid source/target topology changed")
    if value["validation_values_read"] is not False or value["eval_values_read"] is not False:
        raise RuntimeError("grid input role boundary is false")
    artifact_keys = {
        "parent_binding_sha256", "receipt_sha256", "terminal_sha256",
        "authority_artifact_sha256", "authority_logical_sha256", "bundle_sha256",
        "manifest_artifact_sha256", "manifest_logical_sha256", "source_closure_sha256",
    }
    artifacts = value["artifact_binding"]
    if not isinstance(artifacts, dict) or set(artifacts) != artifact_keys or any(
        not isinstance(item, str) or len(item) != 64 or any(c not in "0123456789abcdef" for c in item)
        for item in artifacts.values()
    ) or artifacts["receipt_sha256"] != artifacts["terminal_sha256"]:
        raise RuntimeError("grid artifact binding changed")
    return value


def validate_terminal(
    path: Path, *, expected_source: dict[str, object] | None = None
) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    terminal = json.loads(raw)
    body = {key: value for key, value in terminal.items() if key != "manifest_sha256"}
    exact_keys = {
        "schema", "status", "source_closure", "input_binding", "rank_pairs", "seeds",
        "steps", "learning_rate", "optimizer_device", "controls", "expected_cells",
        "result_cells", "failure_cells", "healthy_cells", "cells",
        "validation_values_read", "eval_values_read", "manifest_sha256",
    }
    if set(terminal) != exact_keys or terminal.get("schema") != "causal_response_factorization_v1_grid_terminal":
        raise RuntimeError("wrong grid schema")
    if terminal.get("status") != "complete_training_only_grid":
        raise RuntimeError("grid is not terminal")
    if logical_sha256(body) != terminal.get("manifest_sha256"):
        raise RuntimeError("grid manifest hash does not replay")
    if terminal.get("validation_values_read") or terminal.get("eval_values_read"):
        raise RuntimeError("grid role boundary is false")
    if expected_source is None:
        from causal_response_factorization_v1_grid_runner import (
            _source_closure, _validate_grid_audit,
        )
        expected_source = _source_closure(require_published=True)
        _validate_grid_audit(expected_source)
    if terminal.get("source_closure") != expected_source:
        raise RuntimeError("grid source closure changed")
    input_binding = _validate_input_binding(terminal.get("input_binding"))
    if terminal.get("steps") != STEPS or terminal.get("learning_rate") != LEARNING_RATE or terminal.get("optimizer_device") != "cuda":
        raise RuntimeError("grid optimizer protocol changed")
    if tuple(map(tuple, terminal.get("rank_pairs", ()))) != RANK_PAIRS:
        raise RuntimeError("grid rank pairs changed")
    if tuple(terminal.get("seeds", ())) != SEEDS:
        raise RuntimeError("grid seeds changed")

    expected = len(RANK_PAIRS) * len(SEEDS)
    cells = terminal.get("cells")
    if terminal.get("expected_cells") != expected or not isinstance(cells, list) or len(cells) != expected:
        raise RuntimeError("grid expected-cell census changed")
    seen: set[tuple[int, int, int]] = set()
    artifacts: set[str] = set()
    results = failures = healthy = 0
    for cell in cells:
        if not isinstance(cell, dict):
            raise RuntimeError("grid cell is not an object")
        ranks = (cell.get("global_rank"), cell.get("private_rank_each_owner"))
        seed = cell.get("seed")
        key = (*ranks, seed)
        if ranks not in RANK_PAIRS or seed not in SEEDS or key in seen:
            raise RuntimeError("grid cell identity changed or duplicated")
        seen.add(key)
        kind = cell.get("kind")
        suffix = ".pt" if kind == "result" else ".failure.json" if kind == "failure" else None
        if suffix is None:
            raise RuntimeError("grid cell kind changed")
        artifact = f"{_cell_stem(*ranks, seed)}{suffix}"
        if cell.get("artifact") != artifact or artifact in artifacts:
            raise RuntimeError("grid artifact identity changed or duplicated")
        artifacts.add(artifact)
        artifact_path = path.parent / artifact
        artifact_raw = artifact_path.read_bytes()
        if len(artifact_raw) != cell.get("bytes") or sha256(artifact_raw) != cell.get("artifact_sha256"):
            raise RuntimeError("grid artifact bytes do not match terminal")
        if cell.get("validation_values_read") is not False or cell.get("eval_values_read") is not False:
            raise RuntimeError("grid cell role boundary is false")
        if cell.get("source_closure_sha256") != expected_source["sha256"] or cell.get("input_binding_sha256") != input_binding["sha256"]:
            raise RuntimeError("grid cell source/input binding changed")
        if cell.get("steps") != STEPS or cell.get("learning_rate") != LEARNING_RATE or cell.get("optimizer_device") != "cuda":
            raise RuntimeError("grid cell optimizer protocol changed")
        if _finite(cell.get("elapsed_seconds"), "cell elapsed_seconds") < 0:
            raise RuntimeError("cell elapsed_seconds is negative")
        if kind == "result":
            if set(cell) != RESULT_CELL_KEYS:
                raise RuntimeError("result terminal-cell schema changed")
            results += 1
            if not isinstance(cell.get("healthy"), bool):
                raise RuntimeError("result health is missing")
            healthy += int(cell["healthy"])
            for metric in (
                "initial_mse", "final_mse", "improvement_fraction",
                "normalized_training_mse", "training_response_rms",
                "worst_owner_pair_nrmse",
            ):
                _finite(cell.get(metric), metric)
            initial = float(cell["initial_mse"])
            final = float(cell["final_mse"])
            if initial <= 0:
                raise RuntimeError("result initial MSE is not positive")
            replay_improvement = (initial - final) / initial
            if not math.isclose(
                float(cell["improvement_fraction"]), replay_improvement,
                rel_tol=1e-12, abs_tol=1e-15,
            ):
                raise RuntimeError("result improvement does not replay")
            if cell.get("minimum_improvement") != MINIMUM_IMPROVEMENT or cell["healthy"] != (
                math.isfinite(final) and replay_improvement >= MINIMUM_IMPROVEMENT
            ):
                raise RuntimeError("result health does not replay")
            if len(cell.get("phase_mse", ())) != 2 or len(cell.get("source_owner_mse", ())) != 6 or len(cell.get("target_owner_mse", ())) != 6:
                raise RuntimeError("registered training slice shape changed")
            owner_pairs = cell.get("owner_pair_nrmse")
            if not isinstance(owner_pairs, list) or len(owner_pairs) != 6 or any(
                not isinstance(row, list) or len(row) != 6 for row in owner_pairs
            ):
                raise RuntimeError("owner-pair matrix shape changed")
            persistent, code = structured_price(*ranks)
            if cell.get("persistent_values") != persistent or cell.get("per_document_values") != code:
                raise RuntimeError("grid price changed")
            if cell.get("amortized_total_values") != persistent + 229 * code:
                raise RuntimeError("amortized price changed")
            if cell.get("strict_dense_matched_rank") != 0 or cell.get("prediction_multiply_adds_per_document") != 4802 * code or cell.get("calibration_cells_training_stage") != 0:
                raise RuntimeError("registered result cost changed")
            if cell.get("registered_validation_calibration_arm_budgets") != [2, 4, 8, 16]:
                raise RuntimeError("registered calibration budgets changed")
            expected_costs = [
                {
                    "arms": arms, "cells": 49 * arms,
                    "normal_equation_multiply_add_upper_bound": 49 * arms * code * (code + 1) + code ** 3,
                }
                for arms in (2, 4, 8, 16)
            ]
            if cell.get("registered_validation_calibration_costs") != expected_costs:
                raise RuntimeError("registered calibration costs changed")
        else:
            if set(cell) != FAILURE_CELL_KEYS or cell.get("schema") != "causal_response_factorization_v1_grid_failure" or cell.get("status") != "failed_training_only":
                raise RuntimeError("failure terminal-cell schema changed")
            failures += 1
            if cell.get("error_type") != "RuntimeError" or cell.get("error_message") not in NUMERICAL_FAILURE_MESSAGES:
                raise RuntimeError("grid failure cause is not registered")
    if seen != {(*ranks, seed) for ranks in RANK_PAIRS for seed in SEEDS}:
        raise RuntimeError("grid rank/seed coverage is incomplete")
    if (terminal.get("result_cells"), terminal.get("failure_cells"), terminal.get("healthy_cells")) != (results, failures, healthy):
        raise RuntimeError("grid terminal counts do not replay")
    expected_names = {".lock", "terminal.json", *artifacts}
    if {item.name for item in path.parent.iterdir()} != expected_names:
        raise RuntimeError("grid directory census changed")
    return terminal, raw


def _range(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "median": median(values), "max": max(values)}


def build(
    grid: Path = GRID, *, expected_source: dict[str, object] | None = None
) -> dict[str, object]:
    terminal, terminal_raw = validate_terminal(grid, expected_source=expected_source)
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for cell in terminal["cells"]:
        grouped[(cell["global_rank"], cell["private_rank_each_owner"])].append(cell)

    controls = terminal["controls"]
    strict_zero = _finite(controls.get("strict_dense_rank_zero_mse"), "strict control")
    mean_mse = _finite(controls.get("observationwise_training_mean_mse"), "mean control")
    if controls.get("observationwise_training_mean_persistent_values") != 4802:
        raise RuntimeError("mean-control price changed")

    rank_rows: list[dict[str, object]] = []
    for ranks in RANK_PAIRS:
        group = sorted(grouped[ranks], key=lambda cell: cell["seed"])
        if len(group) != len(SEEDS) or tuple(cell["seed"] for cell in group) != SEEDS:
            raise RuntimeError("rank-pair seed triplet is incomplete")
        eligible = all(cell["kind"] == "result" and cell["healthy"] for cell in group)
        persistent, code = structured_price(*ranks)
        seed_status = [
            {
                "seed": cell["seed"],
                "kind": cell["kind"],
                "healthy": cell.get("healthy", False),
                "artifact": cell["artifact"],
                "elapsed_seconds": cell["elapsed_seconds"],
                **({
                    "initial_mse": cell["initial_mse"],
                    "final_mse": cell["final_mse"],
                    "improvement_fraction": cell["improvement_fraction"],
                    "normalized_training_mse": cell["normalized_training_mse"],
                    "worst_owner_pair_nrmse": cell["worst_owner_pair_nrmse"],
                } if cell["kind"] == "result" else {
                    "error_type": cell["error_type"],
                    "error_message": cell["error_message"],
                }),
            }
            for cell in group
        ]
        row: dict[str, object] = {
            "global_rank": ranks[0],
            "private_rank_each_owner": ranks[1],
            "family": "joint" if all(ranks) else ("global_only" if ranks[0] else "private_only"),
            "persistent_values": persistent,
            "per_document_values": code,
            "amortized_total_values": persistent + 229 * code,
            "eligible_complete_healthy_three_seed_candidate": eligible,
            "seed_status": seed_status,
            "total_elapsed_seconds": sum(float(cell["elapsed_seconds"]) for cell in group),
        }
        if eligible:
            def values(name: str) -> list[float]:
                return [float(cell[name]) for cell in group]

            owner_pair_medians = [
                [median([float(cell["owner_pair_nrmse"][s][t]) for cell in group]) for t in range(6)]
                for s in range(6)
            ]
            worst_source, worst_target = max(
                ((s, t) for s in range(6) for t in range(6)),
                key=lambda pair: owner_pair_medians[pair[0]][pair[1]],
            )
            row.update({
                "final_mse_range": _range(values("final_mse")),
                "median_final_mse": median(values("final_mse")),
                "normalized_training_mse_range": _range(values("normalized_training_mse")),
                "median_worst_owner_pair_nrmse": median(values("worst_owner_pair_nrmse")),
                "worst_owner_pair_nrmse_range": _range(values("worst_owner_pair_nrmse")),
                "improvement_fraction_range": _range(values("improvement_fraction")),
                "training_response_rms_range": _range(values("training_response_rms")),
                "median_phase_mse": [median([float(cell["phase_mse"][p]) for cell in group]) for p in range(2)],
                "median_source_owner_mse": [median([float(cell["source_owner_mse"][i]) for cell in group]) for i in range(6)],
                "median_target_owner_mse": [median([float(cell["target_owner_mse"][i]) for cell in group]) for i in range(6)],
                "worst_median_owner_pair": {
                    "source_owner": OWNER_LABELS[worst_source],
                    "target_owner": OWNER_LABELS[worst_target],
                    "nrmse": owner_pair_medians[worst_source][worst_target],
                },
                "median_mse_below_strict_rank_zero": strict_zero - median(values("final_mse")),
                "median_mse_below_observationwise_mean": mean_mse - median(values("final_mse")),
            })
        rank_rows.append(row)

    eligible_rows = [row for row in rank_rows if row["eligible_complete_healthy_three_seed_candidate"]]
    pooled_frontier = frontier(eligible_rows, ("median_final_mse",))
    robust_frontier = frontier(
        eligible_rows, ("median_final_mse", "median_worst_owner_pair_nrmse")
    )
    return {
        "schema": "causal_response_factorization_v1_training_analysis",
        "status": "complete_fit_only_no_candidate_selected_or_frozen",
        "analysis_source_closure": analysis_source_closure(),
        "grid_terminal_sha256": sha256(terminal_raw),
        "grid_manifest_sha256": terminal["manifest_sha256"],
        "expected_cells": terminal["expected_cells"],
        "result_cells": terminal["result_cells"],
        "failure_cells": terminal["failure_cells"],
        "healthy_cells": terminal["healthy_cells"],
        "total_optimizer_seconds_including_failures": sum(float(cell["elapsed_seconds"]) for cell in terminal["cells"]),
        "controls_with_explicit_prices": {
            "strict_dense_rank_zero": {"mse": strict_zero, "persistent_values": 0, "per_document_values": 0},
            "observationwise_training_mean": {"mse": mean_mse, "persistent_values": 4802, "per_document_values": 0},
        },
        "rank_pair_rows": rank_rows,
        "eligible_rank_pairs": len(eligible_rows),
        "training_pooled_median_mse_frontier_rank_pairs": pooled_frontier,
        "training_median_mse_and_worst_owner_pair_frontier_rank_pairs": robust_frontier,
        "interpretation_limits": {
            "candidate_selected": False,
            "candidate_library_frozen": False,
            "hierarchy_or_quotient_supported": False,
            "semantic_or_terminal_circuit_claim": False,
            "extraction_removal_ood_or_ledger_credit": False,
            "validation_values_read": False,
            "eval_values_read": False,
            "training_frontiers_do_not_authorize_validation_selection": True,
        },
    }


def _publish_create_only(value: dict[str, object]) -> bytes:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if OUTPUT.exists():
        if OUTPUT.read_bytes() != raw:
            raise RuntimeError("training analysis namespace is already spent differently")
        return raw
    stage = OUTPUT.with_name(f".{OUTPUT.name}.stage.{os.getpid()}")
    try:
        with stage.open("xb") as sink:
            sink.write(raw)
            sink.flush()
            os.fsync(sink.fileno())
        os.link(stage, OUTPUT)
        directory_fd = os.open(OUTPUT.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        stage.unlink(missing_ok=True)
    if OUTPUT.read_bytes() != raw:
        raise RuntimeError("training analysis did not replay")
    return raw


def main() -> None:
    print(_publish_create_only(build()).decode(), end="")


if __name__ == "__main__":
    main()
