from __future__ import annotations

import json
from pathlib import Path

import pytest

import causal_response_factorization_v1_training_analysis as analysis


def _source() -> dict[str, object]:
    return {"schema": "synthetic_source", "sha256": "a" * 64}


def _input_binding() -> dict[str, object]:
    artifact_binding = {
        "parent_binding_sha256": "1" * 64, "receipt_sha256": "2" * 64,
        "terminal_sha256": "2" * 64, "authority_artifact_sha256": "3" * 64,
        "authority_logical_sha256": "4" * 64, "bundle_sha256": "5" * 64,
        "manifest_artifact_sha256": "6" * 64, "manifest_logical_sha256": "7" * 64,
        "source_closure_sha256": "8" * 64,
    }
    body = {
        "artifact_binding": artifact_binding, "response_sha256": "9" * 64,
        "valid_sha256": "a" * 64, "document_ids_sha256": "b" * 64,
        "original_document_indices_sha256": "c" * 64, "source_groups_sha256": "d" * 64,
        "shape": [2, 49, 49, 229], "owner_components": list(analysis.OWNER_LABELS),
        "phases": ["full", "residual"], "source_tags": [f"r.{i}" for i in range(49)],
        "target_tags": [f"r.{i}" for i in range(49)],
        "validation_values_read": False, "eval_values_read": False,
    }
    return {**body, "sha256": analysis.logical_sha256(body)}


def _grid(tmp_path: Path, *, unhealthy: tuple[int, int, int] | None = None) -> Path:
    directory = tmp_path / "grid"
    directory.mkdir()
    (directory / ".lock").write_bytes(b"")
    source = _source()
    input_binding = _input_binding()
    cells = []
    for ranks in analysis.RANK_PAIRS:
        for seed in analysis.SEEDS:
            stem = analysis._cell_stem(*ranks, seed)
            artifact = f"{stem}.pt"
            artifact_raw = stem.encode()
            (directory / artifact).write_bytes(artifact_raw)
            if ranks == (1, 1):
                mse = 0.01 if seed == analysis.SEEDS[0] else 0.5
            elif ranks == (0, 2):
                mse = 0.2
            else:
                mse = 2.0
            is_unhealthy = (*ranks, seed) == unhealthy
            if is_unhealthy:
                mse = 3.0
            persistent, code = analysis.structured_price(*ranks)
            cells.append({
                "kind": "result", "artifact": artifact,
                "artifact_sha256": analysis.sha256(artifact_raw), "bytes": len(artifact_raw),
                "global_rank": ranks[0], "private_rank_each_owner": ranks[1], "seed": seed,
                "source_closure_sha256": source["sha256"],
                "input_binding_sha256": input_binding["sha256"],
                "steps": analysis.STEPS, "learning_rate": analysis.LEARNING_RATE,
                "optimizer_device": "cuda",
                "healthy": not is_unhealthy, "minimum_improvement": analysis.MINIMUM_IMPROVEMENT,
                "elapsed_seconds": 1.0,
                "initial_mse": 3.0, "final_mse": mse,
                "improvement_fraction": (3.0 - mse) / 3.0,
                "normalized_training_mse": mse, "training_response_rms": 1.0,
                "worst_owner_pair_nrmse": mse,
                "phase_mse": [mse, mse], "source_owner_mse": [mse] * 6,
                "target_owner_mse": [mse] * 6,
                "owner_pair_nrmse": [[mse] * 6 for _ in range(6)],
                "persistent_values": persistent, "per_document_values": code,
                "amortized_total_values": persistent + 229 * code,
                "strict_dense_matched_rank": 0,
                "amortized_total_dense_rank_noncontrolling": (persistent + 229 * code) // 5031,
                "prediction_multiply_adds_per_document": 4802 * code,
                "calibration_cells_training_stage": 0,
                "registered_validation_calibration_arm_budgets": [2, 4, 8, 16],
                "registered_validation_calibration_costs": [
                    {
                        "arms": arms, "cells": 49 * arms,
                        "normal_equation_multiply_add_upper_bound": 49 * arms * code * (code + 1) + code ** 3,
                    }
                    for arms in (2, 4, 8, 16)
                ],
                "validation_values_read": False, "eval_values_read": False,
            })
    body = {
        "schema": "causal_response_factorization_v1_grid_terminal",
        "status": "complete_training_only_grid",
        "source_closure": source, "input_binding": input_binding,
        "rank_pairs": [list(pair) for pair in analysis.RANK_PAIRS],
        "seeds": list(analysis.SEEDS), "steps": analysis.STEPS,
        "learning_rate": analysis.LEARNING_RATE, "optimizer_device": "cuda",
        "expected_cells": 51,
        "result_cells": 51, "failure_cells": 0,
        "healthy_cells": 50 if unhealthy else 51, "cells": cells,
        "controls": {
            "strict_dense_rank_zero_mse": 3.0,
            "observationwise_training_mean_mse": 0.1,
            "observationwise_training_mean_persistent_values": 4802,
        },
        "validation_values_read": False, "eval_values_read": False,
    }
    terminal = {**body, "manifest_sha256": analysis.logical_sha256(body)}
    path = directory / "terminal.json"
    path.write_text(json.dumps(terminal), encoding="utf-8")
    return path


def test_frontier_uses_three_seed_median_not_best_seed(tmp_path: Path) -> None:
    receipt = analysis.build(_grid(tmp_path), expected_source=_source())
    frontier = receipt["training_pooled_median_mse_frontier_rank_pairs"]
    assert "g00_p02" in frontier
    joint = next(row for row in receipt["rank_pair_rows"] if analysis.rank_id(row) == "g01_p01")
    assert joint["median_final_mse"] == 0.5


def test_one_unhealthy_seed_excludes_whole_rank_pair(tmp_path: Path) -> None:
    bad = (0, 2, analysis.SEEDS[1])
    receipt = analysis.build(_grid(tmp_path, unhealthy=bad), expected_source=_source())
    row = next(row for row in receipt["rank_pair_rows"] if analysis.rank_id(row) == "g00_p02")
    assert row["eligible_complete_healthy_three_seed_candidate"] is False
    assert "g00_p02" not in receipt["training_pooled_median_mse_frontier_rank_pairs"]


def test_incomplete_self_hashed_terminal_is_rejected(tmp_path: Path) -> None:
    path = _grid(tmp_path)
    terminal = json.loads(path.read_text())
    terminal["cells"] = terminal["cells"][:1]
    body = {key: value for key, value in terminal.items() if key != "manifest_sha256"}
    terminal["manifest_sha256"] = analysis.logical_sha256(body)
    path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected-cell census"):
        analysis.validate_terminal(path, expected_source=_source())


def test_full_census_with_forged_optimizer_protocol_is_rejected(tmp_path: Path) -> None:
    path = _grid(tmp_path)
    terminal = json.loads(path.read_text())
    terminal["steps"] = analysis.STEPS - 1
    for cell in terminal["cells"]:
        cell["steps"] = analysis.STEPS - 1
    body = {key: value for key, value in terminal.items() if key != "manifest_sha256"}
    terminal["manifest_sha256"] = analysis.logical_sha256(body)
    path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(RuntimeError, match="optimizer protocol"):
        analysis.validate_terminal(path, expected_source=_source())


def test_full_census_with_forged_health_is_rejected(tmp_path: Path) -> None:
    path = _grid(tmp_path)
    terminal = json.loads(path.read_text())
    terminal["cells"][0]["healthy"] = True
    terminal["cells"][0]["improvement_fraction"] = -1.0
    body = {key: value for key, value in terminal.items() if key != "manifest_sha256"}
    terminal["manifest_sha256"] = analysis.logical_sha256(body)
    path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(RuntimeError, match="improvement does not replay"):
        analysis.validate_terminal(path, expected_source=_source())
