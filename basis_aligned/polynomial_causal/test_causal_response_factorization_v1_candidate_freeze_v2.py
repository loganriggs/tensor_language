from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causal_response_factorization_v1_candidate_freeze_v2 import (
    build_manifest, publish_create_only, revalidate_inputs, validate_manifest,
)


def _fixture(tmp_path: Path):
    grid = tmp_path / "grid"
    grid.mkdir()
    source = tmp_path / "source.py"
    source.write_text("source\n")
    ranks = [(1, 0), (4, 1)]
    seeds = [11, 12, 13]
    rows = []
    cells = []
    for global_rank, private_rank in ranks:
        status = []
        for seed in seeds:
            name = f"g{global_rank:02d}_p{private_rank:02d}_s{seed}.pt"
            raw = name.encode()
            (grid / name).write_bytes(raw)
            status.append({"seed": seed, "artifact": name})
            cells.append({
                "global_rank": global_rank, "private_rank_each_owner": private_rank,
                "seed": seed, "kind": "result", "healthy": True, "artifact": name,
                "artifact_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "persistent_values": 10, "per_document_values": 2,
            })
        rows.append({
            "global_rank": global_rank, "private_rank_each_owner": private_rank,
            "eligible_complete_healthy_three_seed_candidate": True, "seed_status": status,
        })
    terminal = {"manifest_sha256": "a" * 64, "cells": cells}
    terminal_raw = (json.dumps(terminal, sort_keys=True) + "\n").encode()
    terminal_path = grid / "terminal.json"
    terminal_path.write_bytes(terminal_raw)
    analysis = {
        "status": "complete_fit_only_no_candidate_selected_or_frozen",
        "grid_terminal_sha256": hashlib.sha256(terminal_raw).hexdigest(),
        "grid_manifest_sha256": "a" * 64,
        "training_pooled_median_mse_frontier_rank_pairs": ["g01_p00"],
        "training_median_mse_and_worst_owner_pair_frontier_rank_pairs": ["g04_p01"],
        "rank_pair_rows": rows,
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis))
    builder = lambda _: analysis
    validator = lambda path: (json.loads(path.read_bytes()), path.read_bytes())
    value, snapshot = build_manifest(
        analysis_path, terminal_path, (source,), tmp_path,
        analysis_builder=builder, terminal_validator=validator,
    )
    return value, snapshot, grid, source


def test_build_freezes_union_all_seeds_and_no_scores(tmp_path: Path) -> None:
    value, snapshot, _, _ = _fixture(tmp_path)
    assert value["candidate_rank_pairs"] == [[1, 0], [4, 1]]
    assert value["candidate_program_count"] == 6
    assert {program["seed"] for program in value["candidate_programs"]} == {11, 12, 13}
    assert all("mse" not in key and "nrmse" not in key for p in value["candidate_programs"] for key in p)
    revalidate_inputs(snapshot)


def test_analysis_grid_binding_mismatch_rejects(tmp_path: Path) -> None:
    value, snapshot, _, _ = _fixture(tmp_path)
    analysis = dict(snapshot["published_analysis"])
    analysis["grid_manifest_sha256"] = "b" * 64
    snapshot["analysis_path"].write_text(json.dumps(analysis))
    with pytest.raises(RuntimeError, match="same snapshot"):
        build_manifest(
            snapshot["analysis_path"], snapshot["grid_terminal"],
            tuple(snapshot["source_raw"]), tmp_path,
            analysis_builder=lambda _: analysis,
            terminal_validator=snapshot["terminal_validator"],
        )


def test_program_mutation_before_link_rejects_and_publishes_nothing(tmp_path: Path) -> None:
    value, snapshot, grid, _ = _fixture(tmp_path)
    (grid / value["candidate_programs"][0]["artifact"].split("/")[-1]).write_bytes(b"tampered")
    output = tmp_path / "output.json"
    with pytest.raises(RuntimeError, match="program mutated|bytes"):
        publish_create_only(value, output, lambda: revalidate_inputs(snapshot))
    assert not output.exists()


def test_existing_output_corruption_rejects(tmp_path: Path) -> None:
    value, snapshot, _, _ = _fixture(tmp_path)
    output = tmp_path / "output.json"
    output.write_text("{}\n")
    with pytest.raises(RuntimeError, match="spent differently"):
        publish_create_only(value, output, lambda: revalidate_inputs(snapshot))


def test_post_link_mutation_is_detected_and_failure_is_preserved(tmp_path: Path) -> None:
    value, snapshot, _, source = _fixture(tmp_path)
    output = tmp_path / "output.json"
    calls = 0

    def mutate_after_link() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            source.write_text("mutated\n")
        revalidate_inputs(snapshot)

    with pytest.raises(RuntimeError, match="source mutated"):
        publish_create_only(value, output, mutate_after_link)
    assert output.exists()
    validate_manifest(json.loads(output.read_text()))


def test_manifest_semantic_hash_rejects_tamper(tmp_path: Path) -> None:
    value, _, _, _ = _fixture(tmp_path)
    value["candidate_rank_pair_count"] = 99
    with pytest.raises(RuntimeError, match="manifest hash"):
        validate_manifest(value)
