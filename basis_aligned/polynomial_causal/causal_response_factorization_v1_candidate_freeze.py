#!/usr/bin/env python3
"""Create-only freeze of every nondominated FIT response program."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import causal_response_factorization_v1_training_analysis as training_analysis


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSIS = HERE / "causal_response_factorization_v1_training_analysis.json"
GRID_TERMINAL = HERE / "causal_response_factorization_v1_grid_results" / "terminal.json"
OUTPUT = HERE / "causal_response_factorization_v1_candidate_freeze.json"
SOURCE_PATHS = (
    HERE / "causal_response_factorization_v1_candidate_freeze.py",
    HERE / "test_causal_response_factorization_v1_candidate_freeze.py",
    HERE / "CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_14.md",
    HERE / "causal_response_factorization_v1_training_analysis.py",
    HERE / "test_causal_response_factorization_v1_training_analysis.py",
    ANALYSIS,
    GRID_TERMINAL,
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def logical_sha256(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def source_closure() -> dict[str, object]:
    hashes = {str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in SOURCE_PATHS}
    return {"paths": hashes, "sha256": logical_sha256(hashes)}


def candidate_rank_ids(analysis: dict[str, object]) -> tuple[str, ...]:
    pooled = analysis.get("training_pooled_median_mse_frontier_rank_pairs")
    robust = analysis.get("training_median_mse_and_worst_owner_pair_frontier_rank_pairs")
    if not isinstance(pooled, list) or not isinstance(robust, list) or any(
        not isinstance(item, str) for item in (*pooled, *robust)
    ):
        raise RuntimeError("training frontier schema changed")
    result = tuple(sorted(set(pooled) | set(robust)))
    if not result:
        raise RuntimeError("training frontier is empty")
    return result


def freeze_records(
    analysis: dict[str, object], terminal: dict[str, object], grid_directory: Path,
    root: Path = ROOT,
) -> tuple[list[list[int]], list[dict[str, object]]]:
    rank_ids = candidate_rank_ids(analysis)
    rows = {
        f"g{int(row['global_rank']):02d}_p{int(row['private_rank_each_owner']):02d}": row
        for row in analysis["rank_pair_rows"]
    }
    cells = {
        (int(cell["global_rank"]), int(cell["private_rank_each_owner"]), int(cell["seed"])): cell
        for cell in terminal["cells"]
    }
    rank_pairs = []
    programs = []
    for rank_id in rank_ids:
        row = rows.get(rank_id)
        if row is None or row.get("eligible_complete_healthy_three_seed_candidate") is not True:
            raise RuntimeError("frontier contains an ineligible rank pair")
        ranks = (int(row["global_rank"]), int(row["private_rank_each_owner"]))
        rank_pairs.append(list(ranks))
        seed_status = row.get("seed_status")
        if not isinstance(seed_status, list) or len(seed_status) != 3:
            raise RuntimeError("frontier rank pair lacks its three-seed record")
        for seed_row in seed_status:
            seed = int(seed_row["seed"])
            cell = cells.get((*ranks, seed))
            if cell is None or cell.get("kind") != "result" or cell.get("healthy") is not True or cell.get("artifact") != seed_row.get("artifact"):
                raise RuntimeError("frontier program identity or health changed")
            artifact_path = grid_directory / cell["artifact"]
            raw = artifact_path.read_bytes()
            if len(raw) != cell["bytes"] or sha256(raw) != cell["artifact_sha256"]:
                raise RuntimeError("frontier program bytes changed")
            programs.append({
                "global_rank": ranks[0], "private_rank_each_owner": ranks[1],
                "seed": seed,
                "artifact": str(artifact_path.relative_to(root)),
                "artifact_sha256": cell["artifact_sha256"], "bytes": cell["bytes"],
                "persistent_values": cell["persistent_values"],
                "per_document_values": cell["per_document_values"],
                "training_final_mse": cell["final_mse"],
                "training_worst_owner_pair_nrmse": cell["worst_owner_pair_nrmse"],
            })
    if len(programs) != 3 * len(rank_pairs) or len({(p["global_rank"], p["private_rank_each_owner"], p["seed"]) for p in programs}) != len(programs):
        raise RuntimeError("frozen program census is incomplete or duplicated")
    return rank_pairs, programs


def build_manifest() -> dict[str, object]:
    analysis_raw = ANALYSIS.read_bytes()
    published_analysis = json.loads(analysis_raw)
    replayed_analysis = training_analysis.build()
    if published_analysis != replayed_analysis or published_analysis.get("status") != "complete_fit_only_no_candidate_selected_or_frozen":
        raise RuntimeError("published training analysis does not replay")
    terminal_raw = GRID_TERMINAL.read_bytes()
    terminal = json.loads(terminal_raw)
    rank_pairs, programs = freeze_records(published_analysis, terminal, GRID_TERMINAL.parent)
    body: dict[str, object] = {
        "schema": "causal_response_factorization_v1_candidate_freeze",
        "status": "complete_training_frontier_freeze",
        "source_closure": source_closure(),
        "training_analysis_sha256": sha256(analysis_raw),
        "grid_terminal_sha256": sha256(terminal_raw),
        "grid_manifest_sha256": terminal["manifest_sha256"],
        "selection_rule": "union_of_complete_healthy_three_seed_pooled_and_worst_owner_training_frontiers",
        "candidate_rank_pairs": rank_pairs,
        "candidate_rank_pair_count": len(rank_pairs),
        "candidate_programs": programs,
        "candidate_program_count": len(programs),
        "candidate_selected": False,
        "validation_values_read": False,
        "eval_values_read": False,
    }
    return {**body, "manifest_sha256": logical_sha256(body)}


def main() -> None:
    value = build_manifest()
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if OUTPUT.exists():
        if OUTPUT.read_bytes() != raw:
            raise RuntimeError("candidate freeze namespace is already spent differently")
        print(raw.decode(), end="")
        return
    stage = OUTPUT.with_name(f".{OUTPUT.name}.stage.{os.getpid()}")
    try:
        with stage.open("xb") as sink:
            sink.write(raw); sink.flush(); os.fsync(sink.fileno())
        os.link(stage, OUTPUT)
        directory_fd = os.open(OUTPUT.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        stage.unlink(missing_ok=True)
    print(raw.decode(), end="")


if __name__ == "__main__":
    main()
