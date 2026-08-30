#!/usr/bin/env python3
"""Create-only, mutation-closed freeze of every nondominated FIT program."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import causal_response_factorization_v1_training_analysis as training_analysis


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSIS = HERE / "causal_response_factorization_v1_training_analysis.json"
GRID_TERMINAL = HERE / "causal_response_factorization_v1_grid_results" / "terminal.json"
OUTPUT = HERE / "causal_response_factorization_v1_candidate_freeze_v2.json"
SOURCE_PATHS = (
    HERE / "causal_response_factorization_v1_candidate_freeze_v2.py",
    HERE / "test_causal_response_factorization_v1_candidate_freeze_v2.py",
    HERE / "CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_15.md",
    HERE / "causal_response_factorization_v1_candidate_freeze_v1_failure.json",
    HERE / "causal_response_factorization_v1_training_analysis.py",
    HERE / "test_causal_response_factorization_v1_training_analysis.py",
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def logical_sha256(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _read_sources(paths: tuple[Path, ...], root: Path) -> tuple[dict[str, str], dict[Path, bytes]]:
    raw = {path: path.read_bytes() for path in paths}
    hashes = {str(path.relative_to(root)): sha256(value) for path, value in raw.items()}
    return hashes, raw


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


def _freeze_records(
    analysis: dict[str, object], terminal: dict[str, object], grid_directory: Path, root: Path,
) -> tuple[list[list[int]], list[dict[str, object]], dict[Path, bytes]]:
    rank_ids = candidate_rank_ids(analysis)
    rows = {
        f"g{int(row['global_rank']):02d}_p{int(row['private_rank_each_owner']):02d}": row
        for row in analysis["rank_pair_rows"]
    }
    cells = {
        (int(cell["global_rank"]), int(cell["private_rank_each_owner"]), int(cell["seed"])): cell
        for cell in terminal["cells"]
    }
    rank_pairs: list[list[int]] = []
    programs: list[dict[str, object]] = []
    artifact_raw: dict[Path, bytes] = {}
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
            if (
                cell is None or cell.get("kind") != "result" or cell.get("healthy") is not True
                or cell.get("artifact") != seed_row.get("artifact")
            ):
                raise RuntimeError("frontier program identity or health changed")
            path = grid_directory / str(cell["artifact"])
            raw = path.read_bytes()
            if len(raw) != cell["bytes"] or sha256(raw) != cell["artifact_sha256"]:
                raise RuntimeError("frontier program bytes changed")
            artifact_raw[path] = raw
            programs.append({
                "global_rank": ranks[0],
                "private_rank_each_owner": ranks[1],
                "seed": seed,
                "artifact": str(path.relative_to(root)),
                "artifact_sha256": cell["artifact_sha256"],
                "bytes": cell["bytes"],
                "persistent_values": cell["persistent_values"],
                "per_document_values": cell["per_document_values"],
            })
    identities = {(p["global_rank"], p["private_rank_each_owner"], p["seed"]) for p in programs}
    if len(programs) != 3 * len(rank_pairs) or len(identities) != len(programs):
        raise RuntimeError("frozen program census is incomplete or duplicated")
    return rank_pairs, programs, artifact_raw


def validate_manifest(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("candidate freeze is not an object")
    expected = {
        "schema", "status", "source_closure", "training_analysis_sha256",
        "grid_terminal_sha256", "grid_manifest_sha256", "selection_rule",
        "candidate_rank_pairs", "candidate_rank_pair_count", "candidate_programs",
        "candidate_program_count", "candidate_selected", "validation_values_read",
        "eval_values_read", "manifest_sha256",
    }
    if set(value) != expected or value.get("schema") != "causal_response_factorization_v1_candidate_freeze_v2":
        raise RuntimeError("candidate freeze schema changed")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if logical_sha256(body) != value.get("manifest_sha256"):
        raise RuntimeError("candidate freeze manifest hash does not replay")
    programs = value.get("candidate_programs")
    pairs = value.get("candidate_rank_pairs")
    if not isinstance(programs, list) or not isinstance(pairs, list):
        raise RuntimeError("candidate freeze census is malformed")
    if value.get("candidate_program_count") != len(programs) or value.get("candidate_rank_pair_count") != len(pairs):
        raise RuntimeError("candidate freeze counts do not replay")
    if len(programs) != 3 * len(pairs):
        raise RuntimeError("candidate freeze does not contain three seeds per rank pair")
    program_keys = {
        "global_rank", "private_rank_each_owner", "seed", "artifact",
        "artifact_sha256", "bytes", "persistent_values", "per_document_values",
    }
    if any(not isinstance(p, dict) or set(p) != program_keys for p in programs):
        raise RuntimeError("candidate program schema changed or contains a score")
    if value.get("candidate_selected") is not False or value.get("validation_values_read") is not False or value.get("eval_values_read") is not False:
        raise RuntimeError("candidate freeze role boundary changed")
    return value


def build_manifest(
    analysis_path: Path = ANALYSIS,
    grid_terminal: Path = GRID_TERMINAL,
    source_paths: tuple[Path, ...] = SOURCE_PATHS,
    root: Path = ROOT,
    analysis_builder: Callable[[Path], dict[str, object]] | None = None,
    terminal_validator: Callable[[Path], tuple[dict[str, object], bytes]] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if analysis_builder is None:
        analysis_builder = training_analysis.build
    if terminal_validator is None:
        terminal_validator = training_analysis.validate_terminal
    analysis_raw = analysis_path.read_bytes()
    published_analysis = json.loads(analysis_raw)
    terminal, terminal_raw = terminal_validator(grid_terminal)
    replayed_analysis = analysis_builder(grid_terminal)
    if published_analysis != replayed_analysis or published_analysis.get("status") != "complete_fit_only_no_candidate_selected_or_frozen":
        raise RuntimeError("published training analysis does not replay")
    if published_analysis.get("grid_terminal_sha256") != sha256(terminal_raw) or published_analysis.get("grid_manifest_sha256") != terminal.get("manifest_sha256"):
        raise RuntimeError("training analysis and grid terminal are not the same snapshot")
    source_hashes, source_raw = _read_sources(source_paths, root)
    rank_pairs, programs, artifact_raw = _freeze_records(
        published_analysis, terminal, grid_terminal.parent, root,
    )
    body: dict[str, object] = {
        "schema": "causal_response_factorization_v1_candidate_freeze_v2",
        "status": "complete_training_frontier_freeze_no_scores",
        "source_closure": {"paths": source_hashes, "sha256": logical_sha256(source_hashes)},
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
    manifest = validate_manifest({**body, "manifest_sha256": logical_sha256(body)})
    snapshot = {
        "analysis_path": analysis_path, "analysis_raw": analysis_raw,
        "grid_terminal": grid_terminal, "terminal_raw": terminal_raw,
        "source_raw": source_raw, "artifact_raw": artifact_raw,
        "published_analysis": published_analysis,
        "analysis_builder": analysis_builder, "terminal_validator": terminal_validator,
    }
    return manifest, snapshot


def revalidate_inputs(snapshot: dict[str, object]) -> None:
    analysis_path = snapshot["analysis_path"]
    grid_terminal = snapshot["grid_terminal"]
    if analysis_path.read_bytes() != snapshot["analysis_raw"]:
        raise RuntimeError("training analysis mutated during freeze")
    terminal, terminal_raw = snapshot["terminal_validator"](grid_terminal)
    if terminal_raw != snapshot["terminal_raw"]:
        raise RuntimeError("grid terminal mutated during freeze")
    if snapshot["analysis_builder"](grid_terminal) != snapshot["published_analysis"]:
        raise RuntimeError("training analysis replay mutated during freeze")
    for path, raw in snapshot["source_raw"].items():
        if path.read_bytes() != raw:
            raise RuntimeError("freeze source mutated during freeze")
    for path, raw in snapshot["artifact_raw"].items():
        if path.read_bytes() != raw:
            raise RuntimeError("candidate program mutated during freeze")
    if snapshot["published_analysis"]["grid_terminal_sha256"] != sha256(terminal_raw) or snapshot["published_analysis"]["grid_manifest_sha256"] != terminal["manifest_sha256"]:
        raise RuntimeError("training/grid binding mutated during freeze")


def publish_create_only(value: dict[str, object], output: Path, revalidate: Callable[[], None]) -> bytes:
    validate_manifest(value)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    revalidate()
    if output.exists():
        existing = output.read_bytes()
        if existing != raw:
            raise RuntimeError("candidate freeze v2 namespace is already spent differently")
        validate_manifest(json.loads(existing))
        revalidate()
        return existing
    stage = output.with_name(f".{output.name}.stage.{os.getpid()}")
    try:
        with stage.open("xb") as sink:
            sink.write(raw)
            sink.flush()
            os.fsync(sink.fileno())
        revalidate()
        os.link(stage, output)
        linked = output.read_bytes()
        if linked != raw:
            raise RuntimeError("linked candidate freeze bytes changed")
        validate_manifest(json.loads(linked))
        revalidate()
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        stage.unlink(missing_ok=True)
    return raw


def main() -> None:
    value, snapshot = build_manifest()
    print(publish_create_only(value, OUTPUT, lambda: revalidate_inputs(snapshot)).decode(), end="")


if __name__ == "__main__":
    main()
