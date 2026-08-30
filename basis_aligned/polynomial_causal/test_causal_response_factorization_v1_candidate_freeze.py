from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from causal_response_factorization_v1_candidate_freeze import (
    candidate_rank_ids,
    freeze_records,
)


def _fixture(tmp_path: Path):
    ranks = [(1, 0), (4, 1)]
    seeds = [11, 12, 13]
    rows = []
    cells = []
    for global_rank, private_rank in ranks:
        status = []
        for seed in seeds:
            name = f"g{global_rank:02d}_p{private_rank:02d}_s{seed}.pt"
            raw = name.encode()
            (tmp_path / name).write_bytes(raw)
            status.append({"seed": seed, "artifact": name})
            cells.append({
                "global_rank": global_rank, "private_rank_each_owner": private_rank,
                "seed": seed, "kind": "result", "healthy": True, "artifact": name,
                "artifact_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "persistent_values": 10, "per_document_values": 2,
                "final_mse": 0.2, "worst_owner_pair_nrmse": 0.3,
            })
        rows.append({
            "global_rank": global_rank, "private_rank_each_owner": private_rank,
            "eligible_complete_healthy_three_seed_candidate": True,
            "seed_status": status,
        })
    analysis = {
        "training_pooled_median_mse_frontier_rank_pairs": ["g01_p00"],
        "training_median_mse_and_worst_owner_pair_frontier_rank_pairs": ["g04_p01"],
        "rank_pair_rows": rows,
    }
    return analysis, {"cells": cells}


def test_freeze_uses_union_and_all_three_seeds(tmp_path: Path) -> None:
    analysis, terminal = _fixture(tmp_path)
    ranks, programs = freeze_records(analysis, terminal, tmp_path, root=tmp_path)
    assert ranks == [[1, 0], [4, 1]]
    assert len(programs) == 6
    assert {program["seed"] for program in programs} == {11, 12, 13}


def test_one_unhealthy_seed_rejects_whole_frontier(tmp_path: Path) -> None:
    analysis, terminal = _fixture(tmp_path)
    terminal["cells"][0]["healthy"] = False
    with pytest.raises(RuntimeError, match="identity or health"):
        freeze_records(analysis, terminal, tmp_path, root=tmp_path)


def test_frontier_cannot_be_empty() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        candidate_rank_ids({
            "training_pooled_median_mse_frontier_rank_pairs": [],
            "training_median_mse_and_worst_owner_pair_frontier_rank_pairs": [],
        })


def test_program_byte_tamper_is_rejected(tmp_path: Path) -> None:
    analysis, terminal = _fixture(tmp_path)
    first = tmp_path / terminal["cells"][0]["artifact"]
    first.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="bytes changed"):
        freeze_records(analysis, terminal, tmp_path, root=tmp_path)
