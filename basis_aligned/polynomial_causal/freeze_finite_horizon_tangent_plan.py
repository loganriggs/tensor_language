#!/usr/bin/env python3
"""Freeze the row/document lifecycle for the early-MLP tangent pilot.

This reads only an already-authorized CPU row cache and its authoritative provenance
receipt.  It performs no model forward and grants no GPU or scientific authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from finite_horizon_tangent_response_bank import (
    TANGENT_PROTOCOL,
    TANGENT_PROTOCOL_SHA256,
    TangentResponsePlan,
    allocate_whole_document_splits,
)


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
ROWS = BQ / ".rowcache/fineweb_n96_skip80.pt"
AUTHORITY = BQ / ".rowcache/fineweb_oracle_v2_receipt.json"
OUT = HERE / "finite_horizon_tangent_plan.json"
EXPECTED_RAW_SHA256 = "a703cadb1a5e27497cba43d21bca889a1d765b861c3da311a1dc4dfeb28b21cc"
EXPECTED_FILE_SHA256 = "94bc1fb3e3a6a061541e555295e0af8c50ae6068fdff84e95a69c25844091eda"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def tensor_raw_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def scored_position(row_id: str, seed: int = 2026082803) -> int:
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    return 64 + int.from_bytes(digest[:8], "big") % 192


def main() -> None:
    rows = torch.load(ROWS, map_location="cpu", weights_only=True)
    authority = json.loads(AUTHORITY.read_text())
    provenance = authority["document_provenance"]["sets"]["n96_skip80"]
    entry = authority["entries"]["n96_skip80"]
    observed_file = file_sha256(ROWS)
    observed_raw = tensor_raw_sha256(rows)
    if tuple(rows.shape) != (96, 513) or rows.dtype != torch.int64:
        raise RuntimeError("the frozen tangent rows have the wrong shape or dtype")
    if observed_file != EXPECTED_FILE_SHA256 or observed_raw != EXPECTED_RAW_SHA256:
        raise RuntimeError("the frozen tangent row identity changed")
    if entry["tensor_raw_sha256"] != observed_raw or len(provenance) != len(rows):
        raise RuntimeError("the row cache and authoritative provenance receipt disagree")
    document_ids = tuple(record["document_id"] for record in provenance)
    splits = allocate_whole_document_splits(document_ids)
    row_ids = tuple(f"n96_skip80:{index}" for index in range(len(rows)))
    plan = TangentResponsePlan(
        experiment_id="bilin18-early-mlp-finite-horizon-tangent-v1",
        row_artifact_sha256=observed_raw,
        row_ids=row_ids,
        document_ids=document_ids,
        splits=splits,
        scored_positions=tuple(scored_position(row_id) for row_id in row_ids),
        input_dims=((0, 32), (1, 32), (2, 32)),
        target_site=3,
        probes_per_row=16,
        direction_seed=2026082801,
        probe_seed=2026082802,
        position_seed=2026082803,
    )
    split_summary = {}
    for split in ("primary", "replication"):
        selected = [index for index, role in enumerate(splits) if role == split]
        split_summary[split] = {
            "rows": len(selected),
            "source_documents": len({document_ids[index] for index in selected}),
            "row_indices": selected,
        }
    result = {
        "status": "frozen_cpu_plan_no_gpu_authority",
        "plan_fingerprint": plan.fingerprint,
        "protocol": TANGENT_PROTOCOL,
        "protocol_sha256": TANGENT_PROTOCOL_SHA256,
        "row_cache": {
            "path": str(ROWS),
            "file_sha256": observed_file,
            "tensor_raw_sha256": observed_raw,
            "shape": list(rows.shape),
            "dtype": str(rows.dtype),
        },
        "authority_receipt": {
            "path": str(AUTHORITY),
            "file_sha256": file_sha256(AUTHORITY),
            "authority": authority["authority"],
            "authorized_for_scored_experiments": authority[
                "authorized_for_scored_experiments"
            ],
        },
        "operator": {
            "source_sites": [0, 1, 2],
            "directions_per_site": 32,
            "final_behavioral_target_site": 3,
            "categorical_fisher_probes_per_row": 16,
            "output_score_support": (
                "for row b, all positions t >= frozen_position[b] through 255"
            ),
            "every_direction_on_every_row": True,
            "primary_shape_at_cut3": [split_summary["primary"]["rows"] * 16, 96],
            "replication_shape_at_cut3": [split_summary["replication"]["rows"] * 16, 96],
        },
        "seeds": {
            "covariance_directions": plan.direction_seed,
            "categorical_fisher_probes": plan.probe_seed,
            "scored_positions": plan.position_seed,
        },
        "scored_positions": list(plan.scored_positions),
        "scored_position_rule": (
            "64 + uint64_be(sha256(f'{position_seed}:{row_id}')[:8]) mod 192"
        ),
        "direction_rule": {
            "covariance_support": "exact MLP writes on every row at positions 64:256",
            "psd_rtol": 1e-10,
            "support_rtol": 1e-12,
            "rademacher_seed": "direction_seed + 1000003*site + direction",
            "normalization": "unit coordinate RMS per direction",
        },
        "categorical_fisher_rule": {
            "probe_seeds": [
                plan.probe_seed + index for index in range(plan.probes_per_row)
            ],
            "uniform": (
                "(uint64_be(sha256(f'{seed}:{row_id}:{absolute_position}')[:8])"
                "+0.5)/2**64"
            ),
            "inverse_cdf": (
                "full 50304-way float32 softmax; clamped CDF; searchsorted right=True"
            ),
        },
        "splits": split_summary,
        "whole_document_split": True,
        "unique_source_documents": len(set(document_ids)),
        "remaining_authority": (
            "A separately committed model-side Fisher-VJP collector and independent "
            "lifecycle audit are still required before GPU execution."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "plan_fingerprint": plan.fingerprint,
        "splits": split_summary,
        "unique_source_documents": result["unique_source_documents"],
    }, indent=2))


if __name__ == "__main__":
    main()
