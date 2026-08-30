#!/usr/bin/env python3
"""Outcome-blind executable price audit for factorization-v1 candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "causal_response_factorization_v1_candidate_price_audit.json"
PHASES = 2
SOURCES = 49
TARGETS = 49
TRAIN_DOCUMENTS = 229
OWNER_GROUP_SIZES = (16, 13, 6, 5, 5, 4)
SEEDS = (2026083001, 2026083002, 2026083003)
RANK_PAIRS = (
    *((rank, 0) for rank in (1, 2, 4, 8, 16, 32)),
    *((0, rank) for rank in (1, 2, 4, 8)),
    (1, 1), (2, 1), (4, 1), (4, 2), (8, 2), (8, 4), (16, 4),
)


@dataclass(frozen=True)
class CandidatePrice:
    global_rank: int
    private_rank_each_owner: int
    persistent_values: int
    per_document_values: int
    strict_dense_matched_rank: int
    amortized_total_dense_rank: int


def structured_price(global_rank: int, private_rank: int) -> tuple[int, int]:
    if global_rank < 0 or private_rank < 0 or global_rank + private_rank == 0:
        raise ValueError("candidate ranks must be nonnegative and nonzero together")
    persistent = global_rank * (PHASES + SOURCES + TARGETS)
    persistent += private_rank * sum(
        PHASES + owner_sources + TARGETS for owner_sources in OWNER_GROUP_SIZES
    )
    code = global_rank + len(OWNER_GROUP_SIZES) * private_rank
    return persistent, code


def audit_rows() -> tuple[CandidatePrice, ...]:
    observation_values = PHASES * SOURCES * TARGETS
    rows = []
    for global_rank, private_rank in RANK_PAIRS:
        persistent, code = structured_price(global_rank, private_rank)
        strict = min(persistent // observation_values, code)
        amortized = (persistent + TRAIN_DOCUMENTS * code) // (
            observation_values + TRAIN_DOCUMENTS
        )
        rows.append(CandidatePrice(
            global_rank=global_rank,
            private_rank_each_owner=private_rank,
            persistent_values=persistent,
            per_document_values=code,
            strict_dense_matched_rank=strict,
            amortized_total_dense_rank=amortized,
        ))
    return tuple(rows)


def build_receipt() -> dict[str, object]:
    rows = audit_rows()
    source = Path(__file__).resolve()
    return {
        "schema": "causal_response_factorization_v1_candidate_price_audit",
        "status": "complete_outcome_blind",
        "dimensions": {
            "phases": PHASES,
            "sources": SOURCES,
            "targets": TARGETS,
            "train_documents": TRAIN_DOCUMENTS,
            "owner_group_sizes": list(OWNER_GROUP_SIZES),
            "observation_values_per_dense_rank": PHASES * SOURCES * TARGETS,
        },
        "frozen_seeds": list(SEEDS),
        "structured_rank_pairs": len(rows),
        "structured_fits": len(rows) * len(SEEDS),
        "rows": [asdict(row) for row in rows],
        "conclusions": {
            "strict_dense_matched_rank_zero_for_every_candidate": all(
                row.strict_dense_matched_rank == 0 for row in rows
            ),
            "maximum_structured_persistent_values": max(
                row.persistent_values for row in rows
            ),
            "minimum_dense_rank_one_persistent_values": PHASES * SOURCES * TARGETS,
            "amortized_price_is_a_distinct_noncontrolling_view": True,
            "response_values_read": False,
        },
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("candidate price audit namespace is already spent")
    value = build_receipt()
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with OUTPUT.open("x") as sink:
        sink.write(raw)
    if json.loads(OUTPUT.read_bytes()) != value:
        raise RuntimeError("candidate price audit receipt did not replay")
    print(raw, end="")


if __name__ == "__main__":
    main()
