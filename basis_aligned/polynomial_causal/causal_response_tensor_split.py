"""Freeze a source-document-disjoint split for causal response tensor collection."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import torch


SPLIT_SEED = 184
MIN_COMPONENT_CIRCUITS = 4


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_side(document_id: int, seed: int = SPLIT_SEED) -> str:
    digest = hashlib.sha256(f"{seed}:{document_id}".encode()).digest()
    return "FIT" if (digest[0] & 1) == 0 else "EVAL"


def freeze_split(
    census_state_path: Path,
    curated_rows_path: Path,
    battery_path: Path,
) -> dict[str, object]:
    started = time.monotonic()
    state = torch.load(census_state_path, map_location="cpu", weights_only=False)
    curated = torch.load(curated_rows_path, map_location="cpu", weights_only=False)
    battery = json.loads(battery_path.read_text())
    rows = state["rows"]
    if not torch.equal(rows, curated["rows"]):
        raise ValueError("census-state rows do not exactly equal curated rows")
    if rows.shape[0] != curated["docid"].shape[0]:
        raise ValueError("row/document-ID cardinality mismatch")

    row_document_ids = curated["docid"].to(torch.int64)
    unique_document_ids = sorted(set(map(int, row_document_ids.tolist())))
    side_by_document = {
        document_id: document_side(document_id) for document_id in unique_document_ids
    }
    fit_documents = {d for d, side in side_by_document.items() if side == "FIT"}
    eval_documents = {d for d, side in side_by_document.items() if side == "EVAL"}
    if fit_documents & eval_documents:
        raise AssertionError("document split overlaps")

    positions_per_row = state["basev"].numel() // rows.shape[0]
    if positions_per_row * rows.shape[0] != state["basev"].numel():
        raise ValueError("base-vector length does not factor by row count")
    position_document_ids = row_document_ids[:, None].expand(
        -1, positions_per_row
    ).reshape(-1)
    leaves = {leaf["tag"]: leaf for leaf in state["leaves"]}

    component_counts = collections.Counter(
        entry["best_mean"] for entry in battery["by_tag"].values()
    )
    components = sorted(
        [key for key, count in component_counts.items() if count >= MIN_COMPONENT_CIRCUITS],
        key=lambda key: (-component_counts[key], key),
    )
    by_component: dict[str, object] = {}
    minimum_support = None
    for component in components:
        circuits: dict[str, object] = {}
        tags = [
            tag
            for tag, entry in battery["by_tag"].items()
            if entry["best_mean"] == component and tag in leaves
        ]
        for tag in tags:
            member_indices = leaves[tag]["member"].to(torch.int64)
            member_documents = set(
                map(int, position_document_ids[member_indices].tolist())
            )
            fit_support = len(member_documents & fit_documents)
            eval_support = len(member_documents & eval_documents)
            minimum_support = min(
                fit_support,
                eval_support,
                minimum_support if minimum_support is not None else fit_support,
            )
            circuits[tag] = {
                "member_positions": int(member_indices.numel()),
                "member_unique_documents": len(member_documents),
                "fit_member_documents": fit_support,
                "eval_member_documents": eval_support,
            }
        by_component[component] = {
            "battery_winner_count": component_counts[component],
            "usable_circuit_count": len(circuits),
            "circuits": circuits,
        }

    return {
        "schema": "causal_response_tensor_document_split_v1",
        "claim_boundary": (
            "Metadata-only CPU split. No model import, checkpoint load, forward call, "
            "intervention response, protected outcome, or tensor fit."
        ),
        "split_rule": "FIT iff sha256(f'{seed}:{source_document_id}')[0] is even",
        "split_seed": SPLIT_SEED,
        "parents": {
            "census_state_path": str(census_state_path.resolve()),
            "census_state_sha256": file_sha256(census_state_path),
            "curated_rows_path": str(curated_rows_path.resolve()),
            "curated_rows_sha256": file_sha256(curated_rows_path),
            "battery_path": str(battery_path.resolve()),
            "battery_sha256": file_sha256(battery_path),
        },
        "rows_exactly_match": True,
        "rows": int(rows.shape[0]),
        "positions_per_row": positions_per_row,
        "unique_source_documents": len(unique_document_ids),
        "fit_source_documents": len(fit_documents),
        "eval_source_documents": len(eval_documents),
        "cross_role_document_overlap": 0,
        "components": components,
        "total_usable_circuits": sum(
            entry["usable_circuit_count"] for entry in by_component.values()
        ),
        "minimum_member_document_support_each_role": minimum_support,
        "by_component": by_component,
        "runtime_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1] / "bilinear_quotient"
    parser.add_argument(
        "--census-state", type=Path, default=base / "census_state_diverse.pt"
    )
    parser.add_argument(
        "--curated-rows", type=Path, default=base / "curated_rows.pt"
    )
    parser.add_argument(
        "--battery", type=Path, default=base / "circuits" / "BATTERY.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "causal_response_tensor_document_split.json",
    )
    args = parser.parse_args()
    receipt = freeze_split(args.census_state, args.curated_rows, args.battery)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
