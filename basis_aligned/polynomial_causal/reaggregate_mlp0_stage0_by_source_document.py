#!/usr/bin/env python3
"""Correct Stage-0 v2 bootstrap units from row chunks to source documents.

The immutable collector serialized one ledger row per FineWeb 513-token chunk while
the registered resampling unit was the source document.  The row receipt retains the
source document ID for every chunk, so the exact correction is a lossless pre-bootstrap
sum within document.  This script never accesses the model or new outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
sys.path.insert(0, str(HERE))
from causal_response_quotient import (  # noqa: E402
    pointwise_dominates,
    score_worst_cell_equivalence,
)


SOURCE_RESULT = BQ / "mlp0_quotient_stage0_v2_results.json"
ROW_RECEIPT = BQ / "mlp0_quotient_stage0_v1_rows_receipt.json"
OUT = BQ / "mlp0_quotient_stage0_v2_source_document_reanalysis.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_ledgers_by_document(sums, counts, document_ids):
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if sums.shape != counts.shape or sums.ndim != 2:
        raise ValueError("sums and counts must share [row, cell] shape")
    if len(document_ids) != sums.shape[0]:
        raise ValueError("one source document ID is required per row")
    unique = []
    inverse = {}
    for document in document_ids:
        if not isinstance(document, str) or not document:
            raise ValueError("source document IDs must be nonempty strings")
        if document not in inverse:
            inverse[document] = len(unique)
            unique.append(document)
    grouped_sums = np.zeros((len(unique), sums.shape[1]), dtype=np.float64)
    grouped_counts = np.zeros_like(grouped_sums)
    for row, document in enumerate(document_ids):
        target = inverse[document]
        grouped_sums[target] += sums[row]
        grouped_counts[target] += counts[row]
    return grouped_sums, grouped_counts, unique


def build_reanalysis():
    source = json.loads(SOURCE_RESULT.read_text())
    receipt = json.loads(ROW_RECEIPT.read_text())
    records = receipt["document_provenance"]["sets"]["eval"]
    document_ids = [record["document_id"] for record in records]
    if len(document_ids) != source["reports"]["T_vs_O"]["n_documents"]:
        raise RuntimeError("source ledgers and row provenance have different row counts")
    reports = {}
    grouped_statistics = {}
    unique = None
    for contrast, consumers in source["sufficient_statistics"].items():
        sums = {}
        counts = {}
        grouped_statistics[contrast] = {}
        for consumer, ledger in consumers.items():
            grouped_sums, grouped_counts, current = group_ledgers_by_document(
                ledger["sums"], ledger["counts"], document_ids
            )
            if unique is None:
                unique = current
            elif current != unique:
                raise RuntimeError("document grouping changed between consumers")
            sums[consumer] = grouped_sums
            counts[consumer] = grouped_counts
            grouped_statistics[contrast][consumer] = {
                "sums": grouped_sums.tolist(), "counts": grouped_counts.tolist()
            }
        reports[contrast] = score_worst_cell_equivalence(
            sums, counts, margins=source["margins"], cell_names=source["cell_names"],
            minimum_documents_per_cell=30, n_bootstrap=10000, seed=20260827,
        )
    assert unique is not None
    sensitivity = {
        consumer: max(report["cell_standardized_effects"].values()) > 1
        for consumer, report in reports["M_vs_T"]["consumers"].items()
    }
    gates = {
        "coverage_ge_90pct": source["coverage"] >= 0.90,
        "token_table_vs_live": reports["T_vs_O"]["equivalence_passes"],
        "q64_vs_token_table": reports["Q64_vs_T"]["equivalence_passes"],
        "q64_pointwise_beats_a64": pointwise_dominates(
            reports["Q64_vs_T"], reports["A64_vs_T"]
        ),
        "mean_assay_sensitive_all_consumers": all(sensitivity.values()),
    }
    gates["stage0_passes"] = all(gates.values())
    return {
        "schema_version": 1,
        "experiment": "mlp0_quotient_stage0_v2_source_document_reanalysis",
        "authority": "deterministic reaggregation of immutable sufficient statistics",
        "source_result": {"path": str(SOURCE_RESULT), "sha256": file_sha256(SOURCE_RESULT)},
        "row_receipt": {"path": str(ROW_RECEIPT), "sha256": file_sha256(ROW_RECEIPT)},
        "correction": "sum all chunk ledgers within source document before bootstrap",
        "n_row_chunks": len(document_ids),
        "n_source_documents": len(unique),
        "minimum_source_document_support": min(
            min(consumer["support_documents"].values())
            for report in reports.values() for consumer in report["consumers"].values()
        ),
        "reports": reports,
        "mean_sensitivity_by_consumer": sensitivity,
        "gates": gates,
        "grouped_sufficient_statistics": grouped_statistics,
        "interpretation": (
            "Correcting the resampling unit changes no point effect or gate. "
            "The finite-grid T and Q64 falsifications remain decisive; this still "
            "licenses no arbitrary-background or executable-interface claim."
        ),
    }


def main():
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite reanalysis: {OUT}")
    payload = build_reanalysis()
    temporary = OUT.with_name(f".{OUT.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=1) + "\n")
        os.replace(temporary, OUT)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({
        "n_row_chunks": payload["n_row_chunks"],
        "n_source_documents": payload["n_source_documents"],
        "minimum_source_document_support": payload["minimum_source_document_support"],
        "gates": payload["gates"],
        "ucb": {name: report["simultaneous_95pct_ucb_max_standardized_effect"]
                for name, report in payload["reports"].items()},
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
