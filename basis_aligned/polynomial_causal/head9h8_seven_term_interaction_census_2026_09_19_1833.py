"""CPU replay census of the seven exact head9.8 finite-change terms.

This reads an existing primary artifact and writes a new derived receipt.  It
does not load the model or mutate the source artifact.
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "CITY_FINEWEB_HEAD9_FOLD_V1_ARTIFACT.pt"
ROWS = HERE / "CITY_ATTENTION7_DROP3_FINEWEB_V1_ROWS.json"
OUT = HERE / "HEAD9H8_SEVEN_TERM_INTERACTION_CENSUS_2026-09-19_1833.json"
EXPECTED = {
    ARTIFACT.name: "803ef5267f418cc6eb33ff8cd387290246af891097825e4ad20cb2117ca534fc",
    ROWS.name: "7db2588fec84842e090be85a66d89c0536970c213cea4dbd6fa2f5d340906590",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(terms: torch.Tensor) -> dict:
    single_norms = torch.stack([terms[i].norm() for i in range(3)])
    interaction = terms[3:].sum(0)
    full = terms.sum(0)
    return {
        "single_norms": [float(x) for x in single_norms],
        "interaction_norm": float(interaction.norm()),
        "full_delta_norm": float(full.norm()),
        "interaction_over_smallest_single": float(interaction.norm() / single_norms.min()),
        "interaction_omission_relative_l2": float(interaction.norm() / full.norm()),
        "triple_over_smallest_single": float(terms[6].norm() / single_norms.min()),
    }


def distribution(values: list[float]) -> dict:
    x = torch.tensor(values, dtype=torch.float64)
    return {
        "minimum": float(x.min()),
        "median": float(x.median()),
        "maximum": float(x.max()),
        "count": len(values),
    }


def main() -> None:
    started = datetime.now(timezone.utc)
    tic = time.perf_counter()
    observed = {p.name: sha256(p) for p in (ARTIFACT, ROWS)}
    assert observed == EXPECTED, (observed, EXPECTED)
    assert not OUT.exists()

    artifact = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    terms = artifact["contributions"].to(torch.float64)
    names = artifact["term_names"]
    rows_doc = json.loads(ROWS.read_text())
    rows = rows_doc["rows"]
    contexts = rows_doc["contexts"]
    assert names == [
        "deltaK1", "deltaK2", "deltaV", "deltaK1_deltaK2",
        "deltaK1_deltaV", "deltaK2_deltaV", "deltaK1_deltaK2_deltaV",
    ]
    assert tuple(terms.shape) == (7, 240, 5)
    assert len(contexts) == 20 and len(rows) == 240
    for context_id in range(20):
        ids = [r["context_id"] for r in rows[12 * context_id:12 * (context_id + 1)]]
        assert ids == [context_id] * 12

    cell_interaction = []
    cell_omission = []
    for context_id in range(20):
        cell = summary(terms[:, 12 * context_id:12 * (context_id + 1), 0])
        cell_interaction.append(cell["interaction_over_smallest_single"])
        cell_omission.append(cell["interaction_omission_relative_l2"])

    receipt = {
        "schema": "head9h8.seven_term_interaction_census.v1",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - tic,
        "execution": "CPU-only replay of stored float64 fold contributions; no model load or forward",
        "evidence_type": "fold",
        "evaluation_status": "opened replay",
        "term_names": names,
        "shape": list(terms.shape),
        "distinct_context_cells": len(contexts),
        "rows": len(rows),
        "target": summary(terms[:, :, 0]),
        "all_five_readouts": summary(terms),
        "four_controls": summary(terms[:, :, 1:]),
        "target_cell_distribution": {
            "interaction_over_smallest_single": distribution(cell_interaction),
            "interaction_omission_relative_l2": distribution(cell_omission),
            "cells_interaction_over_0_35": sum(x > 0.35 for x in cell_interaction),
            "cells_omission_over_0_10": sum(x > 0.10 for x in cell_omission),
        },
        "interpretation": (
            "The unique Boolean/Mobius coefficients of the tri-affine local head change "
            "show that collectively omitting the four order>=2 terms is not a small local "
            "approximation on these opened cells. This is replay attribution, not a fresh "
            "causal, random-split-specific, or suffix-fidelity result."
        ),
        "source_sha256": observed,
    }
    OUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
