#!/usr/bin/env python3
"""Post-hoc matched-null audit of corrected context-interaction rank-one energy."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_CORRECTED_V1_RESULT.json"
OUT = ROOT / "SUBJECT_NUMBER_CONTEXT_INTERACTION_RANK_SPECIFICITY_V1_RESULT.json"
NULLS, SEED = 10_000, 20260923


def rank1_energy(matrix):
    interaction = (matrix - matrix.mean(axis=1, keepdims=True)
                   - matrix.mean(axis=0, keepdims=True) + matrix.mean())
    singular = np.linalg.svd(interaction, compute_uv=False)
    return float(singular[0] ** 2 / np.sum(singular ** 2))


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    source = json.loads(SOURCE.read_text())
    matrix = np.asarray([record["alpha"] for record in source["records"]]).reshape(32, 4)
    observed = rank1_energy(matrix); rng = np.random.default_rng(SEED)
    null = np.asarray([rank1_energy(np.stack([row[rng.permutation(4)] for row in matrix]))
                       for _ in range(NULLS)], dtype=np.float64)
    result = {"schema": "subject_number_context_interaction_rank_specificity_v1_result",
              "terminal": "posthoc_rank1_specificity_supported",
              "posthoc": True, "source_terminal": source["terminal"],
              "observed_rank1_energy": observed, "nulls": NULLS, "seed": SEED,
              "null_median": float(np.median(null)),
              "null_q95": float(np.quantile(null, .95)),
              "null_q99": float(np.quantile(null, .99)),
              "null_maximum": float(null.max()),
              "upper_tail_fraction": float(np.mean(null >= observed)),
              "null_float64_sha256": hashlib.sha256(null.tobytes()).hexdigest(),
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": ("Post-hoc red-team of whether four-context rank-one energy exceeds "
                        "within-subject column-permutation structure; descriptive only.")}
    OUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
