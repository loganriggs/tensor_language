#!/usr/bin/env python3
"""Freeze a third bracket construction for the one licensed direct-readout test."""

# BQLANE: cpu
from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
from build_bracket_suffix_free_fresh_corpus_v1 import DELIMITERS, PAIRS, digest, encode, one

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/prior_art/bracket_l13h8_direct_readout_fresh_corpus_v1_rows.json"
SCHEMA = "bracket_l13h8_direct_readout_fresh_corpus_rows_v1"
PREFIXES = (
    "The cartographer", "A watchmaker", "The botanist",
    "One architect", "The geologist", "A composer",
)
WORDS = (
    "anchor", "button", "castle", "dragon", "engine", "forest",
    "glacier", "helmet", "island", "kernel", "lantern", "mirror",
    "orchard", "planet", "quartz", "rocket", "silver", "temple",
)


def build_rows() -> list[dict]:
    rng = random.Random(202609060142)
    rows = []
    for left_index, right_index in PAIRS:
        left, right = DELIMITERS[left_index], DELIMITERS[right_index]
        distractor = DELIMITERS[({0, 1, 2} - {left_index, right_index}).pop()]
        for replicate, prefix in enumerate(PREFIXES):
            w0, w1, w2, w3, w4 = rng.sample(WORDS, 5)
            starts = (
                f"After indexing the archive, {prefix.lower()} fastened",
                f"When the catalog was ready, {prefix.lower()} secured",
                f"Before leaving the workshop, {prefix.lower()} finished",
            )
            common = (
                f"{starts[replicate % 3]} {distractor['open']} the {w0} and the {w1} "
                f"{distractor['close']}; later the supplement continued"
            )
            tail = f"the {w2}, the {w3}, and the {w4} without an ending symbol"
            base = f"{common} {left['open']} {tail}"
            donor = f"{common} {right['open']} {tail}"
            base_ids, donor_ids = encode(base), encode(donor)
            differences = [i for i, (a, b) in enumerate(zip(base_ids, donor_ids)) if a != b]
            if len(base_ids) != len(donor_ids) or len(differences) != 1:
                raise ValueError(f"not a one-token aligned opener substitution: {base!r}")
            coordinates = {
                "family": "archive_completed_distractor_pending_type_substitution",
                "pair": [left_index, right_index], "replicate": replicate,
                "prefix": prefix, "words": [w0, w1, w2, w3, w4],
            }
            rows.append({
                "row_id": digest(coordinates),
                "split": "PROSPECTIVE_DIRECT_READOUT_V1",
                "family_id": "archive_completed_distractor_pending_type_substitution",
                "program_role": "target",
                "base_text": base, "donor_text": donor,
                "base_ids": base_ids, "donor_ids": donor_ids,
                "base_answer": left["close"], "donor_answer": right["close"],
                "base_answer_id": one(left["close"]), "donor_answer_id": one(right["close"]),
                "evaluation_directions": ["base_to_donor", "donor_to_base"],
                "semantic_open_position": differences[0],
                "construction_checks": {
                    "roundtrip": True, "equal_token_length": True,
                    "single_token_difference": True,
                    "completed_distractor_type": distractor["name"],
                },
            })
    counts = Counter((row["base_answer_id"], row["donor_answer_id"]) for row in rows)
    if len(rows) != 36 or len({row["row_id"] for row in rows}) != 36 or len(counts) != 6 or set(counts.values()) != {6}:
        raise ValueError("fresh direct-readout corpus balance failed")
    return rows


def main() -> None:
    if OUT.exists():
        raise ValueError(f"refusing to overwrite {OUT}")
    rows = build_rows()
    value = {
        "schema": SCHEMA,
        "status": "rows_frozen_outcomes_unopened",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "family_id": "archive_completed_distractor_pending_type_substitution",
        "row_count": 36, "endpoint_count": 72,
        "ordered_pair_row_counts": {
            f"{a}->{b}": count for (a, b), count in sorted(
                Counter((row["base_answer_id"], row["donor_answer_id"]) for row in rows).items()
            )
        },
        "model_loaded": False, "model_forwards": 0, "outcomes_opened": [],
        "rows": rows,
    }
    payload = managed.atomic_create_json(OUT, value)
    import hashlib
    print(json.dumps({"rows": 36, "endpoints": 72, "outcomes_opened": [], "sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
