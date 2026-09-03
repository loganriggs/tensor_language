#!/usr/bin/env python3
"""Generate balanced v2 rows using the frozen R543 construction functions."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "ops" / "pending_opener_unique_rows_rung543.py"
spec = importlib.util.spec_from_file_location("r543_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

base.OUT = ROOT / "pending_opener_unique_rows_rung543_v2.json"
base.RECEIPT = ROOT / "pending_opener_unique_rows_rung543_v2_receipt.json"
base.PREREG = ROOT.parent / "polynomial_causal" / "PENDING_OPENER_UNIQUE_FOUR_CLOSER_ROWS_RUNG543_V2_CORRECTION.md"


def balanced_groups(split: str, split_spec: dict) -> list[tuple]:
    del split
    per_pair = split_spec["count"] // len(base.ORDERED_TYPE_PAIRS)
    assert per_pair * len(base.ORDERED_TYPE_PAIRS) == split_spec["count"]
    generator = random.Random(split_spec["seed"])
    selected, seen_invariance = [], set()
    for pair in base.ORDERED_TYPE_PAIRS:
        accepted = 0
        while accepted < per_pair:
            candidate = (
                generator.choice(split_spec["prefixes"]),
                tuple(generator.sample(split_spec["words"], 5)),
                pair,
                generator.randrange(3),
            )
            invariant_identity = (candidate[0], candidate[1], pair[0], candidate[3])
            if invariant_identity in seen_invariance:
                continue
            seen_invariance.add(invariant_identity)
            selected.append(candidate)
            accepted += 1
    return selected


def main() -> None:
    base.semantic_groups = balanced_groups
    base.main()
    rows = json.loads(base.OUT.read_text())
    rows["schema"] = "pending_opener_unique_four_closer_rows_rung543_v2"
    payload = (json.dumps(rows, indent=2, sort_keys=True) + "\n").encode()
    base.OUT.write_bytes(payload)
    receipt = json.loads(base.RECEIPT.read_text())
    receipt["schema"] = "pending_opener_unique_four_closer_rows_rung543_v2_receipt"
    receipt["rows_sha256"] = hashlib.sha256(payload).hexdigest()
    base.RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
