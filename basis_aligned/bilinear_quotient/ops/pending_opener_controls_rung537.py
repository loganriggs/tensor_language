#!/usr/bin/env python3
"""Freeze matched non-opener and wrong-closer controls for rung 537, CPU only."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "bilinear_quotient"
SOURCE = BQ / "pending_opener_multifamily_rows_rung537.json"
OUT = BQ / "pending_opener_controls_rung537.json"
RECEIPT = BQ / "pending_opener_controls_rung537_receipt.json"
ENC = tiktoken.get_encoding("gpt2")


def main() -> None:
    source = json.loads(SOURCE.read_text())
    direct = [row for row in source["rows"] if row["family_id"] == "opener_type_substitution"]
    rows = []
    for row in direct:
        prefix, body = row["base_text"].split(" said ( ", 1)
        base_text = f"{prefix} noted, ( {body}"
        donor_text = f"{prefix} noted: ( {body}"
        base_ids, donor_ids = ENC.encode(base_text), ENC.encode(donor_text)
        differences = [index for index, pair in enumerate(zip(base_ids, donor_ids)) if pair[0] != pair[1]]
        assert len(base_ids) == len(donor_ids)
        assert len(differences) == 1
        assert ENC.decode([base_ids[differences[0]]]) == ","
        assert ENC.decode([donor_ids[differences[0]]]) == ":"
        closer_id = ENC.encode(")")[0]
        wrong_closer_ids = [ENC.encode(token)[0] for token in ("]", "}")]
        rows.append({
            "row_id": f"{row['group_id']}-nonopener-punctuation",
            "group_id": row["group_id"],
            "split": row["split"],
            "family_id": "nonopener_punctuation_substitution",
            "role": "invariance",
            "base_text": base_text,
            "donor_text": donor_text,
            "base_ids": base_ids,
            "donor_ids": donor_ids,
            "answer": ")",
            "answer_id": closer_id,
            "wrong_closer_tokens": ["]", "}"],
            "wrong_closer_ids": wrong_closer_ids,
            "proposed_variable_base": "pending_paren",
            "proposed_variable_donor": "pending_paren",
            "construction_checks": {
                "equal_token_length": True,
                "single_token_difference": True,
                "edit_is_before_opener": differences[0] < base_ids.index(ENC.encode(" (")[0]),
                "opener_and_suffix_identical": base_ids[differences[0] + 1:] == donor_ids[differences[0] + 1:],
            },
        })
    assert len(rows) == source["group_count"]
    result = {
        "schema": "pending_opener_controls_rung537_v1",
        "status": "controls_frozen_outcomes_unopened",
        "source_rows_path": str(SOURCE.relative_to(ROOT)),
        "source_rows_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "controls": {
            "nonopener_punctuation_substitution": (
                "comma-to-colon single-token edit before an unchanged pending parenthesis"
            ),
            "wrong_closer_logits": (
                "square/curly closer logits measured beside the correct parenthesis/quote endpoints"
            ),
            "random_subspace": "same rank and intervention norm, five frozen seeds at model-execution time",
        },
        "row_count": len(rows),
        "rows": rows,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    OUT.write_bytes(payload)
    receipt = {
        "schema": "pending_opener_controls_rung537_receipt_v1",
        "controls_path": str(OUT.relative_to(ROOT)),
        "controls_sha256": hashlib.sha256(payload).hexdigest(),
        "source_rows_sha256": result["source_rows_sha256"],
        "row_count": len(rows),
        "split_counts": dict(sorted(collections.Counter(row["split"] for row in rows).items())),
        "all_single_token": all(row["construction_checks"]["single_token_difference"] for row in rows),
        "all_pending_state_preserved": all(row["proposed_variable_base"] == row["proposed_variable_donor"] for row in rows),
        "outcomes_opened": False,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
