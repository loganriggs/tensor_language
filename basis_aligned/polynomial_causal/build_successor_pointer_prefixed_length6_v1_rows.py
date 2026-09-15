#!/usr/bin/env python3
"""Freeze outcome-blind prefixed length-six rows for successor interaction transfer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "SUCCESSOR_POINTER_PREFIXED_LENGTH6_V1_ROWS.json"
FAMILIES = {
    "month": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "digit": [str(i) for i in range(10)],
}
FINAL_INDICES = {"month": range(5, 10), "digit": range(5, 8)}
PREFIXES = {
    "sequence": "Sequence: ",
    "pattern": "Pattern: ",
    "continue": "Continue: ",
    "items": "Items: ",
}
CONDITIONS = ("coherent", "early_swap_control", "late_swap_incoherent")


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def one(text):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"not one token: {text!r} -> {ids}")
        return ids[0]

    rows = []
    for prefix_id, prefix in PREFIXES.items():
        for family, elements in FAMILIES.items():
            for final_index in FINAL_INDICES[family]:
                coherent = elements[final_index - 5:final_index + 1]
                orders = {
                    "coherent": coherent,
                    "early_swap_control": [coherent[1], coherent[0], *coherent[2:]],
                    "late_swap_incoherent": [*coherent[:3], coherent[4], coherent[3], coherent[5]],
                }
                for condition, ordered in orders.items():
                    body = ordered[0] + "".join(", " + value for value in ordered[1:]) + ","
                    text = prefix + body
                    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                    expected_suffix = []
                    for value in ordered:
                        expected_suffix.extend([one(" " + value), one(",")])
                    if ids[-len(expected_suffix):] != expected_suffix:
                        raise RuntimeError(f"unexpected tokenization: {text!r} -> {ids}")
                    query = len(ids) - 1
                    row = {
                        "row_id": canonical(["prefixed_length6_v1", prefix_id, family, final_index, condition]),
                        "prefix_id": prefix_id,
                        "prefix": prefix,
                        "family": family,
                        "final_index": final_index,
                        "condition": condition,
                        "elements": ordered,
                        "text": text,
                        "token_ids": ids,
                        "last_position": query - 1,
                        "query_position": query,
                        "recipient_element": elements[final_index],
                        "recipient_token_id": one(" " + elements[final_index]),
                        "recipient_answer": elements[final_index + 1],
                        "recipient_answer_id": one(" " + elements[final_index + 1]),
                        "donors": {
                            "forward": {
                                "element": elements[final_index + 1],
                                "token_id": one(" " + elements[final_index + 1]),
                                "answer": elements[final_index + 2],
                                "answer_id": one(" " + elements[final_index + 2]),
                            },
                            "backward": {
                                "element": elements[final_index - 2],
                                "token_id": one(" " + elements[final_index - 2]),
                                "answer": elements[final_index - 1],
                                "answer_id": one(" " + elements[final_index - 1]),
                            },
                        },
                    }
                    if ids[row["last_position"]] != row["recipient_token_id"]:
                        raise RuntimeError("last-token identity mismatch")
                    rows.append(row)

    cell_counts = {
        prefix_id: {
            family: {
                condition: sum(
                    row["prefix_id"] == prefix_id and row["family"] == family and row["condition"] == condition
                    for row in rows
                )
                for condition in CONDITIONS
            }
            for family in FAMILIES
        }
        for prefix_id in PREFIXES
    }
    payload = {
        "schema": "successor_pointer_prefixed_length6_v1_rows",
        "selection_rule": "four fixed label prefixes crossed with every previously defined length-six month/digit endpoint and the same two transpositions; no model outcomes",
        "outcomes_opened": False,
        "model_loaded": False,
        "prefixes": PREFIXES,
        "families": list(FAMILIES),
        "conditions": list(CONDITIONS),
        "directions": ["forward", "backward"],
        "row_count": len(rows),
        "endpoint_count": 2 * len(rows),
        "cell_counts": cell_counts,
        "rows": rows,
    }
    payload["row_manifest_sha256"] = canonical(rows)
    payload["builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(OUT), "rows": len(rows), "manifest": payload["row_manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
