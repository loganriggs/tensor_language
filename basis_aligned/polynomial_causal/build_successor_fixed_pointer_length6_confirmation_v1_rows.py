#!/usr/bin/env python3
"""Freeze outcome-blind length-six rows for successor typed-group confirmation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_ROWS.json"
FAMILIES = {
    "month": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "digit": [str(i) for i in range(10)],
}
FINAL_INDICES = {"month": range(5, 10), "digit": range(5, 8)}
CONDITIONS = ("coherent", "early_swap_control", "late_swap_incoherent")


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main():
    tok = AutoTokenizer.from_pretrained("gpt2")

    def one(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"not one token: {text!r} -> {ids}")
        return ids[0]

    def prompt(elements):
        text = elements[0] + "".join(", " + value for value in elements[1:]) + ","
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) != 2 * len(elements):
            raise RuntimeError(f"unexpected tokenization: {text!r} -> {ids}")
        return text, ids

    rows = []
    for family, elements in FAMILIES.items():
        for final_index in FINAL_INDICES[family]:
            coherent = elements[final_index - 5:final_index + 1]
            orders = {
                "coherent": coherent,
                "early_swap_control": [coherent[1], coherent[0], *coherent[2:]],
                "late_swap_incoherent": [*coherent[:3], coherent[4], coherent[3], coherent[5]],
            }
            for condition in CONDITIONS:
                text, ids = prompt(orders[condition])
                row = {
                    "row_id": canonical(["length6", family, final_index, condition]),
                    "family": family, "final_index": final_index, "condition": condition,
                    "elements": orders[condition], "text": text, "token_ids": ids,
                    "last_position": len(ids) - 2, "query_position": len(ids) - 1,
                    "recipient_element": elements[final_index],
                    "recipient_token_id": one(" " + elements[final_index]),
                    "recipient_answer": elements[final_index + 1],
                    "recipient_answer_id": one(" " + elements[final_index + 1]),
                    "donors": {
                        "forward": {"element": elements[final_index + 1], "token_id": one(" " + elements[final_index + 1]), "answer": elements[final_index + 2], "answer_id": one(" " + elements[final_index + 2])},
                        "backward": {"element": elements[final_index - 2], "token_id": one(" " + elements[final_index - 2]), "answer": elements[final_index - 1], "answer_id": one(" " + elements[final_index - 1])},
                    },
                }
                if ids[row["last_position"]] != row["recipient_token_id"]:
                    raise RuntimeError("last-token identity mismatch")
                rows.append(row)
    counts = {family: {condition: sum(row["family"] == family and row["condition"] == condition for row in rows) for condition in CONDITIONS} for family in FAMILIES}
    payload = {
        "schema": "successor_fixed_pointer_length6_confirmation_v1_rows",
        "selection_rule": "all length-six month endpoints at indices 5..9 and digit endpoints at indices 5..7; fixed first-pair and penultimate-pair transpositions",
        "outcomes_opened": [], "model_loaded": False, "families": list(FAMILIES),
        "conditions": list(CONDITIONS), "directions": ["forward", "backward"],
        "row_count": len(rows), "intervention_count": 2 * len(rows), "cell_counts": counts, "rows": rows,
    }
    payload["row_manifest_sha256"] = canonical(rows)
    payload["builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(OUT), "rows": len(rows), "interventions": 2 * len(rows), "cells": counts, "row_manifest_sha256": payload["row_manifest_sha256"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
