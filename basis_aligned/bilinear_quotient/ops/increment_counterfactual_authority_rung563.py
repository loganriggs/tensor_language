#!/usr/bin/env python3
"""Generate the natural-prompt correction to R562 without opening model outcomes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = ROOT / "bilinear_quotient"
SOURCE = HERE / "increment_counterfactual_authority_rung562.py"
OUT = BQ / "increment_counterfactual_authority_rung563.json"
RECEIPT = BQ / "increment_counterfactual_authority_rung563_receipt.json"
PREREG = ROOT / "polynomial_causal" / "INCREMENT_COUNTERFACTUAL_AUTHORITY_RUNG563_CORRECTION.md"

spec = importlib.util.spec_from_file_location("r562_source", SOURCE)
assert spec and spec.loader
r562 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r562)


def clean_lead(lead: str) -> str:
    return lead.split(" [", 1)[0].removesuffix(" in words")


def natural_prompt(lead: str, values: tuple, words: tuple[str, str, str], style: int) -> str:
    lead = clean_lead(lead)
    a, b, c = values
    w0, w1, w2 = words
    if style == 0:
        return f"{lead}: {a} {w0}; {b} {w1}; {c} {w2}. Next:"
    if style == 1:
        return f"{lead} -- {a} {w0}, then {b} {w1}, then {c} {w2}. Next:"
    if style == 2:
        return f"{lead}: item {a} is {w0}; item {b} is {w1}; item {c} is {w2}. Next item:"
    return f"{lead}. We recorded {a} {w0} / {b} {w1} / {c} {w2}. Continue with:"


def natural_word_prompt(lead: str, values: tuple[int, int, int], words: tuple[str, str, str], style: int) -> str:
    return natural_prompt(clean_lead(lead), tuple(r562.NUMBER_WORD[value] for value in values), words, style)


def main() -> None:
    r562.OUT = OUT
    r562.RECEIPT = RECEIPT
    r562.PREREG = PREREG
    r562.digit_prompt = natural_prompt
    r562.word_prompt = natural_word_prompt
    r562.main()

    rows = json.loads(OUT.read_text())
    rows["schema"] = "increment_counterfactual_authority_rung563_v1"
    rows["correction_of"] = "increment_counterfactual_authority_rung562_v1"
    rows["family_revealing_prompt_labels"] = False
    rows_payload = (json.dumps(rows, indent=2, sort_keys=True) + "\n").encode()
    OUT.write_bytes(rows_payload)

    sequence_ids = [r562.content_id(ids) for row in rows["rows"] for ids in (row["base_ids"], row["donor_ids"])]
    receipt = json.loads(RECEIPT.read_text())
    receipt.update({
        "schema": "increment_counterfactual_authority_rung563_receipt_v1",
        "rows_path": str(OUT.relative_to(ROOT)),
        "rows_sha256": hashlib.sha256(rows_payload).hexdigest(),
        "preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "family_revealing_prompt_labels": False,
        "unique_token_sequence_count": len(set(sequence_ids)),
        "token_sequence_count": len(sequence_ids),
        "endpoint_reuse_within_semantic_groups_allowed": True,
    })
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
