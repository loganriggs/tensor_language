#!/usr/bin/env python3
# BQLANE: cpu
"""Build fresh, model-free rows for a spaced-value successor authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tiktoken


HERE = Path(__file__).resolve().parent
OUT = HERE / "NUMERIC_SUCCESSOR_SPACED_OOD_V1_ROWS.json"
ENC = tiktoken.get_encoding("gpt2")
WORDS = (
    ("amber", "brook", "cedar"), ("dahlia", "estuary", "finch"),
    ("garnet", "heather", "islet"), ("juniper", "keystone", "lilac"),
    ("magnet", "nectar", "onyx"), ("prairie", "quartz", "raven"),
    ("spruce", "thimble", "upland"), ("willow", "yarrow", "zephyr"),
)
NUMBER_WORD = {2:"two", 3:"three", 4:"four", 6:"six", 7:"seven", 8:"eight",
               9:"nine", 10:"ten", 11:"eleven", 12:"twelve"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def endpoint(text, answer):
    ids = ENC.encode(text)
    answer_ids = ENC.encode(answer)
    if ENC.decode(ids) != text or len(answer_ids) != 1:
        raise ValueError((text, answer, answer_ids))
    return {"text":text, "ids":ids, "answer":answer, "answer_id":answer_ids[0]}


def list_text(values, words, headered):
    body = "".join(f"{value}. {word}\n" for value, word in zip(values, words, strict=True))
    return ("Reading queue:\n" + body) if headered else body


def digit_text(value, word, headered):
    return (f"Copy this stable sequence for {word}: {value}, {value}, {value}," if headered
            else f"The {word} sequence stays {value}, {value}, {value},")


def word_text(value, word, headered):
    rendered = NUMBER_WORD[value]
    return (f"Copy this stable sequence for {word}: {rendered}, {rendered}, {rendered}," if headered
            else f"The {word} sequence stays {rendered}, {rendered}, {rendered},")


def main():
    rows = []
    for construction, headered in (("fresh_plain", False), ("fresh_headered", True)):
        for index, words in enumerate(WORDS):
            # The two endpoint states are separated by five, so neither expected
            # successor is the other endpoint's final visible label.
            low = (2, 4, 6) if index % 2 == 0 else (3, 5, 7)
            high = tuple(value + 5 for value in low)
            specs = {
                "list_step_two_spaced": (
                    endpoint(list_text(low, words, headered), str(low[-1] + 1)),
                    endpoint(list_text(high, tuple(reversed(words)), headered), str(high[-1] + 1)),
                ),
                "digit_copy_spaced_control": (
                    endpoint(digit_text(low[0], words[0], headered), f" {low[0]}"),
                    endpoint(digit_text(high[0], words[1], headered), f" {high[0]}"),
                ),
                "word_copy_spaced_control": (
                    endpoint(word_text(low[0], words[0], headered), " " + NUMBER_WORD[low[0]]),
                    endpoint(word_text(high[0], words[1], headered), " " + NUMBER_WORD[high[0]]),
                ),
            }
            for role, (base, donor) in specs.items():
                row_id = digest([construction, index, role, base["text"], donor["text"]])
                rows.append({"row_id":row_id, "construction":construction, "program_role":role,
                             "direction_policy":"evaluate_both", "base":base, "donor":donor})
    manifest = digest(rows)
    result = {"schema":"numeric_successor_spaced_ood_v1_rows", "row_count":len(rows),
              "endpoint_count":2*len(rows), "row_manifest_sha256":manifest, "rows":rows}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key:result[key] for key in ("schema","row_count","endpoint_count","row_manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
