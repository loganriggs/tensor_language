#!/usr/bin/env python3
"""R566 development-only format screen; its prompts can never be circuit evidence.

Pred A: the legacy two-line digit list reaches >=.75 +1 candidate accuracy.
Pred B: at least one nonlegacy digit format reaches >=.75 +1 accuracy.
Pred C: at least one number-word format reaches >=.50 +1 accuracy.
Price: 432 unique sequences, 14 forwards at batch 32, zero backwards.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import tiktoken


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
for search_path in (ROOT, ROOT / "ops", ROOT.parent / "polynomial_causal"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
import bilin18_observed_model_facade as facade  # noqa: E402

OUT = ROOT / "increment_format_exploration_rung566_results.json"
BATCH = 32
FORMATS = (
    "numbered_list_2_digits", "numbered_list_3_digits", "comma_digits", "arrow_digits",
    "comma_words", "sentence_words",
)
RULES = ("plus_one", "copy", "plus_two")
WORDS = ("dogs", "cats", "birds", "books", "trees", "chairs", "apples", "roads", "songs", "doors",
         "coins", "stars", "plants", "shoes", "cups", "maps", "rivers", "clouds", "horses", "ships",
         "walls", "boxes", "hats", "keys")
NUMBER_WORD = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
               "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
               "nineteen", "twenty")
ENC = tiktoken.get_encoding("gpt2")
PRED_KEYS = ("pred_a_legacy_anchor pred_b_nonlegacy_digit_format pred_c_number_word_format "
             "model_forwards next_step").split()


def sequence(rule: str, start: int) -> tuple[tuple[int, ...], int]:
    if rule == "plus_one":
        return (start, start + 1, start + 2), start + 3
    if rule == "copy":
        return (start, start, start), start
    return (start, start + 2, start + 4), start + 6


def prompt(format_name: str, values: tuple[int, ...], answer: int, variant: int) -> tuple[str, str, str]:
    a, b, c = values
    w0, w1, w2 = WORDS[variant], WORDS[(variant + 5) % len(WORDS)], WORDS[(variant + 11) % len(WORDS)]
    if format_name == "numbered_list_2_digits":
        return f"{a}. {w0}\n{b}. {w1}\n", str(c), "digit_plain"
    if format_name == "numbered_list_3_digits":
        return f"{a}. {w0}\n{b}. {w1}\n{c}. {w2}\n", str(answer), "digit_plain"
    if format_name == "comma_digits":
        return f"The {w0} number sequence is {a}, {b}, {c},", f" {answer}", "digit_space"
    if format_name == "arrow_digits":
        return f"For {w0}, continue {a} -> {b} -> {c} ->", f" {answer}", "digit_space"
    rendered = tuple(NUMBER_WORD[value] for value in values)
    answer_word = NUMBER_WORD[answer]
    if format_name == "comma_words":
        return f"The {w0} number sequence is {rendered[0]}, {rendered[1]}, {rendered[2]},", f" {answer_word}", "word_space"
    return (f"For the {w0}, first came {rendered[0]}, then {rendered[1]}, then {rendered[2]}. Next came",
            f" {answer_word}", "word_space")


def candidates(kind: str) -> torch.Tensor:
    strings = ([str(value) for value in range(21)] if kind == "digit_plain" else
               [" " + str(value) for value in range(21)] if kind == "digit_space" else
               [" " + word for word in NUMBER_WORD])
    ids = []
    for text in strings:
        encoded = ENC.encode(text)
        if len(encoded) == 1:
            ids.append(encoded[0])
    return torch.tensor(sorted(set(ids)), dtype=torch.long)


def build() -> list[dict]:
    rows = []
    for format_name in FORMATS:
        for rule in RULES:
            for variant in range(24):
                values, answer = sequence(rule, 1 + variant % 7)
                text, answer_text, kind = prompt(format_name, values, answer, variant)
                answer_ids = ENC.encode(answer_text)
                assert len(answer_ids) == 1 and ENC.decode(ENC.encode(text)) == text
                rows.append({"format": format_name, "rule": rule, "text": text, "ids": ENC.encode(text),
                             "answer_id": answer_ids[0], "answer": answer_text, "candidate_kind": kind})
    assert len(rows) == 432
    assert len({tuple(row["ids"]) for row in rows}) == 432
    return rows


def native_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
    x0, v1 = x, None
    for block in model.transformer.h:
        x, v1 = block(x, v1, x0)
    return (30 * torch.tanh(model.lm_head(F.rms_norm(x, (x.size(-1),))) / 30)).float()


def main() -> None:
    rows = build()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({"status": "dryrun_passed", "rows": len(rows), "forwards": math.ceil(len(rows) / BATCH),
                          "evidence_status": "development_only"}, indent=2))
        return
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    correct, calls = [], 0
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            length = max(len(row["ids"]) for row in chunk)
            tokens = torch.full((len(chunk), length), 50256, dtype=torch.long, device="cuda")
            finals = []
            for index, row in enumerate(chunk):
                tokens[index, :len(row["ids"])] = torch.tensor(row["ids"], device="cuda")
                finals.append(len(row["ids"]) - 1)
            logits = native_logits(model, tokens).cpu()
            calls += 1
            for index, (row, final) in enumerate(zip(chunk, finals, strict=True)):
                pool = candidates(row["candidate_kind"])
                prediction = int(pool[logits[index, final, pool].argmax()])
                correct.append(prediction == row["answer_id"])
    cells = {}
    cursor = 0
    for format_name in FORMATS:
        cells[format_name] = {}
        for rule in RULES:
            values = correct[cursor:cursor + 24]
            cursor += 24
            cells[format_name][rule] = {"n": 24, "accuracy": sum(values) / len(values)}
    pred_a = cells["numbered_list_2_digits"]["plus_one"]["accuracy"] >= .75
    pred_b = max(cells[name]["plus_one"]["accuracy"] for name in FORMATS[1:4]) >= .75
    pred_c = max(cells[name]["plus_one"]["accuracy"] for name in FORMATS[4:]) >= .50
    result = {
        "rung": 566, "stage": "development_only_increment_format_exploration",
        "pred_a_legacy_anchor": pred_a, "pred_b_nonlegacy_digit_format": pred_b,
        "pred_c_number_word_format": pred_c, "cells": cells, "rows": rows,
        "evidence_status": "development_only_never_fit_select_final_or_ood",
        "model_forwards": calls, "model_backwards": 0,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "next_step": "freeze_unseen_rows_from_best_supported_formats",
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: result[key] for key in PRED_KEYS}, indent=2))


if __name__ == "__main__":
    main()
