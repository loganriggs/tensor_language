#!/usr/bin/env python3
"""R564: native increment capability and separation from copy/+2 numeric rules.

Pred A: every digit, number-word, and cross-format +1 endpoint cell has >=.75
numeric-candidate accuracy and a positive bootstrap lower mean margin.
Pred B: both endpoints of +1 surface, copy/repeat, and +2 controls meet the same bar.
Pred C: coherent broken-middle bases meet that bar, while the middle edit reduces
the answer margin in >=.65 of groups with positive bootstrap lower mean reduction.
Null: any opened cell fails. FIT opens first; SELECT only after a complete FIT pass.
Price: at most 960 unique sequences, 30 forwards at batch 32, zero backwards.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
import bilin18_observed_model_facade as facade  # noqa: E402

ROWS = ROOT / "increment_counterfactual_authority_rung563.json"
RECEIPT = ROOT / "increment_counterfactual_authority_rung563_receipt.json"
PREREG = POLY / "INCREMENT_NATIVE_CAPABILITY_RUNG564_PREREGISTRATION.md"
CORRECTION = POLY / "INCREMENT_COUNTERFACTUAL_AUTHORITY_RUNG563_CORRECTION.md"
OUT = ROOT / "increment_native_capability_rung564_results.json"
HASHES = {
    ROWS: "3886eb039a34a67ee0548b3f62c21e35560aa93dba33fe2ebfdf485788e37298",
    RECEIPT: "a692e3b38adea081f63c608e9072be79975623bbf48ad86ef67763b8334af3e4",
    PREREG: "b469f722b63f9f1ca54cb2e7b28a736d62677d058f8e6f350c9ba54bd9c522b9",
    CORRECTION: "79c065271b9bb2e7b2690ad9220edebc8fc8e32868e6406133600436df40953f",
}
TARGET_FAMILIES = ("digit_coherent_shift", "word_coherent_shift", "cross_format_coherent_shift")
CONTROL_FAMILIES = ("operation_preserved_surface_edit", "repeated_number_numeric_control", "step_two_numeric_control")
NECESSITY_FAMILY = "incoherent_middle_number_edit"
BATCH = 32
BOOTSTRAPS = 2000
SEED = 564
EXPECTED_SEQUENCES = {"FIT": 640, "SELECT": 320}
PRED_KEYS = ("pred_0_exact_instrument pred_a_target_capability pred_b_nonincrement_rule_controls "
             "pred_c_middle_number_necessity all_gates_pass evaluated_splits model_forwards next_step").split()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_lower(values: list[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[choices].mean(1), .025))


def answer_candidates() -> torch.Tensor:
    enc = tiktoken.get_encoding("gpt2")
    ids = set()
    for value in range(121):
        encoded = enc.encode(" " + str(value))
        if len(encoded) == 1:
            ids.add(encoded[0])
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
             "nineteen", "twenty")
    for word in words:
        encoded = enc.encode(" " + word)
        assert len(encoded) == 1
        ids.add(encoded[0])
    return torch.tensor(sorted(ids), dtype=torch.long)


def native_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    x = model.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    for block in model.transformer.h:
        x, v1 = block(x, v1, x0)
    logits = model.lm_head(F.rms_norm(x, (x.size(-1),)))
    return (30.0 * torch.tanh(logits / 30.0)).float()


def load_rows(split: str) -> list[dict]:
    for path, expected in HASHES.items():
        if expected == "TO_FILL" or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path} {sha256(path) if path.is_file() else 'missing'}")
    document = json.loads(ROWS.read_text())
    receipt = json.loads(RECEIPT.read_text())
    assert document["model_loaded"] is False and document["outcomes_opened"] == []
    assert receipt["family_revealing_prompt_labels"] is False
    rows = [row for row in document["rows"] if row["split"] == split]
    expected_rows = 448 if split == "FIT" else 224
    assert len(rows) == expected_rows
    return rows


def collect_sequences(rows: list[dict]) -> list[tuple[int, ...]]:
    sequences = {tuple(row[key]) for row in rows for key in ("base_ids", "donor_ids")}
    return sorted(sequences, key=lambda ids: (len(ids), ids))


def evaluate(model: torch.nn.Module, sequences: list[tuple[int, ...]]) -> tuple[dict, int]:
    cache, calls = {}, 0
    with torch.inference_mode():
        for start in range(0, len(sequences), BATCH):
            chunk = sequences[start:start + BATCH]
            length = max(map(len, chunk))
            tokens = torch.full((len(chunk), length), 50256, dtype=torch.long, device="cuda")
            finals = []
            for index, ids in enumerate(chunk):
                tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device="cuda")
                finals.append(len(ids) - 1)
            logits = native_logits(model, tokens)
            calls += 1
            for index, (ids, final) in enumerate(zip(chunk, finals, strict=True)):
                cache[ids] = logits[index, final].detach().cpu()
    return cache, calls


def numeric_margin(logits: torch.Tensor, answer: int, candidates: torch.Tensor) -> float:
    assert int(answer) in set(candidates.tolist())
    alternatives = candidates[candidates != answer]
    return float(logits[answer] - logits[alternatives].max())


def cell_report(values: list[float], seed: int) -> dict:
    report = {
        "n_groups": len(values),
        "correct_fraction": float(np.mean(np.asarray(values) > 0)),
        "mean_numeric_candidate_margin": float(np.mean(values)),
        "bootstrap95_lower_mean_margin": bootstrap_lower(values, seed),
    }
    report["passed"] = bool(report["correct_fraction"] >= .75 and report["bootstrap95_lower_mean_margin"] > 0)
    return report


def score_split(split: str, rows: list[dict], cache: dict, candidates: torch.Tensor) -> dict:
    seed = SEED + (0 if split == "FIT" else 100)
    row_statistics = []
    margins = {}
    for row in rows:
        margins[row["row_id"]] = {}
        for endpoint in ("base", "donor"):
            value = numeric_margin(cache[tuple(row[f"{endpoint}_ids"])], row[f"{endpoint}_answer_id"], candidates)
            margins[row["row_id"]][endpoint] = value
            row_statistics.append({
                "group_id": row["group_id"], "row_id": row["row_id"], "family_id": row["family_id"],
                "endpoint": endpoint, "numeric_candidate_margin": value, "correct": value > 0,
            })

    target_cells, pred_a = {}, True
    for family in TARGET_FAMILIES:
        target_cells[family] = {}
        family_rows = [row for row in rows if row["family_id"] == family]
        for endpoint in ("base", "donor"):
            report = cell_report([margins[row["row_id"]][endpoint] for row in family_rows], seed)
            seed += 1
            target_cells[family][endpoint] = report
            pred_a &= report["passed"]

    control_cells, pred_b = {}, True
    for family in CONTROL_FAMILIES:
        control_cells[family] = {}
        family_rows = [row for row in rows if row["family_id"] == family]
        for endpoint in ("base", "donor"):
            report = cell_report([margins[row["row_id"]][endpoint] for row in family_rows], seed)
            seed += 1
            control_cells[family][endpoint] = report
            pred_b &= report["passed"]

    necessity_rows = [row for row in rows if row["family_id"] == NECESSITY_FAMILY]
    base_values = [margins[row["row_id"]]["base"] for row in necessity_rows]
    drops = [margins[row["row_id"]]["base"] - margins[row["row_id"]]["donor"] for row in necessity_rows]
    base_report = cell_report(base_values, seed)
    necessity = {
        "coherent_base": base_report,
        "n_groups": len(drops),
        "positive_drop_fraction": float(np.mean(np.asarray(drops) > 0)),
        "mean_margin_drop_after_middle_edit": float(np.mean(drops)),
        "bootstrap95_lower_mean_drop": bootstrap_lower(drops, seed + 1),
    }
    necessity["passed"] = bool(
        base_report["passed"] and necessity["positive_drop_fraction"] >= .65
        and necessity["bootstrap95_lower_mean_drop"] > 0
    )
    return {
        "target_capability": bool(pred_a),
        "nonincrement_rule_controls": bool(pred_b),
        "middle_number_necessity": bool(necessity["passed"]),
        "all_pass": bool(pred_a and pred_b and necessity["passed"]),
        "target_cells": target_cells,
        "control_cells": control_cells,
        "middle_number_necessity_details": necessity,
        "row_statistics": row_statistics,
    }


def main() -> None:
    started = time.time()
    fit_rows = load_rows("FIT")
    select_rows = load_rows("SELECT")
    fit_sequences, select_sequences = collect_sequences(fit_rows), collect_sequences(select_rows)
    assert len(fit_sequences) == EXPECTED_SEQUENCES["FIT"]
    assert len(select_sequences) == EXPECTED_SEQUENCES["SELECT"]
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dryrun_passed", "FIT_sequences": len(fit_sequences),
            "SELECT_sequences_conditional": len(select_sequences), "maximum_forwards": 30,
            "FINAL_TEST_or_OOD_opened": False,
        }, indent=2))
        return
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    candidates = answer_candidates()
    fit_cache, fit_calls = evaluate(model, fit_sequences)
    scores = {"FIT": score_split("FIT", fit_rows, fit_cache, candidates)}
    calls = fit_calls
    evaluated_splits = ["FIT"]
    if scores["FIT"]["all_pass"]:
        select_cache, select_calls = evaluate(model, select_sequences)
        scores["SELECT"] = score_split("SELECT", select_rows, select_cache, candidates)
        calls += select_calls
        evaluated_splits.append("SELECT")
    instrument = bool(
        fit_calls == math.ceil(EXPECTED_SEQUENCES["FIT"] / BATCH)
        and calls <= 30 and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
    )
    all_gates = bool(instrument and evaluated_splits == ["FIT", "SELECT"] and scores["SELECT"]["all_pass"])
    result = {
        "rung": 564,
        "stage": "increment_native_capability_and_rule_separation",
        "pred_0_exact_instrument": instrument,
        "pred_a_target_capability": bool(all(value["target_capability"] for value in scores.values())),
        "pred_b_nonincrement_rule_controls": bool(all(value["nonincrement_rule_controls"] for value in scores.values())),
        "pred_c_middle_number_necessity": bool(all(value["middle_number_necessity"] for value in scores.values())),
        "all_gates_pass": all_gates,
        "split_results": scores,
        "evaluated_splits": evaluated_splits,
        "forbidden_splits_opened": [],
        "model_forwards": calls,
        "model_backwards": 0,
        "model_weights_updated": False,
        "unique_sequences": sum(EXPECTED_SEQUENCES[split] for split in evaluated_splits),
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_sha256": {str(path): sha256(path) for path in HASHES},
        "elapsed_seconds": time.time() - started,
        "next_step": "independent_audit_then_factor_localization" if all_gates else "record_capability_null_and_do_not_localize",
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: result[key] for key in PRED_KEYS}, indent=2))


if __name__ == "__main__":
    main()
