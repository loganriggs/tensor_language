#!/usr/bin/env python3
"""R569/R570: independent native gates for list successor and numeric sequence.

Pred A: list two/three-line state-shift endpoint cells pass 75% + bootstrap bars.
Pred B: list surface, middle-label, and repeat endpoint cells pass those bars.
Pred C: list step-two conflicts favor final-label+1 over arithmetic+2 at those bars.
Pred D: sequence digit, word, and cross-format +1 endpoint cells pass those bars.
Pred E: sequence digit/word surface and copy endpoint cells pass those bars.
Pred F: sequence coherent bases pass, and middle edits lower margin in >=65% with
positive bootstrap lower mean. Each hypothesis opens SELECT independently after FIT.
Null: any required cell fails for that hypothesis. FINAL_TEST/OOD remain closed.
Price: at most 935 unique sequence evaluations, 30 forwards, zero backwards.
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

ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
RECEIPT = ROOT / "increment_two_hypothesis_rows_rung567_receipt.json"
OVERLAY = ROOT / "increment_rung568_semantic_role_overlay.json"
LIST_PREREG = POLY / "NUMBERED_LIST_NATIVE_CAPABILITY_RUNG569_PREREGISTRATION.md"
SEQUENCE_PREREG = POLY / "NUMERIC_SEQUENCE_NATIVE_CAPABILITY_RUNG570_PREREGISTRATION.md"
OUT = ROOT / "numeric_two_hypothesis_capability_rung569_570_results.json"
HASHES = {
    ROWS: "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
    RECEIPT: "02b2c37cc23434138accd63e920f417cda10f1c86a4c08c174537149ec2b1072",
    OVERLAY: "90c03d026b4daaae4794b02399967cbd3f9daf8b5412a24e13e594b4ba659765",
    LIST_PREREG: "8b93c3b3ddcf83e587907de7dcdf4160a1cbfe1e40e4e08663650659f441a892",
    SEQUENCE_PREREG: "272689a72c8d89dcc4a7e2e59aa8e65384f7e1764d01a33b891454c0145bbc0a",
}
LIST_ID = "numbered_list_index_successor"
SEQUENCE_ID = "numeric_sequence_continuation"
LIST_TARGETS = ("list_two_line_state_shift", "list_three_line_state_shift")
LIST_INVARIANCES = ("list_surface_preserved", "list_middle_index_break", "list_repeated_index_control")
SEQUENCE_TARGETS = ("sequence_digit_state_shift", "sequence_word_state_shift", "sequence_cross_format_shift")
SEQUENCE_INVARIANCES = ("sequence_digit_surface_preserved", "sequence_word_surface_preserved",
                        "sequence_digit_copy_control", "sequence_word_copy_control")
BATCH = 32
BOOTSTRAPS = 2000
SEED = 569
EXPECTED = {
    LIST_ID: {"FIT": 318, "SELECT": 160},
    SEQUENCE_ID: {"FIT": 280, "SELECT": 177},
}
PRED_KEYS = ("instrument_exact pred_a_list_state_shifts pred_b_list_invariances pred_c_list_step_two_conflict "
             "pred_d_sequence_state_shifts pred_e_sequence_invariances pred_f_sequence_middle_necessity "
             "list_all_gates_pass sequence_all_gates_pass evaluated_splits model_forwards next_step").split()
ENC = tiktoken.get_encoding("gpt2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower(values: list[float], seed: int) -> float:
    data = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(data), size=(BOOTSTRAPS, len(data)))
    return float(np.quantile(data[indices].mean(1), .025))


def standard_report(values: list[float], seed: int) -> dict:
    report = {"n_groups": len(values), "correct_fraction": float(np.mean(np.asarray(values) > 0)),
              "mean_margin": float(np.mean(values)), "bootstrap95_lower_mean_margin": lower(values, seed)}
    report["passed"] = bool(report["correct_fraction"] >= .75 and report["bootstrap95_lower_mean_margin"] > 0)
    return report


def candidates(kind: str) -> torch.Tensor:
    if kind == "plain_digit":
        strings = [str(value) for value in range(101)]
    elif kind == "space_digit":
        strings = [" " + str(value) for value in range(101)]
    else:
        strings = [" " + word for word in ("zero", "one", "two", "three", "four", "five", "six", "seven",
                   "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                   "seventeen", "eighteen", "nineteen", "twenty")]
    ids = [encoded[0] for text in strings if len(encoded := ENC.encode(text)) == 1]
    return torch.tensor(sorted(set(ids)), dtype=torch.long)


CANDIDATES = {kind: candidates(kind) for kind in ("plain_digit", "space_digit", "space_word")}


def answer_kind(answer: str) -> str:
    if not answer.startswith(" "):
        return "plain_digit"
    return "space_digit" if answer.strip().isdigit() else "space_word"


def native_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
    x0, v1 = x, None
    for block in model.transformer.h:
        x, v1 = block(x, v1, x0)
    return (30 * torch.tanh(model.lm_head(F.rms_norm(x, (x.size(-1),))) / 30)).float()


def collect(rows: list[dict], hypothesis: str, split: str) -> tuple[list[dict], list[tuple[int, ...]]]:
    selected = [row for row in rows if row["hypothesis_id"] == hypothesis and row["split"] == split]
    sequences = sorted({tuple(row[key]) for row in selected for key in ("base_ids", "donor_ids")},
                       key=lambda ids: (len(ids), ids))
    assert len(sequences) == EXPECTED[hypothesis][split]
    return selected, sequences


def evaluate(model: torch.nn.Module, sequences: list[tuple[int, ...]]) -> tuple[dict, int]:
    cache, calls = {}, 0
    with torch.inference_mode():
        for start in range(0, len(sequences), BATCH):
            chunk = sequences[start:start + BATCH]
            length = max(map(len, chunk))
            tokens = torch.full((len(chunk), length), 50256, dtype=torch.long, device="cuda")
            finals = []
            for index, ids in enumerate(chunk):
                tokens[index, :len(ids)] = torch.tensor(ids, device="cuda")
                finals.append(len(ids) - 1)
            logits = native_logits(model, tokens)
            calls += 1
            for index, (ids, final) in enumerate(zip(chunk, finals, strict=True)):
                cache[ids] = logits[index, final].detach().cpu()
    return cache, calls


def margin(logits: torch.Tensor, answer_id: int, kind: str) -> float:
    pool = CANDIDATES[kind]
    assert int(answer_id) in set(pool.tolist())
    return float(logits[answer_id] - logits[pool[pool != answer_id]].max())


def endpoint_margins(rows: list[dict], cache: dict) -> tuple[dict, list[dict]]:
    values, statistics = {}, []
    for row in rows:
        values[row["row_id"]] = {}
        for endpoint in ("base", "donor"):
            answer = row[f"{endpoint}_answer"]
            if answer is None:
                continue
            value = margin(cache[tuple(row[f"{endpoint}_ids"])], row[f"{endpoint}_answer_id"], answer_kind(answer))
            values[row["row_id"]][endpoint] = value
            statistics.append({"row_id": row["row_id"], "group_id": row["group_id"],
                               "family_id": row["family_id"], "endpoint": endpoint, "margin": value})
    return values, statistics


def score_list(split: str, rows: list[dict], cache: dict) -> dict:
    values, statistics = endpoint_margins(rows, cache)
    seed = SEED + (0 if split == "FIT" else 100)
    targets, pred_a = {}, True
    for family in LIST_TARGETS:
        targets[family] = {}
        family_rows = [row for row in rows if row["family_id"] == family]
        for endpoint in ("base", "donor"):
            targets[family][endpoint] = standard_report([values[row["row_id"]][endpoint] for row in family_rows], seed)
            pred_a &= targets[family][endpoint]["passed"]
            seed += 1
    invariances, pred_b = {}, True
    for family in LIST_INVARIANCES:
        invariances[family] = {}
        family_rows = [row for row in rows if row["family_id"] == family]
        for endpoint in ("base", "donor"):
            invariances[family][endpoint] = standard_report([values[row["row_id"]][endpoint] for row in family_rows], seed)
            pred_b &= invariances[family][endpoint]["passed"]
            seed += 1
    conflict_rows = [row for row in rows if row["family_id"] == "list_step_two_conflict"]
    conflicts = []
    for item in conflict_rows:
        structural = token_id = item["base_answer_id"]
        arithmetic = ENC.encode(str(item["semantic_details"]["arithmetic_step_two_answer"]))
        assert len(arithmetic) == 1
        for endpoint in ("base", "donor"):
            logits = cache[tuple(item[f"{endpoint}_ids"])]
            conflicts.append(float(logits[structural] - logits[arithmetic[0]]))
    conflict = standard_report(conflicts, seed)
    return {"state_shifts": targets, "invariances": invariances, "step_two_conflict": conflict,
            "state_shifts_pass": bool(pred_a), "invariances_pass": bool(pred_b),
            "step_two_conflict_pass": bool(conflict["passed"]),
            "all_pass": bool(pred_a and pred_b and conflict["passed"]), "row_statistics": statistics}


def score_sequence(split: str, rows: list[dict], cache: dict) -> dict:
    values, statistics = endpoint_margins(rows, cache)
    seed = SEED + 200 + (0 if split == "FIT" else 100)
    targets, pred_d = {}, True
    for family in SEQUENCE_TARGETS:
        targets[family] = {}
        family_rows = [row for row in rows if row["family_id"] == family]
        for endpoint in ("base", "donor"):
            targets[family][endpoint] = standard_report([values[row["row_id"]][endpoint] for row in family_rows], seed)
            pred_d &= targets[family][endpoint]["passed"]
            seed += 1
    invariances, pred_e = {}, True
    for family in SEQUENCE_INVARIANCES:
        invariances[family] = {}
        family_rows = [row for row in rows if row["family_id"] == family]
        for endpoint in ("base", "donor"):
            invariances[family][endpoint] = standard_report([values[row["row_id"]][endpoint] for row in family_rows], seed)
            pred_e &= invariances[family][endpoint]["passed"]
            seed += 1
    middle_rows = [row for row in rows if row["family_id"] == "sequence_middle_value_break"]
    base_values = [values[row["row_id"]]["base"] for row in middle_rows]
    drops = [values[row["row_id"]]["base"] - values[row["row_id"]]["donor"] for row in middle_rows]
    base_report = standard_report(base_values, seed)
    necessity = {"coherent_base": base_report, "n_groups": len(drops),
                 "positive_drop_fraction": float(np.mean(np.asarray(drops) > 0)),
                 "mean_margin_drop": float(np.mean(drops)), "bootstrap95_lower_mean_drop": lower(drops, seed + 1)}
    necessity["passed"] = bool(base_report["passed"] and necessity["positive_drop_fraction"] >= .65
                               and necessity["bootstrap95_lower_mean_drop"] > 0)
    conflict_rows = [row for row in rows if row["family_id"] == "sequence_step_two_conflict"]
    conflict_values = []
    for item in conflict_rows:
        arithmetic = ENC.encode(" " + str(item["semantic_details"]["arithmetic_step_two_answer"]))
        successor = ENC.encode(" " + str(item["semantic_details"]["last_value_successor_answer"]))
        assert len(arithmetic) == len(successor) == 1
        for endpoint in ("base", "donor"):
            logits = cache[tuple(item[f"{endpoint}_ids"])]
            conflict_values.append(float(logits[arithmetic[0]] - logits[successor[0]]))
    conflict = {"n_endpoints": len(conflict_values), "arithmetic_over_successor_fraction": float(np.mean(np.asarray(conflict_values) > 0)),
                "mean_arithmetic_minus_successor_margin": float(np.mean(conflict_values))}
    return {"state_shifts": targets, "invariances": invariances, "middle_necessity": necessity,
            "step_two_characterization": conflict, "state_shifts_pass": bool(pred_d),
            "invariances_pass": bool(pred_e), "middle_necessity_pass": bool(necessity["passed"]),
            "all_pass": bool(pred_d and pred_e and necessity["passed"]), "row_statistics": statistics}


def main() -> None:
    started = time.time()
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    document = json.loads(ROWS.read_text())
    assert document["model_loaded"] is False and document["outcomes_opened"] == []
    bundles = {}
    for hypothesis in (LIST_ID, SEQUENCE_ID):
        bundles[hypothesis] = {split: collect(document["rows"], hypothesis, split) for split in ("FIT", "SELECT")}
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({"status": "dryrun_passed", "expected_sequences": EXPECTED,
                          "maximum_forwards": 30, "FINAL_TEST_or_OOD_opened": False}, indent=2))
        return
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    results, evaluated, calls = {}, {}, 0
    for hypothesis, scorer in ((LIST_ID, score_list), (SEQUENCE_ID, score_sequence)):
        fit_rows, fit_sequences = bundles[hypothesis]["FIT"]
        fit_cache, fit_calls = evaluate(model, fit_sequences)
        calls += fit_calls
        results[hypothesis] = {"FIT": scorer("FIT", fit_rows, fit_cache)}
        evaluated[hypothesis] = ["FIT"]
        if results[hypothesis]["FIT"]["all_pass"]:
            select_rows, select_sequences = bundles[hypothesis]["SELECT"]
            select_cache, select_calls = evaluate(model, select_sequences)
            calls += select_calls
            results[hypothesis]["SELECT"] = scorer("SELECT", select_rows, select_cache)
            evaluated[hypothesis].append("SELECT")
    instrument = bool(calls <= 30 and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
    list_pass = bool(instrument and evaluated[LIST_ID] == ["FIT", "SELECT"] and results[LIST_ID]["SELECT"]["all_pass"])
    sequence_pass = bool(instrument and evaluated[SEQUENCE_ID] == ["FIT", "SELECT"] and results[SEQUENCE_ID]["SELECT"]["all_pass"])
    result = {
        "rungs": [569, 570], "stage": "independent_two_hypothesis_native_capability", "instrument_exact": instrument,
        "pred_a_list_state_shifts": all(value["state_shifts_pass"] for value in results[LIST_ID].values()),
        "pred_b_list_invariances": all(value["invariances_pass"] for value in results[LIST_ID].values()),
        "pred_c_list_step_two_conflict": all(value["step_two_conflict_pass"] for value in results[LIST_ID].values()),
        "pred_d_sequence_state_shifts": all(value["state_shifts_pass"] for value in results[SEQUENCE_ID].values()),
        "pred_e_sequence_invariances": all(value["invariances_pass"] for value in results[SEQUENCE_ID].values()),
        "pred_f_sequence_middle_necessity": all(value["middle_necessity_pass"] for value in results[SEQUENCE_ID].values()),
        "list_all_gates_pass": list_pass, "sequence_all_gates_pass": sequence_pass,
        "hypothesis_results": results, "evaluated_splits": evaluated, "forbidden_splits_opened": [],
        "model_forwards": calls, "model_backwards": 0, "model_weights_updated": False,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_sha256": {str(path): sha256(path) for path in HASHES}, "elapsed_seconds": time.time() - started,
        "next_step": {LIST_ID: "audit_then_legacy_factor_localization" if list_pass else "record_list_capability_null",
                      SEQUENCE_ID: "audit_then_cross_format_site_search" if sequence_pass else "record_sequence_capability_null_or_split"},
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: result[key] for key in PRED_KEYS}, indent=2))


if __name__ == "__main__":
    main()
