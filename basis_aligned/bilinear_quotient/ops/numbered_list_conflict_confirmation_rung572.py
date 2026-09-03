#!/usr/bin/env python3
"""R572 raw-row confirmation of the numbered-list +2 conflict.

Pred A: exact inputs/checkpoint/budget and every aggregate matches R569 within 1e-6.
Pred B: all 64 FIT endpoint margins favor final-label+1 with positive bootstrap lower mean.
Pred C: all 32 SELECT endpoint margins do likewise. Null: any failure blocks localization.
Price: at most 96 unique sequences, 3 forwards, zero backwards; FINAL_TEST/OOD closed.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
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
REFERENCE = ROOT / "numeric_two_hypothesis_capability_rung569_570_results.json"
PREREG = POLY / "NUMBERED_LIST_CONFLICT_CONFIRMATION_RUNG572_PREREGISTRATION.md"
OUT = ROOT / "numbered_list_conflict_confirmation_rung572_results.json"
HASHES = {ROWS: "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
          REFERENCE: "7cc56f22def334673e0035fad7c6a7d1fc58ab8edd3a99744bebd9fb4e6af7e7",
          PREREG: "0cc937cbf532539ed3af8d902476674988a1b16f5e5b7332730bd090bf936b0c"}
ENC = tiktoken.get_encoding("gpt2")
BATCH = 32
BOOTSTRAPS = 2000
SEEDS = {"FIT": 579, "SELECT": 679}
PRED_KEYS = ("pred_a_exact_and_matches_reference pred_b_fit_raw_confirmation pred_c_select_raw_confirmation "
             "all_gates_pass model_forwards next_step").split()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower(values: list[float], seed: int) -> float:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(BOOTSTRAPS, len(data)))
    return float(np.quantile(data[indices].mean(1), .025))


def native_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
    x0, v1 = x, None
    for block in model.transformer.h:
        x, v1 = block(x, v1, x0)
    return (30 * torch.tanh(model.lm_head(F.rms_norm(x, (x.size(-1),))) / 30)).float()


def main() -> None:
    started = time.time()
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    document = json.loads(ROWS.read_text())
    reference = json.loads(REFERENCE.read_text())
    rows = [row for row in document["rows"] if row["family_id"] == "list_step_two_conflict"
            and row["split"] in {"FIT", "SELECT"}]
    sequences = sorted({tuple(row[key]) for row in rows for key in ("base_ids", "donor_ids")},
                       key=lambda ids: (len(ids), ids))
    assert len(rows) == 48 and len(sequences) <= 96
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({"status": "dryrun_passed", "rows": len(rows), "unique_sequences": len(sequences),
                          "maximum_forwards": 3, "FINAL_TEST_or_OOD_opened": False}, indent=2))
        return
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
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
    split_results, exact = {}, True
    for split in ("FIT", "SELECT"):
        raw = []
        for row in [item for item in rows if item["split"] == split]:
            structural = row["base_answer_id"]
            arithmetic = ENC.encode(str(row["semantic_details"]["arithmetic_step_two_answer"]))
            assert len(arithmetic) == 1
            for endpoint in ("base", "donor"):
                value = float(cache[tuple(row[f"{endpoint}_ids"])][structural] - cache[tuple(row[f"{endpoint}_ids"])][arithmetic[0]])
                raw.append({"group_id": row["group_id"], "row_id": row["row_id"], "endpoint": endpoint, "margin": value})
        values = [item["margin"] for item in raw]
        summary = {"n_groups": len(values), "correct_fraction": float(np.mean(np.asarray(values) > 0)),
                   "mean_margin": float(np.mean(values)), "bootstrap95_lower_mean_margin": lower(values, SEEDS[split])}
        summary["passed"] = bool(summary["correct_fraction"] >= .75 and summary["bootstrap95_lower_mean_margin"] > 0)
        old = reference["hypothesis_results"]["numbered_list_index_successor"][split]["step_two_conflict"]
        differences = {key: abs(float(summary[key]) - float(old[key])) for key in
                       ("correct_fraction", "mean_margin", "bootstrap95_lower_mean_margin")}
        match = all(value <= 1e-6 for value in differences.values()) and summary["passed"] == old["passed"]
        exact &= match
        split_results[split] = {"summary": summary, "reference_absolute_differences": differences,
                                "reference_match": match, "raw_rows": raw}
    pred_a = bool(exact and calls <= 3 and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
    pred_b = bool(split_results["FIT"]["summary"]["correct_fraction"] == 1.0
                  and split_results["FIT"]["summary"]["bootstrap95_lower_mean_margin"] > 0)
    pred_c = bool(split_results["SELECT"]["summary"]["correct_fraction"] == 1.0
                  and split_results["SELECT"]["summary"]["bootstrap95_lower_mean_margin"] > 0)
    result = {"rung": 572, "stage": "numbered_list_conflict_raw_confirmation",
              "pred_a_exact_and_matches_reference": pred_a, "pred_b_fit_raw_confirmation": pred_b,
              "pred_c_select_raw_confirmation": pred_c, "all_gates_pass": bool(pred_a and pred_b and pred_c),
              "split_results": split_results, "evaluated_splits": ["FIT", "SELECT"], "forbidden_splits_opened": [],
              "model_forwards": calls, "model_backwards": 0, "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "input_sha256": {str(path): sha256(path) for path in HASHES}, "elapsed_seconds": time.time() - started,
              "next_step": "preregister_exact_list_factor_localization" if pred_a and pred_b and pred_c else "block_list_localization"}
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: result[key] for key in PRED_KEYS}, indent=2))


if __name__ == "__main__":
    main()
