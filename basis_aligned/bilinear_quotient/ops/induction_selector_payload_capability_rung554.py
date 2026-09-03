#!/usr/bin/env python3
"""R554: native capability for the frozen induction selector-by-payload factorial.

Pred A (four-cell capability): in FIT and SELECT separately, every SxP cell has
accuracy >= .75 and a group-bootstrap 95% lower mean correct-vs-other-payload
margin > 0.

Pred B (relation-preserving controls): for irrelevant-source, filler-change,
and lag-extension rows, every split x variant x endpoint cell has accuracy >=
.75 and bootstrap lower mean margin > 0.

Pred C (selected-match necessity/selectivity): in both splits, breaking the
selected match lowers the correct-payload margin in >= .70 of groups with lower
mean > 0, and the paired reduction exceeds the absolute effect of editing the
unselected source with bootstrap lower mean > 0.

Null: any registered cell fails.  This stops a site search rather than licensing
a post-selected component.  Literal experiment price: 864 unique sequences, 27
native model forwards at batch 32, zero backwards, no fitted values, and no
component/rank selection.  Arm names are exact R552 condition/family IDs.  The
measured bars are the accuracies and group-bootstrap bounds above.
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


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
import bilin18_observed_model_facade as facade  # noqa: E402

ROWS = ROOT / "induction_selector_payload_factorial_rows_rung552.json"
RECEIPT = ROOT / "induction_selector_payload_factorial_rows_rung552_receipt.json"
AUDIT = ROOT / "induction_selector_payload_factorial_rows_rung553_audit.json"
PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_CAPABILITY_RUNG554_PREREGISTRATION.md"
OUT = ROOT / "induction_selector_payload_capability_rung554_results.json"
HASHES = {
    ROWS: "6a0a6d2c8a3891ae5d6f787527b35e71c17518548b3b1836042afe730b13c460",
    RECEIPT: "0d42bcaaf7f86390803033ce13bc22d7690700130cd80df74170d6b2d652081a",
    AUDIT: "9fc0376fade6fb204686e164f293f8991caf7bc45c67eedd064f330dffd5d1ea",
    PREREG: "9de3b16299043b6cf96e0cf2c75eb686f2063082e34a51e594fee1b0c0c4f777",
}
SPLITS = ("FIT", "SELECT")
CONDITIONS = ("s0p0", "s0p1", "s1p0", "s1p1")
INVARIANCE_FAMILIES = ("irrelevant_source_edit", "copy_relation_preserved_nuisance_change")
BATCH = 32
BOOTSTRAPS = 2000
SEED = 554
EXPECTED_GROUPS = 108
EXPECTED_SEQUENCES = 864
EXPECTED_FORWARDS = math.ceil(EXPECTED_SEQUENCES / BATCH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_lower(values: list[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[choices].mean(1), .025))


def native_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Exact frozen-model forward, returning logits instead of its training loss."""
    x = model.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    for block in model.transformer.h:
        x, v1 = block(x, v1, x0)
    logits = model.lm_head(F.rms_norm(x, (x.size(-1),)))
    return (30.0 * torch.tanh(logits / 30.0)).float()


def load_authority() -> tuple[dict, list[dict], dict[str, dict]]:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    document = json.loads(ROWS.read_text())
    receipt = json.loads(RECEIPT.read_text())
    audit = json.loads(AUDIT.read_text())
    assert receipt["rows_sha256"] == HASHES[ROWS]
    assert audit["rows_sha256"] == HASHES[ROWS]
    assert audit["all_token_level_factorial_checks_pass"] is True
    assert document["model_loaded"] is False and document["outcomes_opened"] == []
    groups = [group for group in document["groups"] if group["split"] in SPLITS]
    rows = [row for row in document["rows"] if row["split"] in SPLITS]
    by_group = {group["group_id"]: group for group in groups}
    assert len(groups) == EXPECTED_GROUPS and len(by_group) == EXPECTED_GROUPS
    assert {group["split"] for group in groups} == set(SPLITS)
    return document, rows, by_group


def payload_other(group: dict, answer_id: int) -> int:
    payloads = (group["variable_token_ids"]["B"], group["variable_token_ids"]["D"])
    assert answer_id in payloads and payloads[0] != payloads[1]
    return payloads[1] if answer_id == payloads[0] else payloads[0]


def collect_sequences(rows: list[dict], by_group: dict[str, dict]) -> list[tuple[int, ...]]:
    sequences = {
        tuple(condition["ids"])
        for group in by_group.values()
        for condition in group["factorial_conditions"].values()
    }
    for row in rows:
        sequences.add(tuple(row["base_ids"]))
        sequences.add(tuple(row["donor_ids"]))
        assert row["base_ids"] != row["donor_ids"]
    ordered = sorted(sequences, key=lambda ids: (len(ids), ids))
    assert len(ordered) == EXPECTED_SEQUENCES
    return ordered


def evaluate(model: torch.nn.Module, sequences: list[tuple[int, ...]]) -> tuple[dict[tuple[int, ...], torch.Tensor], int]:
    cache: dict[tuple[int, ...], torch.Tensor] = {}
    calls = 0
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
    assert len(cache) == EXPECTED_SEQUENCES
    return cache, calls


def margin(cache: dict[tuple[int, ...], torch.Tensor], ids: list[int], answer: int, other: int) -> float:
    logits = cache[tuple(ids)]
    return float(logits[answer] - logits[other])


def accuracy_report(values: list[float], seed: int) -> dict:
    report = {
        "n_groups": len(values),
        "correct_fraction": float(np.mean(np.asarray(values) > 0)),
        "mean_margin": float(np.mean(values)),
        "bootstrap95_lower_mean_margin": bootstrap_lower(values, seed),
    }
    report["passed"] = bool(
        report["correct_fraction"] >= .75 and report["bootstrap95_lower_mean_margin"] > 0
    )
    return report


def score(by_group: dict[str, dict], rows: list[dict], cache: dict[tuple[int, ...], torch.Tensor]) -> dict:
    seed = SEED
    factorial: dict[str, dict] = {}
    pred_a = True
    for split in SPLITS:
        factorial[split] = {}
        for condition_name in CONDITIONS:
            values = []
            for group in by_group.values():
                if group["split"] != split:
                    continue
                condition = group["factorial_conditions"][condition_name]
                values.append(margin(
                    cache, condition["ids"], condition["answer_id"],
                    payload_other(group, condition["answer_id"]),
                ))
            report = accuracy_report(values, seed)
            seed += 1
            factorial[split][condition_name] = report
            pred_a &= report["passed"]

    invariance: dict[str, dict] = {}
    pred_b = True
    for split in SPLITS:
        invariance[split] = {}
        for family in INVARIANCE_FAMILIES:
            invariance[split][family] = {}
            variants = sorted({
                row["family_variant"] for row in rows
                if row["split"] == split and row["family_id"] == family
            })
            for variant in variants:
                cell = [row for row in rows if row["split"] == split
                        and row["family_id"] == family and row["family_variant"] == variant]
                invariance[split][family][variant] = {}
                for endpoint in ("base", "donor"):
                    values = []
                    for row in cell:
                        group = by_group[row["group_id"]]
                        answer = row[f"{endpoint}_answer_id"]
                        values.append(margin(
                            cache, row[f"{endpoint}_ids"], answer, payload_other(group, answer),
                        ))
                    report = accuracy_report(values, seed)
                    seed += 1
                    invariance[split][family][variant][endpoint] = report
                    pred_b &= report["passed"]

    necessity: dict[str, dict] = {}
    pred_c = True
    for split in SPLITS:
        broken = {
            row["group_id"]: row for row in rows
            if row["split"] == split and row["family_id"] == "match_break_payload_preserved"
        }
        irrelevant = {
            row["group_id"]: row for row in rows
            if row["split"] == split and row["family_id"] == "irrelevant_source_edit"
        }
        assert broken.keys() == irrelevant.keys()
        selected_drops, selective_gaps = [], []
        for group_id in sorted(broken):
            selected, control = broken[group_id], irrelevant[group_id]
            assert selected["base_condition_id"] == control["base_condition_id"]
            assert selected["base_ids"] == control["base_ids"]
            group = by_group[group_id]
            answer = selected["base_answer_id"]
            other = payload_other(group, answer)
            base_margin = margin(cache, selected["base_ids"], answer, other)
            broken_margin = margin(cache, selected["donor_ids"], answer, other)
            control_margin = margin(cache, control["donor_ids"], answer, other)
            selected_drop = base_margin - broken_margin
            selected_drops.append(selected_drop)
            selective_gaps.append(selected_drop - abs(base_margin - control_margin))
        report = {
            "n_groups": len(selected_drops),
            "selected_match_break_positive_fraction": float(np.mean(np.asarray(selected_drops) > 0)),
            "mean_selected_match_margin_drop": float(np.mean(selected_drops)),
            "bootstrap95_lower_mean_selected_match_margin_drop": bootstrap_lower(selected_drops, seed),
            "mean_selective_gap_over_absolute_irrelevant_edit": float(np.mean(selective_gaps)),
            "bootstrap95_lower_mean_selective_gap": bootstrap_lower(selective_gaps, seed + 1),
        }
        seed += 2
        report["passed"] = bool(
            report["selected_match_break_positive_fraction"] >= .70
            and report["bootstrap95_lower_mean_selected_match_margin_drop"] > 0
            and report["bootstrap95_lower_mean_selective_gap"] > 0
        )
        necessity[split] = report
        pred_c &= report["passed"]
    return {
        "pred_a_four_cell_capability": bool(pred_a),
        "pred_b_relation_preserving_controls": bool(pred_b),
        "pred_c_selected_match_necessity_and_selectivity": bool(pred_c),
        "factorial_cells": factorial,
        "relation_preserving_controls": invariance,
        "selected_match_necessity": necessity,
    }


def main() -> None:
    started = time.time()
    _, rows, by_group = load_authority()
    sequences = collect_sequences(rows, by_group)
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dryrun_passed",
            "groups": len(by_group),
            "unique_sequences": len(sequences),
            "expected_forwards": EXPECTED_FORWARDS,
            "evaluated_splits": list(SPLITS),
            "final_or_ood_opened": False,
        }, indent=2))
        return
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    cache, calls = evaluate(model, sequences)
    scores = score(by_group, rows, cache)
    instrument = bool(
        calls == EXPECTED_FORWARDS
        and len(cache) == EXPECTED_SEQUENCES
        and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
    )
    all_gates = bool(
        instrument
        and scores["pred_a_four_cell_capability"]
        and scores["pred_b_relation_preserving_controls"]
        and scores["pred_c_selected_match_necessity_and_selectivity"]
    )
    result = {
        "rung": 554,
        "stage": "induction_selector_payload_native_capability",
        "pred_0_exact_instrument": instrument,
        **scores,
        "all_gates_pass": all_gates,
        "model_forwards": calls,
        "model_backwards": 0,
        "model_weights_updated": False,
        "unique_sequences": len(cache),
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_sha256": {str(path): sha256(path) for path in HASHES},
        "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
        "elapsed_seconds": time.time() - started,
        "next_step": (
            "audit_then_preregister_separate_selector_and_payload_complete_state_ceilings"
            if all_gates else "record_native_capability_null_and_do_not_search_sites"
        ),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({
        key: value for key, value in result.items()
        if key.startswith("pred_") or key in {"all_gates_pass", "model_forwards", "unique_sequences", "next_step"}
    }, indent=2))


if __name__ == "__main__":
    main()
