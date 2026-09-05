#!/usr/bin/env python3
"""TEST reuse screen for the Task14 L11H3 subject-source payload."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_subject_payload_transfers_each_cell pred_c_subject_payload_fails_some_cell

from __future__ import annotations

from collections import defaultdict
import json
import os
import statistics
import sys
from pathlib import Path

import circuit_fast_screen_candidate_task14_test_cross_syntax as authority
import run_task14_head11_3_subject_attractor_score_payload_factorial as parent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/fast_screens/task14_head11_3_subject_payload_test_transfer_v1_result.json"
PRIOR_ART_SHA256 = "0960cb4b6a5893fcd0f0ec10b6b25937cd2e6d3ba0e7cc783c49401abcc1734a"
CONDITIONS = ("subject_payload", "complete_head")


def compile_plan():
    rows = authority.build_rows()
    authority_sha = authority.validate_rows(rows)
    cells = defaultdict(int)
    for row in rows:
        cells[row["cell_id"]] += 1
    if len(rows) != 64 or set(cells.values()) != {16} or len(cells) != 4:
        raise ValueError(f"TEST authority lost four balanced cells: {dict(cells)}")
    if any(row["base_subject_number"] == row["donor_subject_number"] for row in rows):
        raise ValueError("TEST donors no longer reverse subject number")
    if any(row["base_ids"][1] == row["donor_ids"][1] for row in rows):
        raise ValueError("TEST subject tokens are not cross-noun")
    return {
        "schema": "task14_head11_3_subject_payload_test_transfer_plan_v1",
        "candidate_id": "subject_verb.number_agreement.head11_3_subject_payload_test_transfer",
        "split": "TEST_REUSE_NEW_INTERVENTION",
        "screen_tier": "BASIC",
        "row_count": len(rows),
        "authority_sha256": authority_sha,
        "site": {"layer": parent.LAYER, "head": parent.HEAD,
                 "query": "final_prediction_position", "source": "subject_token_index_1"},
        "interventions": {
            "subject_payload": "recipient subject score times donor subject projected value",
            "complete_head": "complete donor head write at the recipient final position",
        },
        "price": {"model_forwards": 3, "example_evaluations": 384,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_directed_is_are_margin", "donor_answer_ce"],
        "bars": {
            "minimum_native_accuracy_each_side_each_cell": .85,
            "maximum_native_replay_absolute_logit_error": 5e-5,
            "maximum_source_term_identity_absolute_error": 5e-5,
            "minimum_complete_head_direction_fraction_each_cell": .75,
            "minimum_subject_payload_margin_recovery_each_cell": .25,
            "minimum_subject_payload_direction_fraction_each_cell": .75,
            "minimum_subject_payload_ce_recovery_each_cell": 0.0,
        },
        "scope": "TEST_REUSE_NEW_INTERVENTION",
        "closed_splits": ["OOD"],
        "limits": (
            "The TEST texts and whole-head result were already opened. This evaluates a new "
            "below-head intervention on those rows; it is not a pristine globally unopened TEST. "
            "OOD remains closed."
        ),
    }


def _pad(rows, side, length, torch, device):
    tokens = torch.full((len(rows), length), 50256, dtype=torch.long, device=device)
    finals = []
    for index, row in enumerate(rows):
        ids = row[f"{side}_ids"]
        tokens[index, :len(ids)] = torch.tensor(ids, device=device)
        finals.append(len(ids) - 1)
    return tokens, torch.tensor(finals, device=device)


def _task_margin(logits, q, answer, foil):
    return float(logits[q, int(answer)] - logits[q, int(foil)])


def _donor_metrics(logits, row, q, torch):
    donor = int(row["donor_answer_id"])
    recipient = int(row["base_answer_id"])
    margin = float(logits[q, donor] - logits[q, recipient])
    ce = float(-torch.log_softmax(logits[q], dim=-1)[donor])
    return margin, ce


def _split_factors(factors, count):
    return ({key: value[:count] for key, value in factors.items()},
            {key: value[count:] for key, value in factors.items()})


def _patch_batch(base_tokens, base_finals, base, donor, torch):
    count = len(base_tokens)
    subject = torch.ones(count, dtype=torch.long, device=base_tokens.device)
    recipient_p, recipient_u = parent._selected(base, subject, torch)
    _donor_p, donor_u = parent._selected(donor, subject, torch)
    payload_terms = recipient_p.unsqueeze(-1) * donor_u
    native_terms = recipient_p.unsqueeze(-1) * recipient_u
    return {
        "tokens": base_tokens.repeat(2, 1),
        "finals": base_finals.repeat(2),
        "source_positions": subject.repeat(2),
        "replacement_terms": torch.cat((payload_terms, native_terms)),
        "replacement_heads": torch.cat((base["head"], donor["head"])),
    }


def _accuracy_cells(rows, native_base, native_donor, base_finals, donor_finals):
    cells = defaultdict(lambda: {"base": [], "donor": []})
    for index, row in enumerate(rows):
        cell = cells[row["cell_id"]]
        cell["base"].append(_task_margin(native_base[index], int(base_finals[index]),
                                                row["base_answer_id"], row["base_foil_id"]) > 0)
        cell["donor"].append(_task_margin(native_donor[index], int(donor_finals[index]),
                                                 row["donor_answer_id"], row["donor_foil_id"]) > 0)
    return {name: {side: sum(values) / len(values) for side, values in sides.items()}
            for name, sides in sorted(cells.items())}


def score(evidence, capability_cells, replay_error, identity_error, bars):
    grouped = defaultdict(list)
    for row in evidence:
        grouped[(row["cell_id"], row["condition"])].append(row)
    cells = {}
    for cell_id in sorted(capability_cells):
        complete = grouped[(cell_id, "complete_head")]
        payload = grouped[(cell_id, "subject_payload")]
        if len(complete) != 16 or len(payload) != 16:
            raise ValueError(f"cell {cell_id} lost its 16 rows per condition")
        complete_margin = statistics.fmean(row["margin_delta"] for row in complete)
        complete_ce = statistics.fmean(row["donor_ce_gain"] for row in complete)
        payload_margin = statistics.fmean(row["margin_delta"] for row in payload)
        payload_ce = statistics.fmean(row["donor_ce_gain"] for row in payload)
        margin_recovery = payload_margin / complete_margin if complete_margin > 0 else None
        ce_recovery = payload_ce / complete_ce if complete_ce > 0 else None
        complete_direction = sum(row["margin_delta"] > 0 for row in complete) / len(complete)
        payload_direction = sum(row["margin_delta"] > 0 for row in payload) / len(payload)
        capability = capability_cells[cell_id]
        capability_passed = min(capability.values()) >= bars["minimum_native_accuracy_each_side_each_cell"]
        ceiling_passed = (complete_margin > 0 and complete_ce > 0 and
                          complete_direction >= bars["minimum_complete_head_direction_fraction_each_cell"])
        payload_passed = (ceiling_passed and margin_recovery is not None and
                          margin_recovery >= bars["minimum_subject_payload_margin_recovery_each_cell"] and
                          payload_direction >= bars["minimum_subject_payload_direction_fraction_each_cell"] and
                          ce_recovery is not None and
                          ce_recovery > bars["minimum_subject_payload_ce_recovery_each_cell"])
        cells[cell_id] = {
            "row_count": len(payload),
            "native_accuracy": capability,
            "native_capability_passed": capability_passed,
            "complete_head": {"mean_margin_delta": complete_margin,
                              "margin_direction_fraction": complete_direction,
                              "mean_donor_ce_gain": complete_ce,
                              "ceiling_passed": ceiling_passed},
            "subject_payload": {"mean_margin_delta": payload_margin,
                                "margin_direction_fraction": payload_direction,
                                "margin_recovery_of_complete_head": margin_recovery,
                                "mean_donor_ce_gain": payload_ce,
                                "ce_recovery_of_complete_head": ce_recovery,
                                "passed": payload_passed},
        }
    instrument = (replay_error <= bars["maximum_native_replay_absolute_logit_error"] and
                  identity_error <= bars["maximum_source_term_identity_absolute_error"] and
                  all(cell["native_capability_passed"] and cell["complete_head"]["ceiling_passed"]
                      for cell in cells.values()))
    transfer = instrument and all(cell["subject_payload"]["passed"] for cell in cells.values())
    return {
        "native_replay_max_absolute_logit_error": replay_error,
        "source_term_identity_max_absolute_error": identity_error,
        "cells": cells,
        "predictions": {
            "pred_a_instrument_live": instrument,
            "pred_b_subject_payload_transfers_each_cell": transfer,
            "pred_c_subject_payload_fails_some_cell": instrument and not transfer,
        },
    }


def evaluate(model, torch, F, facade, plan):
    rows = authority.build_rows()
    device = next(model.parameters()).device
    length = max(len(row[key]) for row in rows for key in ("base_ids", "donor_ids"))
    base_tokens, base_finals = _pad(rows, "base", length, torch, device)
    donor_tokens, donor_finals = _pad(rows, "donor", length, torch, device)

    combined_tokens = torch.cat((base_tokens, donor_tokens))
    combined_finals = torch.cat((base_finals, donor_finals))
    native = parent._native_logits(model, combined_tokens, torch, F)
    replay, factors = parent._factor_forward(
        model, combined_tokens, combined_finals, torch, F, facade,
    )
    count = len(rows)
    native_base, native_donor = native[:count], native[count:]
    base, donor = _split_factors(factors, count)
    patch = _patch_batch(base_tokens, base_finals, base, donor, torch)
    patched, patched_factors = parent._factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        source_positions=patch["source_positions"],
        replacement_terms=patch["replacement_terms"],
        replacement_heads=patch["replacement_heads"],
    )
    patched = patched.view(len(CONDITIONS), count, length, -1)

    replay_error = float((replay - native).abs().max())
    identity_error = max(
        float((torch.einsum("bk,bkd->bd", item["p"], item["u"])-item["head"]).abs().max())
        for item in (base, donor, patched_factors)
    )
    capabilities = _accuracy_cells(
        rows, native_base, native_donor, base_finals, donor_finals,
    )
    evidence = []
    for index, row in enumerate(rows):
        q = int(base_finals[index])
        native_margin, native_ce = _donor_metrics(replay[index], row, q, torch)
        for condition_index, condition in enumerate(CONDITIONS):
            margin, ce = _donor_metrics(patched[condition_index, index], row, q, torch)
            evidence.append({
                "row_id": row["row_id"], "cell_id": row["cell_id"],
                "condition": condition, "native_donor_margin": native_margin,
                "donor_margin": margin, "margin_delta": margin - native_margin,
                "native_donor_ce": native_ce, "donor_ce": ce,
                "donor_ce_gain": native_ce - ce,
            })
    return evidence, capabilities, replay_error, identity_error


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = parent._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True,
    )
    with torch.no_grad():
        evidence, capability, replay, identity = evaluate(model, torch, F, facade, plan)
    scored = score(evidence, capability, replay, identity, plan["bars"])
    predictions = scored["predictions"]
    if not predictions["pred_a_instrument_live"]:
        terminal = "invalid"
    elif predictions["pred_b_subject_payload_transfers_each_cell"]:
        terminal = "subject_payload_test_transfer_screen"
    else:
        terminal = "subject_payload_test_transfer_null"
    result = {
        "schema": "task14_head11_3_subject_payload_test_transfer_result_v1",
        "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "terminal": terminal, "score": scored, "evidence": evidence,
        "evaluated_splits": ["TEST_REUSE_NEW_INTERVENTION"],
        "forbidden_splits_opened": [], "model_forwards": 3,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "score": scored}, indent=2))


if __name__ == "__main__":
    main()
