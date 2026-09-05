#!/usr/bin/env python3
"""Specificity screen for the Task14 L11H3 subject-source payload."""

# BQGATE: EXPERIMENT

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import circuit_fast_screen_candidate_task14_select_cross_noun as authority
import run_task14_head11_3_subject_attractor_score_payload_factorial as parent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/fast_screens/task14_head11_3_subject_payload_number_specificity_v1_result.json"
PRIOR_ART_SHA256 = "5a92eed7bdcca07486ce502c9140ec0006e1684c3c0a6fca09fb308eef3c736c"
CONDITIONS = ("same_number_payload", "opposite_number_payload")
PARENT_RESULT_SHA256 = "3885fe1ed2905cb51fc0e649f7e0107a1b2ecadb52ac30c11344f45de8242202"


def _build_triples_unvalidated():
    panels = authority._panels()
    pairing = authority._cross_noun_pairing(panels)
    output = []
    parent_rows = {(row["target_group_id"], row["target_family"]): row
                   for row in authority.build_rows()}
    for target_group, panel in sorted(
        panels.items(), key=lambda item: int(item[1]["A1"]["group_number"]),
    ):
        donor_group = pairing[target_group]
        for target_family, donor_family in (("A1", "A2"), ("A2", "A1")):
            recipient = authority.matched._endpoint(panel[target_family], "base")
            same = authority.matched._endpoint(panels[donor_group][donor_family], "base")
            opposite = authority.matched._endpoint(panels[donor_group][donor_family], "donor")
            parent_row = parent_rows[(target_group, target_family)]
            if recipient["ids"] != parent_row["base_ids"] \
                    or opposite["ids"] != parent_row["donor_ids"]:
                raise ValueError("opposite condition no longer reproduces the parent row")
            identity = ["task14_subject_payload_specificity_v1", target_group,
                        donor_group, target_family]
            output.append({
                    "row_id": authority.matched.canonical_sha256(identity),
                    "parent_row_id": parent_row["row_id"],
                    "cell_id": f"{target_family.lower()}_{recipient['subject_number']}",
                    "target_group_id": target_group,
                    "donor_group_id": donor_group,
                    "target_family": target_family,
                    "donor_family": donor_family,
                    "base_ids": recipient["ids"],
                    "same_ids": same["ids"],
                    "opposite_ids": opposite["ids"],
                    "base_text": recipient["text"],
                    "same_text": same["text"],
                    "opposite_text": opposite["text"],
                    "base_subject_number": recipient["subject_number"],
                    "same_subject_number": same["subject_number"],
                    "opposite_subject_number": opposite["subject_number"],
                    "base_attractor_plural": recipient["attractor_plural"],
                    "same_attractor_plural": same["attractor_plural"],
                    "opposite_attractor_plural": opposite["attractor_plural"],
                    "base_answer_id": recipient["answer_id"],
                    "base_foil_id": recipient["foil_id"],
                    "opposite_answer_id": opposite["answer_id"],
                })
    return sorted(output, key=lambda row: row["row_id"])


def build_triples():
    rows = _build_triples_unvalidated()
    if len(rows) != 64 or len({row["row_id"] for row in rows}) != 64:
        raise ValueError("specificity authority must contain 64 unique triples")
    cells = defaultdict(int)
    for row in rows:
        cells[row["cell_id"]] += 1
        if row["target_group_id"] == row["donor_group_id"]:
            raise ValueError("recipient and donors use the same noun group")
        if len({row["base_ids"][1], row["same_ids"][1],
                row["opposite_ids"][1]}) != 3:
            raise ValueError("recipient and donors do not use three subject tokens")
        if len(row["same_ids"]) != len(row["opposite_ids"]) \
                or any(left != right for index, (left, right) in
                       enumerate(zip(row["same_ids"], row["opposite_ids"])) if index != 1):
            raise ValueError("same/opposite donors differ outside the subject token")
        if row["base_subject_number"] != row["same_subject_number"] \
                or row["base_subject_number"] == row["opposite_subject_number"]:
            raise ValueError("same/opposite grammatical-number relation changed")
        if len({row["base_attractor_plural"], row["same_attractor_plural"],
                row["opposite_attractor_plural"]}) != 1:
            raise ValueError("attractor plurality is not matched")
        if row["target_family"] == row["donor_family"]:
            raise ValueError("donor syntax is not crossed")
        if row["opposite_answer_id"] != row["base_foil_id"]:
            raise ValueError("opposite donor does not reverse the answer")
    if set(cells.values()) != {16} or len(cells) != 4:
        raise ValueError(f"specificity cells are imbalanced: {dict(cells)}")
    regenerated = _build_triples_unvalidated()
    if authority.matched.canonical_sha256(rows) != authority.matched.canonical_sha256(regenerated):
        raise ValueError("triples differ from exact frozen regeneration")
    return rows


def compile_plan():
    rows = build_triples()
    return {
        "schema": "task14_head11_3_subject_payload_number_specificity_plan_v1",
        "candidate_id": "subject_verb.number_agreement.head11_3_subject_payload_number_specificity",
        "split": "SELECT", "screen_tier": "BASIC", "row_count": len(rows),
        "authority_sha256": authority.matched.canonical_sha256(rows),
        "site": {"layer": parent.LAYER, "head": parent.HEAD,
                 "query": "final_prediction_position", "source": "subject_token_index_1"},
        "intervention": "recipient subject score times donor subject projected value",
        "conditions": list(CONDITIONS),
        "price": {"model_forwards": 3, "example_evaluations": 384,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["is_are_task_margin", "answer_ce",
                     "same_number_absolute_leakage_over_opposite_number_live_effect"],
        "bars": {"minimum_native_accuracy": .85,
                 "minimum_opposite_margin_direction_fraction": .75,
                 "minimum_mean_opposite_margin_effect": .50,
                 "minimum_mean_opposite_ce_gain": .25,
                 "maximum_parent_opposite_reproduction_error": 5e-5,
                 "maximum_same_over_opposite_margin_ratio": .25,
                 "maximum_same_over_opposite_ce_ratio": .25},
        "closed_splits": ["TEST", "OOD"],
        "limits": "SELECT specificity screen only; no held-out/OOD or complete-circuit claim.",
    }


def _pad(rows, key, length, torch, device):
    tokens = torch.full((len(rows), length), 50256, dtype=torch.long, device=device)
    finals = []
    for index, row in enumerate(rows):
        ids = row[key]
        tokens[index, :len(ids)] = torch.tensor(ids, device=device)
        finals.append(len(ids) - 1)
    return tokens, torch.tensor(finals, device=device)


def _task_metrics(logits, row, q, answer_key, torch):
    answer = int(row[answer_key])
    foil = int(row["base_foil_id"] if answer_key == "base_answer_id" else row["base_answer_id"])
    margin = float(logits[q, answer] - logits[q, foil])
    ce = float(-torch.log_softmax(logits[q], dim=-1)[answer])
    return margin, ce


def _payload_terms(recipient, same_donor, opposite_donor, subjects, torch):
    recipient_p, _recipient_u = parent._selected(recipient, subjects, torch)
    _same_p, same_u = parent._selected(same_donor, subjects, torch)
    _opposite_p, opposite_u = parent._selected(opposite_donor, subjects, torch)
    return torch.cat((recipient_p.unsqueeze(-1) * same_u,
                      recipient_p.unsqueeze(-1) * opposite_u))


def score(evidence, replay_error, identity_error, parent_reproduction_error, native_accuracy):
    same = [row for row in evidence if row["condition"] == "same_number_payload"]
    opposite = [row for row in evidence if row["condition"] == "opposite_number_payload"]
    opposite_margin = statistics.fmean(row["directed_margin_delta"] for row in opposite)
    opposite_ce = statistics.fmean(row["answer_ce_gain"] for row in opposite)
    same_margin_abs = statistics.fmean(abs(row["task_margin_delta"]) for row in same)
    same_ce_abs = statistics.fmean(abs(row["answer_ce_gain"]) for row in same)
    margin_ratio = same_margin_abs / opposite_margin if opposite_margin > 0 else None
    ce_ratio = same_ce_abs / opposite_ce if opposite_ce > 0 else None
    by_cell = {}
    for cell_id in sorted({row["cell_id"] for row in evidence}):
        same_cell = [row for row in same if row["cell_id"] == cell_id]
        opposite_cell = [row for row in opposite if row["cell_id"] == cell_id]
        cell_margin = statistics.fmean(row["directed_margin_delta"] for row in opposite_cell)
        cell_ce = statistics.fmean(row["answer_ce_gain"] for row in opposite_cell)
        cell_same_margin = statistics.fmean(abs(row["task_margin_delta"]) for row in same_cell)
        cell_same_ce = statistics.fmean(abs(row["answer_ce_gain"]) for row in same_cell)
        by_cell[cell_id] = {
            "opposite_mean_directed_margin_effect": cell_margin,
            "opposite_mean_donor_answer_ce_gain": cell_ce,
            "same_absolute_margin_over_opposite_effect": (
                cell_same_margin / cell_margin if cell_margin > 0 else None
            ),
            "same_absolute_ce_over_opposite_gain": (
                cell_same_ce / cell_ce if cell_ce > 0 else None
            ),
        }
    live = (native_accuracy >= .85 and replay_error <= 5e-5 and identity_error <= 5e-5
            and parent_reproduction_error <= 5e-5
            and opposite_margin >= .50 and opposite_ce >= .25
            and all(cell["opposite_mean_directed_margin_effect"] > 0
                    and cell["opposite_mean_donor_answer_ce_gain"] > 0
                    for cell in by_cell.values())
            and sum(row["directed_margin_delta"] > 0 for row in opposite) / len(opposite) >= .75)
    specific = (live and margin_ratio <= .25 and ce_ratio <= .25
                and all(cell["same_absolute_margin_over_opposite_effect"] <= .25
                        and cell["same_absolute_ce_over_opposite_gain"] <= .25
                        for cell in by_cell.values()))
    return {
        "native_accuracy": native_accuracy,
        "native_replay_max_absolute_logit_error": replay_error,
        "source_term_identity_max_absolute_error": identity_error,
        "parent_opposite_condition_max_absolute_reproduction_error": parent_reproduction_error,
        "opposite_number": {
            "mean_directed_margin_effect": opposite_margin,
            "margin_direction_fraction": sum(row["directed_margin_delta"] > 0 for row in opposite) / len(opposite),
            "mean_donor_answer_ce_gain": opposite_ce,
        },
        "same_number": {
            "mean_absolute_task_margin_change": same_margin_abs,
            "mean_absolute_base_answer_ce_change": same_ce_abs,
            "absolute_margin_leakage_over_live_effect": margin_ratio,
            "absolute_ce_leakage_over_live_effect": ce_ratio,
        },
        "cells": by_cell,
        "predictions": {
            "pred_a_opposite_number_control_live": live,
            "pred_b_number_specific_payload": specific,
            "pred_c_noun_or_syntax_sensitive_payload": live and not specific,
        },
    }


def _parent_evidence():
    path = ROOT / "circuits/fast_screens/task14_head11_3_subject_attractor_score_payload_factorial_v1_result.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_RESULT_SHA256:
        raise RuntimeError("parent subject-payload result changed")
    result = json.loads(path.read_text())
    return {row["row_id"]: row for row in result["evidence"]
            if row["condition"] == "subject_payload"}


def evaluate(model, torch, F, facade):
    rows = build_triples()
    device = next(model.parameters()).device
    length = max(len(row[key]) for row in rows for key in ("base_ids", "same_ids", "opposite_ids"))
    base_tokens, base_finals = _pad(rows, "base_ids", length, torch, device)
    same_tokens, same_finals = _pad(rows, "same_ids", length, torch, device)
    opposite_tokens, opposite_finals = _pad(rows, "opposite_ids", length, torch, device)
    all_tokens = torch.cat((base_tokens, same_tokens, opposite_tokens))
    all_finals = torch.cat((base_finals, same_finals, opposite_finals))
    replay, factors = parent._factor_forward(model, all_tokens, all_finals, torch, F, facade)
    count = len(rows)
    base_factors = {key: value[:count] for key, value in factors.items()}
    same_factors = {key: value[count:2*count] for key, value in factors.items()}
    opposite_factors = {key: value[2*count:] for key, value in factors.items()}
    subjects = torch.ones(count, dtype=torch.long, device=device)
    replacement = _payload_terms(base_factors, same_factors, opposite_factors, subjects, torch)
    patched, patched_factors = parent._factor_forward(
        model, base_tokens.repeat(2, 1), base_finals.repeat(2), torch, F, facade,
        source_positions=subjects.repeat(2), replacement_terms=replacement,
    )
    identity_error = max(
        float((torch.einsum("bk,bkd->bd", value["p"], value["u"])-value["head"]).abs().max())
        for value in (base_factors, same_factors, opposite_factors, patched_factors)
    )
    native_reference = parent._native_logits(model, base_tokens, torch, F)
    replay_error = float((replay[:count] - native_reference).abs().max())
    evidence, correct = [], []
    parent_evidence = _parent_evidence()
    reproduction_error = 0.0
    for index, row in enumerate(rows):
        q = int(base_finals[index])
        native_margin, native_ce = _task_metrics(replay[index], row, q, "base_answer_id", torch)
        correct.append(native_margin > 0)
        same_margin, same_ce = _task_metrics(patched[index], row, q, "base_answer_id", torch)
        opposite_margin, opposite_ce = _task_metrics(
            patched[count + index], row, q, "opposite_answer_id", torch,
        )
        native_opposite_margin = -native_margin
        native_opposite_ce = float(-torch.log_softmax(replay[index, q], dim=-1)[int(row["opposite_answer_id"])])
        parent_row = parent_evidence[row["parent_row_id"]]
        reproduction_error = max(
            reproduction_error,
            abs((opposite_margin - native_opposite_margin) - parent_row["margin_delta"]),
            abs(opposite_ce - parent_row["donor_ce"]),
            abs(native_opposite_ce - parent_row["native_donor_ce"]),
        )
        evidence.extend((
            {"row_id": row["row_id"], "cell_id": row["cell_id"],
             "condition": "same_number_payload", "native_task_margin": native_margin,
             "task_margin": same_margin, "task_margin_delta": same_margin - native_margin,
             "native_answer_ce": native_ce, "answer_ce": same_ce,
             "answer_ce_gain": native_ce - same_ce, "directed_margin_delta": same_margin - native_margin},
            {"row_id": row["row_id"], "cell_id": row["cell_id"],
             "condition": "opposite_number_payload", "native_task_margin": native_opposite_margin,
             "task_margin": opposite_margin,
             "task_margin_delta": opposite_margin - native_opposite_margin,
             "native_answer_ce": native_opposite_ce, "answer_ce": opposite_ce,
             "answer_ce_gain": native_opposite_ce - opposite_ce,
             "directed_margin_delta": opposite_margin - native_opposite_margin},
        ))
    return evidence, replay_error, identity_error, reproduction_error, sum(correct) / len(correct)


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = parent._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, replay, identity, reproduction, capability = evaluate(model, torch, F, facade)
    scored = score(evidence, replay, identity, reproduction, capability)
    predictions = scored["predictions"]
    if not predictions["pred_a_opposite_number_control_live"]:
        terminal = "invalid_opposite_number_control"
    elif predictions["pred_b_number_specific_payload"]:
        terminal = "number_specific_payload_screen"
    else:
        terminal = "noun_or_syntax_sensitive_payload_null"
    result = {
        "schema": "task14_head11_3_subject_payload_number_specificity_result_v1",
        "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "terminal": terminal, "score": scored, "evidence": evidence,
        "evaluated_splits": ["SELECT_BASIC"], "forbidden_splits_opened": [],
        "model_forwards": plan["price"]["model_forwards"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "score": scored}, indent=2))


if __name__ == "__main__":
    main()
