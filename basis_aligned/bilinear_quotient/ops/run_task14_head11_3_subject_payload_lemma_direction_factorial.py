#!/usr/bin/env python3
"""TEST-reuse lemma-versus-direction factorial for the L11H3 subject payload."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_lemma_conditioning_rescues_weak_cell pred_c_plural_to_singular_asymmetry_persists pred_d_same_lemma_direction_symmetric_rescue pred_e_mixed_result

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import circuit_fast_screen_candidate_task14_test_cross_syntax as authority
import run_task14_head11_3_subject_attractor_score_payload_factorial as factor_parent
import run_task14_head11_3_subject_payload_test_transfer as transfer_parent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/fast_screens/task14_head11_3_subject_payload_lemma_direction_factorial_v1_result.json"
PRIOR_ART_SHA256 = "4603e7b2ce1c087358369f104933b5cce5cadb9bb93d9a1b2cfc22d4cf595d9e"
PARENT_RESULT_SHA256 = "157c907abb796012f2b6ba2b0fb8bd302daa66455d01376467b2acdc283b3c1b"
WEAK_CELL = "pp_plural_to_relative_singular"
CONDITIONS = ("same_lemma_payload", "cross_noun_payload", "complete_head")


def _canonical(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def build_rows():
    rows = authority.build_rows()
    authority.validate_rows(rows)
    sources = {(str(row["group_id"]), str(row["transform_id"])): row
               for row in authority._CANDIDATE._source_rows()}
    output = []
    for row in rows:
        source = sources[(row["group_id"], row["target_family"])]
        same = authority._CANDIDATE._endpoint(source, "donor")
        if same["subject_number"] != row["donor_subject_number"]:
            raise ValueError("same-lemma endpoint does not reverse grammatical number")
        if same["answer_id"] != row["donor_answer_id"]:
            raise ValueError("same-lemma and cross-noun donors disagree on answer")
        if same["ids"][0] != row["base_ids"][0] or same["ids"][1] == row["base_ids"][1]:
            raise ValueError("same-lemma endpoint does not preserve the pre-subject prefix")
        if len(same["ids"]) != len(row["base_ids"]) or any(
            left != right for index, (left, right) in enumerate(zip(same["ids"], row["base_ids"]))
            if index != 1
        ):
            raise ValueError("same-lemma endpoint differs from recipient outside the subject token")
        augmented = dict(row)
        augmented.update(
            same_lemma_ids=same["ids"], same_lemma_text=same["text"],
            same_lemma_answer_id=same["answer_id"],
            same_lemma_foil_id=same["foil_id"],
            same_lemma_semantic_position=same["position"],
            same_lemma_subject_number=same["subject_number"],
        )
        output.append(augmented)
    cells = defaultdict(int)
    for row in output:
        cells[row["cell_id"]] += 1
    if len(output) != 64 or set(cells.values()) != {16} or len(cells) != 4:
        raise ValueError(f"paired TEST authority lost balance: {dict(cells)}")
    return output


def compile_plan():
    rows = build_rows()
    return {
        "schema": "task14_head11_3_subject_payload_lemma_direction_factorial_plan_v1",
        "candidate_id": "subject_verb.number_agreement.head11_3_subject_payload_lemma_direction_factorial",
        "split": "TEST_REUSE_NEW_INTERVENTION",
        "screen_tier": "BASIC",
        "row_count": len(rows),
        "parent_authority_sha256": authority.validate_rows(authority.build_rows()),
        "paired_authority_sha256": _canonical(rows),
        "site": {"layer": factor_parent.LAYER, "head": factor_parent.HEAD,
                 "query": "final_prediction_position", "source": "subject_token_index_1"},
        "conditions": list(CONDITIONS),
        "price": {"model_forwards": 3, "example_evaluations": 576,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_directed_is_are_margin", "donor_answer_ce"],
        "bars": {
            "minimum_native_accuracy_each_source_each_cell": .85,
            "maximum_native_replay_absolute_logit_error": 5e-5,
            "maximum_source_term_identity_absolute_error": 5e-5,
            "maximum_parent_cross_noun_reproduction_error": 5e-5,
            "minimum_complete_head_direction_fraction_each_cell": .75,
            "minimum_material_margin_recovery": .25,
            "minimum_margin_direction_fraction": .75,
            "minimum_lemma_rescue_over_cross_noun": .10,
            "maximum_broad_same_lemma_recovery_range": .15,
            "minimum_positive_ce_recovery": 0.0,
        },
        "opposing_interpretations": {
            "lemma_conditioning": (
                "same-lemma recovery in the prior weak cell is at least .25 and exceeds "
                "cross-noun recovery by at least .10"
            ),
            "plural_to_singular_context_asymmetry": (
                "same-lemma and cross-noun recovery both remain below .25 in the prior weak cell"
            ),
            "same_lemma_direction_symmetric_rescue": (
                "same-lemma recovery repairs the weak cross-noun cell, reaches at least .25 in every cell, "
                "and has an across-cell range at most .15; this is still lemma-conditioned, not abstract cross-lemma transfer"
            ),
        },
        "scope": "TEST_REUSE_NEW_INTERVENTION",
        "closed_splits": ["OOD"],
        "limits": "TEST text is reused; this is not a pristine final test. OOD remains closed.",
    }


def _pad(rows, key, length, torch, device):
    tokens = torch.full((len(rows), length), 50256, dtype=torch.long, device=device)
    finals = []
    for index, row in enumerate(rows):
        ids = row[key]
        tokens[index, :len(ids)] = torch.tensor(ids, device=device)
        finals.append(len(ids) - 1)
    return tokens, torch.tensor(finals, device=device)


def _split_factors(factors, count):
    return tuple({key: value[start:start + count] for key, value in factors.items()}
                 for start in (0, count, 2 * count))


def _patch_batch(base_tokens, base_finals, base, same_lemma, cross_noun, torch):
    count = len(base_tokens)
    subject = torch.ones(count, dtype=torch.long, device=base_tokens.device)
    recipient_p, recipient_u = factor_parent._selected(base, subject, torch)
    _same_p, same_u = factor_parent._selected(same_lemma, subject, torch)
    _cross_p, cross_u = factor_parent._selected(cross_noun, subject, torch)
    return {
        "tokens": base_tokens.repeat(3, 1),
        "finals": base_finals.repeat(3),
        "source_positions": subject.repeat(3),
        "replacement_terms": torch.cat((
            recipient_p.unsqueeze(-1) * same_u,
            recipient_p.unsqueeze(-1) * cross_u,
            recipient_p.unsqueeze(-1) * recipient_u,
        )),
        "replacement_heads": torch.cat((base["head"], base["head"], cross_noun["head"])),
    }


def _capability(rows, native_parts, finals_parts):
    cells = defaultdict(lambda: {"base": [], "same_lemma": [], "cross_noun": []})
    specs = (
        ("base", "base_answer_id", "base_foil_id"),
        ("same_lemma", "same_lemma_answer_id", "same_lemma_foil_id"),
        ("cross_noun", "donor_answer_id", "donor_foil_id"),
    )
    for index, row in enumerate(rows):
        for part, (name, answer_key, foil_key), finals in zip(native_parts, specs, finals_parts):
            q = int(finals[index])
            cells[row["cell_id"]][name].append(
                float(part[index, q, int(row[answer_key])]
                      - part[index, q, int(row[foil_key])]) > 0
            )
    return {cell: {name: sum(values) / len(values) for name, values in groups.items()}
            for cell, groups in sorted(cells.items())}


def _parent_cross_evidence():
    path = transfer_parent.OUT
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_RESULT_SHA256:
        raise RuntimeError("parent TEST subject-payload result changed")
    result = json.loads(path.read_text())
    return {row["row_id"]: row for row in result["evidence"]
            if row["condition"] == "subject_payload"}


def score(evidence, capability, replay_error, identity_error, reproduction_error, bars):
    grouped = defaultdict(list)
    for row in evidence:
        grouped[(row["cell_id"], row["condition"])].append(row)
    cells = {}
    for cell_id in sorted(capability):
        complete = grouped[(cell_id, "complete_head")]
        complete_margin = statistics.fmean(row["margin_delta"] for row in complete)
        complete_ce = statistics.fmean(row["donor_ce_gain"] for row in complete)
        complete_direction = sum(row["margin_delta"] > 0 for row in complete) / len(complete)
        ceiling = complete_margin > 0 and complete_ce > 0 and complete_direction >= \
            bars["minimum_complete_head_direction_fraction_each_cell"]
        arms = {}
        for condition in ("same_lemma_payload", "cross_noun_payload"):
            rows = grouped[(cell_id, condition)]
            margin = statistics.fmean(row["margin_delta"] for row in rows)
            ce = statistics.fmean(row["donor_ce_gain"] for row in rows)
            arms[condition] = {
                "mean_margin_delta": margin,
                "margin_direction_fraction": sum(row["margin_delta"] > 0 for row in rows) / len(rows),
                "margin_recovery_of_complete_head": margin / complete_margin if complete_margin > 0 else None,
                "mean_donor_ce_gain": ce,
                "ce_recovery_of_complete_head": ce / complete_ce if complete_ce > 0 else None,
            }
        cells[cell_id] = {
            "row_count": len(complete), "native_accuracy": capability[cell_id],
            "native_capability_passed": min(capability[cell_id].values()) >=
                bars["minimum_native_accuracy_each_source_each_cell"],
            "complete_head": {"mean_margin_delta": complete_margin,
                              "margin_direction_fraction": complete_direction,
                              "mean_donor_ce_gain": complete_ce, "ceiling_passed": ceiling},
            **arms,
        }
    instrument = (replay_error <= bars["maximum_native_replay_absolute_logit_error"] and
                  identity_error <= bars["maximum_source_term_identity_absolute_error"] and
                  reproduction_error <= bars["maximum_parent_cross_noun_reproduction_error"] and
                  all(cell["native_capability_passed"] and cell["complete_head"]["ceiling_passed"]
                      for cell in cells.values()))
    weak = cells[WEAK_CELL]
    same = weak["same_lemma_payload"]
    cross = weak["cross_noun_payload"]
    same_recovery = same["margin_recovery_of_complete_head"]
    cross_recovery = cross["margin_recovery_of_complete_head"]
    asymmetry = (instrument and same_recovery < bars["minimum_material_margin_recovery"] and
                 cross_recovery < bars["minimum_material_margin_recovery"])
    same_recoveries = [cell["same_lemma_payload"]["margin_recovery_of_complete_head"]
                       for cell in cells.values()]
    direction_symmetric_rescue = (
             instrument and
             same_recovery >= cross_recovery + bars["minimum_lemma_rescue_over_cross_noun"] and
             min(same_recoveries) >= bars["minimum_material_margin_recovery"] and
             max(same_recoveries) - min(same_recoveries) <=
             bars["maximum_broad_same_lemma_recovery_range"] and
             all(cell["same_lemma_payload"]["margin_direction_fraction"] >=
                 bars["minimum_margin_direction_fraction"] and
                 cell["same_lemma_payload"]["ce_recovery_of_complete_head"] >
                 bars["minimum_positive_ce_recovery"] for cell in cells.values()))
    lemma_rescue = (instrument and not direction_symmetric_rescue and
                    same_recovery >= bars["minimum_material_margin_recovery"] and
                    same_recovery >= cross_recovery + bars["minimum_lemma_rescue_over_cross_noun"] and
                    same["margin_direction_fraction"] >= bars["minimum_margin_direction_fraction"] and
                    same["ce_recovery_of_complete_head"] > bars["minimum_positive_ce_recovery"])
    mixed = instrument and not (lemma_rescue or asymmetry or direction_symmetric_rescue)
    return {
        "native_replay_max_absolute_logit_error": replay_error,
        "source_term_identity_max_absolute_error": identity_error,
        "parent_cross_noun_max_absolute_reproduction_error": reproduction_error,
        "cells": cells,
        "predictions": {
            "pred_a_instrument_live": instrument,
            "pred_b_lemma_conditioning_rescues_weak_cell": lemma_rescue,
            "pred_c_plural_to_singular_asymmetry_persists": asymmetry,
            "pred_d_same_lemma_direction_symmetric_rescue": direction_symmetric_rescue,
            "pred_e_mixed_result": mixed,
        },
    }


def evaluate(model, torch, F, facade, plan):
    rows = build_rows()
    device = next(model.parameters()).device
    keys = ("base_ids", "same_lemma_ids", "donor_ids")
    length = max(len(row[key]) for row in rows for key in keys)
    padded = tuple(_pad(rows, key, length, torch, device) for key in keys)
    combined_tokens = torch.cat(tuple(item[0] for item in padded))
    combined_finals = torch.cat(tuple(item[1] for item in padded))
    native = factor_parent._native_logits(model, combined_tokens, torch, F)
    replay, factors = factor_parent._factor_forward(
        model, combined_tokens, combined_finals, torch, F, facade,
    )
    count = len(rows)
    native_parts = tuple(native[start:start + count] for start in (0, count, 2 * count))
    factor_parts = _split_factors(factors, count)
    patch = _patch_batch(padded[0][0], padded[0][1], *factor_parts, torch)
    patched, patched_factors = factor_parent._factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        source_positions=patch["source_positions"],
        replacement_terms=patch["replacement_terms"],
        replacement_heads=patch["replacement_heads"],
    )
    patched = patched.view(len(CONDITIONS), count, length, -1)
    replay_error = float((replay - native).abs().max())
    identity_error = max(
        float((torch.einsum("bk,bkd->bd", item["p"], item["u"])-item["head"]).abs().max())
        for item in (*factor_parts, patched_factors)
    )
    capability = _capability(rows, native_parts, tuple(item[1] for item in padded))
    evidence = []
    parent_rows = _parent_cross_evidence()
    reproduction_error = 0.0
    for index, row in enumerate(rows):
        q = int(padded[0][1][index])
        native_margin, native_ce = transfer_parent._donor_metrics(replay[index], row, q, torch)
        for condition_index, condition in enumerate(CONDITIONS):
            margin, ce = transfer_parent._donor_metrics(
                patched[condition_index, index], row, q, torch,
            )
            record = {
                "row_id": row["row_id"], "cell_id": row["cell_id"],
                "condition": condition, "native_donor_margin": native_margin,
                "donor_margin": margin, "margin_delta": margin - native_margin,
                "native_donor_ce": native_ce, "donor_ce": ce,
                "donor_ce_gain": native_ce - ce,
            }
            evidence.append(record)
            if condition == "cross_noun_payload":
                parent_row = parent_rows[row["row_id"]]
                reproduction_error = max(
                    reproduction_error,
                    abs(record["native_donor_margin"] - parent_row["native_donor_margin"]),
                    abs(record["donor_margin"] - parent_row["donor_margin"]),
                    abs(record["margin_delta"] - parent_row["margin_delta"]),
                    abs(record["native_donor_ce"] - parent_row["native_donor_ce"]),
                    abs(record["donor_ce"] - parent_row["donor_ce"]),
                )
    return evidence, capability, replay_error, identity_error, reproduction_error


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = factor_parent._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True,
    )
    with torch.no_grad():
        evidence, capability, replay, identity, reproduction = evaluate(
            model, torch, F, facade, plan,
        )
    scored = score(evidence, capability, replay, identity, reproduction, plan["bars"])
    predictions = scored["predictions"]
    if not predictions["pred_a_instrument_live"]:
        terminal = "invalid"
    elif predictions["pred_d_same_lemma_direction_symmetric_rescue"]:
        terminal = "same_lemma_direction_symmetric_rescue_screen"
    elif predictions["pred_b_lemma_conditioning_rescues_weak_cell"]:
        terminal = "lemma_conditioning_screen"
    elif predictions["pred_c_plural_to_singular_asymmetry_persists"]:
        terminal = "plural_to_singular_asymmetry_screen"
    else:
        terminal = "mixed_lemma_direction_result"
    result = {
        "schema": "task14_head11_3_subject_payload_lemma_direction_factorial_result_v1",
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
