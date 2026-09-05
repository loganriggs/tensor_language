#!/usr/bin/env python3
"""Select an A1 narrative-tense template on 0:16; score its frozen 16:32 holdout."""

# BQGATE: EXPERIMENT pred_a_selection_complete pred_b_holdout_capable pred_c_no_eligible_template

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Sequence

import circuit_fast_screen_candidate_narrative_tense_fresh_unchanged_carrier as source
import circuit_fast_screen_managed_runner as managed
import run_task14_head11_3_subject_attractor_score_payload_factorial as model_helpers


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/narrative_tense_a1_direct_template_capability_select_holdout_v1.json"
OUT = ROOT / "circuits/fast_screens/narrative_tense_a1_direct_template_capability_select_holdout_v1_result.json"
PRIOR_ART_SHA256 = "3684ca71bc0f83a9aa38459b67985e8b09a70cd5899762a882fae521b8467522"
SOURCE_SHA256 = "3ad1054469ca5f6f4c37558e0a46433d4647ce705e4d2cdbe7e82c653ead336f"
TEMPLATE_ORDER = ("remained", "served_one_purpose", "had_one_purpose")
TEMPLATE_VERBS = {
    "remained": ("remained indoors", "remains indoors"),
    "served_one_purpose": ("served one purpose", "serves one purpose"),
    "had_one_purpose": ("had one purpose", "has one purpose"),
}
MINIMUM_HOLDOUT_ACCURACY_EACH_TENSE = .875


class CapabilityError(ValueError):
    """The frozen capability-selection contract was violated."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt(template_id: str, subject: str, tail: str, past: bool) -> str:
    lead = "Yesterday" if past else "Today"
    phrase = TEMPLATE_VERBS[template_id][0 if past else 1]
    return f"{lead} the {subject} {phrase}. The central purpose of the {tail}"


def build_pairs(template_ids: Sequence[str], start: int, stop: int,
                families: Sequence[str] = ("A1",)):
    if tuple(template_ids) != tuple(x for x in TEMPLATE_ORDER if x in template_ids):
        raise CapabilityError("templates must preserve their frozen tie order")
    if (start, stop) not in {(0, 16), (16, 32)}:
        raise CapabilityError("group boundary differs from the frozen selection design")
    if tuple(families) not in {("A1",), ("A1", "P")}:
        raise CapabilityError("only the frozen A1 or A1+P panels may be built")
    pairs = []
    for template_id in template_ids:
        for group in range(start, stop):
            for family in families:
                base_past = group % 2 == 0
                if family == "A1":
                    direction = "past_to_present" if base_past else "present_to_past"
                    endpoint_specs = (("base", source.SUBJECTS[group], base_past),
                                      ("donor", source.SUBJECTS[group], not base_past))
                else:
                    direction = ("primary_to_alternative" if base_past
                                 else "alternative_to_primary")
                    primary, alternate = source.SUBJECTS[group], source.ALTERNATES[group]
                    left, right = ((primary, alternate) if base_past else (alternate, primary))
                    endpoint_specs = (("base", left, base_past), ("donor", right, base_past))
                endpoints = []
                for side, subject, past in endpoint_specs:
                    prompt = _prompt(template_id, subject, source.TAILS[group], past)
                    answer, foil = ((" was", " is") if past else (" is", " was"))
                    ids = source.builder.ENCODING.encode(prompt)
                    answer_id = source.builder.ENCODING.encode(answer)
                    foil_id = source.builder.ENCODING.encode(foil)
                    if len(answer_id) != 1 or len(foil_id) != 1 \
                            or source.builder.ENCODING.decode(ids) != prompt \
                            or source.builder.ENCODING.encode(prompt + answer) != ids + answer_id:
                        raise CapabilityError("joint tokenization changed")
                    endpoints.append({"side": side, "past": past, "prompt": prompt, "ids": ids,
                                      "answer": answer, "foil": foil, "answer_id": answer_id[0],
                                      "foil_id": foil_id[0]})
                pairs.append({"template_id": template_id, "family": family,
                              "group_number": group, "direction": direction,
                              "endpoints": endpoints})
                base_ids, donor_ids = (endpoint["ids"] for endpoint in endpoints)
                changed = tuple(i for i, values in enumerate(zip(base_ids, donor_ids))
                                if values[0] != values[1])
                expected = (0, 3) if family == "A1" else (2,)
                if len(base_ids) != len(donor_ids) or base_ids[-1] != donor_ids[-1] \
                        or changed != expected:
                    raise CapabilityError("paired token alignment changed")
    counts = {}
    for row in pairs:
        counts[(row["template_id"], row["family"], row["direction"])] = \
            counts.get((row["template_id"], row["family"], row["direction"]), 0) + 1
    expected_directions = {"A1": ("past_to_present", "present_to_past"),
                           "P": ("primary_to_alternative", "alternative_to_primary")}
    if any(counts.get((template, family, direction)) != 8 for template in template_ids
           for family in families for direction in expected_directions[family]):
        raise CapabilityError("directions are not balanced within the half")
    return pairs


def compile_plan():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256 or _sha256(Path(source.__file__)) != SOURCE_SHA256:
        raise CapabilityError("frozen receipt or source authority changed")
    fit = build_pairs(TEMPLATE_ORDER, 0, 16)
    return {
        "schema": "narrative_tense_a1_direct_template_capability_select_holdout_plan_v1",
        "candidate_id": "narrative_tense.a1_direct_template_capability_select_holdout_v1",
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "circuit_interventions": 0, "templates": list(TEMPLATE_ORDER),
        "selection_groups": list(range(16)), "construction_holdout_groups": list(range(16, 32)),
        "fit_pair_count": len(fit), "holdout_pair_count": 32,
        "selection_rule": ["maximum worst A1 direction-by-side accuracy",
                           "eligibility requires every A1 cell at least 7/8",
                           "maximum worst A1 cell mean signed margin", "fixed template order"],
        "holdout_bar": {"minimum_accuracy_each_A1_and_P_direction_by_side_cell":
                        MINIMUM_HOLDOUT_ACCURACY_EACH_TENSE},
        "price": {"model_forwards": 2, "example_evaluations": 160,
                  "backwards": 0, "parameter_updates": 0},
        "interpretation_limit": "Dataset capability only; not circuit or carrier evidence.",
    }


def _summarize(evidence):
    output = {}
    for template in sorted({x["template_id"] for x in evidence}, key=TEMPLATE_ORDER.index):
        items = [x for x in evidence if x["template_id"] == template]
        by_tense = {}
        for tense in ("past", "present"):
            chosen = [x for x in items if x["tense"] == tense]
            by_tense[tense] = {
                "count": len(chosen), "accuracy": statistics.fmean(x["correct"] for x in chosen),
                "mean_full_vocab_CE": statistics.fmean(x["full_vocab_CE"] for x in chosen),
                "mean_answer_minus_foil_margin": statistics.fmean(x["answer_minus_foil_margin"]
                                                                  for x in chosen),
            }
        cells = {}
        for cell in sorted({x["cell_id"] for x in items}):
            chosen = [x for x in items if x["cell_id"] == cell]
            cells[cell] = {
                "count": len(chosen), "accuracy": statistics.fmean(x["correct"] for x in chosen),
                "mean_full_vocab_CE": statistics.fmean(x["full_vocab_CE"] for x in chosen),
                "mean_signed_margin": statistics.fmean(x["answer_minus_foil_margin"] for x in chosen),
            }
        output[template] = {
            "count": len(items), "accuracy": statistics.fmean(x["correct"] for x in items),
            "mean_full_vocab_CE": statistics.fmean(x["full_vocab_CE"] for x in items),
            "mean_answer_minus_foil_margin": statistics.fmean(x["answer_minus_foil_margin"]
                                                              for x in items),
            "by_tense": by_tense, "direction_side_cells": cells,
        }
    return output


def select_template(summary):
    if tuple(summary) != TEMPLATE_ORDER:
        raise CapabilityError("selection summary is incomplete or reordered")
    def key(template):
        item = summary[template]
        a1 = [value for cell, value in item["direction_side_cells"].items()
              if cell.startswith("A1/")]
        worst_accuracy = min(value["accuracy"] for value in a1)
        eligible = worst_accuracy >= MINIMUM_HOLDOUT_ACCURACY_EACH_TENSE
        worst_margin = min(value["mean_signed_margin"] for value in a1)
        return (-int(eligible), -worst_accuracy, -worst_margin, TEMPLATE_ORDER.index(template))
    return min(TEMPLATE_ORDER, key=key)


def fit_template_eligible(summary, selected):
    return min(value["accuracy"] for value in
               summary[selected]["direction_side_cells"].values()) >= \
        MINIMUM_HOLDOUT_ACCURACY_EACH_TENSE


def _tokens(pairs, torch, device):
    endpoints = [(row, endpoint) for row in pairs for endpoint in row["endpoints"]]
    length = max(len(endpoint["ids"]) for _, endpoint in endpoints)
    tokens = torch.full((len(endpoints), length), 50256, dtype=torch.long, device=device)
    finals = []
    for index, (_, endpoint) in enumerate(endpoints):
        ids = endpoint["ids"]
        tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        finals.append(len(ids) - 1)
    return endpoints, tokens, torch.tensor(finals, dtype=torch.long, device=device)


def evaluate_pairs(model, pairs, torch, F):
    device = next(model.parameters()).device
    endpoints, tokens, finals = _tokens(pairs, torch, device)
    logits = model_helpers._native_logits(model, tokens, torch, F)
    evidence = []
    for index, (row, endpoint) in enumerate(endpoints):
        q = int(finals[index]); answer = endpoint["answer_id"]; foil = endpoint["foil_id"]
        ce = float(-torch.log_softmax(logits[index, q], dim=-1)[answer])
        margin = float(logits[index, q, answer] - logits[index, q, foil])
        evidence.append({
            "template_id": row["template_id"], "family": row["family"],
            "cell_id": f"{row['family']}/{row['direction']}/{endpoint['side']}",
            "group_number": row["group_number"],
            "direction": row["direction"], "side": endpoint["side"],
            "tense": "past" if endpoint["past"] else "present",
            "answer": endpoint["answer"], "foil": endpoint["foil"],
            "correct": bool(margin > 0), "answer_minus_foil_margin": margin,
            "full_vocab_CE": ce,
        })
    return evidence


def evaluate_two_stage(model, torch, F):
    """Open holdout only after the frozen FIT eligibility decision succeeds."""
    fit_evidence = evaluate_pairs(model, build_pairs(TEMPLATE_ORDER, 0, 16), torch, F)
    fit_summary = _summarize(fit_evidence)
    selected = select_template(fit_summary)
    eligible = fit_template_eligible(fit_summary, selected)
    if not eligible:
        return fit_evidence, fit_summary, selected, False, [], None
    holdout_evidence = evaluate_pairs(
        model, build_pairs((selected,), 16, 32, ("A1", "P")), torch, F)
    return (fit_evidence, fit_summary, selected, True, holdout_evidence,
            _summarize(holdout_evidence)[selected])


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise CapabilityError(f"{name} must be absent or exactly 1")
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise CapabilityError(f"refusing to overwrite {OUT}")
    torch, F, facade = model_helpers._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        (fit_evidence, fit_summary, selected, fit_eligible,
         holdout_evidence, holdout_summary) = evaluate_two_stage(model, torch, F)
    capable = fit_eligible and all(
        value["accuracy"] >= MINIMUM_HOLDOUT_ACCURACY_EACH_TENSE
        for value in holdout_summary["direction_side_cells"].values())
    terminal = ("fit_no_eligible_template" if not fit_eligible
                else "capable" if capable else "holdout_incapable")
    actual_price = {"model_forwards": 2 if fit_eligible else 1,
                    "example_evaluations": 160 if fit_eligible else 96,
                    "backwards": 0, "parameter_updates": 0}
    result = {
        "schema": "narrative_tense_a1_direct_template_capability_select_holdout_result_v1",
        "candidate_id": plan["candidate_id"], "terminal": terminal,
        "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "selected_template": selected, "fit_summary": fit_summary,
        "holdout_summary": holdout_summary, "fit_evidence": fit_evidence,
        "holdout_evidence": holdout_evidence,
        "fit_selected_template_eligible": fit_eligible,
        "predictions": {"pred_a_selection_complete": fit_eligible,
                        "pred_b_holdout_capable": capable,
                        "pred_c_no_eligible_template": not fit_eligible},
        "maximum_price": plan["price"], "active_price": actual_price,
        "circuit_interventions": 0,
        "scientific_scope": "dataset_capability_only_not_carrier_evidence",
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": result["terminal"], "selected_template": selected,
                      "result_path": OUT.relative_to(ROOT).as_posix(),
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
