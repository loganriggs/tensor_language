#!/usr/bin/env python3
"""Licensed Task14 L11H3 subject-term versus all-other-sources factorial."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_interaction_repairs_p2s pred_c_complement_independently_carries_task pred_d_complement_asymmetry_persists

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Mapping, Sequence

import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_fresh_matched_natural_native_capability as capability
import run_task14_head11_3_subject_attractor_score_payload_factorial as factors


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_fresh_matched_subject_term_vs_complement_factorial_v1.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_term_vs_complement_factorial_v1_result.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_matched_subject_term_vs_complement_factorial_v1_capability_license.json"
PRIOR_ART_SHA256 = "ea9385643db35efc29e006b79b661f24ea9acf6912190c29d0fad480d857c84c"
LICENSE_SHA256 = "9ac6596d92a0b7e75a65a31edebcb480676fa81505cd211028babc56ca1ecf18"
CANDIDATE_ID = "subject_verb.number_agreement.head11_3_fresh_matched_subject_term_vs_complement_factorial_v1"
CONDITIONS = ("native_neither", "opposite_subject_only",
              "opposite_complement_only", "complete_opposite_head")
SELF_POSITION = 8
BARS = {
    "maximum_native_replay_absolute_logit_error": 7e-5,
    "maximum_source_term_sum_absolute_error": 5e-5,
    "maximum_same_batch_native_noop_endpoint_error": 7e-5,
    "maximum_installed_head_absolute_error": 5e-5,
    "maximum_complete_head_vector_absolute_error": 5e-5,
    "minimum_complete_head_mean_donor_margin_improvement": .05,
    "minimum_complete_head_mean_donor_CE_improvement": 0.0,
    "minimum_complete_head_row_improvement_fraction": .75,
    "minimum_live_mean_donor_margin_magnitude": .05,
    "minimum_directional_row_fraction": .75,
    "minimum_positive_mean_donor_CE_improvement": 0.0,
    "minimum_complement_recovery_of_complete_margin": .70,
    "minimum_interaction_repair_of_subject_harm": 1.0,
}


class ComplementFactorialError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    rows = [row for row in capability.authority.build_rows() if row["phase"] == "HOLDOUT"]
    if len(rows) != 16 or {row["group_number"] for row in rows} != set(range(8, 16)):
        raise ComplementFactorialError("runner must use exact licensed HOLDOUT")
    return rows


def validate_preflight():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise ComplementFactorialError("prior-art receipt changed")
    return licensing.validate_causal_preflight(
        capability.build_gate(), capability.CAPABILITY_RESULT, LICENSE,
        expected_license_sha256=LICENSE_SHA256, causal_candidate_id=CANDIDATE_ID)


def compile_plan():
    license_value = validate_preflight()
    return {
        "schema": "task14_head11_3_fresh_matched_subject_term_vs_complement_factorial_plan_v1",
        "candidate_id": CANDIDATE_ID, "split": "LICENSED_HOLDOUT",
        "row_count": len(build_rows()), "conditions": list(CONDITIONS),
        "prior_art_sha256": PRIOR_ART_SHA256, "license_sha256": LICENSE_SHA256,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "preflight_validated": True, "bars": dict(BARS),
        "partition": "subject p_8*u_8 versus sum of all p_j*u_j for j != 8",
        "price": {"model_forwards": 4, "example_evaluations": 192,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_margin_improvement", "donor_full_vocab_CE_improvement"],
        "closed_claims": ["individual_q_or_k", "individual_complement_source_semantics",
                          "necessity", "syntax_generality", "FIT", "rank"],
    }


def _role_batch(rows, torch, device):
    roles = ("recipient", "opposite_same_lemma")
    tokens = torch.cat([torch.tensor([row["endpoints"][role]["ids"] for row in rows],
                                    dtype=torch.long, device=device) for role in roles])
    finals = torch.full((len(tokens),), SELF_POSITION, dtype=torch.long, device=device)
    return tokens, finals


def _compile(recipient_tokens, recipient, opposite, rows, torch):
    recipient_terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    opposite_terms = opposite["p"].unsqueeze(-1) * opposite["u"]
    sr, so = recipient_terms[:, 8], opposite_terms[:, 8]
    source_mask = torch.arange(recipient_terms.shape[1], device=recipient_terms.device) != 8
    cr = recipient_terms[:, source_mask].sum(1)
    co = opposite_terms[:, source_mask].sum(1)
    heads_by_condition = {
        "native_neither": sr+cr,
        "opposite_subject_only": so+cr,
        "opposite_complement_only": sr+co,
        "complete_opposite_head": so+co,
    }
    indices, heads, specs, reinstall = [], [], [], []
    for row_index, row in enumerate(rows):
        for condition in CONDITIONS:
            indices.append(row_index); heads.append(heads_by_condition[condition][row_index])
            specs.append((row_index, condition, f"{row['direction_id']}__{row['template_id']}"))
            reinstall.append(condition == "native_neither")
    index = torch.tensor(indices, dtype=torch.long, device=recipient_tokens.device)
    return {
        "tokens": recipient_tokens[index], "finals": torch.full_like(index, 8),
        "replacement_heads": torch.stack(heads),
        "native_reinstall_mask": torch.tensor(reinstall, dtype=torch.bool,
                                               device=recipient_tokens.device),
        "specs": specs, "sr": sr, "so": so, "cr": cr, "co": co,
    }


def _metrics(logits, row, torch):
    recipient = int(row["endpoints"]["recipient"]["answer_id"])
    donor = int(row["endpoints"]["opposite_same_lemma"]["answer_id"])
    lp = torch.log_softmax(logits, dim=-1)
    return {"donor_margin": float(logits[donor]-logits[recipient]),
            "donor_CE": float(-lp[donor])}


def evaluate(model, torch, F, facade):
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = _role_batch(rows, torch, device)
    native = factors._native_logits(model, tokens, torch, F)
    replay, captured = factors._factor_forward(model, tokens, finals, torch, F, facade)
    recipient = {k: v[:n] for k, v in captured.items()}
    opposite = {k: v[n:] for k, v in captured.items()}
    patch = _compile(tokens[:n], recipient, opposite, rows, torch)
    native_patch = factors._native_logits(model, patch["tokens"], torch, F)
    patched, _ = factors._factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])
    exactness = {
        "native_replay_max_absolute_logit_error": float((native-replay).abs().max()),
        "source_term_sum_max_absolute_error": max(float((
            torch.einsum("bk,bkd->bd", side["p"], side["u"])-side["head"]).abs().max())
            for side in (recipient, opposite)),
        "same_batch_native_noop_endpoint_max_absolute_error": 0.0,
        "installed_head_max_absolute_error": 0.0,
        "complete_head_vector_max_absolute_error": float((patch["so"]+patch["co"]
                                                           - opposite["head"]).abs().max()),
    }
    evidence = []
    expected = {
        "native_neither": patch["sr"]+patch["cr"],
        "opposite_subject_only": patch["so"]+patch["cr"],
        "opposite_complement_only": patch["sr"]+patch["co"],
        "complete_opposite_head": patch["so"]+patch["co"],
    }
    for out_index, (row_index, condition, cell_id) in enumerate(patch["specs"]):
        base = _metrics(native_patch[out_index, 8], rows[row_index], torch)
        value = _metrics(patched[out_index, 8], rows[row_index], torch)
        evidence.append({
            "row_id": rows[row_index]["row_id"], "cell_id": cell_id,
            "condition": condition,
            "donor_margin_improvement": value["donor_margin"]-base["donor_margin"],
            "donor_CE_improvement": base["donor_CE"]-value["donor_CE"],
        })
        exactness["installed_head_max_absolute_error"] = max(
            exactness["installed_head_max_absolute_error"],
            float((patch["replacement_heads"][out_index]-expected[condition][row_index]).abs().max()))
        if condition == "native_neither":
            exactness["same_batch_native_noop_endpoint_max_absolute_error"] = max(
                exactness["same_batch_native_noop_endpoint_max_absolute_error"],
                float((patched[out_index]-native_patch[out_index]).abs().max()))
    return evidence, exactness


def _fraction(values, positive=True):
    return sum((v > 0) if positive else (v < 0) for v in values)/len(values)


def score(evidence: Sequence[Mapping[str, object]], exactness: Mapping[str, float], bars=BARS):
    expected = {(row["row_id"], f"{row['direction_id']}__{row['template_id']}", c)
                for row in build_rows() for c in CONDITIONS}
    observed = [(x.get("row_id"), x.get("cell_id"), x.get("condition")) for x in evidence]
    if len(observed) != len(expected) or set(observed) != expected or len(set(observed)) != len(expected):
        raise ComplementFactorialError("evidence does not cover exact licensed factorial")
    if any(type(x.get(k)) not in (int, float) or not math.isfinite(float(x[k]))
           for x in evidence for k in ("donor_margin_improvement", "donor_CE_improvement")):
        raise ComplementFactorialError("non-finite or missing task metric")
    grouped = defaultdict(dict)
    for x in evidence:
        grouped[x["cell_id"]].setdefault(x["condition"], []).append(x)
    cells = {}
    for cell_id, conditions in sorted(grouped.items()):
        def vals(condition, metric): return [float(x[metric]) for x in conditions[condition]]
        sm, sc = vals("opposite_subject_only", "donor_margin_improvement"), vals("opposite_subject_only", "donor_CE_improvement")
        cm, cc = vals("opposite_complement_only", "donor_margin_improvement"), vals("opposite_complement_only", "donor_CE_improvement")
        fm, fc = vals("complete_opposite_head", "donor_margin_improvement"), vals("complete_opposite_head", "donor_CE_improvement")
        im = [f-c-s for f, c, s in zip(fm, cm, sm)]
        ic = [f-c-s for f, c, s in zip(fc, cc, sc)]
        cells[cell_id] = {
            "subject_only": {"mean_margin": statistics.fmean(sm), "mean_CE": statistics.fmean(sc), "margin_values": sm, "CE_values": sc},
            "complement_only": {"mean_margin": statistics.fmean(cm), "mean_CE": statistics.fmean(cc), "margin_values": cm, "CE_values": cc},
            "interaction": {"mean_margin": statistics.fmean(im), "mean_CE": statistics.fmean(ic), "margin_values": im, "CE_values": ic},
            "complete_head": {"mean_margin": statistics.fmean(fm), "mean_CE": statistics.fmean(fc), "margin_values": fm, "CE_values": fc},
        }
    exact_live = (
        exactness["native_replay_max_absolute_logit_error"] <= bars["maximum_native_replay_absolute_logit_error"] and
        exactness["source_term_sum_max_absolute_error"] <= bars["maximum_source_term_sum_absolute_error"] and
        exactness["same_batch_native_noop_endpoint_max_absolute_error"] <= bars["maximum_same_batch_native_noop_endpoint_error"] and
        exactness["installed_head_max_absolute_error"] <= bars["maximum_installed_head_absolute_error"] and
        exactness["complete_head_vector_max_absolute_error"] <= bars["maximum_complete_head_vector_absolute_error"])
    complete_live = all(c["complete_head"]["mean_margin"] >= bars["minimum_complete_head_mean_donor_margin_improvement"]
        and c["complete_head"]["mean_CE"] >= bars["minimum_complete_head_mean_donor_CE_improvement"]
        and _fraction(c["complete_head"]["margin_values"]) >= bars["minimum_complete_head_row_improvement_fraction"]
        and _fraction(c["complete_head"]["CE_values"]) >= bars["minimum_complete_head_row_improvement_fraction"] for c in cells.values())
    instrument = exact_live and complete_live
    repair = instrument and all(
        c["subject_only"]["mean_margin"] <= -bars["minimum_live_mean_donor_margin_magnitude"]
        and c["subject_only"]["mean_CE"] <= 0
        and _fraction(c["subject_only"]["margin_values"], False) >= bars["minimum_directional_row_fraction"]
        and _fraction(c["subject_only"]["CE_values"], False) >= bars["minimum_directional_row_fraction"]
        and c["interaction"]["mean_margin"] >= abs(c["subject_only"]["mean_margin"])*bars["minimum_interaction_repair_of_subject_harm"]
        and c["interaction"]["mean_CE"] >= bars["minimum_positive_mean_donor_CE_improvement"]
        and _fraction(c["interaction"]["margin_values"]) >= bars["minimum_directional_row_fraction"]
        and _fraction(c["interaction"]["CE_values"]) >= bars["minimum_directional_row_fraction"]
        for cell, c in cells.items() if cell.startswith("plural_to_singular"))
    independent = instrument and all(
        c["complement_only"]["mean_margin"] >= bars["minimum_live_mean_donor_margin_magnitude"]
        and c["complement_only"]["mean_CE"] >= bars["minimum_positive_mean_donor_CE_improvement"]
        and _fraction(c["complement_only"]["margin_values"]) >= bars["minimum_directional_row_fraction"]
        and _fraction(c["complement_only"]["CE_values"]) >= bars["minimum_directional_row_fraction"]
        and c["complement_only"]["mean_margin"]/c["complete_head"]["mean_margin"] >= bars["minimum_complement_recovery_of_complete_margin"]
        for c in cells.values())
    asymmetric = instrument and all(
        (c["complement_only"]["mean_margin"] >= bars["minimum_live_mean_donor_margin_magnitude"]
         and c["complement_only"]["mean_CE"] >= 0
         and _fraction(c["complement_only"]["margin_values"]) >= bars["minimum_directional_row_fraction"]
         and _fraction(c["complement_only"]["CE_values"]) >= bars["minimum_directional_row_fraction"])
        if cell.startswith("singular_to_plural") else
        (c["complement_only"]["mean_margin"] <= -bars["minimum_live_mean_donor_margin_magnitude"]
         and c["complement_only"]["mean_CE"] <= 0
         and _fraction(c["complement_only"]["margin_values"], False) >= bars["minimum_directional_row_fraction"]
         and _fraction(c["complement_only"]["CE_values"], False) >= bars["minimum_directional_row_fraction"])
        for cell, c in cells.items())
    return {**exactness, "cells": cells, "predictions": {
        "pred_a_instrument_live": bool(instrument),
        "pred_b_interaction_repairs_p2s": bool(repair),
        "pred_c_complement_independently_carries_task": bool(independent),
        "pred_d_complement_asymmetry_persists": bool(asymmetric)}}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists(): raise ComplementFactorialError(f"refusing to overwrite {OUT}")
    torch, F, facade = factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad(): evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    terminal = "valid_causal_screen" if scored["predictions"]["pred_a_instrument_live"] else "invalid"
    result = {"schema": "task14_head11_3_fresh_matched_subject_term_vs_complement_factorial_result_v1",
              "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
              "checkpoint_weights_sha256": checkpoint.weights_sha256, "score": scored,
              "evidence": evidence, "evaluated_splits": ["LICENSED_HOLDOUT"],
              "forbidden_splits_opened": [], "model_forwards": 4,
              "causal_interventions": len(evidence)}
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__": main()
