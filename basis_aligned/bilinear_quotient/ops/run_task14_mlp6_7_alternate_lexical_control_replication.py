#!/usr/bin/env python3
"""Alternate lexical-donor stability test for the Task14 MLP6/7 split."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_lexical_entanglement_replicates pred_c_control_family_sensitive pred_d_mlp7_completion_cancels pred_e_reciprocal_mixed_corner_quiet

from __future__ import annotations

from collections import defaultdict
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import circuit_fast_screen_managed_runner as managed
import run_task14_head11_3_fresh_matched_subject_mlp8_mlp6_7_source_factorial as parent


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_alternate_lexical_control_replication_v1.json"
PARENT_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_mlp6_7_source_factorial_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_alternate_lexical_control_replication_v1_result.json"
PRIOR_ART_SHA256 = "7a47cd47f5104565d5281eb5bdbc371cb25f71dc8e71d88e113e4e5094e8e483"
PARENT_RESULT_SHA256 = "eff2b9e7ab76b4335733e8bde6708435a18e6b0da14073373e211d9588994efa"
CANDIDATE_ID = "subject_verb.number_agreement.mlp6_7_alternate_lexical_control_replication_v1"
SUBSETS = ("EAUWY", "EAUWZ", "EAUWYZ")
CONDITIONS = ("recipient",) + tuple(f"lexical_{subset}_full" for subset in SUBSETS)
PRIMARY_CELL = "plural_to_singular__between_below"
ALTERNATE = {
    "rider": ("leader", 3554),
    "sailor": ("guard", 4860),
    "singer": ("judge", 10266),
    "writer": ("owner", 4870),
    "owners": ("sailors", 29996),
    "leaders": ("singers", 39113),
    "guards": ("writers", 8786),
    "judges": ("riders", 13750),
}
BARS = {
    "maximum_numerical_absolute_error": 5e-5,
    "minimum_native_lexical_answer_margin": 0.0,
    "lexical_ratio_boundary": 0.25,
    "maximum_completion_residual_fraction": 0.60,
}


class AlternateLexicalControlError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    rows = copy.deepcopy(parent.build_rows())
    for row in rows:
        recipient_subject = row["endpoints"]["recipient"]["subject"]
        if recipient_subject not in ALTERNATE:
            raise AlternateLexicalControlError("alternate donor rule is incomplete")
        subject, token = ALTERNATE[recipient_subject]
        endpoint = row["endpoints"]["same_number_different_lemma"]
        endpoint["subject"] = subject
        endpoint["ids"][-1] = token
        endpoint["text"] = endpoint["text"].rsplit(" ", 1)[0] + " " + subject
        row["alternate_lexical_control"] = True
    return rows


def validate_preflight():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise AlternateLexicalControlError("prior-art receipt changed")
    if _sha256(PARENT_RESULT) != PARENT_RESULT_SHA256:
        raise AlternateLexicalControlError("parent result changed")
    result = json.loads(PARENT_RESULT.read_text())
    predictions = result.get("score", {}).get("predictions", {})
    if result.get("terminal") != "valid_causal_screen" \
            or predictions.get("pred_a_instrument_and_parent_closure") is not True \
            or predictions.get("pred_g_number_specific") is not False:
        raise AlternateLexicalControlError("parent no longer licenses control replication")


def compile_plan():
    validate_preflight()
    return {
        "schema": "task14_mlp6_7_alternate_lexical_control_replication_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "row_count": 16,
        "conditions": list(CONDITIONS),
        "primary_cell": PRIMARY_CELL,
        "alternate_donor_rule": {key: value[0] for key, value in ALTERNATE.items()},
        "bars": dict(BARS),
        "price": {"physical_model_forwards": 4, "example_evaluations": 224,
                  "causal_interventions": 48, "backwards": 0,
                  "parameter_updates": 0},
        "closed_claims": ["new_independent_text", "semantic_uniqueness", "rank",
                          "compression", "sufficiency_outside_fixed_L11H3_interface"],
    }


def _role_slice(values, start, stop):
    return {key: value[start:stop] for key, value in values.items()}


def _compile_patch(tokens, heads, rows, torch):
    indices, replacements, masks, specs = [], [], [], []
    for row_index, row in enumerate(rows):
        for condition in CONDITIONS:
            indices.append(row_index)
            replacements.append(heads[condition][row_index])
            masks.append(condition == "recipient")
            specs.append((row_index, condition,
                          f"{row['direction_id']}__{row['template_id']}"))
    index = torch.tensor(indices, dtype=torch.long, device=tokens.device)
    return {
        "tokens": tokens[index],
        "finals": torch.full_like(index, parent.SUBJECT_POSITION),
        "replacement_heads": torch.stack(replacements),
        "native_reinstall_mask": torch.tensor(masks, dtype=torch.bool,
                                                device=tokens.device),
        "specs": specs,
    }


def evaluate(model, torch, F, facade):
    rows = build_rows()
    n = len(rows)
    device = next(model.parameters()).device
    tokens, finals = parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    native_roles = parent.factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, role_closure, mlp8_all = parent._decomposed_forward(
        model, tokens, finals, torch, F, facade)
    recipient = _role_slice(captured, 0, n)
    opposite = _role_slice(captured, n, 2 * n)
    lexical = _role_slice(captured, 2 * n, 3 * n)
    input_recipient = _role_slice(mlp8_all, 0, n)
    input_lexical = _role_slice(mlp8_all, 2 * n, 3 * n)
    mlp = model.transformer.h[parent.MLP_LAYER].mlp
    attention = model.transformer.h[parent.LAYER].attn

    heads = {"recipient": parent.grandparent._head_from_slot(
        recipient, opposite, recipient["M8"], attention, projection, torch, F)}
    exactness = {
        "native_role_replay_max_absolute_logit_error": float(
            (native_roles - replay).abs().max()),
        "role_input_state_closure_max_absolute_error": role_closure[
            "input_state_closure_max_absolute_error"],
        "role_input_normalized_closure_max_absolute_error": role_closure[
            "input_normalized_closure_max_absolute_error"],
        "full_source_input_max_absolute_error": 0.0,
        "full_source_output_max_absolute_error": 0.0,
        "full_source_propagated_slot_max_absolute_error": 0.0,
        "full_source_installed_head_max_absolute_error": 0.0,
        "recipient_noop_full_logit_max_absolute_error": 0.0,
    }
    for subset in SUBSETS:
        hybrid, _, endpoint_error = parent._hybrid_input(
            input_recipient, input_lexical, subset, F)
        products, _ = parent.polarized_v2._polarized_products(
            mlp, input_recipient["input"], hybrid, torch, F)
        slots, outputs, _ = parent.polarized_v2._propagated_slots(
            model, mlp, products, recipient["M8"], F,
            native_recipient_output=input_recipient["output"],
            native_source_output=input_lexical["output"] if subset == "EAUWYZ" else None)
        condition = f"lexical_{subset}_full"
        heads[condition] = parent.grandparent._head_from_slot(
            recipient, opposite, slots["full"], attention, projection, torch, F)
        if subset == "EAUWYZ":
            exactness["full_source_input_max_absolute_error"] = endpoint_error
            exactness["full_source_output_max_absolute_error"] = float(
                (outputs["full"] - input_lexical["output"]).abs().max())
            exactness["full_source_propagated_slot_max_absolute_error"] = float(
                (slots["full"][:, parent.SUBJECT_POSITION]
                 - lexical["M8"][:, parent.SUBJECT_POSITION]).abs().max())
            expected_head = parent.grandparent._head_from_slot(
                recipient, opposite, lexical["M8"], attention, projection, torch, F)
            exactness["full_source_installed_head_max_absolute_error"] = float(
                (heads[condition] - expected_head).abs().max())

    patch = _compile_patch(tokens[:n], heads, rows, torch)
    native_patch = parent.factors._native_logits(model, patch["tokens"], torch, F)
    patched, _, _, patch_closure = parent.downstream._decomposed_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])
    exactness["downstream_state_closure_max_absolute_error"] = patch_closure[
        "state_sum_max_absolute_error"]
    exactness["downstream_normalized_closure_max_absolute_error"] = patch_closure[
        "normalized_state_max_absolute_error"]
    mask = patch["native_reinstall_mask"]
    exactness["recipient_noop_full_logit_max_absolute_error"] = float(
        (patched[mask] - native_patch[mask]).abs().max())

    native_lexical_margins = []
    for row_index, row in enumerate(rows):
        metrics = parent.grandparent._both_metrics(
            native_roles[2 * n + row_index, parent.SUBJECT_POSITION], row, torch)
        native_lexical_margins.append(float(metrics["lexical_target_margin"]))

    evidence = []
    for out_index, (row_index, condition, cell_id) in enumerate(patch["specs"]):
        base = parent.grandparent._both_metrics(
            native_patch[out_index, parent.SUBJECT_POSITION], rows[row_index], torch)
        value = parent.grandparent._both_metrics(
            patched[out_index, parent.SUBJECT_POSITION], rows[row_index], torch)
        evidence.append({
            "row_id": rows[row_index]["row_id"],
            "cell_id": cell_id,
            "condition": condition,
            "alternate_lexical_subject": rows[row_index]["endpoints"][
                "same_number_different_lemma"]["subject"],
            "lexical_target_margin_improvement":
                value["lexical_target_margin"] - base["lexical_target_margin"],
            "lexical_target_CE_improvement":
                base["lexical_target_CE"] - value["lexical_target_CE"],
        })
    return evidence, exactness, native_lexical_margins


def score(evidence, exactness, native_lexical_margins, bars=BARS):
    if len(evidence) != 64 or len({(item["row_id"], item["condition"])
                                  for item in evidence}) != 64:
        raise AlternateLexicalControlError("evidence does not cover the frozen 64 rows")
    if any(not math.isfinite(float(item[key])) for item in evidence
           for key in ("lexical_target_margin_improvement",
                       "lexical_target_CE_improvement")):
        raise AlternateLexicalControlError("non-finite outcome")
    grouped = defaultdict(lambda: defaultdict(list))
    for item in evidence:
        grouped[item["cell_id"]][item["condition"]].append(item)
    effects = {}
    for cell_id, conditions in grouped.items():
        effects[cell_id] = {}
        for condition, items in conditions.items():
            effects[cell_id][condition] = {
                metric: statistics.fmean(float(item[metric]) for item in items)
                for metric in ("lexical_target_margin_improvement",
                               "lexical_target_CE_improvement")}
    parent_score = json.loads(PARENT_RESULT.read_text())["score"]
    frozen_opposite = parent_score["cells"][PRIMARY_CELL]["opposite"]["full"][
        "margin"]["effects"]["EAUWYZ"]
    primary = effects[PRIMARY_CELL]
    mixed_y = primary["lexical_EAUWY_full"]["lexical_target_margin_improvement"]
    mixed_z = primary["lexical_EAUWZ_full"]["lexical_target_margin_improvement"]
    completed = primary["lexical_EAUWYZ_full"]["lexical_target_margin_improvement"]
    ratio_y = abs(mixed_y) / max(abs(frozen_opposite), 1e-12)
    ratio_z = abs(mixed_z) / max(abs(frozen_opposite), 1e-12)
    completion_fraction = abs(completed) / max(abs(mixed_y), 1e-12)
    exact_live = all(value <= bars["maximum_numerical_absolute_error"]
                     for value in exactness.values())
    native_live = min(native_lexical_margins) > bars[
        "minimum_native_lexical_answer_margin"]
    instrument = exact_live and native_live
    replicate = instrument and ratio_y >= bars["lexical_ratio_boundary"]
    sensitive = instrument and ratio_y < bars["lexical_ratio_boundary"]
    completion_change = completed - mixed_y
    cancellation = replicate \
        and completion_fraction <= bars["maximum_completion_residual_fraction"] \
        and completion_change * mixed_y < 0
    reciprocal_quiet = instrument and ratio_z < bars["lexical_ratio_boundary"]
    return {
        **exactness,
        "minimum_native_lexical_answer_margin": min(native_lexical_margins),
        "cell_effects": effects,
        "frozen_parent_opposite_full_margin_effect": frozen_opposite,
        "primary_EAUWY_lexical_ratio": ratio_y,
        "primary_EAUWZ_lexical_ratio": ratio_z,
        "primary_completion_residual_fraction": completion_fraction,
        "predictions": {
            "pred_a_instrument_live": bool(instrument),
            "pred_b_lexical_entanglement_replicates": bool(replicate),
            "pred_c_control_family_sensitive": bool(sensitive),
            "pred_d_mlp7_completion_cancels": bool(cancellation),
            "pred_e_reciprocal_mixed_corner_quiet": bool(reciprocal_quiet),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise AlternateLexicalControlError(f"refusing to overwrite {OUT}")
    torch, F, facade = parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness, native_margins = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, native_margins)
    terminal = "valid_causal_screen" if scored["predictions"][
        "pred_a_instrument_live"] else "invalid"
    result = {
        "schema": "task14_mlp6_7_alternate_lexical_control_replication_result_v1",
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": scored,
        "evidence": evidence,
        "evaluated_splits": ["LICENSED_HOLDOUT_REUSED_TEXT_ALTERNATE_CONTROL"],
        "forbidden_splits_opened": [],
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()},
                     sort_keys=True))


if __name__ == "__main__":
    main()
