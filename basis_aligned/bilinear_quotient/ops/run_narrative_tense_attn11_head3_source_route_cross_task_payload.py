#!/usr/bin/env python3
"""Exact source-route and cross-task payload screen for narrative-tense L11H3."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_self_route pred_c_tense_cue_route pred_d_mixed_or_remaining_route pred_e_cross_task_semantic_reuse pred_f_generic_output_token_confound

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Mapping, Sequence

import circuit_fast_screen_candidate_narrative_tense as narrative
import circuit_fast_screen_managed_runner as managed
import run_task14_head11_3_ood_fronted_score_role_factorial as task14
import run_task14_head11_3_subject_attractor_score_payload_factorial as factor


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/narrative_tense_attn11_head3_source_route_cross_task_payload_v1.json"
PARENT = ROOT / "circuits/fast_screens/narrative_tense_attn11_head3_complement_factorial_v1_result.json"
NATIVE_PARENT = ROOT / "circuits/fast_screens/narrative_tense_past_vs_present_v2_result.json"
TASK14_VALUE_PARENT = ROOT / "circuits/fast_screens/task14_head11_3_ood_same_syntax_source_value_atlas_v2_result.json"
TASK14_SCORE_PARENT = ROOT / "circuits/fast_screens/task14_head11_3_ood_fronted_score_role_factorial_v2_result.json"
OUT = ROOT / "circuits/fast_screens/narrative_tense_attn11_head3_source_route_cross_task_payload_v1_result.json"

EXPECTED_SHA256 = {
    PRIOR_ART: "18f97382fa020d7da7b0ab35d0c52537d500296cd3c5364cbb9ba68e4e664345",
    PARENT: "7f31c50639ef20dd35f5bdaa5dcb9024221025db8856043baf104074f5b3c32b",
    NATIVE_PARENT: "5466980e1aa0a59538e4e8fcfb29457814c01e91cbe39bf41a2d42140fc7e71a",
    TASK14_VALUE_PARENT: "e0cdff4a7e71713d3ae8ab58dda72de83ee58930516d13faafbdc279a405ed66",
    TASK14_SCORE_PARENT: "39269428f6948ceae892721c7f79906e1223b97c6eb0033de0c95be545ac1f8e",
}
NARRATIVE_AUTHORITY_SHA256 = "745910973b77cfec0dd945920f72b2c19c06a6e51464df11bd969165233d4a77"
TASK14_AUTHORITY_SHA256 = "c321eead9b4832e42806015f9248b6dbb418acfc78b9a38d6638c2d4a62d1eda"
PAIRING_SHA256 = "fcd7ec77ca5df09e1e0a13274f34b862e280932b1d5ce6457bf02e0e28e87bd1"
LAYER = 11
HEAD = 3
BATCH = 32
TARGET_FAMILIES = ("A1", "A2")
CONTROL_FAMILIES = ("P", "C")
WITHIN_ARMS = (
    "native_noop", "complete_head",
    "S_score", "S_value", "S_joint",
    "T_score", "T_value", "T_joint", "R_joint",
)
CROSS_ARMS = ("task14_singular_value", "task14_plural_value")
ANSWER_IDS = {"is": 318, "was": 373, "are": 389}

BARS = {
    "minimum_native_accuracy_each_A1_A2_side_cell": .85,
    "maximum_native_replay_source_sum_and_install_error": 5e-5,
    "minimum_complete_H3_donorward_fraction_each_target_cell": .75,
    "route_minimum_fraction_of_complete_H3_margin_and_CE_each_target_cell": .50,
    "route_minimum_row_direction_fraction_each_target_cell": .75,
    "route_minimum_advantage_over_competing_joint_route": .15,
    "factor_minimum_fraction_of_its_route_joint_margin_and_CE": .70,
    "factor_minimum_row_direction_fraction": .75,
    "cross_task_matched_minimum_direction_fraction_each_state": .75,
    "cross_task_matched_minimum_fraction_of_within_narrative_S_value": .50,
    "cross_task_unmatched_minimum_recipientward_fraction_each_state": .75,
}


class ScreenError(ValueError):
    """The frozen closure or exact intervention contract was violated."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_closure() -> None:
    for path, expected in EXPECTED_SHA256.items():
        if _sha256(path) != expected:
            raise ScreenError(f"frozen artifact changed: {path}")
    parent = json.loads(PARENT.read_text())
    if parent.get("terminal") != "shared_copular_service" \
            or not parent.get("predictions", {}).get("pred_a_instrument_live"):
        raise ScreenError("parent H3 screen is not the frozen live result")


def build_rows() -> list[dict[str, object]]:
    rows = narrative.build_rows()
    if narrative.validate_rows(rows) != NARRATIVE_AUTHORITY_SHA256 or len(rows) != 128:
        raise ScreenError("narrative authority changed")
    for row in rows:
        base = list(row["base_ids"])
        donor = list(row["donor_ids"])
        if len(base) != len(donor):
            raise ScreenError("narrative pair lengths differ")
        self_position = len(base) - 1
        changed = tuple(i for i, pair in enumerate(zip(base, donor)) if pair[0] != pair[1])
        expected = {"A1": (0, 4), "A2": (3, 5, 7)}.get(str(row["transform_id"]))
        if expected is not None and changed != expected:
            raise ScreenError(f"target tense positions changed: {row['row_id']}")
        if str(row["transform_id"]) in CONTROL_FAMILIES and not changed:
            raise ScreenError("control must contain its registered lexical rewrite")
        if self_position in changed:
            raise ScreenError("final self token differs inside a paired narrative row")
        row["S_positions"] = (self_position,)
        row["T_positions"] = changed
        row["R_positions"] = tuple(i for i in range(len(base))
                                   if i != self_position and i not in changed)
    return rows


def build_pairing(rows: Sequence[Mapping[str, object]] | None = None) -> list[dict[str, object]]:
    narrative_rows = list(rows) if rows is not None else build_rows()
    fronted = sorted(task14.build_rows(), key=lambda row: str(row["group_id"]))
    if task14.compile_plan()["authority_sha256"] != TASK14_AUTHORITY_SHA256 or len(fronted) != 32:
        raise ScreenError("Task14 fronted authority changed")
    records = []
    for row in narrative_rows:
        if str(row["transform_id"]) not in TARGET_FAMILIES:
            continue
        group_number = int(row["group_number"])
        source = fronted[group_number]
        sides = {}
        for side in ("base", "donor"):
            state = str(source[f"{side}_subject_number"])
            if state in sides or state not in {"singular", "plural"}:
                raise ScreenError("Task14 row does not have one endpoint per number state")
            sides[state] = side
        if set(sides) != {"singular", "plural"}:
            raise ScreenError("Task14 row lost singular/plural endpoints")
        matched = "singular" if str(row["donor_answer"]) == " is" else "plural"
        records.append({
            "narrative_row_id": str(row["row_id"]),
            "narrative_group_number": group_number,
            "narrative_family": str(row["transform_id"]),
            "narrative_direction": str(row["direction_id"]),
            "task14_group_id": str(source["group_id"]),
            "singular_side": sides["singular"],
            "plural_side": sides["plural"],
            "matched_task14_state": matched,
            "unmatched_task14_state": "plural" if matched == "singular" else "singular",
        })
    if len(records) != 64 or _canonical(records) != PAIRING_SHA256:
        raise ScreenError("cross-task pairing changed")
    return records


def compile_plan() -> dict[str, object]:
    _validate_closure()
    rows = build_rows()
    pairing = build_pairing(rows)
    return {
        "schema": "narrative_tense_attn11_head3_source_route_cross_task_payload_plan_v1",
        "candidate_id": "narrative_tense.attn11_head3_source_route_and_cross_task_payload_v1",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_SHA256[PRIOR_ART],
        "narrative_authority_sha256": NARRATIVE_AUTHORITY_SHA256,
        "task14_authority_sha256": TASK14_AUTHORITY_SHA256,
        "pairing_sha256": PAIRING_SHA256,
        "row_count": len(rows),
        "target_row_count": len(pairing),
        "layer": LAYER,
        "head": HEAD,
        "source_partition": {
            "S": "final query/self source",
            "T": "all recipient/donor token-ID differences",
            "R": "all remaining causally available sources",
        },
        "arms": list(WITHIN_ARMS + CROSS_ARMS),
        "bars": dict(BARS),
        "price": {"model_forwards": 14, "example_evaluations": 1600,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_directed_was_is_margin", "full_vocabulary_donor_CE",
                     "cross_task_was_is_are_logit_changes", "P_C_normalized_movement"],
        "interpretation_limit": (
            "Failure rejects only this deterministic cross-task alignment. Same-is-only "
            "success is an output-token confound, not shared semantic-state evidence."
        ),
    }


def _pad_endpoints(endpoints, length, torch, device):
    tokens = torch.full((len(endpoints), length), 50256, dtype=torch.long, device=device)
    finals = []
    for index, endpoint in enumerate(endpoints):
        ids = endpoint["ids"]
        tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        finals.append(len(ids) - 1)
    return tokens, torch.tensor(finals, dtype=torch.long, device=device)


def _group_head(base, donor, positions, mode, torch):
    indices = torch.tensor(tuple(positions), dtype=torch.long, device=base["p"].device)
    if indices.numel() == 0:
        return base["head"].clone()
    native = torch.einsum("bk,bkd->bd", base["p"][:, indices], base["u"][:, indices])
    if mode == "score":
        replacement = torch.einsum("bk,bkd->bd", donor["p"][:, indices], base["u"][:, indices])
    elif mode == "value":
        replacement = torch.einsum("bk,bkd->bd", base["p"][:, indices], donor["u"][:, indices])
    elif mode == "joint":
        replacement = torch.einsum("bk,bkd->bd", donor["p"][:, indices], donor["u"][:, indices])
    else:
        raise ScreenError(f"unknown factor mode: {mode}")
    return base["head"] - native + replacement


def _metrics(logits, q, row, torch):
    donor = int(row["donor_answer_id"])
    recipient = int(row["base_answer_id"])
    margin = float(logits[q, donor] - logits[q, recipient])
    ce = float(-torch.log_softmax(logits[q], dim=-1)[donor])
    answer_margin = float(logits[q, int(row["base_answer_id"])]
                          - logits[q, int(row["base_foil_id"])])
    answer_ce = float(-torch.log_softmax(logits[q], dim=-1)[int(row["base_answer_id"])])
    triple = {name: float(logits[q, token]) for name, token in ANSWER_IDS.items()}
    return margin, ce, answer_margin, answer_ce, triple


def _mean(values):
    if not values:
        raise ScreenError("empty score cell")
    return statistics.fmean(values)


def score(evidence, capability, exactness, bars=BARS):
    target = [item for item in evidence if item["family"] in TARGET_FAMILIES]
    controls = [item for item in evidence if item["family"] in CONTROL_FAMILIES]
    cells = sorted({str(item["cell_id"]) for item in target})
    summaries = {}
    for cell in cells:
        summaries[cell] = {}
        for arm in WITHIN_ARMS + CROSS_ARMS:
            items = [x for x in target if x["cell_id"] == cell and x["arm"] == arm]
            if not items:
                continue
            summaries[cell][arm] = {
                "row_count": len(items),
                "mean_margin_delta": _mean([x["margin_delta"] for x in items]),
                "mean_full_vocab_CE_gain": _mean([x["donor_ce_gain"] for x in items]),
                "donorward_fraction": _mean([x["margin_delta"] > 0 for x in items]),
            }
    limit = bars["maximum_native_replay_source_sum_and_install_error"]
    exact_live = bool(exactness) and max(float(value) for value in exactness.values()) <= limit
    native_live = all(
        min(float(side) for side in capability[cell].values()) >=
        bars["minimum_native_accuracy_each_A1_A2_side_cell"] for cell in cells
    )
    complete_live = all(
        summaries[cell]["complete_head"]["mean_margin_delta"] > 0
        and summaries[cell]["complete_head"]["mean_full_vocab_CE_gain"] > 0
        and summaries[cell]["complete_head"]["donorward_fraction"] >=
            bars["minimum_complete_H3_donorward_fraction_each_target_cell"]
        for cell in cells
    )

    control_summary = {}
    controls_live = True
    for family in CONTROL_FAMILIES:
        complete = [abs(x["normalized_control_movement"]) for x in controls
                    if x["family"] == family and x["arm"] == "complete_head"]
        ceiling = _mean(complete)
        arm_values = {}
        for arm in WITHIN_ARMS:
            if arm == "complete_head":
                continue
            values = [abs(x["normalized_control_movement"]) for x in controls
                      if x["family"] == family and x["arm"] == arm]
            movement = _mean(values)
            passed = movement <= ceiling + 1e-12
            arm_values[arm] = {"mean_absolute_normalized_movement": movement,
                               "no_larger_than_complete_H3": passed}
            controls_live &= passed
        control_summary[family] = {"complete_H3_mean_absolute_normalized_movement": ceiling,
                                   "arms": arm_values}

    def route_report(route, competitor):
        output = {"cells": {}, "passed": True}
        for cell in cells:
            joint = summaries[cell][f"{route}_joint"]
            other = summaries[cell][f"{competitor}_joint"]
            complete = summaries[cell]["complete_head"]
            margin_fraction = joint["mean_margin_delta"] / complete["mean_margin_delta"]
            ce_fraction = joint["mean_full_vocab_CE_gain"] / complete["mean_full_vocab_CE_gain"]
            other_margin_fraction = other["mean_margin_delta"] / complete["mean_margin_delta"]
            other_ce_fraction = other["mean_full_vocab_CE_gain"] / complete["mean_full_vocab_CE_gain"]
            passed = (
                margin_fraction >= bars["route_minimum_fraction_of_complete_H3_margin_and_CE_each_target_cell"]
                and ce_fraction >= bars["route_minimum_fraction_of_complete_H3_margin_and_CE_each_target_cell"]
                and joint["donorward_fraction"] >= bars["route_minimum_row_direction_fraction_each_target_cell"]
                and margin_fraction - other_margin_fraction >= bars["route_minimum_advantage_over_competing_joint_route"]
                and ce_fraction - other_ce_fraction >= bars["route_minimum_advantage_over_competing_joint_route"]
            )
            output["cells"][cell] = {"margin_fraction_of_complete": margin_fraction,
                                      "CE_fraction_of_complete": ce_fraction,
                                      "margin_advantage": margin_fraction - other_margin_fraction,
                                      "CE_advantage": ce_fraction - other_ce_fraction,
                                      "donorward_fraction": joint["donorward_fraction"], "passed": passed}
            output["passed"] &= passed
        return output

    route_S = route_report("S", "T")
    route_T = route_report("T", "S")

    def factor_report(route):
        result = {}
        for mode in ("score", "value"):
            per_cell = {}
            passed_all = True
            for cell in cells:
                arm = summaries[cell][f"{route}_{mode}"]
                joint = summaries[cell][f"{route}_joint"]
                mf = arm["mean_margin_delta"] / joint["mean_margin_delta"] \
                    if abs(joint["mean_margin_delta"]) > 1e-12 else float("-inf")
                cf = arm["mean_full_vocab_CE_gain"] / joint["mean_full_vocab_CE_gain"] \
                    if abs(joint["mean_full_vocab_CE_gain"]) > 1e-12 else float("-inf")
                passed = mf >= bars["factor_minimum_fraction_of_its_route_joint_margin_and_CE"] \
                    and cf >= bars["factor_minimum_fraction_of_its_route_joint_margin_and_CE"] \
                    and arm["donorward_fraction"] >= bars["factor_minimum_row_direction_fraction"]
                per_cell[cell] = {"margin_fraction_of_joint": mf, "CE_fraction_of_joint": cf,
                                  "donorward_fraction": arm["donorward_fraction"], "passed": passed}
                passed_all &= passed
            result[mode] = {"cells": per_cell, "passed": passed_all}
        if result["score"]["passed"] and result["value"]["passed"]:
            result["classification"] = "score_and_value_redundant_or_interactive"
        elif result["score"]["passed"]:
            result["classification"] = "score_only"
        elif result["value"]["passed"]:
            result["classification"] = "value_only"
        else:
            result["classification"] = "score_value_composition_required"
        return result

    factor_readout = {"S": factor_report("S"), "T": factor_report("T")}

    cross = {}
    state_passes = {}
    for state, arm in (("is", "task14_singular_value"), ("was", "task14_plural_value")):
        items = [x for x in target if x["target_state"] == state and x["arm"] == arm]
        unmatched_arm = "task14_plural_value" if state == "is" else "task14_singular_value"
        unmatched = [x for x in target if x["target_state"] == state and x["arm"] == unmatched_arm]
        s_value = [x for x in target if x["target_state"] == state and x["arm"] == "S_value"]
        mm, mc = _mean([x["margin_delta"] for x in items]), _mean([x["donor_ce_gain"] for x in items])
        sm, sc = _mean([x["margin_delta"] for x in s_value]), _mean([x["donor_ce_gain"] for x in s_value])
        mf = mm / sm if sm > 1e-12 else float("-inf")
        cf = mc / sc if sc > 1e-12 else float("-inf")
        direction = _mean([x["margin_delta"] > 0 for x in items])
        unmatched_direction = _mean([x["margin_delta"] < 0 for x in unmatched])
        passed = direction >= bars["cross_task_matched_minimum_direction_fraction_each_state"] \
            and unmatched_direction >= bars["cross_task_unmatched_minimum_recipientward_fraction_each_state"] \
            and mf >= bars["cross_task_matched_minimum_fraction_of_within_narrative_S_value"] \
            and cf >= bars["cross_task_matched_minimum_fraction_of_within_narrative_S_value"] and mc > 0
        state_passes[state] = passed
        cross[state] = {"matched_mean_margin_delta": mm, "matched_mean_full_vocab_CE_gain": mc,
                        "margin_fraction_of_narrative_S_value": mf,
                        "CE_fraction_of_narrative_S_value": cf,
                        "matched_donorward_fraction": direction,
                        "unmatched_recipientward_fraction": unmatched_direction, "passed": passed}
    semantic_reuse = all(state_passes.values())
    was_plural = cross["was"]
    generic_confound = (state_passes["is"] and not state_passes["was"]) or (
        was_plural["matched_mean_margin_delta"] > 0
        and was_plural["matched_mean_full_vocab_CE_gain"] <= 0
    )
    instrument = exact_live and native_live and complete_live and controls_live
    predictions = {
        "pred_a_instrument_live": instrument,
        "pred_b_self_route": instrument and route_S["passed"],
        "pred_c_tense_cue_route": instrument and route_T["passed"],
        "pred_d_mixed_or_remaining_route": instrument and not (route_S["passed"] or route_T["passed"]),
        "pred_e_cross_task_semantic_reuse": instrument and semantic_reuse,
        "pred_f_generic_output_token_confound": instrument and generic_confound,
    }
    return {"capability": capability, "exactness": exactness, "target_cells": summaries,
            "controls": control_summary, "routes": {"S": route_S, "T": route_T},
            "factor_readout": factor_readout, "cross_task": cross, "predictions": predictions}


def _load_frozen_native_pairs():
    result = json.loads(NATIVE_PARENT.read_text())
    return {(str(x["row_id"]), str(x["side"])): (float(x["answer_logit"]), float(x["foil_logit"]))
            for x in result["run"]["native_logits"]}


def evaluate(model, torch, F, facade):
    rows = build_rows()
    pairing = build_pairing(rows)
    pair_by_row = {x["narrative_row_id"]: x for x in pairing}
    device = next(model.parameters()).device
    max_length = max(len(row[side]) for row in rows for side in ("base_ids", "donor_ids"))
    base_tokens, base_finals = factor._pad(rows, "base", max_length, torch, device)
    donor_tokens, donor_finals = factor._pad(rows, "donor", max_length, torch, device)

    fronted = {str(row["group_id"]): row for row in task14.build_rows()}
    endpoints = []
    endpoint_keys = []
    for group_id in sorted(fronted):
        row = fronted[group_id]
        for side in ("base", "donor"):
            state = str(row[f"{side}_subject_number"])
            endpoints.append({"ids": row[f"{side}_ids"]})
            endpoint_keys.append((group_id, state))
    task_tokens, task_finals = _pad_endpoints(endpoints, 9, torch, device)
    task_values = {}
    exactness = {"native_replay_max_absolute_error": 0.0,
                 "source_sum_max_absolute_error": 0.0,
                 "installed_noop_max_absolute_error": 0.0,
                 "complete_head_endpoint_reproduction_max_absolute_error": 0.0,
                 "task14_source_sum_max_absolute_error": 0.0}
    for start in range(0, len(endpoints), BATCH):
        _logits, captured = factor._factor_forward(
            model, task_tokens[start:start+BATCH], task_finals[start:start+BATCH], torch, F, facade)
        exactness["task14_source_sum_max_absolute_error"] = max(
            exactness["task14_source_sum_max_absolute_error"],
            float((torch.einsum("bk,bkd->bd", captured["p"], captured["u"])
                   - captured["head"]).abs().max()))
        for local, key in enumerate(endpoint_keys[start:start+BATCH]):
            task_values[key] = captured["u"][local, 8].clone()

    frozen_native = _load_frozen_native_pairs()
    frozen_parent = json.loads(PARENT.read_text())
    frozen_complete_margin = {
        str(item["row_id"]): float(item["margins"]["head3"])
        for item in frozen_parent["evidence"]
    }
    if set(frozen_complete_margin) != {str(row["row_id"]) for row in rows}:
        raise ScreenError("frozen complete-H3 evidence lost exact row coverage")
    evidence = []
    capability_hits = defaultdict(lambda: {"base": [], "donor": []})
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start+BATCH]
        bt, bf = base_tokens[start:start+BATCH], base_finals[start:start+BATCH]
        dt, df = donor_tokens[start:start+BATCH], donor_finals[start:start+BATCH]
        base_logits, base = factor._factor_forward(model, bt, bf, torch, F, facade)
        donor_logits, donor = factor._factor_forward(model, dt, df, torch, F, facade)
        for captured in (base, donor):
            exactness["source_sum_max_absolute_error"] = max(
                exactness["source_sum_max_absolute_error"],
                float((torch.einsum("bk,bkd->bd", captured["p"], captured["u"])
                       - captured["head"]).abs().max()))

        indices, heads, specs = [], [], []
        for local, row in enumerate(chunk):
            groups = {name: row[f"{name}_positions"] for name in ("S", "T", "R")}
            arm_heads = {
                "native_noop": base["head"][local],
                "complete_head": donor["head"][local],
            }
            for route in ("S", "T"):
                for mode in ("score", "value", "joint"):
                    sliced_base = {key: value[local:local+1] for key, value in base.items()}
                    sliced_donor = {key: value[local:local+1] for key, value in donor.items()}
                    arm_heads[f"{route}_{mode}"] = _group_head(
                        sliced_base, sliced_donor, groups[route], mode, torch)[0]
            sliced_base = {key: value[local:local+1] for key, value in base.items()}
            sliced_donor = {key: value[local:local+1] for key, value in donor.items()}
            arm_heads["R_joint"] = _group_head(sliced_base, sliced_donor, groups["R"], "joint", torch)[0]
            for arm in WITHIN_ARMS:
                indices.append(local); heads.append(arm_heads[arm]); specs.append((local, arm, None))
            if str(row["transform_id"]) in TARGET_FAMILIES:
                record = pair_by_row[str(row["row_id"])]
                source = int(row["S_positions"][0])
                native_term = base["p"][local, source] * base["u"][local, source]
                for state in ("singular", "plural"):
                    u = task_values[(str(record["task14_group_id"]), state)]
                    replacement = base["p"][local, source] * u
                    indices.append(local)
                    heads.append(base["head"][local] - native_term + replacement)
                    specs.append((local, f"task14_{state}_value", state))
        index = torch.tensor(indices, dtype=torch.long, device=device)
        patched_logits, patched_factors = factor._factor_forward(
            model, bt[index], bf[index], torch, F, facade,
            replacement_heads=torch.stack(heads))
        exactness["source_sum_max_absolute_error"] = max(
            exactness["source_sum_max_absolute_error"],
            float((torch.einsum("bk,bkd->bd", patched_factors["p"], patched_factors["u"])
                   - patched_factors["head"]).abs().max()))
        noop_cursor = 0
        for output_index, (local, arm, cross_state) in enumerate(specs):
            row = chunk[local]
            q = int(bf[local])
            native = _metrics(base_logits[local], q, row, torch)
            patched = _metrics(patched_logits[output_index], q, row, torch)
            family = str(row["transform_id"])
            item = {"row_id": str(row["row_id"]), "family": family,
                    "cell_id": str(row["capability_cell_id"]), "arm": arm,
                    "margin_delta": patched[0] - native[0], "donor_ce_gain": native[1] - patched[1],
                    "full_vocab_donor_ce": patched[1],
                    "was_is_are_logit_change": {name: patched[4][name] - native[4][name]
                                                 for name in ANSWER_IDS},
                    "target_state": ("is" if str(row["donor_answer"]) == " is" else "was")
                                    if family in TARGET_FAMILIES else None,
                    "cross_task_state": cross_state}
            if family in CONTROL_FAMILIES:
                donor_metric = _metrics(donor_logits[local], int(df[local]), row, torch)
                scale = abs(native[2]) + abs(donor_metric[2])
                if scale <= 1e-12:
                    raise ScreenError("control normalization scale vanished")
                item["normalized_control_movement"] = (patched[2] - native[2]) / scale
                item["full_vocab_base_answer_CE_change"] = patched[3] - native[3]
            evidence.append(item)
            if arm == "native_noop":
                exactness["installed_noop_max_absolute_error"] = max(
                    exactness["installed_noop_max_absolute_error"],
                    float((patched_logits[output_index] - base_logits[local]).abs().max()))
            elif arm == "complete_head":
                # The parent records base-answer-minus-foil. On answer-changing rows
                # that is the negative of this runner's donor-minus-recipient margin;
                # controls retain the parent answer-minus-foil orientation.
                observed = patched[0] if family in TARGET_FAMILIES else patched[2]
                expected = (-frozen_complete_margin[str(row["row_id"])]
                            if family in TARGET_FAMILIES
                            else frozen_complete_margin[str(row["row_id"])])
                exactness["complete_head_endpoint_reproduction_max_absolute_error"] = max(
                    exactness["complete_head_endpoint_reproduction_max_absolute_error"],
                    abs(observed - expected))

        for local, row in enumerate(chunk):
            cell = str(row["capability_cell_id"])
            if str(row["transform_id"]) not in TARGET_FAMILIES:
                continue
            for side, logits, q in (("base", base_logits[local], int(bf[local])),
                                    ("donor", donor_logits[local], int(df[local]))):
                answer = int(row[f"{side}_answer_id"]); foil = int(row[f"{side}_foil_id"])
                capability_hits[cell][side].append(float(logits[q, answer] > logits[q, foil]))
                observed = (float(logits[q, answer]), float(logits[q, foil]))
                expected = frozen_native[(str(row["row_id"]), side)]
                exactness["native_replay_max_absolute_error"] = max(
                    exactness["native_replay_max_absolute_error"],
                    abs(observed[0] - expected[0]), abs(observed[1] - expected[1]))
    capability = {cell: {side: _mean(values) for side, values in sides.items()}
                  for cell, sides in capability_hits.items()}
    return evidence, capability, exactness, pairing


def _terminal(predictions):
    if not predictions["pred_a_instrument_live"]:
        return "invalid"
    if predictions["pred_b_self_route"]:
        if predictions["pred_e_cross_task_semantic_reuse"]:
            return "self_route_cross_task_semantic_reuse_screen"
        if predictions["pred_f_generic_output_token_confound"]:
            return "self_route_generic_output_token_confound"
        return "self_route_without_cross_task_reuse"
    if predictions["pred_c_tense_cue_route"]:
        return "tense_cue_route_screen"
    return "mixed_or_remaining_route_null"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise ScreenError(f"{name} must be absent or exactly 1")
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise ScreenError(f"refusing to overwrite {OUT}")
    torch, F, facade = factor._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, capability, exactness, pairing = evaluate(model, torch, F, facade)
    scored = score(evidence, capability, exactness)
    terminal = _terminal(scored["predictions"])
    result = {
        "schema": "narrative_tense_attn11_head3_source_route_cross_task_payload_result_v1",
        "candidate_id": plan["candidate_id"], "terminal": terminal,
        "plan": plan, "prior_art_sha256": EXPECTED_SHA256[PRIOR_ART],
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "pairing": pairing, "pairing_sha256": PAIRING_SHA256,
        "score": scored, "evidence": evidence,
        "evaluated_splits": ["FIT_BASIC"], "forbidden_splits_opened": [],
        "active_price": plan["price"],
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "result_path": OUT.relative_to(ROOT).as_posix(),
                      "result_sha256": hashlib.sha256(payload).hexdigest(),
                      "active_price": plan["price"]}, sort_keys=True))


if __name__ == "__main__":
    main()
