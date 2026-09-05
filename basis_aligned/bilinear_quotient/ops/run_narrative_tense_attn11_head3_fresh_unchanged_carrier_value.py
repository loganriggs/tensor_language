#!/usr/bin/env python3
"""Exact fresh-confirmation L11H3 unchanged-carrier factor intervention."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_unchanged_carrier_route pred_c_unchanged_carrier_effective_value pred_d_pre_first_negative_control pred_e_between_changes_effective_value pred_f_post_last_change_effective_value pred_g_distributed_R_effective_value pred_h_no_unchanged_carrier_route

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Mapping, Sequence

import circuit_fast_screen_candidate_narrative_tense_fresh_unchanged_carrier as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_head11_3_subject_attractor_score_payload_factorial as factor


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/narrative_tense_attn11_head3_fresh_unchanged_carrier_value_v1.json"
OUT = ROOT / "circuits/fast_screens/narrative_tense_attn11_head3_fresh_unchanged_carrier_value_v1_result.json"
PRIOR_ART_SHA256 = "5978ab3cb345aff98b1af8f457db5db2dad05415cbc3c0dd2026e7747770b62c"
AUTHORITY_SHA256 = "337c0b10f8f0d74bef5c1f3d9fef12b40231ebd78fe8fff38d8fb6af9b5f3178"
LAYER = 11
HEAD = 3
BATCH = 32
TARGET_FAMILIES = ("A1", "A2")
CONTROL_FAMILIES = ("P", "C")
ARMS = (
    "expanded_native", "native_reinstall", "complete_head", "R_score",
    "R_effective_value", "R_joint", "complement_joint",
    "pre_first_change_effective_value", "between_changes_effective_value",
    "post_last_change_effective_value",
)
SELECTIVITY_ARMS = (
    "R_score", "R_effective_value", "R_joint",
    "between_changes_effective_value", "post_last_change_effective_value",
)
BARS = {
    "minimum_native_accuracy_each_direction_side_cell": .85,
    "maximum_source_sum_and_same_batch_native_reinstall_error": 5e-5,
    "minimum_complete_H3_donorward_fraction_each_target_cell": .75,
    "R_joint_minimum_fraction_of_complete_margin_and_CE_each_target_cell": .50,
    "R_joint_minimum_donorward_fraction_each_target_cell": .75,
    "R_joint_minimum_advantage_over_complement_each_target_cell": .15,
    "R_effective_value_minimum_fraction_of_R_joint_margin_and_CE_each_target_cell": .70,
    "R_effective_value_minimum_donorward_fraction_each_target_cell": .75,
    "maximum_each_R_or_causal_subgroup_P_C_absolute_effect_fraction_of_smallest_A1_A2_R_target_effect": .25,
    "causal_subgroup_minimum_fraction_of_R_effective_value_margin_and_CE_each_target_cell": .70,
    "causal_subgroup_minimum_donorward_fraction_each_target_cell": .75,
    "minimum_live_installed_factor_difference_norm_each_tested_P_C_cell": 1e-8,
}


class ScreenError(ValueError):
    """The frozen authority or exact intervention contract was violated."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows() -> list[dict[str, object]]:
    rows = authority.build_rows()
    if authority.validate_rows(rows) != AUTHORITY_SHA256 or len(rows) != 128:
        raise ScreenError("fresh narrative authority changed")
    expected_diffs = {"A1": (0, 3), "A2": (3, 5, 6), "P": (2,), "C": (4, 7)}
    for row in rows:
        family = str(row["transform_id"])
        base, donor = tuple(row["base_ids"]), tuple(row["donor_ids"])
        changed = tuple(i for i, pair in enumerate(zip(base, donor)) if pair[0] != pair[1])
        if changed != expected_diffs[family] or len(base) != len(donor):
            raise ScreenError("fresh token alignment changed")
        self_position = len(base) - 1
        first, last = min(changed), max(changed)
        row["T_positions"] = changed
        row["S_positions"] = (self_position,)
        row["pre_first_change_positions"] = tuple(i for i in range(first) if i not in changed)
        row["between_changes_positions"] = tuple(
            i for i in range(first + 1, last) if i not in changed)
        row["post_last_change_positions"] = tuple(
            i for i in range(last + 1, self_position) if i not in changed)
        row["R_positions"] = tuple(
            i for i in range(self_position) if i not in changed)
        row["complement_positions"] = tuple(changed + (self_position,))
        if any(base[i] != donor[i] for i in row["R_positions"]):
            raise ScreenError("measured carrier contains a changed token")
        if set(row["R_positions"]) & set(row["complement_positions"]):
            raise ScreenError("source roles overlap")
        if set(row["R_positions"]) | set(row["complement_positions"]) != set(range(len(base))):
            raise ScreenError("source roles do not exhaust the prompt")
    return rows


def compile_plan() -> dict[str, object]:
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise ScreenError("prior-art receipt changed")
    rows = build_rows()
    return {
        "schema": "narrative_tense_attn11_head3_fresh_unchanged_carrier_value_plan_v1",
        "candidate_id": "narrative_tense.attn11_head3_fresh_unchanged_carrier_value_v1",
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "prior_art_sha256": PRIOR_ART_SHA256, "authority_sha256": AUTHORITY_SHA256,
        "row_count": len(rows), "split": "fresh_confirmation", "layer": LAYER, "head": HEAD,
        "arms": list(ARMS), "bars": dict(BARS),
        "factor_definition": {
            "score": "donor attention probability times the recipient effective source value",
            "value": "recipient attention probability times the donor effective source value",
            "joint": "donor attention probability times the donor effective source value",
        },
        "price": {"model_forwards": 12, "example_evaluations": 1536,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_directed_was_is_margin", "full_vocabulary_donor_CE_gain",
                     "absolute_P_C_answer_margin_and_full_vocabulary_CE_change",
                     "source_factor_and_same_batch_native_reinstall_exactness"],
    }


def _group_head(base, donor, positions, mode, torch):
    indices = torch.tensor(tuple(positions), dtype=torch.long, device=base["p"].device)
    if indices.numel() == 0:
        return base["head"].clone()
    native = torch.einsum("bk,bkd->bd", base["p"][:, indices], base["u"][:, indices])
    if mode == "score":
        replacement = torch.einsum("bk,bkd->bd", donor["p"][:, indices],
                                   base["u"][:, indices])
    elif mode == "value":
        replacement = torch.einsum("bk,bkd->bd", base["p"][:, indices],
                                   donor["u"][:, indices])
    elif mode == "joint":
        replacement = torch.einsum("bk,bkd->bd", donor["p"][:, indices],
                                   donor["u"][:, indices])
    else:
        raise ScreenError(f"unknown factor mode: {mode}")
    return base["head"] - native + replacement


def _mean(values):
    if not values:
        raise ScreenError("empty score cell")
    return statistics.fmean(values)


def _metrics(logits, q, row, torch):
    donor, recipient = int(row["donor_answer_id"]), int(row["base_answer_id"])
    margin = float(logits[q, donor] - logits[q, recipient])
    donor_ce = float(-torch.log_softmax(logits[q], dim=-1)[donor])
    answer = int(row["base_answer_id"]); foil = int(row["base_foil_id"])
    answer_margin = float(logits[q, answer] - logits[q, foil])
    answer_ce = float(-torch.log_softmax(logits[q], dim=-1)[answer])
    return margin, donor_ce, answer_margin, answer_ce


def _cell_summaries(evidence, families):
    cells = sorted({str(x["cell_id"]) for x in evidence if x["family"] in families})
    output = {}
    for cell in cells:
        output[cell] = {}
        for arm in ARMS:
            items = [x for x in evidence if x["cell_id"] == cell and x["arm"] == arm]
            output[cell][arm] = {
                "row_count": len(items),
                "mean_margin_delta": _mean([x["margin_delta"] for x in items]),
                "mean_full_vocab_CE_gain": _mean([x["donor_ce_gain"] for x in items]),
                "donorward_fraction": _mean([x["margin_delta"] > 0 for x in items]),
            }
    return output


def score(evidence, capability, exactness, liveness, bars=BARS):
    target = _cell_summaries(evidence, TARGET_FAMILIES)
    target_cells = sorted(target)
    capability_live = all(
        min(float(value) for value in sides.values()) >=
        bars["minimum_native_accuracy_each_direction_side_cell"]
        for sides in capability.values()
    )
    exact_live = bool(exactness) and max(float(x) for x in exactness.values()) <= \
        bars["maximum_source_sum_and_same_batch_native_reinstall_error"]
    complete_live = all(
        target[cell]["complete_head"]["mean_margin_delta"] > 0
        and target[cell]["complete_head"]["mean_full_vocab_CE_gain"] > 0
        and target[cell]["complete_head"]["donorward_fraction"] >=
            bars["minimum_complete_H3_donorward_fraction_each_target_cell"]
        for cell in target_cells
    )

    route_cells = {}
    route_live = True
    for cell in target_cells:
        complete, route, other = (target[cell][name] for name in
                                  ("complete_head", "R_joint", "complement_joint"))
        mf = route["mean_margin_delta"] / complete["mean_margin_delta"]
        cf = route["mean_full_vocab_CE_gain"] / complete["mean_full_vocab_CE_gain"]
        omf = other["mean_margin_delta"] / complete["mean_margin_delta"]
        ocf = other["mean_full_vocab_CE_gain"] / complete["mean_full_vocab_CE_gain"]
        passed = (mf >= bars["R_joint_minimum_fraction_of_complete_margin_and_CE_each_target_cell"]
                  and cf >= bars["R_joint_minimum_fraction_of_complete_margin_and_CE_each_target_cell"]
                  and route["donorward_fraction"] >=
                      bars["R_joint_minimum_donorward_fraction_each_target_cell"]
                  and mf - omf >= bars["R_joint_minimum_advantage_over_complement_each_target_cell"]
                  and cf - ocf >= bars["R_joint_minimum_advantage_over_complement_each_target_cell"])
        route_cells[cell] = {"margin_fraction_of_complete": mf,
                             "CE_fraction_of_complete": cf,
                             "margin_advantage_over_complement": mf - omf,
                             "CE_advantage_over_complement": cf - ocf,
                             "donorward_fraction": route["donorward_fraction"], "passed": passed}
        route_live &= passed

    def fraction_report(arm, denominator, minimum_fraction, minimum_direction):
        cells, all_pass = {}, True
        for cell in target_cells:
            value, denom = target[cell][arm], target[cell][denominator]
            mf = value["mean_margin_delta"] / denom["mean_margin_delta"] \
                if denom["mean_margin_delta"] > 1e-12 else float("-inf")
            cf = value["mean_full_vocab_CE_gain"] / denom["mean_full_vocab_CE_gain"] \
                if denom["mean_full_vocab_CE_gain"] > 1e-12 else float("-inf")
            passed = mf >= minimum_fraction and cf >= minimum_fraction \
                and value["donorward_fraction"] >= minimum_direction
            cells[cell] = {"margin_fraction": mf, "CE_fraction": cf,
                           "donorward_fraction": value["donorward_fraction"], "passed": passed}
            all_pass &= passed
        return {"cells": cells, "passed": all_pass}

    value_report = fraction_report(
        "R_effective_value", "R_joint",
        bars["R_effective_value_minimum_fraction_of_R_joint_margin_and_CE_each_target_cell"],
        bars["R_effective_value_minimum_donorward_fraction_each_target_cell"])
    subgroup_reports = {
        name: fraction_report(
            f"{name}_effective_value", "R_effective_value",
            bars["causal_subgroup_minimum_fraction_of_R_effective_value_margin_and_CE_each_target_cell"],
            bars["causal_subgroup_minimum_donorward_fraction_each_target_cell"])
        for name in ("between_changes", "post_last_change")
    }

    # These contrasts cost no forwards: every corner is already in the factorial.
    interactions = {}
    for cell in target_cells:
        native = target[cell]["expanded_native"]
        complete = target[cell]["complete_head"]
        route = target[cell]["R_joint"]
        complement = target[cell]["complement_joint"]
        score_arm = target[cell]["R_score"]
        value_arm = target[cell]["R_effective_value"]
        interactions[cell] = {
            "R_by_complement_margin": complete["mean_margin_delta"]
                - route["mean_margin_delta"] - complement["mean_margin_delta"]
                + native["mean_margin_delta"],
            "R_by_complement_CE": complete["mean_full_vocab_CE_gain"]
                - route["mean_full_vocab_CE_gain"] - complement["mean_full_vocab_CE_gain"]
                + native["mean_full_vocab_CE_gain"],
            "score_by_value_margin": route["mean_margin_delta"]
                - score_arm["mean_margin_delta"] - value_arm["mean_margin_delta"]
                + native["mean_margin_delta"],
            "score_by_value_CE": route["mean_full_vocab_CE_gain"]
                - score_arm["mean_full_vocab_CE_gain"] - value_arm["mean_full_vocab_CE_gain"]
                + native["mean_full_vocab_CE_gain"],
        }

    target_margin_scale = min(abs(target[cell]["R_joint"]["mean_margin_delta"])
                              for cell in target_cells)
    target_ce_scale = min(abs(target[cell]["R_joint"]["mean_full_vocab_CE_gain"])
                          for cell in target_cells)
    control_limits = {
        "answer_margin": bars["maximum_each_R_or_causal_subgroup_P_C_absolute_effect_fraction_of_smallest_A1_A2_R_target_effect"]
                         * target_margin_scale,
        "full_vocab_CE": bars["maximum_each_R_or_causal_subgroup_P_C_absolute_effect_fraction_of_smallest_A1_A2_R_target_effect"]
                         * target_ce_scale,
    }
    control_report = {}
    for family in CONTROL_FAMILIES:
        control_report[family] = {}
        for arm in SELECTIVITY_ARMS:
            items = [x for x in evidence if x["family"] == family and x["arm"] == arm]
            margin = _mean([abs(x["answer_margin_delta"]) for x in items])
            ce = _mean([abs(x["base_answer_CE_change"]) for x in items])
            passed = margin <= control_limits["answer_margin"] \
                and ce <= control_limits["full_vocab_CE"]
            control_report[family][arm] = {
                "mean_absolute_answer_margin_change": margin,
                "mean_absolute_full_vocab_CE_change": ce, "passed": passed,
            }

    instrument = exact_live and capability_live and complete_live
    controls_informative = all(
        float(value) > bars["minimum_live_installed_factor_difference_norm_each_tested_P_C_cell"]
        for cell, value in liveness["minimum_R_factor_difference_norm_by_cell"].items()
        if cell.startswith(("P/", "C/"))
    )
    target_carrier_live = all(
        float(value) > bars["minimum_live_installed_factor_difference_norm_each_tested_P_C_cell"]
        for cell, value in liveness["minimum_R_factor_difference_norm_by_cell"].items()
        if cell.startswith(("A1/", "A2/"))
    )
    def arm_selective(arm):
        return controls_informative and all(
            control_report[family][arm]["passed"] for family in CONTROL_FAMILIES)
    route_confirmed = route_live and target_carrier_live and arm_selective("R_joint")
    value_confirmed = route_confirmed and value_report["passed"] \
        and arm_selective("R_effective_value")
    predictions = {
        "pred_a_instrument_live": instrument,
        "pred_b_unchanged_carrier_route": instrument and route_confirmed,
        "pred_c_unchanged_carrier_effective_value": instrument and value_confirmed,
        "pred_d_pre_first_negative_control": exact_live,
        "pred_e_between_changes_effective_value": instrument and value_confirmed
            and subgroup_reports["between_changes"]["passed"]
            and arm_selective("between_changes_effective_value"),
        "pred_f_post_last_change_effective_value": instrument and value_confirmed
            and subgroup_reports["post_last_change"]["passed"]
            and arm_selective("post_last_change_effective_value"),
        "pred_g_distributed_R_effective_value": instrument and value_confirmed
            and not any(x["passed"] and arm_selective(f"{name}_effective_value")
                        for name, x in subgroup_reports.items()),
        "pred_h_no_unchanged_carrier_route": instrument and not route_confirmed,
    }
    return {"capability": capability, "exactness": exactness, "liveness": liveness,
            "target_cells": target, "route": {"cells": route_cells, "passed": route_live},
            "R_effective_value": value_report, "subgroups": subgroup_reports,
            "interactions": interactions, "control_limits": control_limits,
            "controls": control_report, "controls_informative": controls_informative,
            "target_carrier_live": target_carrier_live,
            "predictions": predictions}


def evaluate(model, torch, F, facade):
    rows = build_rows()
    device = next(model.parameters()).device
    max_length = max(len(row[side]) for row in rows for side in ("base_ids", "donor_ids"))
    base_tokens, base_finals = factor._pad(rows, "base", max_length, torch, device)
    donor_tokens, donor_finals = factor._pad(rows, "donor", max_length, torch, device)
    evidence = []
    capability_hits = defaultdict(lambda: {"base": [], "donor": []})
    exactness = {"source_sum_max_absolute_error": 0.0,
                 "same_batch_native_reinstall_max_absolute_error": 0.0,
                 "pre_first_change_install_max_absolute_error": 0.0}
    factor_differences = defaultdict(list)
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

        indices, heads, masks, native_masks, specs = [], [], [], [], []
        for local, row in enumerate(chunk):
            sliced_base = {k: v[local:local+1] for k, v in base.items()}
            sliced_donor = {k: v[local:local+1] for k, v in donor.items()}
            positions = {name: row[f"{name}_positions"] for name in
                         ("R", "complement", "pre_first_change", "between_changes",
                          "post_last_change")}
            r = torch.tensor(row["R_positions"], dtype=torch.long, device=device)
            difference = torch.cat(((base["p"][local, r] - donor["p"][local, r]).reshape(-1),
                                    (base["u"][local, r] - donor["u"][local, r]).reshape(-1)))
            factor_differences[str(row["capability_cell_id"])].append(
                float(torch.linalg.vector_norm(difference)))
            arm_heads = {
                "expanded_native": base["head"][local],
                "native_reinstall": base["head"][local],
                "complete_head": donor["head"][local],
                "R_score": _group_head(sliced_base, sliced_donor, positions["R"], "score", torch)[0],
                "R_effective_value": _group_head(
                    sliced_base, sliced_donor, positions["R"], "value", torch)[0],
                "R_joint": _group_head(sliced_base, sliced_donor, positions["R"], "joint", torch)[0],
                "complement_joint": _group_head(sliced_base, sliced_donor,
                                                positions["complement"], "joint", torch)[0],
                "pre_first_change_effective_value": _group_head(
                    sliced_base, sliced_donor, positions["pre_first_change"], "value", torch)[0],
                "between_changes_effective_value": _group_head(
                    sliced_base, sliced_donor, positions["between_changes"], "value", torch)[0],
                "post_last_change_effective_value": _group_head(
                    sliced_base, sliced_donor, positions["post_last_change"], "value", torch)[0],
            }
            for arm in ARMS:
                indices.append(local); heads.append(arm_heads[arm]);
                masks.append(arm != "expanded_native")
                native_masks.append(arm == "native_reinstall")
                specs.append((local, arm))
        index = torch.tensor(indices, dtype=torch.long, device=device)
        patched_logits, patched = factor._factor_forward(
            model, bt[index], bf[index], torch, F, facade,
            replacement_heads=torch.stack(heads),
            replacement_head_mask=torch.tensor(masks, dtype=torch.bool, device=device),
            native_reinstall_mask=torch.tensor(native_masks, dtype=torch.bool, device=device))
        exactness["source_sum_max_absolute_error"] = max(
            exactness["source_sum_max_absolute_error"],
            float((torch.einsum("bk,bkd->bd", patched["p"], patched["u"])
                   - patched["head"]).abs().max()))
        expanded_native_logits = {}
        arm_logits = {}
        for output_index, (local, arm) in enumerate(specs):
            row = chunk[local]; q = int(bf[local])
            changed = _metrics(patched_logits[output_index], q, row, torch)
            if arm == "expanded_native":
                expanded_native_logits[local] = patched_logits[output_index]
                native = changed
            else:
                native = _metrics(expanded_native_logits[local], q, row, torch)
            item = {"row_id": str(row["row_id"]), "family": str(row["transform_id"]),
                    "cell_id": str(row["capability_cell_id"]), "arm": arm,
                    "margin_delta": changed[0] - native[0],
                    "donor_ce_gain": native[1] - changed[1],
                    "full_vocab_donor_CE": changed[1],
                    "answer_margin_delta": changed[2] - native[2],
                    "base_answer_CE_change": changed[3] - native[3]}
            evidence.append(item)
            arm_logits[(local, arm)] = patched_logits[output_index]
            if arm == "native_reinstall":
                exactness["same_batch_native_reinstall_max_absolute_error"] = max(
                    exactness["same_batch_native_reinstall_max_absolute_error"],
                    float((patched_logits[output_index] - expanded_native_logits[local]).abs().max()))
        for local in range(len(chunk)):
            exactness["pre_first_change_install_max_absolute_error"] = max(
                exactness["pre_first_change_install_max_absolute_error"],
                float((arm_logits[(local, "pre_first_change_effective_value")]
                       - expanded_native_logits[local]).abs().max()),
            )

        for local, row in enumerate(chunk):
            cell = str(row["capability_cell_id"])
            for side, logits, q in (("base", base_logits[local], int(bf[local])),
                                    ("donor", donor_logits[local], int(df[local]))):
                answer, foil = int(row[f"{side}_answer_id"]), int(row[f"{side}_foil_id"])
                capability_hits[cell][side].append(float(logits[q, answer] > logits[q, foil]))
    capability = {cell: {side: _mean(values) for side, values in sides.items()}
                  for cell, sides in capability_hits.items()}
    liveness = {
        "all_registered_token_differences_nonempty": True,
        "minimum_R_factor_difference_norm_by_cell": {
            cell: min(values) for cell, values in sorted(factor_differences.items())
        },
    }
    return evidence, capability, exactness, liveness


def _terminal(predictions):
    if not predictions["pred_a_instrument_live"]:
        return "invalid"
    if predictions["pred_h_no_unchanged_carrier_route"]:
        return "no_selective_unchanged_carrier_route_null"
    if not predictions["pred_c_unchanged_carrier_effective_value"]:
        return "unchanged_carrier_route_not_effective_value"
    named = [name for name, key in (
        ("between_changes", "pred_e_between_changes_effective_value"),
        ("post_last_change", "pred_f_post_last_change_effective_value"),
    ) if predictions[key]]
    if named:
        return "unchanged_carrier_value_" + "_and_".join(named)
    if predictions["pred_g_distributed_R_effective_value"]:
        return "unchanged_carrier_distributed_R_effective_value"
    return "unchanged_carrier_value_unclassified"


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
        evidence, capability, exactness, liveness = evaluate(model, torch, F, facade)
    scored = score(evidence, capability, exactness, liveness)
    terminal = _terminal(scored["predictions"])
    result = {
        "schema": "narrative_tense_attn11_head3_fresh_unchanged_carrier_value_result_v1",
        "candidate_id": plan["candidate_id"], "terminal": terminal, "plan": plan,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": scored, "evidence": evidence,
        "evaluated_splits": ["FRESH_CONFIRMATION_BASIC"],
        "forbidden_splits_opened": [], "active_price": plan["price"],
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "result_path": OUT.relative_to(ROOT).as_posix(),
                      "result_sha256": hashlib.sha256(payload).hexdigest(),
                      "active_price": plan["price"]}, sort_keys=True))


if __name__ == "__main__":
    main()
