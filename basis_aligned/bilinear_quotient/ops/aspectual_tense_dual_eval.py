#!/usr/bin/env python3
"""Pure shared contracts for paired has/had and is/was program evaluation."""

# BQGATE: LIBRARY
from __future__ import annotations

import math
import statistics


BANKS = ("has_had", "is_was")
FAMILIES = ("A1", "A2", "P", "C")
DIRECTIONS = ("past_to_present", "present_to_past")
METRIC_BY_FAMILY = {"A1": "recovery", "A2": "recovery", "P": "margin_reflection_fraction", "C": "normalized_unrelated_effect"}


class DualEvalError(ValueError):
    pass


def direction_for(row):
    family = row.get("family")
    if family not in FAMILIES:
        raise DualEvalError(f"unknown family: {family!r}")
    if family in ("A1", "A2"):
        direction = row.get("direction_id")
        if direction not in DIRECTIONS:
            raise DualEvalError(f"unknown direction: {direction!r}")
        return direction
    group_number = row.get("group_number")
    if not isinstance(group_number, int) or isinstance(group_number, bool) or group_number < 0:
        raise DualEvalError("group_number must be a nonnegative integer")
    return "present_to_past" if group_number % 2 == 0 else "past_to_present"


def source_side(family):
    if family not in FAMILIES:
        raise DualEvalError(f"unknown family: {family!r}")
    return "donor" if family == "P" else "base"


def requested_token_ids(bank, direction, token_ids_by_bank):
    if bank not in BANKS or direction not in DIRECTIONS:
        raise DualEvalError(f"unknown bank/direction: {bank!r}/{direction!r}")
    tokens = token_ids_by_bank.get(bank)
    names = ("has", "had") if bank == "has_had" else ("is", "was")
    if not isinstance(tokens, dict) or set(tokens) != set(names) or any(not isinstance(tokens[name], int) for name in names):
        raise DualEvalError(f"invalid token map for {bank}")
    present, past = tokens[names[0]], tokens[names[1]]
    return (past, present) if direction == "present_to_past" else (present, past)


def capability_cells(bank, rows, outputs, *, full_two_sided=True):
    if bank not in BANKS or set(outputs) != {"base", "donor"}:
        raise DualEvalError("invalid bank or output sides")
    observations = []
    for index, row in enumerate(rows):
        sides = ("base", "donor") if full_two_sided else (source_side(row["family"]),)
        for side in sides:
            try:
                answer, foil = outputs[side].answer_foil[index]
            except (AttributeError, IndexError, TypeError, ValueError) as error:
                raise DualEvalError("missing capability answer/foil") from error
            answer, foil = float(answer), float(foil)
            if not math.isfinite(answer) or not math.isfinite(foil):
                raise DualEvalError("nonfinite capability score")
            observations.append({"family": row["family"], "direction": direction_for(row), "correct": answer > foil})
    cells = []
    for family in FAMILIES:
        for direction in DIRECTIONS:
            selected = [item for item in observations if item["family"] == family and item["direction"] == direction]
            if not selected:
                raise DualEvalError(f"empty capability cell: {family}/{direction}")
            threshold = 0.75 if family == "C" else 0.85
            correct = sum(item["correct"] for item in selected)
            accuracy = correct / len(selected)
            cells.append({"bank": bank, "family": family, "direction": direction, "correct": correct, "total": len(selected), "accuracy": accuracy, "threshold": threshold, "passed": accuracy >= threshold})
    return cells


def metric_summary(records, key):
    values = [record[key] for record in records]
    if not values or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise DualEvalError(f"empty/nonfinite metric: {key}")
    return {
        "count": len(values),
        f"mean_{key}": statistics.fmean(values),
        f"mean_absolute_{key}": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def summarize_program_records(records):
    summaries = {}
    for bank in BANKS:
        summaries[bank] = {}
        for family in FAMILIES:
            selected = [record for record in records if record.get("bank") == bank and record.get("family") == family]
            summaries[bank][family] = metric_summary(selected, METRIC_BY_FAMILY[family])
    return summaries


def program_bars_pass(summary, *, target_bar=0.75, control_bar=0.20):
    for family in ("A1", "A2", "P"):
        key = METRIC_BY_FAMILY[family]
        if summary[family][f"mean_{key}"] < target_bar or summary[family]["direction_fraction"] < target_bar:
            return False
    return summary["C"]["mean_normalized_unrelated_effect"] <= control_bar


def exact_price(*, rows, forwards, intervention_records, fitted_scalars=0, inherited_gain_scalars=8, basis_scalars=2304):
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (rows, forwards, intervention_records, fitted_scalars, inherited_gain_scalars, basis_scalars)):
        raise DualEvalError("price fields must be nonnegative integers")
    return {
        "model_forwards": forwards, "example_evaluations": 2 * rows, "rows": rows,
        "intervention_records": intervention_records, "inherited_gain_scalars": inherited_gain_scalars,
        "basis_scalars": basis_scalars, "fitted_scalars": fitted_scalars, "grid_evaluations": 0,
        "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0,
    }


def verify_contract():
    token_maps = {"has_had": {"has": 1, "had": 2}, "is_was": {"is": 3, "was": 4}}
    if direction_for({"family": "A1", "direction_id": "past_to_present"}) != "past_to_present":
        raise DualEvalError("A direction contract failed")
    if direction_for({"family": "P", "group_number": 2}) != "present_to_past" or source_side("P") != "donor" or source_side("C") != "base":
        raise DualEvalError("P/C routing contract failed")
    if requested_token_ids("has_had", "present_to_past", token_maps) != (2, 1) or requested_token_ids("is_was", "past_to_present", token_maps) != (3, 4):
        raise DualEvalError("token direction contract failed")
    sample = metric_summary([{"recovery": 0.5}, {"recovery": 1.5}], "recovery")
    if sample != {"count": 2, "mean_recovery": 1.0, "mean_absolute_recovery": 1.0, "direction_fraction": 1.0}:
        raise DualEvalError("summary key contract failed")
    if exact_price(rows=128, forwards=4, intervention_records=128)["example_evaluations"] != 256:
        raise DualEvalError("price contract failed")
    return True

