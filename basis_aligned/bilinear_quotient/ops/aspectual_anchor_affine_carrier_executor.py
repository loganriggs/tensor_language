"""Reusable executor for frozen source-margin aspectual carrier actuators."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import statistics
import time

import run_aspectual_anchor_base_margin_affine_carrier_actuation_v1 as affine


ROOT = Path(__file__).resolve().parents[1]
SITE = "resid:18"
FAMILIES = ("A1", "A2", "P", "C")


class ExecutorError(RuntimeError):
    pass


def summarize(records, key):
    values = [record[key] for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ExecutorError(f"missing/nonfinite {key}")
    return {
        "count": len(values),
        f"mean_{key}": statistics.fmean(values),
        f"mean_absolute_{key}": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def measure(rows, spec, rank1, *, coefficients, token_ids):
    """Measure one sealed population; capability is defined only on scored task sides."""
    if token_ids != affine.TOKEN_IDS:
        raise ExecutorError("fixed token IDs changed")
    started_utc, started = affine.parent.scalar_parent.empirical.component_parent.utc_now(), time.perf_counter()
    backend = affine.parent.producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    q = torch.tensor(rank1["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    if q.shape != (1152,) or abs(float(q.norm()) - 1.0) > 1.0e-4 or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != rank1["basis"]["sha256"]:
        raise ExecutorError("basis reconstruction failed")
    a1_rows = [row for row in rows if row["transform_id"] == "A1"]
    head_ok, head_error = affine.parent.das.verify_head(backend, a1_rows[:8], SITE)
    forward_calls, evaluations, head_evaluations = 1, 8, 0
    target_scale = float(rank1["score"]["families"]["target_scale"])
    records = []
    capability_counts = {
        "A_base": [0, 0],
        "A_donor": [0, 0],
        "P_source": [0, 0],
        "C_actual_base": [0, 0],
    }

    for family in FAMILIES:
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in affine.parent.producer._chunks(family_rows, spec.batch_size):
            base, donor, _ = affine.parent.das.capture_site(backend, chunk, SITE)
            forward_calls += 2
            evaluations += 2 * len(chunk)
            for i, row in enumerate(chunk):
                direction = affine.direction_for(row, family)
                source = donor[i] if family == "P" else base[i]
                confidence = affine.fixed_confidence(backend, source, direction)
                alpha = coefficients[direction]["intercept"] + coefficients[direction]["slope"] * confidence
                if family in ("A1", "A2"):
                    current_id = token_ids["has"] if direction == "present_to_past" else token_ids["had"]
                    other_id = token_ids["had"] if direction == "present_to_past" else token_ids["has"]
                    if (row["base_answer_id"], row["base_foil_id"], row["donor_answer_id"], row["donor_foil_id"]) != (current_id, other_id, other_id, current_id):
                        raise ExecutorError("A row does not match fixed has/had interface")
                    donor_original = affine.parent.pair_logits(backend, donor[i], other_id, current_id)
                    capability_counts["A_base"][0] += int(confidence > 0.0)
                    capability_counts["A_base"][1] += 1
                    capability_counts["A_donor"][0] += int(donor_original[0] > donor_original[1])
                    capability_counts["A_donor"][1] += 1
                    answer_id, foil_id = other_id, current_id
                    base_margin = -confidence
                    donor_margin = donor_original[0] - donor_original[1]
                    head_evaluations += 3
                elif family == "P":
                    capability_counts["P_source"][0] += int(confidence > 0.0)
                    capability_counts["P_source"][1] += 1
                    answer_id = token_ids["had"] if direction == "present_to_past" else token_ids["has"]
                    foil_id = token_ids["has"] if direction == "present_to_past" else token_ids["had"]
                    base_margin, donor_margin = -confidence, None
                    head_evaluations += 2
                else:
                    original = affine.parent.pair_logits(backend, source, row["base_answer_id"], row["base_foil_id"])
                    actual_margin = original[0] - original[1]
                    capability_counts["C_actual_base"][0] += int(actual_margin > 0.0)
                    capability_counts["C_actual_base"][1] += 1
                    answer_id, foil_id = row["base_answer_id"], row["base_foil_id"]
                    base_margin, donor_margin = actual_margin, None
                    head_evaluations += 3
                patched = affine.parent.pair_logits(backend, source + alpha * q, answer_id, foil_id)
                patched_margin = patched[0] - patched[1]
                record = {
                    "family": family,
                    "row_id": str(row["row_id"]),
                    "direction": direction,
                    "fixed_has_had_confidence": confidence,
                    "alpha": alpha,
                    "base_margin": base_margin,
                    "patched_margin": patched_margin,
                    "confirmation_donor_activation_used_by_actuator": False,
                    "row_target_or_foil_used_to_select_alpha": False,
                }
                if family in ("A1", "A2"):
                    record["donor_reference_margin"] = donor_margin
                    record["recovery"] = (patched_margin - base_margin) / (donor_margin - base_margin)
                elif family == "P":
                    record["margin_reflection_fraction"] = (patched_margin - base_margin) / (-2.0 * base_margin)
                else:
                    record["normalized_unrelated_effect"] = abs(patched_margin - base_margin) / target_scale
                records.append(record)
            forward_calls += 1
            evaluations += len(chunk)

    capability = all(correct == total for correct, total in capability_counts.values())
    by_family = {family: [record for record in records if record["family"] == family] for family in FAMILIES}
    summaries = {
        "A1": summarize(by_family["A1"], "recovery"),
        "A2": summarize(by_family["A2"], "recovery"),
        "P": summarize(by_family["P"], "margin_reflection_fraction"),
        "C": summarize(by_family["C"], "normalized_unrelated_effect"),
    }
    return {
        "started_utc": started_utc,
        "finished_utc": affine.parent.scalar_parent.empirical.component_parent.utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "records": records,
        "families": summaries,
        "capability": capability,
        "capability_counts": {key: {"correct": value[0], "total": value[1]} for key, value in capability_counts.items()},
        "head_control": {"passed": head_ok, "max_abs_difference": head_error},
        "target_scale": target_scale,
        "forward_calls": forward_calls,
        "example_evaluations": evaluations,
        "selected_head_pair_evaluations": head_evaluations,
    }
