"""Reusable two-site executor for frozen aspectual upstream-gain programs."""

from __future__ import annotations

import hashlib
import math
import time

import run_aspectual_anchor_base_margin_affine_carrier_actuation_v1 as affine
import run_aspectual_anchor_resid10_margin_to_carrier_gain_v1 as upstream


def _manual_pair(backend, state, first_id, second_id):
    normalized = backend.F.rms_norm(state, (backend.model.config.n_embd,))
    first_raw = (normalized * backend.model.lm_head.weight[first_id]).sum()
    second_raw = (normalized * backend.model.lm_head.weight[second_id]).sum()
    return float(30.0 * backend.torch.tanh(first_raw / 30.0)), float(30.0 * backend.torch.tanh(second_raw / 30.0))


def measure(rows, screen_spec, rank1, *, coefficients):
    started_utc, started = affine.parent.scalar_parent.empirical.component_parent.utc_now(), time.perf_counter()
    backend = affine.parent.producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    q = torch.tensor(rank1["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    if q.shape != (1152,) or abs(float(q.norm()) - 1.0) > 1.0e-4 or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != rank1["basis"]["sha256"]:
        raise RuntimeError("basis reconstruction failed")
    head_ok, head_error = affine.parent.das.verify_head(backend, [row for row in rows if row["transform_id"] == "A1"][:8], "resid:18")
    counted_forwards, evaluations, pair_evaluations = 1, 8, 0
    local_error = 0.0
    target_scale = float(rank1["score"]["families"]["target_scale"])
    capability_records, records = [], []

    for family in ("A1", "A2", "P", "C"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in affine.parent.producer._chunks(family_rows, screen_spec.batch_size):
            base10, donor10, _ = affine.parent.das.capture_site(backend, chunk, "resid:10")
            base18, donor18, _ = affine.parent.das.capture_site(backend, chunk, "resid:18")
            counted_forwards += 4
            evaluations += 4 * len(chunk)
            for i, row in enumerate(chunk):
                direction = affine.direction_for(row, family)
                source10 = donor10[i] if family == "P" else base10[i]
                source18 = donor18[i] if family == "P" else base18[i]
                current_id = upstream.TOKEN_IDS["has"] if direction == "present_to_past" else upstream.TOKEN_IDS["had"]
                other_id = upstream.TOKEN_IDS["had"] if direction == "present_to_past" else upstream.TOKEN_IDS["has"]
                shared_pair = affine.parent.pair_logits(backend, source10, current_id, other_id)
                manual = _manual_pair(backend, source10, current_id, other_id)
                local_error = max(local_error, abs(shared_pair[0] - manual[0]), abs(shared_pair[1] - manual[1]))
                contrast10 = shared_pair[0] - shared_pair[1]
                alpha = coefficients[direction]["intercept"] + coefficients[direction]["slope"] * contrast10
                if family in ("A1", "A2"):
                    base_original = affine.parent.pair_logits(backend, base18[i], current_id, other_id)
                    donor_original = affine.parent.pair_logits(backend, donor18[i], other_id, current_id)
                    base_margin = -(base_original[0] - base_original[1])
                    donor_margin = donor_original[0] - donor_original[1]
                    answer_id, foil_id = other_id, current_id
                    capability_records.extend([
                        {"family": family, "direction": direction, "side": "base", "correct": base_margin < 0.0},
                        {"family": family, "direction": direction, "side": "donor", "correct": donor_margin > 0.0},
                    ])
                    pair_evaluations += 4
                elif family == "P":
                    original = affine.parent.pair_logits(backend, source18, current_id, other_id)
                    current_margin = original[0] - original[1]
                    base_margin, donor_margin = -current_margin, None
                    answer_id, foil_id = other_id, current_id
                    capability_records.append({"family": family, "direction": direction, "side": "source", "correct": current_margin > 0.0})
                    pair_evaluations += 3
                else:
                    original = affine.parent.pair_logits(backend, source18, row["base_answer_id"], row["base_foil_id"])
                    base_margin, donor_margin = original[0] - original[1], None
                    answer_id, foil_id = row["base_answer_id"], row["base_foil_id"]
                    capability_records.append({"family": family, "direction": direction, "side": "actual_base", "correct": base_margin > 0.0})
                    pair_evaluations += 3
                patched = affine.parent.pair_logits(backend, source18 + alpha * q, answer_id, foil_id)
                patched_margin = patched[0] - patched[1]
                record = {"family": family, "row_id": str(row["row_id"]), "direction": direction, "resid10_unembedding_contrast": contrast10, "alpha": alpha, "base_margin": base_margin, "patched_margin": patched_margin, "confirmation_resid18_margin_used_to_select_alpha": False, "confirmation_donor_activation_used_to_select_alpha": False, "row_target_or_foil_used_to_select_alpha": False}
                if family in ("A1", "A2"):
                    record["donor_reference_margin"] = donor_margin
                    record["recovery"] = (patched_margin - base_margin) / (donor_margin - base_margin)
                elif family == "P":
                    record["margin_reflection_fraction"] = (patched_margin - base_margin) / (-2.0 * base_margin)
                else:
                    record["normalized_unrelated_effect"] = abs(patched_margin - base_margin) / target_scale
                records.append(record)
            counted_forwards += 1
            evaluations += len(chunk)

    summaries = {
        "A1": upstream.summarize([r for r in records if r["family"] == "A1"], "recovery"),
        "A2": upstream.summarize([r for r in records if r["family"] == "A2"], "recovery"),
        "P": upstream.summarize([r for r in records if r["family"] == "P"], "margin_reflection_fraction"),
        "C": upstream.summarize([r for r in records if r["family"] == "C"], "normalized_unrelated_effect"),
    }
    capability_cells = []
    for family in ("A1", "A2", "P", "C"):
        for direction in sorted({r["direction"] for r in capability_records if r["family"] == family}):
            cell = [r for r in capability_records if r["family"] == family and r["direction"] == direction]
            accuracy = sum(r["correct"] for r in cell) / len(cell)
            threshold = 0.75 if family == "C" else 0.85
            capability_cells.append({"family": family, "direction": direction, "correct": sum(r["correct"] for r in cell), "total": len(cell), "accuracy": accuracy, "threshold": threshold, "passed": accuracy >= threshold})
    return {"started_utc": started_utc, "finished_utc": affine.parent.scalar_parent.empirical.component_parent.utc_now(), "serial_seconds": time.perf_counter() - started, "records": records, "families": summaries, "capability_cells": capability_cells, "capability": all(cell["passed"] for cell in capability_cells), "local_error": local_error, "head_control": {"passed": head_ok, "max_abs_difference": head_error}, "target_scale": target_scale, "counted_forwards": counted_forwards, "example_evaluations": evaluations, "selected_head_pair_evaluations": pair_evaluations}
