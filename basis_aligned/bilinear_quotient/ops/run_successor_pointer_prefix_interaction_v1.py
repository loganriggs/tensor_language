#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 3forwards120seq; exact pointer x prefix-coherence factorial;0fits.
"""A instrument; B answer preservation; C gated interaction; D independence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

import numpy as np


RUNNER = Path(__file__).resolve()
OPS = RUNNER.parent
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
ROWS = POLY / "SUCCESSOR_POINTER_PREFIX_INTERACTION_V1_ROWS.json"
PREREG = POLY / "SUCCESSOR_POINTER_PREFIX_INTERACTION_V1_PREREGISTRATION.md"
PRIOR = ROOT / "basis_aligned/bilinear_quotient/circuits/prior_art/successor_pointer_prefix_interaction_v1.json"
BINDING = POLY / "SUCCESSOR_POINTER_PREFIX_INTERACTION_V1_BINDING.json"
OUT = POLY / "SUCCESSOR_POINTER_PREFIX_INTERACTION_V1_RESULT.json"
REQUESTED_DRY = bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
DIRECTIONS = ("forward", "backward")
CONDITIONS = ("coherent", "early_swap_control", "late_swap_incoherent")
BARS = {
    "maximum_self_logit_absolute_error": 1e-5,
    "minimum_pointer_delta_norm": 1e-3,
    "minimum_native_and_self_accuracy": 0.75,
    "minimum_backward_late_interaction_scale": 0.20,
    "minimum_backward_late_win_fraction_gain": 0.20,
    "maximum_backward_early_control_scale": 0.15,
    "maximum_forward_late_control_scale": 0.25,
    "maximum_forward_early_control_scale": 0.15,
    "maximum_independent_effect_difference_scale": 0.15,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    for name, expected in binding["files"].items():
        path = {"rows": ROWS, "preregistration": PREREG, "prior_art": PRIOR}[name]
        if digest(path) != expected:
            raise RuntimeError(f"bound {name} changed")
    rows = json.loads(ROWS.read_text())
    if canonical(rows["rows"]) != rows["row_manifest_sha256"]:
        raise RuntimeError("row manifest changed")
    if rows["row_count"] != 30 or rows["intervention_count"] != 60:
        raise RuntimeError("row counts changed")
    if binding["bars"] != BARS or binding["row_manifest_sha256"] != rows["row_manifest_sha256"]:
        raise RuntimeError("binding bars or row manifest changed")
    return binding, rows


def compile_plan():
    binding, rows = load_bound()
    return {
        "schema": "successor_pointer_prefix_interaction_v1_plan",
        "candidate_id": "successor_pointer_prefix_interaction_v1",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "rows": rows["row_count"],
        "directed_interventions": rows["intervention_count"],
        "families": rows["families"],
        "conditions": rows["conditions"],
        "directions": rows["directions"],
        "price": binding["price"],
        "bars": BARS,
    }


def accuracy(logits, answer_ids):
    return float(np.mean(np.argmax(logits, axis=1) == np.asarray(answer_ids)))


def main():
    binding, frozen = load_bound()
    plan = compile_plan()
    if REQUESTED_DRY:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(600)
    sys.path.insert(0, str(ROOT / "basis_aligned/qk_mdl/algo_tasks/semantics_successor"))
    import torch
    import semlib

    torch.set_num_threads(2)
    started = time.perf_counter()
    model, cfg = semlib.get_model()
    device = next(model.parameters()).device
    rows = frozen["rows"]
    tokens = torch.tensor([row["token_ids"] for row in rows], dtype=torch.long, device=device)
    query = torch.tensor(
        [row["query_position"] for row in rows], dtype=torch.long, device=device,
    )
    last = torch.tensor([row["last_position"] for row in rows], dtype=torch.long, device=device)
    ar = torch.arange(len(rows), device=device)

    native_full, _ = semlib.run(model, cfg, tokens)
    native = native_full[ar, query].cpu().numpy()
    recipient_tokens = torch.tensor(
        [row["recipient_token_id"] for row in rows], dtype=torch.long, device=device,
    )
    recipient_v1 = semlib.v1_of_tokens(model, cfg, recipient_tokens)
    self_sub = {
        (layer, head): (recipient_v1[:, head], last)
        for layer in range(1, cfg["n_layer"]) for head in range(cfg["n_head"])
    }
    self_full, _ = semlib.run(model, cfg, tokens, v1_sub=self_sub)
    self_logits = self_full[ar, query].cpu().numpy()

    directed = [(index, direction) for index in range(len(rows)) for direction in DIRECTIONS]
    directed_tokens = tokens[torch.tensor([index for index, _ in directed], device=device)]
    directed_last = last[torch.tensor([index for index, _ in directed], device=device)]
    donor_tokens = torch.tensor([
        rows[index]["donors"][direction]["token_id"] for index, direction in directed
    ], dtype=torch.long, device=device)
    donor_v1 = semlib.v1_of_tokens(model, cfg, donor_tokens)
    donor_sub = {
        (layer, head): (donor_v1[:, head], directed_last)
        for layer in range(1, cfg["n_layer"]) for head in range(cfg["n_head"])
    }
    donor_full, _ = semlib.run(model, cfg, directed_tokens, v1_sub=donor_sub)
    directed_query = query[torch.tensor(
        [index for index, _ in directed], device=device,
    )]
    donor_logits = donor_full[
        torch.arange(len(directed), device=device), directed_query
    ].cpu().numpy()

    self_error = float(np.max(np.abs(native - self_logits)))
    delta_norms = (donor_v1 - recipient_v1[
        torch.tensor([index for index, _ in directed], device=device)
    ]).float().flatten(1).norm(dim=1).cpu().numpy()
    records = []
    for di, (index, direction) in enumerate(directed):
        row = rows[index]
        recipient_answer = int(row["recipient_answer_id"])
        donor_answer = int(row["donors"][direction]["answer_id"])
        before = self_logits[index]
        after = donor_logits[di]
        records.append({
            "row_id": row["row_id"],
            "family": row["family"],
            "final_index": row["final_index"],
            "condition": row["condition"],
            "direction": direction,
            "native_recipient_correct": bool(np.argmax(native[index]) == recipient_answer),
            "self_recipient_correct": bool(np.argmax(before) == recipient_answer),
            "donor_answer_win": bool(np.argmax(after) == donor_answer),
            "native_recipient_minus_donor_margin": float(
                native[index, recipient_answer] - native[index, donor_answer]
            ),
            "self_donor_margin": float(before[donor_answer] - before[recipient_answer]),
            "donor_margin": float(after[donor_answer] - after[recipient_answer]),
            "pointer_effect": float(
                (after[donor_answer] - after[recipient_answer])
                - (before[donor_answer] - before[recipient_answer])
            ),
            "pointer_delta_norm": float(delta_norms[di]),
        })

    capability = {}
    for family in frozen["families"]:
        capability[family] = {}
        for condition in CONDITIONS:
            indices = [i for i, row in enumerate(rows)
                       if row["family"] == family and row["condition"] == condition]
            answers = [rows[i]["recipient_answer_id"] for i in indices]
            capability[family][condition] = {
                "n": len(indices),
                "native_accuracy": accuracy(native[indices], answers),
                "self_accuracy": accuracy(self_logits[indices], answers),
            }

    interactions = {}
    for family in frozen["families"]:
        interactions[family] = {}
        for direction in DIRECTIONS:
            coherent_rows = [
                row for row in records
                if row["family"] == family and row["direction"] == direction
                and row["condition"] == "coherent"
            ]
            scale = statistics.median(abs(row["native_recipient_minus_donor_margin"])
                                      for row in coherent_rows)
            if scale <= 1e-12:
                raise RuntimeError("dead native margin scale")
            by_key = {
                (row["final_index"], row["condition"]): row
                for row in records if row["family"] == family and row["direction"] == direction
            }
            effects = {
                condition: statistics.median(
                    by_key[(index, condition)]["pointer_effect"]
                    for index in sorted({r["final_index"] for r in rows if r["family"] == family})
                ) for condition in CONDITIONS
            }
            wins = {
                condition: statistics.fmean(
                    by_key[(index, condition)]["donor_answer_win"]
                    for index in sorted({r["final_index"] for r in rows if r["family"] == family})
                ) for condition in CONDITIONS
            }
            paired = {}
            for condition in ("early_swap_control", "late_swap_incoherent"):
                differences = [
                    by_key[(index, condition)]["pointer_effect"]
                    - by_key[(index, "coherent")]["pointer_effect"]
                    for index in sorted({r["final_index"] for r in rows if r["family"] == family})
                ]
                paired[condition] = {
                    "median_difference": statistics.median(differences),
                    "median_difference_in_native_margin_units": statistics.median(differences) / scale,
                }
            interactions[family][direction] = {
                "native_margin_scale": scale,
                "median_pointer_effect": effects,
                "donor_answer_win_fraction": wins,
                "paired_differences": paired,
            }

    finite = all(np.isfinite(value) for value in [self_error, *delta_norms]) and all(
        np.isfinite(row[key]) for row in records
        for key in ("self_donor_margin", "donor_margin", "pointer_effect")
    )
    pred_a = finite and self_error <= BARS["maximum_self_logit_absolute_error"] and float(
        np.min(delta_norms)) >= BARS["minimum_pointer_delta_norm"]
    pred_b = all(
        cell[key] >= BARS["minimum_native_and_self_accuracy"]
        for family in capability.values() for cell in family.values()
        for key in ("native_accuracy", "self_accuracy")
    )
    pred_c = True
    pred_d = True
    for family in frozen["families"]:
        back = interactions[family]["backward"]
        forward = interactions[family]["forward"]
        pred_c &= (
            back["paired_differences"]["late_swap_incoherent"]["median_difference_in_native_margin_units"]
            >= BARS["minimum_backward_late_interaction_scale"]
            and back["donor_answer_win_fraction"]["late_swap_incoherent"]
            - back["donor_answer_win_fraction"]["coherent"]
            >= BARS["minimum_backward_late_win_fraction_gain"]
            and abs(back["paired_differences"]["early_swap_control"]["median_difference_in_native_margin_units"])
            <= BARS["maximum_backward_early_control_scale"]
            and abs(forward["paired_differences"]["late_swap_incoherent"]["median_difference_in_native_margin_units"])
            <= BARS["maximum_forward_late_control_scale"]
            and abs(forward["paired_differences"]["early_swap_control"]["median_difference_in_native_margin_units"])
            <= BARS["maximum_forward_early_control_scale"]
        )
        for direction in DIRECTIONS:
            for condition in ("early_swap_control", "late_swap_incoherent"):
                pred_d &= abs(interactions[family][direction]["paired_differences"][condition][
                    "median_difference_in_native_margin_units"
                ]) <= BARS["maximum_independent_effect_difference_scale"]
    predictions = {
        "pred_a_instrument_live": bool(pred_a),
        "pred_b_answer_preserving_prefix_capability": bool(pred_b),
        "pred_c_coherent_run_gate": bool(pred_a and pred_b and pred_c),
        "pred_d_independent_pointer": bool(pred_a and pred_b and pred_d),
    }
    terminal = (
        "invalid" if not pred_a or not pred_b else
        "coherent_run_gate" if pred_c else
        "independent_pointer" if pred_d else
        "mixed_interaction"
    )
    result = {
        "schema": "successor_pointer_prefix_interaction_v1_result",
        "candidate_id": "successor_pointer_prefix_interaction_v1",
        "terminal": terminal,
        "serial_seconds": time.perf_counter() - started,
        "plan": plan,
        "binding_sha256": digest(BINDING),
        "checkpoint": "bilin18",
        "exactness": {
            "maximum_self_logit_absolute_error": self_error,
            "minimum_pointer_delta_norm": float(np.min(delta_norms)),
        },
        "capability": capability,
        "interactions": interactions,
        "predictions": predictions,
        "records": records,
        "counts": {
            "model_forwards": 3,
            "sequence_evaluations": 120,
            "fits": 0,
            "backwards": 0,
            "parameter_updates": 0,
        },
        "claim_boundary": (
            "factorial screen of exact pointer substitution and matched prefix coherence; "
            "no downstream-gate localization, extraction, adoption, rank, or quantization claim"
        ),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(OUT), "terminal": terminal, "predictions": predictions,
        "exactness": result["exactness"], "serial_seconds": result["serial_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
