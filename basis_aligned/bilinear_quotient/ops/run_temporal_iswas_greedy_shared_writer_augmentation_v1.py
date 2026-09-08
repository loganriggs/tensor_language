#!/usr/bin/env python3
"""Greedy augmentation of the licensed three-head dual-command program."""

# BQGATE: EXPERIMENT pred_a_authority_capture_h3_replay_finiteness_and_exact_price pred_b_a_fixed_augmentation_meets_fit_quality_bars pred_c_fit_selected_augmentation_transfers_to_holdout pred_d_selected_augmentation_materially_improves_h3 pred_e_greedy_augmented_union_is_licensed
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import dual_command_head_module_factorial_contract as accounting
import run_temporal_iswas_dual_command_shared_head_module_factorial_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_greedy_shared_writer_augmentation_v1.json"
JOINT_RESULT = ROOT / "circuits/followups/temporal_iswas_three_head_union_joint_composition_v1_result.json"
ATLAS = ROOT / "circuits/followups/temporal_iswas_common_gauge_weight_interface_atlas_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_dual_command_shared_head_module_factorial_v1.py"
ACCOUNTING = ROOT / "ops/dual_command_head_module_factorial_contract.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_greedy_shared_writer_augmentation_v1_result.json"
EXPECTED = {"prior": "c57f209bc08212943cb8ec7d474d77a2cbc47ac5d2df4422bdf0a83fba0f9689",
            "joint_result": "309f30a665faaf2f90203cb7f00e8f38854698e41e01784694563b918bc2424f",
            "atlas": "d951420fef51f9d550237927072bfc721bd73112f87d5f716b3cb7c2ca3483ae",
            "parent_runner": "61cc52e73d51441512b073b75eef86f41ff5941c8f97770b7a062446c47ef849",
            "accounting": "985b824c7d3d8b622dfb92e3522f107e15e5e095b1f40a59a46122ad9af68496",
            "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498"}
ARMS = {
    "base_H3": {9: (1,), 11: (3,), 15: (5,)},
    "H3_plus_L8H1": {8: (1,), 9: (1,), 11: (3,), 15: (5,)},
    "H3_plus_L9H4": {9: (1, 4), 11: (3,), 15: (5,)},
    "H3_plus_both": {8: (1,), 9: (1, 4), 11: (3,), 15: (5,)},
}
ADDED = {"H3_plus_L8H1": 1, "H3_plus_L9H4": 1, "H3_plus_both": 2}
PRICE = {"checkpoint_loads": 1, "model_forwards": 9, "sequence_evaluations": 1152,
         "scored_token_positions": 2304, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_capture_h3_replay_finiteness_and_exact_price",
    "pred_b_a_fixed_augmentation_meets_fit_quality_bars",
    "pred_c_fit_selected_augmentation_transfers_to_holdout",
    "pred_d_selected_augmentation_materially_improves_h3",
    "pred_e_greedy_augmented_union_is_licensed",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def pooled(reports, role, phase, arm):
    return next(row for row in reports if row["role"] == role and row["phase"] == phase
                and row["template_id"] == "ALL" and row["arm"] == arm)


def passes(report, role):
    return (report["signed_recovery"] >= (.80 if role == "temporal" else .65)
            and report["cosine"] >= .98 and report["direction_agreement"] == 1.0
            and report["non_target_to_target_gold_norm"] <= .01)


def select_fit(reports):
    eligible = []
    for arm, added in ADDED.items():
        task = {role: pooled(reports, role, "FIT", arm) for role in ("temporal", "iswas")}
        if all(passes(task[role], role) for role in task):
            score = min(task["temporal"]["signed_recovery"] / .80,
                        task["iswas"]["signed_recovery"] / .65)
            eligible.append((added, -score, arm))
    return min(eligible)[2] if eligible else None


def main():
    files = {"prior": PRIOR, "joint_result": JOINT_RESULT, "atlas": ATLAS,
             "parent_runner": PARENT_RUNNER, "accounting": ACCOUNTING, "producer": PRODUCER}
    observed = {name: sha(path) for name, path in files.items()}
    prior, joint_result, atlas = (json.loads(path.read_text()) for path in (PRIOR, JOINT_RESULT, ATLAS))
    authority = bool(observed == EXPECTED and joint_result.get("terminal") == "distributed_joint_union_licensed"
                     and atlas.get("terminal") == "shared_weight_interface_candidates")
    rows = parent.candidate.build_rows(); endpoints, lookup = parent.endpoint_bank(rows)
    dryrun = {"candidate_id": prior.get("candidate_id"), "dryrun": True,
              "authority_ok": authority, "gpu_accessed": False, "model_loaded": False,
              "queue_touched": False, "arms": ARMS, "price": PRICE}
    if not authority: raise RuntimeError(f"greedy augmentation authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints], dtype=torch.long,
                          device=backend.device)
    capture_layers = tuple(sorted({layer for arm in ARMS.values() for layer in arm}))
    with torch.no_grad(): native_logits, captures = parent._forward(
        backend, tokens, capture_layers=capture_layers)
    native = {role: parent.margins(native_logits, endpoints, role) for role in ("temporal", "iswas")}
    records = []
    for role in ("temporal", "iswas"):
        pairs = parent.pair_indices(endpoints, lookup, role)
        positions = [endpoint[f"{role}_position"] - 1 for _row, _cell, endpoint in endpoints]
        for arm, patch_spec in ARMS.items():
            with torch.no_grad(): logits, _ = parent._forward(
                backend, tokens, captures=captures, pairs=pairs, positions=positions,
                patch_spec=patch_spec)
            patched = {name: parent.margins(logits, endpoints, name) for name in ("temporal", "iswas")}
            effect = patched[role] - native[role]; gold = native[role][pairs] - native[role]
            other = "iswas" if role == "temporal" else "temporal"
            collateral = patched[other] - native[other]
            for index, (row, cell, _endpoint) in enumerate(endpoints):
                records.append({"row_id": row["row_id"], "phase": row["phase"],
                    "template_id": row["template_id"], "cell": cell, "role": role, "arm": arm,
                    "target_effect": float(effect[index]), "target_gold": float(gold[index]),
                    "non_target_effect": float(collateral[index])})
    reports = accounting.aggregate(records)
    selected = select_fit(reports)
    replay_fields = ("signed_recovery", "cosine", "direction_agreement", "relative_residual",
                     "non_target_to_target_gold_norm")
    replay = []
    for role in ("temporal", "iswas"):
        for phase in ("FIT", "HOLDOUT"):
            new = pooled(reports, role, phase, "base_H3")
            old = next(row for row in joint_result["single_reports"] if row["role"] == role
                       and row["phase"] == phase and row["template_id"] == "ALL")
            replay.extend(abs(new[key] - old[key]) for key in replay_fields)
    replay_max = max(replay)
    counters = {"checkpoint_loads": 1, "model_forwards": 9,
                "sequence_evaluations": 9 * len(endpoints),
                "scored_token_positions": 18 * len(endpoints),
                "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
    finite = parent.finite_structure({"reports": reports, "replay": replay_max})
    A = bool(authority and counters == PRICE and finite and replay_max <= 1e-5
             and {label for arm in ARMS.values() for label in arm} == {8, 9, 11, 15})
    B = selected is not None
    C = bool(selected and all(passes(pooled(reports, role, "HOLDOUT", selected), role)
                              for role in ("temporal", "iswas")))
    improvement = {}
    for phase in ("FIT", "HOLDOUT"):
        if selected:
            changes = {role: pooled(reports, role, phase, selected)["signed_recovery"]
                       - pooled(reports, role, phase, "base_H3")["signed_recovery"]
                       for role in ("temporal", "iswas")}
            improvement[phase] = {"by_role": changes, "summed_recovery_gain": sum(changes.values())}
        else: improvement[phase] = {"by_role": {}, "summed_recovery_gain": None}
    D = bool(selected and all(min(improvement[phase]["by_role"].values()) >= -.02
                              and improvement[phase]["summed_recovery_gain"] >= .10
                              for phase in ("FIT", "HOLDOUT")))
    E = A and B and C and D
    predictions = dict(zip(PREDICTION_KEYS, (A, B, C, D, E)))
    terminal = "invalid" if not A else "augmented_union_licensed" if E else "writer_augmentation_null"
    result = {"schema": "temporal_iswas_greedy_shared_writer_augmentation_result_v1",
              "candidate_id": prior["candidate_id"], "started_utc": started_utc,
              "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
              "selected_arm": selected, "fit_selection_order": [item for item in ADDED],
              "improvement": improvement, "parent_replay_max_abs_error": replay_max,
              "reports": reports, "records": records, "predictions": predictions,
              "terminal": terminal, "price": counters}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("selected_arm", "improvement",
        "parent_replay_max_abs_error", "reports", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__": main()
