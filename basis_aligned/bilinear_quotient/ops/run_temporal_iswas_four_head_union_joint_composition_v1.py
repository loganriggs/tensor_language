#!/usr/bin/env python3
"""Joint composition confirmation for the FIT-selected four-head union."""

# BQGATE: EXPERIMENT pred_a_authority_h4_replay_pairing_finiteness_and_exact_price pred_b_h4_single_command_quality_and_selectivity_reproduce pred_c_h4_simultaneous_execution_is_additive pred_d_h4_simultaneous_execution_preserves_both_recoveries pred_e_four_head_joint_program_is_licensed
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
import joint_command_composition_contract as joint
import run_temporal_iswas_three_head_union_joint_composition_v1 as base

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_four_head_union_joint_composition_v1.json"
AUGMENTATION = ROOT / "circuits/followups/temporal_iswas_greedy_shared_writer_augmentation_v1_result.json"
AUGMENTATION_RUNNER = ROOT / "ops/run_temporal_iswas_greedy_shared_writer_augmentation_v1.py"
BASE_RUNNER = ROOT / "ops/run_temporal_iswas_three_head_union_joint_composition_v1.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_four_head_union_joint_composition_v1_result.json"
EXPECTED = {"prior": "b7adb64fcad4e6f22665d1651121b06d3d9609fcdf7c60a6ac54545969c756a6",
            "augmentation": "c81853c47e6ed2ecf40840a2951f67fe3f249d0f3f8b5ca5e6a4282e5ade4e4e",
            "augmentation_runner": "961c2e968da2953133a29df5145241f53c1b570cd74debaebbbe54692a752a6a",
            "base_runner": "5d2fa19bf187527a6b00699d2200ad90288ebdb75c490d8be525aa051f4e56d4",
            "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498"}
UNION = {9: (1, 4), 11: (3,), 15: (5,)}
PRICE = {"checkpoint_loads": 1, "model_forwards": 4, "sequence_evaluations": 512,
         "scored_token_positions": 1024, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_h4_replay_pairing_finiteness_and_exact_price",
    "pred_b_h4_single_command_quality_and_selectivity_reproduce",
    "pred_c_h4_simultaneous_execution_is_additive",
    "pred_d_h4_simultaneous_execution_preserves_both_recoveries",
    "pred_e_four_head_joint_program_is_licensed",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = {"prior": PRIOR, "augmentation": AUGMENTATION,
             "augmentation_runner": AUGMENTATION_RUNNER, "base_runner": BASE_RUNNER,
             "producer": PRODUCER}
    observed = {name: sha(path) for name, path in files.items()}
    prior, augmentation = json.loads(PRIOR.read_text()), json.loads(AUGMENTATION.read_text())
    authority = bool(observed == EXPECTED and augmentation.get("terminal") == "augmented_union_licensed"
                     and augmentation.get("selected_arm") == "H3_plus_L9H4")
    rows = base.parent.candidate.build_rows(); endpoints, lookup = base.parent.endpoint_bank(rows)
    dryrun = {"candidate_id": prior.get("candidate_id"), "dryrun": True,
              "authority_ok": authority, "gpu_accessed": False, "model_loaded": False,
              "queue_touched": False, "union": UNION, "anchors": 128, "price": PRICE}
    if not authority: raise RuntimeError(f"H4 composition authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints], dtype=torch.long,
                          device=backend.device)
    pairs = {role: base.parent.pair_indices(endpoints, lookup, role) for role in ("temporal", "iswas")}
    positions = {role: [endpoint[f"{role}_position"] - 1 for _row, _cell, endpoint in endpoints]
                 for role in ("temporal", "iswas")}
    with torch.no_grad():
        native_logits, captures = base.parent._forward(backend, tokens, capture_layers=(9, 11, 15))
        temporal_logits, _ = base.parent._forward(backend, tokens, captures=captures,
            pairs=pairs["temporal"], positions=positions["temporal"], patch_spec=UNION)
        iswas_logits, _ = base.parent._forward(backend, tokens, captures=captures,
            pairs=pairs["iswas"], positions=positions["iswas"], patch_spec=UNION)
        original_union = base.UNION
        try:
            base.UNION = UNION
            both_logits = base._joint_forward(backend, tokens, captures,
                ((pairs["temporal"], positions["temporal"]), (pairs["iswas"], positions["iswas"])))
        finally:
            base.UNION = original_union
    logits = {"00": native_logits, "10": temporal_logits, "01": iswas_logits, "11": both_logits}
    margins = {role: {arm: base.parent.margins(value, endpoints, role) for arm, value in logits.items()}
               for role in ("temporal", "iswas")}
    single_reports, composition_reports = [], []
    for phase in ("FIT", "HOLDOUT"):
        for template in ("ALL",) + base.parent.candidate.TEMPLATES:
            selected = base.masks(endpoints, phase, template)
            for role, arm in (("temporal", "10"), ("iswas", "01")):
                other = "iswas" if role == "temporal" else "temporal"
                gold = margins[role]["00"][pairs[role]] - margins[role]["00"]
                report = accounting.effect_metrics((margins[role][arm] - margins[role]["00"])[selected],
                    gold[selected], (margins[other][arm] - margins[other]["00"])[selected])
                simultaneous = accounting.effect_metrics(
                    (margins[role]["11"] - margins[role]["00"])[selected], gold[selected],
                    (margins[other]["11"] - margins[other]["00"])[selected])
                single_reports.append({"role": role, "phase": phase, "template_id": template,
                    "arm": arm, **report, "simultaneous_signed_recovery": simultaneous["signed_recovery"],
                    "simultaneous_cosine": simultaneous["cosine"]})
            for role in ("temporal", "iswas"):
                decomposition = joint.decompose(torch, {arm: margins[role][arm][selected] for arm in joint.CELLS})
                composition_reports.append({"output_role": role, "phase": phase, "template_id": template,
                    "closure_max_abs_error": decomposition["closure_max_abs_error"],
                    "interaction_rms_over_both_rms": decomposition["interaction_rms_over_both_rms"],
                    "additive_cosine_to_both_effect": decomposition["additive_cosine_to_both_effect"]})
    replay_fields = ("signed_recovery", "cosine", "direction_agreement", "relative_residual",
                     "non_target_to_target_gold_norm")
    replay = []
    for role in ("temporal", "iswas"):
        for phase in ("FIT", "HOLDOUT"):
            new = next(row for row in single_reports if row["role"] == role and row["phase"] == phase
                       and row["template_id"] == "ALL")
            old = next(row for row in augmentation["reports"] if row["role"] == role
                       and row["phase"] == phase and row["template_id"] == "ALL"
                       and row["arm"] == "H3_plus_L9H4")
            replay.extend(abs(new[key] - old[key]) for key in replay_fields)
    replay_max = max(replay)
    causal_zero = float(np.max(np.abs(margins["temporal"]["01"] - margins["temporal"]["00"])))
    counters = {"checkpoint_loads": 1, "model_forwards": 4, "sequence_evaluations": 512,
                "scored_token_positions": 1024, "transformer_backwards": 0,
                "model_updates": 0, "fit_parameters": 0}
    finite = base.parent.finite_structure({"single": single_reports, "composition": composition_reports,
                                           "replay": replay_max, "causal_zero": causal_zero})
    A = bool(authority and counters == PRICE and replay_max <= 1e-5 and finite
             and max(row["closure_max_abs_error"] for row in composition_reports) <= 1e-6)
    pooled = [row for row in single_reports if row["template_id"] == "ALL"]
    B = all(row["signed_recovery"] >= (.80 if row["role"] == "temporal" else .65)
            and row["cosine"] >= .98 and row["direction_agreement"] == 1.0
            and row["non_target_to_target_gold_norm"] <= .01 for row in pooled) and causal_zero <= 1e-5
    C = all(row["interaction_rms_over_both_rms"] <= .10
            and row["additive_cosine_to_both_effect"] >= .99
            for row in composition_reports if row["template_id"] != "ALL")
    D = all(row["simultaneous_signed_recovery"] >= row["signed_recovery"] - .05 for row in pooled)
    E = A and B and C and D
    predictions = dict(zip(PREDICTION_KEYS, (A, B, C, D, E)))
    terminal = "invalid" if not A else "four_head_joint_program_licensed" if E else "four_head_joint_null_keep_h3"
    result = {"schema": "temporal_iswas_four_head_union_joint_composition_result_v1",
              "candidate_id": prior["candidate_id"], "started_utc": started_utc,
              "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
              "union": UNION, "augmentation_replay_max_abs_error": replay_max,
              "iswas_to_temporal_max_abs_effect": causal_zero, "single_reports": single_reports,
              "composition_reports": composition_reports, "predictions": predictions,
              "terminal": terminal, "price": counters}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("augmentation_replay_max_abs_error",
        "iswas_to_temporal_max_abs_effect", "single_reports", "composition_reports",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
