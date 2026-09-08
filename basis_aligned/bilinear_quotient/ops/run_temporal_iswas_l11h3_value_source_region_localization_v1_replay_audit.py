#!/usr/bin/env python3
"""Audit the invalid localization receipt's mixed-target replay comparison."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_finiteness_and_exact_price pred_b_like_for_like_command_gold_replay_is_exact pred_c_mixed_target_bug_exactly_explains_invalid_receipt pred_d_localization_science_is_recovered_without_rescoring
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import dual_command_head_module_factorial_contract as accounting
import run_temporal_iswas_l11h3_value_source_region_localization_v1 as loc


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_l11h3_value_source_region_localization_v1_replay_audit.json"
INVALID = ROOT / "circuits/followups/temporal_iswas_l11h3_value_source_region_localization_v1_result.json"
LOCALIZATION_RUNNER = ROOT / "ops/run_temporal_iswas_l11h3_value_source_region_localization_v1.py"
FACTOR = ROOT / "circuits/followups/temporal_iswas_h4_reader_factor_factorial_v1_result.json"
ORIGINAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_v2.py"
ACCOUNTING = ROOT / "ops/dual_command_head_module_factorial_contract.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_l11h3_value_source_region_localization_v1_replay_audit_result.json"
EXPECTED = {
    "prior": "d7db98c77ba92d46b9169457d24926fe4362c6462a02d06b8a9f42133d803445",
    "invalid": "74d7e51e227c9799bb6e079edff1fb8b5c49e9e6725960958e0269e19d74882e",
    "localization_runner": "ba51ffec9601444de6cbef51713cf7136b76c6207666e8a6c5364297a7f3296d",
    "factor": "5a1f121d82d799292613d7e386a3d16429457c80ba55a56aaca58a3386118b9b",
    "original_builder": "72da11860ce5bf1c03ea126bc10fcb6c46dd2648169edb00a1e9e1dbeee8a0d0",
    "accounting": "985b824c7d3d8b622dfb92e3522f107e15e5e095b1f40a59a46122ad9af68496",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
PRICE = {"checkpoint_loads": 1, "model_forwards": 3, "sequence_evaluations": 384,
         "scored_token_positions": 768, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
FLOAT_FIELDS = ("signed_recovery", "cosine", "direction_agreement", "relative_residual",
                "non_target_to_target_gold_norm")
PREDICTION_KEYS = (
    "pred_a_authority_pairing_finiteness_and_exact_price",
    "pred_b_like_for_like_command_gold_replay_is_exact",
    "pred_c_mixed_target_bug_exactly_explains_invalid_receipt",
    "pred_d_localization_science_is_recovered_without_rescoring",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def find(rows, role, phase, arm):
    return next(row for row in rows if row["role"] == role and row["phase"] == phase
                and row["template_id"] == "ALL" and row["arm"] == arm)


def run_audit(backend):
    rows = loc.original.build_rows()
    endpoints, lookup = loc.parent.endpoint_bank(rows)
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    torch = backend.torch
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints],
                          dtype=torch.long, device=backend.device)
    pairs = {role: loc.parent.pair_indices(endpoints, lookup, role)
             for role in ("temporal", "iswas")}
    queries = {role: [endpoint[f"{role}_position"] - 1
                      for _row, _cell, endpoint in endpoints]
               for role in ("temporal", "iswas")}
    regions = {role: [loc.source_regions(endpoint["ids"],
                endpoints[pairs[role][index]][2]["ids"], queries[role][index])["full_prefix"]
                for index, (_row, _cell, endpoint) in enumerate(endpoints)]
               for role in ("temporal", "iswas")}
    with torch.no_grad():
        native_logits, captures = loc._forward(backend, tokens, capture=True)
    native = {role: loc.margins(native_logits, endpoints, role)
              for role in ("temporal", "iswas")}
    effects, collateral = {}, {}
    for role in ("temporal", "iswas"):
        with torch.no_grad():
            logits, _ = loc._forward(backend, tokens, donor=captures["v"],
                                     pairs=pairs[role], region_rows=regions[role])
        effects[role] = loc.margins(logits, endpoints, role) - native[role]
        other = "iswas" if role == "temporal" else "temporal"
        collateral[role] = loc.margins(logits, endpoints, other) - native[other]
    return endpoints, pairs, native, effects, collateral, captures, int(tokens.shape[1])


def main():
    paths = {"prior": PRIOR, "invalid": INVALID, "localization_runner": LOCALIZATION_RUNNER,
             "factor": FACTOR, "original_builder": ORIGINAL_BUILDER,
             "accounting": ACCOUNTING, "producer": PRODUCER}
    observed = {name: sha(path) for name, path in paths.items()}
    invalid = json.loads(INVALID.read_text())
    factor = json.loads(FACTOR.read_text())
    authority = bool(observed == EXPECTED and invalid.get("terminal") == "invalid"
        and invalid.get("predictions", {}).get(
            "pred_a_authority_pairing_region_coverage_full_replay_finiteness_and_exact_price") is False
        and factor.get("terminal") == "partial_reader_factor_split")
    dry = {"candidate_id": json.loads(PRIOR.read_text())["candidate_id"], "dryrun": True,
           "authority_ok": authority, "gpu_accessed": False, "model_loaded": False,
           "queue_touched": False, "price": PRICE}
    if not authority:
        raise RuntimeError(f"replay-audit authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = loc.producer.Bilin18TorchBackend.load("cuda")
    endpoints, pairs, native, effects, collateral, captures, token_length = run_audit(backend)
    audit_rows = []
    like_errors, stored_errors, wrong_errors = [], [], []
    invalid_reports = invalid["panels"]["original"]["reports"]
    for role in ("temporal", "iswas"):
        gold = native[role][pairs[role]] - native[role]
        for phase in ("FIT", "HOLDOUT"):
            selected = np.asarray([row["phase"] == phase
                                   for row, _cell, _endpoint in endpoints])
            gold_report = accounting.effect_metrics(effects[role][selected], gold[selected],
                                                     collateral[role][selected])
            self_report = accounting.effect_metrics(effects[role][selected], effects[role][selected],
                                                     collateral[role][selected])
            old = find(factor["reports"], role, phase, "L11H3:v_prefix")
            stored = find(invalid_reports, role, phase, "full_prefix")
            field_errors = {key: abs(float(gold_report[key]) - float(old[key]))
                            for key in FLOAT_FIELDS}
            count_match = int(gold_report["count"]) == int(old["count"])
            like_errors.extend(field_errors.values())
            stored_field_errors = {
                "signed_recovery": abs(gold_report["signed_recovery"]
                                       - stored["command_gold_signed_recovery"]),
                "non_target_to_target_gold_norm": abs(
                    gold_report["non_target_to_target_gold_norm"]
                    - stored["command_gold_non_target_norm_ratio"]),
            }
            stored_errors.extend(stored_field_errors.values())
            wrong_errors.extend((abs(gold_report["signed_recovery"] - old["signed_recovery"]),
                                 abs(self_report["cosine"] - old["cosine"]),
                                 abs(self_report["direction_agreement"]
                                     - old["direction_agreement"])))
            audit_rows.append({"role": role, "phase": phase,
                "command_gold_metrics": gold_report, "immutable_factor_metrics": old,
                "like_for_like_abs_errors": field_errors, "count_match": count_match,
                "invalid_stored_command_gold_abs_errors": stored_field_errors,
                "parent_relative_self_cosine": self_report["cosine"]})
    like_max = max(like_errors)
    stored_max = max(stored_errors)
    reconstructed_wrong_max = max(wrong_errors)
    wrong_error_delta = abs(reconstructed_wrong_max - invalid["full_replay_max_abs_error"])
    counters = PRICE.copy()
    A = bool(authority and len(endpoints) == 128
             and list(captures["v"].shape) == [128, token_length, 1152]
             and counters == PRICE and finite(audit_rows))
    B = bool(like_max <= 1e-5 and all(row["count_match"] for row in audit_rows))
    C = bool(wrong_error_delta <= 1e-12 and stored_max <= 1e-5)
    D = bool(A and B and C)
    predictions = dict(zip(PREDICTION_KEYS, (A, B, C, D)))
    terminal = "analysis_target_mismatch_repaired" if D else "replay_audit_failed"
    recovered = None
    if D:
        recovered = {"source_result_sha256": observed["invalid"],
            "source_result_terminal_remains": "invalid",
            "source_prediction_a_remains": False,
            "source_predictions_b_to_e": {key: value for key, value in invalid["predictions"].items()
                if not key.startswith("pred_a_")},
            "selections": invalid["selections"], "validations": invalid["validations"],
            "interpretation": "stable compositional task-typed value-source regions"}
    result = {"schema": "temporal_iswas_l11h3_value_source_region_localization_replay_audit_result_v1",
        "candidate_id": json.loads(PRIOR.read_text())["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "like_for_like_max_abs_error": like_max, "stored_command_gold_max_abs_error": stored_max,
        "reconstructed_mixed_target_error": reconstructed_wrong_max,
        "invalid_reported_error": invalid["full_replay_max_abs_error"],
        "mixed_target_error_delta": wrong_error_delta, "audit_rows": audit_rows,
        "recovered_v1_outcome": recovered, "predictions": predictions,
        "terminal": terminal, "price": counters}
    atomic_create_json(OUT, result)
    print(json.dumps({"like_for_like_max_abs_error": like_max,
        "stored_command_gold_max_abs_error": stored_max,
        "reconstructed_mixed_target_error": reconstructed_wrong_max,
        "mixed_target_error_delta": wrong_error_delta, "predictions": predictions,
        "terminal": terminal, "price": counters}, sort_keys=True))


if __name__ == "__main__":
    main()
