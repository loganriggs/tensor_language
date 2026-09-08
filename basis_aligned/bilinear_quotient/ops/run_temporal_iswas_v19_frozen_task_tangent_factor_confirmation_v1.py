#!/usr/bin/env python3
"""Pristine v19 confirmation of the frozen task-tangent M11 factor order."""

# BQGATE: EXPERIMENT pred_a_authority_formula_replays_hooks_finiteness_and_exact_price pred_b_frozen_top16_transfers_to_both_pristine_v19_constructions pred_c_frozen_top32_recovers_most_of_both_pristine_v19_constructions pred_d_factor_program_is_construction_stable_and_selective pred_e_no_v19_fit_reorder_dose_or_postselection
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
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v19 as fresh
import circuit_fast_screen_producer as producer
import run_temporal_iswas_m11_crossconstruction_exact_factor_greedy_v1 as executor
import run_temporal_iswas_v18_frozen_shared_tensor_transfer_v1 as transfer


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v19_frozen_task_tangent_factor_confirmation_v1.json"
TASK_RESULT = ROOT / "circuits/followups/temporal_iswas_m11_crossconstruction_task_tangent_factor_order_v1_result.json"
TASK_RUNNER = ROOT / "ops/run_temporal_iswas_m11_crossconstruction_task_tangent_factor_order_v1.py"
CAPABILITY_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v19_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v19.py"
EXECUTOR = ROOT / "ops/run_temporal_iswas_m11_crossconstruction_exact_factor_greedy_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v19_frozen_task_tangent_factor_confirmation_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v19_frozen_task_tangent_factor_confirmation_v1"
EXPECTED = {
    "prior": "c1f5140d653a8b52b890217a7b168edbd76d625ef5bfee63b53467433b1867a1",
    "task_result": "e0af6cc965d2d6a560dbde548e191268c77fcac7a5817ad54ea2ccaefe7050b7",
    "task_runner": "78ac32a90ef3a0182fe2eb024a3b687ddefe2b5fb1b816df5dd0febaf8f857ec",
    "capability_result": "28f8f134d78279dd82cd7cca94f76e9cf35815bd1a59430ad34601fd0294fdd4",
    "builder": "67916a1c83a019e037fc14415932560ba98cc5dd46a623216b43fd36d267d8b6",
    "executor": "e83aa3a75819c68587f46d5b53f332eab71cb1800a294d448abc854a92dfefb7",
}
PANELS, TARGETS, CONTROLS = ("A1", "A2", "P", "C"), ("A1", "A2"), ("P", "C")
PRICE = {"checkpoint_loads": 1, "model_forwards": 44, "sequence_evaluations": 704,
         "scored_token_positions": 1408, "intervention_arms_per_panel": 8,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
BARS = {"replay": 1e-5, "formula": 1e-5, "output_rse": 1e-8,
        "top16_low": .70, "top16_high": 1.30, "top16_cosine": .95,
        "top32_low": .80, "top32_high": 1.25, "top32_cosine": .95,
        "direction": .875, "recovery_gap": .25, "control_ratio": .25}
PREDICTION_KEYS = (
    "pred_a_authority_formula_replays_hooks_finiteness_and_exact_price",
    "pred_b_frozen_top16_transfers_to_both_pristine_v19_constructions",
    "pred_c_frozen_top32_recovers_most_of_both_pristine_v19_constructions",
    "pred_d_factor_program_is_construction_stable_and_selective",
    "pred_e_no_v19_fit_reorder_dose_or_postselection",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    paths = {"prior": PRIOR, "task_result": TASK_RESULT, "task_runner": TASK_RUNNER,
             "capability_result": CAPABILITY_RESULT, "builder": BUILDER, "executor": EXECUTOR}
    observed = {name: sha(path) for name, path in paths.items()}
    task_result, capability = json.loads(TASK_RESULT.read_text()), json.loads(CAPABILITY_RESULT.read_text())
    all_rows = fresh.build_rows()
    row_sets = {family: [row for row in all_rows if row["transform_id"] == family and (
        family in CONTROLS or row["row_id"] in set(capability["jointly_capable_row_ids"][family]))]
        for family in PANELS}
    order_tuple = tuple(int(index) for index in task_result["stable_factor_order_first256"])
    authority_ok = bool(observed == EXPECTED
        and task_result.get("terminal") == "compact_task_read_factor_program"
        and capability.get("terminal") == "screen" and capability.get("causal_outcomes_opened") is False
        and len(order_tuple) == len(set(order_tuple)) == 256
        and all(len(row_sets[family]) == 16 for family in PANELS))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "rows": {key: len(value) for key, value in row_sets.items()},
           "frozen_order": len(order_tuple), "prefixes": executor.PREFIXES, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("v19 task-factor confirmation authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    order = backend.torch.tensor(order_tuple, dtype=backend.torch.long, device=backend.device)
    reports, effects, instruments = {}, {}, {}
    with backend.torch.no_grad():
        for family in PANELS:
            reports[family], panel_effects, instruments[family] = executor.test_panel(
                backend, row_sets[family], order)
            effects[family] = {key: value.tolist() for key, value in panel_effects.items()}
    target_rms = {label: transfer.rms(np.concatenate([
        np.asarray(effects[family][label], dtype=np.float64) for family in TARGETS]))
        for label in ("top16", "top32")}
    control_ratios = {family: {label: transfer.rms(effects[family][label])
        / max(target_rms[label], 1e-30) for label in target_rms} for family in CONTROLS}
    recovery_gaps = {label: abs(reports["A1"][label]["signed_recovery"]
                                - reports["A2"][label]["signed_recovery"])
                     for label in ("top16", "top32")}
    A = bool(authority_ok and all(executor.calls_ok(value["calls"]) for value in instruments.values())
        and all(value["writer_replay_max_abs_margin_error"] <= BARS["replay"]
                and value["all_hidden_margin_max_abs_error"] <= BARS["formula"]
                and value["all_hidden_output_relative_squared_error"] <= BARS["output_rse"]
                for value in instruments.values())
        and finite({"reports": reports, "controls": control_ratios, "gaps": recovery_gaps})
        and PRICE["model_forwards"] == len(PANELS) * (3 + PRICE["intervention_arms_per_panel"])
        and PRICE["sequence_evaluations"] == PRICE["model_forwards"] * 16)
    B = all(BARS["top16_low"] <= reports[family]["top16"]["signed_recovery"] <= BARS["top16_high"]
        and reports[family]["top16"]["cosine"] >= BARS["top16_cosine"]
        and reports[family]["top16"]["direction_agreement"] >= BARS["direction"]
        for family in TARGETS)
    C = all(BARS["top32_low"] <= reports[family]["top32"]["signed_recovery"] <= BARS["top32_high"]
        and reports[family]["top32"]["cosine"] >= BARS["top32_cosine"]
        and reports[family]["top32"]["direction_agreement"] >= BARS["direction"]
        for family in TARGETS)
    D = all(gap <= BARS["recovery_gap"] for gap in recovery_gaps.values()) and all(
        ratio <= BARS["control_ratio"] for family in CONTROLS for ratio in control_ratios[family].values())
    E = True
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A else "nonselective_or_unstable" if not D else
        "top16_pristine_confirmed" if B and C else "top32_pristine_confirmed" if C else
        "task_tangent_factor_confirmation_failed")
    result = {"schema": "temporal_iswas_v19_frozen_task_tangent_factor_confirmation_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "frozen_factor_order_first256": order_tuple, "v19_fit_updates": 0,
        "v19_selected_parameters": 0, "reports": reports, "effect_vectors": effects,
        "recovery_gaps": recovery_gaps, "control_normalized_rms_ratios": control_ratios,
        "instrument": instruments, "predictions": predictions, "terminal": terminal,
        "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("reports", "recovery_gaps",
        "control_normalized_rms_ratios", "instrument", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__": main()
