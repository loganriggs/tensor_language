#!/usr/bin/env python3
"""Zero-forward audit for a legal simultaneous temporal/is-was composition test."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authorities_and_zero_forward_price_pass pred_b_programs_share_a_final_residual_decoder_boundary pred_c_released_artifacts_store_interoperable_physical_command_bases pred_d_a_genuinely_simultaneous_dual_command_population_exists
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, math, os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_joint_command_composition_compatibility_audit_v1.json"
HANKEL = ROOT / "circuits/followups/temporal_iswas_q8_finite_causal_hankel_v1_result.json"
DIRECT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_rank2_direct_residual_final_head_route_v1_result.json"
MANIPULATION = ROOT / "circuits/followups/temporal_iswas_v15_final_rank2_removal_construction_swap_v1_result.json"
TASK_MODES = ROOT / "circuits/followups/temporal_iswas_rank46_task_typed_mode_causal_factorial_v1_result.json"
HANKEL_RUNNER = ROOT / "ops/run_temporal_iswas_q8_finite_causal_hankel_v1.py"
DIRECT_RUNNER = ROOT / "ops/run_temporal_iswas_v15_entry12_rank2_direct_residual_final_head_route_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_joint_command_composition_compatibility_audit_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas_joint_command_composition_compatibility_audit_v1"
EXPECTED = {
    "prior": "0e9daae2072f46e936733a0bb1e4afa077fc8eda2f7d9544f5278d08c2980d75", "hankel": "f8fa10c21c30cd3420648641b4a284ba3cb41152872db8cf77d25213c597bb62",
    "direct": "d6dd9592660cfad1c632a7b6d580b15faa04f3e762629d76984129e4872a56e8",
    "manipulation": "df0f6b4d3a43ea5a2e88281bdf7c95afab99e9329cae7edba5c9aee86480bded",
    "task_modes": "301377a9eded372a3c7a624892ae88089618690336e1da67f573d4e29ae87de4",
    "hankel_runner": "e9303c6fc1a11af4c103c49c5d47b2fdf0937714a80fd33d0549df2fa7216950",
    "direct_runner": "2c4bb81f87dcaa5a79df7d9a39cc18fcb0f154f7bed266140fd274e479d61c8b"
}
FILES = {"prior": PRIOR, "hankel": HANKEL, "direct": DIRECT, "manipulation": MANIPULATION,
         "task_modes": TASK_MODES, "hankel_runner": HANKEL_RUNNER, "direct_runner": DIRECT_RUNNER}
PRICE = {"model_forwards": 0, "transformer_backwards": 0, "model_updates": 0,
         "checkpoint_loads": 0, "example_evaluations": 0, "fit_parameters": 0}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior, hankel, direct, manipulation, task_modes = (
        json.loads(path.read_text()) for path in (PRIOR, HANKEL, DIRECT, MANIPULATION, TASK_MODES))
    hankel_source, direct_source = HANKEL_RUNNER.read_text(), DIRECT_RUNNER.read_text()
    candidates = sorted(path.name for path in (ROOT / "ops").glob("circuit_candidate*.py")
                        if "joint" in path.stem and "temporal" in path.stem and "iswas" in path.stem)
    row_tasks = sorted(set(row.get("task") for row in hankel.get("row_metadata", [])))
    physical_keys = {"hankel": sorted(key for key in hankel if "basis" in key or "command" in key),
                     "v15": sorted(key for key in manipulation if "basis" in key or "command" in key)}
    common_boundary = bool("resid18" in hankel_source and "backend.F.rms_norm" in hankel_source
                           and "backend.model.lm_head" in hankel_source
                           and '"entry12"' in direct_source
                           and "backend.F.rms_norm" in direct_source
                           and "backend.model.lm_head" in direct_source)
    stored_interoperable = bool(physical_keys["hankel"] and physical_keys["v15"])
    simultaneous = bool(candidates)
    A = bool(observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
             and hankel.get("terminal") == "screen"
             and direct.get("terminal") == "direct_residual_final_head_route"
             and manipulation.get("terminal") == "final_rank2_construction_split"
             and task_modes.get("terminal") == "task_mode_causal_factorial_null"
             and row_tasks == ["iswas", "temporal"] and finite({"hankel": hankel["rank_analysis"],
                                                                  "direct": direct["instrument"]})
             and PRICE == {"model_forwards": 0, "transformer_backwards": 0, "model_updates": 0,
                           "checkpoint_loads": 0, "example_evaluations": 0, "fit_parameters": 0})
    predictions = {
        "pred_a_authorities_and_zero_forward_price_pass": A,
        "pred_b_programs_share_a_final_residual_decoder_boundary": common_boundary,
        "pred_c_released_artifacts_store_interoperable_physical_command_bases": stored_interoperable,
        "pred_d_a_genuinely_simultaneous_dual_command_population_exists": simultaneous,
    }
    terminal = ("invalid" if not A else "joint_factorial_ready" if all(predictions.values())
                else "joint_population_and_basis_capture_required" if common_boundary and not stored_interoperable and not simultaneous
                else "joint_population_required" if common_boundary and not simultaneous
                else "basis_capture_required" if common_boundary else "no_common_boundary")
    result = {"schema": "temporal_iswas_joint_command_composition_compatibility_audit_result_v1",
              "candidate_id": CANDIDATE_ID, "started_utc": datetime.now(timezone.utc).isoformat(),
              "authority_sha256": EXPECTED, "audit": {"common_ambient_width": 1152 if common_boundary else None,
              "shared_decoder": "final_rmsnorm_tied_unembedding_softcap" if common_boundary else None,
              "hankel_row_task_values": row_tasks, "simultaneous_candidate_builders": candidates,
              "stored_physical_basis_keys": physical_keys,
              "existing_rows_are_separate_task_commands": row_tasks == ["iswas", "temporal"]},
              "predictions": predictions, "terminal": terminal, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "authority_ok": A, "candidate_id": CANDIDATE_ID,
                          "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
                          "anticipated_terminal": terminal, "price": PRICE}, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    atomic_create_json(OUT, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
