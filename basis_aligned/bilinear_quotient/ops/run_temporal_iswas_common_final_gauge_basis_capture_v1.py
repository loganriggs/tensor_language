#!/usr/bin/env python3
"""Materialize temporal Q8 and v15 is-was command bases in one final gauge."""

# BQGATE: EXPERIMENT pred_a_authority_replay_geometry_finiteness_and_exact_price pred_b_v15_physical_command_span_is_fold_stable pred_c_v15_iswas_span_is_contained_in_temporal_q8 pred_d_literal_basis_storage_roundtrips
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import entry12_response_basis_contract as basis_contract
import run_temporal_iswas_q8_finite_causal_hankel_v1 as q8
import run_temporal_iswas_v15_multidirection_contextual_gram_router_v16_v1 as v15run

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_common_final_gauge_basis_capture_v1.json"
AUDIT = ROOT / "circuits/followups/temporal_iswas_joint_command_composition_compatibility_audit_v1_result.json"
HANKEL = ROOT / "circuits/followups/temporal_iswas_q8_finite_causal_hankel_v1_result.json"
BASIS_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_shared_target_control_response_basis_v1_result.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_common_final_gauge_basis_capture_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas_common_final_gauge_basis_capture_v1"
EXPECTED = {
    "prior": "ca731bcfde6bc0d602ad6820e171e83ea9aa7a068a727810dd1052ad14e9f54a",
    "audit": "dcae033128e12903242aca03acb1cd4a53b493dba9f9554fb826553519903563",
    "hankel": "f8fa10c21c30cd3420648641b4a284ba3cb41152872db8cf77d25213c597bb62",
    "q8_runner": "e9303c6fc1a11af4c103c49c5d47b2fdf0937714a80fd33d0549df2fa7216950",
    "basis_result": "ffaec3f0bede415f7b9b1664ae7ddf1667978dc26f42cfa2f81e9c82f0f77467",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "v15_runner": "2a6a245084826bd8242b326ec5c90022977cac4a7622e4d23a1eaa5653f12a05",
    "basis_contract": "cc05ed73baf02c7ec6f0ec2b9e64a831a1fff6462ab4d8a5ba1f39bc5f70dab1",
}
FILES = {"prior": PRIOR, "audit": AUDIT, "hankel": HANKEL,
         "q8_runner": ROOT / "ops/run_temporal_iswas_q8_finite_causal_hankel_v1.py",
         "basis_result": BASIS_RESULT, "oracles": ORACLES,
         "v15_runner": ROOT / "ops/run_temporal_iswas_v15_multidirection_contextual_gram_router_v16_v1.py",
         "basis_contract": ROOT / "ops/entry12_response_basis_contract.py"}
PRICE = {"checkpoint_loads": 1, "native_capture_forwards": 2,
         "differentiable_transformer_forwards": 6, "transformer_backward_forwards": 0,
         "model_updates": 0, "example_evaluations": 320, "fit_parameters": 0}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def stored_basis(torch, value):
    value = value.detach().float().contiguous()
    column_major = value.T.contiguous().cpu()
    payload = column_major.numpy().tobytes()
    values = column_major.reshape(-1).tolist()
    replay = torch.tensor(values, device=value.device).reshape(value.shape[1], value.shape[0]).T
    return {"shape": list(value.shape), "layout": "columns_then_rows_float32",
            "sha256": hashlib.sha256(payload).hexdigest(), "values": values,
            "roundtrip_max_abs_error": float((value - replay).abs().max())}


def principal(torch, left, right):
    values = torch.linalg.svdvals(left.float().T @ right.float())
    return [float(value) for value in values]


def stack_prefix(torch, delta, context, panel):
    indices = [i for i, row in enumerate(context["rows"]) if row["transform_id"] == panel]
    return torch.cat([delta[index, :int(context["base_batch"].semantic_positions[index]) + 1].float()
                      for index in indices], dim=0)


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior, audit, hankel, basis_result, oracles = (
        json.loads(path.read_text()) for path in (PRIOR, AUDIT, HANKEL, BASIS_RESULT, ORACLES))
    q8_paths = {"prior": q8.PRIOR, "shared_causal": q8.SHARED_CAUSAL,
                "temporal_capability": q8.TEMPORAL_CAPABILITY, "subspace": q8.SUBSPACE,
                "iswas": q8.ISWAS, "v2_capability": q8.V2_CAPABILITY,
                "v3_capability": q8.V3_CAPABILITY, "temporal_builder": q8.TEMPORAL_BUILDER,
                "v2_builder": q8.V2_BUILDER, "v3_builder": q8.V3_BUILDER,
                "atlas_runner": q8.ATLAS_RUNNER, "analytic_runner": q8.ANALYTIC_RUNNER,
                "overlap_runner": q8.OVERLAP_RUNNER}
    q8_authority = {name: sha(path) for name, path in q8_paths.items()} == q8.EXPECTED
    authority = bool(observed == EXPECTED and q8_authority
                     and prior.get("candidate_id") == CANDIDATE_ID
                     and audit.get("terminal") == "joint_population_and_basis_capture_required"
                     and hankel.get("terminal") == "screen"
                     and basis_result.get("terminal") == "p_complement_destroys_target"
                     and oracles.get("terminal") == "construction_conditioned_coordinate")
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "authority_ok": authority,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "basis_shapes": {"temporal_q8": [1152, 8], "v15_fold0": [1152, 2],
                            "v15_fold1": [1152, 2]}, "price": PRICE}
    if not authority: raise RuntimeError(f"authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)

    started = time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters(): parameter.requires_grad_(False)
    torch = backend.torch; counters = {key: 0 for key in PRICE}; counters["checkpoint_loads"] = 1
    native = backend.native
    def counted(batch, *, capture):
        counters["native_capture_forwards"] += 1; counters["example_evaluations"] += len(batch.row_ids)
        return native(batch, capture=capture)
    backend.native = counted

    subspace = json.loads(q8.SUBSPACE.read_text())
    family, _singular, _energy = q8.family_builder.build_family(backend, subspace)
    q = family[8]
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in range(12, 18))
    raw_temporal, orientation_error, _wrong = q8.overlap.residual_modes(backend, q, gain)
    temporal = torch.linalg.qr(raw_temporal.float(), mode="reduced").Q

    rows = v15run.v15.build_rows(); bank = v15run.parent.capture_bank(backend, rows, counters, factors=False)
    contexts, off_states = {}, {}
    expert_states = {panel: {} for panel in ("A1", "A2")}
    for parity in (0, 1):
        context = v15run.parent.subset_context(bank, v15run.parent.row_indices(rows, parity=parity))
        contexts[parity] = context
        _output, states = v15run.state_executor.execute(
            v15run.parent, backend, context, counters, {}, capture=True)
        off_states[parity] = states["entry12"]
        for expert, panel in v15run.EXPERTS.items():
            bases = v15run.dependency.bases_for_evaluation(torch, backend.device, oracles, expert)[parity]
            _output, states = v15run.state_executor.execute(
                v15run.parent, backend, context, counters, bases, capture=True)
            expert_states[panel][parity] = states["entry12"]

    v15_bases, fit_geometry = {}, {}
    for held in (0, 1):
        train = 1 - held; context = contexts[train]; off = off_states[train]
        deltas = {panel: expert_states[panel][train].float() - off.float() for panel in ("A1", "A2")}
        target_a1 = stack_prefix(torch, deltas["A1"], context, "A1").mean(0)
        target_a2 = stack_prefix(torch, deltas["A2"], context, "A2").mean(0)
        p_rows = torch.cat([stack_prefix(torch, deltas[panel], context, "P")
                            for panel in ("A1", "A2")])
        bases, geometry = basis_contract.fit_bases(
            torch, target_a1, target_a2, p_rows, relative_threshold=1e-6)
        v15_bases[held] = bases["construction_union_rank_le_2"].float()
        fit_geometry[str(held)] = geometry

    identity8 = torch.eye(8, device=backend.device); identity2 = torch.eye(2, device=backend.device)
    orthogonality = max(float((temporal.T @ temporal - identity8).abs().max()),
                        *(float((basis.T @ basis - identity2).abs().max()) for basis in v15_bases.values()))
    fold_principal = principal(torch, v15_bases[0], v15_bases[1])
    temporal_principal = {str(fold): principal(torch, temporal, basis)
                          for fold, basis in v15_bases.items()}
    projector_distance = float((v15_bases[0] @ v15_bases[0].T
                                - v15_bases[1] @ v15_bases[1].T).norm())
    joined_singular = torch.linalg.svdvals(torch.cat((temporal, v15_bases[0], v15_bases[1]), dim=1))
    union_rank = int((joined_singular > joined_singular[0] * 1e-6).sum())
    stored = {"temporal_q8": stored_basis(torch, temporal),
              "v15_fold0": stored_basis(torch, v15_bases[0]),
              "v15_fold1": stored_basis(torch, v15_bases[1])}
    counters["fit_parameters"] = 0
    A = bool(authority and v15run.v15.validate_rows(rows) == v15run.V15_ROWS_SHA256
             and all(basis.shape == (1152, 2) for basis in v15_bases.values())
             and temporal.shape == (1152, 8) and orientation_error <= 1e-4
             and orthogonality <= 1e-5 and finite({"fit": fit_geometry,
                                                   "fold": fold_principal,
                                                   "temporal": temporal_principal})
             and counters == PRICE)
    B = all(len(basis) == 2 for basis in temporal_principal.values()) and min(fold_principal) >= .80
    C = all(min(values) >= .75 for values in temporal_principal.values())
    D = all(item["roundtrip_max_abs_error"] <= 1e-6 and len(item["sha256"]) == 64
            for item in stored.values())
    predictions = {
        "pred_a_authority_replay_geometry_finiteness_and_exact_price": A,
        "pred_b_v15_physical_command_span_is_fold_stable": B,
        "pred_c_v15_iswas_span_is_contained_in_temporal_q8": C,
        "pred_d_literal_basis_storage_roundtrips": D,
    }
    terminal = ("invalid" if not A or not D else "shared_q8_command_state" if B and C
                else "task_typed_direct_sum" if B else "fold_typed_iswas_state")
    result = {"schema": "temporal_iswas_common_final_gauge_basis_capture_result_v1",
              "candidate_id": CANDIDATE_ID, "started_utc": datetime.now(timezone.utc).isoformat(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "basis_storage": stored, "geometry": {"temporal_orientation_max_abs_error": orientation_error,
              "orthogonality_max_abs_error": orthogonality, "v15_fold_principal_cosines": fold_principal,
              "temporal_q8_vs_v15_principal_cosines": temporal_principal,
              "v15_projector_frobenius_distance": projector_distance,
              "joint_union_singular_values": [float(x) for x in joined_singular], "joint_union_rank": union_rank,
              "v15_fit_geometry": fit_geometry, "recurrent_gain": gain},
              "predictions": predictions, "terminal": terminal, "price": counters}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("geometry", "predictions", "terminal", "price")},
                     sort_keys=True))


if __name__ == "__main__": main()
