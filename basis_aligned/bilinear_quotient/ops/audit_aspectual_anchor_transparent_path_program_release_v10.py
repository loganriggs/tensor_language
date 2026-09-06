#!/usr/bin/env python3
"""Zero-forward conformance audit for donor-free actuator program v10."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_valid_authority pred_b_executable_inventory pred_c_synthetic_equation_conformance pred_d_actuator_evidence_exact pred_e_price_dependency_scope_exact
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

import aspectual_anchor_transparent_path_program_v9 as v9
import aspectual_anchor_transparent_path_program_v10 as program
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_release_v10.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v10_result.json"
PATHS = {
    "aspectual_anchor_transparent_path_program_v10.py": ROOT / "ops/aspectual_anchor_transparent_path_program_v10.py",
    "aspectual_anchor_transparent_path_program_v10_artifact.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v10_artifact.json",
    "aspectual_anchor_transparent_path_program_release_v9_result.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v9_result.json",
    "aspectual_anchor_rank1_donor_free_margin_reflection_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_rank1_donor_free_margin_reflection_v1_result.json",
}
EXPECTED_PRIOR_SHA256 = "522846018ae9ecc2e4bc8d7277c41b851e0c598bd9b0ec038239601cb9158e38"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict[str, object]:
    return json.loads(PATHS[name].read_text())


def synthetic_conformance() -> tuple[bool, int]:
    generator = torch.Generator().manual_seed(2026090610)
    okay, cases = True, 0
    lm_head = torch.nn.Linear(7, 13, bias=False)
    for batch_size in (1, 3, 8):
        states = torch.randn(batch_size, 7, generator=generator)
        selected = program.exact_selected_margin(states, lm_head, target_id=4, foil_id=9)
        logits = v9.exact_final_logits(states, lm_head)
        expected = logits[:, 4] - logits[:, 9]
        okay = okay and torch.allclose(selected, expected, atol=2.0e-6, rtol=0.0)
        cases += batch_size
    q = torch.randn(7, generator=generator)
    q = q / q.norm()
    grid_points = 9
    for _ in range(8):
        base = torch.randn(7, generator=generator)
        result = program.donor_free_margin_reflection(base, q, lm_head, target_id=4, foil_id=9, budget=5.0, grid_points=grid_points)
        grid = torch.linspace(-5.0, 5.0, grid_points)
        candidates = base[None, :] + grid[:, None] * q[None, :]
        margins = program.exact_selected_margin(candidates, lm_head, target_id=4, foil_id=9)
        base_margin = program.exact_selected_margin(base, lm_head, target_id=4, foil_id=9)
        expected_index = int(torch.argmin(torch.abs(margins + base_margin)))
        okay = okay and result["grid_index"] == expected_index and torch.equal(result["patched_resid18"], candidates[expected_index])
        okay = okay and torch.equal(result["patched_target_margin"], result["target_logit"] - result["foil_logit"])
        cases += 3
    rejection_calls = (
        lambda: program.exact_selected_margin([0.0] * 7, lm_head, target_id=1, foil_id=2),
        lambda: program.exact_selected_margin(torch.zeros(7), lm_head, target_id=1, foil_id=1),
        lambda: program.exact_selected_margin(torch.zeros(7), lm_head, target_id=-1, foil_id=2),
        lambda: program.donor_free_margin_reflection(torch.zeros((1, 7)), q, lm_head, target_id=1, foil_id=2),
        lambda: program.donor_free_margin_reflection(torch.zeros(7), torch.ones(7), lm_head, target_id=1, foil_id=2),
        lambda: program.donor_free_margin_reflection(torch.zeros(7), q, lm_head, target_id=1, foil_id=2, budget=0.0),
        lambda: program.donor_free_margin_reflection(torch.zeros(7), q, lm_head, target_id=1, foil_id=2, grid_points=8),
    )
    for call in rejection_calls:
        rejected = False
        try:
            call()
        except program.ProgramInputError:
            rejected = True
        okay = okay and rejected
        cases += 1
    return bool(okay and cases >= 30), cases


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "synthetic_equation_cases_min": 30}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    observed = {name: sha(path) for name, path in PATHS.items()}
    artifact = load("aspectual_anchor_transparent_path_program_v10_artifact.json")
    parent = load("aspectual_anchor_transparent_path_program_release_v9_result.json")
    evidence = load("aspectual_anchor_rank1_donor_free_margin_reflection_v1_result.json")
    pred_a = sha(PRIOR) == EXPECTED_PRIOR_SHA256 and observed == prior["authority"] and parent["terminal"] == "release" and evidence["terminal"] == "screen" and all(evidence["predictions"].values())
    tree = ast.parse(PATHS["aspectual_anchor_transparent_path_program_v10.py"].read_text())
    imports = sorted(alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    pred_b = imports == ["annotations", "aspectual_anchor_transparent_path_program_v9", "math", "torch"] and functions == {"exact_selected_margin", "donor_free_margin_reflection", "program_manifest"} and program.carrier_amplitude is v9.carrier_amplitude and program.rank1_carrier_projection is v9.rank1_carrier_projection and program.compiled_sparse_suffix_delta is v9.compiled_sparse_suffix_delta
    pred_c, cases = synthetic_conformance()
    panels = evidence["score"]["panels"]
    donor = artifact["evidence"]
    pred_d = (
        program.ACTUATOR_BUDGET == artifact["new_operation"]["budget"] == evidence["actuator"]["budget"]
        and program.ACTUATOR_GRID_POINTS == artifact["price"]["per_actuation_selected_head_evaluations"] == evidence["actuator"]["grid_points"]
        and program.RANK1_BASIS_SHA256 == artifact["new_operation"]["basis_sha256"] == evidence["basis_sha256"]
        and donor["A_recovery"] == {"lexical_A1_heldout": panels["lexical_A1_heldout"]["mean_recovery"], "lexical_A2": panels["lexical_A2"]["mean_recovery"], "fresh_A1": panels["fresh_A1"]["mean_recovery"], "fresh_A2": panels["fresh_A2"]["mean_recovery"]}
        and donor["P_noun_shift_margin_reflection"] == {"lexical": panels["lexical_P_noun_shifted_start"]["mean_margin_reflection_fraction"], "fresh": panels["fresh_P_noun_shifted_start"]["mean_margin_reflection_fraction"]}
        and evidence["actuator"]["confirmation_donor_activation_used"] is False
    )
    price = artifact["price"]
    exclusions = {"raw-text-to-resid18 computation", "whole-model replacement", "unrestricted syntax or vocabulary", "target-token-free autonomous prediction", "complete mediation of program v8", "independent fresh C control"}
    pred_e = price["stored_fit_scalars"] == prior["price"]["stored_fit_scalars"] == 1153 and price["stored_fit_vectors"] == 1 and price["requires_checkpoint_weights"] is True and price["requires_base_resid18"] is True and price["requires_confirmation_donor_activation"] is False and price["requires_target_and_foil_token_ids"] is True and exclusions == set(artifact["scope"]["not_licensed"])
    predictions = {
        "pred_a_hash_bound_valid_authority": pred_a,
        "pred_b_executable_inventory": pred_b,
        "pred_c_synthetic_equation_conformance": pred_c,
        "pred_d_actuator_evidence_exact": pred_d,
        "pred_e_price_dependency_scope_exact": pred_e,
    }
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_transparent_path_program_release_result_v10",
        "candidate_id": prior["candidate_id"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": sha(PRIOR),
        "authority_sha256": observed,
        "program_sha256": observed["aspectual_anchor_transparent_path_program_v10.py"],
        "artifact_sha256": observed["aspectual_anchor_transparent_path_program_v10_artifact.json"],
        "predictions": predictions,
        "synthetic_equation_cases": cases,
        "classification": "valid_executable_donor_free_target_guided_selective_carrier_actuator_program",
        "price": prior["price"],
        "terminal": terminal,
    }
    payload = atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "synthetic_equation_cases": cases, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
