#!/usr/bin/env python3
"""Zero-forward conformance audit for carrier-mediated program v9."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_valid_authority pred_b_executable_inventory pred_c_synthetic_equation_conformance pred_d_carrier_evidence_exact pred_e_price_dependency_scope_exact
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

import aspectual_anchor_transparent_path_program_v8 as v8
import aspectual_anchor_transparent_path_program_v9 as program
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_release_v9.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v9_result.json"
PATHS = {
    "aspectual_anchor_transparent_path_program_v9.py": ROOT / "ops/aspectual_anchor_transparent_path_program_v9.py",
    "aspectual_anchor_transparent_path_program_v9_artifact.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v9_artifact.json",
    "aspectual_anchor_transparent_path_program_release_v8_result.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v8_result.json",
    "aspectual_anchor_das_resid18_rank1_transfer_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json",
    "aspectual_anchor_program_v8_rank1_carrier_mediation_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_program_v8_rank1_carrier_mediation_v1_result.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict[str, object]:
    return json.loads(PATHS[name].read_text())


def synthetic_conformance() -> tuple[bool, int]:
    generator = torch.Generator().manual_seed(2026090609)
    q = torch.randn(17, generator=generator)
    q = q / q.norm()
    okay, cases = True, 0
    for _ in range(10):
        delta = torch.randn(17, generator=generator)
        amplitude = program.carrier_amplitude(delta, q)
        projection = program.rank1_carrier_projection(delta, q)
        okay = okay and torch.allclose(amplitude, torch.dot(delta, q)) and torch.allclose(projection, q * torch.dot(delta, q))
        cases += 2
    lm_head = torch.nn.Linear(5, 9, bias=False)
    q5 = torch.randn(5, generator=generator)
    q5 = q5 / q5.norm()
    lambdas = {boundary: torch.randn((), generator=generator) for boundary in program.SUFFIX_BOUNDARIES}
    attention = {boundary: torch.randn(5, generator=generator) for boundary in program.SUFFIX_SOURCE_BOUNDARIES}
    states = {boundary: tuple(torch.randn(7, generator=generator) for _ in range(4)) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
    weights = {boundary: torch.randn((5, 7), generator=generator) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
    for _ in range(5):
        base, initial = torch.randn(5, generator=generator), torch.randn(5, generator=generator)
        delta = v8.compiled_sparse_suffix_delta(initial, lambda0_by_boundary=lambdas, source_attention_delta_by_boundary=attention, mlp_states_by_boundary=states, down_weight_by_boundary=weights)
        expected = v8.exact_scored_pair(base + q5 * torch.dot(delta, q5), lm_head, answer_id=1, foil_id=7)
        observed = program.compiled_rank1_suffix_scored_pair(base, initial, q5, lm_head, answer_id=1, foil_id=7, lambda0_by_boundary=lambdas, source_attention_delta_by_boundary=attention, mlp_states_by_boundary=states, down_weight_by_boundary=weights)
        okay = okay and torch.equal(observed[0], expected[0]) and torch.equal(observed[1], expected[1])
        cases += 1
    rejection_calls = (
        lambda: program.carrier_amplitude([0.0] * 5, q5),
        lambda: program.carrier_amplitude(torch.zeros(5), [0.0] * 5),
        lambda: program.carrier_amplitude(torch.zeros(6), q5),
        lambda: program.carrier_amplitude(torch.zeros(5), 2.0 * q5),
        lambda: program.compiled_rank1_suffix_scored_pair([0.0] * 5, torch.zeros(5), q5, lm_head, answer_id=1, foil_id=2, lambda0_by_boundary=lambdas, source_attention_delta_by_boundary=attention, mlp_states_by_boundary=states, down_weight_by_boundary=weights),
        lambda: program.compiled_rank1_suffix_scored_pair(torch.zeros(6), torch.zeros(5), q5, lm_head, answer_id=1, foil_id=2, lambda0_by_boundary=lambdas, source_attention_delta_by_boundary=attention, mlp_states_by_boundary=states, down_weight_by_boundary=weights),
        lambda: program.compiled_rank1_suffix_scored_pair(torch.zeros(5), torch.zeros(5), q5, lm_head, answer_id=1, foil_id=1, lambda0_by_boundary=lambdas, source_attention_delta_by_boundary=attention, mlp_states_by_boundary=states, down_weight_by_boundary=weights),
        lambda: program.compiled_rank1_suffix_scored_pair(torch.zeros(5), torch.zeros(5), q5, lm_head, answer_id=1, foil_id=2, lambda0_by_boundary={}, source_attention_delta_by_boundary=attention, mlp_states_by_boundary=states, down_weight_by_boundary=weights),
    )
    for call in rejection_calls:
        rejected = False
        try:
            call()
        except program.ProgramInputError:
            rejected = True
        okay = okay and rejected
        cases += 1
    return bool(okay and cases >= 25), cases


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "synthetic_equation_cases_min": 25}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    observed = {name: sha(path) for name, path in PATHS.items()}
    artifact = load("aspectual_anchor_transparent_path_program_v9_artifact.json")
    parent = load("aspectual_anchor_transparent_path_program_release_v8_result.json")
    carrier = load("aspectual_anchor_das_resid18_rank1_transfer_v1_result.json")
    mediation = load("aspectual_anchor_program_v8_rank1_carrier_mediation_v1_result.json")
    pred_a = observed == prior["authority"] and parent["terminal"] == "release" and carrier["terminal"] == mediation["terminal"] == "screen" and all(all(item["predictions"].values()) for item in (parent, carrier, mediation))
    tree = ast.parse(PATHS["aspectual_anchor_transparent_path_program_v9.py"].read_text())
    imports = sorted(alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    pred_b = imports == ["annotations", "aspectual_anchor_transparent_path_program_v8", "torch"] and functions == {"carrier_amplitude", "rank1_carrier_projection", "compiled_rank1_suffix_scored_pair", "program_manifest"} and program.compiled_sparse_suffix_delta is v8.compiled_sparse_suffix_delta and program.exact_scored_pair is v8.exact_scored_pair
    pred_c, cases = synthetic_conformance()
    evidence = artifact["mediation"]
    pred_d = (
        artifact["carrier"]["basis_sha256"] == carrier["basis"]["sha256"] == program.RANK1_BASIS_SHA256
        and artifact["carrier"]["rank"] == carrier["rank"] == 1
        and evidence["rank1_recovery_fraction"] == mediation["score"]["rank1_to_full_recovery_fraction"]
        and evidence["orthogonal_absolute_fraction"] == mediation["score"]["orthogonal_to_full_absolute_fraction"]
    )
    price = artifact["price"]
    exclusions = {"standalone native-margin prediction", "free-form text", "whole-model replacement", "unrestricted syntax transfer", "complete mediation", "rank greater than one"}
    pred_e = price["stored_fit_scalars"] == 1152 and price["stored_fit_vectors"] == 1 and price["requires_checkpoint_weights"] is True and price["requires_paired_base_and_donor_states"] is True and price["requires_native_intervening_blocks_or_final_readout"] is False and price["requires_base_resid18"] is True and exclusions == set(artifact["scope"]["not_licensed"]) and prior["price"]["stored_fit_scalars"] == 1152
    predictions = {"pred_a_hash_bound_valid_authority": pred_a, "pred_b_executable_inventory": pred_b, "pred_c_synthetic_equation_conformance": pred_c, "pred_d_carrier_evidence_exact": pred_d, "pred_e_price_dependency_scope_exact": pred_e}
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {"schema": "aspectual_anchor_transparent_path_program_release_result_v9", "candidate_id": prior["candidate_id"], "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": sha(PRIOR), "authority_sha256": observed, "program_sha256": observed["aspectual_anchor_transparent_path_program_v9.py"], "artifact_sha256": observed["aspectual_anchor_transparent_path_program_v9_artifact.json"], "predictions": predictions, "synthetic_equation_cases": cases, "classification": "valid_executable_rank1_carrier_mediated_prospectively_transferred_paired_causal_scored_logit_program", "price": prior["price"], "terminal": terminal}
    payload = atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "synthetic_equation_cases": cases, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
