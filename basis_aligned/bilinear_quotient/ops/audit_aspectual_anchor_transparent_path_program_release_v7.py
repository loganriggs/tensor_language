#!/usr/bin/env python3
"""Zero-forward conformance audit for scored-logit program v7."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_valid_authority pred_b_executable_inventory pred_c_synthetic_equation_conformance pred_d_readout_and_conformance_evidence_exact pred_e_price_dependency_scope_exact
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import aspectual_anchor_transparent_path_program_v6 as v6
import aspectual_anchor_transparent_path_program_v7 as program
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_release_v7.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v7_result.json"
PATHS = {
    "aspectual_anchor_transparent_path_program_v7.py": ROOT / "ops/aspectual_anchor_transparent_path_program_v7.py",
    "aspectual_anchor_transparent_path_program_v7_artifact.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v7_artifact.json",
    "aspectual_anchor_transparent_path_program_release_v6_result.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v6_result.json",
    "aspectual_anchor_transparent_path_program_v6_native_conformance_v1_result.json": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v6_native_conformance_v1_result.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict[str, object]:
    return json.loads(PATHS[name].read_text())


def synthetic_conformance() -> tuple[bool, int]:
    generator = torch.Generator().manual_seed(2026090607)
    lm_head = torch.nn.Linear(5, 11, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.randn((11, 5), generator=generator))
    okay, cases = True, 0
    for shape in ((5,), (3, 5)):
        state = torch.randn(shape, generator=generator)
        expected = 30.0 * torch.tanh(lm_head(F.rms_norm(state, (5,))) / 30.0)
        observed = program.exact_final_logits(state, lm_head)
        okay = okay and torch.equal(observed, expected)
        answer, foil = program.exact_scored_pair(state, lm_head, answer_id=2, foil_id=7)
        okay = okay and torch.equal(answer, expected[..., 2]) and torch.equal(foil, expected[..., 7])
        cases += 2
    for _ in range(13):
        base = torch.randn(5, generator=generator)
        initial = torch.randn(5, generator=generator)
        lambdas = {boundary: torch.randn((), generator=generator) for boundary in program.SUFFIX_BOUNDARIES}
        attention = {boundary: torch.randn(5, generator=generator) for boundary in program.SUFFIX_SOURCE_BOUNDARIES}
        states = {boundary: tuple(torch.randn(7, generator=generator) for _ in range(4)) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
        weights = {boundary: torch.randn((5, 7), generator=generator) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
        delta = v6.compiled_sparse_suffix_delta(initial, lambda0_by_boundary=lambdas, source_attention_delta_by_boundary=attention, mlp_states_by_boundary=states, down_weight_by_boundary=weights)
        expected = program.exact_scored_pair(base + delta, lm_head, answer_id=1, foil_id=9)
        observed = program.compiled_sparse_suffix_scored_pair(base, initial, lm_head, answer_id=1, foil_id=9, lambda0_by_boundary=lambdas, source_attention_delta_by_boundary=attention, mlp_states_by_boundary=states, down_weight_by_boundary=weights)
        okay = okay and torch.equal(observed[0], expected[0]) and torch.equal(observed[1], expected[1])
        cases += 1
    valid_lambdas = {boundary: torch.tensor(1.0) for boundary in program.SUFFIX_BOUNDARIES}
    valid_attention = {boundary: torch.zeros(5) for boundary in program.SUFFIX_SOURCE_BOUNDARIES}
    valid_states = {boundary: tuple(torch.zeros(7) for _ in range(4)) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
    valid_weights = {boundary: torch.zeros((5, 7)) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
    rejection_calls = (
        lambda: program.exact_final_logits([0.0] * 5, lm_head),
        lambda: program.exact_final_logits(torch.zeros(5), None),
        lambda: program.exact_scored_pair(torch.zeros(5), lm_head, answer_id=2, foil_id=2),
        lambda: program.exact_scored_pair(torch.zeros(5), lm_head, answer_id=-1, foil_id=2),
        lambda: program.exact_scored_pair(torch.zeros(5), lm_head, answer_id=2, foil_id=99),
        lambda: program.compiled_sparse_suffix_scored_pair([0.0] * 5, torch.zeros(5), lm_head, answer_id=1, foil_id=2, lambda0_by_boundary=valid_lambdas, source_attention_delta_by_boundary=valid_attention, mlp_states_by_boundary=valid_states, down_weight_by_boundary=valid_weights),
        lambda: program.compiled_sparse_suffix_scored_pair(torch.zeros(6), torch.zeros(5), lm_head, answer_id=1, foil_id=2, lambda0_by_boundary=valid_lambdas, source_attention_delta_by_boundary=valid_attention, mlp_states_by_boundary=valid_states, down_weight_by_boundary=valid_weights),
        lambda: program.compiled_sparse_suffix_scored_pair(torch.zeros(5), torch.zeros(5), lm_head, answer_id=1, foil_id=2, lambda0_by_boundary={}, source_attention_delta_by_boundary=valid_attention, mlp_states_by_boundary=valid_states, down_weight_by_boundary=valid_weights),
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
    artifact = load("aspectual_anchor_transparent_path_program_v7_artifact.json")
    parent = load("aspectual_anchor_transparent_path_program_release_v6_result.json")
    conformance = load("aspectual_anchor_transparent_path_program_v6_native_conformance_v1_result.json")
    pred_a = observed == prior["authority"] and parent["terminal"] == "release" and conformance["terminal"] == "screen" and all(parent["predictions"].values()) and all(conformance["predictions"].values())
    tree = ast.parse(PATHS["aspectual_anchor_transparent_path_program_v7.py"].read_text())
    imports = sorted(alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    pred_b = (
        imports == ["annotations", "aspectual_anchor_transparent_path_program_v6", "torch", "torch.nn.functional"]
        and functions == {"exact_final_logits", "exact_scored_pair", "compiled_sparse_suffix_scored_pair", "program_manifest"}
        and program.compiled_sparse_suffix_delta is v6.compiled_sparse_suffix_delta
        and program.program_manifest()["exact_final_readout"] == ("rms_norm", "lm_head", "softcap_30", "answer_foil_index")
    )
    pred_c, cases = synthetic_conformance()
    native = artifact["native_conformance"]
    pred_d = (
        artifact["final_readout"]["equation"] == "30*tanh(lm_head(rms_norm(base_resid18+delta18))/30)"
        and native["reference_logit_max_abs"] == conformance["score"]["reference_logit_max_abs"]
        and native["reference_recovery_max_abs"] == conformance["score"]["reference_recovery_max_abs"]
    )
    price = artifact["price"]
    exclusions = {"standalone native-margin prediction", "free-form text", "new-construction transfer", "whole-model replacement", "prospective attribution of block12 or block14 selection", "empirical fidelity of non-answer/foil vocabulary logits"}
    pred_e = (
        price["stored_fit_scalars"] == price["stored_fit_vectors"] == 0
        and price["requires_checkpoint_weights"] is True and price["requires_paired_base_and_donor_states"] is True
        and price["requires_native_intervening_blocks_or_final_readout"] is False and price["requires_base_resid18"] is True
        and exclusions == set(artifact["scope"]["not_licensed"])
        and prior["price"]["model_forwards"] == prior["price"]["fit_parameters"] == 0
    )
    predictions = {"pred_a_hash_bound_valid_authority": pred_a, "pred_b_executable_inventory": pred_b, "pred_c_synthetic_equation_conformance": pred_c, "pred_d_readout_and_conformance_evidence_exact": pred_d, "pred_e_price_dependency_scope_exact": pred_e}
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_transparent_path_program_release_result_v7", "candidate_id": prior["candidate_id"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": sha(PRIOR),
        "authority_sha256": observed, "program_sha256": observed["aspectual_anchor_transparent_path_program_v7.py"],
        "artifact_sha256": observed["aspectual_anchor_transparent_path_program_v7_artifact.json"], "predictions": predictions,
        "synthetic_equation_cases": cases, "classification": "valid_executable_paired_causal_program_through_answer_foil_scored_logits",
        "price": prior["price"], "terminal": terminal,
    }
    payload = atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "synthetic_equation_cases": cases, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
