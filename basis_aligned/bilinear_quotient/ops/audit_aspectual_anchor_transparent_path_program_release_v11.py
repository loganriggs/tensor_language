#!/usr/bin/env python3
"""Zero-forward conformance audit for operational-quotient program v11."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_hash_bound_valid_authority pred_b_parent_operations_preserved pred_c_equivalence_evidence_exact pred_d_manifest_conformance pred_e_price_and_scope_exact
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import aspectual_anchor_transparent_path_program_v10 as v10
import aspectual_anchor_transparent_path_program_v11 as program
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_transparent_path_program_release_v11.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v11_result.json"
PATHS = {
    "program_v11": ROOT / "ops/aspectual_anchor_transparent_path_program_v11.py",
    "artifact_v11": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_v11_artifact.json",
    "program_v10_release": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v10_result.json",
    "construction_equivalence": ROOT / "circuits/followups/aspectual_anchor_program_v8_cross_construction_variable_interchange_v1_result.json",
    "family_equivalence": ROOT / "circuits/followups/aspectual_anchor_program_v8_cross_family_variable_interchange_v1_result.json",
    "lexicon_equivalence": ROOT / "circuits/followups/aspectual_anchor_program_v8_lexicon_deranged_variable_interchange_v1_result.json",
}
EXPECTED_PRIOR_SHA256 = "7c2eb8fba2d6d757f5e6a1c75d1819eb221abfac40dbb97bb3d98887d5e8760a"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def range_for(result, arms):
    ratios = result["score"]["recovery_fraction_vs_target_full"]
    values = [ratios[panel][arm] for panel in ratios for arm in arms]
    return min(values), max(values)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "manifest_assertions_min": 20}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    observed = {name: sha(path) for name, path in PATHS.items()}
    loaded = {name: json.loads(path.read_text()) for name, path in PATHS.items() if name != "program_v11"}
    pred_a = sha(PRIOR) == EXPECTED_PRIOR_SHA256 and observed == prior["authority"] and loaded["program_v10_release"]["terminal"] == "release" and all(loaded[name]["terminal"] == "screen" for name in ("construction_equivalence", "family_equivalence", "lexicon_equivalence"))
    tree = ast.parse(PATHS["program_v11"].read_text())
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    aliases = ("carrier_amplitude", "rank1_carrier_projection", "compiled_sparse_suffix_delta", "exact_final_logits", "exact_scored_pair", "exact_selected_margin", "donor_free_margin_reflection")
    pred_b = functions == {"operational_quotient_manifest", "program_manifest"} and all(getattr(program, name) is getattr(v10, name) for name in aliases)
    quotient = program.operational_quotient_manifest()
    construction = loaded["construction_equivalence"]
    family = loaded["family_equivalence"]
    lexicon = loaded["lexicon_equivalence"]
    expected_ranges = {
        "construction": (range_for(construction, ("source_full",)), range_for(construction, ("swap_initial", "swap_attention", "swap_mlp"))),
        "A1_A2_family": (range_for(family, ("source_full",)), range_for(family, ("swap_initial", "swap_attention", "swap_mlp"))),
        "reporter_period_lexicon": (range_for(lexicon, ("source_full",)), range_for(lexicon, ("swap_initial", "swap_attention", "swap_mlp"))),
    }
    pred_c = all(tuple(quotient["equivalences"][name]["whole_recovery_fraction_range"]) == ranges[0] and tuple(quotient["equivalences"][name]["single_group_recovery_fraction_range"]) == ranges[1] for name, ranges in expected_ranges.items()) and quotient["direction_fraction"] == 1.0 and set(quotient["variable_groups"]) == {"initial", "attention", "mlp"}
    manifest = program.program_manifest()
    assertions = [
        manifest["program_id"] == program.PROGRAM_ID,
        manifest["operational_quotient"] == quotient,
        manifest["quotient_variable_count"] == 3,
        manifest["native_module_boundaries_are_semantic_units"] is False,
        manifest["donor_free_actuator"] == v10.program_manifest()["donor_free_actuator"],
        manifest["rank1_basis_sha256"] == v10.program_manifest()["rank1_basis_sha256"],
        quotient["downstream_reader"].startswith("blocks10-17"),
        quotient["claim"].startswith("operational equivalence"),
        len(quotient["equivalences"]) == 3,
        quotient["equivalences"]["construction"]["result_sha256"] == program.CONSTRUCTION_EQUIVALENCE_RESULT_SHA256,
        quotient["equivalences"]["A1_A2_family"]["result_sha256"] == program.FAMILY_EQUIVALENCE_RESULT_SHA256,
        quotient["equivalences"]["reporter_period_lexicon"]["result_sha256"] == program.LEXICON_EQUIVALENCE_RESULT_SHA256,
        quotient["equivalences"]["reporter_period_lexicon"]["source_permutation"].endswith("modulo 16"),
        all(low > 0.0 and high >= low for pair in expected_ranges.values() for low, high in pair),
        all(result["score"]["record_count"] == 320 for result in (construction, family, lexicon)),
        all(all(result["predictions"].values()) for result in (construction, family, lexicon)),
        "opposite-direction signed equivalence" in quotient["not_claimed"],
        "unrestricted corpus invariance" in quotient["not_claimed"],
        "gauge-independent coordinate identity" in quotient["not_claimed"],
        loaded["artifact_v11"]["computational_change"].startswith("none"),
        loaded["artifact_v11"]["price"]["additional_runtime_compute"] == 0,
        loaded["artifact_v11"]["price"]["additional_stored_fit_scalars"] == 0,
    ]
    pred_d = len(assertions) >= 20 and all(assertions)
    exclusions = {"unrestricted corpus invariance", "opposite-direction signed equivalence", "gauge-independent coordinate identity", "raw-text-to-resid18 computation", "whole-model replacement"}
    pred_e = prior["price"]["additional_stored_fit_scalars"] == 0 and prior["price"]["additional_runtime_compute"] == 0 and loaded["artifact_v11"]["price"]["quotient_variable_groups"] == 3 and exclusions == set(loaded["artifact_v11"]["scope"]["not_licensed"])
    predictions = {"pred_a_hash_bound_valid_authority": pred_a, "pred_b_parent_operations_preserved": pred_b, "pred_c_equivalence_evidence_exact": pred_c, "pred_d_manifest_conformance": pred_d, "pred_e_price_and_scope_exact": pred_e}
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {"schema": "aspectual_anchor_transparent_path_program_release_result_v11", "candidate_id": prior["candidate_id"], "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": sha(PRIOR), "authority_sha256": observed, "program_sha256": observed["program_v11"], "artifact_sha256": observed["artifact_v11"], "predictions": predictions, "manifest_assertions": len(assertions), "classification": "valid_executable_donor_free_operational_quotient_aspectual_program", "price": prior["price"], "terminal": terminal}
    payload = atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "manifest_assertions": len(assertions), "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
