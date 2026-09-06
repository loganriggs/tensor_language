#!/usr/bin/env python3
"""Exhaustive conformance audit and evidence manifest for the public program."""

# BQGATE: EXPERIMENT pred_a_hash_bound_load_and_inventory pred_b_exhaustive_task14_conformance pred_c_exhaustive_bracket_conformance pred_d_dependency_and_import_boundary pred_e_evidence_and_scope_manifest
# BQLANE: cpu
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import transparent_margin_program as program

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/task14_bracket_transparent_margin_program_release_v7.json"
CORRECTION = ROOT / "circuits/prior_art/task14_bracket_transparent_margin_program_release_v7_count_correction.json"
V6_ARTIFACT = ROOT / "circuits/followups/task14_standalone_bracket_conditioned_hybrid_v6_artifact.json"
V6_RESULT = ROOT / "circuits/followups/task14_standalone_bracket_conditioned_hybrid_v6_result.json"
V5_RESULT = ROOT / "circuits/followups/task14_bracket_margin_actuator_composition_contract_v5_result.json"
V2_RESULT = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_prospective_v2_result.json"
EXECUTABLE = ROOT / "ops/transparent_margin_program.py"
MANIFEST = ROOT / "circuits/followups/task14_bracket_transparent_margin_program_release_v7_manifest.json"
OUT = ROOT / "circuits/followups/task14_bracket_transparent_margin_program_release_v7_result.json"
CANDIDATE_ID = "cross_behavior.task14_bracket_transparent_margin_program_release_v7"
EXPECTED = {PRIOR: "cae124f37809d7684cbef60e0e570ad79f8c1dbb8662ebc6e1574c2fcc1e6b6a", CORRECTION: "4b2dfec59457b7d15d35ed98e3cddf083b645f4d5565778cb88c455b37adb19d", V6_ARTIFACT: "c365321035a7cf7886f3038ac29a76659b9f3c968bf044e6ef84bf448dd5218d", V6_RESULT: "fc13ffdb0998b3b4db662c3104891a1dc780c84d49f1990b3c4f2231beacf19a", V5_RESULT: "cef063b5875912af744a0205b55fc0a04296eedd67563ce36f678315dc47032a", V2_RESULT: "1d2f99a6c965ed0d6794cb83a6fb0c8953d11e9a599e769b02d4a0f612d89ea4"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict, dict, dict, dict]:
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"immutable input changed: {path}")
    artifact = program.load_artifact(); v6, v5, v2 = (json.loads(path.read_text()) for path in (V6_RESULT, V5_RESULT, V2_RESULT))
    if v6["terminal"] != "screen" or v5["terminal"] != "screen" or v2["terminal"] != "null":
        raise ValueError("evidence status changed")
    return artifact, v6, v5, v2


def compile_plan() -> dict:
    load()
    return {"schema": "task14_bracket_transparent_margin_program_release_plan_v7", "candidate_id": CANDIDATE_ID, "prior_art_sha256": EXPECTED[PRIOR], "count_correction_sha256": EXPECTED[CORRECTION], "conformance_cases": {"task14": 64, "bracket": 45}, "stored_fp32_scalars": 22, "price": {"model_forwards": 0, "example_evaluations": 0, "fits": 0, "backwards": 0, "parameter_updates": 0}}


def rejects(callable_) -> bool:
    try:
        callable_()
    except program.ProgramError:
        return True
    return False


def evaluate(artifact: dict, v6: dict, v5: dict, v2: dict) -> tuple[dict, dict]:
    inventory = len(artifact["programs"]["task14"]["native_margin_coefficients"]) + len(artifact["programs"]["task14"]["intervention_effects"]) + len(artifact["programs"]["bracket"]["intervention_effects"])
    load_ok = inventory == artifact["stored_fp32_scalars"] == 22 and program.EXPECTED_SHA256 == EXPECTED[V6_ARTIFACT]
    task_cases = 0; task_ok = True
    subsets = ["".join(values) for length in range(5) for values in itertools.combinations(program.LETTERS, length)]
    for direction in ("singular_to_plural", "plural_to_singular"):
        for background in subsets:
            features = [1.0, 1.0 if direction == "singular_to_plural" else -1.0] + [1.0 if letter in background else 0.0 for letter in program.LETTERS]
            native = sum(feature * coefficient for feature, coefficient in zip(features, artifact["programs"]["task14"]["native_margin_coefficients"]))
            for edit in (False, True):
                observed = program.task14(artifact, direction=direction, background=background, edit=edit)
                key = f"{direction}.cardinality_{len(background)}"; effect = artifact["programs"]["task14"]["intervention_effects"][key] if edit else 0.0
                task_ok &= observed["predicted_native_donorward_margin"] == native and observed["predicted_intervention_effect"] == effect and observed["predicted_counterfactual_donorward_margin"] == native + effect
                task_cases += 1
    task_ok &= task_cases == 64 and rejects(lambda: program.task14(artifact, direction="bad", background="", edit=True)) and rejects(lambda: program.task14(artifact, direction="singular_to_plural", background="EE", edit=True)) and rejects(lambda: program.task14(artifact, direction="singular_to_plural", background="X", edit=True)) and rejects(lambda: program.task14(artifact, direction="singular_to_plural", background="", edit=1))
    bracket_cases = 0; bracket_ok = True
    for baseline in (-10.0, -1.0, 0.0, 1.0, 10.0):
        for recipient in (1, 8, 60):
            for donor in (1, 8, 60):
                observed = program.bracket(artifact, native_unedited_donorward_margin=baseline, recipient_closer_id=recipient, donor_closer_id=donor)
                effect = 0.0 if recipient == donor else artifact["programs"]["bracket"]["intervention_effects"][f"{recipient}->{donor}"]
                bracket_ok &= observed["predicted_intervention_effect"] == effect and observed["predicted_counterfactual_donorward_margin"] == baseline + effect
                bracket_cases += 1
    bracket_ok &= bracket_cases == 45 and rejects(lambda: program.bracket(artifact, native_unedited_donorward_margin=float("nan"), recipient_closer_id=1, donor_closer_id=8)) and rejects(lambda: program.bracket(artifact, native_unedited_donorward_margin=0.0, recipient_closer_id=2, donor_closer_id=8))
    tree = ast.parse(EXECUTABLE.read_text()); imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    forbidden = {"torch", "transformers", "numpy", "requests", "urllib", "socket", "facade", "circuit_fast_screen_managed_runner"}
    dependency_ok = not imports & forbidden and artifact["runtime_dependencies"] == {"task14": ["direction", "E/A/U/W membership", "edit specification"], "bracket": ["native unedited donorward margin", "ordered closer edit"]}
    manifest = {"schema": "task14_bracket_transparent_margin_program_manifest_v7", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "executable": "ops/transparent_margin_program.py", "executable_sha256": sha(EXECUTABLE), "program_artifact": "circuits/followups/task14_standalone_bracket_conditioned_hybrid_v6_artifact.json", "program_artifact_sha256": EXPECTED[V6_ARTIFACT], "stored_fp32_scalars": 22, "runtime_dependencies": artifact["runtime_dependencies"], "prospective_evidence": {"task14_native": v6["score"]["task14_prospective"]["baseline"], "task14_counterfactual": v6["score"]["task14_prospective"]["counterfactual"], "bracket_effect": v6["score"]["bracket_newest_corpus_effect_recurrence"]}, "composition_exhaustive_cases": v5["score"]["exhaustive_cases"], "preserved_nulls": {"combined_standalone_v2": v2["terminal"], "standalone_bracket_native_margin": "closed"}, "closed_claims": artifact["explicitly_not_provided"], "classification": v6["score"]["classification"]}
    manifest_ok = manifest["preserved_nulls"] == {"combined_standalone_v2": "null", "standalone_bracket_native_margin": "closed"} and manifest["composition_exhaustive_cases"] == {"identity": 35, "idempotence": 95, "same_slot_ordered_overwrite": 905, "independent_slot_commutativity": 2250} and manifest["stored_fp32_scalars"] == 22
    predictions = {"pred_a_hash_bound_load_and_inventory": load_ok, "pred_b_exhaustive_task14_conformance": task_ok, "pred_c_exhaustive_bracket_conformance": bracket_ok, "pred_d_dependency_and_import_boundary": dependency_ok, "pred_e_evidence_and_scope_manifest": manifest_ok}
    result = {"conformance_cases": {"task14": task_cases, "bracket": bracket_cases}, "imports": sorted(imports), "forbidden_imports_present": sorted(imports & forbidden), "stored_fp32_scalars": inventory, "runtime_dependencies": artifact["runtime_dependencies"], "classification": v6["score"]["classification"], "predictions": predictions, "terminal": "release" if all(predictions.values()) else "null"}
    return manifest, result


def main() -> None:
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, sort_keys=True)); return
    if MANIFEST.exists() or OUT.exists():
        raise ValueError("refusing overwrite")
    artifact, v6, v5, v2 = load(); manifest, result = evaluate(artifact, v6, v5, v2)
    manifest_bytes = managed.atomic_create_json(MANIFEST, manifest)
    payload = managed.atomic_create_json(OUT, {"schema": "task14_bracket_transparent_margin_program_release_result_v7", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "plan": plan, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "score": result, "terminal": result["terminal"]})
    print(json.dumps({"terminal": result["terminal"], "predictions": result["predictions"], "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
