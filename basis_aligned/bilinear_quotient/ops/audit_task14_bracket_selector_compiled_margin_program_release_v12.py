#!/usr/bin/env python3
"""Hash-bound release audit for the unified selector-compiled margin program."""

# BQGATE: AUDIT pred_a_hash_bound_chain_and_inventory pred_b_unified_api_conformance pred_c_selector_evidence_bound pred_d_dependency_and_import_boundary pred_e_scope_preserved
from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import transparent_margin_program as v7
import transparent_margin_program_v11 as program

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/task14_bracket_selector_compiled_margin_program_release_v12.json"
MANIFEST = ROOT / "circuits/followups/task14_bracket_selector_compiled_margin_program_release_v12_manifest.json"
OUT = ROOT / "circuits/followups/task14_bracket_selector_compiled_margin_program_release_v12_result.json"
EXPECTED = {
    ROOT / "ops/transparent_margin_program_v11.py": "78b58e45dde1e364596e31a39121351695c402191fe34fc9db9a643da9b24ac7",
    ROOT / "circuits/followups/task14_text_direction_selector_program_release_v11_result.json": "b0f6c1e77b7f4a044a07f45d5dca83ee3235555be66b48c34709dc601fc6e6c0",
    ROOT / "circuits/followups/bracket_circuit_source_selector_release_v10_result.json": "fe462f9c6576a068ea3c3a52a20512a68029e04959dafcf4b4ca1c8608c90180",
    ROOT / "circuits/followups/task14_bracket_text_selector_program_release_v9_result.json": "14beedea97bd4cbeb20571e5ca8364c048d1c422ff1ea943e16fa7b0aa410860",
    ROOT / "circuits/followups/task14_bracket_transparent_program_boundary_certificate_v8_result.json": "7364d8dde67343cc3d222f97558f7bdbaa8df5737ad71abb60a35e1948524790",
    ROOT / "circuits/followups/task14_standalone_bracket_conditioned_hybrid_v6_artifact.json": "c365321035a7cf7886f3038ac29a76659b9f3c968bf044e6ef84bf448dd5218d",
}
FORBIDDEN = {"torch", "transformers", "requests", "socket", "subprocess", "urllib", "numpy", "sklearn", "importlib", "runpy", "pickle"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split('.')[0])
    return names


def main() -> None:
    if OUT.exists() or MANIFEST.exists():
        raise ValueError("refusing overwrite")
    prior = json.loads(PRIOR.read_text()); observed = {path.name: sha(path) for path in EXPECTED}
    loaded = [json.loads(path.read_text()) for path in list(EXPECTED)[1:]]
    v11, v10, v9, v8, artifact = loaded
    chain = all(observed[path.name] == digest for path, digest in EXPECTED.items()) and prior["authority"] == {path.name: digest for path, digest in EXPECTED.items()} and [v11["terminal"], v10["terminal"], v9["terminal"], v8["terminal"], artifact["terminal"]] == ["screen", "screen", "release", "certificate", "frozen_hybrid_program"] and artifact["stored_fp32_scalars"] == 22
    frozen = v7.load_artifact()
    requests = [
        {"behavior": "task14", "direction": "singular_to_plural", "background": "EA", "edit": True},
        {"behavior": "task14_text", "text": "Near the pilots beyond the guide the cook", "background": "EA", "edit": True},
        {"behavior": "bracket", "native_unedited_donorward_margin": -7.0, "recipient_closer_id": 8, "donor_closer_id": 60},
        {"behavior": "bracket_text", "text": "The clerk closed [ item ], then opened ( value", "native_unedited_donorward_margin": -7.0, "donor_closer_id": 60},
    ]
    outputs = [program.dispatch(frozen, request) for request in requests]
    api = outputs[0]["predicted_counterfactual_donorward_margin"] == outputs[1]["predicted_counterfactual_donorward_margin"] and outputs[2]["predicted_counterfactual_donorward_margin"] == outputs[3]["predicted_counterfactual_donorward_margin"] and outputs[1]["inferred_direction"] == "singular_to_plural" and outputs[3]["inferred_recipient_closer_id"] == 8
    evidence = v11["score"]["endpoint_texts"] == 96 and v11["score"]["equation_cases"] == 3072 and v11["score"]["equation_failures"] == 0 and v10["score"]["endpoint_cases"] == 2088 and v10["score"]["recipient_failures"] == 0 and v10["score"]["source_failures"] == 0 and v9["score"]["equation_cases"] == 6264 and v9["score"]["equation_failures"] == 0
    imported = imports(ROOT / "ops/transparent_margin_program_v11.py") | imports(ROOT / "ops/transparent_margin_program_v9.py") | imports(ROOT / "ops/transparent_margin_program.py")
    dependency = not (imported & FORBIDDEN)
    scope = v8["certified_boundary"]["current_release"] == "v7 22-scalar hybrid" and v8["certified_boundary"]["bracket"] == "one native unedited donorward margin plus six frozen effects"
    predictions = {"pred_a_hash_bound_chain_and_inventory": chain, "pred_b_unified_api_conformance": api, "pred_c_selector_evidence_bound": evidence, "pred_d_dependency_and_import_boundary": dependency, "pred_e_scope_preserved": scope}
    terminal = "release" if all(predictions.values()) else "invalid"
    manifest_value = {"schema": "task14_bracket_selector_compiled_margin_program_manifest_v12", "candidate_id": prior["candidate_id"], "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "program": {"path": "ops/transparent_margin_program_v11.py", "sha256": EXPECTED[ROOT / "ops/transparent_margin_program_v11.py"]}, "artifact": {"path": "circuits/followups/task14_standalone_bracket_conditioned_hybrid_v6_artifact.json", "sha256": EXPECTED[ROOT / "circuits/followups/task14_standalone_bracket_conditioned_hybrid_v6_artifact.json"], "stored_fp32_scalars": 22, "stored_fp32_bytes": 88}, "interfaces": {"task14_text": {"inputs": ["controlled raw subject text", "E/A/U/W background subset", "edit boolean"], "output": "native/effect/counterfactual donorward margins"}, "bracket_text": {"inputs": ["controlled raw delimiter text", "native unedited donorward margin", "desired donor closer id"], "output": "native/effect/counterfactual donorward margins"}, "bracket_internal_selector": {"inputs": ["controlled raw delimiter text", "native token ids"], "output": ["recipient closer id", "L13H8 semantic opener token position"]}}, "runtime_dependencies": {"task14": ["controlled raw text", "E/A/U/W background/edit specification"], "bracket_margin": ["controlled raw text", "native unedited donorward margin", "desired edit"], "bracket_internal_intervention": ["controlled raw text", "native token IDs", "desired edit", "native prefix/base activation and suffix"]}, "classification": "selector_compiled_controlled_domain_task14_standalone_bracket_baseline_conditioned_predictive_composable_manipulable_margin_program_not_whole_model", "terminal": "release" if terminal == "release" else "invalid"}
    manifest_bytes = managed.atomic_create_json(MANIFEST, manifest_value)
    payload = managed.atomic_create_json(OUT, {"schema": "task14_bracket_selector_compiled_margin_program_release_result_v12", "candidate_id": prior["candidate_id"], "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": sha(PRIOR), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "authority_sha256": observed, "score": {"stored_fp32_scalars": 22, "stored_fp32_bytes": 88, "imports": sorted(imported), "forbidden_imports_present": sorted(imported & FORBIDDEN), "representative_api_cases": len(requests), "bound_selector_cases": {"task14_endpoint_texts": 96, "task14_equations": 3072, "bracket_endpoints": 2088, "bracket_equations": 6264}, "predictions": predictions, "classification": manifest_value["classification"], "terminal": terminal}, "terminal": terminal})
    print(json.dumps({"terminal": terminal, "predictions": predictions, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
