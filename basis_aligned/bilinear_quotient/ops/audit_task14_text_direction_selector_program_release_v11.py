#!/usr/bin/env python3
"""Exhaustive zero-forward audit of Task14 raw-text direction inference."""

# BQGATE: AUDIT pred_a_authority_and_v10_bound pred_b_exact_subject_number_direction pred_c_v7_equation_conformance pred_d_strict_malformed_rejection pred_e_scope_and_price
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import transparent_margin_program as v7
import transparent_margin_program_v11 as v11

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/task14_text_direction_selector_program_release_v11.json"
ROWS = ROOT / "circuits/prior_art/task14_native_baseline_fresh_corpus_v1_rows.json"
V10 = ROOT / "circuits/followups/bracket_circuit_source_selector_release_v10_result.json"
V9 = ROOT / "ops/transparent_margin_program_v9.py"
OUT = ROOT / "circuits/followups/task14_text_direction_selector_program_release_v11_result.json"
EXPECTED = {V9: "52b89038ce7982c34ca7759beffde569c131ff009565810d7b5b8d123a9d250f", V10: "fe462f9c6576a068ea3c3a52a20512a68029e04959dafcf4b4ca1c8608c90180", ROWS: "564d03ae74202b5e0e1be0ce272464362974e2a2f9f6f587c9587319ef829360"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejected(function) -> bool:
    try:
        function()
    except v7.ProgramError:
        return True
    return False


def main() -> None:
    if OUT.exists():
        raise ValueError("refusing overwrite")
    prior = json.loads(PRIOR.read_text())
    observed = {path.name: sha(path) for path in EXPECTED}
    authority = all(observed[path.name] == digest for path, digest in EXPECTED.items()) and prior["authority"] == {path.name: digest for path, digest in EXPECTED.items()} and json.loads(V10.read_text())["terminal"] == "screen"
    rows = json.loads(ROWS.read_text())["rows"]
    artifact = v7.load_artifact(); backgrounds = ["".join(combo) for size in range(5) for combo in itertools.combinations(v7.LETTERS, size)]
    endpoint_cases = equation_cases = number_failures = direction_failures = equation_failures = 0
    for row in rows:
        for endpoint in row["endpoints"].values():
            endpoint_cases += 1
            expected_number = endpoint["subject_number"]
            expected_direction = "singular_to_plural" if expected_number == "singular" else "plural_to_singular"
            if v11.subject_number(endpoint["text"]) != expected_number:
                number_failures += 1
            for background in backgrounds:
                for edit in (False, True):
                    equation_cases += 1
                    new = v11.task14_text(artifact, text=endpoint["text"], background=background, edit=edit)
                    old = v7.task14(artifact, direction=expected_direction, background=background, edit=edit)
                    if new["inferred_direction"] != expected_direction:
                        direction_failures += 1
                    for key in ("predicted_native_donorward_margin", "predicted_intervention_effect", "predicted_counterfactual_donorward_margin", "edit_key"):
                        if new[key] != old[key]:
                            equation_failures += 1; break
    rejection = {"non_string": rejected(lambda: v11.subject_number(4)), "empty": rejected(lambda: v11.subject_number("")), "nonalphabetic": rejected(lambda: v11.subject_number("1234")), "ambiguous_ss": rejected(lambda: v11.subject_number("The final subject is moss"))}
    predictions = {"pred_a_authority_and_v10_bound": authority, "pred_b_exact_subject_number_direction": endpoint_cases == 96 and number_failures == 0 and direction_failures == 0, "pred_c_v7_equation_conformance": equation_cases == 3072 and equation_failures == 0, "pred_d_strict_malformed_rejection": all(rejection.values()), "pred_e_scope_and_price": prior["price"] == {"endpoint_texts": 96, "equation_cases": 3072, "new_fitted_scalars": 0, "model_forwards": 0, "fits": 0}}
    terminal = "screen" if all(predictions.values()) else "invalid"
    value = {"schema": "task14_text_direction_selector_program_release_result_v11", "candidate_id": prior["candidate_id"], "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": sha(PRIOR), "program_sha256": sha(ROOT / "ops/transparent_margin_program_v11.py"), "authority_sha256": observed, "score": {"endpoint_texts": endpoint_cases, "number_failures": number_failures, "direction_failures": direction_failures, "equation_cases": equation_cases, "equation_failures": equation_failures, "rejection": rejection, "dependencies_removed": ["externally supplied Task14 direction"], "dependencies_retained": ["controlled raw Task14 text", "E/A/U/W background membership", "edit/no-edit specification"], "predictions": predictions, "terminal": terminal}, "terminal": terminal}
    payload = managed.atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
