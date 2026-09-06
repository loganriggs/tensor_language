#!/usr/bin/env python3
"""Exhaustive zero-forward audit of the raw-text bracket selector release."""

# BQGATE: AUDIT pred_a_authority_and_v8_boundary pred_b_exact_text_selector pred_c_v7_equation_conformance pred_d_strict_malformed_rejection pred_e_scope_and_price
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import transparent_margin_program as v7
import transparent_margin_program_v9 as v9

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/task14_bracket_text_selector_program_release_v9.json"
OUT = ROOT / "circuits/followups/task14_bracket_text_selector_program_release_v9_result.json"
EXPECTED = {
    ROOT / "ops/transparent_margin_program.py": "ad0aadf4318fae7c13908350ab49a87cad00d20ae5386fed5ec27ba0851ab98d",
    ROOT / "circuits/followups/task14_bracket_transparent_program_boundary_certificate_v8_result.json": "7364d8dde67343cc3d222f97558f7bdbaa8df5737ad71abb60a35e1948524790",
    ROOT / "pending_opener_three_value_fresh_rows_rung545.json": "07b64d2e48a6ca67685c81d3475a064daba612d6fe7ff233efd5b6c157b940a9",
    ROOT / "circuits/prior_art/bracket_suffix_free_fresh_corpus_v1_rows.json": "d808806fd1b05f834cf6ef4fa71465464c0403f66dc13ece8a24cffcc40142f9",
    ROOT / "circuits/prior_art/bracket_absolute_term_fresh_corpus_v1_rows.json": "92ee66ce0a4bf084789bf0c2af394a107a49df19dbf3c9fbfbebbe467d873c76",
    ROOT / "circuits/prior_art/bracket_native_baseline_fresh_corpus_v1_rows.json": "ad246a0ab2affd0a351b971c100c27c2ad09597d0d9e7b84b636e1eb4c8fb399",
    ROOT / "circuits/prior_art/bracket_l13h8_direct_readout_fresh_corpus_v1_rows.json": "09424b15ad797491b4968bdb9e84b3f81f7062a6b42b057e85c725b25c1b4f8c",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect_rejection(function) -> bool:
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
    authority = all(observed[path.name] == digest for path, digest in EXPECTED.items()) and prior["authority"] == {path.name: digest for path, digest in EXPECTED.items()}
    boundary = json.loads(list(EXPECTED)[1].read_text())
    authority = authority and boundary["terminal"] == "certificate"
    rows = []
    corpus_counts = {}
    for path in list(EXPECTED)[2:]:
        value = json.loads(path.read_text()); corpus = value["rows"]
        corpus_counts[path.name] = 2 * len(corpus); rows.extend(corpus)
    artifact = v7.load_artifact()
    selector_cases = equation_cases = selector_failures = equation_failures = 0
    for row in rows:
        for side in ("base", "donor"):
            text = row[f"{side}_text"]; recipient = row[f"{side}_answer_id"]
            selector_cases += 1
            if v9.pending_closer_id(text) != recipient:
                selector_failures += 1
            for donor in (1, 8, 60):
                equation_cases += 1
                new = v9.bracket_text(artifact, text=text, native_unedited_donorward_margin=-3.25, donor_closer_id=donor)
                old = v7.bracket(artifact, native_unedited_donorward_margin=-3.25, recipient_closer_id=recipient, donor_closer_id=donor)
                for key in ("native_unedited_donorward_margin", "predicted_intervention_effect", "predicted_counterfactual_donorward_margin", "edit_key"):
                    if new[key] != old[key]:
                        equation_failures += 1; break
    rejection = {
        "balanced": expect_rejection(lambda: v9.pending_closer_id('finished ( item )')),
        "multiply_pending": expect_rejection(lambda: v9.pending_closer_id('unfinished ( item [ value')),
        "mismatched_close": expect_rejection(lambda: v9.pending_closer_id('broken ( item ]')),
        "non_string": expect_rejection(lambda: v9.pending_closer_id(7)),
        "invalid_target": expect_rejection(lambda: v9.bracket_text(artifact, text='pending ( item', native_unedited_donorward_margin=-1.0, donor_closer_id=999)),
    }
    predictions = {
        "pred_a_authority_and_v8_boundary": authority,
        "pred_b_exact_text_selector": selector_cases == 2088 and selector_failures == 0,
        "pred_c_v7_equation_conformance": equation_cases == 6264 and equation_failures == 0,
        "pred_d_strict_malformed_rejection": all(rejection.values()),
        "pred_e_scope_and_price": prior["price"] == {"stored_fp32_scalars": 22, "new_fitted_scalars": 0, "model_forwards": 0, "fits": 0, "selector_prompt_cases": 2088, "equation_cases": 6264},
    }
    terminal = "release" if all(predictions.values()) else "invalid"
    value = {"schema": "task14_bracket_text_selector_program_release_result_v9", "candidate_id": prior["candidate_id"], "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": sha(PRIOR), "program_sha256": sha(ROOT / "ops/transparent_margin_program_v9.py"), "authority_sha256": observed, "score": {"corpus_endpoint_counts": corpus_counts, "selector_cases": selector_cases, "selector_failures": selector_failures, "equation_cases": equation_cases, "equation_failures": equation_failures, "malformed_rejection": rejection, "stored_fp32_scalars": 22, "new_fitted_scalars": 0, "runtime_dependencies": {"removed": ["externally supplied bracket recipient_closer_id"], "retained": ["raw controlled-domain bracket text", "native unedited donorward margin", "desired donor closer/edit specification"]}, "predictions": predictions, "terminal": terminal}, "terminal": terminal}
    payload = managed.atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
