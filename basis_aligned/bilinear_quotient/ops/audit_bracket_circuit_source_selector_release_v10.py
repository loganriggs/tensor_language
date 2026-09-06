#!/usr/bin/env python3
"""Exhaustive zero-forward audit of the bracket circuit source selector."""

# BQGATE: AUDIT pred_a_v9_and_authority_bound pred_b_exact_recipient_state pred_c_exact_l13h8_source_position pred_d_strict_token_rejection pred_e_dependency_and_price
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import transparent_margin_program as v7
import transparent_bracket_circuit_selector_v10 as selector

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/bracket_circuit_source_selector_release_v10.json"
OUT = ROOT / "circuits/followups/bracket_circuit_source_selector_release_v10_result.json"
EXPECTED = {
    ROOT / "circuits/followups/task14_bracket_text_selector_program_release_v9_result.json": "14beedea97bd4cbeb20571e5ca8364c048d1c422ff1ea943e16fa7b0aa410860",
    ROOT / "ops/transparent_margin_program_v9.py": "52b89038ce7982c34ca7759beffde569c131ff009565810d7b5b8d123a9d250f",
    ROOT / "pending_opener_three_value_fresh_rows_rung545.json": "07b64d2e48a6ca67685c81d3475a064daba612d6fe7ff233efd5b6c157b940a9",
    ROOT / "circuits/prior_art/bracket_suffix_free_fresh_corpus_v1_rows.json": "d808806fd1b05f834cf6ef4fa71465464c0403f66dc13ece8a24cffcc40142f9",
    ROOT / "circuits/prior_art/bracket_absolute_term_fresh_corpus_v1_rows.json": "92ee66ce0a4bf084789bf0c2af394a107a49df19dbf3c9fbfbebbe467d873c76",
    ROOT / "circuits/prior_art/bracket_native_baseline_fresh_corpus_v1_rows.json": "ad246a0ab2affd0a351b971c100c27c2ad09597d0d9e7b84b636e1eb4c8fb399",
    ROOT / "circuits/prior_art/bracket_l13h8_direct_readout_fresh_corpus_v1_rows.json": "09424b15ad797491b4968bdb9e84b3f81f7062a6b42b057e85c725b25c1b4f8c",
}


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
    authority = all(observed[path.name] == digest for path, digest in EXPECTED.items()) and prior["authority"] == {path.name: digest for path, digest in EXPECTED.items()}
    v9 = json.loads(list(EXPECTED)[0].read_text())
    authority = authority and v9["terminal"] == "release"
    recipient_failures = source_failures = cases = 0
    corpus_counts = {}
    for path in list(EXPECTED)[2:]:
        rows = json.loads(path.read_text())["rows"]
        corpus_counts[path.name] = 2 * len(rows)
        for row in rows:
            for side in ("base", "donor"):
                cases += 1
                selected = selector.select(row[f"{side}_text"], row[f"{side}_ids"])
                answer = row[f"{side}_answer_id"]
                expected_opener = selector.OPENER_TOKEN_BY_CLOSER[answer]
                expected_positions = [index for index, token in enumerate(row[f"{side}_ids"]) if token == expected_opener]
                if selected["recipient_closer_id"] != answer:
                    recipient_failures += 1
                if not expected_positions or selected["semantic_open_position"] != expected_positions[-1] or selected["semantic_open_token_id"] != expected_opener:
                    source_failures += 1
    rejection = {
        "non_list": rejected(lambda: selector.select("pending ( item", "not tokens")),
        "empty": rejected(lambda: selector.select("pending ( item", [])),
        "boolean_token": rejected(lambda: selector.select("pending ( item", [True, 357])),
        "missing_opener": rejected(lambda: selector.select("pending ( item", [10, 11, 12])),
        "raw_token_mismatch": rejected(lambda: selector.select("pending [ item", [10, 357, 12])),
    }
    predictions = {
        "pred_a_v9_and_authority_bound": authority,
        "pred_b_exact_recipient_state": cases == 2088 and recipient_failures == 0,
        "pred_c_exact_l13h8_source_position": cases == 2088 and source_failures == 0,
        "pred_d_strict_token_rejection": all(rejection.values()),
        "pred_e_dependency_and_price": prior["price"] == {"endpoint_cases": 2088, "new_fitted_scalars": 0, "model_forwards": 0, "fits": 0},
    }
    terminal = "screen" if all(predictions.values()) else "invalid"
    value = {"schema": "bracket_circuit_source_selector_release_result_v10", "candidate_id": prior["candidate_id"], "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": sha(PRIOR), "selector_sha256": sha(ROOT / "ops/transparent_bracket_circuit_selector_v10.py"), "authority_sha256": observed, "score": {"corpus_endpoint_counts": corpus_counts, "endpoint_cases": cases, "recipient_failures": recipient_failures, "source_failures": source_failures, "rejection": rejection, "dependencies_removed": ["externally supplied recipient closer", "externally supplied semantic opener source position"], "dependencies_retained": ["raw controlled-domain text", "native token IDs", "desired edit specification", "native prefix/base activation and suffix for internal intervention execution"], "predictions": predictions, "terminal": terminal}, "terminal": terminal}
    payload = managed.atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
