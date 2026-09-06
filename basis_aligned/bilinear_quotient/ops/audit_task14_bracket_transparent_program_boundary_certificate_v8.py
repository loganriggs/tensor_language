#!/usr/bin/env python3
"""Zero-forward boundary certificate for the executable hybrid margin program."""

# BQGATE: AUDIT pred_a_release_and_positive_evidence_bound pred_b_two_prospective_baseline_nulls_bound pred_c_causal_localization_preserved pred_d_scope_exact pred_e_next_route_nonlocal
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/task14_bracket_transparent_program_boundary_certificate_v8.json"
OUT = ROOT / "circuits/followups/task14_bracket_transparent_program_boundary_certificate_v8_result.json"
EXPECTED = {
    ROOT / "circuits/followups/task14_bracket_transparent_margin_program_release_v7_result.json": "55ade3e9f15043bc046d89913bd96eec4e2b1d9398d6cbd3bbe63014b6099182",
    ROOT / "circuits/followups/task14_standalone_bracket_conditioned_hybrid_v6_result.json": "fc13ffdb0998b3b4db662c3104891a1dc780c84d49f1990b3c4f2231beacf19a",
    ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_prospective_v2_result.json": "1d2f99a6c965ed0d6794cb83a6fb0c8953d11e9a599e769b02d4a0f612d89ea4",
    ROOT / "circuits/followups/bracket_native_baseline_l13h8_causal_ceiling_newest_v1_result.json": "670ec697b68c0e74f7ee8b11d33dace33c16a726acd7272738d9ef876f214cf1",
    ROOT / "circuits/followups/bracket_l13h8_direct_readout_baseline_program_v1_result.json": "534730d3831526a851571642f50ca00315dbb3a3953f6f0a00888ba7d945171c",
    ROOT / "circuits/followups/bracket_l13h8_direct_readout_baseline_program_v1_artifact.json": "4d5e53e9f09df68204e92eea4b60ee48e7f7685a06555a91404120ba0a2a0da6",
    ROOT / "ops/transparent_margin_program.py": "ad0aadf4318fae7c13908350ab49a87cad00d20ae5386fed5ec27ba0851ab98d",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise ValueError("refusing overwrite")
    prior = json.loads(PRIOR.read_text())
    observed = {path.name: sha(path) for path in EXPECTED}
    if any(observed[path.name] != digest for path, digest in EXPECTED.items()):
        raise ValueError("immutable authority changed")
    if prior["authority"] != {path.name: digest for path, digest in EXPECTED.items()}:
        raise ValueError("prior authority mismatch")
    v7, v6, v2, ceiling, direct, artifact = [json.loads(path.read_text()) for path in list(EXPECTED)[:6]]
    release = v7["terminal"] == "release" and v6["terminal"] == "screen" and v6["score"]["task14_prospective"]["counterfactual"]["cosine"] >= .98 and v6["score"]["bracket_newest_corpus_effect_recurrence"]["cosine"] >= .99
    two_nulls = v2["terminal"] == "null" and direct["terminal"] == "null" and not v2["score"]["predictions"]["pred_d_prospective_absolute_counterfactuals"] and not direct["score"]["predictions"]["pred_d_absolute_counterfactual_precision"]
    causal = ceiling["terminal"] == "screen" and ceiling["score"]["semantic_open"]["damage_positive_fraction"] == 1.0 and direct["score"]["prospective_direct_damage"]["cosine"] >= .98
    scope = artifact["stored_fp32_scalars"] == 6 and direct["score"]["dependency_boundary"]["classification"] == "circuit_conditioned_baseline_and_counterfactual_margin_program_not_standalone_not_whole_model" and v7["score"]["classification"] == "task14_standalone_bracket_baseline_conditioned_predictive_composable_manipulable_margin_program_not_whole_model"
    nonlocal_next = direct["score"]["prospective_native_baseline"]["relative_l2_error"] > direct["plan"]["bars"]["maximum_prospective_baseline_relative_l2"]
    predictions = {"pred_a_release_and_positive_evidence_bound": release, "pred_b_two_prospective_baseline_nulls_bound": two_nulls, "pred_c_causal_localization_preserved": causal, "pred_d_scope_exact": scope, "pred_e_next_route_nonlocal": nonlocal_next}
    terminal = "certificate" if all(predictions.values()) else "invalid"
    value = {"schema": "task14_bracket_transparent_program_boundary_certificate_result_v8", "candidate_id": prior["candidate_id"], "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": sha(PRIOR), "authority_sha256": observed, "predictions": predictions, "terminal": terminal, "certified_boundary": {"current_release": "v7 22-scalar hybrid", "task14": "standalone prospective native and counterfactual margins", "bracket": "one native unedited donorward margin plus six frozen effects", "empirical_minimality_scope": ["five-coefficient semantic-linear no-native-input bracket baseline", "five semantic-zero residual coefficients plus one fixed L13H8 direct-readout gain"], "closed_routes": ["local bracket-baseline feature rescue", "local L13H8 site/rank/readout rescue", "reinterpretation of causal importance as standalone predictability"], "not_claimed": ["universal minimality", "whole-model replacement", "free-form text support"]}, "price": prior["price"]}
    payload = managed.atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
