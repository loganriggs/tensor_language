#!/usr/bin/env python3
"""CPU-only audit of pre-registry bracket/quote evidence relevant to pending opener."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "legacy_bracket_quote_evidence_audit_rung547.json"
FILES = {
    "bracket_match_implementation": ROOT / "bracket_match.py",
    "bracket_match_result": ROOT / "bracket_match_results.json",
    "bracket_pointer_implementation": ROOT / "bracket_pointer_pairs.py",
    "bracket_pointer_result": ROOT / "bracket_pointer_pairs_results.json",
    "bracket_query_rank_implementation": ROOT / "bracket_query_rank.py",
    "bracket_query_rank_result": ROOT / "bracket_query_rank_results.json",
    "quote_head_implementation": ROOT / "quote_close_heads.py",
    "quote_head_result": ROOT / "quote_close_heads_results.json",
    "quote_state_causal_implementation": ROOT / "quote_state_causal.py",
    "quote_state_causal_result": ROOT / "quote_state_causal_results.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path in FILES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    match = json.loads(FILES["bracket_match_result"].read_text())
    pointer = json.loads(FILES["bracket_pointer_result"].read_text())
    rank = json.loads(FILES["bracket_query_rank_result"].read_text())
    quote = json.loads(FILES["quote_head_result"].read_text())
    parity = json.loads(FILES["quote_state_causal_result"].read_text())

    assert match["n_targets"] == 84 and match["n_nested"] == 1
    assert match["pred_a"] and match["pred_b"] and match["pred_c"]
    assert pointer["pred_0"] and not pointer["pred_a"] and not pointer["pred_b"] and pointer["pred_c"]
    assert rank["pred_0"] and not rank["pred_a"] and rank["pred_b"] and rank["kill80_rank"] == 64
    assert quote["pred_a_owner"] and quote["pred_b_directions_agree"] and quote["pred_c_surgical"]
    assert parity["pred_0"] and not parity["pred_a_causal"] and parity["null_ok"]

    source_text = {name: path.read_text() for name, path in FILES.items() if name.endswith("implementation")}
    unsealed = {
        name: {
            "uses_mutable_fineweb_rows": "fineweb_rows(" in text,
            "hardcodes_live_checkpoint_import": "bilin18_joint_removal import m" in text,
            "writes_result_without_checkpoint_hash": "weights_sha256" not in text,
            "no_fit_select_final_ood_authority": not all(word in text for word in ("FIT", "SELECT", "FINAL", "OOD")),
        }
        for name, text in source_text.items()
    }
    assert all(item["uses_mutable_fineweb_rows"] for item in unsealed.values())
    assert all(item["hardcodes_live_checkpoint_import"] for item in unsealed.values())
    assert all(item["writes_result_without_checkpoint_hash"] for item in unsealed.values())

    audit = {
        "rung": 547, "status": "legacy_evidence_audited_no_model_access",
        "files_sha256": {name: sha256(path) for name, path in FILES.items()},
        "provenance_limits": unsealed,
        "bracket_match": {
            "descriptive_effect_nat": match["arms"]["kill_match"]["target"],
            "whole_head_deletion_nat": match["arms"]["delete"]["target"],
            "global_deletion_nat": match["arms"]["delete"]["global"],
            "n_targets": match["n_targets"], "n_nested_control": match["n_nested"],
            "interpretation": "strong matched-edge localization on one unsealed draw; nesting generalization untested (n=1)",
        },
        "bracket_pointer_pairs": {
            "exact_relative_score_error": pointer["rel_score"],
            "top10_match_mass_fraction": pointer["match_top10_share"],
            "top10_pair_set_difference": pointer["pairs_differing"],
            "sparsity_prediction_held": pointer["pred_a"],
            "match_specific_pair_identity_held": pointer["pred_b"],
            "interpretation": "exact dense algebra replay, but the proposed sparse writer-pair pointer failed",
        },
        "bracket_query_rank": {
            "rank8_fraction_of_full_effect": rank["curve"]["8"]["target"] / rank["full_cost"],
            "smallest_rank_reaching_80_percent": rank["kill80_rank"],
            "rank8_prediction_held": rank["pred_a"],
            "interpretation": "query effect is selective but not an eight-dimensional explanation; rank 64 was needed for 80%",
        },
        "quote_head": {
            "l13h8_target_deletion_nat": quote["solo"]["8"]["target"],
            "l13h8_elsewhere_deletion_nat": quote["solo"]["8"]["else"],
            "n_targets": quote["n_targets"],
            "interpretation": "strong unsealed head-ownership evidence; no independent counterfactual or OOD split",
        },
        "quote_parity_direction": {
            "baseline_probability_gap": parity["baseline_gap"],
            "fraction_gap_lost_after_rank1_removal": parity["wquote_lost_frac"],
            "causal_prediction_held": parity["pred_a_causal"],
            "interpretation": "the decoded rank-one quote-parity direction was not a causal carrier",
        },
        "correction": (
            "The 2026-08-30 campaign's bracket Tier-4 label overstates the surviving evidence. L13H8 and its matched "
            "score edge are strong hypotheses, and the 625-term score expansion is numerically exact, but sparse "
            "writer-pair identity failed, rank eight failed, nesting had n=1, and the rows/checkpoint were not sealed. "
            "R545/R546 are therefore confirmation rather than duplication."
        ),
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0, "outcomes_opened": [],
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
