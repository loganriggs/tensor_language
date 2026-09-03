"""Machine-checked multi-counterfactual contract for weight-compiled DAS pilots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "bilinear_quotient" / "circuit_counterfactual_contract_rung536.json"

COMMON_PROMOTION = {
    "causal_variable_first": (
        "Declare one causal variable. Different edits belong to one shared projector only if they "
        "are claimed to change the same variable."
    ),
    "leave_one_family_out": (
        "freeze rank and site, fit on all but one answer-changing family, and require signed causal "
        "transfer to the held-out family without refitting"
    ),
    "shared_private_comparison": (
        "compare family-specific projectors with a shared-plus-private model; shared structure must "
        "transfer, while any private residual may improve only its own family"
    ),
    "cross_family_identification": (
        "compare downstream causal responses and W_D-weighted response geometry; raw basis-vector "
        "equality is not required"
    ),
    "selectivity": (
        "answer-preserving controls and registered unrelated behaviors must remain within frozen bounds"
    ),
    "site_ceiling": (
        "before optimization, native/full-product interchange at the proposed site must move the "
        "registered endpoint in the required direction"
    ),
    "ood": "hold out document, template, token/entity identities, and at least one stimulus family",
    "null": (
        "if no shared projector transfers across families, split the proposed variable or reject it; "
        "do not call a family-specific low-rank subspace the circuit"
    ),
}


def family(
    family_id: str,
    role: str,
    variable_changes: bool,
    answer_changes: bool,
    intervention: str,
    changed_variable: str,
    held_fixed: list[str],
    endpoint: str,
    controls: list[str],
) -> dict:
    return {
        "family_id": family_id,
        "intervention_role": role,
        "proposed_variable_changes": variable_changes,
        "answer_changes": answer_changes,
        "intervention": intervention,
        "changed_variable": changed_variable,
        "held_fixed": held_fixed,
        "endpoint": endpoint,
        "controls": controls,
    }


PILOTS = [
    {
        "circuit_id": "induction_selector_and_payload",
        "status": "highest_data_scale_existing_natural_and_code_rows",
        "candidate_sites": ["equality/QK selector sites", "OV/value write", "later MLP products"],
        "evidence": [
            "circuits/campaign_2026_08_30/02_induction_copy.md",
            "terminal_copy_induction_v2_rows_receipt.json",
            "induction_equality_tensor_final_ood_v2_retry1_result.json",
        ],
        "counterfactual_families": [
            family(
                "two_valid_sources_selector_swap", "interchange", True, True,
                "construct two valid sources A-B and C-D, then change the query from A to C so the answer changes B to D",
                "which equality edge selects a source", ["both source pairs", "payload tokens", "length", "positions"],
                "D-minus-B logit movement and selector-path write",
                ["query an unrelated token", "distance-matched wrong source", "random subspace"],
            ),
            family(
                "payload_swap_match_preserved", "interchange", True, True,
                "keep both A occurrences fixed but replace the earlier follower B by B-prime",
                "payload carried by a fixed equality edge", ["source/query equality", "query token", "lag", "scaffold"],
                "B-prime-minus-B logit movement and delivered value write",
                ["change a non-follower", "shuffle B-prime within matched strata", "first mentions"],
            ),
            family(
                "natural_pair_interchange", "interchange", True, True,
                "pair natural copy-positive positions matched on query class, lag, frequency, and base loss",
                "naturally occurring selector/payload state", ["matching strata", "corpus role", "position band"],
                "donor-answer logit and loss movement",
                ["within-stratum wrong donor", "matched negative donor", "random subspace"],
            ),
            family(
                "match_break_payload_preserved", "necessity", True, False,
                "replace the earlier A by a matched decoy while leaving the current query and original answer B fixed",
                "availability of the equality edge", ["payload B", "current query A", "answer B", "length"],
                "loss of equality/copy evidence for B, not movement toward a new answer",
                ["same edit at irrelevant source", "offset plus/minus one or two", "token derangement"],
            ),
            family(
                "copy_relation_preserved_nuisance_change", "invariance", False, False,
                "change lag, filler, or source location while preserving the same A-to-B copy relation",
                "surface nuisance only", ["query A", "payload B", "copy relation"],
                "invariance of the recovered causal effect",
                ["held-out lag bands", "held-out corpus", "matched token frequencies"],
            ),
        ],
        "split_axes": ["document", "query token", "lag", "natural versus code"],
        "factorial_test": "selector x payload, including their interaction rather than head-by-head effects",
    },
    {
        "circuit_id": "pending_opener_state",
        "status": "strong_existing_DAS_instrument_needs_multi_family_transfer",
        "candidate_sites": ["layer-13 entry", "L13H8 input/output", "downstream MLP products"],
        "evidence": [
            "qk_mdl/algo_tasks/bracket/report.md",
            "qk_mdl/algo_tasks/bracket/das.json",
            "qk_mdl/algo_tasks/semantics_opener/report.md",
        ],
        "counterfactual_families": [
            family(
                "opener_presence_edit", "interchange", True, True,
                "insert or delete an opener in a position-matched prompt so the required closer changes",
                "pending-opener state", ["suffix template", "scoring position", "length after compensating edit"],
                "new-closer minus old-closer logit and held-out interchange recovery",
                ["edit non-opener punctuation", "position-matched filler edit", "random subspace"],
            ),
            family(
                "opener_type_substitution", "interchange", True, True,
                "substitute one opener type for another at equal length, for example parenthesis versus quote",
                "pending closer type", ["opener position", "length", "surrounding text"],
                "donor closer-type logit movement",
                ["non-opener punctuation substitution", "wrong closer", "random subspace"],
            ),
            family(
                "later_matching_closer_reset", "necessity", True, False,
                "insert a later matching closer while keeping the opener and original scoring suffix fixed",
                "whether an opener remains pending", ["opener identity", "suffix", "original answer candidates"],
                "reduction of pending-opener evidence",
                ["nonmatching closer", "punctuation insertion", "distance-matched filler"],
            ),
            family(
                "pending_state_preserved_distance_edit", "invariance", False, False,
                "change filler and opener distance without changing which opener remains pending",
                "distance and wording nuisance", ["pending type", "correct closer", "scoring position"],
                "invariance of closer effect",
                ["held-out distances", "held-out text templates", "brace negative family"],
            ),
        ],
        "split_axes": ["opener type", "template", "distance", "token identity"],
        "factorial_test": "opener type x distance/recency",
    },
    {
        "circuit_id": "successor_pointer_state",
        "status": "existing_multi_family_evidence_but_cross_family_transfer_failed",
        "candidate_sites": ["embedding/block-0 payload", "post-attention-8", "MLP8-11 products"],
        "evidence": [
            "qk_mdl/algo_tasks/successor/stimuli.json",
            "qk_mdl/algo_tasks/successor/report.md",
            "qk_mdl/algo_tasks/semantics_successor/report.md",
        ],
        "counterfactual_families": [
            family(
                "same_family_last_element_swap", "interchange", True, True,
                "replace the final sequence element by another element from the same family",
                "identity pointer supplied to successor lookup", ["prefix", "length", "positions", "family"],
                "donor-successor minus base-successor margin",
                ["position-only edit", "family-matched nonsuccessor", "random subspace"],
            ),
            family(
                "coherent_whole_sequence_shift", "interchange", True, True,
                "shift the whole sequence coherently so the last element and answer change without an intruder conflict",
                "successor pointer in a coherent context", ["format", "length", "family", "coherence"],
                "shifted-successor logit movement",
                ["prefix shift with final fixed", "incoherent sequence", "boundary cases"],
            ),
            family(
                "internal_pointer_imposition", "interchange", True, True,
                "keep the prompt fixed and impose another element's calibrated identity payload internally",
                "internal identity pointer", ["all prompt tokens", "position", "syntax", "family context"],
                "imposed-element successor and full-donor agreement",
                ["zero payload", "uncalibrated payload", "random subspace"],
            ),
            family(
                "prefix_change_final_pointer_preserved", "invariance", False, False,
                "alter the earlier sequence while keeping the final element and correct successor fixed",
                "prefix coherence nuisance", ["identity pointer", "correct successor", "final position"],
                "invariance of pointer effect",
                ["coherent prefix", "out-of-order prefix", "list-end boundary"],
            ),
        ],
        "split_axes": ["element", "template", "weekday/month/alphabet/digit", "natural text"],
        "factorial_test": "token/family identity x sequence coherence",
    },
    {
        "circuit_id": "increment_state",
        "status": "strong_existing_DAS_instrument_needs_operation_vs_digit_separation",
        "candidate_sites": ["post-attention-8", "L8H7/L8H3 paths", "MLP8-14 products"],
        "evidence": [
            "qk_mdl/algo_tasks/increment/report.md",
            "qk_mdl/algo_tasks/increment/s3_das.json",
            "qk_mdl/algo_tasks/increment/s3b_das_postattn.json",
        ],
        "counterfactual_families": [
            family(
                "coherent_constant_shift", "interchange", True, True,
                "shift all relevant list numbers by the same amount so the next number changes coherently",
                "numeric state consumed by increment", ["list structure", "step size", "template", "positions"],
                "shifted-next-number logit and interchange recovery",
                ["noun-only edit", "punctuation-only edit", "random subspace"],
            ),
            family(
                "cross_format_operation_swap", "interchange", True, True,
                "express the same increment relation with digits, number words, or a different list format",
                "format-independent increment state", ["increment operation", "semantic values", "answer relation"],
                "cross-format answer movement and leave-format-out transfer",
                ["nonincrement digit prediction", "format-matched constant sequence", "random subspace"],
            ),
            family(
                "incoherent_one_number_edit", "necessity", True, False,
                "edit one prior number without making a coherent sequence, leaving the original expected answer fixed",
                "support for the increment relation", ["final numeric state", "answer candidates", "template"],
                "loss of increment-specific evidence rather than attraction to a new answer",
                ["noun edit", "separator edit", "generic digit continuation"],
            ),
            family(
                "operation_preserved_surface_edit", "invariance", False, False,
                "change nouns, punctuation, and list wording while preserving numeric state and increment rule",
                "surface nuisance", ["numbers", "increment relation", "correct answer"],
                "invariance of increment effect",
                ["held-out nouns", "held-out punctuation", "different list lengths"],
            ),
        ],
        "split_axes": ["numeric identity", "digit versus word format", "template", "list length"],
        "factorial_test": "numeric payload x increment relation",
    },
]


def validate_pilot(pilot: dict) -> None:
    required = {
        "circuit_id", "status", "candidate_sites", "evidence", "counterfactual_families",
        "split_axes", "factorial_test",
    }
    assert set(pilot) == required
    assert pilot["candidate_sites"] and len(pilot["evidence"]) >= 2
    assert len(pilot["counterfactual_families"]) >= 3
    families = pilot["counterfactual_families"]
    assert sum(f["proposed_variable_changes"] for f in families) >= 2
    assert sum(f["answer_changes"] for f in families) >= 1
    assert any(f["intervention_role"] == "invariance" for f in families)
    ids = [f["family_id"] for f in families]
    assert len(ids) == len(set(ids))
    for item in families:
        assert set(item) == {
            "family_id", "intervention_role", "proposed_variable_changes", "answer_changes",
            "intervention", "changed_variable", "held_fixed", "endpoint", "controls",
        }
        assert item["intervention_role"] in {"interchange", "necessity", "invariance"}
        assert len(item["held_fixed"]) >= 2 and len(item["controls"]) >= 2
        if item["intervention_role"] == "interchange":
            assert item["proposed_variable_changes"] and item["answer_changes"]
        elif item["intervention_role"] == "necessity":
            assert item["proposed_variable_changes"] and not item["answer_changes"]
        else:
            assert not item["proposed_variable_changes"] and not item["answer_changes"]
    assert len(pilot["split_axes"]) >= 3


def main() -> None:
    for pilot in PILOTS:
        validate_pilot(pilot)
    families = [item for pilot in PILOTS for item in pilot["counterfactual_families"]]
    result = {
        "schema": "rung536_multi_counterfactual_contract_v2",
        "status": "contract_validated",
        "pilot_count": len(PILOTS),
        "total_counterfactual_families": len(families),
        "interchange_families": sum(f["intervention_role"] == "interchange" for f in families),
        "necessity_families": sum(f["intervention_role"] == "necessity" for f in families),
        "invariance_families": sum(f["intervention_role"] == "invariance" for f in families),
        "answer_changing_families": sum(f["answer_changes"] for f in families),
        "common_promotion": COMMON_PROMOTION,
        "pilots": PILOTS,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    OUT.write_text(payload)
    print(payload, end="")
    print("result_sha256", hashlib.sha256(payload.encode()).hexdigest())


if __name__ == "__main__":
    main()
