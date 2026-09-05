#!/usr/bin/env python3
"""Idempotently publish the omitted L13H8 semantic chain and FINAL_TEST null."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import circuit_registry_v2 as registry


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
TAG = "task.bracket.pending_opener"
BASE_SHA256 = "ad1ee205cd74b3f0a3430de045c2035c729c035e84e83f3b9ea75ae1ebaafebd"
INTERMEDIATE_V29_SHA256 = "374a8c3551befeb57e8d5ec2066e58505a5b82ffc3a00039c6b2346abb1a6643"
CHECKPOINT = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"

STEMS = {
    "source_region_payload": ("bracket_l13h8_source_region_payload_factorial_v1", "52ace3d1cb66118a2e09e12a7b802d636a4fbef9c32dd63741374826be51f2e0", "e5471038a18c2b7b285723e1bbe41d56c0236cab9d4e665edc2daa7595e6205d"),
    "source_region_select": ("bracket_l13h8_source_region_family_interaction_select_v1", "502329c0fa90a1f72569347f629158f5a93a93fd1e792444af4521eab4748b1d", "8148996681c64edaac3c25e9dda85d122f1a170b3fa57ea0bb0a42bc6c02625a"),
    "open_post_confirm": ("bracket_l13h8_open_post_family_confirm_v1", "a10b795d42ff4f66f052a5acc758323420172e4280ec998f517835c9874eb491", "39d75e204bdc428f9f2c574eaff22483ede1a95c976c78ab409d58f242d66124"),
    "open_zero_removal": ("bracket_l13h8_semantic_open_zero_removal_v1", "4a47c89f908925daa7574c665a7bde963130f2b24fb6a838d1e60fec8a9272b7", "dac4bdf0596884e84e7a282ff36f2027137fda69952d25fa27f198523e967aca"),
    "shared_contrast": ("bracket_l13h8_semantic_open_shared_contrast_v1", "283566ddade304cbde67fd3f88a8fe49d2d4067e73dac6529875f86fa88ab992", "37014b88e9341d63626d5152fca87e2df7de90c7dc76e2b309935bc330806c3a"),
    "shared_contrast_interaction": ("bracket_l13h8_shared_contrast_interaction_factorial_v1", "ada10afe3cd3ff617685cebf64218a23dff7b2553b92e9adea39afd7f6e834a4", "d686a2cb22e96049cb12ba1db5658dde9dac9240639bb5860edc1614faf594a7"),
    "mlp15_mediation": ("bracket_l13h8_mu_delta_mlp15_mediation_v1", "916c38eaf884be3e498b0a53e2e7c3d48fd249e02373c1f4e8d1970608bbd17f", "20543b23a1eea03db22722a17bf4ba5d623ba5809102edf1fcd2bca085524565"),
    "attention_mediation": ("bracket_l13h8_mu_delta_r549_attention_mediation_v1", "176815d395903cef47f13148d6a2d1fb7190d661e075c3851c263dff32466530", "7ae8f092a873b15d3f273196db380e46ef1e7f906b7d837395eb7b3dacf87917"),
    "attention_joint": ("bracket_l13h8_mu_delta_r549_attention_joint_mediation_v1", "c08ffec32d1aaba60cf081dc145c1645a187e5cf5f2a22e7ea2c6eee580759e1", "dfe042e91eed84b0493558916c02d87637c5ea84f02f037d81dfd66e448e59fb"),
    "downstream_module": ("bracket_l13h8_mu_delta_downstream_module_mediation_v1", "66fba6861aa6d5bd0048b7658fcf8440f9adde25f45e5e38d774fc83cfca55e4", "925683ae1e04c9aa3f27b51e22370589ef459aeeacc093e62d5afcb3391ce2b3"),
    "residual_write": ("bracket_l13h8_mu_delta_residual_write_bank_factorial_v1", "eab7b8ef3c54e7583204451cc3659a1c7470c60ac1e6502e508b7269510fae47", "bc38fe115de9f5dac166bbe7b4451592015df67c700e5b297524a1f66a76b17d"),
    "direct_fold_v1": ("bracket_l13h8_mu_delta_direct_readout_fold_v1", "194f9df04b0a264c6510ddae0763c589a827c95a9348591842cd1adfee2bb717", "05b13e9bc0de8c59122a05ae5a423951cae604629b9a56b0f791e9cf743f182d"),
    "direct_fold_v2": ("bracket_l13h8_mu_delta_direct_readout_fold_v2", "ca99d6974cbbe99e137ce8943fc9aa0090d3dab82e38f7ccfbf1f0c0b701b0fd", "31a40ed62409181e7977ca4182f34dbd5d7d9fa0d92c3270888bae53a94a21ae"),
    "pair_centered_final": ("bracket_l13h8_pair_centered_open_term_final_test_v1", "8910899470d14cd7190c290a506307a562017dde0dfc63105447d57ff8b85f63", "e64093354428d62eecd268360a79e8ef6549437babdaf9897d853134a44000f6"),
}


class PublicationError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts():
    out = {}
    for key, (stem, prereg_sha, result_sha) in STEMS.items():
        result_stem = ("bracket_l13h8_shared_contrast_interaction_select_v1"
                       if key == "shared_contrast_interaction" else stem)
        for suffix, folder, digest, kind in (
            ("prior_art", "circuits/prior_art", prereg_sha, "preregistration"),
            ("result", "circuits/fast_screens", result_sha, "screen_result"),
        ):
            file_stem = stem if suffix == "prior_art" else result_stem
            path = f"basis_aligned/bilinear_quotient/{folder}/{file_stem}{'.json' if suffix == 'prior_art' else '_result.json'}"
            if _sha(REPO / path) != digest:
                raise PublicationError(f"artifact changed: {path}")
            out[f"bracket_{key}_{suffix}"] = {"path": path, "sha256": digest,
                                                      "kind": kind, "status": "frozen"}
    return out


def _metric(name, estimate, bar):
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _event(key, verdict, test_type, metrics, notes, *, stage="complete", failure=None):
    return {
        "event_id": f"pending_opener.semantic_chain.{key}.v1",
        "claim_id": "pending_opener_state.v29", "test_type": test_type,
        "stage": stage, "verdict": verdict, "failure_kind": failure,
        "family_ids": ["direct_three_value_type_substitution", "completed_then_reopened_three_value_order",
                       "pending_type_preserved_surface_rewrite", "pending_type_preserved_distance_extension",
                       "pending_type_preserved_nonopener_punctuation"],
        "site_id": "attention13.head8.semantic_pending_opener_term.final_query",
        "split_plan_id": "pending_opener_three_value_fresh_split_r545_v1",
        "evaluation_role": ("held-out FINAL_TEST" if key == "pair_centered_final"
                            else "fresh semantic screen"),
        "metrics": metrics, "prereg_artifact_id": f"bracket_{key}_prior_art",
        "result_artifact_id": f"bracket_{key}_result",
        "input_artifact_ids": [f"bracket_{key}_prior_art"],
        "seed": None, "checkpoint_sha256": CHECKPOINT,
        "supersedes_event_id": None, "replicates_event_id": None,
        "sections": [], "notes": notes,
    }


def build_record():
    path = registry.circuit_path(TAG)
    current = json.loads(path.read_text())
    migrating_intermediate = False
    if current.get("claims", [{}])[-1].get("claim_id") == "pending_opener_state.v29":
        if current["claims"][-1]["causal_variable"]["id"] == \
                "pending_opener_exact_semantic_source_term":
            registry.validate_v2(current)
            return current
        if _sha(path) != INTERMEDIATE_V29_SHA256:
            raise PublicationError("unexpected intermediate v29 dossier")
        migrating_intermediate = True
        current["claims"].pop()
        current["evidence_events"] = current["evidence_events"][:-15]
        for artifact_id in _artifacts():
            current["artifacts"].pop(artifact_id)
    if not migrating_intermediate and _sha(path) != BASE_SHA256:
        raise PublicationError("canonical bracket dossier differs from the audited v28 base")
    record = copy.deepcopy(current)
    additions = _artifacts()
    if set(additions) & set(record["artifacts"]):
        raise PublicationError("new artifact ID collides")
    record["artifacts"].update(additions)
    previous = copy.deepcopy(record["claims"][-1])
    previous.update({
        "claim_id": "pending_opener_state.v29", "revision": 29,
        "status": "site_live", "supersedes": "pending_opener_state.v28",
        "next_missing": "selective necessity remains null; freeze an OOD-capable authority before opening OOD, and test a contrast definition that preserves distance-dependent CE without changing the held-out transfer claim",
    })
    previous["causal_variable"] = {
        "id": "pending_opener_exact_semantic_source_term",
        "domain": "parenthesis, square-bracket, or quote pending state on the frozen R545 distribution",
        "read": "the final causally relevant opener token at the exact L13H8 final-query source term",
        "operation": "write transferable closer-type evidence through the exact post-projection term p_k u_k; selective necessity of the pair-centered contrast is not established",
        "write": "signed evidence among the three matching closer tokens",
        "endpoint": "held-out donor-directed three-closer margin and full-vocabulary CE under exact term interchange",
    }
    site = {
        "site_id": "attention13.head8.semantic_pending_opener_term.final_query",
        "tensor_path": "exact p_k u_k post-output-projection contribution from the semantic pending-opener source to L13H8 at the final query",
        "shape": ["batch", 1152],
        "intervention": "exact donor-term interchange or pair-mean replacement; no fitted projector",
        "ceiling_event_ids": ["pending_opener_three_value_confirmation.r546.l13h8_site.held.v1"],
    }
    previous["candidate_sites"].append(site)
    events = [
        _event("source_region_payload", "null", "composition", [_metric("instrument", 1, "pass"), _metric("prespecified_region_localization", 0, "pass")], "Exhaustive PREFIX/OPEN/POST payload factorial was live but did not hold its original localization claim.", failure="scientific_null"),
        _event("source_region_select", "held", "composition", [_metric("OPEN_family_interaction", 1, "pass")], "Fresh SELECT isolated semantic OPEN as the family interaction."),
        _event("open_post_confirm", "held", "cross_family_transfer", [_metric("opener_payload_cells_passed", 4, "4/4")], "Fresh confirmation held opener payload across both target families and directions."),
        _event("open_zero_removal", "null", "removal", [_metric("target_cells_passed", 4, "4/4"), _metric("same_state_collateral", 0.9724998000919484, "<=0.25")], "Raw semantic-opener zero removal damaged controls almost as much as targets; it is not selectively necessary.", failure="scientific_null"),
        _event("shared_contrast", "null", "composition", [_metric("natural_swap_positive_fraction", 1.0, ">=0.75"), _metric("completed_order_type_common_ratio", 1.8635033226155877, ">=2")], "Exact triplet effect coding transferred type, but the stronger shared/common separation claim failed.", failure="scientific_null"),
        _event("shared_contrast_interaction", "held", "composition", [_metric("additive_oblique", 1, "pass")], "Shared and centered delimiter terms were approximately additive under the frozen interaction bar."),
        _event("mlp15_mediation", "null", "composition", [_metric("MLP15_rescue", 0, "pass required")], "Restoring MLP15 did not mediate either exact semantic factor.", failure="scientific_null"),
        _event("attention_mediation", "null", "composition", [_metric("individual_R549_head_rescue", 0, "pass required")], "No individual R549 attention handle mediated either factor.", failure="scientific_null"),
        _event("attention_joint", "null", "composition", [_metric("joint_R549_head_rescue", 0, "pass required")], "The grouped R549 attention handles also failed mediation.", failure="scientific_null"),
        _event("downstream_module", "null", "composition", [_metric("all_later_module_rescue", 0, "pass required")], "Freezing all later module writes did not preserve the exact factor's closer-axis effect.", failure="scientific_null"),
        _event("residual_write", "held", "compiled_equivalence", [_metric("residual_projection_range", [0.974, 1.224], ">=0.75"), _metric("later_write_projection_range", [-0.221, 0.037], "reported")], "Closer-axis signal is residual-route dominated; full-vocabulary CE shows later compensation."),
        _event("direct_fold_v1", "invalid", "compiled_equivalence", [_metric("instrument_live", 0, "must pass")], "First direct fold used an invalid normalization identity.", stage="invalid", failure="invalid_instrument"),
        _event("direct_fold_v2", "held", "compiled_equivalence", [_metric("direct_fold_projection_range", [0.9838, 1.1197], ">=0.75")], "Corrected exact RMS/unembedding/softcap fold accounts for the isolated residual closer-axis effect."),
        _event("pair_centered_final", "held", "cross_family_transfer", [_metric("open_swap_positive_cells", 24, "24/24"), _metric("open_swap_median_fraction_complete_range", [0.9799421606288387, 1.2119383614076955], ">=0.50")], "On unopened R545 FINAL_TEST, exact opener-term interchange transferred the closer in every ordered-pair cell. OOD remains unopened."),
        _event("pair_centered_final", "null", "removal", [_metric("direct_midpoint_ratio", [0.49089436629864325, 0.49456392450349285], ">=0.50"), _metric("distance_control_midpoint_CE_ratio_max", 3.56249944627227, "<=0.25")], "Pair-centered removal was directionally live but failed the registered direct-family ratio and control-CE selectivity bars; selective necessity is null.", failure="scientific_null"),
    ]
    # The transfer and removal conclusions share one immutable result but need unique IDs.
    events[-2]["event_id"] = "pending_opener.semantic_chain.pair_centered_final_test.transfer.held.v1"
    events[-1]["event_id"] = "pending_opener.semantic_chain.pair_centered_final_test.removal.null.v1"
    previous["evidence_event_ids"] = list(previous["evidence_event_ids"]) + [e["event_id"] for e in events]
    record["claims"].append(previous)
    for event in events:
        event["design_key"] = registry.design_key(record, event)
        event["execution_key"] = registry.execution_key(record, event)
        record["evidence_events"].append(event)
    registry.validate_v2(record)
    return record


def apply(*, regenerate=True):
    path = registry.circuit_path(TAG)
    desired = build_record()
    existing = json.loads(path.read_text())
    if existing != desired:
        with registry._lock("registry"):
            if json.loads(path.read_text()) != existing:
                raise PublicationError("canonical dossier moved during publication")
            registry._atomic_json(path, desired)
    if regenerate:
        registry.rebuild_registry_v2()
        for script in ("make_circuit_coverage.py", "make_circuit_experiment_index.py", "make_circuit_campaign_queue.py"):
            subprocess.run([sys.executable, str(BQ / script)], cwd=REPO, check=True)
    return path


if __name__ == "__main__":
    print(json.dumps({"written": str(apply().relative_to(REPO)), "gpu_used": False}, indent=2))
