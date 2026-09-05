#!/usr/bin/env python3
"""Finite-precision remainder repair for the Task14 E/A/M factorial."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_embedding_carries_task pred_c_attention_carries_task pred_d_MLP_carries_task pred_e_distributed_across_writer_families pred_f_interaction_is_needed pred_g_number_specific pred_h_lexical_collateral

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_fresh_matched_natural_native_capability as capability
import run_task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial as v1
import run_task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v2 as value_v2
import run_task14_head11_3_subject_attractor_score_payload_factorial as factors


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_numerical_repair_v2.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_v2_result.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_matched_subject_current_upstream_writer_factorial_v2_capability_license.json"
PRIOR_ART_SHA256 = "99befade2c755168e6eff45f2c3b58f1df556b709925eeff772bcf1d355606f4"
LICENSE_SHA256 = "bf8f9a3b71846a06ebaf577421547eb19149345d9011e3bf5417bcbd7338abc6"
CANDIDATE_ID = "subject_verb.number_agreement.head11_3_fresh_matched_subject_current_upstream_writer_factorial_v2"
LAYER, HEAD, SELF_POSITION = v1.LAYER, v1.HEAD, v1.SELF_POSITION
CONDITIONS, BARS = v1.CONDITIONS, dict(v1.BARS)


class UpstreamWriterFactorialV2Error(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    return v1.build_rows()


def validate_preflight():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise UpstreamWriterFactorialV2Error("numerical-repair receipt changed")
    return licensing.validate_causal_preflight(
        capability.build_gate(), capability.CAPABILITY_RESULT, LICENSE,
        expected_license_sha256=LICENSE_SHA256, causal_candidate_id=CANDIDATE_ID)


def compile_plan():
    license_value = validate_preflight()
    return {
        "schema": "task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_plan_v2",
        "candidate_id": CANDIDATE_ID, "split": "LICENSED_HOLDOUT",
        "row_count": len(build_rows()), "conditions": list(CONDITIONS),
        "scientific_prior_art_sha256": v1.PRIOR_ART_SHA256,
        "numerical_repair_sha256": PRIOR_ART_SHA256,
        "license_sha256": LICENSE_SHA256,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "preflight_validated": True, "bars": dict(BARS),
        "partition": "exact pre-attention-11 subject state (E+R)+A+M",
        "remainder_rule": "R=reference-(E+A+M); R follows the E donor and is added last",
        "fixed_context": "recipient p_8, cached value branch, and non-subject head complement",
        "numerical_repair_only": True,
        "price": {"model_forwards": 4, "example_evaluations": 480,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["answer_directed_target_margin_improvement",
                     "target_full_vocab_CE_improvement"],
        "closed_claims": ["individual_attention_or_MLP_block", "downstream_reader",
                          "necessity", "syntax_generality", "FIT", "rank", "reconstruction"],
    }


def _decomposed_factor_forward(model, tokens, finals, torch, F, facade, **kwargs):
    logits, captured, projection, old_closure = v1._decomposed_factor_forward(
        model, tokens, finals, torch, F, facade, **kwargs)
    uncorrected = captured["E"] + captured["A"] + captured["M"]
    remainder = captured["raw_state"] - uncorrected
    corrected = uncorrected + remainder
    captured["R"] = remainder.detach().clone()
    corrected_normalized = F.rms_norm(corrected, (corrected.size(-1),))
    closure = {
        "uncorrected_state_max_absolute_error": old_closure["state_sum_max_absolute_error"],
        "state_sum_max_absolute_error": float((corrected - captured["raw_state"]).abs().max()),
        "normalized_state_max_absolute_error": float((
            corrected_normalized - captured["normalized_state"]).abs().max()),
    }
    return logits, captured, projection, closure


def _current_from_state(embedding, attention_sum, mlp_sum, remainder, attention, torch, F):
    base = embedding + attention_sum + mlp_sum
    corrected = base + remainder
    normalized = F.rms_norm(corrected, (corrected.size(-1),))
    batch, length, width = normalized.shape
    head_width = width // 9
    raw = F.linear(normalized, attention.c_v.weight.to(normalized.dtype)).view(
        batch, length, 9, head_width)
    return (1 - attention.lamb) * raw[:, :, HEAD]


def _compile(recipient_tokens, recipient, opposite, lexical, attention, projection,
             rows, torch, F):
    recipient_terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    complement_mask = torch.arange(recipient_terms.shape[1], device=recipient_terms.device) != SELF_POSITION
    complement = recipient_terms[:, complement_mask].sum(1)
    native_p = recipient["p"][:, SELF_POSITION].unsqueeze(-1)
    components = {"r": recipient, "o": opposite, "l": lexical}
    choices = {
        "recipient_EAM": "rrr",
        "opposite_E": "orr", "opposite_A": "ror", "opposite_M": "rro",
        "opposite_EA": "oor", "opposite_EM": "oro", "opposite_AM": "roo",
        "opposite_EAM": "ooo",
        "lexical_E": "lrr", "lexical_A": "rlr", "lexical_M": "rrl",
        "lexical_EAM": "lll",
    }
    heads, current_by_condition = {}, {}
    for condition, choice in choices.items():
        # The fixed convention makes R travel with E, hence choice[0].
        current = _current_from_state(
            components[choice[0]]["E"], components[choice[1]]["A"],
            components[choice[2]]["M"], components[choice[0]]["R"],
            attention, torch, F)
        value = value_v2._project_once(current, recipient["cached_pre"], projection, F)
        heads[condition] = complement + native_p * value[:, SELF_POSITION]
        current_by_condition[condition] = current
    indices, replacements, specs, reinstall = [], [], [], []
    for row_index, row in enumerate(rows):
        for condition in CONDITIONS:
            indices.append(row_index)
            replacements.append(heads[condition][row_index])
            specs.append((row_index, condition, f"{row['direction_id']}__{row['template_id']}"))
            reinstall.append(condition == "recipient_EAM")
    index = torch.tensor(indices, dtype=torch.long, device=recipient_tokens.device)
    return {
        "tokens": recipient_tokens[index], "finals": torch.full_like(index, SELF_POSITION),
        "replacement_heads": torch.stack(replacements),
        "native_reinstall_mask": torch.tensor(reinstall, dtype=torch.bool,
                                               device=recipient_tokens.device),
        "specs": specs, "heads": heads, "current": current_by_condition,
    }


def evaluate(model, torch, F, facade):
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = v1._role_batch(rows, torch, device)
    native = factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, closure = _decomposed_factor_forward(
        model, tokens, finals, torch, F, facade)
    recipient = {key: value[:n] for key, value in captured.items()}
    opposite = {key: value[n:2*n] for key, value in captured.items()}
    lexical = {key: value[2*n:] for key, value in captured.items()}
    attention = model.transformer.h[LAYER].attn
    patch = _compile(tokens[:n], recipient, opposite, lexical, attention, projection,
                     rows, torch, F)
    native_patch = factors._native_logits(model, patch["tokens"], torch, F)
    patched, _, _, patch_closure = _decomposed_factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])
    sides = (recipient, opposite, lexical)
    direct_opposite_value = value_v2._project_once(
        opposite["current_pre"], recipient["cached_pre"], projection, F)
    recipient_terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    complement_mask = torch.arange(recipient_terms.shape[1], device=device) != SELF_POSITION
    direct_opposite_head = recipient_terms[:, complement_mask].sum(1) + \
        recipient["p"][:, SELF_POSITION].unsqueeze(-1) * direct_opposite_value[:, SELF_POSITION]
    exactness = {
        "native_replay_max_absolute_logit_error": float((native - replay).abs().max()),
        "uncorrected_state_max_absolute_error": max(
            closure["uncorrected_state_max_absolute_error"],
            patch_closure["uncorrected_state_max_absolute_error"]),
        "state_sum_max_absolute_error": max(
            closure["state_sum_max_absolute_error"], patch_closure["state_sum_max_absolute_error"]),
        "normalized_state_max_absolute_error": max(
            closure["normalized_state_max_absolute_error"],
            patch_closure["normalized_state_max_absolute_error"]),
        "source_term_sum_max_absolute_error": max(float((
            torch.einsum("bk,bkd->bd", side["p"], side["u"]) - side["head"]
        ).abs().max()) for side in sides),
        "all_donor_current_head_max_absolute_error": float((
            patch["heads"]["opposite_EAM"] - direct_opposite_head).abs().max()),
        "same_batch_native_noop_endpoint_max_absolute_error": 0.0,
        "installed_head_max_absolute_error": 0.0,
    }
    evidence = []
    for out_index, (row_index, condition, cell_id) in enumerate(patch["specs"]):
        base = v1._metrics(native_patch[out_index, SELF_POSITION], rows[row_index], condition, torch)
        value = v1._metrics(patched[out_index, SELF_POSITION], rows[row_index], condition, torch)
        evidence.append({
            "row_id": rows[row_index]["row_id"], "cell_id": cell_id,
            "condition": condition,
            "target_margin_improvement": value["target_margin"] - base["target_margin"],
            "target_CE_improvement": base["target_CE"] - value["target_CE"],
        })
        exactness["installed_head_max_absolute_error"] = max(
            exactness["installed_head_max_absolute_error"], float((
                patch["replacement_heads"][out_index] - patch["heads"][condition][row_index]
            ).abs().max()))
        if condition == "recipient_EAM":
            exactness["same_batch_native_noop_endpoint_max_absolute_error"] = max(
                exactness["same_batch_native_noop_endpoint_max_absolute_error"],
                float((patched[out_index] - native_patch[out_index]).abs().max()))
    return evidence, exactness


def score(evidence, exactness, bars=BARS):
    if "uncorrected_state_max_absolute_error" not in exactness:
        raise UpstreamWriterFactorialV2Error("v2 remainder diagnostic is missing")
    scored = v1.score(evidence, exactness, bars)
    inherited = scored["predictions"]
    # The scientific scorer and thresholds remain frozen in v1. Spell out its
    # registered keys here so the static experiment gate can audit this wrapper.
    scored["predictions"] = {
        "pred_a_instrument_live": bool(inherited["pred_a_instrument_live"]),
        "pred_b_embedding_carries_task": bool(
            inherited["pred_b_embedding_carries_task"]),
        "pred_c_attention_carries_task": bool(
            inherited["pred_c_attention_carries_task"]),
        "pred_d_MLP_carries_task": bool(inherited["pred_d_MLP_carries_task"]),
        "pred_e_distributed_across_writer_families": bool(
            inherited["pred_e_distributed_across_writer_families"]),
        "pred_f_interaction_is_needed": bool(
            inherited["pred_f_interaction_is_needed"]),
        "pred_g_number_specific": bool(inherited["pred_g_number_specific"]),
        "pred_h_lexical_collateral": bool(
            inherited["pred_h_lexical_collateral"]),
    }
    return scored


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise UpstreamWriterFactorialV2Error(f"refusing to overwrite {OUT}")
    torch, F, facade = factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    terminal = "valid_causal_screen" if scored["predictions"]["pred_a_instrument_live"] else "invalid"
    result = {
        "schema": "task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_result_v2",
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256, "score": scored,
        "evidence": evidence, "evaluated_splits": ["LICENSED_HOLDOUT"],
        "forbidden_splits_opened": [], "model_forwards": 4,
        "causal_interventions": len(evidence),
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
