#!/usr/bin/env python3
"""Numerically repaired Task14 L11H3 current/cache value factorial."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_current_branch_carries_task pred_c_cached_branch_carries_task pred_d_interaction_is_needed pred_e_lexical_leakage pred_f_number_specific

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_fresh_matched_natural_native_capability as capability
import run_task14_head11_3_fresh_matched_subject_current_cached_value_factorial as v1
import run_task14_head11_3_subject_attractor_score_payload_factorial as factors
import attention_source_factor_primitive as source_factor


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_fresh_matched_subject_current_cached_value_factorial_numerical_repair_v2.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v2_result.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_matched_subject_current_cached_value_factorial_v2_capability_license.json"
PRIOR_ART_SHA256 = "3516fd368b80676ee0554592f94c2dbacceaea0350e69e7be9a0bbdbb6c81c9c"
LICENSE_SHA256 = "b5ba447870147311b76211b9ff52c4e672f7a095c96488381d8147ef9b403ab4"
CANDIDATE_ID = "subject_verb.number_agreement.head11_3_fresh_matched_subject_current_cached_value_factorial_v2"
LAYER, HEAD, SELF_POSITION = v1.LAYER, v1.HEAD, v1.SELF_POSITION
CONDITIONS, BARS = v1.CONDITIONS, dict(v1.BARS)


class CurrentCachedFactorialV2Error(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    return v1.build_rows()


def validate_preflight():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise CurrentCachedFactorialV2Error("numerical-repair receipt changed")
    return licensing.validate_causal_preflight(
        capability.build_gate(), capability.CAPABILITY_RESULT, LICENSE,
        expected_license_sha256=LICENSE_SHA256, causal_candidate_id=CANDIDATE_ID)


def compile_plan():
    license_value = validate_preflight()
    return {
        "schema": "task14_head11_3_fresh_matched_subject_current_cached_value_factorial_plan_v2",
        "candidate_id": CANDIDATE_ID, "split": "LICENSED_HOLDOUT",
        "row_count": len(build_rows()), "conditions": list(CONDITIONS),
        "scientific_prior_art_sha256": v1.PRIOR_ART_SHA256,
        "numerical_repair_sha256": PRIOR_ART_SHA256,
        "license_sha256": LICENSE_SHA256,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "preflight_validated": True, "bars": dict(BARS),
        "fixed_context": "recipient p_8 and recipient sum over j != 8 of p_j*u_j",
        "value_partition": {
            "current_pre": "(1-lambda_11)*V_11*xhat_11,8 in head-value coordinates",
            "cached_pre": "lambda_11*V_0*xhat_0,8 in head-value coordinates",
            "projection_rule": "add chosen 128-d branches, then apply head-3 c_proj slice once",
        },
        "numerical_repair_only": True,
        "price": {"model_forwards": 4, "example_evaluations": 352,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["answer_directed_target_margin_improvement",
                     "target_full_vocab_CE_improvement", "fixed_are_minus_is_change"],
        "closed_claims": ["individual_q_or_k", "necessity", "syntax_generality",
                          "FIT", "downstream_reader_identity", "rank"],
    }


def _raw_value_branches(state, first_value, attention, torch, F):
    if first_value is None:
        raise CurrentCachedFactorialV2Error("L11 requires the block-0 cached-value bus")
    batch, length, width = state.shape
    heads, head_width = 9, width // 9
    raw = F.linear(state, attention.c_v.weight.to(state.dtype)).view(
        batch, length, heads, head_width)
    bus = first_value.view_as(raw)
    current_pre = (1 - attention.lamb) * raw[:, :, HEAD]
    cached_pre = attention.lamb * bus[:, :, HEAD]
    effective_pre = current_pre + cached_pre
    projection = attention.c_proj.weight[:, HEAD * head_width:(HEAD + 1) * head_width]
    return current_pre, cached_pre, effective_pre, projection


def _factor_forward(model, tokens, finals, torch, F, facade, *, replacement_heads=None,
                    native_reinstall_mask=None):
    captured = {}
    projection = None

    def attention(event):
        nonlocal projection
        if event.site != LAYER:
            return event.block.attn(event.state, event.first_value)
        write, base = source_factor.replay_attention_with_source_factors(
            event.state, event.first_value, event.block.attn, finals, HEAD, torch, F)
        current, cached, effective, projection = _raw_value_branches(
            event.state, event.first_value, event.block.attn, torch, F)
        captured.update({name: value.detach().clone() for name, value in base.items()})
        captured.update({"current_pre": current.detach().clone(),
                         "cached_pre": cached.detach().clone(),
                         "effective_pre": effective.detach().clone()})
        if replacement_heads is not None:
            rows = torch.arange(tokens.size(0), device=tokens.device)
            installed = factors._same_batch_native_heads(
                replacement_heads, base["head"], native_reinstall_mask, torch)
            write[rows, finals] += (installed - base["head"]).to(write.dtype)
        return write, event.first_value

    logits = facade.forward_with_dispatch(
        model, tokens, attention, lambda event: event.block.mlp(event.state),
        require_production=False).float()
    if set(captured) != {"p", "u", "head", "current_pre", "cached_pre", "effective_pre"} \
            or projection is None:
        raise CurrentCachedFactorialV2Error("failed to capture exact raw value branches")
    return logits, captured, projection.detach().clone()


def _project_once(current_pre, cached_pre, projection, F):
    return F.linear((current_pre + cached_pre).float(), projection.float())


def _compile(recipient_tokens, recipient, opposite, lexical, projection, rows, torch, F):
    recipient_terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    complement_mask = torch.arange(recipient_terms.shape[1], device=recipient_terms.device) != SELF_POSITION
    complement = recipient_terms[:, complement_mask].sum(1)
    native_p = recipient["p"][:, SELF_POSITION].unsqueeze(-1)

    def subject(current, cached):
        # Keep the native [batch, source, head-width] projection shape.  Only
        # select source 8 after the single c_proj-slice application.
        value = _project_once(current, cached, projection, F)[:, SELF_POSITION]
        return native_p * value

    heads = {
        "native_value": complement + subject(recipient["current_pre"], recipient["cached_pre"]),
        "opposite_current_only": complement + subject(opposite["current_pre"], recipient["cached_pre"]),
        "opposite_cached_only": complement + subject(recipient["current_pre"], opposite["cached_pre"]),
        "opposite_both": complement + subject(opposite["current_pre"], opposite["cached_pre"]),
        "lexical_current_only": complement + subject(lexical["current_pre"], recipient["cached_pre"]),
        "lexical_cached_only": complement + subject(recipient["current_pre"], lexical["cached_pre"]),
        "lexical_both": complement + subject(lexical["current_pre"], lexical["cached_pre"]),
        "complete_opposite_head": opposite["head"],
    }
    indices, replacements, specs, reinstall = [], [], [], []
    for row_index, row in enumerate(rows):
        for condition in CONDITIONS:
            indices.append(row_index)
            replacements.append(heads[condition][row_index])
            specs.append((row_index, condition, f"{row['direction_id']}__{row['template_id']}"))
            reinstall.append(condition == "native_value")
    index = torch.tensor(indices, dtype=torch.long, device=recipient_tokens.device)
    return {
        "tokens": recipient_tokens[index], "finals": torch.full_like(index, SELF_POSITION),
        "replacement_heads": torch.stack(replacements),
        "native_reinstall_mask": torch.tensor(reinstall, dtype=torch.bool,
                                               device=recipient_tokens.device),
        "specs": specs, "heads": heads,
    }


def evaluate(model, torch, F, facade):
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = v1._role_batch(rows, torch, device)
    native = factors._native_logits(model, tokens, torch, F)
    replay, captured, projection = _factor_forward(model, tokens, finals, torch, F, facade)
    recipient = {key: value[:n] for key, value in captured.items()}
    opposite = {key: value[n:2*n] for key, value in captured.items()}
    lexical = {key: value[2*n:] for key, value in captured.items()}
    patch = _compile(tokens[:n], recipient, opposite, lexical, projection, rows, torch, F)
    native_patch = factors._native_logits(model, patch["tokens"], torch, F)
    patched, _, _ = _factor_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])
    sides = (recipient, opposite, lexical)
    exactness = {
        "native_replay_max_absolute_logit_error": float((native - replay).abs().max()),
        "source_term_sum_max_absolute_error": max(float((
            torch.einsum("bk,bkd->bd", side["p"], side["u"]) - side["head"]
        ).abs().max()) for side in sides),
        "raw_effective_value_max_absolute_error": max(float((
            side["current_pre"] + side["cached_pre"] - side["effective_pre"]
        ).abs().max()) for side in sides),
        "projected_effective_value_max_absolute_error": max(float((
            F.linear(side["effective_pre"].float(), projection.float()) - side["u"]
        ).abs().max()) for side in sides),
        "same_batch_native_noop_endpoint_max_absolute_error": 0.0,
        "installed_head_max_absolute_error": 0.0,
        "complete_head_vector_max_absolute_error": float((
            patch["heads"]["complete_opposite_head"] - opposite["head"]
        ).abs().max()),
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
            "fixed_are_minus_is_change": value["are_minus_is"] - base["are_minus_is"],
        })
        exactness["installed_head_max_absolute_error"] = max(
            exactness["installed_head_max_absolute_error"], float((
                patch["replacement_heads"][out_index] - patch["heads"][condition][row_index]
            ).abs().max()))
        if condition == "native_value":
            exactness["same_batch_native_noop_endpoint_max_absolute_error"] = max(
                exactness["same_batch_native_noop_endpoint_max_absolute_error"],
                float((patched[out_index] - native_patch[out_index]).abs().max()))
    return evidence, exactness


def score(evidence, exactness, bars=BARS):
    required = {"raw_effective_value_max_absolute_error",
                "projected_effective_value_max_absolute_error"}
    if not required.issubset(exactness):
        raise CurrentCachedFactorialV2Error("v2 exactness evidence is incomplete")
    proxy = dict(exactness)
    proxy["value_branch_sum_max_absolute_error"] = max(
        exactness["raw_effective_value_max_absolute_error"],
        exactness["projected_effective_value_max_absolute_error"])
    scored = v1.score(evidence, proxy, bars)
    scored.pop("value_branch_sum_max_absolute_error", None)
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
        raise CurrentCachedFactorialV2Error(f"refusing to overwrite {OUT}")
    torch, F, facade = factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    terminal = "valid_causal_screen" if scored["predictions"]["pred_a_instrument_live"] else "invalid"
    result = {
        "schema": "task14_head11_3_fresh_matched_subject_current_cached_value_factorial_result_v2",
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
