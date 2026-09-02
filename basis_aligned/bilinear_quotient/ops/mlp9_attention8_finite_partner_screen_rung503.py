#!/usr/bin/env python3
"""RUNG503 -- finite raw-source partners of the attention8-driven MLP9 response."""

# BQGATE: EXPERIMENT
# pred_a finite BF16 source-removal instrument and calibrated parent response are valid
# pred_b a complete nonempty <=10-source partner set is selected without top-k
# pred_c the identical source set and its simultaneous finite response confirm
# pred_d the finite group has copy-selective downstream use against payload/position controls
# pred_e the group is only a candidate for a separately registered suffix intervention

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_score_directed_action_graph_rung501 as action_parent
import mlp0_branch_circuit_response_rung481 as circuit_parent
import mlp9_score_response_source_pair_atlas_rung502 as first


PREREG = POLY / "MLP9_ATTENTION8_FINITE_PARTNER_SCREEN_RUNG503_PREREGISTRATION.md"
R502B_SOURCE = ROOT / "ops/mlp9_score_response_source_pair_atlas_rung502b.py"
R502B_RESULT = ROOT / "mlp9_score_response_source_pair_atlas_rung502b_results.json"
R502B_BUNDLE = ROOT / "mlp9_score_response_source_pair_atlas_rung502b_bundle.pt"
PARENT_SOURCE = ROOT / "ops/equality_score_directed_action_graph_rung501.py"
PARENT_RESULT = ROOT / "equality_score_directed_action_graph_rung501_results.json"
CIRCUIT_SOURCE = ROOT / "ops/mlp0_branch_circuit_response_rung481.py"
OUT = ROOT / "mlp9_attention8_finite_partner_screen_rung503_results.json"
BUNDLE = ROOT / "mlp9_attention8_finite_partner_screen_rung503_bundle.pt"
HASHES = {
    PREREG: "e41e1f9eeba0acf7898f517d50f12fa293d10e5c59dfe72a1c47455c05906ac7",
    R502B_SOURCE: "b48cad198641b462d5861ed3bbf1b116d182ebe88e8904ad8f881f68a211b186",
    R502B_RESULT: "ab25bce7350482056e68d8d9d8c4c121dbad8a21256f75dca0ae0b54a2c8bf17",
    R502B_BUNDLE: "17d12ba36677c7c53655504b9bcf7fdfa66da87edac551440d62185b98a4f138",
    PARENT_SOURCE: "97f3946f558f3d61fc952a9b6ddc7c334b51ccc0ccfe5f02c6ecced417f1e077",
    PARENT_RESULT: "b17a9b274e4c61e0b4a3fc68d8ce84ec6f8e76f257c3d898e6a6990492301c4f",
    CIRCUIT_SOURCE: "ef08017a30ceb0c9e4481198fc1d58c5b0bf8cd37707d2223c42db9eb04f1f44",
}
PARTNERS = (
    "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A9",
    "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8",
)
PARTNER_SOURCE_INDICES = tuple(first.SOURCES.index(name) for name in PARTNERS)
BACKGROUNDS = first.BACKGROUNDS
STATES = first.STATES
DOC_QUARTERS = first.DOC_QUARTERS
POSITION_SHIFTS = first.POSITION_SHIFTS
KNOWN_PAIR = first.KNOWN_PAIR
BATCH = first.BATCH
D = first.D
TOKENS = first.TOKENS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(R502B_RESULT.read_text())
    if receipt.get("rung") != "502b" \
            or receipt.get("pred_a_repaired_exact_two_gauge_instrument") is not True \
            or receipt.get("pred_b_background_parent_response_reproduces") is not True \
            or receipt.get("pred_c_cross_gauge_compact_group_confirms") is not False \
            or receipt.get("pred_d_cross_gauge_downstream_use_confirms") is not False \
            or receipt.get("pred_e_candidate_for_finite_group_removal") is not False \
            or receipt.get("strong_null") is not True \
            or receipt.get("next_step") \
            != "source_pair_semantics_not_identified_use_finite_source_factorial_or_float32_control":
        raise RuntimeError("rung502b does not license the finite raw-source route")
    rows, parent_metadata = action_parent.validate_inputs()
    circuit_rows, circuit_masks, tags, validation_tags, _, circuit_metadata = \
        circuit_parent.validate_inputs()
    if not torch.equal(rows, circuit_rows) or len(tags) != 32 or len(validation_tags) != 30:
        raise RuntimeError("rung503 row or 32/30 circuit authority changed")
    if PARTNER_SOURCE_INDICES != (1, 2, 3, 4, 5, 6, 7, 8, 10,
                                  11, 12, 13, 14, 15, 16, 17, 18, 19):
        raise RuntimeError("finite partner vocabulary changed")
    return rows, circuit_masks, tags, {
        "parent": parent_metadata,
        "circuits": circuit_metadata,
        "rung502b_outcomes_loaded_for_selection": False,
        "validation_documents_or_tags_opened": False,
    }


def _raw_partner_sources(model, x0, attention_writes, prior_writes, raw_state):
    if len(attention_writes) != 10 or len(prior_writes) != 9:
        raise RuntimeError("MLP9 raw source count changed")
    embedding_coefficient, write_coefficients = first._source_coefficients(model)
    analytic = [embedding_coefficient * x0.float()]
    analytic.extend(write_coefficients[i] * attention_writes[i].float() for i in range(10))
    analytic.extend(write_coefficients[i] * prior_writes[i].float() for i in range(9))
    analytic = torch.stack(analytic, dim=2)
    raw = raw_state.float()
    raw_round = raw - analytic.sum(2)
    ratio = float(raw_round.double().square().mean().sqrt()
                  / raw.double().square().mean().sqrt().clamp_min(1e-30))
    partners = analytic[:, :, PARTNER_SOURCE_INDICES].detach()
    return raw_state.detach(), partners, ratio


def _forward(model, tokens, scales, *, direct=False, background="early_present",
             state="late_native", gradient_leaf=False, raw_sources=False):
    facade.validate_production_model(model)
    facade.validate_tokens(tokens, production_shape=True)
    if background not in BACKGROUNDS or state not in (*STATES, "late_native"):
        raise ValueError("unregistered action")
    cached = {}
    attention_writes = []
    prior_writes = []
    capture = {}
    diagnostics = {"factor_reconstruction_max": 0.0, "early_edit_rms": 0.0,
                   "late_edit_rms": 0.0, "raw_round_rms_over_raw": 0.0}
    audit = {"native_attention": 0, "replayed_attention": 0, "native_mlp": 0,
             "mlp9_leaves": 0}

    x = model.transformer.wte(tokens)
    x = F.rms_norm(x, (D,))
    x0 = x
    v1 = None
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_state = F.rms_norm(x, (D,))
        if direct or site not in action_parent.factor_parent.stage1.SITE_HEADS:
            write, v1 = block.attn(attention_state, v1)
            audit["native_attention"] += 1
        else:
            write, terms, support, error = action_parent.factor_parent._factor_site(
                attention_state, v1, block.attn, site, tokens)
            audit["replayed_attention"] += 1
            diagnostics["factor_reconstruction_max"] = max(
                diagnostics["factor_reconstruction_max"], error)
            donor, recipient = KNOWN_PAIR
            if site == action_parent.factor_parent.TERMS[donor][1]:
                cached.update(terms[donor])
                if background == "early_absent":
                    edit = terms[donor]["native_term"]
                    write = write - edit
                    diagnostics["early_edit_rms"] = float(edit.float().square().mean().sqrt())
            if site == action_parent.factor_parent.TERMS[recipient][1]:
                if not cached:
                    raise RuntimeError("known donor factors unavailable")
                target = terms[recipient]
                if state != "late_native":
                    replacement = torch.zeros_like(target["factor_term"])
                    if state == "score_donor":
                        replacement = torch.bmm(
                            cached["p"] * scales["score_ratio"] * support, target["u"])
                    elif state == "payload_donor":
                        replacement = torch.bmm(
                            target["p"] * support, cached["u"] * scales["payload_ratio"])
                    edit = replacement.to(write.dtype) - target["native_term"]
                    write = write + edit
                    diagnostics["late_edit_rms"] = float(edit.float().square().mean().sqrt())
        attention_writes.append(write.detach())
        x = x + write
        raw_mlp_state = x
        z = F.rms_norm(x, (D,))
        mlp_write = block.mlp(z)
        audit["native_mlp"] += 1
        if site == 9:
            capture["deployed_write"] = mlp_write.detach()
            if raw_sources:
                raw, partners, ratio = _raw_partner_sources(
                    model, x0.detach(), attention_writes, prior_writes,
                    raw_mlp_state.detach())
                capture["raw_state"] = raw
                capture["partner_sources"] = partners
                diagnostics["raw_round_rms_over_raw"] = ratio
            if gradient_leaf:
                mlp_write = mlp_write.detach().requires_grad_(True)
                capture["leaf"] = mlp_write
                audit["mlp9_leaves"] += 1
        prior_writes.append(mlp_write.detach())
        x = x + mlp_write
    logits = model.lm_head(F.rms_norm(x, (D,)))
    logits = (30.0 * torch.tanh(logits / 30.0)).float()
    if tuple(logits.shape) != (*tokens.shape, facade.LOGIT_VOCAB):
        raise RuntimeError("manual finite-source forward shape changed")
    expected = ({"native_attention": 18, "replayed_attention": 0, "native_mlp": 18,
                 "mlp9_leaves": int(gradient_leaf)} if direct else
                {"native_attention": 15, "replayed_attention": 3, "native_mlp": 18,
                 "mlp9_leaves": int(gradient_leaf)})
    expected_capture = {"deployed_write"}
    if raw_sources:
        expected_capture |= {"raw_state", "partner_sources"}
    if gradient_leaf:
        expected_capture.add("leaf")
    if audit != expected or set(capture) != expected_capture:
        raise RuntimeError(f"manual forward audit failed: {audit}, {set(capture)}")
    return logits, capture, diagnostics, audit


@torch.no_grad()
def _removed_writes(mlp, raw_state, partner_sources, source_indices=None):
    if source_indices is None:
        source_indices = tuple(range(len(PARTNERS)))
    outputs = []
    for index in source_indices:
        edited = (raw_state.float() - partner_sources[:, :, index]).to(raw_state.dtype)
        outputs.append(mlp(F.rms_norm(edited, (D,))).detach())
    if not outputs:
        return torch.empty((0, *raw_state.shape), dtype=raw_state.dtype,
                           device=raw_state.device)
    return torch.stack(outputs, dim=0)


@torch.no_grad()
def _group_removed_write(mlp, raw_state, partner_sources, selected):
    source_sum = partner_sources[:, :, selected].sum(2)
    edited = (raw_state.float() - source_sum).to(raw_state.dtype)
    return mlp(F.rms_norm(edited, (D,))).detach()


def _partner_contributions(absent, score, absent_removed, score_removed):
    delta = absent - score
    removed_delta = absent_removed - score_removed
    return delta.unsqueeze(0) - removed_delta


def _singleton_input_edit_rms(raw_state, partner_sources):
    values = []
    for index in range(len(PARTNERS)):
        edited = (raw_state.float() - partner_sources[:, :, index]).to(raw_state.dtype)
        values.append((edited.float() - raw_state.float()).double().square().mean().sqrt())
    return torch.stack(values)


def _group_input_edit_rms(raw_state, partner_sources, selected):
    edited = (raw_state.float() - partner_sources[:, :, selected].sum(2)).to(raw_state.dtype)
    return float((edited.float() - raw_state.float()).double().square().mean().sqrt())


def _empty_stats(tags):
    stats = first._empty_stats(len(tags))
    stats["pair_response_num"] = torch.zeros(2, 4, len(PARTNERS), dtype=torch.float64)
    stats["pair_payload_num"] = torch.zeros(2, 4, len(PARTNERS), dtype=torch.float64)
    stats["pair_gradient_num"] = torch.zeros(2, 4, len(PARTNERS), dtype=torch.float64)
    return stats


def _select_partners(stats, quarters):
    selected = []
    detail = {}
    for source_index, name in enumerate(PARTNERS):
        holds = True
        per_background = []
        for background in range(2):
            response2 = float(stats["denominators"][background, list(quarters), 1].sum())
            grad_h = float(stats["denominators"][background, list(quarters), 5].sum())
            response = float(stats["pair_response_num"][
                background, list(quarters), source_index].sum()) / max(response2, 1e-30)
            payload = float(stats["pair_payload_num"][
                background, list(quarters), source_index].sum()) / max(response2, 1e-30)
            gradient = float(stats["pair_gradient_num"][
                background, list(quarters), source_index].sum()) \
                / (grad_h if abs(grad_h) > 1e-30 else math.copysign(1e-30, grad_h or 1))
            quarter_signs = [
                bool(stats["pair_response_num"][background, quarter, source_index] > 0
                     and stats["pair_gradient_num"][background, quarter, source_index] > 0)
                for quarter in quarters]
            row_holds = bool(response >= .01 and gradient >= .01
                             and response >= 2 * abs(payload) and all(quarter_signs))
            holds = holds and row_holds
            per_background.append({
                "response_fraction": response,
                "payload_response_fraction": payload,
                "gradient_fraction": gradient,
                "quarter_signs_positive": quarter_signs,
                "holds": row_holds,
            })
        detail[name] = {"backgrounds": per_background, "selected": holds}
        if holds:
            selected.append(source_index)
    return selected, detail


def _confirmation_report(stats, selected):
    reports = []
    signs = []
    for background in range(2):
        background_rows = []
        background_signs = []
        for half in range(2):
            quarter = half + 2
            row = stats["confirmation_local"][background, half]
            shape = first._cosine_residual(float(row[0]), float(row[1]), float(row[2]))
            response_fraction = float(row[2] / max(float(row[1]), 1e-30))
            gradient_fraction = float(row[7]
                                      / (float(row[8]) if abs(float(row[8])) > 1e-30
                                         else math.copysign(1e-30, float(row[8]) or 1)))
            payload_ratio = abs(float(row[5])) / max(abs(float(row[2])), 1e-30)
            background_rows.append({
                **shape,
                "response_fraction_of_complete": response_fraction,
                "gradient_fraction_of_complete": gradient_fraction,
                "payload_to_score_response_fraction": payload_ratio,
                "tokens": int(row[11]),
            })
            background_signs.append({
                PARTNERS[index]: bool(
                    stats["pair_response_num"][background, quarter, index] > 0
                    and stats["pair_gradient_num"][background, quarter, index] > 0)
                for index in selected})
        reports.append(background_rows)
        signs.append(background_signs)
    holds = bool(selected and len(selected) <= 10 and all(
        row["cosine"] >= .75 and row["positive_scale_residual"] <= .70
        and .50 <= row["response_fraction_of_complete"] <= 1.50
        and .50 <= row["gradient_fraction_of_complete"] <= 1.50
        and row["payload_to_score_response_fraction"] <= .50
        for rows in reports for row in rows)
        and all(all(values.values()) for rows in signs for values in rows))
    return holds, {"backgrounds": reports, "selected_source_signs": signs}


def _parent_reports(stats, quarters):
    return [[first._complete_report(stats, background, quarter) for quarter in quarters]
            for background in range(2)]


def _pooled_complete(stats, background, quarters):
    row = stats["denominators"][background, list(quarters)].sum(0)
    return first._cosine_residual(float(row[1]), float(row[0]), float(row[2]))


def _update_audit(calls, audit):
    for key in ("native_attention", "replayed_attention", "native_mlp", "mlp9_leaves"):
        calls[key] += audit[key]


def collect(model, rows, circuit_masks, tags, scales):
    copy_mask = action_parent._task_masks(rows)["copy_positive"]
    stats = _empty_stats(tags)
    diagnostics = {
        "factor_reconstruction_max": 0.0,
        "raw_round_rms_over_raw_max": 0.0,
        "minimum_action_edit_rms": float("inf"),
        "minimum_singleton_input_removal_rms": float("inf"),
        "minimum_singleton_output_change_rms": float("inf"),
        "group_input_removal_rms_min": float("inf"),
        "group_output_change_rms_min": float("inf"),
        "background_native_early_present_max_abs": 0.0,
    }
    calls = {
        "early_present_native": 0, "early_absent_native": 0, "actions": 0,
        "copy_backwards": 0, "circuit_backwards": 0, "source_captures": 0,
        "local_mlp9_singleton_evaluations": 0, "local_mlp9_group_evaluations": 0,
        "native_attention": 0, "replayed_attention": 0, "native_mlp": 0,
        "mlp9_leaves": 0,
    }
    selected = None
    selection_detail = None
    confirmation_opened = False
    device = next(model.parameters()).device
    mlp9 = model.transformer.h[9].mlp

    for start in range(0, 500, BATCH):
        if start == 248:
            selected, selection_detail = _select_partners(stats, (0, 1))
            if not selected or len(selected) > 10:
                break
            confirmation_opened = True
        stop = min(start + BATCH, 500)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        targets = batch_rows[:, 1:].to(device)
        with torch.no_grad():
            _, native_present, native_diag, native_audit = _forward(
                model, tokens, scales, direct=True)
        calls["early_present_native"] += 1
        _update_audit(calls, native_audit)

        for background_index, background in enumerate(BACKGROUNDS):
            if background_index == 0:
                native = native_present
            else:
                with torch.no_grad():
                    _, native, native_diag, native_audit = _forward(
                        model, tokens, scales, background=background, state="late_native")
                calls["early_absent_native"] += 1
                _update_audit(calls, native_audit)
            logits, absent, absent_diag, absent_audit = _forward(
                model, tokens, scales, background=background, state="late_absent",
                gradient_leaf=True, raw_sources=True)
            with torch.no_grad():
                _, score, score_diag, score_audit = _forward(
                    model, tokens, scales, background=background, state="score_donor",
                    raw_sources=True)
                _, payload, payload_diag, payload_audit = _forward(
                    model, tokens, scales, background=background, state="payload_donor",
                    raw_sources=True)
            calls["actions"] += 3
            calls["source_captures"] += 3
            for audit in (absent_audit, score_audit, payload_audit):
                _update_audit(calls, audit)
            for diag in (absent_diag, score_diag, payload_diag):
                diagnostics["factor_reconstruction_max"] = max(
                    diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
                diagnostics["raw_round_rms_over_raw_max"] = max(
                    diagnostics["raw_round_rms_over_raw_max"], diag["raw_round_rms_over_raw"])
                for key in ("early_edit_rms", "late_edit_rms"):
                    if diag[key] > 0:
                        diagnostics["minimum_action_edit_rms"] = min(
                            diagnostics["minimum_action_edit_rms"], diag[key])
            if background_index == 0:
                diagnostics["background_native_early_present_max_abs"] = max(
                    diagnostics["background_native_early_present_max_abs"],
                    float((native["deployed_write"].float()
                           - native_present["deployed_write"].float()).abs().max()))

            removed = {}
            for name, capture in (("absent", absent), ("score", score), ("payload", payload)):
                removed[name] = _removed_writes(
                    mlp9, capture["raw_state"], capture["partner_sources"])
                calls["local_mlp9_singleton_evaluations"] += len(PARTNERS)
                input_rms = _singleton_input_edit_rms(
                    capture["raw_state"], capture["partner_sources"])
                output_rms = (removed[name].float()
                              - capture["deployed_write"].unsqueeze(0).float()) \
                    .double().square().mean((1, 2, 3)).sqrt()
                diagnostics["minimum_singleton_input_removal_rms"] = min(
                    diagnostics["minimum_singleton_input_removal_rms"], float(input_rms.min()))
                diagnostics["minimum_singleton_output_change_rms"] = min(
                    diagnostics["minimum_singleton_output_change_rms"], float(output_rms.min()))

            delta = absent["deployed_write"].float() - score["deployed_write"].float()
            payload_delta = absent["deployed_write"].float() - payload["deployed_write"].float()
            source_response = _partner_contributions(
                absent["deployed_write"].float(), score["deployed_write"].float(),
                removed["absent"].float(), removed["score"].float())
            source_payload = _partner_contributions(
                absent["deployed_write"].float(), payload["deployed_write"].float(),
                removed["absent"].float(), removed["payload"].float())

            group = payload_group = None
            if confirmation_opened:
                group_removed = {}
                for name, capture in (("absent", absent), ("score", score),
                                      ("payload", payload)):
                    group_removed[name] = _group_removed_write(
                        mlp9, capture["raw_state"], capture["partner_sources"], selected)
                    calls["local_mlp9_group_evaluations"] += 1
                    input_rms = _group_input_edit_rms(
                        capture["raw_state"], capture["partner_sources"], selected)
                    output_rms = float((group_removed[name].float()
                                        - capture["deployed_write"].float())
                                       .double().square().mean().sqrt())
                    diagnostics["group_input_removal_rms_min"] = min(
                        diagnostics["group_input_removal_rms_min"], input_rms)
                    diagnostics["group_output_change_rms_min"] = min(
                        diagnostics["group_output_change_rms_min"], output_rms)
                group = delta - (group_removed["absent"].float()
                                 - group_removed["score"].float())
                payload_group = payload_delta - (group_removed["absent"].float()
                                                 - group_removed["payload"].float())

            quarter_selections = first._quarter_selections(copy_mask, start, stop)
            for quarter, selected_cpu in quarter_selections:
                chosen = selected_cpu.to(device)
                first._accumulate_complete(
                    stats, background_index, quarter, chosen,
                    native["deployed_write"], absent["deployed_write"],
                    score["deployed_write"], payload["deployed_write"])
                stats["pair_response_num"][background_index, quarter] += (
                    source_response[:, chosen].double()
                    * delta[chosen].double().unsqueeze(0)).sum((1, 2)).cpu()
                stats["pair_payload_num"][background_index, quarter] += (
                    source_payload[:, chosen].double()
                    * delta[chosen].double().unsqueeze(0)).sum((1, 2)).cpu()
                if confirmation_opened:
                    g, f, p = (group[chosen].double(), delta[chosen].double(),
                               payload_group[chosen].double())
                    row = stats["confirmation_local"][background_index, quarter - 2]
                    row[0] += float(g.square().sum())
                    row[1] += float(f.square().sum())
                    row[2] += float((g * f).sum())
                    row[3] += float((g * f).sum())
                    row[4] += float(f.square().sum())
                    row[5] += float((p * f).sum())
                    row[6] += float(f.square().sum())
                    row[9] += float(p.square().sum())
                    row[10] += float(g.square().sum())
                    row[11] += int(chosen.sum())

            circuit_selections = [] if not confirmation_opened else \
                circuit_parent._batch_selections(circuit_masks, tags, start, stop, 374)
            gradient_jobs = [("copy", job) for job in quarter_selections]
            gradient_jobs += [("circuit", job) for job in circuit_selections]
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none") \
                .view(len(batch_rows), TOKENS)
            for job_index, (kind, job) in enumerate(gradient_jobs):
                if kind == "copy":
                    quarter, selected_cpu = job
                    chosen = selected_cpu.to(device)
                else:
                    half, mask_index, tag_index, selected_cpu = job
                    chosen = selected_cpu.to(device)
                gradient = torch.autograd.grad(
                    nll[chosen].sum(), absent["leaf"],
                    retain_graph=job_index + 1 < len(gradient_jobs), allow_unused=False)[0]
                if kind == "copy":
                    stats["pair_gradient_num"][background_index, quarter] += (
                        source_response.double()
                        * gradient.double().unsqueeze(0)).sum((1, 2, 3)).cpu()
                    complete = float((gradient.float() * delta).sum())
                    stats["denominators"][background_index, quarter, 5] += complete
                    stats["denominators"][background_index, quarter, 7] += 1
                    if confirmation_opened:
                        row = stats["confirmation_local"][background_index, quarter - 2]
                        row[7] += float((gradient.float() * group).sum())
                        row[8] += complete
                    calls["copy_backwards"] += 1
                else:
                    full_values = [
                        float((gradient.float() * delta).sum()),
                        float((gradient.float() * group).sum()),
                        float((gradient.float() * payload_group).sum()),
                    ]
                    full_values.extend(float((gradient.float() * torch.roll(
                        group, shift, dims=1)).sum()) for shift in POSITION_SHIFTS)
                    stats["circuit_sums"][half, background_index, mask_index, tag_index] \
                        += torch.tensor(full_values, dtype=torch.float64)
                    if background_index == 0:
                        stats["circuit_counts"][half, mask_index, tag_index] += int(chosen.sum())
                    calls["circuit_backwards"] += 1
            del logits, nll, absent, score, payload, removed, source_response, source_payload
            if confirmation_opened:
                del group, payload_group, group_removed

    if selected is None:
        selected, selection_detail = _select_partners(stats, (0, 1))
    confirmation_selected = None
    confirmation_detail = None
    if confirmation_opened:
        confirmation_selected, confirmation_detail = _select_partners(stats, (2, 3))
    return (stats, diagnostics, calls, selected, selection_detail,
            confirmation_opened, confirmation_selected, confirmation_detail)


def _dry_run():
    assert len(PARTNERS) == 18
    assert "E" not in PARTNERS and "A8" not in PARTNERS
    assert DOC_QUARTERS == ((0, 124), (124, 248), (248, 374), (374, 500))
    torch.manual_seed(503)
    absent = torch.randn(2, 3, 7)
    score = torch.randn(2, 3, 7)
    absent_removed = torch.randn(18, 2, 3, 7)
    score_removed = torch.randn(18, 2, 3, 7)
    observed = _partner_contributions(absent, score, absent_removed, score_removed)
    expected = (absent - score).unsqueeze(0) - (absent_removed - score_removed)
    torch.testing.assert_close(observed, expected)
    print(json.dumps({
        "status": "dry_run_passed", "rung": 503, "model_loaded": False,
        "source_outcomes_opened": False, "partners": list(PARTNERS),
        "selection_full_model_forwards": 496,
        "conditional_total_full_model_forwards": 1000,
        "selection_local_mlp9_evaluations": 6696,
        "conditional_total_local_mlp9_evaluations": 13878,
        "validation_documents_or_tags_opened": False,
    }, indent=2))


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv[1:]:
        _dry_run()
        return
    if len(sys.argv) != 1:
        raise SystemExit("only --dry-run is supported")
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung503 namespace already exists")
    rows, circuit_masks, tags, metadata = validate_inputs()
    parent_receipt = json.loads(PARENT_RESULT.read_text())
    scales = parent_receipt["frozen_scales"][action_parent.KNOWN_POSITIVE]
    copy_mask = action_parent._task_masks(rows)["copy_positive"]
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    (stats, diagnostics, calls, selected, selection_detail, confirmation_opened,
     confirmation_selected, confirmation_detail) = collect(
         model, rows, circuit_masks, tags, scales)
    if not confirmation_opened:
        diagnostics["group_input_removal_rms_min"] = None
        diagnostics["group_output_change_rms_min"] = None

    stop = 500 if confirmation_opened else 248
    batch_count = stop // BATCH
    expected_copy = sum(len(first._quarter_selections(
        copy_mask, start, min(start + BATCH, stop))) for start in range(0, stop, BATCH)) * 2
    expected_circuit = 0 if not confirmation_opened else sum(
        len(circuit_parent._batch_selections(
            circuit_masks, tags, start, min(start + BATCH, 500), 374))
        for start in range(248, 500, BATCH)) * 2
    expected_calls = {
        "early_present_native": batch_count,
        "early_absent_native": batch_count,
        "actions": batch_count * 6,
        "copy_backwards": expected_copy,
        "circuit_backwards": expected_circuit,
        "source_captures": batch_count * 6,
        "local_mlp9_singleton_evaluations": batch_count * 6 * len(PARTNERS),
        "local_mlp9_group_evaluations": (63 * 6 if confirmation_opened else 0),
        "native_attention": batch_count * 18 + batch_count * 7 * 15,
        "replayed_attention": batch_count * 7 * 3,
        "native_mlp": batch_count * 8 * 18,
        "mlp9_leaves": batch_count * 2,
    }
    calls_exact = calls == expected_calls
    opened_quarters = range(4) if confirmation_opened else range(2)
    parent_reports = _parent_reports(stats, opened_quarters)
    parent_bars = all(
        row["score"]["cosine"] >= .75
        and row["score"]["positive_scale_residual"] <= .70
        and (row["score"]["cosine"] >= row["payload"]["cosine"] + .30
             or row["score"]["positive_scale_residual"]
             <= row["payload"]["positive_scale_residual"] - .30)
        for rows_ in parent_reports for row in rows_)
    parent_analysis = parent_receipt["discovery"]["analysis"][action_parent.KNOWN_POSITIVE]
    parent_differences = []
    phase_quarters = [((0, 1), 0)]
    if confirmation_opened:
        phase_quarters.append(((2, 3), 1))
    for background_index, background in enumerate(BACKGROUNDS):
        for quarters, parent_half in phase_quarters:
            observed = _pooled_complete(stats, background_index, quarters)["cosine"]
            expected = parent_analysis[background]["score_donor"][parent_half]["reader"][
                "copy_positive"]["cosine"]
            parent_differences.append({
                "background": background, "phase": parent_half,
                "observed": observed, "rung501": expected,
                "absolute_difference": abs(observed - expected),
            })
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and calls_exact and parent_bars
        and max(row["absolute_difference"] for row in parent_differences) <= .03
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["raw_round_rms_over_raw_max"] <= .03125
        and diagnostics["minimum_action_edit_rms"] > 0
        and diagnostics["minimum_singleton_input_removal_rms"] > 0
        and diagnostics["minimum_singleton_output_change_rms"] > 0
        and diagnostics["background_native_early_present_max_abs"] == 0.0
        and (not confirmation_opened
             or (diagnostics["group_input_removal_rms_min"] > 0
                 and diagnostics["group_output_change_rms_min"] > 0)))
    pred_b = bool(selected and len(selected) <= 10)
    same_selection = bool(confirmation_opened and confirmation_selected == selected)
    if confirmation_opened:
        group_holds, confirmation = _confirmation_report(stats, selected)
        circuit_holds, circuits = first._circuit_report(stats, tags)
    else:
        group_holds, confirmation = False, None
        circuit_holds, circuits = False, None
    pred_c = bool(same_selection and group_holds)
    pred_d = bool(pred_c and circuit_holds)
    pred_e = bool(pred_a and pred_b and pred_c and pred_d)
    strong_null = bool(not pred_a or not pred_b or not pred_c or not pred_d)

    bundle = {
        "schema": "mlp9_attention8_finite_partner_screen_rung503_stats_v1",
        "stats": stats,
        "raw_tokens_logits_gradients_or_per_token_vectors_included": False,
    }
    torch.save(bundle, BUNDLE)
    selected_names = [PARTNERS[index] for index in selected]
    confirmation_names = None if confirmation_selected is None else [
        PARTNERS[index] for index in confirmation_selected]
    result = {
        "status": "complete", "rung": 503,
        "claim_level": "finite_local_raw_source_partner_screen_not_circuit",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "partner_vocabulary": list(PARTNERS),
        "excluded_from_partner_semantics": ["E", "A8"],
        "document_quarters": [list(value) for value in DOC_QUARTERS],
        "confirmation_opened": confirmation_opened,
        "selection": {"sources": selected_names, "details": selection_detail},
        "confirmation_reselection": {"sources": confirmation_names,
                                     "same_complete_set": same_selection,
                                     "details": confirmation_detail},
        "finite_group_confirmation": confirmation,
        "circuit_fingerprints": circuits,
        "parent_response": parent_reports,
        "rung501_parent_differences": parent_differences,
        "instrument": {**diagnostics, "calls": calls, "expected_calls": expected_calls,
                       "calls_exact": calls_exact},
        'pred_a_finite_source_instrument_and_parent_valid': pred_a,
        'pred_b_compact_partner_set_selected': pred_b,
        'pred_c_partner_identity_and_group_confirm': pred_c,
        'pred_d_selective_downstream_use_confirms': pred_d,
        'pred_e_candidate_for_suffix_intervention': pred_e,
        "strong_null": strong_null,
        "validation_documents_or_tags_opened": False,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "full_model_forwards": calls["early_present_native"]
            + calls["early_absent_native"] + calls["actions"],
            "local_mlp9_evaluations": calls["local_mlp9_singleton_evaluations"]
            + calls["local_mlp9_group_evaluations"],
            "backwards": calls["copy_backwards"] + calls["circuit_backwards"],
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started,
        "next_step": (
            "repair_finite_source_instrument_only" if not pred_a else
            "finite_pair_removal_screen_or_float32_control" if not pred_b else
            "partner_identity_unstable_use_float32_control_or_change_observation"
            if not pred_c else
            "change_downstream_observation_before_circuit_claim" if not pred_d else
            "preregister_heldout_finite_mlp9_partner_suffix_intervention"),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 503,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "selection": selected_names,
        "confirmation_reselection": confirmation_names,
        "execution_price": result["execution_price"],
        "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
