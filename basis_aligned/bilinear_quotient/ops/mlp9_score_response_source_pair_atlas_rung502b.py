#!/usr/bin/env python3
"""RUNG502B -- background-correct, two-gauge MLP9 source-pair atlas repair."""

# BQGATE: EXPERIMENT
# pred_a repaired exact deployed-residual and two-gauge instrument
# pred_b background-specific native references restore the calibrated parent response
# pred_c the compact exact pair group agrees and confirms in both allocation gauges
# pred_d downstream circuit use and pair signs agree in both allocation gauges
# pred_e a cross-gauge survivor is only a candidate for finite causal removal

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


PREREG = POLY / "MLP9_SCORE_RESPONSE_SOURCE_PAIR_ATLAS_RUNG502_PREREGISTRATION.md"
FIRST_SOURCE = ROOT / "ops/mlp9_score_response_source_pair_atlas_rung502.py"
FIRST_RESULT = ROOT / "mlp9_score_response_source_pair_atlas_rung502_results.json"
FIRST_BUNDLE = ROOT / "mlp9_score_response_source_pair_atlas_rung502_bundle.pt"
PARENT_SOURCE = ROOT / "ops/equality_score_directed_action_graph_rung501.py"
PARENT_RESULT = ROOT / "equality_score_directed_action_graph_rung501_results.json"
CIRCUIT_SOURCE = ROOT / "ops/mlp0_branch_circuit_response_rung481.py"
OUT = ROOT / "mlp9_score_response_source_pair_atlas_rung502b_results.json"
BUNDLE = ROOT / "mlp9_score_response_source_pair_atlas_rung502b_bundle.pt"
HASHES = {
    PREREG: "ac32f5d857544596c0d80544218536a064bc32a6fae9244053b36d306c021d67",
    FIRST_SOURCE: "8467077102879bd028360a3626776b2de853342095c86ef186db8785c24ce3a5",
    FIRST_RESULT: "77984dd9d68da79640d72a8c273718b32199d9eb67fea0b7c4038770141099c0",
    FIRST_BUNDLE: "c2d3a35565951218dd7f335bed6adb6322172db9b8fe3f12cf5ae1d4cad2604e",
    PARENT_SOURCE: "97f3946f558f3d61fc952a9b6ddc7c334b51ccc0ccfe5f02c6ecced417f1e077",
    PARENT_RESULT: "b17a9b274e4c61e0b4a3fc68d8ce84ec6f8e76f257c3d898e6a6990492301c4f",
    CIRCUIT_SOURCE: "ef08017a30ceb0c9e4481198fc1d58c5b0bf8cd37707d2223c42db9eb04f1f44",
}
GAUGES = ("E_ABSORBS", "PROPORTIONAL")
SOURCES = first.SOURCES
SOURCE_PAIRS = first.SOURCE_PAIRS
PAIR_NAMES = first.PAIR_NAMES
BACKGROUNDS = first.BACKGROUNDS
STATES = first.STATES
DOC_QUARTERS = first.DOC_QUARTERS
POSITION_SHIFTS = first.POSITION_SHIFTS
MASK_TYPES = first.MASK_TYPES
KNOWN_PAIR = first.KNOWN_PAIR
BATCH = first.BATCH
D = first.D
TOKENS = first.TOKENS
DEPLOYED_BF16_BAR = first.DEPLOYED_BF16_BAR


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
    invalid = json.loads(FIRST_RESULT.read_text())
    if invalid.get("rung") != 502 \
            or invalid.get("pred_a_exact_live_source_pair_instrument") is not False \
            or invalid.get("pred_b_known_mlp9_parent_response_retained") is not False \
            or invalid.get("strong_null") is not True \
            or invalid.get("next_step") != "repair_source_pair_instrument_only" \
            or invalid["instrument"]["numerical_response_rms_over_complete"][0][0] <= .02:
        raise RuntimeError("first rung502 receipt does not license the instrument repair")
    rows, parent_metadata = action_parent.validate_inputs()
    circuit_rows, circuit_masks, tags, validation_tags, _, circuit_metadata = \
        circuit_parent.validate_inputs()
    if not torch.equal(rows, circuit_rows) or len(tags) != 32 or len(validation_tags) != 30:
        raise RuntimeError("rung502b row or 32/30 circuit authority changed")
    return rows, circuit_masks, tags, {"parent": parent_metadata,
                                       "circuits": circuit_metadata,
                                       "allocation_gauges": list(GAUGES),
                                       "first_outcomes_reused_for_scoring": False}


def _norm_weights(sources):
    energy = sources.double().square().sum(-1, keepdim=True)
    return (energy / energy.sum(2, keepdim=True).clamp_min(1e-30)).to(sources.dtype)


def exact_source_gauges(model, x0, attention_writes, prior_writes, raw_state, z):
    if len(attention_writes) != 10 or len(prior_writes) != 9:
        raise RuntimeError("MLP9 source count changed")
    embedding_coefficient, write_coefficients = first._source_coefficients(model)
    analytic = [embedding_coefficient * x0.float()]
    analytic.extend(write_coefficients[i] * attention_writes[i].float() for i in range(10))
    analytic.extend(write_coefficients[i] * prior_writes[i].float() for i in range(9))
    analytic = torch.stack(analytic, dim=2)
    raw = raw_state.float()
    raw_round = raw - analytic.sum(2)
    alpha = (z.float() * raw).sum(-1, keepdim=True) \
        / raw.square().sum(-1, keepdim=True).clamp_min(1e-30)

    e_raw = analytic.clone()
    e_raw[:, :, 0] += raw_round
    e_norm = alpha.unsqueeze(2) * e_raw
    norm_round_e = z.float() - e_norm.sum(2)
    e_norm[:, :, 0] += norm_round_e

    proportional_raw = analytic + _norm_weights(analytic) * raw_round.unsqueeze(2)
    proportional_norm = alpha.unsqueeze(2) * proportional_raw
    norm_round_p = z.float() - proportional_norm.sum(2)
    proportional_norm = proportional_norm \
        + _norm_weights(proportional_norm) * norm_round_p.unsqueeze(2)

    gauges = {"E_ABSORBS": e_norm.detach(),
              "PROPORTIONAL": proportional_norm.detach()}
    analytic_sum = analytic.sum(2)
    original_alpha = (z.float() * analytic_sum).sum(-1, keepdim=True) \
        / analytic_sum.square().sum(-1, keepdim=True).clamp_min(1e-30)
    original_sources = original_alpha.unsqueeze(2) * analytic
    original_numerical = z.float() - original_sources.sum(2)
    diagnostics = {
        "raw_round_rms_over_raw": float(raw_round.double().square().mean().sqrt()
                                         / raw.double().square().mean().sqrt().clamp_min(1e-30)),
        "norm_round_e_rms_over_z": float(norm_round_e.double().square().mean().sqrt()
                                           / z.double().square().mean().sqrt().clamp_min(1e-30)),
        "norm_round_proportional_rms_over_z": float(
            norm_round_p.double().square().mean().sqrt()
            / z.double().square().mean().sqrt().clamp_min(1e-30)),
        "state_closure": {name: first._relative_squared(value.sum(2), z.float())
                          for name, value in gauges.items()},
        "alpha": alpha.detach(),
    }
    return gauges, (original_sources.detach(), original_numerical.detach()), diagnostics


def _source_factors(mlp, gauges, original, deployed_write):
    factors = {}
    zero = torch.zeros_like(next(iter(gauges.values()))[:, :, 0])
    for name, sources in gauges.items():
        factors[name] = first._source_factors(mlp, sources, zero, deployed_write)
    original_factors = first._source_factors(
        mlp, original[0], original[1], deployed_write)
    return factors, original_factors["numerical_output"]


def _forward(model, tokens, scales, *, direct=False, background="early_present",
             state="late_native", gradient_leaf=False, source_factors=False):
    facade.validate_production_model(model)
    facade.validate_tokens(tokens, production_shape=True)
    if background not in BACKGROUNDS or state not in (*STATES, "late_native"):
        raise ValueError("unregistered action")
    cached = {}
    attention_writes = []
    prior_writes = []
    capture = {}
    diagnostics = {"factor_reconstruction_max": 0.0, "early_edit_rms": 0.0,
                   "late_edit_rms": 0.0, "raw_round_rms_over_raw_max": 0.0,
                   "norm_round_e_rms_over_z_max": 0.0,
                   "norm_round_proportional_rms_over_z_max": 0.0,
                   "state_closure": {name: 0.0 for name in GAUGES}}
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
            if source_factors:
                gauges, original, source_diagnostics = exact_source_gauges(
                    model, x0.detach(), attention_writes, prior_writes,
                    raw_mlp_state.detach(), z.detach())
                capture["factors"], capture["explicit_numerical_output"] = _source_factors(
                    block.mlp, gauges, original, mlp_write.detach())
                capture["alpha"] = source_diagnostics.pop("alpha")
                diagnostics["raw_round_rms_over_raw_max"] = source_diagnostics[
                    "raw_round_rms_over_raw"]
                diagnostics["norm_round_e_rms_over_z_max"] = source_diagnostics[
                    "norm_round_e_rms_over_z"]
                diagnostics["norm_round_proportional_rms_over_z_max"] = source_diagnostics[
                    "norm_round_proportional_rms_over_z"]
                diagnostics["state_closure"] = source_diagnostics["state_closure"]
            if gradient_leaf:
                mlp_write = mlp_write.detach().requires_grad_(True)
                capture["leaf"] = mlp_write
                audit["mlp9_leaves"] += 1
        prior_writes.append(mlp_write.detach())
        x = x + mlp_write
    logits = model.lm_head(F.rms_norm(x, (D,)))
    logits = (30.0 * torch.tanh(logits / 30.0)).float()
    if tuple(logits.shape) != (*tokens.shape, facade.LOGIT_VOCAB):
        raise RuntimeError("manual source-closed forward shape changed")
    expected = ({"native_attention": 18, "replayed_attention": 0, "native_mlp": 18,
                 "mlp9_leaves": int(gradient_leaf)} if direct else
                {"native_attention": 15, "replayed_attention": 3, "native_mlp": 18,
                 "mlp9_leaves": int(gradient_leaf)})
    expected_capture = {"deployed_write"}
    if source_factors:
        expected_capture |= {"factors", "alpha", "explicit_numerical_output"}
    if gradient_leaf:
        expected_capture.add("leaf")
    if audit != expected or set(capture) != expected_capture:
        raise RuntimeError(f"manual forward audit failed: {audit}, {set(capture)}")
    return logits, capture, diagnostics, audit


def _update_instrument(diagnostics, capture, diag):
    diagnostics["factor_reconstruction_max"] = max(
        diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
    diagnostics["raw_round_rms_over_raw_max"] = max(
        diagnostics["raw_round_rms_over_raw_max"], diag["raw_round_rms_over_raw_max"])
    diagnostics["norm_round_e_rms_over_z_max"] = max(
        diagnostics["norm_round_e_rms_over_z_max"], diag["norm_round_e_rms_over_z_max"])
    diagnostics["norm_round_proportional_rms_over_z_max"] = max(
        diagnostics["norm_round_proportional_rms_over_z_max"],
        diag["norm_round_proportional_rms_over_z_max"])
    for gauge in GAUGES:
        diagnostics["state_closure"][gauge] = max(
            diagnostics["state_closure"][gauge], diag["state_closure"][gauge])
        factors = capture["factors"][gauge]
        diagnostics["float32_pair_closure"][gauge] = max(
            diagnostics["float32_pair_closure"][gauge], factors["float32_closure"])
        diagnostics["float32_vs_deployed"][gauge] = max(
            diagnostics["float32_vs_deployed"][gauge], factors["deployed_relative_squared"])


def _parent_reports(stats):
    return [[first._complete_report(stats, background, quarter)
             for quarter in range(4)] for background in range(2)]


def _pooled_complete(stats, background, quarters):
    row = stats["denominators"][background, list(quarters)].sum(0)
    return first._cosine_residual(float(row[1]), float(row[0]), float(row[2]))


def collect(model, rows, circuit_masks, tags, scales):
    copy_mask = action_parent._task_masks(rows)["copy_positive"]
    banks = {gauge: first._empty_stats(len(tags)) for gauge in GAUGES}
    diagnostics = {
        "factor_reconstruction_max": 0.0, "minimum_nonzero_edit_rms": float("inf"),
        "raw_round_rms_over_raw_max": 0.0, "norm_round_e_rms_over_z_max": 0.0,
        "norm_round_proportional_rms_over_z_max": 0.0,
        "state_closure": {gauge: 0.0 for gauge in GAUGES},
        "float32_pair_closure": {gauge: 0.0 for gauge in GAUGES},
        "float32_vs_deployed": {gauge: 0.0 for gauge in GAUGES},
        "background_native_early_present_max_abs": 0.0,
        "alpha_stats": torch.zeros(2, len(STATES), 4, 3, dtype=torch.float64),
        "explicit_numerical_response2": torch.zeros(2, 4, dtype=torch.float64),
        "complete_response2": torch.zeros(2, 4, dtype=torch.float64),
    }
    calls = {"early_present_native": 0, "early_absent_native": 0, "actions": 0,
             "copy_backwards": 0, "circuit_backwards": 0, "source_captures": 0,
             "native_attention": 0, "replayed_attention": 0, "native_mlp": 0,
             "mlp9_leaves": 0}
    selected = {gauge: None for gauge in GAUGES}
    selection_detail = {gauge: None for gauge in GAUGES}
    device = next(model.parameters()).device
    mlp9 = model.transformer.h[9].mlp

    for start in range(0, 500, BATCH):
        if start == 248:
            for gauge in GAUGES:
                selected[gauge], selection_detail[gauge] = first._select_pairs(banks[gauge])
        stop = min(start + BATCH, 500)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        targets = batch_rows[:, 1:].to(device)
        with torch.no_grad():
            _, native_present, native_diag, native_audit = _forward(
                model, tokens, scales, direct=True)
        calls["early_present_native"] += 1
        for key in ("native_attention", "replayed_attention", "native_mlp", "mlp9_leaves"):
            calls[key] += native_audit[key]

        for background_index, background in enumerate(BACKGROUNDS):
            if background_index == 0:
                native = native_present
            else:
                with torch.no_grad():
                    _, native, background_diag, background_audit = _forward(
                        model, tokens, scales, background=background, state="late_native")
                calls["early_absent_native"] += 1
                for key in ("native_attention", "replayed_attention", "native_mlp",
                            "mlp9_leaves"):
                    calls[key] += background_audit[key]
            logits, absent, absent_diag, absent_audit = _forward(
                model, tokens, scales, background=background, state="late_absent",
                gradient_leaf=True, source_factors=True)
            with torch.no_grad():
                _, score, score_diag, score_audit = _forward(
                    model, tokens, scales, background=background, state="score_donor",
                    source_factors=True)
                _, payload, payload_diag, payload_audit = _forward(
                    model, tokens, scales, background=background, state="payload_donor",
                    source_factors=True)
            calls["actions"] += 3
            calls["source_captures"] += 3
            for audit in (absent_audit, score_audit, payload_audit):
                for key in ("native_attention", "replayed_attention", "native_mlp",
                            "mlp9_leaves"):
                    calls[key] += audit[key]
            for diag, capture in ((absent_diag, absent), (score_diag, score),
                                  (payload_diag, payload)):
                _update_instrument(diagnostics, capture, diag)
                for key in ("early_edit_rms", "late_edit_rms"):
                    if diag[key] > 0:
                        diagnostics["minimum_nonzero_edit_rms"] = min(
                            diagnostics["minimum_nonzero_edit_rms"], diag[key])
            if background_index == 0:
                diagnostics["background_native_early_present_max_abs"] = max(
                    diagnostics["background_native_early_present_max_abs"],
                    float((native["deployed_write"].float()
                           - native_present["deployed_write"].float()).abs().max()))

            quarter_selections = first._quarter_selections(copy_mask, start, stop)
            document_rows = torch.arange(start, stop)
            for state_index, capture in enumerate((absent, score, payload)):
                alpha = capture["alpha"].double().cpu()
                for quarter, (left, right) in enumerate(DOC_QUARTERS):
                    values = alpha[(document_rows >= left) & (document_rows < right)]
                    if values.numel():
                        diagnostics["alpha_stats"][background_index, state_index, quarter, 0] \
                            += values.sum()
                        diagnostics["alpha_stats"][background_index, state_index, quarter, 1] \
                            += values.square().sum()
                        diagnostics["alpha_stats"][background_index, state_index, quarter, 2] \
                            += values.numel()

            for gauge in GAUGES:
                stats = banks[gauge]
                for quarter, selected_cpu in quarter_selections:
                    selected_copy = selected_cpu.to(device)
                    first._accumulate_complete(
                        stats, background_index, quarter, selected_copy,
                        native["deployed_write"], absent["deployed_write"],
                        score["deployed_write"], payload["deployed_write"])
                    down_ref = first._linear(
                        torch.where(selected_copy[..., None],
                                    absent["deployed_write"].float()
                                    - native["deployed_write"].float(),
                                    torch.zeros_like(absent["deployed_write"].float())),
                        mlp9.Down.weight.float().T)
                    absent_ref = first._unordered_contraction(
                        down_ref, absent["factors"][gauge])
                    score_ref = first._unordered_contraction(
                        down_ref, score["factors"][gauge])
                    payload_ref = first._unordered_contraction(
                        down_ref, payload["factors"][gauge])
                    stats["pair_response_num"][background_index, quarter] += \
                        absent_ref - score_ref
                    stats["pair_payload_num"][background_index, quarter] += \
                        absent_ref - payload_ref
            for quarter, selected_cpu in quarter_selections:
                selected_copy = selected_cpu.to(device)
                numerical_response = (absent["explicit_numerical_output"]
                                      - score["explicit_numerical_output"])
                complete_response = (absent["deployed_write"].float()
                                     - score["deployed_write"].float())
                diagnostics["explicit_numerical_response2"][background_index, quarter] \
                    += float(numerical_response[selected_copy].double().square().sum())
                diagnostics["complete_response2"][background_index, quarter] \
                    += float(complete_response[selected_copy].double().square().sum())

            circuit_selections = [] if start < 248 else circuit_parent._batch_selections(
                circuit_masks, tags, start, stop, 374)
            gradient_jobs = [("copy", job) for job in quarter_selections]
            gradient_jobs += [("circuit", job) for job in circuit_selections]
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none") \
                .view(len(batch_rows), TOKENS)
            group_outputs = {}
            if start >= 248:
                for gauge in GAUGES:
                    pair_indices = selected[gauge]
                    group_outputs[gauge] = {
                        "absent": first._group_output(
                            mlp9, absent["factors"][gauge], pair_indices),
                        "score": first._group_output(
                            mlp9, score["factors"][gauge], pair_indices),
                        "payload": first._group_output(
                            mlp9, payload["factors"][gauge], pair_indices),
                    }
                    group = group_outputs[gauge]["absent"] - group_outputs[gauge]["score"]
                    payload_group = (group_outputs[gauge]["absent"]
                                     - group_outputs[gauge]["payload"])
                    full = absent["deployed_write"].float() - score["deployed_write"].float()
                    reference = absent["deployed_write"].float() - native["deployed_write"].float()
                    for quarter, selected_cpu in quarter_selections:
                        if quarter < 2:
                            continue
                        chosen = selected_cpu.to(device)
                        g, f, r, p = (group[chosen].double(), full[chosen].double(),
                                     reference[chosen].double(), payload_group[chosen].double())
                        row = banks[gauge]["confirmation_local"][background_index, quarter - 2]
                        row[0] += float(g.square().sum())
                        row[1] += float(f.square().sum())
                        row[2] += float((g * f).sum())
                        row[3] += float((g * r).sum())
                        row[4] += float((f * r).sum())
                        row[5] += float((p * r).sum())
                        row[6] += float(r.square().sum())
                        row[9] += float(p.square().sum())
                        row[10] += float(g.square().sum())
                        row[11] += int(chosen.sum())

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
                down_gradient = first._linear(gradient.float(), mlp9.Down.weight.float().T)
                for gauge in GAUGES:
                    stats = banks[gauge]
                    if kind == "copy":
                        absent_grad = first._unordered_contraction(
                            down_gradient, absent["factors"][gauge])
                        score_grad = first._unordered_contraction(
                            down_gradient, score["factors"][gauge])
                        stats["pair_gradient_num"][background_index, quarter] += \
                            absent_grad - score_grad
                        complete = float((gradient.float() *
                                          (absent["deployed_write"].float()
                                           - score["deployed_write"].float())).sum())
                        stats["denominators"][background_index, quarter, 5] += complete
                        stats["denominators"][background_index, quarter, 7] += 1
                        if start >= 248:
                            group = (group_outputs[gauge]["absent"]
                                     - group_outputs[gauge]["score"])
                            row = stats["confirmation_local"][background_index, quarter - 2]
                            row[7] += float((gradient.float() * group).sum())
                            row[8] += complete
                    else:
                        group = group_outputs[gauge]["absent"] - group_outputs[gauge]["score"]
                        payload_group = (group_outputs[gauge]["absent"]
                                         - group_outputs[gauge]["payload"])
                        full = absent["deployed_write"].float() - score["deployed_write"].float()
                        values = [float((gradient.float() * full).sum()),
                                  float((gradient.float() * group).sum()),
                                  float((gradient.float() * payload_group).sum())]
                        values.extend(float((gradient.float() * torch.roll(
                            group, shift, dims=1)).sum()) for shift in POSITION_SHIFTS)
                        stats["circuit_sums"][half, background_index, mask_index, tag_index] \
                            += torch.tensor(values, dtype=torch.float64)
                        if background_index == 0:
                            stats["circuit_counts"][half, mask_index, tag_index] += int(chosen.sum())
                if kind == "copy":
                    calls["copy_backwards"] += 1
                else:
                    calls["circuit_backwards"] += 1
            del logits, nll, absent, score, payload, group_outputs

    for gauge in GAUGES:
        if selected[gauge] is None:
            selected[gauge], selection_detail[gauge] = first._select_pairs(banks[gauge])
    return banks, diagnostics, calls, selected, selection_detail


def _alpha_report(stats):
    output = []
    for background in range(2):
        rows = []
        for quarter in range(4):
            states = {}
            for state_index, state in enumerate(STATES):
                row = stats[background, state_index, quarter]
                count = max(float(row[2]), 1.0)
                mean = float(row[0] / count)
                variance = max(float(row[1] / count) - mean * mean, 0.0)
                states[state] = {"mean": mean, "std": math.sqrt(variance),
                                 "values": int(row[2])}
            states["score_vs_absent_relative_mean_drift"] = abs(
                states["score_donor"]["mean"] - states["late_absent"]["mean"]) \
                / max(abs(states["late_absent"]["mean"]), 1e-30)
            rows.append(states)
        output.append(rows)
    return output


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(SOURCES) == 20 and len(SOURCE_PAIRS) == 210
        assert DOC_QUARTERS == ((0, 124), (124, 248), (248, 374), (374, 500))
        analytic = torch.randn(2, 3, 20, 7)
        raw_round = torch.randn(2, 3, 7) * .01
        raw = analytic.sum(2) + raw_round
        z = raw / raw.square().mean(-1, keepdim=True).sqrt()
        # Test the allocation helper's core identities without a production model.
        proportional_raw = analytic + _norm_weights(analytic) * raw_round.unsqueeze(2)
        torch.testing.assert_close(proportional_raw.sum(2), raw)
        assert GAUGES == ("E_ABSORBS", "PROPORTIONAL")
        print(json.dumps({"status": "dry_run_passed", "rung": "502b",
                          "model_loaded": False, "source_pair_outcomes_opened": False,
                          "gauges": list(GAUGES), "pair_count": 210,
                          "model_forwards": 1000,
                          "validation_documents_or_tags_opened": False}, indent=2))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung502b namespace already exists")
    rows, circuit_masks, tags, metadata = validate_inputs()
    parent_receipt = json.loads(PARENT_RESULT.read_text())
    scales = parent_receipt["frozen_scales"][action_parent.KNOWN_POSITIVE]
    copy_mask = action_parent._task_masks(rows)["copy_positive"]
    expected_copy = sum(len(first._quarter_selections(
        copy_mask, start, min(start + BATCH, 500))) for start in range(0, 500, BATCH)) * 2
    expected_circuit = sum(len(circuit_parent._batch_selections(
        circuit_masks, tags, start, min(start + BATCH, 500), 374))
        for start in range(248, 500, BATCH)) * 2
    expected_calls = {
        "early_present_native": 125, "early_absent_native": 125, "actions": 750,
        "copy_backwards": expected_copy, "circuit_backwards": expected_circuit,
        "source_captures": 750,
        "native_attention": 125 * 18 + 875 * 15,
        "replayed_attention": 875 * 3,
        "native_mlp": 1000 * 18, "mlp9_leaves": 250,
    }
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    banks, diagnostics, calls, selected, selection_detail = collect(
        model, rows, circuit_masks, tags, scales)
    calls_exact = calls == expected_calls
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256 and calls_exact
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["minimum_nonzero_edit_rms"] > 0
        and diagnostics["background_native_early_present_max_abs"] == 0.0
        and all(diagnostics["state_closure"][gauge] <= 1e-12 for gauge in GAUGES)
        and all(diagnostics["float32_pair_closure"][gauge] <= 1e-8 for gauge in GAUGES)
        and all(diagnostics["float32_vs_deployed"][gauge] <= DEPLOYED_BF16_BAR
                for gauge in GAUGES)
        and diagnostics["raw_round_rms_over_raw_max"] <= .03125
        and diagnostics["norm_round_e_rms_over_z_max"] <= .015625
        and diagnostics["norm_round_proportional_rms_over_z_max"] <= .015625)

    reports = {gauge: _parent_reports(banks[gauge]) for gauge in GAUGES}
    parent_bars = all(
        row["score"]["cosine"] >= .75
        and row["score"]["positive_scale_residual"] <= .70
        and (row["score"]["cosine"] >= row["payload"]["cosine"] + .30
             or row["score"]["positive_scale_residual"]
             <= row["payload"]["positive_scale_residual"] - .30)
        for rows_ in reports[GAUGES[0]] for row in rows_)
    parent_differences = []
    parent_analysis = parent_receipt["discovery"]["analysis"][action_parent.KNOWN_POSITIVE]
    for background_index, background in enumerate(BACKGROUNDS):
        for half, quarters in enumerate(((0, 1), (2, 3))):
            observed = _pooled_complete(banks[GAUGES[0]], background_index, quarters)["cosine"]
            expected = parent_analysis[background]["score_donor"][half]["reader"][
                "copy_positive"]["cosine"]
            parent_differences.append({"background": background, "half": half,
                                       "observed": observed, "rung501": expected,
                                       "absolute_difference": abs(observed - expected)})
    pred_b = bool(parent_bars and max(row["absolute_difference"]
                                      for row in parent_differences) <= .03)

    confirmation = {}
    pred_c_by_gauge = {}
    circuits = {}
    pred_d_by_gauge = {}
    for gauge in GAUGES:
        pred_c_by_gauge[gauge], confirmation[gauge] = first._confirmation_report(
            banks[gauge], selected[gauge])
        pred_d_by_gauge[gauge], circuits[gauge] = first._circuit_report(banks[gauge], tags)
    selected_names = {gauge: [PAIR_NAMES[index] for index in selected[gauge]] for gauge in GAUGES}
    same_selection = selected_names[GAUGES[0]] == selected_names[GAUGES[1]]
    cross_gauge_signs = bool(same_selection and all(
        bool(banks[GAUGES[0]]["pair_gradient_num"][background, quarter, pair] > 0)
        == bool(banks[GAUGES[1]]["pair_gradient_num"][background, quarter, pair] > 0)
        for pair in selected[GAUGES[0]] for background in range(2) for quarter in (2, 3)))
    pred_c = bool(same_selection and selected[GAUGES[0]]
                  and all(pred_c_by_gauge.values()))
    pred_d = bool(pred_c and cross_gauge_signs and all(pred_d_by_gauge.values()))
    pred_e = bool(pred_a and pred_b and pred_c and pred_d)
    strong_null = bool(not pred_a or not pred_b or not pred_c or not pred_d)
    alpha_report = _alpha_report(diagnostics.pop("alpha_stats"))
    explicit_numerical_ratio = torch.sqrt(
        diagnostics.pop("explicit_numerical_response2")
        / diagnostics.pop("complete_response2").clamp_min(1e-30)).tolist()
    bundle = {
        "schema": "mlp9_score_response_source_pair_atlas_rung502b_stats_v1",
        "gauge_banks": {gauge: banks[gauge] for gauge in GAUGES},
        "raw_tokens_logits_gradients_or_pair_vectors_included": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": "502b",
        "claim_level": "two_gauge_exact_local_source_pair_screen_not_circuit",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "sources": list(SOURCES),
        "source_pairs": list(PAIR_NAMES), "gauges": list(GAUGES),
        "document_quarters": [list(value) for value in DOC_QUARTERS],
        "selection": {"by_gauge": selected_names, "same_complete_set": same_selection,
                      "details": selection_detail},
        "parent_response": reports, "rung501_parent_differences": parent_differences,
        "confirmation": confirmation, "circuit_fingerprints": circuits,
        "cross_gauge_confirmation_pair_gradient_signs_agree": cross_gauge_signs,
        "normalization_gain": alpha_report,
        "old_explicit_numerical_response_rms_over_complete": explicit_numerical_ratio,
        "instrument": {**diagnostics, "calls": calls, "expected_calls": expected_calls,
                       "calls_exact": calls_exact},
        'pred_a_repaired_exact_two_gauge_instrument': pred_a,
        'pred_b_background_parent_response_reproduces': pred_b,
        'pred_c_cross_gauge_compact_group_confirms': pred_c,
        'pred_d_cross_gauge_downstream_use_confirms': pred_d,
        'pred_e_candidate_for_finite_group_removal': pred_e,
        "strong_null": strong_null,
        "validation_documents_or_tags_opened": False,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {"model_forwards": 1000,
                            "backwards": calls["copy_backwards"] + calls["circuit_backwards"],
                            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                            "deployed_parameters_added": 0, "deployed_parameters_saved": 0},
        "runtime_s": time.time() - started,
        "next_step": (
            "repair_two_gauge_instrument_only" if not pred_a else
            "retire_source_atlas_parent_mismatch" if not pred_b else
            "source_pair_semantics_not_identified_use_finite_source_factorial_or_float32_control"
            if not pred_c else
            "change_downstream_observation_before_circuit_claim" if not pred_d else
            "preregister_finite_mlp9_source_pair_group_removal"),
    }
    dump(result, OUT)
    print(json.dumps({"status": "complete", "rung": "502b",
                      "predictions": {key: value for key, value in result.items()
                                      if key.startswith("pred_")},
                      "strong_null": strong_null, "selection": selected_names,
                      "execution_price": result["execution_price"],
                      "runtime_s": result["runtime_s"], "next_step": result["next_step"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
