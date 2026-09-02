#!/usr/bin/env python3
"""RUNG502 -- exact named-source-pair anatomy of MLP9's copy-score response."""

# BQGATE: EXPERIMENT
# pred_a exact live named-source and source-pair instrument
# pred_b the calibrated complete MLP9 score response survives source capture
# pred_c one compact exact source-pair group is stable without reselection
# pred_d supported downstream circuit gradients confirm location and specificity
# pred_e a passing group is only a candidate for a separately registered finite intervention

from __future__ import annotations

import hashlib
import itertools
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
import equality_score_directed_action_graph_rung501 as parent
import mlp0_branch_circuit_response_rung481 as circuit_parent


PREREG = POLY / "MLP9_SCORE_RESPONSE_SOURCE_PAIR_ATLAS_RUNG502_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/equality_score_directed_action_graph_rung501.py"
PARENT_RESULT = ROOT / "equality_score_directed_action_graph_rung501_results.json"
PARENT_BUNDLE = ROOT / "equality_score_directed_action_graph_rung501_bundle.pt"
CIRCUIT_SOURCE = ROOT / "ops/mlp0_branch_circuit_response_rung481.py"
OUT = ROOT / "mlp9_score_response_source_pair_atlas_rung502_results.json"
BUNDLE = ROOT / "mlp9_score_response_source_pair_atlas_rung502_bundle.pt"
HASHES = {
    PREREG: "b50229786060cbf58d4653aa8f7a5d7c615b3ca4c4ddd7fc923f7a57cc3617d0",
    PARENT_SOURCE: "97f3946f558f3d61fc952a9b6ddc7c334b51ccc0ccfe5f02c6ecced417f1e077",
    PARENT_RESULT: "b17a9b274e4c61e0b4a3fc68d8ce84ec6f8e76f257c3d898e6a6990492301c4f",
    PARENT_BUNDLE: "728d9be2681a60579b743626bb8eb7e8cc09414fdb9c90cf128388f2049f59c5",
    CIRCUIT_SOURCE: "ef08017a30ceb0c9e4481198fc1d58c5b0bf8cd37707d2223c42db9eb04f1f44",
}
SOURCES = ("E",) + tuple(f"A{i}" for i in range(10)) + tuple(f"M{i}" for i in range(9))
SOURCE_PAIRS = tuple(itertools.combinations_with_replacement(range(len(SOURCES)), 2))
PAIR_NAMES = tuple(f"{SOURCES[i]}x{SOURCES[j]}" for i, j in SOURCE_PAIRS)
BACKGROUNDS = parent.BACKGROUNDS
STATES = ("late_absent", "score_donor", "payload_donor")
POSITION_SHIFTS = tuple(range(1, 17))
MASK_TYPES = circuit_parent.MASK_TYPES
DOC_QUARTERS = ((0, 124), (124, 248), (248, 374), (374, 500))
KNOWN_PAIR = parent.PAIRS[parent.PAIR_NAMES.index(parent.KNOWN_POSITIVE)]
BATCH = 4
D = 1152
H = 4608
TOKENS = 256
U = 2.0 ** -8
DEPLOYED_BF16_BAR = 16 * U * U


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_squared(left, right) -> float:
    delta = left.double() - right.double()
    return float(delta.square().sum() / right.double().square().sum().clamp_min(1e-30))


def _cosine_residual(left2: float, right2: float, cross: float) -> dict:
    cosine = cross / math.sqrt(max(left2 * right2, 1e-30))
    scale = cross / max(left2, 1e-30)
    positive = max(scale, 0.0)
    residual = math.sqrt(max(right2 - 2 * positive * cross + positive * positive * left2, 0.0)
                         / max(right2, 1e-30))
    return {"cosine": cosine, "left_to_right_positive_scale": scale,
            "positive_scale_residual": residual}


def _quantile95(values) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)]


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(PARENT_RESULT.read_text())
    if receipt.get("rung") != 501 \
            or receipt.get("pred_a_exact_live_isolated_instrument") is not True \
            or receipt.get("pred_b_calibration_tripwires_reproduce") is not True \
            or receipt.get("pred_c_new_confirmed_directed_edge") is not False \
            or receipt.get("validation_licensed_and_opened") is not False \
            or receipt.get("strong_null") is not True \
            or receipt.get("next_step") \
            != "calibrated_known_edge_isolated_then_decompose_its_mlp9_source_pairs" \
            or receipt["discovery"]["checks"]["confirmed_directed_edges"] \
            != [parent.KNOWN_POSITIVE]:
        raise RuntimeError("rung501 does not license the isolated known-edge source-pair atlas")
    rows, parent_metadata = parent.validate_inputs()
    circuit_rows, circuit_masks, discovery_tags, validation_tags, _, circuit_metadata = \
        circuit_parent.validate_inputs()
    if not torch.equal(rows, circuit_rows):
        raise RuntimeError("rung501 and circuit rows differ")
    if len(discovery_tags) != 32 or len(validation_tags) != 30:
        raise RuntimeError("frozen circuit split changed")
    return rows, circuit_masks, discovery_tags, {
        "parent": parent_metadata, "circuits": circuit_metadata,
        "sources": list(SOURCES), "pair_names": list(PAIR_NAMES),
        "validation_tags_opened": False,
    }


def _source_coefficients(model):
    lambda0 = [block.lambdas[0].detach().float() for block in model.transformer.h[:10]]
    lambda1 = [block.lambdas[1].detach().float() for block in model.transformer.h[:10]]
    embedding = torch.ones_like(lambda0[0])
    for left, skip in zip(lambda0, lambda1):
        embedding = left * embedding + skip
    writes = []
    for site in range(10):
        coefficient = torch.ones_like(lambda0[0])
        for later in range(site + 1, 10):
            coefficient = coefficient * lambda0[later]
        writes.append(coefficient)
    return embedding, tuple(writes)


def _normalized_sources(model, x0, attention_writes, event):
    if len(attention_writes) != 10 or len(event.prior_writes) != 9:
        raise RuntimeError("MLP9 source count changed")
    embedding_coefficient, write_coefficients = _source_coefficients(model)
    raw = [embedding_coefficient * x0.float()]
    raw.extend(write_coefficients[i] * attention_writes[i].float() for i in range(10))
    raw.extend(write_coefficients[i] * event.prior_writes[i].float() for i in range(9))
    raw = torch.stack(raw, dim=2)
    raw_sum = raw.sum(dim=2)
    z = event.state.float()
    gain = (z * raw_sum).sum(-1, keepdim=True) \
        / raw_sum.square().sum(-1, keepdim=True).clamp_min(1e-30)
    sources = gain.unsqueeze(2) * raw
    numerical = z - sources.sum(dim=2)
    error = _relative_squared(sources.sum(dim=2) + numerical, z)
    return sources.detach(), numerical.detach(), gain.detach(), error


def _linear(value, weight):
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def _source_factors(mlp, sources, numerical, deployed_write):
    left = _linear(sources.float(), mlp.Left.weight.float())
    right = _linear(sources.float(), mlp.Right.weight.float())
    left_num = _linear(numerical.float(), mlp.Left.weight.float())
    right_num = _linear(numerical.float(), mlp.Right.weight.float())
    z = sources.sum(2) + numerical
    left_full = left.sum(2) + left_num
    right_full = right.sum(2) + right_num
    full_hidden = left_full * right_full
    semantic_hidden = left.sum(2) * right.sum(2)
    numerical_hidden = full_hidden - semantic_hidden
    independent = _linear(full_hidden, mlp.Down.weight.float()) + mlp.Down_bias.float()
    semantic_output = _linear(semantic_hidden, mlp.Down.weight.float())
    numerical_output = _linear(numerical_hidden, mlp.Down.weight.float())
    rebuilt = semantic_output + numerical_output + mlp.Down_bias.float()
    return {
        "left": left.detach(), "right": right.detach(),
        "numerical_output": numerical_output.detach(),
        "independent_write": independent.detach(),
        "float32_closure": _relative_squared(rebuilt, independent),
        "deployed_relative_squared": _relative_squared(independent, deployed_write.float()),
        "state": z.detach(),
    }


def _unordered_contraction(weight, factors):
    ordered = torch.einsum(
        "bth,btsh,btuh->su", weight.float(), factors["left"], factors["right"])
    values = []
    for left, right in SOURCE_PAIRS:
        value = ordered[left, right]
        if left != right:
            value = value + ordered[right, left]
        values.append(value)
    return torch.stack(values).double().cpu()


def _group_output(mlp, factors, pair_indices):
    hidden = torch.zeros_like(factors["left"][:, :, 0])
    for pair_index in pair_indices:
        left, right = SOURCE_PAIRS[pair_index]
        hidden = hidden + factors["left"][:, :, left] * factors["right"][:, :, right]
        if left != right:
            hidden = hidden + factors["left"][:, :, right] * factors["right"][:, :, left]
    return _linear(hidden, mlp.Down.weight.float())


def _forward(model, tokens, scales, *, direct=False, background="early_present",
             state="late_native", gradient_leaf=False, source_factors=False):
    cached = {}
    attention_writes = []
    capture = {}
    diagnostics = {"factor_reconstruction_max": 0.0, "early_edit_rms": 0.0,
                   "late_edit_rms": 0.0, "state_source_error": 0.0}
    audit = {"native_attention": 0, "replayed_attention": 0, "native_mlp": 0,
             "mlp9_leaves": 0}
    x0 = F.rms_norm(model.transformer.wte(tokens), (D,)).detach()

    def attention(event):
        if direct or event.site not in parent.factor_parent.stage1.SITE_HEADS:
            write, next_value = event.block.attn(event.state, event.first_value)
            audit["native_attention"] += 1
        else:
            write, factors, support, error = parent.factor_parent._factor_site(
                event.state, event.first_value, event.block.attn, event.site, event.tokens)
            audit["replayed_attention"] += 1
            diagnostics["factor_reconstruction_max"] = max(
                diagnostics["factor_reconstruction_max"], error)
            donor, recipient = KNOWN_PAIR
            if event.site == parent.factor_parent.TERMS[donor][1]:
                cached.update(factors[donor])
                if background == "early_absent":
                    edit = factors[donor]["native_term"]
                    write = write - edit
                    diagnostics["early_edit_rms"] = float(edit.float().square().mean().sqrt())
            if event.site == parent.factor_parent.TERMS[recipient][1]:
                if not cached:
                    raise RuntimeError("known donor factors unavailable")
                target = factors[recipient]
                if state != "late_native":
                    replacement = torch.zeros_like(target["factor_term"])
                    if state.endswith("donor"):
                        p, u = target["p"], target["u"]
                        if state == "score_donor":
                            p = cached["p"] * scales["score_ratio"]
                        elif state == "payload_donor":
                            u = cached["u"] * scales["payload_ratio"]
                        replacement = torch.bmm(p * support, u)
                    edit = replacement.to(write.dtype) - target["native_term"]
                    write = write + edit
                    diagnostics["late_edit_rms"] = float(edit.float().square().mean().sqrt())
            next_value = event.first_value
        attention_writes.append(write.detach())
        return write, next_value

    def mlp(event):
        audit["native_mlp"] += 1
        write = event.block.mlp(event.state)
        if event.site == 9:
            sources, numerical, gain, error = _normalized_sources(
                model, x0, attention_writes, event)
            diagnostics["state_source_error"] = error
            capture["deployed_write"] = write.detach()
            if source_factors:
                capture["factors"] = _source_factors(
                    event.block.mlp, sources, numerical, write.detach())
                capture["factors"]["gain"] = gain
            if gradient_leaf:
                write = write.detach().requires_grad_(True)
                capture["leaf"] = write
                audit["mlp9_leaves"] += 1
        return write

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
    expected = ({"native_attention": 18, "replayed_attention": 0, "native_mlp": 18,
                 "mlp9_leaves": int(gradient_leaf)} if direct else
                {"native_attention": 15, "replayed_attention": 3, "native_mlp": 18,
                 "mlp9_leaves": int(gradient_leaf)})
    if audit != expected or set(capture) != ({"deployed_write", "leaf", "factors"}
                                            if gradient_leaf and source_factors else
                                            {"deployed_write", "factors"}
                                            if source_factors else
                                            {"deployed_write", "leaf"}
                                            if gradient_leaf else {"deployed_write"}):
        raise RuntimeError(f"MLP9 source capture audit failed: {audit}, {set(capture)}")
    return logits, capture, diagnostics, audit


def _quarter_selections(mask, start, stop):
    rows = torch.arange(start, stop)
    selections = []
    for quarter, (left, right) in enumerate(DOC_QUARTERS):
        selected = mask[start:stop].clone() & ((rows >= left) & (rows < right))[:, None]
        if bool(selected.any()):
            selections.append((quarter, selected))
    return selections


def _empty_stats(tag_count):
    gain_stats = torch.zeros(2, len(STATES), 4, 5, dtype=torch.float64)
    gain_stats[..., 3] = float("inf")
    gain_stats[..., 4] = -float("inf")
    return {
        "pair_response_num": torch.zeros(2, 4, len(SOURCE_PAIRS), dtype=torch.float64),
        "pair_payload_num": torch.zeros(2, 4, len(SOURCE_PAIRS), dtype=torch.float64),
        "pair_gradient_num": torch.zeros(2, 4, len(SOURCE_PAIRS), dtype=torch.float64),
        "denominators": torch.zeros(2, 4, 8, dtype=torch.float64),
        # r2,h2,rh,p2,rp,grad_h,copy_tokens,copy_gradient_calls
        "confirmation_local": torch.zeros(2, 2, 12, dtype=torch.float64),
        # group2,full2,cross,group_r,full_r,payload_group_r,r2,grad_group,grad_full,
        # payload_group2,score_group2,copy_tokens
        "circuit_sums": torch.zeros(
            2, 2, len(MASK_TYPES), tag_count, 19, dtype=torch.float64),
        # full,group,payload_group,16 shifted groups
        "circuit_counts": torch.zeros(2, len(MASK_TYPES), tag_count, dtype=torch.float64),
        # sum, squared sum, count, minimum, maximum
        "gain_stats": gain_stats,
    }


def _accumulate_complete(stats, background, quarter, selected, native_write,
                         absent_write, score_write, payload_write):
    r = absent_write.float()[selected] - native_write.float()[selected]
    h = absent_write.float()[selected] - score_write.float()[selected]
    p = absent_write.float()[selected] - payload_write.float()[selected]
    row = stats["denominators"][background, quarter]
    row[0] += float(r.double().square().sum())
    row[1] += float(h.double().square().sum())
    row[2] += float((r.double() * h.double()).sum())
    row[3] += float(p.double().square().sum())
    row[4] += float((r.double() * p.double()).sum())
    row[6] += int(selected.sum())
    return r, h, p


def _select_pairs(stats):
    selected = []
    detail = {}
    for pair_index, name in enumerate(PAIR_NAMES):
        holds = True
        per_background = []
        for background in range(2):
            r2 = float(stats["denominators"][background, :2, 0].sum())
            grad_h = float(stats["denominators"][background, :2, 5].sum())
            response = float(stats["pair_response_num"][background, :2, pair_index].sum()) \
                / max(r2, 1e-30)
            payload = float(stats["pair_payload_num"][background, :2, pair_index].sum()) \
                / max(r2, 1e-30)
            gradient = float(stats["pair_gradient_num"][background, :2, pair_index].sum()) \
                / (grad_h if abs(grad_h) > 1e-30 else math.copysign(1e-30, grad_h or 1))
            quarter_signs = [
                bool(stats["pair_response_num"][background, q, pair_index] > 0
                     and stats["pair_gradient_num"][background, q, pair_index] > 0)
                for q in range(2)]
            row_holds = bool(response >= .01 and gradient >= .01
                             and response >= 2 * abs(payload) and all(quarter_signs))
            holds = holds and row_holds
            per_background.append({"response_fraction": response,
                                   "payload_response_fraction": payload,
                                   "gradient_fraction": gradient,
                                   "quarter_signs_positive": quarter_signs,
                                   "holds": row_holds})
        detail[name] = {"backgrounds": per_background, "selected": holds}
        if holds:
            selected.append(pair_index)
    return selected, detail


def collect(model, rows, circuit_masks, tags, scales):
    copy_mask = parent._task_masks(rows)["copy_positive"]
    stats = _empty_stats(len(tags))
    diagnostics = {
        "state_source_relative_squared_max": 0.0,
        "float32_pair_closure_relative_squared_max": 0.0,
        "float32_vs_deployed_relative_squared_max": 0.0,
        "factor_reconstruction_max": 0.0,
        "minimum_nonzero_edit_rms": float("inf"),
        "numerical_response2": torch.zeros(2, 4, dtype=torch.float64),
        "complete_response2": torch.zeros(2, 4, dtype=torch.float64),
    }
    calls = {"native": 0, "actions": 0, "copy_backwards": 0,
             "circuit_backwards": 0, "source_captures": 0,
             "native_attention": 0, "replayed_attention": 0,
             "native_mlp": 0, "mlp9_leaves": 0}
    selected_pairs = None
    selection_detail = None
    device = next(model.parameters()).device
    mlp9 = model.transformer.h[9].mlp

    for start in range(0, 500, BATCH):
        if start == 248 and selected_pairs is None:
            selected_pairs, selection_detail = _select_pairs(stats)
        stop = min(start + BATCH, 500)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        targets = batch_rows[:, 1:].to(device)
        with torch.no_grad():
            _, native, diag, native_audit = _forward(
                model, tokens, scales, direct=True, source_factors=False)
        calls["native"] += 1
        for key in ("native_attention", "replayed_attention", "native_mlp", "mlp9_leaves"):
            calls[key] += native_audit[key]
        diagnostics["state_source_relative_squared_max"] = max(
            diagnostics["state_source_relative_squared_max"], diag["state_source_error"])

        for background_index, background in enumerate(BACKGROUNDS):
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
                diagnostics["state_source_relative_squared_max"] = max(
                    diagnostics["state_source_relative_squared_max"], diag["state_source_error"])
                diagnostics["factor_reconstruction_max"] = max(
                    diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
                for key in ("early_edit_rms", "late_edit_rms"):
                    if diag[key] > 0:
                        diagnostics["minimum_nonzero_edit_rms"] = min(
                            diagnostics["minimum_nonzero_edit_rms"], diag[key])
                diagnostics["float32_pair_closure_relative_squared_max"] = max(
                    diagnostics["float32_pair_closure_relative_squared_max"],
                    capture["factors"]["float32_closure"])
                diagnostics["float32_vs_deployed_relative_squared_max"] = max(
                    diagnostics["float32_vs_deployed_relative_squared_max"],
                    capture["factors"]["deployed_relative_squared"])
            for state_index, capture in enumerate((absent, score, payload)):
                gain = capture["factors"]["gain"].double().cpu()
                document_rows = torch.arange(start, stop)
                for quarter, (left, right) in enumerate(DOC_QUARTERS):
                    chosen = gain[(document_rows >= left) & (document_rows < right)]
                    if not chosen.numel():
                        continue
                    row = stats["gain_stats"][background_index, state_index, quarter]
                    row[0] += chosen.sum()
                    row[1] += chosen.square().sum()
                    row[2] += chosen.numel()
                    row[3] = min(float(row[3]), float(chosen.min()))
                    row[4] = max(float(row[4]), float(chosen.max()))

            absent_write = absent["deployed_write"]
            score_write = score["deployed_write"]
            payload_write = payload["deployed_write"]
            quarter_selections = _quarter_selections(copy_mask, start, stop)
            circuit_selections = [] if start < 248 else circuit_parent._batch_selections(
                circuit_masks, tags, start, stop, 374)
            gradient_jobs = [("copy", item) for item in quarter_selections]
            if selected_pairs is not None:
                gradient_jobs += [("circuit", item) for item in circuit_selections]
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none") \
                .view(len(batch_rows), TOKENS)

            for quarter, selected_cpu in quarter_selections:
                selected = selected_cpu.to(device)
                r, h, _p = _accumulate_complete(
                    stats, background_index, quarter, selected, native["deployed_write"],
                    absent_write, score_write, payload_write)
                down_ref = _linear(
                    torch.where(selected[..., None],
                                absent_write.float() - native["deployed_write"].float(),
                                torch.zeros_like(absent_write.float())),
                    mlp9.Down.weight.float().T)
                absent_ref = _unordered_contraction(down_ref, absent["factors"])
                score_ref = _unordered_contraction(down_ref, score["factors"])
                payload_ref = _unordered_contraction(down_ref, payload["factors"])
                stats["pair_response_num"][background_index, quarter] += absent_ref - score_ref
                stats["pair_payload_num"][background_index, quarter] += absent_ref - payload_ref
                numerical = absent["factors"]["numerical_output"] \
                    - score["factors"]["numerical_output"]
                diagnostics["numerical_response2"][background_index, quarter] += float(
                    numerical[selected].double().square().sum())
                diagnostics["complete_response2"][background_index, quarter] += float(
                    h.double().square().sum())

            group_outputs = None
            if selected_pairs is not None:
                group_outputs = {
                    "absent": _group_output(mlp9, absent["factors"], selected_pairs),
                    "score": _group_output(mlp9, score["factors"], selected_pairs),
                    "payload": _group_output(mlp9, payload["factors"], selected_pairs),
                }
                group_response = group_outputs["absent"] - group_outputs["score"]
                payload_group_response = group_outputs["absent"] - group_outputs["payload"]
                full_response = absent_write.float() - score_write.float()
                reference = absent_write.float() - native["deployed_write"].float()
                for quarter, selected_cpu in quarter_selections:
                    if quarter < 2:
                        continue
                    selected = selected_cpu.to(device)
                    group, full, ref = (group_response[selected].double(),
                                        full_response[selected].double(),
                                        reference[selected].double())
                    pg = payload_group_response[selected].double()
                    row = stats["confirmation_local"][background_index, quarter - 2]
                    row[0] += float(group.square().sum())
                    row[1] += float(full.square().sum())
                    row[2] += float((group * full).sum())
                    row[3] += float((group * ref).sum())
                    row[4] += float((full * ref).sum())
                    row[5] += float((pg * ref).sum())
                    row[6] += float(ref.square().sum())
                    row[9] += float(pg.square().sum())
                    row[10] += float(group.square().sum())
                    row[11] += int(selected.sum())

            for job_index, (kind, job) in enumerate(gradient_jobs):
                if kind == "copy":
                    quarter, selected_cpu = job
                    selected = selected_cpu.to(device)
                else:
                    half, mask_index, tag_index, selected_cpu = job
                    selected = selected_cpu.to(device)
                gradient = torch.autograd.grad(
                    nll[selected].sum(), absent["leaf"],
                    retain_graph=job_index + 1 < len(gradient_jobs), allow_unused=False)[0]
                down_gradient = _linear(gradient.float(), mlp9.Down.weight.float().T)
                if kind == "copy":
                    absent_grad = _unordered_contraction(down_gradient, absent["factors"])
                    score_grad = _unordered_contraction(down_gradient, score["factors"])
                    stats["pair_gradient_num"][background_index, quarter] += \
                        absent_grad - score_grad
                    complete = float((gradient.float() *
                                      (absent_write.float() - score_write.float())).sum())
                    stats["denominators"][background_index, quarter, 5] += complete
                    stats["denominators"][background_index, quarter, 7] += 1
                    calls["copy_backwards"] += 1
                    if selected_pairs is not None and quarter >= 2:
                        group_response = group_outputs["absent"] - group_outputs["score"]
                        row = stats["confirmation_local"][background_index, quarter - 2]
                        row[7] += float((gradient.float() * group_response).sum())
                        row[8] += complete
                else:
                    calls["circuit_backwards"] += 1
                    group_response = group_outputs["absent"] - group_outputs["score"]
                    payload_group_response = group_outputs["absent"] - group_outputs["payload"]
                    full_response = absent_write.float() - score_write.float()
                    values = [
                        float((gradient.float() * full_response).sum()),
                        float((gradient.float() * group_response).sum()),
                        float((gradient.float() * payload_group_response).sum()),
                    ]
                    values.extend(float((gradient.float() * torch.roll(
                        group_response, shift, dims=1)).sum()) for shift in POSITION_SHIFTS)
                    stats["circuit_sums"][half, background_index, mask_index, tag_index] += \
                        torch.tensor(values, dtype=torch.float64)
                    if background_index == 0:
                        stats["circuit_counts"][half, mask_index, tag_index] += int(selected.sum())
            del logits, nll, absent, score, payload, group_outputs

    if selected_pairs is None:
        selected_pairs, selection_detail = _select_pairs(stats)
    return stats, diagnostics, calls, selected_pairs, selection_detail


def _complete_report(stats, background, quarter):
    row = stats["denominators"][background, quarter]
    score = _cosine_residual(float(row[1]), float(row[0]), float(row[2]))
    payload = _cosine_residual(float(row[3]), float(row[0]), float(row[4]))
    score["tokens"] = int(row[6])
    payload["tokens"] = int(row[6])
    return {"score": score, "payload": payload}


def _confirmation_report(stats, selected_pairs):
    reports = []
    pair_signs = []
    for background in range(2):
        background_rows = []
        signs = []
        for half in range(2):
            quarter = half + 2
            row = stats["confirmation_local"][background, half]
            shape = _cosine_residual(float(row[0]), float(row[1]), float(row[2]))
            response_fraction = float(row[3] / max(abs(float(row[4])), 1e-30))
            gradient_fraction = float(row[7] / (float(row[8]) if abs(float(row[8])) > 1e-30 else 1e-30))
            payload_ratio = abs(float(row[5])) / max(abs(float(row[3])), 1e-30)
            background_rows.append({
                **shape, "response_fraction_of_complete": response_fraction,
                "gradient_fraction_of_complete": gradient_fraction,
                "payload_to_score_response_fraction": payload_ratio,
                "tokens": int(row[11]),
            })
            signs.append({
                PAIR_NAMES[pair]: bool(
                    stats["pair_response_num"][background, quarter, pair] > 0
                    and stats["pair_gradient_num"][background, quarter, pair] > 0)
                for pair in selected_pairs})
        reports.append(background_rows)
        pair_signs.append(signs)
    holds = bool(selected_pairs and len(selected_pairs) <= 32 and all(
        row["cosine"] >= .75 and row["positive_scale_residual"] <= .70
        and row["response_fraction_of_complete"] >= .50
        and .50 <= row["gradient_fraction_of_complete"] <= 1.50
        and row["payload_to_score_response_fraction"] <= .50
        for rows in reports for row in rows)
        and all(all(values.values()) for rows in pair_signs for values in rows))
    return holds, {"backgrounds": reports, "selected_pair_signs": pair_signs}


def _circuit_report(stats, tags):
    sums = stats["circuit_sums"]
    counts = stats["circuit_counts"]
    supported = [index for index in range(len(tags)) if bool((counts[:, :, index] > 0).all())]
    unsupported = [tags[index] for index in range(len(tags)) if index not in supported]
    reports = []
    if supported:
        means = sums / counts[:, None, :, :, None].clamp_min(1)
        fingerprints = means[:, :, 0] - means[:, :, 1]
        for half in range(2):
            half_rows = []
            for background in range(2):
                bank = fingerprints[half, background, supported]
                full, group, payload = bank[:, 0], bank[:, 1], bank[:, 2]
                same = _cosine_residual(float(group.square().sum()),
                                        float(full.square().sum()), float((group * full).sum()))
                payload_cos = float((payload * full).sum() /
                                    (payload.square().sum() * full.square().sum()).sqrt().clamp_min(1e-30))
                controls = []
                for shift_index in range(16):
                    shifted = bank[:, 3 + shift_index]
                    controls.append(float((shifted * full).sum() /
                                          (shifted.square().sum() * full.square().sum()).sqrt().clamp_min(1e-30)))
                norm_ratio = float(group.square().sum().sqrt()
                                   / full.square().sum().sqrt().clamp_min(1e-30))
                half_rows.append({
                    **same, "group_to_complete_norm": norm_ratio,
                    "payload_cosine": payload_cos,
                    "position_shift_cosines": controls,
                    "position_shift_q95": _quantile95(controls),
                    "position_margin": same["cosine"] - _quantile95(controls),
                    "payload_margin": same["cosine"] - payload_cos,
                })
            reports.append(half_rows)
    holds = bool(supported and all(
        row["cosine"] >= .75 and row["positive_scale_residual"] <= .70
        and row["group_to_complete_norm"] >= .25
        and row["position_margin"] >= .10 and row["payload_margin"] >= .20
        for rows in reports for row in rows))
    return holds, {"supported_tags": [tags[index] for index in supported],
                   "unsupported_tags": unsupported, "halves": reports}


def _gain_report(stats):
    report = []
    for background in range(2):
        background_rows = []
        for quarter in range(4):
            states = {}
            for state_index, state in enumerate(STATES):
                row = stats["gain_stats"][background, state_index, quarter]
                count = max(float(row[2]), 1.0)
                mean = float(row[0] / count)
                variance = max(float(row[1] / count) - mean * mean, 0.0)
                states[state] = {"mean": mean, "std": math.sqrt(variance),
                                 "min": float(row[3]), "max": float(row[4]),
                                 "values": int(row[2])}
            absent_mean = states["late_absent"]["mean"]
            states["score_vs_absent_relative_mean_drift"] = abs(
                states["score_donor"]["mean"] - absent_mean) / max(abs(absent_mean), 1e-30)
            background_rows.append(states)
        report.append(background_rows)
    return report


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(SOURCES) == 20 and len(SOURCE_PAIRS) == 210
        assert len(set(SOURCE_PAIRS)) == 210 and SOURCE_PAIRS[0] == (0, 0)
        assert SOURCE_PAIRS[-1] == (19, 19) and KNOWN_PAIR == (0, 3)
        tiny_left = torch.randn(2, 3, 20, 5)
        tiny_right = torch.randn(2, 3, 20, 5)
        total = torch.zeros(2, 3, 5)
        for left, right in SOURCE_PAIRS:
            total += tiny_left[:, :, left] * tiny_right[:, :, right]
            if left != right:
                total += tiny_left[:, :, right] * tiny_right[:, :, left]
        torch.testing.assert_close(total, tiny_left.sum(2) * tiny_right.sum(2))
        print(json.dumps({"status": "dry_run_passed", "rung": 502,
                          "model_loaded": False, "source_pair_outcomes_opened": False,
                          "sources": list(SOURCES), "pair_count": len(SOURCE_PAIRS),
                          "forwards": 875, "validation_documents_or_tags_opened": False}, indent=2))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung502 namespace already exists")
    rows, circuit_masks, tags, metadata = validate_inputs()
    parent_receipt = json.loads(PARENT_RESULT.read_text())
    scales = parent_receipt["frozen_scales"][parent.KNOWN_POSITIVE]
    expected_copy_backwards = 0
    copy_mask = parent._task_masks(rows)["copy_positive"]
    for start in range(0, 500, BATCH):
        expected_copy_backwards += len(_quarter_selections(copy_mask, start, min(start + BATCH, 500)))
    expected_copy_backwards *= 2
    expected_circuit_backwards = 0
    for start in range(248, 500, BATCH):
        stop = min(start + BATCH, 500)
        for _half, _kind, _tag, selected in circuit_parent._batch_selections(
                circuit_masks, tags, start, stop, 374):
            expected_circuit_backwards += int(bool(selected.any()))
    expected_circuit_backwards *= 2

    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stats, diagnostics, calls, selected_pairs, selection_detail = collect(
        model, rows, circuit_masks, tags, scales)
    calls_expected = {"native": 125, "actions": 750,
                      "copy_backwards": expected_copy_backwards,
                      "circuit_backwards": expected_circuit_backwards,
                      "source_captures": 750,
                      "native_attention": 125 * 18 + 750 * 15,
                      "replayed_attention": 750 * 3,
                      "native_mlp": 875 * 18,
                      "mlp9_leaves": 250}
    calls_exact = calls == calls_expected
    numerical_ratios = torch.sqrt(
        diagnostics["numerical_response2"]
        / diagnostics["complete_response2"].clamp_min(1e-30))
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256 and calls_exact
        and diagnostics["state_source_relative_squared_max"] <= 1e-12
        and diagnostics["float32_pair_closure_relative_squared_max"] <= 1e-8
        and diagnostics["float32_vs_deployed_relative_squared_max"] <= DEPLOYED_BF16_BAR
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["minimum_nonzero_edit_rms"] > 0
        and bool((numerical_ratios < .02).all()))
    parent_reports = [[_complete_report(stats, background, quarter)
                       for quarter in range(4)] for background in range(2)]
    pred_b = bool(all(
        row["score"]["cosine"] >= .75
        and row["score"]["positive_scale_residual"] <= .70
        and (row["score"]["cosine"] >= row["payload"]["cosine"] + .30
             or row["score"]["positive_scale_residual"]
             <= row["payload"]["positive_scale_residual"] - .30)
        for rows_ in parent_reports for row in rows_))
    pred_c, confirmation = _confirmation_report(stats, selected_pairs)
    pred_d, circuits = _circuit_report(stats, tags)
    pred_e = bool(pred_a and pred_b and pred_c and pred_d)
    strong_null = bool(not pred_a or not pred_b or not pred_c or not pred_d)
    serial_diagnostics = dict(diagnostics)
    serial_diagnostics["numerical_response_rms_over_complete"] = numerical_ratios.tolist()
    serial_diagnostics.pop("numerical_response2")
    serial_diagnostics.pop("complete_response2")
    bundle = {
        "schema": "mlp9_score_response_source_pair_atlas_rung502_stats_v1",
        "pair_response_num": stats["pair_response_num"],
        "pair_payload_num": stats["pair_payload_num"],
        "pair_gradient_num": stats["pair_gradient_num"],
        "denominators": stats["denominators"],
        "confirmation_local": stats["confirmation_local"],
        "circuit_sums": stats["circuit_sums"],
        "circuit_counts": stats["circuit_counts"],
        "gain_stats": stats["gain_stats"],
        "raw_tokens_logits_gradients_or_pair_vectors_included": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 502,
        "claim_level": "exact_local_source_pair_atlas_and_gradient_screen_not_circuit",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "sources": list(SOURCES),
        "source_pairs": list(PAIR_NAMES), "backgrounds": list(BACKGROUNDS),
        "document_quarters": [list(x) for x in DOC_QUARTERS],
        "selection": {"selected_indices": selected_pairs,
                      "selected_pairs": [PAIR_NAMES[i] for i in selected_pairs],
                      "count": len(selected_pairs), "details": selection_detail},
        "parent_response": parent_reports,
        "confirmation": confirmation, "circuit_fingerprints": circuits,
        "normalization_gain": _gain_report(stats),
        "instrument": {**serial_diagnostics, "calls": calls,
                       "expected_calls": calls_expected, "calls_exact": calls_exact},
        'pred_a_exact_live_source_pair_instrument': pred_a,
        'pred_b_known_mlp9_parent_response_retained': pred_b,
        'pred_c_compact_stable_source_pair_group': pred_c,
        'pred_d_downstream_circuit_use_confirms_group': pred_d,
        'pred_e_source_pair_candidate_for_finite_intervention': pred_e,
        "strong_null": strong_null,
        "validation_documents_or_tags_opened": False,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {"model_forwards": calls["native"] + calls["actions"],
                            "backwards": calls["copy_backwards"] + calls["circuit_backwards"],
                            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                            "deployed_parameters_added": 0, "deployed_parameters_saved": 0},
        "runtime_s": time.time() - started,
        "next_step": (
            "repair_source_pair_instrument_only" if not pred_a else
            "retire_source_capture_parent_mismatch" if not pred_b else
            "refine_largest_stable_source_families_without_rank" if not pred_c else
            "change_downstream_observation_before_circuit_claim" if not pred_d else
            "preregister_finite_mlp9_source_pair_group_removal"),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 502,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "selected_pairs": result["selection"]["selected_pairs"],
        "execution_price": result["execution_price"], "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
