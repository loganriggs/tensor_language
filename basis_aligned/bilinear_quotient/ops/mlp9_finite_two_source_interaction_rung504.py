#!/usr/bin/env python3
"""RUNG504 -- finite two-source interactions through MLP9 and the real suffix."""

# BQGATE: EXPERIMENT
# pred_a: exact finite MLP9-plus-suffix instrument and rung503 parent reproduce
# pred_b: a complete nonempty <=10 two-source interaction set is selected without top-k
# pred_c: the identical complete pair set and every finite effect confirm
# pred_d: every selected pair has copy-selective finite downstream circuit fingerprints
# pred_e: selected pairs are candidates only for separately registered held-out execution

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

import mlp9_attention8_finite_partner_screen_rung503 as parent


PREREG = POLY / "MLP9_FINITE_TWO_SOURCE_INTERACTION_RUNG504_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp9_attention8_finite_partner_screen_rung503.py"
PARENT_RESULT = ROOT / "mlp9_attention8_finite_partner_screen_rung503_results.json"
PARENT_BUNDLE = ROOT / "mlp9_attention8_finite_partner_screen_rung503_bundle.pt"
OUT = ROOT / "mlp9_finite_two_source_interaction_rung504_results.json"
BUNDLE = ROOT / "mlp9_finite_two_source_interaction_rung504_bundle.pt"
HASHES = {
    PREREG: "50d42c89a8093491334e1631dcae837dfcb16bf109f89b4ffc0dd1ee7f3365d7",
    PARENT_SOURCE: "dd792fa67be3b8a14b8f552b356d0cac1bd424da3c9899701f85015922413e17",
    PARENT_RESULT: "b320e9706e1de230620a4da98b2c2a4e9e2b811bb7db577849e46a634b94f966",
    PARENT_BUNDLE: "7c59452743345eb29a32bebc30d1a51ae69e2dd8bf92674b98e1b26e33203c3f",
}
PARTNERS = parent.PARTNERS
PAIR_INDICES = tuple(itertools.combinations(range(len(PARTNERS)), 2))
PAIR_NAMES = tuple(f"{PARTNERS[left]}+{PARTNERS[right]}" for left, right in PAIR_INDICES)
SOURCE_SETS = tuple((index,) for index in range(len(PARTNERS))) + PAIR_INDICES
SINGLETON_COUNT = len(PARTNERS)
PAIR_COUNT = len(PAIR_INDICES)
DISCOVERY_BATCHES = 62
CONFIRMATION_BATCHES = 63
ORDINARY_EVALUATIONS_SELECTION = DISCOVERY_BATCHES * 2 * 3 * (1 + len(SOURCE_SETS))
ORDINARY_EVALUATIONS_TOTAL = (
    DISCOVERY_BATCHES + CONFIRMATION_BATCHES) * 2 * 3 * (1 + len(SOURCE_SETS))
POSITION_EVALUATIONS_PER_SELECTED_PAIR = CONFIRMATION_BATCHES * 2 * 2 * 16
BACKGROUNDS = parent.BACKGROUNDS
STATES = parent.STATES
DOC_QUARTERS = parent.DOC_QUARTERS
POSITION_SHIFTS = parent.POSITION_SHIFTS
BATCH = parent.BATCH
D = parent.D
TOKENS = parent.TOKENS
CHUNK = 16
facade = parent.facade
action_parent = parent.action_parent
circuit_parent = parent.circuit_parent
first = parent.first


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
    receipt = json.loads(PARENT_RESULT.read_text())
    required = {
        "status": "complete",
        "rung": 503,
        "pred_a_finite_source_instrument_and_parent_valid": True,
        "pred_b_compact_partner_set_selected": False,
        "pred_c_partner_identity_and_group_confirm": False,
        "pred_d_selective_downstream_use_confirms": False,
        "pred_e_candidate_for_suffix_intervention": False,
        "strong_null": True,
        "next_step": "finite_pair_removal_screen_or_float32_control",
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise RuntimeError("rung503 does not license rung504")
    rows, circuit_masks, tags, metadata = parent.validate_inputs()
    if PARTNERS != parent.PARTNERS or len(PAIR_INDICES) != 153 \
            or len(set(PAIR_INDICES)) != 153 or len(SOURCE_SETS) != 171:
        raise RuntimeError("rung504 pair vocabulary changed")
    return rows, circuit_masks, tags, {
        "parent": metadata,
        "rung503_pair_outcomes_loaded_for_selection": False,
        "validation_documents_or_tags_opened": False,
    }


def finite_effect(native_absent, native_other, removed_absent, removed_other):
    """Return complete state difference and the part lost under every removal."""
    complete = native_absent - native_other
    after_removal = removed_absent - removed_other
    contribution = complete.unsqueeze(0) - after_removal
    return complete, contribution


def finite_mixed(pair_contribution, singleton_contribution):
    """Exact inclusion--exclusion interaction for the frozen unordered pairs."""
    if pair_contribution.shape[0] != PAIR_COUNT \
            or singleton_contribution.shape[0] != SINGLETON_COUNT:
        raise ValueError("candidate leading dimension changed")
    left = torch.tensor([pair[0] for pair in PAIR_INDICES], device=pair_contribution.device)
    right = torch.tensor([pair[1] for pair in PAIR_INDICES], device=pair_contribution.device)
    return pair_contribution - singleton_contribution[left] - singleton_contribution[right]


def source_sums(partner_sources, source_sets=SOURCE_SETS, *, shift=0):
    """Construct the exact raw contribution removed for each registered source set."""
    if partner_sources.shape[2] != SINGLETON_COUNT:
        raise ValueError("partner-source axis changed")
    values = [partner_sources[:, :, indices].sum(2) for indices in source_sets]
    stacked = torch.stack(values, dim=0)
    return torch.roll(stacked, shift, dims=2) if shift else stacked


def split_candidates(values):
    if values.shape[0] != len(SOURCE_SETS):
        raise ValueError("candidate leading dimension changed")
    return values[:SINGLETON_COUNT], values[SINGLETON_COUNT:]


def expected_price(selected_pair_count: int, confirmation_opened: bool):
    if selected_pair_count < 0 or selected_pair_count > PAIR_COUNT:
        raise ValueError("selected pair count is outside the frozen pair vocabulary")
    if not confirmation_opened:
        return {
            "full_model_forwards": 496,
            "mlp9_plus_suffix_evaluations": ORDINARY_EVALUATIONS_SELECTION,
            "backwards": 0,
        }
    if selected_pair_count > 10:
        raise ValueError("diffuse selection cannot open confirmation")
    return {
        "full_model_forwards": 1000,
        "mlp9_plus_suffix_evaluations": (
            ORDINARY_EVALUATIONS_TOTAL
            + POSITION_EVALUATIONS_PER_SELECTED_PAIR * selected_pair_count),
        "backwards": 0,
    }


@torch.no_grad()
def _forward_capture(model, tokens, scales, *, direct=False,
                     background="early_present", state="late_native"):
    """Run the registered action and retain exactly the state needed for a suffix edit."""
    facade.validate_production_model(model)
    facade.validate_tokens(tokens, production_shape=True)
    if background not in BACKGROUNDS or state not in (*STATES, "late_native"):
        raise ValueError("unregistered action")
    cached = {}
    attention_writes = []
    prior_writes = []
    capture = {}
    diagnostics = {
        "factor_reconstruction_max": 0.0,
        "early_edit_rms": 0.0,
        "late_edit_rms": 0.0,
        "raw_round_rms_over_raw": 0.0,
    }
    audit = {"native_attention": 0, "replayed_attention": 0, "native_mlp": 0}

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
            donor, recipient = parent.KNOWN_PAIR
            if site == action_parent.factor_parent.TERMS[donor][1]:
                cached.update(terms[donor])
                if background == "early_absent":
                    edit = terms[donor]["native_term"]
                    write = write - edit
                    diagnostics["early_edit_rms"] = float(
                        edit.float().square().mean().sqrt())
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
                            target["p"] * support,
                            cached["u"] * scales["payload_ratio"])
                    edit = replacement.to(write.dtype) - target["native_term"]
                    write = write + edit
                    diagnostics["late_edit_rms"] = float(
                        edit.float().square().mean().sqrt())
        attention_writes.append(write.detach())
        x = x + write
        raw_mlp_state = x
        mlp_write = block.mlp(F.rms_norm(x, (D,)))
        audit["native_mlp"] += 1
        if site == 9:
            raw, partners, ratio = parent._raw_partner_sources(
                model, x0.detach(), attention_writes, prior_writes,
                raw_mlp_state.detach())
            capture = {
                "raw_state": raw,
                "partner_sources": partners,
                "deployed_write": mlp_write.detach(),
                "x0": x0.detach(),
                "first_value": v1.detach(),
            }
            diagnostics["raw_round_rms_over_raw"] = ratio
        prior_writes.append(mlp_write.detach())
        x = x + mlp_write
    logits = (30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0)).float()
    expected = ({"native_attention": 18, "replayed_attention": 0, "native_mlp": 18}
                if direct else
                {"native_attention": 15, "replayed_attention": 3, "native_mlp": 18})
    expected_capture = {
        "raw_state", "partner_sources", "deployed_write", "x0", "first_value"}
    if audit != expected or set(capture) != expected_capture:
        raise RuntimeError(f"rung504 forward audit failed: {audit}, {set(capture)}")
    if tuple(logits.shape) != (*tokens.shape, facade.LOGIT_VOCAB):
        raise RuntimeError("rung504 forward shape changed")
    return logits, capture, diagnostics, audit


def _repeat_candidates(value, count):
    return value.unsqueeze(0).expand(count, *value.shape).reshape(
        count * value.shape[0], *value.shape[1:])


@torch.no_grad()
def _suffix_candidates(model, capture, targets, source_sets, *, shift=0):
    """Evaluate registered removals through MLP9 and layers10--17 in bounded chunks."""
    writes = []
    losses = []
    minimum_input_edit_rms = float("inf")
    gpu_calls = 0
    mlp9 = model.transformer.h[9].mlp
    for start in range(0, len(source_sets), CHUNK):
        sets = source_sets[start:start + CHUNK]
        sums = source_sums(capture["partner_sources"], sets, shift=shift)
        raw = capture["raw_state"].unsqueeze(0).expand(len(sets), *capture["raw_state"].shape)
        edited = (raw.float() - sums).to(capture["raw_state"].dtype)
        edit_rms = (edited.float() - raw.float()).double().square().mean((1, 2, 3)).sqrt()
        live = edit_rms[edit_rms > 0]
        if live.numel():
            minimum_input_edit_rms = min(minimum_input_edit_rms, float(live.min()))
        flat = edited.reshape(-1, *edited.shape[2:])
        changed_write = mlp9(F.rms_norm(flat, (D,)))
        x = flat + changed_write
        x0 = _repeat_candidates(capture["x0"], len(sets))
        first_value = _repeat_candidates(capture["first_value"], len(sets))
        for block in model.transformer.h[10:]:
            x = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, first_value = block.attn(F.rms_norm(x, (D,)), first_value)
            x = x + attention
            x = x + block.mlp(F.rms_norm(x, (D,)))
        logits = (30.0 * torch.tanh(
            model.lm_head(F.rms_norm(x, (D,))) / 30.0)).float()
        expanded_targets = _repeat_candidates(targets, len(sets))
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), expanded_targets.reshape(-1),
            reduction="none").view(len(sets), targets.shape[0], targets.shape[1])
        writes.append(changed_write.view(
            len(sets), capture["raw_state"].shape[0],
            capture["raw_state"].shape[1], D).detach())
        losses.append(nll.detach())
        gpu_calls += 1
        del sums, raw, edited, flat, x, x0, first_value, logits, expanded_targets, nll
    return (torch.cat(writes), torch.cat(losses), minimum_input_edit_rms,
            {"candidate_evaluations": len(source_sets), "gpu_chunk_calls": gpu_calls})


def _empty_stats(tag_count):
    return {
        # local complete response squared norm; finite copy benefit sum; copy tokens
        "denominators": torch.zeros(2, 4, 3, dtype=torch.float64),
        # singleton response/payload projections, used only for rung503 reproduction
        "singleton_local": torch.zeros(2, 4, SINGLETON_COUNT, 2, dtype=torch.float64),
        # group2, group_cross, mixed2, mixed_cross, payload_group_cross,
        # payload_mixed_cross
        "pair_local": torch.zeros(2, 4, PAIR_COUNT, 6, dtype=torch.float64),
        # finite score group, score mixed, payload group, payload mixed
        "pair_loss": torch.zeros(2, 4, PAIR_COUNT, 4, dtype=torch.float64),
        "parent": parent._empty_stats([f"tag{index}" for index in range(tag_count)]),
        "circuit_sums": None,
        "circuit_counts": None,
    }


def _allocate_circuit_stats(stats, selected_count, tag_count):
    # complete, pair, mixed, payload, and 16 shifted pair controls
    stats["circuit_sums"] = torch.zeros(
        2, 2, 2, tag_count, selected_count, 20, dtype=torch.float64)
    stats["circuit_counts"] = torch.zeros(2, 2, tag_count, dtype=torch.float64)


def _candidate_effects(
        absent_write, score_write, payload_write,
        absent_nll, score_nll, payload_nll,
        absent_removed_write, score_removed_write, payload_removed_write,
        absent_removed_nll, score_removed_nll, payload_removed_nll):
    """Construct all finite local and loss effects before applying an observation mask."""
    delta, score_contribution = finite_effect(
        absent_write.float(), score_write.float(),
        absent_removed_write.float(), score_removed_write.float())
    _, payload_contribution = finite_effect(
        absent_write.float(), payload_write.float(),
        absent_removed_write.float(), payload_removed_write.float())
    score_single, score_pair = split_candidates(score_contribution)
    payload_single, payload_pair = split_candidates(payload_contribution)
    score_mixed = finite_mixed(score_pair, score_single)
    payload_mixed = finite_mixed(payload_pair, payload_single)

    benefit, score_loss_contribution = finite_effect(
        absent_nll.float(), score_nll.float(),
        absent_removed_nll.float(), score_removed_nll.float())
    _, payload_loss_contribution = finite_effect(
        absent_nll.float(), payload_nll.float(),
        absent_removed_nll.float(), payload_removed_nll.float())
    score_loss_single, score_loss_pair = split_candidates(score_loss_contribution)
    payload_loss_single, payload_loss_pair = split_candidates(payload_loss_contribution)
    score_loss_mixed = finite_mixed(score_loss_pair, score_loss_single)
    payload_loss_mixed = finite_mixed(payload_loss_pair, payload_loss_single)

    return {
        "delta": delta,
        "score_single": score_single,
        "payload_single": payload_single,
        "score_pair": score_pair,
        "score_mixed": score_mixed,
        "payload_pair": payload_pair,
        "payload_mixed": payload_mixed,
        "complete_benefit": benefit,
        "pair_benefit": score_loss_pair,
        "mixed_benefit": score_loss_mixed,
        "payload_pair_benefit": payload_loss_pair,
        "payload_mixed_benefit": payload_loss_mixed,
        "absent_write": absent_write,
        "score_write": score_write,
        "payload_write": payload_write,
    }


def _accumulate_candidate_statistics(
        stats, background, quarter, chosen, native_write, effects):
    """Accumulate the already-computed effects on one frozen copy-task slice."""
    delta = effects["delta"]
    score_single = effects["score_single"]
    payload_single = effects["payload_single"]
    score_pair = effects["score_pair"]
    score_mixed = effects["score_mixed"]
    payload_pair = effects["payload_pair"]
    payload_mixed = effects["payload_mixed"]
    benefit = effects["complete_benefit"]
    score_loss_pair = effects["pair_benefit"]
    score_loss_mixed = effects["mixed_benefit"]
    payload_loss_pair = effects["payload_pair_benefit"]
    payload_loss_mixed = effects["payload_mixed_benefit"]

    h = delta[chosen].double()
    score_s = score_single[:, chosen].double()
    payload_s = payload_single[:, chosen].double()
    score_p = score_pair[:, chosen].double()
    score_k = score_mixed[:, chosen].double()
    payload_p = payload_pair[:, chosen].double()
    payload_k = payload_mixed[:, chosen].double()
    stats["denominators"][background, quarter, 0] += float(h.square().sum())
    stats["denominators"][background, quarter, 1] += float(
        benefit[chosen].double().sum())
    stats["denominators"][background, quarter, 2] += int(chosen.sum())
    stats["singleton_local"][background, quarter, :, 0] += (
        score_s * h.unsqueeze(0)).sum((1, 2)).cpu()
    stats["singleton_local"][background, quarter, :, 1] += (
        payload_s * h.unsqueeze(0)).sum((1, 2)).cpu()
    local = stats["pair_local"][background, quarter]
    local[:, 0] += score_p.square().sum((1, 2)).cpu()
    local[:, 1] += (score_p * h.unsqueeze(0)).sum((1, 2)).cpu()
    local[:, 2] += score_k.square().sum((1, 2)).cpu()
    local[:, 3] += (score_k * h.unsqueeze(0)).sum((1, 2)).cpu()
    local[:, 4] += (payload_p * h.unsqueeze(0)).sum((1, 2)).cpu()
    local[:, 5] += (payload_k * h.unsqueeze(0)).sum((1, 2)).cpu()
    loss = stats["pair_loss"][background, quarter]
    loss[:, 0] += score_loss_pair[:, chosen].double().sum(1).cpu()
    loss[:, 1] += score_loss_mixed[:, chosen].double().sum(1).cpu()
    loss[:, 2] += payload_loss_pair[:, chosen].double().sum(1).cpu()
    loss[:, 3] += payload_loss_mixed[:, chosen].double().sum(1).cpu()

    first._accumulate_complete(
        stats["parent"], background, quarter, chosen,
        native_write, effects["absent_write"], effects["score_write"],
        effects["payload_write"])


def _pair_report(stats, pair_index, background, quarters):
    quarter_list = list(quarters)
    delta2 = float(stats["denominators"][background, quarter_list, 0].sum())
    benefit = float(stats["denominators"][background, quarter_list, 1].sum())
    local = stats["pair_local"][background, quarter_list, pair_index].sum(0)
    loss = stats["pair_loss"][background, quarter_list, pair_index].sum(0)
    shape = first._cosine_residual(float(local[0]), delta2, float(local[1]))
    response_fraction = float(local[1]) / max(delta2, 1e-30)
    mixed_response_fraction = float(local[3]) / max(delta2, 1e-30)
    benefit_denominator = (benefit if abs(benefit) > 1e-30
                           else math.copysign(1e-30, benefit or 1.0))
    copy_fraction = float(loss[0]) / benefit_denominator
    mixed_copy_fraction = float(loss[1]) / benefit_denominator
    local_payload_group_ratio = abs(float(local[4])) / max(abs(float(local[1])), 1e-30)
    local_payload_mixed_ratio = abs(float(local[5])) / max(abs(float(local[3])), 1e-30)
    loss_payload_group_ratio = abs(float(loss[2])) / max(abs(float(loss[0])), 1e-30)
    loss_payload_mixed_ratio = abs(float(loss[3])) / max(abs(float(loss[1])), 1e-30)
    signs = []
    for quarter in quarters:
        local_q = stats["pair_local"][background, quarter, pair_index]
        loss_q = stats["pair_loss"][background, quarter, pair_index]
        signs.append({
            "local_group": bool(local_q[1] > 0),
            "local_mixed": bool(local_q[3] > 0),
            "finite_copy_group": bool(loss_q[0] > 0),
            "finite_copy_mixed": bool(loss_q[1] > 0),
        })
    holds = bool(
        shape["cosine"] >= .75 and shape["positive_scale_residual"] <= .70
        and .20 <= response_fraction <= 1.50
        and mixed_response_fraction >= .10
        and .20 <= copy_fraction <= 1.50
        and mixed_copy_fraction >= .10
        and all(all(row.values()) for row in signs)
        and local_payload_group_ratio <= .50
        and local_payload_mixed_ratio <= .50
        and loss_payload_group_ratio <= .50
        and loss_payload_mixed_ratio <= .50)
    return {
        **shape,
        "response_fraction": response_fraction,
        "mixed_response_fraction": mixed_response_fraction,
        "finite_copy_fraction": copy_fraction,
        "mixed_finite_copy_fraction": mixed_copy_fraction,
        "local_payload_group_ratio": local_payload_group_ratio,
        "local_payload_mixed_ratio": local_payload_mixed_ratio,
        "finite_payload_group_ratio": loss_payload_group_ratio,
        "finite_payload_mixed_ratio": loss_payload_mixed_ratio,
        "half_signs": signs,
        "holds": holds,
    }


def _select_pairs(stats, quarters):
    selected = []
    details = {}
    for pair_index, name in enumerate(PAIR_NAMES):
        backgrounds = [
            _pair_report(stats, pair_index, background, quarters)
            for background in range(2)]
        holds = all(row["holds"] for row in backgrounds)
        details[name] = {"backgrounds": backgrounds, "selected": holds}
        if holds:
            selected.append(pair_index)
    return selected, details


def _accumulate_circuit_statistics(
        stats, background, selections, selected_pairs, effects):
    for half, mask_index, tag_index, selected_cpu in selections:
        chosen = selected_cpu.to(effects["complete_benefit"].device)
        if background == 0:
            stats["circuit_counts"][half, mask_index, tag_index] += int(chosen.sum())
        for slot, pair_index in enumerate(selected_pairs):
            values = (
                effects["complete_benefit"][chosen],
                effects["pair_benefit"][pair_index, chosen],
                effects["mixed_benefit"][pair_index, chosen],
                effects["payload_pair_benefit"][pair_index, chosen],
            )
            stats["circuit_sums"][half, background, mask_index, tag_index, slot, :4] += \
                torch.tensor([float(value.double().sum()) for value in values])


def _accumulate_shifted_circuit_statistics(
        stats, background, shift_index, selections, shifted_pair_benefit):
    for half, mask_index, tag_index, selected_cpu in selections:
        chosen = selected_cpu.to(shifted_pair_benefit.device)
        values = shifted_pair_benefit[:, chosen].double().sum(1).cpu()
        stats["circuit_sums"][
            half, background, mask_index, tag_index, :, 4 + shift_index] += values


def _cosine(left, right):
    return float((left * right).sum() /
                 (left.square().sum() * right.square().sum()).sqrt().clamp_min(1e-30))


def _circuit_report(stats, tags, selected_pairs):
    sums = stats["circuit_sums"]
    counts = stats["circuit_counts"]
    supported = [index for index in range(len(tags))
                 if bool((counts[:, :, index] > 0).all())]
    unsupported = [tags[index] for index in range(len(tags)) if index not in supported]
    reports = []
    pair_holds = []
    if supported:
        means = sums / counts[:, None, :, :, None, None].clamp_min(1)
        fingerprints = means[:, :, 0] - means[:, :, 1]
        for half in range(2):
            background_rows = []
            for background in range(2):
                pair_rows = []
                for slot, pair_index in enumerate(selected_pairs):
                    bank = fingerprints[half, background, supported, slot]
                    full, group, mixed, payload = (
                        bank[:, 0], bank[:, 1], bank[:, 2], bank[:, 3])
                    group_shape = first._cosine_residual(
                        float(group.square().sum()), float(full.square().sum()),
                        float((group * full).sum()))
                    mixed_cosine = _cosine(mixed, full)
                    group_norm = float(group.square().sum().sqrt()
                                       / full.square().sum().sqrt().clamp_min(1e-30))
                    mixed_norm = float(mixed.square().sum().sqrt()
                                       / full.square().sum().sqrt().clamp_min(1e-30))
                    payload_cosine = _cosine(payload, full)
                    controls = [
                        _cosine(bank[:, 4 + shift], full) for shift in range(16)]
                    q95 = first._quantile95(controls)
                    row = {
                        "pair": PAIR_NAMES[pair_index],
                        **group_shape,
                        "group_to_complete_norm": group_norm,
                        "mixed_cosine": mixed_cosine,
                        "mixed_to_complete_norm": mixed_norm,
                        "payload_cosine": payload_cosine,
                        "position_shift_cosines": controls,
                        "position_shift_q95": q95,
                        "position_margin": group_shape["cosine"] - q95,
                        "payload_margin": group_shape["cosine"] - payload_cosine,
                    }
                    row["holds"] = bool(
                        row["cosine"] >= .75
                        and row["positive_scale_residual"] <= .70
                        and row["group_to_complete_norm"] >= .25
                        and row["payload_margin"] >= .20
                        and row["position_margin"] >= .10
                        and row["mixed_cosine"] >= .65
                        and row["mixed_to_complete_norm"] >= .10)
                    pair_rows.append(row)
                    pair_holds.append(row["holds"])
                background_rows.append(pair_rows)
            reports.append(background_rows)
    holds = bool(len(supported) == len(tags) and pair_holds and all(pair_holds))
    return holds, {
        "supported_tags": [tags[index] for index in supported],
        "unsupported_tags": unsupported,
        "halves": reports,
    }


def _nll(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
        reduction="none").view(targets.shape)


@torch.no_grad()
def collect(model, rows, circuit_masks, tags, scales):
    copy_mask = action_parent._task_masks(rows)["copy_positive"]
    stats = _empty_stats(len(tags))
    diagnostics = {
        "factor_reconstruction_max": 0.0,
        "raw_round_rms_over_raw_max": 0.0,
        "minimum_action_edit_rms": float("inf"),
        "minimum_candidate_input_edit_rms": float("inf"),
        "zero_removal_write_max_abs": 0.0,
        "zero_removal_nll_max_abs": 0.0,
        "background_native_early_present_write_max_abs": 0.0,
        "background_native_early_present_nll_max_abs": 0.0,
    }
    calls = {
        "early_present_native": 0,
        "early_absent_native": 0,
        "actions": 0,
        "candidate_evaluations": 0,
        "position_evaluations": 0,
        "suffix_gpu_chunk_calls": 0,
        "native_attention": 0,
        "replayed_attention": 0,
        "native_mlp": 0,
    }
    selected_pairs = None
    selection_detail = None
    confirmation_selected = None
    confirmation_detail = None
    confirmation_opened = False
    device = next(model.parameters()).device
    for start in range(0, 500, BATCH):
        if start == 248:
            selected_pairs, selection_detail = _select_pairs(stats, (0, 1))
            if not selected_pairs or len(selected_pairs) > 10:
                break
            confirmation_opened = True
            _allocate_circuit_stats(stats, len(selected_pairs), len(tags))
        stop = min(start + BATCH, 500)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        targets = batch_rows[:, 1:].to(device)

        native_present_logits, native_present, diag, audit = _forward_capture(
            model, tokens, scales, direct=True)
        calls["early_present_native"] += 1
        for key in ("native_attention", "replayed_attention", "native_mlp"):
            calls[key] += audit[key]
        diagnostics["factor_reconstruction_max"] = max(
            diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
        native_present_nll = _nll(native_present_logits, targets)

        for background_index, background in enumerate(BACKGROUNDS):
            if background_index == 0:
                native_logits, native, native_nll = (
                    native_present_logits, native_present, native_present_nll)
            else:
                native_logits, native, diag, audit = _forward_capture(
                    model, tokens, scales, background=background, state="late_native")
                calls["early_absent_native"] += 1
                for key in ("native_attention", "replayed_attention", "native_mlp"):
                    calls[key] += audit[key]
                diagnostics["factor_reconstruction_max"] = max(
                    diagnostics["factor_reconstruction_max"],
                    diag["factor_reconstruction_max"])
                native_nll = _nll(native_logits, targets)

            action_captures = {}
            action_nll = {}
            action_outputs = {}
            for state in STATES:
                logits, capture, diag, audit = _forward_capture(
                    model, tokens, scales, background=background, state=state)
                calls["actions"] += 1
                for key in ("native_attention", "replayed_attention", "native_mlp"):
                    calls[key] += audit[key]
                diagnostics["factor_reconstruction_max"] = max(
                    diagnostics["factor_reconstruction_max"],
                    diag["factor_reconstruction_max"])
                diagnostics["raw_round_rms_over_raw_max"] = max(
                    diagnostics["raw_round_rms_over_raw_max"],
                    diag["raw_round_rms_over_raw"])
                if diag["early_edit_rms"] > 0:
                    diagnostics["minimum_action_edit_rms"] = min(
                        diagnostics["minimum_action_edit_rms"], diag["early_edit_rms"])
                if diag["late_edit_rms"] > 0:
                    diagnostics["minimum_action_edit_rms"] = min(
                        diagnostics["minimum_action_edit_rms"], diag["late_edit_rms"])
                baseline_nll = _nll(logits, targets)
                replay_write, replay_nll, _, replay_calls = _suffix_candidates(
                    model, capture, targets, ((),))
                writes, nll, minimum_edit, suffix_calls = _suffix_candidates(
                    model, capture, targets, SOURCE_SETS)
                diagnostics["minimum_candidate_input_edit_rms"] = min(
                    diagnostics["minimum_candidate_input_edit_rms"], minimum_edit)
                diagnostics["zero_removal_write_max_abs"] = max(
                    diagnostics["zero_removal_write_max_abs"],
                    float((replay_write[0] - capture["deployed_write"]).abs().max()))
                diagnostics["zero_removal_nll_max_abs"] = max(
                    diagnostics["zero_removal_nll_max_abs"],
                    float((replay_nll[0] - baseline_nll).abs().max()))
                calls["candidate_evaluations"] += (
                    replay_calls["candidate_evaluations"]
                    + suffix_calls["candidate_evaluations"])
                calls["suffix_gpu_chunk_calls"] += (
                    replay_calls["gpu_chunk_calls"] + suffix_calls["gpu_chunk_calls"])
                action_captures[state] = capture
                action_nll[state] = baseline_nll
                action_outputs[state] = (writes, nll)
                del logits, replay_write, replay_nll, writes, nll

            absent = action_captures["late_absent"]
            score = action_captures["score_donor"]
            payload = action_captures["payload_donor"]
            absent_writes, absent_removed_nll = action_outputs["late_absent"]
            score_writes, score_removed_nll = action_outputs["score_donor"]
            payload_writes, payload_removed_nll = action_outputs["payload_donor"]
            effects = _candidate_effects(
                absent["deployed_write"], score["deployed_write"],
                payload["deployed_write"], action_nll["late_absent"],
                action_nll["score_donor"], action_nll["payload_donor"],
                absent_writes, score_writes, payload_writes,
                absent_removed_nll, score_removed_nll, payload_removed_nll)
            quarter_selections = first._quarter_selections(copy_mask, start, stop)
            for quarter, selected_cpu in quarter_selections:
                chosen = selected_cpu.to(device)
                _accumulate_candidate_statistics(
                    stats, background_index, quarter, chosen,
                    native["deployed_write"], effects)

            if background_index == 0:
                diagnostics["background_native_early_present_write_max_abs"] = max(
                    diagnostics["background_native_early_present_write_max_abs"],
                    float((native["deployed_write"]
                           - native_present["deployed_write"]).abs().max()))
                diagnostics["background_native_early_present_nll_max_abs"] = max(
                    diagnostics["background_native_early_present_nll_max_abs"],
                    float((native_nll - native_present_nll).abs().max()))

            if confirmation_opened:
                circuit_selections = circuit_parent._batch_selections(
                    circuit_masks, tags, start, stop, 374)
                _accumulate_circuit_statistics(
                    stats, background_index, circuit_selections, selected_pairs, effects)
                selected_sets = tuple(PAIR_INDICES[index] for index in selected_pairs)
                for shift_index, shift in enumerate(POSITION_SHIFTS):
                    shifted_losses = {}
                    for state in ("late_absent", "score_donor"):
                        _, shifted_nll, _, suffix_calls = _suffix_candidates(
                            model, action_captures[state], targets, selected_sets,
                            shift=shift)
                        shifted_losses[state] = shifted_nll
                        calls["position_evaluations"] += suffix_calls[
                            "candidate_evaluations"]
                        calls["suffix_gpu_chunk_calls"] += suffix_calls["gpu_chunk_calls"]
                    _, shifted_pair_benefit = finite_effect(
                        action_nll["late_absent"], action_nll["score_donor"],
                        shifted_losses["late_absent"], shifted_losses["score_donor"])
                    _accumulate_shifted_circuit_statistics(
                        stats, background_index, shift_index, circuit_selections,
                        shifted_pair_benefit)
                    del shifted_losses, shifted_pair_benefit

            del action_captures, action_nll, action_outputs, absent_writes
            del score_writes, payload_writes, absent_removed_nll
            del score_removed_nll, payload_removed_nll, effects
            if background_index == 1:
                del native_logits, native, native_nll
        del native_present_logits, native_present, native_present_nll

    if selected_pairs is None:
        selected_pairs, selection_detail = _select_pairs(stats, (0, 1))
    if confirmation_opened:
        confirmation_selected, confirmation_detail = _select_pairs(stats, (2, 3))
    return {
        "stats": stats,
        "diagnostics": diagnostics,
        "calls": calls,
        "selected_pairs": selected_pairs,
        "selection_detail": selection_detail,
        "confirmation_opened": confirmation_opened,
        "confirmation_selected": confirmation_selected,
        "confirmation_detail": confirmation_detail,
    }


def _dry_run():
    validate_inputs()
    torch.manual_seed(504)
    single = torch.randn(SINGLETON_COUNT, 2, 3)
    synergy = torch.randn(PAIR_COUNT, 2, 3)
    pair = torch.stack([
        single[left] + single[right] for left, right in PAIR_INDICES]) + synergy
    torch.testing.assert_close(finite_mixed(pair, single), synergy)
    sources = torch.randn(2, 3, SINGLETON_COUNT, 5)
    sums = source_sums(sources)
    assert tuple(sums.shape) == (171, 2, 3, 5)
    print(json.dumps({
        "status": "dry_run_core_passed",
        "rung": 504,
        "model_loaded": False,
        "pair_outcomes_opened": False,
        "partner_count": SINGLETON_COUNT,
        "pair_count": PAIR_COUNT,
        "selection_price": expected_price(0, False),
        "conditional_price_at_one_pair": expected_price(1, True),
        "real_execution_enabled": True,
    }, indent=2))


@torch.no_grad()
def _gpu_smoke():
    """Exercise real device placement with zero removals; open no pair outcome."""
    rows, _, _, _ = validate_inputs()
    scales = json.loads(parent.PARENT_RESULT.read_text())["frozen_scales"][
        action_parent.KNOWN_POSITIVE]
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    batch_rows = rows[:BATCH]
    tokens = batch_rows[:, :-1].to("cuda")
    targets = batch_rows[:, 1:].to("cuda")
    maximum_write_error = 0.0
    maximum_nll_error = 0.0
    gpu_chunk_calls = 0
    for state in STATES:
        logits, capture, _, _ = _forward_capture(
            model, tokens, scales, background="early_absent", state=state)
        baseline_nll = _nll(logits, targets)
        writes, nll, _, call_info = _suffix_candidates(
            model, capture, targets, tuple(() for _ in range(CHUNK)))
        maximum_write_error = max(
            maximum_write_error,
            float((writes - capture["deployed_write"].unsqueeze(0)).abs().max()))
        maximum_nll_error = max(
            maximum_nll_error,
            float((nll - baseline_nll.unsqueeze(0)).abs().max()))
        gpu_chunk_calls += call_info["gpu_chunk_calls"]
    if maximum_write_error != 0.0 or maximum_nll_error != 0.0 \
            or gpu_chunk_calls != len(STATES):
        raise RuntimeError("rung504 zero-removal GPU smoke failed")
    print(json.dumps({
        "status": "gpu_smoke_passed",
        "rung": 504,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "pair_outcomes_opened": False,
        "zero_removal_candidates_per_state": CHUNK,
        "zero_removal_write_max_abs": maximum_write_error,
        "zero_removal_nll_max_abs": maximum_nll_error,
        "gpu_chunk_calls": gpu_chunk_calls,
    }, indent=2))


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv[1:]:
        _dry_run()
        return
    if os.environ.get("RUNG504_GPU_SMOKE") == "1":
        _gpu_smoke()
        return
    if len(sys.argv) != 1:
        raise SystemExit("only --dry-run is supported")
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung504 namespace already exists")
    rows, circuit_masks, tags, metadata = validate_inputs()
    parent_receipt = json.loads(PARENT_RESULT.read_text())
    scales = json.loads(parent.PARENT_RESULT.read_text())["frozen_scales"][
        action_parent.KNOWN_POSITIVE]
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    collected = collect(model, rows, circuit_masks, tags, scales)
    stats = collected["stats"]
    diagnostics = collected["diagnostics"]
    calls = collected["calls"]
    selected = collected["selected_pairs"]
    confirmation_opened = collected["confirmation_opened"]
    confirmation_selected = collected["confirmation_selected"]
    selected_count = len(selected)
    batch_count = DISCOVERY_BATCHES + (CONFIRMATION_BATCHES if confirmation_opened else 0)
    expected_calls = {
        "early_present_native": batch_count,
        "early_absent_native": batch_count,
        "actions": batch_count * 6,
        "candidate_evaluations": batch_count * 2 * 3 * (1 + len(SOURCE_SETS)),
        "position_evaluations": (
            CONFIRMATION_BATCHES * 2 * 2 * len(POSITION_SHIFTS) * selected_count
            if confirmation_opened else 0),
        "suffix_gpu_chunk_calls": (
            batch_count * 2 * 3 * (1 + math.ceil(len(SOURCE_SETS) / CHUNK))
            + (CONFIRMATION_BATCHES * 2 * 2 * len(POSITION_SHIFTS)
               * math.ceil(selected_count / CHUNK) if confirmation_opened else 0)),
        "native_attention": batch_count * 123,
        "replayed_attention": batch_count * 21,
        "native_mlp": batch_count * 144,
    }
    calls_exact = calls == expected_calls

    parent_bundle = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=True)
    parent_stats = parent_bundle["stats"]
    singleton_response_reproduces = torch.allclose(
        stats["singleton_local"][:, :2, :, 0],
        parent_stats["pair_response_num"][:, :2], rtol=1e-6, atol=1e-8)
    singleton_payload_reproduces = torch.allclose(
        stats["singleton_local"][:, :2, :, 1],
        parent_stats["pair_payload_num"][:, :2], rtol=1e-6, atol=1e-8)
    parent_denominators_reproduce = torch.allclose(
        stats["parent"]["denominators"][:, :2, :5],
        parent_stats["denominators"][:, :2, :5], rtol=1e-6, atol=1e-8)
    opened_quarters = range(4) if confirmation_opened else range(2)
    parent_reports = parent._parent_reports(stats["parent"], opened_quarters)
    parent_report_differences = []
    for background in range(2):
        for quarter in range(2):
            for action in ("score", "payload"):
                observed = parent_reports[background][quarter][action]["cosine"]
                expected = parent_receipt["parent_response"][background][quarter][action][
                    "cosine"]
                parent_report_differences.append({
                    "background": BACKGROUNDS[background],
                    "quarter": quarter,
                    "action": action,
                    "observed": observed,
                    "rung503": expected,
                    "absolute_difference": abs(observed - expected),
                })

    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and calls_exact
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["raw_round_rms_over_raw_max"] <= .03125
        and diagnostics["minimum_action_edit_rms"] > 0
        and diagnostics["minimum_candidate_input_edit_rms"] > 0
        and diagnostics["zero_removal_write_max_abs"] == 0.0
        and diagnostics["zero_removal_nll_max_abs"] == 0.0
        and diagnostics["background_native_early_present_write_max_abs"] == 0.0
        and diagnostics["background_native_early_present_nll_max_abs"] == 0.0
        and singleton_response_reproduces
        and singleton_payload_reproduces
        and parent_denominators_reproduce
        and max(row["absolute_difference"] for row in parent_report_differences) <= .01)
    pred_b = bool(selected and len(selected) <= 10)
    same_complete_set = bool(
        confirmation_opened and confirmation_selected == selected)
    pred_c = bool(same_complete_set and all(
        row["holds"]
        for detail in collected["confirmation_detail"].values()
        if detail["selected"]
        for row in detail["backgrounds"]))
    if confirmation_opened:
        circuit_holds, circuit_report = _circuit_report(stats, tags, selected)
    else:
        circuit_holds, circuit_report = False, None
    pred_d = bool(pred_c and circuit_holds)
    pred_e = bool(pred_a and pred_b and pred_c and pred_d)
    strong_null = bool(not pred_a or not pred_b or not pred_c or not pred_d)

    stats_to_save = {key: value for key, value in stats.items() if key != "parent"}
    stats_to_save["parent_denominators"] = stats["parent"]["denominators"]
    torch.save({
        "schema": "mlp9_finite_two_source_interaction_rung504_stats_v1",
        "stats": stats_to_save,
        "raw_tokens_logits_gradients_or_per_token_vectors_included": False,
    }, BUNDLE)
    result = {
        "status": "complete",
        "rung": 504,
        "claim_level": "finite_two_source_mlp9_suffix_screen_not_circuit",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "partner_vocabulary": list(PARTNERS),
        "pair_vocabulary": list(PAIR_NAMES),
        "excluded_from_partner_semantics": ["E", "A8"],
        "document_quarters": [list(value) for value in DOC_QUARTERS],
        "confirmation_opened": confirmation_opened,
        "selection": {
            "pairs": [PAIR_NAMES[index] for index in selected],
            "details": collected["selection_detail"],
        },
        "confirmation_reselection": {
            "pairs": (None if confirmation_selected is None else
                      [PAIR_NAMES[index] for index in confirmation_selected]),
            "same_complete_set": same_complete_set,
            "details": collected["confirmation_detail"],
        },
        "finite_circuit_fingerprints": circuit_report,
        "parent_response": parent_reports,
        "rung503_parent_differences": parent_report_differences,
        "instrument": {
            **diagnostics,
            "singleton_response_reproduces": singleton_response_reproduces,
            "singleton_payload_reproduces": singleton_payload_reproduces,
            "parent_denominators_reproduce": parent_denominators_reproduce,
            "calls": calls,
            "expected_calls": expected_calls,
            "calls_exact": calls_exact,
        },
        'pred_a_exact_finite_suffix_instrument_and_parent_reproduce': pred_a,
        'pred_b_compact_two_source_interaction_set_selected': pred_b,
        'pred_c_pair_identity_and_finite_effects_confirm': pred_c,
        'pred_d_selected_pairs_have_selective_downstream_use': pred_d,
        'pred_e_candidate_for_heldout_executable_intervention': pred_e,
        "strong_null": strong_null,
        "validation_documents_or_tags_opened": False,
        "sufficient_statistics": {
            "path": str(BUNDLE),
            "sha256": sha256(BUNDLE),
            "bytes": BUNDLE.stat().st_size,
        },
        "execution_price": {
            **expected_price(selected_count, confirmation_opened),
            "suffix_gpu_chunk_calls": calls["suffix_gpu_chunk_calls"],
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0,
            "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started,
        "next_step": (
            "repair_finite_suffix_instrument_only" if not pred_a else
            "run_float32_explanatory_control_or_change_downstream_observation"
            if not pred_b else
            "pair_identity_unstable_use_float32_control_or_change_observation"
            if not pred_c else
            "change_downstream_observation_before_circuit_claim" if not pred_d else
            "preregister_heldout_pair_intervention_and_sign_gauge_composition"),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete",
        "rung": 504,
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "strong_null": strong_null,
        "selection": result["selection"]["pairs"],
        "confirmation_reselection": result["confirmation_reselection"]["pairs"],
        "execution_price": result["execution_price"],
        "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
