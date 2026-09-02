#!/usr/bin/env python3
"""RUNG459 -- causal equality-score versus value/output transplants."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import circuit_induction_tensor as induction
import equality_term_subset_factorial_stage1 as stage1
import interchange


PREREG = POLY / "EQUALITY_TERM_SCORE_PAYLOAD_RUNG459_PREREGISTRATION.md"
ROWS = ROOT / ".rowcache_induction_equality_tensor_final_ood_v2/final_natural.pt"
ROW_RECEIPT = ROOT / "induction_equality_tensor_final_ood_v2_rows_receipt.json"
OUT = ROOT / "equality_term_score_payload_rung459_results.json"
BUNDLE = ROOT / "equality_term_score_payload_rung459_sufficient_statistics.pt"
TERMS = stage1.TERMS
TERM_NAMES = tuple(row[0] for row in TERMS)
PAIRS = ((0, 2), (0, 3), (1, 2), (1, 3))
PAIR_NAMES = tuple(f"{TERM_NAMES[e]}->{TERM_NAMES[l]}" for e, l in PAIRS)
FACTORS = ("score", "payload")
ARMS = ("base", "reference", "score", "payload", "whole")
COMPONENTS = tuple(
    component for site in range(9, 18) for component in (f"a{site}", f"m{site}")
)
RESPONSE_CELLS = ("all_positive", "matched_negative", "off_target")
CE_CELLS = RESPONSE_CELLS
FIT = slice(0, 96)
VALIDATION = slice(96, 192)
BATCH = 4
D = stage1.D
HEADS = stage1.HEADS
HEAD_DIM = stage1.HEAD_DIM
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = "equality-score-payload-rung459:bootstrap:0"
INTERCHANGE_SEED = 459
HASHES = {
    PREREG: "61863038d1dd038287a9b872501dcfa72042ba6a3bf6b8eda7bbf81c13c1ade3",
    ROW_RECEIPT: "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    ROWS: "5f2813eacc3ec66162c2ce695b978264137c66126fdc25e3d49b4efd44a9d759",
    ROOT / "ops/equality_term_subset_factorial_stage1.py":
        "3caa753cd856ec87899936fe71137ce28e893f86433558f40a815afff61824af",
    ROOT / "ops/interchange.py":
        "df4a8585dd6a557a71be991f12d0547023ae771bfccc591008cc0ab08f08fd29",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    POLY / "circuit_induction_tensor.py":
        "b2d43be8e260bbe4bfece494999d237d93258f676b19e2993eca09655e253e3a",
}


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
    payload, masks, metadata = stage1.validate_inputs()
    if payload.get("role") != "final_natural" or "ood_code" in str(ROWS):
        raise RuntimeError("rung459 loaded a forbidden row role")
    return payload, masks, metadata


def _linear(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def _term_index(site: int, head: int) -> int:
    return next(i for i, (_, s, h) in enumerate(TERMS) if (s, h) == (site, head))


def _factor_site(
    state: torch.Tensor,
    first_value: torch.Tensor,
    attention: torch.nn.Module,
    site: int,
    tokens: torch.Tensor,
):
    batch, length, width = state.shape
    if width != D or first_value.shape != (batch, length, HEADS, HEAD_DIM):
        raise RuntimeError("factor replay interface changed")
    q = _linear(state, attention.c_q.weight).view(batch, length, HEADS, HEAD_DIM)
    k = _linear(state, attention.c_k.weight).view(batch, length, HEADS, HEAD_DIM)
    q2 = _linear(state, attention.c_q2.weight).view(batch, length, HEADS, HEAD_DIM)
    k2 = _linear(state, attention.c_k2.weight).view(batch, length, HEADS, HEAD_DIM)
    raw_value = _linear(state, attention.c_v.weight).view(batch, length, HEADS, HEAD_DIM)
    value = (1 - attention.lamb) * raw_value + attention.lamb * first_value.view_as(raw_value)
    cos, sin = attention.rotary(q)
    module = sys.modules[type(attention).__module__]
    q = module.apply_rotary_emb(F.rms_norm(q, (HEAD_DIM,)), cos, sin)
    k = module.apply_rotary_emb(F.rms_norm(k, (HEAD_DIM,)), cos, sin)
    q2 = module.apply_rotary_emb(F.rms_norm(q2, (HEAD_DIM,)), cos, sin)
    k2 = module.apply_rotary_emb(F.rms_norm(k2, (HEAD_DIM,)), cos, sin)
    score1 = torch.einsum("bqhd,bkhd->bhqk", q, k) / HEAD_DIM
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HEAD_DIM
    pattern = score1 * score2
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=state.device))
    pattern = pattern.masked_fill(~causal, 0)
    heads = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
    flattened = heads.transpose(1, 2).contiguous().view(batch, length, width)
    full_write = _linear(flattened, attention.c_proj.weight)
    support = induction.induction_fetch_mask(tokens)
    factors = {}
    reconstruction_errors = []
    for head in stage1.SITE_HEADS[site]:
        index = _term_index(site, head)
        p = pattern[:, head].float()
        v = value[:, :, head]
        weight = attention.c_proj.weight[:, head * HEAD_DIM:(head + 1) * HEAD_DIM]
        u = F.linear(v.float(), weight.float())
        equality_head = induction.contract_induction_fetch(pattern[:, head], v, tokens)
        native_term = F.linear(equality_head, weight.to(equality_head.dtype))
        factor_term = torch.bmm(p * support, u)
        reference32 = F.linear(
            torch.bmm(p * support, v.float()), weight.float(),
        )
        error = float((factor_term - reference32).square().sum()) / max(
            float(reference32.square().sum()), 1e-30,
        )
        reconstruction_errors.append(error)
        factors[index] = {
            "p": p, "u": u, "native_term": native_term,
            "factor_term": factor_term,
        }
    return full_write, factors, support, max(reconstruction_errors, default=0.0)


def _empty_scale_stats():
    return {
        name: {key: 0.0 for key in (
            "edge_count", "payload_entry_count", "early_p2", "late_p2", "p_cross",
            "early_u2", "late_u2", "u_cross",
        )}
        for name in PAIR_NAMES
    }


def _accumulate_scales(stats, pair_name, early, late, support, positive_mask):
    selected = support & positive_mask.to(support.device).unsqueeze(-1)
    edge_count = int(selected.sum())
    if edge_count == 0:
        return
    p_e, p_l = early["p"], late["p"]
    edge_weight = selected.sum(1).float()
    u_e, u_l = early["u"], late["u"]
    row = stats[pair_name]
    row["edge_count"] += edge_count
    row["payload_entry_count"] += edge_count * D
    row["early_p2"] += float(p_e[selected].square().sum())
    row["late_p2"] += float(p_l[selected].square().sum())
    row["p_cross"] += float((p_e[selected] * p_l[selected]).sum())
    weights = edge_weight.unsqueeze(-1)
    row["early_u2"] += float((weights * u_e.square()).sum())
    row["late_u2"] += float((weights * u_l.square()).sum())
    row["u_cross"] += float((weights * u_e * u_l).sum())


def _finish_scales(stats):
    output = {}
    for name, row in stats.items():
        if row["edge_count"] <= 0 or row["payload_entry_count"] <= 0:
            raise RuntimeError(f"no fitting equality edges for {name}")
        a_e = math.sqrt(row["early_p2"] / row["edge_count"])
        a_l = math.sqrt(row["late_p2"] / row["edge_count"])
        b_e = math.sqrt(row["early_u2"] / row["payload_entry_count"])
        b_l = math.sqrt(row["late_u2"] / row["payload_entry_count"])
        if min(a_e, a_l, b_e, b_l) <= 0 or not all(
            math.isfinite(value) for value in (a_e, a_l, b_e, b_l)
        ):
            raise RuntimeError(f"invalid fitting RMS scale for {name}")
        output[name] = {
            **row,
            "early_score_rms": a_e, "late_score_rms": a_l,
            "early_payload_rms": b_e, "late_payload_rms": b_l,
            "score_ratio": a_l / a_e, "payload_ratio": b_l / b_e,
            "direct_score_cosine": row["p_cross"] / math.sqrt(
                row["early_p2"] * row["late_p2"]
            ),
            "direct_payload_cosine": row["u_cross"] / math.sqrt(
                row["early_u2"] * row["late_u2"]
            ),
        }
    return output


def _record_audit(totals, label, audit, *, analytical, captures):
    expected = {
        "native_attention": 15 if analytical else 18,
        "replayed_attention": 3 if analytical else 0,
        "native_mlp": 18,
        "captures": captures,
    }
    if audit != expected:
        raise RuntimeError(f"forward audit changed for {label}: {audit} != {expected}")
    row = totals.setdefault(label, {"forwards": 0, **{key: 0 for key in expected}})
    row["forwards"] += 1
    for key, value in audit.items():
        row[key] += value


@torch.no_grad()
def run_forward(
    model,
    tokens,
    *,
    pair: tuple[int, int] | None,
    arm: str,
    scales: Mapping[str, float] | None = None,
    capture_keys: Sequence[str] = (),
    scale_callback=None,
):
    if arm not in (*ARMS, "native", "replay", "scale"):
        raise ValueError(f"unknown arm: {arm}")
    analytical = arm != "native"
    if pair is None and arm not in {"native", "replay"}:
        raise ValueError("pairless forward must be native or replay")
    if pair is not None and pair not in PAIRS:
        raise ValueError("unregistered early-to-late pair")
    if arm in {"score", "payload", "whole"} and scales is None:
        raise ValueError("hybrid arm requires frozen scales")
    capture_set = set(capture_keys)
    if len(capture_set) != len(capture_keys) or not capture_set <= set(COMPONENTS):
        raise ValueError("capture identity changed")
    cached_early = {}
    captures = {}
    audit = {"native_attention": 0, "replayed_attention": 0, "native_mlp": 0,
             "captures": 0}
    max_reconstruction = 0.0

    def attention(event):
        nonlocal max_reconstruction
        if analytical and event.site in stage1.SITE_HEADS:
            write, factors, support, reconstruction = _factor_site(
                event.state, event.first_value, event.block.attn, event.site, event.tokens,
            )
            max_reconstruction = max(max_reconstruction, reconstruction)
            audit["replayed_attention"] += 1
            if pair is not None:
                early, late = pair
                early_site = TERMS[early][1]
                late_site = TERMS[late][1]
                if event.site == early_site:
                    cached_early.update(factors[early])
                    write = write - factors[early]["native_term"]
                if event.site == late_site:
                    if not cached_early:
                        raise RuntimeError("early factors were not cached before layer8")
                    late_factor = factors[late]
                    if arm == "scale":
                        if scale_callback is None:
                            raise RuntimeError("scale arm lacks accumulator")
                        scale_callback(cached_early, late_factor, support)
                    elif arm != "reference":
                        write = write - late_factor["native_term"]
                        if arm in {"score", "payload", "whole"}:
                            assert scales is not None
                            p = late_factor["p"]
                            u = late_factor["u"]
                            if arm in {"score", "whole"}:
                                p = cached_early["p"] * scales["score_ratio"]
                            if arm in {"payload", "whole"}:
                                u = cached_early["u"] * scales["payload_ratio"]
                            hybrid = torch.bmm(p * support, u).to(write.dtype)
                            write = write + hybrid
            next_value = event.first_value
        else:
            write, next_value = event.block.attn(event.state, event.first_value)
            audit["native_attention"] += 1
        key = f"a{event.site}"
        if key in capture_set:
            captures[key] = write.detach().clone()
            audit["captures"] += 1
        return write, next_value

    def mlp(event):
        write = event.block.mlp(event.state)
        audit["native_mlp"] += 1
        key = f"m{event.site}"
        if key in capture_set:
            captures[key] = write.detach().clone()
            audit["captures"] += 1
        return write

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
    if set(captures) != capture_set:
        raise RuntimeError("capture set changed")
    return logits, captures, audit, max_reconstruction


def _ce_sums(logits, rows, masks, global_start):
    targets = rows[:, 1:].to(logits.device)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).view(len(rows), -1)
    sums = torch.zeros(len(rows), len(CE_CELLS), dtype=torch.float64)
    counts = torch.zeros_like(sums)
    for local in range(len(rows)):
        for ci, cell in enumerate(CE_CELLS):
            selected = masks[cell][global_start + local]
            sums[local, ci] = nll[local, selected].double().sum().cpu()
            counts[local, ci] = int(selected.sum())
    return sums, counts


def _empty_response_stats(pair_count=len(PAIRS), component_count=len(COMPONENTS)):
    shape = (pair_count, len(FACTORS) + 1, len(RESPONSE_CELLS), component_count)
    return {
        key: torch.zeros(shape, dtype=torch.float64)
        for key in ("ref2", "hyb2", "cross", "write2", "tokens")
    }


def _accumulate_response(stats, pair_slot, captures, masks, start, components=COMPONENTS):
    base, reference = captures["base"], captures["reference"]
    for ci, cell in enumerate(RESPONSE_CELLS):
        selected = masks[cell][start:start + BATCH]
        token_count = int(selected.sum())
        if token_count == 0:
            continue
        for j, component in enumerate(components):
            ref = (reference[component] - base[component])[selected].float()
            writer = reference[component][selected].float()
            for hi, hybrid in enumerate(("score", "payload", "whole")):
                value = (captures[hybrid][component] - base[component])[selected].float()
                stats["ref2"][pair_slot, hi, ci, j] += ref.square().sum().double().cpu()
                stats["hyb2"][pair_slot, hi, ci, j] += value.square().sum().double().cpu()
                stats["cross"][pair_slot, hi, ci, j] += (ref * value).sum().double().cpu()
                stats["write2"][pair_slot, hi, ci, j] += writer.square().sum().double().cpu()
                stats["tokens"][pair_slot, hi, ci, j] += token_count


def _response_row(stats, pair_slot, hybrid_slot, cell_index, component_index):
    a = float(stats["ref2"][pair_slot, hybrid_slot, cell_index, component_index])
    b = float(stats["hyb2"][pair_slot, hybrid_slot, cell_index, component_index])
    cross = float(stats["cross"][pair_slot, hybrid_slot, cell_index, component_index])
    write = float(stats["write2"][pair_slot, hybrid_slot, cell_index, component_index])
    cosine = cross / math.sqrt(max(a * b, 1e-30))
    error = math.sqrt(max(a + b - 2 * cross, 0.0) / max(a, 1e-30))
    return {
        "cosine": cosine,
        "reference_relative_error": error,
        "reference_rms_over_reader_write_rms": math.sqrt(a / max(write, 1e-30)),
        "hybrid_rms_over_reader_write_rms": math.sqrt(b / max(write, 1e-30)),
        "tokens": int(stats["tokens"][pair_slot, hybrid_slot, cell_index, component_index]),
    }


def _pooled_ce(losses, counts, pair_slot, arm):
    ai = ARMS.index(arm)
    return [
        float(losses[pair_slot, ai, :, ci].sum() / counts[:, ci].sum())
        for ci in range(len(CE_CELLS))
    ]


def make_candidates(response_stats, losses, counts):
    candidates = []
    for pi, pair in enumerate(PAIRS):
        ce = {arm: _pooled_ce(losses, counts, pi, arm) for arm in ARMS}
        positive = CE_CELLS.index("all_positive")
        off = CE_CELLS.index("off_target")
        stake = ce["base"][positive] - ce["reference"][positive]
        for fi, factor in enumerate(FACTORS):
            recovery = (
                (ce["base"][positive] - ce[factor][positive]) / stake
                if stake > 0 else None
            )
            off_change = ce[factor][off] - ce["reference"][off]
            for ji, component in enumerate(COMPONENTS):
                reports = {
                    cell: _response_row(response_stats, pi, fi, ci, ji)
                    for ci, cell in enumerate(RESPONSE_CELLS)
                }
                pos = reports["all_positive"]
                margin = pos["cosine"] - max(
                    reports["matched_negative"]["cosine"], reports["off_target"]["cosine"],
                )
                live = min(
                    pos["reference_rms_over_reader_write_rms"],
                    pos["hybrid_rms_over_reader_write_rms"],
                )
                qualified = bool(
                    stake > 0 and recovery is not None and recovery >= .50
                    and abs(off_change) <= .01 and pos["cosine"] >= .75
                    and margin >= .10 and pos["reference_relative_error"] <= .60
                    and live >= 1e-4
                )
                candidates.append({
                    "pair_index": pi, "pair": PAIR_NAMES[pi], "pair_terms": [
                        TERM_NAMES[pair[0]], TERM_NAMES[pair[1]],
                    ],
                    "factor": factor, "component": component,
                    "positive": pos, "matched_negative": reports["matched_negative"],
                    "off_target": reports["off_target"], "task_margin": margin,
                    "minimum_relative_response_rms": live,
                    "reference_stake_nat": stake, "ce_recovery": recovery,
                    "off_target_hybrid_minus_reference_nat": off_change,
                    "qualified": qualified,
                })
    qualified = [row for row in candidates if row["qualified"]]
    selected = sorted(qualified, key=lambda row: (
        -row["task_margin"], row["positive"]["reference_relative_error"],
        -(row["ce_recovery"] or -math.inf), row["pair"], row["factor"], row["component"],
    ))[0] if qualified else None
    return candidates, selected


@torch.no_grad()
def collect_scales(model, payload, masks, audit_totals):
    rows = payload["rows"]
    stats = _empty_scale_stats()
    max_reconstruction = 0.0
    device = next(model.parameters()).device
    for start in range(FIT.start, FIT.stop, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        positive = masks["all_positive"][start:start + BATCH]
        for pi, pair in enumerate(PAIRS):
            def callback(early, late, support, *, name=PAIR_NAMES[pi], positive=positive):
                _accumulate_scales(stats, name, early, late, support, positive)
            logits, _, audit, error = run_forward(
                model, tokens, pair=pair, arm="scale", scale_callback=callback,
            )
            _record_audit(audit_totals, f"fit:scale:{PAIR_NAMES[pi]}", audit,
                          analytical=True, captures=0)
            max_reconstruction = max(max_reconstruction, error)
            del logits
    return _finish_scales(stats), max_reconstruction


@torch.no_grad()
def collect_fit(model, payload, masks, scales, audit_totals):
    rows = payload["rows"]
    docs = FIT.stop - FIT.start
    losses = torch.zeros(len(PAIRS), len(ARMS), docs, len(CE_CELLS), dtype=torch.float64)
    counts = torch.zeros(docs, len(CE_CELLS), dtype=torch.float64)
    response = _empty_response_stats()
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    max_reconstruction = 0.0
    device = next(model.parameters()).device
    for start in range(FIT.start, FIT.stop, BATCH):
        local = start - FIT.start
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = run_forward(model, tokens, pair=None, arm="native")
        _record_audit(audit_totals, "fit:native", audit, analytical=False, captures=0)
        replay_logits, _, audit, error = run_forward(model, tokens, pair=None, arm="replay")
        _record_audit(audit_totals, "fit:replay", audit, analytical=True, captures=0)
        difference = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
        )
        max_reconstruction = max(max_reconstruction, error)
        del native, replay_logits, difference
        for pi, pair in enumerate(PAIRS):
            captures = {}
            for ai, arm in enumerate(ARMS):
                logits, arm_capture, audit, error = run_forward(
                    model, tokens, pair=pair, arm=arm,
                    scales=scales[PAIR_NAMES[pi]], capture_keys=COMPONENTS,
                )
                _record_audit(audit_totals, f"fit:{PAIR_NAMES[pi]}:{arm}", audit,
                              analytical=True, captures=len(COMPONENTS))
                max_reconstruction = max(max_reconstruction, error)
                sums, observed_counts = _ce_sums(logits, batch_rows, masks, start)
                if pi == 0 and ai == 0:
                    counts[local:local + BATCH] = observed_counts
                elif not torch.equal(observed_counts, counts[local:local + BATCH]):
                    raise RuntimeError("fit CE supports changed across arms")
                losses[pi, ai, local:local + BATCH] = sums
                captures[arm] = arm_capture
                del logits
            _accumulate_response(response, pi, captures, masks, start)
            del captures
    return response, losses, counts, replay, max_reconstruction


def _validation_response_report(stats, pair_slot, factor, component):
    fi, ji = FACTORS.index(factor), 0
    reports = {
        cell: _response_row(stats, pair_slot, fi, ci, ji)
        for ci, cell in enumerate(RESPONSE_CELLS)
    }
    positive = reports["all_positive"]
    margin = positive["cosine"] - max(
        reports["matched_negative"]["cosine"], reports["off_target"]["cosine"],
    )
    live = min(
        positive["reference_rms_over_reader_write_rms"],
        positive["hybrid_rms_over_reader_write_rms"],
    )
    return {"component": component, "factor": factor, **reports,
            "task_margin": margin, "minimum_relative_response_rms": live}


def _ce_recovery_rows(losses, counts, pair_slot, factor):
    ci = CE_CELLS.index("all_positive")
    base = losses[pair_slot, ARMS.index("base"), :, ci]
    reference = losses[pair_slot, ARMS.index("reference"), :, ci]
    hybrid = losses[pair_slot, ARMS.index(factor), :, ci]
    denominator = counts[:, ci]
    return base, reference, hybrid, denominator


def bootstrap_recovery(losses, counts, pair_slot, factor):
    base, reference, hybrid, denominator_counts = _ce_recovery_rows(
        losses, counts, pair_slot, factor,
    )
    stake = (base.sum() - reference.sum()) / denominator_counts.sum()
    effect = (base.sum() - hybrid.sum()) / denominator_counts.sum()
    point = float(effect / stake) if float(stake) > 0 else None
    seed = int.from_bytes(hashlib.sha256(BOOTSTRAP_SEED.encode()).digest()[:8], "little")
    generator = torch.Generator().manual_seed(seed)
    chunks = []
    all_positive = True
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        n = min(500, BOOTSTRAP_DRAWS - start)
        draws = torch.randint(len(base), (n, len(base)), generator=generator)
        weights = torch.zeros(n, len(base), dtype=torch.float64)
        weights.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
        denom = weights @ denominator_counts
        stakes = (weights @ (base - reference)) / denom
        effects = (weights @ (base - hybrid)) / denom
        all_positive &= bool((stakes > 0).all())
        chunks.append(torch.where(stakes > 0, effects / stakes, torch.zeros_like(stakes)))
    draws = torch.cat(chunks)
    lower = float(draws.sort().values[math.floor(.025 * BOOTSTRAP_DRAWS)])
    return {
        "reference_stake_nat": float(stake), "hybrid_effect_nat": float(effect),
        "recovery": point, "simultaneous_95_lower": lower,
        "every_bootstrap_reference_stake_positive": all_positive,
        "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
    }


@torch.no_grad()
def collect_validation(model, payload, masks, scales, selected, audit_totals):
    rows = payload["rows"]
    selected_pi = selected["pair_index"]
    selected_pair = PAIRS[selected_pi]
    late = selected_pair[1]
    other_early = next(early for early in (0, 1) if early != selected_pair[0])
    control_pair = (other_early, late)
    control_pi = PAIRS.index(control_pair)
    pair_slots = (selected_pi, control_pi)
    component = selected["component"]
    docs = VALIDATION.stop - VALIDATION.start
    losses = torch.zeros(2, len(ARMS), docs, len(CE_CELLS), dtype=torch.float64)
    counts = torch.zeros(docs, len(CE_CELLS), dtype=torch.float64)
    full = _empty_response_stats(pair_count=2, component_count=1)
    halves = [_empty_response_stats(pair_count=2, component_count=1) for _ in range(2)]
    validation_scale_stats = _empty_scale_stats()
    max_reconstruction = 0.0
    device = next(model.parameters()).device
    for start in range(VALIDATION.start, VALIDATION.stop, BATCH):
        local = start - VALIDATION.start
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        for slot, pi in enumerate(pair_slots):
            pair = PAIRS[pi]
            positive = masks["all_positive"][start:start + BATCH]
            def scale_callback(
                early, late_factor, support, *, name=PAIR_NAMES[pi], positive=positive,
            ):
                _accumulate_scales(
                    validation_scale_stats, name, early, late_factor, support, positive,
                )
            scale_logits, _, audit, error = run_forward(
                model, tokens, pair=pair, arm="scale", scale_callback=scale_callback,
            )
            _record_audit(
                audit_totals, f"validation:scale:{PAIR_NAMES[pi]}", audit,
                analytical=True, captures=0,
            )
            max_reconstruction = max(max_reconstruction, error)
            del scale_logits
            captures = {}
            for ai, arm in enumerate(ARMS):
                logits, arm_capture, audit, error = run_forward(
                    model, tokens, pair=pair, arm=arm, scales=scales[PAIR_NAMES[pi]],
                    capture_keys=(component,),
                )
                _record_audit(audit_totals, f"validation:{PAIR_NAMES[pi]}:{arm}", audit,
                              analytical=True, captures=1)
                max_reconstruction = max(max_reconstruction, error)
                sums, observed_counts = _ce_sums(logits, batch_rows, masks, start)
                if slot == 0 and ai == 0:
                    counts[local:local + BATCH] = observed_counts
                elif not torch.equal(observed_counts, counts[local:local + BATCH]):
                    raise RuntimeError("validation CE supports changed across arms")
                losses[slot, ai, local:local + BATCH] = sums
                captures[arm] = arm_capture
                del logits
            _accumulate_response(full, slot, captures, masks, start, components=(component,))
            half = 0 if local < docs // 2 else 1
            _accumulate_response(halves[half], slot, captures, masks, start,
                                 components=(component,))
            del captures
    validation_scales = _finish_scales({
        PAIR_NAMES[pi]: validation_scale_stats[PAIR_NAMES[pi]] for pi in pair_slots
    })
    return (
        full, halves, losses, counts, pair_slots, validation_scales, max_reconstruction,
    )


def analyze_validation(response, halves, losses, counts, selected, pair_slots):
    factor = selected["factor"]
    full = _validation_response_report(response, 0, factor, selected["component"])
    half_reports = [
        _validation_response_report(stats, 0, factor, selected["component"])
        for stats in halves
    ]
    bootstrap = bootstrap_recovery(losses, counts, 0, factor)
    ci_off = CE_CELLS.index("off_target")
    off_change = float((
        losses[0, ARMS.index(factor), :, ci_off]
        - losses[0, ARMS.index("reference"), :, ci_off]
    ).sum() / counts[:, ci_off].sum())
    half_recoveries = []
    for start, stop in ((0, 48), (48, 96)):
        base, reference, hybrid, denom = _ce_recovery_rows(losses, counts, 0, factor)
        stake = (base[start:stop].sum() - reference[start:stop].sum()) / denom[start:stop].sum()
        effect = (base[start:stop].sum() - hybrid[start:stop].sum()) / denom[start:stop].sum()
        half_recoveries.append(float(effect / stake) if float(stake) > 0 else None)
    selected_discrepancy = []
    control_discrepancy = []
    ci = CE_CELLS.index("all_positive")
    supported = counts[:, ci] > 0
    for slot, target in ((0, selected_discrepancy), (1, control_discrepancy)):
        delta = (
            losses[slot, ARMS.index(factor), :, ci]
            - losses[slot, ARMS.index("reference"), :, ci]
        )[supported] / counts[:, ci][supported]
        target.extend(delta.abs().tolist())
    interchange_result = interchange.commutation(
        selected_discrepancy, control_discrepancy,
        seed=INTERCHANGE_SEED, permutations=10_000,
    )
    interchange_result.update({
        "selected_pair": PAIR_NAMES[pair_slots[0]],
        "between_control_pair": PAIR_NAMES[pair_slots[1]],
        "factor": factor, "seed": INTERCHANGE_SEED, "permutations": 10_000,
    })
    response_pass = bool(
        full["all_positive"]["cosine"] >= .65 and full["task_margin"] >= .05
        and full["all_positive"]["reference_relative_error"] <= .70
        and full["minimum_relative_response_rms"] >= 1e-4
        and all(row["all_positive"]["cosine"] > 0 for row in half_reports)
        and all(value is not None and value > 0 for value in half_recoveries)
    )
    causal_pass = bool(
        bootstrap["every_bootstrap_reference_stake_positive"]
        and bootstrap["recovery"] is not None and bootstrap["recovery"] >= .40
        and bootstrap["simultaneous_95_lower"] > .20 and abs(off_change) <= .01
    )
    interchange_pass = bool(
        interchange_result["separation"] >= 2.0 and interchange_result["p_value"] <= .05
    )
    return {
        "response": full, "response_halves": half_reports,
        "causal_recovery": bootstrap, "recovery_halves": half_recoveries,
        "off_target_hybrid_minus_reference_nat": off_change,
        "interchange": interchange_result,
        "pred_c_response_transfer": response_pass,
        "pred_d_causal_effect": causal_pass,
        "pred_e_between_control": interchange_pass,
    }


def main():
    started = time.time()
    payload, masks, metadata = validate_inputs()
    if len(PAIRS) * len(FACTORS) * len(COMPONENTS) != 144:
        raise RuntimeError("candidate family changed")
    if os.environ.get("BQLIB_DRYRUN") == "1":
        shape = (len(PAIRS), len(ARMS), FIT.stop, len(CE_CELLS))
        losses = torch.ones(shape, dtype=torch.float64)
        counts = torch.ones(FIT.stop, len(CE_CELLS), dtype=torch.float64)
        response = _empty_response_stats()
        response["ref2"].fill_(1.0)
        response["hyb2"].fill_(1.0)
        response["write2"].fill_(100.0)
        response["tokens"].fill_(10)
        # Plant one task-specific score candidate at pair0/m12.
        ji = COMPONENTS.index("m12")
        response["cross"][0, 0, RESPONSE_CELLS.index("all_positive"), ji] = .90
        response["cross"][0, 0, RESPONSE_CELLS.index("matched_negative"), ji] = .20
        response["cross"][0, 0, RESPONSE_CELLS.index("off_target"), ji] = .20
        losses[:, ARMS.index("base"), :, :] = 2.0
        losses[:, ARMS.index("reference"), :, :] = 1.0
        losses[:, ARMS.index("score"), :, :] = 1.4
        losses[:, ARMS.index("payload"), :, :] = 1.8
        # Keep off-target hybrid-reference changes within the registered bar.
        oi = CE_CELLS.index("off_target")
        losses[:, ARMS.index("base"), :, oi] = 1.0
        losses[:, ARMS.index("reference"), :, oi] = 1.0
        losses[:, ARMS.index("score"), :, oi] = 1.0
        losses[:, ARMS.index("payload"), :, oi] = 1.0
        candidates, selected = make_candidates(response, losses, counts)
        if len(candidates) != 144 or selected is None or selected["pair_index"] != 0 \
                or selected["factor"] != "score" or selected["component"] != "m12":
            raise RuntimeError("synthetic factor candidate selection failed")
        print(json.dumps({
            "status": "dry_run_passed", "rung": 459, "model_loaded": False,
            "code_ood_loaded": False, "sealed_opened": False,
            "candidate_count": len(candidates), "selected": selected,
            "input_metadata": metadata,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung459 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    audit_totals = {}
    scales, scale_reconstruction = collect_scales(model, payload, masks, audit_totals)
    response, losses, counts, replay, fit_reconstruction = collect_fit(
        model, payload, masks, scales, audit_totals,
    )
    candidates, selected = make_candidates(response, losses, counts)
    reconstruction = max(scale_reconstruction, fit_reconstruction)
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
    )
    pred_b = selected is not None
    validation = None
    pred_c = pred_d = pred_e = False
    validation_stats = validation_halves = validation_losses = validation_counts = None
    if selected is not None:
        (
            validation_stats, validation_halves, validation_losses, validation_counts,
            pair_slots, validation_scales, error,
        ) = collect_validation(model, payload, masks, scales, selected, audit_totals)
        reconstruction = max(reconstruction, error)
        pred_a &= reconstruction <= 1e-10
        validation = analyze_validation(
            validation_stats, validation_halves, validation_losses, validation_counts,
            selected, pair_slots,
        )
        validation["reported_validation_scales_not_used_by_hybrids"] = validation_scales
        pred_c = validation["pred_c_response_transfer"]
        pred_d = validation["pred_d_causal_effect"]
        pred_e = validation["pred_e_between_control"]
    strong_null = bool(
        not pred_a or selected is None
        or (validation is not None and validation["response"]["all_positive"]["cosine"] < .30)
        or (validation is not None and (
            validation["causal_recovery"]["recovery"] is None
            or validation["causal_recovery"]["recovery"] <= .10
        ))
        or (validation is not None and validation["interchange"]["separation"] <= 1.2)
    )
    bundle = {
        "schema": "equality_term_score_payload_rung459_sufficient_statistics_v1",
        "fit_response_stats": response, "fit_loss_sums": losses, "fit_counts": counts,
        "validation_response_stats": validation_stats,
        "validation_response_halves": validation_halves,
        "validation_loss_sums": validation_losses,
        "validation_counts": validation_counts,
        "raw_rows_tokens_logits_or_hidden_states_included": False,
        "code_ood_loaded": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 459,
        "claim_level": "natural_text_factor_hybrid_screen_and_heldout_interchange_not_ood_or_adoption",
        "input_identity": metadata,
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "code_ood_loaded": False, "sealed_attention0_confirmation_opened": False,
        "pairs": PAIR_NAMES, "frozen_fit_scales": scales,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay,
        "fit_screen": {
            "candidate_count": len(candidates),
            "qualified_count": sum(row["qualified"] for row in candidates),
            "selected": selected, "candidates": candidates,
        },
        "validation": validation,
        "audit_totals": audit_totals,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "outer_forwards": sum(row["forwards"] for row in audit_totals.values()),
            "searched_factor_pair_readers": 144,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_factor_candidate': pred_b,
        'pred_c_response_transfer': pred_c,
        'pred_d_causal_effect': pred_d,
        'pred_e_between_control': pred_e,
        "strong_null": strong_null,
        "runtime_s": time.time() - started,
        "next_step": (
            "freeze_code_ood_factor_confirmation"
            if all((pred_a, pred_b, pred_c, pred_d, pred_e)) and not strong_null
            else "split_qk_product_or_context_condition_factor_pieces"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 459,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "selected": selected, "validation": validation,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "execution_price": result["execution_price"],
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
