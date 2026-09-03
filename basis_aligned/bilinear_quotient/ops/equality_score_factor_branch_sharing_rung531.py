#!/usr/bin/env python3
"""RUNG531 -- test factor-level sharing inside a causally validated equality score."""

# BQGATE: EXPERIMENT

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
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import bilin18_observed_model_facade as facade
import circuit_induction_tensor as induction
import equality_score_directed_action_graph_rung501 as row_parent
import equality_term_score_payload_rung459 as factor_parent


RUNG = 531
TERMS = ("L5H5", "L7H3", "L8H3", "L8H4")
TERM_SITES_HEADS = ((5, 5), (7, 3), (8, 3), (8, 4))
SITE_HEADS = {5: (5,), 7: (3,), 8: (3, 4)}
PAIRS = tuple((source, target) for source in range(4) for target in range(4)
              if source != target)
PAIR_NAMES = tuple(f"{TERMS[source]}->{TERMS[target]}" for source, target in PAIRS)
KNOWN_PRODUCT_PAIRS = ("L5H5->L8H4", "L7H3->L8H4", "L8H3->L8H4")
ASSIGNMENTS = ("direct", "swapped")
SEGMENTS = {"fit": (0, 250), "confirmation_half0": (250, 375),
            "confirmation_half1": (375, 500)}
CONTROLS = ("equality", "key_prefix_reversal", "non_equality_causal")
FIT_SEGMENT = 0
CONFIRMATION_SEGMENTS = (1, 2)
BATCH = 4
ROWS_OPENED = 500
FORWARDS = ROWS_OPENED // BATCH
QUERY_START = 64
HEADS = 9
HEAD_DIM = 128
FACTOR_COUNT = len(TERMS) * 2

PREREG = POLY / "EQUALITY_SCORE_FACTOR_BRANCH_SHARING_RUNG531_PREREGISTRATION.md"
MATH_SOURCE = ROOT / "ops/equality_score_factor_branch_sharing_rung531_math.py"
ROW_SOURCE = ROOT / "ops/equality_score_directed_action_graph_rung501.py"
ROW_RESULT = ROOT / "equality_score_directed_action_graph_rung501_results.json"
SIGN_SOURCE = ROOT / "ops/equality_score_sign_gauge_quotient.py"
SIGN_RESULT = ROOT / "equality_score_sign_gauge_quotient_results.json"
FACTOR_SOURCE = ROOT / "ops/equality_term_subset_factorial_stage1.py"
FACADE_SOURCE = POLY / "bilin18_observed_model_facade.py"
INDUCTION_SOURCE = POLY / "circuit_induction_tensor.py"
MODEL_SOURCE = Path("/workspace/tensor_language/jacclust/tt_model.py")
OUT = ROOT / "equality_score_factor_branch_sharing_rung531_results.json"
BUNDLE = ROOT / "equality_score_factor_branch_sharing_rung531_bundle.pt"
HASHES = {
    PREREG: "5bc02afcaffc04aa62baf48f8f8dcbbb727deec86e7722287177055730b785b3",
    MATH_SOURCE: "83b9cb3549a8626b4f2ffc5814deea1d910b04867d19225cccf50de6b6e32611",
    ROW_SOURCE: "97f3946f558f3d61fc952a9b6ddc7c334b51ccc0ccfe5f02c6ecced417f1e077",
    ROW_RESULT: "b17a9b274e4c61e0b4a3fc68d8ce84ec6f8e76f257c3d898e6a6990492301c4f",
    SIGN_SOURCE: "a6007d7faff747b8467d50ef6ae934a9bc7b617735b8f01c808b417f142a00c7",
    SIGN_RESULT: "eff94038395d4da9571f5ace8c9e69f5a18aae2382c6385b5724c8937d7ef8b9",
    FACTOR_SOURCE: "3caa753cd856ec87899936fe71137ce28e893f86433558f40a815afff61824af",
    FACADE_SOURCE: "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    INDUCTION_SOURCE: "b2d43be8e260bbe4bfece494999d237d93258f676b19e2993eca09655e253e3a",
    MODEL_SOURCE: "49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2",
}

PREDICTION_TEXT = {
    "pred_a_exact_authorized_instrument": "all frozen identities, authorities, and calls pass",
    "pred_b_both_score_factors_shared": "at least one pair shares both factors on both halves",
    "pred_c_exactly_one_score_factor_shared": "at least one pair shares exactly one factor",
    "pred_d_factor_gauges_match_product": "a B/C pair has product-consistent branch gauges",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    model_source_hash = sha256(MODEL_SOURCE)
    sign = json.loads(SIGN_RESULT.read_text())
    if not all(sign.get(key) is True for key in (
        "pred_" + "a_exact_live_sign_gauge_instrument",
        "pred_" + "b_L7H3_score_is_L8H4_computation_up_to_sign",
        "pred_" + "c_gauge_extends_to_L8H3",
    )) or sign.get("strong_null"):
        raise RuntimeError("frozen sign-gauge authority changed")
    rows, metadata = row_parent.validate_inputs()
    if tuple(factor_parent.TERM_NAMES) != TERMS or tuple(factor_parent.TERMS) != tuple(
        (name, site, head) for name, (site, head) in zip(TERMS, TERM_SITES_HEADS)
    ):
        raise RuntimeError("equality-head identity changed")
    if tuple(rows.shape) != (1000, 257) or rows.dtype != torch.long:
        raise RuntimeError("frozen natural census changed")
    return rows, metadata, model_source_hash


def empty_statistics() -> dict[str, torch.Tensor]:
    prefix = (len(SEGMENTS), len(CONTROLS))
    return {
        "factor_source2": torch.zeros((*prefix, FACTOR_COUNT), dtype=torch.float64),
        "factor_target2": torch.zeros((*prefix, FACTOR_COUNT), dtype=torch.float64),
        "factor_cross": torch.zeros((*prefix, FACTOR_COUNT, FACTOR_COUNT), dtype=torch.float64),
        "product_source2": torch.zeros((*prefix, len(TERMS)), dtype=torch.float64),
        "product_target2": torch.zeros((*prefix, len(TERMS)), dtype=torch.float64),
        "product_cross": torch.zeros((*prefix, len(TERMS), len(TERMS)), dtype=torch.float64),
        "edges": torch.zeros(prefix, dtype=torch.int64),
    }


def _score_branches(state: torch.Tensor, attention: torch.nn.Module):
    batch, length, width = state.shape
    if width != 1152 or attention.n_head != HEADS or attention.head_dim != HEAD_DIM:
        raise RuntimeError("attention branch dimensions changed")

    def project(linear):
        return F.linear(state, linear.weight.to(dtype=state.dtype)).view(
            batch, length, HEADS, HEAD_DIM)

    q, k = project(attention.c_q), project(attention.c_k)
    q2, k2 = project(attention.c_q2), project(attention.c_k2)
    cos, sin = attention.rotary(q)
    module = sys.modules[type(attention).__module__]
    q = module.apply_rotary_emb(F.rms_norm(q, (HEAD_DIM,)), cos, sin)
    k = module.apply_rotary_emb(F.rms_norm(k, (HEAD_DIM,)), cos, sin)
    q2 = module.apply_rotary_emb(F.rms_norm(q2, (HEAD_DIM,)), cos, sin)
    k2 = module.apply_rotary_emb(F.rms_norm(k2, (HEAD_DIM,)), cos, sin)
    first = torch.einsum("bqhd,bkhd->bhqk", q, k) / HEAD_DIM
    second = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HEAD_DIM
    return first, second


@torch.no_grad()
def capture_forward(model, tokens: torch.Tensor):
    captures = {}
    support_seen = None
    diagnostics = {"native_attention": 0, "native_mlp": 0, "captured_heads": 0,
                   "branch_product_max_abs": 0.0, "factor_reconstruction_max": 0.0}

    def attention(event):
        nonlocal support_seen
        if event.site in SITE_HEADS:
            first, second = _score_branches(event.state, event.block.attn)
            _write, factors, support, reconstruction = factor_parent._factor_site(
                event.state, event.first_value, event.block.attn, event.site, event.tokens)
            support_seen = support if support_seen is None else support_seen
            if not torch.equal(support_seen, support):
                raise RuntimeError("equality support changed across captured sites")
            diagnostics["factor_reconstruction_max"] = max(
                diagnostics["factor_reconstruction_max"], reconstruction)
            for head in SITE_HEADS[event.site]:
                name = f"L{event.site}H{head}"
                index = TERMS.index(name)
                native_product = (first[:, head] * second[:, head]).float()
                causal = torch.tril(torch.ones(
                    native_product.shape[-2:], dtype=torch.bool, device=native_product.device))
                native_product = native_product.masked_fill(~causal, 0.0)
                parent_product = factors[index]["p"]
                difference = float((native_product - parent_product).abs().max())
                diagnostics["branch_product_max_abs"] = max(
                    diagnostics["branch_product_max_abs"], difference)
                captures[name] = (first[:, head].float().detach(),
                                  second[:, head].float().detach())
                diagnostics["captured_heads"] += 1
        write, next_value = event.block.attn(event.state, event.first_value)
        diagnostics["native_attention"] += 1
        return write, next_value

    def mlp(event):
        diagnostics["native_mlp"] += 1
        return event.block.mlp(event.state)

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
    expected = {"native_attention": 18, "native_mlp": 18, "captured_heads": 4}
    if any(diagnostics[key] != value for key, value in expected.items()):
        raise RuntimeError(f"capture audit changed: {diagnostics}")
    if set(captures) != set(TERMS) or support_seen is None:
        raise RuntimeError("factor capture set changed")
    return logits, captures, support_seen, diagnostics


def _key_prefix_reverse(values: torch.Tensor) -> torch.Tensor:
    length = values.shape[-1]
    query = torch.arange(length, device=values.device)[:, None]
    key = torch.arange(length, device=values.device)[None, :]
    reverse = (query - key).remainder(length)
    return values.gather(-1, reverse.expand(values.shape[0], -1, -1))


def _add_gram_statistics(
    statistics: dict[str, torch.Tensor], segment_index: int, control_index: int,
    source_factors: torch.Tensor, target_factors: torch.Tensor,
    source_products: torch.Tensor, target_products: torch.Tensor, selected: torch.Tensor,
):
    count = int(selected.sum())
    if count <= 0:
        raise RuntimeError("registered segment/control has no selected edges")
    source_factor_values = source_factors[:, selected].double()
    target_factor_values = target_factors[:, selected].double()
    source_product_values = source_products[:, selected].double()
    target_product_values = target_products[:, selected].double()
    statistics["factor_source2"][segment_index, control_index] += (
        source_factor_values.square().sum(-1).cpu())
    statistics["factor_target2"][segment_index, control_index] += (
        target_factor_values.square().sum(-1).cpu())
    statistics["factor_cross"][segment_index, control_index] += (
        source_factor_values @ target_factor_values.T).cpu()
    statistics["product_source2"][segment_index, control_index] += (
        source_product_values.square().sum(-1).cpu())
    statistics["product_target2"][segment_index, control_index] += (
        target_product_values.square().sum(-1).cpu())
    statistics["product_cross"][segment_index, control_index] += (
        source_product_values @ target_product_values.T).cpu()
    statistics["edges"][segment_index, control_index] += count


def accumulate_batch(
    statistics: dict[str, torch.Tensor], captures, support: torch.Tensor, global_start: int,
):
    target_factors = torch.stack([
        branch for name in TERMS for branch in captures[name]
    ])
    target_products = torch.stack([
        captures[name][0] * captures[name][1] for name in TERMS
    ])
    reversed_factors = _key_prefix_reverse(target_factors.flatten(0, 1)).view_as(target_factors)
    reversed_products = torch.stack([
        reversed_factors[2 * index] * reversed_factors[2 * index + 1]
        for index in range(len(TERMS))
    ])
    batch, length, _ = support.shape
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=support.device))
    equality = support.clone()
    equality[:, :QUERY_START] = False
    non_equality = causal.unsqueeze(0).expand(batch, -1, -1) & ~support
    non_equality[:, :QUERY_START] = False
    row_ids = torch.arange(global_start, global_start + batch, device=support.device)

    for segment_index, (_name, (begin, end)) in enumerate(SEGMENTS.items()):
        rows = (row_ids >= begin) & (row_ids < end)
        if not bool(rows.any()):
            continue
        row_mask = rows[:, None, None]
        for control_index, control in enumerate(CONTROLS):
            if control == "equality":
                source_factors, source_products, edge_mask = (
                    target_factors, target_products, equality & row_mask)
            elif control == "key_prefix_reversal":
                source_factors, source_products, edge_mask = (
                    reversed_factors, reversed_products, equality & row_mask)
            else:
                source_factors, source_products, edge_mask = (
                    target_factors, target_products, non_equality & row_mask)
            _add_gram_statistics(
                statistics, segment_index, control_index, source_factors, target_factors,
                source_products, target_products, edge_mask)


def _branch_indices(pair, assignment: str):
    source, target = pair
    target_indices = (2 * target, 2 * target + 1)
    source_indices = ((2 * source, 2 * source + 1) if assignment == "direct"
                      else (2 * source + 1, 2 * source))
    return source_indices, target_indices


def _fit_scale(source2: float, cross: float) -> float:
    if source2 <= 0 or not math.isfinite(source2 + cross):
        raise RuntimeError("non-live scalar fit")
    return cross / source2


def _metric(source2: float, target2: float, cross: float, scale: float):
    if source2 <= 0 or target2 <= 0 or not all(
        math.isfinite(value) for value in (source2, target2, cross, scale)
    ):
        raise RuntimeError("non-live metric")
    prediction2 = scale * scale * source2
    prediction_cross = scale * cross
    cosine = prediction_cross / math.sqrt(max(prediction2 * target2, 1e-30))
    error2 = max(prediction2 + target2 - 2 * prediction_cross, 0.0)
    return {"cosine": cosine, "relative_rmse": math.sqrt(error2 / target2)}


def _branch_stat(statistics, segment, control, source_index, target_index):
    return (
        float(statistics["factor_source2"][segment, control, source_index]),
        float(statistics["factor_target2"][segment, control, target_index]),
        float(statistics["factor_cross"][segment, control, source_index, target_index]),
    )


def _product_stat(statistics, segment, control, source, target):
    return (
        float(statistics["product_source2"][segment, control, source]),
        float(statistics["product_target2"][segment, control, target]),
        float(statistics["product_cross"][segment, control, source, target]),
    )


def _fit_assignment(statistics, pair, assignment, segment=FIT_SEGMENT):
    source_indices, target_indices = _branch_indices(pair, assignment)
    scales, metrics = [], []
    for source_index, target_index in zip(source_indices, target_indices):
        values = _branch_stat(statistics, segment, 0, source_index, target_index)
        scale = _fit_scale(values[0], values[2])
        scales.append(scale)
        metrics.append(_metric(*values, scale))
    return {
        "assignment": assignment,
        "target_first_scale": scales[0],
        "target_second_scale": scales[1],
        "first": metrics[0],
        "second": metrics[1],
        "branch_objective": sum(metric["relative_rmse"] ** 2 for metric in metrics),
    }


def _choose_assignment(statistics, pair, segment=FIT_SEGMENT):
    reports = {assignment: _fit_assignment(statistics, pair, assignment, segment)
               for assignment in ASSIGNMENTS}
    selected = min(ASSIGNMENTS, key=lambda assignment: (
        reports[assignment]["branch_objective"], ASSIGNMENTS.index(assignment)))
    return selected, reports


def _fixed_report(statistics, pair, assignment, scales, segment, control):
    source_indices, target_indices = _branch_indices(pair, assignment)
    factors = []
    for source_index, target_index, scale in zip(source_indices, target_indices, scales):
        factors.append(_metric(
            *_branch_stat(statistics, segment, control, source_index, target_index), scale))
    source, target = pair
    product_values = _product_stat(statistics, segment, control, source, target)
    return {
        "first": factors[0], "second": factors[1],
        "branch_product": _metric(*product_values, scales[0] * scales[1]),
    }


def analyze(statistics):
    reports = {}
    both_candidates, one_candidates, gauge_candidates = [], [], []
    for pair_index, pair in enumerate(PAIRS):
        name = PAIR_NAMES[pair_index]
        selected, fit_reports = _choose_assignment(statistics, pair)
        selected_fit = fit_reports[selected]
        scales = (selected_fit["target_first_scale"], selected_fit["target_second_scale"])
        source, target = pair
        product_fit_values = _product_stat(statistics, FIT_SEGMENT, 0, source, target)
        product_scale = _fit_scale(product_fit_values[0], product_fit_values[2])
        product_scale_difference = abs(scales[0] * scales[1] - product_scale) / max(
            abs(product_scale), 1e-30)
        halves = {}
        factor_passes = [[], []]
        product_passes = []
        independently_selected = []
        gauge_passes = []
        other_factor_bad = [False, False]
        for segment in CONFIRMATION_SEGMENTS:
            segment_name = tuple(SEGMENTS)[segment]
            real = _fixed_report(statistics, pair, selected, scales, segment, 0)
            permuted = _fixed_report(statistics, pair, selected, scales, segment, 1)
            non_equality = _fixed_report(statistics, pair, selected, scales, segment, 2)
            half_selected, _half_fits = _choose_assignment(statistics, pair, segment)
            independently_selected.append(half_selected)
            baseline_values = _product_stat(statistics, segment, 0, source, target)
            baseline = _metric(*baseline_values, product_scale)
            halves[segment_name] = {
                "equality": real,
                "key_prefix_reversal": permuted,
                "non_equality_causal": non_equality,
                "independently_selected_assignment": half_selected,
                "scalar_product_baseline": baseline,
            }
            for branch_index, branch in enumerate(("first", "second")):
                metric = real[branch]
                margin = metric["cosine"] - permuted[branch]["cosine"]
                factor_passes[branch_index].append(
                    metric["cosine"] >= 0.90 and metric["relative_rmse"] <= 0.45
                    and margin >= 0.15)
                if metric["cosine"] < 0.70 or metric["relative_rmse"] > 0.65:
                    other_factor_bad[branch_index] = True
            product_passes.append(
                real["branch_product"]["cosine"] >= 0.90
                and real["branch_product"]["relative_rmse"] <= 0.45)
            gauge_passes.append(
                real["branch_product"]["relative_rmse"]
                <= baseline["relative_rmse"] + 0.05)
        assignment_stable = all(choice == selected for choice in independently_selected)
        stable_factors = [all(values) for values in factor_passes]
        both = all(stable_factors) and all(product_passes) and assignment_stable
        exactly_one = sum(stable_factors) == 1 and other_factor_bad[1 - stable_factors.index(True)]
        gauge = product_scale_difference <= 0.10 and all(gauge_passes)
        if both:
            both_candidates.append(name)
        if exactly_one:
            one_candidates.append(name)
        if gauge and (both or (exactly_one and name in KNOWN_PRODUCT_PAIRS)):
            gauge_candidates.append(name)
        reports[name] = {
            "selected_assignment": selected,
            "fit": fit_reports,
            "target_first_scale": scales[0],
            "target_second_scale": scales[1],
            "branch_scale_product": scales[0] * scales[1],
            "independent_product_scale": product_scale,
            "scale_product_relative_difference": product_scale_difference,
            "confirmation": halves,
            "assignment_stable": assignment_stable,
            "both_factors_shared": both,
            "exactly_one_factor_shared": exactly_one,
            "factor_gauges_match_product": gauge,
        }
    return reports, both_candidates, one_candidates, gauge_candidates


def _bundle(statistics, diagnostics):
    return {
        "schema": "rung531_factor_branch_aggregate_dot_products_v1",
        "statistics": statistics,
        "diagnostics": diagnostics,
        "raw_tokens_logits_states_or_edge_factors_included": False,
        "validation_or_ood_opened": False,
    }


def main():
    started = time.time()
    smoke = os.environ.get("RUNG531_SMOKE") == "1"
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(PAIRS) == 12 and FORWARDS == 125
        assert tuple(SEGMENTS.values()) == ((0, 250), (250, 375), (375, 500))
        assert set(KNOWN_PRODUCT_PAIRS) <= set(PAIR_NAMES)
        print(json.dumps({
            "status": "dry_run_passed", "rung": RUNG, "model_loaded": False,
            "scientific_outcomes_opened": False, "forwards": FORWARDS,
            "predictions": list(PREDICTION_TEXT), "segments": SEGMENTS,
            "controls": CONTROLS,
        }, indent=2, sort_keys=True))
        return
    if not smoke and (OUT.exists() or BUNDLE.exists()):
        raise RuntimeError("rung531 output namespace already exists")
    rows, metadata, model_source_hash = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    statistics = empty_statistics()
    aggregate_diagnostics = {
        "calls": 0, "factor_reconstruction_max": 0.0,
        "branch_product_max_abs": 0.0, "minimum_equality_edges_per_batch": math.inf,
    }
    stop = BATCH if smoke else ROWS_OPENED
    for start in range(0, stop, BATCH):
        tokens = rows[start:start + BATCH, :-1].to("cuda")
        logits, captures, support, diagnostics = capture_forward(model, tokens)
        equality = support.clone()
        equality[:, :QUERY_START] = False
        aggregate_diagnostics["minimum_equality_edges_per_batch"] = min(
            aggregate_diagnostics["minimum_equality_edges_per_batch"], int(equality.sum()))
        aggregate_diagnostics["factor_reconstruction_max"] = max(
            aggregate_diagnostics["factor_reconstruction_max"],
            diagnostics["factor_reconstruction_max"])
        aggregate_diagnostics["branch_product_max_abs"] = max(
            aggregate_diagnostics["branch_product_max_abs"],
            diagnostics["branch_product_max_abs"])
        aggregate_diagnostics["calls"] += 1
        if not smoke:
            accumulate_batch(statistics, captures, support, start)
        del logits, captures, support, tokens
    if smoke:
        print(json.dumps({
            "status": "smoke_passed", "rung": RUNG,
            "scientific_outcomes_opened": False,
            "calls": aggregate_diagnostics["calls"],
            "factor_reconstruction_max": aggregate_diagnostics["factor_reconstruction_max"],
            "branch_product_max_abs": aggregate_diagnostics["branch_product_max_abs"],
            "minimum_equality_edges_per_batch":
                aggregate_diagnostics["minimum_equality_edges_per_batch"],
            "checkpoint_weights_sha256": checkpoint.weights_sha256,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }, indent=2, sort_keys=True))
        return
    reports, both_candidates, one_candidates, gauge_candidates = analyze(statistics)
    aggregate_diagnostics.update({
        "calls_exact": aggregate_diagnostics["calls"] == FORWARDS,
        "all_segment_controls_live": bool((statistics["edges"] > 0).all()),
        "all_factor_norms_live": bool((statistics["factor_source2"] > 0).all()
                                      and (statistics["factor_target2"] > 0).all()),
        "all_product_norms_live": bool((statistics["product_source2"] > 0).all()
                                       and (statistics["product_target2"] > 0).all()),
    })
    pred_a = bool(
        aggregate_diagnostics["calls_exact"]
        and aggregate_diagnostics["all_segment_controls_live"]
        and aggregate_diagnostics["all_factor_norms_live"]
        and aggregate_diagnostics["all_product_norms_live"]
        and aggregate_diagnostics["factor_reconstruction_max"] <= 1e-10
        and aggregate_diagnostics["branch_product_max_abs"] == 0.0
        and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
    pred_b = bool(pred_a and both_candidates)
    pred_c = bool(pred_a and one_candidates)
    pred_d = bool(pred_a and gauge_candidates)
    prediction_values = dict(zip(PREDICTION_TEXT, (pred_a, pred_b, pred_c, pred_d)))
    result = {
        "rung": RUNG,
        "status": "completed",
        **prediction_values,
        "strong_null": bool(pred_a and not pred_b and not pred_c),
        "both_factor_candidates": both_candidates,
        "one_factor_candidates": one_candidates,
        "gauge_consistent_candidates": gauge_candidates,
        "reports": reports,
        "diagnostics": aggregate_diagnostics,
        "segments": SEGMENTS,
        "controls": CONTROLS,
        "known_product_pairs": KNOWN_PRODUCT_PAIRS,
        "price": {
            "model_forwards": aggregate_diagnostics["calls"],
            "backward_passes": 0, "fitted_scalars_per_pair": 3,
            "learned_vector_parameters": 0, "validation_or_ood_forwards": 0,
        },
        "checkpoint": checkpoint,
        "input_metadata": metadata,
        "source_hashes": {str(path): expected for path, expected in HASHES.items()},
        "model_source_sha256": model_source_hash,
        "elapsed_seconds": time.time() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "raw_tokens_logits_states_or_edge_factors_included": False,
    }
    torch.save(_bundle(statistics, aggregate_diagnostics), BUNDLE)
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"], "rung": RUNG,
        **{key: result[key] for key in PREDICTION_TEXT},
        "strong_null": result["strong_null"],
        "both_factor_candidates": both_candidates,
        "one_factor_candidates": one_candidates,
        "gauge_consistent_candidates": gauge_candidates,
        "calls": aggregate_diagnostics["calls"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
