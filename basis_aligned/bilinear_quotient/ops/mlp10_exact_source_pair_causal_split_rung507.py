#!/usr/bin/env python3
"""RUNG507 -- exact MLP10 named-source-pair split with finite causal tests."""

# BQGATE: EXPERIMENT
# pred_a: exact named-source, bilinear, and finite-intervention instrument
# pred_b: 2--8 no-ranking gradient candidates stable across score sources
# pred_c: at least two named terms survive finite confirmation
# pred_d: at least two named terms validate on new documents
# pred_e: at least one confirmation-frozen pair composition predicts validation

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
import equality_score_gauged_downstream_program_rung505 as score_parent
import natural_action_conditioned_later_write_state_atlas_rung506 as state_parent


PREREG = POLY / "MLP10_EXACT_SOURCE_PAIR_CAUSAL_SPLIT_RUNG507_PREREGISTRATION.md"
R506_SOURCE = ROOT / "ops/natural_action_conditioned_later_write_state_atlas_rung506.py"
R506_RESULT = ROOT / "natural_action_conditioned_later_write_state_atlas_rung506_results.json"
R506_BUNDLE = ROOT / "natural_action_conditioned_later_write_state_atlas_rung506_bundle.pt"
R506_AUDIT = ROOT / "rung506_failure_audit_results.json"
R505_SOURCE = ROOT / "ops/equality_score_gauged_downstream_program_rung505.py"
OUT = ROOT / "mlp10_exact_source_pair_causal_split_rung507_results.json"
BUNDLE = ROOT / "mlp10_exact_source_pair_causal_split_rung507_bundle.pt"
HASHES = {
    PREREG: "4bfd001804fde4ab0852172c5fe5242fb523258f1e60cd9aa14c26a94428a8e9",
    R506_SOURCE: "9a17e28312a0e7214e5fc587123e3267e2650b382f3a40daf12ad1a380b1d004",
    R506_RESULT: "f86e5f0303ab0616ea14e3141fd09886ca54d326e8d83ea6c8c13a62f66db75e",
    R506_BUNDLE: "225f73cb885e0e51d76ed329b60b044359a600a18e130846c99dd4c103959093",
    R506_AUDIT: "96f58a2d993a34900c9ef74aec7a0e98d8363155092a882aabd2e570db978946",
    R505_SOURCE: "0c5f6679ec40cb02bd6af1e28b0b41ca2ad7967fd4b6c9d73a4f388153f3e4de",
}

TARGET = 10
D = 1152
H = 4608
TOKENS = 256
BATCH = 4
SOURCES = score_parent.SOURCES
NAMED_SOURCES = ("E",) + tuple(f"A{i}" for i in range(TARGET + 1)) \
    + tuple(f"M{i}" for i in range(TARGET))
SOURCE_PAIRS = tuple(itertools.combinations_with_replacement(range(len(NAMED_SOURCES)), 2))
PAIR_NAMES = tuple(f"{NAMED_SOURCES[left]}x{NAMED_SOURCES[right]}"
                   for left, right in SOURCE_PAIRS)
GRAD_CELLS = (
    "near_positive", "far_positive", "one_predecessor_positive",
    "multiple_predecessor_positive", "off_target",
)
TASK_CELLS = score_parent.CELLS
DISCOVERY = (0, 248, 124)
CONFIRMATION = (248, 496, 372)
VALIDATION = (500, 1000, 750)
MAX_TERMS = 8
U = 2.0 ** -8
DEPLOYED_BF16_BAR = 16 * U * U


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_squared(left, right):
    left = torch.as_tensor(left).double()
    right = torch.as_tensor(right).double()
    return float((left - right).square().sum() / right.square().sum().clamp_min(1e-30))


def _linear(value, weight):
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def _source_coefficients(model):
    lambda0 = [block.lambdas[0].detach().float() for block in model.transformer.h[:TARGET + 1]]
    lambda1 = [block.lambdas[1].detach().float() for block in model.transformer.h[:TARGET + 1]]
    embedding = torch.ones_like(lambda0[0])
    for residual, skip in zip(lambda0, lambda1):
        embedding = residual * embedding + skip
    writes = []
    for site in range(TARGET + 1):
        coefficient = torch.ones_like(lambda0[0])
        for later in range(site + 1, TARGET + 1):
            coefficient = coefficient * lambda0[later]
        writes.append(coefficient)
    return embedding, tuple(writes)


def _normalized_sources(model, x0, attention_writes, prior_writes, raw_state, normalized):
    if len(attention_writes) != TARGET + 1 or len(prior_writes) != TARGET:
        raise RuntimeError("MLP10 source count changed")
    embedding_coefficient, write_coefficients = _source_coefficients(model)
    raw = [embedding_coefficient * x0.float()]
    raw.extend(write_coefficients[i] * attention_writes[i].float()
               for i in range(TARGET + 1))
    raw.extend(write_coefficients[i] * prior_writes[i].float() for i in range(TARGET))
    raw = torch.stack(raw, dim=2)
    raw_sum = raw.sum(2)
    raw_error = _relative_squared(raw_sum, raw_state.float())
    z = normalized.float()
    gain = (z * raw_sum).sum(-1, keepdim=True) \
        / raw_sum.square().sum(-1, keepdim=True).clamp_min(1e-30)
    sources = gain.unsqueeze(2) * raw
    numerical = z - sources.sum(2)
    normalized_error = _relative_squared(sources.sum(2) + numerical, z)
    numerical_ratio = float(
        numerical.square().mean().sqrt() / z.square().mean().sqrt().clamp_min(1e-30))
    return sources.detach(), numerical.detach(), {
        "raw_source_relative_squared": raw_error,
        "normalized_closure_relative_squared": normalized_error,
        "normalized_numerical_rms_ratio": numerical_ratio,
    }


def _source_factors(mlp, sources, numerical, deployed_write):
    left = _linear(sources.float(), mlp.Left.weight.float())
    right = _linear(sources.float(), mlp.Right.weight.float())
    left_num = _linear(numerical.float(), mlp.Left.weight.float())
    right_num = _linear(numerical.float(), mlp.Right.weight.float())
    left_full = left.sum(2) + left_num
    right_full = right.sum(2) + right_num
    full_hidden = left_full * right_full
    semantic_hidden = left.sum(2) * right.sum(2)
    independent = _linear(full_hidden, mlp.Down.weight.float()) + mlp.Down_bias.float()
    semantic_output = _linear(semantic_hidden, mlp.Down.weight.float())
    numerical_output = independent - semantic_output - mlp.Down_bias.float()
    rebuilt = semantic_output + numerical_output + mlp.Down_bias.float()
    return {
        "left": left.detach(), "right": right.detach(),
        "deployed_write": deployed_write.detach(),
        "independent_write": independent.detach(),
        "semantic_output": semantic_output.detach(),
        "numerical_output": numerical_output.detach(),
        "float32_closure": _relative_squared(rebuilt, independent),
        "deployed_relative_squared": _relative_squared(independent, deployed_write.float()),
    }


def _pair_hidden(factors, pair_index):
    left, right = SOURCE_PAIRS[pair_index]
    hidden = factors["left"][:, :, left] * factors["right"][:, :, right]
    if left != right:
        hidden = hidden + factors["left"][:, :, right] * factors["right"][:, :, left]
    return hidden


def _pair_output(mlp, factors, pair_index):
    return _linear(_pair_hidden(factors, pair_index), mlp.Down.weight.float())


def _sum_unordered_pair_hidden(factors):
    """Sum the frozen unordered vocabulary without materializing 253 outputs."""
    total = torch.zeros_like(factors["left"][:, :, 0])
    for pair_index in range(len(SOURCE_PAIRS)):
        total = total + _pair_hidden(factors, pair_index)
    return total


def _unordered_contraction(reader, factors):
    ordered = torch.einsum(
        "bth,btsh,btuh->su", reader.float(), factors["left"], factors["right"])
    values = []
    for left, right in SOURCE_PAIRS:
        value = ordered[left, right]
        if left != right:
            value = value + ordered[right, left]
        values.append(value)
    return torch.stack(values).double().cpu()


def _pair_relationship(left_index, right_index):
    left = set(SOURCE_PAIRS[left_index])
    right = set(SOURCE_PAIRS[right_index])
    shared = sorted(left & right)
    return {
        "left_term": PAIR_NAMES[left_index], "right_term": PAIR_NAMES[right_index],
        "shared_unordered_sources": [NAMED_SOURCES[index] for index in shared],
        "same_left_source": SOURCE_PAIRS[left_index][0] == SOURCE_PAIRS[right_index][0],
        "same_right_source": SOURCE_PAIRS[left_index][1] == SOURCE_PAIRS[right_index][1],
    }


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(R506_RESULT.read_text())
    if not (
        result.get("pred_a_exact_live_conditional_instrument") is True
        and result.get("pred_b_score_actions_recalibrate_new_documents") is True
        and result.get("pred_c_at_least_one_whole_write_edge_confirms") is False
        and result.get("strong_null") is True
        and result.get("next_step") == "split_fixed_writes_into_exact_attention_or_bilinear_terms"
    ):
        raise RuntimeError("rung506 route changed")
    audit = json.loads(R506_AUDIT.read_text())
    if audit.get("new_model_outcomes_opened") is not False \
            or audit.get("next_object") \
            != "exact_MLP10_named_input_source_pair_terms_with_finite_downstream_interventions" \
            or "m10" not in audit.get("task_stable_sites", []):
        raise RuntimeError("rung506 CPU successor audit changed")
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = \
        state_parent.validate_inputs()
    if len(NAMED_SOURCES) != 22 or len(SOURCE_PAIRS) != 253 \
            or SOURCE_PAIRS[0] != (0, 0) or SOURCE_PAIRS[-1] != (21, 21):
        raise RuntimeError("MLP10 source-pair vocabulary changed")
    return rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, {
        **metadata,
        "rung506_result_sha256": sha256(R506_RESULT),
        "rung506_bundle_sha256": sha256(R506_BUNDLE),
        "rung506_audit_sha256": sha256(R506_AUDIT),
        "named_sources": list(NAMED_SOURCES), "pair_names": list(PAIR_NAMES),
    }


def _source_action(scales, action):
    spec = score_parent.SOURCE_ACTIONS[action]
    pair = spec["pair"]
    if pair is None:
        return None, None
    return pair, score_parent.signed_scales(scales, action)


def _forward(
    model,
    tokens,
    scales,
    *,
    action="N",
    absent=False,
    direct=False,
    capture_mlp10=False,
    gradient_leaf=False,
    remove_terms=(),
    absent_capture=None,
):
    if action not in SOURCES or (direct and (action != "N" or absent or remove_terms)):
        raise ValueError("unregistered forward arm")
    if remove_terms and absent_capture is None:
        raise ValueError("term removal requires score-absent MLP10 factors")
    if any(index < 0 or index >= len(SOURCE_PAIRS) for index in remove_terms):
        raise ValueError("term index outside frozen vocabulary")
    pair, scale = _source_action(scales, action)
    cached = {}
    attention_writes = []
    prior_writes = []
    capture = {}
    diagnostics = {
        "factor_reconstruction_max": 0.0,
        "score_edit_rms": 0.0,
        "raw_source_relative_squared": 0.0,
        "normalized_closure_relative_squared": 0.0,
        "normalized_numerical_rms_ratio": 0.0,
        "float32_mlp10_closure": 0.0,
        "deployed_mlp10_relative_squared": 0.0,
        "term_edit_rms": 0.0,
    }
    audit = {"native_attention": 0, "replayed_attention": 0, "mlps": 0,
             "mlp10_captures": 0, "mlp10_term_patches": 0}
    x = model.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    first_value = None
    factor_parent = score_parent.action_parent.factor_parent

    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_state = F.rms_norm(x, (x.size(-1),))
        if direct or site not in factor_parent.stage1.SITE_HEADS:
            attention_write, next_value = block.attn(attention_state, first_value)
            audit["native_attention"] += 1
        else:
            attention_write, factors, support, error = factor_parent._factor_site(
                attention_state, first_value, block.attn, site, tokens)
            audit["replayed_attention"] += 1
            diagnostics["factor_reconstruction_max"] = max(
                diagnostics["factor_reconstruction_max"], error)
            edit_pair = (0, 3) if absent else pair
            if edit_pair is not None:
                donor, recipient = edit_pair
                if site == factor_parent.TERMS[donor][1]:
                    cached.update(factors[donor])
                if site == factor_parent.TERMS[recipient][1]:
                    if not cached:
                        raise RuntimeError("score donor factors unavailable")
                    target = factors[recipient]
                    replacement = torch.zeros_like(target["factor_term"])
                    if not absent:
                        replacement = torch.bmm(
                            cached["p"] * scale["score_ratio"] * support, target["u"])
                    edit = replacement.to(attention_write.dtype) - target["native_term"]
                    attention_write = attention_write + edit
                    diagnostics["score_edit_rms"] = float(edit.float().square().mean().sqrt())
            next_value = first_value
        first_value = next_value
        attention_writes.append(attention_write.detach())
        x = x + attention_write
        raw_state = x
        z = F.rms_norm(raw_state, (raw_state.size(-1),))
        deployed_write = block.mlp(z)
        write = deployed_write
        if site == TARGET:
            sources, numerical, source_diag = _normalized_sources(
                model, x0, attention_writes, prior_writes, raw_state, z)
            factors = _source_factors(block.mlp, sources, numerical, deployed_write)
            for key, value in source_diag.items():
                diagnostics[key] = max(diagnostics[key], value)
            diagnostics["float32_mlp10_closure"] = factors["float32_closure"]
            diagnostics["deployed_mlp10_relative_squared"] = factors["deployed_relative_squared"]
            if remove_terms:
                hidden_delta = torch.zeros_like(factors["left"][:, :, 0])
                for term_index in remove_terms:
                    hidden_delta += _pair_hidden(factors, term_index)
                    hidden_delta -= _pair_hidden(absent_capture["factors"], term_index)
                output_delta = _linear(hidden_delta, block.mlp.Down.weight.float())
                write = write - output_delta.to(write.dtype)
                diagnostics["term_edit_rms"] = float(output_delta.square().mean().sqrt())
                audit["mlp10_term_patches"] += 1
            if gradient_leaf:
                write = write.detach().requires_grad_(True)
            if capture_mlp10:
                capture = {
                    "factors": factors,
                    "write": write,
                    "deployed_write": deployed_write.detach(),
                    "numerical_output": factors["numerical_output"],
                }
                audit["mlp10_captures"] += 1
        prior_writes.append(write)
        x = x + write
        audit["mlps"] += 1

    logits = model.lm_head(F.rms_norm(x, (x.size(-1),)))
    logits = (30.0 * torch.tanh(logits / 30.0)).float()
    if logits.shape[:2] != tokens.shape or not torch.isfinite(logits).all():
        raise RuntimeError("invalid logits from explicit forward")
    expected_attention = (18, 0) if direct else (15, 3)
    if (audit["native_attention"], audit["replayed_attention"]) != expected_attention \
            or audit["mlps"] != 18 \
            or audit["mlp10_captures"] != int(capture_mlp10) \
            or audit["mlp10_term_patches"] != int(bool(remove_terms)):
        raise RuntimeError(f"forward audit changed: {audit}")
    return logits, capture, diagnostics, audit


def _nll(logits, batch_rows):
    targets = batch_rows[:, 1:].to(logits.device)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).view(len(batch_rows), -1)


def _task_sums(nll, local_masks):
    mask = torch.stack([local_masks[cell].to(nll.device).double() for cell in TASK_CELLS], -1)
    return torch.einsum("abp,bpc->abc", nll.double(), mask)


def _empty_diagnostics():
    return {
        "calls": {"direct": 0, "analytical": 0}, "backwards": 0,
        "factor_reconstruction_max": 0.0,
        "raw_source_relative_squared": 0.0,
        "normalized_closure_relative_squared": 0.0,
        "normalized_numerical_rms_ratio": 0.0,
        "float32_mlp10_closure": 0.0,
        "deployed_mlp10_relative_squared": 0.0,
        "score_delta_float32_closure": 0.0,
        "score_delta_deployed_relative_squared": 0.0,
        "native_replay_logit_max_abs": 0.0,
        "native_replay_relative_squared": 0.0,
        "minimum_nonzero_score_edit_rms": math.inf,
        "minimum_nonzero_term_edit_rms": math.inf,
        "term_patches": 0, "zero_term_edits": 0,
    }


def _update_diagnostics(total, row):
    for key in (
        "factor_reconstruction_max", "raw_source_relative_squared",
        "normalized_closure_relative_squared", "normalized_numerical_rms_ratio",
        "float32_mlp10_closure", "deployed_mlp10_relative_squared",
    ):
        total[key] = max(total[key], row[key])
    if row["score_edit_rms"] > 0:
        total["minimum_nonzero_score_edit_rms"] = min(
            total["minimum_nonzero_score_edit_rms"], row["score_edit_rms"])
    if row["term_edit_rms"] > 0:
        total["minimum_nonzero_term_edit_rms"] = min(
            total["minimum_nonzero_term_edit_rms"], row["term_edit_rms"])


def _score_delta_closure(total, current, absent):
    named = current["factors"]["semantic_output"] - absent["factors"]["semantic_output"]
    numerical = current["numerical_output"] - absent["numerical_output"]
    independent = current["factors"]["independent_write"] \
        - absent["factors"]["independent_write"]
    deployed = current["deployed_write"].float() - absent["deployed_write"].float()
    rebuilt = named + numerical
    total["score_delta_float32_closure"] = max(
        total["score_delta_float32_closure"], _relative_squared(rebuilt, independent))
    total["score_delta_deployed_relative_squared"] = max(
        total["score_delta_deployed_relative_squared"], _relative_squared(rebuilt, deployed))


def _calibration(base_task, source_task, counts, bounds):
    fake = {
        "bounds": bounds, "arms": ("intact",),
        "base_task": base_task, "source_task": source_task[:, None],
        "task_counts": counts,
    }
    return state_parent.calibration(fake)


def collect_gradient(model, rows, task_masks, scales):
    lo, hi, split = DISCOVERY
    documents = hi - lo
    attribution = torch.zeros(2, len(SOURCES), len(SOURCE_PAIRS), len(GRAD_CELLS),
                              dtype=torch.float64)
    full = torch.zeros(2, len(SOURCES), len(GRAD_CELLS), dtype=torch.float64)
    counts = torch.zeros(2, len(GRAD_CELLS), dtype=torch.float64)
    base_task = torch.zeros(documents, len(TASK_CELLS), dtype=torch.float64)
    source_task = torch.zeros(len(SOURCES), documents, len(TASK_CELLS), dtype=torch.float64)
    task_counts = torch.zeros(documents, len(TASK_CELLS), dtype=torch.float64)
    diagnostics = _empty_diagnostics()
    device = next(model.parameters()).device
    for start in range(lo, hi, BATCH):
        stop = start + BATCH
        local = start - lo
        half = int(start >= split)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in TASK_CELLS}
        direct_logits, _, direct_diag, _ = _forward(model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        _update_diagnostics(diagnostics, direct_diag)
        absent_logits, absent_capture, absent_diag, _ = _forward(
            model, tokens, scales, action="P", absent=True, capture_mlp10=True)
        diagnostics["calls"]["analytical"] += 1
        _update_diagnostics(diagnostics, absent_diag)
        absent_nll = _nll(absent_logits, batch_rows).detach().cpu()
        base_task[local:local + BATCH] = _task_sums(absent_nll[None], masks)[0]
        task_counts[local:local + BATCH] = torch.stack(
            [masks[cell].sum(1).double() for cell in TASK_CELLS], -1)

        active = []
        for cell_index, cell in enumerate(GRAD_CELLS):
            selected = masks[cell].to(device)
            observed = int(selected.sum())
            counts[half, cell_index] += observed
            if observed:
                active.append((cell_index, selected))

        for source_index, source in enumerate(SOURCES):
            logits, capture, diag, _ = _forward(
                model, tokens, scales, action=source,
                capture_mlp10=True, gradient_leaf=True)
            diagnostics["calls"]["analytical"] += 1
            _update_diagnostics(diagnostics, diag)
            _score_delta_closure(diagnostics, capture, absent_capture)
            if source == "N":
                difference = logits.float() - direct_logits.float()
                diagnostics["native_replay_logit_max_abs"] = max(
                    diagnostics["native_replay_logit_max_abs"], float(difference.abs().max()))
                diagnostics["native_replay_relative_squared"] = max(
                    diagnostics["native_replay_relative_squared"],
                    float(difference.square().sum()) /
                    max(float(direct_logits.float().square().sum()), 1e-30))
            nll = _nll(logits, batch_rows)
            source_task[source_index, local:local + BATCH] = _task_sums(
                nll.detach().cpu()[None], masks)[0]
            write_delta = capture["deployed_write"].float() \
                - absent_capture["deployed_write"].float()
            for active_index, (cell_index, selected) in enumerate(active):
                gradient = torch.autograd.grad(
                    nll[selected].sum(), capture["write"],
                    retain_graph=active_index + 1 < len(active), allow_unused=False)[0]
                diagnostics["backwards"] += 1
                reader = _linear(-gradient.float(), model.transformer.h[TARGET].mlp.Down.weight.float().T)
                attribution[half, source_index, :, cell_index] += (
                    _unordered_contraction(reader, capture["factors"])
                    - _unordered_contraction(reader, absent_capture["factors"]))
                full[half, source_index, cell_index] += float(
                    (-gradient.float() * write_delta).sum())
            del logits, capture, nll
        del direct_logits, absent_logits, absent_capture

    batches = (hi - lo) // BATCH
    expected_backwards = 0
    for start in range(lo, hi, BATCH):
        for cell in GRAD_CELLS:
            expected_backwards += int(bool(task_masks[cell][start:start + BATCH].any())) * len(SOURCES)
    diagnostics["calls_expected"] = {"direct": batches, "analytical": batches * 5}
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["backwards_expected"] = expected_backwards
    diagnostics["backwards_exact"] = diagnostics["backwards"] == expected_backwards
    return {
        "attribution_sums": attribution, "full_sums": full, "gradient_counts": counts,
        "base_task": base_task, "source_task": source_task, "task_counts": task_counts,
        "diagnostics": diagnostics,
    }


def gradient_vector(collection, source, term, window="pooled"):
    source_index = SOURCES.index(source)
    term_index = PAIR_NAMES.index(term)
    halves = range(2) if window == "pooled" else (_window_index(window),)
    sums = collection["attribution_sums"][list(halves), source_index, term_index].sum(0)
    counts = collection["gradient_counts"][list(halves)].sum(0).clamp_min(1)
    return sums[:4] / counts[:4]


def full_gradient_vector(collection, source, window="pooled"):
    source_index = SOURCES.index(source)
    halves = range(2) if window == "pooled" else (_window_index(window),)
    sums = collection["full_sums"][list(halves), source_index].sum(0)
    counts = collection["gradient_counts"][list(halves)].sum(0).clamp_min(1)
    return sums[:4] / counts[:4]


def gradient_all_off(collection, source, term, window="pooled"):
    source_index = SOURCES.index(source)
    term_index = PAIR_NAMES.index(term)
    halves = range(2) if window == "pooled" else (_window_index(window),)
    sums = collection["attribution_sums"][list(halves), source_index, term_index].sum(0)
    counts = collection["gradient_counts"][list(halves)].sum(0).clamp_min(1)
    all_copy = float((sums[0] + sums[1]) / (counts[0] + counts[1]))
    off_target = float(sums[4] / counts[4])
    return all_copy, off_target


def _window_index(window):
    return {"half0": 0, "half1": 1}[window]


def _comparison(left, right):
    return state_parent.comparison(left, right)


def discover_terms(collection):
    checks = {}
    candidates = []
    for term in PAIR_NAMES:
        row = {"sources": {}, "source_comparisons": {}}
        holds = True
        for source in SOURCES:
            pooled = gradient_vector(collection, source, term)
            full = full_gradient_vector(collection, source)
            repeat = _comparison(
                gradient_vector(collection, source, term, "half0"),
                gradient_vector(collection, source, term, "half1"))
            projection = abs(float(torch.dot(pooled, full) / torch.dot(full, full).clamp_min(1e-30)))
            all_copy, off_target = gradient_all_off(collection, source, term)
            source_holds = bool(
                float(torch.linalg.vector_norm(pooled)) >= .00025
                and projection >= .05 and repeat["cosine"] >= .60
                and repeat["norm_ratio"] <= 3
                and abs(all_copy) >= 2 * abs(off_target))
            row["sources"][source] = {
                "task_vector_nat": pooled.tolist(),
                "task_norm_nat": float(torch.linalg.vector_norm(pooled)),
                "projection_fraction_on_full": projection,
                "repeat": repeat, "all_copy_attribution_nat": all_copy,
                "off_target_attribution_nat": off_target, "holds": source_holds,
            }
            holds &= source_holds
        native = gradient_vector(collection, "N", term)
        for source in SOURCES[1:]:
            metric = _comparison(native, gradient_vector(collection, source, term))
            metric["holds"] = bool(metric["cosine"] >= .70 and metric["norm_ratio"] <= 3)
            row["source_comparisons"][f"N:{source}"] = metric
            holds &= metric["holds"]
        row["holds"] = bool(holds)
        checks[term] = row
        if holds:
            candidates.append(term)
    return candidates, checks


def _arm_name(spec):
    return "intact" if not spec else "+".join(PAIR_NAMES[index] for index in spec)


def collect_finite(
    model, rows, task_masks, circuit_masks, circuit_tags, scales, bounds, patch_specs,
):
    lo, hi, split = bounds
    documents = hi - lo
    arms = tuple(_arm_name(spec) for spec in patch_specs)
    if len(set(arms)) != len(arms):
        raise ValueError("duplicate finite arm")
    task = torch.zeros(len(SOURCES), len(arms), documents, len(TASK_CELLS), dtype=torch.float64)
    counts = torch.zeros(documents, len(TASK_CELLS), dtype=torch.float64)
    circuit_sums = None
    circuit_counts = None
    if circuit_tags:
        circuit_sums = torch.zeros(
            len(SOURCES), len(arms), 2, 2, len(circuit_tags), dtype=torch.float64)
        circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    diagnostics = _empty_diagnostics()
    device = next(model.parameters()).device
    for start in range(lo, hi, BATCH):
        stop = start + BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in TASK_CELLS}
        _absent_logits, absent_capture, absent_diag, _ = _forward(
            model, tokens, scales, action="P", absent=True, capture_mlp10=True)
        diagnostics["calls"]["analytical"] += 1
        _update_diagnostics(diagnostics, absent_diag)
        counts[local:local + BATCH] = torch.stack(
            [masks[cell].sum(1).double() for cell in TASK_CELLS], -1)
        nll_rows = []
        for source in SOURCES:
            for spec in patch_specs:
                logits, capture, diag, audit = _forward(
                    model, tokens, scales, action=source, capture_mlp10=True,
                    remove_terms=spec, absent_capture=absent_capture)
                diagnostics["calls"]["analytical"] += 1
                diagnostics["term_patches"] += audit["mlp10_term_patches"]
                diagnostics["zero_term_edits"] += int(
                    bool(spec) and diag["term_edit_rms"] <= 0)
                _update_diagnostics(diagnostics, diag)
                _score_delta_closure(diagnostics, capture, absent_capture)
                nll_rows.append(_nll(logits, batch_rows).detach().cpu())
        nll_stack = torch.stack(nll_rows).view(
            len(SOURCES), len(arms), BATCH, TOKENS)
        task[:, :, local:local + BATCH] = _task_sums(
            nll_stack.view(-1, BATCH, TOKENS), masks).view(
                len(SOURCES), len(arms), BATCH, len(TASK_CELLS))
        if circuit_tags:
            matrix, observed = state_parent._circuit_mask_matrix(
                circuit_masks, circuit_tags, start, stop, bounds)
            circuit_counts += observed
            circuit_sums += torch.matmul(
                nll_stack.view(len(SOURCES) * len(arms), -1).double(), matrix.T,
            ).view(len(SOURCES), len(arms), 2, 2, len(circuit_tags))
        del absent_capture, nll_stack
    batches = (hi - lo) // BATCH
    expected_calls = batches * (1 + len(SOURCES) * len(arms))
    expected_patches = batches * len(SOURCES) * sum(bool(spec) for spec in patch_specs)
    diagnostics["calls_expected"] = {"direct": 0, "analytical": expected_calls}
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["term_patches_expected"] = expected_patches
    diagnostics["term_patches_exact"] = diagnostics["term_patches"] == expected_patches
    return {
        "bounds": bounds, "arms": arms, "patch_specs": patch_specs,
        "task": task, "task_counts": counts,
        "circuit_tags": tuple(circuit_tags), "circuit_sums": circuit_sums,
        "circuit_counts": circuit_counts, "diagnostics": diagnostics,
    }


def finite_vector(target, arm, intact, source, window="pooled", context_only=True):
    source_index = SOURCES.index(source)
    target_index = target["arms"].index(arm)
    intact_index = intact["arms"].index("intact")
    if window == "pooled":
        lo, hi = 0, target["task"].shape[2]
    else:
        bounds = target["bounds"]
        absolute = ((bounds[0], bounds[2]), (bounds[2], bounds[1]))[_window_index(window)]
        lo, hi = absolute[0] - bounds[0], absolute[1] - bounds[0]
    numerator = (target["task"][source_index, target_index, lo:hi]
                 - intact["task"][source_index, intact_index, lo:hi]).sum(0)
    denominator = target["task_counts"][lo:hi].sum(0).clamp_min(1)
    vector = numerator / denominator
    if context_only:
        indices = [TASK_CELLS.index(cell) for cell in GRAD_CELLS[:4]]
        return vector[indices]
    return vector


def finite_all_off(target, arm, intact, source, window="pooled"):
    vector = finite_vector(target, arm, intact, source, window, context_only=False)
    return float(vector[TASK_CELLS.index("all_positive")]), \
        float(vector[TASK_CELLS.index("off_target")])


def confirm_terms(discovery, finite, candidates):
    checks = {}
    confirmed = []
    for term in candidates:
        row = {"sources": {}, "source_comparisons": {}}
        holds = True
        for source in SOURCES:
            vector = finite_vector(finite, term, finite, source)
            repeat = _comparison(
                finite_vector(finite, term, finite, source, "half0"),
                finite_vector(finite, term, finite, source, "half1"))
            gradient_match = _comparison(gradient_vector(discovery, source, term), vector)
            all_copy, off_target = finite_all_off(finite, term, finite, source)
            source_holds = bool(
                float(torch.linalg.vector_norm(vector)) >= .00025
                and gradient_match["cosine"] >= .60
                and repeat["cosine"] >= .50 and repeat["norm_ratio"] <= 3
                and abs(all_copy) >= .00025 and abs(all_copy) >= 2 * abs(off_target))
            row["sources"][source] = {
                "task_vector_nat": vector.tolist(), "gradient_match": gradient_match,
                "repeat": repeat, "all_copy_effect_nat": all_copy,
                "off_target_effect_nat": off_target, "holds": source_holds,
            }
            holds &= source_holds
        native = finite_vector(finite, term, finite, "N")
        for source in SOURCES[1:]:
            metric = _comparison(native, finite_vector(finite, term, finite, source))
            metric["holds"] = bool(metric["cosine"] >= .70 and metric["norm_ratio"] <= 3)
            row["source_comparisons"][f"N:{source}"] = metric
            holds &= metric["holds"]
        row["holds"] = bool(holds)
        checks[term] = row
        if holds:
            confirmed.append(term)
    return confirmed, checks


def validate_terms(confirmation, validation, terms):
    checks = {}
    validated = []
    for term in terms:
        row = {"sources": {}, "source_comparisons": {}}
        holds = True
        for source in SOURCES:
            confirm = finite_vector(confirmation, term, confirmation, source)
            vector = finite_vector(validation, term, validation, source)
            transfer = _comparison(confirm, vector)
            half_cosines = [state_parent.cosine(
                finite_vector(validation, term, validation, source, window), confirm)
                for window in ("half0", "half1")]
            all_copy, off_target = finite_all_off(validation, term, validation, source)
            source_holds = bool(
                transfer["cosine"] >= .60 and transfer["norm_ratio"] <= 3
                and min(half_cosines) > 0 and abs(all_copy) >= .00025
                and abs(all_copy) >= 2 * abs(off_target))
            row["sources"][source] = {
                "task_vector_nat": vector.tolist(), "confirmation_transfer": transfer,
                "half_cosines_to_confirmation": half_cosines,
                "all_copy_effect_nat": all_copy, "off_target_effect_nat": off_target,
                "holds": source_holds,
            }
            holds &= source_holds
        native = finite_vector(validation, term, validation, "N")
        for source in SOURCES[1:]:
            metric = _comparison(native, finite_vector(validation, term, validation, source))
            metric["holds"] = bool(metric["cosine"] >= .65 and metric["norm_ratio"] <= 3)
            row["source_comparisons"][f"N:{source}"] = metric
            holds &= metric["holds"]
        row["holds"] = bool(holds)
        checks[term] = row
        if holds:
            validated.append(term)
    return validated, checks


def _relative_residual(actual, predicted):
    actual = torch.as_tensor(actual, dtype=torch.float64)
    predicted = torch.as_tensor(predicted, dtype=torch.float64)
    return float(torch.linalg.vector_norm(actual - predicted)
                 / torch.linalg.vector_norm(actual).clamp_min(1e-30))


def fit_composition(singletons, pairs, left, right):
    name = f"{left}+{right}"
    lefts, rights, joints = [], [], []
    for source in SOURCES:
        lefts.append(finite_vector(singletons, left, singletons, source))
        rights.append(finite_vector(singletons, right, singletons, source))
        joints.append(finite_vector(pairs, name, singletons, source))
    left_vector, right_vector, joint = map(torch.cat, (lefts, rights, joints))
    summed = left_vector + right_vector
    interaction = joint - summed
    joint_norm = torch.linalg.vector_norm(joint).clamp_min(1e-30)
    interaction_ratio = float(torch.linalg.vector_norm(interaction) / joint_norm)
    left_residual = float(torch.linalg.vector_norm(joint - left_vector) / joint_norm)
    right_residual = float(torch.linalg.vector_norm(joint - right_vector) / joint_norm)
    row = {
        "interaction_over_joint": interaction_ratio,
        "left_redundancy_residual": left_residual,
        "right_redundancy_residual": right_residual,
    }
    if interaction_ratio <= .25:
        row.update({"kind": "additive", "beta": 0.0, "identified": True})
    elif left_residual <= .25:
        row.update({"kind": "left_redundant", "beta": None, "identified": True})
    elif right_residual <= .25:
        row.update({"kind": "right_redundant", "beta": None, "identified": True})
    else:
        beta = float(torch.dot(interaction, summed) / torch.dot(summed, summed).clamp_min(1e-30))
        residual = _relative_residual(interaction, beta * summed)
        identified = bool(abs(beta) >= .25 and -.8 <= beta <= 2 and residual <= .50)
        row.update({"kind": "one_scalar_interaction" if identified else "none",
                    "beta": beta, "scalar_interaction_residual": residual,
                    "identified": identified})
    return row


def predict_composition(rule, left, right):
    if rule["kind"] == "additive":
        return left + right
    if rule["kind"] == "left_redundant":
        return left
    if rule["kind"] == "right_redundant":
        return right
    if rule["kind"] == "one_scalar_interaction":
        return (1 + rule["beta"]) * (left + right)
    raise ValueError("composition rule is unidentified")


def score_composition(validation, left, right, rule):
    name = f"{left}+{right}"
    row = {"sources": {}, "same_output": True}
    holds = bool(rule["identified"])
    for source in SOURCES:
        left_vector = finite_vector(validation, left, validation, source)
        right_vector = finite_vector(validation, right, validation, source)
        joint = finite_vector(validation, name, validation, source)
        predicted = predict_composition(rule, left_vector, right_vector)
        singleton_cosine = state_parent.cosine(left_vector, right_vector)
        half_prediction_cosines = []
        for window in ("half0", "half1"):
            half_left = finite_vector(validation, left, validation, source, window)
            half_right = finite_vector(validation, right, validation, source, window)
            half_joint = finite_vector(validation, name, validation, source, window)
            half_prediction_cosines.append(state_parent.cosine(
                predict_composition(rule, half_left, half_right), half_joint))
        all_copy, off_target = finite_all_off(validation, name, validation, source)
        source_holds = bool(
            state_parent.cosine(predicted, joint) >= .70
            and _relative_residual(joint, predicted) <= .65
            and min(half_prediction_cosines) > 0
            and abs(all_copy) >= .00025 and abs(all_copy) >= 2 * abs(off_target))
        row["sources"][source] = {
            "prediction_cosine": state_parent.cosine(predicted, joint),
            "prediction_relative_residual": _relative_residual(joint, predicted),
            "half_prediction_cosines": half_prediction_cosines,
            "singleton_same_output_cosine": singleton_cosine,
            "all_copy_effect_nat": all_copy, "off_target_effect_nat": off_target,
            "holds": source_holds,
        }
        holds &= source_holds
        row["same_output"] &= singleton_cosine >= .80
    row["holds"] = bool(holds)
    return row


def _phase_instrument(collection, *, gradient=False):
    d = collection["diagnostics"]
    base = bool(
        d["calls_exact"]
        and d["factor_reconstruction_max"] <= 1e-10
        and d["raw_source_relative_squared"] <= DEPLOYED_BF16_BAR
        and d["normalized_closure_relative_squared"] <= 1e-12
        and d["normalized_numerical_rms_ratio"] <= .02
        and d["float32_mlp10_closure"] <= 1e-10
        and d["deployed_mlp10_relative_squared"] <= DEPLOYED_BF16_BAR
        and d["score_delta_float32_closure"] <= 1e-10
        and d["score_delta_deployed_relative_squared"] <= DEPLOYED_BF16_BAR
        and d["minimum_nonzero_score_edit_rms"] > 0)
    if gradient:
        return bool(base and d["backwards_exact"]
                    and d["native_replay_relative_squared"] <= 1e-12
                    and d["native_replay_logit_max_abs"] == 0.0)
    return bool(base and d["term_patches_exact"] and d["zero_term_edits"] == 0
                and (d["term_patches"] == 0 or d["minimum_nonzero_term_edit_rms"] > 0))


def _bundle_finite(collection):
    return {key: collection[key] for key in (
        "bounds", "arms", "patch_specs", "task", "task_counts",
        "circuit_tags", "circuit_sums", "circuit_counts",
    )}


def _gpu_smoke():
    """Exercise the CUDA algebra/patch path while retaining no scientific effects."""
    rows, _task_masks, _circuit_masks, scales, _discovery_tags, _validation_tags, _meta = \
        validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    batch_rows = rows[:BATCH]
    tokens = batch_rows[:, :-1].to("cuda")
    direct_logits, _, direct_diag, _ = _forward(model, tokens, scales, direct=True)
    absent_logits, absent, absent_diag, _ = _forward(
        model, tokens, scales, action="P", absent=True, capture_mlp10=True)
    native_logits, native, native_diag, _ = _forward(
        model, tokens, scales, action="N", capture_mlp10=True)
    score_logits, score, score_diag, _ = _forward(
        model, tokens, scales, action="P", capture_mlp10=True, gradient_leaf=True)

    native_difference = native_logits - direct_logits
    native_relative = float(native_difference.square().sum()) \
        / max(float(direct_logits.square().sum()), 1e-30)
    closure = _empty_diagnostics()
    _score_delta_closure(closure, score, absent)

    # The explicit 253-term expansion is checked on one token. This validates the
    # vocabulary/algebra without turning the smoke into a second scientific run.
    sample = {
        "left": score["factors"]["left"][:1, :1],
        "right": score["factors"]["right"][:1, :1],
    }
    pair_sum = _sum_unordered_pair_hidden(sample)
    named_product = sample["left"].sum(2) * sample["right"].sum(2)
    pair_enumeration_error = _relative_squared(pair_sum, named_product)
    pair_output = _linear(pair_sum, model.transformer.h[TARGET].mlp.Down.weight.float())
    pair_output_error = _relative_squared(
        pair_output, score["factors"]["semantic_output"][:1, :1])

    nll = _nll(score_logits, batch_rows)
    gradient = torch.autograd.grad(nll.sum(), score["write"], allow_unused=False)[0]
    reader = _linear(
        -gradient.float(), model.transformer.h[TARGET].mlp.Down.weight.float().T)
    attributions = _unordered_contraction(reader, score["factors"]) \
        - _unordered_contraction(reader, absent["factors"])

    singleton = PAIR_NAMES.index("A8xA8")
    partner = PAIR_NAMES.index("ExA8")
    _single_logits, _single, single_diag, single_audit = _forward(
        model, tokens, scales, action="P", capture_mlp10=True,
        remove_terms=(singleton,), absent_capture=absent)
    _joint_logits, _joint, joint_diag, joint_audit = _forward(
        model, tokens, scales, action="P", capture_mlp10=True,
        remove_terms=(singleton, partner), absent_capture=absent)

    diagnostics = (direct_diag, absent_diag, native_diag, score_diag)
    passed = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and max(row["factor_reconstruction_max"] for row in diagnostics) <= 1e-10
        and max(row["raw_source_relative_squared"] for row in diagnostics)
        <= DEPLOYED_BF16_BAR
        and max(row["normalized_closure_relative_squared"] for row in diagnostics) <= 1e-12
        and max(row["float32_mlp10_closure"] for row in diagnostics) <= 1e-10
        and max(row["deployed_mlp10_relative_squared"] for row in diagnostics)
        <= DEPLOYED_BF16_BAR
        and closure["score_delta_float32_closure"] <= 1e-10
        and closure["score_delta_deployed_relative_squared"] <= DEPLOYED_BF16_BAR
        and float(native_difference.abs().max()) == 0.0 and native_relative <= 1e-12
        and pair_enumeration_error <= 1e-10 and pair_output_error <= 1e-10
        and attributions.ndim == 1 and attributions.numel() == len(SOURCE_PAIRS)
        and bool(torch.isfinite(attributions).all())
        and single_audit["mlp10_term_patches"] == 1
        and joint_audit["mlp10_term_patches"] == 1
        and single_diag["term_edit_rms"] > 0 and joint_diag["term_edit_rms"] > 0)
    if not passed:
        raise RuntimeError("rung507 CUDA smoke failed")
    print(json.dumps({
        "status": "smoke_passed", "rung": 507,
        "scientific_outcomes_retained": False,
        "native_replay_logit_max_abs": float(native_difference.abs().max()),
        "native_replay_relative_squared": native_relative,
        "factor_reconstruction_max": max(
            row["factor_reconstruction_max"] for row in diagnostics),
        "pair_enumeration_relative_squared": pair_enumeration_error,
        "pair_output_relative_squared": pair_output_error,
        "score_delta_float32_closure": closure["score_delta_float32_closure"],
        "score_delta_deployed_relative_squared": closure[
            "score_delta_deployed_relative_squared"],
        "gradient_attribution_count": int(attributions.numel()),
        "singleton_patch_live": single_diag["term_edit_rms"] > 0,
        "joint_patch_live": joint_diag["term_edit_rms"] > 0,
        "full_forwards": 6, "backwards": 1,
    }, indent=2, sort_keys=True))


def main():
    started = time.time()
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = \
        validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        assert len(NAMED_SOURCES) == 22 and len(SOURCE_PAIRS) == 253
        assert 1369 + 248 * 8 + 500 * 8 + 748 * math.comb(8, 2) == 28297
        print(json.dumps({
            "status": "dry_run_passed", "rung": 507, "model_loaded": False,
            "outcomes_opened": False, "named_sources": list(NAMED_SOURCES),
            "pair_count": len(SOURCE_PAIRS), "maximum_conditional_forwards": 28297,
            "maximum_discovery_backwards": 1240,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung507 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    collections = {}
    collections["gradient_discovery"] = collect_gradient(model, rows, task_masks, scales)
    calibration = _calibration(
        collections["gradient_discovery"]["base_task"],
        collections["gradient_discovery"]["source_task"],
        collections["gradient_discovery"]["task_counts"], DISCOVERY)
    calibration_ok = state_parent.calibration_holds(calibration)
    candidates, discovery_checks = discover_terms(collections["gradient_discovery"])
    discovery_identifying = 2 <= len(candidates) <= MAX_TERMS

    confirmed = []
    confirmation_checks = {}
    validated = []
    validation_checks = {}
    rules = {}
    composition_checks = {}
    predictable_pairs = []
    same_output_pairs = []
    relationships = {}
    if calibration_ok and discovery_identifying:
        candidate_indices = tuple((PAIR_NAMES.index(term),) for term in candidates)
        collections["finite_confirmation"] = collect_finite(
            model, rows, task_masks, circuit_masks, (), scales, CONFIRMATION,
            ((),) + candidate_indices)
        confirmed, confirmation_checks = confirm_terms(
            collections["gradient_discovery"], collections["finite_confirmation"], candidates)
        if 2 <= len(confirmed) <= MAX_TERMS:
            term_index = {term: PAIR_NAMES.index(term) for term in confirmed}
            term_pairs = tuple(itertools.combinations(confirmed, 2))
            pair_specs = tuple((term_index[left], term_index[right]) for left, right in term_pairs)
            collections["pair_confirmation"] = collect_finite(
                model, rows, task_masks, circuit_masks, (), scales, CONFIRMATION, pair_specs)
            for left, right in term_pairs:
                rules[f"{left}+{right}"] = fit_composition(
                    collections["finite_confirmation"], collections["pair_confirmation"], left, right)
                relationships[f"{left}+{right}"] = _pair_relationship(
                    term_index[left], term_index[right])
            validation_specs = ((),) + tuple((term_index[term],) for term in confirmed) + pair_specs
            collections["validation"] = collect_finite(
                model, rows, task_masks, circuit_masks, validation_tags, scales,
                VALIDATION, validation_specs)
            validated, validation_checks = validate_terms(
                collections["finite_confirmation"], collections["validation"], confirmed)
            for left, right in term_pairs:
                name = f"{left}+{right}"
                if left in validated and right in validated and rules[name]["identified"]:
                    composition_checks[name] = score_composition(
                        collections["validation"], left, right, rules[name])
                    if composition_checks[name]["holds"]:
                        predictable_pairs.append(name)
                    if composition_checks[name]["holds"] and composition_checks[name]["same_output"]:
                        same_output_pairs.append(name)

    all_collections = list(collections.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and _phase_instrument(collections["gradient_discovery"], gradient=True)
        and all(_phase_instrument(collection) for name, collection in collections.items()
                if name != "gradient_discovery"))
    pred_b = bool(pred_a and calibration_ok and discovery_identifying)
    pred_c = bool(pred_b and 2 <= len(confirmed) <= MAX_TERMS)
    pred_d = bool(pred_c and len(validated) >= 2)
    pred_e = bool(pred_d and predictable_pairs)
    strong_null = not (pred_a and pred_b and pred_c and pred_d and pred_e)

    if not pred_a:
        next_step = "repair_algebra_or_intervention_instrument_only"
    elif not calibration_ok:
        next_step = "stop_mlp10_assay_preserve_validated_score_gauge"
    elif not discovery_identifying and len(candidates) < 2:
        next_step = "coupled_factor_output_dictionary_with_finite_tests_required"
    elif not discovery_identifying:
        next_step = "add_independent_tasks_without_best_eight_selection"
    elif not pred_c:
        next_step = "replace_gradient_screen_with_registered_input_source_family_factorial"
    elif not pred_d:
        next_step = "preserve_term_screen_as_corpus_specific"
    elif not pred_e:
        next_step = "model_higher_order_term_state_dependence_before_extraction"
    else:
        next_step = "build_executable_mlp10_term_replacement_and_heldout_circuit_edits"

    bundle_payload = {
        "schema": "rung507_exact_mlp10_source_pair_causal_split_stats_v1",
        "gradient_discovery": {
            key: collections["gradient_discovery"][key] for key in (
                "attribution_sums", "full_sums", "gradient_counts",
                "base_task", "source_task", "task_counts",
            )
        },
        "finite": {name: _bundle_finite(collection) for name, collection in collections.items()
                   if name != "gradient_discovery"},
        "raw_tokens_logits_hidden_states_or_weights_included": False,
        "validation_opened": "validation" in collections,
    }
    torch.save(bundle_payload, BUNDLE)
    result = {
        "status": "complete", "rung": 507,
        "claim_level": "exact_named_bilinear_term_finite_causal_split_not_executable_replacement",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "named_sources": list(NAMED_SOURCES),
        "pair_names": list(PAIR_NAMES), "score_sources": list(SOURCES),
        "calibration": calibration,
        "diagnostics": {name: collection["diagnostics"] for name, collection in collections.items()},
        "analysis": {
            "discovery_candidates": candidates, "discovery_identifying": discovery_identifying,
            "discovery_checks": discovery_checks,
            "confirmed_terms": confirmed, "confirmation_checks": confirmation_checks,
            "validated_terms": validated, "validation_checks": validation_checks,
            "composition_rules": rules, "composition_checks": composition_checks,
            "predictable_composition_pairs": predictable_pairs,
            "same_output_pairs": same_output_pairs,
            "input_sharing_relationships": relationships,
        },
        'pred_a_exact_live_decomposition_and_intervention_instrument': pred_a,
        'pred_b_sparse_source_stable_gradient_screen': pred_b,
        'pred_c_at_least_two_terms_finitely_confirm': pred_c,
        'pred_d_at_least_two_terms_validate_new_documents': pred_d,
        'pred_e_at_least_one_pair_composition_predicts_validation': pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "full_forwards": sum(sum(row["diagnostics"]["calls"].values())
                                 for row in all_collections),
            "backwards": collections["gradient_discovery"]["diagnostics"]["backwards"],
            "candidate_count_k": len(candidates), "confirmed_count_q": len(confirmed),
            "confirming_term_pair_count": math.comb(len(confirmed), 2),
            "maximum_conditional_forwards": 28297,
            "fitted_vectors": 0,
            "fitted_scalars": sum(rule["kind"] == "one_scalar_interaction"
                                  for rule in rules.values()),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started, "next_step": next_step,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 507,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e, "strong_null": strong_null,
        "discovery_candidates": candidates, "confirmed_terms": confirmed,
        "validated_terms": validated, "predictable_pairs": predictable_pairs,
        "same_output_pairs": same_output_pairs,
        "execution_price": result["execution_price"], "next_step": next_step,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
