#!/usr/bin/env python3
"""RUNG512 -- localize exact MLP10 branch equivalence at attention11/MLP11."""

# BQGATE: EXPERIMENT
# pred_a: exact branch, consumer-capture, question-basis, and patch instrument is live
# pred_b: at least one fixed consumer-local relation passes discovery
# pred_c: a discovery relation predicts fresh documents without refitting
# pred_d: a confirmed relation passes bidirectional physical consumer substitution
# pred_e: a fixed MLP11 question-form relation passes discovery and confirmation

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import tiktoken
import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import mlp10_score_change_three_branch_factorial_rung511 as r511


PREREG = POLY / "MLP10_BRANCH_FIRST_CONSUMER_QUOTIENT_RUNG512_PREREGISTRATION.md"
PREFLIGHT_ADDENDUM = POLY / "MLP10_BRANCH_FIRST_CONSUMER_QUOTIENT_RUNG512_PREFLIGHT_ADDENDUM.md"
R511_RESULT = ROOT / "mlp10_score_change_three_branch_factorial_rung511_results.json"
R511_BUNDLE = ROOT / "mlp10_score_change_three_branch_factorial_rung511_bundle.pt"
R511_SOURCE = ROOT / "ops/mlp10_score_change_three_branch_factorial_rung511.py"
R511_PREREG = POLY / "MLP10_SCORE_CHANGE_THREE_BRANCH_FACTORIAL_RUNG511_PREREGISTRATION.md"
QUESTION_WRITER_RESULT = ROOT / "slice_writers_results.json"
QUESTION_WRITER_SOURCE = ROOT / "slice_writers.py"
QUESTION_PRODUCT_RESULT = POLY / "question_one_product_results.json"
QUESTION_PRODUCT_SOURCE = POLY / "question_one_product.py"
OUT = ROOT / "mlp10_branch_first_consumer_quotient_rung512_results.json"
BUNDLE = ROOT / "mlp10_branch_first_consumer_quotient_rung512_bundle.pt"

HASHES = {
    PREREG: "b72ab252feb82132602f1f674f594eded0eb419d54319aa7801ad2293f17daf8",
    PREFLIGHT_ADDENDUM: "d31e4fa39273f91afd7c09872f995d285133d7759f7f4bf0700870a5d64d68bd",
    R511_RESULT: "39a6afc592ceea8ed3f79d2928333eb70442ca63f5d147f61635649e57fca6d4",
    R511_BUNDLE: "16a70cb757ba97a6bc72b1b5bf2a35eaae4b7c5538b474254cad4beabb377a6e",
    R511_SOURCE: "6d07301b253c1216ea24e310eb82e1deab5c18baa3ce120b590cfa7fdba95031",
    R511_PREREG: "95a296478a5adc21ef0ef9bf8a1762ddd86e8f1312258733ce1de2eb2d9b4cd4",
    QUESTION_WRITER_RESULT: "f3394570e3122f8fee84f9e30b51574367549a68489a6c9505c67785b72b3cde",
    QUESTION_WRITER_SOURCE: "0b534adeceabd1cedec6977470c277a4dfafa786f787323f8e1f9fba2b0a04ee",
    QUESTION_PRODUCT_RESULT: "f8f58fd96b37eb23f95dc69b140b7b1c5edf9d708f247c3935e502d3ce03a2f5",
    QUESTION_PRODUCT_SOURCE: "4ff0fd56983818dc129d13244db092bf3aa3522f818fd5df71e4f310de5b2f9a",
}

DISCOVERY = r511.DISCOVERY
CONFIRMATION = r511.CONFIRMATION
N_ACTIONS = r511.N_ACTIONS
N_SUBSETS = r511.N_SUBSETS
N_NODES = r511.N_NODES
PRIMARY_SITES = ("a11", "m11", "q11")
GRAM_CHANNELS = ("source", "a11", "m11", "q11", "q11_question")
WINDOWS = ("half0", "half1", "pooled")
QUESTION_EIGENVALUES = (144.8641, -73.8464)
QUESTION_SUPPORT = {
    "discovery": {"half0": 40, "half1": 57},
    "confirmation": {"half0": 26, "half1": 36},
}
TASK_CELLS = r511.r510.r509.parent.TASK_CELLS
COPY_CELLS = ("near_positive", "far_positive", "one_predecessor_positive",
              "multiple_predecessor_positive")
ENCODER = tiktoken.get_encoding("gpt2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def question_token_mask() -> torch.Tensor:
    return torch.tensor([
        bool(re.match(r"^\?$| \?$", ENCODER.decode([token])))
        for token in range(50257)
    ], dtype=torch.bool)


@torch.no_grad()
def build_question_basis(model) -> dict:
    device = next(model.parameters()).device
    token_mask = question_token_mask().to(device)
    # The published form was derived from the checkpoint's stored float32
    # tensors.  The execution model is deliberately bf16 and must not silently
    # redefine that archived semantic object by downcasting these weights.
    state = torch.load(
        facade.DEFAULT_SNAPSHOT / "pytorch_model.bin", map_location="cpu",
        weights_only=True, mmap=True)
    unembedding = state["lm_head.weight"][:50257].to(device=device, dtype=torch.float32)
    direction = unembedding[token_mask].mean(0)
    direction = direction / direction.norm()
    left = state["transformer.h.11.mlp.Left.weight"].to(device=device, dtype=torch.float32)
    right = state["transformer.h.11.mlp.Right.weight"].to(device=device, dtype=torch.float32)
    down = state["transformer.h.11.mlp.Down.weight"].to(device=device, dtype=torch.float32)
    output_weight = direction @ down
    raw = left.T @ (output_weight[:, None] * right)
    del state, unembedding, left, right, down, output_weight
    symmetric = 0.5 * (raw + raw.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    order = eigenvalues.abs().argsort(descending=True)[:2]
    values = eigenvalues[order].contiguous()
    vectors = eigenvectors[:, order].contiguous()
    error = max(abs(float(values[index]) - QUESTION_EIGENVALUES[index]) for index in range(2))
    return {
        "direction": direction.contiguous(), "values": values,
        "vectors": vectors, "maximum_archived_eigenvalue_error": error,
    }


def question_scalar(state: torch.Tensor, basis: dict) -> torch.Tensor:
    coordinates = state.float() @ basis["vectors"]
    return (coordinates.square() * basis["values"]).sum(-1)


@torch.no_grad()
def _consumer_call(model, callable_):
    captured = {}
    block = model.transformer.h[11]

    def attention_pre(_module, arguments):
        captured["a11_input"] = arguments[0].detach().clone()

    def attention_post(_module, _arguments, output):
        captured["a11"] = output[0].detach().clone()

    def mlp_pre(_module, arguments):
        captured["m11_input"] = arguments[0].detach().clone()

    def mlp_post(_module, _arguments, output):
        captured["m11"] = output.detach().clone()

    handles = [
        block.attn.register_forward_pre_hook(attention_pre),
        block.attn.register_forward_hook(attention_post),
        block.mlp.register_forward_pre_hook(mlp_pre),
        block.mlp.register_forward_hook(mlp_post),
    ]
    try:
        result = callable_()
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"a11_input", "a11", "m11_input", "m11"}:
        raise RuntimeError(f"consumer capture changed: {sorted(captured)}")
    return result, captured


def _new_gram_collection() -> dict:
    return {
        "gram": {
            window: {channel: torch.zeros(N_NODES, N_NODES, dtype=torch.float64)
                     for channel in GRAM_CHANNELS}
            for window in WINDOWS
        },
        "base_energy": {
            window: {channel: torch.zeros(N_NODES, dtype=torch.float64)
                     for channel in GRAM_CHANNELS}
            for window in WINDOWS
        },
        "counts": {
            window: {channel: torch.zeros(N_NODES, dtype=torch.float64)
                     for channel in GRAM_CHANNELS}
            for window in WINDOWS
        },
    }


def _masked_flat(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value[mask].reshape(-1)
    return value[mask].reshape(-1)


@torch.no_grad()
def _update_grams(accumulator: dict, effects: dict, baselines: dict,
                  copy_mask: torch.Tensor, question_mask: torch.Tensor, half: str) -> None:
    masks = {channel: copy_mask for channel in GRAM_CHANNELS}
    masks["q11_question"] = question_mask
    for channel in GRAM_CHANNELS:
        mask = masks[channel]
        if int(mask.sum()) == 0:
            continue
        response_rows = [_masked_flat(effects[channel][node], mask).float()
                         for node in range(N_NODES)]
        baseline_rows = [_masked_flat(baselines[channel][node], mask).float()
                         for node in range(N_NODES)]
        width = response_rows[0].numel()
        if width == 0 or any(row.numel() != width for row in response_rows + baseline_rows):
            raise RuntimeError("consumer response width changed")
        response = torch.stack(response_rows)
        gram = (response @ response.T).double().cpu()
        energy = torch.stack([row.square().sum() for row in baseline_rows]).double().cpu()
        count = torch.full((N_NODES,), width, dtype=torch.float64)
        for window in (half, "pooled"):
            accumulator["gram"][window][channel] += gram
            accumulator["base_energy"][window][channel] += energy
            accumulator["counts"][window][channel] += count


def _empty_diagnostics() -> dict:
    row = r511._empty_diagnostics()
    row.update({
        "consumer_forward_captures": 0,
        "consumer_forward_captures_expected": 0,
        "consumer_forward_captures_exact": False,
        "question_support": {window: 0 for window in ("half0", "half1")},
    })
    return row


@torch.no_grad()
def collect_consumers(model, rows, task_masks, circuit_masks, circuit_tags,
                      scales, bounds, basis):
    lo, hi, split = bounds
    documents = hi - lo
    task = torch.zeros(N_ACTIONS, len(r511.ARMS), documents, len(TASK_CELLS),
                       dtype=torch.float64)
    counts = torch.zeros(documents, len(TASK_CELLS), dtype=torch.float64)
    base_task = torch.zeros_like(counts)
    circuit_sums = torch.zeros(
        N_ACTIONS, len(r511.ARMS), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    consumer = _new_gram_collection()
    diagnostics = _empty_diagnostics()
    device = next(model.parameters()).device
    mlp10 = model.transformer.h[r511.r510.r509.parent.TARGET].mlp
    qtokens = question_token_mask()

    for start in range(lo, hi, r511.r510.r509.parent.BATCH):
        stop = start + r511.r510.r509.parent.BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in TASK_CELLS}
        copy_mask = masks["all_positive"].to(device)
        targets = batch_rows[:, 1:]
        valid = torch.ones_like(targets, dtype=torch.bool)
        valid[:, :64] = False
        qmask_cpu = qtokens[targets] & valid
        qmask = qmask_cpu.to(device)
        half = "half0" if start < split else "half1"
        diagnostics["question_support"][half] += int(qmask.sum())

        direct_logits, _, direct_diag, _ = r511.r510.r509.parent._forward(
            model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        r511.r510.r509._update_diagnostics(diagnostics, direct_diag)
        absent_logits, absent, absent_diag, _ = r511._captured_forward(
            model, tokens, scales, action="P", absent=True)
        diagnostics["calls"]["analytical"] += 1
        diagnostics["hooks"] += 1
        r511.r510.r509._update_diagnostics(diagnostics, absent_diag)
        base_task[local:local + len(batch_rows)] = r511.r510.r509.parent._task_sums(
            r511.r510.r509.parent._nll(absent_logits, batch_rows).detach().cpu().unsqueeze(0),
            masks)[0]

        action_nll = []
        effects = {channel: [None] * N_NODES for channel in GRAM_CHANNELS}
        baselines = {channel: [None] * N_NODES for channel in GRAM_CHANNELS}
        for action_index, source in enumerate(r511.r510.r509.parent.SOURCES):
            (current_result, current_consumer) = _consumer_call(
                model, lambda source=source: r511._captured_forward(
                    model, tokens, scales, action=source))
            logits, current, current_diag, _ = current_result
            diagnostics["calls"]["analytical"] += 1
            diagnostics["hooks"] += 1
            diagnostics["consumer_forward_captures"] += 1
            r511.r510.r509._update_diagnostics(diagnostics, current_diag)
            r511.r510.r509.parent._score_delta_closure(diagnostics, current, absent)
            if source == "N":
                diagnostics["native_replay_logit_max_abs"] = max(
                    diagnostics["native_replay_logit_max_abs"],
                    float((logits.float() - direct_logits.float()).abs().max()))
                diagnostics["native_replay_relative_squared"] = max(
                    diagnostics["native_replay_relative_squared"],
                    r511._relative_squared(direct_logits, logits))
            branches, branch_diag = r511.deployed_branches(mlp10, absent, current)
            diagnostics["four_corner_replays"] += 1
            r511._update_branch_diagnostics(diagnostics, branch_diag)
            current_q = question_scalar(current_consumer["m11_input"], basis)
            nll_rows = [r511.r510.r509.parent._nll(logits, batch_rows).detach().cpu()]
            for subset_index in range(N_SUBSETS):
                node = action_index * N_SUBSETS + subset_index
                delta10 = r511.subset_output(branches, subset_index)
                replacement = current["deployed_write"].float() - delta10
                (removed_result, removed_consumer) = _consumer_call(
                    model,
                    lambda source=source, replacement=replacement: (
                        r511.r510.r509.parent.score_parent.run_forward(
                            model, tokens, action=source, scales=scales,
                            patch_writes={"m10": replacement.to(current["deployed_write"].dtype)})))
                edited_logits, _captures, patch_diag, patch_audit = removed_result
                diagnostics["calls"]["analytical"] += 1
                diagnostics["consumer_forward_captures"] += 1
                diagnostics["subset_patches"] += patch_audit["patches"]
                edit_rms = patch_diag["patch_rms_max"]
                diagnostics["zero_term_edits"] += int(edit_rms <= 0)
                if edit_rms > 0:
                    diagnostics["minimum_nonzero_term_edit_rms"] = min(
                        diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
                removed_q = question_scalar(removed_consumer["m11_input"], basis)
                effects["source"][node] = delta10.float()
                effects["a11"][node] = (current_consumer["a11"].float()
                                                  - removed_consumer["a11"].float())
                effects["m11"][node] = (current_consumer["m11"].float()
                                                  - removed_consumer["m11"].float())
                effects["q11"][node] = current_q - removed_q
                effects["q11_question"][node] = current_q - removed_q
                base_values = {
                    "source": current["deployed_write"].float(),
                    "a11": current_consumer["a11"].float(),
                    "m11": current_consumer["m11"].float(),
                    "q11": current_q,
                    "q11_question": current_q,
                }
                for channel in GRAM_CHANNELS:
                    baselines[channel][node] = base_values[channel]
                nll_rows.append(
                    r511.r510.r509.parent._nll(edited_logits, batch_rows).detach().cpu())
            task[action_index, :, local:local + len(batch_rows)] = \
                r511.r510.r509.parent._task_sums(torch.stack(nll_rows), masks)
            action_nll.append(torch.stack(nll_rows))

        _update_grams(consumer, effects, baselines, copy_mask, qmask, half)
        counts[local:local + len(batch_rows)] = torch.stack(
            [masks[cell].sum(1).double() for cell in TASK_CELLS], -1)
        matrix, observed = r511.r510.r509.parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        circuit_counts += observed
        for action_index, nll_stack in enumerate(action_nll):
            circuit_sums[action_index] += torch.matmul(
                nll_stack.view(len(r511.ARMS), -1).double(), matrix.T,
            ).view(len(r511.ARMS), 2, 2, len(circuit_tags))

    batches = documents // r511.r510.r509.parent.BATCH
    diagnostics["calls_expected"] = {
        "direct": batches, "analytical": batches * (1 + N_ACTIONS * (1 + N_SUBSETS))}
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["hooks_expected"] = batches * (1 + N_ACTIONS)
    diagnostics["hooks_exact"] = diagnostics["hooks"] == diagnostics["hooks_expected"]
    diagnostics["four_corner_replays_expected"] = batches * N_ACTIONS
    diagnostics["four_corner_replays_exact"] = (
        diagnostics["four_corner_replays"] == diagnostics["four_corner_replays_expected"])
    diagnostics["subset_patches_expected"] = batches * N_ACTIONS * N_SUBSETS
    diagnostics["subset_patches_exact"] = (
        diagnostics["subset_patches"] == diagnostics["subset_patches_expected"])
    diagnostics["patches"] = diagnostics["subset_patches"]
    diagnostics["patches_expected"] = diagnostics["subset_patches_expected"]
    diagnostics["patches_exact"] = diagnostics["subset_patches_exact"]
    diagnostics["consumer_forward_captures_expected"] = batches * N_ACTIONS * (1 + N_SUBSETS)
    diagnostics["consumer_forward_captures_exact"] = (
        diagnostics["consumer_forward_captures"]
        == diagnostics["consumer_forward_captures_expected"])
    return {
        "bounds": bounds, "arms": r511.ARMS, "task": task,
        "task_counts": counts, "base_task": base_task, "source_task": task[:, 0],
        "circuit_tags": tuple(circuit_tags), "circuit_sums": circuit_sums,
        "circuit_counts": circuit_counts, "consumer": consumer,
        "diagnostics": diagnostics,
    }


def _gram_metrics(collection: dict, channel: str, left: int, right: int,
                  beta: float) -> dict:
    row = {"beta_left_from_right": beta, "windows": {}, "scale_holds": bool(.25 <= abs(beta) <= 4)}
    for window in WINDOWS:
        gram = collection["consumer"]["gram"][window][channel]
        base = collection["consumer"]["base_energy"][window][channel]
        count = collection["consumer"]["counts"][window][channel]
        ll, rr, lr = float(gram[left, left]), float(gram[right, right]), float(gram[left, right])
        cosine = math.copysign(1.0, beta) * lr / max(math.sqrt(max(ll * rr, 0.0)), 1e-30)
        inverse = 1.0 / beta
        forward_sq = max(ll - 2 * beta * lr + beta * beta * rr, 0.0)
        backward_sq = max(rr - 2 * inverse * lr + inverse * inverse * ll, 0.0)
        effect_rms = [math.sqrt(max(float(gram[node, node]), 0.0) / max(float(count[node]), 1.0))
                      for node in (left, right)]
        base_rms = [math.sqrt(max(float(base[node]), 0.0) / max(float(count[node]), 1.0))
                    for node in (left, right)]
        relative_rms = [effect_rms[index] / max(base_rms[index], 1e-30) for index in range(2)]
        row["windows"][window] = {
            "cosine": cosine,
            "left_from_right_relative_residual": math.sqrt(forward_sq / max(ll, 1e-30)),
            "right_from_left_relative_residual": math.sqrt(backward_sq / max(rr, 1e-30)),
            "effect_rms": effect_rms, "intact_rms": base_rms,
            "effect_to_intact_rms": relative_rms,
            "material": bool(min(effect_rms) > 0 and min(relative_rms) >= 1e-4),
        }
    return row


def _metrics_hold(metrics: dict, cosine: float, residual: float,
                  windows=("half0", "half1")) -> bool:
    return bool(metrics["scale_holds"] and all(
        metrics["windows"][window]["material"]
        and metrics["windows"][window]["cosine"] >= cosine
        and max(metrics["windows"][window]["left_from_right_relative_residual"],
                metrics["windows"][window]["right_from_left_relative_residual"]) <= residual
        for window in windows))


def _fit_beta(collection: dict, channel: str, left: int, right: int) -> float:
    gram = collection["consumer"]["gram"]["half0"][channel]
    return float(gram[left, right] / gram[right, right].clamp_min(1e-30))


def discover_relations(collection: dict) -> tuple[list[dict], dict]:
    candidates, checks = [], {}
    for subset in range(N_SUBSETS):
        nodes = [action * N_SUBSETS + subset for action in range(N_ACTIONS)]
        for left, right in itertools.combinations(nodes, 2):
            source_beta = _fit_beta(collection, "source", left, right)
            source_metrics = _gram_metrics(collection, "source", left, right, source_beta)
            source_holds = _metrics_hold(source_metrics, .85, .55)
            for site in PRIMARY_SITES:
                beta = _fit_beta(collection, site, left, right)
                metrics = _gram_metrics(collection, site, left, right, beta)
                holds = _metrics_hold(metrics, .85, .55)
                semantic = None
                if site == "q11":
                    semantic = _gram_metrics(
                        collection, "q11_question", left, right, beta)
                    holds &= _metrics_hold(semantic, .70, .65)
                key = f"{r511.NODE_NAMES[left]} <-> {r511.NODE_NAMES[right]} @ {site}"
                checks[key] = {
                    "consumer": metrics, "question_tokens": semantic,
                    "source_control": source_metrics, "source_holds": source_holds,
                    "holds": bool(holds),
                }
                if holds:
                    candidates.append({
                        "left_node": left, "right_node": right,
                        "left_name": r511.NODE_NAMES[left],
                        "right_name": r511.NODE_NAMES[right],
                        "subset_index": subset, "subset_name": r511.SUBSET_NAMES[subset],
                        "site": site, "beta_left_from_right": beta,
                        "type": "transported" if source_holds else "consumer_convergence",
                    })
    return candidates, {
        "relations": 42, "consumer_tests": 126,
        "candidate_count": len(candidates),
        "consumer_convergence_count": sum(c["type"] == "consumer_convergence" for c in candidates),
        "checks": checks,
    }


def confirm_relations(collection: dict, candidates: list[dict]) -> tuple[list[dict], dict]:
    confirmed, checks = [], {}
    for candidate in candidates:
        metrics = _gram_metrics(
            collection, candidate["site"], candidate["left_node"],
            candidate["right_node"], candidate["beta_left_from_right"])
        holds = _metrics_hold(metrics, .75, .65)
        semantic = None
        if candidate["site"] == "q11":
            semantic = _gram_metrics(
                collection, "q11_question", candidate["left_node"],
                candidate["right_node"], candidate["beta_left_from_right"])
            holds &= _metrics_hold(semantic, .65, .75)
        key = (f"{candidate['left_name']} <-> {candidate['right_name']}"
               f" @ {candidate['site']}")
        checks[key] = {"consumer": metrics, "question_tokens": semantic, "holds": bool(holds)}
        if holds:
            confirmed.append(candidate)
    return confirmed, checks


def replacement_write(site: str, target_consumer: dict, target_delta: torch.Tensor,
                      donor_delta: torch.Tensor, scale: float, basis: dict) -> tuple[str, torch.Tensor]:
    if site in ("a11", "m11"):
        return site, (target_consumer[site].float() - target_delta.float()
                      + scale * donor_delta.float())
    if site == "q11":
        adjustment = scale * donor_delta.float() - target_delta.float()
        return "m11", (target_consumer["m11"].float()
                       + adjustment.unsqueeze(-1) * basis["direction"].float())
    raise ValueError(f"unknown consumer site: {site}")


def _physical_empty(candidate_count: int, documents: int, circuit_count: int) -> dict:
    directions = 2 * candidate_count
    return {
        "intact_task": torch.zeros(N_ACTIONS, documents, len(TASK_CELLS), dtype=torch.float64),
        "substitution_task": torch.zeros(directions, documents, len(TASK_CELLS), dtype=torch.float64),
        "task_counts": torch.zeros(documents, len(TASK_CELLS), dtype=torch.float64),
        "intact_circuit_sums": torch.zeros(N_ACTIONS, 2, 2, circuit_count, dtype=torch.float64),
        "substitution_circuit_sums": torch.zeros(directions, 2, 2, circuit_count, dtype=torch.float64),
        "circuit_counts": torch.zeros(2, 2, circuit_count, dtype=torch.float64),
        "intact_question": torch.zeros(N_ACTIONS, 2, dtype=torch.float64),
        "substitution_question": torch.zeros(directions, 2, dtype=torch.float64),
        "question_count": 0,
    }


@torch.no_grad()
def collect_physical(model, rows, task_masks, circuit_masks, circuit_tags,
                     scales, bounds, basis, candidates):
    lo, hi, split = bounds
    data = _physical_empty(len(candidates), hi - lo, len(circuit_tags))
    data["bounds"] = bounds
    data["directions"] = []
    diagnostics = {
        "calls": 0, "calls_expected": 0, "calls_exact": False,
        "branch_patches": 0, "branch_patches_expected": 0,
        "consumer_patches": 0, "consumer_patches_expected": 0,
        "patches_exact": False, "zero_patch_edits": 0,
        "minimum_patch_rms": math.inf,
        "maximum_patch_capture_error": 0.0,
    }
    for candidate_index, candidate in enumerate(candidates):
        beta = candidate["beta_left_from_right"]
        data["directions"].extend([
            {"candidate": candidate_index, "side": "left_from_right",
             "target": candidate["left_node"], "donor": candidate["right_node"], "scale": beta},
            {"candidate": candidate_index, "side": "right_from_left",
             "target": candidate["right_node"], "donor": candidate["left_node"], "scale": 1.0 / beta},
        ])
    device = next(model.parameters()).device
    mlp10 = model.transformer.h[r511.r510.r509.parent.TARGET].mlp
    qtokens = question_token_mask()

    for start in range(lo, hi, r511.r510.r509.parent.BATCH):
        stop = start + r511.r510.r509.parent.BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in TASK_CELLS}
        targets = batch_rows[:, 1:]
        valid = torch.ones_like(targets, dtype=torch.bool)
        valid[:, :64] = False
        qmask = (qtokens[targets] & valid).to(device)
        absent_result, _absent_consumer = _consumer_call(
            model, lambda: r511._captured_forward(model, tokens, scales, action="P", absent=True))
        _absent_logits, absent, _absent_diag, _ = absent_result
        diagnostics["calls"] += 1

        action_current = {}
        node_delta = {site: [None] * N_NODES for site in PRIMARY_SITES}
        action_nll = {}
        for action_index, source in enumerate(r511.r510.r509.parent.SOURCES):
            current_result, current_consumer = _consumer_call(
                model, lambda source=source: r511._captured_forward(
                    model, tokens, scales, action=source))
            logits, current, _current_diag, _ = current_result
            diagnostics["calls"] += 1
            current_q = question_scalar(current_consumer["m11_input"], basis)
            current_nll = r511.r510.r509.parent._nll(logits, batch_rows).detach().cpu()
            action_nll[action_index] = current_nll
            data["intact_task"][action_index, local:local + len(batch_rows)] = \
                r511.r510.r509.parent._task_sums(current_nll.unsqueeze(0), masks)[0]
            if int(qmask.sum()):
                true_logits = logits.gather(-1, batch_rows[:, 1:].to(device).unsqueeze(-1)).squeeze(-1)
                data["intact_question"][action_index, 0] += float(current_nll.to(device)[qmask].sum())
                data["intact_question"][action_index, 1] += float(true_logits[qmask].sum())
            branches, _branch_diag = r511.deployed_branches(mlp10, absent, current)
            action_current[action_index] = {
                "consumer": current_consumer, "capture": current, "branches": branches,
                "q": current_q, "logits": logits,
            }
            for subset_index in range(N_SUBSETS):
                node = action_index * N_SUBSETS + subset_index
                delta10 = r511.subset_output(branches, subset_index)
                replacement = current["deployed_write"].float() - delta10
                removed_result, removed_consumer = _consumer_call(
                    model, lambda source=source, replacement=replacement: (
                        r511.r510.r509.parent.score_parent.run_forward(
                            model, tokens, action=source, scales=scales,
                            patch_writes={"m10": replacement.to(current["deployed_write"].dtype)})))
                _removed_logits, _captures, removed_diag, removed_audit = removed_result
                diagnostics["calls"] += 1
                diagnostics["branch_patches"] += removed_audit["patches"]
                branch_rms = removed_diag["patch_rms_max"]
                diagnostics["zero_patch_edits"] += int(branch_rms <= 0)
                if branch_rms > 0:
                    diagnostics["minimum_patch_rms"] = min(
                        diagnostics["minimum_patch_rms"], branch_rms)
                node_delta["a11"][node] = current_consumer["a11"].float() - removed_consumer["a11"].float()
                node_delta["m11"][node] = current_consumer["m11"].float() - removed_consumer["m11"].float()
                node_delta["q11"][node] = current_q - question_scalar(
                    removed_consumer["m11_input"], basis)

        data["task_counts"][local:local + len(batch_rows)] = torch.stack(
            [masks[cell].sum(1).double() for cell in TASK_CELLS], -1)
        matrix, observed = r511.r510.r509.parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        data["circuit_counts"] += observed
        for action_index, nll in action_nll.items():
            data["intact_circuit_sums"][action_index] += torch.matmul(
                nll.reshape(1, -1).double(), matrix.T).view(2, 2, len(circuit_tags))

        for direction_index, direction in enumerate(data["directions"]):
            candidate = candidates[direction["candidate"]]
            target_action, _target_subset = r511.node_parts(direction["target"])
            donor_action, _donor_subset = r511.node_parts(direction["donor"])
            target = action_current[target_action]
            key, replacement = replacement_write(
                candidate["site"], target["consumer"],
                node_delta[candidate["site"]][direction["target"]],
                node_delta[candidate["site"]][direction["donor"]],
                direction["scale"], basis)
            logits, captures, diag, audit = r511.r510.r509.parent.score_parent.run_forward(
                model, tokens, action=r511.r510.r509.parent.SOURCES[target_action],
                scales=scales, patch_writes={key: replacement.to(target["consumer"][key].dtype)},
                capture_keys=(key,))
            diagnostics["calls"] += 1
            diagnostics["consumer_patches"] += audit["patches"]
            patch_rms = diag["patch_rms_max"]
            diagnostics["zero_patch_edits"] += int(patch_rms <= 0)
            if patch_rms > 0:
                diagnostics["minimum_patch_rms"] = min(diagnostics["minimum_patch_rms"], patch_rms)
            diagnostics["maximum_patch_capture_error"] = max(
                diagnostics["maximum_patch_capture_error"],
                float((captures[key] - replacement.to(captures[key].dtype)).float().abs().max()))
            nll = r511.r510.r509.parent._nll(logits, batch_rows).detach().cpu()
            data["substitution_task"][direction_index, local:local + len(batch_rows)] = \
                r511.r510.r509.parent._task_sums(nll.unsqueeze(0), masks)[0]
            data["substitution_circuit_sums"][direction_index] += torch.matmul(
                nll.reshape(1, -1).double(), matrix.T).view(2, 2, len(circuit_tags))
            if int(qmask.sum()):
                true_logits = logits.gather(-1, batch_rows[:, 1:].to(device).unsqueeze(-1)).squeeze(-1)
                data["substitution_question"][direction_index, 0] += float(nll.to(device)[qmask].sum())
                data["substitution_question"][direction_index, 1] += float(true_logits[qmask].sum())
        data["question_count"] += int(qmask.sum())

    batches = (hi - lo) // r511.r510.r509.parent.BATCH
    diagnostics["calls_expected"] = batches * (1 + N_ACTIONS * (1 + N_SUBSETS)
                                                 + 2 * len(candidates))
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["branch_patches_expected"] = batches * N_ACTIONS * N_SUBSETS
    diagnostics["consumer_patches_expected"] = batches * 2 * len(candidates)
    diagnostics["patches_exact"] = bool(
        diagnostics["branch_patches"] == diagnostics["branch_patches_expected"]
        and diagnostics["consumer_patches"] == diagnostics["consumer_patches_expected"])
    data["diagnostics"] = diagnostics
    return data


def _window_slice(bounds, window: str) -> slice:
    lo, hi, split = bounds
    if window == "half0":
        return slice(0, split - lo)
    if window == "half1":
        return slice(split - lo, hi - lo)
    return slice(0, hi - lo)


def _physical_task_damage(physical: dict, direction_index: int, window: str) -> torch.Tensor:
    direction = physical["directions"][direction_index]
    target_action, _ = r511.node_parts(direction["target"])
    sl = _window_slice(physical["bounds"], window)
    numerator = (physical["substitution_task"][direction_index, sl]
                 - physical["intact_task"][target_action, sl]).sum(0)
    denominator = physical["task_counts"][sl].sum(0).clamp_min(1)
    values = numerator / denominator
    return values[[TASK_CELLS.index(cell) for cell in COPY_CELLS]]


def _physical_off_target(physical: dict, direction_index: int, window: str) -> float:
    direction = physical["directions"][direction_index]
    target_action, _ = r511.node_parts(direction["target"])
    sl = _window_slice(physical["bounds"], window)
    index = TASK_CELLS.index("off_target")
    numerator = (physical["substitution_task"][direction_index, sl, index]
                 - physical["intact_task"][target_action, sl, index]).sum()
    denominator = physical["task_counts"][sl, index].sum().clamp_min(1)
    return float(numerator / denominator)


def _physical_circuit_damage(physical: dict, direction_index: int, window: str) -> torch.Tensor:
    direction = physical["directions"][direction_index]
    target_action, _ = r511.node_parts(direction["target"])
    if window == "pooled":
        sub = physical["substitution_circuit_sums"][direction_index].sum(0)
        intact = physical["intact_circuit_sums"][target_action].sum(0)
        counts = physical["circuit_counts"].sum(0)
    else:
        half = {"half0": 0, "half1": 1}[window]
        sub = physical["substitution_circuit_sums"][direction_index, half]
        intact = physical["intact_circuit_sums"][target_action, half]
        counts = physical["circuit_counts"][half]
    effect = (sub - intact) / counts.clamp_min(1)
    return effect[0] - effect[1]


def score_physical(physical: dict, confirmation: dict, candidates: list[dict]) -> tuple[list[dict], dict]:
    passing, checks = [], {}
    for candidate_index, candidate in enumerate(candidates):
        row = {"directions": {}, "holds": True}
        for local, side in enumerate(("left_from_right", "right_from_left")):
            direction_index = 2 * candidate_index + local
            target_node = physical["directions"][direction_index]["target"]
            side_row = {"windows": {}, "holds": True}
            for window in WINDOWS:
                task_damage = _physical_task_damage(physical, direction_index, window)
                circuit_damage = _physical_circuit_damage(physical, direction_index, window)
                target_task = r511._task_vector(confirmation, target_node, window)
                target_circuit = r511._circuit_vector(confirmation, target_node, window)
                task_ratio = float(torch.linalg.vector_norm(task_damage)
                                   / torch.linalg.vector_norm(target_task).clamp_min(1e-30))
                circuit_ratio = float(torch.linalg.vector_norm(circuit_damage)
                                      / torch.linalg.vector_norm(target_circuit).clamp_min(1e-30))
                off_target = _physical_off_target(physical, direction_index, window)
                holds = bool(task_ratio <= .50 and circuit_ratio <= .50
                             and abs(off_target) <= .002)
                side_row["windows"][window] = {
                    "task_damage_to_full_removal": task_ratio,
                    "circuit_damage_to_full_removal": circuit_ratio,
                    "off_target_ce_damage_nat": off_target, "holds": holds,
                }
                if window in ("half0", "half1"):
                    side_row["holds"] &= holds
            row["directions"][side] = side_row
            row["holds"] &= side_row["holds"]
        key = (f"{candidate['left_name']} <-> {candidate['right_name']}"
               f" @ {candidate['site']}")
        checks[key] = row
        if row["holds"]:
            passing.append(candidate)
    return passing, checks


def _instrument(collection: dict, basis: dict, expected_support: dict) -> bool:
    diagnostics = collection["diagnostics"]
    return bool(
        r511._instrument(collection)
        and diagnostics["consumer_forward_captures_exact"]
        and diagnostics["question_support"] == expected_support
        and basis["maximum_archived_eigenvalue_error"] <= 1e-3
        and all(torch.isfinite(collection["consumer"]["gram"][window][channel]).all()
                for window in WINDOWS for channel in GRAM_CHANNELS))


def _physical_instrument(physical: dict) -> bool:
    diagnostics = physical["diagnostics"]
    return bool(
        diagnostics["calls_exact"] and diagnostics["patches_exact"]
        and diagnostics["zero_patch_edits"] == 0
        and diagnostics["minimum_patch_rms"] > 0
        and diagnostics["maximum_patch_capture_error"] == 0.0)


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(R511_RESULT.read_text())
    if not (
        result.get("pred_a_exact_live_three_branch_factorial_instrument") is True
        and result.get("pred_b_fixed_same_subset_cross_action_discovery") is False
        and result.get("analysis", {}).get("discovery_summary", {}).get("candidate_count") == 0
        and result.get("execution_price", {}).get("full_forwards") == 2108
        and result.get("next_step")
        == "localize_exact_branches_at_first_downstream_consumer_including_mlp11_question_interface"
    ):
        raise RuntimeError("rung511 zero-relation route changed")
    writer = json.loads(QUESTION_WRITER_RESULT.read_text())
    product = json.loads(QUESTION_PRODUCT_RESULT.read_text())
    if not (writer.get("pred_a_sparse_60") is True
            and writer.get("pred_b_causal_edges") is True
            and writer.get("pred_c_attn_edge_is_circuit_head") is True
            and product.get("predictions", {}).get("A_pair_fp32_exact") is True
            and product.get("predictions", {}).get("B_pair_bf16_stable") is True):
        raise RuntimeError("archived question interface authority changed")
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = \
        r511.validate_inputs()
    qtokens = question_token_mask()
    support = {}
    for name, bounds in (("discovery", DISCOVERY), ("confirmation", CONFIRMATION)):
        lo, hi, split = bounds
        support[name] = {}
        for window, left, right in (("half0", lo, split), ("half1", split, hi)):
            targets = rows[left:right, 1:]
            valid = torch.ones_like(targets, dtype=torch.bool)
            valid[:, :64] = False
            support[name][window] = int((qtokens[targets] & valid).sum())
    if support != QUESTION_SUPPORT:
        raise RuntimeError(f"question support changed: {support}")
    return rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, {
        **metadata, "rung511_result_sha256": sha256(R511_RESULT),
        "rung511_bundle_sha256": sha256(R511_BUNDLE),
        "question_support": support, "primary_sites": list(PRIMARY_SITES),
        "consumer_tests": 126,
    }


def _toy_collection(seed=512) -> dict:
    generator = torch.Generator().manual_seed(seed)
    response = {
        window: {channel: .01 * torch.randn(N_NODES, 64, generator=generator)
                 for channel in GRAM_CHANNELS}
        for window in WINDOWS
    }
    for window in WINDOWS:
        # N::L and P::L become identical only at the first consumer.
        response[window]["a11"][N_SUBSETS] = -2 * response[window]["a11"][0]
    consumer = _new_gram_collection()
    for window in WINDOWS:
        for channel in GRAM_CHANNELS:
            values = response[window][channel].double()
            consumer["gram"][window][channel] = values @ values.T
            consumer["base_energy"][window][channel].fill_(64.0)
            consumer["counts"][window][channel].fill_(64.0)
    return {"consumer": consumer}


def dry_run() -> None:
    toy = _toy_collection()
    candidates, summary = discover_relations(toy)
    planted = [candidate for candidate in candidates
               if candidate["left_node"] == 0 and candidate["right_node"] == N_SUBSETS
               and candidate["site"] == "a11"]
    assert len(planted) == 1 and planted[0]["type"] == "consumer_convergence"
    confirmed, _checks = confirm_relations(toy, planted)
    assert len(confirmed) == 1
    basis = {"direction": torch.tensor([1.0, 0.0])}
    current = {"a11": torch.tensor([[[3.0, 4.0]]]), "m11": torch.tensor([[[5.0, 6.0]]])}
    key, write = replacement_write(
        "a11", current, torch.tensor([[[1.0, 2.0]]]),
        torch.tensor([[[2.0, 1.0]]]), .5, basis)
    assert key == "a11" and torch.equal(write, torch.tensor([[[3.0, 2.5]]]))
    assert summary["consumer_tests"] == 126
    assert 4216 + 2046 + 124 * 126 == 21886
    print(json.dumps({
        "status": "dry_run_passed", "rung": 512, "model_loaded": False,
        "outcomes_opened": False, "consumer_tests": 126,
        "planted_consumer_convergence_recovered": True,
        "maximum_conditional_forwards": 21886,
    }, indent=2, sort_keys=True))


def _bundle_collection(collection: dict) -> dict:
    return {key: value for key, value in collection.items() if key != "diagnostics"}


@torch.no_grad()
def gpu_smoke() -> None:
    rows, task_masks, circuit_masks, scales, discovery_tags, _validation_tags, _metadata = \
        validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    basis = build_question_basis(model)
    bounds = (500, 504, 502)
    collection = collect_consumers(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, bounds, basis)
    candidate = {
        "left_node": 0, "right_node": N_SUBSETS,
        "left_name": r511.NODE_NAMES[0], "right_name": r511.NODE_NAMES[N_SUBSETS],
        "subset_index": 0, "subset_name": r511.SUBSET_NAMES[0],
        "site": "a11", "beta_left_from_right": 1.0,
        "type": "smoke_only",
    }
    physical = collect_physical(
        model, rows, task_masks, circuit_masks, discovery_tags, scales,
        bounds, basis, [candidate])
    checks = {
        "weights": checkpoint.weights_sha256 == facade.WEIGHTS_SHA256,
        "consumer_collection": r511._instrument(collection)
                               and collection["diagnostics"]["consumer_forward_captures_exact"],
        "question_basis": basis["maximum_archived_eigenvalue_error"] <= 1e-3,
        "physical": _physical_instrument(physical),
        "all_twenty_eight_branch_patches": collection["diagnostics"]["subset_patches"] == 28,
        "all_twenty_eight_physical_branch_patches": physical["diagnostics"]["branch_patches"] == 28,
        "both_consumer_substitutions": physical["diagnostics"]["consumer_patches"] == 2,
    }
    passed = all(checks.values())
    print(json.dumps({
        "status": "smoke_passed" if passed else "smoke_failed", "rung": 512,
        "scientific_outcomes_retained": False, "checks": checks,
        "question_eigenvalues": [float(value) for value in basis["values"]],
        "collection_diagnostics": collection["diagnostics"],
        "physical_diagnostics": physical["diagnostics"],
        "full_forwards": sum(collection["diagnostics"]["calls"].values())
                         + physical["diagnostics"]["calls"],
        "backwards": 0,
    }, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(f"rung512 smoke failed: "
                           f"{sorted(name for name, value in checks.items() if not value)}")


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        dry_run()
        return
    if os.environ.get("BQLIB_GPU_SMOKE") == "1" or "--gpu-smoke" in sys.argv:
        gpu_smoke()
        return
    started = time.time()
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = \
        validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung512 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    basis = build_question_basis(model)
    collections = {}
    collections["discovery"] = collect_consumers(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, DISCOVERY, basis)
    discovery_calibration = r511.r510.r509.parent._calibration(
        collections["discovery"]["base_task"], collections["discovery"]["source_task"],
        collections["discovery"]["task_counts"], DISCOVERY)
    discovery_calibration_ok = r511.r510.r509.parent.state_parent.calibration_holds(
        discovery_calibration)
    candidates, discovery_summary = discover_relations(collections["discovery"])

    confirmed, confirmation_checks = [], {}
    confirmation_calibration, confirmation_calibration_ok = {}, False
    if discovery_calibration_ok and candidates:
        collections["confirmation"] = collect_consumers(
            model, rows, task_masks, circuit_masks, validation_tags, scales,
            CONFIRMATION, basis)
        confirmation_calibration = r511.r510.r509.parent._calibration(
            collections["confirmation"]["base_task"],
            collections["confirmation"]["source_task"],
            collections["confirmation"]["task_counts"], CONFIRMATION)
        confirmation_calibration_ok = r511.r510.r509.parent.state_parent.calibration_holds(
            confirmation_calibration)
        confirmed, confirmation_checks = confirm_relations(
            collections["confirmation"], candidates)

    physical_passing, physical_checks = [], {}
    physical = None
    if confirmation_calibration_ok and confirmed:
        physical = collect_physical(
            model, rows, task_masks, circuit_masks, validation_tags, scales,
            CONFIRMATION, basis, confirmed)
        physical_passing, physical_checks = score_physical(
            physical, collections["confirmation"], confirmed)

    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and discovery_calibration_ok
        and _instrument(collections["discovery"], basis, QUESTION_SUPPORT["discovery"])
        and ("confirmation" not in collections or (
            confirmation_calibration_ok
            and _instrument(collections["confirmation"], basis, QUESTION_SUPPORT["confirmation"])))
        and (physical is None or _physical_instrument(physical)))
    pred_b = bool(pred_a and candidates)
    pred_c = bool(pred_b and confirmation_calibration_ok and confirmed)
    pred_d = bool(pred_c and physical_passing)
    pred_e = bool(any(candidate["site"] == "q11" for candidate in confirmed))
    strong_null = not (pred_a and pred_b and pred_c and pred_d)
    if not pred_a:
        next_step = "repair_first_consumer_instrument_only"
    elif not pred_b:
        next_step = "split_attention11_q_k_q2_k2_value_and_mlp11_left_right_product_finitely"
    elif not pred_c:
        next_step = "task_conditioned_nonlinear_consumer_response_with_planted_identifiability_gate"
    elif not pred_d:
        next_step = "split_at_first_consumer_whose_physical_swap_fails"
    elif not pred_e:
        next_step = "retain_consumer_local_variable_without_question_semantics_and_test_ood_code_composition"
    else:
        next_step = "validate_question_consumer_relation_on_ood_code_then_trace_to_22_earlier_writes"

    bundle_payload = {
        "schema": "rung512_first_consumer_quotient_v1",
        "collections": {name: _bundle_collection(collection)
                        for name, collection in collections.items()},
        "candidates": candidates, "confirmed": confirmed,
        "physical": None if physical is None else {
            key: value for key, value in physical.items() if key != "diagnostics"},
        "physical_passing": physical_passing,
        "raw_tokens_logits_hidden_states_or_weights_included": False,
    }
    torch.save(bundle_payload, BUNDLE)
    result = {
        "status": "complete", "rung": 512,
        "claim_level": "consumer_local_equivalence_until_heldout_physical_substitution_passes",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "question_basis": {
            "eigenvalues": [float(value) for value in basis["values"]],
            "maximum_archived_eigenvalue_error": basis["maximum_archived_eigenvalue_error"],
        },
        "calibration": {"discovery": discovery_calibration,
                        "confirmation": confirmation_calibration},
        "calibration_holds": {"discovery": discovery_calibration_ok,
                              "confirmation": confirmation_calibration_ok},
        "diagnostics": {name: collection["diagnostics"]
                        for name, collection in collections.items()},
        "physical_diagnostics": None if physical is None else physical["diagnostics"],
        "analysis": {
            "discovery_summary": discovery_summary,
            "discovery_candidates": candidates,
            "confirmation_checks": confirmation_checks,
            "confirmed_relations": confirmed,
            "physical_checks": physical_checks,
            "physical_relations": physical_passing,
        },
        'pred_a_exact_live_consumer_instrument': pred_a,
        'pred_b_consumer_local_discovery_relation': pred_b,
        'pred_c_frozen_relation_predicts_fresh_documents': pred_c,
        'pred_d_bidirectional_physical_consumer_substitution': pred_d,
        'pred_e_question_form_relation': pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "full_forwards": sum(sum(collection["diagnostics"]["calls"].values())
                                 for collection in collections.values())
                             + (0 if physical is None else physical["diagnostics"]["calls"]),
            "backwards": 0, "consumer_tests": 126,
            "discovery_candidates": len(candidates),
            "confirmed_relations": len(confirmed),
            "physical_relations": len(physical_passing),
            "maximum_conditional_forwards": 21886,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started, "next_step": next_step,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 512,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e, "strong_null": strong_null,
        "discovery_candidates": len(candidates), "confirmed_relations": len(confirmed),
        "physical_relations": len(physical_passing),
        "execution_price": result["execution_price"], "next_step": next_step,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
