#!/usr/bin/env python3
"""RUNG501 -- directed finite-action graph among four equality scores."""

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
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_matcher_causal_action_quotient_rung498 as action_parent
import equality_term_score_payload_rung459 as factor_parent
import rung498_copy_task_portability_diagnosis as mask_parent


TERMS = ("L5H5", "L7H3", "L8H3", "L8H4")
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 2))
PAIR_NAMES = tuple(f"{TERMS[left]}->{TERMS[right]}" for left, right in PAIRS)
KNOWN_POSITIVE = "L5H5->L8H4"
KNOWN_NEGATIVE = "L7H3->L8H4"
PARTITIONS = ((0, 250), (250, 500), (500, 750), (750, 1000))
DISCOVERY = (0, 500, 250)
VALIDATION = (500, 1000, 750)
BATCH = 4
SCALE_FORWARDS = 24
FORWARDS_PER_BATCH = 2 + 9 * len(PAIRS)
DISCOVERY_FORWARDS = 125 * FORWARDS_PER_BATCH
VALIDATION_FORWARDS = 125 * FORWARDS_PER_BATCH
BACKGROUNDS = ("early_present", "early_absent")
STATES = ("late_native", "late_absent", "score_donor", "payload_donor", "whole_donor")
HYBRIDS = ("score_donor", "payload_donor", "whole_donor")
CELLS = ("copy_positive", "noncopy_equality", "all_noncopy")
D = 1152

PREREG = POLY / "EQUALITY_SCORE_DIRECTED_ACTION_GRAPH_RUNG501_PREREGISTRATION.md"
R500_SOURCE = ROOT / "ops/equality_matcher_mlp9_reader_calibration_rung500.py"
R500_RESULT = ROOT / "equality_matcher_mlp9_reader_calibration_rung500_results.json"
R500_BUNDLE = ROOT / "equality_matcher_mlp9_reader_calibration_rung500_bundle.pt"
ACTION_SOURCE = ROOT / "ops/equality_matcher_causal_action_quotient_rung498.py"
MASK_SOURCE = ROOT / "ops/rung498_copy_task_portability_diagnosis.py"
MASK_RESULT = ROOT / "rung498_copy_task_portability_diagnosis_results.json"
FACTOR_SOURCE = ROOT / "ops/equality_term_score_payload_rung459.py"
FACTOR_RESULT = ROOT / "equality_term_score_payload_rung459_results.json"
OUT = ROOT / "equality_score_directed_action_graph_rung501_results.json"
BUNDLE = ROOT / "equality_score_directed_action_graph_rung501_bundle.pt"
HASHES = {
    PREREG: "033d4d538f1a97ba1cfc2efd81bb98338ec0b1db7124521bd58715d1a42fa971",
    R500_SOURCE: "83b520873cabfd167e4da645c0564e267b15c3be98e4a4f8d739133d01f81b0f",
    R500_RESULT: "9e4daa40c2ab88980d29d141eef6317bfae3035e823ac6bd6c8fc57fabcbc7d9",
    R500_BUNDLE: "e7dca9a4f092f21db8e306460c8bd9fa970b04ced5dde2b7e51600bf6b8949cc",
    ACTION_SOURCE: "3186d610b77e1684849a54af79e83ce3d7a6a4338e36b3ec27ce2d7cc8696e59",
    MASK_SOURCE: "93ea53f87264eb0893affd9daf6b06a8099e4d243366c7c4c5f6453f521e5b51",
    MASK_RESULT: "6a733bf55bf7920825dfc0a5ccf367fbf3e419ae98b249a68ca59ac8748ab3a0",
    FACTOR_SOURCE: "9f9e66f689452cbcb14d741792b66eb9ff526dff5472a5938c58a2a4c82620d8",
    FACTOR_RESULT: "f157681ced170cbf8664db5710414a38d4f928f8d15dc0dd2b4d8cea9288aefa",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty_scale_stats():
    keys = ("edge_count", "payload_entry_count", "early_p2", "late_p2", "p_cross",
            "early_u2", "late_u2", "u_cross")
    return {name: {key: 0.0 for key in keys} for name in PAIR_NAMES}


def _empty_reader_stats():
    shape = (len(PAIRS), len(BACKGROUNDS), len(HYBRIDS), 2, len(CELLS))
    return {key: torch.zeros(shape, dtype=torch.float64)
            for key in ("ref2", "hyb2", "cross", "write2", "tokens")}


def _empty_background_stats():
    shape = (len(PAIRS), 2, len(CELLS))
    return {key: torch.zeros(shape, dtype=torch.float64)
            for key in ("present2", "absent2", "cross", "tokens")}


def _reader_report(stats, index):
    ref2 = float(stats["ref2"][index])
    hyb2 = float(stats["hyb2"][index])
    cross = float(stats["cross"][index])
    write2 = float(stats["write2"][index])
    cosine = cross / math.sqrt(max(ref2 * hyb2, 1e-30))
    scale = cross / max(hyb2, 1e-30)
    fitted = max(scale, 0.0)
    residual = math.sqrt(max(ref2 - 2 * fitted * cross + fitted * fitted * hyb2, 0.0)
                         / max(ref2, 1e-30))
    return {
        "cosine": cosine, "positive_fit_scale": scale, "scaled_residual": residual,
        "native_response_rms_over_native_write_rms": math.sqrt(ref2 / max(write2, 1e-30)),
        "hybrid_response_rms_over_native_write_rms": math.sqrt(hyb2 / max(write2, 1e-30)),
        "tokens": int(stats["tokens"][index]),
    }


def _background_report(stats, index):
    present2 = float(stats["present2"][index])
    absent2 = float(stats["absent2"][index])
    cross = float(stats["cross"][index])
    return {
        "cosine": cross / math.sqrt(max(present2 * absent2, 1e-30)),
        "present_over_absent_rms": math.sqrt(present2 / max(absent2, 1e-30)),
        "tokens": int(stats["tokens"][index]),
    }


def _task_masks(rows):
    masks = mask_parent.build_masks(rows)
    valid = torch.zeros_like(masks["copy_positive"])
    valid[:, 64:] = True
    return {
        "copy_positive": masks["copy_positive"],
        "noncopy_equality": masks["noncopy_equality"],
        "all_noncopy": valid & ~masks["copy_positive"],
    }


@torch.no_grad()
def _captured_forward(model, tokens, **kwargs):
    captures = []

    def hook(_module, _inputs, output):
        captures.append(output.detach().clone())

    handle = model.transformer.h[9].mlp.register_forward_hook(hook)
    try:
        logits, diagnostics, audit = run_forward(model, tokens, **kwargs)
    finally:
        handle.remove()
    if len(captures) != 1 or list(captures[0].shape) != [len(tokens), 256, D]:
        raise RuntimeError(f"MLP9 capture count/shape changed: {[x.shape for x in captures]}")
    return logits, captures[0], diagnostics, audit


@torch.no_grad()
def run_forward(model, tokens, *, pair=None, background="early_present",
                state="late_native", scales=None, direct=False):
    if background not in BACKGROUNDS or state not in STATES:
        raise ValueError("unregistered action")
    if direct and pair is not None:
        raise ValueError("direct native forward cannot carry a pair")
    if not direct and pair is not None and pair not in PAIRS:
        raise ValueError("unregistered directed pair")
    if state.endswith("donor") and (pair is None or scales is None):
        raise ValueError("hybrid requires a pair and frozen scales")
    cached = {}
    diagnostics = {"factor_reconstruction_max": 0.0, "early_edit_rms": 0.0,
                   "late_edit_rms": 0.0}
    audit = {"native_attention": 0, "replayed_attention": 0, "native_mlp": 0}

    def attention(event):
        if direct or event.site not in factor_parent.stage1.SITE_HEADS:
            write, next_value = event.block.attn(event.state, event.first_value)
            audit["native_attention"] += 1
            return write, next_value
        write, factors, support, error = factor_parent._factor_site(
            event.state, event.first_value, event.block.attn, event.site, event.tokens)
        audit["replayed_attention"] += 1
        diagnostics["factor_reconstruction_max"] = max(
            diagnostics["factor_reconstruction_max"], error)
        if pair is not None:
            donor, recipient = pair
            if event.site == factor_parent.TERMS[donor][1]:
                cached.update(factors[donor])
                if background == "early_absent":
                    edit = factors[donor]["native_term"]
                    write = write - edit
                    diagnostics["early_edit_rms"] = float(edit.float().square().mean().sqrt())
            if event.site == factor_parent.TERMS[recipient][1]:
                if not cached:
                    raise RuntimeError("donor factors unavailable at recipient")
                target = factors[recipient]
                if state != "late_native":
                    replacement = torch.zeros_like(target["factor_term"])
                    if state.endswith("donor"):
                        p, u = target["p"], target["u"]
                        if state in ("score_donor", "whole_donor"):
                            p = cached["p"] * scales["score_ratio"]
                        if state in ("payload_donor", "whole_donor"):
                            u = cached["u"] * scales["payload_ratio"]
                        replacement = torch.bmm(p * support, u)
                    edit = replacement.to(write.dtype) - target["native_term"]
                    write = write + edit
                    diagnostics["late_edit_rms"] = float(edit.float().square().mean().sqrt())
        return write, event.first_value

    def mlp(event):
        audit["native_mlp"] += 1
        return event.block.mlp(event.state)

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
    expected = ({"native_attention": 18, "replayed_attention": 0, "native_mlp": 18}
                if direct else
                {"native_attention": 15, "replayed_attention": 3, "native_mlp": 18})
    if audit != expected:
        raise RuntimeError(f"forward audit changed: {audit} != {expected}")
    return logits, diagnostics, audit


@torch.no_grad()
def _scale_forward(model, tokens, callbacks):
    factors_by_index = {}
    support_seen = None
    max_reconstruction = 0.0
    audit = {"native_attention": 0, "replayed_attention": 0, "native_mlp": 0}

    def attention(event):
        nonlocal support_seen, max_reconstruction
        if event.site not in factor_parent.stage1.SITE_HEADS:
            write, next_value = event.block.attn(event.state, event.first_value)
            audit["native_attention"] += 1
            return write, next_value
        write, factors, support, error = factor_parent._factor_site(
            event.state, event.first_value, event.block.attn, event.site, event.tokens)
        factors_by_index.update(factors)
        support_seen = support if support_seen is None else support_seen
        if not torch.equal(support_seen, support):
            raise RuntimeError("equality support changed across factor sites")
        max_reconstruction = max(max_reconstruction, error)
        audit["replayed_attention"] += 1
        return write, event.first_value

    def mlp(event):
        audit["native_mlp"] += 1
        return event.block.mlp(event.state)

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
    expected = {"native_attention": 15, "replayed_attention": 3, "native_mlp": 18}
    if audit != expected or set(factors_by_index) != set(range(len(TERMS))) or support_seen is None:
        raise RuntimeError("scale-pass factor capture changed")
    for callback in callbacks:
        callback(factors_by_index, support_seen)
    return logits, max_reconstruction, audit


@torch.no_grad()
def collect_scales(model):
    payload, fit_masks, metadata = factor_parent.validate_inputs()
    rows = payload["rows"]
    stats = _empty_scale_stats()
    diagnostics = {"calls": 0, "factor_reconstruction_max": 0.0}
    device = next(model.parameters()).device
    for start in range(0, 96, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        positive = fit_masks["all_positive"][start:start + BATCH]
        callbacks = []
        for pair_index, pair in enumerate(PAIRS):
            def callback(factors, support, pair=pair, name=PAIR_NAMES[pair_index], positive=positive):
                factor_parent._accumulate_scales(
                    stats, name, factors[pair[0]], factors[pair[1]], support, positive)
            callbacks.append(callback)
        logits, error, _audit = _scale_forward(model, tokens, callbacks)
        diagnostics["calls"] += 1
        diagnostics["factor_reconstruction_max"] = max(
            diagnostics["factor_reconstruction_max"], error)
        del logits
    diagnostics["calls_exact"] = diagnostics["calls"] == SCALE_FORWARDS
    scales = factor_parent._finish_scales(stats)
    diagnostics["all_scales_live"] = all(
        row["edge_count"] > 0 and row["score_ratio"] > 0 and row["payload_ratio"] > 0
        and all(math.isfinite(row[key]) for key in ("score_ratio", "payload_ratio"))
        for row in scales.values())
    return scales, diagnostics, metadata


def _cell_sums(logits, rows, masks):
    targets = rows[:, 1:].to(logits.device)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                          reduction="none").view(len(rows), -1).float().cpu()
    sums = torch.zeros(len(rows), len(CELLS), dtype=torch.float64)
    counts = torch.zeros_like(sums)
    for row in range(len(rows)):
        for cell_index, cell in enumerate(CELLS):
            selected = masks[cell][row]
            sums[row, cell_index] = nll[row, selected].double().sum()
            counts[row, cell_index] = int(selected.sum())
    return sums, counts


def _accumulate_reader(stats, pair_index, background_index, writes, masks, halves):
    absent = writes["late_absent"].float()
    native = writes["late_native"].float()
    reference = absent - native
    for hybrid_index, hybrid in enumerate(HYBRIDS):
        response = absent - writes[hybrid].float()
        for half, condition in enumerate(halves):
            for cell_index, cell in enumerate(CELLS):
                selected = masks[cell] & condition[:, None]
                if not bool(selected.any()):
                    continue
                ref = reference[selected].double()
                hyb = response[selected].double()
                writer = native[selected].double()
                index = (pair_index, background_index, hybrid_index, half, cell_index)
                stats["ref2"][index] += ref.square().sum().cpu()
                stats["hyb2"][index] += hyb.square().sum().cpu()
                stats["cross"][index] += (ref * hyb).sum().cpu()
                stats["write2"][index] += writer.square().sum().cpu()
                stats["tokens"][index] += int(selected.sum())


def _accumulate_background(stats, pair_index, present, absent, masks, halves):
    for half, condition in enumerate(halves):
        for cell_index, cell in enumerate(CELLS):
            selected = masks[cell] & condition[:, None]
            if not bool(selected.any()):
                continue
            left = present[selected].double()
            right = absent[selected].double()
            index = (pair_index, half, cell_index)
            stats["present2"][index] += left.square().sum().cpu()
            stats["absent2"][index] += right.square().sum().cpu()
            stats["cross"][index] += (left * right).sum().cpu()
            stats["tokens"][index] += int(selected.sum())


@torch.no_grad()
def collect_phase(model, rows, scales, bounds):
    start_doc, stop_doc, split = bounds
    docs = stop_doc - start_doc
    task_sums = torch.zeros(
        len(PAIRS), len(BACKGROUNDS), len(STATES), docs, len(CELLS), dtype=torch.float64)
    task_counts = torch.zeros(docs, len(CELLS), dtype=torch.float64)
    reader_stats = _empty_reader_stats()
    background_stats = _empty_background_stats()
    task_masks = _task_masks(rows)
    diagnostics = {
        "native_replay_logit_max_abs": 0.0, "native_replay_mlp9_max_abs": 0.0,
        "factor_reconstruction_max": 0.0, "minimum_nonzero_edit_rms": math.inf,
        "zero_intended_edit_actions": 0, "capture_calls": 0,
    }
    calls = {"native": 0, "analytical": 0}
    device = next(model.parameters()).device
    for start in range(start_doc, stop_doc, BATCH):
        stop = min(start + BATCH, stop_doc)
        local = start - start_doc
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        local_masks_cpu = {cell: task_masks[cell][start:stop] for cell in CELLS}
        local_masks_gpu = {cell: value.to(device) for cell, value in local_masks_cpu.items()}
        global_rows = torch.arange(start, stop, device=device)
        halves = (global_rows < split, global_rows >= split)
        native_logits, native_write, _diag, _audit = _captured_forward(model, tokens, direct=True)
        replay_logits, replay_write, diag, _audit = _captured_forward(model, tokens, pair=None)
        calls["native"] += 1
        calls["analytical"] += 1
        diagnostics["capture_calls"] += 2
        diagnostics["native_replay_logit_max_abs"] = max(
            diagnostics["native_replay_logit_max_abs"],
            float((native_logits.float() - replay_logits.float()).abs().max()))
        diagnostics["native_replay_mlp9_max_abs"] = max(
            diagnostics["native_replay_mlp9_max_abs"],
            float((native_write.float() - replay_write.float()).abs().max()))
        diagnostics["factor_reconstruction_max"] = max(
            diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
        replay_sums, observed_counts = _cell_sums(
            replay_logits, batch_rows, local_masks_cpu)
        task_counts[local:local + len(batch_rows)] = observed_counts
        for pair_index, pair in enumerate(PAIRS):
            score_responses = {}
            for background_index, background in enumerate(BACKGROUNDS):
                writes = {"late_native": replay_write} if background == "early_present" else {}
                sums = {"late_native": replay_sums} if background == "early_present" else {}
                for state in STATES:
                    if background == "early_present" and state == "late_native":
                        continue
                    logits, write, diag, _audit = _captured_forward(
                        model, tokens, pair=pair, background=background, state=state,
                        scales=scales[PAIR_NAMES[pair_index]])
                    calls["analytical"] += 1
                    diagnostics["capture_calls"] += 1
                    diagnostics["factor_reconstruction_max"] = max(
                        diagnostics["factor_reconstruction_max"],
                        diag["factor_reconstruction_max"])
                    if background == "early_absent" and diag["early_edit_rms"] <= 0:
                        diagnostics["zero_intended_edit_actions"] += 1
                    if state != "late_native" and diag["late_edit_rms"] <= 0:
                        diagnostics["zero_intended_edit_actions"] += 1
                    for key in ("early_edit_rms", "late_edit_rms"):
                        if diag[key] > 0:
                            diagnostics["minimum_nonzero_edit_rms"] = min(
                                diagnostics["minimum_nonzero_edit_rms"], diag[key])
                    cell_sums, counts = _cell_sums(logits, batch_rows, local_masks_cpu)
                    if not torch.equal(counts, observed_counts):
                        raise RuntimeError("task supports changed across actions")
                    writes[state] = write
                    sums[state] = cell_sums
                    del logits
                for state_index, state in enumerate(STATES):
                    task_sums[pair_index, background_index, state_index,
                              local:local + len(batch_rows)] = sums[state]
                _accumulate_reader(
                    reader_stats, pair_index, background_index, writes,
                    local_masks_gpu, halves)
                score_responses[background] = (
                    writes["late_absent"].float() - writes["score_donor"].float())
            _accumulate_background(
                background_stats, pair_index, score_responses["early_present"],
                score_responses["early_absent"], local_masks_gpu, halves)
        del native_logits, replay_logits, native_write, replay_write
    batches = math.ceil(docs / BATCH)
    diagnostics["calls"] = calls
    diagnostics["expected_calls"] = {"native": batches, "analytical": 64 * batches}
    diagnostics["calls_exact"] = calls == diagnostics["expected_calls"]
    diagnostics["capture_calls_exact"] = diagnostics["capture_calls"] == 65 * batches
    diagnostics["support"] = {
        cell: [int(task_masks[cell][start_doc:split].sum()),
               int(task_masks[cell][split:stop_doc].sum())] for cell in CELLS}
    return {
        "task_sums": task_sums, "task_counts": task_counts,
        "reader_stats": reader_stats, "background_stats": background_stats,
        "diagnostics": diagnostics,
    }


def analyze_phase(collection):
    task_sums = collection["task_sums"]
    counts = collection["task_counts"]
    docs = counts.shape[0]
    boundary = docs // 2
    analysis = {}
    for pair_index, name in enumerate(PAIR_NAMES):
        analysis[name] = {}
        for background_index, background in enumerate(BACKGROUNDS):
            analysis[name][background] = {}
            absent = task_sums[pair_index, background_index, STATES.index("late_absent")]
            native = task_sums[pair_index, background_index, STATES.index("late_native")]
            for hybrid_index, hybrid in enumerate(HYBRIDS):
                hybrid_sums = task_sums[pair_index, background_index, STATES.index(hybrid)]
                halves = []
                for half, (lo, hi) in enumerate(((0, boundary), (boundary, docs))):
                    ci = CELLS.index("copy_positive")
                    denom = counts[lo:hi, ci].clamp_min(1)
                    native_rows = (absent[lo:hi, ci] - native[lo:hi, ci]) / denom
                    hybrid_rows = (absent[lo:hi, ci] - hybrid_sums[lo:hi, ci]) / denom
                    task = action_parent._fit_report(native_rows, hybrid_rows)
                    native_total = float((absent[lo:hi, ci] - native[lo:hi, ci]).sum())
                    hybrid_total = float((absent[lo:hi, ci] - hybrid_sums[lo:hi, ci]).sum())
                    recovery = hybrid_total / native_total if abs(native_total) > 1e-30 else None
                    off = CELLS.index("all_noncopy")
                    off_change = float(
                        (hybrid_sums[lo:hi, off] - native[lo:hi, off]).sum()
                        / counts[lo:hi, off].sum().clamp_min(1))
                    reader = {
                        cell: _reader_report(
                            collection["reader_stats"],
                            (pair_index, background_index, hybrid_index, half, cell_index))
                        for cell_index, cell in enumerate(CELLS)
                    }
                    halves.append({
                        "equality_recovery": recovery, "task_effect": task,
                        "off_target_signed_mean_hybrid_minus_native_nat": off_change,
                        "reader": reader,
                    })
                analysis[name][background][hybrid] = halves
    background = {
        name: [
            {cell: _background_report(collection["background_stats"],
                                      (pair_index, half, cell_index))
             for cell_index, cell in enumerate(CELLS)}
            for half in range(2)]
        for pair_index, name in enumerate(PAIR_NAMES)
    }
    return analysis, background


def _partition_edge(analysis, background, pair_name, half):
    background_checks = []
    for background_name in BACKGROUNDS:
        score = analysis[pair_name][background_name]["score_donor"][half]
        payload = analysis[pair_name][background_name]["payload_donor"][half]
        copy = score["reader"]["copy_positive"]
        payload_copy = payload["reader"]["copy_positive"]
        task_holds = bool(
            score["equality_recovery"] is not None
            and .65 <= score["equality_recovery"] <= 1.40
            and score["task_effect"]["cosine"] >= .75
            and score["task_effect"]["scaled_residual"] <= .70
            and abs(score["off_target_signed_mean_hybrid_minus_native_nat"]) <= .01)
        reader_holds = bool(
            copy["cosine"] >= .75 and copy["scaled_residual"] <= .70
            and copy["positive_fit_scale"] > 0
            and min(copy["native_response_rms_over_native_write_rms"],
                    copy["hybrid_response_rms_over_native_write_rms"]) >= 1e-4)
        payload_holds = bool(
            copy["cosine"] >= payload_copy["cosine"] + .30
            or copy["scaled_residual"] <= payload_copy["scaled_residual"] - .30)
        specificity_margin = copy["cosine"] - max(
            score["reader"]["noncopy_equality"]["cosine"],
            score["reader"]["all_noncopy"]["cosine"])
        specificity_holds = specificity_margin >= .10
        background_checks.append({
            "background": background_name, "task_holds": task_holds,
            "reader_holds": reader_holds, "payload_rejected": payload_holds,
            "copy_specificity_margin": specificity_margin,
            "copy_specificity_holds": specificity_holds,
        })
    present = analysis[pair_name]["early_present"]["score_donor"][half]
    absent = analysis[pair_name]["early_absent"]["score_donor"][half]
    present_scale = present["reader"]["copy_positive"]["positive_fit_scale"]
    absent_scale = absent["reader"]["copy_positive"]["positive_fit_scale"]
    scale_drift = abs(present_scale - absent_scale) / max(
        abs(present_scale), abs(absent_scale), 1e-30)
    recovery_change = abs(present["equality_recovery"] - absent["equality_recovery"])
    closure = {
        "reader_response_cosine": background[pair_name][half]["copy_positive"]["cosine"],
        "task_recovery_change": recovery_change, "reader_scale_drift": scale_drift,
    }
    closure["holds"] = bool(
        closure["reader_response_cosine"] >= .75
        and recovery_change <= .30 and min(present_scale, absent_scale) > 0
        and scale_drift <= .50)
    edge = bool(
        all(row["task_holds"] and row["reader_holds"] and row["payload_rejected"]
            and row["copy_specificity_holds"] for row in background_checks)
        and closure["holds"])
    return {"edge": edge, "background_checks": background_checks, "closure": closure}


def _negative_tripwire(analysis, pair_name, half):
    rows = []
    for background_name in BACKGROUNDS:
        score = analysis[pair_name][background_name]["score_donor"][half]
        rows.append({
            "background": background_name,
            "task_cosine": score["task_effect"]["cosine"],
            "reader_cosine": score["reader"]["copy_positive"]["cosine"],
            "holds": bool(score["task_effect"]["cosine"] < 0
                          and score["reader"]["copy_positive"]["cosine"] < 0),
        })
    return {"holds": all(row["holds"] for row in rows), "backgrounds": rows}


def score_discovery(analysis, background, collection, scale_diagnostics):
    diagnostics = collection["diagnostics"]
    pred_a = bool(
        diagnostics["native_replay_logit_max_abs"] == 0.0
        and diagnostics["native_replay_mlp9_max_abs"] == 0.0
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["minimum_nonzero_edit_rms"] > 0
        and diagnostics["zero_intended_edit_actions"] == 0
        and diagnostics["calls_exact"] and diagnostics["capture_calls_exact"]
        and scale_diagnostics["calls_exact"] and scale_diagnostics["all_scales_live"]
        and scale_diagnostics["factor_reconstruction_max"] <= 1e-10
        and all(min(values) > 0 for values in diagnostics["support"].values())
        and bool((collection["reader_stats"]["tokens"] > 0).all()))
    partitions = {
        name: [_partition_edge(analysis, background, name, half) for half in range(2)]
        for name in PAIR_NAMES
    }
    negative = [_negative_tripwire(analysis, KNOWN_NEGATIVE, half) for half in range(2)]
    pred_b = bool(
        all(row["edge"] for row in partitions[KNOWN_POSITIVE])
        and all(not row["edge"] for row in partitions[KNOWN_NEGATIVE])
        and all(row["holds"] for row in negative))
    excluded = {KNOWN_POSITIVE, KNOWN_NEGATIVE}
    confirmed_new = [
        name for name in PAIR_NAMES if name not in excluded
        and partitions[name][0]["edge"] and partitions[name][1]["edge"]]
    pred_c = bool(confirmed_new)
    confirmed_directed = [
        name for name in PAIR_NAMES
        if partitions[name][0]["edge"] and partitions[name][1]["edge"]]
    payload_checks = [
        row for name in confirmed_directed for half in range(2)
        for row in partitions[name][half]["background_checks"]]
    equivalences = (["L8H3<->L8H4"] if
                    "L8H3->L8H4" in confirmed_directed
                    and "L8H4->L8H3" in confirmed_directed else [])
    pred_d = bool(pred_c and payload_checks and all(row["payload_rejected"] for row in payload_checks))
    return pred_a, pred_b, pred_c, pred_d, {
        "partitions": partitions, "negative_tripwire": negative,
        "confirmed_new_edges": confirmed_new,
        "confirmed_directed_edges": confirmed_directed,
        "confirmation_mutual_same_layer_equivalences": equivalences,
        "payload_checks": payload_checks,
    }


def score_validation(analysis, background, confirmed_edges):
    partitions = {
        name: [_partition_edge(analysis, background, name, half) for half in range(2)]
        for name in PAIR_NAMES
    }
    negative = [_negative_tripwire(analysis, KNOWN_NEGATIVE, half) for half in range(2)]
    graph_edges = (KNOWN_POSITIVE, *confirmed_edges)
    holds = bool(
        confirmed_edges
        and all(partitions[name][half]["edge"]
                for name in graph_edges for half in range(2))
        and all(not partitions[KNOWN_NEGATIVE][half]["edge"] for half in range(2))
        and all(row["holds"] for row in negative))
    retained = [name for name in graph_edges
                if all(partitions[name][half]["edge"] for half in range(2))]
    final_equivalence = (["L8H3<->L8H4"] if
                         "L8H3->L8H4" in retained and "L8H4->L8H3" in retained else [])
    return holds, {
        "partitions": partitions, "negative_tripwire": negative,
        "retained_edges": retained,
        "mutual_same_layer_equivalences": final_equivalence,
        "validation_uses_registered_partition_edge_interval": [.65, 1.40],
    }


def _serial_collection(collection):
    return {"diagnostics": collection["diagnostics"]}


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    parent = json.loads(R500_RESULT.read_text())
    parent_keys = (
        "pred_a_exact_live_reader_instrument", "pred_b_mlp9_reads_known_score_relation",
        "pred_c_mlp9_rejects_typed_controls", "pred_d_reader_stable_under_early_removal",
        "pred_e_reader_copy_task_selective", "pred_f_named_reader_calibrated")
    if not all(parent.get(key) is True for key in parent_keys) or parent.get("strong_null"):
        raise RuntimeError("rung500 calibration authority changed")
    if tuple(factor_parent.TERM_NAMES) != TERMS:
        raise RuntimeError("equality-term order changed")
    rows, _circuit_masks, _discovery_tags, _validation_tags, _scales, metadata = (
        action_parent.validate_inputs())
    if len(rows) != 1000:
        raise RuntimeError("census rows changed")
    return rows, metadata


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(PAIRS) == 7 and len(set(PAIRS)) == 7
        assert KNOWN_POSITIVE in PAIR_NAMES and KNOWN_NEGATIVE in PAIR_NAMES
        assert FORWARDS_PER_BATCH == 65
        stats = _empty_reader_stats()
        index = (0, 0, 0, 0, 0)
        stats["ref2"][index] = stats["hyb2"][index] = stats["cross"][index] = 4
        stats["write2"][index] = 100
        stats["tokens"][index] = 2
        assert _reader_report(stats, index)["cosine"] == 1.0
        print(json.dumps({
            "status": "dry_run_passed", "rung": 501, "model_loaded": False,
            "candidate_outcomes_opened": False, "pairs": PAIR_NAMES,
            "partitions": PARTITIONS, "scale_forwards": SCALE_FORWARDS,
            "discovery_forwards": DISCOVERY_FORWARDS,
            "conditional_validation_forwards": VALIDATION_FORWARDS,
            "predictions": ["pred_a", "pred_b", "pred_c", "pred_d", "pred_e", "pred_f"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung501 output namespace already exists")
    rows, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scales, scale_diagnostics, scale_metadata = collect_scales(model)
    discovery_collection = collect_phase(model, rows, scales, DISCOVERY)
    discovery, discovery_background = analyze_phase(discovery_collection)
    pred_a, pred_b, pred_c, pred_d, discovery_checks = score_discovery(
        discovery, discovery_background, discovery_collection, scale_diagnostics)
    pred_a = bool(pred_a and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
    validation_licensed = bool(pred_a and pred_b and pred_c and pred_d)
    validation_collection = validation = validation_background = validation_checks = None
    pred_e = False
    if validation_licensed:
        validation_collection = collect_phase(model, rows, scales, VALIDATION)
        validation, validation_background = analyze_phase(validation_collection)
        validation_instrument, *_ = score_discovery(
            validation, validation_background, validation_collection, scale_diagnostics)
        validation_science, validation_checks = score_validation(
            validation, validation_background, discovery_checks["confirmed_new_edges"])
        pred_e = bool(validation_instrument and validation_science)
    pred_f = bool(pred_a and pred_b and pred_c and pred_d and pred_e)
    bundle = {
        "schema": "equality_score_directed_action_graph_rung501_stats_v1",
        "scale_diagnostics": scale_diagnostics, "scales": scales,
        "discovery_task_ce_sums": discovery_collection["task_sums"],
        "discovery_task_counts": discovery_collection["task_counts"],
        "discovery_reader_dot_products": discovery_collection["reader_stats"],
        "discovery_background_dot_products": discovery_collection["background_stats"],
        "validation_task_ce_sums": None if validation_collection is None else
        validation_collection["task_sums"],
        "validation_task_counts": None if validation_collection is None else
        validation_collection["task_counts"],
        "validation_reader_dot_products": None if validation_collection is None else
        validation_collection["reader_stats"],
        "validation_background_dot_products": None if validation_collection is None else
        validation_collection["background_stats"],
        "raw_tokens_logits_or_mlp9_vectors_included": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 501,
        "claim_level": "prospective_directed_score_action_graph_not_head_equivalence_or_compression",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "scale_input_identity": scale_metadata,
        "pairs": list(PAIR_NAMES), "backgrounds": list(BACKGROUNDS),
        "states": list(STATES), "cells": list(CELLS), "frozen_scales": scales,
        "scale_diagnostics": scale_diagnostics,
        "discovery": {
            "bounds": list(DISCOVERY), "collection": _serial_collection(discovery_collection),
            "analysis": discovery, "background_comparisons": discovery_background,
            "checks": discovery_checks,
        },
        "validation": None if validation_collection is None else {
            "bounds": list(VALIDATION), "collection": _serial_collection(validation_collection),
            "analysis": validation, "background_comparisons": validation_background,
            "checks": validation_checks,
        },
        "validation_licensed_and_opened": validation_licensed,
        "off_target_definition": "absolute value of signed mean hybrid-minus-native NLL over all non-copy positions",
        "validation_edge_interval": [.65, 1.40],
        'pred_a_exact_live_isolated_instrument': pred_a,
        'pred_b_calibration_tripwires_reproduce': pred_b,
        'pred_c_new_confirmed_directed_edge': pred_c,
        'pred_d_graph_semantics_typed': pred_d,
        'pred_e_heldout_graph_validation': pred_e,
        'pred_f_reusable_copy_score_graph': pred_f,
        "strong_null": bool(not pred_a or not pred_b or not pred_c or not pred_d),
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "scale_forwards": scale_diagnostics["calls"],
            "discovery_forwards": sum(discovery_collection["diagnostics"]["calls"].values()),
            "validation_forwards": 0 if validation_collection is None else
            sum(validation_collection["diagnostics"]["calls"].values()),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started,
        "next_step": (
            "decompose_validated_score_edges_into_exact_mlp9_source_pair_responses" if pred_f else
            "repair_instrument_only" if not pred_a else
            "retire_directed_graph_assay" if not pred_b else
            "calibrated_known_edge_isolated_then_decompose_its_mlp9_source_pairs" if not pred_c else
            "reject_untyped_score_payload_edges" if not pred_d else
            "discovery_graph_failed_validation_then_decompose_only_known_edge"),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 501,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": result["strong_null"], "validation_opened": validation_licensed,
        "execution_price": result["execution_price"], "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
