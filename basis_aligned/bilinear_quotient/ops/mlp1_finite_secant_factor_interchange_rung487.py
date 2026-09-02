#!/usr/bin/env python3
"""RUNG487 -- exact MLP1 finite-secant factor interchange across T/C/I."""

# BQGATE: EXPERIMENT
# pred_a exact polarization, replay, calls, and physical own-secant identity
# pred_b own finite responses transfer across document halves
# pred_c at least one context-factor or direction-factor interchange edge
# pred_d the factor-sharing graph is stable across discovery halves
# pred_e the frozen graph validates on held-out documents

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
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import mlp0_coupled_block1_bigram_response_rung486 as parent
import mlp0_immediate_consumer_quotient_rung483 as branch_parent
import mlp0_centered_context_anova_factorial as component_parent


PREREG = POLY / "MLP1_FINITE_SECANT_FACTOR_INTERCHANGE_RUNG487_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp0_coupled_block1_bigram_response_rung486.py"
PARENT_RESULT = ROOT / "mlp0_coupled_block1_bigram_response_rung486_results.json"
OUT = ROOT / "mlp1_finite_secant_factor_interchange_rung487_results.json"
HASHES = {
    PREREG: "ba71ae5ce6288a18ef5195f0ee261cb51428ed7a7af409c3ba8f71e82762ce4f",
    PARENT_SOURCE: "4cf42487272688bfb03430e5aa5a27b78421df5b3138a6b71cd4c9a6061a607f",
    PARENT_RESULT: "f36ed7bed41d5908fd5f1da977a6ec72c21414e91829dfdda6fe39cfaf3ec941",
    ROOT / "ops/mlp0_immediate_consumer_quotient_rung483.py":
        "9763502b99b8693826a5985c8f25a3ebe7763c3cd176c3aebeeb140833a61f4c",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
}
BRANCHES = ("T", "C", "I")
UNORDERED_PAIRS = (("T", "C"), ("T", "I"), ("C", "I"))
ORDERED_PAIRS = tuple((target, donor) for pair in UNORDERED_PAIRS
                      for target, donor in (pair, pair[::-1]))
MODES = ("own", "context", "direction", "both")
POSITION_SHIFTS = parent.POSITION_SHIFTS
DISCOVERY_RANGE = (0, 500)
VALIDATION_RANGE = (500, 1000)
SPLIT = 250
BATCH = 4
TOKENS = 256


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_name(pair):
    return f"{pair[0]}<-{pair[1]}"


def _cosine(left, right):
    return parent._cosine(left, right)


def _effect_report(predictor, target):
    return parent.parent.parent._effect_report(predictor, target)


def _relative_squared(left, right):
    return parent.parent.parent._relative_squared(left, right)


def _linear(value, weight):
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def _mlp_write(mlp, state):
    left = _linear(state, mlp.Left.weight)
    right = _linear(state, mlp.Right.weight)
    write = _linear(left * right, mlp.Down.weight)
    return write + mlp.Down_bias.to(device=write.device, dtype=write.dtype)


def _secant(mlp, delta, midpoint):
    left_delta = _linear(delta, mlp.Left.weight)
    right_delta = _linear(delta, mlp.Right.weight)
    left_midpoint = _linear(midpoint, mlp.Left.weight)
    right_midpoint = _linear(midpoint, mlp.Right.weight)
    return _linear(
        left_delta * right_midpoint + left_midpoint * right_delta,
        mlp.Down.weight)


def _secants_for_pair(mlp, states, target, donor):
    native = states["native"]
    target_absent = states[target]
    donor_absent = states[donor]
    delta_target = native - target_absent
    delta_donor = native - donor_absent
    midpoint_target = (native + target_absent) / 2
    midpoint_donor = (native + donor_absent) / 2
    return {
        "own": _secant(mlp, delta_target, midpoint_target),
        "context": _secant(mlp, delta_target, midpoint_donor),
        "direction": _secant(mlp, delta_donor, midpoint_target),
        "both": _secant(mlp, delta_donor, midpoint_donor),
    }, (delta_target, delta_donor, midpoint_target, midpoint_donor)


def validate_inputs():
    for path, expected in HASHES.items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(PARENT_RESULT.read_text())
    if receipt.get("rung") != 486 \
            or receipt.get("pred_a_exact_lawful_instrument") is not True \
            or receipt.get("pred_b_stable_complete_carrier_decomposition") is not True \
            or receipt.get("pred_c_bigram_predicts_T_response") is not False \
            or receipt.get("pred_d_exactly_one_T_I_relation") is not False \
            or receipt.get("pred_e_heldout_documents") is not False \
            or receipt.get("validation_licensed_and_opened") is not False \
            or receipt.get("strong_null") is not True \
            or receipt.get("next_step") != "continuous_live_attention0_state_finite_reader":
        raise RuntimeError("rung486 did not license rung487")
    rows, positive, fit_rows, metadata = parent.parent.parent.validate_inputs()
    return rows, positive, fit_rows, metadata


@torch.no_grad()
def _native_forward(model, tokens, reference):
    capture = {}
    calls = {"attention": 0, "mlp": 0}

    def attention(event):
        calls["attention"] += 1
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 1:
            capture["A"] = write.detach().clone()
        return write, first_value

    def mlp(event):
        calls["mlp"] += 1
        write = event.block.mlp(event.state)
        if event.site == 0:
            capture["D"] = write.detach().clone()
            capture["mlp0_state"] = event.state.detach().clone()
        elif event.site == 1:
            capture["M"] = write.detach().clone()
            capture["z"] = event.state.detach().clone()
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    prefix = branch_parent._native_prefix(model, tokens, reference)
    capture["branches"] = {name: prefix["branches"][name].detach().clone()
                           for name in BRANCHES}
    capture["identity"] = {key: prefix[key] for key in (
        "analytical_num", "analytical_den", "deployed_num", "deployed_den")}
    capture["prefix_errors"] = {
        "D": _relative_squared(prefix["m0"], capture["D"]),
        "A": _relative_squared(prefix["a1"], capture["A"]),
        "M": _relative_squared(prefix["m1"], capture["M"]),
    }
    return logits, capture, calls


@torch.no_grad()
def _absent_forward(model, tokens, native, branch):
    capture = {}
    calls = {"attention": 0, "mlp": 0, "site0_removal": 0}

    def attention(event):
        calls["attention"] += 1
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 1:
            capture["A"] = write.detach().clone()
        return write, first_value

    def mlp(event):
        calls["mlp"] += 1
        write = event.block.mlp(event.state)
        if event.site == 0:
            calls["site0_removal"] += 1
            capture["mlp0_state_error"] = float(
                (event.state - native["mlp0_state"]).abs().max())
            write = native["D"] - branch
            capture["D"] = write.detach().clone()
        elif event.site == 1:
            capture["M"] = write.detach().clone()
            capture["z"] = event.state.detach().clone()
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    return logits, capture, calls


@torch.no_grad()
def _physical_forward(model, tokens, absent, secant):
    calls = {"attention": 0, "mlp": 0, "D": 0, "A": 0, "M": 0}

    def attention(event):
        calls["attention"] += 1
        if event.site != 1:
            return event.block.attn(event.state, event.first_value)
        calls["A"] += 1
        return absent["A"], event.first_value

    def mlp(event):
        calls["mlp"] += 1
        if event.site == 0:
            calls["D"] += 1
            return absent["D"]
        if event.site == 1:
            calls["M"] += 1
            return (absent["M"].float() + secant.float()).to(absent["M"].dtype)
        return event.block.mlp(event.state)

    return facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True), calls
