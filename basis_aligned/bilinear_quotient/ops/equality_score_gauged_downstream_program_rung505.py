#!/usr/bin/env python3
"""RUNG505 -- fixed five-site downstream program across sign-gauged scores."""

# BQGATE: EXPERIMENT
# pred_a: exact live 17,875-forward intervention instrument
# pred_b: all three supplied-score actions remain calibrated and wrong signs anti-align
# pred_c: the frozen rung466 program transfers from code to natural text
# pred_d: task/suppressor/interaction program is invariant across the sign gauge
# pred_e: correct gauge orientation beats wrong-sign controls downstream

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_matcher_copy_task_calibration_rung499 as mask_parent
import equality_score_directed_action_graph_rung501 as action_parent


PREREG = POLY / "EQUALITY_SCORE_GAUGED_DOWNSTREAM_PROGRAM_RUNG505_PREREGISTRATION.md"
ACTION_SOURCE = ROOT / "ops/equality_score_directed_action_graph_rung501.py"
ACTION_RESULT = ROOT / "equality_score_directed_action_graph_rung501_results.json"
GAUGE_SOURCE = ROOT / "ops/equality_score_sign_gauge_validation.py"
GAUGE_RESULT = ROOT / "equality_score_sign_gauge_validation_results.json"
MASK_SOURCE = ROOT / "ops/equality_matcher_copy_task_calibration_rung499.py"
GROUP_SOURCE = ROOT / "ops/equality_correction_group_factorial_rung466.py"
GROUP_RESULT = ROOT / "equality_correction_group_factorial_rung466_results.json"
R504_RESULT = ROOT / "mlp9_finite_two_source_interaction_rung504_results.json"
OUT = ROOT / "equality_score_gauged_downstream_program_rung505_results.json"
BUNDLE = ROOT / "equality_score_gauged_downstream_program_rung505_bundle.pt"
HASHES = {
    PREREG: "75e23241289b5ce0ef611e81982826f42d2f96fa8a7ba6e1838de2589e53bb46",
    ACTION_SOURCE: "97f3946f558f3d61fc952a9b6ddc7c334b51ccc0ccfe5f02c6ecced417f1e077",
    ACTION_RESULT: "b17a9b274e4c61e0b4a3fc68d8ce84ec6f8e76f257c3d898e6a6990492301c4f",
    GAUGE_SOURCE: "d1cc61481884109bd666e4a525de8d4925a911288429b9b48c5865c4b7b99333",
    GAUGE_RESULT: "ba1e7154c7c8d591cd0cf73bd9c3cb883c84b5bf78e21cce7ed1e9c965aa6095",
    MASK_SOURCE: "2c85a758f3083df467761d199f8dd108602ed7528789910f655030497d088904",
    GROUP_SOURCE: "48fd463c04981b601a969e8fa9f1020180c75b0bff68092e11d2d233e54ddd72",
    GROUP_RESULT: "d04acf3637834830f8ee7bd73eaa8a6c435386816ef54fce1d8451b0597132fe",
    R504_RESULT: "b5ff59ec7b86d3eae0dfc6a8f618ececce2ff7c35a66151a78f8201ccf9aeabe",
}

BOUNDS = (500, 1000, 750)
BATCH = 4
DOCUMENTS = BOUNDS[1] - BOUNDS[0]
SITES = ("m8", "m9", "m12", "a14", "m17")
TASK_MASK = 0b00111
SUPPRESSOR_MASK = 0b11000
ALL_MASK = 0b11111
SUBSETS = tuple(range(1 << len(SITES)))
ALL_LATER_SITES = (
    "m8", "a9", "m9", "a10", "m10", "a11", "m11", "a12", "m12",
    "a13", "m13", "a14", "m14", "a15", "m15", "a16", "m16", "a17", "m17",
)
CELLS = (
    "all_positive", "near_positive", "far_positive",
    "one_predecessor_positive", "multiple_predecessor_positive", "off_target",
)
CONTEXT_CELLS = (
    "near_positive", "far_positive",
    "one_predecessor_positive", "multiple_predecessor_positive",
)
SOURCES = ("N", "P", "Z7", "Z8")
WRONG_SIGNS = ("W7", "W8")
SOURCE_ACTIONS = {
    "N": {"pair": None, "sign": 1.0},
    "P": {"pair": (0, 3), "sign": 1.0},
    "Z7": {"pair": (1, 3), "sign": -1.0},
    "Z8": {"pair": (2, 3), "sign": -1.0},
    "W7": {"pair": (1, 3), "sign": 1.0},
    "W8": {"pair": (2, 3), "sign": 1.0},
}
WRONG_MASKS = (0, TASK_MASK, SUPPRESSOR_MASK, ALL_MASK)
BATCHES = DOCUMENTS // BATCH
FORWARDS_PER_BATCH = 2 + 1 + len(SOURCES) * (len(SUBSETS) + 1) \
    + len(WRONG_SIGNS) * len(WRONG_MASKS)
EXPECTED_FORWARDS = BATCHES * FORWARDS_PER_BATCH
EXPECTED_CAPTURES = BATCHES * len(ALL_LATER_SITES)
EXPECTED_PATCHES_PER_BATCH = (
    len(SOURCES) * (len(SITES) * (1 << (len(SITES) - 1)) + len(ALL_LATER_SITES))
    + len(WRONG_SIGNS) * sum(mask.bit_count() for mask in WRONG_MASKS)
)
EXPECTED_PATCHES = BATCHES * EXPECTED_PATCHES_PER_BATCH


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def subset_sites(mask: int) -> tuple[str, ...]:
    if mask < 0 or mask > ALL_MASK:
        raise ValueError("five-site subset mask changed")
    return tuple(site for bit, site in enumerate(SITES) if mask & (1 << bit))


def signed_scales(scales: Mapping[str, Mapping[str, float]], action: str):
    spec = SOURCE_ACTIONS[action]
    pair = spec["pair"]
    if pair is None:
        return None
    name = f"{action_parent.TERMS[pair[0]]}->{action_parent.TERMS[pair[1]]}"
    row = dict(scales[name])
    row["score_ratio"] *= spec["sign"]
    return row


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    r504 = json.loads(R504_RESULT.read_text())
    if r504.get("pred_a_exact_finite_suffix_instrument_and_parent_reproduce") is not True \
            or r504.get("pred_b_compact_two_source_interaction_set_selected") is not False \
            or r504.get("strong_null") is not True:
        raise RuntimeError("rung 504 route identity changed")
    group = json.loads(GROUP_RESULT.read_text())
    if group.get("rung") != 466 or not all(group.get(key) is True for key in (
        "pred_a_instrument", "pred_b_task_group_context", "pred_c_broad_suppressor_role",
        "pred_d_cross_group_interaction", "pred_e_five_site_extraction",
    )) or group.get("strong_null") is not False:
        raise RuntimeError("rung 466 group authority changed")
    gauge = json.loads(GAUGE_RESULT.read_text())
    if not all(gauge.get(key) is True for key in (
        "pred_a_exact_live_validation_instrument", "pred_b_forward_gauge_validates_500_1000",
        "pred_c_reverse_L8H4_to_L8H3_edge_on_discovery",
        "pred_d_reverse_validates_500_1000", "pred_e_mutual_sign_gauge_licensed",
    )) or gauge.get("strong_null") is not False:
        raise RuntimeError("sign-gauge authority changed")
    rows, metadata = action_parent.validate_inputs()
    action = json.loads(ACTION_RESULT.read_text())
    scales = action["frozen_scales"]
    required = ("L5H5->L8H4", "L7H3->L8H4", "L8H3->L8H4")
    if len(rows) != 1000 or any(name not in scales for name in required):
        raise RuntimeError("row or scale authority changed")
    masks = mask_parent.build_task_masks(rows)
    if set(masks) != set(CELLS) or any(list(masks[cell].shape) != [1000, 256] for cell in CELLS):
        raise RuntimeError("natural task masks changed")
    support = {
        cell: [int(masks[cell][lo:hi].sum()) for lo, hi in ((500, 750), (750, 1000))]
        for cell in CELLS
    }
    if any(min(values) <= 0 for values in support.values()):
        raise RuntimeError(f"unsupported task cell: {support}")
    return rows, masks, scales, {
        **metadata,
        "documents": [500, 1000],
        "halves": [[500, 750], [750, 1000]],
        "sites": list(SITES),
        "all_later_sites": list(ALL_LATER_SITES),
        "sources": list(SOURCES),
        "wrong_sign_controls": list(WRONG_SIGNS),
        "support": support,
    }


@torch.no_grad()
def run_forward(
    model,
    tokens,
    *,
    action: str = "N",
    absent: bool = False,
    scales: Mapping[str, Mapping[str, float]] | None = None,
    direct: bool = False,
    capture_keys: Sequence[str] = (),
    patch_writes: Mapping[str, torch.Tensor] | None = None,
):
    if action not in SOURCE_ACTIONS:
        raise ValueError("unregistered score action")
    if direct and (action != "N" or absent or capture_keys or patch_writes):
        raise ValueError("native direct arm cannot carry analytical edits")
    pair = SOURCE_ACTIONS[action]["pair"]
    if (pair is not None or absent) and scales is None:
        raise ValueError("score edit requires frozen scales")
    patch_writes = {} if patch_writes is None else dict(patch_writes)
    capture_set = set(capture_keys)
    if len(capture_set) != len(capture_keys) or not capture_set <= set(ALL_LATER_SITES):
        raise ValueError("capture identity changed")
    if set(patch_writes) - set(ALL_LATER_SITES):
        raise ValueError("patch identity changed")
    cached = {}
    captures = {}
    diagnostics = {
        "factor_reconstruction_max": 0.0,
        "late_edit_rms": 0.0,
        "patch_rms_max": 0.0,
    }
    audit = {
        "native_attention": 0,
        "replayed_attention": 0,
        "native_mlp": 0,
        "captures": 0,
        "patches": 0,
    }
    factor_parent = action_parent.factor_parent

    def patch_and_capture(key: str, write: torch.Tensor) -> torch.Tensor:
        if key in patch_writes:
            replacement = patch_writes[key]
            if replacement.shape != write.shape or replacement.dtype != write.dtype \
                    or replacement.device != write.device or not bool(torch.isfinite(replacement).all()):
                raise RuntimeError(f"malformed patch at {key}")
            diagnostics["patch_rms_max"] = max(
                diagnostics["patch_rms_max"],
                float((replacement.float() - write.float()).square().mean().sqrt()),
            )
            write = replacement
            audit["patches"] += 1
        if key in capture_set:
            captures[key] = write.detach().clone()
            audit["captures"] += 1
        return write

    def attention(event):
        if direct or event.site not in factor_parent.stage1.SITE_HEADS:
            write, next_value = event.block.attn(event.state, event.first_value)
            audit["native_attention"] += 1
        else:
            write, factors, support, error = factor_parent._factor_site(
                event.state, event.first_value, event.block.attn, event.site, event.tokens,
            )
            audit["replayed_attention"] += 1
            diagnostics["factor_reconstruction_max"] = max(
                diagnostics["factor_reconstruction_max"], error)
            edit_pair = (0, 3) if absent else pair
            if edit_pair is not None:
                donor, recipient = edit_pair
                if event.site == factor_parent.TERMS[donor][1]:
                    cached.update(factors[donor])
                if event.site == factor_parent.TERMS[recipient][1]:
                    if not cached:
                        raise RuntimeError("donor factors unavailable at recipient")
                    target = factors[recipient]
                    replacement = torch.zeros_like(target["factor_term"])
                    if not absent:
                        row = signed_scales(scales, action)
                        if row is None:
                            raise RuntimeError("edited score action lacks a scale")
                        replacement = torch.bmm(cached["p"] * row["score_ratio"] * support, target["u"])
                    edit = replacement.to(write.dtype) - target["native_term"]
                    write = write + edit
                    diagnostics["late_edit_rms"] = float(edit.float().square().mean().sqrt())
            next_value = event.first_value
        return patch_and_capture(f"a{event.site}", write), next_value

    def mlp(event):
        write = event.block.mlp(event.state)
        audit["native_mlp"] += 1
        return patch_and_capture(f"m{event.site}", write)

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
    expected = ({"native_attention": 18, "replayed_attention": 0, "native_mlp": 18}
                if direct else
                {"native_attention": 15, "replayed_attention": 3, "native_mlp": 18})
    for key, value in expected.items():
        if audit[key] != value:
            raise RuntimeError(f"forward audit changed at {key}: {audit} != {expected}")
    if audit["captures"] != len(capture_set) or set(captures) != capture_set \
            or audit["patches"] != len(patch_writes):
        raise RuntimeError(f"capture/patch audit changed: {audit}")
    return logits, captures, diagnostics, audit


def cell_sums(logits: torch.Tensor, rows: torch.Tensor, masks: Mapping[str, torch.Tensor]):
    targets = rows[:, 1:].to(logits.device)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).view(len(rows), -1).float().cpu()
    sums = torch.zeros(len(rows), len(CELLS), dtype=torch.float64)
    counts = torch.zeros_like(sums)
    for ci, cell in enumerate(CELLS):
        selected = masks[cell]
        sums[:, ci] = (nll * selected).double().sum(1)
        counts[:, ci] = selected.sum(1).double()
    return sums, counts


def _empty_collection():
    return {
        "base": torch.zeros(DOCUMENTS, len(CELLS), dtype=torch.float64),
        "losses": torch.zeros(len(SOURCES), len(SUBSETS), DOCUMENTS, len(CELLS), dtype=torch.float64),
        "direct": torch.zeros(len(SOURCES), DOCUMENTS, len(CELLS), dtype=torch.float64),
        "wrong": torch.zeros(len(WRONG_SIGNS), len(WRONG_MASKS), DOCUMENTS, len(CELLS), dtype=torch.float64),
        "counts": torch.zeros(DOCUMENTS, len(CELLS), dtype=torch.float64),
    }


@torch.no_grad()
def collect(model, rows, masks, scales):
    data = _empty_collection()
    diagnostics = {
        "native_replay_logit_max_abs": 0.0,
        "native_replay_relative_squared": 0.0,
        "factor_reconstruction_max": 0.0,
        "minimum_score_edit_rms": math.inf,
        "source_patch_rms_max": {source: 0.0 for source in SOURCES},
        "calls": {"native": 0, "analytical": 0},
        "captures": 0,
        "patches": 0,
    }
    device = next(model.parameters()).device
    lo_doc, hi_doc, _split = BOUNDS
    for start in range(lo_doc, hi_doc, BATCH):
        stop = start + BATCH
        local = start - lo_doc
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        local_masks = {cell: masks[cell][start:stop] for cell in CELLS}

        native_logits, _, _, audit = run_forward(model, tokens, direct=True)
        replay_logits, _, replay_diag, replay_audit = run_forward(model, tokens, action="N")
        diagnostics["calls"]["native"] += 1
        diagnostics["calls"]["analytical"] += 1
        diagnostics["factor_reconstruction_max"] = max(
            diagnostics["factor_reconstruction_max"], replay_diag["factor_reconstruction_max"])
        diff = replay_logits.float() - native_logits.float()
        diagnostics["native_replay_logit_max_abs"] = max(
            diagnostics["native_replay_logit_max_abs"], float(diff.abs().max()))
        diagnostics["native_replay_relative_squared"] = max(
            diagnostics["native_replay_relative_squared"],
            float(diff.square().sum()) / max(float(native_logits.float().square().sum()), 1e-30),
        )

        base_logits, absent_writes, diag, audit = run_forward(
            model, tokens, action="P", absent=True, scales=scales,
            capture_keys=ALL_LATER_SITES,
        )
        diagnostics["calls"]["analytical"] += 1
        diagnostics["captures"] += audit["captures"]
        diagnostics["factor_reconstruction_max"] = max(
            diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
        diagnostics["minimum_score_edit_rms"] = min(
            diagnostics["minimum_score_edit_rms"], diag["late_edit_rms"])
        sums, observed = cell_sums(base_logits, batch_rows, local_masks)
        data["base"][local:local + BATCH] = sums
        data["counts"][local:local + BATCH] = observed

        for source_index, source in enumerate(SOURCES):
            for mask in SUBSETS:
                patches = {site: absent_writes[site] for site in subset_sites(mask)}
                logits, _, diag, audit = run_forward(
                    model, tokens, action=source, scales=scales, patch_writes=patches,
                )
                diagnostics["calls"]["analytical"] += 1
                diagnostics["patches"] += audit["patches"]
                diagnostics["source_patch_rms_max"][source] = max(
                    diagnostics["source_patch_rms_max"][source], diag["patch_rms_max"])
                diagnostics["factor_reconstruction_max"] = max(
                    diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
                if diag["late_edit_rms"] > 0:
                    diagnostics["minimum_score_edit_rms"] = min(
                        diagnostics["minimum_score_edit_rms"], diag["late_edit_rms"])
                sums, counts = cell_sums(logits, batch_rows, local_masks)
                if not torch.equal(counts, observed):
                    raise RuntimeError("task support changed across subset actions")
                data["losses"][source_index, mask, local:local + BATCH] = sums
            logits, _, diag, audit = run_forward(
                model, tokens, action=source, scales=scales, patch_writes=absent_writes,
            )
            diagnostics["calls"]["analytical"] += 1
            diagnostics["patches"] += audit["patches"]
            diagnostics["source_patch_rms_max"][source] = max(
                diagnostics["source_patch_rms_max"][source], diag["patch_rms_max"])
            diagnostics["factor_reconstruction_max"] = max(
                diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
            sums, counts = cell_sums(logits, batch_rows, local_masks)
            if not torch.equal(counts, observed):
                raise RuntimeError("task support changed in all-later control")
            data["direct"][source_index, local:local + BATCH] = sums

        for wrong_index, source in enumerate(WRONG_SIGNS):
            for mask_index, mask in enumerate(WRONG_MASKS):
                patches = {site: absent_writes[site] for site in subset_sites(mask)}
                logits, _, diag, audit = run_forward(
                    model, tokens, action=source, scales=scales, patch_writes=patches,
                )
                diagnostics["calls"]["analytical"] += 1
                diagnostics["patches"] += audit["patches"]
                diagnostics["factor_reconstruction_max"] = max(
                    diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
                if diag["late_edit_rms"] > 0:
                    diagnostics["minimum_score_edit_rms"] = min(
                        diagnostics["minimum_score_edit_rms"], diag["late_edit_rms"])
                sums, counts = cell_sums(logits, batch_rows, local_masks)
                if not torch.equal(counts, observed):
                    raise RuntimeError("task support changed in wrong-sign control")
                data["wrong"][wrong_index, mask_index, local:local + BATCH] = sums
        del native_logits, replay_logits, base_logits, absent_writes

    diagnostics["calls_expected"] = {"native": BATCHES, "analytical": EXPECTED_FORWARDS - BATCHES}
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["captures_expected"] = EXPECTED_CAPTURES
    diagnostics["captures_exact"] = diagnostics["captures"] == EXPECTED_CAPTURES
    diagnostics["patches_expected"] = EXPECTED_PATCHES
    diagnostics["patches_exact"] = diagnostics["patches"] == EXPECTED_PATCHES
    diagnostics["support"] = {
        cell: [int(masks[cell][500:750].sum()), int(masks[cell][750:1000].sum())]
        for cell in CELLS
    }
    return data, diagnostics


def effect_report(base, other, counts, lo: int, hi: int):
    report = {}
    for ci, cell in enumerate(CELLS):
        denominator = float(counts[lo:hi, ci].sum())
        report[cell] = {
            "effect_nat": float((base[lo:hi, ci] - other[lo:hi, ci]).sum()) / max(denominator, 1.0),
            "tokens": int(denominator),
        }
    return report


def cosine(left, right):
    left = torch.as_tensor(left, dtype=torch.float64)
    right = torch.as_tensor(right, dtype=torch.float64)
    return float(torch.dot(left, right) / max(
        float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)), 1e-30))


def metrics(left, right):
    left = torch.as_tensor(left, dtype=torch.float64)
    right = torch.as_tensor(right, dtype=torch.float64)
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    return {
        "cosine": cosine(left, right),
        "left_norm": left_norm,
        "right_norm": right_norm,
        "norm_ratio": max(left_norm, right_norm) / max(min(left_norm, right_norm), 1e-30),
        "right_projection_on_left": float(torch.dot(right, left)) / max(float(torch.dot(left, left)), 1e-30),
    }


def fit_report(reference, candidate):
    reference = torch.as_tensor(reference, dtype=torch.float64)
    candidate = torch.as_tensor(candidate, dtype=torch.float64)
    return action_parent.action_parent._fit_report(reference, candidate)


def sign_pattern(vector):
    return bool(vector[0] < 0 and vector[1] > 0 and vector[2] > 0 and vector[3] < 0)


def all_negative(vector):
    return all(value < 0 for value in vector)


def _window(data, lo: int, hi: int):
    reports = {source: {} for source in SOURCES}
    values = {source: {} for source in SOURCES}
    dividends = {source: {} for source in SOURCES}
    vectors = {source: {} for source in SOURCES}
    corrections = {}
    interactions = {}
    for si, source in enumerate(SOURCES):
        for mask in SUBSETS:
            reports[source][mask] = effect_report(
                data["base"], data["losses"][si, mask], data["counts"], lo, hi)
        direct = effect_report(data["base"], data["direct"][si], data["counts"], lo, hi)
        full = reports[source][0]
        for mask in SUBSETS:
            values[source][mask] = {
                cell: full[cell]["effect_nat"] - reports[source][mask][cell]["effect_nat"]
                for cell in CELLS
            }
            vectors[source][mask] = [values[source][mask][cell] for cell in CONTEXT_CELLS]
        for mask in SUBSETS[1:]:
            dividend = {}
            children = []
            child = mask
            while True:
                children.append(child)
                if child == 0:
                    break
                child = (child - 1) & mask
            for cell in CELLS:
                dividend[cell] = sum(
                    (-1) ** (mask.bit_count() - child.bit_count()) * values[source][child][cell]
                    for child in children
                )
            dividends[source][mask] = dividend
        corrections[source] = [
            full[cell]["effect_nat"] - direct[cell]["effect_nat"] for cell in CONTEXT_CELLS
        ]
        interactions[source] = [
            vectors[source][ALL_MASK][i] - vectors[source][TASK_MASK][i]
            - vectors[source][SUPPRESSOR_MASK][i]
            for i in range(len(CONTEXT_CELLS))
        ]

    wrong_reports = {source: {} for source in WRONG_SIGNS}
    wrong_vectors = {source: {} for source in WRONG_SIGNS}
    for wi, source in enumerate(WRONG_SIGNS):
        for mi, mask in enumerate(WRONG_MASKS):
            wrong_reports[source][mask] = effect_report(
                data["base"], data["wrong"][wi, mi], data["counts"], lo, hi)
        full = wrong_reports[source][0]
        for mask in WRONG_MASKS:
            wrong_vectors[source][mask] = [
                full[cell]["effect_nat"] - wrong_reports[source][mask][cell]["effect_nat"]
                for cell in CONTEXT_CELLS
            ]

    calibration = {}
    ci = CELLS.index("all_positive")
    positive = data["counts"][lo:hi, ci] > 0
    denom = data["counts"][lo:hi, ci].clamp_min(1)
    native_rows = (data["base"][lo:hi, ci] - data["losses"][0, 0, lo:hi, ci]) / denom
    native_total = float((data["base"][lo:hi, ci] - data["losses"][0, 0, lo:hi, ci]).sum())
    for si, source in enumerate(SOURCES):
        rows = (data["base"][lo:hi, ci] - data["losses"][si, 0, lo:hi, ci]) / denom
        total = float((data["base"][lo:hi, ci] - data["losses"][si, 0, lo:hi, ci]).sum())
        off = CELLS.index("off_target")
        off_change = float(
            (data["losses"][si, 0, lo:hi, off] - data["losses"][0, 0, lo:hi, off]).sum()
            / data["counts"][lo:hi, off].sum().clamp_min(1)
        )
        calibration[source] = {
            "recovery": total / native_total if abs(native_total) > 1e-30 else None,
            "task_effect": fit_report(native_rows[positive], rows[positive]),
            "off_target_source_minus_native_nat": off_change,
        }
    for wi, source in enumerate(WRONG_SIGNS):
        rows = (data["base"][lo:hi, ci] - data["wrong"][wi, 0, lo:hi, ci]) / denom
        calibration[source] = {"task_effect": fit_report(native_rows[positive], rows[positive])}
    return {
        "reports": reports,
        "subset_values": values,
        "mobius_dividends": dividends,
        "subset_vectors": vectors,
        "correction_vectors": corrections,
        "interaction_vectors": interactions,
        "wrong_reports": wrong_reports,
        "wrong_vectors": wrong_vectors,
        "calibration": calibration,
    }


def _component(window, source, component):
    if component == "I":
        return window["interaction_vectors"][source]
    return window["subset_vectors"][source][component]


def analyze(data):
    pooled = _window(data, 0, DOCUMENTS)
    halves = [_window(data, lo, hi) for lo, hi in ((0, 250), (250, 500))]
    component_masks = {"T": TASK_MASK, "G": SUPPRESSOR_MASK, "ALL": ALL_MASK, "I": "I"}
    source_metrics = {}
    for source in SOURCES:
        source_metrics[source] = {
            "task_to_correction": metrics(
                pooled["correction_vectors"][source], pooled["subset_vectors"][source][TASK_MASK]),
            "suppressor_to_correction": metrics(
                pooled["correction_vectors"][source], pooled["subset_vectors"][source][SUPPRESSOR_MASK]),
            "all_to_correction": metrics(
                pooled["correction_vectors"][source], pooled["subset_vectors"][source][ALL_MASK]),
            "interaction_norm": float(torch.linalg.vector_norm(torch.tensor(
                pooled["interaction_vectors"][source], dtype=torch.float64))),
        }
    comparisons = {}
    for left, right in itertools.combinations(SOURCES, 2):
        comparisons[f"{left}:{right}"] = {
            name: metrics(_component(pooled, left, mask), _component(pooled, right, mask))
            for name, mask in component_masks.items()
        }
    half_comparisons = []
    for half_index, half in enumerate(halves):
        for left, right in itertools.combinations(SOURCES, 2):
            half_comparisons.append({
                "half": half_index,
                "left": left,
                "right": right,
                "components": {
                    name: metrics(_component(half, left, mask), _component(half, right, mask))
                    for name, mask in component_masks.items()
                },
            })
    orientation = []
    for correct, wrong in (("Z7", "W7"), ("Z8", "W8")):
        for half_index, half in enumerate(halves):
            for name, mask in (("T", TASK_MASK), ("ALL", ALL_MASK)):
                correct_cos = cosine(_component(half, "N", mask), _component(half, correct, mask))
                wrong_cos = cosine(_component(half, "N", mask), half["wrong_vectors"][wrong][mask])
                orientation.append({
                    "correct": correct,
                    "wrong": wrong,
                    "half": half_index,
                    "component": name,
                    "correct_cosine": correct_cos,
                    "wrong_cosine": wrong_cos,
                    "margin": correct_cos - wrong_cos,
                })
    return {
        "pooled": pooled,
        "halves": halves,
        "source_metrics": source_metrics,
        "pooled_source_comparisons": comparisons,
        "half_source_comparisons": half_comparisons,
        "orientation_comparisons": orientation,
    }


def score(analysis, diagnostics):
    pooled = analysis["pooled"]
    halves = analysis["halves"]
    pred_a = bool(
        diagnostics["native_replay_logit_max_abs"] == 0.0
        and diagnostics["native_replay_relative_squared"] <= 1e-12
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["minimum_score_edit_rms"] > 0
        and all(value > 0 for value in diagnostics["source_patch_rms_max"].values())
        and diagnostics["calls_exact"] and diagnostics["captures_exact"] and diagnostics["patches_exact"]
        and all(min(values) > 0 for values in diagnostics["support"].values())
    )
    pred_b = bool(all(
        .65 <= half["calibration"][source]["recovery"] <= 1.40
        and half["calibration"][source]["task_effect"]["cosine"] >= .85
        and abs(half["calibration"][source]["off_target_source_minus_native_nat"]) <= .01
        for half in halves for source in ("P", "Z7", "Z8")
    ) and all(
        half["calibration"][source]["task_effect"]["cosine"] <= -.50
        for half in halves for source in WRONG_SIGNS
    ))

    def source_program_holds(source, *, natural_control=False):
        row = analysis["source_metrics"][source]
        pooled_holds = bool(
            sign_pattern(pooled["subset_vectors"][source][TASK_MASK])
            and row["task_to_correction"]["right_norm"] >= .015
            and row["task_to_correction"]["cosine"] >= .80
            and all_negative(pooled["subset_vectors"][source][SUPPRESSOR_MASK])
            and row["suppressor_to_correction"]["cosine"] < .70
            and row["interaction_norm"] >= .005
            and row["all_to_correction"]["cosine"] >= .80
            and .40 <= row["all_to_correction"]["right_projection_on_left"] <= 1.60
        )
        half_holds = all(
            sign_pattern(half["subset_vectors"][source][TASK_MASK])
            and all_negative(half["subset_vectors"][source][SUPPRESSOR_MASK])
            for half in halves
        )
        return pooled_holds and half_holds

    np_comparison = analysis["pooled_source_comparisons"]["N:P"]
    np_similarity = bool(
        all(np_comparison[name]["cosine"] >= .85 and np_comparison[name]["norm_ratio"] <= 2.5
            for name in ("T", "G", "ALL"))
        and np_comparison["I"]["cosine"] >= .75 and np_comparison["I"]["norm_ratio"] <= 2.5
        and all(row["components"][name]["cosine"] > 0
                for row in analysis["half_source_comparisons"]
                if {row["left"], row["right"]} == {"N", "P"}
                for name in ("T", "G", "ALL", "I"))
    )
    pred_c = bool(source_program_holds("N") and source_program_holds("P") and np_similarity)

    gauge_pairs = [key for key in analysis["pooled_source_comparisons"]
                   if "Z7" in key or "Z8" in key]
    gauge_similarity = bool(all(
        comparison[name]["cosine"] >= (.70 if name == "I" else .80)
        and comparison[name]["norm_ratio"] <= 3.0
        for key, comparison in analysis["pooled_source_comparisons"].items()
        if key in gauge_pairs and (key.startswith("N:") or key.startswith("P:"))
        for name in ("T", "G", "ALL", "I")
    ) and all(
        row["components"][name]["cosine"] > 0
        for row in analysis["half_source_comparisons"]
        if (row["left"] in ("N", "P") and row["right"] in ("Z7", "Z8"))
        for name in ("T", "G", "ALL", "I")
    ))
    pred_d = bool(source_program_holds("Z7") and source_program_holds("Z8") and gauge_similarity)
    pred_e = bool(all(row["margin"] >= .25 for row in analysis["orientation_comparisons"]))
    task_norms = [analysis["source_metrics"][source]["task_to_correction"]["right_norm"]
                  for source in ("N", "P")]
    all_inert = all(abs(pooled["subset_values"][source][ALL_MASK][cell]) < 1e-6
                    for source in SOURCES for cell in CONTEXT_CELLS)
    indistinguishable_wrong = all(row["margin"] < .05 for row in analysis["orientation_comparisons"])
    strong_null = bool(
        not pred_a or not pred_b or all(value < .005 for value in task_norms)
        or np_comparison["T"]["cosine"] <= 0 or all_inert or indistinguishable_wrong
    )
    return pred_a, pred_b, pred_c, pred_d, pred_e, strong_null


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(SUBSETS) == 32 and tuple(subset_sites(ALL_MASK)) == SITES
        assert EXPECTED_FORWARDS == 17875 and EXPECTED_PATCHES == 52000
        toy = torch.tensor([1.0, 2.0, -1.0, 3.0])
        assert math.isclose(cosine(toy, toy), 1.0)
        print(json.dumps({
            "status": "dry_run_passed",
            "rung": 505,
            "model_loaded": False,
            "subset_outcomes_opened": False,
            "documents": [500, 1000],
            "sources": SOURCES,
            "wrong_sign_controls": WRONG_SIGNS,
            "subsets": len(SUBSETS),
            "expected_forwards": EXPECTED_FORWARDS,
            "expected_patches": EXPECTED_PATCHES,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung 505 output namespace already exists")
    rows, masks, scales, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data, diagnostics = collect(model, rows, masks, scales)
    analysis = analyze(data)
    pred_a, pred_b, pred_c, pred_d, pred_e, strong_null = score(analysis, diagnostics)
    pred_a = bool(pred_a and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
    strong_null = bool(strong_null or not pred_a)
    bundle = {
        "schema": "equality_score_gauged_downstream_program_rung505_ce_sums_v1",
        **data,
        "raw_rows_tokens_logits_or_hidden_states_included": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete",
        "rung": 505,
        "claim_level": "fixed_natural_five_site_program_across_sign_gauged_scores_not_internal_split_or_compression",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "sources": list(SOURCES),
        "wrong_sign_controls": list(WRONG_SIGNS),
        "sites": list(SITES),
        "all_later_sites": list(ALL_LATER_SITES),
        "subset_masks": list(SUBSETS),
        "context_cells": list(CONTEXT_CELLS),
        "analysis": analysis,
        "instrument": diagnostics,
        'pred_a_exact_live_intervention_instrument': pred_a,
        'pred_b_score_actions_calibrated_in_patch_harness': pred_b,
        'pred_c_fixed_program_transfers_code_to_natural': pred_c,
        'pred_d_program_invariant_across_sign_gauge': pred_d,
        'pred_e_correct_gauge_orientation_specific': pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {
            "path": str(BUNDLE),
            "sha256": sha256(BUNDLE),
            "bytes": BUNDLE.stat().st_size,
        },
        "execution_price": {
            "full_model_forwards": sum(diagnostics["calls"].values()),
            "write_patches": diagnostics["patches"],
            "backwards": 0,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0,
            "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started,
        "next_step": (
            "split_MLP8_9_12_by_cross_source_downstream_finite_effect_then_exact_removal"
            if all((pred_a, pred_b, pred_c, pred_d, pred_e)) else
            "repair_instrument_only" if not pred_a else
            "repair_or_abandon_score_action_patch_assay" if not pred_b else
            "abandon_fixed_five_site_program_as_code_specific" if not pred_c else
            "retain_score_gauge_only_downstream_realization_is_source_dependent" if not pred_d else
            "register_shifted_or_sign_control_before_one_signed_program_claim"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"],
        "rung": 505,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "source_metrics": analysis["source_metrics"],
        "pooled_source_comparisons": analysis["pooled_source_comparisons"],
        "orientation_comparisons": analysis["orientation_comparisons"],
        "instrument": diagnostics,
        "execution_price": result["execution_price"],
        "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
