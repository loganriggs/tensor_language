#!/usr/bin/env python3
"""RUNG498 -- calibrate finite causal-action grouping on the equality matcher."""

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
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import circuit_induction_tensor as induction
import equality_term_score_payload_rung459 as factor_parent
import mlp0_branch_circuit_response_rung481 as census_parent


PREREG = POLY / "EQUALITY_MATCHER_CAUSAL_ACTION_QUOTIENT_RUNG498_PREREGISTRATION.md"
R459_SOURCE = ROOT / "ops/equality_term_score_payload_rung459.py"
R459_RESULT = ROOT / "equality_term_score_payload_rung459_results.json"
R481_SOURCE = ROOT / "ops/mlp0_branch_circuit_response_rung481.py"
R481_RESULT = ROOT / "mlp0_branch_circuit_response_rung481_results.json"
OUT = ROOT / "equality_matcher_causal_action_quotient_rung498_results.json"
BUNDLE = ROOT / "equality_matcher_causal_action_quotient_rung498_bundle.pt"
HASHES = {
    PREREG: "b017baaa76ec6ec787775ece21eed930a8482fed9acf16a37a30e0d81c7a6998",
    R459_SOURCE: "9f9e66f689452cbcb14d741792b66eb9ff526dff5472a5938c58a2a4c82620d8",
    R459_RESULT: "f157681ced170cbf8664db5710414a38d4f928f8d15dc0dd2b4d8cea9288aefa",
    R481_SOURCE: "ef08017a30ceb0c9e4481198fc1d58c5b0bf8cd37707d2223c42db9eb04f1f44",
    R481_RESULT: "2af2e9d934d85223cb01cb731ad2bcbe54b8b90cbf21f6ba6753cc1347e84573",
}
DONORS = ("L5H5", "L7H3")
PAIRS = ((0, 3), (1, 3))
BACKGROUNDS = ("early_present", "early_absent")
STATES = ("late_native", "late_absent", "score_donor", "payload_donor", "whole_donor")
TASKS = ("all_positive", "near_positive", "far_positive",
         "one_predecessor_positive", "multiple_predecessor_positive", "off_target")
DISCOVERY = (0, 500, 250)
VALIDATION = (500, 1000, 750)
BATCH = 4
TOKENS = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cosine(left, right) -> float:
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator.clamp_min(1e-30))


def _fit_report(native, hybrid):
    native = torch.as_tensor(native, dtype=torch.float64).reshape(-1)
    hybrid = torch.as_tensor(hybrid, dtype=torch.float64).reshape(-1)
    scale = float(torch.dot(hybrid, native) / hybrid.square().sum().clamp_min(1e-30))
    fitted = max(scale, 0.0) * hybrid
    residual = float(torch.linalg.vector_norm(native - fitted)
                     / torch.linalg.vector_norm(native).clamp_min(1e-30))
    return {"cosine": _cosine(native, hybrid), "positive_fit_scale": scale,
            "scaled_residual": residual}


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    parent = json.loads(R459_RESULT.read_text())
    if not all(parent.get(key) is True for key in (
        "pred_a_instrument", "pred_b_factor_candidate", "pred_c_response_transfer",
        "pred_d_causal_effect", "pred_e_between_control")) or parent.get("strong_null"):
        raise RuntimeError("rung459 positive authority changed")
    rows, circuit_masks, discovery_tags, validation_tags, _fit, metadata = \
        census_parent.validate_inputs()
    if len(rows) != 1000 or len(discovery_tags) != 32 or len(validation_tags) != 30:
        raise RuntimeError("census partition changed")
    scales = {
        donor: parent["frozen_fit_scales"][f"{donor}->L8H4"] for donor in DONORS
    }
    return rows, circuit_masks, discovery_tags, validation_tags, scales, metadata


def build_task_masks(rows):
    inputs = rows[:, :-1]
    support = induction.induction_fetch_mask(inputs)
    positive = support.any(-1)
    positive[:, :64] = False
    masks = {name: torch.zeros_like(positive) for name in TASKS}
    masks["all_positive"] = positive
    for row_index, row in enumerate(inputs):
        for query in range(64, TOKENS):
            if not bool(positive[row_index, query]):
                continue
            predecessors = torch.nonzero(row[:query] == row[query], as_tuple=False).flatten()
            distance = query - int(predecessors[-1])
            masks["near_positive" if distance <= 16 else "far_positive"][row_index, query] = True
            masks["one_predecessor_positive" if len(predecessors) == 1
                  else "multiple_predecessor_positive"][row_index, query] = True
    masks["off_target"][:, 64:] = ~positive[:, 64:]
    if not torch.equal(masks["near_positive"] | masks["far_positive"], positive):
        raise RuntimeError("distance masks do not partition equality positions")
    if not torch.equal(masks["one_predecessor_positive"]
                       | masks["multiple_predecessor_positive"], positive):
        raise RuntimeError("multiplicity masks do not partition equality positions")
    return masks


def _nll(logits, rows):
    targets = rows[:, 1:].to(logits.device)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           reduction="none").view(len(rows), -1).float().cpu()


@torch.no_grad()
def run_forward(model, tokens, *, pair=None, background="early_present",
                state="late_native", scales=None, direct=False):
    if background not in BACKGROUNDS or state not in STATES:
        raise ValueError("unregistered action")
    if direct and pair is not None:
        raise ValueError("direct native forward cannot carry a donor")
    if not direct and pair not in PAIRS and pair is not None:
        raise ValueError("unregistered donor pair")
    if state.endswith("donor") and (pair is None or scales is None):
        raise ValueError("hybrid requires donor and frozen scales")
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
            early, late = pair
            if event.site == factor_parent.TERMS[early][1]:
                cached.update(factors[early])
                if background == "early_absent":
                    edit = factors[early]["native_term"]
                    write = write - edit
                    diagnostics["early_edit_rms"] = float(edit.float().square().mean().sqrt())
            if event.site == factor_parent.TERMS[late][1]:
                if not cached:
                    raise RuntimeError("donor factors unavailable at recipient")
                late_factor = factors[late]
                if state != "late_native":
                    replacement = torch.zeros_like(late_factor["factor_term"])
                    if state.endswith("donor"):
                        p, u = late_factor["p"], late_factor["u"]
                        if state in ("score_donor", "whole_donor"):
                            p = cached["p"] * scales["score_ratio"]
                        if state in ("payload_donor", "whole_donor"):
                            u = cached["u"] * scales["payload_ratio"]
                        replacement = torch.bmm(p * support, u)
                    edit = replacement.to(write.dtype) - late_factor["native_term"]
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


def _phase_selections(circuit_masks, tags, start, stop, split, positive):
    rows = torch.arange(start, stop)
    output = []
    for half, row_condition in enumerate((rows < split, rows >= split)):
        for tag_index, tag in enumerate(tags):
            pair = []
            for mask_type in ("member", "slice_control"):
                selected = circuit_masks[tag][mask_type].view(1000, TOKENS)[start:stop].clone()
                selected &= positive[start:stop]
                selected &= row_condition[:, None]
                pair.append(selected)
            output.append((half, tag_index, pair[0], pair[1]))
    return output


@torch.no_grad()
def collect_phase(model, rows, circuit_masks, tags, task_masks, scales, bounds):
    start_doc, stop_doc, split = bounds
    docs = stop_doc - start_doc
    nll = torch.zeros(len(DONORS), len(BACKGROUNDS), len(STATES), docs, TOKENS)
    replay = {"logit_max_abs": 0.0, "logit_relative_squared": 0.0}
    diagnostics = {"factor_reconstruction_max": 0.0, "minimum_nonzero_edit_rms": math.inf,
                   "zero_intended_edit_actions": 0}
    calls = {"native": 0, "analytical": 0}
    device = next(model.parameters()).device
    for start in range(start_doc, stop_doc, BATCH):
        stop = min(start + BATCH, stop_doc)
        local = start - start_doc
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        native_logits, _diag, _audit = run_forward(model, tokens, direct=True)
        replay_logits, diag, _audit = run_forward(model, tokens, pair=None)
        calls["native"] += 1
        calls["analytical"] += 1
        difference = native_logits.float() - replay_logits.float()
        replay["logit_max_abs"] = max(replay["logit_max_abs"], float(difference.abs().max()))
        replay["logit_relative_squared"] = max(
            replay["logit_relative_squared"],
            float(difference.double().square().sum()
                  / native_logits.double().square().sum().clamp_min(1e-30)))
        diagnostics["factor_reconstruction_max"] = max(
            diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
        replay_nll = _nll(replay_logits, batch_rows)
        for donor_index, (donor, pair) in enumerate(zip(DONORS, PAIRS)):
            for background_index, background in enumerate(BACKGROUNDS):
                for state_index, state in enumerate(STATES):
                    if background == "early_present" and state == "late_native":
                        nll[donor_index, background_index, state_index, local:local + len(batch_rows)] = replay_nll
                        continue
                    logits, diag, _audit = run_forward(
                        model, tokens, pair=pair, background=background, state=state,
                        scales=scales[donor])
                    calls["analytical"] += 1
                    nll[donor_index, background_index, state_index,
                        local:local + len(batch_rows)] = _nll(logits, batch_rows)
                    diagnostics["factor_reconstruction_max"] = max(
                        diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
                    if background == "early_absent" and diag["early_edit_rms"] <= 0:
                        diagnostics["zero_intended_edit_actions"] += 1
                    if state != "late_native" and diag["late_edit_rms"] <= 0:
                        diagnostics["zero_intended_edit_actions"] += 1
                    for key in ("early_edit_rms", "late_edit_rms"):
                        if diag[key] > 0:
                            diagnostics["minimum_nonzero_edit_rms"] = min(
                                diagnostics["minimum_nonzero_edit_rms"], diag[key])
                    del logits
        del native_logits, replay_logits
    batches = math.ceil(docs / BATCH)
    expected_calls = {"native": batches, "analytical": 19 * batches}
    diagnostics["calls"] = calls
    diagnostics["expected_calls"] = expected_calls
    diagnostics["calls_exact"] = calls == expected_calls
    selections = _phase_selections(
        circuit_masks, tags, start_doc, stop_doc, split, task_masks["all_positive"])
    support = {task: [int(task_masks[task][start_doc:split].sum()),
                      int(task_masks[task][split:stop_doc].sum())] for task in TASKS}
    circuit_support = torch.zeros(2, len(tags), 2, dtype=torch.int64)
    for half, tag_index, member, control in selections:
        circuit_support[half, tag_index, 0] = int(member.sum())
        circuit_support[half, tag_index, 1] = int(control.sum())
    return {"nll": nll, "support": support, "circuit_support": circuit_support,
            "diagnostics": diagnostics, "replay": replay}


def analyze_phase(collection, circuit_masks, tags, task_masks, bounds):
    start_doc, stop_doc, split = bounds
    nll = collection["nll"].double()
    results = {}
    selections = _phase_selections(
        circuit_masks, tags, start_doc, stop_doc, split, task_masks["all_positive"])
    for donor_index, donor in enumerate(DONORS):
        results[donor] = {}
        for background_index, background in enumerate(BACKGROUNDS):
            results[donor][background] = {}
            absent = nll[donor_index, background_index, STATES.index("late_absent")]
            native = nll[donor_index, background_index, STATES.index("late_native")]
            native_effect = absent - native
            for state in ("score_donor", "payload_donor", "whole_donor"):
                hybrid = nll[donor_index, background_index, STATES.index(state)]
                hybrid_effect = absent - hybrid
                halves = []
                for half, (lo, hi) in enumerate(((0, split - start_doc),
                                                  (split - start_doc, stop_doc - start_doc))):
                    positive = task_masks["all_positive"][start_doc + lo:start_doc + hi]
                    off = task_masks["off_target"][start_doc + lo:start_doc + hi]
                    per_doc_native, per_doc_hybrid = [], []
                    for row in range(lo, hi):
                        selected = positive[row - lo]
                        count = int(selected.sum())
                        per_doc_native.append(float(native_effect[row, selected].sum()) / max(count, 1))
                        per_doc_hybrid.append(float(hybrid_effect[row, selected].sum()) / max(count, 1))
                    task_fit = _fit_report(per_doc_native, per_doc_hybrid)
                    native_sum = float(native_effect[lo:hi][positive].sum())
                    hybrid_sum = float(hybrid_effect[lo:hi][positive].sum())
                    recovery = hybrid_sum / native_sum if abs(native_sum) > 1e-30 else None
                    off_change = float((hybrid[lo:hi] - native[lo:hi])[off].mean())
                    fingerprint_native = torch.zeros(len(tags), dtype=torch.float64)
                    fingerprint_hybrid = torch.zeros(len(tags), dtype=torch.float64)
                    for selected_half, tag_index, member, control in selections:
                        if selected_half != half:
                            continue
                        member = member[lo:hi]
                        control = control[lo:hi]
                        fingerprint_native[tag_index] = (
                            native_effect[lo:hi][member].mean() - native_effect[lo:hi][control].mean())
                        fingerprint_hybrid[tag_index] = (
                            hybrid_effect[lo:hi][member].mean() - hybrid_effect[lo:hi][control].mean())
                    halves.append({
                        "equality_recovery": recovery,
                        "task_effect": task_fit,
                        "off_target_hybrid_minus_native_nat": off_change,
                        "circuit_effect": _fit_report(fingerprint_native, fingerprint_hybrid),
                        "native_circuit_fingerprint": fingerprint_native.tolist(),
                        "hybrid_circuit_fingerprint": fingerprint_hybrid.tolist(),
                    })
                results[donor][background][state] = halves
    return results


def score_discovery(analysis, collection):
    instrument = collection["diagnostics"]
    pred_a = bool(
        collection["replay"]["logit_max_abs"] == 0.0
        and collection["replay"]["logit_relative_squared"] <= 1e-12
        and instrument["factor_reconstruction_max"] <= 1e-10
        and instrument["minimum_nonzero_edit_rms"] > 0
        and instrument["zero_intended_edit_actions"] == 0
        and instrument["calls_exact"]
        and all(min(values) > 0 for values in collection["support"].values())
        and bool((collection["circuit_support"] > 0).all()))
    positives = [analysis["L5H5"][background]["score_donor"][half]
                 for background in BACKGROUNDS for half in range(2)]
    pred_b = all(
        row["equality_recovery"] is not None
        and .75 <= row["equality_recovery"] <= 1.30
        and row["task_effect"]["cosine"] >= .75
        and row["task_effect"]["scaled_residual"] <= .70
        and abs(row["off_target_hybrid_minus_native_nat"]) <= .01
        for row in positives)
    separation = []
    for background in BACKGROUNDS:
        for half in range(2):
            positive = analysis["L5H5"][background]["score_donor"][half]
            controls = (
                analysis["L7H3"][background]["score_donor"][half],
                analysis["L5H5"][background]["payload_donor"][half])
            for control in controls:
                similarity_wins = (
                    positive["task_effect"]["cosine"]
                    >= control["task_effect"]["cosine"] + .15
                    or positive["task_effect"]["scaled_residual"]
                    <= control["task_effect"]["scaled_residual"] - .20)
                recovery_wins = (
                    abs(control["equality_recovery"] - 1)
                    >= abs(positive["equality_recovery"] - 1) + .20)
                separation.append(bool(similarity_wins and recovery_wins))
    pred_c = all(separation)
    closure = []
    for half in range(2):
        present = analysis["L5H5"]["early_present"]["score_donor"][half]
        absent = analysis["L5H5"]["early_absent"]["score_donor"][half]
        scales = [present["circuit_effect"]["positive_fit_scale"],
                  absent["circuit_effect"]["positive_fit_scale"]]
        scale_drift = abs(scales[0] - scales[1]) / max(abs(scales[0]), abs(scales[1]), 1e-30)
        background_cosine = _cosine(
            present["hybrid_circuit_fingerprint"], absent["hybrid_circuit_fingerprint"])
        closure.append({
            "half": half, "scale_drift": scale_drift,
            "hybrid_background_cosine": background_cosine,
            "holds": bool(
                present["circuit_effect"]["cosine"] >= .70
                and absent["circuit_effect"]["cosine"] >= .70
                and present["circuit_effect"]["scaled_residual"] <= .75
                and absent["circuit_effect"]["scaled_residual"] <= .75
                and min(scales) > 0 and scale_drift <= .50 and background_cosine >= .75),
        })
    pred_d = all(row["holds"] for row in closure)
    return pred_a, pred_b, pred_c, pred_d, {"control_cells": separation,
                                           "background_closure": closure}


def score_validation(analysis):
    checks = []
    for background in BACKGROUNDS:
        for half in range(2):
            positive = analysis["L5H5"][background]["score_donor"][half]
            controls = (analysis["L7H3"][background]["score_donor"][half],
                        analysis["L5H5"][background]["payload_donor"][half])
            basic = bool(
                .65 <= positive["equality_recovery"] <= 1.40
                and positive["task_effect"]["cosine"] >= .65
                and positive["circuit_effect"]["cosine"] >= .60
                and positive["circuit_effect"]["scaled_residual"] <= .80
                and positive["circuit_effect"]["positive_fit_scale"] > 0)
            control_checks = []
            for control in controls:
                control_checks.append(bool(
                    (positive["task_effect"]["cosine"] >= control["task_effect"]["cosine"] + .10
                     or positive["task_effect"]["scaled_residual"]
                     <= control["task_effect"]["scaled_residual"] - .15)
                    and abs(control["equality_recovery"] - 1)
                    >= abs(positive["equality_recovery"] - 1) + .15))
            checks.append(basic and all(control_checks))
    for half in range(2):
        present = analysis["L5H5"]["early_present"]["score_donor"][half]
        absent = analysis["L5H5"]["early_absent"]["score_donor"][half]
        checks.append(_cosine(present["hybrid_circuit_fingerprint"],
                              absent["hybrid_circuit_fingerprint"]) >= .65)
    return all(checks), checks


def _serial_collection(collection):
    return {"support": collection["support"],
            "circuit_support": collection["circuit_support"].tolist(),
            "diagnostics": collection["diagnostics"], "replay": collection["replay"]}


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(DONORS) == 2 and len(BACKGROUNDS) == 2 and len(STATES) == 5
        assert 1 + 19 == 20 and DISCOVERY == (0, 500, 250) and VALIDATION == (500, 1000, 750)
        native = torch.tensor([1.0, 2.0, -1.0])
        exact = _fit_report(native, native)
        null = _fit_report(native, torch.tensor([-1.0, 0.0, 1.0]))
        assert exact["cosine"] > .999 and exact["scaled_residual"] < 1e-12
        assert null["positive_fit_scale"] < 0 and null["scaled_residual"] == 1.0
        print(json.dumps({
            "status": "dry_run_passed", "rung": 498, "model_loaded": False,
            "discovery_forwards": 2500, "conditional_validation_forwards": 2500,
            "validation_opened": False, "raw_rows_logits_or_hidden_states_stored": False,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung498 output namespace already exists")
    rows, circuit_masks, discovery_tags, validation_tags, scales, metadata = validate_inputs()
    task_masks = build_task_masks(rows)
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    discovery_collection = collect_phase(
        model, rows, circuit_masks, discovery_tags, task_masks, scales, DISCOVERY)
    discovery = analyze_phase(
        discovery_collection, circuit_masks, discovery_tags, task_masks, DISCOVERY)
    pred_a, pred_b, pred_c, pred_d, discovery_checks = score_discovery(
        discovery, discovery_collection)
    pred_a = bool(pred_a and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
    validation_licensed = bool(pred_a and pred_b and pred_c and pred_d)
    validation_collection = validation = None
    validation_checks = []
    pred_e = False
    if validation_licensed:
        validation_collection = collect_phase(
            model, rows, circuit_masks, validation_tags, task_masks, scales, VALIDATION)
        validation = analyze_phase(
            validation_collection, circuit_masks, validation_tags, task_masks, VALIDATION)
        validation_science, validation_checks = score_validation(validation)
        validation_instrument, *_ = score_discovery(validation, validation_collection)
        pred_e = bool(validation_instrument and validation_science)
    pred_f = bool(pred_a and pred_b and pred_c and pred_d and pred_e)
    bundle = {
        "schema": "equality_matcher_causal_action_quotient_rung498_nll_v1",
        "discovery_nll": discovery_collection["nll"],
        "validation_nll": None if validation_collection is None else validation_collection["nll"],
        "raw_rows_tokens_logits_or_hidden_states_included": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 498,
        "claim_level": "known_positive_finite_action_quotient_calibration_not_discovery_or_compression",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "donors": list(DONORS),
        "backgrounds": list(BACKGROUNDS), "states": list(STATES),
        "frozen_scales": scales,
        "discovery": {"bounds": list(DISCOVERY), "tags": discovery_tags,
                      "collection": _serial_collection(discovery_collection),
                      "analysis": discovery, "checks": discovery_checks},
        "validation": None if validation is None else {
            "bounds": list(VALIDATION), "tags": validation_tags,
            "collection": _serial_collection(validation_collection),
            "analysis": validation, "checks": validation_checks},
        "validation_licensed_and_opened": validation_licensed,
        'pred_a_exact_lawful_live_instrument': pred_a,
        'pred_b_known_positive_recovered': pred_b,
        'pred_c_fixed_controls_rejected': pred_c,
        'pred_d_closed_under_early_removal': pred_d,
        'pred_e_heldout_documents_and_circuits': pred_e,
        'pred_f_calibrated_action_quotient': pred_f,
        "strong_null": bool(not pred_a or not pred_b or not pred_c or not pred_d),
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "discovery_forwards": sum(discovery_collection["diagnostics"]["calls"].values()),
            "validation_forwards": 0 if validation_collection is None else
            sum(validation_collection["diagnostics"]["calls"].values()),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started,
        "next_step": (
            "register_new_candidate_search_with_frozen_action_semantics" if pred_f else
            "repair_instrument_only" if not pred_a else
            "abandon_census_action_assay" if not pred_b else
            "change_observation_set_before_quotient_search" if not pred_c else
            "record_background_conditional_reuse_not_quotient" if not pred_d else
            "discovery_specific_calibration_no_search"),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 498,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": result["strong_null"], "validation_opened": validation_licensed,
        "execution_price": result["execution_price"], "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
