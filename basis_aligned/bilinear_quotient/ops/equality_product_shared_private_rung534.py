#!/usr/bin/env python3
"""RUNG534 -- shared equality product plus target-specific score correction."""

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
import equality_factor_to_slot_exchangeability_rung533 as parent
import equality_product_shared_private_rung534_math as math_contract


RUNG = 534
ROLES = parent.ROLES
BACKGROUNDS = parent.BACKGROUNDS
DOCUMENTS_PER_ROLE = parent.DOCUMENTS_PER_ROLE
DOCUMENT_SPLIT = parent.DOCUMENT_SPLIT
TOKENS = parent.TOKENS
BATCH = parent.BATCH
TASK_CELLS = parent.TASK_CELLS
ARMS = (
    "native", "absent", "shared", "private", "shared_key_control",
    "private_key_control", "private_sign_control",
)
BATCHES_PER_ROLE = DOCUMENTS_PER_ROLE // BATCH
FORWARDS_PER_BATCH = 1 + len(BACKGROUNDS) * len(ARMS)
FORWARDS = len(ROLES) * BATCHES_PER_ROLE * FORWARDS_PER_BATCH

PREREG = POLY / "EQUALITY_PRODUCT_SHARED_PRIVATE_RUNG534_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/equality_factor_to_slot_exchangeability_rung533.py"
PARENT_RESULT = ROOT / "equality_factor_to_slot_exchangeability_rung533_results.json"
MATH_SOURCE = ROOT / "ops/equality_product_shared_private_rung534_math.py"
MATH_RESULT = ROOT / "equality_product_shared_private_rung534_math.json"
OUT = ROOT / "equality_product_shared_private_rung534_results.json"
BUNDLE = ROOT / "equality_product_shared_private_rung534_bundle.pt"
HASHES = {
    PREREG: "47d738db728c24b8b8d1105e8467905815312fc6d207edad8680a46b8d7de428",
    PARENT_SOURCE: "6ba3a9e5fa4e0fa23c461610451bfc8d65eea909f14fe563131a1441228528fd",
    PARENT_RESULT: "5c43872a037f662ab93c64915e74419439513393f026654d8ed16c7bdb7f84d0",
    MATH_SOURCE: "61c3dbf88718f979258c0a828d741515deb1fbf011ec237eba2396c849825be4",
    MATH_RESULT: "97cc35d57b677132a6ff6196e1ceb3c264a230005404a6e13d951d77914a2ddf",
}

PREDICTION_TEXT = {
    "pred_a_exact_live_instrument": "exact replay, product split, edits, supports, calls, rows, and sources",
    "pred_b_shared_signal_premise_reproduces": "shared product retains code copy effect but misses negatives",
    "pred_c_private_correction_autonomous_on_code": "private standalone predicts its code marginal effect",
    "pred_d_private_correction_key_specific": "private beats key-reversed and sign controls",
    "pred_e_private_correction_transfers_to_natural": "autonomous private correction transfers to natural text",
    "pred_f_private_correction_survives_redundant_donor": "autonomy survives donor-present backgrounds",
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
    result = json.loads(PARENT_RESULT.read_text())
    if (
        result.get("rung") != 533
        or result.get("pred_a_valid_physical_instrument") is not True
        or result.get("pred_b_product_level_positive_control") is not False
        or result.get("pred_e_branch_exchangeable_downstream_family") is not False
        or result.get("pred_f_donor_background_stability") is not False
    ):
        raise RuntimeError("rung533 result authority changed")
    screen = json.loads(MATH_RESULT.read_text())
    if (
        screen.get("status") != "cpu_screen_passed"
        or screen.get("exact_algebra", {}).get("maximum_recomposition_error", 1) > 4e-16
        or screen.get("model_loaded") is not False
        or screen.get("new_scientific_outcomes_opened") is not False
    ):
        raise RuntimeError("rung534 CPU screen changed")
    payloads, metadata = parent.validate_inputs()
    return payloads, {
        **metadata,
        "rung533_result_sha256": HASHES[PARENT_RESULT],
        "rung534_math_result_sha256": HASHES[MATH_RESULT],
    }


def split_patterns(source_first, source_second, target_first, target_second):
    native = target_first * target_second
    shared = parent.parent.GAMMA * (source_first * source_second)
    private = native - shared
    return native, shared, private


def replacement_pattern(arm, source_first, source_second, target_first, target_second):
    native, shared, private = split_patterns(
        source_first, source_second, target_first, target_second)
    if arm == "native":
        return native
    if arm == "absent":
        return torch.zeros_like(native)
    if arm == "shared":
        return shared
    if arm == "private":
        return private
    if arm == "shared_key_control":
        return parent.math_contract.key_prefix_reverse(shared)
    if arm == "private_key_control":
        return parent.math_contract.key_prefix_reverse(private)
    if arm == "private_sign_control":
        return -private
    raise ValueError(f"unknown arm: {arm}")


@torch.no_grad()
def run_forward(model, tokens, *, background, arm, direct=False):
    if background not in BACKGROUNDS or arm not in ARMS:
        raise ValueError("unregistered background or arm")
    diagnostics = {
        "factor_reconstruction_max": 0.0, "branch_product_max_abs": 0.0,
        "score_recomposition_max_abs": 0.0,
        "donor_edit_rms": 0.0, "target_edit_rms": 0.0,
    }
    audit = {"native_attention": 0, "replayed_attention": 0, "native_mlp": 0}

    def attention(event):
        if direct or event.site not in parent.parent.factor_parent.stage1.SITE_HEADS:
            write, next_value = event.block.attn(event.state, event.first_value)
            audit["native_attention"] += 1
            return write, next_value
        write, factors, support, reconstruction = parent.parent.factor_parent._factor_site(
            event.state, event.first_value, event.block.attn, event.site, event.tokens)
        audit["replayed_attention"] += 1
        diagnostics["factor_reconstruction_max"] = max(
            diagnostics["factor_reconstruction_max"], reconstruction)
        if event.site == 8:
            first, second = parent.parent.factor_screen._score_branches(event.state, event.block.attn)
            source = factors[parent.parent.SOURCE_INDEX]
            target = factors[parent.parent.TARGET_INDEX]
            source_first_native = first[:, parent.parent.SOURCE_HEAD]
            source_second_native = second[:, parent.parent.SOURCE_HEAD]
            target_first_native = first[:, parent.parent.TARGET_HEAD]
            target_second_native = second[:, parent.parent.TARGET_HEAD]
            causal = torch.tril(torch.ones(
                target_first_native.shape[-2:], dtype=torch.bool,
                device=target_first_native.device))
            native_product = parent.parent.native_branch_product(
                target_first_native, target_second_native).masked_fill(~causal, 0.0)
            diagnostics["branch_product_max_abs"] = max(
                diagnostics["branch_product_max_abs"],
                float((native_product - target["p"]).abs().max()))
            source_first = source_first_native.float()
            source_second = source_second_native.float()
            target_first = target_first_native.float()
            target_second = target_second_native.float()
            native, shared, private = split_patterns(
                source_first, source_second, target_first, target_second)
            diagnostics["score_recomposition_max_abs"] = max(
                diagnostics["score_recomposition_max_abs"],
                float((native - (shared + private)).abs().max()))
            if background == "donor_absent":
                write = write - source["native_term"]
                diagnostics["donor_edit_rms"] = float(
                    source["native_term"].float().square().mean().sqrt())
            if arm != "native":
                pattern = replacement_pattern(
                    arm, source_first, source_second, target_first, target_second)
                replacement = torch.bmm(pattern * support, target["u"])
                edit = replacement.to(write.dtype) - target["native_term"]
                write = write + edit
                diagnostics["target_edit_rms"] = float(edit.float().square().mean().sqrt())
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


def empty_collection(payloads):
    return {
        "document_sums": {
            role: torch.zeros(
                len(BACKGROUNDS), len(ARMS), len(TASK_CELLS), DOCUMENTS_PER_ROLE,
                dtype=torch.float64)
            for role in ROLES
        },
        "document_counts": {
            role: torch.stack([
                payloads[role]["copy_cells"][cell].sum(-1).double()
                for cell in TASK_CELLS
            ])
            for role in ROLES
        },
    }


def _accumulate_nll(collection, role, nll, cells, background_index, arm_index, start, stop):
    for cell_index, cell in enumerate(TASK_CELLS):
        selected = cells[cell][start:stop].to(nll.device)
        collection["document_sums"][role][
            background_index, arm_index, cell_index, start:stop] += (
                nll.mul(selected).sum(-1).double().cpu())


@torch.no_grad()
def collect(model, payloads, *, smoke=False):
    collection = empty_collection(payloads)
    diagnostics = {
        "direct_native_calls": 0, "analytical_calls": 0,
        "native_replay_logit_max_abs": 0.0,
        "factor_reconstruction_max": 0.0, "branch_product_max_abs": 0.0,
        "score_recomposition_max_abs": 0.0,
        "minimum_donor_edit_rms": math.inf, "minimum_target_edit_rms": math.inf,
        "zero_intended_edits": 0, "roles_exercised": [],
    }
    roles = ROLES[:1] if smoke else ROLES
    device = next(model.parameters()).device
    for role in roles:
        rows = payloads[role]["rows"]
        cells = payloads[role]["copy_cells"]
        stop_document = BATCH if smoke else DOCUMENTS_PER_ROLE
        diagnostics["roles_exercised"].append(role)
        for start in range(0, stop_document, BATCH):
            stop = start + BATCH
            batch_rows = rows[start:stop]
            tokens = batch_rows[:, :-1].to(device)
            targets = batch_rows[:, 1:].to(device)
            direct_logits, _direct_diagnostics, _direct_audit = run_forward(
                model, tokens, background="donor_present", arm="native", direct=True)
            diagnostics["direct_native_calls"] += 1
            diagnostics["support_accumulator_exercised"] = True
            present_native_logits = None
            for background_index, background in enumerate(BACKGROUNDS):
                for arm_index, arm in enumerate(ARMS):
                    logits, row_diagnostics, _audit = run_forward(
                        model, tokens, background=background, arm=arm)
                    diagnostics["analytical_calls"] += 1
                    for key in (
                        "factor_reconstruction_max", "branch_product_max_abs",
                        "score_recomposition_max_abs",
                    ):
                        diagnostics[key] = max(diagnostics[key], row_diagnostics[key])
                    if background == "donor_absent":
                        if row_diagnostics["donor_edit_rms"] <= 0:
                            diagnostics["zero_intended_edits"] += 1
                        else:
                            diagnostics["minimum_donor_edit_rms"] = min(
                                diagnostics["minimum_donor_edit_rms"],
                                row_diagnostics["donor_edit_rms"])
                    if arm != "native":
                        if row_diagnostics["target_edit_rms"] <= 0:
                            diagnostics["zero_intended_edits"] += 1
                        else:
                            diagnostics["minimum_target_edit_rms"] = min(
                                diagnostics["minimum_target_edit_rms"],
                                row_diagnostics["target_edit_rms"])
                    if background == "donor_present" and arm == "native":
                        present_native_logits = logits
                        diagnostics["native_replay_logit_max_abs"] = max(
                            diagnostics["native_replay_logit_max_abs"],
                            float((direct_logits - logits).abs().max()))
                    if not smoke:
                        nll = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                            reduction="none").view(BATCH, -1)
                        _accumulate_nll(
                            collection, role, nll, cells, background_index, arm_index,
                            start, stop)
                    if logits is not present_native_logits:
                        del logits
            if present_native_logits is None:
                raise RuntimeError("present/native replay was not exercised")
            del direct_logits, present_native_logits, tokens, targets
    expected_batches = 1 if smoke else len(ROLES) * BATCHES_PER_ROLE
    diagnostics["calls_exact"] = bool(
        diagnostics["direct_native_calls"] == expected_batches
        and diagnostics["analytical_calls"]
        == expected_batches * len(BACKGROUNDS) * len(ARMS))
    if not smoke:
        diagnostics["all_document_supports_live"] = all(
            all(int((counts[cell_index, half * DOCUMENT_SPLIT:(half + 1) * DOCUMENT_SPLIT] > 0).sum()) >= 50
                for cell_index in range(len(TASK_CELLS)) for half in range(2))
            for counts in collection["document_counts"].values())
    return collection, diagnostics


def analyze(collection):
    reports, contexts = {}, []
    for role in ROLES:
        counts = collection["document_counts"][role]
        document_ce = collection["document_sums"][role] / counts[None, None].clamp_min(1)
        for background_index, background in enumerate(BACKGROUNDS):
            for half in range(2):
                start, stop = half * DOCUMENT_SPLIT, (half + 1) * DOCUMENT_SPLIT
                cell_reports = {}
                for cell_index, cell in enumerate(TASK_CELLS):
                    live = counts[cell_index, start:stop] > 0
                    values = document_ce[
                        background_index, :, cell_index, start:stop][:, live]
                    effects = values[ARMS.index("absent")][None] - values
                    native_effect = effects[ARMS.index("native")]
                    marginal = native_effect - effects[ARMS.index("shared")]
                    arm_reports = {
                        arm: parent.parent._vector_metrics(native_effect, effects[arm_index])
                        for arm_index, arm in enumerate(ARMS)
                    }
                    private_comparisons = {
                        arm: parent.parent._vector_metrics(marginal, effects[ARMS.index(arm)])
                        for arm in ("private", "private_key_control", "private_sign_control")
                    }
                    cell_reports[cell] = {
                        "arms_vs_native_effect": arm_reports,
                        "private_vs_marginal": private_comparisons,
                        "private_marginal_over_native_norm": float(
                            torch.linalg.vector_norm(marginal)
                            / torch.linalg.vector_norm(native_effect).clamp_min(1e-30)),
                        "documents": int(live.sum()),
                    }
                positive_counts = counts[TASK_CELLS.index("positive"), start:stop].sum()
                positive_sums = collection["document_sums"][role][
                    background_index, :, TASK_CELLS.index("positive"), start:stop].sum(-1)
                positive_ce = positive_sums / positive_counts.clamp_min(1)
                positive_reference = float(
                    positive_ce[ARMS.index("absent")] - positive_ce[ARMS.index("native")])
                positive_recovery = {
                    arm: (
                        float(positive_ce[ARMS.index("absent")] - positive_ce[arm_index])
                        / positive_reference if abs(positive_reference) > 1e-30 else None)
                    for arm_index, arm in enumerate(ARMS)
                }
                matched_counts = counts[
                    TASK_CELLS.index("matched_negative"), start:stop].sum()
                matched_sums = collection["document_sums"][role][
                    background_index, :, TASK_CELLS.index("matched_negative"), start:stop].sum(-1)
                matched_ce = matched_sums / matched_counts.clamp_min(1)
                shared_matched_mismatch = abs(float(
                    matched_ce[ARMS.index("shared")] - matched_ce[ARMS.index("native")]))
                key = f"{role}/{background}/half{half}"
                reports[key] = {
                    "cells": cell_reports,
                    "positive_recovery": positive_recovery,
                    "shared_matched_negative_abs_mean_ce_change_from_native": shared_matched_mismatch,
                }
                contexts.append(reports[key])
    return reports, contexts


def _shared_premise_holds(row):
    shared = row["cells"]["positive"]["arms_vs_native_effect"]["shared"]
    control = row["cells"]["positive"]["arms_vs_native_effect"]["shared_key_control"]
    recovery = row["positive_recovery"]["shared"]
    return bool(
        shared["cosine"] >= 0.85 and shared["relative_error"] <= 0.60
        and recovery is not None and 0.65 <= recovery <= 1.40
        and row["shared_matched_negative_abs_mean_ce_change_from_native"] >= 0.02
        and shared["cosine"] >= control["cosine"] + 0.15)


def _private_holds(row, *, controls):
    cells = ("positive", "matched_negative")
    base = all(
        row["cells"][cell]["private_vs_marginal"]["private"]["cosine"] >= 0.80
        and row["cells"][cell]["private_vs_marginal"]["private"]["relative_error"] <= 0.60
        for cell in cells)
    if not controls:
        return bool(base)
    return bool(base and all(
        row["cells"][cell]["private_vs_marginal"]["private"]["cosine"]
        >= row["cells"][cell]["private_vs_marginal"][control]["cosine"] + 0.15
        for cell in cells for control in ("private_key_control", "private_sign_control")))


def score(reports, contexts, diagnostics, checkpoint_hash):
    pred_a = bool(
        diagnostics["native_replay_logit_max_abs"] == 0.0
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["branch_product_max_abs"] == 0.0
        and diagnostics["score_recomposition_max_abs"] <= 2e-6
        and diagnostics["minimum_donor_edit_rms"] > 0
        and diagnostics["minimum_target_edit_rms"] > 0
        and diagnostics["zero_intended_edits"] == 0
        and diagnostics["calls_exact"]
        and diagnostics["all_document_supports_live"]
        and checkpoint_hash == facade.WEIGHTS_SHA256)
    code_absent = [reports[f"ood_code/donor_absent/half{half}"] for half in range(2)]
    natural_absent = [reports[f"final_natural/donor_absent/half{half}"] for half in range(2)]
    donor_present = [
        reports[f"{role}/donor_present/half{half}"]
        for role in ROLES for half in range(2)]
    pred_b = bool(pred_a and all(_shared_premise_holds(row) for row in code_absent))
    pred_c = bool(pred_a and all(_private_holds(row, controls=False) for row in code_absent))
    pred_d = bool(pred_c and all(_private_holds(row, controls=True) for row in code_absent))
    pred_e = bool(pred_c and pred_d and all(
        _private_holds(row, controls=True) for row in natural_absent))
    pred_f = bool(pred_c and pred_d and all(
        _private_holds(row, controls=True) for row in donor_present))
    predictions = dict(zip(PREDICTION_TEXT, (pred_a, pred_b, pred_c, pred_d, pred_e, pred_f)))
    detail = {
        key: {
            "private_base_cells_passing": sum(
                row["cells"][cell]["private_vs_marginal"]["private"]["cosine"] >= 0.80
                and row["cells"][cell]["private_vs_marginal"]["private"]["relative_error"] <= 0.60
                for cell in ("positive", "matched_negative")),
            "private_control_cells_passing": sum(
                row["cells"][cell]["private_vs_marginal"]["private"]["cosine"]
                >= row["cells"][cell]["private_vs_marginal"][control]["cosine"] + 0.15
                for cell in ("positive", "matched_negative")
                for control in ("private_key_control", "private_sign_control")),
        }
        for key, row in reports.items()
    }
    checks = {
        "shared_premise_code_absent_halves_passing": sum(
            _shared_premise_holds(row) for row in code_absent),
        "private_autonomy_code_absent_halves_passing": sum(
            _private_holds(row, controls=False) for row in code_absent),
        "private_with_controls_code_absent_halves_passing": sum(
            _private_holds(row, controls=True) for row in code_absent),
        "private_with_controls_natural_absent_halves_passing": sum(
            _private_holds(row, controls=True) for row in natural_absent),
        "private_with_controls_donor_present_contexts_passing": sum(
            _private_holds(row, controls=True) for row in donor_present),
        "context_detail": detail,
    }
    return predictions, checks


def main():
    started = time.time()
    smoke = os.environ.get("RUNG534_SMOKE") == "1"
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(ARMS) == 7 and FORWARDS_PER_BATCH == 15 and FORWARDS == 1440
        print(json.dumps({
            "status": "dry_run_passed", "rung": RUNG, "model_loaded": False,
            "scientific_outcomes_opened": False, "forwards": FORWARDS,
            "roles": ROLES, "arms": ARMS, "backgrounds": BACKGROUNDS,
            "predictions": list(PREDICTION_TEXT),
        }, indent=2, sort_keys=True))
        return
    if not smoke and (OUT.exists() or BUNDLE.exists()):
        raise RuntimeError("rung534 output namespace already exists")
    payloads, input_metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    collection, diagnostics = collect(model, payloads, smoke=smoke)
    if smoke:
        instrument_pass = bool(
            diagnostics["calls_exact"]
            and diagnostics.get("support_accumulator_exercised") is True
            and diagnostics["roles_exercised"] == ["final_natural"]
            and diagnostics["native_replay_logit_max_abs"] == 0.0
            and diagnostics["factor_reconstruction_max"] <= 1e-10
            and diagnostics["branch_product_max_abs"] == 0.0
            and diagnostics["score_recomposition_max_abs"] <= 2e-6
            and diagnostics["minimum_donor_edit_rms"] > 0
            and diagnostics["minimum_target_edit_rms"] > 0
            and diagnostics["zero_intended_edits"] == 0
            and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
        print(json.dumps({
            "status": "smoke_passed" if instrument_pass else "smoke_instrument_invalid",
            "instrument_pass": instrument_pass, "rung": RUNG,
            "scientific_outcomes_opened": False,
            "calls": diagnostics["direct_native_calls"] + diagnostics["analytical_calls"],
            "roles_exercised": diagnostics["roles_exercised"],
            "native_replay_logit_max_abs": diagnostics["native_replay_logit_max_abs"],
            "factor_reconstruction_max": diagnostics["factor_reconstruction_max"],
            "branch_product_max_abs": diagnostics["branch_product_max_abs"],
            "score_recomposition_max_abs": diagnostics["score_recomposition_max_abs"],
            "minimum_donor_edit_rms": diagnostics["minimum_donor_edit_rms"],
            "minimum_target_edit_rms": diagnostics["minimum_target_edit_rms"],
            "zero_intended_edits": diagnostics["zero_intended_edits"],
            "checkpoint_weights_sha256": checkpoint.weights_sha256,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }, indent=2, sort_keys=True))
        if not instrument_pass:
            raise RuntimeError("rung534 smoke instrument did not pass")
        return
    reports, contexts = analyze(collection)
    predictions, checks = score(reports, contexts, diagnostics, checkpoint.weights_sha256)
    strong_null = bool(
        predictions["pred_a_exact_live_instrument"]
        and predictions["pred_b_shared_signal_premise_reproduces"]
        and any(all(
            not (
                reports[f"ood_code/donor_absent/half{half}"]["cells"][cell]
                ["private_vs_marginal"]["private"]["cosine"] >= 0.80
                and reports[f"ood_code/donor_absent/half{half}"]["cells"][cell]
                ["private_vs_marginal"]["private"]["relative_error"] <= 0.60)
            for half in range(2)) for cell in ("positive", "matched_negative")))
    result = {
        "status": "completed", "rung": RUNG, **predictions,
        "interaction_only_correction_strong_null": strong_null,
        "invalid": bool(not predictions["pred_a_exact_live_instrument"]
                        or not predictions["pred_b_shared_signal_premise_reproduces"]),
        "reports": reports, "checks": checks, "diagnostics": diagnostics,
        "decomposition": {
            "shared": "gamma*source_product",
            "private": "target_product-gamma*source_product",
            "gamma": parent.parent.GAMMA,
        },
        "roles": list(ROLES), "document_halves": [[0, 96], [96, 192]],
        "task_cells": list(TASK_CELLS),
        "price": {
            "model_forwards": diagnostics["direct_native_calls"] + diagnostics["analytical_calls"],
            "backward_passes": 0, "fitted_vector_parameters": 0,
            "arms_per_background": len(ARMS),
        },
        "checkpoint": checkpoint, "input_metadata": input_metadata,
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "elapsed_seconds": time.time() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "raw_tokens_logits_hidden_states_or_per_token_losses_included": False,
    }
    bundle = {
        "schema": "rung534_shared_private_document_ce_sufficient_statistics_v1",
        "collection": collection, "diagnostics": diagnostics,
        "raw_tokens_logits_hidden_states_or_per_token_losses_included": False,
    }
    torch.save(bundle, BUNDLE)
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"], "rung": RUNG, **predictions,
        "interaction_only_correction_strong_null": strong_null,
        "invalid": result["invalid"], "checks": checks,
        "calls": result["price"]["model_forwards"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
