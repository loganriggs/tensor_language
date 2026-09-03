#!/usr/bin/env python3
"""RUNG532 -- downstream-defined equivalence of equality-score factors."""

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
import equality_score_directed_action_graph_rung501 as task_parent
import equality_score_factor_branch_sharing_rung531 as factor_screen
import equality_term_score_payload_rung459 as factor_parent
import mlp0_branch_circuit_response_rung481 as circuit_parent


RUNG = 532
PAIR_NAME = "L8H3->L8H4"
SOURCE_INDEX = 2
TARGET_INDEX = 3
SOURCE_HEAD = 3
TARGET_HEAD = 4
ALPHA = 1.227983240318439
BETA = -0.8533769036200292
GAMMA = -1.0785167862928777
DIRECT_ALPHA = -1.268044102615207
DIRECT_BETA = 0.6995515454196305
BACKGROUNDS = ("donor_present", "donor_absent")
ARMS = (
    "native", "absent", "swapped_first", "swapped_second", "swapped_both",
    "product_control", "permuted_first", "permuted_second",
    "direct_first", "direct_second",
)
TAG_SET_NAMES = ("discovery_32", "heldout_30")
MASK_TYPES = ("member", "slice_control")
TASK_CELLS = ("copy_positive", "all_noncopy")
DOCUMENTS = (500, 1000)
DOCUMENT_SPLIT = 750
BATCH = 4
FORWARDS_PER_BATCH = 1 + len(BACKGROUNDS) * len(ARMS)
BATCHES = (DOCUMENTS[1] - DOCUMENTS[0]) // BATCH
FORWARDS = FORWARDS_PER_BATCH * BATCHES

PREREG = POLY / "EQUALITY_FACTOR_COMPANION_CAUSAL_EQUIVALENCE_RUNG532_PREREGISTRATION.md"
R531_SOURCE = ROOT / "ops/equality_score_factor_branch_sharing_rung531.py"
R531_RESULT = ROOT / "equality_score_factor_branch_sharing_rung531_results.json"
CIRCUIT_SOURCE = ROOT / "ops/mlp0_branch_circuit_response_rung481.py"
FACTOR_SOURCE = ROOT / "ops/equality_term_score_payload_rung459.py"
TASK_SOURCE = ROOT / "ops/equality_score_directed_action_graph_rung501.py"
R500_SOURCE = ROOT / "ops/equality_matcher_mlp9_reader_calibration_rung500.py"
R500_RESULT = ROOT / "equality_matcher_mlp9_reader_calibration_rung500_results.json"
FACADE_SOURCE = POLY / "bilin18_observed_model_facade.py"
INDUCTION_SOURCE = POLY / "circuit_induction_tensor.py"
MODEL_SOURCE = Path("/workspace/tensor_language/jacclust/tt_model.py")
OUT = ROOT / "equality_factor_companion_causal_equivalence_rung532_results.json"
BUNDLE = ROOT / "equality_factor_companion_causal_equivalence_rung532_bundle.pt"
HASHES = {
    PREREG: "5417fd39f3ebe5827276e03e85d73b7791e53dcf85bfde9dc5d41fcaf0c8ec7e",
    R531_SOURCE: "e2eb9bd2674247c1fa1c0e25a50d4e747b899a2883899f3074bf809bc676f71e",
    R531_RESULT: "016d4e7babaf2fa562ee254e76ea8c354a7448ddb9fb70cf4be6c835c77354ab",
    CIRCUIT_SOURCE: "ef08017a30ceb0c9e4481198fc1d58c5b0bf8cd37707d2223c42db9eb04f1f44",
    FACTOR_SOURCE: "9f9e66f689452cbcb14d741792b66eb9ff526dff5472a5938c58a2a4c82620d8",
    TASK_SOURCE: "97f3946f558f3d61fc952a9b6ddc7c334b51ccc0ccfe5f02c6ecced417f1e077",
    R500_SOURCE: "83b520873cabfd167e4da645c0564e267b15c3be98e4a4f8d739133d01f81b0f",
    R500_RESULT: "9e4daa40c2ab88980d29d141eef6317bfae3035e823ac6bd6c8fc57fabcbc7d9",
    FACADE_SOURCE: "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    INDUCTION_SOURCE: "b2d43be8e260bbe4bfece494999d237d93258f676b19e2993eca09655e253e3a",
    MODEL_SOURCE: "49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2",
}

PREDICTION_TEXT = {
    "pred_a_exact_live_interaction_instrument": "all identities, edits, supports, and calls pass",
    "pred_b_product_control_transfers": "whole product transfers across new rows and 62 circuits",
    "pred_c_source_second_replaces_target_first": "source second factor works with target companion",
    "pred_d_source_first_replaces_target_second": "source first factor works with target companion",
    "pred_e_heldout_interaction_defined_factor": "one factor identity survives all held-out contexts",
    "pred_f_factor_replacements_compose": "two factor replacements have bounded interaction",
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
    r531 = json.loads(R531_RESULT.read_text())
    report = r531["reports"][PAIR_NAME]
    if (
        r531.get("pred_" + "a_exact_authorized_instrument") is not True
        or r531.get("strong_null") is not True
        or report["selected_assignment"] != "swapped"
        or not report["assignment_stable"]
        or report["scale_product_relative_difference"] > 0.10
        or [report["target_first_scale"], report["target_second_scale"],
            report["independent_product_scale"]] != [ALPHA, BETA, GAMMA]
        or report["fit"]["direct"]["target_first_scale"] != DIRECT_ALPHA
        or report["fit"]["direct"]["target_second_scale"] != DIRECT_BETA
    ):
        raise RuntimeError("rung531 pair selection or frozen scales changed")
    r500 = json.loads(R500_RESULT.read_text())
    if not all(r500.get(key) is True for key in (
        "pred_" + "a_exact_live_reader_instrument",
        "pred_" + "b_mlp9_reads_known_score_relation",
        "pred_" + "c_mlp9_rejects_typed_controls",
        "pred_" + "d_reader_stable_under_early_removal",
        "pred_" + "e_reader_copy_task_selective",
        "pred_" + "f_named_reader_calibrated",
    )) or r500.get("strong_null"):
        raise RuntimeError("rung500 downstream reader authority changed")
    rows, circuit_masks, discovery_tags, heldout_tags, _fit_rows, metadata = (
        circuit_parent.validate_inputs())
    if tuple(rows.shape) != (1000, 257) or rows.dtype != torch.long:
        raise RuntimeError("circuit census rows changed")
    if len(discovery_tags) != 32 or len(heldout_tags) != 30:
        raise RuntimeError("62-circuit tag partition changed")
    if set(circuit_masks) != set(discovery_tags) | set(heldout_tags):
        raise RuntimeError("circuit masks do not cover the frozen tag partition")
    return rows, circuit_masks, (tuple(discovery_tags), tuple(heldout_tags)), metadata


def replacement_pattern(arm: str, source_first, source_second, target_first, target_second):
    if arm == "native":
        return target_first * target_second
    if arm == "absent":
        return torch.zeros_like(target_first)
    if arm == "swapped_first":
        return (ALPHA * source_second) * target_second
    if arm == "swapped_second":
        return target_first * (BETA * source_first)
    if arm == "swapped_both":
        return (ALPHA * source_second) * (BETA * source_first)
    if arm == "product_control":
        return GAMMA * (source_first * source_second)
    if arm == "permuted_first":
        return (ALPHA * factor_screen._key_prefix_reverse(source_second)) * target_second
    if arm == "permuted_second":
        return target_first * (BETA * factor_screen._key_prefix_reverse(source_first))
    if arm == "direct_first":
        return (DIRECT_ALPHA * source_first) * target_second
    if arm == "direct_second":
        return target_first * (DIRECT_BETA * source_second)
    raise ValueError(f"unknown arm: {arm}")


def native_branch_product(first: torch.Tensor, second: torch.Tensor):
    """Match deployed attention's multiply-before-FP32-cast order exactly."""
    if first.dtype != second.dtype:
        raise ValueError("native score factors must have the same dtype")
    return (first * second).float()


@torch.no_grad()
def run_forward(model, tokens, *, background: str, arm: str, direct: bool = False):
    if background not in BACKGROUNDS or arm not in ARMS:
        raise ValueError("unregistered background or arm")
    diagnostics = {
        "factor_reconstruction_max": 0.0, "branch_product_max_abs": 0.0,
        "donor_edit_rms": 0.0, "target_edit_rms": 0.0,
    }
    audit = {"native_attention": 0, "replayed_attention": 0, "native_mlp": 0}

    def attention(event):
        if direct or event.site not in factor_parent.stage1.SITE_HEADS:
            write, next_value = event.block.attn(event.state, event.first_value)
            audit["native_attention"] += 1
            return write, next_value
        write, factors, support, reconstruction = factor_parent._factor_site(
            event.state, event.first_value, event.block.attn, event.site, event.tokens)
        audit["replayed_attention"] += 1
        diagnostics["factor_reconstruction_max"] = max(
            diagnostics["factor_reconstruction_max"], reconstruction)
        if event.site == 8:
            first, second = factor_screen._score_branches(event.state, event.block.attn)
            source = factors[SOURCE_INDEX]
            target = factors[TARGET_INDEX]
            source_first_native, source_second_native = first[:, SOURCE_HEAD], second[:, SOURCE_HEAD]
            target_first_native, target_second_native = first[:, TARGET_HEAD], second[:, TARGET_HEAD]
            causal = torch.tril(torch.ones(
                target_first_native.shape[-2:], dtype=torch.bool,
                device=target_first_native.device))
            native_product = native_branch_product(
                target_first_native, target_second_native).masked_fill(~causal, 0.0)
            diagnostics["branch_product_max_abs"] = max(
                diagnostics["branch_product_max_abs"],
                float((native_product - target["p"]).abs().max()))
            source_first, source_second = source_first_native.float(), source_second_native.float()
            target_first, target_second = target_first_native.float(), target_second_native.float()
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


def empty_collection(tags_by_set):
    return {
        "circuit_sums": {
            name: torch.zeros(
                len(BACKGROUNDS), len(ARMS), 2, len(MASK_TYPES), len(tags),
                dtype=torch.float64)
            for name, tags in zip(TAG_SET_NAMES, tags_by_set)
        },
        "circuit_counts": {
            name: torch.zeros(2, len(MASK_TYPES), len(tags), dtype=torch.float64)
            for name, tags in zip(TAG_SET_NAMES, tags_by_set)
        },
        "task_sums": torch.zeros(
            len(BACKGROUNDS), len(ARMS), 2, len(TASK_CELLS), dtype=torch.float64),
        "task_counts": torch.zeros(2, len(TASK_CELLS), dtype=torch.float64),
    }


def _circuit_batch_masks(circuit_masks, tags_by_set, start, stop, device):
    output = {}
    for name, tags in zip(TAG_SET_NAMES, tags_by_set):
        output[name] = torch.stack([
            torch.stack([
                circuit_masks[tag][mask_type].view(1000, 256)[start:stop]
                for tag in tags
            ])
            for mask_type in MASK_TYPES
        ]).to(device)
    return output


def _accumulate_counts(collection, circuit_masks, task_masks, tags_by_set, start, stop, device):
    batch_masks = _circuit_batch_masks(circuit_masks, tags_by_set, start, stop, device)
    row_ids = torch.arange(start, stop, device=device)
    half_rows = (row_ids < DOCUMENT_SPLIT, row_ids >= DOCUMENT_SPLIT)
    for half, rows in enumerate(half_rows):
        selector = rows[None, :, None]
        for name in TAG_SET_NAMES:
            selected = batch_masks[name] & selector
            collection["circuit_counts"][name][half] += selected.sum((1, 2)).cpu()
        for cell_index, cell in enumerate(TASK_CELLS):
            selected = task_masks[cell][start:stop].to(device) & rows[:, None]
            collection["task_counts"][half, cell_index] += int(selected.sum())
    return batch_masks, half_rows


def _accumulate_nll(
    collection, nll, background_index, arm_index, batch_masks, half_rows,
    task_masks, start, stop,
):
    for half, rows in enumerate(half_rows):
        selector = rows[None, :, None]
        for name in TAG_SET_NAMES:
            selected = batch_masks[name] & selector
            collection["circuit_sums"][name][background_index, arm_index, half] += (
                (selected * nll[None, None]).sum((2, 3)).double().cpu())
        for cell_index, cell in enumerate(TASK_CELLS):
            selected = task_masks[cell][start:stop].to(nll.device) & rows[:, None]
            collection["task_sums"][background_index, arm_index, half, cell_index] += (
                nll[selected].double().sum().cpu())


@torch.no_grad()
def collect(model, rows, circuit_masks, tags_by_set, *, smoke=False):
    collection = empty_collection(tags_by_set)
    task_masks = task_parent._task_masks(rows)
    diagnostics = {
        "direct_native_calls": 0, "analytical_calls": 0,
        "native_replay_logit_max_abs": 0.0,
        "factor_reconstruction_max": 0.0, "branch_product_max_abs": 0.0,
        "minimum_donor_edit_rms": math.inf, "minimum_target_edit_rms": math.inf,
        "zero_intended_edits": 0,
    }
    stop_document = DOCUMENTS[0] + BATCH if smoke else DOCUMENTS[1]
    device = next(model.parameters()).device
    for start in range(DOCUMENTS[0], stop_document, BATCH):
        stop = start + BATCH
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        targets = batch_rows[:, 1:].to(device)
        direct_logits, _direct_diagnostics, _direct_audit = run_forward(
            model, tokens, background="donor_present", arm="native", direct=True)
        diagnostics["direct_native_calls"] += 1
        if not smoke:
            batch_masks, half_rows = _accumulate_counts(
                collection, circuit_masks, task_masks, tags_by_set, start, stop, device)
        present_native_logits = None
        for background_index, background in enumerate(BACKGROUNDS):
            for arm_index, arm in enumerate(ARMS):
                logits, row_diagnostics, _audit = run_forward(
                    model, tokens, background=background, arm=arm)
                diagnostics["analytical_calls"] += 1
                diagnostics["factor_reconstruction_max"] = max(
                    diagnostics["factor_reconstruction_max"],
                    row_diagnostics["factor_reconstruction_max"])
                diagnostics["branch_product_max_abs"] = max(
                    diagnostics["branch_product_max_abs"],
                    row_diagnostics["branch_product_max_abs"])
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
                        collection, nll, background_index, arm_index, batch_masks,
                        half_rows, task_masks, start, stop)
                if logits is not present_native_logits:
                    del logits
        if present_native_logits is None:
            raise RuntimeError("present/native replay was not exercised")
        del direct_logits, present_native_logits, tokens, targets
    expected_batches = 1 if smoke else BATCHES
    diagnostics["calls_exact"] = bool(
        diagnostics["direct_native_calls"] == expected_batches
        and diagnostics["analytical_calls"] == expected_batches * len(BACKGROUNDS) * len(ARMS))
    if not smoke:
        diagnostics["all_circuit_supports_live"] = all(
            bool((counts > 0).all()) for counts in collection["circuit_counts"].values())
        diagnostics["all_task_supports_live"] = bool((collection["task_counts"] > 0).all())
    return collection, diagnostics


def _vector_metrics(reference: torch.Tensor, candidate: torch.Tensor):
    reference = torch.as_tensor(reference, dtype=torch.float64)
    candidate = torch.as_tensor(candidate, dtype=torch.float64)
    reference2 = float(reference.square().sum())
    candidate2 = float(candidate.square().sum())
    if reference2 <= 0:
        raise RuntimeError("native circuit effect vector is not live")
    if candidate2 <= 0:
        return {
            "cosine": 0.0, "relative_error": 1.0,
            "reference_rms": math.sqrt(reference2 / reference.numel()),
            "candidate_rms": 0.0,
        }
    cross = float((reference * candidate).sum())
    return {
        "cosine": cross / math.sqrt(reference2 * candidate2),
        "relative_error": math.sqrt(float((reference - candidate).square().sum()) / reference2),
        "reference_rms": math.sqrt(reference2 / reference.numel()),
        "candidate_rms": math.sqrt(candidate2 / candidate.numel()),
    }


def _norm_ratio(value: torch.Tensor, reference: torch.Tensor):
    return float(torch.linalg.vector_norm(value) /
                 torch.linalg.vector_norm(reference).clamp_min(1e-30))


def analyze(collection):
    circuit_ce = {
        name: collection["circuit_sums"][name]
        / collection["circuit_counts"][name][None, None].clamp_min(1)
        for name in TAG_SET_NAMES
    }
    task_ce = collection["task_sums"] / collection["task_counts"][None, None].clamp_min(1)
    reports = {}
    contexts = []
    for tag_set_name in TAG_SET_NAMES:
        values = circuit_ce[tag_set_name]
        for background_index, background in enumerate(BACKGROUNDS):
            for half in range(2):
                native_member = values[
                    background_index, ARMS.index("native"), half, MASK_TYPES.index("member")]
                absent_member = values[
                    background_index, ARMS.index("absent"), half, MASK_TYPES.index("member")]
                native_effect = absent_member - native_member
                native_slice = values[
                    background_index, ARMS.index("native"), half,
                    MASK_TYPES.index("slice_control")]
                absent_task = task_ce[
                    background_index, ARMS.index("absent"), half,
                    TASK_CELLS.index("copy_positive")]
                native_task = task_ce[
                    background_index, ARMS.index("native"), half,
                    TASK_CELLS.index("copy_positive")]
                task_reference = float(absent_task - native_task)
                arm_reports, effects = {}, {}
                for arm_index, arm in enumerate(ARMS):
                    member = values[background_index, arm_index, half, MASK_TYPES.index("member")]
                    effect = absent_member - member
                    effects[arm] = effect
                    slice_values = values[
                        background_index, arm_index, half, MASK_TYPES.index("slice_control")]
                    arm_task = task_ce[
                        background_index, arm_index, half, TASK_CELLS.index("copy_positive")]
                    off_change = float(
                        task_ce[background_index, arm_index, half, TASK_CELLS.index("all_noncopy")]
                        - task_ce[background_index, ARMS.index("native"), half,
                                  TASK_CELLS.index("all_noncopy")])
                    arm_reports[arm] = {
                        "member_effect": _vector_metrics(native_effect, effect),
                        "copy_task_recovery": (
                            float(absent_task - arm_task) / task_reference
                            if abs(task_reference) > 1e-30 else None),
                        "slice_control_mean_abs_ce_change_from_native": float(
                            (slice_values - native_slice).abs().mean()),
                        "all_noncopy_signed_mean_ce_change_from_native": off_change,
                    }
                interaction = (
                    effects["swapped_both"] - effects["swapped_first"]
                    - effects["swapped_second"] + effects["native"])
                composition = {
                    "factorial_interaction_over_native_effect": _norm_ratio(
                        interaction, native_effect),
                    "swapped_both_minus_product_control_over_product_control_effect": _norm_ratio(
                        effects["swapped_both"] - effects["product_control"],
                        effects["product_control"]),
                }
                key = f"{tag_set_name}/{background}/half{half}"
                reports[key] = {"arms": arm_reports, "composition": composition}
                contexts.append(reports[key])
    return reports, contexts


def _base_holds(row, arm):
    report = row["arms"][arm]
    metric = report["member_effect"]
    recovery = report["copy_task_recovery"]
    return bool(
        metric["cosine"] >= 0.85 and metric["relative_error"] <= 0.60
        and recovery is not None and 0.65 <= recovery <= 1.40)


def score(reports, contexts, diagnostics, collection, checkpoint_hash):
    pred_a = bool(
        diagnostics["native_replay_logit_max_abs"] == 0.0
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["branch_product_max_abs"] == 0.0
        and diagnostics["minimum_donor_edit_rms"] > 0
        and diagnostics["minimum_target_edit_rms"] > 0
        and diagnostics["zero_intended_edits"] == 0
        and diagnostics["calls_exact"]
        and diagnostics["all_circuit_supports_live"]
        and diagnostics["all_task_supports_live"]
        and checkpoint_hash == facade.WEIGHTS_SHA256)
    pred_b = bool(pred_a and all(_base_holds(row, "product_control") for row in contexts))
    first_checks, second_checks = [], []
    for row in contexts:
        first = row["arms"]["swapped_first"]
        second = row["arms"]["swapped_second"]
        first_checks.append(bool(
            _base_holds(row, "swapped_first")
            and first["member_effect"]["cosine"]
                >= row["arms"]["permuted_first"]["member_effect"]["cosine"] + 0.15
            and first["member_effect"]["cosine"]
                >= row["arms"]["direct_first"]["member_effect"]["cosine"] + 0.15
            and first["slice_control_mean_abs_ce_change_from_native"] <= 0.01))
        second_checks.append(bool(
            _base_holds(row, "swapped_second")
            and second["member_effect"]["cosine"]
                >= row["arms"]["permuted_second"]["member_effect"]["cosine"] + 0.15
            and second["member_effect"]["cosine"]
                >= row["arms"]["direct_second"]["member_effect"]["cosine"] + 0.15
            and second["slice_control_mean_abs_ce_change_from_native"] <= 0.01))
    pred_c = bool(pred_a and all(first_checks))
    pred_d = bool(pred_a and all(second_checks))
    pred_e = bool(pred_a and pred_b and (pred_c or pred_d))
    pred_f = bool(pred_a and all(
        row["composition"]["factorial_interaction_over_native_effect"] <= 0.30
        and row["composition"][
            "swapped_both_minus_product_control_over_product_control_effect"] <= 0.30
        for row in contexts))
    predictions = dict(zip(PREDICTION_TEXT, (pred_a, pred_b, pred_c, pred_d, pred_e, pred_f)))
    checks = {
        "product_control_contexts_passing": sum(
            _base_holds(row, "product_control") for row in contexts),
        "swapped_first_contexts_passing": sum(first_checks),
        "swapped_second_contexts_passing": sum(second_checks),
        "total_contexts": len(contexts),
        "composition_contexts_passing": sum(
            row["composition"]["factorial_interaction_over_native_effect"] <= 0.30
            and row["composition"][
                "swapped_both_minus_product_control_over_product_control_effect"] <= 0.30
            for row in contexts),
    }
    return predictions, checks


def main():
    started = time.time()
    smoke = os.environ.get("RUNG532_SMOKE") == "1"
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert FORWARDS_PER_BATCH == 21 and FORWARDS == 2625
        assert len(ARMS) == 10 and len(BACKGROUNDS) == 2
        assert DOCUMENTS == (500, 1000) and DOCUMENT_SPLIT == 750
        print(json.dumps({
            "status": "dry_run_passed", "rung": RUNG, "model_loaded": False,
            "scientific_outcomes_opened": False, "forwards": FORWARDS,
            "arms": ARMS, "backgrounds": BACKGROUNDS,
            "predictions": list(PREDICTION_TEXT),
        }, indent=2, sort_keys=True))
        return
    if not smoke and (OUT.exists() or BUNDLE.exists()):
        raise RuntimeError("rung532 output namespace already exists")
    rows, circuit_masks, tags_by_set, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    collection, diagnostics = collect(
        model, rows, circuit_masks, tags_by_set, smoke=smoke)
    if smoke:
        instrument_pass = bool(
            diagnostics["calls_exact"]
            and diagnostics["native_replay_logit_max_abs"] == 0.0
            and diagnostics["factor_reconstruction_max"] <= 1e-10
            and diagnostics["branch_product_max_abs"] == 0.0
            and diagnostics["minimum_donor_edit_rms"] > 0
            and diagnostics["minimum_target_edit_rms"] > 0
            and diagnostics["zero_intended_edits"] == 0
            and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
        print(json.dumps({
            "status": "smoke_passed" if instrument_pass else "smoke_instrument_invalid",
            "instrument_pass": instrument_pass, "rung": RUNG,
            "scientific_outcomes_opened": False,
            "calls": diagnostics["direct_native_calls"] + diagnostics["analytical_calls"],
            "native_replay_logit_max_abs": diagnostics["native_replay_logit_max_abs"],
            "factor_reconstruction_max": diagnostics["factor_reconstruction_max"],
            "branch_product_max_abs": diagnostics["branch_product_max_abs"],
            "minimum_donor_edit_rms": diagnostics["minimum_donor_edit_rms"],
            "minimum_target_edit_rms": diagnostics["minimum_target_edit_rms"],
            "zero_intended_edits": diagnostics["zero_intended_edits"],
            "checkpoint_weights_sha256": checkpoint.weights_sha256,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }, indent=2, sort_keys=True))
        if not instrument_pass:
            raise RuntimeError("rung532 smoke instrument did not pass")
        return
    reports, contexts = analyze(collection)
    predictions, checks = score(
        reports, contexts, diagnostics, collection, checkpoint.weights_sha256)
    result = {
        "status": "completed", "rung": RUNG,
        **predictions,
        "strong_null": bool(
            predictions["pred_a_exact_live_interaction_instrument"]
            and predictions["pred_b_product_control_transfers"]
            and not predictions["pred_c_source_second_replaces_target_first"]
            and not predictions["pred_d_source_first_replaces_target_second"]),
        "reports": reports, "checks": checks, "diagnostics": diagnostics,
        "pair": PAIR_NAME, "frozen_scales": {
            "alpha": ALPHA, "beta": BETA, "gamma": GAMMA,
            "direct_alpha": DIRECT_ALPHA, "direct_beta": DIRECT_BETA,
        },
        "documents": DOCUMENTS, "document_split": DOCUMENT_SPLIT,
        "tag_sets": {name: list(tags) for name, tags in zip(TAG_SET_NAMES, tags_by_set)},
        "mask_types": MASK_TYPES, "task_cells": TASK_CELLS,
        "price": {
            "model_forwards": diagnostics["direct_native_calls"] + diagnostics["analytical_calls"],
            "backward_passes": 0, "fitted_vector_parameters": 0,
            "ood_forwards": 0, "arms_per_background": len(ARMS),
        },
        "checkpoint": checkpoint, "input_metadata": metadata,
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "elapsed_seconds": time.time() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "raw_tokens_logits_states_or_per_token_losses_included": False,
    }
    bundle = {
        "schema": "rung532_62_circuit_ce_sufficient_statistics_v1",
        "collection": collection, "diagnostics": diagnostics,
        "raw_tokens_logits_states_or_per_token_losses_included": False,
        "ood_opened": False,
    }
    torch.save(bundle, BUNDLE)
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"], "rung": RUNG,
        **predictions, "strong_null": result["strong_null"],
        "checks": checks, "calls": result["price"]["model_forwards"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
