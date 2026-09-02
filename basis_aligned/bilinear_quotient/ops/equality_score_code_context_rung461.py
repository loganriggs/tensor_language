#!/usr/bin/env python3
"""RUNG461 -- explanatory context split of the frozen code score transplant."""

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
import equality_score_code_ood_rung460 as parent
import equality_term_subset_factorial_stage1 as stage1


PREREG = POLY / "EQUALITY_SCORE_CODE_CONTEXT_RUNG461_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_score_code_ood_rung460_results.json"
PARENT_SOURCE = ROOT / "ops/equality_score_code_ood_rung460.py"
ROWS = ROOT / ".rowcache_induction_equality_tensor_final_ood_v2/ood_code.pt"
ROW_RECEIPT = ROOT / "induction_equality_tensor_final_ood_v2_rows_receipt.json"
OUT = ROOT / "equality_score_code_context_rung461_results.json"
BUNDLE = ROOT / "equality_score_code_context_rung461_sufficient_statistics.pt"
PAIR = (0, 3)
PAIR_NAME = "L5H5->L8H4"
COMPONENT = "m9"
ARMS = ("base", "reference", "score")
PRIMARY_CELLS = (
    "near_positive", "far_positive",
    "one_predecessor_positive", "multiple_predecessor_positive",
)
COMPANION_CELLS = ("all_positive", "off_target")
CELLS = PRIMARY_CELLS + COMPANION_CELLS
DOCUMENTS = 192
BATCH = 4
D_MODEL = 1152
EXPECTED_FORWARDS = (DOCUMENTS // BATCH) * (2 + len(ARMS))
HASHES = {
    PREREG: "10758c12f176eed5cfd2174ad9baa510b5112d9a82268bcbaa28bee7445b3d4d",
    PARENT_RESULT: "3910e5a9acb462f2527c8d5f8b7d8039ede410f5a90e1853c7ac279b174cf4a1",
    PARENT_SOURCE: "94afe872f485e625ab483d30c37fc57bd13dda72e00449901355ce9cc5f8be2b",
    ROW_RECEIPT: "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    ROWS: "a82642da15dea4c82d486b46f118a55e480e7613e011ed588caa647eed16b660",
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
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("rung") != 460 \
            or result.get("pred_a_instrument") is not True \
            or result.get("pred_b_code_response") is not False \
            or not all(result.get(key) is True for key in (
                "pred_c_code_causal_effect", "pred_d_code_interchange",
                "pred_e_code_score_geometry",
            )) or result.get("strong_null") is not False:
        raise RuntimeError("rung460 registered near-miss identity changed")
    payload, masks, scales, metadata = parent.validate_inputs()
    if PAIR_NAME not in scales:
        raise RuntimeError("frozen selected natural scale missing")
    metadata = {
        **metadata,
        "rung460_result_sha256": sha256(PARENT_RESULT),
        "rung460_source_sha256": sha256(PARENT_SOURCE),
        "context_cells": list(CELLS),
        "reporting_halves": [[0, 96], [96, 192]],
    }
    return payload, masks, scales[PAIR_NAME], metadata


def _empty_response_stats():
    return {
        key: torch.zeros(len(CELLS), dtype=torch.float64)
        for key in ("ref2", "hyb2", "cross", "tokens")
    }


def _accumulate_response(stats, captures, masks, start):
    reference = captures["reference"] - captures["base"]
    hybrid = captures["score"] - captures["base"]
    for ci, cell in enumerate(CELLS):
        selected = masks[cell][start:start + BATCH]
        if not bool(selected.any()):
            continue
        ref = reference[selected].float()
        hyb = hybrid[selected].float()
        stats["ref2"][ci] += ref.square().sum().double().cpu()
        stats["hyb2"][ci] += hyb.square().sum().double().cpu()
        stats["cross"][ci] += (ref * hyb).sum().double().cpu()
        stats["tokens"][ci] += int(selected.sum())


def _ce_sums(logits, rows, masks, global_start):
    targets = rows[:, 1:].to(logits.device)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).view(len(rows), -1)
    sums = torch.zeros(len(rows), len(CELLS), dtype=torch.float64)
    counts = torch.zeros_like(sums)
    for local in range(len(rows)):
        for ci, cell in enumerate(CELLS):
            selected = masks[cell][global_start + local]
            sums[local, ci] = nll[local, selected].double().sum().cpu()
            counts[local, ci] = int(selected.sum())
    return sums, counts


def response_report(stats):
    answer = {}
    for ci, cell in enumerate(CELLS):
        ref2 = float(stats["ref2"][ci])
        hyb2 = float(stats["hyb2"][ci])
        cross = float(stats["cross"][ci])
        tokens = int(stats["tokens"][ci])
        answer[cell] = {
            "cosine": cross / math.sqrt(max(ref2 * hyb2, 1e-30)),
            "reference_relative_error": math.sqrt(
                max(ref2 + hyb2 - 2 * cross, 0.0) / max(ref2, 1e-30)
            ),
            "reference_raw_coordinate_rms": math.sqrt(
                ref2 / max(tokens * D_MODEL, 1)
            ),
            "hybrid_raw_coordinate_rms": math.sqrt(
                hyb2 / max(tokens * D_MODEL, 1)
            ),
            "tokens": tokens,
        }
    return answer


def causal_report(losses, counts, start=0, stop=DOCUMENTS):
    answer = {}
    for ci, cell in enumerate(CELLS):
        denominator = float(counts[start:stop, ci].sum())
        base = float(losses[ARMS.index("base"), start:stop, ci].sum()) / denominator
        reference = float(losses[ARMS.index("reference"), start:stop, ci].sum()) / denominator
        hybrid = float(losses[ARMS.index("score"), start:stop, ci].sum()) / denominator
        stake = base - reference
        effect = base - hybrid
        answer[cell] = {
            "ce_base_nat": base,
            "ce_reference_nat": reference,
            "ce_hybrid_nat": hybrid,
            "native_stake_nat": stake,
            "hybrid_effect_nat": effect,
            "recovery": effect / stake if stake > 0 else None,
            "tokens": int(denominator),
        }
    return answer


def _ordering(report, field):
    return {
        "far_gt_near": report["far_positive"][field] > report["near_positive"][field],
        "one_gt_multiple": (
            report["one_predecessor_positive"][field]
            > report["multiple_predecessor_positive"][field]
        ),
    }


def analyze(response, halves, losses, counts):
    responses = response_report(response)
    response_halves = [response_report(row) for row in halves]
    causal = causal_report(losses, counts)
    causal_halves = [causal_report(losses, counts, start, start + 96) for start in (0, 96)]
    primary_stakes = torch.tensor(
        [causal[cell]["native_stake_nat"] for cell in PRIMARY_CELLS], dtype=torch.float64,
    )
    primary_effects = torch.tensor(
        [causal[cell]["hybrid_effect_nat"] for cell in PRIMARY_CELLS], dtype=torch.float64,
    )
    spearman = stage1.spearman(primary_stakes, primary_effects)
    half_spearman = []
    for report in causal_halves:
        stakes = torch.tensor([report[cell]["native_stake_nat"] for cell in PRIMARY_CELLS])
        effects = torch.tensor([report[cell]["hybrid_effect_nat"] for cell in PRIMARY_CELLS])
        half_spearman.append(stage1.spearman(stakes, effects))
    pooled_order = {
        "native_stake": _ordering(causal, "native_stake_nat"),
        "hybrid_effect": _ordering(causal, "hybrid_effect_nat"),
        "reference_response_size": _ordering(responses, "reference_raw_coordinate_rms"),
        "hybrid_response_size": _ordering(responses, "hybrid_raw_coordinate_rms"),
    }
    half_order = [
        {
            "native_stake": _ordering(report, "native_stake_nat"),
            "hybrid_effect": _ordering(report, "hybrid_effect_nat"),
        }
        for report in causal_halves
    ]
    pred_b = bool(
        all(pooled_order[key][contrast] for key in ("native_stake", "hybrid_effect")
            for contrast in ("far_gt_near", "one_gt_multiple"))
        and all(row[key][contrast] for row in half_order
                for key in ("native_stake", "hybrid_effect")
                for contrast in ("far_gt_near", "one_gt_multiple"))
    )
    pred_c = bool(
        spearman >= .80 and all(value > 0 for value in half_spearman)
        and all(causal[cell]["native_stake_nat"] > 0
                and causal[cell]["hybrid_effect_nat"] > 0
                and causal[cell]["recovery"] is not None
                and .20 <= causal[cell]["recovery"] <= 1.50
                for cell in PRIMARY_CELLS)
    )
    pred_d = bool(
        all(responses[cell]["cosine"] >= .65
                and responses[cell]["reference_relative_error"] <= .75
                and responses[cell]["reference_raw_coordinate_rms"] >= 1e-4
                and responses[cell]["hybrid_raw_coordinate_rms"] >= 1e-4
                for cell in PRIMARY_CELLS)
        and all(report[cell]["cosine"] > 0 for report in response_halves
                for cell in PRIMARY_CELLS)
    )
    stakes = [causal[cell]["native_stake_nat"] for cell in PRIMARY_CELLS]
    ref_sizes = [responses[cell]["reference_raw_coordinate_rms"] for cell in PRIMARY_CELLS]
    pred_e = bool(
        max(stakes) - min(stakes) >= .01
        and max(ref_sizes) / min(ref_sizes) >= 1.10
        and all(pooled_order[key][contrast]
                for key in ("reference_response_size", "hybrid_response_size")
                for contrast in ("far_gt_near", "one_gt_multiple"))
    )
    natural_sign_exists = any(
        pooled_order[key][contrast]
        for key in ("native_stake", "hybrid_effect")
        for contrast in ("far_gt_near", "one_gt_multiple")
    )
    strong_science_null = bool(
        any(causal[cell]["native_stake_nat"] <= 0
            or causal[cell]["hybrid_effect_nat"] <= 0 for cell in PRIMARY_CELLS)
        or spearman <= 0 or not natural_sign_exists
    )
    return {
        "response_by_context": responses,
        "response_by_context_halves": response_halves,
        "causal_by_context": causal,
        "causal_by_context_halves": causal_halves,
        "pooled_ordering": pooled_order,
        "half_ordering": half_order,
        "native_stake_vs_hybrid_effect_spearman": spearman,
        "half_spearman": half_spearman,
        "pred_b_context_order": pred_b,
        "pred_c_causal_tracking": pred_c,
        "pred_d_shared_direction": pred_d,
        "pred_e_amplitude_explanation": pred_e,
        "strong_science_null": strong_science_null,
    }


@torch.no_grad()
def collect(model, payload, masks, scale):
    rows = payload["rows"]
    response = _empty_response_stats()
    halves = [_empty_response_stats(), _empty_response_stats()]
    losses = torch.zeros(len(ARMS), DOCUMENTS, len(CELLS), dtype=torch.float64)
    counts = torch.zeros(DOCUMENTS, len(CELLS), dtype=torch.float64)
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    reconstruction = 0.0
    device = next(model.parameters()).device
    for start in range(0, DOCUMENTS, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = parent.parent.run_forward(model, tokens, pair=None, arm="native")
        parent.parent._record_audit(audit_totals, "code:native", audit,
                                    analytical=False, captures=0)
        replay_logits, _, audit, error = parent.parent.run_forward(
            model, tokens, pair=None, arm="replay",
        )
        parent.parent._record_audit(audit_totals, "code:replay", audit,
                                    analytical=True, captures=0)
        difference = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
        )
        reconstruction = max(reconstruction, error)
        del native, replay_logits, difference
        captures = {}
        for ai, arm in enumerate(ARMS):
            logits, arm_captures, audit, error = parent.parent.run_forward(
                model, tokens, pair=PAIR, arm=arm, scales=scale,
                capture_keys=(COMPONENT,),
            )
            parent.parent._record_audit(
                audit_totals, f"code:{PAIR_NAME}:{arm}", audit,
                analytical=True, captures=1,
            )
            reconstruction = max(reconstruction, error)
            sums, observed = _ce_sums(logits, batch_rows, masks, start)
            if ai == 0:
                counts[start:start + BATCH] = observed
            elif not torch.equal(observed, counts[start:start + BATCH]):
                raise RuntimeError("context supports changed across arms")
            losses[ai, start:start + BATCH] = sums
            captures[arm] = arm_captures[COMPONENT]
            del logits
        _accumulate_response(response, captures, masks, start)
        _accumulate_response(halves[0 if start < 96 else 1], captures, masks, start)
        del captures
    return response, halves, losses, counts, audit_totals, replay, reconstruction


def main():
    started = time.time()
    payload, masks, scale, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 461, "model_loaded": False,
            "ood_code_rows_loaded_for_identity_only": True,
            "new_model_outcomes_opened": False, "sealed_opened": False,
            "pair": PAIR_NAME, "component": COMPONENT,
            "natural_fit_score_ratio": scale["score_ratio"],
            "cells": CELLS, "expected_forwards": EXPECTED_FORWARDS,
            "input_metadata": metadata,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung461 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    response, halves, losses, counts, audit, replay, reconstruction = collect(
        model, payload, masks, scale,
    )
    analysis = analyze(response, halves, losses, counts)
    forwards = sum(row["forwards"] for row in audit.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12
        and reconstruction <= 1e-10
        and forwards == EXPECTED_FORWARDS
    )
    strong_null = bool(not pred_a or analysis["strong_science_null"])
    bundle = {
        "schema": "equality_score_code_context_rung461_sufficient_statistics_v1",
        "response_stats": response, "response_half_stats": halves,
        "loss_sums": losses, "counts": counts,
        "raw_rows_tokens_logits_or_hidden_states_included": False,
        "sealed_attention0_opened": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 461,
        "claim_level": "already_open_code_context_diagnostic_not_ood_confirmation",
        "input_identity": metadata,
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "ood_code_model_outcomes_opened": True,
        "sealed_attention0_confirmation_opened": False,
        "frozen_pair": PAIR_NAME, "frozen_component": COMPONENT,
        "frozen_natural_fit_scale": scale,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "analysis": analysis, "audit_totals": audit,
        "sufficient_statistics": {
            "path": str(BUNDLE), "sha256": sha256(BUNDLE), "bytes": BUNDLE.stat().st_size,
        },
        "execution_price": {
            "outer_forwards": forwards,
            "tested_pairs": 1, "tested_factors": 1, "tested_readers": 1,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_context_order': analysis["pred_b_context_order"],
        'pred_c_causal_tracking': analysis["pred_c_causal_tracking"],
        'pred_d_shared_direction': analysis["pred_d_shared_direction"],
        'pred_e_amplitude_explanation': analysis["pred_e_amplitude_explanation"],
        "strong_null": strong_null,
        "runtime_s": time.time() - started,
        "next_step": (
            "freeze_genuinely_independent_amplitude_sensitive_confirmation_role"
            if all(analysis[key] for key in (
                "pred_b_context_order", "pred_c_causal_tracking",
                "pred_d_shared_direction", "pred_e_amplitude_explanation",
            )) and not strong_null
            else "retain_aggregate_code_result_and_reconsider_context_gate"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 461,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "analysis": analysis,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "execution_price": result["execution_price"],
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
