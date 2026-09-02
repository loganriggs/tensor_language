#!/usr/bin/env python3
"""RUNG460 -- frozen code-OOD confirmation of the shared equality score."""

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

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_term_score_payload_rung459 as parent
import equality_term_subset_factorial_stage1 as stage1
import interchange


PREREG = POLY / "EQUALITY_SCORE_CODE_OOD_RUNG460_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_term_score_payload_rung459_results.json"
PARENT_SOURCE = ROOT / "ops/equality_term_score_payload_rung459.py"
ROWS = ROOT / ".rowcache_induction_equality_tensor_final_ood_v2/ood_code.pt"
ROW_RECEIPT = ROOT / "induction_equality_tensor_final_ood_v2_rows_receipt.json"
OUT = ROOT / "equality_score_code_ood_rung460_results.json"
BUNDLE = ROOT / "equality_score_code_ood_rung460_sufficient_statistics.pt"
SELECTED_PAIR = (0, 3)
CONTROL_PAIR = (1, 3)
PAIRS = (SELECTED_PAIR, CONTROL_PAIR)
PAIR_NAMES = ("L5H5->L8H4", "L7H3->L8H4")
FACTOR = "score"
COMPONENT = "m9"
ARMS = ("base", "reference", "score")
CELLS = ("all_positive", "matched_negative", "off_target")
DOCUMENTS = 192
BATCH = 4
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = "equality-score-code-ood-rung460:bootstrap:0"
INTERCHANGE_SEED = 460
EXPECTED_SUPPORT = {
    "matched_positive": (1132, 187), "matched_negative": (1132, 190),
    "all_positive": (10848, 192), "near_positive": (6492, 190),
    "far_positive": (4356, 192), "one_predecessor_positive": (2499, 192),
    "multiple_predecessor_positive": (8349, 192), "off_target": (24884, 192),
    "all": (36864, 192),
}
HASHES = {
    PREREG: "845947e2922c566c8eba3536733a2b1cf0f2b875408acca47e1e7481a6f8873b",
    PARENT_RESULT: "f157681ced170cbf8664db5710414a38d4f928f8d15dc0dd2b4d8cea9288aefa",
    PARENT_SOURCE: "9f9e66f689452cbcb14d741792b66eb9ff526dff5472a5938c58a2a4c82620d8",
    ROW_RECEIPT: "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    ROWS: "a82642da15dea4c82d486b46f118a55e480e7613e011ed588caa647eed16b660",
    ROOT / "ops/interchange.py":
        "df4a8585dd6a557a71be991f12d0547023ae771bfccc591008cc0ab08f08fd29",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
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
    receipt = json.loads(ROW_RECEIPT.read_text())
    entry = receipt["entries"]["ood_code"]
    if receipt.get("status") != "frozen_before_any_v2_model_forward" \
            or entry.get("file_sha256") != HASHES[ROWS]:
        raise RuntimeError("code row authority changed")
    payload = torch.load(ROWS, map_location="cpu", weights_only=True)
    if payload.get("schema") != "induction_equality_tensor_final_ood_v2_role" \
            or payload.get("role") != "ood_code" \
            or list(payload["rows"].shape) != [DOCUMENTS, 257] \
            or len(payload["records"]) != DOCUMENTS:
        raise RuntimeError("code row payload changed")
    masks = stage1.build_masks(payload["rows"], payload["copy_cells"])
    support = {}
    for name, mask in masks.items():
        observed = (int(mask.sum()), int(mask.any(1).sum()))
        if observed != EXPECTED_SUPPORT[name]:
            raise RuntimeError(f"code task support changed: {name} {observed}")
        support[name] = {
            "tokens": observed[0], "documents": observed[1],
            "mask_sha256": stage1.tensor_sha256(mask),
        }
    parent_result = json.loads(PARENT_RESULT.read_text())
    selected = parent_result["fit_screen"]["selected"]
    if selected is None or selected.get("pair") != PAIR_NAMES[0] \
            or selected.get("factor") != FACTOR or selected.get("component") != COMPONENT \
            or not all(parent_result.get(key) is True for key in (
                "pred_a_instrument", "pred_b_factor_candidate", "pred_c_response_transfer",
                "pred_d_causal_effect", "pred_e_between_control",
            )) or parent_result.get("strong_null") is not False:
        raise RuntimeError("rung459 selected hypothesis changed")
    scales = {name: parent_result["frozen_fit_scales"][name] for name in PAIR_NAMES}
    metadata = {
        "role": "ood_code", "row_file_sha256": sha256(ROWS),
        "row_receipt_sha256": sha256(ROW_RECEIPT), "support": support,
        "parent_result_sha256": sha256(PARENT_RESULT),
        "parent_source_sha256": sha256(PARENT_SOURCE),
        "frozen_selected_pair": PAIR_NAMES[0], "frozen_control_pair": PAIR_NAMES[1],
        "frozen_factor": FACTOR, "frozen_component": COMPONENT,
    }
    return payload, masks, scales, metadata


def _empty_response_stats():
    shape = (2, len(CELLS))
    return {key: torch.zeros(shape, dtype=torch.float64) for key in (
        "ref2", "hyb2", "cross", "write2", "tokens",
    )}


def _accumulate_response(stats, slot, captures, masks, start):
    for ci, cell in enumerate(CELLS):
        selected = masks[cell][start:start + BATCH]
        if not bool(selected.any()):
            continue
        reference = (captures["reference"] - captures["base"])[selected].float()
        hybrid = (captures["score"] - captures["base"])[selected].float()
        writer = captures["reference"][selected].float()
        stats["ref2"][slot, ci] += reference.square().sum().double().cpu()
        stats["hyb2"][slot, ci] += hybrid.square().sum().double().cpu()
        stats["cross"][slot, ci] += (reference * hybrid).sum().double().cpu()
        stats["write2"][slot, ci] += writer.square().sum().double().cpu()
        stats["tokens"][slot, ci] += int(selected.sum())


def _response_row(stats, slot, ci):
    a = float(stats["ref2"][slot, ci])
    b = float(stats["hyb2"][slot, ci])
    cross = float(stats["cross"][slot, ci])
    write = float(stats["write2"][slot, ci])
    return {
        "cosine": cross / math.sqrt(max(a * b, 1e-30)),
        "reference_relative_error": math.sqrt(
            max(a + b - 2 * cross, 0.0) / max(a, 1e-30)
        ),
        "reference_rms_over_reader_write_rms": math.sqrt(a / max(write, 1e-30)),
        "hybrid_rms_over_reader_write_rms": math.sqrt(b / max(write, 1e-30)),
        "tokens": int(stats["tokens"][slot, ci]),
    }


def response_report(stats, slot):
    rows = {cell: _response_row(stats, slot, ci) for ci, cell in enumerate(CELLS)}
    positive = rows["all_positive"]
    margin = positive["cosine"] - max(
        rows["matched_negative"]["cosine"], rows["off_target"]["cosine"],
    )
    live = min(
        positive["reference_rms_over_reader_write_rms"],
        positive["hybrid_rms_over_reader_write_rms"],
    )
    return {**rows, "task_margin": margin, "minimum_relative_response_rms": live}


def _ce_rows(losses, counts, slot, cell="all_positive"):
    ci = CELLS.index(cell)
    base = losses[slot, ARMS.index("base"), :, ci]
    reference = losses[slot, ARMS.index("reference"), :, ci]
    hybrid = losses[slot, ARMS.index("score"), :, ci]
    return base, reference, hybrid, counts[:, ci]


def ce_recovery(losses, counts, slot, start=0, stop=DOCUMENTS):
    base, reference, hybrid, denominator = _ce_rows(losses, counts, slot)
    stake = (base[start:stop].sum() - reference[start:stop].sum()) / denominator[start:stop].sum()
    effect = (base[start:stop].sum() - hybrid[start:stop].sum()) / denominator[start:stop].sum()
    return {
        "reference_stake_nat": float(stake), "hybrid_effect_nat": float(effect),
        "recovery": float(effect / stake) if float(stake) > 0 else None,
    }


def bootstrap_recovery(losses, counts):
    base, reference, hybrid, denominator_counts = _ce_rows(losses, counts, 0)
    point = ce_recovery(losses, counts, 0)
    generator = torch.Generator().manual_seed(
        int.from_bytes(hashlib.sha256(BOOTSTRAP_SEED.encode()).digest()[:8], "little")
    )
    chunks = []
    all_positive = True
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        n = min(500, BOOTSTRAP_DRAWS - start)
        draws = torch.randint(DOCUMENTS, (n, DOCUMENTS), generator=generator)
        weights = torch.zeros(n, DOCUMENTS, dtype=torch.float64)
        weights.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
        denominator = weights @ denominator_counts
        stakes = (weights @ (base - reference)) / denominator
        effects = (weights @ (base - hybrid)) / denominator
        all_positive &= bool((stakes > 0).all())
        chunks.append(torch.where(stakes > 0, effects / stakes, torch.zeros_like(stakes)))
    draws = torch.cat(chunks).sort().values
    return {
        **point, "simultaneous_95_lower": float(draws[math.floor(.025 * BOOTSTRAP_DRAWS)]),
        "every_bootstrap_reference_stake_positive": all_positive,
        "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
    }


@torch.no_grad()
def collect(model, payload, masks, scales):
    rows = payload["rows"]
    response = _empty_response_stats()
    halves = [_empty_response_stats(), _empty_response_stats()]
    losses = torch.zeros(2, len(ARMS), DOCUMENTS, len(CELLS), dtype=torch.float64)
    counts = torch.zeros(DOCUMENTS, len(CELLS), dtype=torch.float64)
    scale_stats = parent._empty_scale_stats()
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    reconstruction = 0.0
    device = next(model.parameters()).device
    for start in range(0, DOCUMENTS, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = parent.run_forward(model, tokens, pair=None, arm="native")
        parent._record_audit(audit_totals, "code:native", audit,
                             analytical=False, captures=0)
        replay_logits, _, audit, error = parent.run_forward(
            model, tokens, pair=None, arm="replay",
        )
        parent._record_audit(audit_totals, "code:replay", audit,
                             analytical=True, captures=0)
        difference = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
        )
        reconstruction = max(reconstruction, error)
        del native, replay_logits, difference
        for slot, pair in enumerate(PAIRS):
            positive = masks["all_positive"][start:start + BATCH]
            def scale_callback(
                early, late, support, *, name=PAIR_NAMES[slot], positive=positive,
            ):
                parent._accumulate_scales(scale_stats, name, early, late, support, positive)
            scale_logits, _, audit, error = parent.run_forward(
                model, tokens, pair=pair, arm="scale", scale_callback=scale_callback,
            )
            parent._record_audit(audit_totals, f"code:scale:{PAIR_NAMES[slot]}", audit,
                                 analytical=True, captures=0)
            reconstruction = max(reconstruction, error)
            del scale_logits
            captures = {}
            for ai, arm in enumerate(ARMS):
                logits, arm_captures, audit, error = parent.run_forward(
                    model, tokens, pair=pair, arm=arm, scales=scales[PAIR_NAMES[slot]],
                    capture_keys=(COMPONENT,),
                )
                parent._record_audit(
                    audit_totals, f"code:{PAIR_NAMES[slot]}:{arm}", audit,
                    analytical=True, captures=1,
                )
                reconstruction = max(reconstruction, error)
                sums, observed_counts = parent._ce_sums(logits, batch_rows, masks, start)
                if slot == 0 and ai == 0:
                    counts[start:start + BATCH] = observed_counts
                elif not torch.equal(observed_counts, counts[start:start + BATCH]):
                    raise RuntimeError("code CE supports changed across arms")
                losses[slot, ai, start:start + BATCH] = sums
                captures[arm] = arm_captures[COMPONENT]
                del logits
            _accumulate_response(response, slot, captures, masks, start)
            _accumulate_response(halves[0 if start < 96 else 1], slot, captures, masks, start)
            del captures
    code_scales = parent._finish_scales({name: scale_stats[name] for name in PAIR_NAMES})
    return response, halves, losses, counts, code_scales, audit_totals, replay, reconstruction


def analyze(response, halves, losses, counts, code_scales):
    selected_response = response_report(response, 0)
    control_response = response_report(response, 1)
    half_responses = [response_report(stats, 0) for stats in halves]
    recovery = bootstrap_recovery(losses, counts)
    half_recoveries = [ce_recovery(losses, counts, 0, start, start + 96)
                       for start in (0, 96)]
    ci_off = CELLS.index("off_target")
    off_change = float((
        losses[0, ARMS.index("score"), :, ci_off]
        - losses[0, ARMS.index("reference"), :, ci_off]
    ).sum() / counts[:, ci_off].sum())
    ci = CELLS.index("all_positive")
    supported = counts[:, ci] > 0
    discrepancies = []
    for slot in range(2):
        delta = (
            losses[slot, ARMS.index("score"), :, ci]
            - losses[slot, ARMS.index("reference"), :, ci]
        )[supported] / counts[:, ci][supported]
        discrepancies.append(delta.abs().tolist())
    interchange_result = interchange.commutation(
        discrepancies[0], discrepancies[1], seed=INTERCHANGE_SEED, permutations=10_000,
    )
    interchange_result.update({
        "selected_pair": PAIR_NAMES[0], "between_control_pair": PAIR_NAMES[1],
        "seed": INTERCHANGE_SEED, "permutations": 10_000,
    })
    selected_direct = code_scales[PAIR_NAMES[0]]["direct_score_cosine"]
    control_direct = code_scales[PAIR_NAMES[1]]["direct_score_cosine"]
    pred_b = bool(
        selected_response["all_positive"]["cosine"] >= .65
        and selected_response["task_margin"] >= .05
        and selected_response["all_positive"]["reference_relative_error"] <= .70
        and selected_response["minimum_relative_response_rms"] >= 1e-4
        and all(row["all_positive"]["cosine"] > 0 and row["task_margin"] > 0
                for row in half_responses)
    )
    pred_c = bool(
        recovery["every_bootstrap_reference_stake_positive"]
        and recovery["recovery"] is not None and recovery["recovery"] >= .40
        and recovery["simultaneous_95_lower"] > .20
        and all(row["recovery"] is not None and row["recovery"] > 0
                for row in half_recoveries)
        and abs(off_change) <= .01
    )
    pred_d = bool(
        interchange_result["separation"] >= 2.0 and interchange_result["p_value"] <= .05
    )
    pred_e = bool(selected_direct >= .60 and selected_direct - control_direct >= .30)
    return {
        "selected_response": selected_response, "control_response": control_response,
        "selected_response_waves": half_responses,
        "selected_causal_recovery": recovery,
        "selected_recovery_waves": half_recoveries,
        "off_target_hybrid_minus_reference_nat": off_change,
        "interchange": interchange_result,
        "code_scale_diagnostics_not_used_by_hybrids": code_scales,
        "direct_score_cosine_gap": selected_direct - control_direct,
        "pred_b_response": pred_b, "pred_c_causal": pred_c,
        "pred_d_interchange": pred_d, "pred_e_geometry": pred_e,
    }


def main():
    started = time.time()
    payload, masks, scales, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        if PAIRS != ((0, 3), (1, 3)) or FACTOR != "score" or COMPONENT != "m9" \
                or set(scales) != set(PAIR_NAMES):
            raise RuntimeError("frozen code confirmation identity changed")
        print(json.dumps({
            "status": "dry_run_passed", "rung": 460, "model_loaded": False,
            "ood_code_rows_loaded_for_identity_only": True,
            "ood_code_model_outcomes_opened": False, "sealed_opened": False,
            "pairs": PAIR_NAMES, "factor": FACTOR, "component": COMPONENT,
            "natural_fit_score_ratios": {
                name: scales[name]["score_ratio"] for name in PAIR_NAMES
            },
            "input_metadata": metadata,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung460 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    response, halves, losses, counts, code_scales, audit, replay, reconstruction = collect(
        model, payload, masks, scales,
    )
    analysis = analyze(response, halves, losses, counts, code_scales)
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
    )
    pred_b = analysis["pred_b_response"]
    pred_c = analysis["pred_c_causal"]
    pred_d = analysis["pred_d_interchange"]
    pred_e = analysis["pred_e_geometry"]
    strong_null = bool(
        not pred_a or analysis["selected_response"]["all_positive"]["cosine"] < .30
        or analysis["selected_causal_recovery"]["recovery"] is None
        or analysis["selected_causal_recovery"]["recovery"] <= .10
        or analysis["interchange"]["separation"] <= 1.2
        or analysis["code_scale_diagnostics_not_used_by_hybrids"][PAIR_NAMES[0]][
            "direct_score_cosine"
        ] <= analysis["code_scale_diagnostics_not_used_by_hybrids"][PAIR_NAMES[1]][
            "direct_score_cosine"
        ]
    )
    bundle = {
        "schema": "equality_score_code_ood_rung460_sufficient_statistics_v1",
        "response_stats": response, "response_wave_stats": halves,
        "loss_sums": losses, "counts": counts,
        "raw_rows_tokens_logits_or_hidden_states_included": False,
        "sealed_attention0_opened": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 460,
        "claim_level": "frozen_code_ood_shared_score_confirmation_not_compression_or_adoption",
        "input_identity": metadata,
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "ood_code_model_outcomes_opened": True,
        "sealed_attention0_confirmation_opened": False,
        "frozen_natural_fit_scales": scales,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "analysis": analysis, "audit_totals": audit,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "outer_forwards": sum(row["forwards"] for row in audit.values()),
            "tested_pairs": 2, "tested_factors": 1, "tested_readers": 1,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_code_response': pred_b,
        'pred_c_code_causal_effect': pred_c,
        'pred_d_code_interchange': pred_d,
        'pred_e_code_score_geometry': pred_e,
        "strong_null": strong_null,
        "runtime_s": time.time() - started,
        "next_step": (
            "split_shared_score_product_into_qk1_and_qk2_feature_branches"
            if all((pred_a, pred_b, pred_c, pred_d, pred_e)) and not strong_null
            else "retain_natural_only_score_identification_and_reconsider_context_conditioning"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 460,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "analysis": analysis,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "execution_price": result["execution_price"],
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
