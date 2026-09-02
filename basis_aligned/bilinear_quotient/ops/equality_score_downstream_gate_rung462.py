#!/usr/bin/env python3
"""RUNG462 -- causal later-write localization of the equality context gate."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
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
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_score_code_context_rung461 as parent
import equality_term_score_payload_rung459 as factor_parent
import equality_term_subset_factorial_stage1 as stage1


PREREG = POLY / "EQUALITY_SCORE_DOWNSTREAM_GATE_RUNG462_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_score_code_context_rung461_results.json"
PARENT_SOURCE = ROOT / "ops/equality_score_code_context_rung461.py"
ROWS = ROOT / ".rowcache_induction_equality_tensor_final_ood_v2/ood_code.pt"
ROW_RECEIPT = ROOT / "induction_equality_tensor_final_ood_v2_rows_receipt.json"
OUT = ROOT / "equality_score_downstream_gate_rung462_results.json"
BUNDLE = ROOT / "equality_score_downstream_gate_rung462_sufficient_statistics.pt"
PAIR = (0, 3)
PAIR_NAME = "L5H5->L8H4"
ARMS = ("base", "reference", "score")
PATCH_MODES = ("reference_patch", "hybrid_patch", "permuted_patch")
PRIMARY_CELLS = parent.PRIMARY_CELLS
CELLS = parent.CELLS
CANDIDATES = ("m8",) + tuple(
    component for site in range(9, 18) for component in (f"a{site}", f"m{site}")
)
DOCUMENTS = 192
BATCH = 4
FIT_DOCUMENTS = 96
VALIDATION_DOCUMENTS = 96
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = "equality-score-downstream-gate-rung462:bootstrap:0"
FIT_EXPECTED_FORWARDS = (FIT_DOCUMENTS // BATCH) * (2 + len(ARMS) + len(CANDIDATES))
VALIDATION_EXPECTED_FORWARDS = (VALIDATION_DOCUMENTS // BATCH) * (
    len(ARMS) + len(PATCH_MODES)
)
HASHES = {
    PREREG: "26152c085a1cd019a9a7f2d7c2f4efece296fc7ceb820836ece70a02147ede1e",
    PARENT_RESULT: "901e7e9f33618db95d62e85c75d283a2f4736dff240ab34d8b617f505018af36",
    PARENT_SOURCE: "8b2021d306769ff653b0bebd82db9539bac66ec78d8e3c16a4fcd3324e4aa29a",
    ROW_RECEIPT: "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    ROWS: "a82642da15dea4c82d486b46f118a55e480e7613e011ed588caa647eed16b660",
    ROOT / "ops/equality_term_score_payload_rung459.py":
        "9f9e66f689452cbcb14d741792b66eb9ff526dff5472a5938c58a2a4c82620d8",
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
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("rung") != 461 \
            or not all(result.get(key) is True for key in (
                "pred_a_instrument", "pred_b_context_order", "pred_c_causal_tracking",
                "pred_d_shared_direction",
            )) or result.get("pred_e_amplitude_explanation") is not False \
            or result.get("strong_null") is not False:
        raise RuntimeError("rung461 registered result identity changed")
    payload, masks, scale, metadata = parent.validate_inputs()
    metadata = {
        **metadata,
        "rung461_result_sha256": sha256(PARENT_RESULT),
        "rung461_source_sha256": sha256(PARENT_SOURCE),
        "fit_documents": [0, 96], "validation_documents": [96, 192],
        "validation_waves": [[96, 144], [144, 192]],
        "candidate_order": list(CANDIDATES),
    }
    return payload, masks, scale, metadata


def _record_audit(totals, label, audit, *, analytical, captures, patches):
    expected = {
        "native_attention": 15 if analytical else 18,
        "replayed_attention": 3 if analytical else 0,
        "native_mlp": 18,
        "captures": captures,
        "patches": patches,
    }
    if audit != expected:
        raise RuntimeError(f"forward audit changed for {label}: {audit} != {expected}")
    row = totals.setdefault(label, {"forwards": 0, **{key: 0 for key in expected}})
    row["forwards"] += 1
    for key, value in audit.items():
        row[key] += value


@torch.no_grad()
def run_forward(
    model,
    tokens,
    *,
    arm: str,
    scale: Mapping[str, float] | None = None,
    capture_keys: Sequence[str] = (),
    patch_key: str | None = None,
    patch_write: torch.Tensor | None = None,
):
    if arm not in (*ARMS, "native", "replay"):
        raise ValueError(f"unknown arm: {arm}")
    analytical = arm != "native"
    if arm == "score" and scale is None:
        raise ValueError("score arm requires frozen scale")
    if (patch_key is None) != (patch_write is None):
        raise ValueError("patch key and write must travel together")
    if patch_key is not None and (arm != "base" or patch_key not in CANDIDATES):
        raise ValueError("only a registered base-trajectory candidate may be patched")
    capture_set = set(capture_keys)
    if len(capture_set) != len(capture_keys) or not capture_set <= set(CANDIDATES):
        raise ValueError("capture identity changed")
    cached_early = {}
    captures = {}
    audit = {
        "native_attention": 0, "replayed_attention": 0, "native_mlp": 0,
        "captures": 0, "patches": 0,
    }
    max_reconstruction = 0.0

    def maybe_patch(key, write):
        if patch_key != key:
            return write
        assert patch_write is not None
        if patch_write.shape != write.shape or patch_write.dtype != write.dtype \
                or patch_write.device != write.device or not bool(
                    torch.isfinite(patch_write).all()
                ):
            raise RuntimeError("cached patch write is malformed")
        audit["patches"] += 1
        return patch_write

    def maybe_capture(key, write):
        if key in capture_set:
            captures[key] = write.detach().clone()
            audit["captures"] += 1

    def attention(event):
        nonlocal max_reconstruction
        if analytical and event.site in stage1.SITE_HEADS:
            write, factors, support, reconstruction = factor_parent._factor_site(
                event.state, event.first_value, event.block.attn, event.site, event.tokens,
            )
            max_reconstruction = max(max_reconstruction, reconstruction)
            audit["replayed_attention"] += 1
            if arm != "replay":
                early, late = PAIR
                early_site = factor_parent.TERMS[early][1]
                late_site = factor_parent.TERMS[late][1]
                if event.site == early_site:
                    cached_early.update(factors[early])
                    write = write - factors[early]["native_term"]
                if event.site == late_site:
                    if not cached_early:
                        raise RuntimeError("early factors were not cached before layer8")
                    late_factor = factors[late]
                    if arm != "reference":
                        write = write - late_factor["native_term"]
                        if arm == "score":
                            assert scale is not None
                            p = cached_early["p"] * scale["score_ratio"]
                            hybrid = torch.bmm(p * support, late_factor["u"]).to(write.dtype)
                            write = write + hybrid
            next_value = event.first_value
        else:
            write, next_value = event.block.attn(event.state, event.first_value)
            audit["native_attention"] += 1
        key = f"a{event.site}"
        write = maybe_patch(key, write)
        maybe_capture(key, write)
        return write, next_value

    def mlp(event):
        write = event.block.mlp(event.state)
        audit["native_mlp"] += 1
        key = f"m{event.site}"
        write = maybe_patch(key, write)
        maybe_capture(key, write)
        return write

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
    if set(captures) != capture_set:
        raise RuntimeError("capture set changed")
    if patch_key is not None and audit["patches"] != 1:
        raise RuntimeError("registered patch did not fire exactly once")
    return logits, captures, audit, max_reconstruction


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


def _empty_response_stats():
    shape = (len(CANDIDATES), len(CELLS))
    return {
        key: torch.zeros(shape, dtype=torch.float64)
        for key in ("ref2", "hyb2", "cross", "tokens")
    }


def _accumulate_response(stats, captures, masks, global_start):
    for ji, key in enumerate(CANDIDATES):
        reference = captures["reference"][key] - captures["base"][key]
        hybrid = captures["score"][key] - captures["base"][key]
        for ci, cell in enumerate(CELLS):
            selected = masks[cell][global_start:global_start + BATCH]
            if not bool(selected.any()):
                continue
            ref = reference[selected].float()
            hyb = hybrid[selected].float()
            stats["ref2"][ji, ci] += ref.square().sum().double().cpu()
            stats["hyb2"][ji, ci] += hyb.square().sum().double().cpu()
            stats["cross"][ji, ci] += (ref * hyb).sum().double().cpu()
            stats["tokens"][ji, ci] += int(selected.sum())


def response_report(stats, candidate):
    ji = CANDIDATES.index(candidate)
    report = {}
    for ci, cell in enumerate(CELLS):
        ref2 = float(stats["ref2"][ji, ci])
        hyb2 = float(stats["hyb2"][ji, ci])
        cross = float(stats["cross"][ji, ci])
        tokens = int(stats["tokens"][ji, ci])
        report[cell] = {
            "cosine": cross / math.sqrt(max(ref2 * hyb2, 1e-30)),
            "reference_raw_coordinate_rms": math.sqrt(
                ref2 / max(tokens * stage1.D, 1)
            ),
            "hybrid_raw_coordinate_rms": math.sqrt(
                hyb2 / max(tokens * stage1.D, 1)
            ),
            "tokens": tokens,
        }
    return report


def effect_report(base, other, counts, start=0, stop=None):
    stop = len(counts) if stop is None else stop
    answer = {}
    for ci, cell in enumerate(CELLS):
        denominator = float(counts[start:stop, ci].sum())
        effect = float(
            (base[start:stop, ci] - other[start:stop, ci]).sum() / denominator
        )
        answer[cell] = {"effect_nat": effect, "tokens": int(denominator)}
    return answer


def native_stake_report(base, reference, counts, start=0, stop=None):
    return effect_report(base, reference, counts, start, stop)


def context_order(report):
    return {
        "far_gt_near": (
            report["far_positive"]["effect_nat"]
            > report["near_positive"]["effect_nat"]
        ),
        "one_gt_multiple": (
            report["one_predecessor_positive"]["effect_nat"]
            > report["multiple_predecessor_positive"]["effect_nat"]
        ),
    }


def screen_candidates(arm_losses, patch_losses, counts, response_stats):
    base = arm_losses[ARMS.index("base")]
    reference = arm_losses[ARMS.index("reference")]
    stakes = native_stake_report(base, reference, counts)
    all_stake = stakes["all_positive"]["effect_nat"]
    rows = []
    for ji, candidate in enumerate(CANDIDATES):
        effects = effect_report(base, patch_losses[ji], counts)
        recovery = effects["all_positive"]["effect_nat"] / all_stake if all_stake > 0 else None
        order = context_order(effects)
        qualifies = bool(
            recovery is not None and recovery >= .10
            and order["far_gt_near"] and order["one_gt_multiple"]
            and abs(effects["off_target"]["effect_nat"]) <= .01
        )
        rows.append({
            "candidate": candidate, "effects": effects,
            "all_positive_recovery": recovery, "context_order": order,
            "qualifies": qualifies,
            "response_companion_not_used_for_selection": response_report(
                response_stats, candidate,
            ),
        })
    qualified = [row for row in rows if row["qualifies"]]
    selected = max(
        qualified,
        key=lambda row: (row["all_positive_recovery"], -CANDIDATES.index(row["candidate"])),
        default=None,
    )
    return {
        "native_stakes": stakes, "candidates": rows,
        "selected": selected, "qualified_count": len(qualified),
    }


def bootstrap_recovery(base, reference, patch, counts):
    ci = CELLS.index("all_positive")
    denominator_counts = counts[:, ci]
    generator = torch.Generator().manual_seed(
        int.from_bytes(hashlib.sha256(BOOTSTRAP_SEED.encode()).digest()[:8], "little")
    )
    chunks = []
    every_stake_positive = True
    for begin in range(0, BOOTSTRAP_DRAWS, 500):
        n = min(500, BOOTSTRAP_DRAWS - begin)
        draws = torch.randint(len(counts), (n, len(counts)), generator=generator)
        weights = torch.zeros(n, len(counts), dtype=torch.float64)
        weights.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
        denominator = weights @ denominator_counts
        stakes = (weights @ (base[:, ci] - reference[:, ci])) / denominator
        effects = (weights @ (base[:, ci] - patch[:, ci])) / denominator
        every_stake_positive &= bool((stakes > 0).all())
        chunks.append(torch.where(stakes > 0, effects / stakes, torch.zeros_like(stakes)))
    draws = torch.cat(chunks).sort().values
    point_stake = float((base[:, ci] - reference[:, ci]).sum() / denominator_counts.sum())
    point_effect = float((base[:, ci] - patch[:, ci]).sum() / denominator_counts.sum())
    return {
        "native_stake_nat": point_stake,
        "patch_effect_nat": point_effect,
        "recovery": point_effect / point_stake if point_stake > 0 else None,
        "simultaneous_95_lower": float(draws[math.floor(.025 * BOOTSTRAP_DRAWS)]),
        "every_bootstrap_native_stake_positive": every_stake_positive,
        "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
    }


def analyze_validation(arm_losses, patch_losses, counts):
    base = arm_losses[ARMS.index("base")]
    reference = arm_losses[ARMS.index("reference")]
    stakes = native_stake_report(base, reference, counts)
    reports = {
        mode: effect_report(base, patch_losses[mi], counts)
        for mi, mode in enumerate(PATCH_MODES)
    }
    waves = []
    for start in (0, 48):
        wave_stakes = native_stake_report(base, reference, counts, start, start + 48)
        wave_reports = {
            mode: effect_report(base, patch_losses[mi], counts, start, start + 48)
            for mi, mode in enumerate(PATCH_MODES)
        }
        stake = wave_stakes["all_positive"]["effect_nat"]
        effect = wave_reports["reference_patch"]["all_positive"]["effect_nat"]
        waves.append({
            "native_stakes": wave_stakes, "patch_effects": wave_reports,
            "reference_patch_recovery": effect / stake if stake > 0 else None,
            "reference_context_order": context_order(wave_reports["reference_patch"]),
        })
    bootstrap = bootstrap_recovery(
        base, reference, patch_losses[PATCH_MODES.index("reference_patch")], counts,
    )
    pooled_order = context_order(reports["reference_patch"])
    ref_values = torch.tensor([
        reports["reference_patch"][cell]["effect_nat"] for cell in PRIMARY_CELLS
    ])
    hybrid_values = torch.tensor([
        reports["hybrid_patch"][cell]["effect_nat"] for cell in PRIMARY_CELLS
    ])
    hybrid_spearman = stage1.spearman(ref_values, hybrid_values)
    correct = reports["reference_patch"]["all_positive"]["effect_nat"]
    hybrid = reports["hybrid_patch"]["all_positive"]["effect_nat"]
    permuted = reports["permuted_patch"]["all_positive"]["effect_nat"]
    separation = abs(correct) / max(abs(permuted), 1e-12)
    margin = correct - abs(permuted)
    hybrid_ratio = hybrid / correct if correct > 0 else None
    pred_c = bool(
        bootstrap["every_bootstrap_native_stake_positive"]
        and bootstrap["recovery"] is not None and bootstrap["recovery"] >= .10
        and bootstrap["simultaneous_95_lower"] > .02
        and all(row["reference_patch_recovery"] is not None
                and row["reference_patch_recovery"] > 0 for row in waves)
        and abs(reports["reference_patch"]["off_target"]["effect_nat"]) <= .01
    )
    pred_d = bool(
        all(pooled_order.values())
        and all(all(row["reference_context_order"].values()) for row in waves)
    )
    pred_e = bool(
        separation >= 2.0 and margin >= .005
        and hybrid_spearman >= .80
        and hybrid_ratio is not None and .50 <= hybrid_ratio <= 1.50
    )
    return {
        "native_stakes": stakes, "patch_effects": reports,
        "reference_patch_bootstrap": bootstrap,
        "waves": waves, "reference_context_order": pooled_order,
        "correct_vs_permuted": {
            "correct_effect_nat": correct, "permuted_effect_nat": permuted,
            "absolute_separation": separation, "signed_margin_nat": margin,
        },
        "hybrid_vs_reference": {
            "hybrid_effect_nat": hybrid, "reference_effect_nat": correct,
            "effect_ratio": hybrid_ratio, "four_cell_spearman": hybrid_spearman,
        },
        "pred_c_heldout_mediation": pred_c,
        "pred_d_context_law": pred_d,
        "pred_e_alignment_and_transplant": pred_e,
        "strong_science_null": bool(
            bootstrap["recovery"] is None or bootstrap["recovery"] <= .03
            or separation <= 1.10 or hybrid_spearman <= 0
        ),
    }


@torch.no_grad()
def collect_fit(model, payload, masks, scale):
    rows = payload["rows"]
    arm_losses = torch.zeros(len(ARMS), FIT_DOCUMENTS, len(CELLS), dtype=torch.float64)
    patch_losses = torch.zeros(
        len(CANDIDATES), FIT_DOCUMENTS, len(CELLS), dtype=torch.float64,
    )
    counts = torch.zeros(FIT_DOCUMENTS, len(CELLS), dtype=torch.float64)
    response = _empty_response_stats()
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    reconstruction = 0.0
    device = next(model.parameters()).device
    for global_start in range(0, FIT_DOCUMENTS, BATCH):
        batch_rows = rows[global_start:global_start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = run_forward(model, tokens, arm="native")
        _record_audit(audit_totals, "fit:native", audit,
                      analytical=False, captures=0, patches=0)
        replay_logits, _, audit, error = run_forward(model, tokens, arm="replay")
        _record_audit(audit_totals, "fit:replay", audit,
                      analytical=True, captures=0, patches=0)
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
            logits, arm_captures, audit, error = run_forward(
                model, tokens, arm=arm, scale=scale, capture_keys=CANDIDATES,
            )
            _record_audit(audit_totals, f"fit:{arm}", audit,
                          analytical=True, captures=len(CANDIDATES), patches=0)
            reconstruction = max(reconstruction, error)
            sums, observed = _ce_sums(logits, batch_rows, masks, global_start)
            local = slice(global_start, global_start + BATCH)
            arm_losses[ai, local] = sums
            if ai == 0:
                counts[local] = observed
            elif not torch.equal(observed, counts[local]):
                raise RuntimeError("fit supports changed across arms")
            captures[arm] = arm_captures
            del logits
        _accumulate_response(response, captures, masks, global_start)
        for ji, candidate in enumerate(CANDIDATES):
            logits, _, audit, error = run_forward(
                model, tokens, arm="base", scale=scale,
                patch_key=candidate, patch_write=captures["reference"][candidate],
            )
            _record_audit(audit_totals, f"fit:reference_patch:{candidate}", audit,
                          analytical=True, captures=0, patches=1)
            reconstruction = max(reconstruction, error)
            sums, observed = _ce_sums(logits, batch_rows, masks, global_start)
            if not torch.equal(observed, counts[global_start:global_start + BATCH]):
                raise RuntimeError("fit patch support changed")
            patch_losses[ji, global_start:global_start + BATCH] = sums
            del logits
        del captures
    return arm_losses, patch_losses, counts, response, audit_totals, replay, reconstruction


@torch.no_grad()
def collect_validation(model, payload, masks, scale, selected, audit_totals):
    rows = payload["rows"]
    arm_losses = torch.zeros(
        len(ARMS), VALIDATION_DOCUMENTS, len(CELLS), dtype=torch.float64,
    )
    patch_losses = torch.zeros(
        len(PATCH_MODES), VALIDATION_DOCUMENTS, len(CELLS), dtype=torch.float64,
    )
    counts = torch.zeros(VALIDATION_DOCUMENTS, len(CELLS), dtype=torch.float64)
    response = _empty_response_stats()
    reconstruction = 0.0
    device = next(model.parameters()).device
    for local_start in range(0, VALIDATION_DOCUMENTS, BATCH):
        global_start = FIT_DOCUMENTS + local_start
        batch_rows = rows[global_start:global_start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        captures = {}
        for ai, arm in enumerate(ARMS):
            logits, arm_captures, audit, error = run_forward(
                model, tokens, arm=arm, scale=scale, capture_keys=CANDIDATES,
            )
            _record_audit(audit_totals, f"validation:{arm}", audit,
                          analytical=True, captures=len(CANDIDATES), patches=0)
            reconstruction = max(reconstruction, error)
            sums, observed = _ce_sums(logits, batch_rows, masks, global_start)
            arm_losses[ai, local_start:local_start + BATCH] = sums
            if ai == 0:
                counts[local_start:local_start + BATCH] = observed
            elif not torch.equal(observed, counts[local_start:local_start + BATCH]):
                raise RuntimeError("validation supports changed across arms")
            captures[arm] = arm_captures
            del logits
        _accumulate_response(response, captures, masks, global_start)
        writes = (
            captures["reference"][selected],
            captures["score"][selected],
            torch.roll(captures["reference"][selected], shifts=1, dims=0),
        )
        for mi, (mode, patch_write) in enumerate(zip(PATCH_MODES, writes)):
            logits, _, audit, error = run_forward(
                model, tokens, arm="base", scale=scale,
                patch_key=selected, patch_write=patch_write,
            )
            _record_audit(audit_totals, f"validation:{mode}:{selected}", audit,
                          analytical=True, captures=0, patches=1)
            reconstruction = max(reconstruction, error)
            sums, observed = _ce_sums(logits, batch_rows, masks, global_start)
            if not torch.equal(observed, counts[local_start:local_start + BATCH]):
                raise RuntimeError("validation patch support changed")
            patch_losses[mi, local_start:local_start + BATCH] = sums
            del logits
        del captures
    return arm_losses, patch_losses, counts, response, reconstruction


def main():
    started = time.time()
    payload, masks, scale, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 462, "model_loaded": False,
            "new_patch_outcomes_opened": False, "sealed_opened": False,
            "pair": PAIR_NAME, "candidate_order": CANDIDATES,
            "fit_expected_forwards": FIT_EXPECTED_FORWARDS,
            "validation_expected_forwards_if_selected": VALIDATION_EXPECTED_FORWARDS,
            "natural_fit_score_ratio": scale["score_ratio"],
            "input_metadata": metadata,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung462 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    fit_arms, fit_patches, fit_counts, fit_response, audit, replay, reconstruction = (
        collect_fit(model, payload, masks, scale)
    )
    fit_screen = screen_candidates(fit_arms, fit_patches, fit_counts, fit_response)
    selected_row = fit_screen["selected"]
    selected = None if selected_row is None else selected_row["candidate"]
    pred_b = selected is not None
    validation = None
    validation_tensors = None
    validation_response = None
    if selected is not None:
        val_arms, val_patches, val_counts, val_response, val_reconstruction = collect_validation(
            model, payload, masks, scale, selected, audit,
        )
        reconstruction = max(reconstruction, val_reconstruction)
        validation = analyze_validation(val_arms, val_patches, val_counts)
        validation["selected_response_companion_not_used_for_predictions"] = response_report(
            val_response, selected,
        )
        validation_tensors = {
            "arm_loss_sums": val_arms, "patch_loss_sums": val_patches,
            "counts": val_counts,
        }
        validation_response = val_response
    forwards = sum(row["forwards"] for row in audit.values())
    expected_forwards = FIT_EXPECTED_FORWARDS + (
        VALIDATION_EXPECTED_FORWARDS if selected is not None else 0
    )
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12
        and reconstruction <= 1e-10
        and forwards == expected_forwards
    )
    pred_c = bool(validation and validation["pred_c_heldout_mediation"])
    pred_d = bool(validation and validation["pred_d_context_law"])
    pred_e = bool(validation and validation["pred_e_alignment_and_transplant"])
    strong_null = bool(
        not pred_a or not pred_b
        or (validation is not None and validation["strong_science_null"])
    )
    bundle = {
        "schema": "equality_score_downstream_gate_rung462_sufficient_statistics_v1",
        "fit": {
            "arm_loss_sums": fit_arms, "patch_loss_sums": fit_patches,
            "counts": fit_counts, "response_stats": fit_response,
        },
        "validation": validation_tensors,
        "validation_response_stats": validation_response,
        "selected_candidate_frozen_before_validation_patch_outcomes": selected,
        "raw_rows_tokens_logits_or_hidden_states_included": False,
        "sealed_attention0_opened": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 462,
        "claim_level": "already_open_code_downstream_mediator_diagnostic",
        "input_identity": metadata,
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "frozen_pair": PAIR_NAME, "frozen_natural_fit_scale": scale,
        "fit_screen": fit_screen,
        "selected_candidate_frozen_before_validation_patch_outcomes": selected,
        "validation": validation,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "audit_totals": audit,
        "sufficient_statistics": {
            "path": str(BUNDLE), "sha256": sha256(BUNDLE), "bytes": BUNDLE.stat().st_size,
        },
        "execution_price": {
            "outer_forwards": forwards,
            "fit_candidates": len(CANDIDATES),
            "validation_candidates": 1 if selected is not None else 0,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_discovery_candidate': pred_b,
        'pred_c_heldout_mediation': pred_c,
        'pred_d_context_law': pred_d,
        'pred_e_alignment_and_transplant': pred_e,
        "strong_null": strong_null,
        "runtime_s": time.time() - started,
        "next_step": (
            "heldout_targeted_mediator_removal_and_interchange"
            if all((pred_a, pred_b, pred_c, pred_d, pred_e)) and not strong_null
            else "test_distributed_cumulative_suffix_boundary_not_rank"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 462,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "selected_candidate": selected,
        "fit_screen": fit_screen,
        "validation": validation,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "execution_price": result["execution_price"],
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
