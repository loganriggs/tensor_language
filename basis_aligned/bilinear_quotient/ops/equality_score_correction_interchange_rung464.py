#!/usr/bin/env python3
"""RUNG464 -- native/hybrid equality source by later-correction interchange."""

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

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_score_path_factorial_rung463 as path_parent


PREREG = POLY / "EQUALITY_SCORE_CORRECTION_INTERCHANGE_RUNG464_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_score_path_factorial_rung463_results.json"
PARENT_SOURCE = ROOT / "ops/equality_score_path_factorial_rung463.py"
OUT = ROOT / "equality_score_correction_interchange_rung464_results.json"
BUNDLE = ROOT / "equality_score_correction_interchange_rung464_sufficient_statistics.pt"
SOURCES = ("0", "N", "H")
SOURCE_ARMS = {"0": "base", "N": "reference", "H": "score"}
CANDIDATES = path_parent.CANDIDATES
CELLS = path_parent.CELLS
PRIMARY_CELLS = path_parent.PRIMARY_CELLS
CONTEXT_CELLS = (
    "near_positive", "far_positive", "one_predecessor_positive",
    "multiple_predecessor_positive",
)
DOCUMENTS = path_parent.DOCUMENTS
BATCH = path_parent.BATCH
EXPECTED_FORWARDS = (DOCUMENTS // BATCH) * (2 + len(SOURCES) + len(SOURCES) ** 2)
HASHES = {
    PREREG: "7b29c21e66bf5493ce8ecd5b96e4ec91b04e2a0c12889d13090ef70687536010",
    PARENT_RESULT: "b83c6c16cd88cb4f3c85c207e6a9866779e8025abe8f7ad65b6bc4e137fb8922",
    PARENT_SOURCE: "92a5c30f1869fd9b6471aca56e24d0488bbe1cdf3b63baff4c93bdaf6a657fec",
    path_parent.ROW_RECEIPT: "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    path_parent.ROWS: "a82642da15dea4c82d486b46f118a55e480e7613e011ed588caa647eed16b660",
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
    if result.get("rung") != 463 or result.get("pred_a_instrument") is not True \
            or any(result.get(key) is not False for key in (
                "pred_b_direct_route", "pred_c_cumulative_suffix",
                "pred_d_mlp_over_attention", "pred_e_dominant_context_law",
            )) or result.get("strong_null") is not False:
        raise RuntimeError("rung463 registered result identity changed")
    payload, masks, scale, metadata = path_parent.validate_inputs()
    metadata = {
        **metadata,
        "rung463_result_sha256": sha256(PARENT_RESULT),
        "rung463_source_sha256": sha256(PARENT_SOURCE),
        "sources": list(SOURCES),
        "source_arms": SOURCE_ARMS,
        "later_write_donors": list(SOURCES),
        "document_halves": [[0, 96], [96, 192]],
    }
    return payload, masks, scale, metadata


@torch.no_grad()
def run_forward(
    model,
    tokens,
    *,
    arm: str,
    scale: Mapping[str, float] | None = None,
    capture_keys: Sequence[str] = (),
    patch_writes: Mapping[str, torch.Tensor] | None = None,
):
    if arm not in ("base", "reference", "score", "native", "replay"):
        raise ValueError(f"unknown arm: {arm}")
    analytical = arm != "native"
    if arm == "score" and scale is None:
        raise ValueError("score arm requires frozen scale")
    patch_writes = {} if patch_writes is None else dict(patch_writes)
    if arm in {"native", "replay"} and patch_writes:
        raise ValueError("instrument arms cannot carry later-write patches")
    if set(patch_writes) - set(CANDIDATES):
        raise ValueError("unregistered multi-write patch key")
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

    def patch_and_capture(key, write):
        if key in patch_writes:
            replacement = patch_writes[key]
            if replacement.shape != write.shape or replacement.dtype != write.dtype \
                    or replacement.device != write.device or not bool(
                        torch.isfinite(replacement).all()
                    ):
                raise RuntimeError(f"malformed patch at {key}")
            write = replacement
            audit["patches"] += 1
        if key in capture_set:
            captures[key] = write.detach().clone()
            audit["captures"] += 1
        return write

    def attention(event):
        nonlocal max_reconstruction
        if analytical and event.site in path_parent.parent.stage1.SITE_HEADS:
            write, factors, support, reconstruction = path_parent.parent.factor_parent._factor_site(
                event.state, event.first_value, event.block.attn, event.site, event.tokens,
            )
            max_reconstruction = max(max_reconstruction, reconstruction)
            audit["replayed_attention"] += 1
            if arm != "replay":
                early, late = path_parent.parent.PAIR
                early_site = path_parent.parent.factor_parent.TERMS[early][1]
                late_site = path_parent.parent.factor_parent.TERMS[late][1]
                if event.site == early_site:
                    cached_early.update(factors[early])
                    write = write - factors[early]["native_term"]
                if event.site == late_site:
                    if not cached_early:
                        raise RuntimeError("early factors missing before layer8")
                    late_factor = factors[late]
                    if arm != "reference":
                        write = write - late_factor["native_term"]
                        if arm == "score":
                            assert scale is not None
                            score = cached_early["p"] * scale["score_ratio"]
                            hybrid = torch.bmm(score * support, late_factor["u"]).to(write.dtype)
                            write = write + hybrid
            next_value = event.first_value
        else:
            write, next_value = event.block.attn(event.state, event.first_value)
            audit["native_attention"] += 1
        return patch_and_capture(f"a{event.site}", write), next_value

    def mlp(event):
        write = event.block.mlp(event.state)
        audit["native_mlp"] += 1
        return patch_and_capture(f"m{event.site}", write)

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
    if set(captures) != capture_set:
        raise RuntimeError("capture set changed")
    if audit["patches"] != len(patch_writes):
        raise RuntimeError("not every declared multi-write patch fired exactly once")
    return logits, captures, audit, max_reconstruction


def _effect_report(base, other, counts, start=0, stop=DOCUMENTS):
    return path_parent.parent.effect_report(base, other, counts, start, stop)


def _cosine_and_ratio(left, right):
    left = torch.tensor(left, dtype=torch.float64)
    right = torch.tensor(right, dtype=torch.float64)
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    cosine = float(torch.dot(left, right) / max(left_norm * right_norm, 1e-30))
    larger_over_smaller = max(left_norm, right_norm) / max(min(left_norm, right_norm), 1e-30)
    return {"cosine": cosine, "left_norm": left_norm, "right_norm": right_norm,
            "larger_over_smaller": larger_over_smaller,
            "right_over_left": right_norm / max(left_norm, 1e-30)}


def _sign_pattern(vector):
    return bool(vector[0] < 0 and vector[1] > 0 and vector[2] > 0 and vector[3] < 0)


def _analyze_window(losses, counts, start, stop):
    base = losses[0, 0]
    reports = {}
    for si, source in enumerate(SOURCES):
        reports[source] = {}
        for wi, donor in enumerate(SOURCES):
            reports[source][donor] = _effect_report(
                base, losses[si, wi], counts, start, stop,
            )
    corrections = {}
    for source in ("N", "H"):
        corrections[source] = {}
        direct = reports[source]["0"]
        for donor in ("N", "H"):
            corrections[source][donor] = {
                cell: reports[source][donor][cell]["effect_nat"]
                - direct[cell]["effect_nat"] for cell in CELLS
            }
    vectors = {
        source: {donor: [corrections[source][donor][cell] for cell in CONTEXT_CELLS]
                 for donor in ("N", "H")}
        for source in ("N", "H")
    }
    matched = _cosine_and_ratio(vectors["N"]["N"], vectors["H"]["H"])
    native_cross = _cosine_and_ratio(vectors["N"]["N"], vectors["N"]["H"])
    hybrid_cross = _cosine_and_ratio(vectors["H"]["H"], vectors["H"]["N"])
    native_stake = reports["N"]["N"]["all_positive"]["effect_nat"]
    hybrid_stake = reports["H"]["H"]["all_positive"]["effect_nat"]
    cross_recovery = {
        "native_source_hybrid_correction": (
            reports["N"]["H"]["all_positive"]["effect_nat"] / native_stake
            if native_stake > 0 else None
        ),
        "hybrid_source_native_correction": (
            reports["H"]["N"]["all_positive"]["effect_nat"] / hybrid_stake
            if hybrid_stake > 0 else None
        ),
    }
    correction_only = {}
    for donor in ("N", "H"):
        row = reports["0"][donor]
        correction_only[donor] = {
            "all_positive_effect_nat": row["all_positive"]["effect_nat"],
            "near_effect_nat": row["near_positive"]["effect_nat"],
            "multiple_effect_nat": row["multiple_predecessor_positive"]["effect_nat"],
        }
    return {
        "reports": reports, "corrections": corrections, "correction_vectors": vectors,
        "matched_correction_comparison": matched,
        "native_source_cross_comparison": native_cross,
        "hybrid_source_cross_comparison": hybrid_cross,
        "cross_recovery": cross_recovery,
        "context_orders": {
            f"{source},{donor}": path_parent.parent.context_order(reports[source][donor])
            for source in ("N", "H") for donor in ("N", "H")
        },
        "correction_only": correction_only,
        "matched_stakes": {"native": native_stake, "hybrid": hybrid_stake},
    }


def analyze(losses, counts):
    pooled = _analyze_window(losses, counts, 0, DOCUMENTS)
    halves = [_analyze_window(losses, counts, start, start + 96) for start in (0, 96)]
    matched_patterns = (
        _sign_pattern(pooled["correction_vectors"]["N"]["N"])
        and _sign_pattern(pooled["correction_vectors"]["H"]["H"])
        and all(_sign_pattern(half["correction_vectors"][source][source])
                for half in halves for source in ("N", "H"))
    )
    pred_b = bool(
        matched_patterns and pooled["matched_correction_comparison"]["cosine"] >= .80
        and pooled["matched_correction_comparison"]["larger_over_smaller"] <= 2.0
        and all(half["matched_correction_comparison"]["cosine"] > 0 for half in halves)
    )
    pred_c = bool(
        pooled["native_source_cross_comparison"]["cosine"] >= .80
        and pooled["hybrid_source_cross_comparison"]["cosine"] >= .80
        and .50 <= pooled["native_source_cross_comparison"]["right_over_left"] <= 1.50
        and .50 <= pooled["hybrid_source_cross_comparison"]["right_over_left"] <= 1.50
        and all(half[key]["cosine"] > 0 for half in halves for key in (
            "native_source_cross_comparison", "hybrid_source_cross_comparison",
        ))
    )
    cross_keys = (("N", "H", "native_source_hybrid_correction"),
                  ("H", "N", "hybrid_source_native_correction"))
    pred_d = bool(
        all(pooled["cross_recovery"][key] is not None
            and pooled["cross_recovery"][key] >= .75 for _, _, key in cross_keys)
        and all(half["cross_recovery"][key] is not None
                and half["cross_recovery"][key] > 0
                for half in halves for _, _, key in cross_keys)
        and all(all(pooled["context_orders"][f"{source},{donor}"].values())
                for source, donor, _ in cross_keys)
        and all(all(half["context_orders"][f"{source},{donor}"].values())
                for half in halves for source, donor, _ in cross_keys)
        and all(abs(pooled["reports"][source][donor]["off_target"]["effect_nat"]) <= .01
                for source, donor, _ in cross_keys)
    )
    def standalone_ok(result, donor):
        row = result["correction_only"][donor]
        signs = row["near_effect_nat"] < 0 and row["multiple_effect_nat"] < 0
        return row["all_positive_effect_nat"] <= 0 or signs
    pred_e = bool(
        all(standalone_ok(pooled, donor) for donor in ("N", "H"))
        and all(half["correction_only"][donor]["near_effect_nat"] < 0
                and half["correction_only"][donor]["multiple_effect_nat"] < 0
                for half in halves for donor in ("N", "H"))
    )
    matched_positive = all(value > 0 for value in pooled["matched_stakes"].values())
    any_cross_order = any(
        all(pooled["context_orders"][f"{source},{donor}"].values())
        for source, donor, _ in cross_keys
    )
    strong_science_null = bool(
        not matched_positive or pooled["matched_correction_comparison"]["cosine"] <= 0
        or any(pooled["cross_recovery"][key] is None
               or pooled["cross_recovery"][key] <= .25 for _, _, key in cross_keys)
        or not any_cross_order
    )
    return {
        "pooled": pooled, "halves": halves,
        "pred_b_common_matched_correction": pred_b,
        "pred_c_correction_interchange": pred_c,
        "pred_d_crossed_complete_circuits": pred_d,
        "pred_e_correction_not_standalone": pred_e,
        "strong_science_null": strong_science_null,
    }


@torch.no_grad()
def collect(model, payload, masks, scale):
    rows = payload["rows"]
    losses = torch.zeros(len(SOURCES), len(SOURCES), DOCUMENTS, len(CELLS), dtype=torch.float64)
    counts = torch.zeros(DOCUMENTS, len(CELLS), dtype=torch.float64)
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    diagonal_replay = {source: {"max_abs": 0.0, "relative_squared": 0.0}
                       for source in SOURCES}
    reconstruction = 0.0
    device = next(model.parameters()).device
    for start in range(0, DOCUMENTS, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = run_forward(model, tokens, arm="native")
        path_parent.parent._record_audit(audit_totals, "interchange:native", audit,
                                         analytical=False, captures=0, patches=0)
        replay_logits, _, audit, error = run_forward(model, tokens, arm="replay")
        path_parent.parent._record_audit(audit_totals, "interchange:replay", audit,
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
        direct_logits = {}
        for source in SOURCES:
            logits, source_captures, audit, error = run_forward(
                model, tokens, arm=SOURCE_ARMS[source], scale=scale,
                capture_keys=CANDIDATES,
            )
            path_parent.parent._record_audit(
                audit_totals, f"interchange:capture:{source}", audit,
                analytical=True, captures=len(CANDIDATES), patches=0,
            )
            reconstruction = max(reconstruction, error)
            captures[source] = source_captures
            direct_logits[source] = logits
        for si, source in enumerate(SOURCES):
            for wi, donor in enumerate(SOURCES):
                logits, _, audit, error = run_forward(
                    model, tokens, arm=SOURCE_ARMS[source], scale=scale,
                    patch_writes=captures[donor],
                )
                path_parent.parent._record_audit(
                    audit_totals, f"interchange:{source},{donor}", audit,
                    analytical=True, captures=0, patches=len(CANDIDATES),
                )
                reconstruction = max(reconstruction, error)
                sums, observed = path_parent.parent._ce_sums(logits, batch_rows, masks, start)
                if si == 0 and wi == 0:
                    counts[start:start + BATCH] = observed
                elif not torch.equal(observed, counts[start:start + BATCH]):
                    raise RuntimeError("interchange support changed")
                losses[si, wi, start:start + BATCH] = sums
                if source == donor:
                    difference = logits - direct_logits[source]
                    row = diagonal_replay[source]
                    row["max_abs"] = max(row["max_abs"], float(difference.abs().max()))
                    row["relative_squared"] = max(
                        row["relative_squared"],
                        float(difference.square().sum())
                        / max(float(direct_logits[source].square().sum()), 1e-30),
                    )
                del logits
        del captures, direct_logits
    return losses, counts, audit_totals, replay, diagonal_replay, reconstruction


def main():
    started = time.time()
    payload, masks, scale, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 464, "model_loaded": False,
            "new_interchange_outcomes_opened": False, "sealed_opened": False,
            "sources": SOURCES, "source_arms": SOURCE_ARMS,
            "later_write_count": len(CANDIDATES), "expected_forwards": EXPECTED_FORWARDS,
            "input_metadata": metadata,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung464 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    losses, counts, audit, replay, diagonal, reconstruction = collect(
        model, payload, masks, scale,
    )
    analysis = analyze(losses, counts)
    forwards = sum(row["forwards"] for row in audit.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and all(row["relative_squared"] <= 1e-12 for row in diagonal.values())
        and forwards == EXPECTED_FORWARDS
    )
    strong_null = bool(not pred_a or analysis["strong_science_null"])
    bundle = {
        "schema": "equality_score_correction_interchange_rung464_sufficient_statistics_v1",
        "loss_sums_source_by_later_write_donor": losses,
        "counts": counts,
        "source_order": SOURCES, "donor_order": SOURCES,
        "raw_rows_tokens_logits_or_hidden_states_included": False,
        "sealed_attention0_opened": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 464,
        "claim_level": "already_open_code_source_correction_interchange",
        "input_identity": metadata,
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "frozen_pair": path_parent.PAIR_NAME,
        "frozen_natural_fit_scale": scale,
        "analysis": analysis,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "diagonal_trajectory_replay": diagonal,
        "audit_totals": audit,
        "sufficient_statistics": {
            "path": str(BUNDLE), "sha256": sha256(BUNDLE), "bytes": BUNDLE.stat().st_size,
        },
        "execution_price": {
            "outer_forwards": forwards, "source_donor_cells": len(SOURCES) ** 2,
            "later_writes_per_patch": len(CANDIDATES),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_common_matched_correction': analysis["pred_b_common_matched_correction"],
        'pred_c_correction_interchange': analysis["pred_c_correction_interchange"],
        'pred_d_crossed_complete_circuits': analysis["pred_d_crossed_complete_circuits"],
        'pred_e_correction_not_standalone': analysis["pred_e_correction_not_standalone"],
        "strong_null": strong_null,
        "runtime_s": time.time() - started,
        "next_step": (
            "independent_source_plus_correction_removal_and_swap"
            if pred_a and analysis["pred_b_common_matched_correction"]
            and analysis["pred_c_correction_interchange"]
            and analysis["pred_d_crossed_complete_circuits"] and not strong_null
            else "localize_source_specific_correction_difference_without_rank_reduction"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 464,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "analysis": analysis,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "diagonal_trajectory_replay": diagonal,
        "execution_price": result["execution_price"],
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
