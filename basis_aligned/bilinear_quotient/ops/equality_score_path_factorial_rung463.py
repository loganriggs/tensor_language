#!/usr/bin/env python3
"""RUNG463 -- direct-residual versus distributed-suffix equality path factorial."""

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
import equality_score_downstream_gate_rung462 as parent


PREREG = POLY / "EQUALITY_SCORE_PATH_FACTORIAL_RUNG463_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_score_downstream_gate_rung462_results.json"
PARENT_SOURCE = ROOT / "ops/equality_score_downstream_gate_rung462.py"
OUT = ROOT / "equality_score_path_factorial_rung463_results.json"
BUNDLE = ROOT / "equality_score_path_factorial_rung463_sufficient_statistics.pt"
ROWS = parent.ROWS
ROW_RECEIPT = parent.ROW_RECEIPT
PAIR_NAME = parent.PAIR_NAME
CELLS = parent.CELLS
PRIMARY_CELLS = parent.PRIMARY_CELLS
CANDIDATES = parent.CANDIDATES
MLP_KEYS = tuple(key for key in CANDIDATES if key.startswith("m"))
ATTENTION_KEYS = tuple(key for key in CANDIDATES if key.startswith("a"))
BASE_ARMS = ("base", "reference")
DOCUMENTS = parent.DOCUMENTS
BATCH = parent.BATCH
BEST_SINGLE_RECOVERY = 0.0563254
EXPECTED_FORWARDS = (DOCUMENTS // BATCH) * (2 + 2 + 1 + 2 + len(CANDIDATES))
HASHES = {
    PREREG: "e9473bb331f75c55d092576527595092fc5baa6743ad1fbe984d9cd9cf874231",
    PARENT_RESULT: "875975ec22712a6041067d4254a2a86428428a7b0d499a91e9dbb602b7bb6acb",
    PARENT_SOURCE: "bf5bf18a05e9dc57bf36b900b5cbe4ef32a63a30497d98fd0f61a679fbad3f23",
    ROW_RECEIPT: "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    ROWS: "a82642da15dea4c82d486b46f118a55e480e7613e011ed588caa647eed16b660",
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
    if result.get("rung") != 462 or result.get("pred_a_instrument") is not True \
            or result.get("pred_b_discovery_candidate") is not False \
            or result.get("strong_null") is not True \
            or result.get("fit_screen", {}).get("qualified_count") != 0:
        raise RuntimeError("rung462 registered null identity changed")
    payload, masks, scale, metadata = parent.validate_inputs()
    metadata = {
        **metadata,
        "rung462_result_sha256": sha256(PARENT_RESULT),
        "rung462_source_sha256": sha256(PARENT_SOURCE),
        "document_halves": [[0, 96], [96, 192]],
        "write_order": list(CANDIDATES),
        "mlp_keys": list(MLP_KEYS), "attention_keys": list(ATTENTION_KEYS),
        "suffix_boundaries": [
            {"boundary": key, "patched_writes": list(CANDIDATES[i:])}
            for i, key in enumerate(CANDIDATES)
        ],
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
    if arm not in ("base", "reference", "native", "replay"):
        raise ValueError(f"unknown arm: {arm}")
    analytical = arm != "native"
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
        if analytical and event.site in parent.stage1.SITE_HEADS:
            write, factors, support, reconstruction = parent.factor_parent._factor_site(
                event.state, event.first_value, event.block.attn, event.site, event.tokens,
            )
            max_reconstruction = max(max_reconstruction, reconstruction)
            audit["replayed_attention"] += 1
            if arm != "replay":
                early, late = parent.PAIR
                early_site = parent.factor_parent.TERMS[early][1]
                late_site = parent.factor_parent.TERMS[late][1]
                if event.site == early_site:
                    cached_early.update(factors[early])
                    write = write - factors[early]["native_term"]
                if event.site == late_site:
                    if not cached_early:
                        raise RuntimeError("early factors missing before layer8")
                    if arm != "reference":
                        write = write - factors[late]["native_term"]
            next_value = event.first_value
        else:
            write, next_value = event.block.attn(event.state, event.first_value)
            audit["native_attention"] += 1
        write = patch_and_capture(f"a{event.site}", write)
        return write, next_value

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


def recovery_report(base, reference, other, counts, start=0, stop=DOCUMENTS):
    stakes = parent.effect_report(base, reference, counts, start, stop)
    effects = parent.effect_report(base, other, counts, start, stop)
    recovery = {}
    for cell in CELLS:
        stake = stakes[cell]["effect_nat"]
        recovery[cell] = effects[cell]["effect_nat"] / stake if stake > 0 else None
    return {"native_stakes": stakes, "effects": effects, "recovery": recovery}


def analyze(base_ref, direct, mlp, attention, suffix, counts):
    base = base_ref[BASE_ARMS.index("base")]
    reference = base_ref[BASE_ARMS.index("reference")]
    pooled = {
        "direct_only": recovery_report(base, reference, direct, counts),
        "mediated_mlp": recovery_report(base, reference, mlp, counts),
        "mediated_attention": recovery_report(base, reference, attention, counts),
    }
    suffix_reports = [
        recovery_report(base, reference, suffix[i], counts) for i in range(len(CANDIDATES))
    ]
    pooled["mediated_all"] = suffix_reports[0]
    halves = []
    for start in (0, 96):
        half = {
            "direct_only": recovery_report(base, reference, direct, counts, start, start + 96),
            "mediated_mlp": recovery_report(base, reference, mlp, counts, start, start + 96),
            "mediated_attention": recovery_report(
                base, reference, attention, counts, start, start + 96,
            ),
            "suffix": [
                recovery_report(base, reference, suffix[i], counts, start, start + 96)
                for i in range(len(CANDIDATES))
            ],
        }
        half["mediated_all"] = half["suffix"][0]
        halves.append(half)
    patched_counts = torch.tensor(
        [len(CANDIDATES) - i for i in range(len(CANDIDATES))], dtype=torch.float64,
    )
    suffix_recoveries = torch.tensor([
        row["recovery"]["all_positive"] for row in suffix_reports
    ], dtype=torch.float64)
    suffix_spearman = parent.stage1.spearman(patched_counts, suffix_recoveries)
    half_spearman = []
    for half in halves:
        values = torch.tensor([
            row["recovery"]["all_positive"] for row in half["suffix"]
        ], dtype=torch.float64)
        half_spearman.append(parent.stage1.spearman(patched_counts, values))
    ascending = list(reversed([
        {"boundary": CANDIDATES[i], "patched_write_count": len(CANDIDATES) - i,
         "all_positive_recovery": suffix_reports[i]["recovery"]["all_positive"]}
        for i in range(len(CANDIDATES))
    ]))
    increments = [
        {
            "from_count": ascending[i - 1]["patched_write_count"],
            "to_count": ascending[i]["patched_write_count"],
            "recovery_increment": (
                ascending[i]["all_positive_recovery"]
                - ascending[i - 1]["all_positive_recovery"]
            ),
        }
        for i in range(1, len(ascending))
    ]
    interaction = {}
    stakes = pooled["direct_only"]["native_stakes"]
    for cell in CELLS:
        interaction[cell] = {
            "full_effect_nat": stakes[cell]["effect_nat"],
            "direct_effect_nat": pooled["direct_only"]["effects"][cell]["effect_nat"],
            "mediated_effect_nat": pooled["mediated_all"]["effects"][cell]["effect_nat"],
            "full_minus_direct_minus_mediated_nat": (
                stakes[cell]["effect_nat"]
                - pooled["direct_only"]["effects"][cell]["effect_nat"]
                - pooled["mediated_all"]["effects"][cell]["effect_nat"]
            ),
        }
    direct_recovery = pooled["direct_only"]["recovery"]["all_positive"]
    mediated_recovery = pooled["mediated_all"]["recovery"]["all_positive"]
    direct_order = parent.context_order(pooled["direct_only"]["effects"])
    mediated_order = parent.context_order(pooled["mediated_all"]["effects"])
    pred_b = bool(
        direct_recovery is not None and direct_recovery >= .50
        and all(half["direct_only"]["recovery"]["all_positive"] is not None
                and half["direct_only"]["recovery"]["all_positive"] > 0 for half in halves)
        and all(direct_order.values())
        and all(all(parent.context_order(half["direct_only"]["effects"]).values())
                for half in halves)
        and abs(pooled["direct_only"]["effects"]["off_target"]["effect_nat"]) <= .01
    )
    pred_c = bool(
        mediated_recovery is not None and mediated_recovery >= .15
        and all(half["mediated_all"]["recovery"]["all_positive"] is not None
                and half["mediated_all"]["recovery"]["all_positive"] > 0 for half in halves)
        and suffix_spearman >= .70 and all(value > 0 for value in half_spearman)
        and mediated_recovery - BEST_SINGLE_RECOVERY >= .05
    )
    mlp_recovery = pooled["mediated_mlp"]["recovery"]["all_positive"]
    mlp_effect = pooled["mediated_mlp"]["effects"]["all_positive"]["effect_nat"]
    attention_effect = pooled["mediated_attention"]["effects"]["all_positive"]["effect_nat"]
    pred_d = bool(
        mlp_recovery is not None and mlp_recovery >= .10 and mlp_effect > attention_effect
        and all(
            half["mediated_mlp"]["effects"]["all_positive"]["effect_nat"]
            > half["mediated_attention"]["effects"]["all_positive"]["effect_nat"]
            for half in halves
        )
    )
    dominant = "direct_only" if (
        pooled["direct_only"]["effects"]["all_positive"]["effect_nat"]
        >= pooled["mediated_all"]["effects"]["all_positive"]["effect_nat"]
    ) else "mediated_all"
    dominant_order = parent.context_order(pooled[dominant]["effects"])
    pred_e = bool(
        all(dominant_order.values())
        and all(all(parent.context_order(half[dominant]["effects"]).values())
                for half in halves)
        and abs(pooled[dominant]["effects"]["off_target"]["effect_nat"]) <= .01
    )
    primary_stakes_positive = all(
        stakes[cell]["effect_nat"] > 0 for cell in PRIMARY_CELLS
    )
    strong_science_null = bool(
        not primary_stakes_positive
        or max(direct_recovery or -math.inf, mediated_recovery or -math.inf) <= .10
        or not (all(direct_order.values()) or all(mediated_order.values()))
    )
    route_classification = (
        "both_direct_and_distributed"
        if pred_b and pred_c else
        "mainly_direct_residual" if pred_b else
        "mainly_distributed_suffix" if pred_c else
        "registered_route_predictions_unresolved"
    )
    return {
        "pooled": pooled,
        "halves": halves,
        "suffix_curve": {
            "boundaries_in_execution_order": [
                {"boundary": CANDIDATES[i], "patched_write_count": len(CANDIDATES) - i,
                 "all_positive_recovery": suffix_reports[i]["recovery"]["all_positive"]}
                for i in range(len(CANDIDATES))
            ],
            "patched_count_vs_recovery_spearman": suffix_spearman,
            "half_spearman": half_spearman,
            "ascending_adjacent_increments": increments,
        },
        "direct_distributed_interaction": interaction,
        "dominant_isolated_route": dominant,
        "route_classification": route_classification,
        "pred_b_direct_route": pred_b,
        "pred_c_cumulative_suffix": pred_c,
        "pred_d_mlp_over_attention": pred_d,
        "pred_e_dominant_context_law": pred_e,
        "strong_science_null": strong_science_null,
    }


@torch.no_grad()
def collect(model, payload, masks, scale):
    rows = payload["rows"]
    base_ref = torch.zeros(len(BASE_ARMS), DOCUMENTS, len(CELLS), dtype=torch.float64)
    direct = torch.zeros(DOCUMENTS, len(CELLS), dtype=torch.float64)
    mlp = torch.zeros_like(direct)
    attention = torch.zeros_like(direct)
    suffix = torch.zeros(len(CANDIDATES), DOCUMENTS, len(CELLS), dtype=torch.float64)
    counts = torch.zeros(DOCUMENTS, len(CELLS), dtype=torch.float64)
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    reconstruction = 0.0
    device = next(model.parameters()).device
    for start in range(0, DOCUMENTS, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = run_forward(model, tokens, arm="native")
        parent._record_audit(audit_totals, "path:native", audit,
                             analytical=False, captures=0, patches=0)
        replay_logits, _, audit, error = run_forward(model, tokens, arm="replay")
        parent._record_audit(audit_totals, "path:replay", audit,
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
        for ai, arm in enumerate(BASE_ARMS):
            logits, arm_captures, audit, error = run_forward(
                model, tokens, arm=arm, scale=scale, capture_keys=CANDIDATES,
            )
            parent._record_audit(audit_totals, f"path:{arm}", audit,
                                 analytical=True, captures=len(CANDIDATES), patches=0)
            reconstruction = max(reconstruction, error)
            sums, observed = parent._ce_sums(logits, batch_rows, masks, start)
            base_ref[ai, start:start + BATCH] = sums
            if ai == 0:
                counts[start:start + BATCH] = observed
            elif not torch.equal(observed, counts[start:start + BATCH]):
                raise RuntimeError("path supports changed across base/reference")
            captures[arm] = arm_captures
            del logits
        fixed_arms = (
            ("direct_only", "reference", {key: captures["base"][key] for key in CANDIDATES}),
            ("mediated_mlp", "base", {key: captures["reference"][key] for key in MLP_KEYS}),
            ("mediated_attention", "base", {
                key: captures["reference"][key] for key in ATTENTION_KEYS
            }),
        )
        fixed_outputs = (direct, mlp, attention)
        for (label, arm, patches), destination in zip(fixed_arms, fixed_outputs):
            logits, _, audit, error = run_forward(
                model, tokens, arm=arm, scale=scale, patch_writes=patches,
            )
            parent._record_audit(audit_totals, f"path:{label}", audit,
                                 analytical=True, captures=0, patches=len(patches))
            reconstruction = max(reconstruction, error)
            sums, observed = parent._ce_sums(logits, batch_rows, masks, start)
            if not torch.equal(observed, counts[start:start + BATCH]):
                raise RuntimeError(f"path support changed for {label}")
            destination[start:start + BATCH] = sums
            del logits
        for i, boundary in enumerate(CANDIDATES):
            patches = {
                key: captures["reference"][key] for key in CANDIDATES[i:]
            }
            logits, _, audit, error = run_forward(
                model, tokens, arm="base", scale=scale, patch_writes=patches,
            )
            parent._record_audit(audit_totals, f"path:suffix:{boundary}", audit,
                                 analytical=True, captures=0, patches=len(patches))
            reconstruction = max(reconstruction, error)
            sums, observed = parent._ce_sums(logits, batch_rows, masks, start)
            if not torch.equal(observed, counts[start:start + BATCH]):
                raise RuntimeError(f"path suffix support changed at {boundary}")
            suffix[i, start:start + BATCH] = sums
            del logits
        del captures
    return base_ref, direct, mlp, attention, suffix, counts, audit_totals, replay, reconstruction


def main():
    started = time.time()
    payload, masks, scale, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 463, "model_loaded": False,
            "new_path_outcomes_opened": False, "sealed_opened": False,
            "pair": PAIR_NAME, "write_order": CANDIDATES,
            "mlp_keys": MLP_KEYS, "attention_keys": ATTENTION_KEYS,
            "expected_forwards": EXPECTED_FORWARDS,
            "natural_fit_score_ratio_hash_bound_but_not_used_to_choose_path": scale["score_ratio"],
            "input_metadata": metadata,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung463 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    base_ref, direct, mlp, attention, suffix, counts, audit, replay, reconstruction = collect(
        model, payload, masks, scale,
    )
    analysis = analyze(base_ref, direct, mlp, attention, suffix, counts)
    forwards = sum(row["forwards"] for row in audit.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12
        and reconstruction <= 1e-10
        and forwards == EXPECTED_FORWARDS
    )
    pred_b = analysis["pred_b_direct_route"]
    pred_c = analysis["pred_c_cumulative_suffix"]
    pred_d = analysis["pred_d_mlp_over_attention"]
    pred_e = analysis["pred_e_dominant_context_law"]
    strong_null = bool(not pred_a or analysis["strong_science_null"])
    bundle = {
        "schema": "equality_score_path_factorial_rung463_sufficient_statistics_v1",
        "base_reference_loss_sums": base_ref,
        "direct_only_loss_sums": direct,
        "mediated_mlp_loss_sums": mlp,
        "mediated_attention_loss_sums": attention,
        "suffix_loss_sums": suffix,
        "counts": counts,
        "raw_rows_tokens_logits_or_hidden_states_included": False,
        "sealed_attention0_opened": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 463,
        "claim_level": "already_open_code_path_factorial_not_ood_confirmation",
        "input_identity": metadata,
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "frozen_pair": PAIR_NAME,
        "analysis": analysis,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "audit_totals": audit,
        "sufficient_statistics": {
            "path": str(BUNDLE), "sha256": sha256(BUNDLE), "bytes": BUNDLE.stat().st_size,
        },
        "execution_price": {
            "outer_forwards": forwards,
            "suffix_boundaries": len(CANDIDATES),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_direct_route': pred_b,
        'pred_c_cumulative_suffix': pred_c,
        'pred_d_mlp_over_attention': pred_d,
        'pred_e_dominant_context_law': pred_e,
        "strong_null": strong_null,
        "runtime_s": time.time() - started,
        "next_step": (
            "heldout_targeted_path_removal_and_interchange"
            if pred_a and pred_e and (pred_b or pred_c) and not strong_null
            else "audit_normalized_state_or_hidden_channels_before_branch_split"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 463,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "analysis": analysis,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "execution_price": result["execution_price"],
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
