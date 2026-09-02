#!/usr/bin/env python3
"""RUNG477B -- split-aware repair of the rung477 discovery response tensor.

Registered after rung477 outcomes:
  pred_a: corrected half support and exact response instrument.
  pred_b: native product terms remain overwhelmingly half/source unstable.
  pred_c: the native-coordinate cross-MLP graph remains empty.
  pred_d: half-summed responses reproduce the original total response.
  pred_e: validation families and raw model data remain unopened.
The original rung477 A=false and strong null are never rescored.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
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
import equality_product_circuit_response_graph_rung477 as parent
import equality_mlp_product_term_group_rung467 as product_parent
import equality_score_correction_interchange_rung464 as source_parent
import equality_score_downstream_gate_rung462 as audit_parent


PREREG = POLY / "EQUALITY_PRODUCT_CIRCUIT_RESPONSE_GRAPH_RUNG477B_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_product_circuit_response_graph_rung477_results.json"
PARENT_BUNDLE = ROOT / "equality_product_circuit_response_graph_rung477_bundle.pt"
PARENT_SOURCE = ROOT / "ops/equality_product_circuit_response_graph_rung477.py"
OUT = ROOT / "equality_product_circuit_response_graph_rung477b_results.json"
BUNDLE = ROOT / "equality_product_circuit_response_graph_rung477b_bundle.pt"
SOURCES = parent.SOURCES
SITES = parent.SITES
MODULES = parent.MODULES
MASK_TYPES = parent.MASK_TYPES
DISCOVERY_STOP = parent.DISCOVERY_STOP
HALF_STOP = parent.HALF_STOP
BATCH = parent.BATCH
TOKENS = parent.TOKENS
DOCUMENTS = 1000
HIDDEN = parent.HIDDEN
EXPECTED_FORWARDS = parent.EXPECTED_FORWARDS
HALVES = ((0, HALF_STOP), (HALF_STOP, DISCOVERY_STOP))
HASHES = {
    PREREG: "77ae0b0f6ee424b7e2dd321123ea93660f38472bcc77d97e8e2a996f3cb0cafc",
    PARENT_RESULT: "c02cc199ec4571725961b4d9b7842fc50cd92713947f44af61f339c34c268f0b",
    PARENT_BUNDLE: "b02c8c77b9476eb876549b284dc522a32752b6571fdfc334867a2483bc2490d1",
    PARENT_SOURCE: "7c76b115977ab102884c2233e31a284e3d509ca2b3ed9291cefe1d47562aa770",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _half_batch_mask(flat_mask, start, stop, half_start, half_stop):
    selected = parent._batch_mask(flat_mask, start, stop).clone()
    rows = torch.arange(start, stop)
    selected &= ((rows >= half_start) & (rows < half_stop))[:, None]
    return selected


def expected_backwards(circuit_masks, tags):
    count = 0
    for start in range(0, DISCOVERY_STOP, BATCH):
        stop = min(start + BATCH, DISCOVERY_STOP)
        for half_start, half_stop in HALVES:
            for tag in tags:
                for mask_type in MASK_TYPES:
                    selected = _half_batch_mask(
                        circuit_masks[tag][mask_type], start, stop, half_start, half_stop,
                    )
                    count += int(bool(selected.any()))
    return count * len(SOURCES)


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    original = json.loads(PARENT_RESULT.read_text())
    if original.get("rung") != 477 or original.get("pred_a_instrument") is not False \
            or any(original.get(key) is not False for key in (
                "pred_b_stable_terms", "pred_c_response_graph",
                "pred_d_leave_family_stable", "pred_e_selective_aggregate",
            )) or original.get("strong_null") is not True:
        raise RuntimeError("rung477 registered receipt changed")
    rows, positive, circuit_masks, scale, tags, validation_tags, metadata = parent.validate_inputs()
    original_bundle = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=False)
    if original_bundle.get("schema") != "rung477_discovery_product_circuit_response_v1" \
            or original_bundle.get("validation_tags_or_responses_included") is not False:
        raise RuntimeError("rung477 bundle schema changed")
    backwards = expected_backwards(circuit_masks, tags)
    metadata = {
        **metadata, "rung477_result_sha256": sha256(PARENT_RESULT),
        "rung477_bundle_sha256": sha256(PARENT_BUNDLE),
        "repair": "split masks inside the single batch crossing row250",
        "expected_split_aware_backwards": backwards,
        "original_rung477_a_remains_false": True,
    }
    return rows, positive, circuit_masks, scale, tags, validation_tags, metadata, original_bundle


def collect_responses(model, rows, circuit_masks, scale, tags, audit_totals, replay):
    sums = torch.zeros(
        2, len(SOURCES), len(MASK_TYPES), len(SITES), HIDDEN, len(tags), dtype=torch.float64,
    )
    counts = torch.zeros(2, len(MASK_TYPES), len(tags), dtype=torch.float64)
    device = next(model.parameters()).device
    reconstruction, contraction_numerator, contraction_denominator, backwards = 0.0, 0.0, 0.0, 0
    for start in range(0, DISCOVERY_STOP, BATCH):
        stop = min(start + BATCH, DISCOVERY_STOP)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = source_parent.run_forward(model, tokens, arm="native")
        audit_parent._record_audit(
            audit_totals, "rung477b:native", audit, analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = source_parent.run_forward(model, tokens, arm="replay")
        audit_parent._record_audit(
            audit_totals, "rung477b:replay", audit, analytical=True, captures=0, patches=0,
        )
        difference = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
        )
        reconstruction = max(reconstruction, error)
        with torch.no_grad():
            _, absent_products, _, audit, error = product_parent.run_term_forward(
                model, tokens, arm="base", capture_products=True,
            )
        parent._record(audit_totals, "rung477b:absent", audit)
        reconstruction = max(reconstruction, error)
        active = []
        for hi, (half_start, half_stop) in enumerate(HALVES):
            for ci, tag in enumerate(tags):
                for ki, mask_type in enumerate(MASK_TYPES):
                    selected = _half_batch_mask(
                        circuit_masks[tag][mask_type], start, stop, half_start, half_stop,
                    ).to(device)
                    observed = int(selected.sum())
                    counts[hi, ki, ci] += observed
                    if observed:
                        active.append((hi, ki, ci, selected))
        for si, source in enumerate(SOURCES):
            with torch.enable_grad():
                logits, products, writes, audit, error = product_parent.run_term_forward(
                    model, tokens, arm=source_parent.SOURCE_ARMS[source], scale=scale,
                    capture_products=True, gradient_writes=True,
                )
                parent._record(audit_totals, f"rung477b:source:{source}", audit)
                reconstruction = max(reconstruction, error)
                nll = parent._nll(logits, batch_rows)
                for ai, (hi, ki, ci, selected) in enumerate(active):
                    gradients = torch.autograd.grad(
                        nll[selected].sum(), tuple(writes[site] for site in SITES),
                        retain_graph=ai + 1 < len(active), allow_unused=False,
                    )
                    backwards += 1
                    for mi, (site, gradient) in enumerate(zip(SITES, gradients)):
                        module = model.transformer.h[MODULES[mi]].mlp
                        delta = (products[site] - absent_products[site]).float()
                        reader = torch.matmul(gradient.float(), module.Down.weight.float())
                        term_response = -(reader * delta).sum((0, 1))
                        direct_delta = torch.matmul(delta, module.Down.weight.float().T)
                        direct_response = -(gradient.float() * direct_delta).sum()
                        mismatch = term_response.sum() - direct_response
                        contraction_numerator += float(mismatch.square())
                        contraction_denominator += float(direct_response.square())
                        sums[hi, si, ki, mi, :, ci] += term_response.double().cpu()
                del logits, products, writes, nll
        del absent_products
    contraction_error = contraction_numerator / max(contraction_denominator, 1e-30)
    return sums, counts, reconstruction, contraction_error, backwards


def main():
    started = time.time()
    rows, _, circuit_masks, scale, tags, validation_tags, metadata, original_bundle = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": "477b", "model_loaded": False,
            "response_outcomes_opened": False, "validation_family_responses_opened": False,
            "sealed_opened": False, "expected_forwards": EXPECTED_FORWARDS,
            "expected_backwards": metadata["expected_split_aware_backwards"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung477b output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    sums, counts, reconstruction, contraction_error, backwards = collect_responses(
        model, rows, circuit_masks, scale, tags, audit_totals, replay,
    )
    analysis = parent.analyze(sums, counts)
    forwards = sum(row["forwards"] for row in audit_totals.values())
    old_sums = original_bundle["response_sums"]
    old_counts = original_bundle["response_counts"]
    total_difference = sums.sum(0) - old_sums.sum(0)
    total_response_relative_squared = float(total_difference.square().sum() /
                                            old_sums.sum(0).square().sum().clamp_min(1e-30))
    total_counts_exact = bool(torch.equal(counts.sum(0), old_counts.sum(0)))
    member_min, control_min = int(counts[:, 0].min()), int(counts[:, 1].min())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and contraction_error <= 1e-8 and torch.isfinite(sums).all()
        and member_min >= 39 and control_min >= 439
        and forwards == EXPECTED_FORWARDS
        and backwards == metadata["expected_split_aware_backwards"]
    )
    pred_b = max(analysis["eligible_counts"].values()) <= 20
    pred_c = all(row["count"] == 0 for row in analysis["graphs"])
    pred_d = total_counts_exact and total_response_relative_squared <= 1e-10
    pred_e = bool(len(validation_tags) == 30)
    torch.save({
        "schema": "rung477b_split_aware_discovery_response_v1",
        "response_sums": sums, "response_counts": counts,
        "sources": list(SOURCES), "mask_types": list(MASK_TYPES),
        "sites": list(SITES), "discovery_tags": tags,
        "validation_tags_or_responses_included": False,
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": "477b",
        "claim_level": "data_integrity_repair_not_scientific_rescore",
        "input_identity": metadata,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "original_rung477_a_remains_false": True,
        "sealed_attention0_confirmation_opened": False,
        "validation_family_product_responses_opened": False,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                   "validation_tags_or_responses_included": False,
                   "raw_tokens_logits_or_hidden_states_included": False},
        "analysis": analysis, "native_replay": replay,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "term_vs_write_contraction_relative_squared": contraction_error,
        "member_support_min": member_min, "control_support_min": control_min,
        "half_summed_response_relative_squared_vs_rung477": total_response_relative_squared,
        "half_summed_counts_exact_vs_rung477": total_counts_exact,
        "audit_totals": audit_totals,
        "execution_price": {"outer_forwards": forwards, "backwards": backwards,
                            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                            "deployed_parameters_saved": 0, "deployed_parameters_added": 0},
        'pred_a_corrected_instrument': pred_a,
        'pred_b_native_instability_robust': pred_b,
        'pred_c_graph_remains_empty': pred_c,
        'pred_d_only_half_allocation_changed': pred_d,
        'pred_e_reserved_outcomes_closed': pred_e,
        "strong_null_native_coordinate_basis_remains": True,
        "runtime_s": time.time() - started,
        "next_step": "sparse_mixed_product_response_directions" if pred_a and pred_d else "repair_data_integrity",
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": "477b",
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "native_coordinate_strong_null_remains": True,
        "analysis": {"eligible_counts": analysis["eligible_counts"],
                     "graphs": analysis["graphs"], "proposed_pair": analysis["proposed_pair"]},
        "instrument": {"replay": replay, "factor_error": reconstruction,
                       "contraction_error": contraction_error, "member_min": member_min,
                       "control_min": control_min, "forwards": forwards, "backwards": backwards,
                       "total_response_relative_squared": total_response_relative_squared,
                       "total_counts_exact": total_counts_exact},
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
