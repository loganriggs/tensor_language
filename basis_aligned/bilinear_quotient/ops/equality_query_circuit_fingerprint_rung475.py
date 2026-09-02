#!/usr/bin/env python3
"""RUNG475 -- downstream 62-circuit fingerprints of equality-query MLP writes.

Registered before opening census query-product intervention outcomes:
  pred_a: frozen census/support and exact intervention instrument.
  pred_b: one MLP pair has a shared downstream behavioral fingerprint.
  pred_c: that pair is stable across fixed document halves.
  pred_d: similarity is selective across behavioral circuits.
  pred_e: the third MLP has a distinct downstream role.
Strong null: invalid, no pair survives difficulty control, or no selective circuit.
Literal deployed price: zero parameters saved and zero added.
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
import circuit_induction_tensor as induction
import equality_query_subtractive_factorial_rung474 as parent
import equality_query_mlp_factorial_rung473 as factorial_parent
import equality_query_position_intervention_rung472 as position_parent
import equality_mlp_product_term_group_rung467 as product_parent
import equality_score_correction_interchange_rung464 as source_parent
import equality_score_downstream_gate_rung462 as audit_parent
import equality_mlp_response_form_rung469 as form_parent


PREREG = POLY / "EQUALITY_QUERY_CIRCUIT_FINGERPRINT_RUNG475_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_query_subtractive_factorial_rung474_results.json"
PARENT_BUNDLE = ROOT / "equality_query_subtractive_factorial_rung474_per_token.pt"
PARENT_SOURCE = ROOT / "ops/equality_query_subtractive_factorial_rung474.py"
CENSUS = ROOT / "census_state_diverse.pt"
BATTERY = ROOT / "circuits/BATTERY.json"
OUT = ROOT / "equality_query_circuit_fingerprint_rung475_results.json"
BUNDLE = ROOT / "equality_query_circuit_fingerprint_rung475_per_position.pt"
SOURCES = parent.SOURCES
MODULES = parent.MODULES
SITES = parent.SITES
TARGETS = (*SITES, "union")
PAIRS = factorial_parent.PAIRS
PAIR_NAMES = factorial_parent.PAIR_NAMES
BATCH = 4
DOCUMENTS = 1000
TOKENS = 256
EXPECTED_POSITIVE = 101_052
EXPECTED_CIRCUITS = 62
FORWARDS_PER_BATCH = 3 + len(SOURCES) * (2 + len(TARGETS))
EXPECTED_BATCHES = DOCUMENTS // BATCH
EXPECTED_FORWARDS = EXPECTED_BATCHES * FORWARDS_PER_BATCH
EXPECTED_PATCH_CALLS_PER_BATCH = len(SOURCES) * (
    len(SITES) + sum(1 if target in SITES else len(SITES) for target in TARGETS)
)
EXPECTED_PATCH_CALLS = EXPECTED_BATCHES * EXPECTED_PATCH_CALLS_PER_BATCH
HASHES = {
    PREREG: "ec65c6a5718dc74f8461907f9c01b215ff206638359f641aa04c25bd3498b961",
    PARENT_RESULT: "17235cf0131d356332738dd6551df4ee60219836fbedbfd01ca27e9750998fb7",
    PARENT_BUNDLE: "c5d2b38a1631df1c4aacf3fd5bf583e91b6c71edaf2afd32881b576025e32647",
    PARENT_SOURCE: "3089bbb3703fa2d11b563d0ec04761f7c198422ba6fb695cbd477bd7c45cc13a",
    CENSUS: "c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b",
    BATTERY: "86d7ac72eeb95f9ec80a3e92ef65e28c0df66a36b9291d2d1d2d01f7bb6c5030",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask_from_indices(indices, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[torch.as_tensor(indices, dtype=torch.long)] = True
    return mask


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(PARENT_RESULT.read_text())
    expected = {
        "pred_a_instrument": True,
        "pred_b_coordinate_stable": False,
        "pred_c_state_mixing": False,
        "pred_d_register_persists": False,
        "pred_e_natural_h_half_stable": False,
        "strong_null": False,
    }
    if result.get("rung") != 474 or any(result.get(key) is not value for key, value in expected.items()):
        raise RuntimeError("rung474 registered verdict changed")
    _, scale, _, _, _, _, metadata = parent.validate_inputs()
    state = torch.load(CENSUS, map_location="cpu", weights_only=False)
    rows = state.get("rows")
    base_ce = state.get("basev")
    leaves = state.get("leaves")
    if list(rows.shape) != [DOCUMENTS, 513] or list(base_ce.shape) != [DOCUMENTS * TOKENS] \
            or not isinstance(leaves, list):
        raise RuntimeError("census state shape changed")
    battery = json.loads(BATTERY.read_text())
    tags = sorted(battery.get("by_tag", {}))
    if len(tags) != EXPECTED_CIRCUITS:
        raise RuntimeError("curated circuit count changed")
    by_tag = {leaf["tag"]: leaf for leaf in leaves}
    if any(tag not in by_tag for tag in tags):
        raise RuntimeError("battery tag absent from census leaves")
    tokens = rows[:, :TOKENS].contiguous()
    positive = induction.induction_fetch_mask(tokens).any(-1).cpu()
    if int(positive.sum()) != EXPECTED_POSITIVE:
        raise RuntimeError("equality-positive support changed")
    flat_positive = positive.flatten()
    circuit_masks = {}
    support = {}
    for tag in tags:
        leaf = by_tag[tag]
        member = _mask_from_indices(leaf["member"], DOCUMENTS * TOKENS)
        slice_mask = _mask_from_indices(leaf["slice"], DOCUMENTS * TOKENS)
        positive_members = member & flat_positive
        positive_offslice = ~slice_mask & flat_positive
        positive_in_slice_control = slice_mask & ~member & flat_positive
        count = int(positive_members.sum())
        if count < 100 or not bool(positive_offslice.any()):
            raise RuntimeError(f"insufficient frozen support for {tag}: {count}")
        circuit_masks[tag] = {
            "member": positive_members,
            "offslice": positive_offslice,
            "slice_control": positive_in_slice_control,
        }
        support[tag] = {
            "positive_members": count,
            "positive_offslice": int(positive_offslice.sum()),
            "positive_slice_control": int(positive_in_slice_control.sum()),
        }
    metadata = {
        **metadata,
        "rung474_result_sha256": sha256(PARENT_RESULT),
        "census_sha256": sha256(CENSUS),
        "battery_sha256": sha256(BATTERY),
        "documents": DOCUMENTS,
        "scored_positions": DOCUMENTS * TOKENS,
        "equality_positive_positions": int(positive.sum()),
        "row_halves": [[0, 500], [500, 1000]],
        "circuit_support": support,
    }
    return rows[:, :TOKENS + 1].contiguous(), base_ce.float(), positive, circuit_masks, scale, metadata


def _record(audit_totals, key, audit, patch_calls=0):
    row = audit_totals.setdefault(key, {"forwards": 0, "position_patch_calls": 0})
    row["forwards"] += 1
    row["position_patch_calls"] += patch_calls


@torch.no_grad()
def collect_effects(model, rows, positive, scale, audit_totals, replay):
    effects = torch.zeros(len(SOURCES), len(TARGETS), DOCUMENTS, TOKENS, dtype=torch.float32)
    device = next(model.parameters()).device
    reconstruction, empty_error, patch_calls = 0.0, 0.0, 0
    for start in range(0, DOCUMENTS, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = source_parent.run_forward(model, tokens, arm="native")
        audit_parent._record_audit(
            audit_totals, "rung475:native", audit, analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = source_parent.run_forward(model, tokens, arm="replay")
        audit_parent._record_audit(
            audit_totals, "rung475:replay", audit, analytical=True, captures=0, patches=0,
        )
        difference = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
        )
        reconstruction = max(reconstruction, error)
        _, absent_products, _, audit, error = product_parent.run_term_forward(
            model, tokens, arm="base", capture_products=True,
        )
        _record(audit_totals, "rung475:absent", audit)
        reconstruction = max(reconstruction, error)
        position_mask = positive[start:start + BATCH].to(device)
        for si, source in enumerate(SOURCES):
            arm = source_parent.SOURCE_ARMS[source]
            source_logits, _, _, audit, error = product_parent.run_term_forward(
                model, tokens, arm=arm, scale=scale,
            )
            _record(audit_totals, f"rung475:source:{source}", audit)
            reconstruction = max(reconstruction, error)
            source_nll = position_parent._nll(source_logits, batch_rows)
            false_mask = torch.zeros_like(position_mask)
            empty_logits, calls, audit, error = position_parent.run_position_patch(
                model, tokens, arm=arm, scale=scale,
                baselines={site: absent_products[site] for site in SITES},
                sites=SITES, position_mask=false_mask,
            )
            count = sum(calls.values())
            _record(audit_totals, f"rung475:empty:{source}", audit, count)
            patch_calls += count
            reconstruction = max(reconstruction, error)
            empty_error = max(empty_error, float((empty_logits - source_logits).abs().max()))
            for ti, target in enumerate(TARGETS):
                sites = (target,) if target in SITES else SITES
                patched, calls, audit, error = position_parent.run_position_patch(
                    model, tokens, arm=arm, scale=scale,
                    baselines={site: absent_products[site] for site in sites},
                    sites=sites, position_mask=position_mask,
                )
                count = sum(calls.values())
                _record(audit_totals, f"rung475:{source}:{target}", audit, count)
                patch_calls += count
                reconstruction = max(reconstruction, error)
                effects[si, ti, start:start + len(batch_rows)] = (
                    position_parent._nll(patched, batch_rows) - source_nll
                ).cpu()
        del absent_products
    return effects, reconstruction, empty_error, patch_calls


def residualize_difficulty(effect, base_ce, positive):
    effect = effect.double().flatten()
    base_ce = base_ce.double().flatten()
    mask = positive.flatten()
    x = base_ce[mask]
    y = effect[mask]
    design = torch.stack((torch.ones_like(x), x), dim=1)
    coefficients = torch.linalg.lstsq(design, y).solution
    prediction = coefficients[0] + coefficients[1] * base_ce
    return (effect - prediction).float(), coefficients.tolist()


def _cosine(left, right):
    return form_parent._cosine(left, right)


def build_fingerprint(effect, masks, row_range):
    start, stop = row_range
    half_mask = torch.zeros(DOCUMENTS, TOKENS, dtype=torch.bool)
    half_mask[start:stop] = True
    half_mask = half_mask.flatten()
    signed, detail = [], {}
    for tag, tag_masks in masks.items():
        member = tag_masks["member"] & half_mask
        offslice = tag_masks["offslice"] & half_mask
        if not bool(member.any()) or not bool(offslice.any()):
            raise RuntimeError(f"empty half support for {tag}")
        signed_member = float(effect[member].mean())
        abs_member = float(effect[member].abs().mean())
        abs_offslice = float(effect[offslice].abs().mean())
        signed.append(signed_member)
        detail[tag] = {
            "signed_member_effect_nat": signed_member,
            "absolute_member_effect_nat": abs_member,
            "absolute_offslice_effect_nat": abs_offslice,
            "member_offslice_ratio": abs_member / max(abs_offslice, 1e-30),
            "positive_members": int(member.sum()),
        }
    return torch.tensor(signed, dtype=torch.float64), detail


def analyze(effects, base_ce, positive, circuit_masks):
    tags = list(circuit_masks)
    reports = {}
    fingerprints = {source: {} for source in SOURCES}
    residual_fingerprints = {source: {} for source in SOURCES}
    halves = {source: {} for source in SOURCES}
    for si, source in enumerate(SOURCES):
        reports[source] = {}
        for ti, target in enumerate(TARGETS):
            effect = effects[si, ti].flatten()
            residual, coefficients = residualize_difficulty(effect, base_ce, positive)
            raw_fp, detail = build_fingerprint(effect, circuit_masks, (0, DOCUMENTS))
            residual_fp, residual_detail = build_fingerprint(
                residual, circuit_masks, (0, DOCUMENTS),
            )
            fingerprints[source][target] = raw_fp
            residual_fingerprints[source][target] = residual_fp
            halves[source][target] = [
                build_fingerprint(effect, circuit_masks, row_range)[0]
                for row_range in ((0, 500), (500, 1000))
            ]
            reports[source][target] = {
                "difficulty_affine_coefficients": coefficients,
                "signed_fingerprint": raw_fp.tolist(),
                "difficulty_residualized_fingerprint": residual_fp.tolist(),
                "circuit_detail": detail,
                "residualized_circuit_detail": residual_detail,
            }
    pair_comparisons = {}
    best_pairs = []
    margin_wins = 0
    for source in SOURCES:
        pair_comparisons[source] = {}
        for kind, bank in (
            ("raw", fingerprints[source]),
            ("difficulty_residualized", residual_fingerprints[source]),
        ):
            rows = []
            for pi, (left, right) in enumerate(PAIRS):
                rows.append({
                    "pair": PAIR_NAMES[pi],
                    "pair_index": pi,
                    "cosine": _cosine(bank[SITES[left]], bank[SITES[right]]),
                })
            rows.sort(key=lambda row: row["cosine"], reverse=True)
            pair_comparisons[source][kind] = rows
            best_pairs.append(rows[0]["pair_index"])
            margin_wins += int(rows[0]["cosine"] >= rows[1]["cosine"] + .10)
    same_pair = len(set(best_pairs)) == 1
    proposed_pair = best_pairs[0] if same_pair else None
    pred_b = bool(
        same_pair
        and all(pair_comparisons[source][kind][0]["cosine"] >= .80
                for source in SOURCES for kind in ("raw", "difficulty_residualized"))
        and margin_wins >= 3
    )
    half_comparisons = []
    half_best_pairs = []
    for source in SOURCES:
        for half in range(2):
            rows = []
            for pi, (left, right) in enumerate(PAIRS):
                rows.append({
                    "pair": PAIR_NAMES[pi], "pair_index": pi,
                    "cosine": _cosine(
                        halves[source][SITES[left]][half],
                        halves[source][SITES[right]][half],
                    ),
                })
            rows.sort(key=lambda row: row["cosine"], reverse=True)
            half_comparisons.append({"source": source, "half": half, "pairs": rows})
            half_best_pairs.append(rows[0]["pair_index"])
    pred_c = bool(
        proposed_pair is not None and all(index == proposed_pair for index in half_best_pairs)
        and all(row["pairs"][0]["cosine"] >= .70 for row in half_comparisons)
    )
    selective_tags = []
    distinct_prefixes = set()
    opposite_tags = []
    if proposed_pair is not None:
        left, right = PAIRS[proposed_pair]
        third = next(index for index in range(len(SITES)) if index not in (left, right))
        for tag_index, tag in enumerate(tags):
            selective = True
            pair_signs = []
            third_signs = []
            for source in SOURCES:
                for index in (left, right):
                    row = reports[source][SITES[index]]["circuit_detail"][tag]
                    selective &= row["member_offslice_ratio"] >= 2.0
                    pair_signs.append(row["signed_member_effect_nat"])
                third_signs.append(
                    reports[source][SITES[third]]["circuit_detail"][tag]["signed_member_effect_nat"]
                )
            same_pair_sign = all(value * pair_signs[0] > 0 for value in pair_signs[1:])
            if selective and same_pair_sign:
                selective_tags.append(tag)
                parts = tag.split(".")
                distinct_prefixes.add(".".join(parts[:2]))
            pair_means = [
                float((fingerprints[source][SITES[left]][tag_index]
                       + fingerprints[source][SITES[right]][tag_index]) / 2)
                for source in SOURCES
            ]
            if all(pair_means[i] * third_signs[i] < 0 for i in range(len(SOURCES))):
                opposite_tags.append(tag)
        proposed_cosines = [
            pair_comparisons[source][kind][0]["cosine"]
            for source in SOURCES for kind in ("raw", "difficulty_residualized")
        ]
        third_cosines = []
        for source in SOURCES:
            for kind in ("raw", "difficulty_residualized"):
                bank = fingerprints[source] if kind == "raw" else residual_fingerprints[source]
                third_cosines.extend([
                    _cosine(bank[SITES[left]], bank[SITES[third]]),
                    _cosine(bank[SITES[right]], bank[SITES[third]]),
                ])
        pred_e = bool(
            min(proposed_cosines) >= max(third_cosines) + .10
            and len(opposite_tags) >= 5
        )
    else:
        pred_e = False
    pred_d = bool(len(selective_tags) >= 10 and len(distinct_prefixes) >= 3)
    any_pair_survives = any(
        all(next(row["cosine"] for row in pair_comparisons[source][kind]
                 if row["pair_index"] == pi) > .50
            for source in SOURCES for kind in ("raw", "difficulty_residualized"))
        for pi in range(len(PAIRS))
    )
    selective_count_any_mlp = max(
        sum(all(reports[source][site]["circuit_detail"][tag]["member_offslice_ratio"] >= 2.0
                for source in SOURCES) for tag in tags)
        for site in SITES
    )
    return {
        "circuit_tags": tags,
        "reports": reports,
        "pair_comparisons": pair_comparisons,
        "best_pair_indices_source_raw_residual_order": best_pairs,
        "proposed_pair": PAIR_NAMES[proposed_pair] if proposed_pair is not None else None,
        "half_comparisons": half_comparisons,
        "selective_pair_circuits": selective_tags,
        "selective_pair_top_level_prefixes": sorted(distinct_prefixes),
        "opposite_pair_vs_third_circuits": opposite_tags,
        "any_pair_survives_difficulty_control": any_pair_survives,
        "max_selective_circuit_count_any_mlp": selective_count_any_mlp,
        "pred_b_downstream_pair": pred_b,
        "pred_c_half_stable": pred_c,
        "pred_d_behaviorally_selective": pred_d,
        "pred_e_distinct_third": pred_e,
    }


def main():
    started = time.time()
    rows, base_ce, positive, circuit_masks, scale, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 475, "model_loaded": False,
            "census_intervention_outcomes_opened": False, "sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "expected_patch_calls": EXPECTED_PATCH_CALLS,
            "equality_positive_positions": int(positive.sum()),
            "circuits": len(circuit_masks),
            "minimum_positive_members": min(
                int(row["member"].sum()) for row in circuit_masks.values()
            ),
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung475 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    effects, reconstruction, empty_error, patch_calls = collect_effects(
        model, rows, positive, scale, audit_totals, replay,
    )
    analysis = analyze(effects, base_ce, positive, circuit_masks)
    forwards = sum(row["forwards"] for row in audit_totals.values())
    observed_calls = sum(row.get("position_patch_calls", 0) for row in audit_totals.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and len(analysis["circuit_tags"]) == EXPECTED_CIRCUITS
        and int(positive.sum()) == EXPECTED_POSITIVE
        and min(int(row["member"].sum()) for row in circuit_masks.values()) >= 100
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and empty_error == 0 and forwards == EXPECTED_FORWARDS
        and observed_calls == EXPECTED_PATCH_CALLS and patch_calls == EXPECTED_PATCH_CALLS
    )
    strong_null = bool(
        not pred_a or not analysis["any_pair_survives_difficulty_control"]
        or analysis["max_selective_circuit_count_any_mlp"] < 5
    )
    torch.save({
        "schema": "rung475_equality_query_circuit_fingerprints_v1",
        "effects": effects,
        "positive_mask": positive,
        "circuit_tags": analysis["circuit_tags"],
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": 475,
        "claim_level": "downstream_behavioral_equivalence_screen",
        "input_identity": metadata,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                   "raw_tokens_logits_or_hidden_states_included": False},
        "analysis": analysis, "native_replay": replay,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "empty_query_mask_max_abs": empty_error,
        "audit_totals": audit_totals,
        "execution_price": {
            "outer_forwards": forwards, "position_patch_calls": observed_calls,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_downstream_pair': analysis["pred_b_downstream_pair"],
        'pred_c_half_stable': analysis["pred_c_half_stable"],
        'pred_d_behaviorally_selective': analysis["pred_d_behaviorally_selective"],
        'pred_e_distinct_third': analysis["pred_e_distinct_third"],
        "strong_null": strong_null,
        "runtime_s": time.time() - started,
        "next_step": (
            "circuit_family_heldout_physical_interchange"
            if pred_a and all(analysis[key] for key in (
                "pred_b_downstream_pair", "pred_c_half_stable",
                "pred_d_behaviorally_selective",
            )) else "within_mlp_downstream_response_split"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 475,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "summary": {
            "pair_comparisons": analysis["pair_comparisons"],
            "proposed_pair": analysis["proposed_pair"],
            "selective_pair_circuit_count": len(analysis["selective_pair_circuits"]),
            "opposite_pair_vs_third_count": len(analysis["opposite_pair_vs_third_circuits"]),
            "max_selective_circuit_count_any_mlp": analysis["max_selective_circuit_count_any_mlp"],
        },
        "instrument": {"replay": replay, "factor_error": reconstruction,
                       "empty_error": empty_error, "forwards": forwards,
                       "patch_calls": observed_calls},
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
