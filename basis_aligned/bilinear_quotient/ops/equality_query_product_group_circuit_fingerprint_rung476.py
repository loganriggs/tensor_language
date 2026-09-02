#!/usr/bin/env python3
"""RUNG476 -- downstream fingerprints of frozen within-MLP product groups.

Registered before opening product-group census outcomes:
  pred_a: exact replay, partition, parent, count, and frozen-input checks.
  pred_b: selected pieces group more strongly than their complete parent MLPs.
  pred_c: the same selected grouping is stable across document halves.
  pred_d: task-selected pieces beat matched amplitude and random controls.
  pred_e: selected pieces separate from their complements and are circuit-selective.
Strong null: invalid instrument, no parent improvement, or no control advantage.
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
import equality_query_circuit_fingerprint_rung475 as parent
import equality_query_position_intervention_rung472 as position_parent
import equality_mlp_product_term_group_rung467 as product_parent
import equality_score_correction_interchange_rung464 as source_parent
import equality_score_downstream_gate_rung462 as audit_parent


PREREG = POLY / "EQUALITY_QUERY_PRODUCT_GROUP_CIRCUIT_FINGERPRINT_RUNG476_PREREGISTRATION.md"
GROUP_RESULT = ROOT / "equality_mlp_product_term_group_rung467_results.json"
PARENT_RESULT = ROOT / "equality_query_circuit_fingerprint_rung475_results.json"
PARENT_BUNDLE = ROOT / "equality_query_circuit_fingerprint_rung475_per_position.pt"
PARENT_SOURCE = ROOT / "ops/equality_query_circuit_fingerprint_rung475.py"
POSITION_SOURCE = ROOT / "ops/equality_query_position_intervention_rung472.py"
PRODUCT_SOURCE = ROOT / "ops/equality_mlp_product_term_group_rung467.py"
OUT = ROOT / "equality_query_product_group_circuit_fingerprint_rung476_results.json"
BUNDLE = ROOT / "equality_query_product_group_circuit_fingerprint_rung476_per_position.pt"
SOURCES = parent.SOURCES
SITES = parent.SITES
MODULES = parent.MODULES
PAIRS = parent.PAIRS
PAIR_NAMES = parent.PAIR_NAMES
KINDS = ("selected", "complement", "amplitude", "random")
HIDDEN = product_parent.HIDDEN
BATCH = parent.BATCH
DOCUMENTS = parent.DOCUMENTS
TOKENS = parent.TOKENS
FORWARDS_PER_BATCH = 3 + len(SOURCES) * (2 + len(KINDS) * len(SITES) + len(SITES))
PATCH_CALLS_PER_BATCH = len(SOURCES) * (len(SITES) + len(KINDS) * len(SITES) + len(SITES))
EXPECTED_FORWARDS = DOCUMENTS // BATCH * FORWARDS_PER_BATCH
EXPECTED_PATCH_CALLS = DOCUMENTS // BATCH * PATCH_CALLS_PER_BATCH
HASHES = {
    PREREG: "e7051629538729e6d3f8fdf184a7ae320924082d5411b4dc44507f8d4ec5f83b",
    GROUP_RESULT: "cc0480fc260c81b0fe512ec694413178de181b767f1dbfec43c56804b1ee5015",
    PARENT_RESULT: "226c611ae2df2ca666ca06a866446165fac7ba322017185376217544ebfc42ed",
    PARENT_BUNDLE: "8b43c479d9748e92cb73cde96ef3100c83edbb91dfddbd3f64e8bf4155f9e36b",
    PARENT_SOURCE: "e2885c8de6d893d763bdc06f6ddcfc67bfa374a70e0a23f13087a09ab2a01edf",
    POSITION_SOURCE: "d154c181ecba5e75982d21a1155563102da51e582a0982cb244d2120005d5c77",
    PRODUCT_SOURCE: "3665fc1b33ebb7bff78f78a9548d75219a43e3a0593e79bed6075a42a821bc8b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_groups(selection):
    groups = {kind: {} for kind in KINDS}
    for site in SITES:
        row = selection[site]
        selected = sorted(int(index) for index in row["selected_indices"])
        amplitude = sorted(int(index) for index in row["amplitude_control_indices"])
        random = sorted(int(index) for index in row["random_control_indices"])
        if len(selected) != len(set(selected)) or not selected \
                or selected[0] < 0 or selected[-1] >= HIDDEN:
            raise RuntimeError(f"malformed selected indices at {site}")
        if len(amplitude) != len(selected) or len(random) != len(selected):
            raise RuntimeError(f"matched control count changed at {site}")
        selected_set = set(selected)
        complement = [index for index in range(HIDDEN) if index not in selected_set]
        if selected_set & set(complement) or sorted(selected + complement) != list(range(HIDDEN)):
            raise RuntimeError(f"selected/complement partition failed at {site}")
        groups["selected"][site] = selected
        groups["complement"][site] = complement
        groups["amplitude"][site] = amplitude
        groups["random"][site] = random
    return groups


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    group_result = json.loads(GROUP_RESULT.read_text())
    if group_result.get("rung") != 467 or not all(group_result.get(key) is True for key in (
        "pred_a_instrument", "pred_b_stable_split", "pred_c_exact_heldout_correction",
        "pred_d_beats_matched_controls", "pred_e_cross_module_composition",
    )) or group_result.get("strong_null") is not False:
        raise RuntimeError("rung467 registered verdict changed")
    parent_result = json.loads(PARENT_RESULT.read_text())
    if parent_result.get("rung") != 475 or parent_result.get("pred_a_instrument") is not True \
            or any(parent_result.get(key) is not False for key in (
                "pred_b_downstream_pair", "pred_c_half_stable",
                "pred_d_behaviorally_selective", "pred_e_distinct_third",
            )) or parent_result.get("strong_null") is not False:
        raise RuntimeError("rung475 registered verdict changed")
    rows, base_ce, positive, circuit_masks, scale, metadata = parent.validate_inputs()
    parent_bundle = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=False)
    parent_effects = parent_bundle.get("effects")
    if list(parent_effects.shape) != [len(SOURCES), 4, DOCUMENTS, TOKENS] \
            or not torch.equal(parent_bundle.get("positive_mask"), positive):
        raise RuntimeError("rung475 per-position bundle changed")
    groups = make_groups(group_result["selection"])
    metadata = {
        **metadata,
        "rung467_result_sha256": sha256(GROUP_RESULT),
        "rung475_result_sha256": sha256(PARENT_RESULT),
        "rung475_bundle_sha256": sha256(PARENT_BUNDLE),
        "group_counts": {kind: {site: len(groups[kind][site]) for site in SITES}
                         for kind in KINDS},
    }
    return rows, base_ce, positive, circuit_masks, scale, groups, parent_effects, metadata


def run_group_position_patch(model, tokens, *, arm, scale, baselines, groups, position_mask):
    if set(baselines) != set(groups) or set(groups) - set(SITES):
        raise ValueError("malformed group position patch")
    handles, calls = [], {site: 0 for site in groups}
    for layer, site in zip(MODULES, SITES):
        if site not in groups:
            continue
        replacement = baselines[site]
        indices = torch.as_tensor(groups[site], dtype=torch.long, device=tokens.device)
        down = model.transformer.h[layer].mlp.Down

        def hook(_module, inputs, name=site, baseline=replacement, index=indices):
            if calls[name] != 0:
                raise RuntimeError(f"duplicate group patch at {name}")
            product = inputs[0]
            if baseline.shape != product.shape or baseline.dtype != product.dtype \
                    or baseline.device != product.device:
                raise RuntimeError(f"baseline product mismatch at {name}")
            chosen = product[position_mask].clone()
            replacement_chosen = baseline[position_mask]
            chosen[:, index] = replacement_chosen[:, index]
            updated = product.clone()
            updated[position_mask] = chosen
            calls[name] += 1
            return (updated,)

        handles.append(down.register_forward_pre_hook(hook))
    try:
        logits, _, audit, error = source_parent.run_forward(model, tokens, arm=arm, scale=scale)
    finally:
        for handle in handles:
            handle.remove()
    if any(value != 1 for value in calls.values()):
        raise RuntimeError("not every group patch fired exactly once")
    return logits, calls, audit, error


def _record(audit_totals, key, audit, patch_calls=0):
    row = audit_totals.setdefault(key, {"forwards": 0, "position_patch_calls": 0})
    row["forwards"] += 1
    row["position_patch_calls"] += patch_calls


@torch.no_grad()
def collect_effects(model, rows, positive, scale, groups, parent_effects, audit_totals, replay):
    effects = torch.zeros(len(SOURCES), len(KINDS), len(SITES), DOCUMENTS, TOKENS)
    device = next(model.parameters()).device
    reconstruction, empty_error, partition_error, patch_calls = 0.0, 0.0, 0.0, 0
    for start in range(0, DOCUMENTS, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = source_parent.run_forward(model, tokens, arm="native")
        audit_parent._record_audit(
            audit_totals, "rung476:native", audit, analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = source_parent.run_forward(model, tokens, arm="replay")
        audit_parent._record_audit(
            audit_totals, "rung476:replay", audit, analytical=True, captures=0, patches=0,
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
        _record(audit_totals, "rung476:absent", audit)
        reconstruction = max(reconstruction, error)
        position_mask = positive[start:start + BATCH].to(device)
        for si, source in enumerate(SOURCES):
            arm = source_parent.SOURCE_ARMS[source]
            source_logits, _, _, audit, error = product_parent.run_term_forward(
                model, tokens, arm=arm, scale=scale,
            )
            _record(audit_totals, f"rung476:source:{source}", audit)
            reconstruction = max(reconstruction, error)
            source_nll = position_parent._nll(source_logits, batch_rows)
            false_mask = torch.zeros_like(position_mask)
            empty_logits, calls, audit, error = run_group_position_patch(
                model, tokens, arm=arm, scale=scale,
                baselines={site: absent_products[site] for site in SITES},
                groups={site: groups["selected"][site] for site in SITES},
                position_mask=false_mask,
            )
            count = sum(calls.values())
            _record(audit_totals, f"rung476:empty:{source}", audit, count)
            patch_calls += count
            reconstruction = max(reconstruction, error)
            empty_error = max(empty_error, float((empty_logits - source_logits).abs().max()))
            for ki, kind in enumerate(KINDS):
                for mi, site in enumerate(SITES):
                    patched, calls, audit, error = run_group_position_patch(
                        model, tokens, arm=arm, scale=scale,
                        baselines={site: absent_products[site]},
                        groups={site: groups[kind][site]}, position_mask=position_mask,
                    )
                    count = sum(calls.values())
                    _record(audit_totals, f"rung476:{source}:{kind}:{site}", audit, count)
                    patch_calls += count
                    reconstruction = max(reconstruction, error)
                    effects[si, ki, mi, start:start + len(batch_rows)] = (
                        position_parent._nll(patched, batch_rows) - source_nll
                    ).cpu()
            for mi, site in enumerate(SITES):
                partition, calls, audit, error = run_group_position_patch(
                    model, tokens, arm=arm, scale=scale,
                    baselines={site: absent_products[site]}, groups={site: range(HIDDEN)},
                    position_mask=position_mask,
                )
                count = sum(calls.values())
                _record(audit_totals, f"rung476:{source}:partition:{site}", audit, count)
                patch_calls += count
                reconstruction = max(reconstruction, error)
                partition_effect = (position_parent._nll(partition, batch_rows) - source_nll).cpu()
                partition_error = max(
                    partition_error,
                    float((partition_effect - parent_effects[si, mi, start:start + len(batch_rows)]).abs().max()),
                )
        del absent_products
    return effects, reconstruction, empty_error, partition_error, patch_calls


def _fingerprints(effects, base_ce, positive, circuit_masks):
    banks, details, halves = {}, {}, {}
    for si, source in enumerate(SOURCES):
        banks[source], details[source], halves[source] = {}, {}, {}
        for ki, kind in enumerate(KINDS):
            banks[source][kind], details[source][kind], halves[source][kind] = {}, {}, {}
            for mi, site in enumerate(SITES):
                effect = effects[si, ki, mi].flatten()
                residual, coefficients = parent.residualize_difficulty(effect, base_ce, positive)
                raw, detail = parent.build_fingerprint(effect, circuit_masks, (0, DOCUMENTS))
                residual_fp, residual_detail = parent.build_fingerprint(
                    residual, circuit_masks, (0, DOCUMENTS),
                )
                banks[source][kind][site] = {"raw": raw, "residual": residual_fp}
                details[source][kind][site] = {
                    "difficulty_affine_coefficients": coefficients,
                    "raw": detail, "residual": residual_detail,
                    "raw_fingerprint": raw.tolist(),
                    "residual_fingerprint": residual_fp.tolist(),
                }
                halves[source][kind][site] = [
                    parent.build_fingerprint(effect, circuit_masks, window)[0]
                    for window in ((0, 500), (500, 1000))
                ]
    return banks, details, halves


def analyze(effects, parent_effects, base_ce, positive, circuit_masks):
    banks, details, halves = _fingerprints(effects, base_ce, positive, circuit_masks)
    parent_banks, _, _ = _fingerprints(
        parent_effects[:, :len(SITES)].unsqueeze(1).expand(-1, len(KINDS), -1, -1, -1),
        base_ce, positive, circuit_masks,
    )
    comparisons, best = {}, []
    for source in SOURCES:
        comparisons[source] = {}
        for kind in KINDS:
            comparisons[source][kind] = {}
            for view in ("raw", "residual"):
                rows = []
                for pi, (left, right) in enumerate(PAIRS):
                    cosine = parent._cosine(
                        banks[source][kind][SITES[left]][view],
                        banks[source][kind][SITES[right]][view],
                    )
                    parent_cosine = parent._cosine(
                        parent_banks[source][kind][SITES[left]][view],
                        parent_banks[source][kind][SITES[right]][view],
                    )
                    rows.append({"pair": PAIR_NAMES[pi], "pair_index": pi, "cosine": cosine,
                                 "complete_parent_cosine": parent_cosine,
                                 "improvement_over_parent": cosine - parent_cosine})
                rows.sort(key=lambda row: row["cosine"], reverse=True)
                comparisons[source][kind][view] = rows
                if kind == "selected":
                    best.append(rows[0]["pair_index"])
    proposed = best[0] if len(set(best)) == 1 else None
    pred_b = bool(proposed is not None and all(
        next(row for row in comparisons[source]["selected"][view]
             if row["pair_index"] == proposed)["cosine"] >= .80
        and next(row for row in comparisons[source]["selected"][view]
                 if row["pair_index"] == proposed)["improvement_over_parent"] >= .15
        for source in SOURCES for view in ("raw", "residual")
    ))
    half_rows, half_best = [], []
    for source in SOURCES:
        for half in range(2):
            rows = []
            for pi, (left, right) in enumerate(PAIRS):
                rows.append({"pair": PAIR_NAMES[pi], "pair_index": pi, "cosine": parent._cosine(
                    halves[source]["selected"][SITES[left]][half],
                    halves[source]["selected"][SITES[right]][half],
                )})
            rows.sort(key=lambda row: row["cosine"], reverse=True)
            half_rows.append({"source": source, "half": half, "pairs": rows})
            half_best.append(rows[0]["pair_index"])
    pred_c = bool(proposed is not None and all(index == proposed for index in half_best)
                  and all(row["pairs"][0]["cosine"] >= .70 for row in half_rows))
    control_margins = []
    complement_margins = []
    within_selected_complement = []
    selective_tags = []
    if proposed is not None:
        left, right = PAIRS[proposed]
        for source in SOURCES:
            for view in ("raw", "residual"):
                selected_cosine = parent._cosine(
                    banks[source]["selected"][SITES[left]][view],
                    banks[source]["selected"][SITES[right]][view],
                )
                for control in ("amplitude", "random"):
                    control_margins.append({"source": source, "view": view, "control": control,
                        "margin": selected_cosine - parent._cosine(
                            banks[source][control][SITES[left]][view],
                            banks[source][control][SITES[right]][view])})
                complement_margins.append({"source": source, "view": view,
                    "margin": selected_cosine - parent._cosine(
                        banks[source]["complement"][SITES[left]][view],
                        banks[source]["complement"][SITES[right]][view])})
            for index in (left, right):
                within_selected_complement.append({"source": source, "site": SITES[index],
                    "cosine": parent._cosine(
                        banks[source]["selected"][SITES[index]]["raw"],
                        banks[source]["complement"][SITES[index]]["raw"])})
        tags = list(circuit_masks)
        for tag in tags:
            values = []
            qualifies = True
            for source in SOURCES:
                for index in (left, right):
                    row = details[source]["selected"][SITES[index]]["raw"][tag]
                    qualifies &= row["member_offslice_ratio"] >= 2.0
                    values.append(row["signed_member_effect_nat"])
            qualifies &= all(value * values[0] > 0 for value in values[1:])
            if qualifies:
                selective_tags.append(tag)
    pred_d = bool(proposed is not None and control_margins
                  and min(row["margin"] for row in control_margins) >= .15)
    pred_e = bool(proposed is not None and complement_margins
                  and min(row["margin"] for row in complement_margins) >= .15
                  and max(row["cosine"] for row in within_selected_complement) <= .50
                  and len(selective_tags) >= 10)
    improves_parent = any(all(
        next(row for row in comparisons[source]["selected"]["residual"]
             if row["pair_index"] == pi)["improvement_over_parent"] >= .05
        for source in SOURCES) for pi in range(len(PAIRS)))
    beats_controls = any(all(
        next(row for row in comparisons[source]["selected"]["residual"]
             if row["pair_index"] == pi)["cosine"]
        > next(row for row in comparisons[source][control]["residual"]
               if row["pair_index"] == pi)["cosine"]
        for source in SOURCES for control in ("amplitude", "random"))
        for pi in range(len(PAIRS)))
    return {
        "details": details, "comparisons": comparisons,
        "selected_half_comparisons": half_rows,
        "proposed_pair": PAIR_NAMES[proposed] if proposed is not None else None,
        "control_margins": control_margins,
        "complement_margins": complement_margins,
        "within_selected_complement": within_selected_complement,
        "selective_circuits": selective_tags,
        "any_pair_improves_parent_residualized_both_sources": improves_parent,
        "any_pair_beats_both_controls_residualized_both_sources": beats_controls,
        "pred_b_selected_group": pred_b, "pred_c_half_stable": pred_c,
        "pred_d_beats_controls": pred_d, "pred_e_separates_complement": pred_e,
    }


def main():
    started = time.time()
    rows, base_ce, positive, circuit_masks, scale, groups, parent_effects, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 476, "model_loaded": False,
            "product_group_outcomes_opened": False, "sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "expected_patch_calls": EXPECTED_PATCH_CALLS,
            "group_counts": metadata["group_counts"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung476 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    effects, reconstruction, empty_error, partition_error, patch_calls = collect_effects(
        model, rows, positive, scale, groups, parent_effects, audit_totals, replay,
    )
    analysis = analyze(effects, parent_effects, base_ce, positive, circuit_masks)
    forwards = sum(row["forwards"] for row in audit_totals.values())
    observed_calls = sum(row.get("position_patch_calls", 0) for row in audit_totals.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and empty_error == 0 and partition_error <= 1e-6
        and forwards == EXPECTED_FORWARDS
        and observed_calls == EXPECTED_PATCH_CALLS and patch_calls == EXPECTED_PATCH_CALLS
    )
    strong_null = bool(
        not pred_a
        or not analysis["any_pair_improves_parent_residualized_both_sources"]
        or not analysis["any_pair_beats_both_controls_residualized_both_sources"]
    )
    torch.save({
        "schema": "rung476_product_group_circuit_fingerprints_v1",
        "effects": effects, "positive_mask": positive,
        "kinds": list(KINDS), "sites": list(SITES),
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": 476,
        "claim_level": "frozen_within_mlp_downstream_grouping_test",
        "input_identity": metadata,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                   "raw_tokens_logits_or_hidden_states_included": False},
        "analysis": analysis, "native_replay": replay,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "empty_query_mask_max_abs": empty_error,
        "selected_plus_complement_parent_max_abs_error_nat": partition_error,
        "audit_totals": audit_totals,
        "execution_price": {"outer_forwards": forwards, "position_patch_calls": observed_calls,
                            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                            "deployed_parameters_saved": 0, "deployed_parameters_added": 0},
        'pred_a_instrument': pred_a,
        'pred_b_selected_group': analysis["pred_b_selected_group"],
        'pred_c_half_stable': analysis["pred_c_half_stable"],
        'pred_d_beats_controls': analysis["pred_d_beats_controls"],
        'pred_e_separates_complement': analysis["pred_e_separates_complement"],
        "strong_null": strong_null, "runtime_s": time.time() - started,
        "next_step": ("circuit_family_heldout_interchange" if pred_a and all(
            analysis[key] for key in ("pred_b_selected_group", "pred_c_half_stable",
                                      "pred_d_beats_controls"))
            else "learn_partition_on_discovery_circuit_families"),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 476,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "summary": {"proposed_pair": analysis["proposed_pair"],
                    "comparisons": analysis["comparisons"],
                    "control_margins": analysis["control_margins"],
                    "complement_margins": analysis["complement_margins"],
                    "selective_circuit_count": len(analysis["selective_circuits"])},
        "instrument": {"replay": replay, "factor_error": reconstruction,
                       "empty_error": empty_error, "partition_error": partition_error,
                       "forwards": forwards, "patch_calls": observed_calls},
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
