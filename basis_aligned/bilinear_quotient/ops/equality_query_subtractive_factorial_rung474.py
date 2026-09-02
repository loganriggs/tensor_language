#!/usr/bin/env python3
"""RUNG474 -- subtractive query-position three-MLP factorial.

Registered before opening subtractive multi-MLP outcomes:
  pred_a: exact/frozen alternate intervention and singleton equivalence.
  pred_b: all-three query effect is stable to causal coordinates.
  pred_c: fixed-baseline state mixing caused natural interaction instability.
  pred_d: register-conditioned composition persists under subtraction.
  pred_e: natural-hybrid half fragility disappears under subtraction.
Strong null: invalid, coordinate-level union disagreement, or no material interaction.
Literal deployed price: zero parameters saved and zero added.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import itertools
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
import equality_query_mlp_factorial_rung473 as parent
import equality_query_position_intervention_rung472 as position_parent
import equality_mlp_product_term_group_rung467 as product_parent
import equality_score_correction_interchange_rung464 as source_parent
import equality_score_downstream_gate_rung462 as audit_parent
import equality_mlp_response_form_rung469 as form_parent


PREREG = POLY / "EQUALITY_QUERY_SUBTRACTIVE_FACTORIAL_RUNG474_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_query_mlp_factorial_rung473_results.json"
PARENT_BUNDLE = ROOT / "equality_query_mlp_factorial_rung473_per_token.pt"
PARENT_SOURCE = ROOT / "ops/equality_query_mlp_factorial_rung473.py"
OUT = ROOT / "equality_query_subtractive_factorial_rung474_results.json"
BUNDLE = ROOT / "equality_query_subtractive_factorial_rung474_per_token.pt"
SOURCES = parent.SOURCES
MODULES = parent.MODULES
SITES = parent.SITES
WINDOWS = parent.WINDOWS
BATCH = parent.BATCH
SUBSETS = tuple(
    indices for size in range(1, len(SITES) + 1)
    for indices in itertools.combinations(range(len(SITES)), size)
)
SUBSET_NAMES = tuple("+".join(SITES[index] for index in indices) for indices in SUBSETS)
SINGLE_INDICES = tuple(SUBSETS.index((index,)) for index in range(len(SITES)))
PAIR_INDICES = tuple(SUBSETS.index(indices) for indices in parent.PAIRS)
UNION_INDEX = SUBSETS.index(tuple(range(len(SITES))))
FORWARDS_PER_BATCH = 3 + len(SOURCES) * (2 + 2 * len(SUBSETS))
EXPECTED_BATCHES = sum((stop - start) // BATCH for _, _, start, stop in WINDOWS)
EXPECTED_FORWARDS = EXPECTED_BATCHES * FORWARDS_PER_BATCH
EXPECTED_PATCH_CALLS_PER_BATCH = len(SOURCES) * (
    len(SITES) + 2 * sum(len(indices) for indices in SUBSETS)
)
EXPECTED_PATCH_CALLS = EXPECTED_BATCHES * EXPECTED_PATCH_CALLS_PER_BATCH
HASHES = {
    PREREG: "7af69ec71b98c098091b3356385f043559a439c4eb71df8d96ec38ccd8e9db92",
    PARENT_RESULT: "8929b3101fd2a2bf2856faf9ba2c8131c861366c2c3364c00ec408ec0fdf0f95",
    PARENT_BUNDLE: "9f816e001fe46892ca8e95c8173a05a2880d7893f86e61cdb5eb4bcf22332ae4",
    PARENT_SOURCE: "fd842ca53b15ffeb7b83f08abec830819fa13ee1f48f292e5971448236562617",
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
    expected = {
        "pred_a_instrument": True,
        "pred_b_stable_pair": False,
        "pred_c_register_source_dependent": True,
        "pred_d_pairwise_sufficient": False,
        "pred_e_half_stable": False,
        "strong_null": False,
    }
    if result.get("rung") != 473 or any(result.get(key) is not value for key, value in expected.items()):
        raise RuntimeError("rung473 registered verdict changed")
    roles, scale, old_effects, selections, old_position_bundle, metadata = parent.validate_inputs()
    old_factorial_bundle = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=True)
    if old_factorial_bundle.get("schema") != "rung473_exact_query_mlp_factorial_v1":
        raise RuntimeError("rung473 bundle schema changed")
    metadata = {
        **metadata,
        "rung473_result_sha256": sha256(PARENT_RESULT),
        "rung473_bundle_sha256": sha256(PARENT_BUNDLE),
        "subsets": list(SUBSET_NAMES),
        "coordinate": "subtract_frozen_intact_source_minus_absent_product_from_current_product",
    }
    return (
        roles, scale, old_effects, selections, old_position_bundle,
        old_factorial_bundle, metadata,
    )


@torch.no_grad()
def run_subtractive_patch(model, tokens, *, arm, scale, deltas, sites, position_mask):
    sites = tuple(sites)
    if set(sites) - set(SITES) or set(deltas) != set(sites):
        raise ValueError("malformed subtractive patch sites")
    handles, calls = [], {site: 0 for site in sites}
    for layer, site in zip(MODULES, SITES):
        if site not in sites:
            continue
        delta = deltas[site]
        down = model.transformer.h[layer].mlp.Down

        def hook(_module, inputs, name=site, frozen_delta=delta):
            if calls[name] != 0:
                raise RuntimeError(f"duplicate subtractive product patch at {name}")
            product = inputs[0]
            if frozen_delta.shape != product.shape or frozen_delta.device != product.device \
                    or frozen_delta.dtype != torch.float32:
                raise RuntimeError(f"subtractive delta mismatch at {name}")
            updated = product.clone()
            current = product[position_mask].float()
            updated[position_mask] = (current - frozen_delta[position_mask]).to(product.dtype)
            calls[name] += 1
            return (updated,)

        handles.append(down.register_forward_pre_hook(hook))
    try:
        logits, _, audit, error = source_parent.run_forward(
            model, tokens, arm=arm, scale=scale,
        )
    finally:
        for handle in handles:
            handle.remove()
    if any(value != 1 for value in calls.values()):
        raise RuntimeError("not every subtractive patch fired exactly once")
    return logits, calls, audit, error


def _record(audit_totals, key, audit, patch_calls=0):
    row = audit_totals.setdefault(key, {"forwards": 0, "subtractive_patch_calls": 0})
    row["forwards"] += 1
    row["subtractive_patch_calls"] += patch_calls


@torch.no_grad()
def collect_window(model, payload, scale, selection, audit_totals, replay):
    coordinates = selection["coordinates"]
    by_doc = {}
    for output_index, (doc, query, _) in enumerate(coordinates):
        by_doc.setdefault(doc, []).append((output_index, query))
    effects = torch.zeros(len(SOURCES), len(SUBSETS), len(coordinates), dtype=torch.float64)
    rows = payload["rows"]
    first_doc = min(by_doc)
    last_doc = max(by_doc) + 1
    device = next(model.parameters()).device
    reconstruction, empty_error, patch_calls = 0.0, 0.0, 0
    for start in range(first_doc, last_doc, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = source_parent.run_forward(model, tokens, arm="native")
        audit_parent._record_audit(
            audit_totals, "rung474:native", audit, analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = source_parent.run_forward(model, tokens, arm="replay")
        audit_parent._record_audit(
            audit_totals, "rung474:replay", audit, analytical=True, captures=0, patches=0,
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
        _record(audit_totals, "rung474:absent", audit)
        reconstruction = max(reconstruction, error)
        slots = []
        for slot in range(2):
            chosen = []
            for doc in range(start, min(start + BATCH, last_doc)):
                if len(by_doc.get(doc, [])) > slot:
                    output_index, query = by_doc[doc][slot]
                    chosen.append((output_index, doc - start, query))
            slots.append(chosen)
        for si, source in enumerate(SOURCES):
            arm = source_parent.SOURCE_ARMS[source]
            source_logits, source_products, _, audit, error = product_parent.run_term_forward(
                model, tokens, arm=arm, scale=scale, capture_products=True,
            )
            _record(audit_totals, f"rung474:source:{source}", audit)
            reconstruction = max(reconstruction, error)
            source_nll = position_parent._nll(source_logits, batch_rows)
            deltas = {
                site: source_products[site].float() - absent_products[site].float()
                for site in SITES
            }
            false_mask = torch.zeros_like(tokens, dtype=torch.bool)
            empty_logits, calls, audit, error = run_subtractive_patch(
                model, tokens, arm=arm, scale=scale, deltas=deltas,
                sites=SITES, position_mask=false_mask,
            )
            count = sum(calls.values())
            _record(audit_totals, f"rung474:empty:{source}", audit, count)
            patch_calls += count
            reconstruction = max(reconstruction, error)
            empty_error = max(empty_error, float((empty_logits - source_logits).abs().max()))
            for slot, chosen in enumerate(slots):
                targets = [(local_doc, query) for _, local_doc, query in chosen]
                query_mask, _, _, _ = position_parent.position_masks(
                    len(batch_rows), tokens.shape[1], targets, device,
                )
                for subset_index, indices in enumerate(SUBSETS):
                    sites = tuple(SITES[index] for index in indices)
                    patched, calls, audit, error = run_subtractive_patch(
                        model, tokens, arm=arm, scale=scale,
                        deltas={site: deltas[site] for site in sites},
                        sites=sites, position_mask=query_mask,
                    )
                    count = sum(calls.values())
                    _record(
                        audit_totals,
                        f"rung474:{source}:slot{slot}:{SUBSET_NAMES[subset_index]}",
                        audit, count,
                    )
                    patch_calls += count
                    reconstruction = max(reconstruction, error)
                    damage = position_parent._nll(patched, batch_rows) - source_nll
                    for output_index, local_doc, query in chosen:
                        effects[si, subset_index, output_index] = float(damage[local_doc, query])
            del source_products, deltas
        del absent_products
    return {
        "effects": effects,
        "empty_patch_max_abs": empty_error,
        "subtractive_patch_calls": patch_calls,
    }, reconstruction


def _metrics(target, prediction):
    values = form_parent._metrics(target, prediction)
    values["pearson"] = position_parent._pearson(target, prediction)
    return values


def analyze(windows, old_position, old_factorial, old_effects, selections):
    reports = {}
    singleton_error, closure_error = 0.0, 0.0
    b_flags, natural_c_flags, e_flags = [], [], []
    code_top, natural_top = [], []
    any_material = False
    subtractive_source_cosines = []
    for name, window in windows.items():
        old_query = old_position["windows"][name]["query"].double()
        old_pairs = old_factorial["windows"][name]["pair_effects"].double()
        old_window = old_effects["windows"][name]
        indices = selections[name]["rung470_indices"]
        coordinates = selections[name]["coordinates"]
        docs = torch.tensor([row[0] for row in coordinates])
        window_start = next(start for candidate, _, start, _ in WINDOWS if candidate == name)
        half_masks = (docs < window_start + 48, docs >= window_start + 48)
        reports[name] = {}
        interaction_vectors = {}
        interaction_norms = {}
        static_total_vectors = {}
        subtractive_total_vectors = {}
        for si, source in enumerate(SOURCES):
            effects = window["effects"][si]
            mains = effects[list(SINGLE_INDICES)]
            pairs = effects[list(PAIR_INDICES)]
            union = effects[UNION_INDEX]
            old_mains = old_query[si, :3]
            old_union = old_query[si, 3]
            singleton_error = max(
                singleton_error, float((mains - old_mains).abs().max()),
            )
            pair_interactions, triple, reconstructed = parent.mobius_terms(mains, pairs, union)
            closure_error = max(closure_error, float((reconstructed - union).abs().max()))
            pair_context = torch.stack([
                position_parent._cell_vector(values, old_window, indices)
                for values in pair_interactions
            ])
            triple_context = position_parent._cell_vector(triple, old_window, indices)
            total = union - mains.sum(0)
            total_context = position_parent._cell_vector(total, old_window, indices)
            static_total = old_union - old_mains.sum(0)
            static_total_context = position_parent._cell_vector(static_total, old_window, indices)
            pair_norms = torch.linalg.vector_norm(pair_context, dim=1)
            triple_norm = float(torch.linalg.vector_norm(triple_context))
            total_norm = float(torch.linalg.vector_norm(total_context))
            static_norm = float(torch.linalg.vector_norm(static_total_context))
            largest = int(torch.argmax(pair_norms))
            largest_values = pair_interactions[largest]
            half_rows = []
            for mask in half_masks:
                selected = largest_values[mask]
                half_rows.append({
                    "signed_mean": float(selected.mean()),
                    "norm": float(torch.linalg.vector_norm(selected)),
                })
            pooled_mean = float(largest_values.mean())
            pooled_norm = float(torch.linalg.vector_norm(largest_values))
            half_stable = bool(
                all(row["signed_mean"] * pooled_mean > 0 for row in half_rows)
                and all(row["norm"] >= .20 * pooled_norm for row in half_rows)
            )
            if name.startswith("natural") and source == "H":
                e_flags.append(half_stable)
            (natural_top if name.startswith("natural") else code_top).append(largest)
            any_material |= bool(max(float(pair_norms.max()), triple_norm) >= .003)
            union_metrics = _metrics(old_union, union)
            union_context_metrics = form_parent._metrics(
                position_parent._cell_vector(old_union, old_window, indices),
                position_parent._cell_vector(union, old_window, indices),
            )
            b_ok = bool(
                union_metrics["pearson"] >= .80
                and union_context_metrics["cosine"] >= .80
                and .50 <= union_context_metrics["projection_on_target"] <= 1.50
            )
            b_flags.append(b_ok)
            interaction_vectors[source] = total_context
            interaction_norms[source] = total_norm
            static_total_vectors[source] = static_total_context
            subtractive_total_vectors[source] = total_context
            reports[name][source] = {
                "singleton_parent_max_abs_error_nat": float((mains - old_mains).abs().max()),
                "union_coordinate_metrics": union_metrics,
                "union_context_coordinate_metrics": union_context_metrics,
                "pair_context_vectors": pair_context.tolist(),
                "pair_context_norms": pair_norms.tolist(),
                "triple_context_vector": triple_context.tolist(),
                "triple_context_norm": triple_norm,
                "total_interaction_context_vector": total_context.tolist(),
                "total_interaction_context_norm": total_norm,
                "static_total_interaction_context_norm": static_norm,
                "interaction_norm_ratio_to_static": total_norm / max(static_norm, 1e-30),
                "largest_pair": parent.PAIR_NAMES[largest],
                "largest_pair_halves": half_rows,
                "largest_pair_half_stable": half_stable,
                "mobius_closure_max_abs": float((reconstructed - union).abs().max()),
            }
        subtractive_cosine = form_parent._cosine(
            interaction_vectors["N"], interaction_vectors["H"],
        )
        static_cosine = form_parent._cosine(
            static_total_vectors["N"], static_total_vectors["H"],
        )
        subtractive_source_cosines.append(subtractive_cosine)
        reduction_both = all(
            reports[name][source]["interaction_norm_ratio_to_static"] <= .75
            for source in SOURCES
        )
        improvement = subtractive_cosine >= static_cosine + .30
        if name.startswith("natural"):
            natural_c_flags.append(bool(reduction_both or improvement))
        reports[name]["source_comparison"] = {
            "subtractive_cosine": subtractive_cosine,
            "static_cosine": static_cosine,
            "cosine_improvement": subtractive_cosine - static_cosine,
            "both_sources_reduce_norm_at_least_25pct": reduction_both,
        }
    pred_b = bool(all(b_flags))
    pred_c = bool(subtractive_source_cosines[0] >= .80 and all(natural_c_flags))
    natural_three_m812 = sum(index == 1 for index in natural_top) >= 3
    natural_low_cosine = any(
        reports[name]["source_comparison"]["subtractive_cosine"] <= .20
        and all(reports[name][source]["total_interaction_context_norm"] >= .003 for source in SOURCES)
        for name, _, _, _ in WINDOWS if name.startswith("natural")
    )
    pred_d = bool(all(index == 0 for index in code_top) and (natural_three_m812 or natural_low_cosine))
    return {
        "reports": reports,
        "subset_names": list(SUBSET_NAMES),
        "singleton_parent_max_abs_error_nat": singleton_error,
        "max_mobius_closure_abs": closure_error,
        "code_largest_pair_indices": code_top,
        "natural_largest_pair_indices": natural_top,
        "subtractive_total_interaction_source_cosines": subtractive_source_cosines,
        "material_interaction_exists": any_material,
        "pred_b_coordinate_stable": pred_b,
        "pred_c_state_mixing": pred_c,
        "pred_d_register_persists": pred_d,
        "pred_e_natural_h_half_stable": bool(all(e_flags)),
    }


def main():
    started = time.time()
    (
        roles, scale, old_effects, selections, old_position,
        old_factorial, metadata,
    ) = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 474, "model_loaded": False,
            "subtractive_outcomes_opened": False, "sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "expected_patch_calls": EXPECTED_PATCH_CALLS,
            "subsets": list(SUBSET_NAMES),
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung474 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    windows, reconstruction = {}, 0.0
    for name, role, _, _ in WINDOWS:
        payload, _ = roles[role]
        windows[name], error = collect_window(
            model, payload, scale, selections[name], audit_totals, replay,
        )
        reconstruction = max(reconstruction, error)
    analysis = analyze(windows, old_position, old_factorial, old_effects, selections)
    forwards = sum(row["forwards"] for row in audit_totals.values())
    observed_calls = sum(row.get("subtractive_patch_calls", 0) for row in audit_totals.values())
    empty_error = max(row["empty_patch_max_abs"] for row in windows.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and empty_error == 0 and analysis["singleton_parent_max_abs_error_nat"] <= 1e-6
        and analysis["max_mobius_closure_abs"] <= 1e-12
        and forwards == EXPECTED_FORWARDS and observed_calls == EXPECTED_PATCH_CALLS
    )
    minimum_union_pearson = min(
        analysis["reports"][name][source]["union_coordinate_metrics"]["pearson"]
        for name, _, _, _ in WINDOWS for source in SOURCES
    )
    strong_null = bool(
        not pred_a or minimum_union_pearson < .50
        or not analysis["material_interaction_exists"]
    )
    torch.save({
        "schema": "rung474_subtractive_query_factorial_v1",
        "windows": windows, "subset_names": list(SUBSET_NAMES),
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": 474,
        "claim_level": "query_circuit_causal_coordinate_stability_test",
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
            "outer_forwards": forwards,
            "subtractive_patch_calls": observed_calls,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_coordinate_stable': analysis["pred_b_coordinate_stable"],
        'pred_c_state_mixing': analysis["pred_c_state_mixing"],
        'pred_d_register_persists': analysis["pred_d_register_persists"],
        'pred_e_natural_h_half_stable': analysis["pred_e_natural_h_half_stable"],
        "strong_null": strong_null,
        "runtime_s": time.time() - started,
        "next_step": (
            "downstream_state_decomposition_with_coordinate_explicit_interface"
            if pred_a and analysis["pred_b_coordinate_stable"]
            else "retain_coordinate_specific_query_circuit_and_test_downstream_equivalence"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 474,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "analysis": analysis,
        "instrument": {"replay": replay, "factor_error": reconstruction,
                       "empty_error": empty_error, "forwards": forwards,
                       "patch_calls": observed_calls},
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
