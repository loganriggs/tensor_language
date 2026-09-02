#!/usr/bin/env python3
"""RUNG473 -- exact three-MLP factorial at the equality query position.

Registered before opening any two-MLP query intervention:
  pred_a: frozen identity, exact replay/factors/no-op/calls/Mobius closure.
  pred_b: one stable MLP pair carries the query interaction.
  pred_c: the interaction is register/source dependent.
  pred_d: main plus pair terms are sufficient; the triple term is dispensable.
  pred_e: the largest pair interaction is stable across document halves.
Strong null: invalid, no material pair, or triple dominates in >=4/6 conditions.
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
import equality_query_position_intervention_rung472 as parent
import equality_mlp_product_term_group_rung467 as product_parent
import equality_score_correction_interchange_rung464 as source_parent
import equality_score_downstream_gate_rung462 as audit_parent
import equality_mlp_response_form_rung469 as form_parent


PREREG = POLY / "EQUALITY_QUERY_MLP_FACTORIAL_RUNG473_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_query_position_intervention_rung472_results.json"
PARENT_BUNDLE = ROOT / "equality_query_position_intervention_rung472_per_token.pt"
PARENT_SOURCE = ROOT / "ops/equality_query_position_intervention_rung472.py"
OUT = ROOT / "equality_query_mlp_factorial_rung473_results.json"
BUNDLE = ROOT / "equality_query_mlp_factorial_rung473_per_token.pt"
SOURCES = parent.SOURCES
MODULES = parent.MODULES
SITES = parent.SITES
WINDOWS = parent.WINDOWS
BATCH = parent.BATCH
PAIRS = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = tuple(f"{SITES[left]}+{SITES[right]}" for left, right in PAIRS)
FORWARDS_PER_BATCH = 3 + len(SOURCES) * (2 + 2 * len(PAIRS))
EXPECTED_BATCHES = sum((stop - start) // BATCH for _, _, start, stop in WINDOWS)
EXPECTED_FORWARDS = EXPECTED_BATCHES * FORWARDS_PER_BATCH
EXPECTED_PATCH_CALLS_PER_BATCH = len(SOURCES) * (
    len(SITES) + 2 * sum(2 for _ in PAIRS)
)
EXPECTED_PATCH_CALLS = EXPECTED_BATCHES * EXPECTED_PATCH_CALLS_PER_BATCH
HASHES = {
    PREREG: "6688b7e308ef393abb66f2c37240eb880db9f5d7c5cbffa67088511749c2048f",
    PARENT_RESULT: "21797e38820cc10387f534c21520feebd64470ca0ebc34147d3755c889699140",
    PARENT_BUNDLE: "fa3625f108f908c26fd90f977b37a691f36d9e463bf4171351ec0c80a0aa84bd",
    PARENT_SOURCE: "d154c181ecba5e75982d21a1155563102da51e582a0982cb244d2120005d5c77",
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
        "pred_b_query_component": True,
        "pred_c_query_advantage": True,
        "pred_d_query_composition": False,
        "pred_e_selectivity": True,
        "strong_null": False,
    }
    if result.get("rung") != 472 or any(result.get(key) is not value for key, value in expected.items()):
        raise RuntimeError("rung472 registered verdict changed")
    roles, scale, old_effects, selections, _, metadata = parent.validate_inputs()
    old_bundle = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=True)
    if old_bundle.get("schema") != "rung472_exact_position_effects_v1":
        raise RuntimeError("rung472 bundle schema changed")
    metadata = {
        **metadata,
        "rung472_result_sha256": sha256(PARENT_RESULT),
        "rung472_bundle_sha256": sha256(PARENT_BUNDLE),
        "pairs": list(PAIR_NAMES),
        "windows": [list(row) for row in WINDOWS],
    }
    return roles, scale, old_effects, selections, old_bundle, metadata


def mobius_terms(mains, pairs, union):
    """Return pair interactions, triple interaction, and exact reconstruction."""
    interactions = []
    for pi, (left, right) in enumerate(PAIRS):
        interactions.append(pairs[pi] - mains[left] - mains[right])
    interactions = torch.stack(interactions)
    triple = union - mains.sum(0) - interactions.sum(0)
    reconstructed = mains.sum(0) + interactions.sum(0) + triple
    return interactions, triple, reconstructed


def _record(audit_totals, key, audit, patch_calls=0):
    row = audit_totals.setdefault(key, {"forwards": 0, "position_patch_calls": 0})
    row["forwards"] += 1
    row["position_patch_calls"] += patch_calls


@torch.no_grad()
def collect_window(model, payload, scale, selection, audit_totals, replay):
    coordinates = selection["coordinates"]
    by_doc = {}
    for output_index, (doc, query, _) in enumerate(coordinates):
        by_doc.setdefault(doc, []).append((output_index, query))
    pair_effects = torch.zeros(len(SOURCES), len(PAIRS), len(coordinates), dtype=torch.float64)
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
            audit_totals, "rung473:native", audit, analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = source_parent.run_forward(model, tokens, arm="replay")
        audit_parent._record_audit(
            audit_totals, "rung473:replay", audit, analytical=True, captures=0, patches=0,
        )
        delta = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(delta.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(delta.square().sum()) / max(float(native.square().sum()), 1e-30),
        )
        reconstruction = max(reconstruction, error)
        _, absent_products, _, audit, error = product_parent.run_term_forward(
            model, tokens, arm="base", capture_products=True,
        )
        _record(audit_totals, "rung473:absent", audit)
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
            source_logits, _, _, audit, error = product_parent.run_term_forward(
                model, tokens, arm=arm, scale=scale,
            )
            _record(audit_totals, f"rung473:source:{source}", audit)
            reconstruction = max(reconstruction, error)
            source_nll = parent._nll(source_logits, batch_rows)
            false_mask = torch.zeros_like(tokens, dtype=torch.bool)
            empty_logits, calls, audit, error = parent.run_position_patch(
                model, tokens, arm=arm, scale=scale,
                baselines={site: absent_products[site] for site in SITES},
                sites=SITES, position_mask=false_mask,
            )
            calls_count = sum(calls.values())
            _record(audit_totals, f"rung473:empty:{source}", audit, calls_count)
            patch_calls += calls_count
            reconstruction = max(reconstruction, error)
            empty_error = max(empty_error, float((empty_logits - source_logits).abs().max()))
            for slot, chosen in enumerate(slots):
                targets = [(local_doc, query) for _, local_doc, query in chosen]
                query_mask, _, _, _ = parent.position_masks(
                    len(batch_rows), tokens.shape[1], targets, device,
                )
                for pi, (left, right) in enumerate(PAIRS):
                    sites = (SITES[left], SITES[right])
                    patched, calls, audit, error = parent.run_position_patch(
                        model, tokens, arm=arm, scale=scale,
                        baselines={site: absent_products[site] for site in sites},
                        sites=sites, position_mask=query_mask,
                    )
                    calls_count = sum(calls.values())
                    _record(
                        audit_totals, f"rung473:{source}:slot{slot}:{PAIR_NAMES[pi]}",
                        audit, calls_count,
                    )
                    patch_calls += calls_count
                    reconstruction = max(reconstruction, error)
                    damage = parent._nll(patched, batch_rows) - source_nll
                    for output_index, local_doc, query in chosen:
                        pair_effects[si, pi, output_index] = float(damage[local_doc, query])
        del absent_products
    return {
        "pair_effects": pair_effects,
        "empty_patch_max_abs": empty_error,
        "position_patch_calls": patch_calls,
    }, reconstruction


def _metrics(target, prediction):
    values = form_parent._metrics(target, prediction)
    values["pearson"] = parent._pearson(target, prediction)
    return values


def analyze(windows, old_bundle, old_effects, selections):
    reports = {}
    largest_pairs, pair_fractions, source_cosines = [], [], []
    d_flags, e_flags, material_pair = [], [], False
    triple_dominant = 0
    natural_localization = False
    for name, window in windows.items():
        old = old_bundle["windows"][name]["query"].double()
        indices = selections[name]["rung470_indices"]
        old_effect_window = old_effects["windows"][name]
        coordinates = selections[name]["coordinates"]
        docs = torch.tensor([row[0] for row in coordinates])
        window_start = next(start for candidate, _, start, _ in WINDOWS if candidate == name)
        half_masks = (docs < window_start + 48, docs >= window_start + 48)
        reports[name] = {}
        pair_context_by_source = {}
        for si, source in enumerate(SOURCES):
            mains = old[si, :3]
            union = old[si, 3]
            pair_interactions, triple, reconstructed = mobius_terms(
                mains, window["pair_effects"][si], union,
            )
            closure = float((reconstructed - union).abs().max())
            pair_context = torch.stack([
                parent._cell_vector(values, old_effect_window, indices)
                for values in pair_interactions
            ])
            triple_context = parent._cell_vector(triple, old_effect_window, indices)
            total_context = parent._cell_vector(
                union - mains.sum(0), old_effect_window, indices,
            )
            pair_norms = torch.linalg.vector_norm(pair_context, dim=1)
            triple_norm = float(torch.linalg.vector_norm(triple_context))
            total_norm = float(torch.linalg.vector_norm(total_context))
            largest = int(torch.argmax(pair_norms))
            fraction = float(pair_norms[largest]) / max(total_norm, 1e-30)
            no_triple = union - triple
            sufficiency = _metrics(union, no_triple)
            d_ok = bool(sufficiency["pearson"] >= .90 and sufficiency["normalized_l2_error"] <= .50)
            d_flags.append(d_ok)
            half_rows = []
            for mask in half_masks:
                values = pair_interactions[largest, mask]
                half_rows.append({
                    "signed_mean": float(values.mean()),
                    "norm": float(torch.linalg.vector_norm(values)),
                })
            pooled_mean = float(pair_interactions[largest].mean())
            same_sign = all(row["signed_mean"] * pooled_mean > 0 for row in half_rows)
            pooled_norm = float(torch.linalg.vector_norm(pair_interactions[largest]))
            e_ok = bool(same_sign and all(row["norm"] >= .20 * pooled_norm for row in half_rows))
            e_flags.append(e_ok)
            pair_context_by_source[source] = pair_context
            largest_pairs.append(largest)
            pair_fractions.append(fraction)
            material_pair |= bool(float(pair_norms.max()) >= .003)
            triple_dominant += int(triple_norm >= float(pair_norms.sum()))
            reports[name][source] = {
                "pair_context_vectors": pair_context.tolist(),
                "pair_context_norms": pair_norms.tolist(),
                "triple_context_vector": triple_context.tolist(),
                "triple_context_norm": triple_norm,
                "total_interaction_context_norm": total_norm,
                "largest_pair": PAIR_NAMES[largest],
                "largest_pair_fraction_of_total": fraction,
                "no_triple_metrics": sufficiency,
                "mobius_closure_max_abs": closure,
                "largest_pair_halves": half_rows,
                "largest_pair_half_stable": e_ok,
            }
        pair_source_rows = []
        for pi, pair_name in enumerate(PAIR_NAMES):
            left = pair_context_by_source["N"][pi]
            right = pair_context_by_source["H"][pi]
            cosine = form_parent._cosine(left, right)
            pair_source_rows.append({"pair": pair_name, "cosine": cosine})
            source_cosines.append(cosine)
            if name.startswith("natural") and cosine <= .20 and max(
                float(torch.linalg.vector_norm(left)), float(torch.linalg.vector_norm(right)),
            ) >= .003:
                natural_localization = True
        reports[name]["pair_source_comparison"] = pair_source_rows
    same_largest = len(set(largest_pairs)) == 1
    stable_pair_sources = all(source_cosines[wi * len(PAIRS) + largest_pairs[2 * wi]] >= .70
                              for wi in range(len(WINDOWS))) if same_largest else False
    pred_b = bool(same_largest and min(pair_fractions) >= .40 and stable_pair_sources)
    old_interactions = [
        old_bundle["windows"][name]["query"][:, 3].double()
        - old_bundle["windows"][name]["query"][:, :3].double().sum(1)
        for name, _, _, _ in WINDOWS
    ]
    old_source_cosines = []
    for wi, (name, _, _, _) in enumerate(WINDOWS):
        indices = selections[name]["rung470_indices"]
        old_window = old_effects["windows"][name]
        vectors = [parent._cell_vector(old_interactions[wi][si], old_window, indices)
                   for si in range(len(SOURCES))]
        old_source_cosines.append(form_parent._cosine(vectors[0], vectors[1]))
    code_largest = set(largest_pairs[:2])
    natural_pair_change = any(index not in code_largest for index in largest_pairs[2:])
    pred_c = bool(
        old_source_cosines[0] >= .80 and min(old_source_cosines[1:]) <= .20
        and (natural_pair_change or natural_localization)
    )
    max_closure = max(
        reports[name][source]["mobius_closure_max_abs"]
        for name, _, _, _ in WINDOWS for source in SOURCES
    )
    return {
        "reports": reports,
        "pair_names": list(PAIR_NAMES),
        "largest_pair_indices_in_window_source_order": largest_pairs,
        "old_total_interaction_source_cosines": old_source_cosines,
        "max_mobius_closure_abs": max_closure,
        "material_pair_exists": material_pair,
        "triple_dominant_condition_count": triple_dominant,
        "pred_b_stable_pair": pred_b,
        "pred_c_register_source_dependent": pred_c,
        "pred_d_pairwise_sufficient": bool(all(d_flags)),
        "pred_e_half_stable": bool(all(e_flags)),
    }


def main():
    started = time.time()
    roles, scale, old_effects, selections, old_bundle, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 473, "model_loaded": False,
            "pair_outcomes_opened": False, "sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "expected_patch_calls": EXPECTED_PATCH_CALLS,
            "pairs": list(PAIR_NAMES),
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung473 output namespace already exists")
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
    analysis = analyze(windows, old_bundle, old_effects, selections)
    forwards = sum(row["forwards"] for row in audit_totals.values())
    observed_patch_calls = sum(row.get("position_patch_calls", 0) for row in audit_totals.values())
    empty_error = max(row["empty_patch_max_abs"] for row in windows.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and empty_error == 0 and analysis["max_mobius_closure_abs"] <= 1e-12
        and forwards == EXPECTED_FORWARDS and observed_patch_calls == EXPECTED_PATCH_CALLS
    )
    strong_null = bool(
        not pred_a or not analysis["material_pair_exists"]
        or analysis["triple_dominant_condition_count"] >= 4
    )
    torch.save({
        "schema": "rung473_exact_query_mlp_factorial_v1",
        "windows": windows,
        "pair_names": list(PAIR_NAMES),
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": 473,
        "claim_level": "exact_query_position_mlp_composition_test",
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
            "query_patch_calls": observed_patch_calls,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_stable_pair': analysis["pred_b_stable_pair"],
        'pred_c_register_source_dependent': analysis["pred_c_register_source_dependent"],
        'pred_d_pairwise_sufficient': analysis["pred_d_pairwise_sufficient"],
        'pred_e_half_stable': analysis["pred_e_half_stable"],
        "strong_null": strong_null,
        "runtime_s": time.time() - started,
        "next_step": (
            "downstream_defined_stable_pair_split"
            if pred_a and analysis["pred_b_stable_pair"] and analysis["pred_d_pairwise_sufficient"]
            else "register_conditioned_downstream_state_query_decomposition"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 473,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "analysis": analysis,
        "instrument": {"replay": replay, "factor_error": reconstruction,
                       "empty_error": empty_error, "forwards": forwards,
                       "patch_calls": observed_patch_calls},
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
