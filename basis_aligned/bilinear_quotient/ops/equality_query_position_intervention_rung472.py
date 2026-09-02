#!/usr/bin/env python3
"""RUNG472 -- exact target-specific query-position MLP interventions.

Registered before opening position-intervention outcomes:
  pred_a: frozen identity, exact replay/factors/no-op/full-prefix/calls.
  pred_b: query-position union is an exact causal component across registers.
  pred_c: query position is more informative than the non-query prefix.
  pred_d: at least two MLPs participate and query composition is resolved.
  pred_e: query-position removal is selective against off-target tokens.
Strong null: invalid, no code query advantage, or no natural query relationship.
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
import equality_paired_response_kernel_rung471 as parent
import equality_mlp_product_term_group_rung467 as product_parent
import equality_score_correction_interchange_rung464 as source_parent
import equality_score_downstream_gate_rung462 as audit_parent
import equality_mlp_response_form_rung469 as form_parent
import equality_context_causal_state_rung470 as context_parent


PREREG = POLY / "EQUALITY_QUERY_POSITION_INTERVENTION_RUNG472_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_paired_response_kernel_rung471_results.json"
PARENT_BUNDLE = ROOT / "equality_paired_response_kernel_rung471.pt"
PARENT_SOURCE = ROOT / "ops/equality_paired_response_kernel_rung471.py"
R470_BUNDLE = parent.PARENT_BUNDLE
OUT = ROOT / "equality_query_position_intervention_rung472_results.json"
BUNDLE = ROOT / "equality_query_position_intervention_rung472_per_token.pt"
SOURCES = parent.SOURCES
MODULES = parent.MODULES
SITES = parent.SITES
CONTEXT_CELLS = parent.CONTEXT_CELLS
WINDOWS = tuple(row for row in parent.WINDOWS if row[0] != "code_discovery")
BATCH = parent.BATCH
ARMS_PER_SLOT_SOURCE = 6
FORWARDS_PER_BATCH = 2 + 1 + len(SOURCES) * (2 + 2 * ARMS_PER_SLOT_SOURCE)
EXPECTED_FORWARDS = sum((stop - start) // BATCH for _, _, start, stop in WINDOWS) \
    * FORWARDS_PER_BATCH
HASHES = {
    PREREG: "f6b760ca6127a00939672c843b98872047146a9b08806b6cf7658c88f04da2c8",
    PARENT_RESULT: "9d7d744ca765ee2289208cdaf5e0d55492f3a94b033b019291dfb0cc444f9523",
    PARENT_BUNDLE: "3c2cb5b530125f6c2bbdba667521989b1197ddc76bf562c5c3789e2e5fb8d8cf",
    PARENT_SOURCE: "e11664dc6d67a4796003124a504651cc7c6ed8769b44d87b657cfe6dde99c642",
    R470_BUNDLE: "227bb79fb60bfdec232d51f0862dbe44073887853e7afee1ab2cc517a4a94118",
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
    if result.get("rung") != 471 or any(result.get(key) is not False for key in (
        "pred_a_instrument", "pred_b_heldout_code", "pred_c_natural",
        "pred_d_spatial_computation", "pred_e_shared_downstream_use",
    )) or result.get("strong_null") is not True:
        raise RuntimeError("rung471 registered verdict changed")
    roles, scale, old_effects, selections, metadata = parent.validate_inputs()
    kernel_bundle = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=True)
    if kernel_bundle.get("schema") != "rung471_paired_response_kernel_v1":
        raise RuntimeError("rung471 bundle schema changed")
    metadata = {
        **metadata, "rung471_result_sha256": sha256(PARENT_RESULT),
        "rung471_bundle_sha256": sha256(PARENT_BUNDLE),
        "windows": [list(row) for row in WINDOWS],
        "position_arms": ["query", "non_query_prefix", "full_prefix"],
        "query_targets": [*SITES, "union"],
    }
    return roles, scale, old_effects, selections, kernel_bundle, metadata


def position_masks(batch_size, length, targets, device):
    query = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
    nonquery = torch.zeros_like(query)
    prefix = torch.zeros_like(query)
    active = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for local_doc, query_position in targets:
        active[local_doc] = True
        query[local_doc, query_position] = True
        nonquery[local_doc, :query_position] = True
        prefix[local_doc, :query_position + 1] = True
    if bool((query & nonquery).any()) or not torch.equal(query | nonquery, prefix):
        raise RuntimeError("position masks do not partition causal prefix")
    return query, nonquery, prefix, active


@torch.no_grad()
def run_position_patch(model, tokens, *, arm, scale, baselines, sites, position_mask):
    sites = tuple(sites)
    if set(sites) - set(SITES) or set(baselines) != set(sites):
        raise ValueError("malformed position patch sites")
    handles, calls = [], {site: 0 for site in sites}
    for layer, site in zip(MODULES, SITES):
        if site not in sites:
            continue
        baseline = baselines[site]
        down = model.transformer.h[layer].mlp.Down

        def hook(_module, inputs, name=site, replacement=baseline):
            if calls[name] != 0:
                raise RuntimeError(f"duplicate product patch at {name}")
            product = inputs[0]
            if replacement.shape != product.shape or replacement.dtype != product.dtype \
                    or replacement.device != product.device:
                raise RuntimeError(f"baseline product mismatch at {name}")
            updated = product.clone()
            updated[position_mask] = replacement[position_mask]
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
        raise RuntimeError("not every position patch fired exactly once")
    return logits, calls, audit, error


def _record(audit_totals, key, audit, product_patch_calls=0):
    row = audit_totals.setdefault(key, {"forwards": 0, "position_patch_calls": 0})
    row["forwards"] += 1
    row["position_patch_calls"] += product_patch_calls


def _nll(logits, rows):
    return context_parent._nll(logits, rows)


@torch.no_grad()
def collect_window(model, payload, masks, scale, selection, audit_totals, replay):
    coordinates = selection["coordinates"]
    by_doc = {}
    for output_index, (doc, query, _) in enumerate(coordinates):
        by_doc.setdefault(doc, []).append((output_index, query))
    n = len(coordinates)
    query_effects = torch.zeros(len(SOURCES), 4, n, dtype=torch.float64)
    nonquery_effects = torch.zeros(len(SOURCES), n, dtype=torch.float64)
    full_effects = torch.zeros(len(SOURCES), n, dtype=torch.float64)
    off_target = torch.zeros(len(SOURCES), n, dtype=torch.float64)
    rows = payload["rows"]
    first_doc = min(by_doc)
    last_doc = max(by_doc) + 1
    device = next(model.parameters()).device
    reconstruction, empty_error = 0.0, 0.0
    expected_patch_calls = 0
    for start in range(first_doc, last_doc, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = source_parent.run_forward(model, tokens, arm="native")
        audit_parent._record_audit(
            audit_totals, "rung472:native", audit, analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = source_parent.run_forward(model, tokens, arm="replay")
        audit_parent._record_audit(
            audit_totals, "rung472:replay", audit, analytical=True, captures=0, patches=0,
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
        _record(audit_totals, "rung472:absent", audit)
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
            _record(audit_totals, f"rung472:source:{source}", audit)
            reconstruction = max(reconstruction, error)
            source_nll = _nll(source_logits, batch_rows)
            false_mask = torch.zeros_like(tokens, dtype=torch.bool)
            empty_logits, calls, audit, error = run_position_patch(
                model, tokens, arm=arm, scale=scale,
                baselines={site: absent_products[site] for site in SITES},
                sites=SITES, position_mask=false_mask,
            )
            _record(audit_totals, f"rung472:empty:{source}", audit, sum(calls.values()))
            expected_patch_calls += len(SITES)
            reconstruction = max(reconstruction, error)
            empty_error = max(empty_error, float((empty_logits - source_logits).abs().max()))
            for slot, chosen in enumerate(slots):
                target_positions = [(local_doc, query) for _, local_doc, query in chosen]
                query_mask, nonquery_mask, prefix_mask, _ = position_masks(
                    len(batch_rows), tokens.shape[1], target_positions, device,
                )
                arm_specs = [
                    (f"query:{site}", (site,), query_mask) for site in SITES
                ] + [
                    ("query:union", SITES, query_mask),
                    ("nonquery:union", SITES, nonquery_mask),
                    ("full:union", SITES, prefix_mask),
                ]
                for arm_name, sites, position_mask in arm_specs:
                    patched, calls, audit, error = run_position_patch(
                        model, tokens, arm=arm, scale=scale,
                        baselines={site: absent_products[site] for site in sites},
                        sites=sites, position_mask=position_mask,
                    )
                    _record(
                        audit_totals, f"rung472:{source}:slot{slot}:{arm_name}", audit,
                        sum(calls.values()),
                    )
                    expected_patch_calls += len(sites)
                    reconstruction = max(reconstruction, error)
                    damage = _nll(patched, batch_rows) - source_nll
                    for output_index, local_doc, query in chosen:
                        value = float(damage[local_doc, query])
                        if arm_name.startswith("query:"):
                            target = arm_name.split(":", 1)[1]
                            ti = SITES.index(target) if target in SITES else 3
                            query_effects[si, ti, output_index] = value
                            if target == "union":
                                selected = masks["off_target"][start + local_doc].to(device)
                                off_target[si, output_index] = float(
                                    damage[local_doc, selected].mean()
                                ) if int(selected.sum()) else 0.0
                        elif arm_name == "nonquery:union":
                            nonquery_effects[si, output_index] = value
                        else:
                            full_effects[si, output_index] = value
        del absent_products
    return {
        "query": query_effects, "nonquery": nonquery_effects, "full": full_effects,
        "off_target": off_target, "empty_patch_max_abs": empty_error,
        "position_patch_calls": expected_patch_calls,
    }, reconstruction


def _pearson(left, right):
    return context_parent.pearson(left, right)


def _metrics(target, proposed):
    target = torch.as_tensor(target, dtype=torch.float64)
    proposed = torch.as_tensor(proposed, dtype=torch.float64)
    base = form_parent._metrics(target, proposed)
    base["pearson"] = _pearson(target, proposed)
    base["rmse"] = float(torch.sqrt(torch.mean((proposed - target) ** 2)))
    return base


def _cell_vector(values, old_window, indices):
    return parent._context_vector(values, old_window, indices)


def analyze(windows, old_effects, selections):
    reports = {}
    b_flags, c_flags, d_positive = [], [], {source: set(SITES) for source in SOURCES}
    d_interaction_flags, e_flags = [], []
    any_natural_positive = False
    for name, window in windows.items():
        indices = selections[name]["rung470_indices"]
        old_window = old_effects["windows"][name]
        exact_individual = old_window["effects"][:, :3, indices]
        reports[name] = {}
        coordinates = selections[name]["coordinates"]
        docs = torch.tensor([row[0] for row in coordinates])
        window_start = next(start for candidate, _, start, _ in WINDOWS if candidate == name)
        half_masks = (docs < window_start + 48, docs >= window_start + 48)
        for si, source in enumerate(SOURCES):
            full = window["full"][si]
            query = window["query"][si, 3]
            nonquery = window["nonquery"][si]
            query_metrics = _metrics(full, query)
            nonquery_metrics = _metrics(full, nonquery)
            exact_cells = _cell_vector(full, old_window, indices)
            query_cells = _cell_vector(query, old_window, indices)
            nonquery_cells = _cell_vector(nonquery, old_window, indices)
            context_metrics = form_parent._metrics(exact_cells, query_cells)
            b_ok = bool(
                query_metrics["pearson"] >= .55 and context_metrics["cosine"] >= .80
                and .25 <= context_metrics["projection_on_target"] <= 1.75
            )
            b_flags.append(b_ok)
            c_ok = bool(
                query_metrics["pearson"] >= nonquery_metrics["pearson"] + .15
                or query_metrics["rmse"] <= .80 * nonquery_metrics["rmse"]
            )
            c_flags.append(c_ok)
            if name.startswith("natural"):
                any_natural_positive |= query_metrics["pearson"] > 0
            individual = {}
            for mi, site in enumerate(SITES):
                metrics = _metrics(exact_individual[si, mi], window["query"][si, mi])
                if metrics["pearson"] <= 0:
                    d_positive[source].discard(site)
                individual[site] = metrics
            query_interaction = query - window["query"][si, :3].sum(0)
            interaction_cells = _cell_vector(query_interaction, old_window, indices)
            interaction_norm = float(torch.linalg.vector_norm(interaction_cells))
            reports[name][source] = {
                "full_context_vector": exact_cells.tolist(),
                "query_context_vector": query_cells.tolist(),
                "nonquery_context_vector": nonquery_cells.tolist(),
                "query_metrics": query_metrics, "nonquery_metrics": nonquery_metrics,
                "query_context_metrics": context_metrics,
                "pred_b_window": b_ok, "pred_c_window": c_ok,
                "individual_query_metrics": individual,
                "query_interaction_vector": interaction_cells.tolist(),
                "query_interaction_norm": interaction_norm,
            }
            target_mean = abs(float(query.mean()))
            off_mean = abs(float(window["off_target"][si].mean()))
            half_order = all(
                abs(float(query[mask].mean())) > abs(float(window["off_target"][si, mask].mean()))
                for mask in half_masks if bool(mask.any())
            )
            e_ok = off_mean <= .5 * target_mean and off_mean <= .01 and half_order
            reports[name][source]["selectivity"] = {
                "absolute_target_mean": target_mean, "absolute_off_target_mean": off_mean,
                "half_target_exceeds_off_target": half_order, "holds": bool(e_ok),
            }
            e_flags.append(bool(e_ok))
        left = reports[name]["N"]["query_interaction_vector"]
        right = reports[name]["H"]["query_interaction_vector"]
        interaction_cosine = form_parent._cosine(left, right)
        interaction_small = all(
            reports[name][source]["query_interaction_norm"] < .003 for source in SOURCES
        )
        same_sign = sum(left) * sum(right) > 0
        interaction_ok = bool(same_sign and (interaction_small or interaction_cosine >= .70))
        reports[name]["interaction_source_comparison"] = {
            "cosine": interaction_cosine, "both_small": interaction_small,
            "same_sum_sign": bool(same_sign), "holds": interaction_ok,
        }
        d_interaction_flags.append(interaction_ok)
    individual_intersection = set(SITES)
    for source in SOURCES:
        individual_intersection &= d_positive[source]
    pred_b = all(b_flags)
    pred_c = all(c_flags)
    pred_d = len(individual_intersection) >= 2 and all(d_interaction_flags)
    pred_e = all(e_flags)
    return {
        "reports": reports,
        "individual_positive_intersection": sorted(individual_intersection),
        "pred_b_query_component": bool(pred_b),
        "pred_c_query_advantage": bool(pred_c),
        "pred_d_query_composition": bool(pred_d),
        "pred_e_selectivity": bool(pred_e),
        "any_natural_positive_query_relationship": bool(any_natural_positive),
    }


def main():
    started = time.time()
    roles, scale, old_effects, selections, _, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 472, "model_loaded": False,
            "position_outcomes_opened": False, "sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "windows": [row[0] for row in WINDOWS],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung472 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    windows, reconstruction = {}, 0.0
    for name, role, _, _ in WINDOWS:
        payload, masks = roles[role]
        windows[name], error = collect_window(
            model, payload, masks, scale, selections[name], audit_totals, replay,
        )
        reconstruction = max(reconstruction, error)
    analysis = analyze(windows, old_effects, selections)
    forwards = sum(row["forwards"] for row in audit_totals.values())
    empty_error = max(row["empty_patch_max_abs"] for row in windows.values())
    parent_error = 0.0
    for name, window in windows.items():
        indices = selections[name]["rung470_indices"]
        expected = old_effects["windows"][name]["effects"][:, 3, indices]
        parent_error = max(parent_error, float((window["full"] - expected).abs().max()))
    observed_patch_calls = sum(row.get("position_patch_calls", 0) for row in audit_totals.values())
    expected_patch_calls = sum(row["position_patch_calls"] for row in windows.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and empty_error == 0 and parent_error <= 1e-9
        and forwards == EXPECTED_FORWARDS and observed_patch_calls == expected_patch_calls
    )
    strong_null = bool(
        not pred_a
        or not all(analysis["reports"]["code_validation"][source]["pred_c_window"]
                   for source in SOURCES)
        or not analysis["any_natural_positive_query_relationship"]
    )
    torch.save({
        "schema": "rung472_exact_position_effects_v1", "windows": windows,
        "selections": {name: selections[name] for name, _, _, _ in WINDOWS},
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": 472,
        "claim_level": "exact_query_position_causal_intervention_test",
        "input_identity": metadata,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                   "raw_tokens_logits_or_hidden_states_included": False},
        "analysis": analysis, "native_replay": replay,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "empty_position_mask_max_abs": empty_error,
        "full_prefix_parent_max_abs_error_nat": parent_error,
        "audit_totals": audit_totals,
        "execution_price": {
            "outer_forwards": forwards, "position_patch_calls": observed_patch_calls,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_query_component': analysis["pred_b_query_component"],
        'pred_c_query_advantage': analysis["pred_c_query_advantage"],
        'pred_d_query_composition': analysis["pred_d_query_composition"],
        'pred_e_selectivity': analysis["pred_e_selectivity"],
        "strong_null": strong_null, "runtime_s": time.time() - started,
        "next_step": (
            "within_query_downstream_defined_split"
            if pred_a and all(analysis[key] for key in (
                "pred_b_query_component", "pred_c_query_advantage",
                "pred_d_query_composition", "pred_e_selectivity",
            )) else "nonlinear_query_context_interaction_or_register_specific_regimes"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 472,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "analysis": analysis,
        "instrument": {"replay": replay, "factor_error": reconstruction,
                       "empty_error": empty_error, "parent_error": parent_error,
                       "forwards": forwards, "patch_calls": observed_patch_calls},
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
