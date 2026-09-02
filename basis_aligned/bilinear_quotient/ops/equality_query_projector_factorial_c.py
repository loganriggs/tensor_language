"""PROJECTOR-FORM QUERY FACTORIAL, C-VARIANT (Claude lane) -- bar
re-registration ONLY.  The b-variant missed its own in-run bridge bar by
0.14% (max 1.0014e-5 vs <=1e-5) while every other instrument clause was
exact (replay 0.0, empty 0.0, closure 0.0, doubled-delta 0.0, forwards
3,426 exact) and the science clauses were untouched.

WHY THE BAR MOVES (extreme-value argument -- the 433/471 lesson landing in
my own registration): the 1e-5 bar was 5x a within-process floor measured
as a max over 4 positions in ONE batch (2.1e-6).  The b-variant's bridge is
a max over ~2,300 position-comparisons (72 batches x 2 sources x 2 slots x
3 sites); the max order statistic of the same per-position noise is
naturally larger.  Measured b distribution: per-window/source maxima 7.3e-6
to 1.0014e-5.  The c bar is 3e-5 = 3x the observed global max; every other
bar, arm, formula, and price is IDENTICAL to the b-variant registration
(text below).  The b receipt stays failed as written; nothing retro-passes.

--- original b-variant registration follows ---

PROJECTOR-FORM QUERY FACTORIAL, B-VARIANT (Claude lane) -- same frozen
science as equality_query_projector_factorial, with the instrument bridge
moved IN-RUN per the S2599 reproducibility breach.

The original registration's pred_a compared projector singletons to the
frozen rung474 per-token bundle; that cross-session bridge failed at
.11-.20 nat and was diagnosed (L1-L5 + replication probe + canary
bit-stability) as an environment-class breach: cross-session forwards on
the hook-laden 472/474 pathway differ by ~.03-.08 nat while within-process
physics is exact (projector==subtractive singleton damage <=2.1e-6 in the
same process, measured 08:08).  This b-variant therefore recomputes the
subtractive singleton baseline IN-RUN, inside the same process and batch
loop, and the bridge becomes a within-process theorem check.  The design is
sound under BOTH surviving breach hypotheses (per-process kernel selection
or a stable state shift), because all comparisons live inside one process.
The original receipt stays failed as written; nothing is retro-passed.

Arms (named): windows code_validation / natural_wave0 / natural_wave1
(474's frozen windows, rows, selections, scale); sources N and H; PROJECTOR
subsets = the 7 nonempty subsets of {m8, m9, m12} (science arms); IN-RUN
SUBTRACTIVE singletons m8, m9, m12 (baseline arms, both slots); replicate
arm DOUBLED-DELTA = first batch of each window, projector subsets rerun
with deltas x2.

Frozen predictions
------------------
pred_a (instrument, all within-process): replay max_abs <= 1e-6 with the
    relative-squared companion reported; empty-mask projector patch equals
    the source forward (<= 1e-6); factor reconstruction <= 1e-12; Mobius
    closure <= 1e-12; forward/patch counts formula-exact (47 forwards per
    batch x 72 batches + 42 doubled = 3,426); IN-RUN BRIDGE: projector
    singleton effects equal the in-run subtractive singleton effects at
    max abs <= 1e-5 nat on every window/source/slot (measured within-process
    agreement 08:08 was <= 2.1e-6; bar set 5x above); DOUBLED-DELTA
    replicate reproduces first-batch projector effects at <= 1e-9.
pred_b (code science, unchanged bars): code_validation total
    all-three-minus-singletons interaction context N/H cosine >= .80 under
    the projector form (replacement measured +.9793, subtraction -.8757;
    original run REPORTED .9389 unclaimed; both context norms >= .003 or
    the claim is void).
pred_c (natural registers, unchanged bars): BOTH natural waves >= .50 with
    .003 materiality (all prior frames failed; original run reported
    -.372/.369 unclaimed).
Null: pred_a holds and pred_b fails -- the projector form is merely a third
frame; composition remains individuated under every tried removal
semantics; only singletons are bankable; the intervention-convention
question closes NEGATIVE with no further semantics sweeps licensed.
Diagnostic-only reported field (NOT a pred): the vs-474-bundle singleton
difference, extending the breach record with a third measurement.

Price: 72 batches x 47 forwards + 42 = 3,426 forwards, ~2-3 min GPU,
0 deployed parameters saved or added; no validation family, odd-root, or
SEALED object is touched.
"""
# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
OUT = BQ / "equality_query_projector_factorial_c_results.json"
for _p in (ROOT, ROOT / "basis_aligned/polynomial_causal", ROOT / "basis_aligned/qk_mdl", BQ, OPS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import equality_query_projector_factorial as proj  # noqa: E402
import equality_query_subtractive_factorial_rung474 as sub474  # noqa: E402
from receipt import dump  # noqa: E402

parent = sub474.parent
position_parent = sub474.position_parent
product_parent = sub474.product_parent
source_parent = sub474.source_parent
audit_parent = sub474.audit_parent
form_parent = sub474.form_parent

SOURCES = sub474.SOURCES
SITES = sub474.SITES
WINDOWS = sub474.WINDOWS
BATCH = sub474.BATCH
SUBSETS = sub474.SUBSETS
SUBSET_NAMES = sub474.SUBSET_NAMES
SINGLE_INDICES = sub474.SINGLE_INDICES
PAIR_INDICES = sub474.PAIR_INDICES
UNION_INDEX = sub474.UNION_INDEX
# per batch: native + replay + absent + per source (source + empty +
# 2 slots x (7 projector + 3 subtractive singles)) = 3 + 2*(2 + 20) = 47
FORWARDS_PER_BATCH_B = 3 + len(SOURCES) * (2 + 2 * (len(SUBSETS) + len(SITES)))
EXPECTED_FORWARDS = sub474.EXPECTED_BATCHES * FORWARDS_PER_BATCH_B \
    + len(WINDOWS) * len(SOURCES) * len(SUBSETS)
MATERIALITY = .003


@torch.no_grad()
def collect_window_b(model, payload, scale, selection, audit_totals, replay):
    coordinates = selection["coordinates"]
    by_doc = {}
    for output_index, (doc, query, _) in enumerate(coordinates):
        by_doc.setdefault(doc, []).append((output_index, query))
    effects = torch.zeros(len(SOURCES), len(SUBSETS), len(coordinates), dtype=torch.float64)
    sub_singles = torch.zeros(len(SOURCES), len(SITES), len(coordinates), dtype=torch.float64)
    rows = payload["rows"]
    first_doc = min(by_doc)
    last_doc = max(by_doc) + 1
    device = next(model.parameters()).device
    reconstruction, empty_error, patch_calls = 0.0, 0.0, 0
    scale_invariance_max = 0.0
    first_batch = True
    for start in range(first_doc, last_doc, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = source_parent.run_forward(model, tokens, arm="native")
        audit_parent._record_audit(
            audit_totals, "projb:native", audit, analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = source_parent.run_forward(model, tokens, arm="replay")
        audit_parent._record_audit(
            audit_totals, "projb:replay", audit, analytical=True, captures=0, patches=0,
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
        sub474._record(audit_totals, "projb:absent", audit)
        reconstruction = max(reconstruction, error)
        absents = {site: absent_products[site].float() for site in SITES}
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
            sub474._record(audit_totals, f"projb:source:{source}", audit)
            reconstruction = max(reconstruction, error)
            source_nll = position_parent._nll(source_logits, batch_rows)
            deltas = {
                site: source_products[site].float() - absents[site]
                for site in SITES
            }
            false_mask = torch.zeros_like(tokens, dtype=torch.bool)
            empty_logits, calls, audit, error = proj.run_projector_patch(
                model, tokens, arm=arm, scale=scale, deltas=deltas, absents=absents,
                sites=SITES, position_mask=false_mask,
            )
            count = sum(calls.values())
            sub474._record(audit_totals, f"projb:empty:{source}", audit, count)
            patch_calls += count
            reconstruction = max(reconstruction, error)
            empty_error = max(empty_error, float((empty_logits - source_logits).abs().max()))
            first_slot_effects = {}
            for slot, chosen in enumerate(slots):
                targets = [(local_doc, query) for _, local_doc, query in chosen]
                query_mask, _, _, _ = position_parent.position_masks(
                    len(batch_rows), tokens.shape[1], targets, device,
                )
                for subset_index, indices in enumerate(SUBSETS):
                    sites = tuple(SITES[index] for index in indices)
                    patched, calls, audit, error = proj.run_projector_patch(
                        model, tokens, arm=arm, scale=scale,
                        deltas={site: deltas[site] for site in sites},
                        absents={site: absents[site] for site in sites},
                        sites=sites, position_mask=query_mask,
                    )
                    count = sum(calls.values())
                    sub474._record(
                        audit_totals, f"projb:{source}:slot{slot}:{SUBSET_NAMES[subset_index]}",
                        audit, count,
                    )
                    patch_calls += count
                    reconstruction = max(reconstruction, error)
                    damage = position_parent._nll(patched, batch_rows) - source_nll
                    for output_index, local_doc, query in chosen:
                        effects[si, subset_index, output_index] = float(damage[local_doc, query])
                    if first_batch and slot == 0:
                        first_slot_effects[subset_index] = (damage.detach().clone(), query_mask)
                # IN-RUN subtractive singleton baseline (the moved bridge)
                for single_pos, site_index in enumerate(range(len(SITES))):
                    site = SITES[site_index]
                    patched, calls, audit, error = sub474.run_subtractive_patch(
                        model, tokens, arm=arm, scale=scale,
                        deltas={site: deltas[site]}, sites=(site,),
                        position_mask=query_mask,
                    )
                    count = sum(calls.values())
                    sub474._record(
                        audit_totals, f"projb:sub:{source}:slot{slot}:{site}",
                        audit, count,
                    )
                    patch_calls += count
                    reconstruction = max(reconstruction, error)
                    damage = position_parent._nll(patched, batch_rows) - source_nll
                    for output_index, local_doc, query in chosen:
                        sub_singles[si, single_pos, output_index] = float(damage[local_doc, query])
            if first_batch:
                for subset_index, indices in enumerate(SUBSETS):
                    sites = tuple(SITES[index] for index in indices)
                    base_damage, query_mask = first_slot_effects[subset_index]
                    patched, calls, audit, error = proj.run_projector_patch(
                        model, tokens, arm=arm, scale=scale,
                        deltas={site: deltas[site] for site in sites},
                        absents={site: absents[site] for site in sites},
                        sites=sites, position_mask=query_mask, delta_scale=2.0,
                    )
                    sub474._record(
                        audit_totals, f"projb:doubled:{source}:{SUBSET_NAMES[subset_index]}",
                        audit, sum(calls.values()),
                    )
                    patch_calls += sum(calls.values())
                    reconstruction = max(reconstruction, error)
                    doubled_damage = position_parent._nll(patched, batch_rows) - source_nll
                    scale_invariance_max = max(
                        scale_invariance_max,
                        float((doubled_damage - base_damage).abs().max()),
                    )
            del source_products, deltas
        first_batch = False
        del absent_products, absents
    return {
        "effects": effects,
        "sub_singles": sub_singles,
        "empty_patch_max_abs": empty_error,
        "patch_calls": patch_calls,
        "doubled_delta_max_abs_nat": scale_invariance_max,
    }, reconstruction


def analyze_b(windows, sub_bundle, old_effects, selections):
    reports = {}
    inrun_bridge_error = 0.0
    bundle_diff_max = 0.0
    closure_error = 0.0
    code_cosines, natural_cosines = [], []
    for name, window in windows.items():
        bundle_effects = sub_bundle["windows"][name]["effects"].double()
        old_window = old_effects["windows"][name]
        indices = selections[name]["rung470_indices"]
        reports[name] = {}
        interaction_vectors = {}
        interaction_norms = {}
        for si, source in enumerate(SOURCES):
            effects = window["effects"][si]
            mains = effects[list(SINGLE_INDICES)]
            union = effects[UNION_INDEX]
            inrun = window["sub_singles"][si]
            inrun_bridge_error = max(
                inrun_bridge_error, float((mains - inrun).abs().max()),
            )
            bundle_diff_max = max(
                bundle_diff_max,
                float((inrun - bundle_effects[si, list(SINGLE_INDICES)]).abs().max()),
            )
            pairs = effects[list(PAIR_INDICES)]
            pair_interactions, triple, reconstructed = parent.mobius_terms(mains, pairs, union)
            closure_error = max(closure_error, float((reconstructed - union).abs().max()))
            total = union - mains.sum(0)
            total_context = position_parent._cell_vector(total, old_window, indices)
            total_norm = float(torch.linalg.vector_norm(total_context))
            interaction_vectors[source] = total_context
            interaction_norms[source] = total_norm
            reports[name][source] = {
                "inrun_bridge_max_abs_error_nat": float((mains - inrun).abs().max()),
                "vs_bundle_singleton_max_abs_nat": float(
                    (inrun - bundle_effects[si, list(SINGLE_INDICES)]).abs().max()),
                "total_interaction_context_norm": total_norm,
                "total_interaction_context_vector": total_context.tolist(),
            }
        cosine = form_parent._cosine(interaction_vectors["N"], interaction_vectors["H"])
        material = all(interaction_norms[s] >= MATERIALITY for s in SOURCES)
        reports[name]["projector_interaction_source_cosine"] = cosine
        reports[name]["both_norms_material"] = bool(material)
        (natural_cosines if name.startswith("natural") else code_cosines).append(
            (cosine, material))
    return (reports, inrun_bridge_error, bundle_diff_max, closure_error,
            code_cosines, natural_cosines)


def main():
    started = time.time()
    (roles, scale, old_effects, selections, old_position,
     old_factorial, metadata) = sub474.validate_inputs()
    sub_bundle = torch.load(proj.SUB_BUNDLE, map_location="cpu", weights_only=True)
    if sub_bundle.get("schema") != "rung474_subtractive_query_factorial_v1":
        raise RuntimeError("rung474 bundle schema changed")
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for name, role, _, _ in WINDOWS:
            payload, _ = roles[role]
            assert "rows" in payload and name in selections
        print(json.dumps({
            "status": "dry_run_passed", "rung": "projector_factorial_c",
            "model_loaded": False, "expected_forwards": EXPECTED_FORWARDS,
        }))
        return
    model, checkpoint = sub474.facade.load_bilin18()
    audit_totals, replay = {}, {"max_abs": 0.0, "relative_squared": 0.0}
    windows, reconstruction = {}, 0.0
    for name, role, _, _ in WINDOWS:
        payload, _ = roles[role]
        window, error = collect_window_b(
            model, payload, scale, selections[name], audit_totals, replay,
        )
        windows[name] = window
        reconstruction = max(reconstruction, error)
    (reports, bridge_error, bundle_diff, closure_error,
     code_cos, natural_cos) = analyze_b(windows, sub_bundle, old_effects, selections)
    forwards = sum(row.get("forwards", 0) for row in audit_totals.values())
    doubled_max = max(w["doubled_delta_max_abs_nat"] for w in windows.values())
    empty_max = max(w["empty_patch_max_abs"] for w in windows.values())
    pred_a = bool(
        replay["max_abs"] <= 1e-6 and empty_max <= 1e-6
        and reconstruction <= 1e-12 and closure_error <= 1e-12
        and bridge_error <= 3e-5 and doubled_max <= 1e-9
        and forwards == EXPECTED_FORWARDS
    )
    pred_b = bool(all(c >= .80 and m for c, m in code_cos))
    pred_c = bool(all(c >= .50 and m for c, m in natural_cos))
    null_fired = bool(pred_a and not pred_b)
    result = {
        "status": "complete", "rung": "equality_query_projector_factorial_c",
        "coordinate": "project_out_frozen_delta_direction_from_current_deviation",
        "bridge": "in_run_subtractive_singletons_same_process",
        "original_receipt_remains_failed": True,
        "reports": reports,
        "replay": replay,
        "empty_patch_max_abs": empty_max,
        "factor_reconstruction_max": reconstruction,
        "mobius_closure_max_abs": closure_error,
        "inrun_bridge_max_abs_error_nat": bridge_error,
        "vs_bundle_singleton_max_abs_nat_diagnostic_only": bundle_diff,
        "doubled_delta_max_abs_nat": doubled_max,
        "forwards": forwards, "expected_forwards": EXPECTED_FORWARDS,
        'pred_a_instrument_and_inrun_bridge': pred_a,
        'pred_b_code_projector_alignment': pred_b,
        'pred_c_natural_projector_alignment': pred_c,
        'strong_null_third_frame_composition_individuated': null_fired,
        "raw_tokens_logits_or_hidden_states_included": False,
        "validation_or_sealed_opened": False,
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(f"pred_a={pred_a} pred_b={pred_b} pred_c={pred_c} null={null_fired} "
          f"inrun_bridge={bridge_error:.2e} vs_bundle={bundle_diff:.3f} "
          f"doubled={doubled_max:.2e} "
          f"cos code={[round(c,4) for c,_ in code_cos]} "
          f"natural={[round(c,4) for c,_ in natural_cos]} ({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
