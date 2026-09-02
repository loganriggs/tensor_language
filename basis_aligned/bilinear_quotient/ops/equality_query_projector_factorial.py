"""PROJECTOR-FORM QUERY FACTORIAL (Claude parallel-probe lane) -- the third
intervention semantics for the 472/474 frame-relativity dispute, registered
frozen before any outcome is opened.

Rung 474 proved single-MLP query effects are causal-coordinate invariant while
every pair/triple Mobius dividend is frame-relative: the code-register N/H
total-interaction context cosine is +.9793 under fixed REPLACEMENT (472/473)
and -.8757 under frozen SUBTRACTION (474).  Both forms remove the full frozen
delta; they differ in what downstream sites see after an upstream edit.  The
PROJECTOR form registered here removes, at each patched site and query
position, only the component of the CURRENT deviation (current product minus
frozen absent product) along the frozen per-position delta DIRECTION:

    updated = current - dhat * <current - absent, dhat>,   dhat = delta/|delta|

Properties fixed by construction (not measured): (i) it depends only on the
delta's direction, so it is invariant to any positive rescaling of the frozen
delta (checked in-run by a doubled-delta replicate); (ii) at any SINGLETON
removal the current deviation equals the frozen delta exactly (nothing
upstream is patched), so projector singletons must reproduce rung474's
singleton effects to float exactness -- a theorem-grade bridge to the frozen
474 per-token bundle that fails loudly if the hook is wrong; (iii) under
COMPOSED removal it keeps the orthogonal part of upstream-induced change,
which is exactly where the two prior frames disagree.

Arms (named): windows code_validation / natural_wave0 / natural_wave1
(474's frozen windows, rows, selections, scale); sources N (native matcher)
and H (transplanted); subsets the 7 nonempty site subsets of
{m8, m9, m12}; replicate arm DOUBLED-DELTA = first batch of each window
rerun with frozen deltas scaled x2 (projector effects must be identical).

Frozen predictions
------------------
pred_a (instrument + theorem bridge): replay max_abs 0 within float (bar
    <= 1e-6); empty-mask patch equals the source forward (max_abs <= 1e-6);
    factor reconstruction <= 1e-12; Mobius closure <= 1e-12; forward/patch
    counts formula-exact; SINGLETON projector effects equal the rung474
    bundle singletons at max abs <= 1e-6 nat on every window/source; the
    DOUBLED-DELTA replicate reproduces first-batch subset effects at
    max abs <= 1e-9 nat.
pred_b (code science -- the open question): code_validation total
    all-three-minus-singletons interaction context N/H cosine >= .80 under
    the projector form (replacement measured +.9793, subtraction -.8757;
    both context norms reported; claim void if either norm < .003, the 474
    materiality floor).
pred_c (natural registers): BOTH natural waves have projector-form N/H
    total-interaction context cosine >= .50 with both norms >= .003
    (replacement measured -.8748/+.0742, subtraction +.281/-.143 -- every
    prior frame failed at least one wave).
Null: pred_a holds but pred_b fails -- the projector form is merely a third
frame; pair/triple composition remains individuated under every tried
removal semantics (replacement, subtraction, projection); only singletons
are bankable; the intervention-convention question CLOSES NEGATIVE and no
further semantics sweeps are licensed.

Price: 72 batches x 35 forwards = 2,520 forwards + 42 doubled-delta
replicate forwards = 2,562 total, ~60-90s GPU, 0 deployed parameters saved
or added; receipt only (no tokens/logits/hidden states); no validation
family, odd-root, or SEALED object is touched.
"""
# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
OUT = BQ / "equality_query_projector_factorial_results.json"
SUB_RESULT = BQ / "equality_query_subtractive_factorial_rung474_results.json"
SUB_BUNDLE = BQ / "equality_query_subtractive_factorial_rung474_per_token.pt"

for _p in (ROOT, ROOT / "basis_aligned/polynomial_causal", ROOT / "basis_aligned/qk_mdl", BQ, OPS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import equality_query_subtractive_factorial_rung474 as sub474  # noqa: E402
from receipt import dump  # noqa: E402

parent = sub474.parent
position_parent = sub474.position_parent
product_parent = sub474.product_parent
source_parent = sub474.source_parent
audit_parent = sub474.audit_parent
form_parent = sub474.form_parent

SOURCES = sub474.SOURCES
MODULES = sub474.MODULES
SITES = sub474.SITES
WINDOWS = sub474.WINDOWS
BATCH = sub474.BATCH
SUBSETS = sub474.SUBSETS
SUBSET_NAMES = sub474.SUBSET_NAMES
SINGLE_INDICES = sub474.SINGLE_INDICES
PAIR_INDICES = sub474.PAIR_INDICES
UNION_INDEX = sub474.UNION_INDEX
FORWARDS_PER_BATCH = sub474.FORWARDS_PER_BATCH
EXPECTED_BATCHES = sub474.EXPECTED_BATCHES
EXPECTED_FORWARDS = EXPECTED_BATCHES * FORWARDS_PER_BATCH + len(WINDOWS) * len(SOURCES) * len(SUBSETS)
NORM_FLOOR = 1e-12
MATERIALITY = .003


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_projector_patch(model, tokens, *, arm, scale, deltas, absents, sites,
                        position_mask, delta_scale=1.0):
    """474's subtractive patch with the update rule swapped to the projector
    form: remove only the component of (current - absent) along the frozen
    per-position delta direction.  delta_scale multiplies the frozen delta
    BEFORE normalization; the projector must be invariant to it."""
    sites = tuple(sites)
    if set(sites) - set(SITES) or set(deltas) != set(sites) or set(absents) != set(sites):
        raise ValueError("malformed projector patch sites")
    handles, calls = [], {site: 0 for site in sites}
    for layer, site in zip(MODULES, SITES):
        if site not in sites:
            continue
        delta = deltas[site]
        absent = absents[site]
        down = model.transformer.h[layer].mlp.Down

        def hook(_module, inputs, name=site, frozen_delta=delta, frozen_absent=absent):
            if calls[name] != 0:
                raise RuntimeError(f"duplicate projector patch at {name}")
            product = inputs[0]
            if frozen_delta.shape != product.shape or frozen_delta.device != product.device \
                    or frozen_delta.dtype != torch.float32:
                raise RuntimeError(f"projector delta mismatch at {name}")
            updated = product.clone()
            current = product[position_mask].float()
            direction = frozen_delta[position_mask] * float(delta_scale)
            norms = direction.norm(dim=-1, keepdim=True)
            unit = torch.where(norms > NORM_FLOOR, direction / norms.clamp_min(NORM_FLOOR),
                               torch.zeros_like(direction))
            deviation = current - frozen_absent[position_mask].float()
            component = (deviation * unit).sum(-1, keepdim=True)
            updated[position_mask] = (current - unit * component).to(product.dtype)
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
        raise RuntimeError("not every projector patch fired exactly once")
    return logits, calls, audit, error


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
    scale_invariance_max = 0.0
    first_batch = True
    for start in range(first_doc, last_doc, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = source_parent.run_forward(model, tokens, arm="native")
        audit_parent._record_audit(
            audit_totals, "proj:native", audit, analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = source_parent.run_forward(model, tokens, arm="replay")
        audit_parent._record_audit(
            audit_totals, "proj:replay", audit, analytical=True, captures=0, patches=0,
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
        sub474._record(audit_totals, "proj:absent", audit)
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
            sub474._record(audit_totals, f"proj:source:{source}", audit)
            reconstruction = max(reconstruction, error)
            source_nll = position_parent._nll(source_logits, batch_rows)
            deltas = {
                site: source_products[site].float() - absents[site]
                for site in SITES
            }
            false_mask = torch.zeros_like(tokens, dtype=torch.bool)
            empty_logits, calls, audit, error = run_projector_patch(
                model, tokens, arm=arm, scale=scale, deltas=deltas, absents=absents,
                sites=SITES, position_mask=false_mask,
            )
            count = sum(calls.values())
            sub474._record(audit_totals, f"proj:empty:{source}", audit, count)
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
                    patched, calls, audit, error = run_projector_patch(
                        model, tokens, arm=arm, scale=scale,
                        deltas={site: deltas[site] for site in sites},
                        absents={site: absents[site] for site in sites},
                        sites=sites, position_mask=query_mask,
                    )
                    count = sum(calls.values())
                    sub474._record(
                        audit_totals, f"proj:{source}:slot{slot}:{SUBSET_NAMES[subset_index]}",
                        audit, count,
                    )
                    patch_calls += count
                    reconstruction = max(reconstruction, error)
                    damage = position_parent._nll(patched, batch_rows) - source_nll
                    for output_index, local_doc, query in chosen:
                        effects[si, subset_index, output_index] = float(damage[local_doc, query])
                    if first_batch and slot == 0:
                        first_slot_effects[subset_index] = (
                            [row[:2] for row in chosen],
                            damage.detach().clone(),
                            query_mask,
                        )
            if first_batch:
                # DOUBLED-DELTA replicate: projector effects must be identical.
                for subset_index, indices in enumerate(SUBSETS):
                    sites = tuple(SITES[index] for index in indices)
                    chosen_meta, base_damage, query_mask = first_slot_effects[subset_index]
                    patched, calls, audit, error = run_projector_patch(
                        model, tokens, arm=arm, scale=scale,
                        deltas={site: deltas[site] for site in sites},
                        absents={site: absents[site] for site in sites},
                        sites=sites, position_mask=query_mask, delta_scale=2.0,
                    )
                    sub474._record(
                        audit_totals, f"proj:doubled:{source}:{SUBSET_NAMES[subset_index]}",
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
        "empty_patch_max_abs": empty_error,
        "projector_patch_calls": patch_calls,
        "doubled_delta_max_abs_nat": scale_invariance_max,
    }, reconstruction


def analyze(windows, sub_bundle, old_effects, selections):
    reports = {}
    singleton_bridge_error = 0.0
    closure_error = 0.0
    code_cosines, natural_cosines = [], []
    norms_ok = True
    for name, window in windows.items():
        sub_effects = sub_bundle["windows"][name]["effects"].double()
        old_window = old_effects["windows"][name]
        indices = selections[name]["rung470_indices"]
        reports[name] = {}
        interaction_vectors = {}
        interaction_norms = {}
        for si, source in enumerate(SOURCES):
            effects = window["effects"][si]
            mains = effects[list(SINGLE_INDICES)]
            union = effects[UNION_INDEX]
            sub_mains = sub_effects[si, list(SINGLE_INDICES)]
            singleton_bridge_error = max(
                singleton_bridge_error, float((mains - sub_mains).abs().max()),
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
                "singleton_bridge_max_abs_error_nat": float((mains - sub_mains).abs().max()),
                "total_interaction_context_vector": total_context.tolist(),
                "total_interaction_context_norm": total_norm,
                "triple_context_norm": float(torch.linalg.vector_norm(
                    position_parent._cell_vector(triple, old_window, indices))),
                "mobius_closure_max_abs": float((reconstructed - union).abs().max()),
            }
        cosine = form_parent._cosine(interaction_vectors["N"], interaction_vectors["H"])
        material = all(interaction_norms[s] >= MATERIALITY for s in SOURCES)
        reports[name]["projector_interaction_source_cosine"] = cosine
        reports[name]["both_norms_material"] = bool(material)
        if name.startswith("natural"):
            natural_cosines.append((cosine, material))
        else:
            code_cosines.append((cosine, material))
            norms_ok = norms_ok and material
    return reports, singleton_bridge_error, closure_error, code_cosines, natural_cosines


def main():
    started = time.time()
    (roles, scale, old_effects, selections, old_position,
     old_factorial, metadata) = sub474.validate_inputs()
    sub_result = json.loads(SUB_RESULT.read_text())
    expected = {
        "pred_a_instrument": True, "pred_b_coordinate_stable": False,
        "pred_c_state_mixing": False, "pred_d_register_persists": False,
        "pred_e_natural_h_half_stable": False, "strong_null": False,
    }
    if sub_result.get("rung") != 474 or any(
            sub_result.get(key) is not value for key, value in expected.items()):
        raise RuntimeError("rung474 registered verdict changed")
    sub_bundle = torch.load(SUB_BUNDLE, map_location="cpu", weights_only=True)
    if sub_bundle.get("schema") != "rung474_subtractive_query_factorial_v1":
        raise RuntimeError("rung474 bundle schema changed")
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": "projector_factorial",
            "model_loaded": False, "projector_outcomes_opened": False,
            "sealed_opened": False, "expected_forwards": EXPECTED_FORWARDS,
        }))
        return

    model, checkpoint = sub474.facade.load_bilin18()
    audit_totals, replay = {}, {"max_abs": 0.0, "relative_squared": 0.0}
    windows, reconstruction = {}, 0.0
    for name, _, _, _ in WINDOWS:
        window, error = collect_window(
            model, roles[name], scale, selections[name], audit_totals, replay,
        )
        windows[name] = window
        reconstruction = max(reconstruction, error)
    reports, bridge_error, closure_error, code_cos, natural_cos = analyze(
        windows, sub_bundle, old_effects, selections,
    )
    forwards = sum(row.get("forwards", 0) for row in audit_totals.values())
    doubled_max = max(w["doubled_delta_max_abs_nat"] for w in windows.values())
    empty_max = max(w["empty_patch_max_abs"] for w in windows.values())
    pred_a = bool(
        replay["max_abs"] <= 1e-6 and empty_max <= 1e-6
        and reconstruction <= 1e-12 and closure_error <= 1e-12
        and bridge_error <= 1e-6 and doubled_max <= 1e-9
        and forwards == EXPECTED_FORWARDS
    )
    pred_b = bool(all(c >= .80 and m for c, m in code_cos))
    pred_c = bool(all(c >= .50 and m for c, m in natural_cos))
    null_fired = bool(pred_a and not pred_b)
    result = {
        "status": "complete", "rung": "equality_query_projector_factorial",
        "coordinate": "project_out_frozen_delta_direction_from_current_deviation",
        "rung474_result_sha256": sha256(SUB_RESULT),
        "rung474_bundle_sha256": sha256(SUB_BUNDLE),
        "metadata": metadata,
        "reports": reports,
        "replay": replay,
        "empty_patch_max_abs": empty_max,
        "factor_reconstruction_max": reconstruction,
        "mobius_closure_max_abs": closure_error,
        "singleton_bridge_max_abs_error_nat": bridge_error,
        "doubled_delta_max_abs_nat": doubled_max,
        "forwards": forwards, "expected_forwards": EXPECTED_FORWARDS,
        "measured_context": {
            "replacement_code_cosine": .9793, "subtractive_code_cosine": -.8757,
            "replacement_natural_cosines": [-.8748, .0742],
            "subtractive_natural_cosines": [.281, -.143],
        },
        'pred_a_instrument_and_theorem_bridge': pred_a,
        'pred_b_code_projector_alignment': pred_b,
        'pred_c_natural_projector_alignment': pred_c,
        'strong_null_third_frame_composition_individuated': null_fired,
        "raw_tokens_logits_or_hidden_states_included": False,
        "validation_or_sealed_opened": False,
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(f"pred_a={pred_a} pred_b={pred_b} pred_c={pred_c} null={null_fired} "
          f"bridge={bridge_error:.2e} doubled={doubled_max:.2e} "
          f"cosines code={[round(c,4) for c,_ in code_cos]} "
          f"natural={[round(c,4) for c,_ in natural_cos]} ({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
