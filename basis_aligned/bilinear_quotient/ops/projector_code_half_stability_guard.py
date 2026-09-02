"""PROJECTOR CODE HALF-STABILITY GUARD (Claude lane) -- guard rung for the
S2602 c-variant claim, per the standing guard discipline (421/425/427: every
major positive gets an independent stress test before anyone builds on it).

The c-variant claimed: under projector-form removal, code-register N/H
total-interaction context cosine = .938876 (pooled over code_validation docs
96:192).  The recurring fragility of this arc has been DOCUMENT-HALF
instability (473/474 natural-H half-sign failures).  This guard recomputes
the same projector collection on code_validation only and scores the
alignment PER DOCUMENT HALF (docs 96:144 vs 144:192, the 48/48 convention of
the 47x-era windows).

Arms (named): projector subsets {m8,m9,m12} x sizes 1..3 (7 subsets), both
slots, sources N and H; in-run subtractive singleton baseline (3 sites x 2
slots per source); DOUBLED-DELTA replicate on the first batch; halves
half0 = docs < 144, half1 = docs >= 144.

Frozen predictions
------------------
pred_a (instrument + reproduction): replay and empty-mask <= 1e-6 (relative
    companion reported); factor <= 1e-12; Mobius closure <= 1e-12; in-run
    bridge <= 3e-5 (the c bar); doubled-delta <= 1e-9; forwards == 1,142
    (24 batches x 47 + 14); POOLED code N/H cosine within .015 wobble of the
    c-variant's .938876 (post-shift determinism predicts near-exact).
pred_b (the guard): N/H total-interaction context cosine >= .80 in BOTH
    document halves.
pred_c (materiality per half): all four half x source interaction context
    norms >= .003.
Null: pred_a holds and (pred_b or pred_c fails) -- the .9389 alignment is a
POOLED artifact: the S2602 claim is DEMOTED to pooled-code-only pending
diagnosis (the pooled claim itself stays scored as registered; nothing is
retracted by this guard).

Price: 1,142 forwards, ~45-90s GPU, 0 deployed parameters; code_validation
window only; no validation family, odd-root, or SEALED object is touched.
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
OUT = BQ / "projector_code_half_stability_guard_results.json"
for _p in (ROOT, ROOT / "basis_aligned/polynomial_causal", ROOT / "basis_aligned/qk_mdl", BQ, OPS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import equality_query_projector_factorial_b as bmod  # noqa: E402
import equality_query_subtractive_factorial_rung474 as sub474  # noqa: E402
from receipt import dump  # noqa: E402

position_parent = sub474.position_parent
form_parent = sub474.form_parent
parent = sub474.parent

WINDOW = "code_validation"
HALF_BOUNDARY = 144   # window 96:192, 48/48 halves
C_POOLED_COSINE = 0.938876
EXPECTED_FORWARDS = 24 * bmod.FORWARDS_PER_BATCH_B + len(sub474.SOURCES) * len(sub474.SUBSETS)
MATERIALITY = .003


def main():
    started = time.time()
    (roles, scale, old_effects, selections, old_position,
     old_factorial, metadata) = sub474.validate_inputs()
    role = next(role for name, role, _, _ in sub474.WINDOWS if name == WINDOW)
    payload, _ = roles[role]
    selection = selections[WINDOW]
    coordinates = selection["coordinates"]
    docs = torch.tensor([row[0] for row in coordinates])
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert "rows" in payload and int(docs.min()) >= 96 and int(docs.max()) < 192
        halves = [(docs < HALF_BOUNDARY).sum().item(), (docs >= HALF_BOUNDARY).sum().item()]
        assert min(halves) >= 40, f"half support too thin: {halves}"
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "projector_code_half_stability_guard",
                          "model_loaded": False, "half_support": halves,
                          "expected_forwards": EXPECTED_FORWARDS}))
        return
    model, checkpoint = sub474.facade.load_bilin18()
    audit_totals, replay = {}, {"max_abs": 0.0, "relative_squared": 0.0}
    window, reconstruction = bmod.collect_window_b(
        model, payload, scale, selection, audit_totals, replay,
    )
    forwards = sum(row.get("forwards", 0) for row in audit_totals.values())
    old_window = old_effects["windows"][WINDOW]
    indices = torch.as_tensor(selection["rung470_indices"])
    half_masks = (docs < HALF_BOUNDARY, docs >= HALF_BOUNDARY)

    bridge = 0.0
    closure = 0.0
    result_reports = {}
    half_cosines, half_norms = [], {}
    pooled_cosine = None
    vectors = {}
    for si, source in enumerate(sub474.SOURCES):
        effects = window["effects"][si]
        mains = effects[list(sub474.SINGLE_INDICES)]
        union = effects[sub474.UNION_INDEX]
        bridge = max(bridge, float((mains - window["sub_singles"][si]).abs().max()))
        pair_interactions, triple, reconstructed = parent.mobius_terms(
            mains, effects[list(sub474.PAIR_INDICES)], union)
        closure = max(closure, float((reconstructed - union).abs().max()))
        total = union - mains.sum(0)
        vectors[source] = {"pooled": position_parent._cell_vector(total, old_window, indices)}
        for hi, mask in enumerate(half_masks):
            vec = position_parent._cell_vector(total[mask], old_window, indices[mask])
            vectors[source][f"half{hi}"] = vec
            half_norms[f"{source}_half{hi}"] = float(torch.linalg.vector_norm(vec))
    pooled_cosine = form_parent._cosine(vectors["N"]["pooled"], vectors["H"]["pooled"])
    for hi in range(2):
        half_cosines.append(form_parent._cosine(
            vectors["N"][f"half{hi}"], vectors["H"][f"half{hi}"]))
    result_reports = {
        "pooled_cosine": pooled_cosine,
        "half_cosines": half_cosines,
        "half_norms": half_norms,
        "half_vectors": {s: {k: v.tolist() for k, v in vectors[s].items()}
                         for s in vectors},
    }
    empty_max = window["empty_patch_max_abs"]
    doubled_max = window["doubled_delta_max_abs_nat"]
    pred_a = bool(
        replay["max_abs"] <= 1e-6 and empty_max <= 1e-6
        and reconstruction <= 1e-12 and closure <= 1e-12
        and bridge <= 3e-5 and doubled_max <= 1e-9
        and forwards == EXPECTED_FORWARDS
        and abs(pooled_cosine - C_POOLED_COSINE) <= .015
    )
    pred_b = bool(all(c >= .80 for c in half_cosines))
    pred_c = bool(all(v >= MATERIALITY for v in half_norms.values()))
    null_fired = bool(pred_a and not (pred_b and pred_c))
    result = {
        "status": "complete", "rung": "projector_code_half_stability_guard",
        "guards_claim": "S2602 c-variant pooled code cosine .938876",
        "reports": result_reports,
        "replay": replay, "empty_patch_max_abs": empty_max,
        "factor_reconstruction_max": reconstruction,
        "mobius_closure_max_abs": closure,
        "inrun_bridge_max_abs_error_nat": bridge,
        "doubled_delta_max_abs_nat": doubled_max,
        "forwards": forwards, "expected_forwards": EXPECTED_FORWARDS,
        'pred_a_instrument_and_pooled_reproduction': pred_a,
        'pred_b_half_stable_alignment': pred_b,
        'pred_c_half_materiality': pred_c,
        'strong_null_pooled_only_demotion': null_fired,
        "raw_tokens_logits_or_hidden_states_included": False,
        "validation_or_sealed_opened": False,
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(f"pred_a={pred_a} pred_b={pred_b} pred_c={pred_c} null={null_fired} "
          f"pooled={pooled_cosine:.6f} halves={[round(c,4) for c in half_cosines]} "
          f"norms={ {k: round(v,4) for k, v in half_norms.items()} } ({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
