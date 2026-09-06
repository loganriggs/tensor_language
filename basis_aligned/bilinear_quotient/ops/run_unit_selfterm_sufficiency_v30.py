#!/usr/bin/env python3
# BQGATE: frozen predictions; sets (v9), stacks (v28/v29) and the algebraic decomposition fixed before the run.
"""v30, Tier 4: is the bilinear SELF-TERM of the converter stack sufficient for the early sets' conversion?

v29: stack conversion is quadratic in the write (slope 2.08 polarity, 2.04 voice; quantifier 1.55). Tier 4 asks
for the exact replayed algebraic expansion and an executable sufficiency test (TIER_RUBRIC). Each stack MLP is
ungated Bilinear: M(u) = Down[L(u) * R(u)] + bias. With u_b the base normalized input and u_p the input under
the exact-set patch (both CAPTURED at the row's final position from replayed runs), w = u_p - u_b and, exactly,
    M(u_p) - M(u_b) = Down[L(u_b) R(w) + L(w) R(u_b)]  +  Down[L(w) R(w)]  =  cross + self.
Executable test: forward with the set patched and every stack MLP's output REPLACED at the position by
m_b + self (cross dropped), and separately by m_b + cross (self dropped). All other modules live. The
quadratic law predicts self-only ~ live and cross-only ~ stack-frozen for polarity/voice, mixed for quantifier.
Caveat (registered): the replacement is static -- layer l's self-term is the one from the full live run,
not recomputed from the self-only trajectory; this is the replayed expansion the rubric names, not a
self-consistent one. The 09:07 head in polarity's stack is left live in every arm (it is not an MLP).

REGISTERED BEFORE THE RUN
    pred_a_identity      max over rows and stack layers of |DeltaM - cross - self| / |DeltaM| <= 1e-3 (float32
                         model; a miss means the model is gated or the capture is at the wrong tensor).
    pred_b_self_suffices rec(self-only) / rec_live >= 0.80 for polarity AND voice. Worked: 0.585 live, 0.50 self-only
                         -> 0.85 True; 0.40 -> 0.68 False.
    pred_c_cross_inert   rec(cross-only) <= rec_stack_frozen + 0.20 * (rec_live - rec_stack_frozen) for polarity
                         AND voice. Worked polarity: frozen 0.245, live 0.585 -> bar 0.313; cross-only 0.28 True.
    pred_d_quantifier_mixed  quantifier self share = (rec(self-only) - rec_stack_frozen) / (rec_live - rec_stack_frozen)
                         in [0.30, 0.80] (v29 slope 1.55 sits between 1 and 2). Worked: 0.55 True; 0.9 False.
    pred_e_self_vector_shared  per stack MLP, the self-term vectors are one direction across rows: mean pairwise
                         cosine of self (rows sign-aligned geometrically by the layer's DeltaM against row 0) >= 0.70 at every stack layer for
                         polarity and voice. Worked: 0.78, 0.81, 0.74, 0.90 -> True; one layer 0.5 -> False.
    Reading rule. a,b,c True: the conversion IS Down[L(w)R(w)] on the write -- a closed-form weight statement
    (Tier 4 for the stack); the remaining Tier 4 gap is the write's own algebra inside the set's heads. b False:
    the quadratic scaling arises from cross-terms whose context factor itself carries the write (residual
    feedback), and the self-consistent expansion is needed.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier2_characterization_v23 as v23
import run_unit_tier3_readers_v25 as v25
import run_unit_converter_law_v28 as v28

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_selfterm_sufficiency_v30_result.json"
SETS = {k: v23.SETS[k] for k in ("quantifier_number", "polarity_licensing", "voice_frame")}
STACK = dict(v28.STACK); STACK["quantifier_number"] = [f"mlp:{l:02d}" for l in range(11, 15)]
EARLY = ("polarity_licensing", "voice_frame")
ID_TOL, SELF_BAR, CROSS_BAR, MIXED, COS_BAR = 1e-3, 0.80, 0.20, (0.30, 0.80), 0.70
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 30, 1000


def _plan():
    return {"candidate_id": "corpus.unit_selfterm_sufficiency_v30", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _capture(backend, prep, layers, **kw):
    """Run forward_units with kw, capturing each stack MLP's normalized INPUT and raw OUTPUT at the row position."""
    torch, model = backend.torch, backend.model
    pos = list(prep.base_batch.semantic_positions)
    idx = torch.arange(len(pos), device=backend.device)
    p = torch.tensor(pos, device=backend.device)
    ins, outs, handles = {}, {}, []
    for l in layers:
        mlp = model.transformer.h[l].mlp
        handles.append(mlp.register_forward_pre_hook(lambda m, a, l=l: ins.__setitem__(l, a[0][idx, p].detach().float().clone())))
        handles.append(mlp.register_forward_hook(lambda m, a, o, l=l: outs.__setitem__(l, o[idx, p].detach().float().clone())))
    try:
        af = g.forward_units(backend, prep.base_batch, **kw)
    finally:
        for h in handles: h.remove()
    return v25._rec(prep, [-(float(a) - float(f)) for a, f in af.tolist()]), ins, outs


def _terms(mlp, u_b, u_p):
    """Exact bilinear expansion of M(u_p) - M(u_b) into cross and self terms (float32)."""
    WL, WR, WD = mlp.Left.weight.float(), mlp.Right.weight.float(), mlp.Down.weight.float()
    w = u_p - u_b
    Lb, Rb, Lw, Rw = u_b @ WL.T, u_b @ WR.T, w @ WL.T, w @ WR.T
    cross = (Lb * Rw + Lw * Rb) @ WD.T
    self_ = (Lw * Rw) @ WD.T
    return cross, self_


def _mean_pair_cos(torch, vecs, ref):
    """Rows alternate direction; sign-align GEOMETRICALLY by the reference delta's dot with row 0."""
    signs = torch.sign(ref @ ref[0]).clamp_min(0) * 2 - 1
    v = vecs * signs[:, None]
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    c = v @ v.T
    n = v.shape[0]
    return float((c.sum() - n) / (n * (n - 1)))


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    assert not model.config.gated, "self-term algebra needs the ungated Bilinear MLP"
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        stack_mlps = [m for m in STACK[name] if m.startswith("mlp:")]
        layers = [g.unit_layer(m) for m in stack_mlps]
        cache = v25._merged_cache(prep, units)
        rec_base, ins_b, outs_b = _capture(backend, prep, layers)
        rec_live, ins_p, outs_p = _capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache)
        # stack-frozen reference: stack MLP outputs replaced by their base values (exact, static)
        frozen = dict(cache)
        self_cache, cross_cache = dict(cache), dict(cache)
        ident, layer_stats = [], {}
        for l, m in zip(layers, stack_mlps):
            mlp = model.transformer.h[l].mlp
            cross, self_ = _terms(mlp, ins_b[l], ins_p[l])
            dM = outs_p[l] - outs_b[l]
            rel = ((dM - cross - self_).norm(dim=1) / dM.norm(dim=1).clamp_min(1e-12))
            ident.append(float(rel.max()))
            for i, rid in enumerate(prep.base_batch.row_ids):
                frozen[(rid, m)] = outs_b[l][i]
                self_cache[(rid, m)] = outs_b[l][i] + self_[i]
                cross_cache[(rid, m)] = outs_b[l][i] + cross[i]
            layer_stats[m] = {"identity_rel_err_max": float(rel.max()),
                              "self_norm_over_delta": float((self_.norm(dim=1) / dM.norm(dim=1).clamp_min(1e-12)).mean()),
                              "cross_norm_over_delta": float((cross.norm(dim=1) / dM.norm(dim=1).clamp_min(1e-12)).mean()),
                              "self_pair_cos": _mean_pair_cos(torch, self_, dM),
                              "cross_pair_cos": _mean_pair_cos(torch, cross, dM),
                              "delta_pair_cos": _mean_pair_cos(torch, dM, dM),
                              "write_norm_mean": float((ins_p[l] - ins_b[l]).norm(dim=1).mean())}
        all_units = list(units) + stack_mlps
        def rec_with(c):
            return _capture(backend, prep, layers, units=all_units, donor_cache=c, base_cache=prep.base_cache)[0]
        rec_frozen, rec_self, rec_cross = rec_with(frozen), rec_with(self_cache), rec_with(cross_cache)
        conv = rec_live - rec_frozen
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "rows": len(prep.rows),
                        "rec": {"base": rec_base, "live": rec_live, "stack_frozen": rec_frozen, "self_only": rec_self, "cross_only": rec_cross},
                        "conversion": conv, "self_share": (rec_self - rec_frozen) / conv if conv else None,
                        "cross_share": (rec_cross - rec_frozen) / conv if conv else None,
                        "self_over_live": rec_self / rec_live if rec_live else None,
                        "identity_rel_err_max": max(ident), "layers": layer_stats}
        print(name, {k: round(v, 3) for k, v in report[name]["rec"].items()}, "self_share %.2f cross_share %.2f ident %.1e" %
              (report[name]["self_share"], report[name]["cross_share"], max(ident)), flush=True)
        for m, s in layer_stats.items():
            print("  ", m, {k: round(v, 3) for k, v in s.items()}, flush=True)

    predictions = {
        'pred_a_identity': all(report[n]["identity_rel_err_max"] <= ID_TOL for n in SETS),
        'pred_b_self_suffices': all(report[n]["self_over_live"] is not None and report[n]["self_over_live"] >= SELF_BAR for n in EARLY),
        'pred_c_cross_inert': all(report[n]["rec"]["cross_only"] <= report[n]["rec"]["stack_frozen"] + CROSS_BAR * report[n]["conversion"] for n in EARLY),
        'pred_d_quantifier_mixed': report["quantifier_number"]["self_share"] is not None
                                   and MIXED[0] <= report["quantifier_number"]["self_share"] <= MIXED[1],
        'pred_e_self_vector_shared': all(s["self_pair_cos"] >= COS_BAR for n in EARLY for s in report[n]["layers"].values()),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_selfterm_sufficiency_result_v1",
              "candidate_id": "corpus.unit_selfterm_sufficiency_v30", "semantics": "exact_set_plus_static_replayed_stack_output_replacement",
              "bars": {"identity": ID_TOL, "self_over_live": SELF_BAR, "cross_share_max": CROSS_BAR, "quantifier_mixed": list(MIXED), "pair_cos": COS_BAR},
              "shares": {n: {"self": report[n]["self_share"], "cross": report[n]["cross_share"]} for n in SETS},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "shares": result["shares"], "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
