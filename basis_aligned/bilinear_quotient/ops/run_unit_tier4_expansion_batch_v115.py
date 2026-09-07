#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets (v112/v113 receipts), the expansion, the selection rule and every bar fixed before the run.
"""v115 (Tier-4 batch): exact replayed expansion of each chosen head's write into offset x upstream-writer products.

TIER_RUBRIC Tier 4 = Tier 3 plus "an exact replayed algebraic expansion of the relevant products of upstream
contributions and an executable sufficiency test". For the 13 batch-lifted all-rows sets (v112: 7, v113: 6) plus the
quantifier instrument (v113 units), every chosen unit is a head. bilin18's attention is SQUARED (pattern =
(q.k/D)(q2.k2/D), causal, no softmax) with the value residual v = (1-lam) c_v(n_l) + lam v1 (v1 = layer-0 value).
At the read position t the head's c_proj input is exactly
    z = sum_s P[t,s] v[s],   v[s] = (1-lam) W_V^h n_l[s] + lam v1^h[s],   n_l[s] = live_l[s] / r_l[s],
    live_l[s] = sum_j c_{l,j} w_j[s]  (the residual is re-mixed each layer: x_l = live_l + attn_l + mlp_l,
    live_l = lam0_l x_{l-1} + lam1_l x0, so c_{l,j} is the product of the intermediate lam0s; r_l = the rms radius),
so with writers j in {emb} u {attn_k, mlp_k : k < l} and offsets o = t - s the write is the exact sum of PRODUCTS
    T[o, j] = P[t, t-o] * [(1-lam) W_V^h (c_j w_j[t-o] / r_l[t-o])]   (+ lam P[t, t-o] v1^h[t-o] for j = emb),
one product of the pattern (a function of the whole upstream state at t and s) with one upstream writer's contribution.
Both sides are expanded separately (base and donor may differ in length: terms are keyed by offset from the read
position, missing entries are 0), so Delta z = sum_{o,j} (T^D - T^B) exactly. Share of a term := <Delta T, Delta z> /
|Delta z|^2 (the shares sum to 1 per row). SELECTION (fixed rule, EVEN rows): rank (o, j) terms by mean share, take
the shortest prefix reaching cumulative share >= 0.90, at most K=6 per head. SUFFICIENCY REPLAY (ODD rows): the head's
patched value is z^B + sum_{selected} Delta T (an exact replacement through `forward_units`), every head of the set at
once; recovery is measured against the exact set interchange on the same rows. Two further exact splits are replayed:
value arm z^B + sum_o P^D_o (v^D_o - v^B_o) and pattern arm z^B + sum_o (P^D_o - P^B_o) v^B_o (they sum to exact;
for rows of unequal length the pairing by offset is exact but the prefix tokens are shifted by one -- reported).
Cue offsets := offsets at which base and donor tokens differ (offset-aligned; the extra token of a longer prompt counts).

REGISTERED BEFORE THE RUN (14 behaviours)
    pred_a_identity          max over sets, heads, rows, sides of ||sum_{o,j} T - z_captured|| / ||z|| <= 1e-3, and
                             the full-term replay on ODD reproduces exact within 0.02 on 14/14. Worked: 2e-6 True; 0.03 False.
    pred_b_value_not_pattern value arm >= 0.80 of exact AND |pattern arm| <= 0.30 of exact on >= 10 of 14.
                             Worked: (0.91, 0.12) True; (0.55, 0.48) False.
    pred_c_sparse_suffices   the selected terms (<= 6 per head, chosen on EVEN) replayed on ODD give >= 0.80 of exact on
                             >= 8 of 14. Worked: 0.84 True; 0.61 False.
    pred_d_cue_local         the cue offsets alone carry >= 0.80 of the mean share on >= 8 of 14 (share summed over
                             writers at cue offsets, mean over the set's heads). Worked: 0.86 True; 0.45 False.
    pred_e_writer_sparsity   among selected terms, the top-2 writers (by summed selected share, per set) carry >= 0.70 of
                             the selected share on >= 10 of 14. Worked: 0.78 True; 0.52 False.
    Prior: a 85%; b 50%; c 45%; d 40% (downstream positions inherit the cue through earlier heads); e 55%.
    Reading. a AND c True on a set: that set's head-level Tier-4 expansion is exact, replayed and sufficient (with the
    selected terms named); c False with a True: the write is diffuse over offsets/writers at rank K -- report the
    cumulative-share curve, do not raise K.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_batch_v112 as v112

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier4_expansion_batch_v115_result.json"
V112 = ROOT / "circuits/followups/unit_tier3_batch_v112_result.json"
V113 = ROOT / "circuits/followups/unit_tier3_batch_row2_v113_result.json"
SHARE_TARGET, K, IDENT_TOL, INSTR_TOL, VALUE_MIN, PATTERN_MAX, SUFF_MIN, CUE_MIN, WRITER_MIN = 0.90, 6, 1e-3, 0.02, 0.80, 0.30, 0.80, 0.80, 0.70
K_B, K_C, K_D, K_E = 10, 8, 8, 10
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 64000
C, H, D = g.N_EMBD, 9, g.HEAD_DIM


def _plan():
    return {"candidate_id": "corpus.unit_tier4_expansion_batch_v115", "behaviours": 14,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def sets():
    a, b = json.loads(V112.read_text()), json.loads(V113.read_text())
    allrows = lambda r: [n for n, v in r["behaviours"].items() if all(v.get("rows", {}).get(k) for k in ("row2", "row3", "row4", "row5"))]
    out = {n: a["behaviours"][n]["units"] for n in allrows(a)}
    out.update({n: b["behaviours"][n]["units"] for n in allrows(b) if n != "quantifier_number"})
    out["quantifier_number"] = b["behaviours"]["quantifier_number"]["units"]
    return out


def capture(backend, batch, heads_by_layer):
    """Exact model forward; returns per chosen head: pattern row at t, captured z, per-(offset, writer) terms,
    and the value/pattern split ingredients. Keys: (rid, unit) -> dict."""
    torch, F, model = backend.torch, backend.F, backend.model
    tt = sys.modules[type(model.transformer.h[0].attn).__module__]
    tokens, lengths = backend._tensor_batch(batch)
    n = len(batch.row_ids)
    positions = list(batch.semantic_positions)
    idx = torch.arange(n, device=tokens.device)
    pos = torch.tensor(positions, device=tokens.device)
    out = {}
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (C,))
        x0, v1 = x, None
        writers = [("emb", x0)]
        cx = {"emb": 1.0}                        # live_l = sum_j coef[j] w_j: the residual is RE-MIXED each layer
        for l, block in enumerate(model.transformer.h):
            lam0, lam1 = block.lambdas[0], block.lambdas[1]
            live = lam0 * x + lam1 * x0
            coef = {j: lam0 * c for j, c in cx.items()}
            coef["emb"] = coef["emb"] + lam1
            xn = F.rms_norm(live, (C,))
            r = (xn.norm(dim=-1, keepdim=True) / live.norm(dim=-1, keepdim=True).clamp_min(1e-12))   # xn = live * r  (1/r_l)
            at = block.attn
            if l in heads_by_layer:
                B, T, _ = xn.shape
                q = at.c_q(xn).view(B, T, H, D); k = at.c_k(xn).view(B, T, H, D)
                q2 = at.c_q2(xn).view(B, T, H, D); k2 = at.c_k2(xn).view(B, T, H, D)
                cos, sin = at.rotary(q)
                q, k = tt.apply_rotary_emb(F.rms_norm(q, (D,)), cos, sin), tt.apply_rotary_emb(F.rms_norm(k, (D,)), cos, sin)
                q2, k2 = tt.apply_rotary_emb(F.rms_norm(q2, (D,)), cos, sin), tt.apply_rotary_emb(F.rms_norm(k2, (D,)), cos, sin)
                scores = torch.einsum("bqhd,bkhd->bhqk", q, k); scores2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2)
                pattern = (scores / D) * (scores2 / D)
                pattern = pattern.masked_fill(~torch.tril(torch.ones(T, T, device=pattern.device, dtype=torch.bool)), 0.0)
                v_own = at.c_v(xn).view(B, T, H, D)
                v1_here = v_own if v1 is None else v1.view_as(v_own)
                lam = at.lamb
                vmix = (1 - lam) * v_own + lam * v1_here
                z_all = torch.einsum("bhqk,bkhd->bhqd", pattern, vmix)
                # per-writer value-path contributions at every position: c_v is linear, n_l = r * live
                for h in heads_by_layer[l]:
                    P = pattern[idx, h, pos]                                    # (n, T)  pattern row at t
                    z = z_all[idx, h, pos]                                      # (n, D)
                    terms = {}
                    for name, w in writers:
                        cj = coef[name]
                        u = at.c_v(cj * w * r).view(B, T, H, D)[:, :, h]         # (n, T, D)
                        contrib = (1 - lam) * P.unsqueeze(-1) * u
                        if name == "emb":
                            contrib = contrib + lam * P.unsqueeze(-1) * v1_here[:, :, h]
                        terms[name] = contrib                                    # (n, T, D), indexed by source s
                    vh = vmix[:, :, h]                                          # (n, T, D)
                    for i, rid in enumerate(batch.row_ids):
                        t = positions[i]
                        rec = {"z": z[i].float().clone(), "P": P[i, :t + 1].flip(0).float().clone(),     # offset-indexed
                               "v": vh[i, :t + 1].flip(0).float().clone(),
                               "terms": {name: terms[name][i, :t + 1].flip(0).float().clone() for name in terms},
                               "tokens": tokens[i, :t + 1].flip(0).tolist()}
                        out[(rid, f"attn:{l:02d}:head:{h:02d}")] = rec
            attention, v1 = at(xn, v1)
            x = live + attention
            writers.append((f"attn_{l}", attention))
            mlp = block.mlp(F.rms_norm(x, (C,)))
            x = x + mlp
            writers.append((f"mlp_{l}", mlp))
            cx = dict(coef); cx[f"attn_{l}"] = 1.0; cx[f"mlp_{l}"] = 1.0
    return out


def pad(a, T):
    torch = __import__("torch")
    return a if a.shape[0] == T else torch.cat([a, a.new_zeros((T - a.shape[0],) + tuple(a.shape[1:]))])


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    global OUT
    smoke = os.environ.get("V115_SMOKE")          # local CPU smoke test of the code path only; never the science run
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = sets()
    if smoke:
        S = {k: S[k] for k in list(S)[:1]}; OUT = Path(smoke)
    R, ident_max = {}, 0.0
    for n, units in S.items():
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")[:8] if smoke else g.rows_of(m, "A1")
        prep = {"even": g.prepare(backend, a1[0::2]), "odd": g.prepare(backend, a1[1::2])}
        hb = {}
        for u in units:
            hb.setdefault(g.unit_layer(u), []).append(int(u.rsplit(":", 1)[1]))
        cap = {sp: {"base": capture(backend, prep[sp].base_batch, hb), "donor": capture(backend, prep[sp].donor_batch, hb)} for sp in prep}

        def deltas(sp):
            """per unit: list over rows of (dz (D,), dT {(o,j): (D,)}, cue offsets, unequal flag, value-arm, pattern-arm)."""
            P = prep[sp]
            outu = {}
            for u in units:
                rows = []
                for rid in P.base_batch.row_ids:
                    b, d = cap[sp]["base"][(rid, u)], cap[sp]["donor"][(rid, u)]
                    T = max(b["P"].shape[0], d["P"].shape[0])
                    dz = d["z"] - b["z"]
                    dT = {}
                    for j in set(b["terms"]) | set(d["terms"]):
                        tb = pad(b["terms"][j], T) if j in b["terms"] else None
                        td = pad(d["terms"][j], T) if j in d["terms"] else None
                        diff = (td if td is not None else 0) - (tb if tb is not None else 0)
                        for o in range(T):
                            dT[(o, j)] = diff[o]
                    Pb, Pd = pad(b["P"], T), pad(d["P"], T)
                    vb, vd = pad(b["v"], T), pad(d["v"], T)
                    value_arm = (Pd.unsqueeze(-1) * (vd - vb)).sum(0)
                    pattern_arm = ((Pd - Pb).unsqueeze(-1) * vb).sum(0)
                    tb_, td_ = b["tokens"], d["tokens"]
                    cue = [o for o in range(T) if o >= len(tb_) or o >= len(td_) or tb_[o] != td_[o]]
                    rows.append({"dz": dz, "dT": dT, "cue": cue, "unequal": len(tb_) != len(td_), "value": value_arm, "pattern": pattern_arm,
                                 "zb": b["z"], "zd": d["z"], "sum_b": sum(b["terms"].values()).sum(0), "sum_d": sum(d["terms"].values()).sum(0)})
                outu[u] = rows
            return outu

        E, O = deltas("even"), deltas("odd")
        # identity check on both splits and sides
        for sp in (E, O):
            for u in units:
                for row in sp[u]:
                    for zz, ss in ((row["zb"], row["sum_b"]), (row["zd"], row["sum_d"])):
                        ident_max = max(ident_max, float((ss - zz).norm() / zz.norm().clamp_min(1e-12)))
        # shares on EVEN -> selection (fixed rule)
        share, selected, curves, cue_share = {}, {}, {}, {}
        for u in units:
            rows = E[u]
            sh, cs = {}, 0.0
            for row in rows:
                dz2 = float(row["dz"].dot(row["dz"])) or 1e-12
                for k, v in row["dT"].items():
                    sh[k] = sh.get(k, 0.0) + float(v.dot(row["dz"])) / dz2 / len(rows)
                cs += sum(float(v.dot(row["dz"])) / dz2 for k, v in row["dT"].items() if k[0] in row["cue"]) / len(rows)
            ranked = sorted(sh.items(), key=lambda kv: -kv[1])
            cum, sel = 0.0, []
            for k, v in ranked:
                if cum >= SHARE_TARGET or len(sel) >= K:
                    break
                sel.append(k); cum += v
            share[u], selected[u], cue_share[u] = sh, sel, cs
            curves[u] = [[o, j, round(v, 4)] for (o, j), v in ranked[:12]]

        def replay(build):
            P = prep["odd"]
            synth = {}
            for u in units:
                for rid, row in zip(P.base_batch.row_ids, O[u]):
                    synth[(rid, u)] = build(u, row).cpu()
            out = g.forward_units(backend, P.base_batch, units=units, donor_cache=synth, base_cache=P.base_cache)
            return g.recovery(P, [-(float(a) - float(f)) for a, f in out.tolist()])

        exact = g.recovery(prep["odd"], g.patched_axis(backend, prep["odd"], units))
        full = replay(lambda u, row: row["zb"] + sum(row["dT"].values()))
        value = replay(lambda u, row: row["zb"] + row["value"])
        pattern = replay(lambda u, row: row["zb"] + row["pattern"])
        sel_rec = replay(lambda u, row: row["zb"] + sum((row["dT"][k] for k in selected[u] if k in row["dT"]), row["zb"] * 0))
        frac = lambda v: round(v / exact, 3) if abs(exact) > 1e-6 else None
        by_writer = {}
        for u in units:
            for (o, j) in selected[u]:
                by_writer[j] = by_writer.get(j, 0.0) + share[u][(o, j)]
        tot_sel = sum(by_writer.values()) or 1e-12
        top2 = sum(sorted(by_writer.values(), reverse=True)[:2]) / tot_sel
        R[n] = {"units": units, "exact_odd": round(exact, 3), "full_replay": frac(full), "value_arm": frac(value), "pattern_arm": frac(pattern),
                "selected_replay": frac(sel_rec), "selected": {u: [[o, j, round(share[u][(o, j)], 4)] for o, j in selected[u]] for u in units},
                "selected_share": {u: round(sum(share[u][k] for k in selected[u]), 4) for u in units},
                "share_curves": curves, "cue_share": {u: round(cue_share[u], 3) for u in units},
                "cue_share_mean": round(sum(cue_share.values()) / len(units), 3), "writer_share_selected": {j: round(v, 4) for j, v in by_writer.items()},
                "top2_writer_frac": round(top2, 3), "unequal_rows_odd": sum(row["unequal"] for row in O[units[0]]), "rows_odd": len(O[units[0]]),
                "seconds": round(time.perf_counter() - t1, 1)}
        print(n, "exact", round(exact, 3), "full", frac(full), "value", frac(value), "pattern", frac(pattern), "selected", frac(sel_rec),
              "cue", R[n]["cue_share_mean"], "top2w", R[n]["top2_writer_frac"], round(time.perf_counter() - t0), "s", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"partial": True, "behaviours": R, "ident_max": ident_max}, indent=2, sort_keys=True, default=str) + "\n")

    ok = lambda v, bar: v is not None and v >= bar
    predictions = {
        'pred_a_identity': ident_max <= IDENT_TOL and all(R[n]["full_replay"] is not None and abs(R[n]["full_replay"] - 1.0) <= INSTR_TOL for n in R),
        'pred_b_value_not_pattern': sum(ok(R[n]["value_arm"], VALUE_MIN) and R[n]["pattern_arm"] is not None and abs(R[n]["pattern_arm"]) <= PATTERN_MAX for n in R) >= K_B,
        'pred_c_sparse_suffices': sum(ok(R[n]["selected_replay"], SUFF_MIN) for n in R) >= K_C,
        'pred_d_cue_local': sum(R[n]["cue_share_mean"] >= CUE_MIN for n in R) >= K_D,
        'pred_e_writer_sparsity': sum(R[n]["top2_writer_frac"] >= WRITER_MIN for n in R) >= K_E,
    }
    tier4 = [n for n in R if ok(R[n]["selected_replay"], SUFF_MIN)]
    summary = {n: (R[n]["exact_odd"], R[n]["full_replay"], R[n]["value_arm"], R[n]["pattern_arm"], R[n]["selected_replay"], R[n]["cue_share_mean"], R[n]["top2_writer_frac"]) for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_tier4_expansion_batch_result_v1", "candidate_id": "corpus.unit_tier4_expansion_batch_v115",
              "ident_max": ident_max, "tier4_sufficient": tier4, "counts": {"value_not_pattern": sum(ok(R[n]["value_arm"], VALUE_MIN) and R[n]["pattern_arm"] is not None and abs(R[n]["pattern_arm"]) <= PATTERN_MAX for n in R),
              "sparse_suffices": len(tier4), "cue_local": sum(R[n]["cue_share_mean"] >= CUE_MIN for n in R), "writer_sparse": sum(R[n]["top2_writer_frac"] >= WRITER_MIN for n in R)},
              "summary_columns": ["exact_odd", "full_replay", "value_arm", "pattern_arm", "selected_replay", "cue_share_mean", "top2_writer_frac"], "summary": summary,
              "behaviours": R, "bars": {"share_target": SHARE_TARGET, "k": K, "ident_tol": IDENT_TOL, "instr_tol": INSTR_TOL, "value_min": VALUE_MIN, "pattern_max": PATTERN_MAX,
              "suff_min": SUFF_MIN, "cue_min": CUE_MIN, "writer_min": WRITER_MIN, "k_b": K_B, "k_c": K_C, "k_d": K_D, "k_e": K_E},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "tier4_sufficient": tier4, "summary": summary, "ident_max": ident_max, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
