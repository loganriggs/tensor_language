#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets from v115, near-offset window and bars fixed before the run; ODD rows.
"""v121: at the NEAR offsets (1-3), is a head's interchange term a pattern change or a content change?
v120 refuted the carrier relay that v117/v119 read into the share table: the late heads' large Delta T terms at
offsets 1-3 do not come from donor content carried to those positions (clamping the positions' attention/MLPs to base
costs <= 0.19). The exact per-offset term of a head is T[o] = P[o] v[o] (v115 identity), so
    Delta T[o] = (P^D[o] - P^B[o]) v^B[o]  +  P^D[o] (v^D[o] - v^B[o])   =:  pattern[o] + content[o]     (exact)
v115's value/pattern arms were whole-head (all offsets at once). This rung splits PER OFFSET and replays each half at
the near offsets only (all other offsets keep their full Delta T), `forward_units` synthetic head value z^B + sum:
    full            z^B + sum_o Delta T[o]                                      (instrument = exact)
    near_pattern    near offsets keep pattern[o] only; far offsets full
    near_content    near offsets keep content[o] only; far offsets full
    near_none       near offsets dropped; far offsets full
    far_none        far offsets dropped; near offsets full  (how much the near offsets carry on their own)
Far-cue sets (v117 misses): narrative_tense, possessive_argument, possessive_verbfinal. Near window {1, 2, 3} fixed,
EXCLUDING cue offsets row-wise (a near offset whose base/donor tokens differ is a cue, not a carrier candidate; the CPU
smoke on 4 degree_frame rows showed near_none 0.06 with the cue inside the window -- the exclusion was added BEFORE
enqueue and is disclosed here).
REGISTERED BEFORE THE RUN (14 sets; recoveries as fractions of the exact set interchange on ODD)
    pred_a_instrument       identity |z - sum_o P[o]v[o]| / |z| <= 1e-3 on both sides and full replay within 0.02 of
                            exact on 14/14.                                            Worked: 3e-6, 1.001 True; 0.03 False.
    pred_b_pattern_far_cue  near_pattern >= 0.80 on 3/3 far-cue sets.                  Worked: 0.86 True; 0.71 False.
    pred_c_content_far_cue  near_content <= 0.50 on >= 2/3 far-cue sets.               Worked: 0.32 True; 0.62 False.
    pred_d_near_dispensable near_none >= 0.80 on >= 8 of the 11 other sets.            Worked: 0.88 True; 0.74 False.
    pred_e_near_alone       far_none <= 0.50 on 14/14 (near offsets never suffice alone). Worked: 0.21 True; 0.55 False.
    Prior: a 85%; b 50%; c 55%; d 50%; e 65%.
    Reading: b+c True name the far-cue sets' late heads as PATTERN readers of unchanged near content (the donor changes
    where they look, not what is there); b False with c True says both halves are needed there (interaction; report,
    no new arm). d False says near offsets matter broadly and v117's struct arm passed for another reason -- report.
"""
from __future__ import annotations

import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_batch_v112 as v112
import run_unit_tier4_expansion_batch_v115 as v115

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier4_near_offset_pattern_v121_result.json"
FAR_CUE = ("narrative_tense", "possessive_argument", "possessive_verbfinal")
NEAR = (1, 2, 3)
IDENT_TOL, INSTR_TOL, PAT_MIN, CONT_MAX, NONE_MIN, ALONE_MAX, K_C, K_D = 1e-3, 0.02, 0.80, 0.50, 0.80, 0.50, 2, 8
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 64000


def _plan():
    return {"candidate_id": "corpus.unit_tier4_near_offset_pattern_v121", "behaviours": 14,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def split(cap, prep, units):
    """per unit: rows of (zb, pattern[o], content[o], identity residuals)."""
    pad = v115.pad
    out = {}
    for u in units:
        rows = []
        for rid in prep.base_batch.row_ids:
            b, d = cap["base"][(rid, u)], cap["donor"][(rid, u)]
            T = max(b["P"].shape[0], d["P"].shape[0])
            PB, PD = pad(b["P"], T), pad(d["P"], T)
            vB, vD = pad(b["v"], T), pad(d["v"], T)
            idb = float((b["z"] - (b["P"].unsqueeze(-1) * b["v"]).sum(0)).norm() / max(float(b["z"].norm()), 1e-12))
            idd = float((d["z"] - (d["P"].unsqueeze(-1) * d["v"]).sum(0)).norm() / max(float(d["z"].norm()), 1e-12))
            pattern = (PD - PB).unsqueeze(-1) * vB            # (T, D)
            content = PD.unsqueeze(-1) * (vD - vB)
            tb_, td_ = b["tokens"], d["tokens"]
            cue = [o for o in range(T) if o >= len(tb_) or o >= len(td_) or tb_[o] != td_[o]]
            rows.append({"zb": b["z"], "pattern": pattern, "content": content, "ident": max(idb, idd), "T": T, "cue": cue})
        out[u] = rows
    return out


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V121_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    if smoke:
        S = {k: S[k] for k in list(S)[:1]}
    R = {}
    for n, units in S.items():
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")[1::2]
        if smoke:
            a1 = a1[:4]
        P = g.prepare(backend, a1)
        hb = {}
        for u in units:
            hb.setdefault(g.unit_layer(u), []).append(int(u.rsplit(":", 1)[1]))
        cap = {"base": v115.capture(backend, P.base_batch, hb), "donor": v115.capture(backend, P.donor_batch, hb)}
        O = split(cap, P, units)
        ident = max(row["ident"] for u in units for row in O[u])

        def replay(near_keep, far_keep):
            synth = {}
            for u in units:
                for rid, row in zip(P.base_batch.row_ids, O[u]):
                    val = row["zb"].clone()
                    for o in range(row["T"]):
                        keep = near_keep if (o in NEAR and o not in row["cue"]) else far_keep
                        if "pattern" in keep:
                            val = val + row["pattern"][o]
                        if "content" in keep:
                            val = val + row["content"][o]
                    synth[(rid, u)] = val.cpu()
            out = g.forward_units(backend, P.base_batch, units=units, donor_cache=synth, base_cache=P.base_cache)
            return g.recovery(P, [-(float(a) - float(f)) for a, f in out.tolist()])

        exact = g.recovery(P, g.patched_axis(backend, P, units))
        frac = lambda v: round(v / exact, 3) if abs(exact) > 1e-6 else None
        both = ("pattern", "content")
        arms = {"full": frac(replay(both, both)), "near_pattern": frac(replay(("pattern",), both)),
                "near_content": frac(replay(("content",), both)), "near_none": frac(replay((), both)),
                "far_none": frac(replay(both, ()))}
        R[n] = {"units": units, "kind": "far_cue" if n in FAR_CUE else "other", "exact_odd": round(exact, 3),
                "identity_rel_max": ident, "arms": arms, "rows_odd": len(a1), "seconds": round(time.perf_counter() - t1, 1)}
        print(n, R[n]["kind"], "exact", round(exact, 3), f"ident {ident:.1e}", arms, round(time.perf_counter() - t0), "s", flush=True)

    ok = lambda x: x is not None
    far = [n for n in FAR_CUE if n in R]
    other = [n for n in R if n not in FAR_CUE]
    inst = [n for n, v in R.items() if v["identity_rel_max"] <= IDENT_TOL and ok(v["arms"]["full"]) and abs(v["arms"]["full"] - 1.0) <= INSTR_TOL]
    pat = [n for n in far if ok(R[n]["arms"]["near_pattern"]) and R[n]["arms"]["near_pattern"] >= PAT_MIN]
    con = [n for n in far if ok(R[n]["arms"]["near_content"]) and R[n]["arms"]["near_content"] <= CONT_MAX]
    none = [n for n in other if ok(R[n]["arms"]["near_none"]) and R[n]["arms"]["near_none"] >= NONE_MIN]
    alone = [n for n, v in R.items() if ok(v["arms"]["far_none"]) and v["arms"]["far_none"] <= ALONE_MAX]
    predictions = {
        "pred_a_instrument": len(inst) == len(R),
        "pred_b_pattern_far_cue": len(pat) == len(far) and bool(far),
        "pred_c_content_far_cue": len(con) >= K_C,
        "pred_d_near_dispensable": len(none) >= K_D,
        "pred_e_near_alone": len(alone) == len(R),
    }
    result = {"predictions": predictions, "schema": "unit_tier4_near_offset_pattern_v121",
              "candidate_id": "corpus.unit_tier4_near_offset_pattern_v121",
              "bars": {"ident_tol": IDENT_TOL, "instr_tol": INSTR_TOL, "pattern_min": PAT_MIN, "content_max": CONT_MAX,
                       "none_min": NONE_MIN, "alone_max": ALONE_MAX, "K": [K_C, K_D], "near": list(NEAR)},
              "far_cue_sets": list(FAR_CUE),
              "counts": {"instrument": len(inst), "pattern": len(pat), "content_small": len(con), "near_dispensable": len(none),
                         "near_alone_small": len(alone), "n": len(R)},
              "summary": {n: [v["exact_odd"], v["arms"]["full"], v["arms"]["near_pattern"], v["arms"]["near_content"],
                              v["arms"]["near_none"], v["arms"]["far_none"]] for n, v in R.items()},
              "summary_columns": ["exact_odd", "full", "near_pattern", "near_content", "near_none", "far_none"],
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
