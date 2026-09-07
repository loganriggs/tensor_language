#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, arms and bars fixed before the run; expansion code reused from v115 unchanged.
"""v116: offset-grouped sufficiency of the v115 head-write expansion (which positions must a head read from?).

v115 established the exact expansion z = sum_{o,j} T[o,j] (identity 8e-7, full replay = exact on 14/14) and that the
interchange effect rides the VALUE path (13/14). Sparse (offset, writer) products (<= 6 per head) were sufficient on
only 2/14: each head's value is spread over emb + several early MLPs at 2-3 offsets. The registered reading rule was
"report the curve, do not raise K". This rung asks the coarser, position-level question with the SAME expansion and
the SAME exact-replacement replay (`forward_units` with a synthetic head value z^B + selected Delta T), ODD rows,
selection (where needed) on EVEN:
    cue arm        all writers at the CUE offsets (offset-aligned positions whose base/donor tokens differ)
    top2 arm       all writers at the two offsets with the largest mean EVEN share (per head)
    emb arm        the embedding writer only, at every offset
    identity arm   writers {emb, mlp_0..mlp_4} at the cue offsets (the cue token's identity as the early MLPs transform it)
    v115 arm       v115's selected (o, j) products, recomputed here (instrument: must reproduce the v115 receipt)

REGISTERED BEFORE THE RUN (14 behaviours; recoveries as fractions of the exact set interchange on ODD)
    pred_a_cue_sufficient      cue arm >= 0.80 on >= 8 of 14.       Worked: 0.85 True; 0.61 False.
    pred_b_two_offsets         top2 arm >= 0.80 on >= 10 of 14.     Worked: 0.83 True; 0.70 False.
    pred_c_embedding_half      emb arm >= 0.50 on >= 8 of 14.       Worked: 0.55 True; 0.31 False.
    pred_d_cue_identity        identity arm >= 0.80 on >= 6 of 14.  Worked: 0.81 True; 0.66 False.
    pred_e_instrument          v115 arm within 0.01 of the v115 receipt's selected_replay on 14/14, and the full-term
                               replay within 0.02 of exact on 14/14. Worked: |0.641 - 0.641| True; 0.03 False.
    Prior: a 45%; b 50%; c 50%; d 35%; e 85%.
    Reading: a True names the head-level Tier-4 statement "the set copies the cue positions' upstream content"; a False
    with b True says the heads read a non-cue position (a downstream carrier) -- name it from the receipt, no new arm.
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
OUT = ROOT / "circuits/followups/unit_tier4_offset_sufficiency_v116_result.json"
V115 = ROOT / "circuits/followups/unit_tier4_expansion_batch_v115_result.json"
CUE_MIN, TOP2_MIN, EMB_MIN, ID_MIN, INSTR_V115, INSTR_FULL, K_A, K_B, K_C, K_D = 0.80, 0.80, 0.50, 0.80, 0.01, 0.02, 8, 10, 8, 6
EARLY = {"emb", "mlp_0", "mlp_1", "mlp_2", "mlp_3", "mlp_4"}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 64000


def _plan():
    return {"candidate_id": "corpus.unit_tier4_offset_sufficiency_v116", "behaviours": 14,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def deltas(cap_side, prep, units):
    """per unit: rows of (zb, dz, dT {(o,j)}, cue offsets)."""
    pad = v115.pad
    outu = {}
    for u in units:
        rows = []
        for rid in prep.base_batch.row_ids:
            b, d = cap_side["base"][(rid, u)], cap_side["donor"][(rid, u)]
            T = max(b["P"].shape[0], d["P"].shape[0])
            dT = {}
            for j in set(b["terms"]) | set(d["terms"]):
                diff = (pad(d["terms"][j], T) if j in d["terms"] else 0) - (pad(b["terms"][j], T) if j in b["terms"] else 0)
                for o in range(T):
                    dT[(o, j)] = diff[o]
            tb_, td_ = b["tokens"], d["tokens"]
            cue = [o for o in range(T) if o >= len(tb_) or o >= len(td_) or tb_[o] != td_[o]]
            rows.append({"zb": b["z"], "dz": d["z"] - b["z"], "dT": dT, "cue": cue})
        outu[u] = rows
    return outu


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    prior = json.loads(V115.read_text())["behaviours"]
    R = {}
    for n, units in S.items():
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")
        prep = {"even": g.prepare(backend, a1[0::2]), "odd": g.prepare(backend, a1[1::2])}
        hb = {}
        for u in units:
            hb.setdefault(g.unit_layer(u), []).append(int(u.rsplit(":", 1)[1]))
        cap = {sp: {"base": v115.capture(backend, prep[sp].base_batch, hb), "donor": v115.capture(backend, prep[sp].donor_batch, hb)} for sp in prep}
        E, O = deltas(cap["even"], prep["even"], units), deltas(cap["odd"], prep["odd"], units)
        # EVEN shares by offset (all writers) -> top-2 offsets per head
        top2 = {}
        for u in units:
            sh = {}
            for row in E[u]:
                dz2 = float(row["dz"].dot(row["dz"])) or 1e-12
                for (o, j), v in row["dT"].items():
                    sh[o] = sh.get(o, 0.0) + float(v.dot(row["dz"])) / dz2 / len(E[u])
            top2[u] = [o for o, _ in sorted(sh.items(), key=lambda kv: -kv[1])[:2]]
        v115_sel = {u: [tuple(x[:2]) for x in prior[n]["selected"][u]] for u in units}

        def replay(keep):
            P = prep["odd"]
            synth = {}
            for u in units:
                for rid, row in zip(P.base_batch.row_ids, O[u]):
                    synth[(rid, u)] = (row["zb"] + sum((v for k, v in row["dT"].items() if keep(u, row, k)), row["zb"] * 0)).cpu()
            out = g.forward_units(backend, P.base_batch, units=units, donor_cache=synth, base_cache=P.base_cache)
            return g.recovery(P, [-(float(a) - float(f)) for a, f in out.tolist()])

        exact = g.recovery(prep["odd"], g.patched_axis(backend, prep["odd"], units))
        frac = lambda v: round(v / exact, 3) if abs(exact) > 1e-6 else None
        arms = {"full": replay(lambda u, row, k: True),
                "cue": replay(lambda u, row, k: k[0] in row["cue"]),
                "top2": replay(lambda u, row, k: k[0] in top2[u]),
                "emb": replay(lambda u, row, k: k[1] == "emb"),
                "identity": replay(lambda u, row, k: k[0] in row["cue"] and k[1] in EARLY),
                "v115": replay(lambda u, row, k: k in v115_sel[u])}
        arms = {k: frac(v) for k, v in arms.items()}
        R[n] = {"units": units, "exact_odd": round(exact, 3), "arms": arms, "top2_offsets": top2,
                "cue_offsets_odd": sorted({o for row in O[units[0]] for o in row["cue"]}),
                "v115_selected_replay": prior[n]["selected_replay"], "seconds": round(time.perf_counter() - t1, 1)}
        print(n, "exact", round(exact, 3), arms, "top2", {u.split(":")[1] + ":" + u.split(":")[3]: v for u, v in top2.items()}, round(time.perf_counter() - t0), "s", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"partial": True, "behaviours": R}, indent=2, sort_keys=True, default=str) + "\n")

    ok = lambda v, bar: v is not None and v >= bar
    counts = {"cue": sum(ok(R[n]["arms"]["cue"], CUE_MIN) for n in R), "top2": sum(ok(R[n]["arms"]["top2"], TOP2_MIN) for n in R),
              "emb": sum(ok(R[n]["arms"]["emb"], EMB_MIN) for n in R), "identity": sum(ok(R[n]["arms"]["identity"], ID_MIN) for n in R)}
    predictions = {
        'pred_a_cue_sufficient': counts["cue"] >= K_A,
        'pred_b_two_offsets': counts["top2"] >= K_B,
        'pred_c_embedding_half': counts["emb"] >= K_C,
        'pred_d_cue_identity': counts["identity"] >= K_D,
        'pred_e_instrument': all(R[n]["arms"]["v115"] is not None and R[n]["v115_selected_replay"] is not None
                                 and abs(R[n]["arms"]["v115"] - R[n]["v115_selected_replay"]) <= INSTR_V115
                                 and R[n]["arms"]["full"] is not None and abs(R[n]["arms"]["full"] - 1.0) <= INSTR_FULL for n in R),
    }
    summary = {n: R[n]["arms"] for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_tier4_offset_sufficiency_result_v1", "candidate_id": "corpus.unit_tier4_offset_sufficiency_v116",
              "counts": counts, "cue_sufficient": [n for n in R if ok(R[n]["arms"]["cue"], CUE_MIN)], "summary": summary, "behaviours": R,
              "bars": {"cue_min": CUE_MIN, "top2_min": TOP2_MIN, "emb_min": EMB_MIN, "id_min": ID_MIN, "instr_v115": INSTR_V115, "instr_full": INSTR_FULL,
                       "k": {"a": K_A, "b": K_B, "c": K_C, "d": K_D}, "early": sorted(EARLY)},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "counts": counts, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
