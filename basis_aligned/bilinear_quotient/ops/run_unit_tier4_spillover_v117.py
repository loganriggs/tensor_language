#!/usr/bin/env python3
# BQGATE: five frozen predictions; arms and bars fixed before the run; expansion and replay code reused from v115/v116 unchanged.
"""v117: the spillover hypothesis -- distant-cue heads read cue + the following token + the read position.

v116: the cue offsets suffice only when the cue is adjacent to the read position (3/14); for distant cues the
per-head top-2 offsets were the cue, the token AFTER the cue (offset cue-1) and the read position (offset 0). The
hypothesis this rung tests (not an arm chosen after the fact but a structural claim): by layers 4-5 the early MLPs
have written the cue's content onto the next token and onto the read position, and the set's heads read those three
positions through their value path. Same exact expansion (v115.capture), same exact-replacement replay (ODD), any
selection on EVEN.
    struct arm       offsets cue u {c - 1 : c in cue} u {0}, all writers
    top3 arm         the three offsets with the largest mean EVEN share per head, all writers
    struct_early     the struct offsets, writers of layer <= 4 only ({emb, attn_0..4, mlp_0..4})
    read arm         offset 0 only, all writers
    v116 arm         v116's cue arm recomputed (instrument)

REGISTERED BEFORE THE RUN (14 behaviours; fractions of the exact set interchange on ODD)
    pred_a_struct_sufficient   struct arm >= 0.80 on >= 10 of 14.       Worked: 0.86 True; 0.71 False.
    pred_b_top3                top3 arm >= 0.80 on >= 10 of 14.         Worked: 0.83 True; 0.74 False.
    pred_c_early_writers       struct_early >= 0.80 on >= 6 of 14.      Worked: 0.82 True; 0.55 False.
    pred_d_read_position       read arm >= 0.50 on >= 6 of 14.          Worked: 0.62 True; 0.18 False.
    pred_e_instrument          v116 arm within 0.01 of the v116 receipt's cue arm on 14/14 and full within 0.02 of exact.
    Prior: a 50%; b 55%; c 35%; d 40%; e 85%.
    Reading: a True = the head-level Tier-4 statement for distant-cue sets is "read cue, next token and read position";
    a False with b True = a fourth position matters -- name it from the receipt; both False = the sources are spread over
    >3 positions and the head-level story is a distributed read, reported as such.
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
import run_unit_tier4_offset_sufficiency_v116 as v116

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier4_spillover_v117_result.json"
V116 = ROOT / "circuits/followups/unit_tier4_offset_sufficiency_v116_result.json"
STRUCT_MIN, TOP3_MIN, EARLY_MIN, READ_MIN, INSTR_V116, INSTR_FULL, K_A, K_B, K_C, K_D = 0.80, 0.80, 0.80, 0.50, 0.01, 0.02, 10, 10, 6, 6
EARLY_LAYER = 4
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 2000, 64000


def _plan():
    return {"candidate_id": "corpus.unit_tier4_spillover_v117", "behaviours": 14,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def early(j):
    return j == "emb" or int(j.split("_")[1]) <= EARLY_LAYER


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    S = v115.sets()
    prior = json.loads(V116.read_text())["behaviours"]
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
        E, O = v116.deltas(cap["even"], prep["even"], units), v116.deltas(cap["odd"], prep["odd"], units)
        top3 = {}
        for u in units:
            sh = {}
            for row in E[u]:
                dz2 = float(row["dz"].dot(row["dz"])) or 1e-12
                for (o, j), v in row["dT"].items():
                    sh[o] = sh.get(o, 0.0) + float(v.dot(row["dz"])) / dz2 / len(E[u])
            top3[u] = [o for o, _ in sorted(sh.items(), key=lambda kv: -kv[1])[:3]]

        def struct(row):
            return set(row["cue"]) | {c - 1 for c in row["cue"] if c >= 1} | {0}

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
                "struct": replay(lambda u, row, k: k[0] in struct(row)),
                "top3": replay(lambda u, row, k: k[0] in top3[u]),
                "struct_early": replay(lambda u, row, k: k[0] in struct(row) and early(k[1])),
                "read": replay(lambda u, row, k: k[0] == 0),
                "v116_cue": replay(lambda u, row, k: k[0] in row["cue"])}
        arms = {k: frac(v) for k, v in arms.items()}
        R[n] = {"units": units, "exact_odd": round(exact, 3), "arms": arms, "top3_offsets": top3,
                "struct_offsets_odd": sorted({o for row in O[units[0]] for o in struct(row)}),
                "v116_cue": prior[n]["arms"]["cue"], "seconds": round(time.perf_counter() - t1, 1)}
        print(n, "exact", round(exact, 3), arms, round(time.perf_counter() - t0), "s", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"partial": True, "behaviours": R}, indent=2, sort_keys=True, default=str) + "\n")

    ok = lambda v, bar: v is not None and v >= bar
    counts = {"struct": sum(ok(R[n]["arms"]["struct"], STRUCT_MIN) for n in R), "top3": sum(ok(R[n]["arms"]["top3"], TOP3_MIN) for n in R),
              "struct_early": sum(ok(R[n]["arms"]["struct_early"], EARLY_MIN) for n in R), "read": sum(ok(R[n]["arms"]["read"], READ_MIN) for n in R)}
    predictions = {
        'pred_a_struct_sufficient': counts["struct"] >= K_A,
        'pred_b_top3': counts["top3"] >= K_B,
        'pred_c_early_writers': counts["struct_early"] >= K_C,
        'pred_d_read_position': counts["read"] >= K_D,
        'pred_e_instrument': all(R[n]["arms"]["v116_cue"] is not None and R[n]["v116_cue"] is not None
                                 and abs(R[n]["arms"]["v116_cue"] - R[n]["v116_cue"]) <= INSTR_V116
                                 and R[n]["arms"]["full"] is not None and abs(R[n]["arms"]["full"] - 1.0) <= INSTR_FULL for n in R),
    }
    summary = {n: R[n]["arms"] for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_tier4_spillover_result_v1", "candidate_id": "corpus.unit_tier4_spillover_v117",
              "counts": counts, "struct_sufficient": [n for n in R if ok(R[n]["arms"]["struct"], STRUCT_MIN)], "summary": summary, "behaviours": R,
              "bars": {"struct_min": STRUCT_MIN, "top3_min": TOP3_MIN, "early_min": EARLY_MIN, "read_min": READ_MIN, "instr_v116": INSTR_V116,
                       "instr_full": INSTR_FULL, "early_layer": EARLY_LAYER, "k": {"a": K_A, "b": K_B, "c": K_C, "d": K_D}},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "counts": counts, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
