#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: the cue route PER CUE COLUMN, on the three multi-cue sets -- does the route follow the cue token's class?

v141 (23 reader-set pairs, 9 one-cue sets): every counting reader on a noun-cue set (number, possessives) reads the
layer-writes route w; every reader on a function-word-cue set (either/neither, so/too, every/all) reads the token-
embedding route e -- 18/18, but confounded with behaviour family and with head identity. The three sets v141 skipped
have several cue columns each, of different classes, so one head can be tested on two classes at once:
  interrogative  'The director did request' / 'Did the director request'   columns 0 (The/Did: function), 1, 2 (mixed pairs)
  narrative      'Every winter ... stands' / 'Last winter ... stood'          column 0 (Every/Last: function), column 4 (verb: content)
  preposition    'The stew will consist' / 'The limit will depend'           column 1 (noun), column 3 (verb): both content
Readers per set from the v129 receipt (cue-column loss >= 0.04). Per (set, head, column): caches captured at that
column on Dc_j / Bc_j; w_j, e_j, live_j from the column's own P[t, c_j] v[c_j]; v139 slot-adds at t on the base batch.

Registered before the run (a triple counts when |m_we| >= 0.05; classes fixed above, not from the result):
  pred_a_instrument   m_none = 0 and |m_rand| <= 0.03 on every triple; 0.8 <= m_we / m_live <= 1.2 on counting triples
  pred_b_token_class  every counting FUNCTION column has ratio_e >= 0.6 and every counting CONTENT column has
                      ratio_w >= 0.6 (mixed columns excluded; >= 3 counting classed triples, else False)
  pred_c_narrative    15:05 counts on both columns and reads e on column 0 (ratio_e >= 0.6), w on column 4 (ratio_w >= 0.6)
  pred_d_preposition  06:03 has ratio_w >= 0.6 on every counting preposition column (>= 1)
  pred_e_interrogative column 0 (The/Did) is e-majority (ratio_e >= 0.6) on every counting interrogative reader (>= 1)
Prior: a firm; b the hypothesis; c its sharpest form (same head, two classes) -- genuinely open: a head may have one
readout axis and read whichever route carries it; d likely (verb selects the preposition, computed); e unsure
(03:00 is a cross-term reader, v130).
Smoke: V142_SMOKE=<out.json>, V142_SMOKE_SET=<set> -> CPU, 4 rows.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_batch_v112 as v112
import run_unit_tier5_carrier_relay_v120 as v120
import run_unit_tier5_near_carrier_heads_v123 as v123
import run_unit_tier5_near_value_source_v131 as v131
import run_unit_tier5_near_value_mid_remainder_v132 as v132

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_cue_route_by_column_v142_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
CUE_MIN, COUNT_MIN, RAND_MAX, REPRO_LO, REPRO_HI, ROUTE_MAJ, MIN_CLASSED = 0.04, 0.05, 0.03, 0.8, 1.2, 0.6, 3
SETS = ("interrogative_licensing", "narrative_tense", "preposition_selection")
CLASS = {("interrogative_licensing", 0): "function", ("narrative_tense", 0): "function", ("narrative_tense", 4): "content",
         ("preposition_selection", 1): "content", ("preposition_selection", 3): "content"}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def pairs_from_v129():
    r = json.loads(V129_RECEIPT.read_text())["behaviours"]
    return {n: [h for h, v in b["per_head"].items() if v.get("cue", 0) >= CUE_MIN] for n, b in r.items() if "skipped" not in b and n in SETS}


def _plan():
    return {"candidate_id": "corpus.unit_tier5_cue_route_by_column_v142", "behaviours": 3, "pairs": 9,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V142_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    sets = pairs_from_v129()
    if smoke:
        pick = os.environ.get("V142_SMOKE_SET") or "interrogative_licensing"
        sets = {pick: sets[pick]}
    R, skipped = {}, {}
    for n, readers in sets.items():
        t1 = time.perf_counter()
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        rows, geo = [], []
        for r in v123.rows_of(m, 1, smoke):
            t = r["donor_semantic_position"]
            diff = [p for p in range(t) if r["base_ids"][p] != r["donor_ids"][p]]
            rows.append(r); geo.append({"t": t, "cues": diff})
        cols = sorted({tuple(ge["cues"]) for ge in geo})
        if len(rows) < 4 or len(cols) != 1:
            skipped[n] = f"{len(rows)} rows, cue column sets {cols}"; print(n, "skipped", skipped[n], flush=True); continue
        cols = list(cols[0])
        prep = g.prepare(backend, rows)
        D, B = prep.donor_batch, prep.base_batch
        d_ax, b_ax = prep.donor_axis, prep.base_axis
        top = max(g.unit_layer(h) for h in readers)
        below_all = list(range(0, top))
        tpos = [ge["t"] for ge in geo]
        C = {}
        for c in cols:
            cuepos = tuple([c] * len(rows))
            Dc, Bc = dataclasses.replace(D, semantic_positions=cuepos), dataclasses.replace(B, semantic_positions=cuepos)
            C[c] = {"Dc": Dc, "Bc": Bc, "hd": v120.head_cache(backend, Dc, below_all), "md": v120.mlp_cache(backend, Dc, below_all),
                    "hb": v120.head_cache(backend, Bc, below_all), "mb": v120.mlp_cache(backend, Bc, below_all)}

        def cue_items(units, donor, c):
            hc, mc = (C[c]["hd"], C[c]["md"]) if donor else (C[c]["hb"], C[c]["mb"])
            return ([(u, 0, hc) for u in units if u.startswith("attn")], [(u, 0, mc) for u in units if u.startswith("mlp")])

        for reader in readers:
            layer, h = g.unit_layer(reader), int(reader.rsplit(":", 1)[1])
            units = v132.units_of(range(0, layer))
            PB, VB = v131.capture_with_clamp(backend, B, [], [], layer)
            PD, VD = v131.capture_with_clamp(backend, D, [], [], layer)

            for c in cols:
                def delta(P, V):
                    return [P[i, h, ge["t"], c] * V[i, c, h, :] - PB[i, h, ge["t"], c] * VB[i, c, h, :] for i, ge in enumerate(geo)]

                live = delta(PD, VD)
                hi, mi = cue_items(units, True, c)
                P, V = v131.capture_with_clamp(backend, C[c]["Bc"], hi, mi, layer)
                w_cue = delta(P, V)
                hi, mi = cue_items(units, False, c)
                P, V = v131.capture_with_clamp(backend, C[c]["Dc"], hi, mi, layer)
                e_cue = delta(P, V)
                gen = torch.Generator().manual_seed(0)
                rand = [torch.randn(w.shape, generator=gen).to(w.device) * (w.norm() / g.HEAD_DIM ** 0.5) for w in w_cue]

                def add_margin(vecs):
                    def pre(_m, args):
                        v = args[0].clone()
                        for i, vec in enumerate(vecs):
                            v[i, tpos[i], h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM] += vec.to(v.device, v.dtype)
                        return (v,) + tuple(args[1:])
                    hdl = backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(pre)
                    try:
                        out = g.forward_units(backend, B)
                    finally:
                        hdl.remove()
                    vals = [float(a) - float(f) for a, f in out.tolist()]
                    per = [(-p - bb) / (dd - bb) for dd, bb, p in zip(d_ax, b_ax, vals) if abs(dd - bb) > 1e-6]
                    return round(sum(per) / len(per), 3)

                A = {"column": c, "class": CLASS.get((n, c), "mixed"),
                     "m_none": add_margin([torch.zeros_like(w) for w in w_cue]), "m_live": add_margin(live), "m_w": add_margin(w_cue),
                     "m_e": add_margin(e_cue), "m_we": add_margin([w + e for w, e in zip(w_cue, e_cue)]), "m_rand": add_margin(rand)}
                cos = [float(torch.nn.functional.cosine_similarity(w, e, dim=0)) for w, e in zip(w_cue, e_cue)]
                A["rowcos_mean"] = round(sum(cos) / len(cos), 3)
                ok = abs(A["m_we"]) >= COUNT_MIN
                A["counts"] = ok
                A["ratio_w"] = round(A["m_w"] / A["m_we"], 3) if ok else None
                A["ratio_e"] = round(A["m_e"] / A["m_we"], 3) if ok else None
                A["repro"] = round(A["m_we"] / A["m_live"], 3) if abs(A["m_live"]) > 1e-6 else None
                R[(n, reader, c)] = A
                print(n, reader, A, round(time.perf_counter() - t0), "s", flush=True)

    triples = {f"{n}|{r}|{c}": A for (n, r, c), A in R.items()}
    counting = {k: A for k, A in R.items() if A["counts"]}
    pred_a = bool(R) and all(A["m_none"] == 0 and abs(A["m_rand"]) <= RAND_MAX for A in R.values()) and \
        all(A["repro"] is not None and REPRO_LO <= A["repro"] <= REPRO_HI for A in counting.values())
    classed = {k: A for k, A in counting.items() if A["class"] != "mixed"}
    class_ok = [A["ratio_e"] >= ROUTE_MAJ if A["class"] == "function" else A["ratio_w"] >= ROUTE_MAJ for A in classed.values()]
    pred_b = len(classed) >= MIN_CLASSED and all(class_ok)
    nar = {c: A for (n, r, c), A in counting.items() if n == "narrative_tense" and r == "attn:15:head:05"}
    pred_c = 0 in nar and 4 in nar and nar[0]["ratio_e"] >= ROUTE_MAJ and nar[4]["ratio_w"] >= ROUTE_MAJ
    prep_rows = [A for (n, r, c), A in counting.items() if n == "preposition_selection" and r == "attn:06:head:03"]
    pred_d = len(prep_rows) >= 1 and all(A["ratio_w"] >= ROUTE_MAJ for A in prep_rows)
    inter = [A for (n, r, c), A in counting.items() if n == "interrogative_licensing" and c == 0]
    pred_e = len(inter) >= 1 and all(A["ratio_e"] >= ROUTE_MAJ for A in inter)
    predictions = {"pred_a_instrument": pred_a, "pred_b_token_class": pred_b, "pred_c_narrative": pred_c,
                   "pred_d_preposition": pred_d, "pred_e_interrogative": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_cue_route_by_column_v142", "candidate_id": "corpus.unit_tier5_cue_route_by_column_v142",
              "bars": {"cue_min": CUE_MIN, "count_min": COUNT_MIN, "rand_max": RAND_MAX, "repro": [REPRO_LO, REPRO_HI], "route_maj": ROUTE_MAJ, "min_classed": MIN_CLASSED},
              "classes": {f"{n}|{c}": k for (n, c), k in CLASS.items()},
              "counts": {"triples": len(R), "counting": len(counting), "classed": len(classed), "class_ok": sum(class_ok), "narrative_cols": sorted(nar),
                         "prep_cols": len(prep_rows), "inter_col0": len(inter)},
              "skipped": skipped, "triples_detail": triples, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
