#!/usr/bin/env python3
# BQGATE: five frozen predictions; families and unit list fixed before the run; cue positions from the token diff, no offsets.
"""Tier-5: where is each behaviour's CUE FEATURE computed? Embedding-vs-writes source map at the cue position for eight hub behaviours.

v166 (lexical number): the noun embedding alone equals the whole donor (0.997/1.000 with every cue-position write following it)
but is NOT read directly -- with all 22 noun-position writes clamped to base the donor embedding is inert (whole 0.088/0.163);
the writes alone carry it (0.96/1.01 of full); the largest single source is mlp:01 (0.56/0.65 whole), every attention unit at
the noun <= 0.013. That was one family. bilin18 re-enters x0 = rms_norm(wte) at EVERY block, so a token-identity cue
(both/either, than/as, to/that ...) could in principle be read straight off the embedding by the heads that carry it. This rung
runs the same three conditions plus a 37-unit single-source scan (embed + attn/mlp writes of all 18 layers, clamped at the cue
position only) on eight single-token-cue hub behaviours, both parities, whole-model recovery: coordination_agreement,
correlative_both_either, correlative_both_neither, correlative_either_neither, degree_frame, finiteness_selection,
polarity_state, possessive_adjacent. Cue positions per row from g.cue_positions (common prefix/suffix); rows whose cue spans
more than one token on either side are dropped and counted (possessive_adjacent has 1 unequal-length row per parity).
Row population (ops/row_population.py, A1 parity 0): coordination 'cue columns {2: 1}', both_either '{3: 1}', both_neither
'{3: 1}', either_neither '{4: 1}', degree_frame '{5: 1}', finiteness '{2: 1}', polarity '{3: 1}', possessive_adjacent
'rows 16 (unequal/misaligned 1); {1: 15}'.
Smoke (CPU, 4 rows per parity, disclosed): direct (embed only, writes base) / computed (writes only): coordination 0.33/0.54, 0.51/0.69;
both_either 0.71/0.24, 0.80/0.33; both_neither 0.67/0.25, 0.78/0.38; either_neither 0.77/0.16, 0.85/0.24; degree_frame 0.49/0.12,
0.71/0.39; finiteness 0.81/0.54, 0.45/0.21; polarity 0.50/0.41, 0.52/0.58; possessive_adjacent 0.14/0.84, 0.17/0.83. Instrument
1.000 on all 16; the embedding alone (writes live) 1.000 on all 16. Largest write mlp:01 on 14/16 (degree p0 mlp:00 at 0.013,
finiteness p1 attn:06 at 0.038); largest attention unit <= 0.044 everywhere; possessive mlp:01 0.70/0.61 = 0.83/0.74 of computed.
Singles-sum / computed 0.7-3.2 (unstable where computed is small; not registered). 83 CPU-s.

Registered before the run (16 cells = 8 families x 2 parities; 14 FUNCTION-WORD cells = the seven non-possessive families, 2 LEXICAL
cells = possessive_adjacent; the bars encode what I believe after the 4-row smoke -- my pre-smoke prior was v166's "not read
directly" for every family, and the smoke refuted it on every function-word cell, so the registered claim is the dichotomy):
  pred_a_instrument       donor embedding + all 36 cue-position writes clamped to donor = the interchange: whole-model 0.90-1.05
                          on every cell (the cue position is the only differing position, so this must hold or the map is void)
  pred_b_direct_dichotomy donor embedding with all 36 writes clamped to base recovers >= 0.30 on every function-word cell (the
                          token identity is read straight off the embedding / x0 re-entry) and <= 0.25 on both lexical cells
  pred_c_computed_dichotomy  writes clamped to donor with the base embedding recover 0.05-0.70 of the instrument on every
                          function-word cell and 0.75-1.00 on both lexical cells (the lexical cue is computed, as in v166)
  pred_d_largest_is_early_mlp  the largest single-unit write source is an MLP in layers 0-4 on >= 14 of 16 cells
  pred_e_lexical_mlp01    possessive_adjacent, both parities: mlp:01 is the largest single write and holds 0.5-1.0 of the
                          writes-only value (v166's lexical-number mechanism replicated on a second lexical family)
Reported, unregistered: the full 37-unit map per cell, the largest attention unit, the largest single's share of writes-only,
dropped-row counts and cue positions.
Smoke: V170_SMOKE=<out.json> -> CPU, V170_SMOKE_ROWS rows per parity (default 4).
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_cue_source_map_eight_v170_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state", "possessive_adjacent")
N_LAYERS = 18
UNITS = ["embed"] + [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"instr_band": [0.90, 1.05], "fw_direct_min": 0.30, "lex_direct_max": 0.25, "fw_computed_band": [0.05, 0.70], "lex_computed_band": [0.75, 1.00],
        "early_mlp_layers": [0, 4], "early_mlp_min_cells": 14, "lex_mlp01_share_band": [0.5, 1.0]}
LEXICAL = ("possessive_adjacent",)
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 1200, 24000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_cue_source_map_eight_v170", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V170_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V170_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    R = {}

    def cell(fam, par):
        """one (family, parity): own scope so every closure binds this cell's batch and positions (v167 lesson)."""
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[fam]}")
        a1 = cut(g.rows_of(m, "A1")[0 if par == "p0" else 1::2])
        prep = g.prepare(backend, a1)
        fb, fd = g.cue_positions(prep.base_batch, prep.donor_batch, which="first")
        lb, ld = g.cue_positions(prep.base_batch, prep.donor_batch, which="last")
        keep = [i for i in range(len(a1)) if fb[i] == lb[i] and fd[i] == ld[i]]
        dropped = len(a1) - len(keep)
        if dropped:
            a1 = [a1[i] for i in keep]
            prep = g.prepare(backend, a1)
            lb, ld = g.cue_positions(prep.base_batch, prep.donor_batch, which="last")
        batch, db = prep.base_batch, prep.donor_batch
        rows = len(batch.row_ids)
        ar = torch.arange(rows, device=backend.device)
        cb_t, cd_t = torch.tensor(lb, device=backend.device), torch.tensor(ld, device=backend.device)
        def capture(bt, pos_t):
            cap = {}
            hs = [model.transformer.wte.register_forward_hook(lambda m_, a, o: cap.__setitem__("embed", o[ar, pos_t].detach().clone()))]
            for l in range(N_LAYERS):
                hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda m_, a, l=l: cap.__setitem__(f"attn:{l:02d}", a[0][ar, pos_t].detach().clone())))
                hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(lambda m_, a, o, l=l: cap.__setitem__(f"mlp:{l:02d}", o[ar, pos_t].detach().clone())))
            try:
                with torch.no_grad():
                    g.forward_units(backend, bt, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return cap
        cap, capb = capture(db, cd_t), capture(batch, cb_t)
        def clamped(spec):
            """spec {unit: 'donor'|'base'}: clamp those cue-position writes (base rows, base cue position); whole-model recovery."""
            hs = []
            val = lambda u: (cap if spec[u] == "donor" else capb)[u]
            if "embed" in spec:
                def eh(m_, a, o):
                    o = o.clone(); o[ar, cb_t] = val("embed").to(o.dtype); return o
                hs.append(model.transformer.wte.register_forward_hook(eh))
            for u in spec:
                if u == "embed": continue
                kind, l = u.split(":"); l = int(l)
                if kind == "attn":
                    def ph(m_, a, u=u):
                        v = a[0].clone(); v[ar, cb_t] = val(u).to(v.dtype); return (v,) + tuple(a[1:])
                    hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(ph))
                else:
                    def mh(m_, a, o, u=u):
                        o = o.clone(); o[ar, cb_t] = val(u).to(o.dtype); return o
                    hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(mh))
            try:
                with torch.no_grad():
                    out = g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        writes = [u for u in UNITS if u != "embed"]
        single = {u: clamped({u: "donor"}) for u in UNITS}
        instr = clamped({u: "donor" for u in UNITS})
        direct = clamped({"embed": "donor", **{u: "base" for u in writes}})
        computed = clamped({u: "donor" for u in writes})
        best = max(writes, key=lambda u: single[u])
        best_attn = max((u for u in writes if u.startswith("attn")), key=lambda u: single[u])
        return {"rows": rows, "dropped_multi_token_rows": dropped, "cue_positions_base": lb, "cue_positions_donor": ld,
                "instrument_embed_and_writes": instr, "direct_embed_only": direct, "computed_writes_only": computed,
                "computed_over_instrument": round(computed / instr, 3) if abs(instr) > 1e-6 else None,
                "single": single, "largest_write": best, "largest_write_value": single[best],
                "largest_write_share_of_computed": round(single[best] / computed, 3) if abs(computed) > 1e-6 else None,
                "largest_attn": best_attn, "largest_attn_value": single[best_attn],
                "singles_sum_over_computed": round(sum(single[u] for u in writes) / computed, 3) if abs(computed) > 1e-6 else None}

    for fam in FAMILIES:
        for par in ("p0", "p1"):
            S = cell(fam, par)
            R[f"{fam}:{par}"] = S
            print(fam, par, {k: v for k, v in S.items() if k not in ("single", "cue_positions_base", "cue_positions_donor")}, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_cue_source_map_eight_v170", "candidate_id": "corpus.unit_tier5_cue_source_map_eight_v170",
              "bars": BARS, "families": list(FAMILIES), "units": UNITS, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    cells = list(R.values())
    fw = [S for k, S in R.items() if k.split(":")[0] not in LEXICAL]
    lex = [S for k, S in R.items() if k.split(":")[0] in LEXICAL]
    pred_a = all(B["instr_band"][0] <= S["instrument_embed_and_writes"] <= B["instr_band"][1] for S in cells)
    pred_b = all(S["direct_embed_only"] >= B["fw_direct_min"] for S in fw) and all(S["direct_embed_only"] <= B["lex_direct_max"] for S in lex)
    c = lambda S: S["computed_over_instrument"] if S["computed_over_instrument"] is not None else float("nan")
    pred_c = all(B["fw_computed_band"][0] <= c(S) <= B["fw_computed_band"][1] for S in fw) and all(B["lex_computed_band"][0] <= c(S) <= B["lex_computed_band"][1] for S in lex)
    def early_mlp(S):
        kind, l = S["largest_write"].split(":")
        return kind == "mlp" and B["early_mlp_layers"][0] <= int(l) <= B["early_mlp_layers"][1]
    pred_d = sum(early_mlp(S) for S in cells) >= B["early_mlp_min_cells"]
    pred_e = all(S["largest_write"] == "mlp:01" and S["largest_write_share_of_computed"] is not None
                 and B["lex_mlp01_share_band"][0] <= S["largest_write_share_of_computed"] <= B["lex_mlp01_share_band"][1] for S in lex)
    return {"pred_a_instrument": pred_a, "pred_b_direct_dichotomy": pred_b, "pred_c_computed_dichotomy": pred_c,
            "pred_d_largest_is_early_mlp": pred_d, "pred_e_lexical_mlp01": pred_e}


if __name__ == "__main__":
    main()
