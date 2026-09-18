#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_factor_closure pred_b_right_factor_reads_the_embedding pred_c_heads_write_little_of_the_right_factor pred_d_left_factor_reads_the_embedding
"""Pronoun gender he/she DoD (v171): what do the two factors of MLP-8 unit 3152 (the male-noun detector, v164 / v167) READ at the noun?

u_3152(x) = (L . x)(R . x) with x = rms(x_8[noun]) and x_8 = live_8 + attn_8 an exact lambda-weighted sum of writers: embedding, heads of blocks 0-7
(as block totals), MLPs 0-7, and the nine heads of block 8 (v16's `writers_at_block8_input`). Each factor is linear in x, so the oriented contrast
(male - female over aligned pairs) of L . x and of R . x splits exactly by writer, scaled by 1 / rms(x_8). If the embedding carries the factors, the
detector reads the token itself and MLP 8's gender unit closes to the input; if MLPs 4-7 carry them, one more layer of the port opens.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_factor_closure                    sum of writer terms = the captured factor value within relative 1e-3, every row, both factors
    pred_b_right_factor_reads_the_embedding  embedding share of the R-factor contrast >= 0.50 (R has cos -0.38 with the embedding gender axis, v167)
    pred_c_heads_write_little_of_the_right_factor  attention writers (blocks 0-8 together) <= 0.25 of the R-factor contrast
    pred_d_left_factor_reads_the_embedding   embedding share of the L-factor contrast >= 0.30
PRICE (registered maximum): 2 batches x 1 positional trace = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_aspectual_dod_mlp8_pair_fold_v16 as v16
import run_pronoun_gender_dod_battery_v71 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_gender_dod_unit3152_input_fold_v171_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_unit3152_input_fold_v171"
UNIT = 3152
CLOSURE_TOL, EMB_R_MIN, HEADS_R_MAX, EMB_L_MIN = 1e-3, 0.50, 0.25, 0.30
FORWARDS_MAX = 4
WRITERS = v16.WRITERS
PREDICTIONS = {"pred_a_factor_closure": "<= 1e-3", "pred_b_right_factor_reads_the_embedding": ">= 0.50", "pred_c_heads_write_little_of_the_right_factor": "<= 0.25", "pred_d_left_factor_reads_the_embedding": ">= 0.30"}


def main() -> None:
    rows, he, she = g.build()
    nouns = {L._single(" " + w) for p in g.PAIRS for w in p}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "writers": WRITERS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "emb_r_min": EMB_R_MIN, "heads_r_max": HEADS_R_MAX, "emb_l_min": EMB_L_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend)
    mlp = model.transformer.h[8].mlp; Lrow, Rrow = mlp.Left.weight.detach().float()[UNIT], mlp.Right.weight.detach().float()[UNIT]
    forwards, traces = 0, []
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        traces.extend(L.forward_trace_positions(fw, chunk, lambda rw: [noun_of(rw)], upto_layer=9, head_write_layers=(8,))); forwards += 1
    closure, per_row = 0.0, []
    for row, tr in zip(rows, traces):
        pos = noun_of(row); C = v16.writers_at_block8_input(tr, pos)
        x8 = sum(C.values()); rms = float(x8.pow(2).mean().sqrt()); xin = x8 / rms
        entry = {}
        for name, w in (("L", Lrow), ("R", Rrow)):
            w = w.to(xin.device); true = float(w @ xin); terms = {k: float(w @ v) / rms for k, v in C.items()}
            closure = max(closure, abs(sum(terms.values()) - true) / max(abs(true), 1e-6)); entry[name] = terms
        per_row.append(entry)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    report = {}
    for name in ("L", "R"):
        totals = {k: 0.0 for k in WRITERS}; contrast = 0.0
        for i, row in enumerate(rows):
            if not row.present: continue
            j = partner[(row.construction, row.group, False)]
            for k in WRITERS: totals[k] += per_row[i][name][k] - per_row[j][name][k]
            contrast += sum(per_row[i][name].values()) - sum(per_row[j][name].values())
        shares = {k: v / contrast for k, v in totals.items()}
        heads = sum(v for k, v in shares.items() if k.startswith("attn")); mlps = sum(v for k, v in shares.items() if k.startswith("mlp"))
        ranked = sorted(shares, key=lambda k: -abs(shares[k]))
        report[name] = {"contrast": contrast, "embed": shares["embed"], "heads": heads, "mlps": mlps, "top": [(k, shares[k]) for k in ranked[:8]], "shares": shares}
        print(name, "contrast", round(contrast, 3), "embed", round(shares["embed"], 3), "heads", round(heads, 3), "mlps", round(mlps, 3), "top", [(k, round(v, 3)) for k, v in report[name]["top"][:6]])
    predictions = {"pred_a_factor_closure": closure <= CLOSURE_TOL, "pred_b_right_factor_reads_the_embedding": report["R"]["embed"] >= EMB_R_MIN,
                   "pred_c_heads_write_little_of_the_right_factor": abs(report["R"]["heads"]) <= HEADS_R_MAX, "pred_d_left_factor_reads_the_embedding": report["L"]["embed"] >= EMB_L_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_gender_dod_unit3152_input_fold_result_v171", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "factors": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
