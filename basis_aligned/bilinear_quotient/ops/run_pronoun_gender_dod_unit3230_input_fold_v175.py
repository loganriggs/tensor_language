#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_factor_closure pred_b_embedding_carries_the_factors pred_c_no_single_earlier_module_above_030
"""Pronoun gender he/she DoD (v175): what do the two factors of MLP-6 unit 3230 (v173: ~82% of MLP 6's input to the male-noun detector 3152) READ at
the noun? Exact writer fold of rms(x_6[noun]) -- x_6 = live_6 + attn_6 = lambda-mixed sum of the embedding, heads of blocks 0-5, MLPs 0-5 (block totals) and
the nine heads of block 6 -- projected on 3230's Left and Right rows, oriented male - female. If the embedding dominates, the chain
token -> 3230 -> 3152 -> 9.6 closes to the input at unit grain; otherwise the next writer is named.
PREDICTIONS (scored as written; failures preserved; priors unsure): pred_a closure <= 1e-3 both factors; pred_b embedding share >= 0.50 for both factors;
pred_c no single non-embedding writer (a head of block 6, or one earlier block's attention or MLP total) carries >= 0.30 of either factor.
PRICE (registered maximum): 2 batches x 1 positional trace = 2 forwards; bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_pronoun_gender_dod_battery_v71 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_gender_dod_unit3230_input_fold_v175_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_unit3230_input_fold_v175"
UNIT, LAYER = 3230, 6
CLOSURE_TOL, EMB_MIN, SINGLE_MAX = 1e-3, 0.50, 0.30
FORWARDS_MAX = 4
WRITERS = ["embed"] + [f"{k}:{l:02d}" for l in range(LAYER) for k in ("attn", "mlp")] + [f"attnhead:{LAYER:02d}:{h}" for h in range(9)]
PREDICTIONS = {"pred_a_factor_closure": "<= 1e-3", "pred_b_embedding_carries_the_factors": ">= 0.50 x 2", "pred_c_no_single_earlier_module_above_030": "< 0.30"}


def writers_at_block_input(tr, pos, layer):
    x0 = tr[("embed", pos)]; C = {"embed": x0.clone()}
    for l in range(layer):
        l0, l1 = tr[f"lambda0:{l:02d}"], tr[f"lambda1:{l:02d}"]
        for k in C: C[k] = l0 * C[k]
        C["embed"] = C["embed"] + l1 * x0; C[f"attn:{l:02d}"] = tr[(f"attn:{l:02d}", pos)].clone(); C[f"mlp:{l:02d}"] = tr[(f"mlp:{l:02d}", pos)].clone()
    l0, l1 = tr[f"lambda0:{layer:02d}"], tr[f"lambda1:{layer:02d}"]
    for k in C: C[k] = l0 * C[k]
    C["embed"] = C["embed"] + l1 * x0
    for h in range(9): C[f"attnhead:{layer:02d}:{h}"] = tr[(f"attnhead:{layer:02d}:{h}", pos)].clone()
    return C


def main() -> None:
    rows, he, she = g.build()
    nouns = {L._single(" " + w) for p in g.PAIRS for w in p}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "layer": LAYER, "writers": WRITERS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "emb_min": EMB_MIN, "single_max": SINGLE_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); model = backend.model
    fw = L.ManualForward(backend)
    mlp = model.transformer.h[LAYER].mlp; Lrow, Rrow = mlp.Left.weight.detach().float()[UNIT], mlp.Right.weight.detach().float()[UNIT]
    forwards, traces = 0, []
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        traces.extend(L.forward_trace_positions(fw, chunk, lambda rw: [noun_of(rw)], upto_layer=LAYER + 1, head_write_layers=(LAYER,))); forwards += 1
    closure, per_row = 0.0, []
    for row, tr in zip(rows, traces):
        pos = noun_of(row); C = writers_at_block_input(tr, pos, LAYER)
        x = sum(C.values()); rms = float(x.pow(2).mean().sqrt()); xin = x / rms
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
        shares = {k: v / contrast for k, v in totals.items()}; ranked = sorted(shares, key=lambda k: -abs(shares[k]))
        report[name] = {"contrast": contrast, "embed": shares["embed"], "largest_non_embed": max((abs(v) for k, v in shares.items() if k != "embed"), default=0.0), "top": [(k, shares[k]) for k in ranked[:8]], "shares": shares}
        print(name, "contrast", round(contrast, 3), "embed", round(shares["embed"], 3), "top", [(k, round(v, 3)) for k, v in report[name]["top"][:7]])
    predictions = {"pred_a_factor_closure": closure <= CLOSURE_TOL, "pred_b_embedding_carries_the_factors": all(r["embed"] >= EMB_MIN for r in report.values()), "pred_c_no_single_earlier_module_above_030": all(r["largest_non_embed"] < SINGLE_MAX for r in report.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_gender_dod_unit3230_input_fold_result_v175", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "factors": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
