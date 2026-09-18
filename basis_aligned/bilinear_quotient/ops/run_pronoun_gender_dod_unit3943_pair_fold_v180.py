#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_top_pair_involves_the_embedding pred_c_mlp6_pairs_are_net_negative pred_d_pairs_with_8_1_carry_030
"""Pronoun gender he/she DoD (v180): the FEMALE-noun detector 3943's PRODUCT (v178's fold on the other MLP-8 gender unit; same predictions: is it the same embedding x MLP-7 structure damped by MLP 6?), exactly, as writer-pair terms. u_3943(noun) = (L . x)(R . x) with x = x_8[noun] / rms
and x_8 a sum of 26 writers (embedding, attn / mlp totals of blocks 0-7, the nine heads of block 8; v16's decomposition): u = sum_{a,b} (L . C_a)(R . C_b) / rms^2.
Pooled male - female contrast of every pair term; this is the quantity the edits act on, so it carries the signs the factor-level fold (v171) could not:
v177 showed MLP-6 unit 3230 DAMPS the detector although it feeds both factors.
PREDICTIONS (scored as written; failures preserved; priors unsure except c, which v177 implies)
    pred_a_pair_closure                    sum of pair terms = captured u_3943(noun) within relative 1e-3, every row
    pred_b_top_pair_involves_the_embedding the largest |pooled pair term| has the embedding as a factor
    pred_c_mlp6_pairs_are_net_negative     the net share of pairs with mlp:06 as a factor is < 0 (the damping seen in v177)
    pred_d_pairs_with_8_1_carry_030        pairs with attnhead:08:1 as a factor carry >= 0.30 of the contrast
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
OUT = ROOT / "circuits/followups/pronoun_gender_dod_unit3943_pair_fold_v180_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_unit3943_pair_fold_v180"
UNIT, CLOSURE_TOL, H81_MIN = 3943, 1e-3, 0.30
FORWARDS_MAX = 4
WRITERS = v16.WRITERS
PREDICTIONS = {"pred_a_pair_closure": "<= 1e-3", "pred_b_top_pair_involves_the_embedding": "embed factor", "pred_c_mlp6_pairs_are_net_negative": "< 0", "pred_d_pairs_with_8_1_carry_030": ">= 0.30"}


def main() -> None:
    rows, he, she = g.build()
    nouns = {L._single(" " + w) for p in g.PAIRS for w in p}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "writers": WRITERS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "h81_min": H81_MIN}}
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
    n = len(WRITERS); closure, tables = 0.0, []
    for row, tr in zip(rows, traces):
        pos = noun_of(row); C = v16.writers_at_block8_input(tr, pos)
        x = sum(C.values()); rms = float(x.pow(2).mean().sqrt())
        M = torch.stack([C[w] for w in WRITERS]).to(Lrow.device)
        la, rb = M @ Lrow, M @ Rrow
        T = torch.outer(la, rb) / (rms * rms)
        xin = F.rms_norm(x.to(Lrow.device), (x.shape[-1],)); true = float((Lrow @ xin) * (Rrow @ xin))
        closure = max(closure, abs(float(T.sum()) - true) / max(abs(true), 1e-6)); tables.append(T.cpu())
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    acc = torch.zeros(n, n)
    for i, row in enumerate(rows):
        if not row.present: continue
        acc += tables[i] - tables[partner[(row.construction, row.group, False)]]
    contrast = float(acc.sum()); sym = (acc + acc.T) / 2
    shares = {}
    for a in range(n):
        for b in range(a, n):
            shares[f"{WRITERS[a]} x {WRITERS[b]}"] = float(sym[a, b] * (1 if a == b else 2)) / contrast
    ranked = sorted(shares, key=lambda k: -abs(shares[k]))
    factor_share = lambda name: sum(v for k, v in shares.items() if name in k.split(" x "))
    report = {"contrast": contrast, "top12": [(k, shares[k]) for k in ranked[:12]], "embed_pairs": factor_share("embed"), "mlp6_pairs": factor_share("mlp:06"), "head_8_1_pairs": factor_share("attnhead:08:1"), "attn6_pairs": factor_share("attn:06"), "mlp7_pairs": factor_share("mlp:07")}
    print("contrast", round(contrast, 2), "top", [(k, round(v, 3)) for k, v in report["top12"][:8]]); print("factor shares: embed", round(report["embed_pairs"], 3), "mlp:06", round(report["mlp6_pairs"], 3), "8.1", round(report["head_8_1_pairs"], 3), "attn:06", round(report["attn6_pairs"], 3), "mlp:07", round(report["mlp7_pairs"], 3))
    predictions = {"pred_a_pair_closure": closure <= CLOSURE_TOL, "pred_b_top_pair_involves_the_embedding": "embed" in ranked[0].split(" x "), "pred_c_mlp6_pairs_are_net_negative": report["mlp6_pairs"] < 0, "pred_d_pairs_with_8_1_carry_030": report["head_8_1_pairs"] >= H81_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_gender_dod_unit3152_pair_fold_result_v178", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "shares": shares, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
