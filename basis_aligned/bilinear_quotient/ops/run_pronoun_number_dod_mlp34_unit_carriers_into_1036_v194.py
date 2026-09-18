#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_carrier_closure pred_b_top_10_units_carry_half pred_c_top_50_units_carry_080 pred_d_largest_unit_supports
"""Pronoun number they/he DoD (v194): carrier-level unit census of MLP 3 and MLP 4 into MLP-5 unit 1036 (v192: the largest MLP-5 carrier of the number
into 2483; v193: 1036 is carried by mlp:03 21% and mlp:04 19%). v192's identity per source block s: unit j's carrier over a pair is da_j mB + mA db_j with
c_j = (prod of lam0 over blocks s+1..5) h_j Down_s[:, j]; the units of a block sum to that block's carrier term (closure, tolerance 1e-2 after v192's fp32
lesson: 3.9e-3 with 4608 summands). Both sources captured in one pass.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_carrier_closure           per-pair identity within relative 1e-2, both sources
    pred_b_top_10_units_carry_half   top-10 |pooled| units carry >= 0.50 of the block's carrier total, both sources
    pred_c_top_50_units_carry_080    top-50 >= 0.80, both
    pred_d_largest_unit_supports     the largest unit has the sign of the block's carrier total, both
PRICE (registered maximum): 3 batches x 1 forward = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp34_unit_carriers_into_1036_v194_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp34_unit_carriers_into_1036_v194"
UNIT, SRCS, DST, CLOSURE_TOL, TOP10_MIN, TOP50_MIN, BATCH = 1036, (3, 4), 5, 1e-2, 0.50, 0.80, 32
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_carrier_closure": "<= 1e-2 x 2", "pred_b_top_10_units_carry_half": ">= 0.50", "pred_c_top_50_units_carry_080": ">= 0.80", "pred_d_largest_unit_supports": "sign of the MLP-5 carrier total"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "source_layers": SRCS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top10_min": TOP10_MIN, "top50_min": TOP50_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); blocks = model.transformer.h; dst = blocks[DST].mlp
    Lrow, Rrow = dst.Left.weight.detach().float()[UNIT], dst.Right.weight.detach().float()[UNIT]
    srcs = {}
    for s_ in SRCS:
        m = blocks[s_].mlp; Dw, bb = m.Down.weight.detach().float(), m.Down_bias.detach().float(); scale = 1.0
        for l in range(s_ + 1, DST + 1): scale *= float(blocks[l].lambdas[0])
        srcs[s_] = (m, scale * (Lrow @ Dw), scale * (Rrow @ Dw), scale * float(Lrow @ bb), scale * float(Rrow @ bb))
    per_row, forwards = {s_: [] for s_ in SRCS}, 0
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); pos = [noun_of(r) for r in chunk]; idx = torch.arange(len(chunk))
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_, hs = x, None, {}
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l in srcs: hs[l] = dod_units.hidden(model, srcs[l][0], xin)[idx, pos].float()
                if l == DST: xd = x[idx, pos].float(); break
                x = x + block.mlp(xin)
            forwards += 1
            for i in range(len(chunk)):
                xi = xd[i].to(Lrow.device); rms = float(xi.pow(2).mean().sqrt()); A, B = float(Lrow @ xi) / rms, float(Rrow @ xi) / rms
                for s_, (m, lD, rD, lb, rb) in srcs.items():
                    hi = hs[s_][i].to(Lrow.device)
                    per_row[s_].append({"A": A, "B": B, "a": (lD * hi / rms).cpu(), "b": (rD * hi / rms).cpu(), "a_bias": lb / rms, "b_bias": rb / rms})
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    report, closure = {}, 0.0
    predictions = {"pred_a_carrier_closure": True, "pred_b_top_10_units_carry_half": True, "pred_c_top_50_units_carry_080": True, "pred_d_largest_unit_supports": True}
    for s_ in SRCS:
        carrier = torch.zeros(len(srcs[s_][1])); total, n = 0.0, 0
        for i, row in enumerate(rows):
            if not row.present: continue
            P, S = per_row[s_][i], per_row[s_][partner[(row.construction, row.group, False)]]
            mA, mB = (P["A"] + S["A"]) / 2, (P["B"] + S["B"]) / 2
            terms = (P["a"] - S["a"]) * mB + mA * (P["b"] - S["b"])
            aP, aS = float(P["a"].sum()) + P["a_bias"], float(S["a"].sum()) + S["a_bias"]; bP, bS = float(P["b"].sum()) + P["b_bias"], float(S["b"].sum()) + S["b_bias"]
            ref = (aP - aS) * mB + mA * (bP - bS)
            closure = max(closure, abs(float(terms.sum()) - ref) / max(abs(ref), 1e-6)); carrier += terms; total += ref; n += 1
        order = torch.argsort(carrier.abs(), descending=True); share = lambda k: float(carrier[order[:k]].sum()) / total
        top = [(int(j), float(carrier[j])) for j in order[:40]]
        report[str(s_)] = {"carrier_total": total, "shares": {str(k): share(k) for k in (5, 10, 20, 50, 100, 500)}, "top_units": top}
        print("mlp", s_, "carrier total", round(total, 3), "top-10", round(share(10), 3), "top-50", round(share(50), 3), "top", [(j, round(v, 2)) for j, v in top[:10]])
        predictions["pred_b_top_10_units_carry_half"] &= share(10) >= TOP10_MIN; predictions["pred_c_top_50_units_carry_080"] &= share(50) >= TOP50_MIN; predictions["pred_d_largest_unit_supports"] &= (top[0][1] > 0) == (total > 0)
    predictions["pred_a_carrier_closure"] = closure <= CLOSURE_TOL
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp5_unit_carriers_into_2483_result_v192", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "sources": report, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
