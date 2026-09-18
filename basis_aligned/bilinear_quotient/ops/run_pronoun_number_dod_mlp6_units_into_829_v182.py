#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_linear_closure pred_b_top_10_units_carry_half pred_c_top_50_units_carry_080 pred_d_largest_unit_supports
"""Pronoun number they/he DoD (v182): MLP 6 at UNIT grain as an input of the plural detector, MLP-8 unit 829, at PRODUCT level.
v181: pairs with mlp:06 carry 45% of 829's pooled plural-singular product contrast at the noun (embedding 26%, head 8.1 -1%). Which MLP-6 units?
v177's lesson: a factor-level census cannot give the sign of a unit's effect on the product, so this census is the exact leave-one-unit-out change of
the product: with x = x_8[noun] (pre-norm), c_j = lam0_8 lam0_7 h6_j Down6[:, j] the unit's write as it reaches block 8, and rms = rms(x) held,
    D_j = [ (L . c_j)(R . x) + (L . x)(R . c_j) - (L . c_j)(R . c_j) ] / rms^2  = u_829(x) - u_829(x - c_j).
Pooled plural - singular contrast of D_j per unit (aligned pairs, v76 rows). Closure: the linear parts summed over units plus the Down-bias terms
equal (L . m)(R . x) + (L . x)(R . m) with m the full MLP-6 write, every row.
PREDICTIONS (scored as written; failures preserved; priors unsure): pred_a closure <= 1e-3 every row; pred_b top-10 units carry >= 0.50 of the
pooled sum of D_j; pred_c top-50 >= 0.80; pred_d the largest |pooled D_j| is positive (that unit supports the detector rather than damping it).
PRICE (registered maximum): 3 batches x 1 forward = 3 forwards; bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp6_units_into_829_v182_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp6_units_into_829_v182"
UNIT, SRC, DST, CLOSURE_TOL, TOP10_MIN, TOP50_MIN, BATCH = 829, 6, 8, 1e-3, 0.50, 0.80, 32
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_linear_closure": "<= 1e-3", "pred_b_top_10_units_carry_half": ">= 0.50", "pred_c_top_50_units_carry_080": ">= 0.80", "pred_d_largest_unit_supports": "> 0"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "source_layer": SRC, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top10_min": TOP10_MIN, "top50_min": TOP50_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); blocks = model.transformer.h
    mlp8, mlp6 = blocks[DST].mlp, blocks[SRC].mlp
    Lrow, Rrow = mlp8.Left.weight.detach().float()[UNIT], mlp8.Right.weight.detach().float()[UNIT]
    D6, b6 = mlp6.Down.weight.detach().float(), mlp6.Down_bias.detach().float()
    scale = float(blocks[7].lambdas[0]) * float(blocks[8].lambdas[0])
    lD, rD = scale * (Lrow @ D6), scale * (Rrow @ D6)          # per-unit (L . Down6[:, j]), (R . Down6[:, j]) as they reach block 8
    lb, rb = scale * float(Lrow @ b6), scale * float(Rrow @ b6)
    per_row, closure, forwards = [], 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); pos = [noun_of(r) for r in chunk]; idx = torch.arange(len(chunk))
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_, h6 = x, None, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l == SRC:
                    h6 = dod_units.hidden(model, mlp6, xin)[idx, pos].float()
                if l == DST:
                    x8 = x[idx, pos].float(); break
                x = x + block.mlp(xin)
            forwards += 1
            for i in range(len(chunk)):
                xi, hi = x8[i].to(Lrow.device), h6[i].to(Lrow.device); rms2 = float(xi.pow(2).mean())
                Lx, Rx = float(Lrow @ xi), float(Rrow @ xi)
                lc, rc = lD * hi, rD * hi
                lin = lc * Rx + Lx * rc; D = (lin - lc * rc) / rms2
                m = scale * (D6 @ hi + b6); true = float(Lrow @ m) * Rx + Lx * float(Rrow @ m)
                recon = float(lin.sum()) + lb * Rx + Lx * rb
                closure = max(closure, abs(recon - true) / max(abs(true), 1e-6)); per_row.append({"noun": D.cpu()})
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    total = dod_units.pooled_contrast(rows, per_row, partner, "noun"); pooled = float(total.sum())
    order = torch.argsort(total.abs(), descending=True); share = lambda k: float(total[order[:k]].sum()) / pooled
    top = [(int(j), float(total[j])) for j in order[:40]]
    print("pooled sum of D_j", round(pooled, 3), "top-10", round(share(10), 3), "top-50", round(share(50), 3), "top", [(j, round(v, 2)) for j, v in top[:10]])
    predictions = {"pred_a_linear_closure": closure <= CLOSURE_TOL, "pred_b_top_10_units_carry_half": share(10) >= TOP10_MIN, "pred_c_top_50_units_carry_080": share(50) >= TOP50_MIN, "pred_d_largest_unit_supports": top[0][1] > 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp6_units_into_829_result_v182", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "pooled_sum": pooled, "shares": {str(k): share(k) for k in (5, 10, 20, 50, 100, 500)},
                               "top_units": top, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
