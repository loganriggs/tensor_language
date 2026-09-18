#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_carrier_closure pred_b_top_10_units_carry_half pred_c_trio_in_top5 pred_d_largest_unit_supports pred_e_trio_carries_030
"""Pronoun number they/he DoD (v241): the MLP-6 inputs of the plural detector 829 at the VERB. v203 / v204: 829 re-fires at the verb and its carriers there are the
verb-position MLPs (MLP 6 13%, MLP 7 25%, MLP 5 18%, MLP 4 14%); at the noun MLP 6's part is carried by the trio {2483, 2826, 4131} (v182, 57%). Are the same MLP-6 units
the verb-site inputs (one number circuit re-run at a second position), or different ones? Carrier-level unit census (v192's identity) of MLP 6 into 829 at the VERB
(noun + 1), pooled plural - singular.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_carrier_closure           unit terms sum to MLP 6's carrier term within relative 1e-2, every pair
    pred_b_top_10_units_carry_half   top-10 |pooled| units carry >= 0.50 of MLP 6's carrier total at the verb
    pred_c_trio_in_top5              at least two of {2483, 2826, 4131} are among the top-5 |pooled| units at the verb
    pred_d_largest_unit_supports     the largest |pooled| unit has the sign of MLP 6's carrier total
    pred_e_trio_carries_030          the trio's summed carrier share at the verb is >= 0.30 of MLP 6's total (noun: 0.57)
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
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp6_unit_carriers_into_829_verb_v241_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp6_unit_carriers_into_829_verb_v241"
UNIT, SRC, DST, CLOSURE_TOL, TOP10_MIN, TRIO_MIN, BATCH = 829, 6, 8, 1e-2, 0.50, 0.30, 32
TRIO = (2483, 2826, 4131)
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_carrier_closure": "<= 1e-2", "pred_b_top_10_units_carry_half": ">= 0.50", "pred_c_trio_in_top5": ">= 2 of 3", "pred_d_largest_unit_supports": "sign of the total", "pred_e_trio_carries_030": ">= 0.30"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns) + 1     # the VERB position
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "source_layer": SRC, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top10_min": TOP10_MIN, "trio_min": TRIO_MIN}, "position": "verb"}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); blocks = model.transformer.h; mlp6, mlp5 = blocks[DST].mlp, blocks[SRC].mlp   # names kept from v192: mlp6 = the read block (8), mlp5 = the source block (7)
    Lrow, Rrow = mlp6.Left.weight.detach().float()[UNIT], mlp6.Right.weight.detach().float()[UNIT]
    D5, b5 = mlp5.Down.weight.detach().float(), mlp5.Down_bias.detach().float(); scale = float(blocks[7].lambdas[0]) * float(blocks[8].lambdas[0])
    lD, rD = scale * (Lrow @ D5), scale * (Rrow @ D5); lb, rb = scale * float(Lrow @ b5), scale * float(Rrow @ b5)
    per_row, forwards = [], 0
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); pos = [noun_of(r) for r in chunk]; idx = torch.arange(len(chunk))
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_, h5 = x, None, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l == SRC: h5 = dod_units.hidden(model, mlp5, xin)[idx, pos].float()
                if l == DST: x6 = x[idx, pos].float(); break
                x = x + block.mlp(xin)
            forwards += 1
            for i in range(len(chunk)):
                xi, hi = x6[i].to(Lrow.device), h5[i].to(Lrow.device); rms = float(xi.pow(2).mean().sqrt())
                per_row.append({"A": float(Lrow @ xi) / rms, "B": float(Rrow @ xi) / rms, "a": (lD * hi / rms).cpu(), "b": (rD * hi / rms).cpu(), "a_bias": lb / rms, "b_bias": rb / rms})
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    carrier = torch.zeros(len(lD)); total, closure, n = 0.0, 0.0, 0
    for i, row in enumerate(rows):
        if not row.present: continue
        P, S = per_row[i], per_row[partner[(row.construction, row.group, False)]]
        mA, mB = (P["A"] + S["A"]) / 2, (P["B"] + S["B"]) / 2
        terms = (P["a"] - S["a"]) * mB + mA * (P["b"] - S["b"])
        aP, aS = float(P["a"].sum()) + P["a_bias"], float(S["a"].sum()) + S["a_bias"]; bP, bS = float(P["b"].sum()) + P["b_bias"], float(S["b"].sum()) + S["b_bias"]
        ref = (aP - aS) * mB + mA * (bP - bS)                      # MLP 5's carrier term for this pair (v191's quantity)
        closure = max(closure, abs(float(terms.sum()) - ref) / max(abs(ref), 1e-6)); carrier += terms; total += ref; n += 1
    order = torch.argsort(carrier.abs(), descending=True); share = lambda k: float(carrier[order[:k]].sum()) / total
    top = [(int(j), float(carrier[j])) for j in order[:40]]
    print("MLP-5 carrier total", round(total, 3), "pairs", n, "top-10", round(share(10), 3), "top-50", round(share(50), 3), "top", [(j, round(v, 2)) for j, v in top[:10]])
    trio = sum(float(carrier[j]) for j in TRIO); trio_share = trio / total; top5 = [j for j, _ in top[:5]]
    print("trio carrier at the verb", round(trio, 3), "share", round(trio_share, 4), "in top-5:", [j for j in TRIO if j in top5])
    predictions = {"pred_a_carrier_closure": closure <= CLOSURE_TOL, "pred_b_top_10_units_carry_half": share(10) >= TOP10_MIN, "pred_c_trio_in_top5": sum(1 for j in TRIO if j in top5) >= 2, "pred_d_largest_unit_supports": (top[0][1] > 0) == (total > 0), "pred_e_trio_carries_030": trio_share >= TRIO_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp5_unit_carriers_into_829_verb_result_v241", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "carrier_total": total, "shares": {str(k): share(k) for k in (5, 10, 20, 50, 100, 500)},
                               "top_units": top, "trio_carrier": trio, "trio_share": trio_share, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
