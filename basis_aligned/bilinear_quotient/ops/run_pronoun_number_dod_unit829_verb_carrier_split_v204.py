#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_carrier_closure pred_b_attention_carries_040_at_verb pred_c_mlp_stack_carries_at_most_050 pred_d_one_attention_block_leads
"""Pronoun number they/he DoD (v204): what carries the plural feature from the noun to the VERB? v203: the same MLP-8 detectors re-fire at the verb
(829 33%, 953 25% of MLP 8's write on 9.6's direction there). At the verb the noun is one position back, so the feature can only arrive through
attention. v188's carrier split (exact; `dod_units.carrier_split`) of u_829 at the verb position (noun + 1) over the 48 aligned pairs: which writers'
own change carries the plural - singular contrast there -- the attention totals of blocks 0-7, the block-8 heads, or (as at the noun, v188) the MLPs?
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_carrier_closure                carrier shares sum to 1 within 1e-3; per-pair identity within relative 1e-3
    pred_b_attention_carries_040_at_verb  attention writers (attn:00..07 + the nine block-8 heads) together carry >= 0.40 (at the noun: attn:06 + 8.1 <= 0.05)
    pred_c_mlp_stack_carries_at_most_050  mlp:00..07 together carry <= 0.50
    pred_d_one_attention_block_leads      the largest single attention writer carries >= 0.15
PRICE (registered maximum): 3 batches x 1 positional trace = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_aspectual_dod_mlp8_pair_fold_v16 as v16
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_unit829_verb_carrier_split_v204_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_unit829_verb_carrier_split_v204"
UNIT, CLOSURE_TOL, ATTN_MIN, MLP_MAX, LEAD_MIN = 829, 1e-3, 0.40, 0.50, 0.15
FORWARDS_MAX = 4
WRITERS = v16.WRITERS
PREDICTIONS = {"pred_a_carrier_closure": "<= 1e-3", "pred_b_attention_carries_040_at_verb": ">= 0.40", "pred_c_mlp_stack_carries_at_most_050": "<= 0.50", "pred_d_one_attention_block_leads": ">= 0.15"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns) + 1     # the VERB position
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "writers": WRITERS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "attn_min": ATTN_MIN, "mlp_max": MLP_MAX, "lead_min": LEAD_MIN}}
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
    n = len(WRITERS); closure, factors = 0.0, []
    for row, tr in zip(rows, traces):
        pos = noun_of(row); C = v16.writers_at_block8_input(tr, pos)
        x = sum(C.values()); rms = float(x.pow(2).mean().sqrt())
        M = torch.stack([C[w] for w in WRITERS]).to(Lrow.device)
        factors.append(((M @ Lrow) / rms, (M @ Rrow) / rms))
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    carrier, mass = torch.zeros(n), torch.zeros(n); contrast = 0.0
    for i, row in enumerate(rows):
        if not row.present: continue
        j = partner[(row.construction, row.group, False)]
        aP, bP = factors[i]; aS, bS = factors[j]
        ref = float(aP.sum() * bP.sum() - aS.sum() * bS.sum())
        da, db, ma, mb = aP - aS, bP - bS, (aP + aS) / 2, (bP + bS) / 2
        left, right = da * float(mb.sum()), db * float(ma.sum())          # per-writer: own-change x mean partner total, for each factor
        closure = max(closure, abs(float(left.sum() + right.sum()) - ref) / max(abs(ref), 1e-6))
        carrier += (left + right).cpu(); contrast += ref
        Tm = torch.outer(aP, bP) - torch.outer(aS, bS); mass += ((Tm + Tm.T) / 2).sum(1).cpu()
    carrier_share = {w: float(carrier[k]) / contrast for k, w in enumerate(WRITERS)}
    mass_share = {w: float(mass[k]) / contrast for k, w in enumerate(WRITERS)}            # = v181's factor_share (pairs containing w, diagonal once)
    ranked = sorted(carrier_share, key=lambda w: -abs(carrier_share[w]))
    report = {"contrast": contrast, "carrier_share": carrier_share, "mass_share": mass_share, "carrier_sum": sum(carrier_share.values()), "ranked": ranked}
    print("contrast", round(contrast, 2), "carrier sum", round(report["carrier_sum"], 4)); print("carrier", [(w, round(carrier_share[w], 3), "mass", round(mass_share[w], 3)) for w in ranked[:10]])
    attn = {w: v for w, v in carrier_share.items() if w.startswith("attn")}; mlps = sum(v for w, v in carrier_share.items() if w.startswith("mlp")); lead = max(attn, key=lambda w: abs(attn[w]))
    report.update({"attention_total": sum(attn.values()), "mlp_total": mlps, "embed": carrier_share["embed"], "lead_attention": (lead, attn[lead])}); print("attention", round(sum(attn.values()), 3), "mlps", round(mlps, 3), "embed", round(carrier_share["embed"], 3), "lead", lead, round(attn[lead], 3))
    predictions = {"pred_a_carrier_closure": closure <= CLOSURE_TOL and abs(report["carrier_sum"] - 1) <= CLOSURE_TOL, "pred_b_attention_carries_040_at_verb": sum(attn.values()) >= ATTN_MIN, "pred_c_mlp_stack_carries_at_most_050": mlps <= MLP_MAX, "pred_d_one_attention_block_leads": abs(attn[lead]) >= LEAD_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_unit829_carrier_split_result_v188", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
