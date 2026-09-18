#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_one_head_per_block_leads pred_c_leading_heads_read_the_noun pred_d_leading_heads_are_contextual
"""Pronoun number they/he DoD (v243): WHICH block-5 (and block-4) head feeds MLP-6 unit 69 at the verb? v242: at the verb unit 69 (38% of MLP 6's part of the plural
detector there, v241) is carried by attn:05 21% and attn:04 18%. v205 did this fold for the detector's own gradient and found 4.5 (a copy from the noun) and 5.3 (a
re-read of the verb). Same fold with reader r_row = unit 69's product gradient at the block-6 input at the verb (weights + native trace), every head of blocks 4 and 5
at the verb query, split by source position (noun / earlier / the verb itself) and value branch.
PREDICTIONS (scored as written; failures preserved; priors from v205)
    pred_a_fold_closure                  per head, source terms sum to r_row . (head write) within relative 1e-3, every row
    pred_b_one_head_per_block_leads      in each of blocks 4 and 5 the largest head carries >= 0.50 of the block's pooled contrast (|share|)
    pred_c_leading_heads_read_the_noun   for both leading heads the noun position carries >= 0.60 (a copy, not a re-read). Prior: unsure (5.3 re-read the verb for 829).
    pred_d_leading_heads_are_contextual  for both leading heads the token-only branch carries <= 0.50
PRICE (registered maximum): 3 batches x (positional trace + 2 attention factors) = 9 forwards; 0 backwards; 0 fits. Bar <= 12.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_mlp6_unit_pair_fold_v183 as v183
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_unit69_verb_head_source_fold_v243_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_unit69_verb_head_source_fold_v243"
UNIT, BLOCKS, CLOSURE_TOL, LEAD_MIN, NOUN_MIN, INHERITED_MAX, BATCH = 69, (4, 5), 1e-3, 0.50, 0.60, 0.50, 32
FORWARDS_MAX = 12
PREDICTIONS = {"pred_a_fold_closure": "<= 1e-3", "pred_b_one_head_per_block_leads": ">= 0.50 x 2", "pred_c_leading_heads_read_the_noun": ">= 0.60 x 2", "pred_d_leading_heads_are_contextual": "<= 0.50 x 2"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns); verb_of = lambda row: noun_of(row) + 1
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "blocks": BLOCKS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "lead_min": LEAD_MIN, "noun_min": NOUN_MIN, "inherited_max": INHERITED_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); fw.backend = backend
    mlp = model.transformer.h[6].mlp; Lrow, Rrow = mlp.Left.weight.detach().float()[UNIT], mlp.Right.weight.detach().float()[UNIT]
    Wo = {b: model.transformer.h[b].attn.c_proj.weight.detach().float() for b in BLOCKS}
    forwards, closure, per_row = 0, 0.0, []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        traces = L.forward_trace_positions(fw, chunk, lambda rw: [verb_of(rw)], upto_layer=7, head_write_layers=(6,)); forwards += 1
        facts = {}
        for b in BLOCKS:
            facts[b] = L.attention_factors(fw, chunk, b); forwards += 1
        for i, (row, tr) in enumerate(zip(chunk, traces)):
            pos, npos = verb_of(row), noun_of(row); C = v183.writers_at_block_input(tr, pos, 6); x = sum(C.values()).to(Lrow.device); rms2 = float(x.pow(2).mean())
            r = (float(Rrow @ x) * Lrow + float(Lrow @ x) * Rrow) / rms2
            entry = {}
            for b in BLOCKS:
                f = facts[b]; lam = f["lamb"]
                # the block's attention write at the verb, per head, as it reaches block 8: scale by lambda0 of blocks b+1..8
                scale = 1.0
                for l in range(b + 1, 7): scale *= float(model.transformer.h[l].lambdas[0])
                for h in range(9):
                    d = scale * (Wo[b][:, h * L.HEAD_DIM:(h + 1) * L.HEAD_DIM].T @ r)
                    p, tc, ti = L._source_terms(f, i, pos, h, d, torch)
                    z = ((p[:, None] * ((1 - lam) * f["v_cur"][i, :pos + 1, h].float() + lam * f["v1"][i, :pos + 1, h].float())).sum(0))
                    direct = float(d.to(z.device) @ z); total = float(tc.sum() + ti.sum())
                    closure = max(closure, abs(total - direct) / max(abs(direct), 1e-6))
                    entry[f"{b}.{h}"] = {"total": total, "noun": float(tc[npos] + ti[npos]), "self": float(tc[pos] + ti[pos]), "earlier": float(tc[:npos].sum() + ti[:npos].sum()), "inherited": float(ti.sum())}
            per_row.append(entry)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    keys = list(per_row[0].keys()); pooled = {k: {q: 0.0 for q in ("total", "noun", "self", "earlier", "inherited")} for k in keys}
    for i, row in enumerate(rows):
        if not row.present: continue
        j = partner[(row.construction, row.group, False)]
        for k in keys:
            for q in pooled[k]: pooled[k][q] += per_row[i][k][q] - per_row[j][k][q]
    report, predictions = {}, {"pred_a_fold_closure": closure <= CLOSURE_TOL, "pred_b_one_head_per_block_leads": True, "pred_c_leading_heads_read_the_noun": True, "pred_d_leading_heads_are_contextual": True}
    for b in BLOCKS:
        heads = [k for k in keys if k.startswith(f"{b}.")]; block_total = sum(pooled[k]["total"] for k in heads)
        shares = {k: pooled[k]["total"] / block_total for k in heads}; lead = max(heads, key=lambda k: abs(shares[k]))
        t = pooled[lead]["total"]; noun_frac = pooled[lead]["noun"] / t; inh_frac = pooled[lead]["inherited"] / t
        report[str(b)] = {"block_total": block_total, "head_shares": shares, "lead": lead, "lead_noun_fraction": noun_frac, "lead_self_fraction": pooled[lead]["self"] / t, "lead_earlier_fraction": pooled[lead]["earlier"] / t, "lead_inherited_fraction": inh_frac,
                          "per_head": {k: {q: v / block_total for q, v in pooled[k].items()} for k in heads}}
        print("block", b, "total", round(block_total, 2), "shares", {k: round(v, 3) for k, v in sorted(shares.items(), key=lambda kv: -abs(kv[1]))}, "lead", lead, "noun", round(noun_frac, 3), "self", round(pooled[lead]["self"] / t, 3), "token-only", round(inh_frac, 3))
        predictions["pred_b_one_head_per_block_leads"] &= abs(shares[lead]) >= LEAD_MIN; predictions["pred_c_leading_heads_read_the_noun"] &= noun_frac >= NOUN_MIN; predictions["pred_d_leading_heads_are_contextual"] &= inh_frac <= INHERITED_MAX
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_unit69_verb_head_source_fold_result_v243", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "blocks": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
