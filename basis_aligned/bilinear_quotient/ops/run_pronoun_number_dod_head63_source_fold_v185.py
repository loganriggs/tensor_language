#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_self_position_carries_080 pred_c_token_only_branch_carries_050 pred_d_head_supports_unit
"""Pronoun number they/he DoD (v185): SOURCE FOLD of head 6.3 at the noun, into MLP-6 unit 2483 (the plural detector's largest MLP-6 input; v183:
pairs with head 6.3 carry 32% of 2483's product contrast at the noun, the one named non-MLP input of the number chain).
Reader: the exact linear part of 2483's product in head 6.3's write c at the noun, r_row = ((R . x)L + (L . x)R) / rms^2 with x = x_6[noun] (pre-norm;
writers from the block-6 input trace, v183) -- r_row . c = (L . c)(R . x) + (L . x)(R . c) / rms^2, the two cross terms of the pair fold. Pulled back
through O_h to the 128-d head slice and split by source position (the noun itself vs the earlier tokens) and value branch (current block state vs
token-only block-0 value), `_source_terms` (v80 / v112 fold). Registered reading: 6.3 is a self-position token copier here, as 6.1 is on the gender
line (v174: 90% token-only).
PREDICTIONS (scored as written; failures preserved)
    pred_a_fold_closure                 sum of source terms = r_row . c within relative 1e-3, every row
    pred_b_self_position_carries_080    the noun position carries >= 0.80 of the pooled plural - singular contrast
    pred_c_token_only_branch_carries_050 the token-only (block-0 value) branch carries >= 0.50 of it
    pred_d_head_supports_unit           the pooled contrast of r . c has the sign of 2483's own pooled product contrast (negative on plural - singular, v183)
PRICE (registered maximum): 3 batches x (positional trace + attention factors) = 6 forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_pronoun_number_dod_mlp6_unit_pair_fold_v183 as v183
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_head63_source_fold_v185_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_head63_source_fold_v185"
UNIT, LAYER, HEAD, CLOSURE_TOL, SELF_MIN, INHERITED_MIN, BATCH = 2483, 6, 3, 1e-3, 0.80, 0.50, 32
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_fold_closure": "<= 1e-3", "pred_b_self_position_carries_080": ">= 0.80", "pred_c_token_only_branch_carries_050": ">= 0.50", "pred_d_head_supports_unit": "sign of 2483's contrast"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "head": f"{LAYER}.{HEAD}", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "self_min": SELF_MIN, "inherited_min": INHERITED_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); fw.backend = backend
    mlp = model.transformer.h[LAYER].mlp; Lrow, Rrow = mlp.Left.weight.detach().float()[UNIT], mlp.Right.weight.detach().float()[UNIT]
    Wo = model.transformer.h[LAYER].attn.c_proj.weight.detach().float()[:, HEAD * L.HEAD_DIM:(HEAD + 1) * L.HEAD_DIM]      # (D, 128): write = Wo @ z
    forwards, closure, per_row, unit_contrast_terms = 0, 0.0, [], []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        traces = L.forward_trace_positions(fw, chunk, lambda rw: [noun_of(rw)], upto_layer=LAYER + 1, head_write_layers=(LAYER,)); forwards += 1
        f = L.attention_factors(fw, chunk, LAYER); forwards += 1
        for i, (row, tr) in enumerate(zip(chunk, traces)):
            pos = noun_of(row); C = v183.writers_at_block_input(tr, pos, LAYER); x = sum(C.values()).to(Lrow.device); rms2 = float(x.pow(2).mean())
            Lx, Rx = float(Lrow @ x), float(Rrow @ x); r = (Rx * Lrow + Lx * Rrow) / rms2
            c = C[f"attnhead:{LAYER:02d}:{HEAD}"].to(Lrow.device); direct = float(r @ c)
            d = Wo.T @ r
            p, tc, ti = L._source_terms(f, i, pos, HEAD, d, torch)
            total = float(tc.sum() + ti.sum()); closure = max(closure, abs(total - direct) / max(abs(direct), 1e-6))
            xin = F.rms_norm(x, (x.shape[-1],)); u = float((Lrow @ xin) * (Rrow @ xin))
            per_row.append({"self_cur": float(tc[pos]), "self_inh": float(ti[pos]), "other_cur": float(tc[:pos].sum()), "other_inh": float(ti[:pos].sum()), "total": total, "pattern_self": float(p[pos]), "unit": u})
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    pooled = {k: 0.0 for k in ("self_cur", "self_inh", "other_cur", "other_inh", "total", "unit")}; n = 0
    for i, row in enumerate(rows):
        if not row.present: continue
        j = partner[(row.construction, row.group, False)]
        for k in pooled: pooled[k] += per_row[i][k] - per_row[j][k]
        n += 1
    tot = pooled["total"]; shares = {k: pooled[k] / tot for k in ("self_cur", "self_inh", "other_cur", "other_inh")}
    self_share, inh_share = shares["self_cur"] + shares["self_inh"], shares["self_inh"] + shares["other_inh"]
    report = {"pairs": n, "pooled": pooled, "shares": shares, "self_share": self_share, "inherited_share": inh_share, "head_share_of_unit_contrast": tot / pooled["unit"], "mean_pattern_self": sum(r["pattern_self"] for r in per_row) / len(per_row)}
    print("pooled r.c", round(tot, 2), "unit contrast", round(pooled["unit"], 2), "share", round(tot / pooled["unit"], 3), "self", round(self_share, 3), "inherited", round(inh_share, 3), {k: round(v, 3) for k, v in shares.items()}, "mean self pattern", round(report["mean_pattern_self"], 3))
    predictions = {"pred_a_fold_closure": closure <= CLOSURE_TOL, "pred_b_self_position_carries_080": self_share >= SELF_MIN, "pred_c_token_only_branch_carries_050": inh_share >= INHERITED_MIN, "pred_d_head_supports_unit": (tot > 0) == (pooled["unit"] > 0)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_head63_source_fold_result_v185", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "per_row": per_row, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
