#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_split_closure pred_c_table_part_dominates_for_493 pred_d_table_part_dominates_for_1036_and_829 pred_e_signs_agree_everywhere
"""MLP 1's part of the other agreement units' factors: table vs remainder for 493, 1036, 829 (v343). v341: for 3465, 81-85% of MLP 1's contribution to each
factor is its context-free entry. Completeness over the chain's other lexical units (v332-v337): 493 (MLP 3), 1036 (MLP 5), 829 (MLP 8), same exact split
at the noun of the v76 rows, with the lambda chain from block 1 to each unit's block. Registered reading: the lookup entry dominates for all of them.
PREDICTIONS (scored as written; failures preserved; priors from v341)
    pred_a_closure                              the manual forward reproduces the model's they - he margin (2.048) within 1e-3 (instrument)
    pred_b_split_closure                        table part + remainder part = MLP 1's term within relative 1e-3 on every pair, unit and factor
    pred_c_table_part_dominates_for_493         for 493 the table share is >= 0.60 on both factors
    pred_d_table_part_dominates_for_1036_and_829  for 1036 and 829 the table share is >= 0.60 on both factors. Prior: unsure -- these are 2 / 5 blocks further from MLP 1.
    pred_e_signs_agree_everywhere               the table part's pooled contribution has the sign of MLP 1's total for every unit and factor
PRICE (registered maximum): 3 row batches x 1 forward + 1 table batch = 4 forwards; 0 backwards; 0 fits. Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_table_vs_remainder_other_units_v343_result.json"
CANDIDATE_ID = "pronoun_number.mlp1_table_vs_remainder_other_units_v343"
UNITS = {3: (493,), 5: (1036,), 8: (829,)}; BATCH = 32
CLOSURE_TOL, NATIVE_M, SPLIT_TOL, TABLE_MIN = 1e-3, 2.0481, 1e-3, 0.60
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_closure": "margin 2.048 +- 1e-3", "pred_b_split_closure": "<= 1e-3", "pred_c_table_part_dominates_for_493": ">= 0.60 x 2", "pred_d_table_part_dominates_for_1036_and_829": ">= 0.60 x 4", "pred_e_signs_agree_everywhere": "all"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": {str(k): list(v) for k, v in UNITS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "native_m": NATIVE_M, "split_tol": SPLIT_TOL, "table_min": TABLE_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    lam = [(float(b.lambdas[0]), float(b.lambdas[1])) for b in blocks]
    W1 = {l: [] for l in UNITS}; XIN = {l: [] for l in UNITS}; margins = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pn = torch.tensor([noun_of(r) for r in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None; m1 = None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                if l in UNITS: XIN[l].append(x[idx, pn].float().cpu()); W1[l].append(m1[idx, pn].float().cpu())
                m = block.mlp(F.rms_norm(x, (model.config.n_embd,))); x = x + m
                if l == 1: m1 = m
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
            margins += [float(logits[i, r.final, L._single(" they")] - logits[i, r.final, L._single(" he")]) for i, r in enumerate(chunk)]
            forwards += 1
    base = sum((m if r.present else -m) for m, r in zip(margins, rows)) / len(rows); closure = abs(base - NATIVE_M)
    noun_ids = sorted({row.ids[noun_of(row)] for row in rows}); tok = torch.tensor(noun_ids, device="cuda"); tab = v289.capture(backend, tok.unsqueeze(1), torch.zeros(len(noun_ids), dtype=torch.long, device="cuda")); forwards += 1
    tindex = {t: i for i, t in enumerate(noun_ids)}; T = torch.stack([tab["mlp1"][tindex[row.ids[noun_of(row)]]] for row in rows])
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r in enumerate(rows) if r.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    per, sclos = {}, 0.0
    for l in UNITS:
        scale = 1.0
        for j in range(2, l + 1): scale *= lam[j][0]
        Xl = torch.cat(XIN[l]); rms = Xl.pow(2).mean(1).sqrt(); Wm = torch.cat(W1[l]); alpha = (Wm * (scale * T)).sum(1) / ((scale * T) ** 2).sum(1); A = alpha[:, None] * scale * T; Rm = Wm - A
        for u in UNITS[l]:
            Lr, Rr = blocks[l].mlp.Left.weight.detach().float()[u].cpu(), blocks[l].mlp.Right.weight.detach().float()[u].cpu()
            for name, vec in (("L", Lr), ("R", Rr)):
                term = lambda M: float(((M @ vec)[plural] / rms[plural] - (M @ vec)[sing] / rms[sing]).sum())
                tot, ta, tr = term(Wm), term(A), term(Rm); sclos = max(sclos, abs(ta + tr - tot) / max(abs(tot), 1e-6))
                per[f"mlp{l}.{u}.{name}"] = {"mlp1_contrast_term": tot, "table_share": ta / tot, "remainder_share": tr / tot}
    report = {"closure_margin_gap": closure, "split_closure": sclos, "per_unit_factor": per, "alpha_median_at_block3": float(alpha.median())}
    print(json.dumps(report, indent=1))
    ts = lambda u, n: per[[k for k in per if k.split(".")[1] == str(u)][0].rsplit(".", 1)[0] + f".{n}"]["table_share"]
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_split_closure": sclos <= SPLIT_TOL, "pred_c_table_part_dominates_for_493": all(ts(493, n) >= TABLE_MIN for n in ("L", "R")),
                   "pred_d_table_part_dominates_for_1036_and_829": all(ts(u, n) >= TABLE_MIN for u in (1036, 829) for n in ("L", "R")), "pred_e_signs_agree_everywhere": all(v["table_share"] > 0 for v in per.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_table_vs_remainder_other_units_result_v343", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
