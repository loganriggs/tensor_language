#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_unit_closure pred_c_same_leaders_as_at_the_noun pred_d_mlp4_has_a_named_leader pred_e_each_layer_spread
"""Which units rebuild the number at the token after the noun? (v385). v384: at the post-noun position MLP-8 unit 829 is fed mostly by MLPs 4-7. Per-unit
exact fold of MLPs 4 / 5 / 6 / 7 into 829's two factors at noun+1 on the v76 rows: top units per layer, their ranks against the noun-position leaders
(1036, 2483, 1779; v368), and MLP 4's leaders (new at this site).
PREDICTIONS (scored as written; failures preserved; priors from v368)
    pred_a_closure                     the manual forward reproduces the they - he margin (2.048) within 1e-3 (instrument)
    pred_b_unit_closure                per-unit terms sum to each layer's term within relative 1e-3 on every pair and factor
    pred_c_same_leaders_as_at_the_noun 1036 (MLP 5), 2483 (MLP 6) and 1779 (MLP 7) each rank in the top 3 of their layer on at least one factor at noun+1
    pred_d_mlp4_has_a_named_leader     MLP 4's top unit carries >= 0.10 of MLP 4's summed |contrast| on the L factor
    pred_e_each_layer_spread           for each layer the top-10 units carry <= 0.60 of the layer's summed |contrast| on L
PRICE (registered maximum): 3 row batches x 1 forward = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp4567_units_into_829_next_token_v385_result.json"
CANDIDATE_ID = "pronoun_number.mlp4567_units_into_829_next_token_v385"
UNIT, DST, BATCH = 829, 8, 32
CLOSURE_TOL, NATIVE_M, UNIT_TOL, TOP10_MAX, LEAD_MIN, SRCS = 1e-3, 2.0481, 1e-3, 0.60, 0.10, (4, 5, 6, 7)
NOUN_LEADERS = {5: 1036, 6: 2483, 7: 1779}
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_closure": "margin 2.048 +- 1e-3", "pred_b_unit_closure": "<= 1e-3", "pred_c_same_leaders_as_at_the_noun": "top 3 x 3", "pred_d_mlp4_has_a_named_leader": ">= 0.10", "pred_e_each_layer_spread": "top-10 <= 0.60 x 4"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "dst": DST, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "native_m": NATIVE_M, "unit_tol": UNIT_TOL, "top10_max": TOP10_MAX, "lead_min": LEAD_MIN}, "srcs": list(SRCS), "position": "noun+1"}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    lam = [(float(b.lambdas[0]), float(b.lambdas[1])) for b in blocks]; mlpd = blocks[DST].mlp
    Lr, Rr = mlpd.Left.weight.detach().float()[UNIT].cpu(), mlpd.Right.weight.detach().float()[UNIT].cpu()
    H = {l: [] for l in SRCS}; X8 = []; margins = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pn = torch.tensor([noun_of(r) + 1 for r in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l == DST: X8.append(x[idx, pn].float().cpu())
                if l in SRCS: H[l].append(dod_units.hidden(model, block.mlp, xin)[idx, pn].float().cpu())
                x = x + block.mlp(xin)
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
            margins += [float(logits[i, r.final, L._single(" they")] - logits[i, r.final, L._single(" he")]) for i, r in enumerate(chunk)]
            forwards += 1
    base = sum((m if r.present else -m) for m, r in zip(margins, rows)) / len(rows); closure = abs(base - NATIVE_M)
    X8 = torch.cat(X8); rms = X8.pow(2).mean(1).sqrt(); partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r in enumerate(rows) if r.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    per, uclos = {}, 0.0
    for l in SRCS:
        scale = 1.0
        for j in range(l + 1, DST + 1): scale *= lam[j][0]
        Dw, b = blocks[l].mlp.Down.weight.detach().float().cpu(), blocks[l].mlp.Down_bias.detach().float().cpu(); h = torch.cat(H[l]); per[str(l)] = {}
        for name, vec in (("L", Lr), ("R", Rr)):
            coef = scale * (Dw.T @ vec)                                    # per-unit projection of its write on the factor row
            unit_term = h * coef.unsqueeze(0) / rms.unsqueeze(1)          # [rows, 4608]
            full = scale * ((h @ Dw.T + b) @ vec) / rms
            uclos = max(uclos, float(((unit_term.sum(1) + scale * float(b @ vec) / rms - full).abs() / full.abs().clamp_min(1e-6)).max()))
            pooled = (unit_term[plural] - unit_term[sing]).sum(0); order = torch.argsort(pooled.abs(), descending=True); total = float(pooled.abs().sum())
            per[str(l)][name] = {"top10": [(int(j), float(pooled[j])) for j in order[:10]], "top10_share": float(pooled.abs()[order[:10]].sum() / total), "top1_share": float(pooled.abs()[order[0]] / total),
                                 "rank_noun_leader": int((pooled.abs() > pooled.abs()[NOUN_LEADERS[l]]).sum()) + 1 if l in NOUN_LEADERS else None}
    report = {"closure_margin_gap": closure, "unit_closure": uclos, "per_layer": per}
    print(json.dumps({k: v for k, v in report.items() if k != "per_layer"}, indent=1)); print(json.dumps({l: {f: (v["top10"][:4], round(v["top10_share"], 3), v["rank_noun_leader"]) for f, v in d.items()} for l, d in per.items()}, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_unit_closure": uclos <= UNIT_TOL, "pred_c_same_leaders_as_at_the_noun": all(min(per[str(l)][f]["rank_noun_leader"] for f in ("L", "R")) <= 3 for l in NOUN_LEADERS),
                   "pred_d_mlp4_has_a_named_leader": per["4"]["L"]["top1_share"] >= LEAD_MIN, "pred_e_each_layer_spread": all(per[str(l)]["L"]["top10_share"] <= TOP10_MAX for l in SRCS)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp4567_units_into_829_next_token_result_v385", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
