#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_unit_closure pred_c_1036_leads_mlp5 pred_d_each_layer_spread pred_e_top_units_consistent_across_factors
"""Which units of MLPs 5, 6 and 7 feed MLP-8 unit 829's factors? (v368). v367: MLPs 5-7 carry half of each of 829's factor contrasts at the noun. Each MLP's
write is Down h + bias, so its term in each factor splits exactly by unit (lambda-chain to block 8, over the pair's rms). Per layer: the plural - singular
contrast per unit on each factor, top-k shares, and the rank of the named MLP-5 unit 1036 and the MLP-7 units named in the temporal work (1250 / 3364 / 1884).
PREDICTIONS (scored as written; failures preserved; priors from v194 / v278)
    pred_a_closure               the manual forward reproduces the model's they - he margin (2.048) within 1e-3 (instrument)
    pred_b_unit_closure          per-unit terms sum to each layer's term within relative 1e-3 on every pair and factor
    pred_c_1036_leads_mlp5       unit 1036 ranks first among MLP-5 units on at least one factor
    pred_d_each_layer_spread     for each of MLPs 5 / 6 / 7 the top-10 units carry <= 0.60 of that layer's summed |contrast| on the L factor
    pred_e_top_units_consistent_across_factors  for each layer the top-5 sets on L and R share >= 3 units. Prior: unsure.
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
OUT = ROOT / "circuits/followups/mlp567_units_into_829_v368_result.json"
CANDIDATE_ID = "pronoun_number.mlp567_units_into_829_v368"
UNIT, DST, BATCH = 829, 8, 32
CLOSURE_TOL, NATIVE_M, UNIT_TOL, TOP10_MAX, SHARE_MIN, SRCS = 1e-3, 2.0481, 1e-3, 0.60, 3, (5, 6, 7)
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_closure": "margin 2.048 +- 1e-3", "pred_b_unit_closure": "<= 1e-3", "pred_c_1036_leads_mlp5": "rank 1 on L or R", "pred_d_each_layer_spread": "top-10 <= 0.60 x 3", "pred_e_top_units_consistent_across_factors": ">= 3 of 5 shared x 3"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "dst": DST, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "native_m": NATIVE_M, "unit_tol": UNIT_TOL, "top10_max": TOP10_MAX, "share_min": SHARE_MIN}, "srcs": list(SRCS)}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    lam = [(float(b.lambdas[0]), float(b.lambdas[1])) for b in blocks]; mlpd = blocks[DST].mlp
    Lr, Rr = mlpd.Left.weight.detach().float()[UNIT].cpu(), mlpd.Right.weight.detach().float()[UNIT].cpu()
    H = {l: [] for l in SRCS}; X8 = []; margins = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pn = torch.tensor([noun_of(r) for r in chunk])
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
            per[str(l)][name] = {"top10": [(int(j), float(pooled[j])) for j in order[:10]], "top10_share": float(pooled.abs()[order[:10]].sum() / total), "rank_1036": int((pooled.abs() > pooled.abs()[1036]).sum()) + 1 if l == 5 else None,
                                 "rank_temporal": {str(u): int((pooled.abs() > pooled.abs()[u]).sum()) + 1 for u in (1250, 3364, 1884)} if l == 7 else None}
    shared = {str(l): len({j for j, _ in per[str(l)]["L"]["top10"][:5]} & {j for j, _ in per[str(l)]["R"]["top10"][:5]}) for l in SRCS}
    report = {"closure_margin_gap": closure, "unit_closure": uclos, "per_layer": per, "top5_shared_L_R": shared}
    print(json.dumps({k: v for k, v in report.items() if k != "per_layer"}, indent=1)); print(json.dumps({l: {f: (v["top10"][:5], round(v["top10_share"], 3), v["rank_1036"], v["rank_temporal"]) for f, v in d.items()} for l, d in per.items()}, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_unit_closure": uclos <= UNIT_TOL, "pred_c_1036_leads_mlp5": any(per["5"][f]["rank_1036"] == 1 for f in ("L", "R")),
                   "pred_d_each_layer_spread": all(per[str(l)]["L"]["top10_share"] <= TOP10_MAX for l in SRCS), "pred_e_top_units_consistent_across_factors": all(v >= SHARE_MIN for v in shared.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp567_units_into_829_result_v368", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
