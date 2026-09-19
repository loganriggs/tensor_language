#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_writer_closure pred_c_attention_carries_at_next_token pred_d_attention_4_among_top3 pred_e_mlps_below_half_at_next_token
"""What feeds MLP 8's number unit at the token AFTER the noun (v384). v383: at the post-noun key the number that reader 9.6 takes is written mostly by MLP 8 at
that position. MLP 8 there has no noun to look up, so its factors must receive the noun's number by attention. Exact writer fold of unit 829's two factors at
the noun+1 position on the v76 rows (embedding, attention 0-8 totals, MLP 0-7 totals), plural - singular over pairs -- v367's instrument one token later.
PREDICTIONS (scored as written; failures preserved; priors from section 4.8 / v367 / v383)
    pred_a_closure                       the manual forward reproduces the they - he margin (2.048) within 1e-3 (instrument)
    pred_b_writer_closure                the writer terms reproduce each factor's pair contrast within relative 1e-3 on every pair
    pred_c_attention_carries_at_next_token  attention 0-8 together carry >= 0.40 of each factor's contrast at noun+1 (against 0.03-0.05 at the noun, v367)
    pred_d_attention_4_among_top3        attention 4 (head 4.5's block) is among the three largest writers of at least one factor
    pred_e_mlps_below_half_at_next_token MLPs 0-7 together carry <= 0.50 of each factor (they re-process what attention brought). Prior: unsure.
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
OUT = ROOT / "circuits/followups/factor_writers_829_next_token_v384_result.json"
CANDIDATE_ID = "pronoun_number.factor_writers_829_next_token_v384"
UNIT, DST, BATCH = 829, 8, 32
CLOSURE_TOL, NATIVE_M, WRITER_TOL, ATTN_MIN, MLP_MAX = 1e-3, 2.0481, 1e-3, 0.40, 0.50
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_closure": "margin 2.048 +- 1e-3", "pred_b_writer_closure": "<= 1e-3", "pred_c_attention_carries_at_next_token": ">= 0.40 x 2", "pred_d_attention_4_among_top3": "top 3 on one factor", "pred_e_mlps_below_half_at_next_token": "<= 0.50 x 2"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "dst": DST, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "native_m": NATIVE_M, "writer_tol": WRITER_TOL, "attn_min": ATTN_MIN, "mlp_max": MLP_MAX}, "position": "noun+1"}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    lam = [(float(b.lambdas[0]), float(b.lambdas[1])) for b in blocks]; mlpd = blocks[DST].mlp
    Lr, Rr = mlpd.Left.weight.detach().float()[UNIT].cpu(), mlpd.Right.weight.detach().float()[UNIT].cpu()
    W = {"embedding": []}; W.update({f"attn{l}": [] for l in range(DST + 1)}); W.update({f"mlp{l}": [] for l in range(DST)}); X3 = []; margins = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pn = torch.tensor([noun_of(r) + 1 for r in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None; parts = {k: torch.zeros_like(x) for k in W}
            parts["embedding"] = x.clone()                       # x0 seeds the stream; its lambda-chain re-injections are added below
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                for k in parts: parts[k] = block.lambdas[0] * parts[k]
                parts["embedding"] = parts["embedding"] + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                if l <= DST: parts[f"attn{l}"] = parts[f"attn{l}"] + attention
                if l == DST:
                    X3.append(x[idx, pn].float().cpu())
                    for k in W: W[k].append(parts[k][idx, pn].float().cpu())
                m = block.mlp(F.rms_norm(x, (model.config.n_embd,))); x = x + m
                if l < DST: parts[f"mlp{l}"] = parts[f"mlp{l}"] + m
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
            margins += [float(logits[i, r.final, L._single(" they")] - logits[i, r.final, L._single(" he")]) for i, r in enumerate(chunk)]
            forwards += 1
    base = sum((m if r.present else -m) for m, r in zip(margins, rows)) / len(rows); closure = abs(base - NATIVE_M)
    X3 = torch.cat(X3); W = {k: torch.cat(v) for k, v in W.items()}; recon = sum(W.values()); assert float(((recon - X3).norm(dim=1) / X3.norm(dim=1)).max()) < 1e-3, "writer sum != x3"
    rms = X3.pow(2).mean(1).sqrt(); partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r in enumerate(rows) if r.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    per, wclos = {}, 0.0
    for name, vec in (("L", Lr), ("R", Rr)):
        f = (X3 @ vec) / rms; df = f[plural] - f[sing]; tot = float(df.sum()); terms = {}
        for k, w in W.items():
            proj = (w @ vec); terms[k] = float((proj[plural] / rms[plural] - proj[sing] / rms[sing]).sum()) / tot           # each writer's term, with the pair's own rms on each side (exact split; no separate rms term)
        wclos = max(wclos, abs(sum(terms.values()) - 1.0))
        per[name] = {"pooled_contrast": tot, "writer_shares": terms, "mlps_total": sum(v for k, v in terms.items() if k.startswith("mlp")), "attn_total": sum(v for k, v in terms.items() if k.startswith("attn")), "attn4": terms["attn4"], "top3": sorted(terms, key=lambda k: -abs(terms[k]))[:3]}
    report = {"closure_margin_gap": closure, "writer_closure": wclos, "per_factor": per}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_writer_closure": wclos <= WRITER_TOL, "pred_c_attention_carries_at_next_token": all(per[n]["attn_total"] >= ATTN_MIN for n in ("L", "R")),
                   "pred_d_attention_4_among_top3": any("attn4" in per[n]["top3"] for n in ("L", "R")), "pred_e_mlps_below_half_at_next_token": all(per[n]["mlps_total"] <= MLP_MAX for n in ("L", "R"))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "factor_writers_829_next_token_result_v384", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
