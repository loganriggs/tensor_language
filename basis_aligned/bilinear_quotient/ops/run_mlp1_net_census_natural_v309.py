#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_net_closure_on_text pred_b_gain_head_leads_on_text pred_c_head_share_on_text pred_d_units_lose_in_proportion_on_text pred_e_top200_share_on_text
"""MLP 1: the net per-unit census on natural text (v309). v306-v308 (filler contexts): the net per-unit change delta_j = (D_j . T^)[h_j(context) - h_j(alone)]
sums exactly to (alpha - 1)||T||; units 3289 / 624 lead it (22%), the top 200 hold 48%, and restoring sets in census order repairs the gain and the
direction while random sets do nothing. OOD: the same census at the 2,944 positions of the 128 natural rows (single-token activations from the
unique-token table passes). Reported: closure, the pooled ordering, the rank and share of 3289 / 624, the top-200 share, and the per-unit
lookup-vs-loss correlation.
PREDICTIONS (scored as written; failures preserved; priors from v306)
    pred_a_net_closure_on_text            sum_j delta_j = (alpha - 1)||T|| within relative 1e-3 at every position
    pred_b_gain_head_leads_on_text        3289 and 624 are the two largest |pooled delta_j| on text
    pred_c_head_share_on_text             the pair carries >= 0.15 of the pooled net change on text
    pred_d_units_lose_in_proportion_on_text  Pearson r over units between pooled lookup_j (alone) and pooled delta_j <= -0.90
    pred_e_top200_share_on_text           the top 200 carry between 0.35 and 0.60 of the pooled net change (spread, as on filler)
PRICE (registered maximum): 2 natural batches + <= 12 unique-token table batches = 14 forwards; 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_net_census_natural_v309_result.json"
CANDIDATE_ID = "mlp1.token_table.net_census_natural_v309"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, HEAD_MIN, R_MAX, TOP_LO, TOP_HI, BATCH = 1e-3, 0.15, -0.90, 0.35, 0.60, 256; HEAD = (3289, 624)
FORWARDS_MAX = 16
PREDICTIONS = {"pred_a_net_closure_on_text": "<= 1e-3", "pred_b_gain_head_leads_on_text": "ranks 1 and 2", "pred_c_head_share_on_text": ">= 0.15", "pred_d_units_lose_in_proportion_on_text": "r <= -0.90", "pred_e_top200_share_on_text": "0.35-0.60"}


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "natural": [p.name for p in NATURAL], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "head_min": HEAD_MIN, "r_max": R_MAX, "top_lo": TOP_LO, "top_hi": TOP_HI}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    nat = torch.tensor([r["ids"] for r in recs], device="cuda"); Tlen = nat.shape[1]
    writes, x1s, shares, toks, closure = [], [], [], [], 0.0; store = {}
    def wrap(l):
        orig = model.transformer.h[l].attn.squared_attention
        def f(q, k, v, q2, k2):
            B, T, H, D = q.shape; pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / D) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / D)
            pat = pat.masked_fill(torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool)).logical_not(), 0.0)
            own = torch.diagonal(pat, dim1=-2, dim2=-1).abs(); store[l] = (own / pat.abs().sum(-1).clamp_min(1e-9)).mean(1).float().cpu(); return orig(q, k, v, q2, k2)   # [B, T]: own-key share per position
        return f
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]
            x = F.rms_norm(model.transformer.wte(chunk), (model.config.n_embd,)); x0, v1_ = x, None
            for l in (0, 1):
                block = model.transformer.h[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; orig = block.attn.squared_attention; block.attn.squared_attention = wrap(l)
                try: attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                finally: block.attn.squared_attention = orig
                x = live + attention; m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                if l == 1: x1 = x
                x = x + m
            forwards += 1
            writes.append(m[:, 1:].reshape(-1, m.shape[-1]).float().cpu()); x1s.append(x1[:, 1:].reshape(-1, m.shape[-1]).float().cpu()); toks.append(chunk[:, 1:].reshape(-1).cpu()); shares.append(((store[0] + store[1]) / 2)[:, 1:].reshape(-1))
    W, X1, tok, S = torch.cat(writes), torch.cat(x1s), torch.cat(toks), torch.cat(shares)
    uniq = sorted(set(tok.tolist())); tab = {}
    for s0 in range(0, len(uniq), BATCH):
        ids = torch.tensor(uniq[s0:s0 + BATCH], device="cuda").unsqueeze(1); c = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1
        for k, v in c.items(): tab.setdefault(k, []).append(v)
    tab = {k: torch.cat(v) for k, v in tab.items()}; tindex = {t: i for i, t in enumerate(uniq)}; ti = torch.tensor([tindex[t] for t in tok.tolist()])
    T = tab["mlp1"][ti]; alpha = (W * T).sum(1) / (T * T).sum(1)
    mlp = model.transformer.h[1].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    n = F.rms_norm(X1, (X1.shape[-1],)); n_tab = F.rms_norm(tab["x1"][ti], (X1.shape[-1],))
    h_ctx = (n @ Lw.T) * (n @ Rw.T); h_tab = (n_tab @ Lw.T) * (n_tab @ Rw.T); That = T / T.norm(dim=1, keepdim=True); DT = That @ Dw
    delta = DT * (h_ctx - h_tab); look = DT * h_tab
    closure = float(((delta.sum(1) - (alpha - 1) * T.norm(dim=1)).abs() / ((alpha - 1) * T.norm(dim=1)).abs().clamp_min(1e-6)).max())
    pd, pl = delta.sum(0), look.sum(0); order = torch.argsort(pd.abs(), descending=True)
    def pearson(a, b): a, b = a - a.mean(), b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm()))
    ranks = {str(u): int((pd.abs() > pd.abs()[u]).sum()) + 1 for u in HEAD}; head_share = float(sum(pd[u] for u in HEAD) / pd.sum())
    shares = {str(m): float(pd[order[:m]].sum() / pd.sum()) for m in (2, 10, 50, 200, 500, 1000, 2000)}
    report = {"positions": int(W.shape[0]), "closure_max": closure, "alpha_median": float(alpha.median()), "pooled_net_per_position": float(pd.sum() / W.shape[0]), "head_ranks": ranks, "head_share": head_share, "top_shares": shares,
              "pearson_lookup_vs_net": pearson(pl, pd), "units_negative_fraction": float((pd < 0).float().mean()), "top12_units": [(int(j), float(pd[j] / W.shape[0]), float(pl[j] / W.shape[0])) for j in order[:12]]}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_net_closure_on_text": closure <= CLOSURE_TOL, "pred_b_gain_head_leads_on_text": set(ranks.values()) == {1, 2}, "pred_c_head_share_on_text": head_share >= HEAD_MIN,
                   "pred_d_units_lose_in_proportion_on_text": report["pearson_lookup_vs_net"] <= R_MAX, "pred_e_top200_share_on_text": TOP_LO <= shares["200"] <= TOP_HI}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_net_census_natural_result_v309", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
