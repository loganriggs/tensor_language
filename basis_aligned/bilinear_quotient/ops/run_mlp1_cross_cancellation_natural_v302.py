#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_table_term_near_full_on_text pred_c_cross_negative_on_text pred_d_cross_tracks_context_size_on_text pred_e_decomposition_closes_on_text
"""MLP 1: the cross-term cancellation on natural text (v302). v301 (filler contexts): MLP 1's write = gamma^2 T + cross + only with gamma^2 (the token's own
quadratic lookup term after normalisation) 0.86-0.93, the token x context cross term projecting -0.44 to -0.75 on the lookup T (negative in 100% of
rows, r 0.68 with the context input's size), and the context^2 term +0.07 to +0.20. OOD test at the 2,944 positions of the 128 natural rows: the same
exact three-term expansion per position (table entries from single-token passes over the unique tokens), alpha = gamma^2 + proj_T(cross) + proj_T(only).
PREDICTIONS (scored as written; failures preserved; priors from v301; the closure bar is set at the float32 level v301 reached)
    pred_a_closure                          the block-1 recurrence reproduces the captured write within relative 1e-4 (instrument)
    pred_b_table_term_near_full_on_text     gamma^2 median >= 0.75
    pred_c_cross_negative_on_text           proj_T(cross) < 0 for >= 0.90 of positions
    pred_d_cross_tracks_context_size_on_text  Pearson r(proj_T(cross), -||c||) >= 0.40 over positions
    pred_e_decomposition_closes_on_text     gamma^2 T + cross + only = W within relative 0.02 at every position
PRICE (registered maximum): 2 natural batches + unique tokens (<= 2,944) / 256 <= 12 table batches = 14 forwards; 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_cross_cancellation_natural_v302_result.json"
CANDIDATE_ID = "mlp1.token_table.cross_cancellation_natural_v302"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, GAMMA_MIN, NEG_MIN, R_MIN, DECOMP_TOL, BATCH = 1e-4, 0.75, 0.90, 0.40, 0.02, 256
FORWARDS_MAX = 16
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_table_term_near_full_on_text": ">= 0.75", "pred_c_cross_negative_on_text": ">= 0.90", "pred_d_cross_tracks_context_size_on_text": "r >= 0.40", "pred_e_decomposition_closes_on_text": "<= 0.02"}


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "natural": [p.name for p in NATURAL], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "gamma_min": GAMMA_MIN, "neg_min": NEG_MIN, "r_min": R_MIN, "decomp_tol": DECOMP_TOL}}
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
    rec = mlp(F.rms_norm(X1.to("cuda"), (X1.shape[-1],))).float().cpu(); closure = float(((rec - W).norm(dim=1) / W.norm(dim=1)).max())
    n = F.rms_norm(X1, (X1.shape[-1],)); n_tab = F.rms_norm(tab["x1"][ti], (X1.shape[-1],)); gamma = (n * n_tab).sum(1) / (n_tab * n_tab).sum(1); tp = gamma[:, None] * n_tab; cc = n - tp
    Lt2, Rt2, Lc, Rc = tp @ Lw.T, tp @ Rw.T, cc @ Lw.T, cc @ Rw.T
    quad = (Lt2 * Rt2) @ Dw.T; cross = (Lt2 * Rc + Lc * Rt2) @ Dw.T; only = (Lc * Rc) @ Dw.T
    decomp = float((((quad + cross + only) - W).norm(dim=1) / W.norm(dim=1)).max())
    pT = lambda A: (A * T).sum(1) / (T * T).sum(1); g2, pc, po = pT(quad), pT(cross), pT(only)
    def pearson(a, b): a, b = a - a.mean(), b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm()))
    r = pearson(pc, -cc.norm(dim=1))
    import tiktoken
    tk = tiktoken.get_encoding("gpt2"); word = torch.tensor([tk.decode([t]).strip().isalpha() for t in tok.tolist()])
    report = {"positions": int(W.shape[0]), "closure_max": closure, "decomposition_max_rel_err": decomp, "alpha_median": float(alpha.median()), "gamma2_median": float(g2.median()), "cross_proj_median": float(pc.median()), "cross_negative_fraction": float((pc < 0).float().mean()),
              "only_proj_median": float(po.median()), "pearson_crossproj_vs_neg_cnorm": r, "c_norm_median": float(cc.norm(dim=1).median()), "word_fraction": float(word.float().mean()),
              "by_kind": {"word": {"gamma2": float(g2[word].median()), "cross": float(pc[word].median()), "only": float(po[word].median()), "alpha": float(alpha[word].median())}, "rest": {"gamma2": float(g2[~word].median()), "cross": float(pc[~word].median()), "only": float(po[~word].median()), "alpha": float(alpha[~word].median())}},
              "by_position_median": {"alpha": [float(alpha.view(-1, Tlen - 1)[:, p_].median()) for p_ in range(Tlen - 1)], "gamma2": [float(g2.view(-1, Tlen - 1)[:, p_].median()) for p_ in range(Tlen - 1)], "cross": [float(pc.view(-1, Tlen - 1)[:, p_].median()) for p_ in range(Tlen - 1)], "only": [float(po.view(-1, Tlen - 1)[:, p_].median()) for p_ in range(Tlen - 1)]}}
    print(json.dumps({k: v for k, v in report.items() if k != "by_position_median"}, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_table_term_near_full_on_text": report["gamma2_median"] >= GAMMA_MIN, "pred_c_cross_negative_on_text": report["cross_negative_fraction"] >= NEG_MIN,
                   "pred_d_cross_tracks_context_size_on_text": r >= R_MIN, "pred_e_decomposition_closes_on_text": decomp <= DECOMP_TOL}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_cross_cancellation_natural_result_v302", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
