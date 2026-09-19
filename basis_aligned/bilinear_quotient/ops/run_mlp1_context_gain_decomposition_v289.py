#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_cross_terms_dominate pred_c_change_is_mostly_gain pred_d_gain_is_uniform_across_tokens
"""MLP 1: how one context token changes its write (v289). v288: at the noun of the v76 rows MLP 1's write keeps the direction of its single-token table entry
(cos 0.87) but not its magnitude (0.62x; the number contrast 1/1.6). MLP 1 is bilinear, so with the normalised block-1 input written as n = t + c (t = the single-token
entry's normalised input, c = the context change, exact), each unit is u_j(n) = (L_j.t)(R_j.t) + [(L_j.t)(R_j.c) + (L_j.c)(R_j.t)] + (L_j.c)(R_j.c), and the write change
W_ctx - W_table = Down [cross + context-only] exactly. A "gain" reading says the cross terms dominate and act mostly as a scalar on the table entry. Reported per row:
norm shares of the cross and context-only terms in the change; the scalar projection alpha = (W . T)/||T||^2 and the residual after it (1 - cos^2); alpha's spread.
PREDICTIONS (scored as written; failures preserved; priors from v288)
    pred_a_pair_closure               Down[cross + context-only] = W_ctx - W_table within relative 1e-3, every row
    pred_b_cross_terms_dominate       the cross terms carry >= 0.70 of the change's norm (context-only <= 0.30)
    pred_c_change_is_mostly_gain      after the best scalar alpha per row, the residual energy 1 - cos^2 is <= 0.30 of the in-context write's energy (median)
    pred_d_gain_is_uniform_across_tokens  alpha's coefficient of variation across the 32 nouns is <= 0.25 (one context gain, not token-specific rescaling)
PRICE (registered maximum): 3 row batches + 1 token batch = 4 forwards; 0 backwards; 0 fits (alpha is a projection coefficient, not a fitted parameter). Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_context_gain_decomposition_v289_result.json"
CANDIDATE_ID = "mlp1.token_table.context_gain_decomposition_v289"
CLOSURE_TOL, CROSS_MIN, RESID_MAX, CV_MAX, BATCH = 1e-3, 0.70, 0.30, 0.25, 32
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_pair_closure": "<= 1e-3", "pred_b_cross_terms_dominate": ">= 0.70", "pred_c_change_is_mostly_gain": "<= 0.30", "pred_d_gain_is_uniform_across_tokens": "cv <= 0.25"}


def capture(backend, tokens, pos):
    """writes at `pos` (per row) for blocks 0-1: x0, attn0, mlp0, attn1, x1, mlp1."""
    torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h; idx = torch.arange(tokens.shape[0]); out = {}
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None; out["x0"] = x0[idx, pos].float().cpu()
        for l in (0, 1):
            block = blocks[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            out[f"attn{l}"] = attention[idx, pos].float().cpu(); m = block.mlp(xin); out[f"mlp{l}"] = m[idx, pos].float().cpu()
            if l == 1: out["x1"] = x[idx, pos].float().cpu()
            x = x + m
    return out


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "cross_min": CROSS_MIN, "resid_max": RESID_MAX, "cv_max": CV_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch; fw = L.ManualForward(backend); forwards = 0
    ctx = {}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); pos = torch.tensor([noun_of(r) for r in chunk], device=tokens.device)
        c = capture(backend, tokens, pos); forwards += 1
        for k, v in c.items(): ctx.setdefault(k, []).append(v)
    ctx = {k: torch.cat(v) for k, v in ctx.items()}
    noun_ids = sorted({row.ids[noun_of(row)] for row in rows}); tok = torch.tensor(noun_ids, device="cuda").unsqueeze(1)
    tab = capture(backend, tok, torch.zeros(len(noun_ids), dtype=torch.long, device="cuda")); forwards += 1; tindex = {t: i for i, t in enumerate(noun_ids)}
    model = backend.model; F = backend.F; mlp = model.transformer.h[1].mlp
    Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    W = ctx["mlp1"]; T = torch.stack([tab["mlp1"][tindex[row.ids[noun_of(row)]]] for row in rows])
    n_ctx = F.rms_norm(ctx["x1"], (ctx["x1"].shape[-1],)); n_tab = F.rms_norm(torch.stack([tab["x1"][tindex[row.ids[noun_of(row)]]] for row in rows]), (ctx["x1"].shape[-1],))
    c = n_ctx - n_tab; Lt, Rt, Lc, Rc = n_tab @ Lw.T, n_tab @ Rw.T, c @ Lw.T, c @ Rw.T
    cross = (Lt * Rc + Lc * Rt) @ Dw.T; ctx_only = (Lc * Rc) @ Dw.T; change = W - T
    closure = float((((cross + ctx_only) - change).norm(dim=1) / change.norm(dim=1)).max())
    cross_share = float((cross.norm(dim=1) / (cross.norm(dim=1) + ctx_only.norm(dim=1))).median())
    alpha = (W * T).sum(1) / (T * T).sum(1); cos2 = ((W * T).sum(1) ** 2) / ((W * W).sum(1) * (T * T).sum(1)); resid = 1 - cos2
    alpha_by_token = {}
    for i, row in enumerate(rows): alpha_by_token.setdefault(row.ids[noun_of(row)], []).append(float(alpha[i]))
    al = torch.tensor([sum(v) / len(v) for v in alpha_by_token.values()]); cv = float(al.std() / al.mean())
    report = {"closure_max": closure, "cross_share_median": cross_share, "cross_norm_mean": float(cross.norm(dim=1).mean()), "context_only_norm_mean": float(ctx_only.norm(dim=1).mean()), "change_norm_mean": float(change.norm(dim=1).mean()),
              "alpha_median": float(alpha.median()), "alpha_cv_across_tokens": cv, "alpha_min_max": [float(al.min()), float(al.max())], "residual_after_scalar_median": float(resid.median()), "residual_after_scalar_mean": float(resid.mean()),
              "cross_along_T_share": float(((cross * T).sum(1) / (cross.norm(dim=1) * T.norm(dim=1))).abs().median())}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items()})
    predictions = {"pred_a_pair_closure": closure <= CLOSURE_TOL, "pred_b_cross_terms_dominate": cross_share >= CROSS_MIN, "pred_c_change_is_mostly_gain": float(resid.median()) <= RESID_MAX, "pred_d_gain_is_uniform_across_tokens": cv <= CV_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_context_gain_decomposition_result_v289", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
