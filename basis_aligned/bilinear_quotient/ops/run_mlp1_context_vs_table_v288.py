#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_recurrence_closure pred_b_table_matches_context_write pred_c_number_difference_matches pred_d_context_part_is_attention_borne
"""MLP 1: in-context write vs the context-free table (v288). RESULTS §250: MLP 1 is 79% a token table on corpus text. Here the question is asked for the rows the number
line is built on: at the noun of the v76 rows ("The traders lost ...", noun at position 1 after "The"), how close is MLP 1's write to its single-token table entry
(the same token at position 0, alone)? Both are captured natively (3 batches of rows + 1 batch of the 32 distinct noun tokens). Reported: per-noun relative error
and cosine between the in-context write and the table entry; the same for the plural - singular DIFFERENCE (the number contrast, which is what the downstream
detectors read); and the exact writer split of the in-context minus table gap at x1 (the block-1 pre-MLP residual) -- the gap can only come from what "The" adds
through attention 0 / 1 (the embedding and MLP 0 are position-free for a token-only input up to attention).
PREDICTIONS (scored as written; failures preserved; priors from §250: 79% on corpus, expected higher here with one context token)
    pred_a_recurrence_closure           x1 in context = lambda-recurrence of its writers (embedding, attn0, mlp0, attn1) within relative 1e-4, every row
    pred_b_table_matches_context_write  median over the 96 rows of ||write_ctx - write_table|| / ||write_ctx|| <= 0.25 and median cosine >= 0.95
    pred_c_number_difference_matches    pooled over the 48 pairs: cos(mean d_ctx, mean d_table) >= 0.90 and the table's difference carries >= 0.75 of the in-context difference's norm
    pred_d_context_part_is_attention_borne  the x1 gap (context minus single-token) is >= 0.90 attention (attn0 + attn1) by norm share of the writer split
PRICE (registered maximum): 3 row batches + 1 token batch = 4 forwards; 0 backwards; 0 fits. Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_context_vs_table_v288_result.json"
CANDIDATE_ID = "mlp1.token_table.context_vs_table_v288"
CLOSURE_TOL, ERR_MAX, COS_MIN, DIFF_COS_MIN, DIFF_NORM_MIN, ATTN_MIN, BATCH = 1e-4, 0.25, 0.95, 0.90, 0.75, 0.90, 32
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_recurrence_closure": "<= 1e-4", "pred_b_table_matches_context_write": "err <= 0.25, cos >= 0.95", "pred_c_number_difference_matches": "cos >= 0.90, norm share >= 0.75", "pred_d_context_part_is_attention_borne": ">= 0.90"}


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
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "err_max": ERR_MAX, "cos_min": COS_MIN, "diff_cos_min": DIFF_COS_MIN, "diff_norm_min": DIFF_NORM_MIN, "attn_min": ATTN_MIN}}
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
    blocks = backend.model.transformer.h; l00, l01, l10, l11 = (float(blocks[0].lambdas[0]), float(blocks[0].lambdas[1]), float(blocks[1].lambdas[0]), float(blocks[1].lambdas[1]))
    recon = l10 * (l00 * ctx["x0"] + l01 * ctx["x0"] + ctx["attn0"] + ctx["mlp0"]) + l11 * ctx["x0"] + ctx["attn1"]
    closure = float(((recon - ctx["x1"]).norm(dim=1) / ctx["x1"].norm(dim=1)).max())
    T = torch.stack([tab["mlp1"][tindex[row.ids[noun_of(row)]]] for row in rows]); W = ctx["mlp1"]
    err = (W - T).norm(dim=1) / W.norm(dim=1); cos = torch.nn.functional.cosine_similarity(W, T, dim=1)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pairs = [(i, partner[(row.construction, row.group, False)]) for i, row in enumerate(rows) if row.present]
    d_ctx = torch.stack([W[i] - W[j] for i, j in pairs]); d_tab = torch.stack([T[i] - T[j] for i, j in pairs])
    dcos = float(torch.nn.functional.cosine_similarity(d_ctx.mean(0), d_tab.mean(0), dim=0)); dnorm = float(d_tab.mean(0).norm() / d_ctx.mean(0).norm())
    pair_cos = float(torch.nn.functional.cosine_similarity(d_ctx, d_tab, dim=1).median())
    gap = {k: ctx[k] - torch.stack([tab[k][tindex[row.ids[noun_of(row)]]] for row in rows]) for k in ("x0", "attn0", "mlp0", "attn1")}
    gnorm = {k: float(v.norm(dim=1).mean()) for k, v in gap.items()}; wts = {"x0": l10 * (l00 + l01) + l11, "attn0": l10, "mlp0": l10, "attn1": 1.0}
    contrib = {k: abs(wts[k]) * gnorm[k] for k in gnorm}; attn_share = (contrib["attn0"] + contrib["attn1"]) / sum(contrib.values())
    report = {"closure_max": closure, "err_median": float(err.median()), "err_mean": float(err.mean()), "cos_median": float(cos.median()), "cos_min": float(cos.min()), "diff_mean_cos": dcos, "diff_norm_share": dnorm, "diff_pairwise_cos_median": pair_cos,
              "x1_gap_writer_norms": gnorm, "x1_gap_attention_share": attn_share, "write_norm_mean": float(W.norm(dim=1).mean()), "table_norm_mean": float(T.norm(dim=1).mean())}
    print({k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 2) for kk, vv in v.items()}) for k, v in report.items()})
    predictions = {"pred_a_recurrence_closure": closure <= CLOSURE_TOL, "pred_b_table_matches_context_write": float(err.median()) <= ERR_MAX and float(cos.median()) >= COS_MIN, "pred_c_number_difference_matches": dcos >= DIFF_COS_MIN and dnorm >= DIFF_NORM_MIN, "pred_d_context_part_is_attention_borne": attn_share >= ATTN_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_context_vs_table_result_v288", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
