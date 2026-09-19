#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_alpha_tracks_self_share_on_text pred_c_alpha_within_025_of_share pred_d_direction_kept_on_text pred_e_content_words_track_better
"""MLP 1: does the self-share law hold on natural text? (v297). v289-v291 (filler contexts): MLP 1's write at a token = alpha x table(token) + R with alpha
tracking the token's own-key share of the bilinear attention pattern (blocks 0 + 1). OOD test on the 128 natural rows (v272 FineWeb + v273 Pile, 24 tokens):
at every position >= 1, alpha_p = projection of MLP 1's write on the token's single-token table entry, s_p = own-key share (`dod_units.attention_self_share`);
Spearman-free statistics: Pearson r over positions, median |alpha - s|, median cosine(write, table); split by token kind (alphabetic word tokens vs the rest).
PREDICTIONS (scored as written; failures preserved; priors from v291 where the gap grew with context length: 0.03 -> 0.20)
    pred_a_closure                        the block-1 recurrence reproduces the captured x1 within relative 1e-4 (instrument)
    pred_b_alpha_tracks_self_share_on_text  Pearson r(alpha, s) >= 0.50 over the ~2,944 positions
    pred_c_alpha_within_025_of_share      median |alpha - s| <= 0.25
    pred_d_direction_kept_on_text         median cosine(write, table entry) >= 0.60
    pred_e_content_words_track_better     r(alpha, s) for alphabetic word tokens exceeds r for the rest by >= 0.10. Prior: unsure.
PRICE (registered maximum): 2 natural batches (pattern hook, all positions at once) + unique tokens (<= 2,944) / 256 <= 12 table batches = 14 forwards; 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_self_share_law_natural_v297_result.json"
CANDIDATE_ID = "mlp1.token_table.self_share_law_natural_v297"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, R_MIN, GAP_MAX, COS_MIN, KIND_GAP, BATCH = 1e-4, 0.50, 0.25, 0.60, 0.10, 256
FORWARDS_MAX = 16
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_alpha_tracks_self_share_on_text": "r >= 0.50", "pred_c_alpha_within_025_of_share": "<= 0.25", "pred_d_direction_kept_on_text": ">= 0.60", "pred_e_content_words_track_better": ">= 0.10"}


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "natural": [p.name for p in NATURAL], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "r_min": R_MIN, "gap_max": GAP_MAX, "cos_min": COS_MIN, "kind_gap": KIND_GAP}}
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
    T = tab["mlp1"][ti]; alpha = (W * T).sum(1) / (T * T).sum(1); cos = (W * T).sum(1) / (W.norm(dim=1) * T.norm(dim=1))
    # instrument: the MLP-1 write recomputed from the captured x1 equals the captured write
    mlp = model.transformer.h[1].mlp; rec = mlp(F.rms_norm(X1.to("cuda"), (X1.shape[-1],))).float().cpu(); closure = float(((rec - W).norm(dim=1) / W.norm(dim=1)).max())
    def pearson(a, b): a, b = a - a.mean(), b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm()))
    import tiktoken
    tk = tiktoken.get_encoding("gpt2"); word = torch.tensor([tk.decode([t]).strip().isalpha() for t in tok.tolist()])
    r_all, r_word, r_rest = pearson(alpha, S), pearson(alpha[word], S[word]), pearson(alpha[~word], S[~word])
    report = {"positions": int(W.shape[0]), "closure_max": closure, "pearson_alpha_share": r_all, "pearson_word": r_word, "pearson_rest": r_rest, "word_fraction": float(word.float().mean()), "alpha_median": float(alpha.median()), "share_median": float(S.median()),
              "gap_median": float((alpha - S).abs().median()), "cos_median": float(cos.median()), "cos_word_median": float(cos[word].median()), "cos_rest_median": float(cos[~word].median()),
              "alpha_by_position_median": [float(alpha.view(-1, Tlen - 1)[:, p_].median()) for p_ in range(Tlen - 1)], "share_by_position_median": [float(S.view(-1, Tlen - 1)[:, p_].median()) for p_ in range(Tlen - 1)]}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_alpha_tracks_self_share_on_text": r_all >= R_MIN, "pred_c_alpha_within_025_of_share": report["gap_median"] <= GAP_MAX, "pred_d_direction_kept_on_text": report["cos_median"] >= COS_MIN,
                   "pred_e_content_words_track_better": r_word - r_rest >= KIND_GAP}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_self_share_law_natural_result_v297", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
