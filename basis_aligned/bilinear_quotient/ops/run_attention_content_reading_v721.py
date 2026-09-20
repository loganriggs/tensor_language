#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_tables_replay pred_b_rank_r_tables_faithful pred_c_layer0_separable pred_d_layer1_content_interacts pred_e_interaction_rank_small
"""Attention lane, v721: what the CONTENT directions of the fitted pattern program say — exact token tables for the 18 heads of layers 0-1.

The factored program (v718) gives each head four rank-r maps (r = 16, or 64 for the content heads). At layers 0 and 1 the attention input is
an exact function of the single token (v609: rel-L2 2e-7 against the model's hooks), so the program's content pattern between a query token
t_i and a key token t_j at offset d is an exact table: P_r(t_i, t_j; d) = (q R_d k / 128)(q2 R_d k2 / 128) with q = rms(n(t_i) W_q,r^T) etc.
and R_d the rotary at offset d. On the 1024 most frequent tokens (unigram weights from the 480 x 513 skip80 rows) and offsets d in {1, 2, 4,
16}, each table is decomposed (weighted two-way ANOVA) into mean + query-only A(t_i) + key-only B(t_j) + interaction I(t_i, t_j); we report the
variance shares, the interaction's 90%-energy rank, the correlation of the rank-r table with the NATIVE table (full-rank maps, same
construction — the fold-level fidelity), and the top tokens / pairs as data for a reading. FOLD only (13 forwards for the single-token
tables). Readings are not claimed here; v660/v661 showed table orderings are not readings until an edit confirms them.
PREDICTIONS (scored as written; failures preserved)
    pred_a_tables_replay         the single-token attention-0 input replays v609's X0 construction: rel-L2 of X0 vs the model's own MLP-0 input hook on 64 tokens <= 1e-5
    pred_b_rank_r_tables_faithful at d = 1, the rank-r table correlates >= 0.9 with the native table for >= 12 of the 18 heads. Prior: unsure
    pred_c_layer0_separable      at d = 1, >= 5 layer-0 heads have separable share (mean + A + B) >= 0.8 (v623's gated positional filters). Prior: likely
    pred_d_layer1_content_interacts heads 1.1 and 1.4 (rank 64) have interaction share >= 0.3 at d = 1 (bigram matching, not gating). Prior: unsure
    pred_e_interaction_rank_small every head's interaction table at d = 1 has 90%-energy rank <= 32. Prior: unsure
PRICE (registered maximum): 13 forwards (single-token tables, batch 4096) + 1 hook forward (64 tokens); 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import embedding_forward_lib as EF

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_content_reading_v721_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
CANDIDATE_ID = "attention.content_reading_v721"
FORWARDS_MAX = 16
LAYERS = (0, 1); H = 9
GRID, OFFSETS, TOPK = 1024, (1, 2, 4, 16), 6
REPLAY_TOL, CORR_MIN, N_FAITHFUL, SEP_MIN, N_SEP, INTER_MIN, RANK_MAX = 1e-5, 0.9, 12, 0.8, 5, 0.3, 32
PREDICTIONS = {"pred_a_tables_replay": "rel-L2 <= 1e-5", "pred_b_rank_r_tables_faithful": ">= 12 heads with corr >= 0.9 at d=1", "pred_c_layer0_separable": ">= 5 layer-0 heads separable >= 0.8",
               "pred_d_layer1_content_interacts": "1.1 and 1.4 interaction >= 0.3", "pred_e_interaction_rank_small": "all 90%-ranks <= 32"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "grid": GRID, "offsets": list(OFFSETS),
            "bars": {"replay_tol": REPLAY_TOL, "corr_min": CORR_MIN, "n_faithful": N_FAITHFUL, "sep_min": SEP_MIN, "n_sep": N_SEP, "inter_min": INTER_MIN, "rank_max": RANK_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    D = model.config.n_embd; hd = D // H; V = model.config.vocab_size; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    with torch.no_grad():
        tabs = EF.single_token_tables(model, dev=dev); forwards += tabs["forwards"]
        # replay check against the model's own MLP-0 input hook on 64 tokens
        cap = {}
        hk = blocks[0].mlp.register_forward_pre_hook(lambda m, a: cap.__setitem__("x", a[0].detach()))
        ids = torch.arange(64, device=dev); model(ids[:, None], ids[:, None]); forwards += 1; hk.remove()
        x_hook = cap["x"][:, 0]; x_tab = EF.rms(tabs["X0"][:64]); rel = float((x_hook.float() - x_tab).norm() / x_tab.norm())
        n_in = {0: EF.rms(tabs["fam0"]["r"]), 1: EF.rms(tabs["fam1"]["r"] + tabs["fam1"]["a0"] + tabs["fam1"]["m0"])}
        p, _, _ = EF.unigram_weights(FIT_ROWS, V, dev=dev); grid = torch.argsort(p, descending=True)[:GRID]; w = p[grid] / p[grid].sum(); w = w.float()
        fac = torch.load(PROGS718, map_location=dev)["factored"]
        cos_all, sin_all = blocks[0].attn.rotary(torch.zeros(1, max(OFFSETS) + 1, 1, hd, device=dev))
        cos_all, sin_all = cos_all[0, :, 0].float(), sin_all[0, :, 0].float()

        def rot(x, d):
            c, s_ = cos_all[d], sin_all[d]; h2 = hd // 2; x1, x2 = x[:, :h2], x[:, h2:]
            return torch.cat([x1 * c + x2 * s_, -x1 * s_ + x2 * c], 1)

        def table(l, h, maps, d):
            n = n_in[l][grid]
            q = F.rms_norm(n @ maps["c_q"].T, (hd,)); k = F.rms_norm(n @ maps["c_k"].T, (hd,)); q2 = F.rms_norm(n @ maps["c_q2"].T, (hd,)); k2 = F.rms_norm(n @ maps["c_k2"].T, (hd,))
            q, q2 = rot(q, d), rot(q2, d)                      # key at position 0 (identity rotation), query at position d
            return ((q @ k.T) / hd) * ((q2 @ k2.T) / hd)       # [GRID query, GRID key]

        def anova(T):
            mu = float(w @ T @ w); A = (T @ w) - mu; B = (w @ T) - mu; I = T - mu - A[:, None] - B[None, :]
            var = float(((T - mu) ** 2 * w[:, None] * w[None, :]).sum()); sA = float((A ** 2 * w).sum()); sB = float((B ** 2 * w).sum()); sI = float((I ** 2 * w[:, None] * w[None, :]).sum())
            S = torch.linalg.svdvals(I * w.sqrt()[:, None] * w.sqrt()[None, :]); cum = (S ** 2).cumsum(0) / (S ** 2).sum(); r90 = int((cum < 0.9).sum()) + 1
            return {"mean": mu, "var": var, "share_A": sA / max(var, 1e-30), "share_B": sB / max(var, 1e-30), "share_I": sI / max(var, 1e-30), "rank90_I": r90}, A, B, I

        def dec(i):
            return enc.decode([int(grid[i])])

        report = {"replay_rel_l2": rel, "heads": {}}
        for l in LAYERS:
            at = blocks[l].attn
            for h in range(H):
                key = f"{l}.{h}"; f_ = fac[key]
                maps_r = {n_: (f_[n_][0].to(dev).float() @ f_[n_][1].to(dev).float()) for n_ in ("c_q", "c_k", "c_q2", "c_k2")}
                maps_n = {n_: getattr(at, n_).weight[h * hd:(h + 1) * hd].detach().float() for n_ in ("c_q", "c_k", "c_q2", "c_k2")}
                r = int(f_["c_q"][0].shape[1]); entry = {"rank": r, "offsets": {}}
                for d in OFFSETS:
                    Tr, Tn = table(l, h, maps_r, d), table(l, h, maps_n, d)
                    corr = float(torch.corrcoef(torch.stack([Tr.flatten(), Tn.flatten()]))[0, 1])
                    st, A, B, I = anova(Tr); stn, _, _, _ = anova(Tn)
                    top_q = [dec(i) for i in torch.argsort(A, descending=True)[:TOPK].tolist()]; bot_q = [dec(i) for i in torch.argsort(A)[:TOPK].tolist()]
                    top_k = [dec(i) for i in torch.argsort(B, descending=True)[:TOPK].tolist()]; bot_k = [dec(i) for i in torch.argsort(B)[:TOPK].tolist()]
                    Iw = I * w.sqrt()[:, None] * w.sqrt()[None, :]; flat = torch.argsort(Iw.flatten(), descending=True)[:TOPK]
                    top_pairs = [(dec(int(ix // GRID)), dec(int(ix % GRID)), float(I.flatten()[ix])) for ix in flat]
                    entry["offsets"][str(d)] = {"corr_rank_r_vs_native": corr, "rank_r": st, "native": stn, "top_query": top_q, "bottom_query": bot_q, "top_key": top_k, "bottom_key": bot_k, "top_interaction_pairs": top_pairs}
                report["heads"][key] = entry
                s1 = entry["offsets"]["1"]["rank_r"]
                print(f"{key} (r={r}) d=1: corr(rank-r, native) {entry['offsets']['1']['corr_rank_r_vs_native']:.3f} | shares A {s1['share_A']:.2f} B {s1['share_B']:.2f} I {s1['share_I']:.2f} (native I {entry['offsets']['1']['native']['share_I']:.2f}) rank90(I) {s1['rank90_I']} | top query {entry['offsets']['1']['top_query'][:4]} | top key {entry['offsets']['1']['top_key'][:4]} | top pairs {[(a, b) for a, b, _ in entry['offsets']['1']['top_interaction_pairs'][:3]]}")
    d1 = {k: v["offsets"]["1"] for k, v in report["heads"].items()}
    n_faithful = sum(1 for v in d1.values() if v["corr_rank_r_vs_native"] >= CORR_MIN)
    n_sep = sum(1 for k, v in d1.items() if k.startswith("0.") and (1 - v["rank_r"]["share_I"]) >= SEP_MIN)
    inter_ok = all(d1[k]["rank_r"]["share_I"] >= INTER_MIN for k in ("1.1", "1.4"))
    rank_ok = all(v["rank_r"]["rank90_I"] <= RANK_MAX for v in d1.values())
    print(f"replay rel-L2 {rel:.2e} | faithful heads at d=1: {n_faithful}/18 | separable layer-0 heads: {n_sep}/9 | 1.1 / 1.4 interaction {d1['1.1']['rank_r']['share_I']:.2f} / {d1['1.4']['rank_r']['share_I']:.2f} | max rank90(I) {max(v['rank_r']['rank90_I'] for v in d1.values())}")
    predictions = {"pred_a_tables_replay": rel <= REPLAY_TOL, "pred_b_rank_r_tables_faithful": n_faithful >= N_FAITHFUL, "pred_c_layer0_separable": n_sep >= N_SEP,
                   "pred_d_layer1_content_interacts": inter_ok, "pred_e_interaction_rank_small": rank_ok}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    report.update({"n_faithful": n_faithful, "n_separable_layer0": n_sep})
    OUT.write_text(json.dumps({"schema": "attention_content_reading_result_v721", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
