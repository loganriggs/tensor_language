#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_factors_shared pred_b_program_fits_grid pred_c_head03_edit_cheap pred_d_five_heads_edit pred_e_window_free
"""Embedding-forward folding, rung 15 (v623): the GATED POSITIONAL FILTER program for the separable layer-0 heads — grid test, then a CE edit.

v622: the separable layer-0 heads (0.3 previous-token; 0.6 / 0.8 short window; 0.4 medium; 0.7 long ~1/d) have rank-one squared patterns at small
offsets. If the rank-one factors are SHARED across offsets, each head's whole pattern is a product of three tables,
    S_h^(d)(t,s) ~ kappa_h(d) A_h(t) B_h(s),      A_h, B_h over the vocabulary, kappa_h over the offset d = 1..512,
i.e. a positional filter kappa_h over per-token content, gated by the current token — the concrete "shift-operator template" (Logan, 19 Sep).
Construction, closed-form and loss-free: on the v612 4096 x 4096 grid, top singular pair (u, v) of S^(1); A(t) = S^(1)(t, grid_s) . v^ for every
vocabulary token, B(s) = u^ . S^(1)(grid_t, s); kappa(d) = <S^(d), A B^T> / ||A B^T||^2 on the grid for every d = 1..512 (one scalar per head and
offset; the grid residual is reported). Then the EDIT: inside the real model, block 0's squared_attention keeps its native diagonal (d = 0) and
replaces every off-diagonal pattern entry of a chosen head by kappa(i - j) A(tok_i) B(tok_j); values and the output projection stay native.
Priced in CE on the 192 x 512 skip7000 rows: head 0.3 alone, each other separable head alone, all five together, and all five with the filter
cut at a 32-position window. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_factors_shared     |cos(u_d, u_1)| >= 0.9 and |cos(v_d, v_1)| >= 0.9 for head 0.3 at d <= 4 and for head 0.7 at d <= 16. Prior: unsure
    pred_b_program_fits_grid  grid relative residual of kappa(d) A B^T <= 0.3 for head 0.3 at d in {1, 2} and head 0.7 at d <= 8. Prior: likely if (a)
    pred_c_head03_edit_cheap  replacing head 0.3's off-diagonal pattern costs <= 0.02 nats (its full ablation is +0.112, head_0_3_fold). Prior: unsure
    pred_d_five_heads_edit    replacing all five separable heads costs <= 0.06 nats. Prior: unsure
    pred_e_window_free        the 32-window version costs <= 1.25 x the full version + 0.005. Prior: likely
PRICE (registered maximum): 13 batched T=1 table forwards; 5 heads x 10 grid SVDs (4096^2) + 5 x 512 grid inner products; edits 8 configs x 6
batches = 48 forwards; total 61 forwards; 0 backwards; 0 loss fits (tables and kappa are closed-form projections of exact pattern objects).
Bar <= 70.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import embedding_forward_lib as EF

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt"
ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.gated_filter_edit_v623"
FORWARDS_MAX = 70
BATCH, EBATCH = 4096, 32
N_SCORE = 4096
HEADS = (3, 4, 6, 7, 8)
D_SVD = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
MAX_D = 512
WINDOW = 32
COS_MIN, RESID_MAX, H03_MAX, FIVE_MAX, WIN_RATIO, WIN_ABS = 0.9, 0.3, 0.02, 0.06, 1.25, 0.005
PREDICTIONS = {"pred_a_factors_shared": ">= 0.9 cosines (0.3: d<=4, 0.7: d<=16)", "pred_b_program_fits_grid": "resid <= 0.3", "pred_c_head03_edit_cheap": "<= 0.02",
               "pred_d_five_heads_edit": "<= 0.06", "pred_e_window_free": "<= 1.25x + 0.005"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "heads": list(HEADS), "d_svd": list(D_SVD), "window": WINDOW,
            "bars": {"cos_min": COS_MIN, "resid_max": RESID_MAX, "h03_max": H03_MAX, "five_max": FIVE_MAX, "win_ratio": WIN_RATIO, "win_abs": WIN_ABS}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; H = model.config.n_head; hd = D // H; V = model.config.vocab_size; dev = "cuda"
    b0 = model.transformer.h[0]; a = b0.attn; rms = EF.rms; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    with torch.no_grad():
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))

        def rot(x, d):
            cos, sin = (d * inv_freq).cos().bfloat16().float(), (d * inv_freq).sin().bfloat16().float()
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], -1)

        E = model.transformer.wte.weight.detach().float(); lam0 = b0.lambdas.detach().float()
        T = {k: torch.empty(V, H, hd, device=dev) for k in ("q", "q2", "k", "k2")}
        for s0 in range(0, V, BATCH):
            ids = torch.arange(s0, min(s0 + BATCH, V), device=dev)
            x0 = rms(E[ids]); n = rms(lam0[0] * x0 + lam0[1] * x0)
            T["q"][ids], T["k"][ids] = rms(a.c_q(n).view(-1, H, hd)), rms(a.c_k(n).view(-1, H, hd))
            T["q2"][ids], T["k2"][ids] = rms(a.c_q2(n).view(-1, H, hd)), rms(a.c_k2(n).view(-1, H, hd))
            forwards += 1
        p_uni, _, _ = EF.unigram_weights(ROWS, V, dev=dev)
        gen = torch.Generator(device=dev).manual_seed(6120)
        ts = torch.multinomial(p_uni.float(), N_SCORE, replacement=False, generator=gen); ss = torch.multinomial(p_uni.float(), N_SCORE, replacement=False, generator=gen)

        def pattern(h, t_ids, s_ids, d):                                          # [len(t), len(s)]
            qr, q2r = rot(T["q"][t_ids, h], d), rot(T["q2"][t_ids, h], d)
            return ((qr @ T["k"][s_ids, h].T) / hd) * ((q2r @ T["k2"][s_ids, h].T) / hd)

        tables, grid = {}, {}
        for h in HEADS:
            S1 = pattern(h, ts, ss, 1); U1, sv1, Vh1 = torch.linalg.svd(S1, full_matrices=False); u1, v1 = U1[:, 0], Vh1[0]
            cos = {}
            for d in D_SVD:
                Sd = pattern(h, ts, ss, d); Ud, svd_, Vhd = torch.linalg.svd(Sd, full_matrices=False)
                cos[d] = {"u": abs(float(u1 @ Ud[:, 0])), "v": abs(float(v1 @ Vhd[0])), "sep": float(svd_[0] ** 2 / svd_.square().sum())}
            # vocabulary-wide factors from the d=1 rank-one pair
            A = torch.empty(V, device=dev); B = torch.empty(V, device=dev)
            for s0 in range(0, V, BATCH):
                ids = torch.arange(s0, min(s0 + BATCH, V), device=dev)
                A[ids] = pattern(h, ids, ss, 1) @ v1; B[ids] = u1 @ pattern(h, ts, ids, 1)
            AB = torch.outer(A[ts], B[ss]); nAB = float(AB.square().sum())
            kappa = torch.zeros(MAX_D + 1, device=dev); resid = {}
            for d in range(1, MAX_D + 1):
                Sd = pattern(h, ts, ss, d); k_ = float((Sd * AB).sum() / nAB); kappa[d] = k_
                if d in D_SVD:
                    resid[d] = float((Sd - k_ * AB).square().sum() / Sd.square().sum())
            tables[h] = (A, B, kappa); grid[h] = {"cos": cos, "resid": resid, "kappa_head": kappa[1:33].tolist()}
            print(f"head 0.{h}: cos(u_d,u_1)/cos(v_d,v_1)/sep: " + " ".join(f"d{d}={cos[d]['u']:.2f}/{cos[d]['v']:.2f}/{cos[d]['sep']:.2f}" for d in D_SVD))
            print(f"          grid residual of kappa A B^T: " + " ".join(f"d{d}={resid[d]:.3f}" for d in D_SVD) + f" | kappa(1..8) {[round(x, 4) for x in kappa[1:9].tolist()]}")

        # ---- the edit: override off-diagonal patterns of chosen heads inside block 0's squared attention ---------------------
        state = {"idx": None, "heads": (), "window": None}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        native_sq = a.squared_attention

        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            scores = torch.einsum("bqhd,bkhd->bhqk", q, k); scores2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2)
            pat = (scores / Dn) * (scores2 / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
            if state["heads"]:
                idx = state["idx"]; pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0)      # [T, T]
                off = causal & (dmat > 0)
                for h in state["heads"]:
                    A, Bt, kappa = tables[h]; kap = kappa.clone()
                    if state["window"]:
                        kap[state["window"] + 1:] = 0.0
                    prog = kap[dmat][None] * A[idx][:, :, None] * Bt[idx][:, None, :]                                               # [B, T, T]
                    pat[:, h] = torch.where(off[None], prog, pat[:, h])
            return torch.einsum("bhqk,bkhd->bhqd", pat, v)

        a.squared_attention = patched

        def ce(heads, window=None):
            state["heads"], state["window"] = heads, window
            total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            return total / n, fw

        native, fw = ce(()); forwards += fw; print(f"native CE (patched attention, no override) {native:.5f}")
        edits = {}
        for h in HEADS:
            v_, fw = ce((h,)); forwards += fw; edits[f"head0.{h}"] = v_ - native
        v_, fw = ce(HEADS); forwards += fw; edits["five"] = v_ - native
        v_, fw = ce(HEADS, WINDOW); forwards += fw; edits[f"five_window{WINDOW}"] = v_ - native
        a.squared_attention = native_sq; pre.remove()
        print("CE added:", {k: round(v, 4) for k, v in edits.items()})
        disk_guard.guard_torch_save({f"head{h}_{n}": t.cpu() for h, (A, Bt, kappa) in tables.items() for n, t in (("A", A), ("B", Bt), ("kappa", kappa))}, str(OUT_PT), "v623 tables")
    g3, g7 = grid[3], grid[7]
    predictions = {"pred_a_factors_shared": all(g3["cos"][d]["u"] >= COS_MIN and g3["cos"][d]["v"] >= COS_MIN for d in D_SVD if d <= 4) and all(g7["cos"][d]["u"] >= COS_MIN and g7["cos"][d]["v"] >= COS_MIN for d in D_SVD if d <= 16),
                   "pred_b_program_fits_grid": all(g3["resid"][d] <= RESID_MAX for d in (1, 2)) and all(g7["resid"][d] <= RESID_MAX for d in D_SVD if d <= 8),
                   "pred_c_head03_edit_cheap": edits["head0.3"] <= H03_MAX,
                   "pred_d_five_heads_edit": edits["five"] <= FIVE_MAX,
                   "pred_e_window_free": edits[f"five_window{WINDOW}"] <= WIN_RATIO * edits["five"] + WIN_ABS}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_gated_filter_edit_result_v623", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "native_replay_err": abs(native - NATIVE_V615), "edits": edits, "grid": {str(h): {"cos": {str(d): v for d, v in g["cos"].items()}, "resid": {str(d): v for d, v in g["resid"].items()}, "kappa_1_32": g["kappa_head"]} for h, g in grid.items()}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
