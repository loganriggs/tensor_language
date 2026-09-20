#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exactness pred_b_interaction_t_rank pred_c_interaction_s_rank pred_d_core_retained pred_e_cc_t_rank
"""Embedding-forward folding, rung 5 (v613): Tucker profile of MLP-0's CROSS-TOKEN INTERACTION write on the (cur, prev) grid.

v612 found the previous-token channel into MLP-0 separable, c(t,s) ~ a(t) g(s) (t-mode rank 1, s-mode 129, o-mode 116 at 90%), and split MLP-0's
bigram write exactly into token-only / interaction / pure-previous product terms (0.75 / 0.12 / 0.08 of the energy). This rung folds the
interaction term itself — the first genuinely two-token bilinear object in the model — as a 3-mode tensor on the same 1024 x 1024 unigram grid:
    I[o,t,s] = Down[(L u(t)) o (R c(t,s)) + (L c(t,s)) o (R u(t))]_o / rho(t,s)^2       (rho = rms of the MLP-0 input u + c, kept exactly per pair)
and likewise U[o,t,s] = Down[(L u) o (R u)]/rho^2 (token-only; its s-dependence is only through rho) and C[o,t,s] = Down[(L c) o (R c)]/rho^2
(pure-previous). Nothing is materialised: two chunked passes over the grid accumulate the three mode Grams (HOSVD frames) and then the projected
core, so the (r_t, r_s, r_o) Tucker retained energy is exact for the sampled pairs. Reported per term: 90/95/99% mode ranks and retained energy at
(8,32,32), (16,64,64), (32,128,128), (64,256,256).
PREDICTIONS (scored as written; failures preserved)
    pred_a_exactness          U + I + C reproduces the bias-free write on every chunk (rel-L2 <= 1e-5) and each term's three mode Grams agree in
                              trace (rel <= 1e-6) (instrument)
    pred_b_interaction_t_rank the interaction term's 90%-energy mode-t rank <= 32 (the current token modulates the previous-token write through
                              few features, inheriting the rank-1 gain). Prior: unsure
    pred_c_interaction_s_rank the interaction term's 90%-energy mode-s rank <= 256 (about the cross term's 129 pushed through L/R). Prior: likely
    pred_d_core_retained      the (32,128,128) HOSVD core of the interaction term retains >= 0.50 of its energy. Prior: unsure
    pred_e_cc_t_rank          the pure-previous term's 90%-energy mode-t rank <= 2 (it is a(t)^2 x a prev-only vector if separability holds). Prior: likely
PRICE (registered maximum): 13 batched T=1 manual table forwards (block-0 tables as in v612); grid passes are weights x tables only. 0 real model
forwards; 0 backwards; 0 fits. Bar <= 14.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import embedding_forward_lib as EF

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_interaction_tucker_v613_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_interaction_tucker_v613_frames.pt"
ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
CANDIDATE_ID = "embedding_forward.interaction_tucker_v613"
FORWARDS_MAX = 14
BATCH = 4096
N_GRID = 1024
CH = 64
CORES = ((8, 32, 32), (16, 64, 64), (32, 128, 128), (64, 256, 256))
EXACT_TOL, TRACE_TOL, T_RANK_MAX, S_RANK_MAX, CORE_MIN, CC_T_RANK_MAX = 1e-5, 1e-6, 32, 256, 0.50, 2
PREDICTIONS = {"pred_a_exactness": "<= 1e-5 / 1e-6", "pred_b_interaction_t_rank": "<= 32", "pred_c_interaction_s_rank": "<= 256",
               "pred_d_core_retained": ">= 0.50 at (32,128,128)", "pred_e_cc_t_rank": "<= 2"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "n_grid": N_GRID, "cores": [list(c) for c in CORES],
            "bars": {"exact_tol": EXACT_TOL, "trace_tol": TRACE_TOL, "t_rank_max": T_RANK_MAX, "s_rank_max": S_RANK_MAX, "core_min": CORE_MIN, "cc_t_rank_max": CC_T_RANK_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; H = model.config.n_head; hd = D // H; V = model.config.vocab_size; dev = "cuda"
    b0 = blocks[0]; a = b0.attn; rms = EF.rms; forwards = 0
    with torch.no_grad():
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
        cos1, sin1 = inv_freq.cos().bfloat16().float(), inv_freq.sin().bfloat16().float()

        def rot1(x):
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            return torch.cat([x1 * cos1 + x2 * sin1, -x1 * sin1 + x2 * cos1], -1)

        E = model.transformer.wte.weight.detach().float(); lam0 = b0.lambdas.detach().float()
        T = {k: torch.empty(V, H, hd, device=dev) for k in ("qr", "q2r", "k", "k2", "v")}
        U = torch.empty(V, D, device=dev)
        for s0 in range(0, V, BATCH):
            ids = torch.arange(s0, min(s0 + BATCH, V), device=dev)
            x0 = rms(E[ids]); live = lam0[0] * x0 + lam0[1] * x0; n = rms(live)
            q, k, q2, k2, v = (a.c_q(n).view(-1, H, hd), a.c_k(n).view(-1, H, hd), a.c_q2(n).view(-1, H, hd), a.c_k2(n).view(-1, H, hd), a.c_v(n).view(-1, H, hd))
            qh, kh, q2h, k2h = rms(q), rms(k), rms(q2), rms(k2)
            T["qr"][ids], T["q2r"][ids], T["k"][ids], T["k2"][ids], T["v"][ids] = rot1(qh), rot1(q2h), kh, k2h, v
            alpha_self = ((T["qr"][ids] * rot1(kh)).sum(-1) / hd) * ((T["q2r"][ids] * rot1(k2h)).sum(-1) / hd)
            U[ids] = live + a.c_proj((alpha_self[..., None] * v).reshape(-1, D))
            forwards += 1
        L0, R0, Dw0 = b0.mlp.Left.weight.detach().float(), b0.mlp.Right.weight.detach().float(), b0.mlp.Down.weight.detach().float()
        eps = torch.finfo(torch.float32).eps

        def terms(t_ids, s_ids):
            """Exact U / I / C terms [N, D] for pairs (t, s), plus the bias-free write for the exactness check."""
            S = ((T["qr"][t_ids] * T["k"][s_ids]).sum(-1) / hd) * ((T["q2r"][t_ids] * T["k2"][s_ids]).sum(-1) / hd)
            c = a.c_proj((S[..., None] * T["v"][s_ids]).reshape(-1, D)); u = U[t_ids]; x = u + c
            rho2 = x.square().mean(-1, keepdim=True) + eps; n = x / rho2.sqrt()
            Lu, Ru, Lc, Rc = u @ L0.T, u @ R0.T, c @ L0.T, c @ R0.T
            uu = ((Lu * Ru) / rho2) @ Dw0.T; inter = ((Lu * Rc + Lc * Ru) / rho2) @ Dw0.T; cc = ((Lc * Rc) / rho2) @ Dw0.T
            write = ((n @ L0.T) * (n @ R0.T)) @ Dw0.T
            return {"uu": uu, "inter": inter, "cc": cc}, write

        p_uni, _, _ = EF.unigram_weights(ROWS, V, dev=dev)
        gen = torch.Generator(device=dev).manual_seed(6120)                              # same grid draw as v612
        tg = torch.multinomial(p_uni.float(), 4096, replacement=False, generator=gen)[:N_GRID]
        sg = torch.multinomial(p_uni.float(), 4096, replacement=False, generator=gen)[:N_GRID]
        names = ("uu", "inter", "cc")
        # ---- pass 1: mode Grams ----------------------------------------------------------------------------------------------
        Gt = {k: torch.zeros(N_GRID, N_GRID, dtype=torch.float64, device=dev) for k in names}
        Gs = {k: torch.zeros(N_GRID, N_GRID, dtype=torch.float64, device=dev) for k in names}
        Go = {k: torch.zeros(D, D, dtype=torch.float64, device=dev) for k in names}
        exact_max = 0.0
        for s0 in range(0, N_GRID, CH):                                                  # s-chunks: Gt and Go
            sc = sg[s0:s0 + CH]; tt = tg.repeat_interleave(len(sc)); ss = sc.repeat(N_GRID)
            tm, write = terms(tt, ss)
            exact_max = max(exact_max, float((tm["uu"] + tm["inter"] + tm["cc"] - write).norm() / write.norm()))
            for k in names:
                x = tm[k].double(); Go[k] += x.T @ x; xt = x.view(N_GRID, -1); Gt[k] += xt @ xt.T
        for t0_ in range(0, N_GRID, CH):                                                 # t-chunks: Gs
            tc = tg[t0_:t0_ + CH]; tt = tc.repeat_interleave(N_GRID); ss = sg.repeat(len(tc))
            tm, _ = terms(tt, ss)
            for k in names:
                xs = tm[k].double().view(len(tc), N_GRID, D).permute(1, 0, 2).reshape(N_GRID, -1); Gs[k] += xs @ xs.T
        frames, profile = {}, {}
        for k in names:
            et, Ut = torch.linalg.eigh(Gt[k]); es, Us = torch.linalg.eigh(Gs[k]); eo, Uo = torch.linalg.eigh(Go[k])
            tr = (float(Gt[k].trace()), float(Gs[k].trace()), float(Go[k].trace()))
            profile[k] = {"energy": tr[0], "trace_rel_spread": (max(tr) - min(tr)) / max(tr), "mode_t": EF.energy_ranks(et), "mode_s": EF.energy_ranks(es), "mode_o": EF.energy_ranks(eo),
                          "top_t_evals_frac": (et.flip(0)[:8] / et.sum()).tolist()}
            frames[k] = (Ut.flip(1)[:, :max(c[0] for c in CORES)], Us.flip(1)[:, :max(c[1] for c in CORES)], Uo.flip(1)[:, :max(c[2] for c in CORES)])
            print(f"{k:5s}: energy {tr[0]:.4g} | 90/95/99% ranks t {profile[k]['mode_t']} s {profile[k]['mode_s']} o {profile[k]['mode_o']} | top t-evals {[round(v, 3) for v in profile[k]['top_t_evals_frac'][:4]]}")
        # ---- pass 2: projected cores ---------------------------------------------------------------------------------------------
        rt, rs, ro = (max(c[i] for c in CORES) for i in range(3))
        core = {k: torch.zeros(rt, rs, ro, dtype=torch.float64, device=dev) for k in names}
        for s0 in range(0, N_GRID, CH):
            sc = sg[s0:s0 + CH]; tt = tg.repeat_interleave(len(sc)); ss = sc.repeat(N_GRID)
            tm, _ = terms(tt, ss)
            for k in names:
                Pt, Ps, Po = frames[k]
                x = tm[k].double().view(N_GRID, len(sc), D) @ Po                              # [Nt, ch, ro]
                x = torch.einsum("tp,tcq->pcq", Pt, x)                                       # [rt, ch, ro]
                core[k] += torch.einsum("cs,pcq->psq", Ps[s0:s0 + CH], x)                    # [rt, rs, ro]
        retained = {k: {f"{c[0]}x{c[1]}x{c[2]}": float(core[k][:c[0], :c[1], :c[2]].square().sum() / profile[k]["energy"]) for c in CORES} for k in names}
        for k in names:
            print(f"{k:5s} retained: {retained[k]}")
        disk_guard.guard_torch_save({k: {"Pt": v[0].float().cpu(), "Ps": v[1].float().cpu(), "Po": v[2].float().cpu()} for k, v in frames.items()} | {"grid_t": tg.cpu(), "grid_s": sg.cpu()},
                                    str(OUT_PT), "v613 frames")

    report = {"profile": profile, "retained": retained, "exact_max_rel": exact_max, "energy_fractions": {k: profile[k]["energy"] / sum(profile[j]["energy"] for j in names) for k in names}}
    predictions = {"pred_a_exactness": exact_max <= EXACT_TOL and all(profile[k]["trace_rel_spread"] <= TRACE_TOL for k in names),
                   "pred_b_interaction_t_rank": profile["inter"]["mode_t"]["0.9"] <= T_RANK_MAX,
                   "pred_c_interaction_s_rank": profile["inter"]["mode_s"]["0.9"] <= S_RANK_MAX,
                   "pred_d_core_retained": retained["inter"]["32x128x128"] >= CORE_MIN,
                   "pred_e_cc_t_rank": profile["cc"]["mode_t"]["0.9"] <= CC_T_RANK_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_interaction_tucker_result_v613", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
