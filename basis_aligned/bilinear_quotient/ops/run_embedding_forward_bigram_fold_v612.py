#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_bigram_replay pred_b_cross_share_moderate pred_c_interaction_over_pure pred_d_natural_over_random pred_e_score_rank
"""Embedding-forward folding, rung 4 (v612): the TWO-POSITION (prev, cur) fold at MLP-0 — the first genuinely cross-token tensor.

With tokens (s at position 0, t at position 1), everything at position 1 up to MLP-0's input is an exact function of the pair. The previous
token enters ONLY through attention-0's cross term: per head h, S_h(t,s) v_h(s) with the unnormalised squared pattern
    S_h(t,s) = (rot1(q^_h(t)) . k^_h(s) / 128) x (rot1(q^2_h(t)) . k^2_h(s) / 128)          (rotary offset 1; the model's bf16 cos/sin are replicated)
so MLP-0's input is x = u(t) + c(t,s), u = live0(t) + O_0 [alpha_self(t) v(t)] (a table), c = O_0 sum_h S_h(t,s) v_h(s). Its write splits exactly
into product terms of the normalised input n = (u + c)/rho:  Down[(Lu o Ru) + (Lu o Rc + Lc o Ru) + (Lc o Rc)]/rho^2 + bias  — token-only, INTERACTION
(current x previous), and pure-previous. This is the concrete form of the framework's "attention outputs as coordinates z": the cross-token
coordinates are the 9 x 128 numbers S_h v_h(s), and the bilinear MLP couples them with the current token's coordinates through L, R.
Rows: natural bigrams = every adjacent pair in the 480 x 513 FineWeb row cache (245k instances, unique pairs weighted by count); a RANDOM-PAIR null
keeps every t and redraws s from the unigram distribution. Score-matrix ranks: per head, the 4096 x 4096 sampled pattern matrix (t, s drawn from the
unigram distribution without replacement), SVD energy ranks vs its two rank-<=128 factors. Cross-term Tucker profile: mode-t / mode-s / mode-o energy
ranks of c(t,s) on a 1024 x 1024 unigram-sampled grid (Grams only).
PREDICTIONS (scored as written; failures preserved)
    pred_a_bigram_replay            the manual pair computation matches the model's own MLP-0 input (forward-pre-hook, position 1) on 512 natural
                                    bigrams run as real T=2 sequences, rel-L2 <= 1e-4 (instrument)
    pred_b_cross_share_moderate     median over natural bigrams of ||c||^2 / ||u + c||^2 <= 0.25 (block 0 stays mostly a current-token function; the
                                    previous token is a moderate perturbation). Prior: unsure
    pred_c_interaction_over_pure    in the bias-free write, energy(interaction term) >= 2 x energy(pure-previous term) over natural bigrams (the
                                    previous token acts through its product with the current token, not on its own). Prior: likely
    pred_d_natural_over_random      mean cross share on natural bigrams >= 1.2 x the mean on random pairs (attention-0 attends more to real previous
                                    tokens than to unigram-matched random ones). Prior: unsure
    pred_e_score_rank               every head's sampled 4096^2 pattern matrix has 90%-energy rank <= 256 (well under the 128^2 Hadamard bound).
                                    Prior: unsure
PRICE (registered maximum): 13 batched T=1 manual table forwards + 1 real T=2 verification forward (512 rows) = 14; the pair census, score SVDs
and grid Grams are weights x tables only. 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import embedding_forward_lib as EF

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_bigram_fold_v612_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_bigram_fold_v612_tensors.pt"
ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
CANDIDATE_ID = "embedding_forward.bigram_fold_v612"
FORWARDS_MAX = 16
BATCH = 4096
N_VERIFY = 512
N_SCORE = 4096
N_GRID = 1024
REPLAY_TOL, SHARE_MAX, INTER_RATIO, NAT_RATIO, SCORE_RANK_MAX = 1e-4, 0.25, 2.0, 1.2, 256
PREDICTIONS = {"pred_a_bigram_replay": "<= 1e-4", "pred_b_cross_share_moderate": "median <= 0.25", "pred_c_interaction_over_pure": ">= 2x",
               "pred_d_natural_over_random": ">= 1.2x", "pred_e_score_rank": "<= 256 x 9 heads"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "n_verify": N_VERIFY, "n_score": N_SCORE, "n_grid": N_GRID,
            "bars": {"replay_tol": REPLAY_TOL, "share_max": SHARE_MAX, "inter_ratio": INTER_RATIO, "nat_ratio": NAT_RATIO, "score_rank_max": SCORE_RANK_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; H = model.config.n_head; hd = D // H; V = model.config.vocab_size; dev = "cuda"
    b0 = blocks[0]; a = b0.attn; rms = EF.rms
    forwards = 0
    with torch.no_grad():
        # ---- rotary at offset 1, replicating the model's bf16 cos/sin --------------------------------------------------------------
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
        cos1, sin1 = (1.0 * inv_freq).cos().bfloat16().float(), (1.0 * inv_freq).sin().bfloat16().float()

        def rot1(x):                                                             # x [..., hd]
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            return torch.cat([x1 * cos1 + x2 * sin1, -x1 * sin1 + x2 * cos1], -1)

        # ---- block-0 tables over the vocabulary --------------------------------------------------------------------------------
        E = model.transformer.wte.weight.detach().float(); lam0 = b0.lambdas.detach().float()
        T = {k: torch.empty(V, H, hd, device=dev) for k in ("qr", "q2r", "k", "k2", "kr", "k2r", "v")}
        LIVE, U = torch.empty(V, D, device=dev), torch.empty(V, D, device=dev)
        for s0 in range(0, V, BATCH):
            ids = torch.arange(s0, min(s0 + BATCH, V), device=dev)
            x0 = rms(E[ids]); live = lam0[0] * x0 + lam0[1] * x0; n = rms(live)
            q, k, q2, k2, v = (a.c_q(n).view(-1, H, hd), a.c_k(n).view(-1, H, hd), a.c_q2(n).view(-1, H, hd), a.c_k2(n).view(-1, H, hd), a.c_v(n).view(-1, H, hd))
            qh, kh, q2h, k2h = rms(q), rms(k), rms(q2), rms(k2)
            T["qr"][ids], T["q2r"][ids], T["k"][ids], T["k2"][ids] = rot1(qh), rot1(q2h), kh, k2h
            T["kr"][ids], T["k2r"][ids], T["v"][ids] = rot1(kh), rot1(k2h), v
            alpha_self = ((T["qr"][ids] * T["kr"][ids]).sum(-1) / hd) * ((T["q2r"][ids] * T["k2r"][ids]).sum(-1) / hd)       # [B, H] at position 1
            LIVE[ids] = live; U[ids] = live + a.c_proj((alpha_self[..., None] * v).reshape(-1, D))
            forwards += 1
        L0, R0, Dw0 = b0.mlp.Left.weight.detach().float(), b0.mlp.Right.weight.detach().float(), b0.mlp.Down.weight.detach().float()
        bias0 = b0.mlp.Down_bias.detach().float()

        def cross(t_ids, s_ids):                                                 # [N, H] scores and [N, D] c(t,s)
            S = ((T["qr"][t_ids] * T["k"][s_ids]).sum(-1) / hd) * ((T["q2r"][t_ids] * T["k2"][s_ids]).sum(-1) / hd)
            return S, a.c_proj((S[..., None] * T["v"][s_ids]).reshape(-1, D))

        # ---- natural bigrams + random-pair null ---------------------------------------------------------------------------------
        rows = torch.load(ROWS, map_location="cpu").long()
        pairs = torch.stack([rows[:, :-1].reshape(-1), rows[:, 1:].reshape(-1)], 1)          # (s, t)
        uniq, counts = torch.unique(pairs, dim=0, return_counts=True)
        w = (counts.double() / counts.sum()).to(dev); S_ids, T_ids = uniq[:, 0].to(dev), uniq[:, 1].to(dev)
        p_uni, n_tok, n_seen = EF.unigram_weights(ROWS, V, dev=dev)
        gen = torch.Generator(device=dev).manual_seed(612)
        S_rand = torch.multinomial(p_uni.float(), len(S_ids), replacement=True, generator=gen)
        print(f"bigram instances {int(counts.sum())}, unique pairs {len(uniq)}, unigram vocab seen {n_seen}")

        def census(s_ids, t_ids, wts):
            acc = {k: 0.0 for k in ("share_num", "share_den", "uu", "inter", "cc", "write", "uu_inter", "uu_cc", "inter_cc")}
            shares, head_share = [], torch.zeros(H, dtype=torch.float64, device=dev)
            for s0 in range(0, len(s_ids), BATCH):
                si, ti, pw = s_ids[s0:s0 + BATCH], t_ids[s0:s0 + BATCH], wts[s0:s0 + BATCH]
                S, c = cross(ti, si); u = U[ti]; x = u + c
                rho2 = (x.square().mean(-1, keepdim=True) + torch.finfo(torch.float32).eps)                 # rms^2 (F.rms_norm eps convention)
                n = x / rho2.sqrt()
                Lu, Ru, Lc, Rc = u @ L0.T, u @ R0.T, c @ L0.T, c @ R0.T
                t_uu = ((Lu * Ru) / rho2) @ Dw0.T; t_in = ((Lu * Rc + Lc * Ru) / rho2) @ Dw0.T; t_cc = ((Lc * Rc) / rho2) @ Dw0.T
                write = ((n @ L0.T) * (n @ R0.T)) @ Dw0.T
                assert float((t_uu + t_in + t_cc - write).norm() / write.norm()) < 1e-3
                sh = c.square().sum(-1) / x.square().sum(-1); shares.append(sh.double())
                acc["share_num"] += float((c.square().sum(-1).double() * pw).sum()); acc["share_den"] += float((x.square().sum(-1).double() * pw).sum())
                for key, tt in (("uu", t_uu), ("inter", t_in), ("cc", t_cc), ("write", write)):
                    acc[key] += float((tt.square().sum(-1).double() * pw).sum())
                acc["uu_inter"] += float(((t_uu * t_in).sum(-1).double() * pw).sum()); acc["uu_cc"] += float(((t_uu * t_cc).sum(-1).double() * pw).sum())
                acc["inter_cc"] += float(((t_in * t_cc).sum(-1).double() * pw).sum())
                per_head = (S[..., None] * T["v"][si]).reshape(-1, H, hd)                                   # pre-c_proj head outputs
                hs = torch.stack([a.c_proj.weight[:, h * hd:(h + 1) * hd].float() @ per_head[:, h].T for h in range(H)], 0).square().sum(1)   # [H, N]
                head_share += (hs.double() * pw).sum(1)
            shares = torch.cat(shares); order = torch.argsort(shares); cw = wts[order].cumsum(0)
            median = float(shares[order][int((cw < 0.5).sum())])
            W = acc["write"]
            return {"share_mean": acc["share_num"] / acc["share_den"], "share_median": median, "write_energy": W,
                    "fraction_uu": acc["uu"] / W, "fraction_interaction": acc["inter"] / W, "fraction_cc": acc["cc"] / W,
                    "cross_uu_inter": 2 * acc["uu_inter"] / W, "cross_uu_cc": 2 * acc["uu_cc"] / W, "cross_inter_cc": 2 * acc["inter_cc"] / W,
                    "head_share_of_cross": (head_share / head_share.sum()).tolist()}

        nat = census(S_ids, T_ids, w); rnd = census(S_rand, T_ids, w)
        print("natural:", {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in nat.items()})
        print("random :", {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in rnd.items()})

        # ---- replay against the real model on 512 natural bigrams as T=2 sequences ----------------------------------------------
        top = torch.argsort(counts, descending=True)[:N_VERIFY]
        vs, vt = uniq[top, 0].to(dev), uniq[top, 1].to(dev)
        seq = torch.stack([vs, vt], 1)
        cache = {}; hook = b0.mlp.register_forward_pre_hook(lambda m, args: cache.__setitem__("n", args[0].detach().float().clone()))
        model(seq, seq.clone()); hook.remove(); forwards += 1
        _, c_v = cross(vt, vs); n_manual = rms(U[vt] + c_v)
        replay = float((n_manual - cache["n"][:, 1]).norm() / cache["n"][:, 1].norm())
        print("replay rel-L2 (MLP-0 input, position 1):", replay)

        # ---- score-matrix ranks per head on a 4096 x 4096 unigram-sampled grid -------------------------------------------------
        gen2 = torch.Generator(device=dev).manual_seed(6120)
        ts = torch.multinomial(p_uni.float(), N_SCORE, replacement=False, generator=gen2); ss = torch.multinomial(p_uni.float(), N_SCORE, replacement=False, generator=gen2)
        score_ranks = []
        for h in range(H):
            F1 = (T["qr"][ts, h] @ T["k"][ss, h].T) / hd; F2 = (T["q2r"][ts, h] @ T["k2"][ss, h].T) / hd
            sv = torch.linalg.svdvals((F1 * F2).double()); sv1 = torch.linalg.svdvals(F1.double()); sv2 = torch.linalg.svdvals(F2.double())
            score_ranks.append({"head": h, "pattern": EF.energy_ranks(sv.square().flip(0)), "factor1": EF.energy_ranks(sv1.square().flip(0)), "factor2": EF.energy_ranks(sv2.square().flip(0)),
                                "pattern_rms": float((F1 * F2).square().mean().sqrt()), "top_sv": sv[:8].tolist()})
            print(f"head 0.{h}: pattern 90/95/99% rank {score_ranks[-1]['pattern']} | factors {sv1.square().cumsum(0).div(sv1.square().sum()).lt(0.9).sum().item() + 1}/{sv2.square().cumsum(0).div(sv2.square().sum()).lt(0.9).sum().item() + 1} | rms {score_ranks[-1]['pattern_rms']:.4f}")

        # ---- cross-term Tucker profile on a 1024 x 1024 grid ----------------------------------------------------------------------
        tg, sg = ts[:N_GRID], ss[:N_GRID]
        Gt = torch.zeros(N_GRID, N_GRID, dtype=torch.float64, device=dev); Gs = torch.zeros_like(Gt); Go = torch.zeros(D, D, dtype=torch.float64, device=dev)
        CH = 64
        for s0 in range(0, N_GRID, CH):
            sc = sg[s0:s0 + CH]; tt = tg.repeat_interleave(len(sc)); ssx = sc.repeat(N_GRID)
            _, c = cross(tt, ssx); c = c.view(N_GRID, len(sc) * D).double()
            Gt += c @ c.T; Go += c.view(-1, D).T @ c.view(-1, D)
        for t0_ in range(0, N_GRID, CH):
            tc = tg[t0_:t0_ + CH]; tt = tc.repeat_interleave(N_GRID); ssx = sg.repeat(len(tc))
            _, c = cross(tt, ssx); c = c.view(len(tc), N_GRID, D).permute(1, 0, 2).reshape(N_GRID, -1)
            Gs += c @ c.T
        tucker = {"mode_t": EF.energy_ranks(torch.linalg.eigvalsh(Gt)), "mode_s": EF.energy_ranks(torch.linalg.eigvalsh(Gs)), "mode_o": EF.energy_ranks(torch.linalg.eigvalsh(Go)),
                  "energy_check_rel": float(abs(Gt.trace() - Gs.trace()) / Gt.trace())}
        print("cross-term Tucker profile (t / s / o):", tucker)
        disk_guard.guard_torch_save({"score_ranks": score_ranks, "grid_t": tg.cpu(), "grid_s": sg.cpu(), "Go": Go.float().cpu()}, str(OUT_PT), "v612 tensors")

    report = {"natural": nat, "random": rnd, "replay_rel_l2": replay, "score_ranks": score_ranks, "cross_term_tucker": tucker,
              "bigrams": {"instances": int(counts.sum()), "unique": int(len(uniq))}}
    predictions = {"pred_a_bigram_replay": replay <= REPLAY_TOL,
                   "pred_b_cross_share_moderate": nat["share_median"] <= SHARE_MAX,
                   "pred_c_interaction_over_pure": nat["fraction_interaction"] >= INTER_RATIO * nat["fraction_cc"],
                   "pred_d_natural_over_random": nat["share_mean"] >= NAT_RATIO * rnd["share_mean"],
                   "pred_e_score_rank": all(r["pattern"]["0.9"] <= SCORE_RANK_MAX for r in score_ranks)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_bigram_fold_result_v612", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
