#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_prev_share_at_mlp1_input pred_c_attention1_direct_dominates pred_d_write_prev_share pred_e_a1_cross_over_self
"""Embedding-forward folding, rung 6 (v614): the bigram fold pushed to MLP-1 — by which PATH does the previous token reach MLP-1?

With (s at position 0, t at position 1) everything at position 1 up to MLP-1's input is exact per pair: position 0 sees only s (all tables,
including block-1's keys / mixed values there); position 1 runs block 0 with attention-0's offset-1 channel (v612), MLP-0 (v613's split), then
block 1 with attention-1 reading x1(s) at position 0 and x1(t,s) at itself, and the token branch v1 (block-0 values) mixed into every value.
MLP-1's pre-norm input splits exactly into source families:  r(t) [pure-embedding lambda paths]  +  a0(t,s) = lambda0 O_0[self + cross]
+  m0(t,s) = lambda0 (Down_0 m + b)  +  a1(t,s) = O_1[self(t,s) vmix(t,s) + cross(t,s) vmix(s)],  a1 further split self / cross.
PREVIOUS-TOKEN DEPENDENCE of any f(t,s) on the 1024 x 1024 grid is its variance over s at fixed t:  Var_s f = sum ||f||^2 - sum_t ||mean_s f||^2
(exact from per-t sums; families interfere, so the total is reported next to the family values). Degree expansion by hops, Logan's first pick
("is it a two-hop path or a sum of one-hop shortcuts"): hop-0 direct (a0 cross on the residual), hop-0 relayed through MLP-0 (m0), hop-1 direct
(a1 cross), hop-1 through the prev-modified query (a1 self).
PREDICTIONS (scored as written; failures preserved; grid = the v612 unigram 1024 x 1024 draw)
    pred_a_replay                      manual MLP-1 normalised input at position 1 matches the model's own T=2 forward (forward-pre-hook) on the
                                       512 most frequent natural bigrams, rel-L2 <= 1e-4 (instrument)
    pred_b_prev_share_at_mlp1_input    Var_s / total energy of the pre-norm MLP-1 input <= 0.25 (v612: 0.24 median at MLP-0). Prior: unsure
    pred_c_attention1_direct_dominates Var_s(a1) >= Var_s(m0): attention-1's own channel carries more previous-token variance into MLP-1 than
                                       MLP-0's relayed write after the lambda chain. Prior: unsure
    pred_d_write_prev_share            Var_s share of MLP-1's write >= Var_s share of its normalised input (the bilinear map amplifies cross-token
                                       dependence through cur x prev products). Prior: likely
    pred_e_a1_cross_over_self          within a1, Var_s(cross) >= 2 x Var_s(self) (attention-1's previous-token dependence enters mainly by
                                       attending to position 0, not through the prev-modified query at position 1). Prior: likely
PRICE (registered maximum): 13 batched T=1 manual table forwards (blocks 0-1 at position 0) + 1 real T=2 verification forward (512 rows) = 14;
grid passes are weights x tables only. 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import embedding_forward_lib as EF

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_bigram_mlp1_v614_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_bigram_mlp1_v614_tensors.pt"
ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
CANDIDATE_ID = "embedding_forward.bigram_mlp1_v614"
FORWARDS_MAX = 16
BATCH = 4096
N_VERIFY = 512
N_GRID = 1024
CH = 32
REPLAY_TOL, SHARE_MAX, CROSS_SELF_RATIO = 1e-4, 0.25, 2.0
PREDICTIONS = {"pred_a_replay": "<= 1e-4", "pred_b_prev_share_at_mlp1_input": "<= 0.25", "pred_c_attention1_direct_dominates": "Var(a1) >= Var(m0)",
               "pred_d_write_prev_share": "write share >= input share", "pred_e_a1_cross_over_self": ">= 2x"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "n_verify": N_VERIFY, "n_grid": N_GRID,
            "bars": {"replay_tol": REPLAY_TOL, "share_max": SHARE_MAX, "cross_self_ratio": CROSS_SELF_RATIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; H = model.config.n_head; hd = D // H; V = model.config.vocab_size; dev = "cuda"
    b0, b1 = blocks[0], blocks[1]; A0, A1 = b0.attn, b1.attn; rms = EF.rms; forwards = 0
    eps = torch.finfo(torch.float32).eps
    with torch.no_grad():
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
        cos1, sin1 = inv_freq.cos().bfloat16().float(), inv_freq.sin().bfloat16().float()

        def rot1(x):
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            return torch.cat([x1 * cos1 + x2 * sin1, -x1 * sin1 + x2 * cos1], -1)

        def heads(lin, n):
            return lin(n).view(-1, H, hd)

        E = model.transformer.wte.weight.detach().float(); lam0, lam1 = b0.lambdas.detach().float(), b1.lambdas.detach().float()
        # ---- tables: block 0 at position 1 for token t (rotated q, self pattern), block 0/1 at position 0 for token s --------------
        T = {k: torch.empty(V, H, hd, device=dev) for k in ("q0r", "q20r", "k0", "k20", "v0", "k1", "k21", "vmix1")}
        X0, LIVE0, U0, X1S = (torch.empty(V, D, device=dev) for _ in range(4))
        for s0 in range(0, V, BATCH):
            ids = torch.arange(s0, min(s0 + BATCH, V), device=dev)
            x0 = rms(E[ids]); live = lam0[0] * x0 + lam0[1] * x0; n = rms(live)
            q, k, q2, k2, v = heads(A0.c_q, n), heads(A0.c_k, n), heads(A0.c_q2, n), heads(A0.c_k2, n), heads(A0.c_v, n)
            qh, kh, q2h, k2h = rms(q), rms(k), rms(q2), rms(k2)
            T["q0r"][ids], T["q20r"][ids], T["k0"][ids], T["k20"][ids], T["v0"][ids] = rot1(qh), rot1(q2h), kh, k2h, v
            alpha1 = ((rot1(qh) * rot1(kh)).sum(-1) / hd) * ((rot1(q2h) * rot1(k2h)).sum(-1) / hd)          # self pattern at position 1
            alpha0 = ((qh * kh).sum(-1) / hd) * ((q2h * k2h).sum(-1) / hd)                                  # self pattern at position 0
            X0[ids], LIVE0[ids] = x0, live
            U0[ids] = live + A0.c_proj((alpha1[..., None] * v).reshape(-1, D))                                # MLP-0 input at pos 1, current-token part
            xa = live + A0.c_proj((alpha0[..., None] * v).reshape(-1, D)); n0 = rms(xa)                      # position 0: token s alone
            x1s = xa + b0.mlp.Down(b0.mlp.Left(n0) * b0.mlp.Right(n0)) + b0.mlp.Down_bias; X1S[ids] = x1s
            n1 = rms(lam1[0] * x1s + lam1[1] * x0)
            T["k1"][ids], T["k21"][ids] = rms(heads(A1.c_k, n1)), rms(heads(A1.c_k2, n1))
            T["vmix1"][ids] = (1 - A1.lamb) * heads(A1.c_v, n1) + A1.lamb * v
            forwards += 1

        def pair(t_ids, s_ids):
            """Exact MLP-1 pre-norm input at position 1 for pairs (t, s), split by source family; plus MLP-1's bias-free write."""
            S0 = ((T["q0r"][t_ids] * T["k0"][s_ids]).sum(-1) / hd) * ((T["q20r"][t_ids] * T["k20"][s_ids]).sum(-1) / hd)
            c0 = A0.c_proj((S0[..., None] * T["v0"][s_ids]).reshape(-1, D))
            u = U0[t_ids]; xa = u + c0; n0 = rms(xa)
            w0 = b0.mlp.Down(b0.mlp.Left(n0) * b0.mlp.Right(n0)) + b0.mlp.Down_bias
            x1 = xa + w0; x0 = X0[t_ids]
            live1 = lam1[0] * x1 + lam1[1] * x0; n1 = rms(live1)
            q1, k1, q21, k21 = rot1(rms(heads(A1.c_q, n1))), rot1(rms(heads(A1.c_k, n1))), rot1(rms(heads(A1.c_q2, n1))), rot1(rms(heads(A1.c_k2, n1)))
            vm = (1 - A1.lamb) * heads(A1.c_v, n1) + A1.lamb * T["v0"][t_ids]
            s_self = ((q1 * k1).sum(-1) / hd) * ((q21 * k21).sum(-1) / hd)
            s_cross = ((q1 * T["k1"][s_ids]).sum(-1) / hd) * ((q21 * T["k21"][s_ids]).sum(-1) / hd)
            a1_self = A1.c_proj((s_self[..., None] * vm).reshape(-1, D)); a1_cross = A1.c_proj((s_cross[..., None] * T["vmix1"][s_ids]).reshape(-1, D))
            fam = {"r": lam1[0] * LIVE0[t_ids] + lam1[1] * x0, "a0": lam1[0] * (u - LIVE0[t_ids] + c0), "m0": lam1[0] * w0, "a1_self": a1_self, "a1_cross": a1_cross}
            X = sum(fam.values()); nX = rms(X)
            write = b1.mlp.Down(b1.mlp.Left(nX) * b1.mlp.Right(nX))
            return fam, X, nX, write

        # ---- replay on the 512 most frequent natural bigrams (real T=2 forward) ------------------------------------------------
        rows = torch.load(ROWS, map_location="cpu").long()
        pairs = torch.stack([rows[:, :-1].reshape(-1), rows[:, 1:].reshape(-1)], 1)
        uniq, counts = torch.unique(pairs, dim=0, return_counts=True)
        top = torch.argsort(counts, descending=True)[:N_VERIFY]; vs, vt = uniq[top, 0].to(dev), uniq[top, 1].to(dev)
        seq = torch.stack([vs, vt], 1); cache = {}
        hook = b1.mlp.register_forward_pre_hook(lambda m, args: cache.__setitem__("n", args[0].detach().float().clone()))
        model(seq, seq.clone()); hook.remove(); forwards += 1
        _, _, nX_v, _ = pair(vt, vs)
        replay = float((nX_v - cache["n"][:, 1]).norm() / cache["n"][:, 1].norm())
        print("replay rel-L2 (MLP-1 normalised input, position 1):", replay)

        # ---- grid: previous-token variance by family -------------------------------------------------------------------------
        p_uni, _, _ = EF.unigram_weights(ROWS, V, dev=dev)
        gen = torch.Generator(device=dev).manual_seed(6120)
        tg = torch.multinomial(p_uni.float(), 4096, replacement=False, generator=gen)[:N_GRID]
        sg = torch.multinomial(p_uni.float(), 4096, replacement=False, generator=gen)[:N_GRID]
        keys = ("r", "a0", "m0", "a1_self", "a1_cross", "a1", "X", "nX", "write")
        E2 = {k: 0.0 for k in keys}; M = {k: torch.zeros(N_GRID, D, dtype=torch.float64, device=dev) for k in keys}
        for s0 in range(0, N_GRID, CH):
            sc = sg[s0:s0 + CH]; tt = tg.repeat_interleave(len(sc)); ss = sc.repeat(N_GRID)
            fam, X, nX, write = pair(tt, ss)
            fam = dict(fam); fam["a1"] = fam["a1_self"] + fam["a1_cross"]; fam["X"] = X; fam["nX"] = nX; fam["write"] = write
            for k in keys:
                x = fam[k].double().view(N_GRID, len(sc), D)
                E2[k] += float(x.square().sum()); M[k] += x.sum(1)
        var = {k: E2[k] - float(M[k].square().sum()) / N_GRID for k in keys}
        share = {k: var[k] / E2[k] for k in keys}
        rms_tab = {k: (E2[k] / (N_GRID * N_GRID)) ** 0.5 for k in keys}
        print("energy rms per family:", {k: round(v, 3) for k, v in rms_tab.items()})
        print("prev-variance (Var_s):", {k: f"{v:.4g}" for k, v in var.items()})
        print("prev-variance share  :", {k: round(v, 4) for k, v in share.items()})
        print("Var_s(a1)/Var_s(m0) =", var["a1"] / var["m0"], "| a1 cross/self =", var["a1_cross"] / var["a1_self"], "| sum of family Var / Var(X) =", sum(var[k] for k in ("r", "a0", "m0", "a1")) / var["X"])
        disk_guard.guard_torch_save({"mean_over_s": {k: M[k].float().cpu() / N_GRID for k in ("X", "write")}, "grid_t": tg.cpu(), "grid_s": sg.cpu()}, str(OUT_PT), "v614 tensors")

    report = {"replay_rel_l2": replay, "energy": E2, "prev_variance": var, "prev_variance_share": share, "rms": rms_tab,
              "family_var_sum_over_total": sum(var[k] for k in ("r", "a0", "m0", "a1")) / var["X"], "lambdas": {"block0": lam0.tolist(), "block1": lam1.tolist(), "attn1_lamb": float(A1.lamb)}}
    predictions = {"pred_a_replay": replay <= REPLAY_TOL,
                   "pred_b_prev_share_at_mlp1_input": share["X"] <= SHARE_MAX,
                   "pred_c_attention1_direct_dominates": var["a1"] >= var["m0"],
                   "pred_d_write_prev_share": share["write"] >= share["nX"],
                   "pred_e_a1_cross_over_self": var["a1_cross"] >= CROSS_SELF_RATIO * var["a1_self"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_bigram_mlp1_result_v614", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
