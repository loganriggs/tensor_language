"""Embedding-forward folding, rung 46 (v657): the hop census at MLP-2 — by which paths does the previous token reach MLP-2?

v614 at MLP-1: previous-token variance enters 55% through attention-1's own offset-1 channel, 30% through the MLP-0 relay, ~0 directly from
attention-0's residual write; the bilinear map amplifies it x5.5. One block deeper, exact per pair (s at position 0, t at position 1) through
blocks 0-2 (position 0 is a table through block 2; position 1 runs the pair). MLP-2's pre-norm input splits exactly into families:
r(t) [pure-embedding lambda paths] + a0 + m0 + a1 + m1 (relays through the lambda chain) + a2_self + a2_cross (attention-2 at position 1 reading
itself / position 0). Var_s per family on the 1024 x 1024 grid; replay of MLP-2's normalised input vs the model's own T=2 forward. CE not involved.
PREDICTIONS (scored as written; failures preserved)
    pred_a_replay                   manual MLP-2 normalised input at position 1 matches the model's T=2 forward on the 512 most frequent bigrams, rel-L2 <= 1e-4 (instrument)
    pred_b_prev_share_at_mlp2_input Var_s / total energy of the pre-norm MLP-2 input <= 0.25. Prior: unsure
    pred_c_relays_dominate          Var_s(a1) + Var_s(m0) + Var_s(m1) >= Var_s(a2_self + a2_cross): the relayed previous-token content outweighs attention-2's own channel. Prior: unsure
    pred_d_write_prev_share         Var_s share of MLP-2's write >= Var_s share of its normalised input. Prior: likely
    pred_e_a2_cross_over_self       within attention-2, Var_s(cross) >= 2 x Var_s(self). Prior: likely
PRICE (registered maximum): 13 batched T=1 table forwards (blocks 0-2 at position 0) + 1 real T=2 verification forward = 14; grid passes are
weights x tables only. 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import embedding_forward_lib as EF

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_bigram_mlp2_v657_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_bigram_mlp2_v657_tensors.pt"
ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
CANDIDATE_ID = "embedding_forward.bigram_mlp2_v657"
FORWARDS_MAX = 16
BATCH = 4096
N_VERIFY = 512
N_GRID = 1024
CH = 32
REPLAY_TOL, SHARE_MAX, CROSS_SELF_RATIO = 1e-4, 0.25, 2.0
PREDICTIONS = {"pred_a_replay": "<= 1e-4", "pred_b_prev_share_at_mlp2_input": "<= 0.25", "pred_c_relays_dominate": "Var(a1+m0+m1) >= Var(a2)",
               "pred_d_write_prev_share": "write share >= input share", "pred_e_a2_cross_over_self": ">= 2x"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "n_verify": N_VERIFY, "n_grid": N_GRID,
            "bars": {"replay_tol": REPLAY_TOL, "share_max": SHARE_MAX, "cross_self_ratio": CROSS_SELF_RATIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; H = model.config.n_head; hd = D // H; V = model.config.vocab_size; dev = "cuda"
    b0, b1, b2 = blocks[0], blocks[1], blocks[2]; A0, A1, A2 = b0.attn, b1.attn, b2.attn; rms = EF.rms; forwards = 0
    eps = torch.finfo(torch.float32).eps
    with torch.no_grad():
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
        cos1, sin1 = inv_freq.cos().bfloat16().float(), inv_freq.sin().bfloat16().float()

        def rot1(x):
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            return torch.cat([x1 * cos1 + x2 * sin1, -x1 * sin1 + x2 * cos1], -1)

        def heads(lin, n):
            return lin(n).view(-1, H, hd)

        E = model.transformer.wte.weight.detach().float(); lam0, lam1, lam2 = b0.lambdas.detach().float(), b1.lambdas.detach().float(), b2.lambdas.detach().float()
        # ---- tables: block 0 at position 1 for token t (rotated q, self pattern), blocks 0/1/2 at position 0 for token s ------------
        T = {k: torch.empty(V, H, hd, device=dev) for k in ("q0r", "q20r", "k0", "k20", "v0", "k1", "k21", "vmix1", "k2", "k22", "vmix2")}
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
            # block 1 at position 0 (token s alone), then block-2 keys / mixed values
            q1s, k1s, q21s, k21s = rms(heads(A1.c_q, n1)), rms(heads(A1.c_k, n1)), rms(heads(A1.c_q2, n1)), rms(heads(A1.c_k2, n1))
            a1s = ((q1s * k1s).sum(-1) / hd) * ((q21s * k21s).sum(-1) / hd)
            live1s = lam1[0] * x1s + lam1[1] * x0; xb = live1s + A1.c_proj((a1s[..., None] * T["vmix1"][ids]).reshape(-1, D)); nb = rms(xb)
            x2s = xb + b1.mlp.Down(b1.mlp.Left(nb) * b1.mlp.Right(nb)) + b1.mlp.Down_bias
            n2 = rms(lam2[0] * x2s + lam2[1] * x0)
            T["k2"][ids], T["k22"][ids] = rms(heads(A2.c_k, n2)), rms(heads(A2.c_k2, n2))
            T["vmix2"][ids] = (1 - A2.lamb) * heads(A2.c_v, n2) + A2.lamb * v
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
            X1 = live1 + a1_self + a1_cross; nX1 = rms(X1); w1 = b1.mlp.Down(b1.mlp.Left(nX1) * b1.mlp.Right(nX1)) + b1.mlp.Down_bias
            x2 = X1 + w1; live2 = lam2[0] * x2 + lam2[1] * x0; n2 = rms(live2)
            q2_, k2_, q22_, k22_ = rot1(rms(heads(A2.c_q, n2))), rot1(rms(heads(A2.c_k, n2))), rot1(rms(heads(A2.c_q2, n2))), rot1(rms(heads(A2.c_k2, n2)))
            vm2 = (1 - A2.lamb) * heads(A2.c_v, n2) + A2.lamb * T["v0"][t_ids]
            s2_self = ((q2_ * k2_).sum(-1) / hd) * ((q22_ * k22_).sum(-1) / hd); s2_cross = ((q2_ * T["k2"][s_ids]).sum(-1) / hd) * ((q22_ * T["k22"][s_ids]).sum(-1) / hd)
            a2_self = A2.c_proj((s2_self[..., None] * vm2).reshape(-1, D)); a2_cross = A2.c_proj((s2_cross[..., None] * T["vmix2"][s_ids]).reshape(-1, D))
            L2 = lam2[0]
            fam = {"r": L2 * (lam1[0] * LIVE0[t_ids] + lam1[1] * x0) + lam2[1] * x0, "a0": L2 * lam1[0] * (u - LIVE0[t_ids] + c0), "m0": L2 * lam1[0] * w0,
                   "a1": L2 * (a1_self + a1_cross), "m1": L2 * w1, "a2_self": a2_self, "a2_cross": a2_cross}
            X = sum(fam.values()); nX = rms(X)
            write = b2.mlp.Down(b2.mlp.Left(nX) * b2.mlp.Right(nX))
            return fam, X, nX, write

        # ---- replay on the 512 most frequent natural bigrams (real T=2 forward) ------------------------------------------------
        rows = torch.load(ROWS, map_location="cpu").long()
        pairs = torch.stack([rows[:, :-1].reshape(-1), rows[:, 1:].reshape(-1)], 1)
        uniq, counts = torch.unique(pairs, dim=0, return_counts=True)
        top = torch.argsort(counts, descending=True)[:N_VERIFY]; vs, vt = uniq[top, 0].to(dev), uniq[top, 1].to(dev)
        seq = torch.stack([vs, vt], 1); cache = {}
        hook = b2.mlp.register_forward_pre_hook(lambda m, args: cache.__setitem__("n", args[0].detach().float().clone()))
        model(seq, seq.clone()); hook.remove(); forwards += 1
        _, _, nX_v, _ = pair(vt, vs)
        replay = float((nX_v - cache["n"][:, 1]).norm() / cache["n"][:, 1].norm())
        print("replay rel-L2 (MLP-2 normalised input, position 1):", replay)

        # ---- grid: previous-token variance by family -------------------------------------------------------------------------
        p_uni, _, _ = EF.unigram_weights(ROWS, V, dev=dev)
        gen = torch.Generator(device=dev).manual_seed(6120)
        tg = torch.multinomial(p_uni.float(), 4096, replacement=False, generator=gen)[:N_GRID]
        sg = torch.multinomial(p_uni.float(), 4096, replacement=False, generator=gen)[:N_GRID]
        keys = ("r", "a0", "m0", "a1", "m1", "a2_self", "a2_cross", "a2", "relays", "X", "nX", "write")
        E2 = {k: 0.0 for k in keys}; M = {k: torch.zeros(N_GRID, D, dtype=torch.float64, device=dev) for k in keys}
        for s0 in range(0, N_GRID, CH):
            sc = sg[s0:s0 + CH]; tt = tg.repeat_interleave(len(sc)); ss = sc.repeat(N_GRID)
            fam, X, nX, write = pair(tt, ss)
            fam = dict(fam); fam["a2"] = fam["a2_self"] + fam["a2_cross"]; fam["relays"] = fam["a1"] + fam["m0"] + fam["m1"]; fam["X"] = X; fam["nX"] = nX; fam["write"] = write
            for k in keys:
                x = fam[k].double().view(N_GRID, len(sc), D)
                E2[k] += float(x.square().sum()); M[k] += x.sum(1)
        var = {k: E2[k] - float(M[k].square().sum()) / N_GRID for k in keys}
        share = {k: var[k] / E2[k] for k in keys}
        rms_tab = {k: (E2[k] / (N_GRID * N_GRID)) ** 0.5 for k in keys}
        print("energy rms per family:", {k: round(v, 3) for k, v in rms_tab.items()})
        print("prev-variance (Var_s):", {k: f"{v:.4g}" for k, v in var.items()})
        print("prev-variance share  :", {k: round(v, 4) for k, v in share.items()})
        print("relays (a1+m0+m1) / a2 =", var["relays"] / var["a2"], "| a2 cross/self =", var["a2_cross"] / var["a2_self"], "| sum of family Var / Var(X) =", sum(var[k] for k in ("r", "a0", "m0", "a1", "m1", "a2")) / var["X"])
        disk_guard.guard_torch_save({"mean_over_s": {k: M[k].float().cpu() / N_GRID for k in ("X", "write")}, "grid_t": tg.cpu(), "grid_s": sg.cpu()}, str(OUT_PT), "v614 tensors")

    report = {"replay_rel_l2": replay, "energy": E2, "prev_variance": var, "prev_variance_share": share, "rms": rms_tab,
              "family_var_sum_over_total": sum(var[k] for k in ("r", "a0", "m0", "a1", "m1", "a2")) / var["X"], "lambdas": {"block0": lam0.tolist(), "block1": lam1.tolist(), "block2": lam2.tolist()}}
    predictions = {"pred_a_replay": replay <= REPLAY_TOL,
                   "pred_b_prev_share_at_mlp2_input": share["X"] <= SHARE_MAX,
                   "pred_c_relays_dominate": var["relays"] >= var["a2"],
                   "pred_d_write_prev_share": share["write"] >= share["nX"],
                   "pred_e_a2_cross_over_self": var["a2_cross"] >= CROSS_SELF_RATIO * var["a2_self"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_bigram_mlp2_result_v657", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
