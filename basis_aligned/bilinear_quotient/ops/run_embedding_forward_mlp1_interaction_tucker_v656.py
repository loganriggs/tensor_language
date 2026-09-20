"""Embedding-forward folding, rung 45 (v656): the Tucker profile of MLP-1's previous-token-dependent write — the layer-1 mirror of v613.

v613: MLP-0's cur x prev interaction write on the 1024 x 1024 unigram grid has mode ranks (t 381, s 104, o 764) at 90% energy and a 32 x 128 x 128
HOSVD core retains 0.20. v614: MLP-1's write is 20% previous-token-dependent (variance over s at fixed t). This rung takes MLP-1's exact bigram
write W(t, s) on the same grid, splits it into the s-mean part M(t) and the previous-token-dependent part R(t, s) = W - M (exact), and profiles R
as a 3-mode tensor with the v613 machinery (two chunked passes: mode Grams, then the projected core). Replay of the MLP-1 input vs the model's own
T=2 forward is kept as the instrument.
PREDICTIONS (scored as written; failures preserved)
    pred_a_replay                   manual MLP-1 normalised input matches the model's T=2 forward on the 512 most frequent bigrams, rel-L2 <= 1e-4 (instrument)
    pred_b_prev_part_energy_replays the prev-dependent share of MLP-1's write energy on the grid is within +-0.05 of v614's 0.204 (instrument)
    pred_c_s_mode_narrower_than_mlp0 R's 90% mode-s rank <= 104 (the previous-token content is at most as wide as at MLP-0). Prior: unsure
    pred_d_o_mode_wide              R's 90% mode-o rank >= 500 (the prev-dependent write spreads over most of the residual, as at MLP-0). Prior: likely
    pred_e_core_retention           a (32, 128, 128) core of R retains <= 0.35 (dense, as at MLP-0). Prior: unsure
PRICE (registered maximum): 13 batched T=1 table forwards + 1 real T=2 verification forward = 14; grid passes are weights x tables only. 0 backwards;
0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import embedding_forward_lib as EF

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_mlp1_interaction_tucker_v656_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_mlp1_interaction_tucker_v656_frames.pt"
ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
CANDIDATE_ID = "embedding_forward.mlp1_interaction_tucker_v656"
FORWARDS_MAX = 16
BATCH = 4096
N_VERIFY = 512
N_GRID = 1024
CH = 32
REPLAY_TOL, SHARE_REF, S_MAX, O_MIN, CORE_MAX = 1e-4, 0.204, 104, 500, 0.35
CORES = ((8, 32, 32), (16, 64, 64), (32, 128, 128), (64, 256, 256))
PREDICTIONS = {"pred_a_replay": "<= 1e-4", "pred_b_prev_part_energy_replays": "0.204 +-0.05", "pred_c_s_mode_narrower_than_mlp0": "<= 104",
               "pred_d_o_mode_wide": ">= 500", "pred_e_core_retention": "<= 0.35 at (32,128,128)"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "n_verify": N_VERIFY, "n_grid": N_GRID,
            "bars": {"replay_tol": REPLAY_TOL, "share_ref": SHARE_REF, "s_max": S_MAX, "o_min": O_MIN, "core_max": CORE_MAX}, "cores": [list(c) for c in CORES]}
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

        # ---- grid: MLP-1's write, split into s-mean and previous-token-dependent parts; Tucker profile of the latter -----------------
        p_uni, _, _ = EF.unigram_weights(ROWS, V, dev=dev)
        gen = torch.Generator(device=dev).manual_seed(6120)
        tg = torch.multinomial(p_uni.float(), 4096, replacement=False, generator=gen)[:N_GRID]
        sg = torch.multinomial(p_uni.float(), 4096, replacement=False, generator=gen)[:N_GRID]
        # pass 0: the s-mean of the write per t (exact), and total energy
        M = torch.zeros(N_GRID, D, dtype=torch.float64, device=dev); E_tot = 0.0
        for s0 in range(0, N_GRID, CH):
            sc = sg[s0:s0 + CH]; tt = tg.repeat_interleave(len(sc)); ss = sc.repeat(N_GRID)
            _, _, _, write = pair(tt, ss); w = write.double().view(N_GRID, len(sc), D); M += w.sum(1); E_tot += float(w.square().sum())
        M /= N_GRID; E_mean = float(M.square().sum()) * N_GRID; E_prev = E_tot - E_mean; prev_share = E_prev / E_tot
        print(f"MLP-1 write on the grid: total energy {E_tot:.4g}, prev-dependent share {prev_share:.4f} (v614: 0.204)")
        # pass 1: mode Grams of R = W - M
        Gt = torch.zeros(N_GRID, N_GRID, dtype=torch.float64, device=dev); Gs = torch.zeros_like(Gt); Go = torch.zeros(D, D, dtype=torch.float64, device=dev)
        for s0 in range(0, N_GRID, CH):
            sc = sg[s0:s0 + CH]; tt = tg.repeat_interleave(len(sc)); ss = sc.repeat(N_GRID)
            _, _, _, write = pair(tt, ss); R = write.double().view(N_GRID, len(sc), D) - M[:, None, :]
            Go += R.reshape(-1, D).T @ R.reshape(-1, D); Rt = R.reshape(N_GRID, -1); Gt += Rt @ Rt.T
        for t0_ in range(0, N_GRID, CH):
            tc = tg[t0_:t0_ + CH]; tt = tc.repeat_interleave(N_GRID); ss = sg.repeat(len(tc))
            _, _, _, write = pair(tt, ss); R = write.double().view(len(tc), N_GRID, D) - M[t0_:t0_ + CH, None, :]
            Rs = R.permute(1, 0, 2).reshape(N_GRID, -1); Gs += Rs @ Rs.T
        et, Ut = torch.linalg.eigh(Gt); es, Us = torch.linalg.eigh(Gs); eo, Uo = torch.linalg.eigh(Go)
        profile = {"energy": float(Gt.trace()), "trace_rel_spread": float((max(Gt.trace(), Gs.trace(), Go.trace()) - min(Gt.trace(), Gs.trace(), Go.trace())) / Gt.trace()),
                   "mode_t": EF.energy_ranks(et), "mode_s": EF.energy_ranks(es), "mode_o": EF.energy_ranks(eo), "top_t_evals_frac": (et.flip(0)[:8] / et.sum()).tolist()}
        print(f"prev-dependent write R: 90/95/99% ranks t {profile['mode_t']} s {profile['mode_s']} o {profile['mode_o']} | top t-evals {[round(v, 3) for v in profile['top_t_evals_frac'][:4]]}")
        rt, rs, ro = (max(c[i] for c in CORES) for i in range(3)); Pt, Ps, Po = Ut.flip(1)[:, :rt], Us.flip(1)[:, :rs], Uo.flip(1)[:, :ro]
        core = torch.zeros(rt, rs, ro, dtype=torch.float64, device=dev)
        for s0 in range(0, N_GRID, CH):
            sc = sg[s0:s0 + CH]; tt = tg.repeat_interleave(len(sc)); ss = sc.repeat(N_GRID)
            _, _, _, write = pair(tt, ss); R = write.double().view(N_GRID, len(sc), D) - M[:, None, :]
            x = R @ Po; x = torch.einsum("tp,tcq->pcq", Pt, x); core += torch.einsum("cs,pcq->psq", Ps[s0:s0 + CH], x)
        retained = {f"{c[0]}x{c[1]}x{c[2]}": float(core[:c[0], :c[1], :c[2]].square().sum() / profile["energy"]) for c in CORES}
        print("retained:", retained)
        disk_guard.guard_torch_save({"Pt": Pt.float().cpu(), "Ps": Ps.float().cpu(), "Po": Po.float().cpu(), "grid_t": tg.cpu(), "grid_s": sg.cpu()}, str(OUT_PT), "v656 frames")

    report = {"replay_rel_l2": replay, "prev_share_of_write_energy": prev_share, "profile": profile, "retained": retained}
    predictions = {"pred_a_replay": replay <= REPLAY_TOL, "pred_b_prev_part_energy_replays": abs(prev_share - SHARE_REF) <= 0.05,
                   "pred_c_s_mode_narrower_than_mlp0": profile["mode_s"]["0.9"] <= S_MAX, "pred_d_o_mode_wide": profile["mode_o"]["0.9"] >= O_MIN,
                   "pred_e_core_retention": retained["32x128x128"] <= CORE_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_mlp1_interaction_tucker_result_v656", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
