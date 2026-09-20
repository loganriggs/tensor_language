#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_some_layer1_heads_separable pred_c_separable_edit_per_head pred_d_separable_set_edit pred_e_best_grid_fit
"""Embedding-forward folding, rung 16 (v624): gated positional filters one layer deeper — the nine LAYER-1 heads from single-token tables.

v622/v623: five layer-0 heads are gated positional filters kappa(d) A(t) B(s), installable at +0.0094 nats. At layer 1 the queries and keys read
x1, which in context carries attention-0's mixing (24% of MLP-0's input, v612) — but the SINGLE-TOKEN block-1 tables (v614's x1s -> n1 -> q/k,
exact for a token alone) still define a token x position pattern S_h^(d)(t,s) for every head. This rung (i) measures its kernel and separability
on the v612 grid for all nine layer-1 heads at d in {1, 2, 4, 8, 16, 32, 64}; (ii) for every head whose d=1 separable fraction >= 0.8 (rule
registered, set data-dependent) builds the same three-table program (A, B over the vocabulary from the d=1 rank-one pair; kappa(d), d = 1..512)
and installs it inside block 1's squared attention (native diagonal, native values incl. the v1 token branch, native output projection), priced in
CE on the 192 x 512 skip7000 rows: each separable head alone and the whole separable set. The edit measures how much of a layer-1 head's pattern is
token x position. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays            native CE through the patched block-1 attention within 0.002 of 3.13241 (instrument)
    pred_b_some_layer1_heads_separable at least 2 of the 9 layer-1 heads have single-token separable fraction >= 0.8 at d = 1. Prior: unsure
    pred_c_separable_edit_per_head   every separable head's pattern replacement costs <= 0.02 nats. Prior: unsure (context-dependence of q/k)
    pred_d_separable_set_edit        the whole separable set replaced at once costs <= 0.05 nats. Prior: unsure
    pred_e_best_grid_fit             the best separable head's grid residual at d = 1 is <= 0.2 (as good as layer 0's 0.06-0.12). Prior: unsure
    (if no head is separable, pred_c / pred_d / pred_e score False by construction)
PRICE (registered maximum): 13 batched T=1 table forwards (blocks 0-1 at position 0); 9 x 7 grid SVDs; edits <= (1 + 9 + 1) configs x 6 batches
= 66 forwards; total <= 79 forwards; 0 backwards; 0 loss fits. Bar <= 85.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import embedding_forward_lib as EF

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt"
ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.layer1_filters_v624"
FORWARDS_MAX = 85
BATCH, EBATCH = 4096, 32
N_SCORE = 4096
D_SVD = (1, 2, 4, 8, 16, 32, 64)
MAX_D = 512
SEP_RULE, REPLAY_TOL, MIN_SEP_HEADS, PER_HEAD_MAX, SET_MAX, FIT_MAX = 0.8, 0.002, 2, 0.02, 0.05, 0.2
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_some_layer1_heads_separable": ">= 2 heads sep >= 0.8 at d=1", "pred_c_separable_edit_per_head": "<= 0.02 each",
               "pred_d_separable_set_edit": "<= 0.05", "pred_e_best_grid_fit": "<= 0.2"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "d_svd": list(D_SVD), "sep_rule": SEP_RULE,
            "bars": {"replay_tol": REPLAY_TOL, "min_sep_heads": MIN_SEP_HEADS, "per_head_max": PER_HEAD_MAX, "set_max": SET_MAX, "fit_max": FIT_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; H = model.config.n_head; hd = D // H; V = model.config.vocab_size; dev = "cuda"
    b0, b1 = model.transformer.h[0], model.transformer.h[1]; A0, A1 = b0.attn, b1.attn; rms = EF.rms; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    with torch.no_grad():
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))

        def rot(x, d):
            cos, sin = (d * inv_freq).cos().bfloat16().float(), (d * inv_freq).sin().bfloat16().float()
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], -1)

        # ---- single-token block-1 query / key tables (token alone at position 0, exact) ------------------------------------------
        E = model.transformer.wte.weight.detach().float(); lam0, lam1 = b0.lambdas.detach().float(), b1.lambdas.detach().float()
        T = {k: torch.empty(V, H, hd, device=dev) for k in ("q", "q2", "k", "k2")}
        for s0 in range(0, V, BATCH):
            ids = torch.arange(s0, min(s0 + BATCH, V), device=dev)
            x0 = rms(E[ids]); live = lam0[0] * x0 + lam0[1] * x0; n = rms(live)
            q, k, q2, k2, v = (A0.c_q(n).view(-1, H, hd), A0.c_k(n).view(-1, H, hd), A0.c_q2(n).view(-1, H, hd), A0.c_k2(n).view(-1, H, hd), A0.c_v(n).view(-1, H, hd))
            qh, kh, q2h, k2h = rms(q), rms(k), rms(q2), rms(k2)
            alpha0 = ((qh * kh).sum(-1) / hd) * ((q2h * k2h).sum(-1) / hd)
            xa = live + A0.c_proj((alpha0[..., None] * v).reshape(-1, D)); n0 = rms(xa)
            x1s = xa + b0.mlp.Down(b0.mlp.Left(n0) * b0.mlp.Right(n0)) + b0.mlp.Down_bias
            n1 = rms(lam1[0] * x1s + lam1[1] * x0)
            T["q"][ids], T["k"][ids] = rms(A1.c_q(n1).view(-1, H, hd)), rms(A1.c_k(n1).view(-1, H, hd))
            T["q2"][ids], T["k2"][ids] = rms(A1.c_q2(n1).view(-1, H, hd)), rms(A1.c_k2(n1).view(-1, H, hd))
            forwards += 1
        p_uni, _, _ = EF.unigram_weights(ROWS, V, dev=dev)
        gen = torch.Generator(device=dev).manual_seed(6120)
        ts = torch.multinomial(p_uni.float(), N_SCORE, replacement=False, generator=gen); ss = torch.multinomial(p_uni.float(), N_SCORE, replacement=False, generator=gen)

        def pattern(h, t_ids, s_ids, d):
            qr, q2r = rot(T["q"][t_ids, h], d), rot(T["q2"][t_ids, h], d)
            return ((qr @ T["k"][s_ids, h].T) / hd) * ((q2r @ T["k2"][s_ids, h].T) / hd)

        kernels, tables, grid = {}, {}, {}
        for h in range(H):
            kernels[h] = {}
            for d in D_SVD:
                Sd = pattern(h, ts, ss, d); sv = torch.linalg.svdvals(Sd); e = sv.square(); c = e.cumsum(0) / e.sum()
                kernels[h][d] = {"rms": float(Sd.square().mean().sqrt()), "sep": float(e[0] / e.sum()), "rank90": int((c < 0.9).sum()) + 1}
            print(f"head 1.{h}: " + " ".join(f"d{d}={kernels[h][d]['rms']:.4f}/{kernels[h][d]['sep']:.2f}/r{kernels[h][d]['rank90']}" for d in D_SVD))
        separable = [h for h in range(H) if kernels[h][1]["sep"] >= SEP_RULE]
        print("separable set (rule sep(d=1) >= 0.8):", separable)
        for h in separable:
            S1 = pattern(h, ts, ss, 1); U1, sv1, Vh1 = torch.linalg.svd(S1, full_matrices=False); u1, v1 = U1[:, 0], Vh1[0]
            A = torch.empty(V, device=dev); Bt = torch.empty(V, device=dev)
            for s0 in range(0, V, BATCH):
                ids = torch.arange(s0, min(s0 + BATCH, V), device=dev)
                A[ids] = pattern(h, ids, ss, 1) @ v1; Bt[ids] = u1 @ pattern(h, ts, ids, 1)
            AB = torch.outer(A[ts], Bt[ss]); nAB = float(AB.square().sum()); kappa = torch.zeros(MAX_D + 1, device=dev); resid = {}
            for d in range(1, MAX_D + 1):
                Sd = pattern(h, ts, ss, d); k_ = float((Sd * AB).sum() / nAB); kappa[d] = k_
                if d in D_SVD:
                    resid[d] = float((Sd - k_ * AB).square().sum() / Sd.square().sum())
            tables[h] = (A, Bt, kappa); grid[h] = {"resid": resid, "kappa_1_16": kappa[1:17].tolist()}
            print(f"  head 1.{h} program: residual " + " ".join(f"d{d}={resid[d]:.3f}" for d in D_SVD) + f" | kappa(1..8) {[round(x, 4) for x in kappa[1:9].tolist()]}")

        state = {"idx": None, "heads": ()}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        native_sq = A1.squared_attention

        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
            if state["heads"]:
                idx = state["idx"]; pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
                for h in state["heads"]:
                    A, Bt, kappa = tables[h]
                    prog = kappa[dmat][None] * A[idx][:, :, None] * Bt[idx][:, None, :]
                    pat[:, h] = torch.where(off[None], prog, pat[:, h])
            return torch.einsum("bhqk,bkhd->bhqd", pat, v)

        A1.squared_attention = patched

        def ce(heads):
            state["heads"] = heads; total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            return total / n, fw

        native, fw = ce(()); forwards += fw; print(f"native CE (patched block-1 attention, no override) {native:.5f}")
        edits = {}
        for h in separable:
            v_, fw = ce((h,)); forwards += fw; edits[f"head1.{h}"] = v_ - native
        if separable:
            v_, fw = ce(tuple(separable)); forwards += fw; edits["set"] = v_ - native
        A1.squared_attention = native_sq; pre.remove()
        print("CE added:", {k: round(v, 4) for k, v in edits.items()})
        disk_guard.guard_torch_save({f"head{h}_{n}": t.cpu() for h, (A, Bt, kappa) in tables.items() for n, t in (("A", A), ("B", Bt), ("kappa", kappa))} | {"kernels": kernels}, str(OUT_PT), "v624 tables")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_some_layer1_heads_separable": len(separable) >= MIN_SEP_HEADS,
                   "pred_c_separable_edit_per_head": bool(separable) and all(edits[f"head1.{h}"] <= PER_HEAD_MAX for h in separable),
                   "pred_d_separable_set_edit": bool(separable) and edits["set"] <= SET_MAX,
                   "pred_e_best_grid_fit": bool(separable) and min(grid[h]["resid"][1] for h in separable) <= FIT_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_layer1_filters_result_v624", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": edits, "separable": separable, "kernels": {str(h): {str(d): v for d, v in kd.items()} for h, kd in kernels.items()},
                                          "grid": {str(h): {"resid": {str(d): v for d, v in g["resid"].items()}, "kappa_1_16": g["kappa_1_16"]} for h, g in grid.items()}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
