#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_some_layer2_heads_separable pred_c_separable_programs_cheap pred_d_layer2_set_under_bar pred_e_composes_with_twelve
"""Embedding-forward folding, rung 20 (v628): the attention atlas at LAYER 2 — gated filters, running means, content heads.

Same construction as v623/v624 one block deeper: single-token block-2 query / key tables (token alone at position 0, exact through blocks 0-1
including attention-1's self term with the v1 token branch), the positional kernel and separability of every layer-2 head on the v612 grid,
the three-table program kappa(d) A(t) B(s) for every head with d=1 separable fraction >= 0.8 (rule registered, set data-dependent), and — new —
for EVERY head a kernel-only program (its real mean signed pattern per offset on 64 fit rows, no token dependence), which names running-mean
type heads (v626/v627) automatically. Edits on 192 x 512 skip7000: each separable head's token program, each head's kernel-only program, the
layer-2 program set (token programs of the separable heads), and that set on top of the twelve-head block-0/1 program (v627). CE ADDED.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays           native CE through the patched attentions within 0.002 of 3.13241 (instrument)
    pred_b_some_layer2_heads_separable >= 3 of the 9 layer-2 heads have single-token separable fraction >= 0.8 at d = 1. Prior: unsure
    pred_c_separable_programs_cheap for every separable head, min(token program, kernel-only) <= 0.02 (the head is token x position OR a pure
                                    positional averager). Prior: unsure — deeper heads may be genuinely contextual
    pred_d_layer2_set_under_bar     the layer-2 token-program set costs <= 0.05. Prior: unsure
    pred_e_composes_with_twelve     twelve + layer-2 set <= 1.5 x (twelve re-measured + layer-2 set). Prior: likely
PRICE (registered maximum): 13 batched T=1 table forwards; 2 capture forwards (64 fit rows); edits <= (1 + 9 + 9 + 1 + 2) x 6 = 132; total <= 147
forwards; 0 backwards; 0 loss fits. Bar <= 150.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard
import embedding_forward_lib as EF

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_layer2_filters_v628_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_layer2_filters_v628_tables.pt"
T0 = ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt"
T1 = ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt"
ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.layer2_filters_v628"
FORWARDS_MAX = 150
BATCH, EBATCH, N_CAP, Q_MIN = 4096, 32, 64, 8
N_SCORE = 4096
D_SVD = (1, 2, 4, 8, 16, 32, 64)
MAX_D = 512
LAYER = 2
L0_SET, L1_SET, MEAN_HEAD = (3, 4, 6, 7, 8), (0, 1, 3, 5, 6, 7), 8
SEP_RULE, REPLAY_TOL, MIN_SEP, CHEAP, SET_MAX, COMP = 0.8, 0.002, 3, 0.02, 0.05, 1.5
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_some_layer2_heads_separable": ">= 3", "pred_c_separable_programs_cheap": "min(program, kernel) <= 0.02 each",
               "pred_d_layer2_set_under_bar": "<= 0.05", "pred_e_composes_with_twelve": "<= 1.5 x sum"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "layer": LAYER, "d_svd": list(D_SVD), "sep_rule": SEP_RULE,
            "bars": {"replay_tol": REPLAY_TOL, "min_sep": MIN_SEP, "cheap": CHEAP, "set_max": SET_MAX, "comp": COMP}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; H = model.config.n_head; hd = D // H; V = model.config.vocab_size; dev = "cuda"
    blocks = model.transformer.h; b0, b1, b2 = blocks[0], blocks[1], blocks[2]; A0, A1, A2 = b0.attn, b1.attn, b2.attn; rms = EF.rms; forwards = 0
    fit = torch.load(ROWS, map_location="cpu").long(); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    t0_ = torch.load(T0, map_location=dev); t1_ = torch.load(T1, map_location=dev)
    tables = {0: {h: (t0_[f"head{h}_A"], t0_[f"head{h}_B"], t0_[f"head{h}_kappa"]) for h in L0_SET}, 1: {h: (t1_[f"head{h}_A"], t1_[f"head{h}_B"], t1_[f"head{h}_kappa"]) for h in L1_SET}, 2: {}}
    with torch.no_grad():
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))

        def rot(x, d):
            cos, sin = (d * inv_freq).cos().bfloat16().float(), (d * inv_freq).sin().bfloat16().float()
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], -1)

        def heads(lin, n):
            return lin(n).view(-1, H, hd)

        def self_attn(A, n, v1):
            q, k, q2, k2, v = heads(A.c_q, n), heads(A.c_k, n), heads(A.c_q2, n), heads(A.c_k2, n), heads(A.c_v, n)
            if v1 is None:
                v1 = v
            vm = (1 - A.lamb) * v + A.lamb * v1
            alpha = ((rms(q) * rms(k)).sum(-1) / hd) * ((rms(q2) * rms(k2)).sum(-1) / hd)
            return A.c_proj((alpha[..., None] * vm).reshape(-1, D)), v1

        def mlp(b, x):
            n = rms(x); return b.mlp.Down(b.mlp.Left(n) * b.mlp.Right(n)) + b.mlp.Down_bias

        E = model.transformer.wte.weight.detach().float()
        T = {k: torch.empty(V, H, hd, device=dev) for k in ("q", "q2", "k", "k2")}
        for s0 in range(0, V, BATCH):
            ids = torch.arange(s0, min(s0 + BATCH, V), device=dev)
            x0 = rms(E[ids]); x = x0; v1 = None
            for b in (b0, b1):
                live = b.lambdas[0] * x + b.lambdas[1] * x0; y, v1 = self_attn(b.attn, rms(live), v1); x = live + y; x = x + mlp(b, x)
            n2 = rms(b2.lambdas[0] * x + b2.lambdas[1] * x0)
            T["q"][ids], T["k"][ids] = rms(heads(A2.c_q, n2)), rms(heads(A2.c_k, n2)); T["q2"][ids], T["k2"][ids] = rms(heads(A2.c_q2, n2)), rms(heads(A2.c_k2, n2))
            forwards += 1
        p_uni, _, _ = EF.unigram_weights(ROWS, V, dev=dev)
        gen = torch.Generator(device=dev).manual_seed(6120)
        ts = torch.multinomial(p_uni.float(), N_SCORE, replacement=False, generator=gen); ss = torch.multinomial(p_uni.float(), N_SCORE, replacement=False, generator=gen)

        def pattern(h, t_ids, s_ids, d):
            qr, q2r = rot(T["q"][t_ids, h], d), rot(T["q2"][t_ids, h], d)
            return ((qr @ T["k"][s_ids, h].T) / hd) * ((q2r @ T["k2"][s_ids, h].T) / hd)

        kernels, grid = {}, {}
        for h in range(H):
            kernels[h] = {}
            for d in D_SVD:
                Sd = pattern(h, ts, ss, d); sv = torch.linalg.svdvals(Sd); e = sv.square(); c = e.cumsum(0) / e.sum()
                kernels[h][d] = {"rms": float(Sd.square().mean().sqrt()), "sep": float(e[0] / e.sum()), "rank90": int((c < 0.9).sum()) + 1}
            print(f"head {LAYER}.{h}: " + " ".join(f"d{d}={kernels[h][d]['rms']:.4f}/{kernels[h][d]['sep']:.2f}/r{kernels[h][d]['rank90']}" for d in D_SVD))
        separable = [h for h in range(H) if kernels[h][1]["sep"] >= SEP_RULE]; print("separable set:", separable)
        for h in separable:
            S1 = pattern(h, ts, ss, 1); U1, _, Vh1 = torch.linalg.svd(S1, full_matrices=False); u1, v1 = U1[:, 0], Vh1[0]
            A = torch.empty(V, device=dev); Bt = torch.empty(V, device=dev)
            for s0 in range(0, V, BATCH):
                ids = torch.arange(s0, min(s0 + BATCH, V), device=dev); A[ids] = pattern(h, ids, ss, 1) @ v1; Bt[ids] = u1 @ pattern(h, ts, ids, 1)
            AB = torch.outer(A[ts], Bt[ss]); nAB = float(AB.square().sum()); kappa = torch.zeros(MAX_D + 1, device=dev); resid = {}
            for d in range(1, MAX_D + 1):
                Sd = pattern(h, ts, ss, d); k_ = float((Sd * AB).sum() / nAB); kappa[d] = k_
                if d in D_SVD:
                    resid[d] = float((Sd - k_ * AB).square().sum() / Sd.square().sum())
            tables[2][h] = (A, Bt, kappa); grid[h] = {"resid": resid, "kappa_1_16": kappa[1:17].tolist()}
            print(f"  head {LAYER}.{h} program residual " + " ".join(f"d{d}={resid[d]:.3f}" for d in D_SVD) + f" | kappa(1..8) {[round(x, 4) for x in kappa[1:9].tolist()]}")

        # ---- patched attentions for layers 0, 1, 2; capture layer-2 real patterns ------------------------------------------------
        state = {"idx": None, "heads": {0: (), 1: (), 2: ()}, "mean18": False, "kernel_heads": (), "capture": None, "kbar": {}}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        natives = {l: blocks[l].attn.squared_attention for l in (0, 1, 2)}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (causal & (dmat > 0))[None]
                if l == 2 and state["capture"] is not None:
                    state["capture"].append(pat.detach().clone())
                idx = state["idx"]
                for h in state["heads"][l]:
                    A, Bt, kappa = tables[l][h]; prog = kappa[dmat][None] * A[idx][:, :, None] * Bt[idx][:, None, :]
                    pat[:, h] = torch.where(off, prog, pat[:, h])
                if l == 1 and state["mean18"]:
                    pat[:, MEAN_HEAD] = torch.where(off, (-1.0 / pos.clamp_min(1).float())[None, :, None].expand(Bn, Tn, Tn), pat[:, MEAN_HEAD])
                if l == 2:
                    for h in state["kernel_heads"]:
                        pat[:, h] = torch.where(off, state["kbar"][h][dmat][None].expand(Bn, -1, -1), pat[:, h])
                return torch.einsum("bhqk,bkhd->bhqd", pat, v)
            return patched

        for l in (0, 1, 2):
            blocks[l].attn.squared_attention = make_patched(l)
        state["capture"] = []
        for s in range(0, N_CAP, EBATCH):
            idx = fit[s:s + EBATCH, :-1].to(dev); model(idx, fit[s:s + EBATCH, 1:].to(dev)); forwards += 1
        reals = torch.cat(state["capture"]); state["capture"] = None                                                   # [N, H, T, T]
        Tn = reals.shape[-1]; pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0)
        kbar = {}; rowsum = {}
        for h in range(H):
            kb = torch.zeros(Tn + 1, device=dev)
            for d in range(1, Tn):
                m = (dmat == d) & (pos >= Q_MIN)[:, None]; kb[d] = float(reals[:, h][:, m].mean())
            kbar[h] = kb; rowsum[h] = float((reals[:, h] * (dmat > 0)[None]).sum(-1)[:, Q_MIN:].mean())
        state["kbar"] = kbar
        print("real kernels kbar(1..6) and mean row sum: " + " | ".join(f"{LAYER}.{h}: {[round(x, 4) for x in kbar[h][1:7].tolist()]} sum {rowsum[h]:+.3f}" for h in range(H)))

        def ce(cfg):
            state.update(cfg); total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            state.update({"heads": {0: (), 1: (), 2: ()}, "mean18": False, "kernel_heads": ()})
            return total / n, fw

        base = {"heads": {0: (), 1: (), 2: ()}, "mean18": False, "kernel_heads": ()}
        native, fw = ce(dict(base)); forwards += fw; e = {}
        for h in separable:
            v_, fw = ce({"heads": {0: (), 1: (), 2: (h,)}}); forwards += fw; e[f"prog|{h}"] = v_ - native
        for h in range(H):
            v_, fw = ce({"kernel_heads": (h,)}); forwards += fw; e[f"kernel|{h}"] = v_ - native
        if separable:
            v_, fw = ce({"heads": {0: (), 1: (), 2: tuple(separable)}}); forwards += fw; e["set"] = v_ - native
        v_, fw = ce({"heads": {0: L0_SET, 1: L1_SET, 2: ()}, "mean18": True}); forwards += fw; e["twelve"] = v_ - native
        if separable:
            v_, fw = ce({"heads": {0: L0_SET, 1: L1_SET, 2: tuple(separable)}, "mean18": True}); forwards += fw; e["twelve_plus_set"] = v_ - native
        for l in (0, 1, 2):
            blocks[l].attn.squared_attention = natives[l]
        pre.remove()
        print(f"native {native:.5f} | " + " ".join(f"{k}={v:+.4f}" for k, v in e.items()))
        disk_guard.guard_torch_save({f"head{h}_{n}": t.cpu() for h, (A, Bt, kappa) in tables[2].items() for n, t in (("A", A), ("B", Bt), ("kappa", kappa))} | {f"kbar{h}": kbar[h].cpu() for h in range(H)} | {"kernels": kernels}, str(OUT_PT), "v628 tables")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_some_layer2_heads_separable": len(separable) >= MIN_SEP,
                   "pred_c_separable_programs_cheap": bool(separable) and all(min(e[f"prog|{h}"], e[f"kernel|{h}"]) <= CHEAP for h in separable),
                   "pred_d_layer2_set_under_bar": bool(separable) and e["set"] <= SET_MAX,
                   "pred_e_composes_with_twelve": bool(separable) and e["twelve_plus_set"] <= COMP * (e["twelve"] + e["set"])}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_layer2_filters_result_v628", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": e, "separable": separable, "kernels": {str(h): {str(d): v for d, v in kd.items()} for h, kd in kernels.items()},
                                          "grid": {str(h): g for h, g in grid.items()}, "real_kbar_1_16": {str(h): kbar[h][1:17].tolist() for h in range(H)}, "real_rowsum": {str(h): rowsum[h] for h in range(H)}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
