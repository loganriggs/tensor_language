"""Embedding-forward folding, rung 25 (v633): no tables at all — fitted FIXED KERNELS for the 19 positional heads, and for all 27 early heads.

v632: refitting only the kernel shapes and one gain per head (9.8k numbers) prices the 19-head program at +0.050, and refitting 1.6M vocabulary
entries does no better. This rung drops the token tables entirely: pattern_h(i, j) = kappa_h(i - j) for j < i, kappa initialised at the real
mean kernel of v629 and fitted jointly against the model's CE (model frozen; same schedule: 300 steps, batch 8, Adam 0.02 -> 0.002 cosine; fit rows
skip80 + skip11000 = 672 x 512; held-out 192 x 512 skip7000 every 50 steps; endpoints registered):
    (c) POSITIONAL-19: the 19 positional heads (content heads native) — 19 x 513 = 9,747 numbers
    (d) ALL-27: every head of blocks 0-2 — 27 x 513 = 13,851 numbers (the whole early attention pattern bank as fitted positional priors)
CE ADDED, lower is better. Closed-form comparison points (v630 / v629): positional-19 kernel-only 0.231; all-27 0.355.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays            native CE within 0.002 of 3.13241 (instrument)
    pred_b_no_overfit                both arms: held-out at every evaluation after step 50 <= the step-50 value + 0.002. Prior: likely
    pred_c_positional_kernels_under_bar (c) endpoint <= 0.08 (token gating is worth < 0.03 once kernels are fitted). Prior: unsure
    pred_d_all27_under_bar           (d) endpoint <= 0.15. Prior: unsure
    pred_e_content_heads_cost        (d) endpoint - (c) endpoint >= 0.03 (the content heads' joint content dependence survives fitting). Prior: unsure
PRICE (registered maximum): 2 arms x 300 steps = 600 forwards + 600 BACKWARDS; evaluation 2 x 7 x 6 + 6 = 90; total 690 forwards, 600 backwards;
fit parameters 9,747 (c) and 13,851 (d). Bars: forwards <= 700, backwards <= 600.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_kernel_refit_v633_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_kernel_refit_v633_kernels.pt"
KER = ROOT / "circuits/followups/embedding_forward_kernel_bank_v629_kernels.pt"
TAB = {0: ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt", 1: ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt",
       2: ROOT / "circuits/followups/embedding_forward_layer2_filters_v628_tables.pt"}
FIT_ROWS = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.kernel_refit_v633"
FORWARDS_MAX, BACKWARDS_MAX = 700, 600
LAMBDA = 1e-2
STEPS, TBATCH, EBATCH, EVAL_EVERY = 300, 8, 32, 50
LR, LR_MIN = 0.02, 0.002
LAYERS = (0, 1, 2)
TOKEN_PROG = {0: (3, 4, 6, 7, 8), 1: (0, 1, 3, 5, 6, 7), 2: (0, 2, 3, 4, 8)}
KERNEL = {0: (), 1: (8,), 2: (6, 7)}
REPLAY_TOL, UPTURN_TOL, POS_BAR, ALL_BAR, CONTENT_MIN = 0.002, 0.002, 0.08, 0.15, 0.03
POSITIONAL = {0: (3, 4, 6, 7, 8), 1: (0, 1, 3, 5, 6, 7, 8), 2: (0, 2, 3, 4, 6, 7, 8)}
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_no_overfit": "no upturn > 0.002 x 2 arms", "pred_c_positional_kernels_under_bar": "<= 0.08",
               "pred_d_all27_under_bar": "<= 0.15", "pred_e_content_heads_cost": "(d) - (c) >= 0.03"}


def main() -> None:
    n_pos = sum(len(v) for v in POSITIONAL.values()) * 513; n_all = 27 * 513
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": n_all,
            "fit_parameters_by_arm": {"positional19": n_pos, "all27": n_all}, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "steps": STEPS, "lr": [LR, LR_MIN], "bars": {"replay_tol": REPLAY_TOL, "upturn_tol": UPTURN_TOL, "pos_bar": POS_BAR, "all_bar": ALL_BAR, "content_min": CONTENT_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    H = model.config.n_head; dev = "cuda"; blocks = model.transformer.h; forwards = 0; backwards = 0
    fit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    ker = torch.load(KER, map_location=dev)
    closed = {(l, h): ("kernel", None, None, ker[f"kbar_{l}_{h}"]) for l in LAYERS for h in range(H)}
    state = {"idx": None, "on": False, "current": None, "active": set()}
    pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
    natives = {l: blocks[l].attn.squared_attention for l in LAYERS}

    def make_patched(l):
        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat = pat.masked_fill(~causal, 0.0)
            if state["on"]:
                pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (causal & (dmat > 0))[None]; idx = state["idx"]
                cols = []
                for h in range(Hn):
                    if (l, h) in state["active"]:
                        kind, A, Bt, kappa = state["current"]((l, h))
                        prog = kappa[dmat][None].expand(Bn, -1, -1) if kind == "kernel" else kappa[dmat][None] * A[idx][:, :, None] * Bt[idx][:, None, :]
                        cols.append(torch.where(off, prog, pat[:, h]))
                    else:
                        cols.append(pat[:, h])
                pat = torch.stack(cols, 1)
            return torch.einsum("bhqk,bkhd->bhqd", pat, v)
        return patched

    for l in LAYERS:
        blocks[l].attn.squared_attention = make_patched(l)

    def ce(on):
        state["on"] = on; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        return total / n, fw

    def run_arm(name):
        nonlocal forwards, backwards
        active = {(l, h) for l in LAYERS for h in (POSITIONAL[l] if name == "positional19" else range(H))}
        params, var = [], {}
        for key in active:
            c = torch.zeros_like(closed[key][3], requires_grad=True); params.append(c); var[key] = c

        def current(key):
            return ("kernel", None, None, closed[key][3] * (1 + var[key]))

        state["current"] = current; state["active"] = active
        step0, fw = ce(True); forwards += fw
        opt = torch.optim.Adam(params, lr=LR); curve, evals = [], {0: step0 - native}
        gen = torch.Generator().manual_seed(632); order = torch.randperm(fit.shape[0], generator=gen)
        for step in range(1, STEPS + 1):
            lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * (step - 1) / STEPS))
            for gr in opt.param_groups:
                gr["lr"] = lr
            s0 = ((step - 1) * TBATCH) % fit.shape[0]; sel = order[s0:s0 + TBATCH]
            if len(sel) < TBATCH:
                sel = order[:TBATCH]
            idx = fit[sel, :-1].to(dev); tgt = fit[sel, 1:].to(dev); state["on"] = True
            with torch.enable_grad():
                loss = model(idx, tgt)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            forwards += 1; backwards += 1; curve.append(float(loss))
            if step % EVAL_EVERY == 0:
                v_, fw = ce(True); forwards += fw; evals[step] = v_ - native
                print(f"[{name}] step {step}: train {sum(curve[-EVAL_EVERY:]) / EVAL_EVERY:.4f} | held-out CE added {evals[step]:+.4f} | lr {lr:.4f}")
        tables = {f"{l}.{h}": current((l, h))[3].detach().cpu() for (l, h) in active}
        return {"step0_added": step0 - native, "heldout_added_by_step": {str(k): v for k, v in evals.items()}, "train_curve": curve}, tables

    native, fw = ce(False); forwards += fw; print(f"native {native:.5f}")
    results, saved = {}, {}
    for arm in ("positional19", "all27"):
        results[arm], saved[arm] = run_arm(arm)
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    pre.remove()
    disk_guard.guard_torch_save(saved, str(OUT_PT), "v633 refit kernels")
    pa = results["positional19"]["heldout_added_by_step"]; al = results["all27"]["heldout_added_by_step"]
    end_p, end_a = pa[str(STEPS)], al[str(STEPS)]
    print(f"endpoints: positional-19 kernels {end_p:+.4f} (step-0 {pa['0']:+.4f}) | all-27 kernels {end_a:+.4f} (step-0 {al['0']:+.4f})")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_no_overfit": all(x[str(s)] <= x["50"] + UPTURN_TOL for x in (pa, al) for s in range(100, STEPS + 1, EVAL_EVERY)),
                   "pred_c_positional_kernels_under_bar": end_p <= POS_BAR, "pred_d_all27_under_bar": end_a <= ALL_BAR, "pred_e_content_heads_cost": end_a - end_p >= CONTENT_MIN}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "embedding_forward_kernel_refit_result_v633", "candidate_id": CANDIDATE_ID, "plan": plan, "report": {"native": native, "arms": results},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
