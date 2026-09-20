"""Embedding-forward folding, rung 24 (v632): CONTROLLED refits of the 19-head program — few parameters vs regularised tables, more fit rows.

v631: refitting the three tables per head against the loss took the 19-head program from +0.136 to +0.046 (step 50) and then overfit to +0.062
(1.6M vocabulary entries vs 246k fit tokens). Two controlled arms, same form and schedule (300 steps, batch 8, Adam 0.02 -> 0.002 cosine),
fit rows = skip80 (480) + skip11000 (192) = 672 x 512 (skip7000 stays the only test set; held-out curve every 50 steps; ENDPOINTS registered):
    (a) KAPPA-ONLY: kappa_h = kappa0_h (1 + c_h) plus one scalar gain g_h on A_h B_h per token head — 19 x 513 + 16 = 9.8k parameters; cannot memorise
    (b) REGULARISED TABLES: as v631 (a, b, c per head) with an L2 pull lambda ||a||^2 + ||b||^2 toward the closed-form values, lambda = 1e-2
Fit parameters: (a) 9,763; (b) 1,619,475 (tables only; model frozen). CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native CE within 0.002 of 3.13241 (instrument)
    pred_b_kappa_only_no_overfit (a): held-out CE added at every evaluation after step 50 is <= the step-50 value + 0.002 (no upturn). Prior: likely
    pred_c_kappa_only_under_bar  (a): endpoint <= 0.07. Prior: unsure
    pred_d_regularised_under_bar (b): endpoint <= 0.05. Prior: unsure
    pred_e_tables_beat_kappa_only (b) endpoint <= (a) endpoint (the token tables carry information beyond a per-head scale). Prior: unsure
PRICE (registered maximum): 2 arms x 300 training steps = 600 forwards + 600 BACKWARDS; evaluation 2 x 7 x 6 + 6 = 90 forwards; total 690 forwards,
600 backwards; fit parameters 9,763 (a) and 1,619,475 (b). Bars: forwards <= 700, backwards <= 600.

from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_controlled_refit_v632_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_controlled_refit_v632_tables.pt"
KER = ROOT / "circuits/followups/embedding_forward_kernel_bank_v629_kernels.pt"
TAB = {0: ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt", 1: ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt",
       2: ROOT / "circuits/followups/embedding_forward_layer2_filters_v628_tables.pt"}
FIT_ROWS = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.controlled_refit_v632"
FORWARDS_MAX, BACKWARDS_MAX = 700, 600
LAMBDA = 1e-2
STEPS, TBATCH, EBATCH, EVAL_EVERY = 300, 8, 32, 50
LR, LR_MIN = 0.02, 0.002
LAYERS = (0, 1, 2)
TOKEN_PROG = {0: (3, 4, 6, 7, 8), 1: (0, 1, 3, 5, 6, 7), 2: (0, 2, 3, 4, 8)}
KERNEL = {0: (), 1: (8,), 2: (6, 7)}
REPLAY_TOL, UPTURN_TOL, KAPPA_BAR, REG_BAR = 0.002, 0.002, 0.07, 0.05
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_kappa_only_no_overfit": "no upturn > 0.002", "pred_c_kappa_only_under_bar": "<= 0.07",
               "pred_d_regularised_under_bar": "<= 0.05", "pred_e_tables_beat_kappa_only": "(b) <= (a)"}


def main() -> None:
    n_full = sum(len(v) for v in TOKEN_PROG.values()) * (2 * 50304 + 513) + sum(len(v) for v in KERNEL.values()) * 513
    n_kappa = (sum(len(v) for v in TOKEN_PROG.values()) + sum(len(v) for v in KERNEL.values())) * 513 + sum(len(v) for v in TOKEN_PROG.values())
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": n_full,
            "fit_parameters_by_arm": {"kappa_only": n_kappa, "regularised": n_full}, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "steps": STEPS, "lr": [LR, LR_MIN], "lambda": LAMBDA, "bars": {"replay_tol": REPLAY_TOL, "upturn_tol": UPTURN_TOL, "kappa_bar": KAPPA_BAR, "reg_bar": REG_BAR}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    H = model.config.n_head; dev = "cuda"; blocks = model.transformer.h; forwards = 0; backwards = 0
    fit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    ker = torch.load(KER, map_location=dev)
    closed = {}
    for l, p in TAB.items():
        t_ = torch.load(p, map_location=dev)
        for h in TOKEN_PROG[l]:
            closed[(l, h)] = ("token", t_[f"head{h}_A"], t_[f"head{h}_B"], t_[f"head{h}_kappa"])
        for h in KERNEL[l]:
            closed[(l, h)] = ("kernel", None, None, ker[f"kbar_{l}_{h}"])
    state = {"idx": None, "on": False, "current": None}
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
                    if (l, h) in closed:
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
        params, var = [], {}
        for key, (kind, A0, B0, k0) in closed.items():
            c = torch.zeros_like(k0, requires_grad=True); params.append(c)
            if kind == "token":
                if name == "kappa_only":
                    g = torch.zeros((), device=dev, requires_grad=True); params.append(g); var[key] = (c, g, None, None)
                else:
                    a, b = torch.zeros_like(A0, requires_grad=True), torch.zeros_like(B0, requires_grad=True); params += [a, b]; var[key] = (c, None, a, b)
            else:
                var[key] = (c, None, None, None)

        def current(key):
            kind, A0, B0, k0 = closed[key]; c, g, a, b = var[key]; kappa = k0 * (1 + c)
            if kind == "kernel":
                return (kind, None, None, kappa)
            if name == "kappa_only":
                return (kind, A0 * torch.exp(g), B0, kappa)
            return (kind, A0 * torch.exp(a), B0 * torch.exp(b), kappa)

        state["current"] = current
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
                if name == "regularised":
                    loss = loss + LAMBDA * sum((v[2].square().sum() + v[3].square().sum()) for v in var.values() if v[2] is not None) / n_full
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            forwards += 1; backwards += 1; curve.append(float(loss))
            if step % EVAL_EVERY == 0:
                v_, fw = ce(True); forwards += fw; evals[step] = v_ - native
                print(f"[{name}] step {step}: train {sum(curve[-EVAL_EVERY:]) / EVAL_EVERY:.4f} | held-out CE added {evals[step]:+.4f} | lr {lr:.4f}")
        tables = {f"{l}.{h}": {"kind": current((l, h))[0], "kappa": current((l, h))[3].detach().cpu(), "A": None if current((l, h))[1] is None else current((l, h))[1].detach().cpu(),
                               "B": None if current((l, h))[2] is None else current((l, h))[2].detach().cpu()} for (l, h) in closed}
        return {"step0_added": step0 - native, "heldout_added_by_step": {str(k): v for k, v in evals.items()}, "train_curve": curve}, tables

    native, fw = ce(False); forwards += fw; print(f"native {native:.5f}")
    results, saved = {}, {}
    for arm in ("kappa_only", "regularised"):
        results[arm], saved[arm] = run_arm(arm)
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    pre.remove()
    disk_guard.guard_torch_save(saved, str(OUT_PT), "v632 refit tables")
    ka = results["kappa_only"]["heldout_added_by_step"]; rg = results["regularised"]["heldout_added_by_step"]
    end_k, end_r = ka[str(STEPS)], rg[str(STEPS)]
    print(f"endpoints: kappa-only {end_k:+.4f} | regularised tables {end_r:+.4f} | step-0 {ka['0']:+.4f}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_kappa_only_no_overfit": all(ka[str(s)] <= ka["50"] + UPTURN_TOL for s in range(100, STEPS + 1, EVAL_EVERY)),
                   "pred_c_kappa_only_under_bar": end_k <= KAPPA_BAR, "pred_d_regularised_under_bar": end_r <= REG_BAR, "pred_e_tables_beat_kappa_only": end_r <= end_k}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "embedding_forward_controlled_refit_result_v632", "candidate_id": CANDIDATE_ID, "plan": plan, "report": {"native": native, "arms": results},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
