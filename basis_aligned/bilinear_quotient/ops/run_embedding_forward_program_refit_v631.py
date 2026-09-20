#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_refit_converges pred_c_refit_halves_union pred_d_refit_under_bar pred_e_kernels_keep_shape
"""Embedding-forward folding, rung 23 (v631): FIT — is the gated-positional-filter FORM the limit, or its closed-form parameters?

v630: the 19-head best program (token-gated three-table programs for 16 heads, fixed kernels for 1.8 / 2.6 / 2.7; content heads native) costs
+0.136 jointly although every head costs <= 0.009 alone. Here the same form is kept and its tables are refit JOINTLY against the model's own CE:
    pattern_h(i, j) = kappa_h(i - j) A_h(tok_i) B_h(tok_j)   (16 heads)      pattern_h(i, j) = kappa_h(i - j)   (3 heads)
with A = A0 exp(a), B = B0 exp(b), kappa = kappa0 (1 + c) (a, b, c initialised at 0, so step 0 IS the closed-form program), the rest of the
model frozen, Adam on the 480 x 512 skip80 fit rows (batch 8 rows, 300 steps = 5 epochs, lr 0.02 cosine to 0.002), evaluated on the held-out
192 x 512 skip7000 rows every 50 steps. Fit parameters: 16 x (2V + 513) + 3 x 513 = 1.62M table values (no model weight changes). Convergence
criterion (registered): mean training CE over steps 251-300 improves on steps 201-250 by < 1e-3. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays   native CE within 0.002 of 3.13241 (instrument)
    pred_b_refit_converges  the registered convergence criterion is met within the 300 steps. Prior: likely
    pred_c_refit_halves_union held-out CE added of the refit program <= 0.5 x the step-0 (closed-form) value re-measured here (v630: 0.136). Prior: unsure
    pred_d_refit_under_bar  held-out CE added of the refit program <= 0.05 (the form suffices; only the parameters were wrong). Prior: unsure
    pred_e_kernels_keep_shape for the five layer-0 token-program heads, the refit kappa keeps the closed-form shape: cosine(kappa_refit[1:33], kappa0[1:33]) >= 0.9 for every one. Prior: likely
PRICE (registered maximum): 300 training steps (batch 8 rows) = 300 forwards + 300 BACKWARDS; evaluation 7 x 6 = 42 forwards + native 6; total 348
forwards, 300 backwards; 1.62M fit parameters (tables only). Bars: forwards <= 360, backwards <= 300.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_program_refit_v631_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_program_refit_v631_tables.pt"
KER = ROOT / "circuits/followups/embedding_forward_kernel_bank_v629_kernels.pt"
TAB = {0: ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt", 1: ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt",
       2: ROOT / "circuits/followups/embedding_forward_layer2_filters_v628_tables.pt"}
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.program_refit_v631"
FORWARDS_MAX, BACKWARDS_MAX = 360, 300
STEPS, TBATCH, EBATCH, EVAL_EVERY = 300, 8, 32, 50
LR, LR_MIN = 0.02, 0.002
LAYERS = (0, 1, 2)
TOKEN_PROG = {0: (3, 4, 6, 7, 8), 1: (0, 1, 3, 5, 6, 7), 2: (0, 2, 3, 4, 8)}
KERNEL = {0: (), 1: (8,), 2: (6, 7)}
REPLAY_TOL, CONV_TOL, HALF, BAR, COS_MIN = 0.002, 1e-3, 0.5, 0.05, 0.9
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_refit_converges": "< 1e-3 over the last 50 steps", "pred_c_refit_halves_union": "<= 0.5 x step-0",
               "pred_d_refit_under_bar": "<= 0.05", "pred_e_kernels_keep_shape": "cos >= 0.9 x 5 layer-0 heads"}


def main() -> None:
    n_fit = sum(len(v) for v in TOKEN_PROG.values()) * (2 * 50304 + 513) + sum(len(v) for v in KERNEL.values()) * 513
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": n_fit,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "steps": STEPS, "lr": [LR, LR_MIN],
            "bars": {"replay_tol": REPLAY_TOL, "conv_tol": CONV_TOL, "half": HALF, "bar": BAR, "cos_min": COS_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    H = model.config.n_head; dev = "cuda"; blocks = model.transformer.h; forwards = 0; backwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long(); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    ker = torch.load(KER, map_location=dev)
    base, params = {}, []
    for l, p in TAB.items():
        t_ = torch.load(p, map_location=dev)
        for h in TOKEN_PROG[l]:
            A0, B0, k0 = t_[f"head{h}_A"], t_[f"head{h}_B"], t_[f"head{h}_kappa"]
            a, b, c = (torch.zeros_like(A0, requires_grad=True), torch.zeros_like(B0, requires_grad=True), torch.zeros_like(k0, requires_grad=True))
            base[(l, h)] = ("token", A0, B0, k0, a, b, c); params += [a, b, c]
        for h in KERNEL[l]:
            k0 = ker[f"kbar_{l}_{h}"]; c = torch.zeros_like(k0, requires_grad=True); base[(l, h)] = ("kernel", None, None, k0, None, None, c); params.append(c)
    state = {"idx": None, "on": False}
    pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
    natives = {l: blocks[l].attn.squared_attention for l in LAYERS}

    def current(key):
        kind, A0, B0, k0, a, b, c = base[key]
        kappa = k0 * (1 + c)
        return (kind, None if A0 is None else A0 * torch.exp(a), None if B0 is None else B0 * torch.exp(b), kappa)

    def make_patched(l):
        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat = pat.masked_fill(~causal, 0.0)
            if state["on"]:
                pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (causal & (dmat > 0))[None]; idx = state["idx"]
                cols = []
                for h in range(Hn):
                    if (l, h) in base:
                        kind, A, Bt, kappa = current((l, h))
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

    native, fw = ce(False); forwards += fw
    step0, fw = ce(True); forwards += fw
    print(f"native {native:.5f} | step-0 (closed-form) program CE added {step0 - native:+.4f}")
    opt = torch.optim.Adam(params, lr=LR)
    curve, evals = [], {0: step0 - native}
    gen = torch.Generator().manual_seed(631); order = torch.randperm(fit.shape[0], generator=gen)
    for step in range(1, STEPS + 1):
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * (step - 1) / STEPS))
        for g in opt.param_groups:
            g["lr"] = lr
        sel = order[((step - 1) * TBATCH) % fit.shape[0]:((step - 1) * TBATCH) % fit.shape[0] + TBATCH]
        if len(sel) < TBATCH:
            sel = order[:TBATCH]
        idx = fit[sel, :-1].to(dev); tgt = fit[sel, 1:].to(dev)
        state["on"] = True
        with torch.enable_grad():
            loss = model(idx, tgt); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        forwards += 1; backwards += 1; curve.append(float(loss))
        if step % EVAL_EVERY == 0:
            v_, fw = ce(True); forwards += fw; evals[step] = v_ - native
            print(f"step {step}: train CE (last {EVAL_EVERY} mean) {sum(curve[-EVAL_EVERY:]) / EVAL_EVERY:.4f} | held-out CE added {evals[step]:+.4f} | lr {lr:.4f}")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    pre.remove()
    conv = (sum(curve[200:250]) / 50) - (sum(curve[250:300]) / 50)
    final = evals[STEPS]
    with torch.no_grad():
        cos = {}
        for h in TOKEN_PROG[0]:
            _, _, _, k0, _, _, c = base[(0, h)]; kr = (k0 * (1 + c))[1:33]; cos[h] = float(torch.nn.functional.cosine_similarity(kr, k0[1:33], dim=0))
        disk_guard.guard_torch_save({f"{l}.{h}": {"kind": current((l, h))[0], "A": None if current((l, h))[1] is None else current((l, h))[1].cpu(),
                                                  "B": None if current((l, h))[2] is None else current((l, h))[2].cpu(), "kappa": current((l, h))[3].cpu()} for (l, h) in base}, str(OUT_PT), "v631 refit tables")
    print(f"convergence (mean 201-250 minus mean 251-300) = {conv:+.5f}; final held-out CE added {final:+.4f}; layer-0 kappa shape cosines {cos}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_refit_converges": conv < CONV_TOL,
                   "pred_c_refit_halves_union": final <= HALF * (step0 - native), "pred_d_refit_under_bar": final <= BAR,
                   "pred_e_kernels_keep_shape": all(v >= COS_MIN for v in cos.values())}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "embedding_forward_program_refit_result_v631", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "step0_added": step0 - native, "heldout_added_by_step": {str(k): v for k, v in evals.items()}, "train_curve": curve, "convergence_delta": conv, "kappa_cos_layer0": {str(h): v for h, v in cos.items()}},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
