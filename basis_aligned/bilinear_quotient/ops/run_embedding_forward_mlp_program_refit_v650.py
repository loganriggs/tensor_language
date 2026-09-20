#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_step0_replays_v619 pred_c_no_overfit pred_d_stack_composes_when_fitted pred_e_under_quarter
"""Embedding-forward folding, rung 39 (v650): does the early-MLP projection program compose once its maps are FITTED?

v617-v620: per-layer context-PCA projection programs for MLP-0/1/2 — input rank 256, hidden (product) rank 512, output rank 256 — cost 0.043 /
0.104 / 0.050 alone but 0.406 stacked, and cascade-fitted frames made it worse. v631-v633 showed for attention that the same kind of joint cost was
mostly wrong parameters, recoverable by a joint refit. Here the three layers' six maps are refit jointly against the model's CE:
    x' = mu_x + (x - mu_x) A B^T     (A, B: 1152 x 256)        m' = mu_m + (m - mu_m) C D^T   (4608 x 512)        w' = mu_w + (w - mu_w) E F^T   (1152 x 256)
initialised at the context-PCA frames (A = B = P etc., so step 0 IS v619's SMALL stack), model frozen, Adam 3e-4 -> 3e-5 cosine, 300 steps, batch
8, fit rows skip80 + skip11000 (672 x 512), held-out 192 x 512 skip7000 every 50 steps, endpoint registered. Fit parameters = the program:
3 x [2 x 1152 x 256 + 2 x 4608 x 512 + 2 x 1152 x 256] = 17.7M values (vs the three MLPs' 47.8M). CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native CE within 0.002 of 3.13241 (instrument)
    pred_b_step0_replays_v619      step 0 within 0.01 of 0.406 (instrument)
    pred_c_no_overfit              held-out at every evaluation after step 50 <= the step-50 value + 0.002. Prior: unsure (17.7M parameters)
    pred_d_stack_composes_when_fitted endpoint <= 1.5 x the sum of the per-layer projection costs (0.197): <= 0.30. Prior: unsure
    pred_e_under_quarter           endpoint <= 0.25. Prior: unsure
PRICE (registered maximum): 300 steps = 300 forwards + 300 BACKWARDS; evaluation 7 x 6 + 6 = 48; total 348 forwards, 300 backwards; 17.7M fit
parameters (the program itself). Bars: forwards <= 360, backwards <= 300.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_mlp_program_refit_v650_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_mlp_program_refit_v650_maps.pt"
FRAMES = ROOT / "circuits/followups/embedding_forward_layer_sweep_v618_frames.pt"
FIT_ROWS = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615, V619_SMALL, SUM_SINGLES = 3.13241, 0.4064, 0.197
CANDIDATE_ID = "embedding_forward.mlp_program_refit_v650"
FORWARDS_MAX, BACKWARDS_MAX = 360, 300
STEPS, TBATCH, EBATCH, EVAL_EVERY = 300, 8, 32, 50
LR, LR_MIN = 3e-4, 3e-5
LAYERS = (0, 1, 2)
R_IN, K_HID, R_OUT = 256, 512, 256
REPLAY_TOL, STEP0_TOL, UPTURN_TOL, COMP, QUARTER = 0.002, 0.01, 0.002, 1.5, 0.25
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_step0_replays_v619": "+-0.01 of 0.406", "pred_c_no_overfit": "no upturn > 0.002",
               "pred_d_stack_composes_when_fitted": "<= 0.30", "pred_e_under_quarter": "<= 0.25"}


def main() -> None:
    n_fit = 3 * (2 * 1152 * R_IN + 2 * 4608 * K_HID + 2 * 1152 * R_OUT)
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": n_fit,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "steps": STEPS, "lr": [LR, LR_MIN], "ranks": [R_IN, K_HID, R_OUT],
            "bars": {"replay_tol": REPLAY_TOL, "step0_tol": STEP0_TOL, "upturn_tol": UPTURN_TOL, "comp": COMP, "quarter": QUARTER}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    dev = "cuda"; blocks = model.transformer.h; forwards = 0; backwards = 0
    fit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    fr = torch.load(FRAMES, map_location=dev)
    maps, params, means = {}, [], {}
    for l in LAYERS:
        for key, r in (("x", R_IN), ("m", K_HID), ("w", R_OUT)):
            P0 = fr[f"mlp{l}_{key}_frame"][:, :r].float().contiguous(); means[(l, key)] = fr[f"mlp{l}_{key}_mean"].float()
            A = P0.clone().requires_grad_(True); B = P0.clone().requires_grad_(True); maps[(l, key)] = (A, B); params += [A, B]
    state = {"on": False}
    hooks = []
    for l in LAYERS:
        mlp = blocks[l].mlp
        hooks.append(mlp.register_forward_pre_hook(lambda m, a, l=l: (means[(l, "x")] + (a[0] - means[(l, "x")]) @ maps[(l, "x")][0] @ maps[(l, "x")][1].T,) if state["on"] else None))
        hooks.append(mlp.Down.register_forward_pre_hook(lambda m, a, l=l: (means[(l, "m")] + (a[0] - means[(l, "m")]) @ maps[(l, "m")][0] @ maps[(l, "m")][1].T,) if state["on"] else None))
        hooks.append(mlp.register_forward_hook(lambda m, a, o, l=l: (means[(l, "w")] + (o - means[(l, "w")]) @ maps[(l, "w")][0] @ maps[(l, "w")][1].T) if state["on"] else None))

    def ce(on):
        state["on"] = on; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        return total / n, fw

    native, fw = ce(False); forwards += fw
    step0, fw = ce(True); forwards += fw
    print(f"native {native:.5f} | step-0 (context-PCA small stack) CE added {step0 - native:+.4f}")
    opt = torch.optim.Adam(params, lr=LR); curve, evals = [], {0: step0 - native}
    gen = torch.Generator().manual_seed(650); order = torch.randperm(fit.shape[0], generator=gen)
    for step in range(1, STEPS + 1):
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * (step - 1) / STEPS))
        for g in opt.param_groups:
            g["lr"] = lr
        s0 = ((step - 1) * TBATCH) % fit.shape[0]; sel = order[s0:s0 + TBATCH]
        if len(sel) < TBATCH:
            sel = order[:TBATCH]
        idx = fit[sel, :-1].to(dev); tgt = fit[sel, 1:].to(dev); state["on"] = True
        with torch.enable_grad():
            loss = model(idx, tgt); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        forwards += 1; backwards += 1; curve.append(float(loss))
        if step % EVAL_EVERY == 0:
            v_, fw = ce(True); forwards += fw; evals[step] = v_ - native
            print(f"step {step}: train {sum(curve[-EVAL_EVERY:]) / EVAL_EVERY:.4f} | held-out CE added {evals[step]:+.4f} | lr {lr:.5f}")
    for h in hooks:
        h.remove()
    final = evals[STEPS]
    with torch.no_grad():
        disk_guard.guard_torch_save({f"mlp{l}_{key}_{n}": t.detach().cpu() for (l, key), (A, B) in maps.items() for n, t in (("A", A), ("B", B))}, str(OUT_PT), "v650 fitted maps")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_step0_replays_v619": abs((step0 - native) - V619_SMALL) <= STEP0_TOL,
                   "pred_c_no_overfit": all(evals[s] <= evals[50] + UPTURN_TOL for s in range(100, STEPS + 1, EVAL_EVERY)),
                   "pred_d_stack_composes_when_fitted": final <= COMP * SUM_SINGLES, "pred_e_under_quarter": final <= QUARTER}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "embedding_forward_mlp_program_refit_result_v650", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "step0_added": step0 - native, "heldout_added_by_step": {str(k): v for k, v in evals.items()}, "train_curve": curve},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
