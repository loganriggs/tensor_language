"""Embedding-forward folding, rung 40 (v651): the early-MLP program with a proper stopping rule, at two sizes.

v650: the jointly refit (256, 512, 256) projection program for MLP-0/1/2 reached +0.17 on held-out at step 100 and then overfit. Selecting the
step on the held-out rows would be selection on the test set, so here 96 of the 672 fit rows are held out as VALIDATION (never trained on),
the reported step is the validation minimum (evaluated every 25 steps), and the held-out skip7000 number at that step is the registered endpoint.
Two arms: LARGE (256, 512, 256; 17.7M values) and SMALL (128, 256, 128; 8.8M values, 5.4x fewer than the three MLPs). Same optimiser (Adam
3e-4 -> 3e-5 cosine, 300 steps, batch 8; 576 training rows). CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native CE within 0.002 of 3.13241 (instrument)
    pred_b_validation_tracks_heldout for both arms the validation-chosen step is within 50 steps of the held-out minimum (the stopping rule works). Prior: likely
    pred_c_large_at_chosen_step    LARGE held-out at the validation-chosen step <= 0.20. Prior: likely (v650: 0.173 at its best)
    pred_d_small_at_chosen_step    SMALL held-out at the validation-chosen step <= 0.35. Prior: unsure
    pred_e_small_within_double     SMALL endpoint <= 2 x LARGE endpoint (halving every rank less than doubles the cost). Prior: unsure
PRICE (registered maximum): 2 arms x 300 steps = 600 forwards + 600 BACKWARDS; validation 2 x 12 x 3 = 72; held-out 2 x 12 x 6 = 144 + 6; total 822
forwards, 600 backwards; fit parameters 17.7M (LARGE) and 8.8M (SMALL). Bars: forwards <= 840, backwards <= 600.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_mlp_program_stopped_v651_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_mlp_program_stopped_v651_maps.pt"
FRAMES = ROOT / "circuits/followups/embedding_forward_layer_sweep_v618_frames.pt"
FIT_ROWS = (ROOT / ".rowcache/fineweb_n480_skip80.pt", ROOT / ".rowcache/fineweb_n192_skip11000.pt")
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615, V619_SMALL, SUM_SINGLES = 3.13241, 0.4064, 0.197
CANDIDATE_ID = "embedding_forward.mlp_program_stopped_v651"
FORWARDS_MAX, BACKWARDS_MAX = 840, 600
N_VAL = 96
STEPS, TBATCH, EBATCH, EVAL_EVERY = 300, 8, 32, 25
LR, LR_MIN = 3e-4, 3e-5
LAYERS = (0, 1, 2)
ARMS = {"large": (256, 512, 256), "small": (128, 256, 128)}
REPLAY_TOL, TRACK_STEPS, LARGE_BAR, SMALL_BAR, DOUBLE = 0.002, 50, 0.20, 0.35, 2.0
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_validation_tracks_heldout": "within 50 steps x 2", "pred_c_large_at_chosen_step": "<= 0.20",
               "pred_d_small_at_chosen_step": "<= 0.35", "pred_e_small_within_double": "small <= 2 x large"}


def main() -> None:
    n_fit = {k: 3 * (2 * 1152 * r[0] + 2 * 4608 * r[1] + 2 * 1152 * r[2]) for k, r in ARMS.items()}
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": BACKWARDS_MAX, "model_updates": 0, "fit_parameters": n_fit["large"],
            "fit_parameters_by_arm": n_fit, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "steps": STEPS, "lr": [LR, LR_MIN],
            "arms": {k: list(v) for k, v in ARMS.items()}, "n_val": N_VAL, "bars": {"replay_tol": REPLAY_TOL, "track_steps": TRACK_STEPS, "large_bar": LARGE_BAR, "small_bar": SMALL_BAR, "double": DOUBLE}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    dev = "cuda"; blocks = model.transformer.h; forwards = 0; backwards = 0
    allfit = torch.cat([torch.load(p, map_location="cpu").long() for p in FIT_ROWS]); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    gen = torch.Generator().manual_seed(651); perm = torch.randperm(allfit.shape[0], generator=gen)
    val = allfit[perm[:N_VAL]]; fit = allfit[perm[N_VAL:]]
    fr = torch.load(FRAMES, map_location=dev)
    means = {(l, key): fr[f"mlp{l}_{key}_mean"].float() for l in LAYERS for key in ("x", "m", "w")}
    state = {"on": False, "maps": None}
    hooks = []
    for l in LAYERS:
        mlp = blocks[l].mlp
        hooks.append(mlp.register_forward_pre_hook(lambda m, a, l=l: (means[(l, "x")] + (a[0] - means[(l, "x")]) @ state["maps"][(l, "x")][0] @ state["maps"][(l, "x")][1].T,) if state["on"] else None))
        hooks.append(mlp.Down.register_forward_pre_hook(lambda m, a, l=l: (means[(l, "m")] + (a[0] - means[(l, "m")]) @ state["maps"][(l, "m")][0] @ state["maps"][(l, "m")][1].T,) if state["on"] else None))
        hooks.append(mlp.register_forward_hook(lambda m, a, o, l=l: (means[(l, "w")] + (o - means[(l, "w")]) @ state["maps"][(l, "w")][0] @ state["maps"][(l, "w")][1].T) if state["on"] else None))

    def ce(rows, on):
        state["on"] = on; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, rows.shape[0], EBATCH):
                idx = rows[s:s + EBATCH, :-1].to(dev); total += float(model(idx, rows[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        return total / n, fw

    native, fw = ce(ev, False); forwards += fw
    results, saved = {}, {}
    for arm, (r_in, k_hid, r_out) in ARMS.items():
        maps, params = {}, []
        for l in LAYERS:
            for key, r in (("x", r_in), ("m", k_hid), ("w", r_out)):
                P0 = fr[f"mlp{l}_{key}_frame"][:, :r].float().contiguous()
                A = P0.clone().requires_grad_(True); B = P0.clone().requires_grad_(True); maps[(l, key)] = (A, B); params += [A, B]
        state["maps"] = maps
        opt = torch.optim.Adam(params, lr=LR); curve, val_curve, ho_curve = [], {}, {}
        v0, fw = ce(val, True); forwards += fw; h0, fw = ce(ev, True); forwards += fw; val_curve[0], ho_curve[0] = v0, h0 - native
        gen2 = torch.Generator().manual_seed(650); order = torch.randperm(fit.shape[0], generator=gen2)
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
                v_, fw = ce(val, True); forwards += fw; h_, fw = ce(ev, True); forwards += fw; val_curve[step], ho_curve[step] = v_, h_ - native
        chosen = min(val_curve, key=val_curve.get); ho_min_step = min(ho_curve, key=ho_curve.get)
        results[arm] = {"validation_curve": {str(k): v for k, v in val_curve.items()}, "heldout_curve": {str(k): v for k, v in ho_curve.items()}, "chosen_step": chosen,
                        "heldout_at_chosen": ho_curve[chosen], "heldout_min_step": ho_min_step, "heldout_min": ho_curve[ho_min_step], "train_curve": curve, "values": n_fit[arm]}
        saved[arm] = {f"mlp{l}_{key}_{n}": t.detach().cpu() for (l, key), (A, B) in maps.items() for n, t in (("A", A), ("B", B))}
        print(f"[{arm}] step 0 held-out {ho_curve[0]:+.4f} | validation-chosen step {chosen} -> held-out {ho_curve[chosen]:+.4f} (held-out minimum {ho_curve[ho_min_step]:+.4f} at step {ho_min_step}) | values {n_fit[arm] / 1e6:.1f}M")
    for h in hooks:
        h.remove()
    disk_guard.guard_torch_save(saved, str(OUT_PT), "v651 fitted maps (endpoint of training, not the chosen step)")
    L, S = results["large"], results["small"]
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_validation_tracks_heldout": all(abs(r["chosen_step"] - r["heldout_min_step"]) <= TRACK_STEPS for r in (L, S)),
                   "pred_c_large_at_chosen_step": L["heldout_at_chosen"] <= LARGE_BAR, "pred_d_small_at_chosen_step": S["heldout_at_chosen"] <= SMALL_BAR,
                   "pred_e_small_within_double": S["heldout_at_chosen"] <= DOUBLE * L["heldout_at_chosen"]}
    if forwards > FORWARDS_MAX or backwards > BACKWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} fwd / {backwards} bwd")
    OUT.write_text(json.dumps({"schema": "embedding_forward_mlp_program_stopped_result_v651", "candidate_id": CANDIDATE_ID, "plan": plan, "report": {"native": native, "arms": results},
                               "predictions": predictions, "forwards": forwards, "backwards": backwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards, "backwards": backwards}, indent=2))


if __name__ == "__main__":
    main()
