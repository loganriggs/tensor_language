#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_cascade_removes_excess_small pred_c_cascade_stack_additive pred_d_cascade_large_cheap pred_e_pair_localises_excess
"""Embedding-forward folding, rung 12 (v620): CASCADE-fitted frames — does per-layer compression compose when the frames are fitted in order?

v619: independent context-PCA programs for MLP-0/1/2 cost 0.043 / 0.104 / 0.050 alone but 0.406 stacked (SMALL = in 256, hidden 512, out 256);
LARGE (512, 1024, 512) 0.083 stacked vs 0.056 summed. Hypothesis: MLP-1's and MLP-2's frames were fitted on NATIVE activations; once MLP-0 is
projected, their inputs move off that distribution. Cascade fit: (1) install MLP-0's program, capture MLP-1's input / product / write statistics
under it, build MLP-1's frames; (2) install MLP-0 + MLP-1, capture and build MLP-2's frames. Still pure projections — the statistics are means and
covariances of activations, never the loss. Done separately for SMALL and LARGE. Also the MLP-0 + MLP-1 pair (independent vs cascade) to localise
where the excess arises. CE on skip7000 (both stacks also on skip11000). CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved; skip7000 unless stated)
    pred_a_native_replays            native CE within 0.002 of 3.13241 (instrument)
    pred_b_cascade_removes_excess_small cascade SMALL stack <= 0.75 x independent SMALL stack (re-measured here; v619: 0.406). Prior: likely
    pred_c_cascade_stack_additive    cascade SMALL stack <= 1.5 x the sum of v619's per-layer joints re-measured here (0.197). Prior: unsure
    pred_d_cascade_large_cheap       cascade LARGE stack <= 0.05 on skip7000 AND skip11000. Prior: unsure
    pred_e_pair_localises_excess     independent MLP-0 + MLP-1 pair costs >= 1.5 x (MLP-0 joint + MLP-1 joint) — the excess is already present
                                     at the first hand-off — AND the cascade pair removes >= 50% of that excess. Prior: likely
PRICE (registered maximum): 4 capture passes x 15 = 60 forwards; eval configs = native + 3 per-layer SMALL joints + independent {pair, stack} x
{SMALL, LARGE} + cascade {pair, stack} x {SMALL, LARGE} = 12 configs x 6 batches = 72 on skip7000, + 4 stacks x 6 = 24 on skip11000; total 156
forwards; 0 backwards; 0 fits. Bar <= 170.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_cascade_frames_v620_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_cascade_frames_v620_frames.pt"
FRAMES = ROOT / "circuits/followups/embedding_forward_layer_sweep_v618_frames.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt", "skip11000": ROOT / ".rowcache/fineweb_n192_skip11000.pt"}
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.cascade_frames_v620"
FORWARDS_MAX = 170
BATCH = 32
LAYERS = (0, 1, 2)
SIZES = {"small": (256, 512, 256), "large": (512, 1024, 512)}
REPLAY_TOL, EXCESS_RATIO, ADD_RATIO, LARGE_MAX, PAIR_RATIO, PAIR_REMOVE = 0.002, 0.75, 1.5, 0.05, 1.5, 0.5
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_cascade_removes_excess_small": "<= 0.75 x independent", "pred_c_cascade_stack_additive": "<= 1.5 x sum of joints",
               "pred_d_cascade_large_cheap": "<= 0.05 on both sets", "pred_e_pair_localises_excess": "pair >= 1.5 x sum and cascade removes >= 50%"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "layers": list(LAYERS), "sizes": {k: list(v) for k, v in SIZES.items()},
            "bars": {"replay_tol": REPLAY_TOL, "excess_ratio": EXCESS_RATIO, "add_ratio": ADD_RATIO, "large_max": LARGE_MAX, "pair_ratio": PAIR_RATIO, "pair_remove": PAIR_REMOVE}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    ind = torch.load(FRAMES, map_location=dev)                                            # independent frames (v618)
    fit = torch.load(FIT_ROWS, map_location="cpu").long(); rows = {k: torch.load(p, map_location="cpu").long() for k, p in EVAL.items()}
    HID = blocks[0].mlp.Left.weight.shape[0]
    with torch.no_grad():
        def hooks_for(spec, frames):
            """spec: {layer: (r_in, k, r_out)}; frames: {layer: dict with x/m/w mean+frame}"""
            hs = []
            for l, (ri, k, ro) in spec.items():
                mlp = blocks[l].mlp; f = frames[l]
                mu, P = f["x_mean"], f["x_frame"][:, :ri]; hs.append(mlp.register_forward_pre_hook(lambda m, a, P=P, mu=mu: (mu + (a[0] - mu) @ P @ P.T,)))
                mu, Q = f["m_mean"], f["m_frame"][:, :k]; hs.append(mlp.Down.register_forward_pre_hook(lambda m, a, Q=Q, mu=mu: (mu + (a[0] - mu) @ Q @ Q.T,)))
                mu, W = f["w_mean"], f["w_frame"][:, :ro]; hs.append(mlp.register_forward_hook(lambda m, a, o, W=W, mu=mu: mu + (o - mu) @ W @ W.T))
            return hs

        ind_frames = {l: {f"{key}_{part}": ind[f"mlp{l}_{key}_{part}"] for key in ("x", "m", "w") for part in ("mean", "frame")} for l in LAYERS}

        def capture_layer(l, active_hooks, top):
            """Mean / PCA frame of MLP-l's input, product and write under the currently installed hooks."""
            stat = {key: {"n": 0, "s": torch.zeros(dim, dtype=torch.float64, device=dev), "ss": torch.zeros(dim, dim, dtype=torch.float64, device=dev)} for key, dim in (("x", D), ("m", HID), ("w", D))}

            def acc(key, val):
                v = val.reshape(-1, val.shape[-1]).double(); stat[key]["n"] += v.shape[0]; stat[key]["s"] += v.sum(0); stat[key]["ss"] += v.T @ v

            mlp = blocks[l].mlp
            hs = [mlp.register_forward_pre_hook(lambda m, a: acc("x", a[0])), mlp.Down.register_forward_pre_hook(lambda m, a: acc("m", a[0])), mlp.register_forward_hook(lambda m, a, o: acc("w", o))]
            fw = 0
            for s in range(0, fit.shape[0], BATCH):
                model(fit[s:s + BATCH, :-1].to(dev), fit[s:s + BATCH, 1:].to(dev)); fw += 1
            for h in hs + active_hooks:
                h.remove()
            out = {}
            for key, st in stat.items():
                mean = st["s"] / st["n"]; cov = st["ss"] / st["n"] - torch.outer(mean, mean)
                U = torch.linalg.eigh(cov)[1].flip(1)
                out[f"{key}_mean"] = mean.float(); out[f"{key}_frame"] = U[:, :top[key]].float()
            return out, fw

        cas_frames = {}
        for size, (ri, k, ro) in SIZES.items():
            frames = {0: ind_frames[0]}
            for l in (1, 2):
                spec = {j: (ri, k, ro) for j in range(l)}
                f, fw = capture_layer(l, hooks_for(spec, frames), {"x": ri, "m": k, "w": ro}); forwards += fw; frames[l] = f
            cas_frames[size] = frames
            print(f"cascade frames built for {size}")

        def ce(rws, hs):
            total = 0.0; n = 0; fw = 0
            for s in range(0, rws.shape[0], BATCH):
                idx = rws[s:s + BATCH, :-1].to(dev); total += float(model(idx, rws[s:s + BATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            for h in hs:
                h.remove()
            return total / n, fw

        configs = {"native": ({}, ind_frames)}
        for l in LAYERS:
            configs[f"mlp{l}|joint|small"] = ({l: SIZES["small"]}, ind_frames)
        for size, dims in SIZES.items():
            configs[f"pair|independent|{size}"] = ({0: dims, 1: dims}, ind_frames); configs[f"pair|cascade|{size}"] = ({0: dims, 1: dims}, cas_frames[size])
            configs[f"stack|independent|{size}"] = ({l: dims for l in LAYERS}, ind_frames); configs[f"stack|cascade|{size}"] = ({l: dims for l in LAYERS}, cas_frames[size])
        results = {}
        for name, rws in rows.items():
            todo = list(configs) if name == "skip7000" else ["native"] + [c for c in configs if c.startswith("stack|")]
            vals = {}
            for cname in todo:
                spec, frames = configs[cname]; v, fw = ce(rws, hooks_for(spec, frames)); forwards += fw; vals[cname] = v
            native = vals["native"]; results[name] = {"native": native, "added": {c: v - native for c, v in vals.items()}}
            print(f"{name}: native {native:.5f} | " + " ".join(f"{c}={v - native:+.4f}" for c, v in vals.items() if c != "native"))
        disk_guard.guard_torch_save({f"{size}_mlp{l}_{key}": t.cpu() for size, fr in cas_frames.items() for l, f in fr.items() if l > 0 for key, t in f.items()}, str(OUT_PT), "v620 cascade frames")
    a = results["skip7000"]["added"]; b = results["skip11000"]["added"]
    sum_joints = sum(a[f"mlp{l}|joint|small"] for l in LAYERS); pair_sum = a["mlp0|joint|small"] + a["mlp1|joint|small"]
    pair_excess = a["pair|independent|small"] - pair_sum
    predictions = {"pred_a_native_replays": abs(results["skip7000"]["native"] - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_cascade_removes_excess_small": a["stack|cascade|small"] <= EXCESS_RATIO * a["stack|independent|small"],
                   "pred_c_cascade_stack_additive": a["stack|cascade|small"] <= ADD_RATIO * sum_joints,
                   "pred_d_cascade_large_cheap": a["stack|cascade|large"] <= LARGE_MAX and b["stack|cascade|large"] <= LARGE_MAX,
                   "pred_e_pair_localises_excess": a["pair|independent|small"] >= PAIR_RATIO * pair_sum and (a["pair|independent|small"] - a["pair|cascade|small"]) >= PAIR_REMOVE * pair_excess}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_cascade_frames_result_v620", "candidate_id": CANDIDATE_ID, "plan": plan, "report": results,
                               "derived": {"sum_joints_small": sum_joints, "pair_sum_small": pair_sum, "pair_excess_independent": pair_excess},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
