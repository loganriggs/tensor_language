#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_additivity pred_c_256_256_cheap_and_transfers pred_d_128_128_bounded pred_e_monotone
"""Embedding-forward folding, rung 9 (v617): joint (r_in, r_out) context-PCA compression of MLP-0 with a literal price, additivity, and transfer.

v616: pure projections onto CONTEXT PCA frames are cheap for MLP-0 (IN r=256 0.019 nats, OUT r=256 0.011, IN r=128 0.056, OUT r=128 0.036).
This rung installs both at once — x^' = mu_x + P P^T (x^ - mu_x) before L/R and write' = mu_w + W W^T (write - mu_w) after Down — on the 3 x 3 grid
r_in, r_out in {64, 128, 256} (v616 saved 256 frame columns), and prices each cell literally: the compressed map is L P (4608 x r_in), R P, W^T Down (r_out x 4608), plus the two
means, i.e. 4608 (2 r_in + r_out) + 2 x 1152 values against MLP-0's native 3 x 4608 x 1152 = 15.9M. Additivity (joint vs in + out) says whether the
two projections interact; a second held-out set (skip11000, 192 x 512) says whether frames fitted on the skip80 rows transfer. Frames are v616's
(saved), so no new capture. CE ADDED above the native forward on the same rows; lower is better.
PREDICTIONS (scored as written; failures preserved; skip7000 unless stated)
    pred_a_native_replays                 native CE on skip7000 within 0.002 of 3.13241 (instrument); skip11000's native is recorded, not predicted
    pred_b_additivity                     joint (256, 256) CE added <= 1.5 x (IN-only 256 + OUT-only 256) measured in this run. Prior: likely
    pred_c_256_256_cheap_and_transfers    joint (256, 256) CE added <= 0.05 on skip7000 AND <= 0.05 on skip11000 (price 5.9M values, 2.7x fewer). Prior: likely
    pred_d_128_128_bounded                joint (128, 128) CE added <= 0.15 on skip7000 (price 3.0M, 5.3x fewer). Prior: unsure
    pred_e_monotone                       CE added is non-increasing in r_in at fixed r_out and in r_out at fixed r_in, all 9 cells. Prior: likely
PRICE (registered maximum): (9 joint + 2 single-arm + native) = 12 configs x 2 eval sets x 6 batches = 144 forwards; 0 backwards; 0 fits. Bar <= 150.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_joint_context_edit_v617_result.json"
FRAMES = ROOT / "circuits/followups/embedding_forward_context_frame_edit_v616_frames.pt"
EVAL = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt", "skip11000": ROOT / ".rowcache/fineweb_n192_skip11000.pt"}
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.joint_context_edit_v617"
FORWARDS_MAX = 150
BATCH = 32
RANKS = (64, 128, 256)
REPLAY_TOL, ADD_RATIO, CHEAP, BOUNDED = 0.002, 1.5, 0.05, 0.15
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_additivity": "joint <= 1.5 x (in + out)", "pred_c_256_256_cheap_and_transfers": "<= 0.05 on both sets",
               "pred_d_128_128_bounded": "<= 0.15", "pred_e_monotone": "non-increasing in each rank"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "ranks": list(RANKS), "eval_sets": list(EVAL),
            "bars": {"replay_tol": REPLAY_TOL, "add_ratio": ADD_RATIO, "cheap": CHEAP, "bounded": BOUNDED}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; dev = "cuda"; b0 = model.transformer.h[0]; forwards = 0
    fr = torch.load(FRAMES, map_location=dev)
    rows = {k: torch.load(p, map_location="cpu").long() for k, p in EVAL.items()}
    HID = b0.mlp.Left.weight.shape[0]; native_params = 3 * HID * D
    with torch.no_grad():
        mu_x, Px, mu_w, Pw = fr["ctx_x_mean"].float(), fr["ctx_x_frame"].float(), fr["ctx_w_mean"].float(), fr["ctx_w_frame"].float()
        assert Px.shape[1] >= max(RANKS) and Pw.shape[1] >= max(RANKS), "v616 frames hold fewer directions than requested"

        def ce(rws, hooks):
            total = 0.0; n = 0; fw = 0
            for s in range(0, rws.shape[0], BATCH):
                idx = rws[s:s + BATCH, :-1].to(dev); tgt = rws[s:s + BATCH, 1:].to(dev)
                total += float(model(idx, tgt)) * idx.numel(); n += idx.numel(); fw += 1
            for h in hooks:
                h.remove()
            return total / n, fw

        def hooks_for(r_in, r_out):
            hs = []
            if r_in:
                P = Px[:, :r_in]; hs.append(b0.mlp.register_forward_pre_hook(lambda m, a, P=P: (mu_x + (a[0] - mu_x) @ P @ P.T,)))
            if r_out:
                W = Pw[:, :r_out]; hs.append(b0.mlp.register_forward_hook(lambda m, a, o, W=W: mu_w + (o - mu_w) @ W @ W.T))
            return hs

        configs = [(0, 0), (256, 0), (0, 256)] + [(ri, ro) for ri in RANKS for ro in RANKS]
        results = {}
        for name, rws in rows.items():
            res = {}
            for ri, ro in configs:
                val, fw = ce(rws, hooks_for(ri, ro)); forwards += fw; res[f"{ri}x{ro}"] = val
            native = res["0x0"]
            results[name] = {"native": native, "added": {k: v - native for k, v in res.items()},
                             "price": {f"{ri}x{ro}": (HID * (2 * ri + ro) + 2 * D) for ri in RANKS for ro in RANKS}, "native_params": native_params}
            print(f"{name}: native {native:.5f} | in256 {res['256x0'] - native:+.4f} out256 {res['0x256'] - native:+.4f}")
            for ri in RANKS:
                print(f"  r_in={ri:3d}: " + " ".join(f"r_out={ro}: {res[f'{ri}x{ro}'] - native:+.4f} ({native_params / (HID * (2 * ri + ro) + 2 * D):.1f}x)" for ro in RANKS))
    a = results["skip7000"]["added"]; b = results["skip11000"]["added"]
    mono = all(a[f"{RANKS[i]}x{ro}"] >= a[f"{RANKS[i + 1]}x{ro}"] for i in range(2) for ro in RANKS) and all(a[f"{ri}x{RANKS[i]}"] >= a[f"{ri}x{RANKS[i + 1]}"] for i in range(2) for ri in RANKS)
    predictions = {"pred_a_native_replays": abs(results["skip7000"]["native"] - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_additivity": a["256x256"] <= ADD_RATIO * (a["256x0"] + a["0x256"]),
                   "pred_c_256_256_cheap_and_transfers": a["256x256"] <= CHEAP and b["256x256"] <= CHEAP,
                   "pred_d_128_128_bounded": a["128x128"] <= BOUNDED,
                   "pred_e_monotone": mono}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_joint_context_edit_result_v617", "candidate_id": CANDIDATE_ID, "plan": plan, "report": results,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
