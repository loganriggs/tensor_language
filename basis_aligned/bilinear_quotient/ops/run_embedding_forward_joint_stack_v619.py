#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_three_mode_additive pred_c_stack_additive pred_d_stack_large_cheap pred_e_stack_small_bounded
"""Embedding-forward folding, rung 11 (v619): joint three-mode projections per early MLP, and all three MLPs compressed at once.

v618 priced each mode of MLP-0/1/2 separately (context-PCA projections on the normalised input, the 4608 product coordinates, and the write).
This rung installs all three at once per layer — a (r_in, k, r_out) projected bilinear program with literal price 4608 (2 r_in + r_out) + 5760 k
values (L P, R P, Q^T, Down Q, W^T Down; means extra) — at SMALL (256, 512, 256) and LARGE (512, 1024, 512), then stacks the three layers. Each
projection reuses v618's saved frames (no new capture). Additivity across modes (joint vs the v618 single-arm sum, re-measured here) and across
layers (stack vs the per-layer joints) says whether the early MLPs can be compressed independently. Both held-out sets. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved; skip7000 unless stated)
    pred_a_native_replays     native CE within 0.002 of 3.13241 (instrument)
    pred_b_three_mode_additive per layer, SMALL joint <= 1.5 x (IN-256 + HID-512 + OUT-256 singles measured in this run), all three layers. Prior: likely
    pred_c_stack_additive     SMALL stack of the three layers <= 1.5 x the sum of the three SMALL per-layer joints. Prior: unsure — MLP-1 reads MLP-0's
                              write, so errors could compound (or cancel)
    pred_d_stack_large_cheap  LARGE stack (512, 1024, 512) x 3 layers costs <= 0.05 on skip7000 AND on skip11000. Prior: likely
    pred_e_stack_small_bounded SMALL stack (256, 512, 256) x 3 layers costs <= 0.25 on skip7000. Prior: unsure
PRICE (registered maximum): configs = native + 9 singles + 6 per-layer joints + 2 stacks = 18, x 2 eval sets x 6 batches = 216 forwards; 0 backwards;
0 fits. Bar <= 230.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_joint_stack_v619_result.json"
FRAMES = ROOT / "circuits/followups/embedding_forward_layer_sweep_v618_frames.pt"
EVAL = {"skip7000": ROOT / ".rowcache/fineweb_n192_skip7000.pt", "skip11000": ROOT / ".rowcache/fineweb_n192_skip11000.pt"}
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.joint_stack_v619"
FORWARDS_MAX = 230
BATCH = 32
LAYERS = (0, 1, 2)
SMALL, LARGE = (256, 512, 256), (512, 1024, 512)
REPLAY_TOL, ADD_RATIO, LARGE_MAX, SMALL_MAX = 0.002, 1.5, 0.05, 0.25
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_three_mode_additive": "joint <= 1.5 x singles, x3", "pred_c_stack_additive": "stack <= 1.5 x sum of joints",
               "pred_d_stack_large_cheap": "<= 0.05 on both sets", "pred_e_stack_small_bounded": "<= 0.25"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "layers": list(LAYERS), "small": list(SMALL), "large": list(LARGE),
            "bars": {"replay_tol": REPLAY_TOL, "add_ratio": ADD_RATIO, "large_max": LARGE_MAX, "small_max": SMALL_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    fr = torch.load(FRAMES, map_location=dev)
    rows = {k: torch.load(p, map_location="cpu").long() for k, p in EVAL.items()}
    HID = blocks[0].mlp.Left.weight.shape[0]
    with torch.no_grad():
        def hooks_for(spec):
            """spec: {layer: (r_in or 0, k or 0, r_out or 0)}"""
            hs = []
            for l, (ri, k, ro) in spec.items():
                mlp = blocks[l].mlp
                if ri:
                    mu, P = fr[f"mlp{l}_x_mean"], fr[f"mlp{l}_x_frame"][:, :ri]
                    hs.append(mlp.register_forward_pre_hook(lambda m, a, P=P, mu=mu: (mu + (a[0] - mu) @ P @ P.T,)))
                if k:
                    mu, Q = fr[f"mlp{l}_m_mean"], fr[f"mlp{l}_m_frame"][:, :k]
                    hs.append(mlp.Down.register_forward_pre_hook(lambda m, a, Q=Q, mu=mu: (mu + (a[0] - mu) @ Q @ Q.T,)))
                if ro:
                    mu, W = fr[f"mlp{l}_w_mean"], fr[f"mlp{l}_w_frame"][:, :ro]
                    hs.append(mlp.register_forward_hook(lambda m, a, o, W=W, mu=mu: mu + (o - mu) @ W @ W.T))
            return hs

        def ce(rws, hs):
            total = 0.0; n = 0; fw = 0
            for s in range(0, rws.shape[0], BATCH):
                idx = rws[s:s + BATCH, :-1].to(dev); total += float(model(idx, rws[s:s + BATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            for h in hs:
                h.remove()
            return total / n, fw

        configs = {"native": {}}
        for l in LAYERS:
            configs[f"mlp{l}|in|{SMALL[0]}"] = {l: (SMALL[0], 0, 0)}; configs[f"mlp{l}|hid|{SMALL[1]}"] = {l: (0, SMALL[1], 0)}; configs[f"mlp{l}|out|{SMALL[2]}"] = {l: (0, 0, SMALL[2])}
            configs[f"mlp{l}|joint|small"] = {l: SMALL}; configs[f"mlp{l}|joint|large"] = {l: LARGE}
        configs["stack|small"] = {l: SMALL for l in LAYERS}; configs["stack|large"] = {l: LARGE for l in LAYERS}
        price = lambda ri, k, ro: HID * (2 * ri + ro) + (D + HID) * k
        results = {}
        for name, rws in rows.items():
            vals = {}
            for cname, spec in configs.items():
                v, fw = ce(rws, hooks_for(spec)); forwards += fw; vals[cname] = v
            native = vals["native"]; added = {k: v - native for k, v in vals.items()}
            results[name] = {"native": native, "added": added}
            print(f"{name}: native {native:.5f}")
            for l in LAYERS:
                print(f"  MLP-{l}: singles in {added[f'mlp{l}|in|256']:+.4f} hid {added[f'mlp{l}|hid|512']:+.4f} out {added[f'mlp{l}|out|256']:+.4f} | joint small {added[f'mlp{l}|joint|small']:+.4f} large {added[f'mlp{l}|joint|large']:+.4f}")
            print(f"  STACK small {added['stack|small']:+.4f} (price {3 * price(*SMALL) / 1e6:.2f}M vs {3 * 3 * HID * D / 1e6:.1f}M, {3 * 3 * HID * D / (3 * price(*SMALL)):.1f}x) | large {added['stack|large']:+.4f} ({3 * 3 * HID * D / (3 * price(*LARGE)):.1f}x)")
    a = results["skip7000"]["added"]; b = results["skip11000"]["added"]
    predictions = {"pred_a_native_replays": abs(results["skip7000"]["native"] - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_three_mode_additive": all(a[f"mlp{l}|joint|small"] <= ADD_RATIO * (a[f"mlp{l}|in|256"] + a[f"mlp{l}|hid|512"] + a[f"mlp{l}|out|256"]) for l in LAYERS),
                   "pred_c_stack_additive": a["stack|small"] <= ADD_RATIO * sum(a[f"mlp{l}|joint|small"] for l in LAYERS),
                   "pred_d_stack_large_cheap": a["stack|large"] <= LARGE_MAX and b["stack|large"] <= LARGE_MAX,
                   "pred_e_stack_small_bounded": a["stack|small"] <= SMALL_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_joint_stack_result_v619", "candidate_id": CANDIDATE_ID, "plan": plan, "report": results,
                               "prices": {"small_per_layer": price(*SMALL), "large_per_layer": price(*LARGE), "native_per_layer": 3 * HID * D},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
