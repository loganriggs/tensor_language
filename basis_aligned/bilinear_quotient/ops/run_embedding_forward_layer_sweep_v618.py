#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_in256_cheap_deeper pred_c_out256_cheap_all pred_d_hidden1024_cheap_all pred_e_mlp0_cheapest_input
"""Embedding-forward folding, rung 10 (v618): context-PCA projections of MLP-0 / MLP-1 / MLP-2 on all three modes — input, hidden (product), output.

v616/v617 established for MLP-0 that context-PCA projections are cheap under CE (IN 256: 0.019, OUT 256: 0.011, joint additive). This rung asks
whether the same holds for the next two bilinear MLPs and adds the mode the bilinear map actually lives on: the 4608 PRODUCT coordinates
m = (L x^) o (R x^). A rank-k PCA bottleneck on m before Down (m' = mu_m + Q Q^T (m - mu_m)) measures how many product directions each MLP's
write uses; because the write's rank is at most 1152 anyway, only k < 1152 is informative, so k in {256, 512, 1024}. Statistics (mean and
covariance of the normalised input, the product vector and the write) come from ONE hooked pass over the 480 x 512 skip80 fit rows; CE on the
192 x 512 skip7000 held-out rows. Pure projections, no fitting. CE ADDED above the native forward, lower is better. Prices per arm: IN r ->
2 x 4608 r; OUT r -> 4608 r; HIDDEN k -> 1152 k + 4608 k (Down Q and Q^T).
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays       native CE within 0.002 of 3.13241 (instrument)
    pred_b_in256_cheap_deeper   IN r=256 costs <= 0.05 for MLP-1 and for MLP-2 (the cheapness of v616 persists past layer 0). Prior: unsure
    pred_c_out256_cheap_all     OUT r=256 costs <= 0.03 for each of MLP-0, MLP-1, MLP-2. Prior: likely
    pred_d_hidden1024_cheap_all HIDDEN k=1024 costs <= 0.02 for each of the three MLPs (each write uses at most ~1024 product directions
                                at negligible cost). Prior: unsure
    pred_e_mlp0_cheapest_input  IN r=128: MLP-0's cost is the smallest of the three (deeper MLPs read richer contextual inputs). Prior: unsure
PRICE (registered maximum): capture 15 forwards + (27 edits + native) x 6 batches = 168; total 183 forwards; 0 backwards; 0 fits. Bar <= 200.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_layer_sweep_v618_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_layer_sweep_v618_frames.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.layer_sweep_v618"
FORWARDS_MAX = 200
BATCH = 32
LAYERS = (0, 1, 2)
R_IO = (128, 256, 512)
K_HID = (256, 512, 1024)
REPLAY_TOL, IN_MAX, OUT_MAX, HID_MAX = 0.002, 0.05, 0.03, 0.02
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_in256_cheap_deeper": "<= 0.05 x 2 layers", "pred_c_out256_cheap_all": "<= 0.03 x 3",
               "pred_d_hidden1024_cheap_all": "<= 0.02 x 3", "pred_e_mlp0_cheapest_input": "MLP-0 min at r=128"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "layers": list(LAYERS), "r_io": list(R_IO), "k_hid": list(K_HID),
            "bars": {"replay_tol": REPLAY_TOL, "in_max": IN_MAX, "out_max": OUT_MAX, "hid_max": HID_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long(); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    HID = blocks[0].mlp.Left.weight.shape[0]
    with torch.no_grad():
        # ---- one capture pass: mean / covariance of input (D), product (HID), write (D) per layer ---------------------------------
        stat = {}
        for l in LAYERS:
            for key, dim in (("x", D), ("m", HID), ("w", D)):
                stat[(l, key)] = {"n": 0, "s": torch.zeros(dim, dtype=torch.float64, device=dev), "ss": torch.zeros(dim, dim, dtype=torch.float64, device=dev)}

        def acc(k, val):
            v = val.reshape(-1, val.shape[-1]).double(); stat[k]["n"] += v.shape[0]; stat[k]["s"] += v.sum(0); stat[k]["ss"] += v.T @ v

        hooks = []
        for l in LAYERS:
            mlp = blocks[l].mlp
            hooks.append(mlp.register_forward_pre_hook(lambda mod, a, l=l: acc((l, "x"), a[0])))
            hooks.append(mlp.Down.register_forward_pre_hook(lambda mod, a, l=l: acc((l, "m"), a[0])))
            hooks.append(mlp.register_forward_hook(lambda mod, a, o, l=l: acc((l, "w"), o)))
        for s in range(0, fit.shape[0], BATCH):
            model(fit[s:s + BATCH, :-1].to(dev), fit[s:s + BATCH, 1:].to(dev)); forwards += 1
        for h in hooks:
            h.remove()
        frames = {}; pca_ranks = {}
        for (l, key), st in stat.items():
            mean = st["s"] / st["n"]; cov = st["ss"] / st["n"] - torch.outer(mean, mean)
            evals, U = torch.linalg.eigh(cov); evals, U = evals.flip(0), U.flip(1)
            top = max(R_IO) if key != "m" else max(K_HID)
            frames[(l, key)] = (mean.float(), U[:, :top].float())
            c = evals.clamp_min(0).cumsum(0) / evals.clamp_min(0).sum()
            pca_ranks[f"mlp{l}_{key}"] = {"0.9": int((c < 0.9).sum()) + 1, "0.95": int((c < 0.95).sum()) + 1, "0.99": int((c < 0.99).sum()) + 1}
            del st["ss"]
        print("context PCA 90/95/99% ranks:", pca_ranks)

        def ce(hs):
            total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], BATCH):
                idx = ev[s:s + BATCH, :-1].to(dev); total += float(model(idx, ev[s:s + BATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            for h in hs:
                h.remove()
            return total / n, fw

        native, fw = ce([]); forwards += fw; print(f"native CE {native:.5f}")
        edits = {}
        for l in LAYERS:
            mlp = blocks[l].mlp
            for r in R_IO:
                mu, P = frames[(l, "x")]; P = P[:, :r]
                h = mlp.register_forward_pre_hook(lambda mod, a, P=P, mu=mu: (mu + (a[0] - mu) @ P @ P.T,))
                v, fw = ce([h]); forwards += fw; edits[f"mlp{l}|in|{r}"] = v - native
                mu, W = frames[(l, "w")]; W = W[:, :r]
                h = mlp.register_forward_hook(lambda mod, a, o, W=W, mu=mu: mu + (o - mu) @ W @ W.T)
                v, fw = ce([h]); forwards += fw; edits[f"mlp{l}|out|{r}"] = v - native
            for k in K_HID:
                mu, Q = frames[(l, "m")]; Q = Q[:, :k]
                h = mlp.Down.register_forward_pre_hook(lambda mod, a, Q=Q, mu=mu: (mu + (a[0] - mu) @ Q @ Q.T,))
                v, fw = ce([h]); forwards += fw; edits[f"mlp{l}|hid|{k}"] = v - native
            print(f"MLP-{l}: IN " + " ".join(f"{r}={edits[f'mlp{l}|in|{r}']:+.4f}" for r in R_IO) + " | OUT " + " ".join(f"{r}={edits[f'mlp{l}|out|{r}']:+.4f}" for r in R_IO)
                  + " | HID " + " ".join(f"{k}={edits[f'mlp{l}|hid|{k}']:+.4f}" for k in K_HID))
        disk_guard.guard_torch_save({f"mlp{l}_{key}_{part}": t.cpu() for (l, key), (mu, U) in frames.items() for part, t in (("mean", mu), ("frame", U))}, str(OUT_PT), "v618 frames")
    e = edits
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_in256_cheap_deeper": e["mlp1|in|256"] <= IN_MAX and e["mlp2|in|256"] <= IN_MAX,
                   "pred_c_out256_cheap_all": all(e[f"mlp{l}|out|256"] <= OUT_MAX for l in LAYERS),
                   "pred_d_hidden1024_cheap_all": all(e[f"mlp{l}|hid|1024"] <= HID_MAX for l in LAYERS),
                   "pred_e_mlp0_cheapest_input": e["mlp0|in|128"] <= min(e["mlp1|in|128"], e["mlp2|in|128"])}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_layer_sweep_result_v618", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": edits, "context_pca_ranks": pca_ranks,
                                          "prices": {"in": {r: 2 * HID * r for r in R_IO}, "out": {r: HID * r for r in R_IO}, "hid": {k: (D + HID) * k for k in K_HID}, "native": 3 * HID * D}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
