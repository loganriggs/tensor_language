#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_57_is_costliest pred_c_layer5_carries_most pred_d_median_head_small pred_e_kernel_recovery_median
"""Attention lane, v701: the VALUE of every head — CE cost of MEAN ABLATION — and the kernel programs' recovery relative to it.

Logan (20 Sep): value each head by its CE loss under mean ablation; report what a program recovers relative to that. Mean ablation here: the
head's pre-projection output z_h (128-d) is replaced at every position by its mean over the 480 x 512 skip80 fit rows (one 128-vector per head;
c_proj is linear so this equals replacing the head's residual write by its mean). Priced one head at a time on the 192 x 512 skip7000 rows.
Relative recovery of the closed-form kernel program (v629 / v634 / v641-v644 single-head costs, same rows) = 1 - cost_kernel / cost_mean_ablation,
reported per head where the mean-ablation cost >= 0.005 (below that the ratio is noise). CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native CE within 0.002 of 3.13241 (instrument)
    pred_b_57_is_costliest       head 5.7's mean-ablation cost is the largest of the 162 (the ledger's deletion census: 0.916, eight times the next). Prior: likely
    pred_c_layer5_carries_most   layer 5's nine heads sum to the largest per-layer mean-ablation cost. Prior: likely
    pred_d_median_head_small     the median head costs <= 0.01 under mean ablation (most heads are individually cheap; redundancy). Prior: likely
    pred_e_kernel_recovery_median among heads with mean-ablation cost >= 0.02, the median kernel-program recovery >= 0.5. Prior: unsure
PRICE (registered maximum): 2 capture forwards (64 fit rows) — the means come from 480 rows in 15 forwards — total 15 capture + (1 + 162) x 6 = 978 eval
forwards; total 993 forwards; 0 backwards; 0 fits. Bar <= 1010.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
OUT_PT = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
KERNEL_RESULTS = ["embedding_forward_kernel_bank_v629_result.json", "embedding_forward_kernel_atlas_deep_v634_result.json", "embedding_forward_kernel_atlas_mid_v641_result.json",
                  "embedding_forward_kernel_atlas_v642_result.json", "embedding_forward_kernel_atlas_v643_result.json", "embedding_forward_kernel_atlas_v644_result.json"]
NATIVE_V615 = 3.13241
CANDIDATE_ID = "attention.mean_ablation_v701"
FORWARDS_MAX = 1010
EBATCH = 32
LAYERS = tuple(range(18))
REPLAY_TOL, MEDIAN_MAX, VALUE_FLOOR, REC_MIN = 0.002, 0.01, 0.02, 0.5
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_57_is_costliest": "max of 162", "pred_c_layer5_carries_most": "max layer sum",
               "pred_d_median_head_small": "median <= 0.01", "pred_e_kernel_recovery_median": "median recovery >= 0.5 over heads >= 0.02"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"replay_tol": REPLAY_TOL, "median_max": MEDIAN_MAX, "value_floor": VALUE_FLOOR, "rec_min": REC_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    H = model.config.n_head; D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long(); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    with torch.no_grad():
        state = {"ablate": None, "capture": False}
        sums = {l: torch.zeros(H, hd, dtype=torch.float64, device=dev) for l in LAYERS}; n_tok = 0
        natives = {l: blocks[l].attn.squared_attention for l in LAYERS}
        means = {}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                z = torch.einsum("bhqk,bkhd->bhqd", pat, v)
                if state["capture"]:
                    sums[l] += z.double().sum(dim=(0, 2))
                if state["ablate"] is not None and state["ablate"][0] == l:
                    h = state["ablate"][1]; z = z.clone(); z[:, h] = means[(l, h)]
                return z
            return patched

        for l in LAYERS:
            blocks[l].attn.squared_attention = make_patched(l)
        state["capture"] = True
        for s in range(0, fit.shape[0], EBATCH):
            idx = fit[s:s + EBATCH, :-1].to(dev); model(idx, fit[s:s + EBATCH, 1:].to(dev)); forwards += 1; n_tok += idx.numel()
        state["capture"] = False
        for l in LAYERS:
            for h in range(H):
                means[(l, h)] = (sums[l][h] / n_tok).float()

        def ce(ablate):
            state["ablate"] = ablate; total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            state["ablate"] = None
            return total / n, fw

        native, fw = ce(None); forwards += fw
        value = {}
        for l in LAYERS:
            for h in range(H):
                v_, fw = ce((l, h)); forwards += fw; value[f"{l}.{h}"] = v_ - native
            print(f"layer {l} mean-ablation costs: " + " ".join(f"{l}.{h}={value[f'{l}.{h}']:+.4f}" for h in range(H)))
        for l in LAYERS:
            blocks[l].attn.squared_attention = natives[l]
        disk_guard.guard_torch_save({f"mean_{l}_{h}": means[(l, h)].cpu() for l in LAYERS for h in range(H)}, str(OUT_PT), "v701 head means")
    kernel = {}
    for f in KERNEL_RESULTS:
        kernel.update({k: v for k, v in json.load(open(ROOT / "circuits/followups" / f))["report"]["edits"].items() if "." in k})
    recovery = {k: (1 - kernel[k] / value[k]) if value[k] >= 0.005 and k in kernel else None for k in value}
    layer_sum = {l: sum(value[f"{l}.{h}"] for h in range(H)) for l in LAYERS}
    vals = sorted(value.values()); median = vals[len(vals) // 2]
    big = [recovery[k] for k in value if value[k] >= VALUE_FLOOR and recovery[k] is not None]; rec_med = sorted(big)[len(big) // 2] if big else float("nan")
    top = sorted(value, key=value.get, reverse=True)[:12]
    print(f"native {native:.5f} | top heads by mean-ablation cost: " + " ".join(f"{k}:{value[k]:.3f}" for k in top))
    print(f"layer sums: " + " ".join(f"{l}:{layer_sum[l]:.3f}" for l in LAYERS) + f" | median head {median:.4f} | heads >= 0.02: {len(big)}, median kernel recovery {rec_med:.2f}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_57_is_costliest": max(value, key=value.get) == "5.7",
                   "pred_c_layer5_carries_most": max(layer_sum, key=layer_sum.get) == 5, "pred_d_median_head_small": median <= MEDIAN_MAX,
                   "pred_e_kernel_recovery_median": bool(big) and rec_med >= REC_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_mean_ablation_result_v701", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "mean_ablation_cost": value, "kernel_cost": kernel, "kernel_recovery": recovery, "layer_sum": {str(k): v for k, v in layer_sum.items()}, "median_head": median, "median_recovery_big": rec_med, "n_big": len(big)},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
