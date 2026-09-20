#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_twelve_head_program_replays pred_c_layer5_bank_replays pred_d_all162_fitted_replays pred_e_mlp_program_replays
"""Embedding-forward folding, rung 51 (v663): LANE CANARY — one run that replays the lane's registered numbers from its saved artifacts.

Replays, on the 192 x 512 skip7000 rows (CE ADDED above native, lower is better):
    native CE                                                         3.13241 (v615)
    twelve-head block-0/1 program (v623/v624 tables; 1.8 := -1/i)      +0.0364 (v627)
    layer-5 kernel bank (v634 kernels)                                 +0.163  (v634)
    all 162 fitted kernels (v646 arm all162)                           +0.956  (v646)
    early-MLP (256, 512, 256) program at v650's fitted maps (endpoint) +0.2027 (v650 step 300; the validation-chosen step-75 maps were not saved)
Use: run after any change to the runner, the model load path, or the artifacts, to confirm the lane's state; every number should replay within
0.003 (the canary tolerance covers float non-determinism in batched attention).
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays              |native - 3.13241| <= 0.002
    pred_b_twelve_head_program_replays |added - 0.0364| <= 0.003
    pred_c_layer5_bank_replays         |added - 0.1632| <= 0.003
    pred_d_all162_fitted_replays       |added - 0.9562| <= 0.003
    pred_e_mlp_program_replays         |added - 0.2027| <= 0.003
PRICE (registered maximum): 5 configs x 6 eval batches = 30 forwards; 0 backwards; 0 fits. Bar <= 34.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_lane_canary_v663_result.json"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
T0 = ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt"
T1 = ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt"
K5 = ROOT / "circuits/followups/embedding_forward_kernel_atlas_deep_v634_kernels.pt"
KALL = ROOT / "circuits/followups/embedding_forward_model_kernel_refit_v646_kernels.pt"
MLP = ROOT / "circuits/followups/embedding_forward_mlp_program_refit_v650_maps.pt"
FRAMES = ROOT / "circuits/followups/embedding_forward_layer_sweep_v618_frames.pt"
REF = {"native": 3.13241, "twelve": 0.0364, "bank5": 0.1632, "all162": 0.9562, "mlp": 0.2027}
CANDIDATE_ID = "embedding_forward.lane_canary_v663"
FORWARDS_MAX = 34
EBATCH = 32
TOL_NATIVE, TOL = 0.002, 0.003
L0_SET, L1_SET, MEAN_HEAD = (3, 4, 6, 7, 8), (0, 1, 3, 5, 6, 7), 8
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_twelve_head_program_replays": "+-0.003", "pred_c_layer5_bank_replays": "+-0.003",
               "pred_d_all162_fitted_replays": "+-0.003", "pred_e_mlp_program_replays": "+-0.003"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "reference": REF, "bars": {"tol_native": TOL_NATIVE, "tol": TOL}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    H = model.config.n_head; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    t0_ = torch.load(T0, map_location=dev); t1_ = torch.load(T1, map_location=dev); k5 = torch.load(K5, map_location=dev); kall = torch.load(KALL, map_location=dev)["all162"]
    mlp = torch.load(MLP, map_location=dev); fr = torch.load(FRAMES, map_location=dev)
    tables = {0: {h: (t0_[f"head{h}_A"], t0_[f"head{h}_B"], t0_[f"head{h}_kappa"]) for h in L0_SET}, 1: {h: (t1_[f"head{h}_A"], t1_[f"head{h}_B"], t1_[f"head{h}_kappa"]) for h in L1_SET}}
    with torch.no_grad():
        state = {"idx": None, "mode": None}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        natives = {l: blocks[l].attn.squared_attention for l in range(18)}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                mode = state["mode"]
                if mode:
                    pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (causal & (dmat > 0))[None]; idx = state["idx"]
                    if mode == "twelve" and l in (0, 1):
                        for h in tables[l]:
                            A, Bt, kappa = tables[l][h]; prog = kappa[dmat][None] * A[idx][:, :, None] * Bt[idx][:, None, :]
                            pat[:, h] = torch.where(off, prog, pat[:, h])
                        if l == 1:
                            pat[:, MEAN_HEAD] = torch.where(off, (-1.0 / pos.clamp_min(1).float())[None, :, None].expand(Bn, Tn, Tn), pat[:, MEAN_HEAD])
                    elif mode == "bank5" and l == 5:
                        for h in range(Hn):
                            pat[:, h] = torch.where(off, k5[f"kbar_5_{h}"][dmat][None].expand(Bn, -1, -1), pat[:, h])
                    elif mode == "all162":
                        for h in range(Hn):
                            pat[:, h] = torch.where(off, kall[f"{l}.{h}"][dmat][None].expand(Bn, -1, -1), pat[:, h])
                return torch.einsum("bhqk,bkhd->bhqd", pat, v)
            return patched

        for l in range(18):
            blocks[l].attn.squared_attention = make_patched(l)
        means = {(l, key): fr[f"mlp{l}_{key}_mean"].float() for l in (0, 1, 2) for key in ("x", "m", "w")}
        maps = {(l, key): (mlp[f"mlp{l}_{key}_A"].to(dev), mlp[f"mlp{l}_{key}_B"].to(dev)) for l in (0, 1, 2) for key in ("x", "m", "w")}
        hooks = []
        for l in (0, 1, 2):
            m_ = blocks[l].mlp
            hooks.append(m_.register_forward_pre_hook(lambda m, a, l=l: (means[(l, "x")] + (a[0] - means[(l, "x")]) @ maps[(l, "x")][0] @ maps[(l, "x")][1].T,) if state["mode"] == "mlp" else None))
            hooks.append(m_.Down.register_forward_pre_hook(lambda m, a, l=l: (means[(l, "m")] + (a[0] - means[(l, "m")]) @ maps[(l, "m")][0] @ maps[(l, "m")][1].T,) if state["mode"] == "mlp" else None))
            hooks.append(m_.register_forward_hook(lambda m, a, o, l=l: (means[(l, "w")] + (o - means[(l, "w")]) @ maps[(l, "w")][0] @ maps[(l, "w")][1].T) if state["mode"] == "mlp" else None))

        def ce(mode):
            state["mode"] = mode; total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            state["mode"] = None
            return total / n, fw

        native, fw = ce(None); forwards += fw; got = {"native": native}
        for mode in ("twelve", "bank5", "all162", "mlp"):
            v_, fw = ce(mode); forwards += fw; got[mode] = v_ - native
        for l in range(18):
            blocks[l].attn.squared_attention = natives[l]
        for h in hooks:
            h.remove();
        pre.remove()
        print("canary: " + " | ".join(f"{k}: {got[k]:.4f} (ref {REF[k]:.4f}, diff {got[k] - REF[k]:+.4f})" for k in got))
    predictions = {"pred_a_native_replays": abs(got["native"] - REF["native"]) <= TOL_NATIVE, "pred_b_twelve_head_program_replays": abs(got["twelve"] - REF["twelve"]) <= TOL,
                   "pred_c_layer5_bank_replays": abs(got["bank5"] - REF["bank5"]) <= TOL, "pred_d_all162_fitted_replays": abs(got["all162"] - REF["all162"]) <= TOL,
                   "pred_e_mlp_program_replays": abs(got["mlp"] - REF["mlp"]) <= TOL}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_lane_canary_result_v663", "candidate_id": CANDIDATE_ID, "plan": plan, "report": {"got": got, "reference": REF},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
