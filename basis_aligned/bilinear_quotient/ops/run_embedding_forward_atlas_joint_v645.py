#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_deep_patterns_do_vary pred_c_all162_under_two pred_d_layer5_is_the_content pred_e_deep_union_cheap
"""Embedding-forward folding, rung 34 (v645): the atlas' two missing checks — do deep patterns actually vary, and what does the WHOLE model cost?

v629 / v634 / v641-v644: 159 of 162 heads are singly replaceable by their mean positional kernel at <= 0.011 nats; per-layer banks 0.004-0.05
except layer 5 (0.163). Two checks before this is a claim. (1) CONTROL: per head, the variance of the real off-diagonal pattern around its kernel
across 64 rows, relative to the kernel's energy — VR = E[(P - kbar)^2] / E[kbar^2] (queries >= 8). If deep-layer patterns barely vary (VR << 1),
freezing them is trivially cheap and says nothing about content. (2) JOINT: all 162 kernels at once; all but layer 5 (153); layers 6-17 (108);
layers 0-4 (45). Kernels are re-captured in one pass over all layers (2 forwards). CE on 192 x 512 skip7000. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays      native CE within 0.002 of 3.13241 (instrument)
    pred_b_deep_patterns_do_vary median VR over the 54 heads of layers 12-17 >= 0.5 (their patterns are genuinely context-dependent; the cheapness is
                               robustness, not constancy). Prior: unsure — this is the control
    pred_c_all162_under_two    all 162 kernels at once <= 2.0 nats. Prior: unsure
    pred_d_layer5_is_the_content all-but-layer-5 <= all-162 - 0.10 (layer 5's patterns carry a distinct, non-redundant share). Prior: likely
    pred_e_deep_union_cheap    layers 6-17 (108 heads) <= 0.30. Prior: unsure
PRICE (registered maximum): 2 capture forwards + (1 + 4) x 6 = 30 eval forwards; total 32; 0 backwards; 0 fits. Bar <= 36.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_atlas_joint_v645_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_atlas_joint_v645_kernels.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.atlas_joint_v645"
FORWARDS_MAX = 36
EBATCH, N_CAP, Q_MIN = 32, 64, 8
LAYERS = tuple(range(18))
REPLAY_TOL, VR_MIN, ALL_MAX, L5_GAP, DEEP_MAX = 0.002, 0.5, 2.0, 0.10, 0.30
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_deep_patterns_do_vary": "median VR (12-17) >= 0.5", "pred_c_all162_under_two": "<= 2.0",
               "pred_d_layer5_is_the_content": "all-but-5 <= all - 0.10", "pred_e_deep_union_cheap": "layers 6-17 <= 0.30"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"replay_tol": REPLAY_TOL, "vr_min": VR_MIN, "all_max": ALL_MAX, "l5_gap": L5_GAP, "deep_max": DEEP_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    H = model.config.n_head; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long(); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    with torch.no_grad():
        state = {"heads": {l: () for l in LAYERS}, "capture": None, "kbar": {}}
        natives = {l: blocks[l].attn.squared_attention for l in LAYERS}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                if state["capture"] is not None:
                    state["capture"][l].append(pat.detach().clone())
                if state["heads"][l]:
                    pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (causal & (dmat > 0))[None]
                    for h in state["heads"][l]:
                        pat[:, h] = torch.where(off, state["kbar"][(l, h)][dmat][None].expand(Bn, -1, -1), pat[:, h])
                return torch.einsum("bhqk,bkhd->bhqd", pat, v)
            return patched

        for l in LAYERS:
            blocks[l].attn.squared_attention = make_patched(l)
        # ---- capture all layers (one pass), kernels and variance ratios ------------------------------------------------------------
        state["capture"] = {l: [] for l in LAYERS}
        for s in range(0, N_CAP, EBATCH):
            idx = fit[s:s + EBATCH, :-1].to(dev); model(idx, fit[s:s + EBATCH, 1:].to(dev)); forwards += 1
        Tn = state["capture"][0][0].shape[-1]; pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0)
        qm = (pos >= Q_MIN); off_q = ((dmat > 0) & qm[:, None])
        # per-offset means via scatter over the offset index (fast): for each head, sum over rows and entries with the same d
        dflat = dmat[off_q]                                                                         # offsets of the counted entries
        counts = torch.bincount(dflat, minlength=Tn + 1).double()
        VR = {}
        for l in LAYERS:
            P = torch.cat(state["capture"][l]); state["capture"][l] = None                           # [B, H, T, T]
            for h in range(H):
                X = P[:, h][:, off_q]                                                                # [B, n]
                sums = torch.zeros(Tn + 1, dtype=torch.float64, device=dev).index_add_(0, dflat, X.double().sum(0))
                kb = torch.where(counts > 0, sums / counts.clamp_min(1) / P.shape[0], torch.zeros_like(sums)).float()
                state["kbar"][(l, h)] = kb
                resid = X - kb[dflat][None]
                VR[(l, h)] = float(resid.square().mean() / (kb[dflat].square().mean() + 1e-30))
            del P
        state["capture"] = None
        vr_deep = sorted(VR[(l, h)] for l in range(12, 18) for h in range(H)); med_deep = vr_deep[len(vr_deep) // 2]
        vr_by_layer = {l: sorted(VR[(l, h)] for h in range(H))[4] for l in LAYERS}
        print("median pattern-variance ratio by layer:", {l: round(v, 2) for l, v in vr_by_layer.items()}); print(f"median VR over layers 12-17: {med_deep:.3f}")

        def ce(cfg):
            state["heads"] = {l: cfg.get(l, ()) for l in LAYERS}; total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            return total / n, fw

        allh = tuple(range(H))
        native, fw = ce({}); forwards += fw; e = {}
        for name, ls in (("all162", LAYERS), ("all_but_5", tuple(l for l in LAYERS if l != 5)), ("layers6_17", tuple(range(6, 18))), ("layers0_4", tuple(range(5)))):
            v_, fw = ce({l: allh for l in ls}); forwards += fw; e[name] = v_ - native
        for l in LAYERS:
            blocks[l].attn.squared_attention = natives[l]
        print(f"native {native:.5f} | joint kernel edits: " + " ".join(f"{k}={v:+.4f}" for k, v in e.items()))
        disk_guard.guard_torch_save({f"kbar_{l}_{h}": state["kbar"][(l, h)].cpu() for l in LAYERS for h in range(H)} | {"VR": {f"{l}.{h}": v for (l, h), v in VR.items()}}, str(OUT_PT), "v645 kernels")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_deep_patterns_do_vary": med_deep >= VR_MIN,
                   "pred_c_all162_under_two": e["all162"] <= ALL_MAX, "pred_d_layer5_is_the_content": e["all_but_5"] <= e["all162"] - L5_GAP, "pred_e_deep_union_cheap": e["layers6_17"] <= DEEP_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_atlas_joint_result_v645", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": e, "variance_ratio": {f"{l}.{h}": v for (l, h), v in VR.items()}, "median_vr_by_layer": {str(l): v for l, v in vr_by_layer.items()}, "median_vr_deep": med_deep},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
