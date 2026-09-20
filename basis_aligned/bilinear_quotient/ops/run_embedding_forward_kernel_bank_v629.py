#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_layer0_content_heads_need_content pred_c_layer1_content_head_needs_content pred_d_layer2_bank_cheap pred_e_union_composes
"""Embedding-forward folding, rung 21 (v629): how much of blocks 0-2 attention is a bank of FIXED POSITIONAL KERNELS?

v628: at layer 2, replacing a head's off-diagonal pattern by its real mean kernel per offset (one 512-vector, no token dependence) costs <= 0.009
for eight of nine heads. This rung does the same for all 27 heads of blocks 0-2: capture the real patterns on 64 fit rows (2 forwards), form
kbar_h(d) per head, and price on 192 x 512 skip7000: every head singly, each layer's 9-head bank, and the 27-head union. Where the kernel-only
edit is expensive, the head needs token or content dependence; where it is cheap, the head is a positional filter to the loss. Price of the
whole bank: 27 x 512 = 13.8k values (vs 27 x 4 x 1152 x 128 = 16M native pattern weights). CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays                 native CE within 0.002 of 3.13241 (instrument)
    pred_b_layer0_content_heads_need_content kernel-only costs >= 0.02 for each of 0.0, 0.1, 0.2, 0.5 AND <= 0.01 for each of 0.3, 0.4, 0.6, 0.7, 0.8. Prior: unsure
    pred_c_layer1_content_head_needs_content kernel-only costs >= 0.02 for 1.4 AND <= 0.01 for each of the other eight layer-1 heads. Prior: unsure
    pred_d_layer2_bank_cheap              the layer-2 nine-head bank costs <= 0.06 (singles sum to ~0.047 in v628). Prior: likely
    pred_e_union_composes                 the 27-head union costs <= 2 x (layer-0 bank + layer-1 bank + layer-2 bank). Prior: unsure
PRICE (registered maximum): 2 capture forwards + (1 + 27 + 3 + 1) x 6 = 192 eval forwards; total 194; 0 backwards; 0 fits. Bar <= 200.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_kernel_bank_v629_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_kernel_bank_v629_kernels.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.kernel_bank_v629"
FORWARDS_MAX = 200
EBATCH, N_CAP, Q_MIN = 32, 64, 8
LAYERS = (0, 1, 2)
L0_CONTENT, L0_POS, L1_CONTENT = (0, 1, 2, 5), (3, 4, 6, 7, 8), (4,)
REPLAY_TOL, NEED, FREE, BANK2_MAX, UNION_RATIO = 0.002, 0.02, 0.01, 0.06, 2.0
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_layer0_content_heads_need_content": ">= 0.02 content / <= 0.01 positional", "pred_c_layer1_content_head_needs_content": "1.4 >= 0.02, others <= 0.01",
               "pred_d_layer2_bank_cheap": "<= 0.06", "pred_e_union_composes": "<= 2 x sum of banks"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "layers": list(LAYERS),
            "bars": {"replay_tol": REPLAY_TOL, "need": NEED, "free": FREE, "bank2_max": BANK2_MAX, "union_ratio": UNION_RATIO}}
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
        state["capture"] = {l: [] for l in LAYERS}
        for s in range(0, N_CAP, EBATCH):
            idx = fit[s:s + EBATCH, :-1].to(dev); model(idx, fit[s:s + EBATCH, 1:].to(dev)); forwards += 1
        reals = {l: torch.cat(state["capture"][l]) for l in LAYERS}; state["capture"] = None
        Tn = reals[0].shape[-1]; pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0)
        masks = {d: ((dmat == d) & (pos >= Q_MIN)[:, None]) for d in range(1, Tn)}
        rowsum = {}
        for l in LAYERS:
            for h in range(H):
                kb = torch.zeros(Tn + 1, device=dev); R = reals[l][:, h]
                for d in range(1, Tn):
                    kb[d] = float(R[:, masks[d]].mean())
                state["kbar"][(l, h)] = kb; rowsum[(l, h)] = float((R * (dmat > 0)[None]).sum(-1)[:, Q_MIN:].mean())
        del reals

        def ce(cfg):
            state["heads"] = cfg; total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            return total / n, fw

        empty = {l: () for l in LAYERS}
        native, fw = ce(dict(empty)); forwards += fw; e = {}
        for l in LAYERS:
            for h in range(H):
                cfg = dict(empty); cfg[l] = (h,); v_, fw = ce(cfg); forwards += fw; e[f"{l}.{h}"] = v_ - native
            print(f"layer {l} kernel-only singles: " + " ".join(f"{l}.{h}={e[f'{l}.{h}']:+.4f}(sum{rowsum[(l, h)]:+.2f})" for h in range(H)))
        for l in LAYERS:
            cfg = dict(empty); cfg[l] = tuple(range(H)); v_, fw = ce(cfg); forwards += fw; e[f"bank{l}"] = v_ - native
        v_, fw = ce({l: tuple(range(H)) for l in LAYERS}); forwards += fw; e["union"] = v_ - native
        for l in LAYERS:
            blocks[l].attn.squared_attention = natives[l]
        print(f"native {native:.5f} | banks: " + " ".join(f"layer{l}={e[f'bank{l}']:+.4f}" for l in LAYERS) + f" | union {e['union']:+.4f}")
        disk_guard.guard_torch_save({f"kbar_{l}_{h}": state["kbar"][(l, h)].cpu() for l in LAYERS for h in range(H)} | {"rowsum": {f"{l}.{h}": v for (l, h), v in rowsum.items()}}, str(OUT_PT), "v629 kernels")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_layer0_content_heads_need_content": all(e[f"0.{h}"] >= NEED for h in L0_CONTENT) and all(e[f"0.{h}"] <= FREE for h in L0_POS),
                   "pred_c_layer1_content_head_needs_content": all(e[f"1.{h}"] >= NEED for h in L1_CONTENT) and all(e[f"1.{h}"] <= FREE for h in range(H) if h not in L1_CONTENT),
                   "pred_d_layer2_bank_cheap": e["bank2"] <= BANK2_MAX,
                   "pred_e_union_composes": e["union"] <= UNION_RATIO * sum(e[f"bank{l}"] for l in LAYERS)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_kernel_bank_result_v629", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": e, "rowsum": {f"{l}.{h}": v for (l, h), v in rowsum.items()}, "price_values": 27 * 512},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
