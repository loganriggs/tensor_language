"""Embedding-forward folding, rung 26 (v634): the fixed-kernel atlas continued to LAYERS 3-5 — where does positional structure stop paying?

v629: in blocks 0-2 every head is singly replaceable by its real mean kernel per offset (<= 0.009; the content head 2.5 at 0.025), and the
per-layer banks cost 0.03-0.05. This rung repeats the closed-form kernel-only census for layers 3, 4, 5 (capture on 64 fit rows; singles and
the 9-head bank per layer; CE on 192 x 512 skip7000) to see how the count of "positional" heads (single cost <= 0.01) and the bank cost change
with depth. CE ADDED, lower is better. No fitting; kernels are means of the real patterns.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native CE within 0.002 of 3.13241 (instrument)
    pred_b_fewer_cheap_heads_with_depth the number of heads with single cost <= 0.01 is non-increasing from layer 3 to 4 to 5 and at layer 5 is <= 6 of 9. Prior: unsure
    pred_c_banks_grow_with_depth   bank3 <= bank4 <= bank5. Prior: unsure
    pred_d_some_deep_positional_heads at each of layers 3-5 at least 3 heads have single cost <= 0.01 (fixed positional taps persist). Prior: likely
    pred_e_layer5_bank_expensive   the layer-5 bank costs >= 0.10 (content dependence dominates by layer 5). Prior: unsure
PRICE (registered maximum): 2 capture forwards + (1 + 27 + 3) x 6 = 186 eval forwards; total 188; 0 backwards; 0 fits. Bar <= 200.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_kernel_atlas_deep_v634_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_kernel_atlas_deep_v634_kernels.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.kernel_atlas_deep_v634"
FORWARDS_MAX = 200
EBATCH, N_CAP, Q_MIN = 32, 64, 8
LAYERS = (3, 4, 5)
REPLAY_TOL, FREE, MAX_CHEAP5, MIN_CHEAP, BANK5_MIN = 0.002, 0.01, 6, 3, 0.10
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_fewer_cheap_heads_with_depth": "non-increasing, <= 6 at layer 5", "pred_c_banks_grow_with_depth": "bank3 <= bank4 <= bank5",
               "pred_d_some_deep_positional_heads": ">= 3 cheap heads per layer", "pred_e_layer5_bank_expensive": ">= 0.10"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "layers": list(LAYERS),
            "bars": {"replay_tol": REPLAY_TOL, "free": FREE, "max_cheap5": MAX_CHEAP5, "min_cheap": MIN_CHEAP, "bank5_min": BANK5_MIN}}
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
        for l in LAYERS:
            blocks[l].attn.squared_attention = natives[l]
        print(f"native {native:.5f} | banks: " + " ".join(f"layer{l}={e[f'bank{l}']:+.4f}" for l in LAYERS))
        disk_guard.guard_torch_save({f"kbar_{l}_{h}": state["kbar"][(l, h)].cpu() for l in LAYERS for h in range(H)} | {"rowsum": {f"{l}.{h}": v for (l, h), v in rowsum.items()}}, str(OUT_PT), "v634 kernels")
    cheap = {l: sum(1 for h in range(H) if e[f"{l}.{h}"] <= FREE) for l in LAYERS}
    print("cheap heads per layer:", cheap)
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_fewer_cheap_heads_with_depth": cheap[3] >= cheap[4] >= cheap[5] and cheap[5] <= MAX_CHEAP5,
                   "pred_c_banks_grow_with_depth": e["bank3"] <= e["bank4"] <= e["bank5"],
                   "pred_d_some_deep_positional_heads": all(cheap[l] >= MIN_CHEAP for l in LAYERS),
                   "pred_e_layer5_bank_expensive": e["bank5"] >= BANK5_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_kernel_atlas_deep_result_v634", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": e, "cheap_per_layer": {str(k): v for k, v in cheap.items()}, "rowsum": {f"{l}.{h}": v for (l, h), v in rowsum.items()}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
