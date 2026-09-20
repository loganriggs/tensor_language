"""Embedding-forward folding, rung 43 (v654): robustness of the kernel atlas to the kernel-estimation rows.

Every atlas number (v629, v634, v641-v644) used kernels estimated on the FIRST 64 rows of the skip80 cache. This rung re-estimates the kernels
for layers 0-2 and 5 on a DISJOINT 64 rows (rows 64-127) and re-prices the singles and the four banks on the same held-out rows. If the numbers move
little, the atlas is a property of the model, not of the 64 rows. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays   native CE within 0.002 of 3.13241 (instrument)
    pred_b_singles_stable   >= 90% of the 36 single-head costs are within +-0.005 of their first-64-row values (v629 / v634). Prior: likely
    pred_c_banks_stable     each of the four banks within +-0.015 of its first-64-row value (0.032 / 0.034 / 0.049 / 0.163). Prior: likely
    pred_d_57_still_critical head 5.7 single cost >= 0.05. Prior: likely
    pred_e_layer5_bank_stable layer-5 bank >= 0.12. Prior: likely
PRICE (registered maximum): 2 capture forwards + (1 + 36 + 4) x 6 = 246 eval forwards; total 248; 0 backwards; 0 fits. Bar <= 260.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_atlas_robustness_v654_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_atlas_robustness_v654_kernels.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.atlas_robustness_v654"
FORWARDS_MAX = 260
ROW_OFFSET = 64
REF = {0: ROOT / "circuits/followups/embedding_forward_kernel_bank_v629_result.json", 5: ROOT / "circuits/followups/embedding_forward_kernel_atlas_deep_v634_result.json"}
EBATCH, N_CAP, Q_MIN = 32, 64, 8
LAYERS = (0, 1, 2, 5)
REPLAY_TOL, SINGLE_TOL, SINGLE_FRAC, BANK_TOL, CRIT, L5_MIN = 0.002, 0.005, 0.9, 0.015, 0.05, 0.12
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_singles_stable": ">= 90% within 0.005", "pred_c_banks_stable": "within 0.015 x 4",
               "pred_d_57_still_critical": ">= 0.05", "pred_e_layer5_bank_stable": ">= 0.12"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "layers": list(LAYERS),
            "bars": {"replay_tol": REPLAY_TOL, "single_tol": SINGLE_TOL, "single_frac": SINGLE_FRAC, "bank_tol": BANK_TOL, "crit": CRIT, "l5_min": L5_MIN}, "row_offset": ROW_OFFSET}
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
        for s in range(ROW_OFFSET, ROW_OFFSET + N_CAP, EBATCH):
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
        disk_guard.guard_torch_save({f"kbar_{l}_{h}": state["kbar"][(l, h)].cpu() for l in LAYERS for h in range(H)} | {"rowsum": {f"{l}.{h}": v for (l, h), v in rowsum.items()}}, str(OUT_PT), "v654 kernels")
    ref = {}
    for src in set(REF.values()):
        ref.update(json.load(open(src))["report"]["edits"])
    diffs = {k: e[k] - ref[k] for k in e if k in ref}
    singles_ok = sum(1 for l in LAYERS for h in range(H) if abs(diffs[f"{l}.{h}"]) <= SINGLE_TOL) / (len(LAYERS) * H)
    print("max |diff| singles:", max(abs(diffs[f"{l}.{h}"]) for l in LAYERS for h in range(H)), "| bank diffs:", {l: round(diffs[f"bank{l}"], 4) for l in LAYERS}, "| fraction of singles within 0.005:", singles_ok)
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_singles_stable": singles_ok >= SINGLE_FRAC,
                   "pred_c_banks_stable": all(abs(diffs[f"bank{l}"]) <= BANK_TOL for l in LAYERS), "pred_d_57_still_critical": e["5.7"] >= CRIT, "pred_e_layer5_bank_stable": e["bank5"] >= L5_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_atlas_robustness_result_v654", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": e, "diffs_vs_first64": diffs, "rowsum": {f"{l}.{h}": v for (l, h), v in rowsum.items()}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
