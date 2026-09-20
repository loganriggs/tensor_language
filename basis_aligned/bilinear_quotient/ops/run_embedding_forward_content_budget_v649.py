#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_baseline_replays_v646 pred_c_k18_recovers_half pred_d_k54_recovers_most pred_e_ranking_transfers
"""Embedding-forward folding, rung 38 (v649): the content budget of bilin18's attention — how many heads need content-dependent patterns?

v646-v648: with all 162 patterns replaced by fitted positional kernels the model costs +0.956; restoring heads one at a time recovers modest,
overlapping amounts (8.3 +0.058, 14.4 +0.044, 5.7 -0.154). This rung builds the priced program: rank every head by its single restoration
recovery measured on FIT rows (96 skip80 rows, 3 batches per config, 162 configs), keep the top-k native, fitted kernels elsewhere, and price
k = 9, 18, 36, 54 on the held-out 192 x 512 skip7000 rows (never used for the ranking). Heads with negative fit-row recovery (harmful alone) are
ranked last. Price of a kept head: its native QK weights (4 x 1152 x 128 = 590k values) vs 513 for a kernel. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native CE within 0.002 of 3.13241 (instrument)
    pred_b_baseline_replays_v646 the all-kernels baseline on held-out within 0.005 of 0.956 (instrument)
    pred_c_k18_recovers_half     keeping the top 18 heads native (11%) recovers >= 0.5 x 0.956 on held-out. Prior: unsure
    pred_d_k54_recovers_most     keeping the top 54 (one third) recovers >= 0.8 x 0.956. Prior: unsure
    pred_e_ranking_transfers     the k = 18 held-out recovery >= 0.8 x the sum of its members' fit-row single recoveries is NOT expected (overlap);
                                 registered instead: held-out recovery at k = 18 >= the held-out recovery of the same k = 18 chosen at random (a
                                 seed-640 random subset), by >= 0.10. Prior: likely
PRICE (registered maximum): ranking 162 configs x 3 batches (96 fit rows) = 486 forwards + native/baseline on fit rows 6; held-out: native + baseline
+ 4 top-k + 1 random-18 = 7 x 6 = 42; total 534 forwards; 0 backwards; 0 fits (the ranking is a measurement, not an optimisation). Bar <= 550.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_content_budget_v649_result.json"
KER = ROOT / "circuits/followups/embedding_forward_model_kernel_refit_v646_kernels.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615, V646_ALL = 3.13241, 0.9562
CANDIDATE_ID = "embedding_forward.content_budget_v649"
FORWARDS_MAX = 550
EBATCH, N_RANK = 32, 96
LAYERS = tuple(range(18))
KS = (9, 18, 36, 54)
REPLAY_TOL, BASE_TOL, HALF, MOST, RAND_GAP = 0.002, 0.005, 0.5, 0.8, 0.10
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_baseline_replays_v646": "+-0.005", "pred_c_k18_recovers_half": ">= 0.5 x 0.956",
               "pred_d_k54_recovers_most": ">= 0.8 x 0.956", "pred_e_ranking_transfers": "top-18 beats random-18 by >= 0.10"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "ks": list(KS), "n_rank_rows": N_RANK,
            "bars": {"replay_tol": REPLAY_TOL, "base_tol": BASE_TOL, "half": HALF, "most": MOST, "rand_gap": RAND_GAP}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    H = model.config.n_head; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long()[:N_RANK]; ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    ker = torch.load(KER, map_location=dev)["all162"]; kbar = {(l, h): ker[f"{l}.{h}"].to(dev) for l in LAYERS for h in range(H)}
    with torch.no_grad():
        state = {"kernel_heads": set()}
        natives = {l: blocks[l].attn.squared_attention for l in LAYERS}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                heads = [h for h in range(Hn) if (l, h) in state["kernel_heads"]]
                if heads:
                    pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (causal & (dmat > 0))[None]
                    for h in heads:
                        pat[:, h] = torch.where(off, kbar[(l, h)][dmat][None].expand(Bn, -1, -1), pat[:, h])
                return torch.einsum("bhqk,bkhd->bhqd", pat, v)
            return patched

        for l in LAYERS:
            blocks[l].attn.squared_attention = make_patched(l)

        def ce(rows, kernel_heads):
            state["kernel_heads"] = set(kernel_heads); total = 0.0; n = 0; fw = 0
            for s in range(0, rows.shape[0], EBATCH):
                idx = rows[s:s + EBATCH, :-1].to(dev); total += float(model(idx, rows[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            return total / n, fw

        ALLK = {(l, h) for l in LAYERS for h in range(H)}
        # ---- ranking on fit rows ---------------------------------------------------------------------------------------------------
        nat_f, fw = ce(fit, set()); forwards += fw; base_f, fw = ce(fit, ALLK); forwards += fw
        rec_fit = {}
        for key in sorted(ALLK):
            v_, fw = ce(fit, ALLK - {key}); forwards += fw; rec_fit[key] = base_f - v_
        order = sorted(ALLK, key=lambda k: -rec_fit[k])
        print(f"fit rows: native {nat_f:.4f}, all-kernels +{base_f - nat_f:.4f}; top 20 by restoration: " + " ".join(f"{l}.{h}:{rec_fit[(l, h)]:+.3f}" for (l, h) in order[:20]))
        # ---- held-out pricing ------------------------------------------------------------------------------------------------------
        native, fw = ce(ev, set()); forwards += fw; base, fw = ce(ev, ALLK); forwards += fw; base_added = base - native
        res = {}
        for k in KS:
            keep = set(order[:k]); v_, fw = ce(ev, ALLK - keep); forwards += fw; res[f"top{k}"] = {"ce_added": v_ - native, "recovered": base - v_, "heads": [f"{l}.{h}" for (l, h) in order[:k]],
                                                                                            "price_values": k * 4 * 1152 * 128 + (162 - k) * 513}
        gen = torch.Generator().manual_seed(640); perm = torch.randperm(162, generator=gen).tolist(); rand18 = {sorted(ALLK)[i] for i in perm[:18]}
        v_, fw = ce(ev, ALLK - rand18); forwards += fw; res["random18"] = {"ce_added": v_ - native, "recovered": base - v_, "heads": [f"{l}.{h}" for (l, h) in sorted(rand18)]}
        for l in LAYERS:
            blocks[l].attn.squared_attention = natives[l]
        print(f"held-out: native {native:.5f}, all-kernels +{base_added:.4f} | " + " ".join(f"top{k}: +{res[f'top{k}']['ce_added']:.4f} (recovers {res[f'top{k}']['recovered']:.3f})" for k in KS) + f" | random18: +{res['random18']['ce_added']:.4f}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_baseline_replays_v646": abs(base_added - V646_ALL) <= BASE_TOL,
                   "pred_c_k18_recovers_half": res["top18"]["recovered"] >= HALF * base_added, "pred_d_k54_recovers_most": res["top54"]["recovered"] >= MOST * base_added,
                   "pred_e_ranking_transfers": res["top18"]["recovered"] - res["random18"]["recovered"] >= RAND_GAP}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_content_budget_result_v649", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "all_kernels_added": base_added, "fit_ranking": {f"{l}.{h}": rec_fit[(l, h)] for (l, h) in order}, "budget": res},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
