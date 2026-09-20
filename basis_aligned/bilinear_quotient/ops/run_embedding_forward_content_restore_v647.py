#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_baseline_replays_v646 pred_c_layer5_restores_most pred_d_early_layers_matter pred_e_restorations_superadditive
"""Embedding-forward folding, rung 36 (v647): which layers' content carries the 0.96? Restoring native patterns onto the all-fitted-kernels model.

v646: with all 162 attention patterns replaced by fitted positional kernels the model costs +0.956. The mirror of the bank census (which removed one
layer's content from the native model): starting from the ALL-KERNELS model, restore the native patterns of ONE layer at a time (18 configs) and
of groups (layers 0-4; layers 0-5; layers 6-17), and measure how much of the 0.956 each restoration recovers. This is the joint-context importance
of a layer's pattern content. Fitted kernels are v646's (arm all162). CE on 192 x 512 skip7000. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native CE within 0.002 of 3.13241 (instrument)
    pred_b_baseline_replays_v646   the all-kernels baseline replays v646's endpoint within 0.005 (0.956) (instrument)
    pred_c_layer5_restores_most    restoring layer 5 alone recovers the most of any single layer, and >= 0.15. Prior: unsure
    pred_d_early_layers_matter     restoring layers 0-4 together recovers >= 0.40. Prior: likely (their content was 0.37 when everything else was native)
    pred_e_restorations_superadditive the sum of the 18 single-layer recoveries <= 0.956 (single restorations under-recover; content is needed jointly). Prior: unsure
PRICE (registered maximum): (1 native + 1 baseline + 18 singles + 3 groups) x 6 = 138 forwards; 0 backwards; 0 fits. Bar <= 150.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_content_restore_v647_result.json"
KER = ROOT / "circuits/followups/embedding_forward_model_kernel_refit_v646_kernels.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615, V646_ALL = 3.13241, 0.9562
CANDIDATE_ID = "embedding_forward.content_restore_v647"
FORWARDS_MAX = 150
EBATCH = 32
LAYERS = tuple(range(18))
REPLAY_TOL, BASE_TOL, L5_MIN, EARLY_MIN = 0.002, 0.005, 0.15, 0.40
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_baseline_replays_v646": "+-0.005 of 0.956", "pred_c_layer5_restores_most": "max single and >= 0.15",
               "pred_d_early_layers_matter": ">= 0.40", "pred_e_restorations_superadditive": "sum of singles <= 0.956"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"replay_tol": REPLAY_TOL, "base_tol": BASE_TOL, "l5_min": L5_MIN, "early_min": EARLY_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    H = model.config.n_head; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    ker = torch.load(KER, map_location=dev)["all162"]; kbar = {(l, h): ker[f"{l}.{h}"].to(dev) for l in LAYERS for h in range(H)}
    with torch.no_grad():
        state = {"kernel_layers": ()}
        natives = {l: blocks[l].attn.squared_attention for l in LAYERS}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                if l in state["kernel_layers"]:
                    pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (causal & (dmat > 0))[None]
                    for h in range(Hn):
                        pat[:, h] = torch.where(off, kbar[(l, h)][dmat][None].expand(Bn, -1, -1), pat[:, h])
                return torch.einsum("bhqk,bkhd->bhqd", pat, v)
            return patched

        for l in LAYERS:
            blocks[l].attn.squared_attention = make_patched(l)

        def ce(kernel_layers):
            state["kernel_layers"] = tuple(kernel_layers); total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            return total / n, fw

        native, fw = ce(()); forwards += fw
        base, fw = ce(LAYERS); forwards += fw; base_added = base - native
        rec = {}
        for l in LAYERS:
            v_, fw = ce(tuple(x for x in LAYERS if x != l)); forwards += fw; rec[f"restore_{l}"] = base - v_
        for name, grp in (("restore_0_4", range(5)), ("restore_0_5", range(6)), ("restore_6_17", range(6, 18))):
            v_, fw = ce(tuple(x for x in LAYERS if x not in grp)); forwards += fw; rec[name] = base - v_
        for l in LAYERS:
            blocks[l].attn.squared_attention = natives[l]
        singles = {l: rec[f"restore_{l}"] for l in LAYERS}
        print(f"native {native:.5f} | all-kernels baseline +{base_added:.4f} | single-layer recoveries: " + " ".join(f"{l}:{singles[l]:.3f}" for l in LAYERS))
        print(f"groups: 0-4 {rec['restore_0_4']:.3f}, 0-5 {rec['restore_0_5']:.3f}, 6-17 {rec['restore_6_17']:.3f} | sum of singles {sum(singles.values()):.3f}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_baseline_replays_v646": abs(base_added - V646_ALL) <= BASE_TOL,
                   "pred_c_layer5_restores_most": max(singles, key=singles.get) == 5 and singles[5] >= L5_MIN, "pred_d_early_layers_matter": rec["restore_0_4"] >= EARLY_MIN,
                   "pred_e_restorations_superadditive": sum(singles.values()) <= base_added}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_content_restore_result_v647", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "all_kernels_added": base_added, "recoveries": rec},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
