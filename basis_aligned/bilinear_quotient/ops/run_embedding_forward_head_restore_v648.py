"""Embedding-forward folding, rung 37 (v648): per-head restorations onto the all-kernels model in layers 5, 8 and 14.

v647: restoring a whole layer's native patterns onto the all-fitted-kernels model (+0.956) recovers 0.151 (layer 8), 0.107 (layer 14) and -0.105
(layer 5, which helps only once layers 0-4 are native). This rung restores ONE HEAD at a time in those three layers (27 configs; the other
161 patterns stay fitted kernels) and, per layer, all nine (replaying v647). Recovery = baseline cost minus cost with that head native. The
joint-context mirror of the single-head census. CE on 192 x 512 skip7000. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native CE within 0.002 of 3.13241 (instrument)
    pred_b_baseline_replays_v646 the all-kernels baseline within 0.005 of 0.956 (instrument)
    pred_c_layer8_has_a_lead_head some layer-8 head alone recovers >= 0.05. Prior: unsure
    pred_d_57_hurts_alone        restoring 5.7 alone makes the model worse (recovery <= -0.02). Prior: likely (v647: the whole layer hurts)
    pred_e_head_sums_under_layer for layers 8 and 14, the sum of the nine single-head recoveries <= the whole-layer recovery + 0.02 (heads are needed
                                 together even within a layer). Prior: unsure
PRICE (registered maximum): (1 native + 1 baseline + 27 heads + 3 layers) x 6 = 192 forwards; 0 backwards; 0 fits. Bar <= 200.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_head_restore_v648_result.json"
KER = ROOT / "circuits/followups/embedding_forward_model_kernel_refit_v646_kernels.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615, V646_ALL = 3.13241, 0.9562
CANDIDATE_ID = "embedding_forward.head_restore_v648"
FORWARDS_MAX = 200
TARGET_LAYERS = (5, 8, 14)
EBATCH = 32
LAYERS = tuple(range(18))
REPLAY_TOL, BASE_TOL, LEAD_MIN, HURT_MAX, SUM_SLACK = 0.002, 0.005, 0.05, -0.02, 0.02
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_baseline_replays_v646": "+-0.005 of 0.956", "pred_c_layer8_has_a_lead_head": "some 8.h >= 0.05",
               "pred_d_57_hurts_alone": "<= -0.02", "pred_e_head_sums_under_layer": "sum(heads) <= layer + 0.02 for 8 and 14"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"replay_tol": REPLAY_TOL, "base_tol": BASE_TOL, "lead_min": LEAD_MIN, "hurt_max": HURT_MAX, "sum_slack": SUM_SLACK}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    H = model.config.n_head; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long()
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

        def ce(kernel_heads):
            state["kernel_heads"] = set(kernel_heads); total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            return total / n, fw

        ALLK = {(l, h) for l in LAYERS for h in range(H)}
        native, fw = ce(set()); forwards += fw
        base, fw = ce(ALLK); forwards += fw; base_added = base - native
        rec = {}
        for l in TARGET_LAYERS:
            for h in range(H):
                v_, fw = ce(ALLK - {(l, h)}); forwards += fw; rec[f"{l}.{h}"] = base - v_
            v_, fw = ce(ALLK - {(l, h) for h in range(H)}); forwards += fw; rec[f"layer{l}"] = base - v_
            print(f"layer {l}: whole-layer recovery {rec[f'layer{l}']:.3f} | heads: " + " ".join(f"{l}.{h}:{rec[f'{l}.{h}']:+.3f}" for h in range(H)) + f" | sum {sum(rec[f'{l}.{h}'] for h in range(H)):+.3f}")
        for l in LAYERS:
            blocks[l].attn.squared_attention = natives[l]
        print(f"native {native:.5f} | all-kernels baseline +{base_added:.4f}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_baseline_replays_v646": abs(base_added - V646_ALL) <= BASE_TOL,
                   "pred_c_layer8_has_a_lead_head": max(rec[f"8.{h}"] for h in range(H)) >= LEAD_MIN, "pred_d_57_hurts_alone": rec["5.7"] <= HURT_MAX,
                   "pred_e_head_sums_under_layer": all(sum(rec[f"{l}.{h}"] for h in range(H)) <= rec[f"layer{l}"] + SUM_SLACK for l in (8, 14))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_head_restore_result_v648", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "all_kernels_added": base_added, "recoveries": rec},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
