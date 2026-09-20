#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_positional_union_cheap pred_c_content_union_expensive pred_d_token_gating_helps_jointly pred_e_pair01_composes
"""Embedding-forward folding, rung 22 (v630): where does the 0.36 come from? Positional heads vs content heads, kernels vs token-gated tables.

v629: all 27 early heads are cheap to replace one at a time by fixed kernels, but the 27-head kernel union costs 0.355 (banks 0.032 / 0.034 /
0.049). This rung splits the union: (i) the 19 POSITIONAL heads (separable in v622 / v624 / v628: 0.3 0.4 0.6 0.7 0.8 | 1.0 1.1 1.3 1.5 1.6 1.7
1.8 | 2.0 2.2 2.3 2.4 2.6 2.7 2.8) kernel-only, content heads native; (ii) the 8 CONTENT heads (0.0 0.1 0.2 0.5 | 1.2 1.4 | 2.1 2.5) kernel-only,
positional heads native; (iii) the BEST-PROGRAM union — token-gated three-table programs where they were cheap (layer 0: 0.3 0.4 0.6 0.7 0.8;
layer 1: 1.0 1.1 1.3 1.5 1.6 1.7; layer 2: 2.0 2.2 2.3 2.4 2.8), fixed kernels for the running-mean / scale-broken heads (1.8, 2.6, 2.7), content
heads native — i.e. the same 19 heads as (i) but with token gating kept; (iv) the layer-0 + layer-1 kernel banks together (18 heads). All on the
192 x 512 skip7000 rows. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native CE within 0.002 of 3.13241 (instrument)
    pred_b_positional_union_cheap  (i) 19 positional heads kernel-only <= 0.10. Prior: unsure
    pred_c_content_union_expensive (ii) 8 content heads kernel-only >= 0.10 (the content dependence lives in the content heads, jointly). Prior: unsure
    pred_d_token_gating_helps_jointly (iii) best-program union <= 0.5 x (i) (token gating matters jointly even though singles did not show it). Prior: unsure
    pred_e_pair01_composes         (iv) layer-0 + layer-1 kernel banks <= 1.5 x (bank0 + bank1 re-measured) — compounding starts only with layer 2. Prior: unsure
PRICE (registered maximum): (1 + 4 + 2) x 6 = 42 eval forwards (kernels and tables loaded from v623/v624/v628/v629); 0 backwards; 0 fits. Bar <= 50.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_union_split_v630_result.json"
KER = ROOT / "circuits/followups/embedding_forward_kernel_bank_v629_kernels.pt"
TAB = {0: ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt", 1: ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt",
       2: ROOT / "circuits/followups/embedding_forward_layer2_filters_v628_tables.pt"}
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.union_split_v630"
FORWARDS_MAX = 50
EBATCH = 32
LAYERS = (0, 1, 2)
POSITIONAL = {0: (3, 4, 6, 7, 8), 1: (0, 1, 3, 5, 6, 7, 8), 2: (0, 2, 3, 4, 6, 7, 8)}
CONTENT = {0: (0, 1, 2, 5), 1: (2, 4), 2: (1, 5)}
TOKEN_PROG = {0: (3, 4, 6, 7, 8), 1: (0, 1, 3, 5, 6, 7), 2: (0, 2, 3, 4, 8)}
KERNEL_IN_BEST = {0: (), 1: (8,), 2: (6, 7)}
REPLAY_TOL, POS_MAX, CON_MIN, GATE_RATIO, PAIR_RATIO = 0.002, 0.10, 0.10, 0.5, 1.5
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_positional_union_cheap": "<= 0.10", "pred_c_content_union_expensive": ">= 0.10",
               "pred_d_token_gating_helps_jointly": "<= 0.5 x positional kernel union", "pred_e_pair01_composes": "<= 1.5 x (bank0 + bank1)"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "positional": {str(k): list(v) for k, v in POSITIONAL.items()},
            "content": {str(k): list(v) for k, v in CONTENT.items()}, "bars": {"replay_tol": REPLAY_TOL, "pos_max": POS_MAX, "con_min": CON_MIN, "gate_ratio": GATE_RATIO, "pair_ratio": PAIR_RATIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    H = model.config.n_head; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    ker = torch.load(KER, map_location=dev); kbar = {(l, h): ker[f"kbar_{l}_{h}"] for l in LAYERS for h in range(H)}
    tables = {}
    for l, p in TAB.items():
        t_ = torch.load(p, map_location=dev)
        for h in TOKEN_PROG[l]:
            tables[(l, h)] = (t_[f"head{h}_A"], t_[f"head{h}_B"], t_[f"head{h}_kappa"])
    with torch.no_grad():
        state = {"idx": None, "kernel": {l: () for l in LAYERS}, "token": {l: () for l in LAYERS}}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        natives = {l: blocks[l].attn.squared_attention for l in LAYERS}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                if state["kernel"][l] or state["token"][l]:
                    pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (causal & (dmat > 0))[None]; idx = state["idx"]
                    for h in state["kernel"][l]:
                        pat[:, h] = torch.where(off, kbar[(l, h)][dmat][None].expand(Bn, -1, -1), pat[:, h])
                    for h in state["token"][l]:
                        A, Bt, kappa = tables[(l, h)]; prog = kappa[dmat][None] * A[idx][:, :, None] * Bt[idx][:, None, :]
                        pat[:, h] = torch.where(off, prog, pat[:, h])
                return torch.einsum("bhqk,bkhd->bhqd", pat, v)
            return patched

        for l in LAYERS:
            blocks[l].attn.squared_attention = make_patched(l)

        def ce(kernel, token):
            state["kernel"] = {l: kernel.get(l, ()) for l in LAYERS}; state["token"] = {l: token.get(l, ()) for l in LAYERS}
            total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            return total / n, fw

        native, fw = ce({}, {}); forwards += fw; e = {}
        e["positional_kernels"], fw = ce(POSITIONAL, {}); forwards += fw; e["positional_kernels"] -= native
        e["content_kernels"], fw = ce(CONTENT, {}); forwards += fw; e["content_kernels"] -= native
        e["best_program"], fw = ce(KERNEL_IN_BEST, TOKEN_PROG); forwards += fw; e["best_program"] -= native
        e["bank0"], fw = ce({0: tuple(range(H))}, {}); forwards += fw; e["bank0"] -= native
        e["bank1"], fw = ce({1: tuple(range(H))}, {}); forwards += fw; e["bank1"] -= native
        e["bank01"], fw = ce({0: tuple(range(H)), 1: tuple(range(H))}, {}); forwards += fw; e["bank01"] -= native
        for l in LAYERS:
            blocks[l].attn.squared_attention = natives[l]
        pre.remove()
        print(f"native {native:.5f} | " + " ".join(f"{k}={v:+.4f}" for k, v in e.items()))
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_positional_union_cheap": e["positional_kernels"] <= POS_MAX,
                   "pred_c_content_union_expensive": e["content_kernels"] >= CON_MIN, "pred_d_token_gating_helps_jointly": e["best_program"] <= GATE_RATIO * e["positional_kernels"],
                   "pred_e_pair01_composes": e["bank01"] <= PAIR_RATIO * (e["bank0"] + e["bank1"])}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_union_split_result_v630", "candidate_id": CANDIDATE_ID, "plan": plan, "report": {"native": native, "edits": e},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
