#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_rank8_median_recovery pred_c_rank32_median_recovery pred_d_sink_write_is_low_rank pred_e_joint_rank32_cost
"""Attention lane, v708: the OTHER side of every head — the rank of its WRITE — valued against mean ablation.

v701-v707 simplified the pattern (QK) side only: values and c_proj stayed native, 47.8M numbers across 162 heads (c_v + c_proj rows).
This rung asks how many directions each head actually writes. Edit: the head's 128-d pre-projection output z_h is projected onto the top-r
right singular vectors of its c_proj block W_o,h (1152 x 128) — z_h := V_r V_r^T z_h — so the head's residual write W_o,h z_h is confined
to r directions (the value mix with block-0's values passes through untouched; the pattern is native). Closed-form (SVD of the weights),
no fit. One head at a time at r = 8 and r = 32 on the 192 x 512 skip7000 rows; recovery = 1 - cost / mean-ablation value (v701), reported
where value >= 0.005. Then two joint edits (single edits nominate, joint edits certify): every head at the smallest rank in {8, 32} whose
single-head recovery >= 0.9 (else native), and every head at rank 32. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native CE within 0.002 of 3.13241 (instrument)
    pred_b_rank8_median_recovery among heads with value >= 0.02, the median rank-8 recovery >= 0.5. Prior: unsure
    pred_c_rank32_median_recovery among heads with value >= 0.005, the median rank-32 recovery >= 0.9. Prior: likely
    pred_d_sink_write_is_low_rank head 5.7 (sink, value norm 771 at position 0) has rank-8 recovery >= 0.9. Prior: likely
    pred_e_joint_rank32_cost     all 162 heads at write-rank 32 jointly cost <= 0.3 (a quarter of the numbers). Prior: unsure
PRICE (registered maximum): native 6; 162 heads x 2 ranks x 6 = 1944; two joint edits 12; total 1962 forwards; 0 backwards; 0 fits. Bar <= 1990.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_write_rank_v708_result.json"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "attention.write_rank_v708"
FORWARDS_MAX = 1990
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
RANKS = (8, 32)
REPLAY_TOL, VALUE_BIG, VALUE_FLOOR, REC8, REC32, SELECT, JOINT32 = 0.002, 0.02, 0.005, 0.5, 0.9, 0.9, 0.3
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_rank8_median_recovery": ">= 0.5 over value >= 0.02", "pred_c_rank32_median_recovery": ">= 0.9 over value >= 0.005",
               "pred_d_sink_write_is_low_rank": "5.7 rank-8 recovery >= 0.9", "pred_e_joint_rank32_cost": "<= 0.3"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "ranks": list(RANKS),
            "bars": {"replay_tol": REPLAY_TOL, "value_big": VALUE_BIG, "value_floor": VALUE_FLOOR, "rec8": REC8, "rec32": REC32, "select": SELECT, "joint32": JOINT32}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    value = json.load(open(V701))["report"]["mean_ablation_cost"]
    ALL = [(l, h) for l in LAYERS for h in range(H)]
    with torch.no_grad():
        proj = {}
        for (l, h) in ALL:
            Wo = blocks[l].attn.c_proj.weight[:, h * hd:(h + 1) * hd].detach().float()      # [1152, 128]
            _, _, Vh = torch.linalg.svd(Wo, full_matrices=False)                           # rows of Vh: right singular vectors in z-space
            for r in RANKS:
                Vr = Vh[:r]; proj[(l, h, r)] = (Vr.T @ Vr)                                   # [128, 128]
        state = {"edits": {}}
        natives = {l: blocks[l].attn.squared_attention for l in LAYERS}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                z = torch.einsum("bhqk,bkhd->bhqd", pat, v)
                hs = [(h, r) for (al, h), r in state["edits"].items() if al == l]
                if hs:
                    z = z.clone()
                    for h, r in hs:
                        z[:, h] = (z[:, h].float() @ proj[(l, h, r)]).to(z.dtype)
                return z
            return patched

        for l in LAYERS:
            blocks[l].attn.squared_attention = make_patched(l)

        def ce(edits):
            state["edits"] = edits; total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            state["edits"] = {}
            return total / n, fw

        native, fw = ce({}); forwards += fw
        cost = {}
        for (l, h) in ALL:
            for r in RANKS:
                c_, fw = ce({(l, h): r}); forwards += fw; cost[(l, h, r)] = c_ - native
            print(f"{l}.{h}: value {value[f'{l}.{h}']:+.4f} | write rank 8 {cost[(l, h, 8)]:+.4f} rank 32 {cost[(l, h, 32)]:+.4f}")
        def rec(k, r):
            return 1 - cost[(k[0], k[1], r)] / value[f"{k[0]}.{k[1]}"]
        select = {}
        for k in ALL:
            v_ = value[f"{k[0]}.{k[1]}"]
            if v_ < VALUE_FLOOR:
                select[k] = 8
            else:
                select[k] = next((r for r in RANKS if rec(k, r) >= SELECT), None)
        joint_sel = {k: r for k, r in select.items() if r is not None}
        c_sel, fw = ce(joint_sel); forwards += fw
        c_32, fw = ce({k: 32 for k in ALL}); forwards += fw
        for l in LAYERS:
            blocks[l].attn.squared_attention = natives[l]
    big = [k for k in ALL if value[f"{k[0]}.{k[1]}"] >= VALUE_BIG]; mid = [k for k in ALL if value[f"{k[0]}.{k[1]}"] >= VALUE_FLOOR]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
    med8 = med([rec(k, 8) for k in big]); med32 = med([rec(k, 32) for k in mid])
    counts = {str(r): sum(1 for r_ in joint_sel.values() if r_ == r) for r in RANKS}; counts["native"] = len(ALL) - len(joint_sel)
    nums_sel = sum(2 * r * (hd + D) if k in joint_sel else 2 * hd * D for k in ALL for r in [joint_sel.get(k, 0)])
    print(f"native {native:.5f} | rank-8 median recovery (value >= 0.02, n={len(big)}) {med8:.2f} | rank-32 median recovery (value >= 0.005, n={len(mid)}) {med32:.2f} | 5.7 rank-8 recovery {rec((5, 7), 8):.2f}")
    print(f"joint at selection {counts}: {c_sel - native:+.4f} ({nums_sel / 1e6:.1f}M write numbers vs 47.8M) | joint all rank 32: {c_32 - native:+.4f}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_rank8_median_recovery": med8 >= REC8, "pred_c_rank32_median_recovery": med32 >= REC32,
                   "pred_d_sink_write_is_low_rank": rec((5, 7), 8) >= REC32, "pred_e_joint_rank32_cost": (c_32 - native) <= JOINT32}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_write_rank_result_v708", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "heads": {f"{l}.{h}": {"value": value[f"{l}.{h}"], **{f"cost_{r}": cost[(l, h, r)] for r in RANKS},
                                          **{f"recovery_{r}": (rec((l, h), r) if value[f"{l}.{h}"] >= VALUE_FLOOR else None) for r in RANKS}} for (l, h) in ALL},
                                          "selection": {f"{k[0]}.{k[1]}": (select[k] if select[k] is not None else "native") for k in ALL}, "selection_counts": counts,
                                          "joint_selection_cost": c_sel - native, "joint_selection_numbers": nums_sel, "joint_rank32_cost": c_32 - native, "median_rec8_big": med8, "median_rec32_mid": med32},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
