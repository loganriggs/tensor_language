#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_rank8_median_recovery pred_c_rank32_median_recovery pred_d_sink_variation_is_low_rank pred_e_joint_selection_cost
"""Attention lane, v710: the CENTERED write rank — each head's write as its mean plus r directions of its own variation.

v708 projected z_h onto c_proj's top singular directions and found the write side 'not low-rank' — but for the sink head 5.7 the projection
cost +1.48 while deleting the head's whole variation (mean ablation) costs 0.012: the head's write is a large near-constant vector that
mean ablation keeps and the projection removes. The value of a head under mean ablation is the value of its VARIATION around its mean, so
the write ladder must start from the mean. Edit: z_h := mean_h + M_r (z_h - mean_h), where M_r = pinv(W_o,h) Q_r Q_r^T W_o,h projects the
head's residual write onto the top-r eigenvectors Q_r of the covariance of its own write W_o,h (z_h - mean_h), estimated on 64 fit rows
(mean_h from v701's 480 rows). Rank 0 = mean ablation exactly; rank 128 = native. One head at a time at r = 8 and 32 on skip7000;
recovery = 1 - cost / value; then joint edits: every head at the smallest rank with single recovery >= 0.9 (else native), all at 32, all at 8.
CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native CE within 0.002 of 3.13241 (instrument)
    pred_b_rank8_median_recovery   among heads with value >= 0.02, the median rank-8 recovery >= 0.5. Prior: unsure
    pred_c_rank32_median_recovery  among heads with value >= 0.005, the median rank-32 recovery >= 0.9. Prior: likely
    pred_d_sink_variation_is_low_rank head 5.7's rank-8 recovery >= 0.9 (its variation around the sink read is low-dimensional). Prior: likely
    pred_e_joint_selection_cost    the joint edit at selection costs <= 0.3. Prior: unsure
PRICE (registered maximum): native 6; covariance capture 2; 162 x 2 x 6 = 1944; three joint edits 18; total 1970 forwards; 0 backwards; 0 fits. Bar <= 1995.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_write_rank_centered_v710_result.json"
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "attention.write_rank_centered_v710"
FORWARDS_MAX = 1995
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
RANKS = (8, 32)
REPLAY_TOL, VALUE_BIG, VALUE_FLOOR, REC8, REC32, SELECT, JOINT32 = 0.002, 0.02, 0.005, 0.5, 0.9, 0.9, 0.3
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_rank8_median_recovery": ">= 0.5 over value >= 0.02", "pred_c_rank32_median_recovery": ">= 0.9 over value >= 0.005",
               "pred_d_sink_variation_is_low_rank": "5.7 rank-8 recovery >= 0.9", "pred_e_joint_selection_cost": "<= 0.3"}


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
    means = torch.load(MEANS, map_location=dev); fit64 = torch.load(FIT_ROWS, map_location="cpu").long()[:64]
    with torch.no_grad():
        state = {"edits": {}, "capture": False}
        cov = {(l, h): torch.zeros(hd, hd, dtype=torch.float64, device=dev) for (l, h) in ALL}; n_tok = [0]
        natives = {l: blocks[l].attn.squared_attention for l in LAYERS}
        proj = {}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                z = torch.einsum("bhqk,bkhd->bhqd", pat, v)
                if state["capture"]:
                    for h in range(Hn):
                        zc = (z[:, h].float() - means[f"mean_{l}_{h}"]).reshape(-1, Dn).double(); cov[(l, h)] += zc.T @ zc
                    if l == 0:
                        n_tok[0] += Bn * Tn
                hs = [(h, r) for (al, h), r in state["edits"].items() if al == l]
                if hs:
                    z = z.clone()
                    for h, r in hs:
                        m = means[f"mean_{l}_{h}"]; z[:, h] = (m + (z[:, h].float() - m) @ proj[(l, h, r)]).to(z.dtype)
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

        state["capture"] = True
        for s in range(0, 64, EBATCH):
            idx = fit64[s:s + EBATCH, :-1].to(dev); model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
        state["capture"] = False
        for (l, h) in ALL:
            Wo = blocks[l].attn.c_proj.weight[:, h * hd:(h + 1) * hd].detach().double()          # [1152, 128]
            C = cov[(l, h)] / n_tok[0]; G = Wo @ C @ Wo.T                                        # covariance of the head's write
            evals, Q = torch.linalg.eigh(G); Q = Q[:, torch.argsort(evals, descending=True)]
            Wp = torch.linalg.pinv(Wo)                                                          # [128, 1152]
            for r in RANKS:
                Qr = Q[:, :r]; proj[(l, h, r)] = (Wp @ Qr @ Qr.T @ Wo).T.float().contiguous()   # z-space, applied as z @ M^T
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
        c_8, fw = ce({k: 8 for k in ALL}); forwards += fw
        for l in LAYERS:
            blocks[l].attn.squared_attention = natives[l]
    big = [k for k in ALL if value[f"{k[0]}.{k[1]}"] >= VALUE_BIG]; mid = [k for k in ALL if value[f"{k[0]}.{k[1]}"] >= VALUE_FLOOR]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
    med8 = med([rec(k, 8) for k in big]); med32 = med([rec(k, 32) for k in mid])
    counts = {str(r): sum(1 for r_ in joint_sel.values() if r_ == r) for r in RANKS}; counts["native"] = len(ALL) - len(joint_sel)
    nums_sel = sum(2 * r * (hd + D) if k in joint_sel else 2 * hd * D for k in ALL for r in [joint_sel.get(k, 0)])
    print(f"native {native:.5f} | rank-8 median recovery (value >= 0.02, n={len(big)}) {med8:.2f} | rank-32 median recovery (value >= 0.005, n={len(mid)}) {med32:.2f} | 5.7 rank-8 recovery {rec((5, 7), 8):.2f}")
    print(f"joint at selection {counts}: {c_sel - native:+.4f} ({nums_sel / 1e6:.1f}M write numbers vs 47.8M) | joint all rank 32: {c_32 - native:+.4f} | all rank 8: {c_8 - native:+.4f}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_rank8_median_recovery": med8 >= REC8, "pred_c_rank32_median_recovery": med32 >= REC32,
                   "pred_d_sink_variation_is_low_rank": rec((5, 7), 8) >= REC32, "pred_e_joint_selection_cost": (c_sel - native) <= JOINT32}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_write_rank_centered_result_v710", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "heads": {f"{l}.{h}": {"value": value[f"{l}.{h}"], **{f"cost_{r}": cost[(l, h, r)] for r in RANKS},
                                          **{f"recovery_{r}": (rec((l, h), r) if value[f"{l}.{h}"] >= VALUE_FLOOR else None) for r in RANKS}} for (l, h) in ALL},
                                          "selection": {f"{k[0]}.{k[1]}": (select[k] if select[k] is not None else "native") for k in ALL}, "selection_counts": counts,
                                          "joint_selection_cost": c_sel - native, "joint_selection_numbers": nums_sel, "joint_rank32_cost": c_32 - native, "joint_rank8_cost": c_8 - native, "median_rec8_big": med8, "median_rec32_mid": med32},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
