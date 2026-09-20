#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_indicator_recovers_half pred_c_indicator_beats_kernel pred_d_induction_attracts pred_e_rank16_still_better
"""Attention lane, v731: head 5.5's induction indicator at larger beta (v730: best at the grid edge 0.1, recovery 0.40).

Edit: 5.5's off-diagonal pattern := kappa(d) + beta [tok_i == tok_{j-1}] (the key's previous token equals the query token), beta in
{0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0}, one head inside the native model; against its value 0.0118 (v701) and its closed-form rank-16
program. Response only. CE ADDED; recovery = 1 - cost / value.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native within 0.002 of 3.13241 (instrument)
    pred_b_indicator_recovers_half at the best beta the induction program recovers >= 0.5 of 5.5's value. Prior: likely
    pred_c_indicator_beats_kernel  best cost <= kernel-only (v730: +0.0114) - 0.001. Prior: certain given v730 (registered anyway)
    pred_d_induction_attracts      the best beta is >= 0.2 (the read is strong). Prior: likely
    pred_e_rank16_still_better     the rank-16 closed-form program costs >= 0.001 less than the best indicator program. Prior: unsure
PRICE (registered maximum): native 6; 8 betas x 6 = 48; rank-16 kappa_r 2 + 6; total 62 forwards; 0 backwards; 0 fits. Bar <= 80.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_induction_55_v731_result.json"
PROGS702 = ROOT / "circuits/followups/attention_program_ladder_v702_programs.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "attention.induction_55_v731"
FORWARDS_MAX = 80
EBATCH = 32
L_, H_ = 5, 5; H = 9
BETAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
REPLAY_TOL, REC_MIN, GAIN_MIN, R16_MIN = 0.002, 0.5, 0.001, 0.001
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_indicator_recovers_half": "recovery >= 0.5", "pred_c_indicator_beats_kernel": "best - kernel-only <= -0.001",
               "pred_d_induction_attracts": "best beta >= 0.2", "pred_e_rank16_still_better": "rank-16 <= best - 0.001"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "head": [L_, H_], "betas": list(BETAS),
            "bars": {"replay_tol": REPLAY_TOL, "rec_min": REC_MIN, "gain_min": GAIN_MIN, "r16_min": R16_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; hd = D // H; dev = "cuda"; blk = model.transformer.h[L_]; at = blk.attn; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long(); fit64 = torch.load(FIT_ROWS, map_location="cpu").long()[:64]
    value = json.load(open(V701))["report"]["mean_ablation_cost"][f"{L_}.{H_}"]
    kappa = torch.load(PROGS702, map_location=dev)[f"kappa_{L_}_{H_}"].float()
    state = {"idx": None, "mode": None, "beta": 0.0, "n": None, "kr": None}
    pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
    nh = at.register_forward_pre_hook(lambda m, a: state.__setitem__("n", a[0]))
    native_sq = at.squared_attention; maps16 = AP.truncated_maps(at, H_, 16, hd)

    def patched(q, k, v, q2, k2):
        Bn, Tn, Hn, Dn = q.shape
        pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
        causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat = pat.masked_fill(~causal, 0.0)
        if state["mode"] is not None:
            pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (causal & (dmat > 0))[None]
            if state["mode"] == "indicator":
                prev = torch.cat([torch.full_like(state["idx"][:, :1], -1), state["idx"][:, :-1]], 1)
                prog = kappa[dmat][None] + state["beta"] * (state["idx"][:, :, None] == prev[:, None, :]).float()
            else:
                pr = AP.lowrank_pattern(at, state["n"], maps16, hd, causal); prog = kappa[dmat][None] + (pr - state["kr"][dmat][None])
            pat = pat.clone(); pat[:, H_] = torch.where(off, prog, pat[:, H_])
        return torch.einsum("bhqk,bkhd->bhqd", pat, v)

    at.squared_attention = patched

    def ce(mode, beta=0.0):
        state["mode"] = mode; state["beta"] = beta; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        state["mode"] = None
        return total / n, fw

    native, fw = ce(None); forwards += fw
    costs = {}
    for b in BETAS:
        c_, fw = ce("indicator", b); forwards += fw; costs[b] = c_ - native
        print(f"beta {b:+.2f}: cost {costs[b]:+.4f} (recovery {1 - costs[b] / value:.2f})")
    acc = 0; nb = 0
    with torch.no_grad():
        for s in range(0, 64, EBATCH):
            idx = fit64[s:s + EBATCH, :-1].to(dev); state["mode"] = None; model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
            Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
            acc = acc + AP.offset_mean(AP.lowrank_pattern(at, state["n"], maps16, hd, causal), dmat, off, Tn); nb += 1
    state["kr"] = acc / nb
    c16, fw = ce("lowrank"); forwards += fw; c16 -= native
    best = min(costs, key=costs.get)
    print(f"native {native:.5f} | value {value:.4f} | kernel-only {costs[0.0]:+.4f} | best beta {best:+.2f}: {costs[best]:+.4f} (recovery {1 - costs[best] / value:.2f}) | rank-16 closed-form {c16:+.4f} (recovery {1 - c16 / value:.2f})")
    at.squared_attention = native_sq; pre.remove(); nh.remove()
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_indicator_recovers_half": (1 - costs[best] / value) >= REC_MIN,
                   "pred_c_indicator_beats_kernel": costs[best] <= costs[0.0] - GAIN_MIN, "pred_d_induction_attracts": best >= 0.2, "pred_e_rank16_still_better": c16 <= costs[best] - R16_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_same_token_edit_result_v728", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "value": value, "indicator_costs": {str(b): c for b, c in costs.items()}, "best_beta": best, "best_cost": costs[best], "best_recovery": 1 - costs[best] / value, "rank16_cost": c16},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
