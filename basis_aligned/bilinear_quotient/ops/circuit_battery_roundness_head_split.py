#!/usr/bin/env python
"""circuit_battery_roundness_head_split -- does the roundness feature live in attention 8's SAME two heads?

SS2842: attention 8 carries the step-versus-plus-one switch, ranking 1st of 36 in both the percent and the bare format and recovering
.589 of the decision. SS2820: attention 8's write decomposes exactly into nine head writes and heads {3, 7} are its top-2 on 6 of 7
behaviours, with the class and margin head maps identical. So attention 8 is doing at least two separable jobs in our measurements --
writing which item was last, and carrying whether that item is round -- and the question is whether they share heads.

Because c_proj is linear without bias, a head-level interchange is exact: the donor's slice of the concatenated head outputs is swapped
into the base run before the projection, one head at a time.

# BQGATE: EXPERIMENT  pred_a_a_head_carries_the_switch pred_b_it_is_the_r576_pair
#                     pred_c_two_heads_suffice pred_d_the_head_is_format_invariant
#                     pred_e_all_heads_equal_the_whole_component

SIGN CONVENTION: REC = (ld_patch - ld_base) / max(ld_donor - ld_base, 1e-3) with ld = logit(plus-one) - logit(step); 0 = no effect,
1 = full switch, HIGHER = the head carries more of it. No CE and no SS312 L2; nothing installs.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_ROUNDNESS_HEAD_SPLIT_PREREGISTRATION.md
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_roundness_head_split.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
import circuit_battery_roundness_localisation as RL
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_ROUNDNESS_HEAD_SPLIT_PREREGISTRATION.md"
RLOC = ROOT / "circuit_battery_roundness_localisation_results.json"
RUNG = "circuit_battery_roundness_head_split"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "2feb48d22a9a280ad85ec6d808e8b721325c577c099c7e99f72ad3fb890e42ca",
          RLOC: "331454aac1ce218d9194255e19c81c53eca38d99cc6c2b685ff2d9e0ac12788c",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NH, HD = R.D, R.NH, R.HD
LAYER = 8
R576_HEADS = (3, 7)
WHOLE_REC = 0.5888893604278564      # SS2842's whole-component recovery for attention 8
BARS = {"top_rec": 0.40, "top2_share": 0.60, "exact_tol": 0.05}
NULLS = {"top_rec_le": 0.15, "top2_share_le": 0.30}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def run_headpatch(m, tokens, finals, donor_ycat=None, heads=()):
    """Forward with the donor's slice of attention LAYER's concatenated head outputs swapped in."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    cap = {}
    blk8 = m.transformer.h[LAYER]

    def pre(_mod, args):
        y = args[0]
        cap["y"] = y
        if donor_ycat is None or not heads:
            return None
        y2 = y.clone()
        for h in heads:
            y2[..., h * HD:(h + 1) * HD] = donor_ycat[..., h * HD:(h + 1) * HD]
        return (y2,)

    hk = blk8.attn.c_proj.register_forward_pre_hook(pre)
    try:
        for site, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
            x = x + write
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    finally:
        hk.remove()
    logits = (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float()
    return logits, cap.get("y")


def main():
    t0 = time.time()
    check_hashes()
    m = R.load_model().to(DEV).eval()
    fwd = [0]
    results = {}
    for fmt in RL.FORMATS:
        rows = RL.build_pairs(fmt)
        if SMOKE:
            rows = rows[:6]
        per, allh = defaultdict(list), []
        for b in CB.batches(rows):
            bid, bfin, bans = CB.pack(b, "base")
            did, dfin, dans = CB.pack(b, "donor")
            ldon, dy = run_headpatch(m, did, dfin); fwd[0] += 1
            lb, _ = run_headpatch(m, bid, bfin); fwd[0] += 1
            base_ld = (lb.gather(1, dans[:, None]) - lb.gather(1, bans[:, None])).squeeze(1)
            don_ld = (ldon.gather(1, dans[:, None]) - ldon.gather(1, bans[:, None])).squeeze(1)
            denom = (don_ld - base_ld).clamp_min(1e-3)
            for h in range(NH):
                lp, _ = run_headpatch(m, bid, bfin, donor_ycat=dy, heads=(h,)); fwd[0] += 1
                ld = (lp.gather(1, dans[:, None]) - lp.gather(1, bans[:, None])).squeeze(1)
                per[f"head{h}"].append(((ld - base_ld) / denom).cpu().numpy())
            lp, _ = run_headpatch(m, bid, bfin, donor_ycat=dy, heads=tuple(range(NH))); fwd[0] += 1
            ld = (lp.gather(1, dans[:, None]) - lp.gather(1, bans[:, None])).squeeze(1)
            allh.append(((ld - base_ld) / denom).cpu().numpy())
        rec = {k: float(np.concatenate(v).mean()) for k, v in per.items()}
        order = sorted(rec, key=lambda k: -rec[k])
        pos = sum(v for v in rec.values() if v > 0) or 1e-9
        results[fmt] = {"recovery": rec, "order": order, "top": order[0], "top_recovery": rec[order[0]],
                        "top2_heads": sorted(int(k[4:]) for k in order[:2]),
                        "top2_share": sum(rec[k] for k in order[:2]) / pos,
                        "all_heads_recovery": float(np.concatenate(allh).mean()),
                        "rows": len(rows)}
        p = results[fmt]
        print(f"[rhead] {fmt:8s} top={p['top']}({p['top_recovery']:.3f}) top2={p['top2_heads']} "
              f"share={p['top2_share']:.2f} all_heads={p['all_heads_recovery']:.3f}", flush=True)

    med = lambda k: float(np.median([results[f][k] for f in results]))
    pairs = [tuple(results[f]["top2_heads"]) for f in results]
    is_r576 = all(set(p) == set(R576_HEADS) for p in pairs)
    tops = [results[f]["top"] for f in results]
    exact = max(abs(results[f]["all_heads_recovery"] - WHOLE_REC) for f in results)
    preds = {
        'pred_a_a_head_carries_the_switch': bool(med("top_recovery") >= BARS["top_rec"]),
        'pred_b_it_is_the_r576_pair': bool(is_r576),
        'pred_c_two_heads_suffice': bool(med("top2_share") >= BARS["top2_share"]),
        'pred_d_the_head_is_format_invariant': bool(len(set(tops)) == 1),
        'pred_e_all_heads_equal_the_whole_component': bool(exact <= BARS["exact_tol"]),
    }
    nulls = {
        "a_null_no_head_carries_it": bool(med("top_recovery") <= NULLS["top_rec_le"]),
        "b_null_different_pair": bool(not is_r576),
        "c_null_spread_over_heads": bool(med("top2_share") <= NULLS["top2_share_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "layer": LAYER, "r576_heads": list(R576_HEADS), "whole_component_recovery_2842": WHOLE_REC,
              "summary": {"tops": tops, "top2_pairs": [list(p) for p in pairs],
                          "matches_r576_pair": is_r576,
                          "all_heads_vs_whole_component_max_dev": exact,
                          "medians": {k: med(k) for k in ("top_recovery", "top2_share",
                                                          "all_heads_recovery")},
                          "orders": {f: results[f]["order"] for f in results}},
              "formats_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1200])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: {NH} head interchanges of attention {LAYER} on the roundness minimal pairs, "
              f"two formats; no model loaded")
        sys.exit(0)
    main()
