#!/usr/bin/env python
"""circuit_battery_roundness_localisation -- WHICH component decides step-continuation versus plus-one?

SS2841: with the step fixed at 10, this model continues a percentage run BY ITS STEP when the values are multiples of ten (1.000, 6/6)
and NEVER does so otherwise (.000, 0/48); on non-round values it adds one instead (.313). Roundness switches it between two behaviours.
That makes a minimal pair: "10% 20% 30%" -> " 40" (step) against "11% 21% 31%" -> " 32" (plus one), IDENTICAL in token length and
differing by one digit per term. This rung runs the battery's interchange-localisation stage on that pair to find which of the 36
components carries the switch, scored as normalized logit-difference recovery between the two answers.

# BQGATE: EXPERIMENT  pred_a_the_switch_is_localised pred_b_the_switch_is_not_the_item_writer
#                     pred_c_the_switch_is_concentrated pred_d_the_leader_is_format_invariant
#                     pred_e_patching_everything_recovers_the_donor

SIGN CONVENTION: recovery REC = (ld_patch - ld_base) / max(ld_donor - ld_base, 1e-3) with ld = logit(plus-one answer) - logit(step
answer); REC is 0 for no effect and 1 for a full switch, HIGHER MEANS THE COMPONENT CARRIES MORE OF THE SWITCH. No CE and no SS312 L2
here; nothing installs.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_ROUNDNESS_LOCALISATION_PREREGISTRATION.md
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_roundness_localisation.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_ROUNDNESS_LOCALISATION_PREREGISTRATION.md"
ROUNDCAP = ROOT / "circuit_battery_roundness_capability_results.json"
RUNG = "circuit_battery_roundness_localisation"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "a1dcc07ade3f795069309dc282fee6d4e197ed9968f279140dd81fbc14ce3e15",
          ROUNDCAP: "f099b983dcb1fa35c112d6c1dd3565f6024bb6ea5836b72e4bb050c537ae6923",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
ENC = BANK.ENC
NL = R.NL
COMPONENTS = [(kd, l) for l in range(NL) for kd in ("attn", "mlp")]
STEP = 10
STARTS = [10, 20, 30, 40, 50, 60]
OFFSETS = [1, 2, 3, 4]
FORMATS = {"percent": lambda a, b, c: f"{a}% {b}% {c}%",
           "bare": lambda a, b, c: f"{a} {b} {c}"}
BARS = {"top_rec": 0.50, "top3_share": 0.60, "attn8_rank": 3, "format_agree": 1, "all_rec": 0.90}
NULLS = {"top_rec_le": 0.20, "top3_share_le": 0.30, "attn8_top": 1, "all_rec_le": 0.50}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def build_pairs(fmt):
    """Round base (answer = last + STEP) against its +offset twin (answer = last + 1)."""
    mk = FORMATS[fmt]
    out = []
    for s in STARTS:
        for off in OFFSETS:
            b = mk(s, s + STEP, s + 2 * STEP)
            d = mk(s + off, s + off + STEP, s + off + 2 * STEP)
            ba, da = f" {s + 3 * STEP}", f" {s + off + 2 * STEP + 1}"
            if len(ENC.encode(b)) != len(ENC.encode(d)):
                continue
            if any(len(ENC.encode(x)) != 1 for x in (ba, da)) or ba == da:
                continue
            if ENC.encode(b + ba) != ENC.encode(b) + ENC.encode(ba):
                continue
            if ENC.encode(d + da) != ENC.encode(d) + ENC.encode(da):
                continue
            out.append({"base_text": b, "donor_text": d, "base_answer": ba, "donor_answer": da,
                        "base_ids": ENC.encode(b), "donor_ids": ENC.encode(d),
                        "base_answer_id": ENC.encode(ba)[0], "donor_answer_id": ENC.encode(da)[0]})
    return out


def main():
    t0 = time.time()
    check_hashes()
    m = R.load_model().to(DEV).eval()
    fwd = [0]
    results = {}
    for fmt in FORMATS:
        rows = build_pairs(fmt)
        if SMOKE:
            rows = rows[:6]
        per = defaultdict(list)
        allrec, kept, native_ok = [], 0, 0
        for b in CB.batches(rows):
            bid, bfin, bans = CB.pack(b, "base")
            did, dfin, dans = CB.pack(b, "donor")
            dcache = {}
            ldon = CB.run(m, did, dfin, cache=dcache); fwd[0] += 1
            lb = CB.run(m, bid, bfin); fwd[0] += 1
            base_ld = (lb.gather(1, dans[:, None]) - lb.gather(1, bans[:, None])).squeeze(1)
            don_ld = (ldon.gather(1, dans[:, None]) - ldon.gather(1, bans[:, None])).squeeze(1)
            native_ok += int(((base_ld < 0) & (don_ld > 0)).sum())     # base prefers step, donor prefers +1
            denom = (don_ld - base_ld).clamp_min(1e-3)
            kept += len(b)
            for comp in COMPONENTS:
                lp = CB.run(m, bid, bfin, patch={comp: dcache[comp]}); fwd[0] += 1
                ld = (lp.gather(1, dans[:, None]) - lp.gather(1, bans[:, None])).squeeze(1)
                per[f"{comp[0]}{comp[1]}"].append(((ld - base_ld) / denom).cpu().numpy())
            lp = CB.run(m, bid, bfin, patch=dcache); fwd[0] += 1
            ld = (lp.gather(1, dans[:, None]) - lp.gather(1, bans[:, None])).squeeze(1)
            allrec.append(((ld - base_ld) / denom).cpu().numpy())
        rec = {k: float(np.concatenate(v).mean()) for k, v in per.items()}
        order = sorted(rec, key=lambda k: -rec[k])
        pos = sum(v for v in rec.values() if v > 0) or 1e-9
        results[fmt] = {"recovery": rec, "order_top8": order[:8], "top": order[0],
                        "top_recovery": rec[order[0]],
                        "top3_share": sum(rec[k] for k in order[:3]) / pos,
                        "attn8_rank": order.index("attn8") + 1, "attn8_recovery": rec["attn8"],
                        "all_components_recovery": float(np.concatenate(allrec).mean()),
                        "rows": kept, "rows_with_expected_native_preference": native_ok}
        p = results[fmt]
        print(f"[rloc] {fmt:8s} top={p['top']}({p['top_recovery']:.2f}) top3={p['top3_share']:.2f} "
              f"attn8_rank={p['attn8_rank']} all={p['all_components_recovery']:.2f} "
              f"native_ok={native_ok}/{kept}", flush=True)

    tops = [results[f]["top"] for f in results]
    agree = len(set(tops)) == 1
    med = lambda k: float(np.median([results[f][k] for f in results]))
    preds = {
        'pred_a_the_switch_is_localised': bool(med("top_recovery") >= BARS["top_rec"]),
        'pred_b_the_switch_is_not_the_item_writer': bool(min(results[f]["attn8_rank"] for f in results) > BARS["attn8_rank"]),
        'pred_c_the_switch_is_concentrated': bool(med("top3_share") >= BARS["top3_share"]),
        'pred_d_the_leader_is_format_invariant': bool(agree),
        'pred_e_patching_everything_recovers_the_donor': bool(med("all_components_recovery") >= BARS["all_rec"]),
    }
    nulls = {
        "a_null_not_localised": bool(med("top_recovery") <= NULLS["top_rec_le"]),
        "b_null_attn8_leads": bool(any(results[f]["attn8_rank"] <= NULLS["attn8_top"] for f in results)),
        "c_null_diffuse": bool(med("top3_share") <= NULLS["top3_share_le"]),
        "e_null_instrument_fails": bool(med("all_components_recovery") <= NULLS["all_rec_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "step": STEP, "starts": STARTS, "offsets": OFFSETS, "formats": sorted(FORMATS),
              "summary": {"tops": tops, "leader_agrees_across_formats": agree,
                          "medians": {k: med(k) for k in ("top_recovery", "top3_share",
                                                          "all_components_recovery")},
                          "attn8_ranks": {f: results[f]["attn8_rank"] for f in results},
                          "order_top8": {f: results[f]["order_top8"] for f in results}},
              "formats_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1200])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: {len(FORMATS)} formats x {len(COMPONENTS)} component interchange patches on "
              f"round/non-round minimal pairs; no model loaded")
        sys.exit(0)
    main()
