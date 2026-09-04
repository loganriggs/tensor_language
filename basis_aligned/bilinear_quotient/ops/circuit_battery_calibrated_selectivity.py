"""circuit_battery_calibrated_selectivity -- the protocol repair SS2851 forces: an arm below the ceiling, and a verified control.

SS2851 showed the battery's selectivity stage is unmeasurable as built. Its writer arm removes a median 1.207x the native target
margin and 2.552x the native control margin, so both sides of the ratio are pinned and six of eight behaviours read exactly 1.00;
and the copy control's own native margin is a median .39 -- .09 on the paren list, .20 on the numbered list, and NEGATIVE (-.72)
on month, where the model does not natively give the copy answer at all.

This rung repairs both, as a protocol v3 amendment, and re-scores every capable behaviour:

  CALIBRATED ARM -- on FIT rows, walk a fixed ladder of arm strengths (the write removed from all reader edges, then half, then
  a quarter, then an eighth) and take the LARGEST whose target saturation is <= .80; score selectivity on SELECT with that arm.
  VERIFIED CONTROL -- a control family enters max(|d_P|, |d_C|) only if its own native margin on SELECT is >= .50 and positive.
  A behaviour whose copy control fails that test is scored on P alone and flagged, rather than silently carrying a term measured
  against a wrong-signed baseline.

# BQGATE: EXPERIMENT  pred_a_calibration_lands_below_the_ceiling pred_b_some_behaviour_is_now_selective
#                     pred_c_enough_copy_controls_are_usable pred_d_the_correction_is_material
#                     pred_e_the_full_arm_still_reproduces_the_battery

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's own answer; saturation = d_m /
max(m_NATIVE, .5); ratio = max over USABLE controls of |d_control| / max(d_A1, .5), LOWER IS MORE SELECTIVE. No CE and no SS312
L2; nothing installs.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_PROTOCOL_V3_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_calibrated_selectivity.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
import fastload
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_PROTOCOL_V3_PREREGISTRATION.md"
SAT = ROOT / "circuit_battery_writer_arm_saturation_results.json"
BANK21 = ROOT / "circuit_battery_v2_bank21_results.json"
RUNG = "circuit_battery_calibrated_selectivity"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "284651385e12e379caf8ae8508fe983c31d9a49a723be08a665a4f1d4b5915a5",
          SAT: "764035134854e0196e3ff0aed1200e14b7d26ad10beaaf93ad3fd52d9a3a46ab",
          BANK21: "7c1db82061bb6cac010fa7d2141cfc8e2faa4330759bdfe5aed299bab9a94d50",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
NLAY = R.NL
PER_CELL = 4 if SMOKE else 24
CEILING = 0.80          # the calibrated arm must leave at least 20% of the native margin standing
CTRL_MIN = 0.50         # a control family must natively achieve at least this margin to be usable
BARS = {"sat": 0.80, "selective": 0.25, "n_selective": 1, "usable_controls": 4, "material": 0.15,
        "repro": 0.15, "floor": 0.5}
NULLS = {"sat_ge": 1.00, "n_selective_le": 0, "usable_le": 1, "material_le": 0.05}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def main():
    t0 = time.time()
    check_hashes()
    b21 = json.load(open(BANK21))
    tasks = [t for t in b21["summary"]["capable"] if b21["tasks"][t]["writer"] == "attn8"]
    m = fastload.load_model_fast().to(DEV).eval()
    fwd = [0]
    readers = [("mlp", 8)] + [(kd, l) for l in range(9, NLAY) for kd in ("attn", "mlp")]
    LADDER = [("full", tuple(readers) + ("direct",)),
              ("half", tuple(readers[:max(1, len(readers) // 2)])),
              ("quarter", tuple(readers[:max(1, len(readers) // 4)])),
              ("eighth", tuple(readers[:max(1, len(readers) // 8)]))]
    results = {}
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        fams = set(BANK.TASKS[tid].families)
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)

        def measure(split, fam, arm):
            dm, nat = [], []
            for b in CB.batches([r for r in rows if r["family"] == fam and r["split"] == split]):
                ids, fin, ans = CB.pack(b, "base")
                lg = CB.run(m, ids, fin); fwd[0] += 1
                mn = CB.margins(lg, ans, cand)
                nat.append(mn.cpu().numpy())
                if arm is None:
                    continue
                lg2 = CB.run(m, ids, fin, writer=("attn", 8), removed=arm); fwd[0] += 1
                dm.append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
            n = float(np.concatenate(nat).mean()) if nat else float("nan")
            d = float(np.concatenate(dm).mean()) if dm else float("nan")
            return n, d

        # ---- CALIBRATE on FIT ----
        chosen, ladder_sat = None, {}
        for name, arm in LADDER:
            n, d = measure("FIT", "A1", arm)
            ladder_sat[name] = d / max(n, BARS["floor"])
            if chosen is None and ladder_sat[name] <= CEILING:
                chosen = (name, arm)
        if chosen is None:
            chosen = LADDER[-1]
        # ---- SCORE on SELECT with the chosen arm ----
        natives, dmg = {}, {}
        for fam in ("A1", "P", "C"):
            if fam not in fams:
                continue
            n, d = measure("SELECT", fam, chosen[1])
            natives[fam], dmg[fam] = n, d
        usable = [f for f in ("P", "C") if f in natives and natives[f] >= CTRL_MIN]
        ratio = (max(abs(dmg[f]) for f in usable) / max(dmg["A1"], BARS["floor"])) if usable else float("nan")
        # the FULL arm on the same rows, for the instrument check against SS2840
        _n, dfull = measure("SELECT", "A1", LADDER[0][1])
        fullr = {}
        for f in ("P", "C"):
            if f in natives:
                _n2, d2 = measure("SELECT", f, LADDER[0][1])
                fullr[f] = d2
        full_ratio = (max(abs(v) for v in fullr.values()) / max(dfull, BARS["floor"])) if fullr else float("nan")
        results[tid] = {"arm": chosen[0], "ladder_saturation_fit": ladder_sat,
                        "select_saturation": dmg["A1"] / max(natives["A1"], BARS["floor"]),
                        "native": natives, "damage": dmg, "usable_controls": usable,
                        "ratio": ratio, "full_arm_ratio": full_ratio,
                        "battery_ratio": b21["tasks"][tid]["selectivity_ratio"]}
        p = results[tid]
        print(f"[calib] {tid:30s} arm={p['arm']:8s} sat={p['select_saturation']:.2f} "
              f"usable={usable} ratio={ratio:.2f} (battery {p['battery_ratio']:.2f})", flush=True)

    sats = [results[t]["select_saturation"] for t in results]
    ratios = {t: results[t]["ratio"] for t in results if not np.isnan(results[t]["ratio"])}
    selective = [t for t, v in ratios.items() if v <= BARS["selective"]]
    usable_c = [t for t in results if "C" in results[t]["usable_controls"]]
    material = [abs(ratios[t] - results[t]["battery_ratio"]) for t in ratios]
    repro = max((abs(results[t]["full_arm_ratio"] - results[t]["battery_ratio"])
                 for t in results if not np.isnan(results[t]["full_arm_ratio"])), default=float("nan"))
    preds = {
        'pred_a_calibration_lands_below_the_ceiling': bool(float(np.median(sats)) <= BARS["sat"]),
        'pred_b_some_behaviour_is_now_selective': bool(len(selective) >= BARS["n_selective"]),
        'pred_c_enough_copy_controls_are_usable': bool(len(usable_c) >= BARS["usable_controls"]),
        'pred_d_the_correction_is_material': bool(material and float(np.median(material)) >= BARS["material"]),
        'pred_e_the_full_arm_still_reproduces_the_battery': bool(repro <= BARS["repro"]),
    }
    nulls = {
        "a_null_still_saturated": bool(float(np.median(sats)) >= NULLS["sat_ge"]),
        "b_null_still_none_selective": bool(len(selective) <= NULLS["n_selective_le"]),
        "c_null_controls_unusable": bool(len(usable_c) <= NULLS["usable_le"]),
        "d_null_immaterial": bool(material and float(np.median(material)) <= NULLS["material_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "ceiling": CEILING, "control_min_margin": CTRL_MIN, "tasks": sorted(results),
              "summary": {"median_select_saturation": float(np.median(sats)),
                          "selective_behaviours": selective, "usable_copy_controls": usable_c,
                          "median_ratio_change": float(np.median(material)) if material else None,
                          "max_full_arm_repro_gap": repro,
                          "arms_chosen": {t: results[t]["arm"] for t in results},
                          "ratios": ratios},
              "tasks_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1300])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: calibrated selectivity (arm ladder to saturation <= {CEILING}, controls verified at "
              f">= {CTRL_MIN} native margin) over SS2840's capable attn8 behaviours; no model loaded")
        sys.exit(0)
    main()
