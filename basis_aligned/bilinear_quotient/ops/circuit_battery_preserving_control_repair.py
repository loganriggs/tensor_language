"""circuit_battery_preserving_control_repair -- the answer-preserving CONTROL was the TARGET CONDITION ITSELF.

Found while smoke-testing the protocol-v4 selectivity re-score (that rung is WITHDRAWN UNRUN; see the v5 preregistration): for five of
seven behaviours the A1 and P families read IDENTICAL native margins AND identical damages, to every printed digit. The cause is
structural, not numerical. The bank stores a group's four families as transformations of ONE situation, and the transformation lives in
the DONOR: A1, A2 and P all share the same `base_text`/`base_answer`; only C has a base of its own. Every scoring rung so far has
called `pack(b, "base")` for EVERY family, so "the answer-preserving control" was measured on the target's own prompts, and

    ratio = |d_P| / max(d_A1, .5) = 1.000 identically, by construction, not by measurement.

SS2852 read exactly 1.00 on six of eight behaviours and its docstring attributes that to both sides being saturated. The real cause is
that numerator and denominator are the same number. This rung therefore does NOT assume the defect -- it MEASURES it, with an
independent physical control that needs no GPU at all (byte-identity of the prompt sets), as the standing rule requires of any
correction that would flip a published conclusion. The correct answer-preserving condition is the P family's DONOR: same answer,
different causal variable ("64) beacon / 65) mosaic" -> 66 against "64) jasmine / 65) vine" -> 66).

It then re-scores selectivity with BOTH repairs in place -- the donor-side preserving control and SS2857's repaired copy control --
and reports what the campaign's standing negative becomes.

# BQGATE: EXPERIMENT  pred_a_the_preserving_control_is_byte_identical_to_the_target
#                     pred_b_the_identity_forces_the_damage_to_coincide
#                     pred_c_the_donor_side_control_is_informative
#                     pred_d_the_corrected_score_changes_the_verdict
#                     pred_e_the_derived_rows_are_valid

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's own answer; ratio = max over USABLE controls of
|d_control| / max(d_A1, .5), LOWER IS MORE SELECTIVE. No CE and no SS312 L2; nothing installs. (Frontier convention, per standing rule
though unused here: frontier L2 is CE ADDED ABOVE THE REAL MODEL, LOWER IS BETTER, SS2135; frontier is norm-2304 at 2.6735, SS2125.)
Preregistration: polynomial_causal/CIRCUIT_BATTERY_PROTOCOL_V5_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_preserving_control_repair.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
import circuit_battery_copy_control_redesign as CCR
import fastload
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_PROTOCOL_V5_PREREGISTRATION.md"
CALIB = ROOT / "circuit_battery_calibrated_selectivity_results.json"
CCRES = ROOT / "circuit_battery_copy_control_redesign_results.json"
RUNG = "circuit_battery_preserving_control_repair"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "d81e6db5f48abbbb99c348f09a6bd0d6c11f86d54f560b3fd33cbd2241a54373",
          CALIB: "bb493ffa74ac41381155a8c72a915aa92cc6714200fee0b9e19e590be892ee4f",
          CCRES: "3058347f253d5aae521bd3c81b70c77ab0a485c44269088405f4e259a23a96ec",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
NLAY = R.NL
ENC = BANK.ENC
PER_CELL = 4 if SMOKE else 24          # SS2852's value, so A1 reproduces exactly
CEILING = 0.80                          # SS2852's calibration ceiling, unchanged
CTRL_MIN = 0.50                         # SS2852's control usability bar, unchanged
ATOMICS = 0.015                         # registered CUDA-atomics reproduction tolerance
BARS = {"identical": 1.00, "n_identical": 5, "coincide": ATOMICS, "informative": 0.15,
        "n_verdict": 1, "selective": 0.25, "floor": 0.5}
NULLS = {"n_identical_le": 0, "informative_le": 0.05, "n_verdict_le": 0}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def prompt_set(rows, fam, split, side):
    """The literal (text, answer) pairs a scoring rung would feed the model for this family."""
    return [(r[f"{side}_text"], r[f"{side}_answer"]) for r in rows
            if r["family"] == fam and r["split"] == split]


def census():
    """No-GPU physical control: how often is the answer-preserving control byte-identical to the target?"""
    out = {}
    for tid in sorted(BANK.TASKS):
        fams = set(BANK.TASKS[tid].families)
        if not {"A1", "P"} <= fams:
            continue
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        a1 = set(prompt_set(rows, "A1", "SELECT", "base"))
        pb = prompt_set(rows, "P", "SELECT", "base")
        pd = prompt_set(rows, "P", "SELECT", "donor")
        out[tid] = {"n_p": len(pb),
                    "frac_base_in_a1": (sum(1 for x in pb if x in a1) / len(pb)) if pb else float("nan"),
                    "frac_donor_in_a1": (sum(1 for x in pd if x in a1) / len(pd)) if pd else float("nan"),
                    "donor_answer_matches_base": (sum(1 for r in rows
                                                      if r["family"] == "P" and r["split"] == "SELECT"
                                                      and r["donor_answer"] == r["base_answer"])
                                                  / len(pb)) if pb else float("nan")}
    return out


def main():
    t0 = time.time()
    check_hashes()
    cen = census()
    calib = json.load(open(CALIB))
    ccr = json.load(open(CCRES))
    tasks = [t for t in calib["tasks"] if "per_design" in ccr["tasks_detail"].get(t, {})]
    m = fastload.load_model_fast().to(DEV).eval()
    fwd = [0]
    readers = [("mlp", 8)] + [(kd, l) for l in range(9, NLAY) for kd in ("attn", "mlp")]
    LADDER = [("full", tuple(readers) + ("direct",)),
              ("half", tuple(readers[:max(1, len(readers) // 2)])),
              ("quarter", tuple(readers[:max(1, len(readers) // 4)])),
              ("eighth", tuple(readers[:max(1, len(readers) // 8)]))]
    results, valid_total, valid_bad = {}, 0, 0
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        cand = torch.tensor(sorted({ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)

        def measure(pairs, arm):
            keep = [(x, y) for x, y in pairs
                    if len(ENC.encode(y)) == 1 and ENC.encode(x + y) == ENC.encode(x) + ENC.encode(y)]
            if not keep:
                return float("nan"), float("nan"), 0
            by_len = {}
            for x, y in keep:
                by_len.setdefault(len(ENC.encode(x)), []).append((x, y))
            nat, dm = [], []
            for grp in by_len.values():
                for i in range(0, len(grp), 32):
                    ch = grp[i:i + 32]
                    ids = torch.tensor([ENC.encode(x) for x, _ in ch], device=DEV)
                    fin = torch.full((len(ch),), ids.size(1) - 1, device=DEV, dtype=torch.long)
                    ans = torch.tensor([ENC.encode(y)[0] for _, y in ch], device=DEV)
                    lg = CB.run(m, ids, fin); fwd[0] += 1
                    mn = CB.margins(lg, ans, cand)
                    nat.append(mn.cpu().numpy())
                    if arm is not None:
                        lg2 = CB.run(m, ids, fin, writer=("attn", 8), removed=arm); fwd[0] += 1
                        dm.append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
            return (float(np.concatenate(nat).mean()),
                    float(np.concatenate(dm).mean()) if dm else float("nan"), len(keep))

        def c3_pairs(split):
            out = []
            for r in rows:
                if r["family"] == "C" and r["split"] == split:
                    v = CCR.variants(r).get("v2_triple")
                    if v:
                        out.append(v)
            return out

        # ---- pred_e: validity of the derived copy-control rows ----
        a1_ans = {r["group_id"]: r["base_answer"] for r in rows if r["family"] == "A1" and r["split"] == "SELECT"}
        for r in rows:
            if r["family"] != "C" or r["split"] != "SELECT":
                continue
            v = CCR.variants(r).get("v2_triple")
            if v is None:
                continue
            x, y = v
            valid_total += 1
            if not (len(ENC.encode(y)) == 1 and ENC.encode(x + y) == ENC.encode(x) + ENC.encode(y)
                    and a1_ans.get(r["group_id"]) != y
                    and y.strip() in x.translate(str.maketrans("\n.:+=()[]{},%", "             ")).split()):
                valid_bad += 1

        # ---- calibrate exactly as SS2852 ----
        chosen = None
        for name, arm in LADDER:
            n, d, _ = measure(prompt_set(rows, "A1", "FIT", "base"), arm)
            if chosen is None and d / max(n, BARS["floor"]) <= CEILING:
                chosen = (name, arm)
        if chosen is None:
            chosen = LADDER[-1]
        # ---- score every condition on SELECT ----
        conds = {"A1": prompt_set(rows, "A1", "SELECT", "base"),
                 "P_base": prompt_set(rows, "P", "SELECT", "base"),
                 "P_donor": prompt_set(rows, "P", "SELECT", "donor"),
                 "C_base": prompt_set(rows, "C", "SELECT", "base"),
                 "C3": c3_pairs("SELECT")}
        nat, dmg = {}, {}
        for k, pairs in conds.items():
            n, d, kk = measure(pairs, chosen[1])
            if kk:
                nat[k], dmg[k] = n, d
        old_us = [f for f in ("P_base", "C_base") if f in nat and nat[f] >= CTRL_MIN]
        new_us = [f for f in ("P_donor", "C3") if f in nat and nat[f] >= CTRL_MIN]
        r_old = (max(abs(dmg[f]) for f in old_us) / max(dmg["A1"], BARS["floor"])) if old_us else float("nan")
        r_new = (max(abs(dmg[f]) for f in new_us) / max(dmg["A1"], BARS["floor"])) if new_us else float("nan")
        results[tid] = {"arm": chosen[0], "arm_2852": calib["tasks_detail"][tid]["arm"],
                        "native": nat, "damage": dmg, "usable_old": old_us, "usable_new": new_us,
                        "ratio_old": r_old, "ratio_new": r_new,
                        "ratio_2852": calib["tasks_detail"][tid]["ratio"],
                        "census": cen.get(tid, {})}
        p = results[tid]
        print(f"[v5] {tid:30s} arm={chosen[0]:8s} identical={p['census'].get('frac_base_in_a1'):.2f} "
              f"dP_base-dA1={dmg.get('P_base', float('nan')) - dmg['A1']:+.4f} "
              f"old={r_old:.2f} new={r_new:.2f} (SS2852 {p['ratio_2852']:.2f})", flush=True)

    live = {t: v for t, v in results.items() if not np.isnan(v["ratio_new"])}
    n_ident = sum(1 for v in cen.values() if v["frac_base_in_a1"] >= BARS["identical"])
    coincide = [abs(v["damage"]["P_base"] - v["damage"]["A1"]) for v in results.values()
                if v["census"].get("frac_base_in_a1", 0) >= BARS["identical"] and "P_base" in v["damage"]]
    inform = [abs(v["damage"]["P_donor"] - v["damage"]["A1"]) for v in results.values() if "P_donor" in v["damage"]]
    selective = [t for t, v in live.items() if v["ratio_new"] <= BARS["selective"]]
    flipped = [t for t, v in live.items()
               if (v["ratio_new"] <= BARS["selective"]) != (v["ratio_2852"] <= BARS["selective"])]
    a1_drift = max(abs(v["native"]["A1"] - calib["tasks_detail"][t]["native"]["A1"]) for t, v in results.items())
    arm_same = all(v["arm"] == v["arm_2852"] for v in results.values())
    preds = {
        'pred_a_the_preserving_control_is_byte_identical_to_the_target': bool(n_ident >= BARS["n_identical"]),
        'pred_b_the_identity_forces_the_damage_to_coincide': bool(coincide and max(coincide) <= BARS["coincide"]),
        'pred_c_the_donor_side_control_is_informative': bool(inform and float(np.median(inform)) >= BARS["informative"]),
        'pred_d_the_corrected_score_changes_the_verdict': bool(len(flipped) >= BARS["n_verdict"]),
        'pred_e_the_derived_rows_are_valid': bool(valid_total > 0 and valid_bad == 0),
    }
    nulls = {
        "a_null_no_task_is_degenerate": bool(n_ident <= NULLS["n_identical_le"]),
        "c_null_donor_control_is_also_the_target": bool(inform and float(np.median(inform)) <= NULLS["informative_le"]),
        "d_null_verdict_unchanged": bool(len(flipped) <= NULLS["n_verdict_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "ceiling": CEILING, "control_min_margin": CTRL_MIN,
              "tasks": sorted(results), "census_all_tasks": cen,
              "summary": {"n_tasks_censused": len(cen), "n_degenerate_preserving_control": n_ident,
                          "max_dP_base_minus_dA1": max(coincide) if coincide else None,
                          "median_donor_control_separation": float(np.median(inform)) if inform else None,
                          "selective_corrected": selective, "verdict_flipped": flipped,
                          "n_live": len(live), "max_a1_native_drift": a1_drift, "arm_reproduces": arm_same,
                          "ratios_new": {t: v["ratio_new"] for t, v in live.items()},
                          "ratios_2852": {t: v["ratio_2852"] for t, v in live.items()},
                          "derived_rows_checked": valid_total, "derived_rows_invalid": valid_bad},
              "tasks_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"],
                      "price": result["price"]}, indent=1)[:1800])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: measures whether the answer-preserving control is byte-identical to the target "
              f"(no-GPU census) and re-scores with the donor-side control + SS2857's repaired copy control")
        sys.exit(0)
    main()
