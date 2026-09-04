"""circuit_battery_donor_control_sweep -- what selectivity reads across the WHOLE bank once the control is a real control.

Companion to circuit_battery_preserving_control_repair (protocol v5), which establishes the defect on the seven attn8 successor
behaviours. This rung asks the same question of all 21 behaviours, each at ITS OWN identified writer, with the battery's FULL arm --
the exact configuration whose `selectivity_ratio` SS2840 published and later rungs cite.

The defect, restated: the bank stores a group's families as transformations of ONE situation and puts the transformation in the DONOR,
so A1/A2/P share `base_text`. Scoring rungs call `pack(b, "base")` for every family, so the answer-preserving control is the target's
own prompts. In the landed SS2840 receipt this is visible without running anything: `control_d_m["P"]` is BITWISE identical to
`split_d_m["FULL"]` for 9 of the 16 tasks carrying both, and every such task with a positive value reads `selectivity_ratio` exactly
1.000. This rung measures the donor-side control -- same answer, different causal variable -- and reports what the bank's selectivity
actually is.

# BQGATE: EXPERIMENT  pred_a_the_structural_census_predicts_the_landed_receipt
#                     pred_b_the_degenerate_control_reproduces_the_target
#                     pred_c_the_donor_control_separates_from_the_target
#                     pred_d_some_behaviour_is_selective_once_corrected
#                     pred_e_the_correction_is_not_a_uniform_shift

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's own answer; ratio = max over usable controls of
|d_control| / max(d_A1, .5), LOWER IS MORE SELECTIVE. Note SS2840 used a SIGNED max over controls, which this rung reproduces for the
comparison arm and replaces with |.| for the corrected one, stating both. No CE and no SS312 L2; nothing installs. (Frontier
convention, per standing rule though unused here: frontier L2 is CE ADDED ABOVE THE REAL MODEL, LOWER IS BETTER, SS2135.)
Preregistration: polynomial_causal/CIRCUIT_BATTERY_DONOR_CONTROL_SWEEP_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_donor_control_sweep.py")
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
PREREG = R.POLY / "CIRCUIT_BATTERY_DONOR_CONTROL_SWEEP_PREREGISTRATION.md"
BANK21 = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_donor_control_sweep"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "224dc3e52856d2e8425aaba1ea738a0b6b0abde3dd24b83f21e6eb8668154a25",
          BANK21: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
NLAY = R.NL
ENC = BANK.ENC
PER_CELL = 4 if SMOKE else 24
ATOMICS = 0.015
BARS = {"census_agreement": 1.00, "reproduce": 0.05, "separate": 0.15, "n_separate": 6,
        "selective": 0.25, "n_selective": 1, "spread": 0.20, "floor": 0.5}
NULLS = {"census_agreement_lt": 1.00, "n_separate_le": 2, "n_selective_le": 0, "spread_le": 0.05}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def prompt_set(rows, fam, split, side):
    return [(r[f"{side}_text"], r[f"{side}_answer"]) for r in rows
            if r["family"] == fam and r["split"] == split]


def main():
    t0 = time.time()
    check_hashes()
    b21 = json.load(open(BANK21))
    m = fastload.load_model_fast().to(DEV).eval()
    fwd = [0]
    results = {}
    for tid in sorted(BANK.TASKS):
        pub = b21["tasks"].get(tid, {})
        fams = set(BANK.TASKS[tid].families)
        if not {"A1", "P"} <= fams or "writer" not in pub:
            continue
        kind = "".join(c for c in pub["writer"] if not c.isdigit())
        layer = int("".join(c for c in pub["writer"] if c.isdigit()))
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        cand = torch.tensor(sorted({ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        readers = [(kd, l) for l in range(layer, NLAY) for kd in ("attn", "mlp")
                   if not (l == layer and kd == kind)]
        FULL = tuple(readers) + ("direct",)

        def measure(pairs):
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
                    lg2 = CB.run(m, ids, fin, writer=(kind, layer), removed=FULL); fwd[0] += 1
                    nat.append(mn.cpu().numpy())
                    dm.append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
            return (float(np.concatenate(nat).mean()), float(np.concatenate(dm).mean()), len(keep))

        a1 = set(prompt_set(rows, "A1", "SELECT", "base"))
        pb = prompt_set(rows, "P", "SELECT", "base")
        frac = (sum(1 for x in pb if x in a1) / len(pb)) if pb else float("nan")
        cond = {"A1": prompt_set(rows, "A1", "SELECT", "base"), "P_base": pb,
                "P_donor": prompt_set(rows, "P", "SELECT", "donor")}
        if "C" in fams:
            cond["C_base"] = prompt_set(rows, "C", "SELECT", "base")
        nat, dmg = {}, {}
        for k, pairs in cond.items():
            n, d, kk = measure(pairs)
            if kk:
                nat[k], dmg[k] = n, d
        den = max(dmg["A1"], BARS["floor"])
        r_deg = max([dmg[k] for k in ("P_base", "C_base") if k in dmg]) / den      # SS2840's SIGNED max
        r_cor = max([abs(dmg[k]) for k in ("P_donor", "C_base") if k in dmg]) / den
        results[tid] = {"writer": pub["writer"], "capable": pub.get("capable"),
                        "frac_base_in_a1": frac, "native": nat, "damage": dmg,
                        "ratio_degenerate": r_deg, "ratio_corrected": r_cor,
                        "ratio_published": pub.get("selectivity_ratio"),
                        "published_P": pub.get("control_d_m", {}).get("P"),
                        "published_FULL": pub.get("split_d_m", {}).get("FULL"),
                        "published_P_is_FULL": (pub.get("control_d_m", {}).get("P")
                                                == pub.get("split_d_m", {}).get("FULL"))}
        p = results[tid]
        print(f"[sweep] {tid:32s} {pub['writer']:6s} ident={frac:.2f} pubPisFULL={p['published_P_is_FULL']} "
              f"deg={r_deg:+.2f} cor={r_cor:+.2f} (pub {p['ratio_published']})", flush=True)

    # pred_a: the no-GPU structural census predicts the landed receipt's bitwise coincidence, task for task
    agree = [(v["frac_base_in_a1"] >= 1.0) == bool(v["published_P_is_FULL"])
             for v in results.values() if v["published_P_is_FULL"] is not None]
    agreement = float(np.mean(agree)) if agree else float("nan")
    repro = [abs(v["damage"]["P_base"] - v["damage"]["A1"]) for v in results.values()
             if v["frac_base_in_a1"] >= 1.0]
    sep = [abs(v["damage"]["P_donor"] - v["damage"]["A1"]) for v in results.values() if "P_donor" in v["damage"]]
    cap = {t: v for t, v in results.items() if v["capable"]}
    selective = [t for t, v in cap.items() if v["ratio_corrected"] <= BARS["selective"]]
    spread = (float(np.percentile([v["ratio_corrected"] for v in cap.values()], 75)
                    - np.percentile([v["ratio_corrected"] for v in cap.values()], 25)) if cap else float("nan"))
    preds = {
        'pred_a_the_structural_census_predicts_the_landed_receipt': bool(agreement >= BARS["census_agreement"]),
        'pred_b_the_degenerate_control_reproduces_the_target': bool(repro and max(repro) <= ATOMICS),
        'pred_c_the_donor_control_separates_from_the_target': bool(
            sum(1 for s in sep if s >= BARS["separate"]) >= BARS["n_separate"]),
        'pred_d_some_behaviour_is_selective_once_corrected': bool(len(selective) >= BARS["n_selective"]),
        'pred_e_the_correction_is_not_a_uniform_shift': bool(spread >= BARS["spread"]),
    }
    nulls = {
        "a_null_census_does_not_predict": bool(agreement < NULLS["census_agreement_lt"]),
        "c_null_donor_control_tracks_target": bool(sum(1 for s in sep if s >= BARS["separate"]) <= NULLS["n_separate_le"]),
        "d_null_still_none_selective": bool(len(selective) <= NULLS["n_selective_le"]),
        "e_null_uniform_shift": bool(spread <= NULLS["spread_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "tasks": sorted(results), "smoke": SMOKE,
              "summary": {"n_tasks": len(results), "census_receipt_agreement": agreement,
                          "max_degenerate_reproduction_gap": max(repro) if repro else None,
                          "n_degenerate": sum(1 for v in results.values() if v["frac_base_in_a1"] >= 1.0),
                          "median_donor_separation": float(np.median(sep)) if sep else None,
                          "n_donor_separating": sum(1 for s in sep if s >= BARS["separate"]),
                          "selective_corrected": selective, "n_capable": len(cap),
                          "corrected_iqr": spread,
                          "ratio_corrected": {t: v["ratio_corrected"] for t, v in cap.items()},
                          "ratio_published": {t: v["ratio_published"] for t, v in cap.items()}},
              "tasks_detail": results,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"],
                      "price": result["price"]}, indent=1)[:1600])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: all 21 behaviours at their own writers, FULL arm, degenerate vs donor-side "
              f"answer-preserving control")
        sys.exit(0)
    main()
