#!/usr/bin/env python
"""circuit_battery_class_mass_localisation -- if the READ does not gate the answer class, what does?

SS2828: removing mlp10/mlp11's entire read of attention 8's write costs .0148 nats of candidate-class mass while moving the within-class
margin by 2.6 units -- the read decides WHICH member of the class, not whether the class applies. So the type gate ("a list label goes
here") is somewhere else. This rung localises it the same way the battery localises writers, but with the class-mass metric instead of
the margin: every one of the 36 components has its final-position write ablated outright and the loss of
log sum_{v in candidates} p(v) is measured on OOD rows. The margin damage of the same arms is recorded alongside, so the two maps can
be compared directly -- the question is whether the class gate and the member selector are the SAME components or different ones.

# BQGATE: EXPERIMENT  pred_a_class_mass_is_localised pred_b_class_gate_is_not_the_member_selector
#                     pred_c_class_gate_is_early pred_d_class_gate_is_shared_across_behaviours
#                     pred_e_attention8_is_not_the_class_gate

SIGN CONVENTION: class-mass damage d_c = logmass_NATIVE - logmass_arm in NATS, POSITIVE = the arm REMOVES class mass; margin damage
d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS. Nothing installs into the SS312 frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_CLASS_MASS_LOCALISATION_PREREGISTRATION.md
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_class_mass_localisation.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_CLASS_MASS_LOCALISATION_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_class_mass_localisation"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "81db11ee1d79ca8e674b13b7e64ab816ded134e6811f35c954b044899cd1d670",
          BATTERY: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
NL = R.NL
COMPONENTS = [(kd, l) for l in range(NL) for kd in ("attn", "mlp")]
PER_CELL = 4 if SMOKE else 16
BARS = {"top3_share": 0.60, "overlap": 0.50, "early_layer": 8, "shared_tasks": 4, "floor": 0.05}
NULLS = {"top3_share_le": 0.30, "overlap_ge": 0.80, "shared_tasks_le": 1}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def logmass(logits, cand):
    lp = torch.log_softmax(logits, dim=-1)
    return torch.logsumexp(lp[:, cand], dim=-1)


def main():
    t0 = time.time()
    check_hashes()
    b2 = json.load(open(BATTERY))
    tasks = [t for t in b2["summary"]["capable"] if b2["tasks"][t]["writer"] == "attn8"]
    m = R.load_model().to(DEV).eval()
    fwd = [0]
    results = {}
    for tid in tasks:
        rows = [r for r in BANK.build_rows(tid, per_cell=PER_CELL)
                if r["family"] == "A1" and r["split"] == "OOD"]
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        cls, mar = defaultdict(list), defaultdict(list)
        for b in CB.batches(rows):
            ids, fin, ans = CB.pack(b, "base")
            lg = CB.run(m, ids, fin); fwd[0] += 1
            cn, mn = logmass(lg, cand), CB.margins(lg, ans, cand)
            for comp in COMPONENTS:
                lg2 = CB.run(m, ids, fin, writer=comp, ablate=True); fwd[0] += 1
                name = f"{comp[0]}{comp[1]}"
                cls[name].append((cn - logmass(lg2, cand)).cpu().numpy())
                mar[name].append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
        c = {k: float(np.concatenate(v).mean()) for k, v in cls.items()}
        mm = {k: float(np.concatenate(v).mean()) for k, v in mar.items()}
        ctop = sorted(c, key=lambda k: -c[k])
        mtop = sorted(mm, key=lambda k: -mm[k])
        tot = max(sum(v for v in c.values() if v > 0), BARS["floor"])
        results[tid] = {
            "class_damage": c, "margin_damage": mm,
            "class_top": ctop[:6], "margin_top": mtop[:6],
            "class_top3_share": sum(c[k] for k in ctop[:3]) / tot,
            "overlap_top3": len(set(ctop[:3]) & set(mtop[:3])) / 3.0,
            "class_leader_layer": int("".join(ch for ch in ctop[0] if ch.isdigit())),
            "attn8_class_damage": c.get("attn8", float("nan")),
            "attn8_class_rank": ctop.index("attn8") + 1,
            "attn8_margin_rank": mtop.index("attn8") + 1,
            "rows": len(rows),
        }
        p = results[tid]
        print(f"[classloc] {tid:28s} class_top={p['class_top'][:3]} margin_top={p['margin_top'][:3]} "
              f"top3={p['class_top3_share']:.2f} overlap={p['overlap_top3']:.2f} "
              f"attn8_class_rank={p['attn8_class_rank']}", flush=True)

    leaders = defaultdict(int)
    for t in results:
        leaders[results[t]["class_top"][0]] += 1
    mode, mode_n = (max(leaders.items(), key=lambda kv: kv[1]) if leaders else ("", 0))
    med = lambda k: float(np.median([results[t][k] for t in results])) if results else float("nan")
    early = [t for t in results if results[t]["class_leader_layer"] <= BARS["early_layer"]]
    preds = {
        'pred_a_class_mass_is_localised': bool(med("class_top3_share") >= BARS["top3_share"]),
        'pred_b_class_gate_is_not_the_member_selector': bool(med("overlap_top3") <= BARS["overlap"]),
        'pred_c_class_gate_is_early': bool(len(early) >= BARS["shared_tasks"]),
        'pred_d_class_gate_is_shared_across_behaviours': bool(mode_n >= BARS["shared_tasks"]),
        'pred_e_attention8_is_not_the_class_gate':
            bool(results and float(np.median([results[t]["attn8_class_rank"] for t in results])) > 3),
    }
    nulls = {
        "a_null_diffuse": bool(med("class_top3_share") <= NULLS["top3_share_le"]),
        "b_null_same_components": bool(med("overlap_top3") >= NULLS["overlap_ge"]),
        "d_null_not_shared": bool(mode_n <= NULLS["shared_tasks_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "summary": {"tasks": sorted(results), "class_leaders": dict(leaders),
                          "mode_leader": mode, "mode_count": mode_n, "early_tasks": early,
                          "medians": {k: med(k) for k in ("class_top3_share", "overlap_top3",
                                                          "attn8_class_damage", "attn8_class_rank",
                                                          "attn8_margin_rank")}},
              "tasks": results, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "leaders": dict(leaders),
                      "medians": result["summary"]["medians"]}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2817 capable attn8-writer behaviours x {len(COMPONENTS)} component ablations, "
              f"class-mass and margin metrics on OOD; no model loaded")
        sys.exit(0)
    main()
