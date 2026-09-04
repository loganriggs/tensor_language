"""circuit_battery_selective_band_overlap -- if per-component selectivity transports, WHICH components form the selective set, and is that SET stable?

SS2870 established a dissociation: the per-COMPONENT selectivity ranking survives a population change (rho .763 between OOD and TEST,
SS2868), while the per-BEHAVIOUR ordering collapses (rho .253). And SS2862/SS2864/SS2867/SS2869 have shown four times over that the
ARGMIN of that component ranking never reproduces -- 0 of 7, every time.

A stable ranking with an unstable minimum is what many components sharing a similar true value looks like. This rung tests that
directly, on the object the evidence says is robust: not the single most selective component, but the SET. It scores all 36 components
on all four populations (FIT, SELECT, TEST, OOD -- the last built from held-out vocabulary pools) and measures the Jaccard overlap of
the top-k selective sets between populations, against the overlap a random k-subset would give.

If the top-k set is stable while its argmin is not, "the selective band" is a real structure and the campaign should name a SET rather
than a component. If the set is no more stable than random, then the rho .763 ranking is carried entirely by the inert tail and there
is no band to name -- which would retire the account SS2867-SS2870 have been converging on.

# BQGATE: EXPERIMENT  pred_a_the_top_set_beats_a_random_subset
#                     pred_b_the_top_set_survives_the_population_change
#                     pred_c_the_band_is_contiguous_in_depth
#                     pred_d_the_argmin_is_unstable_where_the_set_is_not
#                     pred_e_inert_components_are_gated_out

SIGN CONVENTION: d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that condition's own answer, NEGATIVE = it HELPS. selectivity =
|d_C| / max(d_A1, .5), LOWER = MORE SELECTIVE. No CE and no SS312 L2; nothing installs. (Frontier: L2 is CE ADDED ABOVE THE REAL MODEL,
LOWER IS BETTER, SS2135; norm-2304 at 2.6735, SS2125 stands.)
Preregistration: polynomial_causal/CIRCUIT_BATTERY_SELECTIVE_BAND_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/"
            "circuit_battery_selective_band_overlap.py")
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
PREREG = R.POLY / "CIRCUIT_BATTERY_SELECTIVE_BAND_PREREGISTRATION.md"
V6RES = ROOT / "circuit_battery_variable_dependence_selectivity_results.json"
CALIB = ROOT / "circuit_battery_calibrated_selectivity_results.json"
RUNG = "circuit_battery_selective_band_overlap"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "bac8e6212360a45e25ad4b731cefcee7027af9bc4da6742111848c91e127d64d", V6RES: "940c1dc148d89d6458c6eecf1052f8f49701d1ffa149a8b3a0bffc4bd28d7e74",
          CALIB: "bb493ffa74ac41381155a8c72a915aa92cc6714200fee0b9e19e590be892ee4f",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
NLAY = R.NL
ENC = BANK.ENC
PER_CELL = 4 if SMOKE else 24
SPLITS = ("FIT", "SELECT", "TEST", "OOD")
WRITER = ("attn", 8)
LIVE_MIN = 0.10          # admissibility: a stand-in must DO something before its ratio is eligible (SS2820's lesson)
BARS = {"over_random": 0.25, "pop_overlap": 0.40, "contiguity": 0.75, "n_argmin_same": 1, "floor": 0.5}
NULLS = {"over_random_le": 0.05, "pop_overlap_le": 0.20, "contiguity_le": 0.50, "n_argmin_same_ge": 4}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def main():
    t0 = time.time()
    check_hashes()
    v6 = json.load(open(V6RES))
    tasks = sorted(v6["tasks"])
    m = fastload.load_model_fast().to(DEV).eval()
    fwd = [0]
    comps = [(kd, l) for l in range(NLAY) for kd in ("attn", "mlp")]
    results = {}
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        cand = torch.tensor(sorted({ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)

        def pairs(fam, sp):
            if fam == "C3":
                return [v for r in rows if r["family"] == "C" and r["split"] == sp
                        for v in [CCR.variants(r).get("v2_triple")] if v]
            return [(r["base_text"], r["base_answer"]) for r in rows
                    if r["family"] == fam and r["split"] == sp]

        def measure(ps, stand_in):
            keep = [(x, y) for x, y in ps
                    if len(ENC.encode(y)) == 1 and ENC.encode(x + y) == ENC.encode(x) + ENC.encode(y)]
            if not keep:
                return float("nan")
            by_len = {}
            for x, y in keep:
                by_len.setdefault(len(ENC.encode(x)), []).append((x, y))
            dm = []
            for grp in by_len.values():
                for i in range(0, len(grp), 32):
                    ch = grp[i:i + 32]
                    ids = torch.tensor([ENC.encode(x) for x, _ in ch], device=DEV)
                    fin = torch.full((len(ch),), ids.size(1) - 1, device=DEV, dtype=torch.long)
                    ans = torch.tensor([ENC.encode(y)[0] for _, y in ch], device=DEV)
                    lg = CB.run(m, ids, fin); fwd[0] += 1
                    readers = [(kd, l) for l in range(stand_in[1], NLAY) for kd in ("attn", "mlp")
                               if not (l == stand_in[1] and kd == stand_in[0])]
                    lg2 = CB.run(m, ids, fin, writer=stand_in, removed=tuple(readers) + ("direct",)); fwd[0] += 1
                    dm.append((CB.margins(lg, ans, cand) - CB.margins(lg2, ans, cand)).cpu().numpy())
            return float(np.concatenate(dm).mean())

        per_split = {}
        for sp in SPLITS:
            a1p, c3p = pairs("A1", sp), pairs("C3", sp)
            per_comp = {}
            for c in comps:
                da1 = measure(a1p, c)
                dc = measure(c3p, c)
                live = abs(da1) >= LIVE_MIN
                per_comp[f"{c[0]}{c[1]}"] = {"d_A1": da1, "d_C": dc, "live": bool(live),
                                             "selectivity": (abs(dc) / max(da1, BARS["floor"])) if live else None}
            per_split[sp] = per_comp
        results[tid] = per_split
        w = f"{WRITER[0]}{WRITER[1]}"
        vals = [v["selectivity"] for v in per_split["OOD"].values() if v["selectivity"] is not None]
        me = per_split["OOD"][w]["selectivity"]
        pct = (float(np.mean([v < me for v in vals])) if (vals and me is not None) else float("nan"))
        print(f"[null] {tid:30s} attn8 sel={me if me is None else round(me,3)} "
              f"percentile={pct:.3f} live_components={len(vals)}/{len(comps)}", flush=True)

    w = f"{WRITER[0]}{WRITER[1]}"
    K = 8
    rng = np.random.default_rng(20260904)

    def topk(ps, sp):
        v = {k: x["selectivity"] for k, x in ps[sp].items() if x["selectivity"] is not None}
        return set(sorted(v, key=v.get)[:K]), v

    def jac(a, b):
        return len(a & b) / len(a | b) if (a | b) else float("nan")

    pairs = [("SELECT", "TEST"), ("SELECT", "OOD"), ("TEST", "OOD"), ("FIT", "OOD")]
    overlaps, rand_overlaps, contig, argmin_same = {}, {}, {}, {}
    for tid, ps in results.items():
        sets = {sp: topk(ps, sp)[0] for sp in SPLITS if sp in ps}
        live = {sp: set(topk(ps, sp)[1]) for sp in SPLITS if sp in ps}
        overlaps[tid] = {f"{a}|{b}": jac(sets[a], sets[b]) for a, b in pairs if a in sets and b in sets}
        # a random k-subset of the SAME live pool, so the comparison is not inflated by pool-size differences
        rnd = []
        for a, b in pairs:
            if a in sets and b in sets:
                for _ in range(50):
                    ra = set(rng.choice(sorted(live[a]), size=min(K, len(live[a])), replace=False))
                    rb = set(rng.choice(sorted(live[b]), size=min(K, len(live[b])), replace=False))
                    rnd.append(jac(ra, rb))
        rand_overlaps[tid] = float(np.mean(rnd)) if rnd else float("nan")
        # contiguity: fraction of the OOD top-k lying in a single window of consecutive layers
        oo = sets.get("OOD", set())
        lay = sorted(int("".join(c for c in x if c.isdigit())) for x in oo)
        contig[tid] = (float(max(sum(1 for l in lay if lo <= l <= lo + 7) for lo in range(NLAY)) / len(lay))
                       if lay else float("nan"))
        am = {sp: (min(topk(ps, sp)[1], key=topk(ps, sp)[1].get) if topk(ps, sp)[1] else None) for sp in sets}
        argmin_same[tid] = bool(len({v for v in am.values() if v}) == 1)
    med_ov = {k: float(np.median([o[k] for o in overlaps.values() if k in o and not np.isnan(o[k])]))
              for k in [f"{a}|{b}" for a, b in pairs]}
    med_rand = float(np.median([v for v in rand_overlaps.values() if not np.isnan(v)]))
    pop_ov = float(np.median([o["TEST|OOD"] for o in overlaps.values() if "TEST|OOD" in o]))
    med_contig = float(np.median([v for v in contig.values() if not np.isnan(v)]))
    n_argmin_same = sum(1 for v in argmin_same.values() if v)
    preds = {
        'pred_a_the_top_set_beats_a_random_subset': bool(
            med_ov.get("SELECT|TEST", float("nan")) - med_rand >= BARS["over_random"]),
        'pred_b_the_top_set_survives_the_population_change': bool(pop_ov >= BARS["pop_overlap"]),
        'pred_c_the_band_is_contiguous_in_depth': bool(med_contig >= BARS["contiguity"]),
        'pred_d_the_argmin_is_unstable_where_the_set_is_not': bool(n_argmin_same <= BARS["n_argmin_same"]),
        'pred_e_inert_components_are_gated_out': bool(
            any(v["selectivity"] is None for ps in results.values() for sp in ps for v in ps[sp].values())),
    }
    nulls = {
        "a_null_top_set_is_random": bool(med_ov.get("SELECT|TEST", float("nan")) - med_rand <= NULLS["over_random_le"]),
        "b_null_set_does_not_survive_population": bool(pop_ov <= NULLS["pop_overlap_le"]),
        "c_null_band_is_scattered": bool(med_contig <= NULLS["contiguity_le"]),
        "d_null_argmin_is_stable": bool(n_argmin_same >= NULLS["n_argmin_same_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "live_min": LIVE_MIN, "tasks": tasks, "splits": list(SPLITS), "smoke": SMOKE,
              "summary": {"k": K, "median_jaccard_by_pair": med_ov,
                          "median_random_subset_jaccard": med_rand,
                          "population_overlap_test_vs_ood": pop_ov,
                          "median_depth_contiguity": med_contig,
                          "n_behaviours_with_one_argmin": n_argmin_same,
                          "per_task_overlaps": overlaps, "per_task_contiguity": contig,
                          "n_behaviours": len(results)},
              "tasks_detail": results,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"],
                      "price": result["price"]}, indent=1)[:1500])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: the v6 selectivity metric computed with all 36 components standing in as the "
              f"writer, so attn8 is read against a distribution rather than an inherited bar")
        sys.exit(0)
    main()
