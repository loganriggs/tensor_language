#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets (v80 hub+8), recipe, rows and bars fixed before the run.
"""v96: why are the number and interrogative directions one-sided -- a default answer at the mean, or a missing dimension?

v95: mean-ablating the rank-1 full-specificity direction costs 1.30 nat on 'was' rows but 0.16 on 'were' rows (quantifier
hub+8), and 1.62 on 'whether' vs 0.42 on 'that' (complementizer hub+8); verb_preposition is two-sided (0.94 / 0.81). Two
mechanisms would produce this. (i) READOUT DEFAULT: the direction carries both values, but at the pooled mean the model's
readout already says the unmarked answer ('were', 'that'), so halving the signal barely moves that side and flips the other.
(ii) MISSING DIMENSION: the rank-1 direction carries only the marked value; the unmarked side's signal lives elsewhere in the
set, so interchange in the unmarked->marked direction is recovered and marked->unmarked is not. They separate on the
reverse interchange (base/donor sides swapped on the same ODD sentences) and on per-document flips at the mean point.

Recipe per set (identical to v80/v95): rank 1 per block, fit on pooled EVEN A1 + verb variants, complement 1.0, own C EVEN +
other five A1 EVEN controls at 30 each, 120 steps, lr 0.05, seed 0. All evaluation on ODD A1 rows (out of sample); 'forward'
= the rows as authored (base = unmarked answer, donor = marked), 'reverse' = `g.swap_base_donor` (base = marked).
Extraction fraction = rank-1 recovery / exact-set recovery in each direction. Flip = after mean-ablation the removed state
prefers the foil over the answer (fraction of documents on that side). Diff-in-means direction (v51-style) is the optimizer
control: same split, no fitting.

REGISTERED BEFORE THE RUN (quantifier_number, verb_complementizer, verb_preposition; hub+8 units from v80)
    pred_a_reproduce         pooled ODD A1 removal within 0.02 of v95 for all three sets.
    pred_b_symmetric_extract for quantifier AND complementizer the rank-1/exact fraction is >= 0.80 in BOTH directions and the
                             two directions differ by <= 0.15 (mechanism i). Worked: 0.93 / 0.88 True; 0.95 / 0.55 False.
    pred_c_default_flip      for quantifier AND complementizer the marked side's flip fraction is >= 0.50 and the unmarked
                             side's is <= 0.10 (the mean point reads as the unmarked answer). Worked: 0.70 / 0.05 True; 0.30 / 0.05 False.
    pred_d_two_sided_control verb_preposition's two flip fractions differ by <= 0.25.
    pred_e_dim_same_shape    the diff-in-means direction's side ratio is >= 3 for quantifier and >= 2 for complementizer
                             (the asymmetry belongs to the set's delta, not to the optimizer). Worked: 5.1 / 2.6 True; 1.8 / 2.6 False.
    Prior: a True; b ~70%; c ~60%; d ~65%; e ~70%.
    Reading: b True + c True = readout default (the tier-table removal numbers are honest but side-pooled; row 4 siblings on the
    unmarked side are default-side controls). b False = missing dimension: a registered rank-2 hypothesis for that set would
    then be licensed by a stated capability failure, not by this null.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_tier2_characterization_v23 as v23

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_default_side_v96_result.json"
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
V95 = ROOT / "circuits/followups/unit_six_sets_side_split_v95_result.json"
NAMES = ("quantifier_number", "verb_complementizer", "verb_preposition")
MARKED = {"quantifier_number": "was", "verb_complementizer": "whether", "verb_preposition": None}
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
REPRO, EXT_MIN, EXT_GAP, FLIP_HI, FLIP_LO, CTRL_GAP, DIM_RATIO = 0.02, 0.80, 0.15, 0.50, 0.10, 0.25, {"quantifier_number": 3.0, "verb_complementizer": 2.0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 700, 50000


def _plan():
    return {"candidate_id": "corpus.unit_default_side_v96", "lambda": LAM,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 3 * 2 * STEPS, "model_updates": 0, "fit_parameters": 3 * 13 * 128, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def removal_docs(backend, prep, units, q, mu):
    """Per-side: CE damage list, flip list (removed state prefers the foil), natural-preference list."""
    torch = backend.torch
    F = torch.nn.functional
    out = {}
    for side in ("base", "donor"):
        batch = prep.base_batch if side == "base" else prep.donor_batch
        cache = prep.base_cache if side == "base" else prep.donor_cache
        bg = dict(cache)
        for rid in batch.row_ids:
            for u in units:
                bg[(rid, u)] = mu[u]
        ans = torch.tensor(batch.answer_ids, device=backend.device)
        foil = torch.tensor(batch.foil_ids, device=backend.device)
        _, nat = g.forward_units(backend, batch, units=[], return_logits=True)
        _, rem = g.forward_units(backend, batch, units=units, donor_cache=bg, base_cache=cache, q=q, return_logits=True)
        lp_n, lp_r = F.log_softmax(nat.float(), -1), F.log_softmax(rem.float(), -1)
        i = torch.arange(len(batch.row_ids), device=backend.device)
        ce = (lp_n[i, ans] - lp_r[i, ans]).tolist()
        out[side] = {"ce": sum(ce) / len(ce), "flip": (lp_r[i, ans] < lp_r[i, foil]).float().mean().item(),
                     "natural_pref": (lp_n[i, ans] > lp_n[i, foil]).float().mean().item(), "documents": len(ce)}
    return out


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    v80 = json.loads(V80.read_text())["sets"]
    v95 = json.loads(V95.read_text())["summary"]
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")}}
    cross_even = {n: g.prepare(backend, g.rows_of(m, "A1")[0::2]) for n, m in modules.items()}

    R = {}
    for name in NAMES:
        m, units = modules[name], v80[name]["units"]
        a1 = g.rows_of(m, "A1")
        maps = v15.SETS[name][2] if name in v15.SETS else ()
        pool = g.prepare(backend, a1[0::2] + [r for mp in maps for r in g.lexical_variant(a1, mp)[0::2]])
        even_c = g.prepare(backend, g.rows_of(m, "C")[0::2])
        xc = (even_c,) + tuple(cross_even[n] for n in modules if n != name)
        mu = {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (pool.base_cache, pool.donor_cache) for rid in pool.base_batch.row_ids]).mean(0) for u in units}
        q, _ = g.fit_block_subspace_constrained(backend, pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW, controls=xc, control_weight=LAM * len(xc), mu=mu)
        q_dim = g.block_diff_in_means(backend, pool, units)

        odd = a1[1::2]
        fwd, rev = g.prepare(backend, odd), g.prepare(backend, g.swap_base_donor(odd))
        ext = {}
        for lab, prep in (("forward", fwd), ("reverse", rev)):
            exact = g.recovery(prep, g.patched_axis(backend, prep, units))
            sub = g.recovery(prep, g.patched_axis(backend, prep, units, q=q))
            ext[lab] = {"base_answer": odd[0]["base_answer"].strip() if lab == "forward" else odd[0]["donor_answer"].strip(),
                        "donor_answer": odd[0]["donor_answer"].strip() if lab == "forward" else odd[0]["base_answer"].strip(),
                        "exact": exact, "rank1": sub, "fraction": sub / exact if abs(exact) > 1e-6 else None}
        rem = removal_docs(backend, fwd, units, q, mu)
        rem_dim = removal_docs(backend, fwd, units, q_dim, mu)
        ans = {"base": odd[0]["base_answer"].strip(), "donor": odd[0]["donor_answer"].strip()}
        by = {ans[s]: rem[s] for s in ("base", "donor")}
        by_dim = {ans[s]: rem_dim[s] for s in ("base", "donor")}
        pooled = (rem["base"]["ce"] + rem["donor"]["ce"]) / 2
        ratio = lambda d: max(d[ans["base"]]["ce"], d[ans["donor"]]["ce"]) / max(min(d[ans["base"]]["ce"], d[ans["donor"]]["ce"]), 1e-6)
        R[name] = {"units": units, "answers": ans, "marked": MARKED[name], "extraction": ext, "removal_by_answer": by, "removal_dim_by_answer": by_dim,
                   "pooled": pooled, "v95_pooled": v95[name]["pooled"], "ratio": ratio(by), "ratio_dim": ratio(by_dim)}
        print(name, "ext", {k: round(v["fraction"], 3) for k, v in ext.items()}, "ce", {k: round(v["ce"], 3) for k, v in by.items()},
              "flip", {k: round(v["flip"], 2) for k, v in by.items()}, "dim ce", {k: round(v["ce"], 3) for k, v in by_dim.items()}, round(time.perf_counter() - t0), "s", flush=True)

    def fr(n):
        return R[n]["extraction"]["forward"]["fraction"], R[n]["extraction"]["reverse"]["fraction"]

    def flips(n):
        mk = MARKED[n]; um = [a for a in R[n]["answers"].values() if a != mk][0]
        return R[n]["removal_by_answer"][mk]["flip"], R[n]["removal_by_answer"][um]["flip"]

    one_sided = ("quantifier_number", "verb_complementizer")
    pf = [R["verb_preposition"]["removal_by_answer"][a]["flip"] for a in R["verb_preposition"]["answers"].values()]
    predictions = {
        'pred_a_reproduce': all(abs(R[n]["pooled"] - R[n]["v95_pooled"]) <= REPRO for n in NAMES),
        'pred_b_symmetric_extract': all(f is not None and r is not None and min(f, r) >= EXT_MIN and abs(f - r) <= EXT_GAP for f, r in map(fr, one_sided)),
        'pred_c_default_flip': all(mk >= FLIP_HI and um <= FLIP_LO for mk, um in map(flips, one_sided)),
        'pred_d_two_sided_control': abs(pf[0] - pf[1]) <= CTRL_GAP,
        'pred_e_dim_same_shape': all(R[n]["ratio_dim"] >= DIM_RATIO[n] for n in one_sided),
    }
    summary = {n: {"extraction_fraction": {k: round(v["fraction"], 3) if v["fraction"] is not None else None for k, v in R[n]["extraction"].items()},
                   "exact": {k: round(v["exact"], 3) for k, v in R[n]["extraction"].items()},
                   "ce": {k: round(v["ce"], 3) for k, v in R[n]["removal_by_answer"].items()}, "flip": {k: round(v["flip"], 3) for k, v in R[n]["removal_by_answer"].items()},
                   "natural_pref": {k: round(v["natural_pref"], 3) for k, v in R[n]["removal_by_answer"].items()},
                   "dim_ce": {k: round(v["ce"], 3) for k, v in R[n]["removal_dim_by_answer"].items()}, "ratio": round(R[n]["ratio"], 2), "ratio_dim": round(R[n]["ratio_dim"], 2)} for n in NAMES}
    result = {"predictions": predictions, "schema": "circuit_unit_default_side_result_v1", "candidate_id": "corpus.unit_default_side_v96", "summary": summary, "sets": R,
              "bars": {"repro": REPRO, "ext_min": EXT_MIN, "ext_gap": EXT_GAP, "flip_hi": FLIP_HI, "flip_lo": FLIP_LO, "ctrl_gap": CTRL_GAP, "dim_ratio": DIM_RATIO},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
