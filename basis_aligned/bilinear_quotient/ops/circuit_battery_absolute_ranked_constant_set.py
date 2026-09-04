#!/usr/bin/env python
"""circuit_battery_absolute_ranked_constant_set -- the correction SS2837 forces: rank by ABSOLUTE cost, not by a recovery ratio.

SS2836 measured, one component at a time, which writes a single fixed vector can replace: attn5 .942, attn1 .915, mlp16 .811, mlp0 .735
recovered, against a median of -.008 over the 27 that passed a geometric screen -- and it explicitly recorded that it "says nothing
about replacing several at once, which is where interactions (and SS2818's super-additivity) would appear". This rung does that: it
replaces the k best individually-replaceable writes with their own mean vectors SIMULTANEOUSLY, for k = 1, 2, 3, 4, 6, 8, against a
size-matched random control. A compiled tensor program needs the joint number, not the marginal one.

# BQGATE: EXPERIMENT  pred_a_absolute_beats_ratio_ranking pred_b_absolute_beats_random
#                     pred_c_more_writes_can_be_constants pred_d_composition_stays_super_additive
#                     pred_e_instrument_reproduces_native_ce_matched

SIGN CONVENTION: d_ce = CE_arm - CE_NATIVE in nats, POSITIVE = the arm HURTS. NOT the SS312 frontier's L2 (CE added above the real model
by an installed approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735, SS2135); nothing installs; DIAGNOSTICS only;
metric-constructed bases/spans remain CLOSED (SS2118 lineage).
Preregistration: polynomial_causal/CIRCUIT_BATTERY_ABSOLUTE_RANKED_CONSTANT_SET_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_absolute_ranked_constant_set.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_ABSOLUTE_RANKED_CONSTANT_SET_PREREGISTRATION.md"
CENSUS = ROOT / "circuit_battery_constant_write_census_results.json"
RUNG = "circuit_battery_absolute_ranked_constant_set"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "b4343a88d3e98745ca150d9f4ea6dfb09232f69fe55c5cbf7b6e6886f0028160",
          CENSUS: "ab7cc4a16d879c981772fe4e65aab3c28b733000436fdbd49600dc21943e78e0",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NL, V, T = R.D, R.NL, R.V, R.T
NAT = ROOT / ".rowcache_terminal_copy_induction_v2/fit_natural.pt"
KS = (1, 2, 3, 4, 6, 8)
SEED = 2837
NFIT = 8 if SMOKE else 24
NEVAL = 8 if SMOKE else 24
CHUNK = 8
RATIO_K4 = 1.386037190755208      # SS2837's ratio-ranked joint cost at k=4
RANDOM_K4 = 0.6812637647          # SS2837's size-matched random control at k=4
BARS = {"beat_ratio": 0.30, "beat_random": 0.0, "n_cheap": 6, "cheap_nats": 1.50,
        "superadd": 0.0, "ce_tol": 0.01}
NULLS = {"beat_ratio_le": 0.0, "beat_random_ge": 0.30, "n_cheap_le": 3, "superadd_le": -0.20}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def parse(name):
    kind = "attn" if name.startswith("attn") else "mlp"
    return (kind, int(name[len(kind):]))


@torch.no_grad()
def doc_forward(m, idx, targets=(), mode=None, consts=None, collect=None):
    """targets: set of (kind, layer).  mode: None | 'zero' | 'const'."""
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0 = x; v1 = None
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        key = ("attn", site)
        if key in targets:
            if collect is not None:
                collect.setdefault(key, []).append(write.reshape(-1, D).float())
            if mode == "zero":
                write = torch.zeros_like(write)
            elif mode == "const":
                write = consts[key].to(write.dtype).expand_as(write)
        x = x + write
        out = blk.mlp(F.rms_norm(x, (D,)))
        key = ("mlp", site)
        if key in targets:
            if collect is not None:
                collect.setdefault(key, []).append(out.reshape(-1, D).float())
            if mode == "zero":
                out = torch.zeros_like(out)
            elif mode == "const":
                out = consts[key].to(out.dtype).expand_as(out)
        x = x + out
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def main():
    t0 = time.time()
    check_hashes()
    census = json.load(open(CENSUS))
    comps = census["components"]
    # THE CORRECTION: rank by the ABSOLUTE nats a constant costs, not by 1 - const/zero.
    order_abs = sorted(comps, key=lambda n: comps[n]["const_damage"])
    order_ratio = sorted((n for n in comps if not np.isnan(comps[n]["recovered"])),
                         key=lambda n: -comps[n]["recovered"])
    m = R.load_model().to(DEV).eval()
    t = torch.load(NAT, map_location="cpu")
    t = (t["rows"] if isinstance(t, dict) else t).long()
    fit_docs, eval_docs = t[:NFIT], t[NFIT:NFIT + NEVAL]
    fwd = [0]

    def ce_of(docs, **kw):
        s_, n_ = 0.0, 0
        for i in range(0, docs.shape[0], CHUNK):
            idx = docs[i:i + CHUNK, :T - 1].to(DEV)
            tgt = docs[i:i + CHUNK, 1:T].to(DEV)
            lg = doc_forward(m, idx, **kw); fwd[0] += 1
            s_ += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum"))
            n_ += tgt.numel()
        return s_ / n_

    ce_eval = ce_of(eval_docs)
    idx0 = eval_docs[:CHUNK, :T - 1].to(DEV); tgt0 = eval_docs[:CHUNK, 1:T].to(DEV)
    lg0 = doc_forward(m, idx0); fwd[0] += 1
    ce_manual = float(F.cross_entropy(lg0.reshape(-1, V), tgt0.reshape(-1)))
    ce_module = float(m(idx0.contiguous(), tgt0.contiguous()))

    g = np.random.default_rng(SEED)
    all_names = sorted(comps)
    rows = {}
    for k in KS:
        sets = {"absolute": [parse(n) for n in order_abs[:k]],
                "ratio": [parse(n) for n in order_ratio[:k]],
                "random": [parse(n) for n in g.choice(all_names, size=k, replace=False)]}
        out = {"names_absolute": order_abs[:k], "names_ratio": order_ratio[:k],
               "sum_of_individual_const_absolute": float(sum(comps[n]["const_damage"] for n in order_abs[:k])),
               "sum_of_individual_zero_absolute": float(sum(comps[n]["zero_damage"] for n in order_abs[:k]))}
        for label, targets in sets.items():
            coll = {}
            for i in range(0, fit_docs.shape[0], CHUNK):
                doc_forward(m, fit_docs[i:i + CHUNK, :T - 1].to(DEV), targets=set(targets), collect=coll); fwd[0] += 1
            consts = {kk: torch.cat(vv, 0).mean(0) for kk, vv in coll.items()}
            del coll
            out[f"{label}_const"] = ce_of(eval_docs, targets=set(targets), mode="const", consts=consts) - ce_eval
        out["absolute_zero"] = ce_of(eval_docs, targets=set(sets["absolute"]), mode="zero") - ce_eval
        rows[f"k{k}"] = out
        print(f"[absrank] k={k} abs={out['absolute_const']:+.4f} ratio={out['ratio_const']:+.4f} "
              f"rnd={out['random_const']:+.4f} abs_zero={out['absolute_zero']:+.4f} "
              f"sum_ind={out['sum_of_individual_const_absolute']:+.4f}", flush=True)

    k4 = rows["k4"]
    beat_ratio = RATIO_K4 - k4["absolute_const"]
    beat_random = RANDOM_K4 - k4["absolute_const"]
    superadd = k4["absolute_const"] - k4["sum_of_individual_const_absolute"]
    n_cheap = max([k for k in KS if rows[f"k{k}"]["absolute_const"] <= BARS["cheap_nats"]], default=0)
    preds = {
        'pred_a_absolute_beats_ratio_ranking': bool(beat_ratio >= BARS["beat_ratio"]),
        'pred_b_absolute_beats_random': bool(beat_random >= BARS["beat_random"]),
        'pred_c_more_writes_can_be_constants': bool(n_cheap >= BARS["n_cheap"]),
        'pred_d_composition_stays_super_additive': bool(superadd >= BARS["superadd"]),
        'pred_e_instrument_reproduces_native_ce_matched': bool(abs(ce_manual - ce_module) <= BARS["ce_tol"]),
    }
    nulls = {
        "a_null_ratio_as_good": bool(beat_ratio <= NULLS["beat_ratio_le"]),
        "b_null_random_still_better": bool(beat_random <= -NULLS["beat_random_ge"]),
        "c_null_few_constants": bool(n_cheap <= NULLS["n_cheap_le"]),
        "d_null_subadditive": bool(superadd <= NULLS["superadd_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "ks": list(KS), "seed": SEED,
              "order_by_absolute_const_cost": order_abs[:10],
              "order_by_recovery_ratio": order_ratio[:10],
              "reference_from_2837": {"ratio_k4": RATIO_K4, "random_k4": RANDOM_K4},
              "n_fit_docs": int(fit_docs.shape[0]), "n_eval_docs": int(eval_docs.shape[0]),
              "ce_eval_native": ce_eval, "ce_manual_chunk": ce_manual, "ce_module_chunk": ce_module,
              "summary": {"beat_ratio_nats": beat_ratio, "beat_random_nats": beat_random,
                          "superadditivity_nats": superadd, "largest_cheap_k": n_cheap,
                          "curve_absolute": {f"k{k}": rows[f"k{k}"]["absolute_const"] for k in KS}},
              "arms": rows, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": sum(KS) * 3 * D,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1200])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: joint constants for sets ranked by ABSOLUTE cost vs SS2837's ratio ranking vs random, "
              f"k={KS}; no model loaded")
        sys.exit(0)
    main()
