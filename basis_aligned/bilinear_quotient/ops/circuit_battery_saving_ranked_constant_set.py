#!/usr/bin/env python
"""circuit_battery_saving_ranked_constant_set -- rank by the DIFFERENCE keeping makes, the rule SS2838 extracted after six failures.

SS2836 measured, one component at a time, which writes a single fixed vector can replace: attn5 .942, attn1 .915, mlp16 .811, mlp0 .735
recovered, against a median of -.008 over the 27 that passed a geometric screen -- and it explicitly recorded that it "says nothing
about replacing several at once, which is where interactions (and SS2818's super-additivity) would appear". This rung does that: it
replaces the k best individually-replaceable writes with their own mean vectors SIMULTANEOUSLY, for k = 1, 2, 3, 4, 6, 8, against a
size-matched random control. A compiled tensor program needs the joint number, not the marginal one.

# BQGATE: EXPERIMENT  pred_a_saving_set_is_worth_keeping pred_b_saving_beats_both_earlier_rankings
#                     pred_c_the_set_is_not_degenerate pred_d_composition_stays_super_additive
#                     pred_e_instrument_reproduces_native_ce_matched

SIGN CONVENTION: d_ce = CE_arm - CE_NATIVE in nats, POSITIVE = the arm HURTS. NOT the SS312 frontier's L2 (CE added above the real model
by an installed approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735, SS2135); nothing installs; DIAGNOSTICS only;
metric-constructed bases/spans remain CLOSED (SS2118 lineage).
Preregistration: polynomial_causal/CIRCUIT_BATTERY_SAVING_RANKED_CONSTANT_SET_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_saving_ranked_constant_set.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_SAVING_RANKED_CONSTANT_SET_PREREGISTRATION.md"
CENSUS = ROOT / "circuit_battery_constant_write_census_results.json"
RUNG = "circuit_battery_saving_ranked_constant_set"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "36606761f0cbcba54a57d10df503963e2d1ab5b1c2ef7ab327c31923a1a52c49",
          CENSUS: "ab7cc4a16d879c981772fe4e65aab3c28b733000436fdbd49600dc21943e78e0",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NL, V, T = R.D, R.NL, R.V, R.T
NAT = ROOT / ".rowcache_terminal_copy_induction_v2/fit_natural.pt"
KS = (1, 2, 3, 4, 6, 8)
SEED = 2838
NFIT = 8 if SMOKE else 24
NEVAL = 8 if SMOKE else 24
CHUNK = 8
RATIO_K4 = 1.386037190755208      # SS2837's ratio-ranked joint cost at k=4
ABS_K4 = 0.06972972551981593      # SS2838's absolute-ranked joint cost at k=4 (degenerate set)
BARS = {"saving_k4": 4.0, "beat_both": 0.0, "delete_cost": 3.0, "superadd": 0.0, "ce_tol": 0.01}
NULLS = {"saving_k4_le": 1.0, "beat_both_le": -0.30, "delete_cost_le": 1.0, "superadd_le": -0.20}


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
    # THE RULE FROM SS2838: rank by the DIFFERENCE keeping makes -- nats saved by a constant over deletion.
    order_abs = sorted(comps, key=lambda n: -(comps[n]["zero_damage"] - comps[n]["const_damage"]))
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
    saving = k4["absolute_zero"] - k4["absolute_const"]          # nats the constants SAVE over deleting the same set
    beat_ratio = RATIO_K4 - k4["absolute_const"]
    superadd = k4["absolute_const"] - k4["sum_of_individual_const_absolute"]
    preds = {
        'pred_a_saving_set_is_worth_keeping': bool(saving >= BARS["saving_k4"]),
        'pred_b_saving_beats_both_earlier_rankings': bool(beat_ratio >= BARS["beat_both"]),
        'pred_c_the_set_is_not_degenerate': bool(k4["absolute_zero"] >= BARS["delete_cost"]),
        'pred_d_composition_stays_super_additive': bool(superadd >= BARS["superadd"]),
        'pred_e_instrument_reproduces_native_ce_matched': bool(abs(ce_manual - ce_module) <= BARS["ce_tol"]),
    }
    nulls = {
        "a_null_no_saving": bool(saving <= NULLS["saving_k4_le"]),
        "b_null_worse_than_ratio": bool(beat_ratio <= NULLS["beat_both_le"]),
        "c_null_degenerate_set": bool(k4["absolute_zero"] <= NULLS["delete_cost_le"]),
        "d_null_subadditive": bool(superadd <= NULLS["superadd_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "ks": list(KS), "seed": SEED,
              "order_by_saving": order_abs[:10],
              "order_by_recovery_ratio": order_ratio[:10],
              "reference_prior_rankings": {"ratio_k4_2837": RATIO_K4, "absolute_k4_2838": ABS_K4},
              "n_fit_docs": int(fit_docs.shape[0]), "n_eval_docs": int(eval_docs.shape[0]),
              "ce_eval_native": ce_eval, "ce_manual_chunk": ce_manual, "ce_module_chunk": ce_module,
              "summary": {"saving_nats_k4": saving, "beat_ratio_nats": beat_ratio,
                          "delete_cost_k4": k4["absolute_zero"], "superadditivity_nats": superadd,
                          "curve_const": {f"k{k}": rows[f"k{k}"]["absolute_const"] for k in KS},
                          "curve_delete": {f"k{k}": rows[f"k{k}"]["absolute_zero"] for k in KS},
                          "curve_saving": {f"k{k}": rows[f"k{k}"]["absolute_zero"] - rows[f"k{k}"]["absolute_const"]
                                           for k in KS}},
              "arms": rows, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": sum(KS) * 3 * D,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1200])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: joint constants for sets ranked by SAVING (zero - const) vs SS2837/SS2838's rankings "
              f"and random, k={KS}; no model loaded")
        sys.exit(0)
    main()
