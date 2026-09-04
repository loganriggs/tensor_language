#!/usr/bin/env python
"""circuit_battery_joint_constant_replacement -- can SEVERAL writes be constants at once?

SS2836 measured, one component at a time, which writes a single fixed vector can replace: attn5 .942, attn1 .915, mlp16 .811, mlp0 .735
recovered, against a median of -.008 over the 27 that passed a geometric screen -- and it explicitly recorded that it "says nothing
about replacing several at once, which is where interactions (and SS2818's super-additivity) would appear". This rung does that: it
replaces the k best individually-replaceable writes with their own mean vectors SIMULTANEOUSLY, for k = 1, 2, 3, 4, 6, 8, against a
size-matched random control. A compiled tensor program needs the joint number, not the marginal one.

# BQGATE: EXPERIMENT  pred_a_joint_beats_deleting_them pred_b_joint_is_worse_than_the_sum_of_parts
#                     pred_c_random_sets_are_worse pred_d_the_curve_degrades_gracefully
#                     pred_e_instrument_reproduces_native_ce_matched

SIGN CONVENTION: d_ce = CE_arm - CE_NATIVE in nats, POSITIVE = the arm HURTS. NOT the SS312 frontier's L2 (CE added above the real model
by an installed approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735, SS2135); nothing installs; DIAGNOSTICS only;
metric-constructed bases/spans remain CLOSED (SS2118 lineage).
Preregistration: polynomial_causal/CIRCUIT_BATTERY_JOINT_CONSTANT_REPLACEMENT_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_joint_constant_replacement.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_JOINT_CONSTANT_REPLACEMENT_PREREGISTRATION.md"
CENSUS = ROOT / "circuit_battery_constant_write_census_results.json"
RUNG = "circuit_battery_joint_constant_replacement"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "eeb413a0e9278eec307ff92c5da8a0274ef1d27583c125092b294dfe92eecbb7",
          CENSUS: "ab7cc4a16d879c981772fe4e65aab3c28b733000436fdbd49600dc21943e78e0",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NL, V, T = R.D, R.NL, R.V, R.T
NAT = ROOT / ".rowcache_terminal_copy_induction_v2/fit_natural.pt"
KS = (1, 2, 3, 4, 6, 8)
SEED = 2836
NFIT = 8 if SMOKE else 24
NEVAL = 8 if SMOKE else 24
CHUNK = 8
BARS = {"beat_delete": 0.50, "superadd": 0.0, "random_gap": 0.30, "graceful": 2.0, "ce_tol": 0.01}
NULLS = {"beat_delete_le": 0.0, "superadd_le": -0.20, "random_gap_le": 0.0, "graceful_ge": 5.0}


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
    order = sorted((n for n in comps if not np.isnan(comps[n]["recovered"])),
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
        chosen = [parse(n) for n in order[:k]]
        rnd = [parse(n) for n in g.choice(all_names, size=k, replace=False)]
        out = {}
        for label, targets in (("best", chosen), ("random", rnd)):
            coll = {}
            for i in range(0, fit_docs.shape[0], CHUNK):
                doc_forward(m, fit_docs[i:i + CHUNK, :T - 1].to(DEV), targets=set(targets), collect=coll); fwd[0] += 1
            consts = {kk: torch.cat(vv, 0).mean(0) for kk, vv in coll.items()}
            del coll
            out[f"{label}_const"] = ce_of(eval_docs, targets=set(targets), mode="const", consts=consts) - ce_eval
            out[f"{label}_zero"] = ce_of(eval_docs, targets=set(targets), mode="zero") - ce_eval
        out["names_best"] = order[:k]
        out["sum_of_individual_const"] = float(sum(comps[n]["const_damage"] for n in order[:k]))
        out["sum_of_individual_zero"] = float(sum(comps[n]["zero_damage"] for n in order[:k]))
        rows[f"k{k}"] = out
        print(f"[joint] k={k} best_const={out['best_const']:+.4f} best_zero={out['best_zero']:+.4f} "
              f"sum_ind_const={out['sum_of_individual_const']:+.4f} rnd_const={out['random_const']:+.4f}", flush=True)

    k4 = rows["k4"]
    beat_delete = k4["best_zero"] - k4["best_const"]
    superadd = k4["best_const"] - k4["sum_of_individual_const"]
    random_gap = k4["random_const"] - k4["best_const"]
    graceful = rows["k8"]["best_const"] / max(k4["best_const"], 1e-9)
    preds = {
        'pred_a_joint_beats_deleting_them': bool(beat_delete >= BARS["beat_delete"]),
        'pred_b_joint_is_worse_than_the_sum_of_parts': bool(superadd >= BARS["superadd"]),
        'pred_c_random_sets_are_worse': bool(random_gap >= BARS["random_gap"]),
        'pred_d_the_curve_degrades_gracefully': bool(graceful <= BARS["graceful"]),
        'pred_e_instrument_reproduces_native_ce_matched': bool(abs(ce_manual - ce_module) <= BARS["ce_tol"]),
    }
    nulls = {
        "a_null_no_gain_over_deleting": bool(beat_delete <= NULLS["beat_delete_le"]),
        "b_null_subadditive": bool(superadd <= NULLS["superadd_le"]),
        "c_null_random_as_good": bool(random_gap <= NULLS["random_gap_le"]),
        "d_null_blows_up": bool(graceful >= NULLS["graceful_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "ks": list(KS), "seed": SEED, "order_by_individual_recovery": order[:10],
              "n_fit_docs": int(fit_docs.shape[0]), "n_eval_docs": int(eval_docs.shape[0]),
              "ce_eval_native": ce_eval, "ce_manual_chunk": ce_manual, "ce_module_chunk": ce_module,
              "summary": {"beat_delete_nats": beat_delete, "superadditivity_nats": superadd,
                          "random_gap_nats": random_gap, "k8_over_k4": graceful,
                          "curve": {f"k{k}": rows[f"k{k}"]["best_const"] for k in KS}},
              "arms": rows, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0,
                        "fitted_parameters": sum(KS) * 2 * D,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1200])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: joint constant replacement of the k best individually-replaceable writes, k={KS}, "
              f"vs size-matched random sets; no model loaded")
        sys.exit(0)
    main()
