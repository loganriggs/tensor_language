#!/usr/bin/env python
"""circuit_battery_roundness_capability -- is bilin18's numeric competence a function of ROUNDNESS?

SS2840: I probed "10% 20% 30%" and the model continued it correctly by the STEP; the bank then drew split-disjoint starts, produced
"13% 23% 33%", and capability collapsed from a hand-probe's apparent success to .07. That is one datapoint suggesting the behaviour the
model has is not "continue a percentage run" but "continue a ROUND percentage run". SS2817 separately established that on bare numeric
runs the model answers LAST + 1 rather than LAST + STEP (capability .92 vs .06). If roundness is the hidden variable, both facts are the
same fact, and several earlier capability numbers need reading differently.

This rung measures capability directly as a function of the start value's roundness, in four numeric formats, with the step held fixed
within each comparison. It is a MODEL-PROPERTY measurement, not a circuit claim: it uses its own value sets rather than the bank's
frozen splits, and it makes no localisation or selectivity claim.

# BQGATE: EXPERIMENT  pred_a_percent_runs_need_round_starts pred_b_roundness_is_general
#                     pred_c_successor_is_roundness_robust pred_d_tens_beat_fives
#                     pred_e_instrument_reproduces_argmax

SIGN CONVENTION: capability is argmax accuracy over the format's own candidate vocabulary, HIGHER IS BETTER. No CE and no SS312 L2 here;
nothing installs.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_ROUNDNESS_CAPABILITY_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_roundness_capability.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_ROUNDNESS_CAPABILITY_PREREGISTRATION.md"
BANK21 = ROOT / "circuit_battery_v2_bank21_results.json"
RUNG = "circuit_battery_roundness_capability"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "c489588643d2b28f95cbf2abf8b7103d05c637f9ece92d9ca5d4a3900697605e",
          BANK21: "7c1db82061bb6cac010fa7d2141cfc8e2faa4330759bdfe5aed299bab9a94d50",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
ENC = BANK.ENC
STEP = 10
BARS = {"percent_gap": 0.30, "general_formats": 3, "format_gap": 0.15, "successor_gap": 0.15,
        "tens_over_fives": 0.10}
NULLS = {"percent_gap_le": 0.05, "general_formats_le": 1, "successor_gap_ge": 0.40,
         "tens_over_fives_le": -0.10}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


# --- value classes, fixed before the run ------------------------------------------------- #
TENS = [n for n in range(10, 100) if n % 10 == 0]                 # 10, 20, ... 90
FIVES = [n for n in range(10, 100) if n % 10 == 5]                # 15, 25, ... 95
OTHER = [n for n in range(11, 100) if n % 5 != 0]                 # everything else

FORMATS = {
    "percent_run":  (lambda s, k: f"{s}% {s + k}% {s + 2 * k}%", lambda s, k: f" {s + 3 * k}"),
    "bare_run":     (lambda s, k: f"{s} {s + k} {s + 2 * k}",     lambda s, k: f" {s + 3 * k}"),
    "numbered_list":(lambda s, k: f"{s}. alpha\n{s + k}. beta\n", lambda s, k: f"{s + 2 * k}"),
    "keyed_line":   (lambda s, k: f"Step {s}\nStep {s + k}\nStep", lambda s, k: f" {s + 2 * k}"),
}
SUCC_FORMATS = {                                                  # the LAST + 1 behaviour of SS2817
    "bare_run_succ": (lambda s, k: f"{s} {s + k} {s + 2 * k}", lambda s, k: f" {s + 2 * k + 1}"),
}


@torch.no_grad()
def accuracy(m, prompts, answers, cand):
    ok, n = 0, 0
    rows = [(p, a) for p, a in zip(prompts, answers)
            if len(ENC.encode(a)) == 1 and ENC.encode(p + a) == ENC.encode(p) + ENC.encode(a)]
    by_len = {}
    for p, a in rows:
        by_len.setdefault(len(ENC.encode(p)), []).append((p, a))
    fwd = 0
    for group in by_len.values():
        for i in range(0, len(group), 32):
            chunk = group[i:i + 32]
            ids = torch.tensor([ENC.encode(p) for p, _ in chunk], device=DEV)
            fin = torch.full((len(chunk),), ids.size(1) - 1, device=DEV, dtype=torch.long)
            ans = torch.tensor([ENC.encode(a)[0] for _, a in chunk], device=DEV)
            lg = CB.run(m, ids, fin); fwd += 1
            pick = cand[lg[:, cand].argmax(1)]
            ok += int((pick == ans).sum()); n += len(chunk)
    return (ok / n if n else float("nan")), n, fwd


def main():
    t0 = time.time()
    check_hashes()
    m = R.load_model().to(DEV).eval()
    cand = torch.tensor(sorted({ENC.encode(s)[0] for s in
                                [str(i) for i in range(0, 200)] + [f" {i}" for i in range(0, 200)]
                                if len(ENC.encode(s)) == 1}), device=DEV)
    fwd = 0
    classes = {"tens": TENS, "fives": FIVES, "other": OTHER}
    if SMOKE:
        classes = {k: v[:3] for k, v in classes.items()}
    res = {}
    for fname, (mk, ans) in {**FORMATS, **SUCC_FORMATS}.items():
        res[fname] = {}
        for cname, vals in classes.items():
            vals = [s for s in vals if s + 3 * STEP < 100]
            prompts = [mk(s, STEP) for s in vals]
            answers = [ans(s, STEP) for s in vals]
            acc, n, f = accuracy(m, prompts, answers, cand); fwd += f
            res[fname][cname] = {"accuracy": acc, "n": n}
            print(f"[round] {fname:16s} {cname:6s} acc={acc:.3f} (n={n})", flush=True)

    gap = lambda f: res[f]["tens"]["accuracy"] - res[f]["other"]["accuracy"]
    percent_gap = gap("percent_run")
    general = [f for f in FORMATS if gap(f) >= BARS["format_gap"]]
    succ_gap = gap("bare_run_succ")
    tens_fives = res["percent_run"]["tens"]["accuracy"] - res["percent_run"]["fives"]["accuracy"]
    preds = {
        'pred_a_percent_runs_need_round_starts': bool(percent_gap >= BARS["percent_gap"]),
        'pred_b_roundness_is_general': bool(len(general) >= BARS["general_formats"]),
        'pred_c_successor_is_roundness_robust': bool(abs(succ_gap) <= BARS["successor_gap"]),
        'pred_d_tens_beat_fives': bool(tens_fives >= BARS["tens_over_fives"]),
        'pred_e_instrument_reproduces_argmax': bool(all(
            res[f][c]["n"] > 0 for f in res for c in res[f])),
    }
    nulls = {
        "a_null_no_percent_gap": bool(percent_gap <= NULLS["percent_gap_le"]),
        "b_null_not_general": bool(len(general) <= NULLS["general_formats_le"]),
        "c_null_successor_also_round": bool(abs(succ_gap) >= NULLS["successor_gap_ge"]),
        "d_null_fives_beat_tens": bool(tens_fives <= NULLS["tens_over_fives_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "step": STEP, "classes": {k: len(v) for k, v in classes.items()},
              "summary": {"percent_gap_tens_minus_other": percent_gap,
                          "formats_with_roundness_gap": general,
                          "successor_gap": succ_gap,
                          "percent_tens_minus_fives": tens_fives,
                          "gaps_by_format": {f: gap(f) for f in FORMATS}},
              "accuracy": res, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd, "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1200])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: {len(FORMATS) + len(SUCC_FORMATS)} numeric formats x 3 roundness classes "
              f"(tens, fives, other) at step {STEP}; capability only; no model loaded")
        sys.exit(0)
    main()
