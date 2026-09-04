#!/usr/bin/env python
"""circuit_battery_successor_full_sweep -- the localisation the successor lineage never ran: ALL 36 components, with controls.

SS2848 found that mlp1's removal costs the numbered-list successor 3.504 margin units -- seven times mlp8's .503 and MORE than
removing attention 8's write entirely (2.646). mlp1 is UPSTREAM of layer 8 and cannot read attention 8's write, so SS2818, SS2819
and SS2821 never measured it: those sections swept only READERS of that write. The successor circuit as described is therefore
missing a term larger than the writer it was built around.

But an early MLP's whole-component ablation damages everything, so raw damage cannot distinguish "part of this circuit" from
"the model needs it to work at all". This rung sweeps all 36 components on the bank's frozen splits and scores each one the way
the battery scores a writer: A1 damage together with the answer-preserving family P and the copy control C, with the
admissibility gate SS2821 forced after SS2820 crowned an inert head.

# BQGATE: EXPERIMENT  pred_a_mlp1_damage_replicates pred_b_mlp1_is_not_selective
#                     pred_c_some_upstream_component_is_selective pred_d_the_ranking_is_task_stable
#                     pred_e_attn8_reproduces_the_battery

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm in margin units, POSITIVE = the arm HURTS that family's own answer; selectivity
ratio = max(|d_P|, |d_C|) / max(d_A1, .5), LOWER IS MORE SELECTIVE. No CE and no SS312 L2; nothing installs.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_SUCCESSOR_FULL_SWEEP_PREREGISTRATION.md
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_successor_full_sweep.py")
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
PREREG = R.POLY / "CIRCUIT_BATTERY_SUCCESSOR_FULL_SWEEP_PREREGISTRATION.md"
UNI = ROOT / "circuit_battery_lineage_unification_results.json"
BANK21 = ROOT / "circuit_battery_v2_bank21_results.json"
RUNG = "circuit_battery_successor_full_sweep"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "fbc82058916b81a353de0bc58c6d7015ea77ea3322705823d8eb8c01bb7876d8",
          UNI: "f9a467b6ea48fc6204c29c36eb712aad3ec98e4724b1b87ad6e1d300fef6e669",
          BANK21: "7c1db82061bb6cac010fa7d2141cfc8e2faa4330759bdfe5aed299bab9a94d50",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NLAY = R.D, R.NL
COMPONENTS = [(kd, l) for l in range(NLAY) for kd in ("attn", "mlp")]
TASKS = ("numbered_list.index_successor", "paren_list.index_successor")
SPLIT = "OOD"
PER_CELL = 4 if SMOKE else 24
MLP1_REF = 3.503512382507324          # SS2848's value on the same rows
BARS = {"mlp1_tol": 0.30, "not_selective": 0.75, "selective": 0.25, "admit": 0.25,
        "rho": 0.50, "repro": 0.30, "floor": 0.5}
NULLS = {"mlp1_selective_le": 0.25, "upstream_selective_none": 0, "rho_le": 0.0}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def run(m, tokens, finals, remove=()):
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        if ("attn", site) in remove:
            write = torch.zeros_like(write)
        x = x + write
        out = blk.mlp(F.rms_norm(x, (D,)))
        if ("mlp", site) in remove:
            out = torch.zeros_like(out)
        x = x + out
    return (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float()


def spearman(a, b):
    rk = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    ra, rb = rk(a), rk(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) and np.std(rb) else float("nan")


def main():
    t0 = time.time()
    check_hashes()
    m = fastload.load_model_fast().to(DEV).eval()
    fwd = [0]
    results = {}
    for tid in TASKS:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        fams = set(BANK.TASKS[tid].families)
        cells = {f: [r for r in rows if r["family"] == f and r["split"] == SPLIT]
                 for f in ("A1", "P", "C") if f in fams}
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        acc = {f: defaultdict(list) for f in cells}
        for fam, rws in cells.items():
            for b in CB.batches(rws):
                ids, fin, ans = CB.pack(b, "base")
                lg = run(m, ids, fin); fwd[0] += 1
                mn = CB.margins(lg, ans, cand)
                for comp in COMPONENTS:
                    lg2 = run(m, ids, fin, remove=(comp,)); fwd[0] += 1
                    acc[fam][f"{comp[0]}{comp[1]}"].append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
        dmg = {f: {k: float(np.concatenate(v).mean()) for k, v in d.items()} for f, d in acc.items()}
        names = sorted(dmg["A1"])
        rows_out = {}
        for n in names:
            a1 = dmg["A1"][n]
            ctrl = max(abs(dmg.get("P", {}).get(n, 0.0)), abs(dmg.get("C", {}).get(n, 0.0)))
            layer = int("".join(ch for ch in n if ch.isdigit()))
            rows_out[n] = {"a1": a1, "ratio": ctrl / max(a1, BARS["floor"]), "layer": layer,
                           "admissible": bool(a1 >= BARS["admit"] * max(dmg["A1"]["attn8"], BARS["floor"]))}
        results[tid] = {"damage": dmg, "components": rows_out,
                        "top8_by_damage": sorted(names, key=lambda n: -rows_out[n]["a1"])[:8]}
        p = results[tid]
        print(f"[sweep] {tid:32s} top: " +
              ", ".join(f"{n}={rows_out[n]['a1']:.2f}/r{rows_out[n]['ratio']:.2f}" for n in p["top8_by_damage"][:5]),
              flush=True)

    t0n = TASKS[0]
    c0 = results[t0n]["components"]
    mlp1_dmg = c0["mlp1"]["a1"]
    mlp1_ratio = c0["mlp1"]["ratio"]
    attn8_dmg = c0["attn8"]["a1"]
    upstream_selective = [n for n, v in c0.items()
                          if v["layer"] < 8 and v["admissible"] and v["ratio"] <= BARS["selective"]]
    names = sorted(c0)
    rho = spearman([c0[n]["a1"] for n in names],
                   [results[TASKS[1]]["components"][n]["a1"] for n in names])
    b21 = json.load(open(BANK21))["tasks"][t0n]["split_d_m"]["FULL"]
    repro = abs(attn8_dmg - b21) / max(b21, BARS["floor"])
    preds = {
        'pred_a_mlp1_damage_replicates': bool(abs(mlp1_dmg - MLP1_REF) / max(MLP1_REF, BARS["floor"]) <= BARS["mlp1_tol"]),
        'pred_b_mlp1_is_not_selective': bool(mlp1_ratio >= BARS["not_selective"]),
        'pred_c_some_upstream_component_is_selective': bool(len(upstream_selective) >= 1),
        'pred_d_the_ranking_is_task_stable': bool(rho >= BARS["rho"]),
        'pred_e_attn8_reproduces_the_battery': bool(repro <= BARS["repro"]),
    }
    nulls = {
        "b_null_mlp1_is_selective": bool(mlp1_ratio <= NULLS["mlp1_selective_le"]),
        "c_null_no_upstream_selective": bool(len(upstream_selective) == NULLS["upstream_selective_none"]),
        "d_null_no_stability": bool(rho <= NULLS["rho_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "tasks": list(TASKS), "split": SPLIT, "mlp1_reference_2848": MLP1_REF,
              "summary": {"mlp1_damage": mlp1_dmg, "mlp1_ratio": mlp1_ratio, "attn8_damage": attn8_dmg,
                          "upstream_selective": upstream_selective, "rho_task_stability": rho,
                          "battery_reference_full": b21, "repro_gap_fraction": repro,
                          "top8": {t: results[t]["top8_by_damage"] for t in results}},
              "tasks_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1200])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: {len(COMPONENTS)} whole-component ablations x {len(TASKS)} successor tasks x "
              f"A1/P/C on {SPLIT}; no model loaded")
        sys.exit(0)
    main()
