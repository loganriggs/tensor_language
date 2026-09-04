#!/usr/bin/env python
"""circuit_battery_remainder_class_gate -- what is the GENERIC four-fifths of the read doing?

SS2826/SS2827: the read of attention 8's write splits into a rank-1 unfitted low-energy task-specific component along
u = W_U[answer] - W_U[competitor] (a fifth of the block's margin damage, 2.4x its specificity) and a generic remainder of four fifths
that has resisted units, lens, SVD, in-sample subspaces and further competitor axes. This rung asks what the remainder DOES rather than
where it lives, with a second measurement the campaign has not used: the CANDIDATE-CLASS MASS, log sum_{v in candidates} p(v) -- how
much probability the model puts on the task's answer class at all, independent of which member it picks.

Hypothesis under test: the remainder maintains the TYPE ("a list label goes here") while the rank-1 causal axis selects the MEMBER
("the next one"). If so the remainder's damage is class-mass damage and the causal axis's is not.

# BQGATE: EXPERIMENT  pred_a_remainder_carries_the_class_mass pred_b_causal_axis_spares_the_class
#                     pred_c_causal_axis_is_within_class pred_d_random_direction_does_neither
#                     pred_e_the_two_parts_add_up

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS; class-mass damage d_c = logmass_NATIVE - logmass_arm in nats,
POSITIVE = the arm REMOVES class mass. Nothing installs into the SS312 frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_REMAINDER_CLASS_GATE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_remainder_class_gate.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
import circuit_battery_causal_direction_read as CD
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_REMAINDER_CLASS_GATE_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_remainder_class_gate"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "464c2f93495b5a6a939c5489d790bb3536a2ff6b9e7879c1f945c2422236928b",
          BATTERY: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D = R.D
READERS = (10, 11)
SEED = 2827
PER_CELL = 4 if SMOKE else 16
BARS = {"class_ratio": 3.0, "causal_class_nats": 0.15, "within_gain": 0.10, "random_nats": 0.05,
        "additivity": 0.20, "floor": 0.05}
NULLS = {"class_ratio_le": 1.0, "causal_class_ge": 0.50, "within_gain_le": 0.0, "random_ge": 0.30}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def logmass(logits, cand):
    """log of the total probability on the task's candidate class."""
    lp = torch.log_softmax(logits, dim=-1)
    return torch.logsumexp(lp[:, cand], dim=-1)


def arms(m, rows, cand, layer, g, fwd):
    """Native, ALL, CAUSAL (along u), REMAINDER (orthogonal complement of u), RANDOM: margin and class mass."""
    out = {k: {"dm": [], "dc": []} for k in ("ALL", "CAUSAL", "REMAINDER", "RANDOM")}
    for b in CB.batches(rows):
        ids, fin, ans = CB.pack(b, "base")
        lg = CD.run_dir(m, ids, fin, layer, None); fwd[0] += 1
        mn, cn = CB.margins(lg, ans, cand), logmass(lg, cand)
        U, _c = CD.directions(m, lg, ans, cand)
        v = torch.randn(len(b), D, generator=g).to(DEV)
        Ur = v / v.norm(dim=-1, keepdim=True)
        for name, arg in (("ALL", "all"), ("CAUSAL", U), ("REMAINDER", ("perp", U)), ("RANDOM", Ur)):
            lg2 = run_arm(m, ids, fin, layer, arg); fwd[0] += 1
            out[name]["dm"].append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
            out[name]["dc"].append((cn - logmass(lg2, cand)).cpu().numpy())
    return {k: {kk: float(np.concatenate(vv).mean()) for kk, vv in v.items()} for k, v in out.items()}


@torch.no_grad()
def run_arm(m, tokens, finals, layer, spec):
    """spec: 'all' | U (B,D) remove along U | ('perp', U) remove the complement of U."""
    import torch.nn.functional as F
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None; W = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if W is not None:
            W = blk.lambdas[0] * W
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        if site == 8:
            W = torch.zeros_like(x); W[ar, finals] = write[ar, finals]
        x = x + write
        if site == layer and W is not None:
            nat = blk.mlp(F.rms_norm(x, (D,)))
            rem = blk.mlp(F.rms_norm(x - W, (D,)))
            d = (nat - rem)[ar, finals].float()
            if isinstance(spec, str):
                proj = d
            elif isinstance(spec, tuple):
                U = spec[1]
                proj = d - (d * U).sum(-1, keepdim=True) * U
            else:
                proj = (d * spec).sum(-1, keepdim=True) * spec
            out = nat.clone()
            out[ar, finals] = nat[ar, finals] - proj.to(nat.dtype)
        else:
            out = blk.mlp(F.rms_norm(x, (D,)))
        x = x + out
    return (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float()


def main():
    t0 = time.time()
    check_hashes()
    b2 = json.load(open(BATTERY))
    tasks = [t for t in b2["summary"]["capable"] if b2["tasks"][t]["writer"] == "attn8"]
    m = R.load_model().to(DEV).eval()
    g = torch.Generator(device="cpu").manual_seed(SEED)
    fwd = [0]
    results = {}
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        per_layer = {}
        for layer in READERS:
            ood = [r for r in rows if r["family"] == "A1" and r["split"] == "OOD"]
            a = arms(m, ood, cand, layer, g, fwd)
            cls = lambda k: a[k]["dc"]
            mar = lambda k: a[k]["dm"]
            per_layer[f"mlp{layer}"] = {
                "arms": a,
                "class_ratio": abs(cls("REMAINDER")) / max(abs(cls("CAUSAL")), BARS["floor"]),
                "causal_class_nats": abs(cls("CAUSAL")),
                "remainder_class_nats": abs(cls("REMAINDER")),
                "all_class_nats": abs(cls("ALL")),
                "margin_per_nat_causal": mar("CAUSAL") / max(abs(cls("CAUSAL")), BARS["floor"]),
                "margin_per_nat_remainder": mar("REMAINDER") / max(abs(cls("REMAINDER")), BARS["floor"]),
                "within_gain": (mar("CAUSAL") / max(abs(cls("CAUSAL")), BARS["floor"]))
                               - (mar("REMAINDER") / max(abs(cls("REMAINDER")), BARS["floor"])),
                "random_class_nats": abs(cls("RANDOM")),
                "additivity_gap": abs(cls("CAUSAL") + cls("REMAINDER") - cls("ALL")),
            }
            p = per_layer[f"mlp{layer}"]
            print(f"[class] {tid:28s} mlp{layer} rem_nats={p['remainder_class_nats']:.3f} "
                  f"caus_nats={p['causal_class_nats']:.3f} ratio={p['class_ratio']:.1f} "
                  f"within_gain={p['within_gain']:.2f} add_gap={p['additivity_gap']:.3f}", flush=True)
        results[tid] = per_layer

    flat = [results[t][l] for t in results for l in results[t]]
    med = lambda k: float(np.median([f[k] for f in flat])) if flat else float("nan")
    preds = {
        'pred_a_remainder_carries_the_class_mass': bool(med("class_ratio") >= BARS["class_ratio"]),
        'pred_b_causal_axis_spares_the_class': bool(med("causal_class_nats") <= BARS["causal_class_nats"]),
        'pred_c_causal_axis_is_within_class': bool(med("within_gain") >= BARS["within_gain"]),
        'pred_d_random_direction_does_neither': bool(med("random_class_nats") <= BARS["random_nats"]),
        'pred_e_the_two_parts_add_up': bool(med("additivity_gap") <= BARS["additivity"]),
    }
    nulls = {
        "a_null_ratio_le_1": bool(med("class_ratio") <= NULLS["class_ratio_le"]),
        "b_null_causal_class_ge_.5": bool(med("causal_class_nats") >= NULLS["causal_class_ge"]),
        "c_null_no_within_gain": bool(med("within_gain") <= NULLS["within_gain_le"]),
        "d_null_random_ge_.3": bool(med("random_class_nats") >= NULLS["random_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "writer": "attn8", "readers": [f"mlp{l}" for l in READERS], "seed": SEED,
              "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "summary": {"tasks": sorted(results), "n_cells": len(flat),
                          "medians": {k: med(k) for k in
                                      ("class_ratio", "causal_class_nats", "remainder_class_nats",
                                       "all_class_nats", "margin_per_nat_causal",
                                       "margin_per_nat_remainder", "within_gain",
                                       "random_class_nats", "additivity_gap")}},
              "tasks": results, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "medians": result["summary"]["medians"]}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2817 capable attn8-writer behaviours x readers {READERS}; "
              f"CAUSAL vs REMAINDER on margin AND candidate-class mass; no model loaded")
        sys.exit(0)
    main()
