#!/usr/bin/env python
"""circuit_battery_writer_head_split -- WHICH HEADS of attention 8 write the last salient item, and do they rescue selectivity?

SS2817: attention 8 is the FIT-chosen writer for 7 of 8 capable behaviours and none of them is writer-selective. SS2819: selectivity is
in the read and grows with depth (mlp11 best on 6 of 7). The remaining granularity below the writer is the HEAD. Attention 8's write is
y = c_proj(concat_h o_h) with no bias, so it decomposes EXACTLY into nine additive head writes; each is carried through the same
residual path-patching instrument used since SS2808 and removed from every reader edge plus the direct path. Two questions: is the same
head set the writer across surface forms (component re-use one level finer), and does head granularity make the write selective
(the negative that SS2817 and SS2819 predict)?

# BQGATE: EXPERIMENT  pred_a_head_decomposition_is_exact pred_b_two_heads_carry_the_write
#                     pred_c_the_head_pair_is_shared pred_d_numbered_list_replicates_r576_heads
#                     pred_e_heads_do_not_rescue_selectivity

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's own answer. Nothing installs into the SS312
frontier. Preregistration: polynomial_causal/CIRCUIT_BATTERY_WRITER_HEAD_SPLIT_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_writer_head_split.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_WRITER_HEAD_SPLIT_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_writer_head_split"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "dfb4a264dab13f1ea403062b452d267121dd2edaf41eb0899acd4b0d090c56f3",
          BATTERY: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NH, HD, NL = R.D, R.NH, R.HD, R.NL
LAYER = 8
EVAL_SPLIT = "OOD"
FAMILIES = ("A1", "P", "C")
PER_CELL = 4 if SMOKE else 16
R576_HEADS = (3, 7)                     # Codex's R576 / SS2808 finding for the numbered list
BARS = {"exact_tol": 1e-4, "top2_share": 0.60, "shared_pairs": 4, "rescue_margin": 0.25,
        "floor": 0.5}
NULLS = {"top2_share_le": 0.35, "shared_pairs_le": 1, "selective_heads_ge": 4}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def head_writes(m, tokens, finals):
    """Exact additive decomposition of attention LAYER's write at the final position into heads."""
    blk = m.transformer.h[LAYER]
    cap = {}

    def pre(_mod, args):
        cap["ycat"] = args[0]

    h = blk.attn.c_proj.register_forward_pre_hook(pre)
    try:
        x = F.rms_norm(m.transformer.wte(tokens), (D,))
        x0 = x; v1 = None
        for site, b in enumerate(m.transformer.h):
            x = b.lambdas[0] * x + b.lambdas[1] * x0
            write, v1 = b.attn(F.rms_norm(x, (D,)), v1)
            if site == LAYER:
                break
            x = x + write
            x = x + b.mlp(F.rms_norm(x, (D,)))
    finally:
        h.remove()
    ar = torch.arange(tokens.size(0), device=tokens.device)
    ycat = cap["ycat"][ar, finals]                      # (B, D) concatenated head outputs
    per = []
    for hd in range(NH):
        mask = torch.zeros_like(ycat)
        mask[:, hd * HD:(hd + 1) * HD] = ycat[:, hd * HD:(hd + 1) * HD]
        per.append(blk.attn.c_proj(mask))
    return torch.stack(per, 1), blk.attn.c_proj(ycat)   # (B, NH, D), (B, D)


@torch.no_grad()
def run_with_W(m, tokens, finals, W0, remove):
    """The SS2808 instrument with an EXTERNALLY supplied write W0 at the final position."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None; Wj = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if Wj is not None:
            Wj = blk.lambdas[0] * Wj
        xr = x - Wj if (Wj is not None and remove and ("attn", site) in remove) else x
        write, v1 = blk.attn(F.rms_norm(xr, (D,)), v1)
        if site == LAYER and W0 is not None:
            Wj = torch.zeros_like(x); Wj[ar, finals] = W0.to(x.dtype)
        x = x + write
        xm = x - Wj if (Wj is not None and remove and ("mlp", site) in remove) else x
        x = x + blk.mlp(F.rms_norm(xm, (D,)))
    xf = x - Wj if (Wj is not None and remove and "direct" in remove) else x
    return (30.0 * torch.tanh(m.lm_head(F.rms_norm(xf, (D,))) / 30.0))[ar, finals].float()


def main():
    t0 = time.time()
    check_hashes()
    b2 = json.load(open(BATTERY))
    tasks = [t for t in b2["summary"]["capable"] if b2["tasks"][t]["writer"] == "attn8"]
    m = R.load_model().to(DEV).eval()
    readers = [("mlp", LAYER)] + [(kd, l) for l in range(LAYER + 1, NL) for kd in ("attn", "mlp")]
    FULL = tuple(readers) + ("direct",)
    fwd = 0
    results = {}
    devs = []
    for tid in tasks:
        rows = BANK.build_rows(tid, per_cell=PER_CELL)
        fams = set(BANK.TASKS[tid].families)
        cells = {f: [r for r in rows if r["family"] == f and r["split"] == EVAL_SPLIT]
                 for f in FAMILIES if f in fams}
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        acc = {f: {"whole": [], **{f"head{h}": [] for h in range(NH)}} for f in cells}
        for fam, rws in cells.items():
            for b in CB.batches(rws):
                ids, fin, ans = CB.pack(b, "base")
                per, whole = head_writes(m, ids, fin); fwd += 1
                devs.append((per.sum(1) - whole).abs().max().item())
                lg = run_with_W(m, ids, fin, None, None); fwd += 1
                mn = CB.margins(lg, ans, cand)
                lgw = run_with_W(m, ids, fin, whole, FULL); fwd += 1
                acc[fam]["whole"].append((mn - CB.margins(lgw, ans, cand)).cpu().numpy())
                for hd in range(NH):
                    lgh = run_with_W(m, ids, fin, per[:, hd], FULL); fwd += 1
                    acc[fam][f"head{hd}"].append((mn - CB.margins(lgh, ans, cand)).cpu().numpy())
        dm = {f: {k: float(np.concatenate(v).mean()) for k, v in d.items()} for f, d in acc.items()}
        heads = sorted(((h, dm["A1"][f"head{h}"]) for h in range(NH)), key=lambda kv: -kv[1])
        top2 = tuple(sorted(h for h, _ in heads[:2]))
        whole = max(dm["A1"]["whole"], BARS["floor"])
        def ratio(key):
            ctrl = max(abs(dm.get("P", {}).get(key, 0.0)), abs(dm.get("C", {}).get(key, 0.0)))
            return ctrl / max(dm["A1"][key], BARS["floor"])
        hr = {f"head{h}": ratio(f"head{h}") for h in range(NH)}
        best_head = min(hr, key=lambda k: hr[k])
        results[tid] = {
            "head_damage_A1": {f"head{h}": dm["A1"][f"head{h}"] for h in range(NH)},
            "whole_damage_A1": dm["A1"]["whole"], "damage_by_family": dm,
            "head_ladder": [[h, round(v, 4)] for h, v in heads],
            "top2_heads": list(top2),
            "top2_share": (heads[0][1] + heads[1][1]) / whole,
            "whole_ratio": ratio("whole"), "head_ratios": hr,
            "best_head": best_head, "best_head_ratio": hr[best_head],
            "rescue": ratio("whole") - hr[best_head],
            "rows": {f: len(v) for f, v in cells.items()},
        }
        print(f"[heads] {tid:30s} top2={top2} share={results[tid]['top2_share']:.2f} "
              f"whole_ratio={results[tid]['whole_ratio']:.2f} best_head={best_head}:{hr[best_head]:.2f}", flush=True)

    pairs = {}
    for t in results:
        pairs[tuple(results[t]["top2_heads"])] = pairs.get(tuple(results[t]["top2_heads"]), 0) + 1
    mode_pair, mode_n = (max(pairs.items(), key=lambda kv: kv[1]) if pairs else ((), 0))
    med = lambda k: float(np.median([results[t][k] for t in results])) if results else float("nan")
    nl = results.get("numbered_list.index_successor")
    selective_heads = [t for t in results if results[t]["best_head_ratio"] <= 0.25]
    preds = {
        'pred_a_head_decomposition_is_exact': bool(devs and max(devs) <= BARS["exact_tol"]),
        'pred_b_two_heads_carry_the_write': bool(med("top2_share") >= BARS["top2_share"]),
        'pred_c_the_head_pair_is_shared': bool(mode_n >= BARS["shared_pairs"]),
        'pred_d_numbered_list_replicates_r576_heads':
            bool(nl and tuple(nl["top2_heads"]) == R576_HEADS),
        'pred_e_heads_do_not_rescue_selectivity': bool(med("rescue") <= BARS["rescue_margin"]),
    }
    nulls = {
        "b_null_top2_le_.35": bool(med("top2_share") <= NULLS["top2_share_le"]),
        "c_null_pair_not_shared": bool(mode_n <= NULLS["shared_pairs_le"]),
        "e_null_heads_are_selective": bool(len(selective_heads) >= NULLS["selective_heads_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "layer": LAYER, "eval_split": EVAL_SPLIT, "r576_heads": list(R576_HEADS),
              "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "summary": {"tasks": sorted(results), "top2_pairs": {str(k): v for k, v in pairs.items()},
                          "mode_pair": list(mode_pair), "mode_count": mode_n,
                          "max_decomposition_dev": float(max(devs)) if devs else None,
                          "selective_head_tasks": selective_heads,
                          "medians": {k: med(k) for k in ("top2_share", "whole_ratio",
                                                          "best_head_ratio", "rescue")}},
              "tasks": results, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd, "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "medians": result["summary"]["medians"],
                      "pairs": {str(k): v for k, v in pairs.items()}}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2817 capable attn8-writer behaviours x {NH} head writes x {FAMILIES} on {EVAL_SPLIT}; no model loaded")
        sys.exit(0)
    main()
