#!/usr/bin/env python
"""circuit_battery_writer_head_split -- WHICH HEADS of attention 8 write the last salient item, and do they rescue selectivity?

SS2817: attention 8 is the FIT-chosen writer for 7 of 8 capable behaviours and none of them is writer-selective. SS2819: selectivity is
in the read and grows with depth (mlp11 best on 6 of 7). The remaining granularity below the writer is the HEAD. Attention 8's write is
y = c_proj(concat_h o_h) with no bias, so it decomposes EXACTLY into nine additive head writes; each is carried through the same
residual path-patching instrument used since SS2808 and removed from every reader edge plus the direct path. Two questions: is the same
head set the writer across surface forms (component re-use one level finer), and does head granularity make the write selective
(the negative that SS2817 and SS2819 predict)?

# BQGATE: EXPERIMENT  pred_a_head_decomposition_is_exact pred_b_two_heads_carry_the_class_gate
#                     pred_c_the_class_head_set_is_shared pred_d_class_heads_differ_from_margin_heads
#                     pred_e_class_gate_heads_are_few

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS that family's own answer. Nothing installs into the SS312
frontier. Preregistration: polynomial_causal/CIRCUIT_BATTERY_WRITER_HEAD_SPLIT_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_attn5_head_class_split.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_ATTN5_HEAD_CLASS_SPLIT_PREREGISTRATION.md"
BATTERY = ROOT / "circuit_battery_v2_results.json"
RUNG = "circuit_battery_attn5_head_class_split"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "ca5a84128d1a54efbcfb0c8a31d1df63ec6307b89f037c059e5b4f71a21dd5d8",
          BATTERY: "5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NH, HD, NL = R.D, R.NH, R.HD, R.NL
LAYER = 5                      # the class gate (SS2829, SS2830), not the item writer
EVAL_SPLIT = "OOD"
FAMILIES = ("A1",)
PER_CELL = 4 if SMOKE else 16
ATTN8_HEADS = (3, 7)                    # SS2820's writer pair at layer 8, used only as a contrast
BARS = {"exact_tol": 1e-3, "top2_share": 0.60, "shared_pairs": 4, "overlap": 0.50,
        "top3_share": 0.80, "floor": 0.05}
NULLS = {"top2_share_le": 0.35, "shared_pairs_le": 1, "overlap_ge": 0.90, "top3_share_le": 0.50}


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


def logmass(logits, cand):
    lp = torch.log_softmax(logits, dim=-1)
    return torch.logsumexp(lp[:, cand], dim=-1)


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
        rows = [r for r in BANK.build_rows(tid, per_cell=PER_CELL)
                if r["family"] == "A1" and r["split"] == EVAL_SPLIT]
        cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(tid)}), device=DEV)
        cls = {f"head{h}": [] for h in range(NH)}
        cls["whole"] = []
        mar = {k: [] for k in cls}
        for b in CB.batches(rows):
            ids, fin, ans = CB.pack(b, "base")
            per, whole = head_writes(m, ids, fin); fwd += 1
            devs.append((per.sum(1) - whole).abs().max().item())
            lg = run_with_W(m, ids, fin, None, None); fwd += 1
            cn, mn = logmass(lg, cand), CB.margins(lg, ans, cand)
            lgw = run_with_W(m, ids, fin, whole, FULL); fwd += 1
            cls["whole"].append((cn - logmass(lgw, cand)).cpu().numpy())
            mar["whole"].append((mn - CB.margins(lgw, ans, cand)).cpu().numpy())
            for hd in range(NH):
                lgh = run_with_W(m, ids, fin, per[:, hd], FULL); fwd += 1
                cls[f"head{hd}"].append((cn - logmass(lgh, cand)).cpu().numpy())
                mar[f"head{hd}"].append((mn - CB.margins(lgh, ans, cand)).cpu().numpy())
        c = {k: float(np.concatenate(v).mean()) for k, v in cls.items()}
        mm = {k: float(np.concatenate(v).mean()) for k, v in mar.items()}
        ctop = sorted((k for k in c if k != "whole"), key=lambda k: -c[k])
        mtop = sorted((k for k in mm if k != "whole"), key=lambda k: -mm[k])
        whole = max(c["whole"], BARS["floor"])
        results[tid] = {
            "class_damage": c, "margin_damage": mm,
            "class_head_order": ctop, "margin_head_order": mtop,
            "top2_class_heads": sorted(ctop[:2]),
            "top2_class_share": (c[ctop[0]] + c[ctop[1]]) / whole,
            "top3_class_share": (c[ctop[0]] + c[ctop[1]] + c[ctop[2]]) / whole,
            "class_margin_overlap_top3": len(set(ctop[:3]) & set(mtop[:3])) / 3.0,
            "rows": len(rows),
        }
        p = results[tid]
        print(f"[a5heads] {tid:28s} class_top={ctop[:3]} margin_top={mtop[:3]} "
              f"top2={p['top2_class_share']:.2f} top3={p['top3_class_share']:.2f} "
              f"overlap={p['class_margin_overlap_top3']:.2f}", flush=True)

    pairs = {}
    for t in results:
        k = tuple(results[t]["top2_class_heads"])
        pairs[k] = pairs.get(k, 0) + 1
    mode_pair, mode_n = (max(pairs.items(), key=lambda kv: kv[1]) if pairs else ((), 0))
    med = lambda k: float(np.median([results[t][k] for t in results])) if results else float("nan")
    preds = {
        'pred_a_head_decomposition_is_exact': bool(devs and max(devs) <= BARS["exact_tol"]),
        'pred_b_two_heads_carry_the_class_gate': bool(med("top2_class_share") >= BARS["top2_share"]),
        'pred_c_the_class_head_set_is_shared': bool(mode_n >= BARS["shared_pairs"]),
        'pred_d_class_heads_differ_from_margin_heads':
            bool(med("class_margin_overlap_top3") <= BARS["overlap"]),
        'pred_e_class_gate_heads_are_few': bool(med("top3_class_share") >= BARS["top3_share"]),
    }
    nulls = {
        "b_null_top2_le_.35": bool(med("top2_class_share") <= NULLS["top2_share_le"]),
        "c_null_pair_not_shared": bool(mode_n <= NULLS["shared_pairs_le"]),
        "d_null_same_heads": bool(med("class_margin_overlap_top3") >= NULLS["overlap_ge"]),
        "e_null_diffuse": bool(med("top3_class_share") <= NULLS["top3_share_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "layer": LAYER, "eval_split": EVAL_SPLIT, "attn8_heads_for_contrast": list(ATTN8_HEADS),
              "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "summary": {"tasks": sorted(results), "top2_class_pairs": {str(k): v for k, v in pairs.items()},
                          "mode_pair": list(mode_pair), "mode_count": mode_n,
                          "max_decomposition_dev": float(max(devs)) if devs else None,
                          "medians": {k: med(k) for k in ("top2_class_share", "top3_class_share",
                                                          "class_margin_overlap_top3")}},
              "tasks": results, "per_cell": PER_CELL, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd, "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "medians": result["summary"]["medians"],
                      "pairs": {str(k): v for k, v in pairs.items()}}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: SS2817 capable behaviours x {NH} head writes of attention {LAYER}, "
              f"class-mass and margin metrics on {EVAL_SPLIT}; no model loaded")
        sys.exit(0)
    main()
