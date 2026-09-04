#!/usr/bin/env python
"""circuit_battery_roundness_decision_ladder -- the feature is in heads {3,7}; WHERE IS THE DECISION?

SS2846: editing both heads of the {3, 7} pair -- and even swapping both slices outright, the largest possible intervention confined to
the pair -- flips only .208 of held-out non-round prompts to step continuation. The feature lives there (SS2843, SS2844) and the decision
does not. SS2842 measured LOGIT recovery for every component but never a FLIP rate, so nothing in this lineage yet says how much of the
model has to be moved before the behaviour actually changes.

This rung measures the flip rate up a ladder of progressively larger donor patches, from the pair to the whole model, so the answer is a
curve rather than an argument.

# BQGATE: EXPERIMENT  pred_a_the_component_beats_the_pair pred_b_adding_readers_beats_the_component
#                     pred_c_everything_reproduces_the_donor pred_d_the_ladder_is_monotone
#                     pred_e_the_pair_replicates

SIGN CONVENTION: flip rate is the fraction of held-out non-round prompts whose argmax over the numeric vocabulary becomes the STEP
answer, HIGHER MEANS THE INTERVENTION DECIDED THE BEHAVIOUR. No CE and no SS312 L2; nothing installs.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_ROUNDNESS_DECISION_LADDER_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_roundness_decision_ladder.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
import circuit_battery_roundness_localisation as RL
import fastload
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_ROUNDNESS_DECISION_LADDER_PREREGISTRATION.md"
TWOHEAD = ROOT / "circuit_battery_roundness_two_head_edit_results.json"
RLOC = ROOT / "circuit_battery_roundness_localisation_results.json"
RUNG = "circuit_battery_roundness_decision_ladder"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "fe7f6d92043e38765f85ee05d89bf675cd0ce0b3a89e588725ff753631d7df3e",
          TWOHEAD: "7c96fe4a4108f3b4a55bb51d2d547f738642c1d733e9857ec50aa3b8b68f3dd5",
          RLOC: "331454aac1ce218d9194255e19c81c53eca38d99cc6c2b685ff2d9e0ac12788c",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NH, HD, NLAY = R.D, R.NH, R.HD, R.NL
LAYER, HEADS = 8, (3, 7)
PAIR_REF = 0.20833333333333331          # SS2846's SWAP_PAIR flip rate
ENC = BANK.ENC
BARS = {"component": 0.40, "with_readers": 0.70, "everything": 0.95, "pair_tol": 0.10}
NULLS = {"component_le": 0.25, "with_readers_le": 0.40, "everything_le": 0.70}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def run_patch(m, tokens, finals, donor=None, heads=None, comps=()):
    """donor: dict of cached donor outputs. heads: swap only these heads of attention LAYER.
    comps: whole components (kind, layer) to swap outright."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    cap = {}

    def pre(_mod, args):
        y = args[0]
        cap["y"] = y
        if donor is None or not heads:
            return None
        y2 = y.clone()
        for h in heads:
            y2[..., h * HD:(h + 1) * HD] = donor["ycat"][..., h * HD:(h + 1) * HD]
        return (y2,)

    hk = m.transformer.h[LAYER].attn.c_proj.register_forward_pre_hook(pre)
    try:
        for site, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
            if donor is not None and ("attn", site) in comps:
                write = donor[("attn", site)]
            if donor is None:
                cap[("attn", site)] = write
            x = x + write
            out = blk.mlp(F.rms_norm(x, (D,)))
            if donor is not None and ("mlp", site) in comps:
                out = donor[("mlp", site)]
            if donor is None:
                cap[("mlp", site)] = out
            x = x + out
    finally:
        hk.remove()
    logits = (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float()
    return logits, cap


def main():
    t0 = time.time()
    check_hashes()
    rl = json.load(open(RLOC))
    m = fastload.load_model_fast().to(DEV).eval()
    cand = torch.tensor(sorted({ENC.encode(s)[0] for s in [f" {i}" for i in range(0, 200)]
                                if len(ENC.encode(s)) == 1}), device=DEV)
    fwd = [0]
    ALL = [(kd, l) for l in range(NLAY) for kd in ("attn", "mlp")]
    results = {}
    for fmt in RL.FORMATS:
        rows = RL.build_pairs(fmt)
        if SMOKE:
            rows = rows[:8]
        # SS2845/SS2846 fitted on the first half and scored on the second; pred_e is a replication of
        # SS2846's pair number, so it must be measured on the SAME population -- the held-out half.
        half = max(1, len(rows) // 2)
        rows = rows[half:] or rows[:half]
        # the top-3 non-attn8 components of this format's SS2842 recovery map, fixed by that receipt
        order = [n for n in rl["formats_detail"][fmt]["order_top8"] if n != "attn8"][:3]
        readers = [(("attn" if n.startswith("attn") else "mlp"),
                    int(n[len("attn"):] if n.startswith("attn") else n[len("mlp"):])) for n in order]
        rungs = {"pair": ("heads", HEADS, ()),
                 "component": ("comps", None, (("attn", LAYER),)),
                 "component_plus_readers": ("comps", None, tuple([("attn", LAYER)] + readers)),
                 "everything": ("comps", None, tuple(ALL))}
        out = {"readers_used": order}
        for name, (kind, heads, comps) in rungs.items():
            hit, tot = 0, 0
            for b in CB.batches(rows):
                did, dfin, dans = CB.pack(b, "donor")      # the NON-round prompt is edited
                bid, bfin, bans = CB.pack(b, "base")       # the round twin supplies the donor values
                _lb, cache = run_patch(m, bid, bfin); fwd[0] += 1
                _l2, cap2 = run_patch(m, bid, bfin); fwd[0] += 1
                cache["ycat"] = cap2.get("y") if cap2.get("y") is not None else None
                if cache["ycat"] is None:
                    ar = torch.arange(len(b), device=DEV)
                    cache["ycat"] = _l2.new_zeros(len(b), 1, D)
                lg, _ = run_patch(m, did, dfin, donor=cache,
                                  heads=(heads if kind == "heads" else None), comps=comps); fwd[0] += 1
                pick = cand[lg[:, cand].argmax(1)]
                hit += int((pick == bans).sum()); tot += len(b)
            out[name] = hit / tot if tot else float("nan")
        results[fmt] = out
        print(f"[ladder] {fmt:8s} pair={out['pair']:.3f} comp={out['component']:.3f} "
              f"+readers={out['component_plus_readers']:.3f} all={out['everything']:.3f} "
              f"(readers {order})", flush=True)

    med = lambda k: float(np.median([results[f][k] for f in results]))
    ladder_ok = all(results[f]["pair"] <= results[f]["component"] + 1e-9
                    and results[f]["component"] <= results[f]["component_plus_readers"] + 1e-9
                    and results[f]["component_plus_readers"] <= results[f]["everything"] + 1e-9
                    for f in results)
    preds = {
        'pred_a_the_component_beats_the_pair': bool(med("component") >= BARS["component"]),
        'pred_b_adding_readers_beats_the_component': bool(med("component_plus_readers") >= BARS["with_readers"]),
        'pred_c_everything_reproduces_the_donor': bool(med("everything") >= BARS["everything"]),
        'pred_d_the_ladder_is_monotone': bool(ladder_ok),
        'pred_e_the_pair_replicates': bool(abs(med("pair") - PAIR_REF) <= BARS["pair_tol"]),
    }
    nulls = {
        "a_null_component_no_better": bool(med("component") <= NULLS["component_le"]),
        "b_null_readers_add_nothing": bool(med("component_plus_readers") <= NULLS["with_readers_le"]),
        "c_null_everything_fails": bool(med("everything") <= NULLS["everything_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "layer": LAYER, "heads": list(HEADS), "pair_reference_2846": PAIR_REF,
              "summary": {"medians": {k: med(k) for k in
                                      ("pair", "component", "component_plus_readers", "everything")},
                          "ladder_monotone": ladder_ok},
              "formats_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1000])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: flip-rate ladder from heads {HEADS} to all {2 * NLAY} components on the roundness "
              f"minimal pairs, two formats; no model loaded")
        sys.exit(0)
    main()
