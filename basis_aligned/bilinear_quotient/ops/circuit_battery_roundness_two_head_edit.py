#!/usr/bin/env python
"""circuit_battery_roundness_two_head_edit -- edit BOTH heads of the pair, and repair SS2845's failed scaling bound.

SS2841: roundness switches this model between two behaviours -- multiples of ten are continued BY THE STEP (1.000, 6/6) and everything
else by PLUS ONE (.313), with no overlap. SS2842/SS2843/SS2844 localised the switch to attention 8, then heads {3, 7}, then to a single
128-dimensional direction inside head 3 that carries .874 of that head's effect on held-out pairs and transports across surfaces at
|cos| .974.

Localisation is not manipulation. This rung asks whether that vector is a HANDLE: inject it into head 3's output on a NON-ROUND prompt
the model would continue by plus-one, and see whether it switches to step continuation -- and inject its negative into a ROUND prompt and
see whether it switches the other way. The direction and its scale are fitted on one half of the prompts and every arm is scored on the
held-out half. This is the campaign's stated goal ("predictive, MANIPULABLE, editable") reduced to one falsifiable question.

# BQGATE: EXPERIMENT  pred_a_two_heads_flip_more_than_one pred_b_two_head_gain_exceeds_one_head
#                     pred_c_the_sanity_bound_now_holds pred_d_random_pair_is_inert
#                     pred_e_the_edit_is_monotone_in_alpha

SIGN CONVENTION: flip rate is the fraction of held-out prompts whose argmax over the numeric candidate set becomes the STEP answer
(for the forward edit) or the PLUS-ONE answer (for the reverse edit); HIGHER MEANS THE EDIT WORKED. REC is the logit-difference recovery
of SS2842/SS2844. No CE and no SS312 L2; nothing installs into the frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_ROUNDNESS_TWO_HEAD_EDIT_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_roundness_two_head_edit.py")
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
PREREG = R.POLY / "CIRCUIT_BATTERY_ROUNDNESS_TWO_HEAD_EDIT_PREREGISTRATION.md"
STEER = ROOT / "circuit_battery_roundness_steering_results.json"
RUNG = "circuit_battery_roundness_two_head_edit"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "e1a32e3d6f4c37bb5def39afb2dd62c4ddb70a1e5ebdde42514b42645ce37736",
          STEER: "7aa2134d911b3ba2fd1144e11af93e88647522258e603611b5f4d2423ce846db",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NH, HD = R.D, R.NH, R.HD
LAYER, HEADS = 8, (3, 7)      # SS2820/SS2843: the pair that writes identity AND roundness
SEED = 2845
ENC = BANK.ENC
ONE_HEAD_FLIP = 0.125          # SS2845 median flip rate, head 3 alone
ONE_HEAD_GAIN = 0.7703808546066284   # SS2845 median logit gain, head 3 alone
BARS = {"flip": 0.30, "gain_over_one_head": 0.30, "random_flip": 0.05, "alpha_mono": 0.0}
NULLS = {"flip_le": 0.10, "gain_over_one_head_le": 0.0, "random_flip_ge": 0.30}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def run_edit(m, tokens, finals, donor_y=None, mode=None, dirs=None, alpha=1.0, heads=HEADS):
    """mode: None | 'swap' (donor slices for `heads`) | 'add' (add alpha*dirs[h] to each head slice)."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    cap = {}

    def pre(_mod, args):
        y = args[0]
        cap["y"] = y
        if mode is None:
            return None
        y2 = y.clone()
        for h in heads:
            sl = slice(h * HD, (h + 1) * HD)
            if mode == "swap":
                y2[..., sl] = donor_y[..., sl]
            else:
                y2[..., sl] = y[..., sl] + (alpha * dirs[h]).to(y.dtype)
        return (y2,)

    hk = m.transformer.h[LAYER].attn.c_proj.register_forward_pre_hook(pre)
    try:
        for site, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
            x = x + write
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    finally:
        hk.remove()
    return (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float(), cap.get("y")


def main():
    t0 = time.time()
    check_hashes()
    m = fastload.load_model_fast().to(DEV).eval()
    g = torch.Generator(device="cpu").manual_seed(SEED)
    cand = torch.tensor(sorted({ENC.encode(s)[0] for s in [f" {i}" for i in range(0, 200)]
                                if len(ENC.encode(s)) == 1}), device=DEV)
    fwd = [0]
    results = {}
    for fmt in RL.FORMATS:
        rows = RL.build_pairs(fmt)
        if SMOKE:
            rows = rows[:8]
        half = max(1, len(rows) // 2)
        fit_rows, ev = rows[:half], rows[half:] or rows[:half]

        # ROUND-WARD directions and per-head scales, fitted on the fit half only
        acc = {h: [] for h in HEADS}
        for b in CB.batches(fit_rows):
            bid, bfin, _ = CB.pack(b, "base")
            did, dfin, _ = CB.pack(b, "donor")
            _l, dy = run_edit(m, did, dfin); fwd[0] += 1
            _l, by = run_edit(m, bid, bfin); fwd[0] += 1
            ar = torch.arange(len(b), device=DEV)
            for h in HEADS:
                sl = slice(h * HD, (h + 1) * HD)
                acc[h].append((by[ar, bfin][:, sl] - dy[ar, dfin][:, sl]).float())
        dirs, alphas = {}, {}
        for h in HEADS:
            Dm = torch.cat(acc[h], 0)
            u = Dm.mean(0); u = u / u.norm().clamp_min(1e-9)
            dirs[h] = u
            alphas[h] = float((Dm @ u).mean())
        rnd = {}
        for h in HEADS:
            v = torch.randn(HD, generator=g).to(DEV)
            rnd[h] = v / v.norm()

        def measure(dirs_, alpha_mult, heads_, mode="add"):
            hit, tot, gains = 0, 0, []
            for b in CB.batches(ev):
                ids, fin, tgt = CB.pack(b, "donor")               # edit the NON-round prompt
                want = torch.tensor([ENC.encode(r["base_answer"])[0] for r in b], device=DEV)
                bid, bfin, _ = CB.pack(b, "base")
                lg0, _ = run_edit(m, ids, fin); fwd[0] += 1
                if mode == "swap":
                    _l, by = run_edit(m, bid, bfin); fwd[0] += 1
                    lg1, _ = run_edit(m, ids, fin, donor_y=by, mode="swap", heads=heads_); fwd[0] += 1
                else:
                    scaled = {h: dirs_[h] * (alphas[h] * alpha_mult) for h in heads_}
                    lg1, _ = run_edit(m, ids, fin, mode="add",
                                      dirs={h: scaled[h] for h in heads_}, alpha=1.0, heads=heads_); fwd[0] += 1
                pick = cand[lg1[:, cand].argmax(1)]
                hit += int((pick == want).sum()); tot += len(b)
                d0 = (lg0.gather(1, want[:, None]) - lg0.gather(1, tgt[:, None])).squeeze(1)
                d1 = (lg1.gather(1, want[:, None]) - lg1.gather(1, tgt[:, None])).squeeze(1)
                gains.append((d1 - d0).cpu().numpy())
            return (hit / tot if tot else float("nan")), float(np.concatenate(gains).mean())

        both_flip, both_gain = measure(dirs, 1.0, HEADS)
        half_flip, half_gain = measure(dirs, 0.5, HEADS)
        one_flip, one_gain = measure(dirs, 1.0, (HEADS[0],))
        rnd_flip, rnd_gain = measure(rnd, 1.0, HEADS)
        swap_flip, swap_gain = measure(dirs, 1.0, HEADS, mode="swap")
        results[fmt] = {"two_head_flip": both_flip, "two_head_gain": both_gain,
                        "half_alpha_flip": half_flip, "half_alpha_gain": half_gain,
                        "one_head_flip": one_flip, "one_head_gain": one_gain,
                        "random_pair_flip": rnd_flip, "random_pair_gain": rnd_gain,
                        "swap_pair_flip": swap_flip, "swap_pair_gain": swap_gain,
                        "gain_over_one_head": both_gain - one_gain,
                        "bound_holds": bool(swap_gain >= both_gain - 1e-6),
                        "alphas": alphas, "n_fit": int(Dm.shape[0]), "n_eval": len(ev)}
        p = results[fmt]
        print(f"[2head] {fmt:8s} two={both_flip:.3f}/{both_gain:.2f} one={one_flip:.3f}/{one_gain:.2f} "
              f"swap={swap_flip:.3f}/{swap_gain:.2f} rnd={rnd_flip:.3f} half={half_gain:.2f} "
              f"bound_ok={p['bound_holds']}", flush=True)

    med = lambda k: float(np.median([results[f][k] for f in results]))
    preds = {
        'pred_a_two_heads_flip_more_than_one': bool(med("two_head_flip") >= BARS["flip"]),
        'pred_b_two_head_gain_exceeds_one_head': bool(med("gain_over_one_head") >= BARS["gain_over_one_head"]),
        'pred_c_the_sanity_bound_now_holds': bool(all(results[f]["bound_holds"] for f in results)),
        'pred_d_random_pair_is_inert': bool(med("random_pair_flip") <= BARS["random_flip"]),
        'pred_e_the_edit_is_monotone_in_alpha': bool(all(
            results[f]["two_head_gain"] > results[f]["half_alpha_gain"] for f in results)),
    }
    nulls = {
        "a_null_no_gain_from_two": bool(med("two_head_flip") <= NULLS["flip_le"]),
        "b_null_second_head_adds_nothing": bool(med("gain_over_one_head") <= NULLS["gain_over_one_head_le"]),
        "d_null_random_flips": bool(med("random_pair_flip") >= NULLS["random_flip_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "layer": LAYER, "heads": list(HEADS), "seed": SEED,
              "reference_2845": {"one_head_flip": ONE_HEAD_FLIP, "one_head_gain": ONE_HEAD_GAIN},
              "summary": {"medians": {k: med(k) for k in
                                      ("two_head_flip", "two_head_gain", "one_head_flip", "one_head_gain",
                                       "gain_over_one_head", "swap_pair_flip", "swap_pair_gain",
                                       "random_pair_flip", "half_alpha_gain")},
                          "bound_holds_all_formats": all(results[f]["bound_holds"] for f in results)},
              "formats_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0,
                        "fitted_parameters": len(RL.FORMATS) * len(HEADS) * (HD + 1),
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1100])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: inject round-ward directions into heads {HEADS} of attention {LAYER} on held-out "
              f"non-round prompts, against one-head, random-pair, half-alpha and full-swap arms; no model loaded")
        sys.exit(0)
    main()
