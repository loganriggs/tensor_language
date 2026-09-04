#!/usr/bin/env python
"""circuit_battery_roundness_steering -- can the roundness vector be USED? The first edit test of this lineage.

SS2841: roundness switches this model between two behaviours -- multiples of ten are continued BY THE STEP (1.000, 6/6) and everything
else by PLUS ONE (.313), with no overlap. SS2842/SS2843/SS2844 localised the switch to attention 8, then heads {3, 7}, then to a single
128-dimensional direction inside head 3 that carries .874 of that head's effect on held-out pairs and transports across surfaces at
|cos| .974.

Localisation is not manipulation. This rung asks whether that vector is a HANDLE: inject it into head 3's output on a NON-ROUND prompt
the model would continue by plus-one, and see whether it switches to step continuation -- and inject its negative into a ROUND prompt and
see whether it switches the other way. The direction and its scale are fitted on one half of the prompts and every arm is scored on the
held-out half. This is the campaign's stated goal ("predictive, MANIPULABLE, editable") reduced to one falsifiable question.

# BQGATE: EXPERIMENT  pred_a_injection_flips_non_round_prompts pred_b_random_direction_does_not
#                     pred_c_injection_recovers_the_full_swap pred_d_the_reverse_edit_works
#                     pred_e_full_swap_bounds_the_injection

SIGN CONVENTION: flip rate is the fraction of held-out prompts whose argmax over the numeric candidate set becomes the STEP answer
(for the forward edit) or the PLUS-ONE answer (for the reverse edit); HIGHER MEANS THE EDIT WORKED. REC is the logit-difference recovery
of SS2842/SS2844. No CE and no SS312 L2; nothing installs into the frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_ROUNDNESS_STEERING_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_roundness_steering.py")
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
PREREG = R.POLY / "CIRCUIT_BATTERY_ROUNDNESS_STEERING_PREREGISTRATION.md"
RDIR = ROOT / "circuit_battery_roundness_direction_results.json"
RUNG = "circuit_battery_roundness_steering"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "97bb6c170b92fc29b0ae644f0d5d3372d90209c5f89800df06934b4a83975d9b",
          RDIR: "232e023b58e6b429d936143d1b139c6c417d8efe77fa7ef1716eb4c9264d3205",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NH, HD = R.D, R.NH, R.HD
LAYER, HEAD = 8, 3
SEED = 2844
ENC = BANK.ENC
BARS = {"flip": 0.50, "random_flip": 0.05, "rec_frac": 0.50, "reverse_flip": 0.30}
NULLS = {"flip_le": 0.10, "random_flip_ge": 0.30, "rec_frac_le": 0.15, "reverse_flip_le": 0.05}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def run_edit(m, tokens, finals, donor_y=None, mode=None, u=None, alpha=0.0):
    """mode: None | 'swap' (donor head slice) | 'add' (add alpha*u to the head slice at every position)."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    cap = {}
    sl = slice(HEAD * HD, (HEAD + 1) * HD)

    def pre(_mod, args):
        y = args[0]
        cap["y"] = y
        if mode is None:
            return None
        y2 = y.clone()
        if mode == "swap":
            y2[..., sl] = donor_y[..., sl]
        else:
            y2[..., sl] = y[..., sl] + (alpha * u).to(y.dtype)
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
    sl = slice(HEAD * HD, (HEAD + 1) * HD)
    results = {}
    for fmt in RL.FORMATS:
        rows = RL.build_pairs(fmt)
        if SMOKE:
            rows = rows[:8]
        half = max(1, len(rows) // 2)
        fit_rows, ev = rows[:half], rows[half:] or rows[:half]

        # fit the direction AND its scale on the fit half only
        deltas = []
        for b in CB.batches(fit_rows):
            bid, bfin, _ = CB.pack(b, "base")
            did, dfin, _ = CB.pack(b, "donor")
            _l, dy = run_edit(m, did, dfin); fwd[0] += 1
            _l, by = run_edit(m, bid, bfin); fwd[0] += 1
            ar = torch.arange(len(b), device=DEV)
            # ROUND-WARD by construction: base is the round twin, donor the non-round one. SS2844 fitted the
            # opposite sign (donor - base); adding THAT to a non-round prompt pushes it further from the step
            # answer, which the smoke showed as a logit gain of -0.26. The preregistration's intent is
            # "inject ... and ask whether it switches to step continuation", so the injected direction must
            # point round-ward; u here is exactly -1 x SS2844's vector and every number below is unaffected in
            # magnitude. Disclosed in the ledger section.
            deltas.append((by[ar, bfin][:, sl] - dy[ar, dfin][:, sl]).float())
        Dm = torch.cat(deltas, 0)
        u = Dm.mean(0); nrm = u.norm().clamp_min(1e-9); u = u / nrm
        alpha = float((Dm @ u).mean())                      # one fitted scalar: the mean projection
        rv = torch.randn(HD, generator=g).to(DEV); rv = rv / rv.norm()

        def flips(rows_, direction, which, alph):
            """which='step' -> the base (round) answer; 'plusone' -> the donor answer."""
            hit, tot, recs = 0, 0, []
            for b in CB.batches(rows_):
                # the NON-round prompt is the donor of the pair; the round one is the base
                if which == "step":
                    ids, fin, tgt = CB.pack(b, "donor")     # edit a non-round prompt
                    want = torch.tensor([ENC.encode(r["base_answer"])[0] for r in b], device=DEV)
                    other = tgt
                else:
                    ids, fin, tgt = CB.pack(b, "base")      # edit a round prompt
                    want = torch.tensor([ENC.encode(r["donor_answer"])[0] for r in b], device=DEV)
                    other = tgt
                lg0, _ = run_edit(m, ids, fin); fwd[0] += 1
                lg1, _ = run_edit(m, ids, fin, mode="add", u=direction, alpha=alph); fwd[0] += 1
                pick = cand[lg1[:, cand].argmax(1)]
                hit += int((pick == want).sum()); tot += len(b)
                d0 = (lg0.gather(1, want[:, None]) - lg0.gather(1, other[:, None])).squeeze(1)
                d1 = (lg1.gather(1, want[:, None]) - lg1.gather(1, other[:, None])).squeeze(1)
                recs.append((d1 - d0).cpu().numpy())
            return (hit / tot if tot else float("nan")), float(np.concatenate(recs).mean())

        flip, dl = flips(ev, u, "step", alpha)
        rflip, rdl = flips(ev, rv, "step", alpha)
        revflip, revdl = flips(ev, -u, "plusone", alpha)   # away from round, on round prompts

        # the SS2843 upper bound: swap head 3's whole slice from the round twin into the non-round prompt
        hit, tot, swrec = 0, 0, []
        for b in CB.batches(ev):
            did, dfin, dans = CB.pack(b, "donor")
            bid, bfin, bans = CB.pack(b, "base")
            _l, by = run_edit(m, bid, bfin); fwd[0] += 1
            lg0, _ = run_edit(m, did, dfin); fwd[0] += 1
            lg1, _ = run_edit(m, did, dfin, donor_y=by, mode="swap"); fwd[0] += 1
            pick = cand[lg1[:, cand].argmax(1)]
            hit += int((pick == bans).sum()); tot += len(b)
            d0 = (lg0.gather(1, bans[:, None]) - lg0.gather(1, dans[:, None])).squeeze(1)
            d1 = (lg1.gather(1, bans[:, None]) - lg1.gather(1, dans[:, None])).squeeze(1)
            swrec.append((d1 - d0).cpu().numpy())
        swap_flip = hit / tot if tot else float("nan")
        swap_dl = float(np.concatenate(swrec).mean())

        results[fmt] = {"flip_rate": flip, "random_flip_rate": rflip, "reverse_flip_rate": revflip,
                        "swap_flip_rate": swap_flip, "logit_gain": dl, "random_logit_gain": rdl,
                        "reverse_logit_gain": revdl, "swap_logit_gain": swap_dl,
                        "rec_fraction_of_swap": dl / swap_dl if abs(swap_dl) > 1e-9 else float("nan"),
                        "alpha": alpha, "n_fit": int(Dm.shape[0]), "n_eval": len(ev)}
        p = results[fmt]
        print(f"[steer] {fmt:8s} flip={flip:.3f} rand={rflip:.3f} rev={revflip:.3f} swap={swap_flip:.3f} "
              f"dlogit={dl:.2f} frac_of_swap={p['rec_fraction_of_swap']:.3f}", flush=True)

    med = lambda k: float(np.median([results[f][k] for f in results]))
    preds = {
        'pred_a_injection_flips_non_round_prompts': bool(med("flip_rate") >= BARS["flip"]),
        'pred_b_random_direction_does_not': bool(med("random_flip_rate") <= BARS["random_flip"]),
        'pred_c_injection_recovers_the_full_swap': bool(med("rec_fraction_of_swap") >= BARS["rec_frac"]),
        'pred_d_the_reverse_edit_works': bool(med("reverse_flip_rate") >= BARS["reverse_flip"]),
        'pred_e_full_swap_bounds_the_injection': bool(all(
            results[f]["swap_logit_gain"] >= results[f]["logit_gain"] - 1e-6 for f in results)),
    }
    nulls = {
        "a_null_no_flip": bool(med("flip_rate") <= NULLS["flip_le"]),
        "b_null_random_flips": bool(med("random_flip_rate") >= NULLS["random_flip_ge"]),
        "c_null_no_recovery": bool(med("rec_fraction_of_swap") <= NULLS["rec_frac_le"]),
        "d_null_reverse_fails": bool(med("reverse_flip_rate") <= NULLS["reverse_flip_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "layer": LAYER, "head": HEAD, "seed": SEED,
              "summary": {"medians": {k: med(k) for k in
                                      ("flip_rate", "random_flip_rate", "reverse_flip_rate",
                                       "swap_flip_rate", "logit_gain", "rec_fraction_of_swap")}},
              "formats_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0,
                        "fitted_parameters": len(RL.FORMATS) * (HD + 1),
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1100])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: inject the SS2844 roundness direction into head {HEAD} of attention {LAYER} on held-out "
              f"non-round prompts (and its negative on round ones), two formats; no model loaded")
        sys.exit(0)
    main()
