#!/usr/bin/env python
"""circuit_battery_roundness_direction -- is the roundness switch ONE DIRECTION inside head 3's output?

SS2842/SS2843: the step-versus-plus-one switch lives in attention 8, and inside it in heads {3, 7} -- the same pair that writes the last
item's identity (SS2820, Codex's R576) -- with head 3 leading both formats and the pair holding .925 of the positive head recovery. That
locates the feature but does not say what it is. SS2826 showed that for a READER, a single unfitted direction carried a fifth of the
damage at 2.4x the block's specificity while holding .0021 of the energy; the analogous question on the WRITE side has never been asked.

This rung asks it: the per-row difference between the donor's and the base's head-3 output is the "roundness delta"; a MEAN direction is
fitted on half the pairs and the switch is then driven by projecting each row's delta onto that one fixed direction, scored on the
held-out half. If a single fixed direction carries the switch, the feature is compilable.

# BQGATE: EXPERIMENT  pred_a_one_fitted_direction_carries_the_switch pred_b_random_direction_is_inert
#                     pred_c_the_direction_transports_across_formats pred_d_it_is_not_the_bulk_output
#                     pred_e_full_delta_reproduces_the_head_patch

SIGN CONVENTION: REC = (ld_patch - ld_base) / max(ld_donor - ld_base, 1e-3) with ld = logit(plus-one) - logit(step); 0 = no effect,
1 = full switch. No CE and no SS312 L2; nothing installs.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_ROUNDNESS_DIRECTION_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_roundness_direction.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_roundness_localisation as RL
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_ROUNDNESS_DIRECTION_PREREGISTRATION.md"
RHEAD = ROOT / "circuit_battery_roundness_head_split_results.json"
RUNG = "circuit_battery_roundness_direction"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "a93c4d819475bcd4351891c4e8d97d22a7185d5c0c0a24b13db804ed7ce810fd",
          RHEAD: "5e076b400295c62d9936bb69f433d207d03111e979253d677a0ebb187109d89e",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NH, HD = R.D, R.NH, R.HD
LAYER, HEAD = 8, 3
SEED = 2843
BARS = {"heldout_frac": 0.50, "random_rec": 0.10, "transport_cos": 0.50, "bulk_cos": 0.80,
        "exact_tol": 0.02}
NULLS = {"heldout_frac_le": 0.15, "random_rec_ge": 0.40, "transport_cos_le": 0.10}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def run_dirpatch(m, tokens, finals, donor_y=None, mode=None, u=None):
    """mode: None | 'head' (swap head HEAD's whole slice) | 'proj' (swap only its component along u)."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    cap = {}
    sl = slice(HEAD * HD, (HEAD + 1) * HD)

    def pre(_mod, args):
        y = args[0]
        cap["y"] = y
        if donor_y is None or mode is None:
            return None
        y2 = y.clone()
        if mode == "head":
            y2[..., sl] = donor_y[..., sl]
        else:
            delta = (donor_y[..., sl] - y[..., sl]).float()
            proj = (delta * u).sum(-1, keepdim=True) * u
            y2[..., sl] = y[..., sl] + proj.to(y.dtype)
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
    m = R.load_model().to(DEV).eval()
    g = torch.Generator(device="cpu").manual_seed(SEED)
    fwd = [0]
    fitted_dirs, results = {}, {}
    for fmt in RL.FORMATS:
        rows = RL.build_pairs(fmt)
        if SMOKE:
            rows = rows[:8]
        half = max(1, len(rows) // 2)
        fit_rows, eval_rows = rows[:half], rows[half:] or rows[:half]
        sl = slice(HEAD * HD, (HEAD + 1) * HD)

        deltas, bulks = [], []
        for b in CB.batches(fit_rows):
            bid, bfin, _ = CB.pack(b, "base")
            did, dfin, _ = CB.pack(b, "donor")
            _ld, dy = run_dirpatch(m, did, dfin); fwd[0] += 1
            _lb, by = run_dirpatch(m, bid, bfin); fwd[0] += 1
            ar = torch.arange(len(b), device=DEV)
            deltas.append((dy[ar, dfin][:, sl] - by[ar, bfin][:, sl]).float())
            bulks.append(by[ar, bfin][:, sl].float())
        Dm = torch.cat(deltas, 0); Bk = torch.cat(bulks, 0)
        u = Dm.mean(0); u = u / u.norm().clamp_min(1e-9)
        bulk = Bk.mean(0); bulk = bulk / bulk.norm().clamp_min(1e-9)
        bulk_cos = float(abs(torch.dot(u, bulk)))
        rv = torch.randn(HD, generator=g).to(DEV); rv = rv / rv.norm()
        fitted_dirs[fmt] = u

        def rec_of(mode, uu):
            acc = []
            for b in CB.batches(eval_rows):
                bid, bfin, bans = CB.pack(b, "base")
                did, dfin, dans = CB.pack(b, "donor")
                ldon, dy = run_dirpatch(m, did, dfin); fwd[0] += 1
                lb, _ = run_dirpatch(m, bid, bfin); fwd[0] += 1
                base_ld = (lb.gather(1, dans[:, None]) - lb.gather(1, bans[:, None])).squeeze(1)
                don_ld = (ldon.gather(1, dans[:, None]) - ldon.gather(1, bans[:, None])).squeeze(1)
                den = (don_ld - base_ld).clamp_min(1e-3)
                lp, _ = run_dirpatch(m, bid, bfin, donor_y=dy, mode=mode, u=uu); fwd[0] += 1
                ld = (lp.gather(1, dans[:, None]) - lp.gather(1, bans[:, None])).squeeze(1)
                acc.append(((ld - base_ld) / den).cpu().numpy())
            return float(np.concatenate(acc).mean()) if acc else float("nan")

        head_rec = rec_of("head", None)
        proj_rec = rec_of("proj", u)
        rand_rec = rec_of("proj", rv)
        results[fmt] = {"head_recovery": head_rec, "fitted_direction_recovery": proj_rec,
                        "random_direction_recovery": rand_rec,
                        "fraction_of_head": proj_rec / head_rec if abs(head_rec) > 1e-9 else float("nan"),
                        "cos_direction_vs_bulk_output": bulk_cos,
                        "n_fit": int(Dm.shape[0]), "n_eval": len(eval_rows)}
        p = results[fmt]
        print(f"[rdir] {fmt:8s} head={head_rec:.3f} fitted_dir={proj_rec:.3f} "
              f"frac={p['fraction_of_head']:.3f} rand={rand_rec:.3f} bulk_cos={bulk_cos:.3f}", flush=True)

    fmts = sorted(fitted_dirs)
    transport = float(abs(torch.dot(fitted_dirs[fmts[0]], fitted_dirs[fmts[1]]))) if len(fmts) == 2 else float("nan")
    med = lambda k: float(np.median([results[f][k] for f in results]))
    preds = {
        'pred_a_one_fitted_direction_carries_the_switch': bool(med("fraction_of_head") >= BARS["heldout_frac"]),
        'pred_b_random_direction_is_inert': bool(med("random_direction_recovery") <= BARS["random_rec"]),
        'pred_c_the_direction_transports_across_formats': bool(transport >= BARS["transport_cos"]),
        'pred_d_it_is_not_the_bulk_output': bool(med("cos_direction_vs_bulk_output") <= BARS["bulk_cos"]),
        'pred_e_full_delta_reproduces_the_head_patch': bool(all(
            abs(results[f]["head_recovery"]) > 1e-6 for f in results)),
    }
    nulls = {
        "a_null_direction_fails": bool(med("fraction_of_head") <= NULLS["heldout_frac_le"]),
        "b_null_random_works": bool(med("random_direction_recovery") >= NULLS["random_rec_ge"]),
        "c_null_no_transport": bool(transport <= NULLS["transport_cos_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "layer": LAYER, "head": HEAD, "seed": SEED,
              "summary": {"transport_cosine": transport,
                          "medians": {k: med(k) for k in ("head_recovery", "fitted_direction_recovery",
                                                          "fraction_of_head", "random_direction_recovery",
                                                          "cos_direction_vs_bulk_output")}},
              "formats_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": len(RL.FORMATS) * HD,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1100])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: head {HEAD} of attention {LAYER}, one mean roundness direction fitted on half the pairs "
              f"and scored on the held-out half, two formats; no model loaded")
        sys.exit(0)
    main()
