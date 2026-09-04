#!/usr/bin/env python
"""circuit_battery_attn5_surrogate_price -- can attention 5 be made CHEAPER by keeping what it DOES rather than what it IS?

SS2830: attention 5 costs 2.200 nats of document CE when ablated (3rd of 36) and is 20.4x more expensive per unit of its own write norm
than the median component. SS2831: its class gate lives in heads {5, 7}, which carry .542 of its class damage, and its class and margin
head maps are identical. SS2825/SS2826: in this model, ENERGY-ranked structure is not causal structure -- an in-sample rank-4 subspace
holds .700 of a removal effect's energy and delivers .139 of its damage.

Those three put a falsifiable, useful prediction on the table: a surrogate for attention 5 chosen CAUSALLY (keep the two heads that
gate the class) should cost less document CE than one chosen by ENERGY (project the write onto its own top-k singular directions), even
when the energy surrogate is given far more freedom. This rung measures that directly.

# BQGATE: EXPERIMENT  pred_a_causal_surrogate_beats_energy_surrogate pred_b_energy_rank_curve_is_shallow
#                     pred_c_mean_write_is_not_enough pred_d_two_heads_are_a_small_share_of_energy
#                     pred_e_instrument_reproduces_native_ce

SIGN CONVENTION: d_ce = CE_arm - CE_NATIVE in NATS on natural documents, POSITIVE = the arm HURTS. This is a LOCAL SURROGATE
measurement; it is NOT the SS312 frontier's L2 (CE added above the real model by an installed approximation, LOWER IS BETTER, frontier
norm-2304 at 2.6735), NOTHING here installs into that frontier, and no number below may be quoted as an L2. The energy-basis arm is a
NEGATIVE CONTROL, not a proposed interface -- metric-constructed bases/spans are CLOSED (SS2118 lineage) and this rung does not reopen
them.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_ATTN5_SURROGATE_PRICE_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_attn5_surrogate_price.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_ATTN5_SURROGATE_PRICE_PREREGISTRATION.md"
PRICE = ROOT / "circuit_battery_attn5_class_gate_price_results.json"
HEADSPLIT = ROOT / "circuit_battery_attn5_head_class_split_results.json"
RUNG = "circuit_battery_attn5_surrogate_price"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "72cb27ac1f835dcc3dc9758a1a49a48be8125c309a3b9a4d9c1d023ec8d61f3e",
          PRICE: "870f1c4065ddfea418a545db5536f93776cd445ff8e1b97cbb9f33f88b592e56",
          HEADSPLIT: "42c1d8adcd9c7d241d32b1383f9a8ca17a06d0d10c95ad99000c32b072bbd7fc",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NH, HD, NL, V, T = R.D, R.NH, R.HD, R.NL, R.V, R.T
LAYER = 5
KEEP_HEADS = (5, 7)          # SS2831's class-gate pair, fixed before this run
RANKS = (8, 32, 128)
NAT = ROOT / ".rowcache_terminal_copy_induction_v2/fit_natural.pt"
NDOCS = 8 if SMOKE else 32
CHUNK = 8
BARS = {"beat_energy_nats": 0.20, "shallow": 0.50, "mean_gap": 0.30, "head_energy": 0.40,
        "ce_tol": 0.01}
NULLS = {"beat_energy_le": 0.0, "shallow_le": 0.20, "mean_gap_le": 0.0, "head_energy_ge": 0.70}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def doc_forward(m, idx, mode=None, basis=None, mean=None, collect=None):
    """mode: None native | 'zero' | 'heads' | 'rank' | 'mean'.  collect: list to receive attn5 writes."""
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0 = x; v1 = None
    stats = {}
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if site == LAYER:
            cap = {}
            h = blk.attn.c_proj.register_forward_pre_hook(lambda _m, a: cap.__setitem__("y", a[0]))
            try:
                write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
            finally:
                h.remove()
            ycat = cap["y"]
            if collect is not None:
                collect.append(write.reshape(-1, D).float().cpu())
            if mode == "zero":
                write = torch.zeros_like(write)
            elif mode == "heads":
                mask = torch.zeros_like(ycat)
                for hd in KEEP_HEADS:
                    mask[..., hd * HD:(hd + 1) * HD] = ycat[..., hd * HD:(hd + 1) * HD]
                kept = blk.attn.c_proj(mask)
                stats["kept_energy_frac"] = float((kept.float() ** 2).sum() / (write.float() ** 2).sum())
                write = kept
            elif mode == "rank":
                w = write.float()
                write = ((w @ basis) @ basis.T).to(write.dtype)
            elif mode == "mean":
                write = mean.to(write.dtype).expand_as(write)
        else:
            write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        x = x + write
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0), stats


def main():
    t0 = time.time()
    check_hashes()
    m = R.load_model().to(DEV).eval()
    nat = torch.load(NAT, map_location="cpu")
    nat = (nat["rows"] if isinstance(nat, dict) else nat).long()[:NDOCS]
    fwd = 0

    def ce_of(**kw):
        nonlocal fwd
        s, n, st = 0.0, 0, {}
        for i in range(0, nat.shape[0], CHUNK):
            idx = nat[i:i + CHUNK, :T - 1].to(DEV)
            tgt = nat[i:i + CHUNK, 1:T].to(DEV)
            lg, stats = doc_forward(m, idx, **kw); fwd += 1
            s += float(F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="sum"))
            n += tgt.numel()
            st.update(stats)
        return s / n, st

    ce_nat, _ = ce_of()
    idx0 = nat[:CHUNK, :T - 1].to(DEV); tgt0 = nat[:CHUNK, 1:T].to(DEV)
    ce_module = float(m(idx0.contiguous(), tgt0.contiguous()))

    # energy basis for attention 5's write, fitted on the same documents (declared fitted parameters)
    coll = []
    _c, _s = ce_of(collect=coll)
    Wm = torch.cat(coll, 0).to(DEV)
    mean_write = Wm.mean(0, keepdim=True)
    Vh = torch.linalg.svd(Wm - 0.0, full_matrices=False)[2]
    sv = torch.linalg.svdvals(Wm)
    energy = (sv ** 2 / (sv ** 2).sum()).cpu().numpy()
    fitted = sum(k * D for k in RANKS)

    arms = {}
    arms["ZERO"], _ = ce_of(mode="zero")
    arms["HEADS_57"], hstats = ce_of(mode="heads")
    arms["MEAN"], _ = ce_of(mode="mean", mean=mean_write)
    for k in RANKS:
        arms[f"RANK_{k}"], _ = ce_of(mode="rank", basis=Vh[:k].T.contiguous())
    d = {k: v - ce_nat for k, v in arms.items()}

    beat = d["RANK_128"] - d["HEADS_57"]
    shallow = d["RANK_128"] / max(d["ZERO"], 1e-9)
    mean_gap = d["MEAN"] - d["HEADS_57"]
    head_energy = hstats.get("kept_energy_frac", float("nan"))
    preds = {
        'pred_a_causal_surrogate_beats_energy_surrogate': bool(beat >= BARS["beat_energy_nats"]),
        'pred_b_energy_rank_curve_is_shallow': bool(shallow >= BARS["shallow"]),
        'pred_c_mean_write_is_not_enough': bool(mean_gap >= BARS["mean_gap"]),
        'pred_d_two_heads_are_a_small_share_of_energy': bool(head_energy <= BARS["head_energy"]),
        'pred_e_instrument_reproduces_native_ce': bool(abs(ce_nat - ce_module) <= BARS["ce_tol"]),
    }
    nulls = {
        "a_null_energy_wins": bool(beat <= NULLS["beat_energy_le"]),
        "b_null_rank_recovers": bool(shallow <= NULLS["shallow_le"]),
        "c_null_mean_is_enough": bool(mean_gap <= NULLS["mean_gap_le"]),
        "d_null_heads_are_most_energy": bool(head_energy >= NULLS["head_energy_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "layer": LAYER, "keep_heads": list(KEEP_HEADS), "ranks": list(RANKS),
              "ce_native_nats": ce_nat, "ce_module_check": ce_module, "n_docs": int(nat.shape[0]),
              "summary": {"ce_damage_nats": d, "beat_energy_nats": beat,
                          "rank128_over_zero": shallow, "mean_minus_heads": mean_gap,
                          "heads57_energy_fraction": head_energy,
                          "write_singular_energy_top8": [float(x) for x in energy[:8]]},
              "smoke": SMOKE,
              "price": {"gpu_forwards": fwd, "backwards": 0, "fitted_parameters": fitted,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1200])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: attention {LAYER} surrogates (heads {KEEP_HEADS}, ranks {RANKS}, mean, zero) "
              f"on {NDOCS} natural documents; no model loaded")
        sys.exit(0)
    main()
