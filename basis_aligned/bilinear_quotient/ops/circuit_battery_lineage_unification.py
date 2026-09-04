#!/usr/bin/env python
"""circuit_battery_lineage_unification -- are the successor computation and the roundness switch ONE mechanism?

Two lineages met at SS2847. The roundness switch's deciding readers are mlp8, mlp9 (+mlp1) (SS2847); the successor
computation's readers are mlp8 > mlp9 > mlp10 > mlp11 as a 2-of-4 redundant threshold with specificity rising in depth
(SS2818, SS2819, SS2821). Attention 8 now carries TWO features -- which item was last (SS2808/R576, SS2820) and whether it is
round (SS2842-SS2844) -- and the same MLP stack performs TWO computations, +1 and +step. Nobody has tested whether that is one
function of two inputs or two functions sharing hardware, and the difference matters for how small a compiled program can be.

Two measurements, on the same reader set:
  PROFILE   -- each reader's contribution to the roundness decision and to the successor margin, correlated across tasks.
  CHANNEL   -- the discriminating arm: remove SS2844's roundness direction from attention 8's write and ask whether the
               SUCCESSOR task is damaged. Roundness is irrelevant to a numbered list, so damage there means the two
               computations share a channel; no damage means attention 8 writes two separable features into one stream.

# BQGATE: EXPERIMENT  pred_a_reader_profiles_correlate pred_b_roundness_direction_is_separable
#                     pred_c_the_top_reader_is_shared pred_d_random_direction_is_inert
#                     pred_e_full_removal_reproduces_the_battery

SIGN CONVENTION: successor damage d_m = m_NATIVE - m_arm in margin units, POSITIVE = the arm HURTS the successor answer;
roundness contribution is the SS2842 logit-difference recovery, HIGHER = the reader carries more of the switch. No CE and no
SS312 L2; nothing installs into the frontier.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_LINEAGE_UNIFICATION_PREREGISTRATION.md
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_lineage_unification.py")
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
PREREG = R.POLY / "CIRCUIT_BATTERY_LINEAGE_UNIFICATION_PREREGISTRATION.md"
LADDER = ROOT / "circuit_battery_roundness_decision_ladder_results.json"
BANK21 = ROOT / "circuit_battery_v2_bank21_results.json"
RUNG = "circuit_battery_lineage_unification"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "4845dc73f72e420c04e7c127c10b1db46f4bce4ba63959fc67b09b80e666674c",
          LADDER: "7d6f806ddb5258518f9893cebbd2aa8b8d35668f6e1a34e97e7a26cbd0585fe7",
          BANK21: "7c1db82061bb6cac010fa7d2141cfc8e2faa4330759bdfe5aed299bab9a94d50",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, HD, NLAY = R.D, R.HD, R.NL
LAYER, HEAD = 8, 3
READERS = (("mlp", 8), ("mlp", 9), ("mlp", 10), ("mlp", 11), ("mlp", 1))
SUCCESSOR_TASK = "numbered_list.index_successor"
SEED = 2847
PER_CELL = 4 if SMOKE else 24   # the battery used 24; pred_e compares to its number, so the OOD row set must MATCH
ENC = BANK.ENC
BARS = {"rho": 0.50, "separable": 0.10, "random": 0.05, "repro": 0.30, "floor": 0.5}
NULLS = {"rho_le": 0.0, "separable_ge": 0.40, "random_ge": 0.30}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def run(m, tokens, finals, *, patch=None, drop_dir=None, remove=()):
    """patch: {(kind,layer): tensor} donor swaps. drop_dir: remove this direction from head HEAD's slice.
    remove: components whose whole output is zeroed (successor-damage arms)."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    cap = {}
    sl = slice(HEAD * HD, (HEAD + 1) * HD)

    def pre(_mod, args):
        y = args[0]
        cap["y"] = y
        if drop_dir is None:
            return None
        y2 = y.clone()
        seg = y2[..., sl].float()
        y2[..., sl] = (seg - (seg * drop_dir).sum(-1, keepdim=True) * drop_dir).to(y.dtype)
        return (y2,)

    hk = m.transformer.h[LAYER].attn.c_proj.register_forward_pre_hook(pre)
    try:
        for site, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
            if patch is not None and ("attn", site) in patch:
                write = patch[("attn", site)]
            if patch is None:
                cap[("attn", site)] = write
            if ("attn", site) in remove:
                write = torch.zeros_like(write)
            x = x + write
            out = blk.mlp(F.rms_norm(x, (D,)))
            if patch is not None and ("mlp", site) in patch:
                out = patch[("mlp", site)]
            if patch is None:
                cap[("mlp", site)] = out
            if ("mlp", site) in remove:
                out = torch.zeros_like(out)
            x = x + out
    finally:
        hk.remove()
    logits = (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float()
    return logits, cap


def spearman(a, b):
    rk = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    ra, rb = rk(a), rk(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) and np.std(rb) else float("nan")


def main():
    t0 = time.time()
    check_hashes()
    m = fastload.load_model_fast().to(DEV).eval()
    g = torch.Generator(device="cpu").manual_seed(SEED)
    fwd = [0]

    # ---- fit the roundness direction on the percent pairs' fit half (as SS2844/SS2846 did) ----
    rows = RL.build_pairs("percent")
    half = max(1, len(rows) // 2)
    fit_rows, ev_rows = rows[:half], rows[half:] or rows[:half]
    sl = slice(HEAD * HD, (HEAD + 1) * HD)
    deltas = []
    for b in CB.batches(fit_rows):
        bid, bfin, _ = CB.pack(b, "base")
        did, dfin, _ = CB.pack(b, "donor")
        _l, dcap = run(m, did, dfin); fwd[0] += 1
        _l, bcap = run(m, bid, bfin); fwd[0] += 1
        dy, by = dcap["y"], bcap["y"]        # run() returns the cache dict; "y" is c_proj's input
        ar = torch.arange(len(b), device=DEV)
        deltas.append((by[ar, bfin][:, sl] - dy[ar, dfin][:, sl]).float())
    Dm = torch.cat(deltas, 0)
    u = Dm.mean(0); u = u / u.norm().clamp_min(1e-9)
    rv = torch.randn(HD, generator=g).to(DEV); rv = rv / rv.norm()

    # ---- PROFILE 1: each reader's contribution to the ROUNDNESS decision (SS2842 recovery) ----
    round_rec = {}
    per = defaultdict(list)
    for b in CB.batches(ev_rows):
        bid, bfin, bans = CB.pack(b, "base")
        did, dfin, dans = CB.pack(b, "donor")
        ldon, dcache = run(m, did, dfin); fwd[0] += 1
        lb, _ = run(m, bid, bfin); fwd[0] += 1
        base_ld = (lb.gather(1, dans[:, None]) - lb.gather(1, bans[:, None])).squeeze(1)
        don_ld = (ldon.gather(1, dans[:, None]) - ldon.gather(1, bans[:, None])).squeeze(1)
        den = (don_ld - base_ld).clamp_min(1e-3)
        for comp in READERS:
            lp, _ = run(m, bid, bfin, patch={comp: dcache[comp]}); fwd[0] += 1
            ld = (lp.gather(1, dans[:, None]) - lp.gather(1, bans[:, None])).squeeze(1)
            per[f"{comp[0]}{comp[1]}"].append(((ld - base_ld) / den).cpu().numpy())
    round_rec = {k: float(np.concatenate(v).mean()) for k, v in per.items()}

    # ---- PROFILE 2: each reader's SUCCESSOR damage on the bank's frozen OOD rows ----
    srows = [r for r in BANK.build_rows(SUCCESSOR_TASK, per_cell=PER_CELL)
             if r["family"] == "A1" and r["split"] == "OOD"]
    cand = torch.tensor(sorted({ENC.encode(s)[0] for s in BANK.candidate_strings(SUCCESSOR_TASK)}), device=DEV)
    succ = defaultdict(list)
    drops = defaultdict(list)
    for b in CB.batches(srows):
        ids, fin, ans = CB.pack(b, "base")
        lg, _ = run(m, ids, fin); fwd[0] += 1
        mn = CB.margins(lg, ans, cand)
        for comp in READERS:
            lg2, _ = run(m, ids, fin, remove=(comp,)); fwd[0] += 1
            succ[f"{comp[0]}{comp[1]}"].append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
        lgF, _ = run(m, ids, fin, remove=tuple(READERS)); fwd[0] += 1
        drops["FULL_READERS"].append((mn - CB.margins(lgF, ans, cand)).cpu().numpy())
        # the discriminating arm and its control
        lgD, _ = run(m, ids, fin, drop_dir=u); fwd[0] += 1
        drops["DROP_ROUNDNESS_DIR"].append((mn - CB.margins(lgD, ans, cand)).cpu().numpy())
        lgR, _ = run(m, ids, fin, drop_dir=rv); fwd[0] += 1
        drops["DROP_RANDOM_DIR"].append((mn - CB.margins(lgR, ans, cand)).cpu().numpy())
        # pred_e is an instrument check against SS2840's FULL value, so it must use SS2840's ARM: the
        # battery removes the writer's FINAL-POSITION write from every reader edge (CB.run ablate=True,
        # verified equal to the all-edges removal by SS2817's pred_a), NOT a whole-component zeroing at
        # every position. My first implementation zeroed all positions and compared to the battery's
        # number -- a mismatched-arm comparison, the same family as SS2832's and SS2843's. Corrected here
        # before the registered run and disclosed in the ledger.
        lgA = CB.run(m, ids, fin, writer=("attn", LAYER), ablate=True); fwd[0] += 1
        drops["REMOVE_ATTN8"].append((mn - CB.margins(lgA, ans, cand)).cpu().numpy())
    succ_dmg = {k: float(np.concatenate(v).mean()) for k, v in succ.items()}
    arms = {k: float(np.concatenate(v).mean()) for k, v in drops.items()}

    names = sorted(round_rec)
    rho = spearman([round_rec[n] for n in names], [succ_dmg[n] for n in names])
    full = max(arms["FULL_READERS"], BARS["floor"])
    sep = arms["DROP_ROUNDNESS_DIR"] / full
    rnd = arms["DROP_RANDOM_DIR"] / full
    top_round = max(names, key=lambda n: round_rec[n])
    top_succ = max(names, key=lambda n: succ_dmg[n])
    b21 = json.load(open(BANK21))["tasks"][SUCCESSOR_TASK]
    repro = abs(arms["REMOVE_ATTN8"] - b21["split_d_m"]["FULL"]) / max(b21["split_d_m"]["FULL"], BARS["floor"])

    preds = {
        'pred_a_reader_profiles_correlate': bool(rho >= BARS["rho"]),
        'pred_b_roundness_direction_is_separable': bool(sep <= BARS["separable"]),
        'pred_c_the_top_reader_is_shared': bool(top_round == top_succ),
        'pred_d_random_direction_is_inert': bool(abs(rnd) <= BARS["random"]),
        'pred_e_full_removal_reproduces_the_battery': bool(repro <= BARS["repro"]),
    }
    nulls = {
        "a_null_no_correlation": bool(rho <= NULLS["rho_le"]),
        "b_null_shared_channel": bool(sep >= NULLS["separable_ge"]),
        "d_null_random_hurts": bool(abs(rnd) >= NULLS["random_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "layer": LAYER, "head": HEAD, "readers": [f"{k}{l}" for k, l in READERS],
              "successor_task": SUCCESSOR_TASK, "seed": SEED,
              "summary": {"rho_roundness_vs_successor": rho,
                          "roundness_recovery": round_rec, "successor_damage": succ_dmg,
                          "top_reader_roundness": top_round, "top_reader_successor": top_succ,
                          "arms": arms, "separability_fraction": sep,
                          "random_fraction": rnd, "battery_reference_full": b21["split_d_m"]["FULL"],
                          "repro_gap_fraction": repro},
              "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": HD,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1400])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: reader profiles for the roundness decision and the successor margin over "
              f"{[f'{k}{l}' for k, l in READERS]}, plus the roundness-direction removal arm; no model loaded")
        sys.exit(0)
    main()
