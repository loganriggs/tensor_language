"""COMMUTANT NULL CALIBRATION (Claude instrument lane) -- measured bars for the
gauge-aware commutant screen (Codex rung 479 registered 07:20).

Rung 479 will ask whether a task-reader Q(g) family in a fixed 32-dim
equality-state subspace has a nontrivial approximate commutant, with blocks
stable across views and better than independently conjugated controls.  Its
bars should cite MEASURED null levels, not guessed ones (the 428/429/432
window lesson).  This rung measures, at exactly n=32 and CPU-only, with NO
model access and NO unopened objects:

  (1) the false-positive rate of the commutant probe (ops/commutant.py) on
      GENERIC symmetric families -- arm G(k), k in {4, 8, 16} matrices,
      8 seeds each: fraction with commutant_dim > 1 at tol 1e-8;
  (2) the INTERPOLATION FLOOR: even when commutant_dim == 1, one can force a
      best-gap 2-split from the lowest non-identity mode of the commutator
      operator L = sum_i K_i^2, K_i = kron(I,A_i) - kron(A_i,I), and measure
      its off-block mass.  The generic distribution of this forced off-mass
      (report q05/q50/q95) is the noise-geometry analog of rung 478's
      alignment-destroyed q95 = .339 -- the level a 479 survivor must beat;
  (3) the DETECTION CURVE: planted two-block families (sizes 20+12, common
      random orthogonal conjugation, k=8) plus eps * GOE noise,
      eps in {0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1}, 8 seeds: recovered block
      sizes and off-block mass per eps;
  (4) the CONTROL ARM C: the same planted construction but each matrix
      INDEPENDENTLY conjugated (destroys the common basis, preserves each
      spectrum) -- the direct analog of 479's registered control; 8 seeds.

Arms (named): G(4), G(8), G(16); P(0), P(1e-4), P(1e-3), P(1e-2), P(3e-2),
P(1e-1); C.  All matrices are unit-Frobenius-normalized 32x32 symmetric.

Frozen predictions
------------------
pred_a (instrument exactness): every P(0) seed recovers block sizes [20, 12]
    with off-block mass <= 1e-9, and lambda2_rel (second-smallest L
    eigenvalue over the mean L eigenvalue, identity mode deflated) is at
    least 100x the corresponding generic-median lambda2_rel... NO -- stated
    as frozen literal: every P(0) seed has commutant_dim >= 2.
pred_b (null cleanliness): commutant_dim == 1 in >= 7/8 seeds for EVERY
    generic arm G(4)/G(8)/G(16) AND >= 7/8 seeds of control arm C.
pred_c (detection monotonicity): median forced/recovered off-block mass is
    non-decreasing in eps across the P grid, and every P(eps<=1e-3) seed
    recovers sizes [20, 12] with off-block mass <= 3e-3.

Null: >= 2/8 generic seeds in any G arm show commutant_dim > 1, or any P(0)
seed fails exact recovery -- then the probe is NOT calibrated at n=32 and
rung 479 must not cite it for bars.

Price: CPU only, zero model forwards/backwards, zero deployed parameters,
target < 6 min; writes commutant_null_calibration_results.json (no tokens,
logits, hidden states, or validation/SEALED objects touched).
"""
# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
OUT = BQ / "commutant_null_calibration_results.json"
COMMUTANT = OPS / "commutant.py"
N = 32
BLOCKS = (20, 12)
K_GENERIC = (4, 8, 16)
K_PLANTED = 8
EPS_GRID = (0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
SEEDS_PER_ARM = 8
TOL = 1e-8


def _sym(M):
    return 0.5 * (M + M.T)


def _unit(A):
    return A / max(np.linalg.norm(A), 1e-30)


def _generic_family(rng, k):
    return [_unit(_sym(rng.standard_normal((N, N)))) for _ in range(k)]


def _planted_family(rng, eps, independent_conjugation=False):
    common = np.linalg.qr(rng.standard_normal((N, N)))[0]
    fam = []
    for _ in range(K_PLANTED):
        B = np.zeros((N, N))
        a, b = BLOCKS
        B[:a, :a] = _sym(rng.standard_normal((a, a)))
        B[a:, a:] = _sym(rng.standard_normal((b, b)))
        O = (np.linalg.qr(rng.standard_normal((N, N)))[0]
             if independent_conjugation else common)
        A = O @ B @ O.T + eps * _sym(rng.standard_normal((N, N)))
        fam.append(_unit(A))
    return fam


def _l_operator(fam):
    eye = np.eye(N)
    L = np.zeros((N * N, N * N))
    for A in fam:
        K = np.kron(eye, A) - np.kron(A, eye)
        L += K @ K
    return L


def _forced_split_offmass(fam, L):
    """Best-gap 2-split off-block mass from the lowest non-identity L mode."""
    w, V = np.linalg.eigh(L)
    ident = np.eye(N).reshape(-1)
    ident /= np.linalg.norm(ident)
    for j in range(min(8, len(w))):
        v = V[:, j]
        v = v - ident * float(ident @ v)
        C = _sym(v.reshape(N, N))
        if np.linalg.norm(C) > 1e-6:
            break
    else:
        return 1.0, float(w[1] / max(w.mean(), 1e-30))
    ew, B = np.linalg.eigh(C / np.linalg.norm(C))
    gaps = np.diff(ew)
    cut = int(np.argmax(gaps)) + 1
    mask = np.zeros((N, N), dtype=bool)
    mask[:cut, :cut] = True
    mask[cut:, cut:] = True
    off = max(np.linalg.norm(np.where(mask, 0.0, B.T @ A @ B))
              / np.linalg.norm(A) for A in fam)
    lam2_rel = float(w[1] / max(w.mean(), 1e-30))
    return float(off), lam2_rel


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert COMMUTANT.exists(), "ops/commutant.py must be on disk"
        assert sum(BLOCKS) == N and len(EPS_GRID) == 6 and SEEDS_PER_ARM == 8
        assert 0.0 in EPS_GRID and max(EPS_GRID) == 1e-1
        print("COMMUTANT NULL CALIBRATION | dry run: G(4/8/16), P(6 eps), C; "
              "n=32, 8 seeds/arm, CPU-only, no model")
        return

    started = time.time()
    sys.path.insert(0, str(OPS))
    from commutant import commutant_blocks

    result = {"rung": "commutant_null_calibration", "n": N,
              "seeds_per_arm": SEEDS_PER_ARM, "tol": TOL, "arms": {}}

    generic_offs = []
    for k in K_GENERIC:
        arm = {"false_positive_seeds": 0, "forced_offmass": [],
               "lambda2_rel": []}
        for s in range(SEEDS_PER_ARM):
            rng = np.random.default_rng(479_000 + 100 * k + s)
            fam = _generic_family(rng, k)
            r = commutant_blocks(fam, tol=TOL)
            if r["commutant_dim"] > 1:
                arm["false_positive_seeds"] += 1
            off, lam = _forced_split_offmass(fam, _l_operator(fam))
            arm["forced_offmass"].append(off)
            arm["lambda2_rel"].append(lam)
            generic_offs.append(off)
        result["arms"][f"G({k})"] = arm

    planted = {}
    for eps in EPS_GRID:
        arm = {"recovered_sizes": [], "offmass": [], "commutant_dim": []}
        for s in range(SEEDS_PER_ARM):
            rng = np.random.default_rng(479_500 + int(eps * 1e6) + s)
            fam = _planted_family(rng, eps)
            r = commutant_blocks(fam, tol=max(TOL, (10 * eps) ** 2))
            arm["recovered_sizes"].append(r["block_sizes"])
            arm["commutant_dim"].append(r["commutant_dim"])
            if r["commutant_dim"] > 1:
                arm["offmass"].append(r["off_block_mass"])
            else:
                off, _ = _forced_split_offmass(fam, _l_operator(fam))
                arm["offmass"].append(off)
        arm["median_offmass"] = float(np.median(arm["offmass"]))
        planted[f"P({eps:g})"] = arm
    result["arms"].update(planted)

    ctrl = {"false_positive_seeds": 0, "forced_offmass": []}
    for s in range(SEEDS_PER_ARM):
        rng = np.random.default_rng(479_900 + s)
        fam = _planted_family(rng, 0.0, independent_conjugation=True)
        r = commutant_blocks(fam, tol=TOL)
        if r["commutant_dim"] > 1:
            ctrl["false_positive_seeds"] += 1
        off, _ = _forced_split_offmass(fam, _l_operator(fam))
        ctrl["forced_offmass"].append(off)
    result["arms"]["C"] = ctrl

    # Deliverable: the generic interpolation floor for 479's bars.
    result["generic_forced_offmass_q05_q50_q95"] = [
        float(np.quantile(generic_offs, q)) for q in (.05, .50, .95)]

    p0 = planted["P(0)"]
    pred_a = all(sz == [20, 12] for sz in p0["recovered_sizes"]) and all(
        o <= 1e-9 for o in p0["offmass"]) and all(
        c >= 2 for c in p0["commutant_dim"])
    gen_ok = all(result["arms"][f"G({k})"]["false_positive_seeds"] <= 1
                 for k in K_GENERIC)
    pred_b = gen_ok and ctrl["false_positive_seeds"] <= 1
    medians = [planted[f"P({eps:g})"]["median_offmass"] for eps in EPS_GRID]
    small = all(
        sz == [20, 12] and o <= 3e-3
        for eps in EPS_GRID if eps <= 1e-3
        for sz, o in zip(planted[f"P({eps:g})"]["recovered_sizes"],
                         planted[f"P({eps:g})"]["offmass"]))
    pred_c = all(b >= a - 1e-12 for a, b in zip(medians, medians[1:])) and small
    null_fired = (not pred_a) or any(
        result["arms"][f"G({k})"]["false_positive_seeds"] >= 2
        for k in K_GENERIC)

    result.update({
        'pred_a_planted_exact_recovery': bool(pred_a),
        'pred_b_generic_and_control_clean': bool(pred_b),
        'pred_c_detection_monotone_and_small_eps_exact': bool(pred_c),
        'strong_null_probe_uncalibrated': bool(null_fired),
        "runtime_s": time.time() - started,
    })
    OUT.write_text(json.dumps(result, indent=1))
    print(f"pred_a={pred_a} pred_b={pred_b} pred_c={pred_c} "
          f"null={null_fired} generic_off_q={result['generic_forced_offmass_q05_q50_q95']} "
          f"({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
