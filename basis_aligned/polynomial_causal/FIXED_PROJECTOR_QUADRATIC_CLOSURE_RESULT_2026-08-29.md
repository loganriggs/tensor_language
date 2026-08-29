# Fixed-projector quadratic closure pilot — result

**Run:** 2026-08-29 07:56 UTC  
**Preregistered source commit:** `58fc0787`  
**Status:** completed negative CPU weights-only pilot

## What was tested

For each of native MLP1 and MLP2, the test asked whether a previously frozen
activation-derived subspace $P$ and its orthogonal complement $Q=I-P$ define nearly
independent quadratic programs. For the symmetric bilinear polarization $B$ of the MLP,
the proposed direct sum was

$$
B_P(x,y)=P B(Px,Py)+Q B(Qx,Qy).
$$

The reported leakage is the squared Frobenius error of this decomposition divided by
the full quadratic tensor energy, estimated by seeded Gaussian contractions. Lower is
better. A matched Haar projector is a random subspace of the same dimension; beating
it is essential because any low-rank split removes some tensor blocks mechanically.

MLP1 used the frozen rank-64 MLP0 correction basis. MLP2 used the frozen rank-64 MLP1
correction basis. Both were also tested with the rank-128 QR union of those bases.

## Preregistered 64-sample outcome

Each entry shows the two independent Gaussian seeds. The Haar column spans two matched
Haar projectors and both Gaussian seeds.

| Layer/projector | Frozen-basis leakage | Matched-Haar leakage | Decision |
|---|---:|---:|---|
| MLP1, upstream rank 64 | 0.18186 / 0.18657 | 0.15620--0.15873 | fail: worse than random |
| MLP1, union rank 128 | 0.34849 / 0.35762 | 0.29291--0.29853 | fail both gates |
| MLP2, upstream rank 64 | 0.19045 / 0.18937 | 0.15533--0.15816 | fail: worse than random |
| MLP2, union rank 128 | 0.33030 / 0.33080 | 0.29122--0.29956 | fail both gates |

The rank-64 arms satisfy the absolute `<=0.25` condition but fail the required
two-times-Haar advantage in the opposite direction: leakage is about 16--21% larger
than random. Both rank-128 unions exceed `0.25` and are also worse than random.

## Doubled-sample robustness check

After the preregistered result was known, the same computation was rerun with 128
samples. This is labeled confirmatory and does not redefine the gates.

| Layer/projector | Frozen-basis leakage | Matched-Haar leakage |
|---|---:|---:|
| MLP1, upstream rank 64 | 0.18327 / 0.18636 | 0.15588--0.15779 |
| MLP1, union rank 128 | 0.35081 / 0.35375 | 0.29401--0.29749 |
| MLP2, upstream rank 64 | 0.18993 / 0.18972 | 0.15632--0.15724 |
| MLP2, union rank 128 | 0.33026 / 0.32933 | 0.29326--0.29782 |

Every qualitative and quantitative conclusion survives doubling. Seed variation is
small relative to the candidate-versus-Haar gap.

## Interpretation

The v3 rank-64 correction bases were useful activation/output coordinates in their
original experiment, but they are **not** boundaries between approximately independent
quadratic subcircuits of the next MLP. Adding both bases makes the algebraic separation
worse, not better. This is evidence against using those bases for an HOSVD/direct-sum
canonicalization or assuming that their coordinates can be edited independently with
low collateral.

This does not say MLP1/MLP2 lack useful hierarchical or sparse structure. It says the
projector must be selected using the downstream causal interface—or a joint
reachable/observable objective—rather than inherited from variance/PCA correction
energy. The next candidate remains causal port balancing, whose basis is explicitly
chosen for directions that are both physically reachable and downstream observable.

Four known-answer tests passed before real weights were inspected: exact direct sum,
pure cross-block failure, deterministic orthonormal controls, and reciprocal gate-gauge
invariance.

