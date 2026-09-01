# Rung 403 preregistration — exact branch errors of the rank-448 MLP0 program

Date: 2026-09-01 15:28 UTC  
Convention: cross-entropy (CE) added above the real model is damage; lower is better.

## Decision

Rung 401 split native MLP0 exactly into a fixed-mean-gain token main effect `T`, context main effect `C`, centered
token-by-context interaction `I`, normalization-gain modulation `S`, and an always-retained constant/vector-residual
term. Rung 402 found that `I` has one dominant attention head but a stable distributed tail. Separately, rung 328's
rank-448 shared-input MLP0 program was validated behaviorally and later passed signed and composition gates before
being dominated on the global frontier.

The missing bridge is to locate the rank-448 program's error inside the exact `T/C/I/S` grammar. If token main
effect error dominates, the next program needs an exact or higher-rank token-private path. If interaction error
dominates, a projection trained to preserve the distributed token-by-context metric is justified. Context-main,
gain, or auxiliary dominance imply different objects. This is a diagnostic of a fixed program, not a rank sweep.

## Fixed compact program

Reconstruct the identical rung-328 rank-448 program using:

- `.rowcache/fineweb_n192_skip11000.pt`, documents `[0,24)`, positions `0:256`;
- the frozen contextual covariance routine;
- the frozen reduced-rank regression routine at rank 448;
- the native MLP0 `Down` map and bias.

For an input `z` in 1,152 dimensions, the compact computation is

`c = Encoder z`,

`P(z,z) = Down((Left_small c) * (Right_small c)) + bias`,

where `c` has 448 coordinates, each small side has 4,608 outputs, and `*` is coordinatewise multiplication. The
literal MLP0 price is

`1152*448 + 2*4608*448 + 4608*1152 + 1152 = 9,954,432 values`,

versus 15,926,400 native values, a saving of 5,971,968 values. The audit must reproduce rank 448, every tensor
shape, the fit slice, and rung328's covariance retained-energy diagnostic `0.9011108875` within `2e-6`.

## Exact five-factor physical assay

Build the same product-reference moments for the native bilinear map `N` and compact map `P`. For each map `q`,
write its bias-free output exactly as

`F_q = K_q + T_q + C_q + I_q + S_q + R_q`,

where `K` is the fixed reference constant and `R` is the explicit vector normalization residual from rung401.
Define the four named changes `delta_b = b_P - b_N` for `b` in `{T,C,I,S}`. Define one auxiliary closing change

`delta_A = deployed_P - deployed_N - delta_T - delta_C - delta_I - delta_S`.

`A` therefore contains the changed fixed constant, changed explicit normalization residual, and the exact BF16
rounding closure. It prevents those quantities from being silently assigned to one named branch.

For every subset `U` of `{T,C,I,S,A}`, physically return

`BF16(deployed_N + sum(delta_b for b in U))`.

This gives 32 arms on the unchanged rung401 FIT and disjoint SELECT documents. `EMPTY` must be the exact native
endpoint and `T+C+I+S+A` the exact compact endpoint. Report pooled document CE, five-factor Shapley CE damage,
all Möbius interaction terms, direct output-error energies for every factor, and FIT/SELECT rank transport.

## Frozen predictions

### A — exact instrument and fixed-program identity

- Native and compact analytical identities have relative mean-squared error at most `1e-8` on FIT and SELECT.
- EMPTY and FULL reproduce direct native and direct rank-448 BF16 states exactly; their pooled CE differences from
  direct endpoints are at most `1e-6`.
- Every arm has the expected live call census.
- Rank, shapes, fit population, price, and retained-energy diagnostic match the fixed program above.

### B — the known compact program remains useful on these documents

- Rank-448 total CE damage is nonnegative and at most `.030 nat` on SELECT.
- FIT and SELECT total-damage signs agree and differ in magnitude by at most `.015 nat`.

### C — token-private error is the leading named obstruction

- `T` is the largest positive named Shapley damage on both FIT and SELECT.
- SELECT `T` damage is at least `.002 nat`.
- Spearman correlation of the four named branch-damage rankings between FIT and SELECT is at least `.75`.

This is the prospective prediction from rungs394–399: token identity is preserved, but its quadratic/private
correction is broad, while response-aware low-rank rotations did not improve it.

### D — the named grammar, rather than numerical closure, localizes the error

- The largest positive named branch supplies at least 40% of the positive `{T,C,I,S}` Shapley total on both roles.
- Absolute auxiliary Shapley damage is at most `.005 nat` on both roles.

## Strong null and decisions

The strong null fires if A fails, SELECT total damage has magnitude below `.001 nat`, named FIT/SELECT Spearman is
nonpositive, or absolute auxiliary Shapley damage is at least the sum of absolute named-branch Shapley damages on
either role.

- A/B/C/D with null false routes to an exact-token or token-private bypass before another shared input rank.
- A/B/D with C failed but stable `I` dominance routes to a distributed-interaction-weighted projection.
- Stable `C` or `S` dominance routes to context-only or normalization-aware preservation respectively.
- Auxiliary dominance means the four named grammar does not localize compact-program error; inspect its constant,
  vector-normalization residual, and BF16 closure separately.
- Instability or a null closes branch-targeted p448 work. Do not tune rank, bars, fit rows, or branch definitions.

No outcome of this diagnostic itself licenses a new compressor, adoption point, or source-position expansion.
