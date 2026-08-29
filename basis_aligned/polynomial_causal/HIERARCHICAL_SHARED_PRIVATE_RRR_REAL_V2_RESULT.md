# Hierarchical shared/private RRR v2 — result and interpretation

**Completed:** 2026-08-29 07:17 UTC  
**Scope:** repeatedly exposed discovery roles only; no validation, final,
generalization, semantic-coordinate, extraction, or removal authority.

## Short answer

At the three tested rank-512-scale storage budgets, a 128-dimensional shared output
trunk plus site-private residual directions does **not** beat the all-private
exact-price endpoint.  It consistently sits between the all-private and all-shared
programs.  The registered primary, typed, and large-budget predictions all fail while
every integrity control passes.

This is a clean negative result for this specific hierarchy and budget scale.  It says
private site specialization is worth more than amortizing a 128-direction global
dictionary here.  It does not contradict the earlier positive low-rank result: flat
sharing helped at the much tighter global rank-64/rank-128 prices, whereas this run
spent as much map storage as global rank 512 or more.

## Computation in plain language

For each of the 36 component sites, the fit constructs an output-merit matrix

\[
M_j=C_j^\top(G+\lambda I)^{-1}C_j.
\]

A direction with a large eigenvalue in \(M_j\) is an output direction that helps
predict site \(j\)'s residual write from the token embedding.  The hierarchy first
takes \(q_0\) directions from the summed merit \(\sum_j M_j\).  Those directions are
stored once as a shared output dictionary.  It then projects them out and gives each
site private directions from

\[
(I-P)M_j(I-P), \qquad P=V_0V_0^\top.
\]

At fixed storage, adding one shared direction is expensive: it needs one output vector
plus 36 site input vectors.  Its cost equals 18.5 private rank slots, since a private
slot needs one input and one output vector at only one site.  Thus moving from
\(q_0=0\) to \(q_0=128\) gives up exactly 2,368 private slots.  The experiment asks
whether reuse of those 128 directions compensates for that lost specialization in
whole-program next-token CE.

## Exact CE results

Lower CE is better.  Values are nats per scored token.

| Map budget | Arm | Shared rank | Total private slots | skip7000 | skip11000 | skip1200 |
|---|---|---:|---:|---:|---:|---:|
| 21,823,488 | all-private exact-price | 0 | 9,472 | 5.977554 | 5.948485 | 5.968654 |
| 21,823,488 | hierarchical | 128 | 7,104 | 5.984077 | 5.955484 | 5.972883 |
| 21,823,488 | all-shared | 512 | 0 | 5.989667 | 5.962682 | 5.982282 |
| 22,413,312 | all-private exact-price | 0 | 9,728 | 5.976827 | 5.947498 | 5.967798 |
| 22,413,312 | hierarchical | 128 | 7,360 | 5.982518 | 5.953620 | 5.971534 |
| 42,467,328 | all-private exact-price | 0 | 18,432 | 5.966009 | 5.935591 | 5.959479 |
| 42,467,328 | hierarchical | 128 | 16,064 | 5.966630 | 5.936274 | 5.960103 |

At the global-rank-512 price, the hierarchy is better than all-shared by
0.00559/0.00720/0.00940 nat, but worse than all-private by
0.00652/0.00700/0.00423 nat.  At the typed price it is worse than all-private by
0.00569/0.00612/0.00374 nat.  At the large independent-rank-512 price it is worse by
0.00062/0.00068/0.00062 nat.

The fit-optimal all-private allocation at the largest price is nearly tied with the
parent uniform rank-512 allocation: it is worse by only
0.000200/0.000236/0.000014 nat.  That difference is descriptive, not an identity
control, because one allocation is nonuniform and the parent is exactly 512 per site.

## What the rank allocations show

The fit allocates private capacity very unevenly.

| Budget/arm | Minimum private rank | Middle two ranks | Maximum private rank |
|---|---:|---:|---:|
| global price, all-private | 0 | 106, 135 | 1,076 |
| global price, shared-128 | 0 | 48, 68 | 978 |
| large price, all-private | 34 | 409, 456 | 1,128 |
| large price, shared-128 | 18 | 357, 370 | 1,024 |

This is evidence against a uniform site grammar.  At compressed prices some sites are
assigned no private output direction while others receive almost the full 1,152
dimensions.  The allocation is based only on fit merit, not held-out CE.

All registered shared, private, and allocation boundary eigengaps are strictly
positive.  Therefore the fitted projectors are conditionally identified for these fit
matrices.  This does not make individual basis columns semantic: signs and within-tied
subspace rotations remain gauge choices, and discovery-only stability is not OOD
stability.

## Controls and artifact identity

- Covered-token CE spread across all seven arms is exactly zero on each role.
- Literal all-private and all-shared factor endpoints pass.
- All three parent CE replay controls pass within the frozen 0.002-nat tolerance.
- Integrity conjunction is true; no CE predicate is allowed to promote without it.
- Runtime: 421.75 seconds.
- Peak allocated CUDA: 4,217,080,320 bytes.
- Authority file SHA256:
  `a591dd6b50c9aa4aa44546232c61748976cdc8790ebac136f64fd7949dea0379`.
- Result file SHA256:
  `86fcfc4b9d1032801a1192ab5a0a07d0878c01f7a63fa220ce4ef189d6835a38`.
- Receipt file SHA256:
  `de90b352855ca5126a35e2bd05fd88cef632bdef2902a2b5c7bca331ffc0d92f`.
- Internal receipt SHA256:
  `6fa7c20f84c67124011c9bb2c5d4cb792cc01aabc755d0694b7391ec6a8eea12`.

The earlier v1 result is deliberately non-creditable: it failed JSON replay before
receipt.  V2 reran every fit and evaluation under a one-change container recovery.

## Consequence for the strategy

The result prunes a tempting but too-simple claim: a single 128-direction trunk is not
the right way to simplify all 36 maps at rank-512-scale prices.  The useful structure
is currently narrower:

1. repeated directions help under severe storage pressure;
2. site-private directions dominate once the map budget is large;
3. attention/MLP typing and a flat global-plus-private hierarchy do not supply the
   missing semantic decomposition.

The one nonredundant follow-up is a **tight-budget** hierarchy, because the earlier
flat shared maps actually beat exact-price private maps at global rank-64/rank-128
prices.  Interior shared ranks such as 16/32/48 at the rank-64 price and 32/64/96 at
the rank-128 price would test whether a small shared trunk plus private suffix beats
both endpoints in the regime where reuse is already known to pay.  This should be a
bounded diagnostic, not an indefinite hierarchy search.

For whole-model reverse engineering, the next higher-return branches are the
fresh-document native-Down behavioral-port test and one behavior-anchored terminal
circuit.  Both test causal transfer/editability rather than spending another interval
only refining context-free map CE.

