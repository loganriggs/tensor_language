# Rung 466: task-shaped correction group versus broad suppressors

Status: prospective explanatory design, frozen after rung 465 and before opening any multi-site removal outcome. It
uses the already-open code role and five sites selected by predeclared properties of the rung-465 receipt. It is a
cross-module grouping and interaction test, not rank reduction or compression.

## Fixed groups and motivation

Rung 465 found that the causal-necessity ordering of all 19 later writes is almost identical for native and
transplanted equality sources (Spearman `.974`). It also separated two properties that should not be conflated:

- MLP17 is the largest necessary site under both sources, but its removal has a broad negative profile in every
  context and only `.48/.40` cosine with the full context correction.
- MLP8, MLP9, and MLP12 are the three largest sites that have the full correction's exact signed pattern under both
  sources and at least `.94` cosine with it.

Freeze the **task-shaped group** `T = {MLP8, MLP9, MLP12}`. Freeze the **broad-suppressor controls**
`G = {attention14, MLP17}`: they are the two largest all-negative necessity sites under both sources. No other site,
group, or threshold may be selected after this registration.

## Exact five-site removal factorial

On all 192 code documents and fixed halves `0:96` / `96:192`, cache absent (`0`), native-source (`N`), and hybrid-
source (`H`) writes exactly as in rung 465. For each source `s` in `{N,H}`, run all `2^5 = 32` subsets of the fixed
five sites. A site included in subset `A` is replaced by its same-document absent-trajectory write; every unlisted
site runs normally, and all later modules recompute after each patch.

For context cell `c`, define

`v_s(A,c) = E_s(full,c) - E_s(remove A,c)`.

Thus `v_s(A,c)` is the signed causal role of jointly removing subset `A`; `v_s(empty,c)=0`. Positive values mean
the removed set was useful, while negative values mean it normally suppresses an over-strong effect.

For every nonempty subset, compute its Möbius/Harsanyi interaction

`d_s(A,c) = sum_{B subseteq A} (-1)^(|A|-|B|) v_s(B,c)`.

This is the exact finite-difference interaction after all lower-order effects are subtracted. It prevents us from
pretending that the MLP/attention contributions add. Only dividends above the established order-dependent numerical
floor are interpreted; all raw subset effects remain reported.

The four-context vector order is `(near, far, one predecessor, multiple predecessors)`. The complete later-program
correction `K_s` is inherited from rung 465 as full source effect minus direct-only effect.

## Registered predictions

### A. Exact instrument

All parent/preregistration/model/row hashes hold; native replay error is at most `1e-12`; equality reconstruction
error is at most `1e-10`; empty-subset effects are exactly zero; singleton effects reproduce rung 465 within
`1e-10 nat`; all32 subsets execute once for each source with exact patch identities/census; direct/full effects
reproduce the parent; and SEALED remains closed.

### B. The task-shaped group jointly carries the context correction

For both sources, `v_s(T)` has the `near-/far+/one+/multiple-` sign pattern pooled and in both halves, has four-cell
norm at least `.04 nat`, and has cosine at least `.90` with `K_s` pooled and positive cosine in both halves. Native
and hybrid `v(T)` vectors have cosine at least `.90` pooled and positive in both halves, with larger/smaller norm
ratio at most `2.0`.

### C. The broad-suppressor group is a distinct causal role

For both sources, all four entries of `v_s(G)` are negative pooled and in both halves. Native and hybrid `v(G)`
vectors have cosine at least `.90` pooled and positive in both halves. Their cosine with `K_s` must be below `.70`
for both sources. This is the preregistered contrast that separates general suppression from context shaping.

### D. Task shaping and broad suppression interact

For both sources, the cross-group interaction vector

`I_s = v_s(T union G) - v_s(T) - v_s(G)`

must have norm at least `.01 nat`. Native and hybrid interaction vectors must have cosine at least `.80` pooled and
positive in both halves. This clause tests compositionality rather than adding marginal importances.

### E. The fixed five-site program explains most of the distributed correction

For both sources, `v_s(T union G)` must have cosine at least `.85` with `K_s`; its projection magnitude onto `K_s`,
`dot(v,K)/dot(K,K)`, must lie in `[.50, 1.50]`; and these cosines must be positive in both halves. The five-site
program's native/hybrid vectors must have cosine at least `.85` pooled. This is an extraction-style sufficiency test
for the fixed group, not a parameter-saving claim.

The strong null is an invalid instrument; nonpositive full-source all-positive benefit; `v(T)` norm below `.01 nat`
for both sources; native/hybrid `v(T)` cosine at most zero; or all five-site effects numerically inert.

## Decision and next split

- B/C pass: the downstream program separates into a task-shaped MLP8/9/12 group and a broad attention14/MLP17
  suppressor group across both matcher sources.
- D pass: their composition contains a stable interaction that must be retained as an edge in the circuit program;
  singleton importances cannot price or explain it.
- E pass: the five fixed sites are a useful executable boundary for the shared correction and license a task-
  conditioned within-MLP split of MLP8/9/12.
- B passes but E fails: the task-shaped group is real but the remaining 14 sites carry necessary background; split
  the named MLPs while keeping an explicit background interface.
- B fails: the single-site context profiles do not compose into a group; move to a state-level predictive quotient
  rather than tuning ranks or selecting a different subset post hoc.

Even a full pass is already-open-code circuit identification with zero saved parameters. A later within-MLP split
must be defined by task-conditioned downstream effects and selective intervention, not by rank alone.
