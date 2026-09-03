# Rung519 preregistration: exact MLP0 interaction partners for one circuit

**Frozen:** 2026-09-03 03:05 UTC, after the valid rung518 result and before any rung519 model forward.

## Decision and circuit target

Rung518 found that none of990 complete head-by-source pieces is downstream-interchangeable across four copy tasks and
32 circuit effects. Two pairs match the task effects across both document halves but zero pair matches the circuit
effects even in one half. The fixed45-piece vocabulary is therefore too coarse; its thresholds and rank are not tuned.

This rung changes the object from whole pieces to the exact bilinear interactions inside one piece. The target circuit
is `r.2.0.2`, documented in `CIRCUITS_INDEX.md` as an attention8-family input-enrichment circuit involving early
attention. It was chosen from prior circuit documentation, not by searching rung518 effects. Rung518 discovery data
then mechanically choose `H4.DISTANT_SAME`: among all45 atoms, it has the largest minimum absolute `r.2.0.2` circuit
effect subject to one sign across SINGLE/DROP and both discovery halves. The four effects are
`.012520590,.003909141,.017181007,.004190039` nat. This selection is training evidence only.

Frozen dependencies:

- rung518 valid result SHA-256 `52e4d3677713a8cfa8ec2064e071a19dbb6534d71764338f7f26ecef3ea3f623`;
- rung518 bundle SHA-256 `fe9851946cdc8248cf9ea151d768589f886a1e41576c56748148ff6d24565329`;
- rung518 source SHA-256 `6294a208fdd0a4facdb93929305296bacbbcc2dc83e59ce376697cc67cd71b65`;
- rung518 preregistration SHA-256 `54ee23d84dcb515917b563690aef1c6c8e0a53909cabda59088825404ad7e382`;
- circuit index SHA-256 `e3e510bbf549c851efcd818169650f0e28b3866a22ae4a8d856fd66de87e87a0`.

## Exact computation

For each token position, let `x` be the pre-MLP0 state and `z = RMSNorm(x)`. The attention contribution to `x` is
already split into45 exact head-by-source atoms. Apply the native normalization gain to the current-token state and
the45 atoms, and define one numerical residual so these47 normalized sources sum exactly to `z`:

`z = z_TOKEN + sum_(j=0)^44 z_j + z_NUMERICAL`.

The numerical residual contains the retained attention arithmetic remainder and the residual left by representing
RMS normalization with one scalar gain. It is reported but never given a semantic label.

Write the bias-free MLP0 bilinear map as

`B(u,v) = Down[(Left u) * (Right v)]`.

For selected source `i = H4.DISTANT_SAME`, deleting `z_i` while holding the native normalized coordinates fixed
changes the float32 MLP output by exactly47 terms:

`B(z_i,z_i) + sum_(s != i) (B(z_i,z_s) + B(z_s,z_i))`.

They are the selected source's self-interaction and its interactions with TOKEN, the other44 attention atoms, and
NUMERICAL. Two separately named closing terms are then added:

1. `NORMALIZATION`: the float32 difference between deleting `z_i` at fixed gain and recomputing RMSNorm after deleting
   the raw attention atom;
2. `DEPLOYMENT_ROUNDING`: the remaining difference between the deployed BF16 MLP outputs and the float32 calculation.

The49 terms must sum to `native_MLP0_write - atom_dropped_MLP0_write`. Removing their sum from the native MLP0 write
must reproduce the original rung518 `DROP::H4.DISTANT_SAME` logits exactly. Individual interventions subtract one
term from the native MLP0 output and run layers1--17 normally. Attention0's direct residual write remains native.

## Data and measurements

Discovery uses the already-open rung518 documents500:748, split500:624 and624:748. It measures NATIVE,
WHOLE_ATOM_DROP, and all49 single-term removals on the four frozen copy tasks and the same32 discovery circuit tags.
The target circuit effect is member-token CE change minus matched-control-token CE change.

Confirmation is conditionally opened only after Prediction B. It uses rung518-unopened documents752:1000, split
752:876 and876:1000. It measures unchanged candidate terms against all62 circuit tags. The62 `(member, control)` mask
pairs must have62 distinct hashes; otherwise repeated masks are collapsed before ranks and controls are computed.

For a semantic bilinear term, define its signed target recovery fraction in a half as

`rho = term_removal_effect(r.2.0.2) / whole_atom_drop_effect(r.2.0.2)`.

A discovery candidate must have `rho >= .15` in both halves, the larger `rho` at most twice the smaller, target
absolute-effect rank at most4 among the32 distinct circuit masks in both halves, and target magnitude at least twice
the median circuit magnitude in both halves. `NORMALIZATION`, `DEPLOYMENT_ROUNDING`, and the NUMERICAL interaction are
reported controls and cannot satisfy B. Sixteen fixed seeds `519100..519115` permute each term's32 circuit identities
independently while retaining halves and the whole-atom denominator. B additionally requires the real candidate count
to exceed the higher-interpolation95th percentile of the16 control counts. No top-k list is used.

Confirmation keeps the discovery set and thresholds fixed, except that target rank at most8 among62 masks preserves
the same approximately13% rank quantile. Every candidate must pass both new document halves. Candidate identities are
not reselected.

If one to eight terms confirm, evaluate every subset of the confirmed terms on discovery and confirmation. For each
half, record the target effect for all `2^k` subsets and its exact Boolean-lattice/Möbius interaction coefficients.
Normalize each nonempty subset-effect vector by the full confirmed-set effect. The complete discovery-half0 normalized
profile must predict discovery-half1 and both confirmation profiles with cosine at least`.90` and relative error at
most`.35`. The full confirmed-set removal must recover between`.60` and`1.40` of the whole-atom target effect with the
same sign in both confirmation halves.

Selective manipulation additionally requires the full-set target effect to rank at most8 of62 absolute circuit
effects and exceed the median circuit magnitude by at least2x in both confirmation halves. Its added CE on the frozen
off-target task mask must be at most`.002` nat per half.

## Frozen predictions

### A — exact, live interaction instrument

The47 normalized sources close at relative squared error at most`1e-10`;47 semantic bilinear terms close the fixed-
gain selected-source difference at most`1e-10`; all49 terms close the deployed whole-atom MLP0 difference at most
`1e-8`; their sum reproduces rung518 whole-atom-drop logits exactly; every single-term edit is live; native replay,
call counts, rows, supports, target selection, circuit partition, and dependency hashes match. Eight planted49-term
tables recover their exact candidate and subset-interaction sets before model outcome.

### B — a small circuit-specific bilinear support

Between one and eight eligible semantic bilinear terms pass all discovery recovery, split-stability, circuit-rank, and
specificity rules, and the real count strictly exceeds the fixed permutation-control95th percentile.

### C — held-out term identification

At least one frozen B term passes the unchanged recovery/stability/specificity rules on both new document halves and
all62 circuit masks. No confirmation-only term can enter.

### D — predictable finite composition

The complete discovery subset-effect profile predicts both confirmation halves at the registered cosine/error bars,
and the full confirmed set recovers60--140% of the whole-atom target effect with the same sign in both halves.

### E — selective target-circuit manipulation

The joint finite removal ranks in the top8 of62 circuit effects, is at least2x the median circuit magnitude, and adds
at most`.002` nat CE on the off-target mask in each confirmation half.

## Controls, price, and stopping rules

The exact closures, whole-drop replay, planted recovery, target mask, all circuit-mask identities, two document halves,
permutation seeds, and every threshold are frozen before a model forward. SINGLE and DROP effects from rung518 are not
averaged. Discovery is51 arms per four-document batch: NATIVE, WHOLE_ATOM_DROP, and49 term removals, or3,162 forwards.
Conditional confirmation is the same3,162 forwards. A full subset factorial costs at most`2^8 * 62 = 15,872`
forwards per phase; repeated NATIVE and singleton arms are retained for an independent replay rather than subtracted
from the price. Maximum total is38,068 forwards,0 backwards,0 trained values,0 deployed values added or saved.

- A failure repairs only exactness, support, or replay; no model result is interpreted.
- If A holds and B fails, the selected piece does not expose a small target-specific bilinear support. Do not lower
  thresholds, change target/atom, add rank, or choose the best terms post hoc.
- If B holds but C fails, retain a discovery screen only and do not run the subset factorial.
- If C holds but D fails, the terms are individually associated but do not form a portable compositional program.
- If D holds but E fails, retain a predictive interaction model without a selective circuit-removal claim.
- A--E would identify a circuit-specific interaction program, not adoption or compression. A later executable
  replacement must still compose with other circuits and earn literal storage/compute credit.

The registered strong null is A true with B false, or confirmation finding no term. The next route after that null is
not another source refinement: move to a task-defined state transition or an attention Q/K/Q2/K2/value factor
vocabulary whose units are tested by finite downstream use.

## Pre-outcome whole-drop replay and price correction — 2026-09-03 03:14 UTC

Prediction A requires the sum of all49 terms to reproduce the whole-atom-drop logits, not merely the MLP0 output
tensor. The collector therefore includes a separately dispatched `TERM_SUM_DROP` arm in addition to NATIVE,
WHOLE_ATOM_DROP, and49 individual term removals. This makes52 arms and`52*62 = 3,224` forwards per discovery or
confirmation phase. The corrected maximum is `3,224*2 + 2*(256*62) = 38,192` forwards. The added arm is an exact
replay check only; it enters no task/circuit selection, candidate, or composition statistic. All terms, rows, target,
thresholds, controls, and conditional gates above are unchanged.
