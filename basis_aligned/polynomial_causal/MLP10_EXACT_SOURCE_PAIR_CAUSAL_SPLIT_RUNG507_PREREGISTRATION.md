# Rung 507: split MLP10 into exact named input-pair computations and test them causally

Status: prospectively frozen after the rung-506 strong null and its hash-pinned CPU clause audit, before any MLP10
source-pair attribution or intervention outcome is computed.

## Why MLP10 and what changes

Rung506 physically removed all19 downstream writes under four interchangeable equality-score sources. Every write
was live, but none had a stable fingerprint across the current32 downstream circuit tags and document repeats, so no
whole-write relation was identified. A CPU audit using only the already-open sufficient statistics found that seven
MLP writes nevertheless have stable four-context copy-task effects. MLP10 is one of them: its minimum repeat cosine
is`.744` and minimum score-source cosine is`.834`. The user also named MLP10 as the concrete example for decomposing
a later bilinear layer into the earlier writes entering its two input branches.

This rung follows the registered zero-edge route by splitting MLP10 internally. It does not rerun whole-write
grouping, lower rung506's bars, or select from its unstable circuit coordinates. It asks which exact earlier-source
pairs create MLP10's equality-dependent write, which of those terms have repeatable finite copy effects, and how two
identified terms combine under joint removal.

## Exact MLP10 computation

Let `r` be the residual stream immediately before MLP10's RMS normalization. With the model's learned residual-mix
coefficients included, write

`r = r_E + sum_{i=0}^{10} r_Ai + sum_{i=0}^{9} r_Mi`.

`E` is the accumulated embedding/skip contribution, `Ai` is attention block `i`'s output write, and `Mi` is MLP
block `i`'s output write. There are exactly `1+11+10 = 22` named sources. The implementation must reconstruct these
coefficients from the shipped model and compare their sum directly with the captured pre-normalization residual.

The model has unweighted RMS normalization, so for a fixed token its normalized MLP10 input `z` is proportional to
`r`. Define the exact scalar

`g = dot(z,r) / dot(r,r)`

and named normalized sources `z_s = g r_s`. The small floating-point remainder is stored explicitly as

`z_num = z - sum_s z_s`.

MLP10 is the bilinear map

`MLP10(z) = Down[(Left z) * (Right z)] + bias`,

where `*` is coordinatewise multiplication, `z` has dimension1,152, the product layer has4,608 coordinates, and the
output returns to1,152 residual-stream coordinates.

For ordered named sources,

`B(s,t) = Down[(Left z_s) * (Right z_t)]`.

Combine the two orders for `s != t`. The22 self/cross choices therefore give

`22*23/2 = 253`

unordered named source-pair terms. Their sum plus the explicitly stored terms involving `z_num` and the bias must
reconstruct the independent float32 MLP10 output. The numerical terms are reported separately and are never allowed
to enter the named selector.

For score action `a`, let `B_a(p)` be named pair term `p`, and let `B_0(p)` be the same term on the same document when
the L8H4 equality score is absent. The score-dependent term is

`delta_a(p) = B_a(p) - B_0(p)`.

Removing named term `p` means subtracting `delta_a(p)` from the deployed MLP10 write, casting only at that final
write boundary, and recomputing layers11--17 normally. Removing two terms subtracts both before recomputation. This
keeps the exact native background and measures downstream nonlinear interactions.

## Fixed score actions and data partitions

Use the same four score actions and scales as rungs505--506: native `N`, positive L5H5 replacement `P`, and correctly
negated L7H3/L8H3 replacements `Z7/Z8`. No scale is fit.

- gradient discovery: documents`0:248`, reported as`0:124` and`124:248`;
- finite singleton confirmation: documents`248:496`, reported as`248:372` and`372:496`;
- documents`496:500` remain unused;
- finite singleton and pair validation: documents`500:1000`, reported as`500:750` and`750:1000`.

Use the fixed task vector `(near copy, far copy, one earlier match, multiple earlier matches)`, plus all-copy and
off-target masks. The32/30 circuit-tag partitions are not used for selection because rung506 showed that even the
same whole write does not have a repeatable direction in those coordinates. On validation, their member-minus-matched-
control effects are stored as a diagnostic for every surviving term, but they do not gate this rung.

## Discovery is a gradient screen, not the causal result

On discovery only, compute the gradient of mean loss in each task cell with respect to the complete MLP10 output
write. For named term `p`, the first-order prediction for removing it is

`A_a(p,c) = -mean_{x in cell c} dot(grad_loss_a(x), delta_a(p,x))`.

Also compute the same prediction for removing the complete equality-dependent MLP10 write. The gradient is used only
to avoid thousands of blind suffix recomputations. It cannot establish a circuit, because one-component attribution
includes background-dependent mediator interactions and ignores curvature.

A named term is eligible only if, for every score source:

- its four-context attribution norm is at least`.00025` nat. This is approximately10% of rung506's smallest
  observed complete-MLP10 task norm (`.00249` nat), rather than a post-outcome term threshold;
- its absolute projection on the complete-MLP10 attribution is at least`.05` of that attribution;
- its two discovery-repeat vectors have cosine at least`.60` and norm ratio at most3; and
- its all-copy attribution magnitude is at least twice its off-target magnitude.

For `P/Z7/Z8`, its pooled vector must have cosine at least`.70` with `N` and norm ratio at most3. Retain every
eligible term; do not rank. Discovery is identifying only when it returns between2 and8 terms. Zero or one cannot
test interactions; more than8 is a non-sparse screen and is not permission to keep eight.

## Finite confirmation and validation

Only discovery terms receive finite interventions on documents`248:496`. For term `p`, source `a`, and task cell
`c`, define

`V_a(p,c) = mean[L_after_removing_p - L_intact]`.

A term confirms only if, for every source:

- finite task-vector norm is at least`.00025` nat;
- finite-vector cosine with the discovery gradient prediction is at least`.60`;
- its two confirmation-repeat vectors have cosine at least`.50` and norm ratio at most3;
- all-copy magnitude is at least`.00025` nat and at least twice off-target magnitude; and
- its pooled vector under `P/Z7/Z8` has cosine at least`.70` with `N` and norm ratio at most3.

Retain every confirming term without reselection. Pair analysis proceeds only if between2 and8 terms confirm.

On documents`500:1000`, each confirming singleton must retain finite task-vector cosine at least`.60` with
confirmation, positive cosine in both validation halves, norm ratio at most3, source-to-native cosine at least`.65`,
and the same all-copy/off-target selectivity. No term is added.

## Exact multiple-mediator interaction

For every unordered pair of confirming terms, run the joint finite removal on confirmation and validation. Define

`J_a(p,q) = V_a({p,q}) - V_a(p) - V_a(q)`.

This is the exact interaction that a single-mediator patch cannot separate from the mediator's background-dependent
effect. Every pair is retained and reported; no interaction is ranked.

Choose the first applicable composition rule from confirmation, concatenating all four score-source task vectors:

1. `additive` if `||J||/||V({p,q})|| <= .25`;
2. `p redundant` if `||V({p,q})-V(p)||/||V({p,q})|| <= .25`;
3. `q redundant` by the symmetric rule;
4. `one-scalar interaction` with
   `beta = dot(J,V(p)+V(q))/||V(p)+V(q)||^2`, requiring `|beta|>=.25`, `-.8<=beta<=2`, and
   `||J-beta(V(p)+V(q))||/||J||<=.50`.

On validation, the frozen rule must predict every source's joint task vector with cosine at least`.70` and relative
residual at most`.65`, preserve the all-copy/off-target selectivity, and have positive prediction cosine in both
document halves.

A pair is a same-output candidate only if its two singleton task vectors additionally have cosine at least`.80` for
every source on confirmation and validation. Independently of same-output status, report whether its two named
bilinear terms share their Left input source, Right input source, either unordered source, or neither. This explicitly
separates “different inputs with the same downstream effect” from “one shared input composed with different partners.”

## Numerical and causal controls

The instrument must verify:

- exact22-source identity and253-pair enumeration;
- raw-source sum versus captured pre-normalization residual, normalized-source plus numerical-remainder closure, and
  float32 bilinear reconstruction;
- named pair changes plus numerical-remainder change reconstruct the complete MLP10 score-dependent output within
  the registered BF16 deployment bound `16*(2^-8)^2` relative squared error;
- direct native versus analytical replay relative-squared logit error at most`1e-12`;
- every selected term patch and joint patch fires exactly once, is finite, and changes the MLP10 write;
- all conditional forward/backward/capture/patch counts and task supports; and
- the four score actions recalibrate on discovery: recovery in`[.65,1.40]`, per-document all-copy cosine at least
  `.85`, and off-target change at most`.01` nat.

## Literal price

Batch size is4. Discovery uses62 batches, each with one direct native forward, one score-absent capture, and four
intact score-source forwards with gradients: `372` full forwards and at most`1,240` task-cell backward calls.

If discovery returns `k` terms (`2<=k<=8`), finite confirmation costs
`62*(1+4*(1+k)) = 310+248k` forwards. If `q` terms confirm, confirmation pair measurement costs
`62*(1+4*choose(q,2))`. Validation of every singleton and pair costs
`125*(1+4*(1+q+choose(q,2)))` forwards. A run reaching validation therefore costs exactly

`1,369 + 248k + 500q + 748*choose(q,2)` full forwards,

at most`28,297` for`k=q=8`, plus at most1,240 backwards in discovery. It fits no vector, permits at most one scalar
per confirming pair, and adds/saves zero deployed parameters.

## Registered predictions and result routes

### A. Exact and live decomposition/intervention instrument

Every numerical, hash, support, shape, liveness, and conditional price clause above holds.

### B. The gradient screen is sparse and score-source stable

The score actions recalibrate, and the no-ranking discovery rule returns between2 and8 named source-pair terms.

### C. At least two terms survive finite confirmation

Between2 and8 terms clear every finite confirmation rule. This is the internal MLP10 split screen; gradient-only
terms do not count.

### D. At least two terms validate on new documents

At least two confirming terms pass every validation and selectivity rule without reselection.

### E. At least one two-term composition rule predicts validation

At least one pair has a confirmation-frozen additive, redundant, or one-scalar interaction rule that predicts its
joint validation effect. Report same-output and shared-input classifications separately.

The strong null is A false; score recalibration false; fewer than2 or more than8 discovery terms; fewer than2 finite
confirmation terms; fewer than2 validation terms; or no predictable pair interaction.

- A false: repair only the algebra or intervention instrument.
- A true/B false with too few terms: the exact MLP10 pair vocabulary does not expose a sparse task split under this
  score action; move to a coupled factor/output dictionary on the exact term tensors, with finite interventions still
  required before circuit language.
- A true/B false with too many terms: the task masks are too coarse to identify a small program; add independently
  defined tasks rather than selecting eight terms.
- B true/C false: gradient attribution did not predict finite causal effects; retain the exact algebra but replace
  gradient screening with a smaller registered factorial based on input-source families.
- C true/D false: the term split is corpus-dependent; preserve it as a screen only.
- C/D true/E false: identified singleton terms do not have a simple multiple-mediator composition law; model their
  higher-order state dependence before extraction.
- A--E true: identified named input-pair terms and their finite composition become candidates for an executable MLP10
  replacement followed by held-out 62-circuit diagnostics and selective joint edits.

No result licenses rank reduction, quantization, threshold changes, best-eight selection, or calling a gradient
attribution a circuit.
