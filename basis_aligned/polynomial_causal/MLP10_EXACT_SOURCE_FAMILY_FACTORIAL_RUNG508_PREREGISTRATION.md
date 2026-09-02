# Rung 508: finite exact source-family terms inside MLP10

Status: prospectively frozen after rung507's lawful strong null, before any exact family-term removal outcome on
documents 500:1000 is computed.

## Question and reason for changing the object

Rung507 expanded MLP10 into all253 exact unordered pairs among its22 named earlier inputs. A no-ranking gradient
screen found exactly `A7×A8` and `A8×A8`, but neither survived finite removal on new documents. The gradient was a
useful search calculation, not a circuit: its local linear prediction did not capture the finite suffix response.
The registered route is therefore to replace gradient screening with a smaller finite factorial over input-source
families. This rung does not lower a threshold, rank terms, or choose more exact pairs from the failed screen.

The circuit targets are within-MLP splitting, stable grouping of exact computations, held-out causal prediction,
selective manipulation, and predictable composition. Literal rank, parameter count, and reconstruction error are not
selection objectives.

## Fixed source families

Partition the22 named sources before reading any new family-removal effect:

- `E = {E}`;
- `A_pre = {A0,A1,A2,A3,A4}`;
- `A_eq = {A5,A6,A7,A8}`;
- `A_post = {A9,A10}`;
- `M_pre = {M0,M1,M2,M3,M4}`;
- `M_post = {M5,M6,M7,M8,M9}`.

The boundary at layer5 is fixed by the established equality-score circuit beginning at attention5, not by rung507's
two exact candidates. `A_eq` is the contiguous attention interval from that donor through the attention8 recipient;
it includes attention6 instead of selecting only known favorable heads. Attention and MLP sources remain separate.
These six disjoint families cover every named source exactly once.

For family pair `F,G`, define the exact MLP10 output term as the sum of all named `B(s,t)` whose two sources belong
to `F,G`, with both Left/Right orders included when the families differ. Equivalently, if `L_F` and `R_F` are the
sums of the Left and Right factors for sources in `F`,

- `H(F,F) = L_F * R_F`;
- `H(F,G) = L_F * R_G + L_G * R_F` for `F != G`;
- `Y(F,G) = Down[H(F,G)]`.

There are `6*7/2 = 21` disjoint family-pair terms, and their sum must equal the sum of all253 named terms. The input
numerical remainder and deployed-output rounding remainder from rung507 remain explicit diagnostics and cannot enter
any family term or selector.

## Physical finite intervention

For score source `a`, let `Y_a(F,G)` be a family term and `Y_0(F,G)` the same term when the equality score is absent.
Removing it subtracts `Y_a(F,G)-Y_0(F,G)` from the actual deployed BF16 MLP10 write and then recomputes layers11--17.
The `FULL_NAMED` arm subtracts the sum of all21 changes. It is a target/reference arm, not a selectable term.

For task cell `c`, the finite effect is

`V_a(F,G,c) = mean[CE_after_removal - CE_intact]`.

Positive means that exact family term helped the prediction. This is measured by real forward passes; no gradient or
first-order approximation enters selection.

Use the four already calibrated score implementations `N,P,Z7,Z8`. Rung507 did not open its validation phase, so the
new family-removal outcomes use:

- discovery documents `500:748`, with repeats `500:624` and `624:748`;
- documents `748:752` unused;
- confirmation documents `752:1000`, with repeats `752:876` and `876:1000`.

The task vector is `(near copy, far copy, one previous match, multiple previous matches)`, plus all-copy and
off-target masks. The32/30 circuit coordinates remain diagnostic only because rung506 showed they were unstable as
a selector at whole-write grain.

## Discovery without ranking

A family term is retained only if, for every score source:

- its four-context finite-effect norm is at least`.00025` nat;
- its absolute projection onto the `FULL_NAMED` finite-effect vector is at least`.05` of that vector;
- its two discovery-repeat vectors have cosine at least`.50` and norm ratio at most3; and
- its all-copy effect has magnitude at least`.00025` nat and at least twice its off-target magnitude.

For `P,Z7,Z8`, the pooled vector must have cosine at least`.70` with `N` and norm ratio at most3. Retain every passer;
do not rank. The family split is identifying only with2--8 retained terms. Zero/one cannot test composition, and more
than8 is a non-identifying task description rather than permission to take the best eight.

## Confirmation and multiple-mediator interaction

Every retained singleton is remeasured without reselection on confirmation. It confirms only if, for every source:

- its confirmation norm is at least`.00025` nat;
- its pooled confirmation vector has cosine at least`.60` with discovery and norm ratio at most3;
- its two confirmation repeats have cosine at least`.50` and norm ratio at most3;
- its all-copy magnitude is at least`.00025` nat and at least twice off-target; and
- `P,Z7,Z8` each have cosine at least`.65` with `N` and norm ratio at most3.

If2--8 terms confirm, remove every unordered pair jointly on both discovery and confirmation documents. Define the
finite suffix interaction

`J_a(p,q) = V_a({p,q}) - V_a(p) - V_a(q)`.

Fit the first applicable rule on discovery after concatenating all four source vectors:

1. additive if `||J||/||V({p,q})|| <= .25`;
2. left redundant if `||V({p,q})-V(p)||/||V({p,q})|| <= .25`;
3. right redundant by the symmetric rule;
4. one-scalar interaction, using `J = beta*(V(p)+V(q))`, if `|beta|>=.25`, `-.8<=beta<=2`, and relative residual
   at most`.50`.

The frozen rule predicts confirmation only if, for every score source, prediction cosine is at least`.70`, relative
residual at most`.65`, both confirmation-half cosines are positive, and the joint all-copy effect remains at least
`.00025` nat and twice off-target.

Report a same-output candidate only when the two singleton task vectors have cosine at least`.80` under every source
on both phases. Separately report whether the family terms share a Left family, Right family, any family, or none.

## Numerical and causal controls

The instrument must verify all frozen hashes and the rung507 A/B-true, C-false route; exact6-family coverage and21
disjoint terms; the21-term sum versus the253-term sum; raw/normalized/float32/deployed closures under rung507's
repaired rules; exact native analytical replay; every requested family, full, and joint edit firing once and changing
the write; exact task supports, call counts, and patch counts; and score recalibration on both phases with recovery
in`[.65,1.40]`, per-document all-copy cosine at least`.85`, and off-target change at most`.01` nat.

## Literal price

Each248-document singleton phase has62 batches. Per batch it runs one score-absent capture and, under four sources,
`intact + FULL_NAMED + 21 family terms`: `62*(1+4*23)=5,766` forwards. If `k` terms pass discovery, each pair phase
costs `62*(1+4*choose(k,2))`. Reaching confirmation and pair prediction therefore costs exactly

`11,656 + 496*choose(k,2)` full forwards,

at most`25,544` for`k=8`, with0 backwards,0 fitted vectors, at most one fitted scalar per pair, and0 deployed
parameters added or saved. If discovery returns outside2--8, the run stops after5,766 forwards.

## Registered predictions and routes

### A. Exact and live finite family-term instrument

Every numerical, support, replay, liveness, hash, and conditional-price check passes.

### B. A sparse finite family split exists

The no-ranking discovery rule retains2--8 of the21 exact family terms.

### C. At least two family terms confirm

Between2 and8 terms pass every held-out singleton rule without reselection.

### D. At least one simple multiple-mediator rule predicts confirmation

At least one confirming pair's discovery-frozen additive, redundant, or one-scalar rule predicts its joint finite
confirmation effect.

### E. The equality-attention family participates in a confirmed term

At least one confirming family term contains `A_eq`. This tests whether the known equality-score interval enters an
identified MLP10 computation after exact-term gradients failed.

The strong null is A false, failed score calibration, a discovery count outside2--8, fewer than2 confirming terms,
no predictable pair, or no confirmed `A_eq` term.

- A false: repair only the family algebra or intervention instrument.
- B false with too few terms: move to a coupled Left/Right/output dictionary whose atoms must predict finite effects;
  do not lower bars or return to gradients.
- B false with too many terms: add independently defined task/circuit outcomes; do not take eight.
- B true/C false: the architecture family split is corpus-specific; preserve it as a screen only.
- C true/D false: model higher-order suffix state dependence before extraction.
- D true/E false: the stable family terms are not the expected equality-input path; audit the normalization-mediated
  route before giving them semantics.
- A--E true: refine only confirmed family terms into smaller exact sources/heads and build an executable held-out
  MLP10 replacement candidate.

No outcome licenses rank reduction, quantization, threshold changes, top-k selection, or calling a reconstruction a
circuit.
