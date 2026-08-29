# Hourly strategic review — 2026-08-29 22:55 UTC

## Bottom line

The new eight-arm result rejected native+C512 local-write MSE as the mechanism for
fixing MLP0×MLP2 composition.  This review then used the receipt-bound document ledger
to ask what kind of correction remains plausible.  The answer is sharp:

> The composition interaction is diffuse across documents and almost the same
> document-level phenomenon for FULL512, CONTINUE512, and ROBUST512.  It is not a rare
> input subset and is not explained by ordinary document difficulty.

That prunes sparse gating and makes a shared downstream-observable interface metric the
highest-return next mathematical move.

## How much of the model is explained

The strict balance sheet is unchanged:

- 36/36 structural sites have at least one intervention;
- **5.348245316%** of stored parameters are certified removable under current rules;
- **10.923302467%** of the model's causal cross-entropy effect is assigned to named
  mechanisms;
- **4.72714 nat / 89.077%** remains unexplained;
- **0/68** terminal actions has been extracted, selectively removed, OOD-transported,
  and certified end to end.

The distinction is important: structural coverage is complete, but predictive and
manipulable explanation is still sparse.

## New CPU result

For document $d$ and MLP2 program $P$, the analysis computed

$$
i_{P,d}=\Delta CE_d(C512+P)-\Delta CE_d(C512)-\Delta CE_d(P).
$$

The analysis lock, exact ledger/receipt hashes, thresholds, and claim boundary were
committed before reading these document-level values.  Runtime was **3.13 seconds**.

### Diffuse rather than gated

| Program | Mean interaction | Effective documents | Top 10% absolute-mass share | Positive documents |
|---|---:|---:|---:|---:|
| FULL512 | 0.008569 | 107.7 | 30.8% | 83.9% |
| CONTINUE512 | 0.007698 | 118.4 | 27.8% | 78.1% |
| ROBUST512 | 0.007442 | 117.6 | 28.0% | 79.2% |

All pass the frozen diffuse rule.  None approaches the sparse-gate criterion of 75%
of absolute mass in the largest 10% of documents.

### One shared document mode

Pairwise Pearson correlations of the CE interaction vectors are:

- FULL--CONTINUE: `0.8430`;
- FULL--ROBUST: `0.8560`;
- CONTINUE--ROBUST: `0.9098`.

The first singular direction explains **91.20%** of document-centered interaction
energy (94.77% without centering).  Changing training changes the mean and some noise,
but not the basic document ordering.  This is evidence for one shared failure geometry
across the rank-512 program family—not proof of one residual-space direction.

### What does not predict it

Native document NLL has Pearson correlations from `-0.1046` to `-0.0223`; standalone
C512 dCE has correlations from `-0.3529` to `-0.3225`.  The frozen simple-difficulty
rule fails.  ROBUST's document-level reduction correlates only `0.3779` with the
absolute FULL interaction, below the 0.50 targeting bar.  Thus robust training did not
selectively repair the documents where composition was worst.

Result artifact:
`mlp0_mlp2_interaction_geometry_v1_result.json`.

## Largest remaining gaps

1. **Missing observable metric.** We know raw local MSE is wrong, but have not yet
   measured the downstream adjoint metric that identifies consequential MLP2 errors.
2. **Unexplained shared interface.** The document-level interaction is common across
   fits, but its residual-space/tensor direction and downstream consumers are unnamed.
3. **Incomplete late-consumer bank.** Copy and attention-interface probes exist;
   capitalization, numeric, syntax, and entity consumers are not yet independently
   sufficient/necessary/OOD-validated.
4. **Composition beyond MLP2.** Independent MLP1 and MLP2 simplifications have not been
   placed in a receipt-backed joint triangle with C512 under equal-price controls.
5. **OOD and terminal actions.** No proposed quotient or compressed program yet
   predicts sealed OOD compositions or supports certified selective removal.
6. **Global algebraic obstruction.** Rank-512 cannot equal the native MLP polynomial
   globally; observable/reachable restriction is mandatory, not optional.

## Ranked top five actions

### 1. Repair and replicate the shipped-program MLP2 rank-allocation test

The GPU lane finished ranks 1/16/128/768 and a constant-row baseline at MLP2 inside the
full 36-site program in 443.6 seconds.  Discovery values say rank 128 costs only
`0.000205 / 0.000322 / 0.000189` nat relative to rank 768, and a constant row differs
from rank 768 by `-0.000119 / +0.000020 / -0.000284` nat.  That would be a major
allocation simplification, but the registered `pred_z_controls` failed: the intended
same-spec inert control was vacuous.  The result is therefore **not certified**.  The
highest-return action is a source-closed recovery with a genuinely non-vacuous inert
pair and the same three roles.  It is cheap, directly executable, and sharply
falsifiable; no ledger should move from the current discovery artifact.

### 2. Consumer-adjoint weighted polarization pilot

Use randomized VJP/JVP sketches to measure

$$
W_g^{1/2}J_gA_yP^{1/2}
$$

for MLP2 on native and C512 backgrounds, where $g$ includes centered logits and the
validated attention 5/6 responses.  Compare spectrum and held-out finite-response
prediction against local-MSE rank 512.  This directly addresses the rejected objective,
retains tensor-rank certificates, costs one moderate GPU pass, and is cheaply falsified
before training another full program.

### 3. Fit the mixed interaction functional, not the main effects

Use the intervention-lattice mixed difference

$$
\mathcal I(C,P)=CE(C+P)-CE(C)-CE(P)+CE(N)
$$

as the target for a rank-16/32 correction or sensitivity factorization.  The shared
91% document mode raises expected information gain, but a residual-space low-rank mode
has not yet been shown.  First require held-out directional derivatives to predict the
finite document interaction; otherwise prune before full fitting.

### 4. Run a controlled C512 × MLP1 × MLP2 composition triangle

Independent early-component compressions are not a whole program unless they compose.
A small factorial using the best equal-price MLP1 and continued MLP2 programs would
test whether the shared interface is specific to MLP0→MLP2 or is a general early-layer
quotient failure.  It has high causal relevance and moderate GPU cost, but follows the
observable pilot because another undirected composition table would localize without
explaining.

### 5. Expand the late-consumer bank and test a causal quotient

Add independently verified capitalization/numeric/syntax/entity consumers, then learn
early-state equivalence from all-but-one consumer/intervention and predict the withheld
one.  This is the best route to semantic extraction and selective removal, but it is
slower and currently blocked by missing consumers.  Finite Hankel/minimal-realization
work remains downstream of at least three independent consumers.

## Pruned branches

- More unweighted CP/Tucker/HOSVD or raw rank-512 fitting: globally impossible and
  redundant with measured negatives.
- More native+C512 local-write MSE: matched continuation explained the improvement.
- Sparse document gates for the MLP0×MLP2 interaction: the frozen concentration rules
  reject this interpretation.
- Rank-one semantic interpretation of attention 6: the correct constant-row denominator
  shows high-dimensional content; rank one represented presence.
- Reopening Family F or unchecked eight-hour-plan code as if it were evidence: the
  deadline audit is historical and only receipt-backed cells count.

## Executed action and next handoff

The CPU interaction-geometry analysis was frozen, tested (3/3), executed, and produced
a create-only result.  During it, the GPU allocation test finished with a promising
rank-128 result but a failed/vacuous control.  The immediate handoff is to repair that
control and replicate without changing the scientific arms or price bar.  If it
survives, record the executable allocation simplification; then freeze the
consumer-adjoint sketch rather than launching another local-MSE refit.
