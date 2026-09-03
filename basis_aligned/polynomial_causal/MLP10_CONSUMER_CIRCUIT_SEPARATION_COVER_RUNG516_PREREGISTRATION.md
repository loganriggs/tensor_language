# Rung516 preregistration: which known circuits force exact consumer terms apart?

**Frozen:** 2026-09-03 01:25 UTC, before any rung515 model outcome was available.

## Conditional route

Run this CPU-only analysis only if rung515's instrument passes, its discovery candidate count is exactly zero, and
its registered next step leaves the MLP10-consumer descent. If rung515 finds a discovery or substitution-valid pair,
do not open rung516; the positive causal relation takes priority.

Rung516 uses only rung515's already-computed discovery finite-removal table: documents500:748 split at624 and the
same32 discovery circuit identities on both halves. It performs no model forward, backward, fit, or intervention.
The30 confirmation circuit families and documents752:1000 are absent on the zero-pair route and remain unopened.
Accordingly this rung may establish document stability for named discovery-circuit witnesses, but not generalization
to new circuit families.

## Question

Rung515 can return zero pairs for several different reasons: the term-removal effects may already disagree on the
copy task, one side may be too small, or the32 known circuit effects may distinguish terms that look the same at the
task level. This rung isolates the last case. It asks:

1. how many exact-term pairs remain plausible after materiality, scale, and task-only tests;
2. how many of those pairs the circuit fingerprints force apart;
3. which particular circuit coordinates provide those separations; and
4. whether a small witness set chosen on one document half reproduces the same pair separations on the other half.

This is about cross-module grouping/splitting and stable identification. It does not optimize rank, retained
variance, reconstruction error, storage, or CE.

## Frozen pair universe and scale

Reuse exactly rung515's17,460 comparisons:

`6 branch subsets * 3 action relations * (31^2 attention11 pairs + 3^2 MLP11 pairs)`.

For every pair `(u,v)`, fit the single scale on the half0 32-circuit vectors exactly as rung515:

`beta = <C0(u), C0(v)> / ||C0(v)||^2`.

No scale is refit for a circuit subset or for half1. A pair is **task-compatible** if:

- both nodes pass rung515's pooled materiality bars: circuit RMS at least`.0005` nat and four-task norm at
  least`.00025` nat;
- `.25 <= |beta| <= 4`;
- on both document halves, signed task cosine is at least`.70`; and
- on both halves, both directional task relative residuals are at most`.65`.

This population is frozen before looking at which circuit coordinates reject it.

For a circuit-coordinate set `S`, use the same fixed beta and compute signed cosine plus both directional relative
residuals on `C_w(u)[S]` and `C_w(v)[S]`. The set has enough signal for a pair only when each restricted vector
contains at least10% of its corresponding full32-coordinate squared norm. On half0, `S` **witnesses a split** when
it has enough signal and violates at least one original half0 circuit gate: cosine below`.90` or either relative
residual above`.35`. On half1 the frozen test uses the original half1 gates: cosine below`.80` or either residual
above`.50`.

## Frozen greedy separation cover

Using half0 only, start with no circuit coordinates. At each step add the unused coordinate that maximizes the number
of task-compatible pairs witnessed by the accumulated set; ties use the original circuit-tag order. Freeze the first
16 coordinates. Report coverage at `k in {1,2,4,8,16,32}` on half0 and, without changing the selected order or beta,
on half1.

For16 controls (seeds51600--51615), draw a random permutation of the32 circuit coordinates and evaluate its prefixes
at the same six sizes. Controls use the same task-compatible population and fixed betas. No best seed is selected;
their maximum half1 coverage at `k=8` is the family-wide comparison.

As a descriptive semantic-stability check, repeat greedy selection using half1 only and report the Jaccard overlap
between the half0 and half1 top-eight tag sets. This repeat cannot change the prospective half0-selected result.

## Predictions

- **pred_a — exact replay:** the source hashes and zero-pair route match, all17,460 pair identities are reconstructed,
  and a direct reimplementation reproduces rung515's material-node count, candidate count, and top20 quality margins
  to absolute error at most`1e-10`.
- **pred_b — circuits add a real split:** at least64 pairs are task-compatible, and at least50% of them fail the full32
  half0 circuit gate. If fewer pairs survive task-only tests, or fewer than half are circuit-rejected, the circuit
  battery is not the main reason this quotient failed.
- **pred_c — a compact document-stable witness set:** the half0-selected top eight witness at least75% of the
  full32-half0-rejected task-compatible pairs; the same eight witness at least65% of those pairs on half1; and that
  half1 fraction exceeds the largest of the16 random-set controls by at least`.10`.
- **pred_d — witness identities are stable:** the independently selected half0 and half1 top-eight circuit-tag sets
  have Jaccard overlap at least`.50`.

## Instrument gate, null, and claim ceiling

Eight planted response tables are generated before model statistics are read. Each has task-compatible pairs whose
splits are caused by a planted set of eight circuit coordinates shared across halves, plus distractor coordinates.
The exact planted eight-set must be the greedy prefix in all eight cases and achieve pred_c's coverage/control bars.
Failure voids model interpretation.

If pred_a fails, repair only the replay. If pred_a holds but pred_b fails, the useful conclusion is that task effects
or materiality already split the terms; do not blame the circuit coordinates. If pred_b holds but pred_c or pred_d
fails, circuit separation is real but diffuse or document-unstable; do not name a compact witness set. Only all four
predictions permit the descriptive statement that a small, named set of existing circuits repeatedly forces these
consumer terms apart across documents.

Even a full pass does not identify an executable circuit component, validate the unopened30 circuit families, or
prove a globally minimal quotient. It supplies a map of which already-known circuit computations constrain this
failed grouping. Any extraction or substitution claim still requires a separately registered physical experiment.

## Price and next routes

Maximum price: zero model forwards, zero backwards, zero deployed parameters, one pass over the stored response
tensors, `6 * 32` greedy coordinate evaluations plus96 control prefixes, and eight planted tests.

- Full pass: use the named witnesses to define the observations for a task-defined downstream state-transition test.
- pred_b true but compactness/stability false: leave this exact-term vocabulary; the known circuits distinguish it in
  a distributed way.
- pred_b false: leave this descent immediately; its failure is already explained before circuit identity matters.

No route permits threshold relaxation, a larger signed-support search, rank reduction, an SAE, quantization, or a
new GPU collection solely to decorate this null.
