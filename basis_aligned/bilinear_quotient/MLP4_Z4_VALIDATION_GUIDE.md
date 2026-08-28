# Interpreting the frozen MLP4 `z4` validation

## What is being replaced

At layer 4 the model first constructs a residual state and RMS-normalizes it to
`z4` in 1,152 dimensions. The checkpoint MLP is exactly a partially symmetric
quadratic map

\[
y_4(z)=b+\sum_{j=1}^{4608} c_j(a_j^\top z)(b_j^\top z).
\]

Thus each hidden unit is one rank-one term of a third-order tensor with two input
legs and one output leg. “Native” candidates retain a fit-frozen prefix of these
checkpoint terms. They are executable replacements, not probes: during candidate
evaluation the original MLP4 computation is bypassed.

The comparison has three program languages:

1. **Affine:** `z @ W + bias`, represented by nested reduced-rank SVDs. This tests
   how much of MLP4's on-distribution action is predictable without any quadratic
   interaction.
2. **Native product:** a prefix of checkpoint factors selected on fit rows by
   centered hidden variance times output-vector norm. This preserves checkpoint
   tensor directions but not all 4,608 terms.
3. **Seeded random product:** fixed SHAKE256-generated Rademacher input factors with
   only output factors and bias learned and serialized. This tests whether bilinear
   capacity itself explains performance, without inheriting checkpoint directions.

The native/random comparison uses actual compressed byte lengths. For each native
prefix, its control is the largest preregistered random prefix whose serialized
program is no larger. This is intentionally stricter than matching component count.

## Metrics

All scores are next-token cross entropy on the separately frozen skip39000 rows,
scoring positions 64--255. Let `L_live` be the intact checkpoint loss and `L_mean`
the loss when MLP4 is replaced by its fit-row mean output. Reported fidelity is

\[
F=1-\frac{L_{candidate}-L_{live}}{L_{mean}-L_{live}}.
\]

`F = 1` matches live loss, `F = 0` is no better than the constant control, and a
negative value is worse than that control. Fidelity is only a normalized behavioral
score; it is not a fraction of weights, variance, mechanism, or information recovered.

## Preregistered decisions

| Observation | Interpretation | Implementation consequence |
|---|---|---|
| Native wins at least 4/5 actual-bit pairs | Checkpoint tensor directions carry reusable rate--distortion structure beyond a generic quadratic basis. | Port the same canonical codec and frozen-prefix selection protocol to MLP1, then close MLP4's remaining operational lanes. |
| Native and random are comparable | Most apparent gain is generic quadratic feature capacity, not checkpoint-specific factor semantics. | Prefer the portable seeded basis when its smaller graph is operationally adequate; do not call native factors semantic clusters. |
| Random wins most pairs | The checkpoint CP parameterization is a poor compression coordinate at these rates, or the fit-only native ordering is wrong out of sample. | Block direct MLP1 transfer. Seek a separately preregistered tensor normal form or typed input/output anchoring; never reorder using exposed validation. |
| Affine reaches high fidelity | MLP4 is locally close to an affine operator on natural `z4`, despite being globally quadratic. | Treat the affine map as the leading operational replacement; quadratic residuals must justify their extra bits and causal claims. |
| Affine is weak while products are strong | Multiplicative interactions are operationally necessary at this boundary. | Preserve product primitives in the compiler rather than explaining MLP4 as clustering followed by a linear readout. |
| Any family materially worsens with capacity | Fit ordering or quantized prefix nesting does not transfer reliably. | Fail the frozen family monotonicity gate; do not select the best exposed prefix as a repaired frontier. |

The formal gates are: native wins at least four of five matched-bit pairs; no adjacent
capacity worsens CE by more than 0.01 within a family; and the full-rank affine arm
reaches at least 0.65 fidelity.

## What a pass still does not establish

A held-out pass closes only one operational lane. It does not show that the candidate:

- composes with the attention and other MLP replacements;
- contains the behavior under extraction or selectively removes it;
- transfers to another distribution or context length;
- has human-semantic factors;
- is globally minimal under CP equivalence.

The native byte price removes explicit component scale/sign, input-leg swap, and
component permutation gauges. Global uniqueness of a partially symmetric CP
decomposition is not proved, so it remains a **conditional known-gauge price**, not
an unconditional quotient-MDL result. This distinction follows the classical tensor
identifiability problem studied by [Kruskal (1977)](https://doi.org/10.1016/0024-3795(77)90069-6)
and [Comon et al. (2008)](https://doi.org/10.1137/060661569).

## Whole-model next step

If the representation passes, choose the smallest preregistered point that remains
competitive on held-out loss and test it unchanged in the composed model. Then run
extraction, matched removal, and OOD lanes; only a five-lane candidate can reduce the
whole-model unexplained operational description length. If the representation fails,
stop local MLP4 tuning and return to MLP1 or global composition with the failure
recorded as evidence against checkpoint-component prefixes.
