# Frozen shared-parent branch removal screen

Registered on the board at 17:21:55 UTC, 11 September 2026, before native outcomes.
This tests selective manipulation and whether an algebraically shared reader supports
different branches. It does not fit weights or identify a circuit by itself.

Candidate: parent 1, canonical branches 0 and 1, in
`SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt`. Their coefficient energies are
53.67% and 46.33% of this node. The signed SVD factors are used together, so their
arbitrary joint sign does not affect the executed contribution. Physical writers
are obtained by solving with the saved output whitener.

$$
\delta_j(x)=(u^\top x)(p_j^\top x)c_j.
$$

Remove branch 0, branch 1, or their sum from native MLP17 output. Keep the native
bias, residual, RMS normalization, full unembedding, and tanh tail. The background
is the real model, not an extracted surrogate. Positive CE added means damage.

Rows were frozen before model outcomes in `SHARED_NODE_PARENT1_FINEWEB_V1_ROWS.pt`
and its JSON provenance. Start with the top 12 positive real-token writer loadings
per branch; exclude their overlapping token ` offers` from both. The historical
FineWeb cache `fineweb_n192_skip11000.pt` has 27 documents containing both families
at target positions at least 128. Select 24 with seed 4301. In each document use
the first family-0 target, nearest family-1 target, and nearest target outside both
families, breaking ties toward the earlier position. Take 128 preceding tokens
per target. These are 72 endpoints in 24 document units, stored family first.
Nearby controls can overlap context, so bootstrap whole documents with the same
indices across families. Intervals are descriptive, not additional registered bars.

- **A:** row provenance, exact counts and finite scores; native and branch-0
  physical model-hook endpoint logits agree with manual native-prefix/tail
  execution to relative error at most 1e-5. MLP inputs agree within 1e-6.
- **B, conditional on A:** the actual next token is in the native top 20 at
  least half the time in each of the two target families.
- **C, conditional on A/B:** each family's own branch removal adds at least
  0.02 mean CE; own-minus-other branch damage is at least 0.01 in each family;
  each individual removal causes at most 0.05 mean absolute CE change on controls.

Report all signed effects even when a bar fails. The null is that distinct
algebraic branches do not selectively support these token families in native
contexts. Weight-selected output signs do not determine actual contribution signs:
the product of input reads can reverse them. Token readouts alone partly select
the expected output preference, so even a positive screen needs stronger controls.
Do not interpret a miss as absence of shared computation. Inspect native capability,
signed branch activity and control effects before deciding the discriminating repair.

Price: 9 batches of 8 native prefixes plus two physical controls on the first
batch: **11 body forwards, 88 sequences, 128 input tokens**. All interventions
reuse endpoint states. No native-body fitting; 900-second alarm. The standalone
two-branch bank has 5,760 coefficients, with the native model and adapters retained.
Store only endpoint ports, scores and provenance. No claim of fresh/OOD prediction,
isolated sufficiency, stable identification, or additive CE effects.
