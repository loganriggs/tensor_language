# MLP10: current understanding and duplicate-work map

**Updated:** 2026-09-02 22:55 UTC

This dossier consolidates the existing MLP10 results that were previously scattered across the main ledger. It is
the required duplicate-work check for new MLP10 experiments. Numerical screens are kept separate from causal circuit
claims.

## Exact computation and dimensions

At one token position, MLP10 receives a normalized residual-stream vector `z` with1152 entries. It computes

`y = Down[(Left z) * (Right z)] + bias`,

where `Left` and `Right` each map1152 inputs to4608 numbers, `*` multiplies matching coordinates, and `Down` maps the
4608 products back to1152 residual-stream coordinates. The native weights contain

`4608*1152 + 4608*1152 + 1152*4608 + 1152 = 15,926,400`

stored numbers. The4608 entries are product features, not the model hidden dimension; the model hidden/residual
dimension is1152.

The normalized input can also be written exactly as22 earlier sources: embedding `E`, attention outputs `A0..A10`,
and MLP outputs `M0..M9`, plus an explicitly tracked numerical remainder. Expanding `Left(z)*Right(z)` gives253
unordered source-pair terms. For distinct sources `s,t`, the exact output term is

`Down[(Left z_s)*(Right z_t) + (Left z_t)*(Right z_s)]`.

For `s=t`, it is `Down[(Left z_s)*(Right z_s)]`. These terms are an exact algebraic coordinate system. They are not
automatically circuits because downstream computation can treat their effects differently across data or model
backgrounds.

## What generic approximation work already established

### Whole-write importance and output-map screens

- In the old all-layer functional-output-map profile (§713, replicated at32k tokens in §717), ablating MLP10 cost
  only about`.03-.05` nat and recovering80% of that small benefit required512 output directions. Because the total
  benefit was tiny, `r80=512` does **not** establish an important high-rank computation.
- A stream-linear surrogate in §1483 recovered`.4009` of the mean-ablation gap; adding random-projection quadratic
  features improved that by only`.0065`, for optimized fidelity`.4129`. This closed generic “add a quadratic
  correction” as an explanation of the missing half.
- The invariant weight-tensor screen in §2482 retained`.59437` of product-mode energy in the top2304 of4608 product
  directions and`.73133` of output-mode energy in the top512 of1152 output directions. MLP10 had the lowest top512
  output retention of all18 MLPs. These are gauge-invariant geometric screens, not causal groupings or executable
  circuit results.

Therefore another rank sweep, output projection, generic ridge map, random quadratic map, or Tucker corner would
repeat a closed question unless it follows an independently identified causal circuit and is used only to price that
circuit.

## Known circuit roles

### Question-mark circuit: a known MLP11 consumer

Section1597 identified a two-dimensional quadratic form at MLP11 that controls the question-mark output channel.
Four earlier public writes supplied71.79% of its attributed input: attention10, attention9, MLP9, and MLP10. MLP10 was
the fourth writer, with attributed magnitude112.91 versus581.31/405.34/366.50 for the other three. Removing the
two-dimensional subspace from all four writers raised question-class loss by`.8144` nat with approximately zero
global loss change; a matched mid-ranked four-writer control caused only`.0858` nat.

This is a real cross-module circuit precedent, but it does not isolate which internal MLP10 products supply that
subspace: the physical edit acted on the four writers jointly. For any later “which consumer distinguishes MLP10
terms?” experiment, the MLP11 question-form reader is the first already-known interface to test, not a new discovery.
It must not be assumed to be MLP10's only consumer.

### Equality/copy circuit: small public-write role, unresolved internal realization

- In §2580, copying only MLP10's whole write from the equality-present run into an equality-absent run recovered
  `5.63%` of the context-dependent equality effect, the best single later public write but below the registered10%
  sufficiency threshold. The complete correction is distributed and interaction-dependent.
- Rung506 (§2637) found MLP10's four copy-context task effect stable enough to place it descriptively with the early
  `{MLP8,MLP9,MLP10}` cluster. Its32-circuit member-minus-control fingerprint at whole-write grain did not repeat,
  so this was not a whole-MLP grouping claim.
- Rung507 (§2638) constructed all253 exact input-pair terms. A no-ranking gradient rule selected `A7*A8` and
  `A8*A8`, but both failed finite causal confirmation. Gradient attribution does not identify the finite terms.
- Rung508 (§2639) grouped all253 terms into21 pairs of six architecture-defined source families and removed all21
  finitely. Zero terms repeated across document halves and the four calibrated equality-score implementations. Even
  `A_eq*A_eq` ranged from native repeat cosine`.975` to`-.474` under another implementation.
- Rung509 (§2641) tried a coupled Left/Right/downstream dictionary. The free dictionary was restart-stable but wrong
  on a known toy; the convex-hull repair also failed a favorable eight-anchor ground-truth test. No model outcome was
  opened. Latent dictionary, seed, penalty, and atom-count sweeps are closed for this route.

The standing equality result is therefore precise: MLP10 is causally involved as one piece of a distributed later
program, and its253-term algebra is exact, but gradients, broad source families, and the tested hidden dictionary do
not identify stable internal circuit units.

## Live experiment and what it can add

Rung510 tests all1,012 directly observed `(score implementation, exact source-pair term)` removals. Thirty-two of the
existing62 circuit families and one document split propose pairwise equivalences;30 different families and new
documents test them. A pair only counts after its exact MLP10 output changes can be substituted bidirectionally
through the real layers11--17 suffix.

This is non-duplicate because it defines a unit by downstream interchangeability rather than activation geometry,
gradient size, source location, or a fitted latent basis. Its possible claims are deliberately narrow:

- two different exact terms are one downstream variable;
- one exact term keeps the same meaning across score implementations; or
- no small pairwise downstream quotient exists at the registered scale.

If response-similar pairs fail the physical swap, the next consumer-local test should include the already identified
MLP11 question-form interface as a fixed hypothesis and compare it with every other layer11--17 attention/MLP
consumer. If no pair exists, the registered alternative is a physical test of predeclared signed multi-term
combinations, not another rank or dictionary search.

## Do-not-repeat list

- Do not interpret the old `r80=512`, top2304 product energy, or top512 output energy as a circuit.
- Do not rerun generic linear, quadratic, Tucker, or output-rank sweeps as discovery.
- Do not select exact products by gradient magnitude and call them causal.
- Do not retry the six architecture families with relaxed thresholds.
- Do not tune the rung509 atom count, seeds, initialization, or penalties.
- Do not claim MLP10 alone implements the equality context law or the question-mark circuit.
- Do not treat native MLP/head boundaries as the final semantic basis.

## Primary local evidence

- `deep_mid_sweep_results.json` and ledger §1483 — stream-linear and quadratic surrogate.
- `slice_writers_results.json` and ledger §§1597--1603 — MLP10 to MLP11 question-form circuit.
- `mlp_mode_concentration_depth_profile_results.json` and ledger §2482 — exact tensor-mode spectra.
- ledger §§2580--2586 — distributed equality correction and its MLP8/9/12 interaction structure.
- rung506--509 receipts and ledger §§2637--2641 — whole-write, exact-term, family, and dictionary results.
- rung510 preregistration — live observable downstream-equivalence test.
