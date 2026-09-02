# Rung 492 preregistration — does the named attention1 source define a real MLP1 reader path?

## Question

Rung491 identifies attention1's residual write (`A1`) as the unique named native-state source necessary for both the
token-only (`T`) and token-by-context (`I`) MLP0 responses at MLP1. That result removes the term
`B(delta_b,z_A1)` from MLP1's output response while holding one branch-absent downstream background fixed. It is a
validated causal attribution, but two questions remain before treating it as an executable path:

1. Does editing A1 where it is actually read—at MLP1's input—change the complete branch effect in the way expected
   from an actual attention1 knockout?
2. Which branch group uses this path? Rung491 also found A1 necessary for C inside an insufficient full-C response,
   so this experiment must not assume that the path is exclusive to T and I.

For native normalized MLP1 state `z_N`, branch-absent state `z_b`, and the frozen native A1 state contribution `a`,
define the branch change `delta_b=z_N-z_b`. MLP1's exact quadratic response is

`OWN_b = MLP1(z_N)-MLP1(z_b)`.

Subtract the same `a` only inside MLP1 in both trajectories, while leaving attention1's residual write present:

`READER_b = MLP1(z_N-a)-MLP1(z_b-a)`.

Quadratic polarization predicts exactly

`READER_b = OWN_b-B(delta_b,a)`.

This is a selective reader edit: attention1 is still written into the residual stream and remains available to every
later module; only its contribution to MLP1's input calculation is removed. Compare it with a complete attention1
knockout, which sets attention1's write to zero in both native and branch-absent trajectories, naturally recomputes
MLP1, and therefore removes both the A1→MLP1 path and attention1's direct residual path.

## Physical interventions and measurements

Use branches `T,C,I,S`, adding S (the exact input-normalization correction branch) as an unselected control. For each
batch run:

- the normal native model and each normal branch-absent model;
- the selective A1-reader edit in the native and every branch-absent trajectory;
- sixteen controls that instead subtract the same A1-derived state shifted to another token position, again in both
  trajectories; and
- a complete attention1 knockout in the native and every branch-absent trajectory.

Every condition recomputes layers2--17. For branch `b`, define per-token CE benefit

`effect_normal_b = CE(absent_b)-CE(native)`,

and define `effect_reader_b` and `effect_A1_knockout_b` analogously within their own edited native baselines. The two
changes caused by the interventions are

`reader_modulation_b = effect_normal_b-effect_reader_b`,

`knockout_modulation_b = effect_normal_b-effect_A1_knockout_b`.

Comparing changes within each intervention's own native baseline prevents an overall native-model CE shift from
being mistaken for a change in the branch computation. Compare the reader modulation with the full-knockout
modulation by cosine, best-scale adjusted error, RMS ratio, and the sixteen shifted-position controls.

Also report native-model collateral change for the selective edit and the full knockout. The selective edit is
structurally narrower because it retains attention1's direct residual write, but aggregate CE preservation is a
measured outcome rather than assumed semantic selectivity.

## Data scope

Use the same hash-bound 1,000 documents. Discovery is documents0:500 split at250. Documents500:1000 open only after
the edited-path instrument and T/I discovery clauses pass and the supported branch set is frozen. Rung491 has already
used these documents to select A1, but none of the selective-input or full-attention1-knockout outcomes exists yet.
Validation is therefore prospective for intervention type, not independent source selection, globally virgin data,
or new-corpus OOD evidence. Final and sealed roles remain closed.

## Frozen predictions

### A — exact and lawful intervention

- All model, row, rung491 source/result, and preregistration hashes match; rung491 has A--E true, strong null false,
  and selected source set exactly `{A1}`.
- Normal native/absent prefixes reproduce exactly, the T/C/I/S branch identity retains its existing float32 and
  deployed bounds, and all call/injection counts are exact.
- In float32, the selective input-edit write difference equals `OWN-B(delta,a)` at relative squared error at most
  `1e-8` for every branch. The same difference after deployed BF16 injection must remain within `16u^2`, where
  `u=2^-8`; the complete normal OWN write retains its `4u^2` bound.
- The selective intervention leaves attention1's captured residual write unchanged bit-for-bit. The full knockout
  has exactly zero attention1 write at site1. Every same-position intervention and every shifted control is live.

### B — real attention1 dependence for T and I

In each discovery half, the complete attention1 knockout must materially change both T and I branch effects:

- `RMS(knockout_modulation)/RMS(effect_normal) >= .10`; and
- the knockout modulation must have nonzero sign-aligned covariance with the normal effect.

This clause tests whether the named source is a causal upstream dependency rather than only a convenient output-term
attribution.

### C — the selective A1→MLP1 reader edit captures that dependence

For T and I in each discovery half:

- reader modulation versus full-knockout modulation cosine must be at least`.60` and best-scale adjusted error at
  most`.80`;
- `RMS(reader_modulation)/RMS(knockout_modulation) >= .25`; and
- its cosine with the full-knockout modulation must beat the95th percentile of the sixteen shifted-position reader
  edits by at least`.10`.

These are deliberately weaker than rung491's output-term attribution bars because the full knockout also removes
attention1's direct residual path. Passing means the selectively retained A1→MLP1 path explains a stable, positional
part of the real source knockout; it does not require that this path explains all A1 behavior.

### D — stable branch grouping and narrower collateral

Apply the complete B/C support definition to all `T,C,I,S` branches separately in documents0:250 and250:500. The
supported branch set must be identical across halves and include T and I. C and S membership is not predicted in
advance; it determines whether the path is T/I-focused or broader. Freeze the complete set without dropping an
unexpected member.

In each half, the selective native reader edit must also have lower per-token CE-change RMS than the complete
attention1 knockout. This is only aggregate collateral evidence plus structural preservation of the direct residual
path; it is not yet proof that every unrelated semantic circuit is preserved.

### E — prospective intervention-type validation

Open documents500:1000 only if A--D hold. The exact frozen branch set and every T/I B/C clause must reproduce in both
validation quarters. No threshold or branch membership may change. The selective native edit must again have lower
CE-change RMS than the full knockout.

The strong null fires if A, B, C, or D fails. If E fails, the discovery path remains a screen. A full pass identifies
an executable A1→MLP1 interface edit and its branch group; it licenses new-corpus/OOD and downstream semantic
preservation tests, not compression. If actual A1 dependence passes but the selective reader comparison fails, keep
rung491 as a local MLP1-output attribution and do not call it a portable path. If the knockout itself does not affect
T/I as predicted, the upstream-source interpretation is falsified.

## Price and anti-rank gate

At batch size4, each phase runs125 normal native batches,500 normal branch-absent batches,2,125 selective native
batches (same-position plus16 shifts),8,500 selective branch-absent batches,125 full-knockout native batches, and500
full-knockout branch-absent batches:11,875 full-model forwards per phase. Conditional validation repeats this only if
licensed. Store per-token effects only as contracted sufficient statistics plus exact audit fields. Add and remove
zero deployed parameters.

This experiment advances computational specification, cross-branch grouping, extraction, selective manipulation,
and held-out causal prediction. It neither learns nor evaluates a lower-rank, sparse, quantized, or compressed basis.
