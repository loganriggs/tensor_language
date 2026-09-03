# Parallel probe: at what granularity is the MLP10 causal fingerprint cross-half STABLE?

**Status:** prospectively frozen after the §2655 CPU probe found MLP0 per-term causal fingerprints correlate
only 0.106 across document halves (single terms and any linear combination fail to localize), and after §2656
(rung520) found the joint source-star effect is ~8-10x different from the sum of its 22 singletons. BEFORE any
cross-half correlation of star effects is computed. CPU-only, zero model forwards, zero deployed parameters.
Owner: Claude parallel lane. This resolves the route fork raised on the board: is the finer-grain instability
an N/power problem (more documents fix it) or genuinely sub-source (only activation-subspace DAS can pool it)?

## Object (frozen, from the rung520 discovery bundle)

Bundle `mlp10_source_star_causal_quotient_rung520_bundle.pt` (`7838deca…`), key
`collections/discovery`. Axes were validated against the published result: `circuit_sums` is
`[action=4, arm=23, half=2, member/control=2, circuit=32]` and `circuit_counts` is `[half, member/control,
circuit]`; arm 0 is `intact`, arms 1..22 are the 22 named MLP10 input sources; member is index 0 (support
~300 tokens/half), control is index 1 (~1550). The per-node, per-half circuit-effect coordinate is the exact
member-minus-control removal effect

`eff[a, s, h, c] = (memMean[a,s,h,c] - memMean[a,0,h,c]) - (ctrlMean[a,s,h,c] - ctrlMean[a,0,h,c])`

where `memMean = circuit_sums[...,member,...] / circuit_counts[...,member,...]` and likewise control. This is
Codex's own coordinate: pooling both halves reproduces his `material_nodes = 83/88` under the frozen rung506--510
material rule (pooled circuit RMS >= .0005 nat AND pooled four-task norm >= .00025 nat, tasks = columns 1..4).

There are 88 nodes (4 actions x 22 sources). For each node the two 32-vectors `eff[a,s,0,:]` and `eff[a,s,1,:]`
are its half-0 and half-1 circuit fingerprints. Its cross-half stability is the Pearson correlation
`rho[a,s] = corr(eff[a,s,0,:], eff[a,s,1,:])` over the 32 circuits.

## Reference point

§2655 measured the analogous per-TERM cross-half correlation of the MLP0 `H4.DISTANT_SAME` source at `0.106`.
Source stars aggregate 22 terms each, an intermediate granularity between single term (unstable) and whole
source. The question is whether aggregation to source level restores a stable object.

## Frozen predictions (with measured bars)

- **A — instrument exactness.** Bundle SHA256 = `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06`
  and result SHA256 = `1c8de74a90ca8eac167274b7fc6b84f6ed3634d5c0baf679d1d457aaf39b2a3b`; the reconstructed
  member/control/half axes reproduce the published `material_nodes == 83` of 88 exactly under the frozen rule
  above; and every count used is positive.

- **B — source-star granularity is cross-half STABLE.** Over the 83 material nodes, the median cross-half
  Pearson correlation `median rho >= 0.50`. (This is the "half the variance is a real shared object" bar and is
  far above the §2655 single-term reference of 0.106; the contrast is the science.)

- **C — the stability is not a support-geometry artifact.** The real median `rho` exceeds the 95th percentile
  of the median under 200 hash-fixed permutations of the 32 circuit labels of the half-1 vector (fixed seeds
  5090..5289). The permuted-label median is the null of "any two 32-vectors over the same supports correlate."

`strong_null = not (A and B and C)`.

## Reading and routes (frozen)

- A false: repair only the named reconstruction clause; no interpretation.
- A true, B false (stars are ALSO unstable, median rho well under 0.5): the instability is not specific to term
  granularity — the per-node causal fingerprint is underpowered at ~250 discovery documents. The decisive
  consequence: NO grouping/reuse/DAS claim at MLP10 is trustworthy until the per-node instrument is powered up
  (many more documents); recommend Codex raise document count before the next finer-grain rung, and I will
  re-run this stability check CPU-side on the higher-N bundle.
- A,B true, C false: stars correlate but no better than the support-geometry null — treat the apparent
  stability as an artifact and fall to the B-false route.
- A,B,C true: source-star level IS the stable, real granularity while sub-source (§2655) is not. Finer-grain
  below a source must therefore POOL within the source — i.e. activation-subspace DAS inside a source's
  contribution, not term enumeration — and cross-circuit REUSE is measurable at source level. Hand Codex the
  per-node stability table and a source-level reuse analysis becomes the natural next CPU probe.

No outcome licenses lowering any rung520 bar, promoting a correlation to a circuit (physical substitution is
Codex's GPU step), a rank sweep, an SAE, or a compression claim.

## Literal price

Zero model forwards, zero backwards, zero deployed parameters. Pure NumPy over a cached tensor plus 200 label
permutations; CPU, < 1 second.

## Frozen inputs

- rung520 bundle SHA256: `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06`
- rung520 result SHA256: `1c8de74a90ca8eac167274b7fc6b84f6ed3634d5c0baf679d1d457aaf39b2a3b`
