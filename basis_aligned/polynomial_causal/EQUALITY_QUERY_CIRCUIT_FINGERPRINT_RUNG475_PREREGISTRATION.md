# Rung 475 preregistration — downstream 62-circuit fingerprints of the equality-query writes

Registered after rung 474 and before running any query-product intervention on the circuit-census rows.

## Question

Rungs 472 and 474 show that the individual MLP8, MLP9, and MLP12 query-position effects are invariant to two causal
coordinates, while pair and triple interaction terms are coordinate-dependent. Native MLP boundaries therefore give
three reliable interventions but do not tell us whether downstream computation treats two of their writes as the same
variable.

Use the existing 62 curated behavioral circuits as downstream readouts. Two MLP query components count as a proposed
group only if their signed causal effects across those circuits agree on held-out rows, both matcher sources, and after
removing shared token-difficulty effects. This implements an interaction-determined basis rather than grouping by
weight geometry, rank, or native module name.

This rung is a screen for downstream equivalence. A positive must be followed by physical interchange/extraction on
frozen circuit families before identification.

## Frozen rows and support

Use the hash-bound `census_state_diverse.pt`: 1,000 rows × 256 scored positions and the exact 62 tags already present
in `circuits/BATTERY.json`. A CPU-only pre-outcome audit found 101,052 equality-successor query positions, with every
curated circuit having at least 135 positive member positions (median 378). Freeze rows 0:500 and 500:1000 as the two
stability halves. No circuit definition, member mask, or row may be changed after intervention outcomes open.

## Exact computation

For both frozen equality matcher sources N and H:

1. capture the equality-absent products of MLP8/9/12;
2. at every query position with at least one exact equality-successor edge, replace the complete product of MLP8,
   MLP9, MLP12, or their union by its equality-absent value;
3. rerun all later computation and save per-position CE change;
4. for each of 62 circuit member masks, form a signed fingerprint coordinate: mean CE change on equality-positive
   members. Also report mean absolute member effect, mean absolute positive effect outside the circuit slice, and their
   ratio;
5. repeat the signed fingerprint after regressing per-position CE change on native per-position CE using one affine
   fit over all equality-positive positions. This is only a shared-difficulty control; it does not fit circuit labels;
6. compare MLP fingerprints by cosine on the full grid and separately on the fixed row halves.

Removing all equality-positive positions at once is a global circuit intervention, analogous to the existing whole-
component battery. The fingerprint is not claimed to be a single-position effect.

## Frozen predictions

### A — valid census intervention

- all source/preregistration/census/BATTERY hashes match;
- exactly 62 registered tags are scored;
- equality-positive support is 101,052 positions and every circuit has at least 100 positive members;
- native replay relative squared error is at most `1e-12`, factor reconstruction error at most `1e-10`, and an empty
  mask changes no logit;
- every requested product replacement fires once per batch;
- observed forwards and patch calls equal the formulas printed before model load;
- SEALED attention-0 confirmation remains unopened.

### B — a downstream-equivalent MLP pair exists

The same MLP pair has the largest fingerprint cosine under N and H, and its cosine is at least `.80` for both raw and
difficulty-residualized fingerprints under each source. It beats the next pair by at least `.10` in at least three of
the four source × raw/residual comparisons.

### C — the pair is stable across documents

The same pair remains largest in both fixed row halves under both sources; all four half cosines are at least `.70`.

### D — the similarity is behaviorally selective rather than a global CE direction

For at least 10 of 62 circuits, both proposed-pair MLPs have member/off-slice absolute-effect ratio at least `2.0` and
the same signed member effect under both sources. At least three such circuits come from different top-level tag
prefixes.

### E — the third MLP has a distinct downstream role

Under both sources and both raw/residual fingerprints, the proposed pair's cosine exceeds each comparison involving
the third MLP by at least `.10`, and at least five circuits have opposite signed member effects between the pair mean
and the third MLP under both sources.

## Strong null and routing

The strong null fires if A fails, if no pair exceeds `.50` cosine for both raw and residual fingerprints under either
source, or if fewer than five circuits have ratio at least 2.0 for any MLP. A+B+C+D licenses one circuit-family-
heldout physical interchange test of the proposed shared downstream variable. A without B/C retains three separate
query interventions and routes to within-MLP downstream-response splitting. No threshold/rank/atom sweep follows.

## Price

Diagnostic only: zero deployed parameters saved or added. Report model forwards, patch calls, runtime, peak GPU
memory, and the saved per-position CE-effect tensor. Execute only through the managed runner.
