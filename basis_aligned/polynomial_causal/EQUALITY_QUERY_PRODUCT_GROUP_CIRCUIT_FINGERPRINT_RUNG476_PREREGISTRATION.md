# Rung 476 preregistration — downstream fingerprints of frozen within-MLP product groups

Registered after rung 475 and before running any within-MLP product-group intervention on the 62-circuit census.

## Question

Rung 475 found no stable downstream grouping at the level of complete MLP8, MLP9, and MLP12 query writes. Rung 467
already supplied a narrower candidate: 450, 426, and 482 exact bilinear product coordinates selected on code by their
task-conditioned downstream effects, plus frozen matched-count largest-amplitude and random controls. Those selected
coordinates predicted held-out code behavior but did not transfer in magnitude or control separation to natural text.

This rung asks a different, downstream-defined question on the existing diverse circuit census: do those frozen
within-MLP pieces have more similar signed effects across the 62 curated circuits than their complete parent MLPs,
their complements, or matched controls? This is a test of grouping and decomposition, not rank reduction.

## Frozen data and components

- Use exactly the 1,000 rows, 101,052 equality-positive positions, 62 circuit masks, difficulty control, and fixed
  row halves from rung 475.
- Use exactly rung 467's selected, amplitude-control, and random-control indices; no gradients, refitting, threshold
  changes, top-K fallback, or index search.
- Define each complement as the other coordinates among the 4,608 products in that MLP. Selected and complement must
  be disjoint and cover all 4,608 coordinates.

## Exact computation

For each native/transplanted equality matcher source and each MLP8/9/12, replace only the requested product coordinates
at all equality-positive query positions by their same-document equality-absent values. Run four singleton arms per
MLP: selected, complement, amplitude control, and random control. Also run selected+complement together as an exact
whole-MLP identity check against rung 475.

For each arm, save the per-position CE change and compute the same raw and native-CE-residualized 62-number signed
fingerprints used in rung 475, on all rows and both fixed halves. Comparisons are between the same named arm across
different MLPs; no circuit labels are fit.

## Frozen predictions

### A — exact intervention and partition

- all frozen source, preregistration, rung467, rung475, census, and battery hashes match;
- replay relative squared error is at most `1e-12`, factor reconstruction error at most `1e-10`, and empty masks
  change no logit;
- selected and complement are a disjoint exact partition of 4,608 coordinates for each MLP;
- selected+complement per-position effects reproduce rung475's complete-parent effects to maximum absolute error at
  most `1e-6` nat;
- observed forwards and patch calls equal the pre-model formulas, and SEALED attention-0 confirmation stays closed.

### B — selected pieces group more strongly than whole MLPs

The same selected-piece MLP pair is largest for raw and difficulty-residualized fingerprints under both sources. All
four cosines are at least `.80`, and each exceeds the corresponding complete-parent pair cosine from rung475 by at
least `.15`.

### C — selected grouping is stable across documents

The same selected pair remains largest in both fixed row halves under both sources, with all four half cosines at
least `.70`.

### D — task selection beats matched controls

For the proposed pair, selected-piece cosine exceeds both the amplitude-control and random-control cosine by at least
`.15` in every source, for both raw and difficulty-residualized fingerprints.

### E — the decomposition separates selected computation from its complement

For the proposed pair, selected-piece cosine exceeds the corresponding complement-pair cosine by at least `.15` in
both sources and both raw/residualized views. Within each of the proposed pair's MLPs, selected-versus-complement
fingerprint cosine is at most `.50` under both sources. At least ten circuits have member/off-slice effect ratio at
least `2.0` for both selected pieces with a common sign under both sources.

## Strong null and routing

The strong null fires if A fails, or if no selected pair improves over its complete-parent pair by `.05` in both
sources after difficulty residualization, or if the selected pair fails to beat either matched control in both
sources after residualization. A+B+C+D supports a circuit-family-heldout interchange test. A without B/C/D closes
the rung467 coordinate split as a downstream grouping and routes to learning a within-MLP partition from discovery
circuit families, with disjoint circuit families reserved for validation. No rank, term-count, or threshold sweep.

## Price

Diagnostic only: zero deployed parameters saved or added. Report model forwards, patch calls, runtime, peak GPU
memory, and the saved per-position effect tensor. Execute only through the managed runner.
