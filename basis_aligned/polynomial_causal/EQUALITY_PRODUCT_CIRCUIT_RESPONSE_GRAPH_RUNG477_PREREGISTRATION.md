# Rung 477 preregistration — per-product downstream circuit-response graph

Registered after rung476 and before computing any individual MLP product-coordinate response on the 62-circuit
census. This is discovery on frozen circuit families; it does not itself identify or adopt a circuit.

## Question

Rung476 showed that the code-selected product groups from rung467 concentrate the code equality correction but do not
concentrate cross-MLP downstream commonality. Instead of tuning that failed selection, measure each product
coordinate by how the existing downstream behavioral circuits read it. Ask whether individual product coordinates in
different MLPs have stable, task-selective response profiles that support a cross-MLP equivalence graph.

This is grouping/splitting by downstream use, not rank reduction. Any discovered group must later pass exact removal
and interchange on held-out circuit families before it can be called a circuit.

## Frozen discovery/validation separation

The 62 circuit tags have top-level forms `r.<family>.*`. Freeze complete top-level families by parity:

- discovery: the 32 tags whose integer family is even (`0,2,4,6,8,18`);
- reserved validation: the 30 tags whose integer family is odd (`1,3,5,7,11,13,23`).

Rung477 opens only discovery tags and census documents0:500. Documents0:250 and250:500 are fixed stability halves.
Every discovery member mask has at least39 equality-positive examples per half; every matched within-slice nonmember
control has at least439. The validation tags and documents500:1000 remain unopened at product-coordinate grain.

## Exact first-order response tensor

For each source `s` in native/transplanted matcher, MLP `m` in8/9/12, product coordinate `j`, discovery circuit `c`,
half `h`, and mask type member/control, compute

`F[h,s,m,j,c] = mean -<gradient of circuit CE, Down_m[:,j] * (z_source[j]-z_absent[j])>`.

Here `z=Left(x)*Right(x)` is the exact4,608-coordinate bilinear product. The control mask is the circuit's frozen
equality-positive in-slice nonmember set. The task-selective profile is member response minus control response. No
weights, circuit labels, thresholds, or product coordinates are fit during collection.

This response is a proposal statistic, not an exact finite removal. Exact validation is a successor rung.

## Frozen response graph

For each term and each of the four source×document-half views, center its 32-number selective profile across circuits
and normalize it to unit length. A term is eligible when its two half-profile cosines are at least`.50` under both
sources and the cosine between its half-averaged source profiles is at least`.50`.

For every pair of MLPs, compute cross-MLP term similarities separately in all four views. Join two terms only when:

1. each is the other's highest average-cosine neighbor across the other MLP (mutual nearest neighbors); and
2. their cosine is at least`.70` in every source×half view.

The proposed MLP pair is the one with the most qualifying joins; ties use lexicographic order8+9,8+12,9+12. The
candidate group is the two endpoint sets. There is no requested group size and no top-K fallback.

For a multiple-comparison control, repeat the graph count for16 frozen seeds after independently permuting the32
circuit coordinates of the second MLP in every view. This preserves each term's magnitude and within-MLP structure
but destroys circuit alignment across MLPs.

## Frozen predictions

### A — valid response tensor

- all source/preregistration/census/BATTERY/rung476 hashes match;
- discovery/validation tag counts are32/30 and all support minima above match;
- native replay relative squared error is at most`1e-12` and factor reconstruction error at most`1e-10`;
- term contraction versus direct MLP-write contraction has relative squared error at most`1e-8`;
- every CPU-predicted nonempty backward executes, all tensor entries are finite, SEALED attention0 stays closed, and
  validation-family product responses remain unopened.

### B — stable product terms exist

At least230 of4,608 terms are eligible in each of at least two MLPs, and at least half of those eligible terms have
nonzero selective-profile norm in every view.

### C — a cross-MLP response graph exceeds the alignment-destroyed control

The winning MLP pair has at least32 qualifying mutual-nearest joins and its count is at least four times the 95th
percentile of the16 permuted-coordinate counts.

### D — the graph is not supported by one circuit family

Recompute the winning-pair graph after leaving out each of the six discovery top-level families. The same MLP pair
wins in at least five of six omissions, and at least five omissions retain Jaccard`.50` or greater with the complete
candidate endpoint sets in both MLPs.

### E — the proposed groups have a stable task-selective aggregate response

For each proposed endpoint set, the summed member-minus-control profile has source cosine at least`.80`, half cosine
at least`.70` under each source, and member-profile norm at least1.5 times its matched control-profile norm. All six
checks must hold for both endpoint MLPs.

## Strong null and routing

The strong null fires if A fails, fewer than two MLPs have230 eligible terms, or no pair has at least16 qualifying
joins while exceeding twice the permuted 95th percentile. A+C+D+E licenses exact group removals on reserved odd
circuit families and documents500:1000, followed by physical interchange if removal fingerprints match. A without C
closes duplicated native product coordinates and routes to sparse mixed-product directions in the same response space;
it does not authorize a rank sweep or reopening rung467 thresholds.

## Price

Discovery/identification probe only: zero deployed parameters saved or added. Save the compact response tensor and
graph receipt without tokens, logits, hidden states, or validation-family product responses. Execute only through the
managed runner.
