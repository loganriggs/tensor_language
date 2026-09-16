# Subject-number L11H3 late-writer OOD Möbius V1

## Purpose

Resolve the dominant `late_writes_8_10` port from the valid five-port V3
decomposition into a sparse graph over native writers, while testing selection
out of distribution instead of reporting an in-panel greedy fit.

## Frozen design

The corrected 128-row authority, frozen embedding-number intervention, frozen
rank-one L11H3 writer, and fixed L11H3 subject-source score are inherited. The
subject value before L11 is split into eight binary ports:

1. the propagated embedding plus layer 0–7 writes;
2. attention 8;
3. MLP 8;
4. attention 9;
5. MLP 9;
6. attention 10;
7. MLP 10;
8. the inherited first-value bus.

All 256 corners and all 255 exact Möbius dividends are computed. MLP10 is the
float64 closed residual component; its correction against the independently
propagated MLP10 write is measured and must be at most `1e-5` relative L2.
Only main effects and pair interactions (36 candidates) are eligible for sparse
selection. Six unscaled terms are selected greedily on pair indices 0–7 in the
`near` and `behind` contexts. No coefficient is fitted.

The same six terms are evaluated without reselection on:

- context OOD: pairs 0–7, `under` and `above`;
- lexical OOD: pairs 8–15, `near` and `behind`;
- joint OOD: pairs 8–15, `under` and `above`.

Each panel contains 32 rows. The identical selection/evaluation procedure is
also run for 16 orthogonal equal-norm random writer readouts.

## Registered gates

- Instrument: finite; exact prices/checkpoint; closed reconstruction at most
  `1e-10`; gauge correction relative L2 at most `1e-5`; native value replay and
  absolute Möbius closure at most `1e-4`; relative closure at most `1e-5`; and
  synthetic closure at most `1e-12`.
- Discovery compactness: six-term relative L2 at most `0.25` with positive
  aligned recovery.
- Context, lexical, and joint OOD: each relative L2 at most `0.25`, with
  positive aligned recovery, using the frozen discovery terms.
- Writer specificity: target joint-OOD error is at least `0.10` below the
  median joint-OOD error of the 16 random readouts.

Passing the OOD gates identifies an extractable sparse intermediate formula
with its native ports counted. It does not by itself establish behavioral
removal; that remains inherited evidence for the embedding node and L11H3 edge.
Failure of specificity licenses the structural/OOD decomposition but not a
claim that the selected terms are behavior-unique reusable atoms.
