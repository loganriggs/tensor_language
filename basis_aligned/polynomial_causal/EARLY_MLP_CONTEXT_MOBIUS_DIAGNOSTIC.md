# Early-MLP/context hierarchical Möbius diagnostic

**Status: descriptive post-outcome analysis. This is not a new prospective pass.**

The exact product-poset Möbius transform writes each measured interaction as
a sum of contributions that appear for the first time at a particular early-MLP
subset and suffix replacement set. The transform is exact; simplicity comes only
from predicting cells after retaining a small, stable subset of its terms.

The early side is a genuine three-factor Boolean lattice. The suffix side is
only a registry of nested macro-replacements, not a physical factorial: for
example, MLP-only layers 3--8 were never measured. Suffix Möbius coefficients
therefore mix broad MLP main effects with attention-by-MLP synergy. They are
useful macro-contrasts, not identified per-site mechanisms or tensor-program terms.
The zeta basis is also non-orthogonal, so squared coefficient size is not
Parseval energy. Simplicity below is judged by held-cell prediction instead.

## Main findings

The two complete interaction grids have Pearson correlation **0.9963** across their 49 non-anchor cells.
This measures transport across disjoint document populations before fitting any
new values on the target role.

### Singular spectrum (descriptive only)

| role | rank-1 energy | rank-2 | rank-3 | rank-4 |
|---|---:|---:|---:|---:|
| skip7000 | 0.8523 | 0.9586 | 0.9901 | 0.9998 |
| skip11000 | 0.8707 | 0.9665 | 0.9918 | 0.9998 |

High in-sample singular energy is not enough: the prospective fixed-pivot rank
test already failed. The remaining question is whether a *sparse hierarchical*
basis predicts an omitted intervention more reliably.

### Leave-one-cell-out sparse Möbius prediction

Each of the 49 non-anchor intervention cells is omitted in turn. Orthogonal
matching pursuit selects a fixed number of Möbius terms using only the other 48
cells. NRE is RMSE divided by the zero-interaction baseline; below 1 is useful.

| role | terms | LOO NRE | LOO R2 |
|---|---:|---:|---:|
| skip7000 | 1 | 0.7103 | 0.0526 |
| skip7000 | 2 | 0.6243 | 0.2680 |
| skip7000 | 4 | 0.5628 | 0.4052 |
| skip7000 | 8 | 0.4449 | 0.6283 |
| skip7000 | 12 | 0.4100 | 0.6843 |
| skip7000 | 16 | 0.3654 | 0.7493 |
| skip7000 | 24 | 0.4683 | 0.5881 |
| skip7000 | 32 | 0.5667 | 0.3970 |
| skip11000 | 1 | 0.7054 | 0.0376 |
| skip11000 | 2 | 0.6159 | 0.2661 |
| skip11000 | 4 | 0.5677 | 0.3765 |
| skip11000 | 8 | 0.4230 | 0.6539 |
| skip11000 | 12 | 0.4086 | 0.6771 |
| skip11000 | 16 | 0.3469 | 0.7672 |
| skip11000 | 24 | 0.3904 | 0.7052 |
| skip11000 | 32 | 0.4917 | 0.5323 |

### Fixed early interaction order is not enough

Keeping all suffix macro-contrasts but only singleton early-MLP terms uses 21
coefficients; allowing early pairs uses 42. These hereditary models perform
poorly despite their larger sizes:

| role | maximum early order | terms | LOO NRE | LOO R2 |
|---|---:|---:|---:|---:|
| skip7000 | 1 | 21 | 0.9774 | -0.7939 |
| skip7000 | 2 | 42 | 0.9291 | -0.6210 |
| skip11000 | 1 | 21 | 0.9680 | -0.8126 |
| skip11000 | 2 | 42 | 0.8248 | -0.3159 |

The useful sparsity is structured across both early subsets and suffix
macro-contrasts; it is not merely a low-degree polynomial in MLP0/1/2.

### Cross-document support transport

Support is selected on one role. `Direct` copies both that sparse grammar and its
coefficients to the other role. `Refit` preserves only the grammar but allows the
target role to re-estimate its coefficient values. This separates stable structure
from stable numerical calibration.

| direction | terms | direct NRE | refit NRE |
|---|---:|---:|---:|
| skip7000_to_skip11000 | 1 | 0.6992 | 0.6967 |
| skip7000_to_skip11000 | 2 | 0.6135 | 0.6100 |
| skip7000_to_skip11000 | 4 | 0.4570 | 0.4512 |
| skip7000_to_skip11000 | 8 | 0.3288 | 0.3183 |
| skip7000_to_skip11000 | 12 | 0.2511 | 0.2353 |
| skip7000_to_skip11000 | 16 | 0.1943 | 0.1728 |
| skip7000_to_skip11000 | 24 | 0.1344 | 0.0943 |
| skip7000_to_skip11000 | 32 | 0.1014 | 0.0326 |
| skip11000_to_skip7000 | 1 | 0.7056 | 0.7027 |
| skip11000_to_skip7000 | 2 | 0.6228 | 0.6189 |
| skip11000_to_skip7000 | 4 | 0.4976 | 0.4908 |
| skip11000_to_skip7000 | 8 | 0.3004 | 0.2851 |
| skip11000_to_skip7000 | 12 | 0.2375 | 0.2176 |
| skip11000_to_skip7000 | 16 | 0.1967 | 0.1689 |
| skip11000_to_skip7000 | 24 | 0.1366 | 0.0898 |
| skip11000_to_skip7000 | 32 | 0.1127 | 0.0441 |

The full source-role grid predicts the other role at NRE about 0.10, while the
16-term direct transfers are about 0.19--0.20. These are strong same-corpus,
disjoint-document transfers, but they are not a new corpus or semantic OOD test.

### Bootstrap-stable eight-term candidates

These terms are selected in at least 80% of 1,000 independent document
bootstraps on both roles when OMP is limited to eight terms:

| macro-contrast | skip7000 | skip11000 |
|---|---:|---:|
| MLP0 x MLP3 | 0.8860 | 0.9990 |
| MLP0+MLP1 x MLP3 | 0.9160 | 1.0000 |
| MLP1 x additional attention4-8 beyond attention3 | 0.9880 | 1.0000 |
| MLP1+MLP2 x attention3 | 1.0000 | 1.0000 |
| MLP2 x additional attention4-8 beyond attention3 | 1.0000 | 1.0000 |
| MLP2 x local attention3-MLP3 synergy | 0.9980 | 1.0000 |

These names inherit the suffix-aliasing warning above. An `additional MLP3-8
/ broad-block synergy` coefficient cannot distinguish a broad MLP main effect
from its interaction with the attention sites bundled in that mask.

## Claim boundary and next test

This analysis may nominate a sparse grammar, but every cell was already visible
when the diagnostic was designed. A genuine result requires freezing the support
rule and testing new suffix masks or an adjacent layer boundary. Token/logit-vector
outcomes should accompany CE so scalar averaging cannot hide incompatible behavior.
The cheapest de-aliasing test adds the missing MLP-only layers-3--8 suffix and
crosses it with all eight early prefixes: eight new masks on each role. Together
with the existing empty, attention-only, and all-sites columns, this completes
the broad attention-by-MLP square. Its support rule and gates must be frozen
before those new outcomes are opened.
