# Terminal copy/induction v1: prospective screening amendment

**Status:** prospective and outcome-blind. This document narrows the first E4 run; it
does not authorize model execution by itself.

## 1. Exact claim

This run tests whether a frozen attention-head candidate has a causal, behavior-specific
effect on terminal next-token copying. It is an **attention-only, copy-only E4.1/E4.2
screen**.

It does not test or claim:

- capitalization or number-formatting coverage;
- a late-MLP causal screen;
- a standalone extracted program;
- selective circuit removal or transplant;
- completion of E4.2's three-behavior requirement or E4.3.

The late MLP is left native in every arm. E4.3 is deferred until a behavior/site pair
passes this screen.

## 2. Frozen inputs and roles

Natural roles contain 192 distinct 257-token rows each:

1. `fit_natural`: fit-only token frequencies and per-position head-write means;
2. `selection_natural`: choose one candidate and no numerical threshold;
3. `final_natural`: one-shot in-distribution replication;
4. `ood_code`: 192 file-disjoint rows from tracked Python in this repository.

`ood_code` is only a **single-repository Python register shift**. It is not broad code
generalization evidence. Generated, vendored, cached, result, and experiment files must
be excluded; rows must be file-clustered and exact/normalized-near duplicates excluded.
If the repository cannot supply 192 eligible source files, the role freezes as a design
failure rather than silently weakening the population.

The exact eligible-path predicate keeps tracked `.py` files except paths under
`archive/`, `basis_aligned/bilinear_quotient/`, `basis_aligned/polynomial_causal/`,
`data_*`, `figures/`, `logs/`, `runs/`, `runs_*`, and `tests/`; any path component named
`__pycache__`, `generated`, `vendor`, `third_party`, `runlogs`, or `results`; and files
named `test_*.py` or `*_test.py`. File order is SHA256 order under the frozen code seed.
The audit binds a manifest hash over every eligible path, exact blob hash, and
normalized-Python hash.

For normalized-Python identity, tokenization drops comments, indentation, line breaks,
and encoding/end markers; every string literal becomes `STRING`, every numeric literal
becomes `NUMBER`, and other token text is retained. Tokenization failures fall back to
collapsed Unicode-replacement text. Equal normalized hashes are duplicates. Prior code
paths are found recursively in every registered JSON; their tracked blobs seed the
normalized exclusion set. New code rows are additionally exact-row and prefix32
disjoint from all prior rows and all three new natural roles.

Fit frequencies count query tokens over columns 0--255 and target tokens over columns
1--256 separately. Count zero has its own bin. All natural roles use those immutable
fit-only histograms.

## 3. Natural copy label and matched control

At scored position \(p\in\{64,\ldots,255\}\), let

\[
q=x_p,\qquad y=x_{p+1},\qquad
j=\max\{i<p:x_i=q\}.
\]

The cell is positive exactly when \(x_{j+1}=y\). Thus an older \(q\to y\) witness
does not override a nearer contradictory successor.

Positive and negative cells are matched on:

- 16-position query-position bin;
- base-2 bin of nearest-query distance;
- separate fit-only frequency bins for \(q\) and \(y\);
- base-2 bin of the number of earlier occurrences of \(q\).

Matching is deterministic and document-balanced. A retained stratum must contain at
least two source documents in each polarity. Before model access, each scored role must
retain at least 24 documents and 48 positions in each of the positive and matched-
negative cells. Failure aborts that role; there is no post-outcome coarsening or row
enlargement.

## 4. Causal estimand

For the same input positions, define

\[
\tau_+=\operatorname{CE}_{\mathrm{ablated},+}
       -\operatorname{CE}_{\mathrm{native},+},
\]

\[
\tau_-=\operatorname{CE}_{\mathrm{ablated},-}
       -\operatorname{CE}_{\mathrm{native},-},
\qquad
S=\tau_+-\tau_-.
\]

Here \(\tau_+\) is the causal copy effect and \(S\) is its matched-control specificity.
Matching is not itself a causal estimator. Native and ablated reductions must bind the
same exact row/position support digest; equal counts are insufficient.

CE, target log-probability, top-1 accuracy, native-to-arm KL, and off-target CE are
reported separately. No weighted sum of these currencies is used.

## 5. Intervention and candidate bank

For selected heads \(H\) in one layer, the physical residual write is

\[
w_{\mathrm{candidate}}
=w_{\mathrm{full,native}}-\sum_{h\in H}w_h+\mu_H(p),
\]

where \(\mu_H(p)\) is the fit-role mean selected-head write at query position \(p\).
The bit-identical unpartitioned full write is the base. The separately accumulated head
sum is used only for the subtraction and its measured bfloat16 discrepancy remains an
integrity control. For `L8H3+L8H4`, both heads are changed in one layer transaction.

The eight frozen candidates are the six individual heads
`L5H5`, `L7H3`, `L8H3`, `L8H4`, `L13H0`, `L14H7`, plus the registered four-head set
and the registered late pair. All other attention heads and all MLPs remain native.

Define the collateral margin

\[
C=0.01-\tau_{\mathrm{off}}.
\]

On `selection_natural`, a candidate passes only if simultaneous 95% document-bootstrap
lower bounds for \(\tau_+\), \(S\), and \(C\) are all positive. Thus the 0.01-nat
off-target rule is a population-level collateral gate, not merely an observed-support
point estimate. Among passers choose the largest lower bound for \(S\); ties use the
lexicographic candidate name. If none passes, E4 copy localization is negative and no
final role is opened.

The bootstrap resamples source documents, not positions or matched pairs. Point
estimates pool token losses across documents rather than averaging document means.
It uses 10,000 shared resamples, seed
`terminal-copy-v1-document-bootstrap:0`, and a one-sided simultaneous basic band over
the 24 positively oriented coordinates
\((\tau_+,S,C)\times 8\) candidates. For replicate \(b\),

\[
T_b=\max_j(\hat\theta^*_{bj}-\hat\theta_j).
\]

Sort all 10,000 values, take the 9,500th value (zero-based index 9,499) without
interpolation, and set \(\operatorname{LCB}_j=\hat\theta_j-T_{(9500)}\). A replicate
with a zero denominator fails the run; it is never dropped or redrawn.

After selection is sealed, `final_natural` and `ood_code` use independent
document-cluster draws within each role but one replicatewise maximum over the six
coordinates \((\tau_+,S,C)\times\{\text{final},\text{OOD}\}\). All six simultaneous
lower bounds must be positive for joint ID/OOD replication. Synthetic crossover scores
are descriptive in v1 and do not enter selection or the replication gate.

## 6. Reciprocal synthetic challenge

Each synthetic item crosses two associations:

\[
\{q\to y,\ r\to z\}
\quad\text{versus}\quad
\{q\to z,\ r\to y\},
\]

while holding length, token multiset, current query \(q\), and observed next token
fixed. At the current query, score both possible successors and compute

\[
D=
\left[\log p(y)-\log p(z)\right]_{q\to y,r\to z}
-
\left[\log p(y)-\log p(z)\right]_{q\to z,r\to y}.
\]

The four frozen position templates are `(8,32,80)`, `(12,44,96)`, `(20,52,128)`, and
`(28,60,160)`, interpreted as first-query, reciprocal-query, and current-query
positions. Token banks contain four distinct nonspecial IDs per item, are disjoint
across items and roles, and are absent from the unmodified base row.

This challenge supports only a claim about the **joint crossed association**. It does
not isolate one edge or test the reciprocal query as a separate outcome.

## 7. Opening order and stopping rules

The immutable order is:

1. publish row authorities and token-only labels;
2. publish the pre-outcome scientific authority;
3. run `fit_natural` and publish the mean-write receipt;
4. run all eight candidates on `selection_natural`, publish selection, then seal one;
5. only after a passer exists, open `final_natural`, `ood_code`, and their reciprocal
   synthetic challenges once;
6. publish result, manifest, and terminal receipt last.

Any support, identity, hook-cleanup, source-closure, namespace, or all-head integrity
failure stops the transaction. It cannot be repaired inside the same authority.
