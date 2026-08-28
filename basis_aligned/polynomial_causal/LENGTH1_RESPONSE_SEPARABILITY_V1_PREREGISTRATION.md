# Length-1 response separability v1 — preregistration

## Status and scope

This document freezes a non-promotive response-law test before any site-by-amplitude
curve toward the deployed length-1 table is observed. It licenses no final-role, OOD,
semantic, edit, removal, or whole-program credit.

The source hypothesis is the committed empirical-token-mean analysis
`substitution_response_separability_v1_results.json`, source SHA-256
`7e620ee9a6be5023c746e2728f8b9a84d6af3d2791f75cbb15e1f463dee6a6db`.
That source used a different table object. Its numerical law is frozen here rather
than refit to the length-1 outcome.

## Object

For all 34 sites \(s\in\{\mathrm{attn}1,\mathrm{mlp}1,\ldots,
\mathrm{attn}17,\mathrm{mlp}17\}\), use the exact deployed full-rank length-1 row bank
from the settled context-free/output-nearest-neighbour/rank-64 program. At amplitudes

\[
\alpha\in(1,0.9,0.75,0.5,0.25,0),
\]

replace the site output on compiler-covered positions by

\[
y_s(\alpha)=r_s^{\mathrm{L1}}(t)
 +\alpha\bigl(y_s-r_s^{\mathrm{L1}}(t)\bigr).
\]

All other sites remain live. Uncovered tokens remain live. Every arm uses the same
documents, scored suffix, denominator, row order, checkpoint, and table bytes.

For each document retain paired CE sufficient statistics and define

\[
D_{s,\alpha}=\operatorname{CE}(y_s(\alpha))-
\operatorname{CE}(y_s(1)).
\]

Top-1 is descriptive only. CE is primary because small amplitudes may not cross argmax
thresholds.

## Frozen prediction

The empirical-mean response gave the anchored law

\[
f(\alpha)=(1-\alpha)^p,\qquad p=2.882399707788177.
\]

No exponent, intercept, site weight, or threshold may be refit for the primary test.
Using only the observed \(\alpha=0.25\) response, predict full replacement by

\[
\widehat D_{s,0}=
\frac{D_{s,0.25}}{0.75^{2.882399707788177}}.
\]

Primary pass requires all of:

1. Spearman rank correlation between \(\widehat D_{s,0}\) and \(D_{s,0}\) at least
   0.95 across the 34 sites;
2. relative \(\ell_2\) error at most 0.35;
3. a document bootstrap lower 95% bound on the Spearman correlation above 0.90;
4. no component class has relative \(\ell_2\) error above 0.45 when attention and MLP
   sites are scored separately.

## Secondary tensor diagnostic

Form the 34-by-5 matrix with columns
\(\alpha=(0.9,0.75,0.5,0.25,0)\). Report its complete singular spectrum without
threshold tuning. The preregistered secondary prediction is at least 98% rank-one
Frobenius energy. Also report attention-only, MLP-only, layers 1–6, and layers 7–17
rank-one energy.

This secondary statistic is descriptive unless the primary frozen predictor passes;
a matrix can have high global rank-one energy merely because its largest column
dominates the norm.

## Controls and failure policy

- \(\alpha=1\) must reproduce the paired live CE to numerical tolerance \(10^{-6}\).
- The length-1 table hash, coverage hash/count, model hash, row-role receipt, suffix
  denominator, site order, and amplitude order must be recorded before outcomes.
- Exactly the frozen covered positions may change. Uncovered positions must remain
  byte-identical to live outputs.
- Per-document sufficient statistics must be retained before aggregation.
- Curves need not be monotone: S1840 showed small beneficial shrinkage, so monotonicity
  is reported rather than used as a validity gate.
- Any source, coverage, support, alpha-1, denominator, or publication failure makes the
  result void. A failed scientific prediction with controls true is a negative result.

## Consequence

A pass licenses a compact scalar response model—one site amplitude and one shared
nonlinear link—for this in-domain length-1 intervention. It does not identify the
site's semantic computation or prove an internal one-dimensional state. A failure
prunes transfer of the empirical-mean rank-one law and returns priority to the
finite-amplitude cut-rank and 68-action causal response tensors.
