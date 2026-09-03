# Rung 536 Stage-B preregistration: label and power preflight before real-model product-space DAS

**Registered:** 2026-09-03 13:34 UTC; corrected before model execution at 13:36 UTC

**Owner:** Codex

**Status:** planning calculation complete; label-boundary audit and portable-target implementation begin; real DAS
fitting remains closed

## Correction made before execution

The first draft proposed applying circuit `r.2.0.2` to 4,888 new FineWeb documents. That is not a valid operation.
The 62 circuit masks are stored as member/control indices on the fixed 1,000-row census, not as portable semantic
label functions. Moreover, `r.2.0.2` has only a weak surface program (balanced accuracy 0.693) and no surviving
behavioral interpretation in its dossier. No model run used the invalid proposal.

This preregistration now keeps two authorities separate:

1. Fresh documents may be used only with portable quantities computed directly from the model and input, initially
   MLP0's exact token-only branch $T\equiv\mathrm{TT}$ and token-by-context branch $I\equiv\mathrm{X}$.
2. The 62 circuit masks may be used only on their frozen census rows: 32 discovery masks for secondary response
   fingerprints and 30 masks for held-out circuit evaluation.

## Why a power gate is necessary

The completed MLP0 49-term probe fit document half 0 exactly but localized 0/32 circuits on half 1. Its per-term
target-circuit effect pattern had split-half correlation $r=0.10594$ using 124 documents per half. Product-space DAS
could pool signal differently, but optimizing a 4,608-dimensional subspace against an unstable target would still
fit sampling noise.

Under the Spearman--Brown planning model,

$$
r_m=\frac{mr}{1+(m-1)r},
$$

where $m$ multiplies independent sample count. The old $r$ implies approximately 2,096 total documents for expected
reliability 0.5, 4,888 for 0.7, and 8,376 for 0.8. These are planning estimates, not guarantees. The 4,888-document
number is a provisional data budget, not authorization to fit DAS.

## Stage B1: exact label-boundary audit

Before any backward pass, the implementation must prove:

- the fresh-document target is computed by the exact deployed $T/C/I/S$ MLP0 decomposition and is not derived from
  old census indices or an unvalidated classifier;
- the discovery and held-out circuit masks are evaluated only on their original row authority;
- document, token, and split hashes are saved;
- the product activation has shape $({\rm batch},{\rm tokens},4608)$;
- the proposed donor/base construction changes $T$ or $I$ in the intended direction on a planted minibatch.

Failure closes Stage B without repair-by-threshold.

## Stage B2: portable gradient-reliability screen

For each preregistered target $q\in\{T,I\}$, define a signed loss contrast $L_q$ from its exact removal or exchange
intervention. At MLP0's 4,608-dimensional product activation $g$, average its gradient separately in two fixed
2,444-document halves:

$$
v_{q,h}=\mathbb{E}_{d\in h}\!\left[\nabla_g L_q(d)\right].
$$

This vector is only a first-order diagnostic of whether the target supplies a reproducible learning signal; it is
not a circuit, a replacement, or evidence of low dimension. The exact $L_T$ and $L_I$ donor/base construction,
token supports, forward/backward price, and positive controls must be frozen in an implementation addendum before
the first model backward pass.

The two targets are reported separately; no best-target selection is allowed. For each target:

- **A — valid instrument:** all hashes, supports, shapes, liveness checks, positive controls, and exact call counts
  reconcile; each half has the preregistered minimum support.
- **B — stable learning signal:**

  $$
  \operatorname{cos}(v_{q,0},v_{q,1})\ge 0.70,
  \qquad
  0.5\le\frac{\lVert v_{q,0}\rVert_2}{\lVert v_{q,1}\rVert_2}\le2.0.
  $$

- **C — causal specificity:** the same gradient or eventual projector must predict the registered $q$ intervention
  better than dimension-matched random directions, shuffled donor/base labels, and native-unit subsets. The precise
  statistic and threshold belong in the implementation addendum because they depend on the exact target construction.

If A fails, repair only the instrument. If A passes and B fails, real-model product-space DAS for that target is not
authorized; more optimization steps, ranks, or seeds cannot repair an unstable learning signal. If B passes, freeze
a separate DAS preregistration before fitting. That later registration must specify the dimension ladder, held-out
documents and code OOD, the 32/30 frozen-census circuit split, random/native controls, cross-seed projector stability,
weight-compilation equivalence, selective removal and exchange, joint installation, and literal storage and compute.

This gate makes no compression or identification claim and adds no deployed parameters.
