# Prospective MLP2 causal-mechanism-reduction experiment

Frozen on 2026-08-29 before assigning document roles, tokenizing rows, capturing
MLP2 products or suffix responses, or opening any MLP2 CMR outcome.

## Question and exact executable

Can 512 of MLP2's 4,608 native bilinear products preserve the downstream behavior
of MLP2 when the omitted products are replaced by their fit-set means and those
constant writes are folded into the bias?

For MLP2,

$$
a_j(x)=(L_jx)(R_jx),\qquad y(x)=b+\sum_jD_{:j}a_j(x).
$$

Fit $\mu_j=\mathbb E_{\rm FIT}[a_j]$.  For a retained set $K$ of exactly 512
channels and omitted set $S$, the candidate is

$$
y_K(x)=b+D_{:S}\mu_S+\sum_{j\in K}D_{:j}a_j(x).
$$

The executable must physically use only the 512 retained rows of `Left` and
`Right`, the 512 retained columns of `Down`, and the folded 1,152-value bias.  It
may not call native MLP2 or compute then mask all 4,608 products.  Its fixed-grammar
price is $3456\cdot512+1152=1,770,624$ scalar values, 512 bilinear products per
token, support indices, and declared coefficient precision.

This first experiment uses the native upstream trajectory.  Native success does not
establish composition with C512; that is a later frozen $2\times2$ cross using the
same MLP2 program.

## Fresh roles and statistical unit

Use four source-document-disjoint FineWeb roles, 192 documents each:

1. `FIT_MEAN`: fit $\mu$ only;
2. `FIT_SELECTOR`: fit selector scores/support only;
3. `VALIDATION`: first finite consequence test;
4. `REPLICATION`: opened only if the frozen validation gate passes.

Only positions 64--255 are scored.  The primary paired uncertainty unit is the
source document.  Validation reports nested 48- and 96-document prefixes before the
full 192-document result so a conclusion that reverses under doubling is explicit.
All selectors share identical documents, logits, probes, precision, and price.

## Five frozen selectors

Write the mean-centered native MLP2 map as

$$
y(x)=b+D\mu+\sum_jD_{:j}(a_j(x)-\mu_j).
$$

For a suffix probe $s_{c,p}$, define the trajectory-complete response

$$
E_{(c,p),j}=\sum_q(a_j(x_{c,q})-\mu_j)
D_{:j}^{\mathsf T}\frac{\partial s_{c,p}}{\partial y_{c,q}}.
$$

The sum over every token position is mandatory because the same product is deleted
everywhere and attention couples positions.

- **SUFFIX:** select 512 columns by context-balanced ridge leverage of $E$, with
  target response rank 256 and the same numerical rank/ridge rules used by the
  completed MLP1 global-gate assay.
- **LOCAL:** select the largest 512
  $\operatorname{Var}(a_j)\lVert D_{:j}\rVert_2^2$ scores.
- **RMS:** select the largest 512
  $\mathbb E[a_j^2]\lVert D_{:j}\rVert_2^2$ scores.
- **MASS:** select the largest 512
  $\lVert L_j\rVert_2^2\lVert R_j\rVert_2^2
  \lVert D_{:j}\rVert_2^2$ scores.
- **RANDOM/DERANGED:** a hash-random support and a fixed-point-free
  factor/product derangement, both at 512 channels.

LOCAL, RMS, and MASS are immediate-write or weight scores—not final-logit risk.
SUFFIX is a tangent selector—not a finite-deletion certificate.

## Finite measurements and certificate

Evaluate native, zero-MLP2, and every compiled candidate on identical rows.  Record
paired CE, teacher KL, centered final-logit NRMSE, top-1 agreement and accuracy,
frequency strata, copy-positive/repeat-negative/nonrepeat cells, and worst-document
harm.  Measure actual joint squared post-softcap-logit distortion $D_2$.

Choose an $\epsilon$ grid using `FIT_SELECTOR` only.  On evaluation rows report

$$
\Pr(\text{top-1 mismatch})\leq
\Pr(m_{\rm native}\leq2\epsilon)+D_2/\epsilon^2
$$

using actual native margins and actual finite compiled logits.  No local MLP-write
quantity may substitute for $D_2$.

Also run signed endpoint fractions $0.1$ and $0.25$.  Their observed final-logit
direction must agree with the suffix tangent before a full deletion receives causal
credit.  Report true joint local write distortion versus the sum of singleton LOCAL
scores; material disagreement rejects singleton additivity, not constant folding.

## Frozen decision gates

SUFFIX512 advances only if all hold on validation and then independently on
replication:

1. its document-bootstrap simultaneous lower confidence bound gives at least 5%
   lower teacher KL than every equal-price control;
2. $|\Delta\mathrm{CE}|\leq0.02$, teacher KL $\leq0.02$, centered final-logit
   NRMSE $\leq0.10$, and no registered cell has more than 0.02 nat collateral CE;
3. top-1 agreement is at least 0.90 and the best fit-frozen margin-certificate lower
   bound is also at least 0.90;
4. signed small edits agree with the tangent direction and full deletion does not
   reverse it;
5. reciprocal native-channel rescaling/permutation exactly replays selector ranks
   and the materialized compiled function within the registered numerical tolerance;
6. no native MLP2 call occurs in the candidate and complete price/source/call
   receipts replay.

If SUFFIX beats controls but singleton additivity fails, the successor is block CMR.
If constant folding fails absolute consequence gates, prune 512-channel native-basis
compression at MLP2 and move to response-conditioned factors or a larger retained
budget.  If it passes natively, cross the unchanged program with C512 before any
whole-model composability claim.

