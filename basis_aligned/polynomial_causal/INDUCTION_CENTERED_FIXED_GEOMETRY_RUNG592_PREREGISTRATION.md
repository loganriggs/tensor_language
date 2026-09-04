# Rung 592 preregistration: centered equality-factor interchange at fixed geometry

**Frozen:** 2026-09-04 UTC, before any R592 implementation, model call, or outcome

**Status:** prospective successor to the numerically invalid R585 instrument

## Question and claim boundary

R592 asks whether the registered equality-copy contribution of four fixed induction sites can be separated into an
attention-coefficient factor and a projected-content factor that transfer in the predicted ways on held-out prompts.
The sites, rows, directions, semantic roles, target/control families, FIT-first split rule, bootstrap cells, scientific
thresholds, and nulls are inherited unchanged from the R585 replacement authority.

This is an **output-space equality-factor interchange**. A held result would identify causal control of the registered
partial output factor under these counterfactuals. It would not establish a complete attention-pattern swap, a
query/key state that realizes the swapped coefficients, literal removal and reinsertion of the native attention term,
individual-head necessity, circuit sufficiency, FINAL/OOD generalization, or a smaller executable model.

The inherited scientific authority is frozen by these exact SHA-256 hashes:

- R585 replacement amendment: `98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf`;
- R585 row/scoring manifest: `7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962`;
- R591 findings: `99c01c5efc03d3011dd562636a41a38ad181a7b35d4d9ab37a1da69ce26f425f`;
- handoff version 7: `595b43156117e0ba2e568972f76af81ac4e716ed5537861ac48f13b23d4ed9fd`;
- centered-factor derivation: `afb816361603d880dea8dd5daa30b90e841f686d4935da8684ac78c3839a78ca`;
- independent centered-factor review: `375ba4bb36655caf1807f978ff38b1aee0f85adc1e82f335663f30a00cf3eec0`;
- fixed-geometry successor design: `075ae15bf31bc8ef4da625e3499ff125bc35225f6ab0846fa5bef45773876ad9`.

No R585 scientific result exists, and no R585 outcome is an authority for R592. R591 was a FIT-only numerical
diagnostic and is used only to repair the instrument.

## Why R585 cannot be interpreted

R591 separated R585's replay/native discrepancy into two measured terms at the unchanged absolute final-logit
tolerance $10^{-5}$:

1. replaying the nominal equality contribution with a different floating-point contraction order changed same-batch
   logits by as much as $1.811981201171875\times10^{-5}$; and
2. changing the physical padded length changed untouched-model logits by as much as
   $2.8848648071289062\times10^{-5}$ in the controlled panel.

Their vector sum reproduced the total discrepancy with maximum residual
$4.547473508864641\times10^{-13}$. Fixed-shape batch membership and a read-only factor observer both changed logits by
exactly zero. R592 repairs the two active causes rather than widening the tolerance.

## Exact factor computation

For endpoint $x$, site $h\in\{\mathrm{L5H5},\mathrm{L7H3},\mathrm{L8H3},\mathrm{L8H4}\}$, and registered equality
source role $r\in\{A,C\}$, define

$$
e^x_{h,r}=p_h^x(q_x,k_x(r))\,\mathbf 1[t^x_{k_x(r)-1}=t^x_{q_x}],
\qquad
u^x_{h,r}=W_h^O v_h^x(k_x(r))\in\mathbb R^{1152}.
$$

$e$ is the continuous attention coefficient on a registered matching source. $u$ is the source's 128-dimensional
value after projection by the head output matrix into the 1,152-dimensional residual stream. The registered partial
output contribution at one site is

$$
B_h(E_x,U_x)=\sum_{r\in\{A,C\}} e^x_{h,r}u^x_{h,r}\in\mathbb R^{1152}.
$$

For recipient $x$ and donor $y$, R592 adds one of these centered changes to the untouched native head write:

$$
\begin{aligned}
\Delta_h^{\mathrm{coefficient}}&=B_h(E_y,U_x)-B_h(E_x,U_x),\\
\Delta_h^{\mathrm{projected\ content}}&=B_h(E_x,U_y)-B_h(E_x,U_x),\\
\Delta_h^{\mathrm{joint}}&=B_h(E_y,U_y)-B_h(E_x,U_x).
\end{aligned}
$$

The replay condition is constructed as `zeros_like(B_h(E_x,U_x))`, not by subtracting two separately evaluated
expressions. At L8H3 and L8H4, both changes are computed from the same pre-modification layer-8 state and applied in one
transaction. Semantic roles, not absolute token positions, align donor and recipient factors.

The exact bilinear interaction is

$$
\Delta_h^{\mathrm{mixed}}
=\Delta_h^{\mathrm{joint}}
-\Delta_h^{\mathrm{coefficient}}
-\Delta_h^{\mathrm{projected\ content}}
=\sum_r(e^y_{h,r}-e^x_{h,r})(u^y_{h,r}-u^x_{h,r}).
$$

This identity must be reconstructed from saved row-level factors. It is the part of the write that requires changing
both factors together; it must not be assigned to either single-factor arm.

## Fixed physical execution

Every scientific forward uses token tensors padded to physical sequence width 30. There is no synthetic filler and no
duplicate row:

- FIT endpoint capture: 1,728 endpoints in 54 batches of shape $[32,30]$;
- FIT directed calls: 3,744 directions in 117 batches of shape $[32,30]$;
- SELECT endpoint capture: 864 endpoints in 27 batches of shape $[32,30]$; and
- SELECT directed calls: 1,872 directions in 58 batches of $[32,30]$ plus the registered final batch $[16,30]$.

For each directed batch, the following five calls receive byte-identical tokens, row order, batch size, padding, and
query positions:

1. untouched native;
2. literal-zero self replay;
3. coefficient interchange;
4. projected-content interchange; and
5. joint interchange.

The complete token tensor is hashed before the first call and verified before every paired call. The read-only endpoint
capture returns the untouched native attention write. Each later directed native call re-observes the recipient factor;
its $B_h(E_x,U_x)$ must match the cached value within $10^{-5}$ at every site. Because every registered row appears in
both directions, every endpoint is checked as a recipient.

Native and zero-replay full-vocabulary logits must agree elementwise within $10^{-5}$ for every directed example. The
scientific arms are compared with their paired zero-replay condition. Any geometry, cache/live, factor, hook,
nonfinite, support, or replay mismatch is `invalid_instrument`, not a scientific null.

The native equality write and $B(E_x,U_x)$ are also compared and reported. A difference above $10^{-5}$ continues to
block the stronger claim of literal removal and reinsertion, but R592 does not inject that difference into the model.

## Scientific predictions and nulls

All R585 cell definitions and numerical gates remain unchanged; the old `score` arm is now named `coefficient`, and
the old `payload` arm is now named `projected_content`. In plain language:

- On selector-changing prompts, coefficient and joint interchange should move the answer toward the donor, while
  projected-content interchange is the registered no-op where the content is unchanged.
- On payload-changing prompts with the match structure preserved, projected-content and joint interchange should move
  the answer toward the donor, while coefficient interchange must remain small.
- On joint selector-and-payload changes, the measured joint effect must satisfy the saved bilinear interaction identity
  and the inherited joint-composition gates.
- On match-breaking and active answer-preserving controls, the inherited signed effects, activity requirements, and
  selectivity limits apply without pooling directions or recipient conditions.
- Broad vocabulary damage, failed active controls, nonpositive native denominators, or failure to repeat the FIT result
  on SELECT prevents an identification claim.

The scientific null is the inherited R585 null: the factors do not transfer selectively under the registered
counterfactuals, even when the instrument is valid. Instrument failure and scientific null remain separate terminal
classes. FIT is scored first; SELECT is opened only if all registered FIT validity and scientific gates allow it.
FINAL and OOD remain closed.

## Literal price

Endpoint capture costs one pass over endpoints. Every directed batch costs two controls (native and zero replay) plus
three scientific arms:

$$
\begin{aligned}
\mathrm{FIT}&=54+5(117)=639,\\
\mathrm{SELECT}&=27+5(59)=322,\\
\mathrm{maximum}&=961.
\end{aligned}
$$

There are zero backward passes and zero weight updates. This is a causal identification experiment, not a proposed
compressed implementation, so the 961 forwards are an experimental cost rather than the storage price of a resulting
program.

## Before any GPU execution

The implementation must be written prospectively, hash-bind its complete executable dependency closure before import,
execute immutable verified bytes, and remain outcome-blind in every dry-run call path. Independent tests must plant at
least these failures: changed pad width, reordered or filled partial batch, cache/live factor drift, nonzero self delta,
unpaired native/replay, old contraction-order subtraction, omitted site or role, same-layer ordering drift, wrong arm
name or claim scope, incomplete row/bootstrap evidence, and atomic-publication failure.

An independent reviewer must reconstruct the 639/322 call schedule, exact row and operation censuses, factor formulas,
scientific decisions, and strongest licensed claim from immutable committed bytes. Only an approved exact commit may be
sent to the managed GPU queue.
