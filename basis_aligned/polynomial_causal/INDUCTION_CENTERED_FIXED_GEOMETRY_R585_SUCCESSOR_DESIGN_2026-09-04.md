# Prospective design review: minimal fixed-geometry centered-factor successor to R585

Date: 2026-09-04 UTC

Evidence used: R591 FIT-only diagnostic log only

R591 log SHA-256:
`85403a5ed99e76734229d3063fa9cd666d005eb5d30507f74d4ad6b7a4257002`

Status: **design approved prospectively, conditional on a separately frozen
implementation and independent exact-byte review**

This review is CPU/design-only. It did not load the model or CUDA, inspect an R585
scientific outcome, enqueue work, or edit R590.

## What R591 requires

R591 isolated two independent numerical causes of the aborted R585 execution at the
unchanged absolute tolerance $10^{-5}$:

1. The old replay hook changed final logits by as much as
   $1.811981201171875\times10^{-5}$ even on the same batch. Locally, projecting the
   role values before summing differed from summing in head space before projection
   by as much as $9.1552734375\times10^{-5}$.
2. Changing the padded sequence length changed native final logits by as much as
   $2.8848648071289062\times10^{-5}$ on the controlled panel and
   $4.9114227294921875\times10^{-5}$ over FIT.

Fixed-shape batch membership and a read-only factor observer both changed the panel
logits by exactly zero. The decomposition of the total error into hook and
batch/padding terms closed to $4.547473508864641\times10^{-13}$.

Consequently, a successor must remove both causes. Fixing only padding or only the
hook is insufficient, and the $10^{-5}$ tolerance must not be widened.

## Frozen computation proposed

Keep the R585 row, endpoint, direction, semantic-role, target/control, bootstrap,
FIT-first, and scoring authorities unchanged. Change only the numerical execution
and the name of the intervention.

### One physical tensor geometry

Every scientific forward uses physical sequence width 30:

- every endpoint-capture batch has 32 endpoints and shape $[32,30]$;
- every full directed chunk has shape $[32,30]$;
- SELECT's final 16-direction chunk remains the registered partial batch, with shape
  $[16,30]$; no unregistered filler examples are invented; and
- a directed native baseline, literal-zero replay, and all three scientific arms use
  byte-identical input tokens, row order, batch size, padding, and query positions.

The hash of each complete token tensor must be saved before execution and checked
again in every paired call. “Batch size 32” means registered chunks of at most 32,
not permission to change the authority by filling a partial chunk. Within each chunk,
the native, replay, and three arm calls have exactly the same physical tensor shape.

### Read-only factor capture

At the four registered sites, capture for each endpoint and semantic role

$$
E_x=(e_{x,A},e_{x,C}),\qquad
U_x=(u_{x,A},u_{x,C}),
$$

and define the registered output-space equality factor

$$
B(E_x,U_x)=e_{x,A}u_{x,A}+e_{x,C}u_{x,C}.
$$

The capture dispatcher must return the untouched native attention write. Its logits
therefore serve as the canonical endpoint-native measurements used in the R585
denominators; the old second, length-sorted native-comparator pass is removed.

The capture must use the same physical sequence width 30 as the interventions. It
does not need the same neighboring examples or batch size as every directed chunk:
the model has no cross-example operation, and R591's fixed-shape membership contrast
was exactly zero. That observation is not enough to skip a full-population check,
however. Each directed native-baseline forward must re-observe the recipient factor
and compare its output-space value $B(E_x,U_x)$ with the endpoint cache. Every
endpoint is available in recipient position because each stored row registers both
directions, so this checks both eventual recipient and donor caches without another
forward. It also directly tests the final SELECT partial batch. A nonfinite value or
discrepancy above $10^{-5}$ is an instrument abort, not a scientific null.

Raw $E$ and $U$ differences should also be saved, but the hard comparison is on
$B(E,U)$ because that is the operational mediator and has one common residual-stream
unit. Separate coefficient and projected-content units must not be conflated.

### Centered intervention

For recipient $x$ and donor $y$, apply at each registered site:

$$
\begin{aligned}
\Delta_{\mathrm{coefficient}}&=B(E_y,U_x)-B(E_x,U_x),\\
\Delta_{\mathrm{projected\ content}}&=B(E_x,U_y)-B(E_x,U_x),\\
\Delta_{\mathrm{joint}}&=B(E_y,U_y)-B(E_x,U_x).
\end{aligned}
$$

The native attention write is otherwise untouched and receives only the sum of
these four site deltas at the registered query position. Self interchange must be
constructed as `zeros_like(B(E_x,U_x))`, not by subtracting two separately evaluated
nominally equal contractions.

Following handoff v7, the arms are named **registered equality-factor coefficient
swap**, **registered projected-content swap**, and **registered joint output-factor
swap**. They are interventions on a partial output-space mediator. They are not
called an attention-score swap, value swap, complete attention-pattern swap,
realizable query/key-state swap, or literal native remove-and-insert.

In particular, the observed native equality contribution $C_x$ may differ in
floating-point arithmetic from $B(E_x,U_x)$. The centered operation is

$$
h_x+B_{\mathrm{new}}-B(E_x,U_x),
$$

not $h_x-C_x+B_{\mathrm{new}}$. The old $B(E_x,U_x)-C_x$ discrepancy must remain
reported, and a value above $10^{-5}$ continues to rule out the stronger literal
remove-and-insert claim. It is no longer injected into the model and is not silently
reclassified as success.

## Why the directed baseline must be paired

R585 previously measured an arm in directed-ID batches but subtracted a replay
measurement made in an endpoint-ID batch. R591 showed membership was null on a
256-endpoint panel at fixed shape, but it did not establish that identity over all
5,616 directions, all factor tensors, or SELECT's half batch. At a $10^{-5}$
instrument threshold, that limited null cannot license an unpaired causal contrast.

Therefore each directed chunk gets both a read-only native forward and a centered
self-replay forward. They use exactly the same token tensor as the three scientific
intervention forwards. Self replay executes the repaired hook path while applying a
bitwise-zero delta. Its full vocabulary logits must agree with the paired native
forward within $10^{-5}$ for every direction; otherwise execution hard-aborts. After
that check, each arm's numerator, CE change, and vocabulary-wide change are computed
relative to self replay, so the only intended difference is the registered inserted
delta. The canonical endpoint captures remain the source of donor-versus-recipient
native denominators.

The replay forward is necessary. Merely constructing a zero tensor in CPU tests does
not test dispatcher selection, site coverage, write indexing, dtype conversion, or
the end-to-end effect of the hook. R591 was needed precisely because an
algebraically-null-looking replay path was not numerically null. Keeping native and
self replay as separate paired calls closes that failure mode without changing the
tolerance.

Reusing the endpoint-capture logits as directed baselines would save forwards, but
would leave an avoidable geometry term inside the claimed causal effect. It is not
the minimal *valid* successor.

## Exact forward price

The unchanged authority contains 1,728 FIT endpoints, 3,744 FIT directions, 864
SELECT endpoints, and 1,872 SELECT directions. With batch size 32:

$$
\begin{aligned}
\mathrm{FIT}
&=\left\lceil\frac{1728}{32}\right\rceil
 +2\left\lceil\frac{3744}{32}\right\rceil
 +3\left\lceil\frac{3744}{32}\right\rceil\\
&=54+234+351=639,\\[4pt]
\mathrm{SELECT}
&=\left\lceil\frac{864}{32}\right\rceil
 +2\left\lceil\frac{1872}{32}\right\rceil
 +3\left\lceil\frac{1872}{32}\right\rceil\\
&=27+118+177=322.
\end{aligned}
$$

Thus a FIT instrument/science stop costs exactly 639 forwards and a complete
FIT+SELECT execution costs exactly 961. There are zero backward passes and zero
updates. Relative to R585's 459/231 schedule, exact native/self pairing adds two
117/59 directed passes but removes the obsolete 54/27 length-sorted comparator, a
net increase of 180 FIT, 91 SELECT, and 271 maximum forwards.

## Opposing predictions and kill conditions

The successor distinguishes the proposed factorization from numerical artifacts as
follows:

1. **Geometry prediction.** At fixed sequence width 30, endpoint-capture and
   directed native measurements for the same endpoint, and cached versus
   directed-live $B(E,U)$ values, remain within $10^{-5}$, including SELECT's final
   $[16,30]$ chunk. A larger discrepancy kills the instrument; it is not repaired by
   averaging or a larger tolerance.
2. **Zero prediction.** Every self-interchange delta is bitwise zero, a read-only
   observer returns the native write unchanged, and paired self-replay/native full
   logits differ by at most $10^{-5}$. Any failure kills the instrument.
3. **Coefficient/content separation.** On registered selector-changing families,
   the registered equality-factor coefficient arm should transfer the selected
   answer more strongly than the projected-content arm. On payload-changing,
   match-preserving families, the projected-content arm should transfer the payload
   more strongly than the coefficient arm. Equal broad movement by both arms argues
   for shared difficulty or damage, not the proposed decomposition.
4. **Composition prediction.** The joint arm should obey the frozen bilinear
   identities and the registered joint-diagonal interaction tests. Failure of the
   mixed identity or absence of the registered interaction rejects the factorized
   composition account.
5. **Selectivity prediction.** Native denominators, active controls, answer-preserving
   controls, vocabulary-wide change, and FIT-to-SELECT replication retain their
   frozen R585 gates. Target transfer accompanied by broad control damage is not a
   circuit identification result.
6. **Scope prediction.** A held result licenses only control of the registered
   output-space equality factor. Literal removal, complete-pattern sufficiency,
   query/key realizability, and OOD generalization remain unestablished until their
   separately named interventions pass.

## Implementation freeze requirements

Before any successor model run, a prospective amendment should hash-bind this
design, R591's exact log, the v7 addendum, the complete authority and dependency
closure, the exact per-chunk token-tensor manifest, the new 639/322 price, and the
evidence schema. Independent tests should plant a changed pad width, an unauthorized
filled partial batch, reordered directed batch, cache/live factor mismatch, unpaired
native/replay, nonzero self delta, old `term-canonical` hook, and a forbidden stronger
claim. Publication
remains receipt-last and FIT-first; FINAL and OOD remain closed.

Reference hashes:

- R591 findings: `99c01c5efc03d3011dd562636a41a38ad181a7b35d4d9ab37a1da69ce26f425f`;
- handoff v7 addendum: `595b43156117e0ba2e568972f76af81ac4e716ed5537861ac48f13b23d4ed9fd`;
- centered derivation: `afb816361603d880dea8dd5daa30b90e841f686d4935da8684ac78c3839a78ca`;
- independent centered review: `375ba4bb36655caf1807f978ff38b1aee0f85adc1e82f335663f30a00cf3eec0`.
