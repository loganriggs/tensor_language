# Independent review: R592 logit-topology amendment

**Reviewed:** 2026-09-04 UTC, prospectively and before any repaired R592 execution or outcome

**Amendment commit:** `0f33456126fa2ea5233798d937e61f7dd6a0ea93`

**Amendment SHA-256:** `15219749dd1d696e52c3129052cadce6758b7186390303eace216d98c953188e`

**Verdict:** **APPROVED as the exact narrow specification repair**

This approval applies only to the topology/evidence amendment. It does not approve an R592 implementation or authorize
model execution. The five non-topology implementation blockers from the review at commit `1c07919a4` remain open.

## Independent topology check

The pinned facade bytes have SHA-256
`b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c`. Their model-free syntax fixes:

- tokenizer-facing vocabulary: 50,257;
- checkpoint/logit vocabulary: 50,304;
- expected checkpoint configuration `vocab_size`: 50,304; and
- unsliced return of the full checkpoint logit tensor from `forward_with_dispatch`.

R592 measures model outputs, native/replay equality, structural identities, and whole-vocabulary RMS. Therefore 50,304
is the correct last dimension. The amendment explicitly forbids slicing to the tokenizer prefix, padding a shorter
vector, or excluding any checkpoint output coordinate.

## Independent byte arithmetic

One float32 full-logit row costs

$$
50{,}304\times4=201{,}216\ \text{bytes}.
$$

Thus a 32-row call contains 6,438,912 data bytes and the 16-row SELECT tail contains 3,219,456 data bytes. For the
four replay-relative differences per direction:

$$
\begin{aligned}
\mathrm{FIT} &= 3{,}744\times4\times50{,}304\times4
             = 3{,}013{,}410{,}816,\\
\mathrm{SELECT} &= 1{,}872\times4\times50{,}304\times4
                = 1{,}506{,}705{,}408,\\
\mathrm{combined} &= 4{,}520{,}116{,}224\ \text{bytes}.
\end{aligned}
$$

The unchanged hook arrays total 414,056,448 bytes, and the unchanged directed live projected-content arrays total
207,028,224 bytes. The corrected principal maximum is therefore

$$
4{,}520{,}116{,}224+414{,}056{,}448+207{,}028{,}224
=5{,}141{,}200{,}896\ \text{bytes}.
$$

These are data bytes and intentionally exclude NumPy headers, JSON, receipts, and filesystem metadata. The amendment's
increases over the old 50,257-wide calculations—2,815,488 FIT, 1,407,744 SELECT, and 4,223,232 combined—also recompute
exactly.

## Coverage of the correction

The supersession is complete and consistent. It changes every object whose last axis is a model logit coordinate:

- per-call endpoint, native, replay, score, payload, and joint logits become `[b,50304]`;
- complete logit differences become `[N_d,4,50304]` in the unchanged difference order;
- native/replay and structural equalities cover all 50,304 coordinates;
- every `vocab_size` is integer 50,304;
- RMS divides the float64 squared-difference sum by 50,304;
- FIT-frozen RMS scales inherit the corrected definition; and
- a nonfinite logit mask has the same `[b,50304]` shape as its raw array.

The commit changes exactly one new specification file and no executable bytes. Its boundary expressly preserves all
rows, endpoints, directions, sites, roles, arms, centered formulas, support and transport checks, tolerances, target and
control cells, bootstrap identities and draws, scientific thresholds, FIT-first rule, terminal precedence, publication
rules, claim scope, and model-call accounting. The price remains 639 FIT, 322 SELECT, 961 maximum, zero backwards, and
zero updates; FINAL and OOD remain closed.

## Approval scope and next gate

The exact amendment is internally sufficient to repair the frozen topology mistake and may be used by a prospective
implementation repair. That implementation must still independently close and be reviewed for:

1. authoritative static-gate acceptance;
2. hard abort rather than diagnostic publication when mandatory observations are missing;
3. content binding of every invalid-evidence byte;
4. raw evidence that independently reconstructs the native-attention hard gate; and
5. explicit fsync of completed memmap evidence.

No implementation, model, Torch, checkpoint, CUDA/GPU, queue, split, bootstrap, or outcome was opened by this review.
