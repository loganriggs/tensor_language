# Rung 502 preregistration — exact source-pair anatomy of MLP9's copy-score response

## Question and claim level

Rung 501 recovered the calibrated `L5H5 score -> L8H4` relation in every discovery partition and found no additional
directed score edge that met all of the task, MLP9, payload, and copy-specificity conditions. The next question is:

> Which pairs of actual earlier residual-stream writes are multiplied inside MLP9 to produce its known response to
> that shared copy score?

This rung is an exact local decomposition plus a downstream-use screen. It may identify a frozen source-pair group
for a later finite removal experiment. It cannot by itself establish a circuit, an interchangeable implementation,
or a compressed model.

## Frozen authority and data

- Model, rows, copy masks, equality factors, and finite actions are inherited hash-exactly from rungs 498--501.
- Rung 501 result SHA256 is `b17a9b274e4c61e0b4a3fc68d8ce84ec6f8e76f257c3d898e6a6990492301c4f` and
  bundle SHA256 is `728d9be2681a60579b743626bb8eb7e8cc09414fdb9c90cf128388f2049f59c5`.
- Rung 501 must have A/B true, C false, validation closed, and exactly one confirmed edge:
  `L5H5->L8H4`.
- Discovery uses documents `0:500`. Source-pair selection uses `0:250`; `250:500` confirms without reselection.
- The 32 circuit tags already assigned to discovery in `circuits/BATTERY.json` are used only after their member and
  matched-control masks pass the frozen support checks. The 30 validation tags and documents `500:1000` remain
  closed. A later finite intervention, not this gradient screen, may open them.

## The exact 210 source pairs

Immediately before MLP9, the unnormalized residual is an exact weighted sum of 20 semantic sources:

1. the normalized token embedding/skip stream `E`;
2. output-projected attention writes `A0,...,A9`; and
3. bilinear MLP writes `M0,...,M8`.

The scalar residual-mixing coefficients from every intervening block are included in each source. For each forward
state, let `r=sum_s r_s` be that unnormalized residual and let `z` be the actual deployed normalized MLP9 input. Fit
the single least-squares gain `g=<z,r>/<r,r>` per token and define `z_s=g r_s`. Retain

`z_num = z - sum_s z_s`

exactly; it is a numerical accounting term, not a semantic source.

For MLP9's bilinear map

`q(z)=Down9[(Left9 z)*(Right9 z)] + bias9`,

define one unordered semantic term for every `s<=t`:

- `Q(s,s)=Down9[(Left9 z_s)*(Right9 z_s)]`;
- `Q(s,t)=Down9[(Left9 z_s)*(Right9 z_t) + (Left9 z_t)*(Right9 z_s)]` for `s<t`.

There are `20*21/2 = 210` terms. A separately reported `NUMERICAL` group contains every term involving `z_num`.
The 210 semantic terms, the numerical group, and the unchanged bias must reconstruct the deployed MLP9 write.

Run the known score hybrid, its recipient-absent reference, and the L5 payload control under both early-present and
early-absent backgrounds. For term `p`, the score response is

`h_p = Q_p(absent) - Q_p(score_hybrid)`.

The terms plus the numerical group sum exactly to the complete MLP9 response `h`. The native reference response is
`r = MLP9(absent) - MLP9(native)`, matching rungs 500--501. The payload response is decomposed the same way.

## Gradient shortlist and circuit fingerprints

Gradients are used only to decide which exact terms deserve finite testing. They are not causal evidence by
themselves.

For the copy-position CE in the recipient-absent trajectory, let `g_copy=d CE_copy/d MLP9_write`. For each pair and
background report:

`response_fraction(p) = <h_p,r> / <r,r>`,

`gradient_fraction(p) = <g_copy,h_p> / <g_copy,h>`.

The second quantity is the first-order fraction of the complete score hybrid's predicted copy benefit. For each of
the 32 discovery circuit tags, also differentiate `member CE - matched-control CE` with respect to the same MLP9
write and contract it with every `h_p`. This gives each pair a downstream circuit-use fingerprint. Tags with empty
member or control support in either half are reported and excluded by the fixed support rule; no result-dependent
tag may be dropped.

On documents `0:250`, a semantic pair enters the frozen shortlist only if, in both early backgrounds:

- its response and gradient fractions are positive and each at least `.01`;
- its score-response fraction is at least twice the absolute fraction produced by the corresponding payload term;
- its contribution is nonzero on copy positions; and
- both signed fractions have the same sign in each fixed 125-document quarter.

Freeze every passing pair, not merely the best one. The set must contain at most 32 pairs; a larger set is a diffuse
atlas and fails the compact-group clause rather than being truncated. The numerical group can never be selected.

## Frozen predictions

### A — exact, live source-pair instrument

- All frozen hashes, row roles, actions, source names, masks, circuit-tag assignments, calls, and backward counts
  match.
- The 20 raw semantic sources are derived from the exact residual recurrence and registered scalar coefficients.
  Adding `z_num` after the common gain reconstructs the deployed normalized state at relative squared error at most
  `1e-12`; repeated deployed BF16 additions are therefore retained rather than silently assigned to a named source.
- In float32, the 210 semantic pair terms plus the numerical group and bias reconstruct an independently evaluated
  float32 MLP9 write at relative squared error at most `1e-8`; their state-by-state differences reconstruct the
  independent float32 score and payload responses at the same bound. The independent float32 write versus the
  captured deployed-BF16 write is reported separately and must have relative squared error at most
  `16*(2^-8)^2 = 0.000244140625`.
- Native/action replays agree with rung 501's exact path, every intended edit and complete response is live, and the
  `NUMERICAL` response RMS is below `2%` of the complete score response in both halves and backgrounds.

### B — the known parent response remains calibrated

In both selection halves and both early backgrounds, the complete score response retains cosine at least `.75` and
positive-scale residual at most `.70` against the native MLP9 response, while the complete payload response loses by
at least `.30` cosine or `.30` residual. This repeats the parent observation inside the source-capture instrument;
it is not a new edge search.

### C — a compact semantic source-pair group is stable

The selection rule produces a nonempty set of at most 32 semantic pairs. Without changing it, on documents
`250:500`, in both halves and both early backgrounds:

- the summed group response has cosine at least `.75` and positive-scale residual at most `.70` against the complete
  score response;
- it accounts for at least `.50` of `<h,r>/<r,r>` and between `.50` and `1.50` of the complete signed copy-gradient
  contraction;
- every selected pair preserves the sign of both registered fractions; and
- the same group applied to the payload action has at most half the score group's response-direction fraction.

### D — downstream circuit use confirms location and specificity

Stack the support-qualified circuit member-minus-control contractions for the frozen group and for the complete score
response. On documents `250:500`, their cosine is at least `.75`, positive-scale residual at most `.70`, and the
group fingerprint norm is at least `.25` of the complete fingerprint norm in both early backgrounds. The same-position
group cosine must exceed the 95th percentile of 16 fixed token-position rolls by at least `.10`. The payload group's
fingerprint cosine with the complete score response must be at least `.20` lower. Report all 32 tag coordinates and
all unsupported tags; do not retain only favorable circuits.

### E — interpretation

E is true only if A--D hold. The selected terms are then called a **source-pair candidate group for the MLP9 copy
response**. The next rung must subtract that exact frozen group from MLP9, recompute layers 10--17, and test held-out
copy behavior plus unrelated-circuit preservation with partner-present/absent, payload, and position controls. No
individual source, head, or native bilinear unit is called a circuit at this stage.

## Nulls and routing

- A failure repairs only the source accounting, precision, replay, or call instrument.
- A true/B false means the known MLP9 observation did not survive source capture; retire this implementation before
  reading pair outcomes.
- A/B true/C false means the exact response is diffuse or unstable at the 20-source grain. Next refine only the
  largest stable source families into projected heads/MLP0 branches and use a held-out simplicity objective; do not
  run a rank sweep.
- A--C true/D false means local anatomy exists but the present 32 circuit probes do not identify its downstream use.
  Keep the exact group as anatomy, change the causal observation in a separately registered experiment, and do not
  claim a circuit.
- A--D true licenses the finite group intervention described in E. It does not license compression or removal yet.

## Literal price and storage

Discovery has 125 batches. Each batch runs native once and, for each of two early backgrounds, captures recipient
absent, score hybrid, and payload hybrid: exactly `7*125 = 875` model forwards. Circuit gradients are taken only at
the two recipient-absent MLP9 writes. The exact backward count is computed from the frozen nonempty masks before
model loading and asserted in the receipt. Pair construction and gradient contractions are standalone tensor
operations, not model forwards. Store only dot-product sufficient statistics, support counts, source/pair names,
and audit values—never raw tokens, logits, gradients, or 1,152-dimensional per-token pair vectors. This rung adds
and saves zero deployed parameters.

## Pre-outcome batch-alignment addendum — 2026-09-02 19:01 UTC

Implementation review found that the originally written `0:250/250:500` boundary is not divisible by the production
batch size of four. Splitting the model loop there would either run an invalid two-document production batch, repeat
two documents, or retain two full MLP9 source-factor graphs across the boundary. Before any rung 502 model or
source-pair outcome was opened, the split is changed to the nearest lower aligned boundary:

- selection documents `0:248`, with fixed halves `0:124` and `124:248`;
- confirmation documents `248:500`, with fixed halves `248:374` and `374:500`.

All 500 documents are still used exactly once in globally aligned four-document batches. The 875-forward price,
source definitions, 210 pairs, selection rule, gradients, controls, thresholds, circuit tags, and closed validation
set are unchanged. The implementation must assert these four exact intervals. This is a pre-outcome batching repair,
not a result-dependent scientific change.

## Instrument-repair addendum — rung 502b, 2026-09-02 19:12 UTC

The first rung502 receipt is preserved at result SHA256
`77984dd9d68da79640d72a8c273718b32199d9eb67fea0b7c4038770141099c0` and bundle SHA256
`c2d3a35565951218dd7f335bed6adb6322172db9b8fe3f12cf5ae1d4cad2604e`. It is instrument-invalid and none of
its pair, group, or circuit outcomes may be reused to pass rung502b.

Two independent instrument failures require a distinct namespace and complete rerun:

1. The first implementation compared the early-absent score response with the fully native MLP9 write. The registered
   parent response instead requires the `late_native` write in the same early-absent background. Rung502b therefore
   runs one native early-present trajectory plus four early-absent trajectories (`late_native`, `late_absent`, score,
   payload), for eight model forwards per batch and exactly1,000 forwards total.
2. The explicit numerical source carried10.9--13.3% of the small MLP9 response, above the frozen2% ceiling. Whole-state
   closure, float32/BF16 write agreement, and normalization-gain drift all passed, so silently discarding that term or
   loosening2% is forbidden. Rung502b replaces the invalid source-accounting instrument with two exact allocation
   gauges and requires the scientific answer to agree across them.

### Exact deployed residual and two source-allocation gauges

Rung502b owns a source-closed copy of the production forward loop so it can capture the exact deployed BF16 residual
`x` immediately after attention9 and before MLP9 RMS normalization. The19 explicit attention/MLP sources retain the
same registered residual-mixing coefficients. Let `e0` be the analytic embedding/skip contribution and let

`raw_round = x.float() - e0 - sum(explicit_write_sources)`.

Let `alpha=<z,x>/<x,x>` for the deployed normalized state `z`, and let

`norm_round = z - alpha*x.float()`.

Both complements are reported separately. They are implementation arithmetic, not semantic sources. Construct two
20-source gauges:

- `E_ABSORBS`: add all `raw_round` to `E`, multiply every raw source by `alpha`, then add all `norm_round` to `E`;
- `PROPORTIONAL`: distribute `raw_round` among all20 raw sources in proportion to their per-token squared norms;
  after multiplying by `alpha`, distribute `norm_round` in proportion to the resulting per-token squared norms.

Each gauge sums exactly to the same deployed `z`. Every gauge independently produces the same named210-pair list and
must meet the old float32 pair-closure and float32-versus-BF16 write bounds. Neither creates or selects a `NUMERICAL`
source. Report the raw and normalized complement RMS relative to `x` and `z`, and their response contribution under
the old explicit-numerical convention, but these diagnostics cannot be called semantic computation.

### Frozen repaired predictions

`A_b` holds only if all hashes, data roles, source names, edits, exact intervals, and the corrected1,000-forward price
match; background-native replay is exact; both gauges reconstruct every deployed normalized state to `1e-12`, every
independent float32 MLP9 write to `1e-8`, and deployed BF16 writes to the old `16u^2` bound; all responses/gradients
are live and finite. Because the model's RMS normalization is unweighted, the captured raw-side complement is also
an instrument check rather than a free remainder: `raw_round` RMS divided by deployed raw-residual RMS must be at
most `8u=.03125`, and `norm_round` RMS divided by normalized-state RMS must be at most `4u=.015625`, where
`u=2^-8` is the BF16 unit roundoff. Larger errors indicate a missing/misweighted source and fail A_b. The first
receipt's `<2% NUMERICAL` clause cannot pass retroactively and is replaced only by the exact raw-residual capture,
these roundoff bounds, the two exact allocations, and the cross-gauge requirements below.

`B_b` repeats the old parent-response and payload-rejection bars in all four fixed document halves and both backgrounds,
using each background's own `late_native` write. It must numerically reproduce rung501's background-specific MLP9
cosines to absolute error at most`.03`.

`C_b` applies the original per-pair selection and compact-group rules independently in `E_ABSORBS` and `PROPORTIONAL`.
The complete selected pair-name sets must be identical, nonempty, and at most32. Without reselection, that same set
must meet every original C confirmation bar in both gauges, and every selected pair must preserve both registered
signs across halves and gauges. No union, intersection, or best gauge is permitted after seeing outcomes.

`D_b` applies the original32-tag member-minus-control,16-position-roll, norm, residual, and payload bars independently
in both gauges. All support-qualified coordinates are retained. Both gauges must pass and must agree on the sign of
every selected-pair copy-gradient contribution in every confirmation half/background.

`E_b` is true only if `A_b--D_b` hold. It licenses the same separately preregistered finite MLP9 group-removal test and
nothing stronger. Gauge disagreement means the semantic source-pair atlas is not identified at deployed precision;
the next object is an exact finite upstream-source factorial or a clearly labeled float32 explanatory tensor, not a
third allocation, rank sweep, or threshold change.

The data,20 source names,210 pairs,0:248 selection,248:500 confirmation, pair thresholds, compactness ceiling,
payload controls, circuit masks, position controls, closed documents500:1000/30 validation tags, and zero deployed-
parameter claim remain unchanged. Store sufficient statistics for both gauges and no raw token/logit/vector data.
