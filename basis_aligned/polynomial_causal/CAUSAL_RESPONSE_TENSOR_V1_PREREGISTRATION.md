# Causal-response tensor v1 — prospective preregistration

Frozen before any v1 model outcome is collected.  This document specifies the
discovery instrument needed to fit and test shared/private causal tensor programs.  It
does **not** authorize execution by itself.

## Question

Across the 49 already registered circuits owned by `a8`, `a16`, `m16`, `a3`,
`m14`, and `m13`, is there a small set of signed causal response patterns that can be
stored once and sparsely reused?  A response pattern is useful only if it predicts
held-out intervention effects; similarity between activation directions is not enough.

## Frozen data roles and lineage

The source grid is exactly 1,000 rows of 256 predictions from 688 source documents.
Document roles are fixed by
`causal_response_tensor_document_split.json` (SHA-256
`3cb829ce5c9627f787e804e4e2ca44098030c629933f14df2c3a7fb07283317c`):

- split seed: 184;
- FIT: 343 source documents;
- EVAL: 345 source documents;
- source-document overlap: zero;
- every repeated row from one source document stays in one role;
- all 49 circuits have at least 149 member documents in each role.

The execution authority must bind and recheck these parents byte for byte:

| parent | SHA-256 |
|---|---|
| `census_state_diverse.pt` | `c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b` |
| `curated_rows.pt` | `faaf89f38ddf1471234a1d30d978213367a566a9927bb3c73b274ab32afaa9dd` |
| `circuits/BATTERY.json` | `86d7ac72eeb95f9ec80a3e92ef65e28c0df66a36b9291d2d1d2d01f7bb6c5030` |
| bilin18 config | `428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c` |
| bilin18 weights | `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3` |

FIT may estimate directions and later fit tensor factors.  EVAL may only measure
frozen interventions and evaluate frozen fits.  All uncertainty resamples whole
source documents.

## Exact source intervention

For each circuit tag (s), let (c(s)) be its frozen owner component.  On FIT only,
compute the component-write contrast

\[
v_s = \mathbb E[y_{c(s)}\mid\text{member of }s]
      -\mathbb E[y_{c(s)}\mid\text{outside the frozen slice of }s],
\qquad u_s=\frac{v_s}{\lVert v_s\rVert_2}.
\]

For each component, stack its unit vectors as rows and let (q_c) be the first right
singular vector, with sign fixed by making its largest-magnitude coordinate positive.
The residual phase is

\[
r_s=\frac{u_s-\langle u_s,q_{c(s)}\rangle q_{c(s)}}
          {\lVert u_s-\langle u_s,q_{c(s)}\rangle q_{c(s)}\rVert_2}.
\]

The two frozen phases are `full` with direction (u_s), and `residual` with direction
(r_s).  At every sequence position of source component (c(s)), replace its native
write (y) by the rank-one projection deletion

\[
y' = y-\langle y,d_s\rangle d_s,
\]

where (d_s=u_s) or (r_s).  The amplitude is exactly 1.  There is no top-k routing,
decoded label, gradient fit, target-dependent intervention, or post-hoc sign choice.
All other components run their exact native program once on the resulting trajectory.

Preregistered failure: if a FIT contrast is nonfinite/zero, or its residual norm is at
most (10^{-6}), v1 fails rather than silently changing its intervention family.

## Exact target response cell

For every phase (p), source (s), target circuit (t), EVAL source document (d),
and prediction position (i), define

\[
\Delta\mathrm{CE}_{p,s,d,i}
=\mathrm{CE}(y_i\mid x_{<i};\text{projection}_{p,s})
-\mathrm{CE}(y_i\mid x_{<i};\text{native}).
\]

Positive values mean the deleted direction helped prediction.  The target member mask
is the frozen circuit membership mask.  The target off mask is exactly the complement
of that target's frozen slice mask.  For each document, store four additive sums and
two static counts:

\[
\sum_{i\in M_t}\Delta\mathrm{CE},\quad
\sum_{i\in M_t}|\Delta\mathrm{CE}|,\quad |M_t|,
\qquad
\sum_{i\in O_t}\Delta\mathrm{CE},\quad
\sum_{i\in O_t}|\Delta\mathrm{CE}|,\quad |O_t|.
\]

The signed target contrast is member mean minus off-slice mean.  Absolute
concentration is diagnostic only; it is not additive and will not be the fitted tensor.
Documents with zero member support remain explicit unsupported cells.

## Collection acceptance gates

The result must have the complete dense shape
`[2 phases, 49 sources, 49 targets, 345 EVAL documents]`, with an explicit validity
mask from member counts.  It passes collection only if:

1. every stored numeric value is finite and absolute sums obey the triangle inequality;
2. FIT and EVAL source-document sets are disjoint and match the frozen split;
3. every outer forward calls every one of 18 native attention and 18 native MLP sites
   exactly once;
4. each projection is dispatched only at its frozen source component;
5. the model has zero forward hooks before and after, and its state/config/checkpoint
   hashes are unchanged;
6. a semantic reload reproduces schema, shape, ordering, masks, counts, and hashes;
7. no raw tokens, targets, activations, or logits enter the published artifact;
8. authority, result, failure, and receipt use create-only mutually exclusive terminal
   publication, with the receipt written last.

## Analysis boundary

This is a discovery response tensor, not a confirmatory circuit result.  Candidate
CP/block-term/shared-private/DAG factorizations may be fit on FIT responses, ranked by
literal stored/executable price, and compared on EVAL signed contrasts.  No factor gets
a semantic name or terminal-circuit tier from response reconstruction alone.  It must
subsequently predict a fresh intervention, support extraction or selective removal,
and transport OOD.  No confirmatory p-values will be reported from model choices made
using this v1 tensor.
