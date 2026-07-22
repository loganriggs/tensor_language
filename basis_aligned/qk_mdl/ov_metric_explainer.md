# How OV information is folded into the QK metrics, what the cancellation index measures, and how to include cancellation *correctly*

Standalone explainer (Logan request, 2026-07-22). Context: we compress the layer-0 query/key
factor tables and want a cheap weight-side metric that predicts held-out cross-entropy. Folding
the output-value (OV) circuit into the metric made prediction *worse* (Spearman 0.95 → 0.57),
which is counterintuitive — the OV circuit is literally what reads the attention pattern. This
file walks through (1) exactly how OV is folded in, (2) exactly what the cancellation index
measures, and (3) a derivation showing the two existing OV rungs are opposite limiting cases of
the *correct* metric — one forbids cancellation entirely, one over-credits it — and what the
correct in-between metric is.

---

## 1. The physical object: what a pattern error actually does

At layer 0 everything is exact and weight-only. For a context, the attention output at query
position `i` for head `h` is

    out_i = Σ_j P(i,j) · u_{t_j}        with   u_t = W_o^h W_v^h ê_t ∈ R^1152,

where `P` is the (unnormalized, no-softmax) pattern and `u_t` is the OV output vector of token
`t` — what attending to `t` actually writes into the residual stream. (At layer 0 the value-bus
mixing is the identity, so this is exact.)

So if compression perturbs the pattern by ΔP, the *physical* error at query `i` is a **vector
sum**:

    e_i = Σ_j ΔP(i,j) · u_{t_j}.                                          (★)

Everything below is about how to summarize (★) into one number without running data through the
model. Two facts about (★) drive everything:

- Errors on tokens with small ‖u‖ (the OV null space) don't matter.
- Errors on different tokens can **cancel**: if ΔP(i,j)·u_j and ΔP(i,j′)·u_{j′} point opposite
  ways in R^1152, their sum is harmless even though each is large.

## 2. The two ways we folded OV in

Both metrics start from a sampled pattern-error matrix ΔP (2048 sampled tokens × 2048 sampled
tokens, per head; rotary variants also exist) and the matrix `U` (2048 × 1152) of OV output
vectors. They differ in how they collapse the token sum:

**Rung "pat_ov" — the norm (diagonal) weighting.**

    m_diag = Σ_{i,j} ΔP(i,j)² · ‖u_j‖²   /   (same with P)

Each token's error is charged by its OV magnitude, **independently** — the sum over `j` is a sum
of squares. This respects the null space (‖u‖≈0 ⇒ no charge) but **forbids cancellation
entirely**: two errors that would annihilate in (★) are both charged in full.

**Rung "pat_gram" — the full Gram (coherent) metric.**

    m_gram = Σ_i ‖ Σ_j ΔP(i,j) · u_j ‖²   /   (same with P)     =  ‖ΔP·U‖²_F / ‖P·U‖²_F

This computes (★) literally over the whole token sample and squares the *resulting vector* —
cancellation and null space are handled exactly... **for one particular fictitious context**: the
context that contains *every sampled token simultaneously, once each*. That assumption is the
flaw, and section 4 makes it precise.

Empirically both rungs predict FineWeb ΔCE at Spearman **0.57** versus **0.95** for plain factor
FVU. They fail identically but for opposite reasons.

## 3. The cancellation index — what it measures

For each compression arm we report

    cancel = m_gram numerator / m_diag numerator
           = ‖ΔP·U‖²_F  /  Σ ΔP(i,j)²‖u_j‖².

Reading: if the per-token error vectors ΔP(i,j)·u_j were mutually orthogonal, the ratio would be
exactly 1. Ratio < 1 means net cancellation; ratio > 1 means the error vectors *reinforce* (point
the same way, so the vector sum is larger than the sum-of-squares suggests). Measured values:

| object | cancel index |
|---|---|
| the TRUE pattern (signal itself) | **31.6** |
| merges (K=2048, two-stage) | ≈ 16 |
| dictionaries (n=1024, k=8) | ≈ 13–14 |
| SVD (rank 16…128) | ≈ **10–11** |

Two things to notice. First, everything is far above 1: the OV vectors of different tokens are
highly correlated (they live in a low-dimensional cone — each head's `u` matrix has rank ≤ 128),
so *sums reinforce by default*; genuine "cancellation" here means *reinforcing less than the
signal does*. Second, the families differ systematically: **SVD's residual errors are the least
coherent through OV** (10–11 vs the dictionaries' 13–14). This is why the Gram metric flatters
SVD relative to the dictionaries: it awards SVD a ~25–30% coherence discount that, empirically,
held-out cross-entropy does not honor.

So the index is a *diagnostic*: when two arms differ a lot in cancel index, any post-OV energy
comparison between them is suspect. But it begs the question you asked — shouldn't the Gram
metric's accounting of cancellation be *correct*, since (★) is the physical error? Why does
honoring cancellation exactly make the prediction worse?

## 4. Why full cancellation is too generous: contexts are finite samples

The resolution: (★) is the error for a **specific context** — a specific multiset of tokens
`t_1 … t_T` at specific positions. The Gram metric evaluates (★) for one fictitious context
containing all 2048 sampled tokens with equal weight. Real contexts are **length-T samples**
(T = 512 in the frozen regime) drawn roughly from the unigram distribution `q`. Cancellation
that holds *summed over the whole vocabulary* need not hold *within a given draw* — a context
containing "music-atom tokens" but not the tokens whose errors would have cancelled them gets
the full error.

Make this precise. Fix query token `i`; write `c_j = ΔP(i,j)·u_j` for the per-token error
vector, and let the context's keys be T i.i.d. draws from the unigram distribution `q` (the
no-softmax architecture makes this model apt: the output really is a plain sum over positions —
this is also exactly why the model degrades past T≈512). Then the expected squared error is

    E‖e_i‖²  =  T · ( E_q‖c‖² − ‖μ‖² )  +  T² · ‖μ‖²,        μ = E_q[c_j].       (†)

The two existing rungs are the two terms of (†) in isolation:

- **m_diag is the first (variance) term** — the part of the error that behaves like a random
  walk across contexts. It accumulates as **T** and **never cancels** in any single context: it
  is the context-to-context scatter, and squared error is charged for scatter regardless of sign.
- **m_gram is the second (mean²) term** — the *systematic* component of the error, identical in
  every large context. This part **does** cancel exactly as the Gram metric says, and it
  accumulates as **T²**.

So: the norm rung forbids cancellation everywhere (it charges the mean component as if it were
scatter); the Gram rung credits cancellation everywhere (it treats the scatter as if it were
systematic and lets it cancel). Physically, cancellation applies **only to the mean component**.
That is the "cancellation part of the OV matrix" you asked about, and it can be folded in
properly:

## 5. The corrected metric: context-expected OV error ("pat_ctx")

Charge each arm by the expectation (†), queries weighted by their own frequency:

    m_ctx = Σ_i q_i [ T·(s_i − ‖μ_i‖²) + T²·‖μ_i‖² ]  /  (same functional of the true P)

with, per query token i (all weight-only plus unigram statistics):

    μ_i = Σ_j q_j ΔP(i,j) u_j          (mean error vector — the part that truly cancels)
    s_i = Σ_j q_j ΔP(i,j)² ‖u_j‖²      (second moment — the scatter floor)

Inputs: the factor tables, the OV vectors `u`, the unigram frequencies `q`, and T = 512. Nothing
else. This is the exact expected squared layer-0 output error under the i.i.d.-context model —
cancellation included precisely where it physically operates (the T² mean term) and excluded
precisely where it doesn't (the T variance term). It also subsumes the frequency fix from the
ladder (rare tokens are down-weighted by `q` in both terms).

**Pre-registered expectation** (stated before the run): SVD's low cancel index means its error is
relatively scatter-dominated; m_ctx re-charges that scatter at weight T without cancellation, so
m_ctx should *undo the Gram metric's SVD discount* and rank arms closer to the truth than either
pure rung. If instead m_ctx still lands near 0.57, the i.i.d.-context model is the wrong
approximation (real co-occurrence structure matters), and the next refinement is replacing the
unigram `q` with document-level co-occurrence statistics — at which point the metric stops being
weight-plus-unigram and starts being data-conditional, with the usual ledger cost.

## 6. Known approximations (in honesty order)

1. **i.i.d. keys**: real contexts have topical co-occurrence (music tokens cluster). This
   *underestimates* within-context coherence for topically-clustered errors — notably the
   dictionaries', whose atoms are topics. Testable by comparing m_ctx's residual misranking
   against a co-occurrence-corrected version.
2. **Query marginalized independently** of its context (a query token co-occurs with correlated
   keys). Same refinement path.
3. **Rotary/mask**: (†) is stated pre-rotary; the ladder's rope machinery can layer offsets in
   (pair-count weighted) if the pre-rotary version proves insufficient.
4. Downstream nonlinearity: (†) is the layer-0 *output* error; CE response to that error is
   assumed monotone in its magnitude. All rungs share this assumption; plain factor FVU's 0.95
   suggests it is not the binding constraint at current accuracy.

Implementation: `qk_ovweight.py`, rung `pat_ctx`; results land in `qk_ovweight.json` and the
ladder table of `RESULTS_l0_mdl.md`.
