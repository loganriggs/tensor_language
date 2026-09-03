# Rung 526 preregistration — group MLP0 token-by-context operators by downstream circuit use

**Frozen:** 2026-09-03 10:02 UTC  
**Owner:** Codex  
**Claim level:** downstream-conditioned tangent screen; finite causal identification remains a successor  
**GPU:** shared managed runner only

## Question and duplicate-work boundary

Rung 525 found stable similarities between the exact token-by-context operators, but those similarities were worse
than simply pairing tokens with nearby token vectors. Its physical substitution is therefore closed. The present
question is different:

> Can two tokens that are far apart in the residual stream nevertheless be grouped because their exact MLP0
> token-by-context outputs have the same predicted effects on known downstream circuits?

Rung 406 used the aggregate next-token cross-entropy gradient to choose a rank-448 shared input basis. It did not
group tokens or use the 62 circuit outcomes. Rung 481 measured the finite effects of the four complete MLP0 branches
on those circuits. It did not split the interaction branch by token-dependent operator. Rungs 485--493 established
that MLP1 is a real downstream reader of the interaction branch, but categorical token and previous-token tables do
not predict that response. Rung 526 instead uses the existing circuit member/control masks as a downstream metric on
the exact operator from rung 525. It neither reduces rank nor reconstructs activations.

## Exact object and computational definition

For token `t`, rung 525 defined the exact centered interaction operator

`K_t da = g_bar D0[(L0 de_t) * (R0 da) + (L0 da) * (R0 de_t)]`,

where `de_t` is the token-base deviation, `da` is an attention0-context deviation, `L0`, `R0`, and `D0` are MLP0's
two readers and output map, and `*` is elementwise multiplication.

For circuit `c` on a fixed document set, define the native downstream score

`Y_c = mean(NLL on c.member) - mean(NLL on c.slice_control)`.

Here NLL is the per-token negative log likelihood, numerically the same quantity averaged to obtain cross-entropy
loss. Capture MLP0's complete output write `m_i` at every position `i` as an autograd leaf without changing its
value, and compute

`G_c,i = derivative of Y_c with respect to m_i`.

The downstream-conditioned signature of hypothetical token operator `t` is

`S_H[t,c] = sum_i G_H[c,i]^T K_t da_H,i`.

This is the exact directional derivative of the circuit score in the `K_t da` direction at the native model state.
It includes the complete suffix after MLP0, including attention1 and later layers, because `G` is backpropagated from
the final logits. It is still a first-order screen: a finite replacement can differ because normalization, attention,
and later bilinear layers respond nonlinearly.

The signature can be contracted without materializing every 1,152-dimensional `K_t da_i`. If
`u_t=L0 de_t`, `v_t=R0 de_t`, `a_i=L0 da_i`, `b_i=R0 da_i`, and `w_c,i=D0^T G_c,i`, then

`S_H[t,c] = g_bar [u_t^T sum_i(b_i*w_c,i) + v_t^T sum_i(a_i*w_c,i)]`.

Thus each circuit produces two 4,608-dimensional accumulated vectors, followed by two matrix products over all
50,257 tokens. This identity must match explicit construction on frozen toy tensors to relative squared error at
most `1e-10` in float64 and `1e-5` in the model's float32 analysis path.

## Data separation and circuit separation

Use the already frozen 1,000-document circuit corpus and its existing 32 discovery plus 30 validation circuit tags.

- `D0`: discovery tags, documents `0:124`; candidate selection only.
- `D1`: the same discovery tags, documents `124:248`; held-out-document scoring.
- `V0`: validation tags, documents `500:750`; unopened unless the discovery gate passes.
- `V1`: validation tags, documents `750:1000`; unopened unless the discovery gate passes.

All circuit member and slice-control counts are computed before model loading and must be positive in every used
half. Token donors and receivers retain rung 525's split: donor when token ID modulo 5 is nonzero, receiver otherwise.

For each document/circuit block, divide member and control masks by their full-half counts before backpropagation, so
`Y_c` is exactly a difference of means rather than a support-size proxy. Standardize each circuit coordinate using
donor tokens within that half only.

## Candidate and controls

- Candidate: nearest donor under standardized `D0` signature distance, constrained to raw token-base cosine at most
  `0.50`.
- Raw-token control: donor with highest raw token-base cosine.
- Far-random controls: 16 deterministic donors satisfying the same raw-cosine ceiling.
- Scrambled-signature control: independently permute the 32 discovery-circuit coordinates for every token before
  candidate search, preserving each coordinate's marginal distribution while breaking the coupled downstream
  fingerprint.
- Rung-525 control: report how often the selected donor equals rung 525's task-free operator donor. This is diagnostic
  only and is not a pass condition.

All identities are frozen from `D0` before reading `D1`, `V0`, or `V1` distances.

## Planted and gradient-instrument gates

Before model loading, generate 256 tokens in 32 known downstream-response classes. Raw token vectors are nuisance-
dominated and far within a class. `D0`, `D1`, `V0`, and `V1` have independently generated response coordinates but
share class identity. The same search must recover the correct class for at least 95% of receivers in `D0`, have
candidate/raw median distance at most `0.20` in all three unseen banks, and fall to at most 25% correct after the
per-token coordinate scrambling.

On a frozen small differentiable network before the real collection, autograd contraction and an independently
formed explicit directional derivative must agree to relative squared error at most `1e-10` in float64. For the real
model, the identity MLP0-output leaf must reproduce native logits exactly, all gradient/signature entries must be
finite, every circuit must have a nonzero gradient, and the aggregate contraction check must be at most `1e-5`.

Failure of any instrument gate stops without reading `D1`, `V0`, or `V1` as an outcome.

## Registered predictions

### A — exact, live, leakage-free instrument

- all dependency, checkpoint, runner, row, mask, and circuit-partition hashes match;
- planted and differentiable-toy gates pass before model loading;
- the four document ranges and the discovery/validation circuit families are disjoint as specified;
- member/control supports are positive and their normalized mask weights sum to `+1/-1` per circuit and half;
- identity-leaf replay, gradient census, exact contraction, donor eligibility, and raw-cosine ceilings pass; and
- no `D1`, validation-circuit outcome, finite token swap, FINAL row, or sealed outcome is opened before its gate.

### B — downstream grouping transfers to new documents for the same circuits

For frozen `D0` candidate pairs evaluated only in `D1`:

- median candidate distance is at most 75% of the raw-token-control distance;
- median candidate distance is at most 35% of the far-random distance;
- median candidate distance is at most 75% of the scrambled-control distance;
- at least 5% of receivers beat the fifth percentile of their 16 far-random controls; and
- receiver-wise `D0`/`D1` candidate distances have Spearman correlation at least `0.40`.

### C — the grouping transfers to held-out circuits

Open `V0` and `V1` only if A and B pass. In each validation half, using the unchanged `D0` donor identities:

- candidate/raw median distance ratio is at most `0.85`;
- candidate/far-random median distance ratio is at most `0.50`;
- candidate/scrambled median distance ratio is at most `0.85`; and
- at least 5% of receivers beat their far-random fifth percentile.

The two validation-half candidate-distance vectors must have Spearman correlation at least `0.40`.

### D — the result is a reusable quotient rather than isolated pairs

- at least 100 donors are selected by at least two receivers;
- at least 1,000 receivers belong to repeated-donor groups;
- searches using the first and second 16 discovery circuits separately yield full-`D1` median distances differing
  by at most 25%; and
- at least 80% of selected donors differ from rung 525's task-free donors, demonstrating that this is a genuinely
  downstream-conditioned coordinate rather than a relabeling of the closed metric.

## Null, decision, and circuit relevance

The discovery strong null fires if the exact/planted instrument fails, if the `D1` candidate/raw ratio is at least
`0.95`, or if fewer than 1% of receivers beat the far-random fifth percentile. Do not tune circuit weights, token
split, raw-cosine ceiling, thresholds, or signature dimension after a null.

- If A or B fails: validation circuits remain unopened. Close this tangent metric and change to an exact finite
  context-only intervention or a predictive-state quotient defined directly by finite downstream outcomes.
- If A/B pass but C or D fails: retain only the discovery screen; no physical token grouping is licensed.
- If A/B/C/D all pass: rung 527 may perform finite natural-context swaps `K_t da -> K_u da` for the frozen groups,
  recomputing the entire suffix and testing held-out circuit effects, overall cross-entropy damage, unrelated-circuit
  preservation, redundancy, and pairwise interactions. Only that successor can identify or manipulate a circuit.

Rung 526 directly addresses cross-boundary grouping, held-out prediction, and stable identification. It receives no
adoption, extraction, selective-manipulation, composition, or simplicity credit by itself. Literal price is the
stored token-to-group assignment plus occupied representative signatures, reported against a full signature table;
the native model comparison remains zero saved values until finite replacement succeeds.
