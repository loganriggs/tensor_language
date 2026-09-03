# Rung 525 preregistration — exact token-by-context operator quotient

**Frozen:** 2026-09-03 09:43 UTC  
**Owner:** Codex  
**Claim level:** weight/function-space grouping screen; no circuit or compression claim  
**GPU:** managed runner only

## Goal and duplicate-work boundary

The goal is to explain MLP0 as a computation, not merely reduce its rank. Rungs 394--399 already establish the
length-one token-only anatomy: common operating-point write `M`, compact degree-one token modulation `L`, and broad
token-private quadratic correction `Q`. Rungs 400--401 already give the exact natural-context branches `T`, `C`,
`I`, and `S`. Rungs 481--493 already show that whole branches, immediate-consumer Jacobians, categorical current or
previous-token tables, and native attention/MLP boundaries do not provide the desired circuit grouping.

Rung 525 does not repeat any of those questions. It constructs the exact linear operator that maps attention0
context into MLP0's centered token-by-context branch for every vocabulary token. It asks whether different tokens
have the same context-dependent MLP0 computation even when their token vectors are far apart. A positive result
would provide candidate token groups for a separately registered physical downstream interchange. A negative result
would close task-free grouping of this operator and move to the context-only branch or an explicitly
downstream-conditioned operator metric.

## Exact operator

For token `t`, define the token base

`e_t = (lambda_0,0 + lambda_0,1) RMSNorm(Embedding[t])`.

Reuse rung 401's FIT reference means `mu_e`, `mu_a` and mean squared normalization gain `g_bar`. Let
`de_t = e_t - mu_e` and `da = a - mu_a`, where `a` is attention0's output write. With MLP0 weights `Left`, `Right`,
and `Down`, the centered interaction is

`I_t(da) = g_bar Down[(Left de_t) * (Right da) + (Left da) * (Right de_t)]`.

For fixed `t`, this is a linear map from the 1,152-dimensional context deviation to the 1,152-dimensional MLP0
write. Call that map `K_t`. It is invariant to hidden-unit permutation, reciprocal Left/Right scaling, and swapping
Left with Right. We compare tokens through `K_t`, not by native hidden-unit labels.

## Randomized exact distance estimator

Form two disjoint probe banks, A from rung401 FIT documents and B from rung401 SELECT documents. Each bank contains
256 deterministically selected context deviations. Use the same 256 deterministic unit-norm Rademacher output
directions `q_j` in both banks. For token `t`, bank `H` stores scalars

`s_H[t,j] = q_j^T K_t da_H,j`.

Each scalar is computed by contraction without forming a `1152 x 1152` operator. The implementation must also form
the explicit `I_t(da)` for 64 frozen token/probe pairs and match the scalar contraction to relative squared error at
most `1e-10`.

For each bank, standardize every probe coordinate using only donor tokens. The squared operator distance between
tokens is the mean squared difference of their 256 standardized scalars. Bank A chooses candidates; bank B is an
unseen context distribution used only to score those frozen pairs.

## Token split, candidate, and controls

- Real vocabulary: token IDs `0:50257`.
- Donors: token ID modulo 5 is nonzero (`40,205` tokens).
- Receivers: token ID modulo 5 is zero (`10,052` tokens).
- Candidate: nearest donor under bank-A operator distance subject to raw token-base cosine at most `0.50`.
- Raw control: nearest donor in raw token-base cosine, with no operator information.
- Random control: one deterministic donor satisfying the same `<=0.50` raw-cosine condition.
- Deranged control: independently permute bank-A probe coordinates for each of four fixed blocks before candidate
  selection, preserving each coordinate's marginal distribution but destroying the coupled operator signature.

All candidate identities are frozen before bank-B distances are read. Candidate selection uses no CE, circuit tag,
activation-patching result, or downstream label.

## Planted identifiability gate

Before loading the model, generate a CPU float64 bilinear toy with 256 tokens in 32 known eight-token operator
classes. Token vectors within a class are deliberately far in raw cosine, but their `K_t` maps are identical up to
small independent noise. Apply the same two-bank sketch and candidate rule.

The toy instrument passes only if at least 95% of receivers select a donor from the correct operator class, the
bank-B candidate distance is at most 20% of the raw-control distance, and deranging probe coordinates reduces
correct-class selection to at most 25%. Failure stops before model loading.

## Registered real-model predictions

### A — exact and lawful instrument

- all checkpoint, row-role, parent, preregistration, and runner hashes match;
- FIT and SELECT document identities are disjoint and each probe bank has exactly 256 unique `(document,position)`
  contexts drawn before any token-pair result;
- all 50,257 token operators have finite, nonconstant sketches in both banks;
- the 64 explicit scalar identities have relative squared error at most `1e-10`;
- every receiver has an eligible far candidate and far random control;
- selected candidate raw cosine is at most `0.5001`; and
- no downstream model call, circuit tag, FINAL row, or sealed outcome is used.

### B — the operator grouping transfers across contexts

On frozen bank-A candidate pairs evaluated only with bank B:

- median candidate distance is at most 75% of median raw-control distance;
- median candidate distance is at most 25% of median far-random distance;
- median candidate distance is at most 75% of median deranged-candidate distance; and
- at least 5% of receivers have candidate distance below the fifth percentile of their 16 fixed far-random controls.

### C — the result describes reusable groups, not isolated coincidences

- at least 100 donor tokens are each selected by at least two receivers;
- at least 1,000 receivers belong to such repeated-donor groups;
- receiver-wise bank-A and bank-B candidate distances have Spearman correlation at least `0.50`; and
- rerunning candidate search on the first and second 128 bank-A probes yields selected donors whose full-bank-B
  distances differ by at most 20% in median.

## Null, interpretation, and next action

The strong null fires if the exact/planted instrument fails, if the candidate does not beat the raw control by 5%
on bank B, or if fewer than 1% of receivers beat the far-random fifth percentile. Do not tune sketch width, raw
cosine, class count, token split, or thresholds after a null.

A/B/C all passing would identify a task-free functional quotient of the exact token-by-context operator. It still
would not be a circuit. The only licensed successor is a physical natural-context substitution: replace `I_t(a)`
by `I_u(a)` for the frozen grouped token pairs, recompute the complete suffix, and require held-out attention1,
MLP1, final-CE, and 62-circuit-effect preservation with raw/deranged/random controls.

If A holds but B or C fails, task-free operator similarity does not define useful token groups. Move to either the
context-only quadratic operator or a downstream-conditioned metric built from the already validated finite MLP1
responses; do not return to rank, SAE reconstruction, token k-means, or native-head grouping.

## Price and relation to simplicity

Rung 525 adds no deployed values and saves none. It records the observed number and size distribution of repeated
donor groups and an explicit descriptive code length: token-to-group assignments plus the two `4608`-coordinate
token-side factor vectors for every occupied representative. This is compared with a full cached per-token factor
table, but not with native MLP0—the native shared weights remain cheaper than such a cache. Description length is a
screening statistic only. A simplicity claim requires the physical successor to preserve downstream computation and
eventually beat the native executable price.
