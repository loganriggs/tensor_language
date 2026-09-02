# Rung 481 preregistration — exact MLP0 branches measured by downstream circuit use

Conditional registration written while rung480 is still running and before any rung481 model outcome is collected.
Run this only if rung480's in-process instrument is valid and its registered scientific strong null fires. A full
rung480 pass instead routes to exact odd-family attention0 removal. A bridge-only rung480 failure routes first to its
already-queued in-process bridge repair.

## Question

MLP0 has an exact natural-context decomposition into token-only, context-only, token-by-context, and normalization
terms. We know their aggregate CE effects, but not which of the existing downstream circuits use each term. The
native MLP boundary is too coarse, and rank reduction does not answer this question.

Rung481 asks:

1. Which exact MLP0 branches affect which downstream circuits?
2. Does downstream computation distinguish the token-only branch `T` from the token-by-context branch `I`, or treat
   them as two ways of producing the same variable?
3. Which branch pairs interact after all later layers recompute, and are those interaction patterns stable enough to
   justify separate decomposition paths?

This is a branch-level causal map and grouping/splitting decision. It saves no parameters and does not select a rank.

## Existing results that must not be repeated

Rung401 established the exact decomposition. Earlier MLP0 work already rejects a universal sparse token code,
whole-vocabulary interchange, response-weighted PCA, fixed low-rank quadratic producers, and a Tucker refactor.
Rung481 may reuse their exact algebra and controls, but it may not tune sparse support, rank, Tucker dimensions, or
reconstruction loss.

## Exact intervention

Write the pre-normalization MLP0 input as token contribution `e` plus attention0 context contribution `a`. The native
bilinear map is

`G(e,a) = Down((Left(e+a)) * (Right(e+a)))`.

Rung401 fixes moments on its original fitting rows and decomposes the deployed MLP0 write as

`constant + T(e) + C(a) + I(e,a) + S(e,a) + numerical_residual + bias`.

The numerical residual is the explicit quadratic contribution of the small non-collinear part left after expressing
RMSNorm as a scalar multiple of `e+a`; it remains unchanged in every intervention. The definitions are:

- `T`: token main effect at the average normalization gain;
- `C`: context main effect at the average gain;
- `I`: centered bilinear interaction between the particular token and particular context;
- `S`: change caused by the example's normalization gain differing from its average.

For every subset `S0` of `{T,C,I,S}`, call the model with only those branches retained, while keeping the constant,
numerical residual, MLP0 bias, attention0 write, token reinjection, and all unrelated computation fixed. Every later
attention and MLP module recomputes normally. The full subset must reproduce the native model exactly. All16 subsets
are evaluated; no subset is chosen after seeing its result.

## Circuit-specific quantities

Use the frozen 62-tag circuit battery and its two existing position masks:

- `member`: positions where that circuit definition applies;
- `slice_control`: matched positions from the same data slice where it does not apply.

For branch subset `S0`, circuit `c`, mask `m`, and document half `h`, let

`P_h,c,m(S0) = - mean CE_h,c,m(S0)`.

Higher `P` is better. From the complete16-arm table, compute the ordinary Shapley contribution of each branch. This
is the branch's average marginal effect over every order in which the other three branches could be restored.

The circuit-selective response profile of branch `b` is

`p_b[c] = Shapley_b(member,c) - Shapley_b(slice_control,c)`.

All profile cosines and difficulty regressions center these values across circuits first, so they compare which
circuits differ rather than a shared average branch effect. Norm ratios, magnitudes, and signs use the uncentered
effects and are reported separately.

Also compute the context-averaged pair interaction

`J_bd[c] = mean over U subset of the other two branches of`

`P(U+b+d) - P(U+b) - P(U+d) + P(U)`.

A positive value means the two branches help more together than the sum of their separate effects for that circuit;
a negative value means their effects partly substitute or cancel. This is downstream nonlinear interaction, not an
interaction asserted from activation similarity.

Report raw member and control effects separately, the member-minus-control profiles, pooled CE, Shapley effects, all
pair interactions, and complete higher-order Möbius terms. CE is always measured relative to the real full model;
larger damage after removing useful computation means a more positive branch benefit.

## Data split and conditional validation

The discovery calculation uses only the32 tags whose top-level roots are `{0,2,4,6,8,18}` and documents0:500,
split exactly at document250. This matches the repaired split convention from rung477b. The30 tags with roots
`{1,3,5,7,11,13,23}` and documents500:1000 remain unopened unless the discovery gates below license validation.

If licensed, run the same frozen16 arms on those30 tags and documents500:1000, split at750. No threshold, branch
pair, sign convention, or formula may change between discovery and validation. Thus all62 circuits are ultimately
used, while30 remain an honest circuit-family and document holdout during selection.

No FINAL or SEALED model outcomes are used.

## Controls

1. **Matched positions:** branch effects on `member` and `slice_control` positions are retained separately; only their
   difference is called circuit-selective.
2. **Circuit-label permutation:** for16 fixed seeds `2026090281..2026090296`, permute the circuit labels in the second document half before
   calculating cross-half profile cosine. This keeps effect sizes and branch geometry but destroys circuit identity.
3. **Token-difficulty residual:** regress each branch profile on the full-model member-minus-control CE profile using
   only the first document half, then apply the frozen coefficient to the second half. Stability must hold both before
   and after removing this one-dimensional difficulty signal.
4. **Exact native baseline:** in every batch, separately call the unmodified model and require the full `T+C+I+S`
   arm to reproduce its logits and per-position CE.
5. **Branch reconstruction:** the constant, four branches, retained normalization residual, and bias must reproduce
   the directly evaluated native MLP0 write at the registered float and deployed dtypes.

## Opposing scientific hypotheses and frozen gates

### A — valid exact instrument

- all rung401, circuit-battery, census-row, mask, model, and source hashes match;
- the analytical MLP0 decomposition has relative squared error at most`1e-8`;
- the complete branch sum reproduces the deployed MLP0 write to relative squared error at most`1e-5` and the full-arm
  logits reproduce the separately called native logits to relative squared error at most`1e-12`;
- all16 arms execute with the exact registered forward/module-call counts, every branch knob changes its MLP0 write,
  the document250 boundary is allocated correctly, and every discovery tag has at least39 member and439 matched
  control positions in each half; and
- validation circuits/documents, FINAL, and SEALED remain unopened before the discovery decision.

### B — the important branch profiles are stable and circuit-selective

For both `T` and `I`:

- raw and token-difficulty-residualized `p_b` have cross-half cosine at least`.70`;
- each cross-half cosine exceeds the95th percentile of its16 circuit-label permutations by at least`.15`; and
- the branch's raw member-effect norm is at least`1.25` times its matched-control-effect norm in each half.

Also, `T` and `I` must be the two largest branches by median absolute circuit-selective Shapley effect in both halves.

### C-split — downstream computation distinguishes token and token-by-context paths

In both discovery halves, `abs(cosine(p_T,p_I)) <= .70`, and at least eight of the32 circuits have opposite-signed
`T` and `I` selective effects in both halves.

### C-shared — downstream computation treats them as one variable

In both discovery halves, `cosine(p_T,p_I) >= .90`. Fit
`alpha=<p_T_half0,p_I_half0>/<p_T_half0,p_T_half0>` and require
`norm(p_I_half1-alpha*p_T_half1)/norm(p_I_half1) <= .35`; at most three second-half circuits may have opposite signs
between `p_I_half1` and `alpha*p_T_half1`.

These are opposing hypotheses. The interval between them is deliberately unresolved: neither “some similarity” nor
“some difference” licenses grouping or splitting.

### D — branch interactions determine whether separate follow-ups are needed

For every pair, report the member-minus-control `J_bd` profile and its size relative to the smaller of the two
single-branch profile norms. A pair is material and stable when its relative norm is at least`.20`, its raw and
difficulty-residualized cross-half cosines are at least`.60`, and both exceed their label-permutation95th percentiles
by`.15`. Predict at least one stable material pair involving `I`.

This gate does not force every branch to share one decomposition. Its result fixes priority:

- a stable material pair is analyzed jointly in the next decomposition;
- a stable branch with no material pair is analyzed independently; and
- an unstable or negligible branch is deprioritized rather than compressed.

### E — the selected grouping/splitting relation validates on new circuits and documents

Open the30 validation tags only if A and B hold and exactly one of C-split or C-shared holds. On documents500:1000:

- `T` and `I` again satisfy B's raw/residualized stability and control margins;
- whichever of C-split or C-shared won on discovery satisfies the same thresholds on the30 new circuits; and
- any branch pair selected as material by D has at least`.50` profile cosine between the two validation halves, and
  the sign of its median validation member-minus-control interaction matches the sign of its pooled discovery
  median.

## Null and routing

The instrument null fires if A fails. Preserve the receipt and repair only the named instrument; do not interpret
branch profiles.

The scientific strong null fires if B fails or neither opposing C hypothesis holds. In that case, the exact
`T/C/I/S` formulas remain valid causal accounting, but the62-circuit battery does not identify a stable grouping of
their downstream uses. Do not tune rank, sparsity, or thresholds. The next object must change, for example to exact
consumer-specific Jacobians at attention1 and MLP1 rather than circuit-tag averages.

If B and C-split hold, fit separate downstream-response decompositions for `T` and `I`, joining only the branch pairs
licensed by D. If B and C-shared hold, seek one operational downstream variable shared across `T` and `I` while
retaining their distinct exact input formulas. In either case E is required before any selective removal or semantic
name is claimed.

The next decomposition must use its held-out downstream response and exact physical removal as its success criteria.
A lower rank, fewer values, better reconstruction, or lower ordinary CE is not enough.

## Price

Discovery uses16 exact branch arms plus one separate native call per batch on500 documents. Conditional validation
repeats the same computation on500 disjoint documents. The receipt stores only aggregated CE sums/counts, branch
response profiles, interactions, controls, hashes, and execution audits—no raw tokens, logits, activations, or hidden
states. Deployed parameters saved and added are both zero.
