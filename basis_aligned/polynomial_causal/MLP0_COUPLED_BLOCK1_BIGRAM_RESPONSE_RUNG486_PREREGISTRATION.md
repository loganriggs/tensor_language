# Rung 486 preregistration — coupled block-1 carriers and previous×current-token response

Written after rung485's registered strong null and before any rung486 model outcome.

## Why this is new rather than another MLP0 compression

Rung485 exactly reconstructed MLP1's two multiplicative sides. For both `T` and `I`, either side alone correlated
about `.90--.92` with the complete route, but its best rescaled error was `.40--.44`, above the frozen `.35` fit
bar. Neither side was selected. The average three-term `(Left,Right,Left×Right)` profiles were almost identical
between T and I (cross-branch cosine `.9998`) and almost perfectly stable across halves, while a scalar T-to-I
prediction failed with held-out relative error `.952`. Thus the average MLP1 anatomy is shared but does not explain
the example-level computation. Current-token means also failed to predict T's downstream effect: cross-half token
profile cosine `.425` unweighted/`.672` support-weighted, per-position correlation `.028`, and RMSE change `-0.85%`
relative to one global mean.

The older MLP0 dossier must constrain the next step. It already found:

- an exact current-token table plus a current×attended-context correction;
- on covered examples, `(previous token,current token)` explains substantially more of that correction than current
  token alone;
- 98.3% of MLP0's causal effect is mediated by blocks1--3; and
- a rank-256 quadratic program can recover 97.9% of MLP0's function, but its features are distributed and their
  informal names failed a code-level verification test.

Therefore rung486 does **not** refit a rank, sparse code, Tucker model, or token table. It tests whether the old named
previous×current variable predicts the exact finite downstream response, and it measures the three block-1 carriers
jointly so nonlinear interactions are not assigned to one consumer by construction.

## Exact branches and exact carrier cube

Reuse rung401's exact MLP0 branches `T` (current-token main effect), `C` (attention0-context main effect), and `I`
(centered current-token×context interaction). `S`, the much smaller normalization-gain term, remains in the
background in this rung; this experiment does not claim full MLP0 completion.

For each branch `b`, run two complete block-1 trajectories:

1. native; and
2. `b` absent at MLP0, with attention1 and MLP1 allowed to recompute normally.

Capture the native and absent versions of three writes:

- `D`: MLP0's direct residual write;
- `A`: attention1's output write; and
- `M`: MLP1's output write.

Then run all eight choices of native versus absent `(D,A,M)`. Each choice injects the captured write at its named
site; layers2--17 recompute normally. The empty corner reproduces the natural branch-absent trajectory. The full
corner reproduces the native state immediately after MLP1 and must reproduce native logits. Mixed corners are
finite, explicitly named cross-state interventions.

For per-token performance `y(S)=-CE(S)`, compute the seven Möbius effects

`D, A, M, D×A, D×M, A×M, D×A×M`.

Their sum must equal the complete native-minus-absent branch effect. No proper subset is selected and no term is
discarded. The resulting object is a response tensor indexed by branch, occurrence, and the seven physical carrier
terms.

## Frozen data and previous×current support

Use the same hash-bound1,000 documents and256 input positions as rungs481--485. Discovery is documents0:500 split
at250. Position0 is excluded because it has no within-row predecessor.

Before model outcomes, freeze the287 ordered `(previous token,current token)` pairs that occur at least eight times
in each discovery half. They cover `7,859` and `8,292` positions. Of these,269 occur at least four times in each
reserved validation quarter, covering `8,016` and `7,626` positions. These exact sets are recomputed from input IDs
and their identities are stored in the receipt.

On documents0:250, estimate for every supported pair:

- its mean complete-route effect; and
- its mean seven-term carrier vector.

Apply those values unchanged to occurrences in documents250:500. The matched baseline uses means indexed only by
the current token, fitted on exactly the same supported half0 positions. Also report a previous-token-only baseline
and one global mean. The primary comparison is pair versus current-token: it asks whether named local context adds
predictive information beyond the token lookup that failed in rung485.

Use the same16 cyclic position shifts as a same-position control. All mean-profile cosines are reported both
unweighted and weighted by the square root of the smaller half's pair count.

## Frozen predictions

### A — valid exact instrument

- All parent, data, source, model, and preregistration hashes match. In particular, rung485 source/result hashes are
  `2449de7b...` and `a1ecf427...`, rung485 has A true, B/C/D/E false, validation closed, and its registered next step.
- Native and absent captures reproduce their separately executed trajectories; all-native and all-absent cube
  corners reproduce native and absent logits with relative squared error at most`1e-12`.
- MLP0 analytical/deployed branch identities retain their `1e-8/1e-5` limits.
- All calls and injections are exact. Möbius closure is at most`1e-8` for every branch and half.
- T, C, and I complete routes and all seven carrier terms are nonzero. Validation and final evaluation data remain
  unopened before the discovery decision.

### B — the complete carrier decomposition is stable

For T, C, and I, normalize the seven signed mean carrier effects by their total absolute size. Cross-half profile
cosine must be at least`.90` and every absolute share may change by at most`.15`. This is a stability condition on
the full finite response, not a claim that a small subset suffices.

### C — previous×current context predicts T's downstream response

On frozen supported occurrences in documents250:500, the half0 pair means must:

- reduce complete-route RMSE by at least`10%` versus half0 current-token means;
- correlate with complete-route effects at Pearson at least`.30`; and
- exceed the95th percentile of the16 position-shift cosine controls by at least`.15`.

For the flattened occurrence×seven carrier tensor, pair means must reduce RMSE by at least`10%` versus current-token
means and have cosine at least`.50`. Report the same measurements for C and I without requiring them to pass C.

### D-shared — T and I are two inputs to one downstream carrier law

For each half, compare the supported-pair×seven mean response tensors for T and I using the frozen minimum-count
weights. D-shared holds if cosine is at least`.90` in both halves and one scalar fitted T-to-I on half0 predicts the
half1 I tensor with relative error at most`.35`.

### D-split — T and I require different downstream carrier laws

D-split holds if the weighted tensor cosine is at most`.60` in both halves and, on half1 occurrences, each branch's
own half0 pair predictor beats the other branch's predictor by at least`.10` relative RMSE. T and I may share an
average carrier profile without satisfying D-shared; intermediate results identify no relation.

The D outcomes are mutually exclusive.

### E — held-out documents and pairs

Open documents500:1000 only if A/B/C hold and exactly one D outcome holds. Keep the original269 supported pairs,
document0:250 pair/current means, carrier definitions, relation, and all thresholds frozen. In each validation
quarter require pair-over-current RMSE improvement at least`5%`, positive complete-route correlation, flattened
carrier cosine at least`.35`, B's carrier-profile stability, and the same D relation. No refitting is allowed.

## Nulls and next actions

The scientific strong null fires if A, B, or C fails or neither D outcome holds.

- If A alone fails, repair only the named capture/accounting defect.
- If B fails, retain the complete per-example response tensor; do not average carrier effects into a circuit.
- If C fails, the named local bigram does not explain downstream use. Move to a continuous live-attention0-state
  reader along the exact finite branch path, using held-out causal prediction rather than another categorical table.
- If C passes but neither D relation holds, keep T and I separate and test their context-conditioned readers
  independently.
- A validated D-shared licenses exact cross-branch carrier interchange; a validated D-split licenses separate
  branch-specific extraction. Either must preserve unrelated behavior in a later removal/interchange rung.

This rung identifies or rejects a named contextual variable and a full finite path decomposition. It saves no
parameters and makes no claim from low rank, reconstruction error, or the number of stored means.

## Price and stored data

Discovery uses one native capture plus, for each of three branches, one absent capture and eight carrier-cube arms:
`3,500` full-model forwards at batch size4. Conditional validation costs the same. Store contracted CE effects,
carrier profiles, pair counts/means, controls, hashes, and audits only—no logits, hidden states, or raw token rows.
