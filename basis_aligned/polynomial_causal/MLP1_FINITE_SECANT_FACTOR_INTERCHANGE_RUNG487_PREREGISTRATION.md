# Rung 487 preregistration — exact MLP1 finite-secant factor interchange

Written after rung486's registered strong null and before any rung487 model outcome.

## Question and evidence-determined scope

Rung486 measured the complete finite block-1 carrier cube for MLP0 branches T, C, and I. Its seven-term profiles
were stable, but neither current token nor ordered previous×current token predicted held-out effects. The categorical
routes are therefore closed at this grain.

The carrier-mask order is `D, A, D×A, M, D×M, A×M, D×A×M`, where D is the direct MLP0 residual write, A is
attention1's write, and M is MLP1's write. The dominant mean term was **M**, not D×A: M's absolute-profile shares
were `.526/.752/.595` for T/C/I. This selects MLP1's exact finite response as the first continuous object. It does
not erase the other carrier terms or claim that MLP1 alone is the whole circuit.

Rung487 asks whether differences between T, C, and I at this dominant carrier live mainly in:

1. the branch-induced change entering MLP1; or
2. the live state that multiplies and reads that change.

This directly addresses “different inputs, same downstream use” versus “same input factor, different use.” It fits
no rank, sparse code, token table, or requested number of components.

## Exact quadratic polarization

For normalized MLP1 input `z`, write

`q(z) = Down1[(Left1 z) elementwise-multiplied-by (Right1 z)] + bias1`.

For branch `b` in `{T,C,I}`, let `z_N` be the native input and `z_b` the input when b is absent and attention1
recomputes naturally. Define

`delta_b = z_N - z_b`, and `mid_b = (z_N + z_b)/2`.

Then the finite MLP1 change is exactly

`q(z_N)-q(z_b) = B(delta_b,mid_b)`, where

`B(delta,mid) = Down1[(Left1 delta)*(Right1 mid) + (Left1 mid)*(Right1 delta)]`.

The bias cancels. Reciprocal rescaling and joint permutation of MLP1 product coordinates leave B unchanged; a global
Left/Right exchange swaps the two displayed summands but not their sum. This is an exact continuous secant, not a
first-order derivative.

## Cross-factor writes and physical tests

For every ordered pair of distinct branches `(b,d)`, construct:

- `OWN_b = B(delta_b,mid_b)`;
- `CONTEXT_b<-d = B(delta_b,mid_d)`, keeping b's change but using d's live-state factor;
- `DIRECTION_b<-d = B(delta_d,mid_b)`, keeping b's live state but using d's change; and
- `BOTH_b<-d = B(delta_d,mid_d)`, the donor's complete finite secant.

Keep target branch b absent at MLP0 and let attention1 recompute in b's absent trajectory. At MLP1, start from
`q(z_b)` and add each constructed secant, then let layers2--17 recompute normally. `OWN_b` must reproduce injection
of the native MLP1 write. The cross writes are exact bilinear cross-factor interventions, not claimed model states.

Run all three unordered branch pairs in both directions. Report exact MLP1-write error, centered output-vector
cosine/error, downstream per-token CE-effect cosine/error, equality-positive results, and16 cyclic position-shift
controls. No cross factor is selected from a search: every T--C, T--I, and C--I comparison is fixed in advance.

## Data split

Use the same hash-bound1,000 documents. Discovery is documents0:500 split at250. Documents500:1000 remain unopened
unless the discovery license holds. Final evaluation data remain unopened.

## Frozen predictions

### A — exact and lawful instrument

- Rung486 source/result, all ancestor, row, model, and preregistration hashes match; rung486 has A/B true, C/D/E
  false, validation closed, and the registered continuous-state next step.
- Native and each branch-absent trajectory replay exactly. The polarization identity has float32 relative squared
  error at most`1e-8` and deployed-BF16 error at most`1e-5` for T/C/I.
- Injecting `q(z_b)+OWN_b` reproduces the native MLP1 write and its physical suffix result at relative squared error
  at most`1e-5`; calls and injections are exact.
- Every own and cross secant write and physical CE effect is nonzero. Validation and final data stay unopened before
  the discovery decision.

### B — stable own finite responses

For T, C, and I, OWN's per-token physical effect must have cross-half cosine at least`.80`, with a half0-fitted
scalar predicting half1 at relative error at most`.50`. This prevents declaring factor interchange on an unstable
target.

### C-context — a pair shares the live-state factor

For at least one unordered branch pair, `CONTEXT_b<-d` must predict OWN_b in both directions and both discovery halves
with physical-effect cosine at least`.80` and best scalar-adjusted relative error at most`.50`. In every cell its
cosine must exceed `DIRECTION_b<-d` by at least`.15`. Its same-position MLP1-write cosine must exceed the95th
percentile of the16 position-shifted donor-midpoint controls by at least`.15`.

This means the two branch-induced changes remain distinct while their continuous live-state multipliers are
interchangeable at MLP1.

### C-direction — a pair shares the branch-change factor

Symmetrically, for at least one unordered pair, `DIRECTION_b<-d` must meet the same `.80/.50` physical bars in both
directions and halves, beat `CONTEXT_b<-d` by `.15`, and beat16 position-shifted donor-direction write controls by
`.15`.

This means the two live states use an interchangeable change direction differently.

For a given pair, C-context and C-direction are mutually exclusive. Intermediate or asymmetric results identify no
shared factor. Multiple different pairs may qualify if they independently meet every fixed clause.

### D — factor-sharing graph stability

At least one C edge must exist. Recompute the graph separately in documents0:250 and250:500; the same edge type and
unordered branch pair must occur in both halves. Report all edges and all failures, not only the best pair.

### E — held-out documents

Open documents500:1000 only if A/B/D hold. Freeze the complete discovery graph, factor definitions, and thresholds.
In each validation quarter, every claimed edge must again satisfy its `.80/.50`, opposite-swap margin, and
same-position control clauses. No new edge is added during validation. The graph validates only if every discovery
edge transfers.

## Nulls and routing

The scientific strong null fires if A, B, or D fails.

- If A alone fails, repair only the named polarization/replay/accounting defect.
- A validated context-sharing edge licenses cross-document live-state-factor interchange within that branch pair.
- A validated direction-sharing edge licenses cross-document branch-change-factor interchange.
- If no pair shares either factor, keep T/C/I separate and move within each branch: pair the exact finite secant with
  the suffix's integrated response along that secant, then test continuous state-conditioned prediction across
  examples. Do not return to token/bigram tables, side subsets, or rank tuning.

Even a positive result is factor localization, not a complete MLP0 explanation or compression claim. It must later
support cross-document extraction and selective intervention without unrelated damage.

## Price and stored data

Per batch, run one native trajectory, three branch-absent trajectories, and four physical secant arms for each of six
ordered branch pairs: `3,500` discovery forwards at batch size4, plus the same only if validation opens. Store only
contracted write/effect statistics, controls, hashes, and call audits—no logits, hidden states, or raw token rows.
