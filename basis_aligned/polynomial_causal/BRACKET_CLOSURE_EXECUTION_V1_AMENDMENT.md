# Bracket closure execution v1 inference amendment

Status: prospectively frozen before model, row, or outcome access.

This amendment resolves the role, bootstrap, and extraction definitions created by
the fresh-row amendment.

1. FIT is metadata-only and is never forwarded. SELECT and OOD are the only forward
   roles. Each contains prespecified prose and code coordinates. The original
   separate-domain and synthetic roles are superseded; no synthetic claim is made.
2. Each bootstrap replicate independently resamples source documents within SELECT
   and within OOD. Within a role, prose/code and every cell/arm coordinate share the
   same document multiplicities. Domain is a coordinate, not a resampling stratum.
3. The replicatewise maximum absolute error is taken jointly over every SELECT and
   OOD promotive coordinate. There are exactly 20,000 draws using seed 2026083013 and
   zero-based order statistic 18,999, without interpolation.
4. There is no rescue. SELECT must pass every registered natural integrity, exact
   replay, call-ledger, support, compatible-target, family/type, specificity,
   collateral, and fixed-null gate. OOD must independently pass integrity, replay,
   calls, and support; compatible-target and specificity LCBs must exceed zero in
   prose and code; collateral UCB must be at most 0.01 nat; and fixed-null separation
   must pass. Both role conjunctions are required. Failure of either is terminal
   nonpromotion.

For each role/domain let `D=CE(delete)-CE(native)`. Exact executable extraction is
`E=(CE(delete)-CE(stored_all_heads))/D`. This is expected near one after replay passes:
it certifies the stored H8 contribution added to the identical stored eight-head
background with zero native L13 calls. It is not compression or standalone sufficiency.
OOD requires E point at least 0.60, E LCB at least 0.40, and OOD E at least 50% of
SELECT E. The last requirement is the stronger simultaneous-family lower bound
`LCB(E_OOD - 0.5 E_SELECT) >= 0`, not a pointwise retention check. Define spectral recovery
`R=(CE(delete)-CE(spectral_derangement))/D`. Its UCB must be below `0.5*E_point`, and
the true-versus-null normalized benefit `E-R` must have LCB above zero. Replay maximum
logit error and teacher KL remain separate integrity gates. V1 can earn exact
extraction, OOD confirmation, and removal evidence, but no simplification credit.

All ratios and the specificity maximum are recomputed inside every bootstrap draw
from document-balanced arm CE means. A nonpositive compatible deletion denominator
at the point or in any draw is terminal unevaluable, not clipped. The joint family
contains, per applicable role/domain, deletion stake, specificity, collateral margin,
true-versus-null benefit, spectral margin, the OOD extraction LCB floor, the OOD/SELECT
retention contrast, and SELECT family-specific deletion stakes. SELECT family gates
remain point-positive as originally registered, but their coordinates remain inside
the joint maximum-error family. No OOD family-specific positivity gate is added.

Execution uses consecutive batches of four rows in exact frozen role order, with a
short final batch only if the authority-bound row count is not divisible by four.
Each batch executes the exact arm order native, stored replay, deletion, spectral
derangement. This batching is numerical/call currency only and cannot select rows,
programs, masks, thresholds, or bootstrap multiplicities.
The spectral control permutation is contiguous CPU int64 `[128]`; its dtype, shape,
and exact ordered bytes must match the authority hash before materialization. The
authority separately binds the resulting full dense program state.

“Raw float32 logits” means the exact Bilin18 facade output after its single native
`30*tanh(raw/30)` output softcap; “uncapped” in the original text forbids any second
metric-side cap and does not remove that deployed model operation. CE and teacher KL
convert these float32 facade outputs to float64 before log-softmax and reduction.

The execution authority additionally binds the exact row authority and independent
row audit, not only the terminal row receipt.  Before every role load and at each
publication guard, execution must replay the receipt's authority, audit, committed
source, candidate-source identity, delimiter registry, and historical-exclusion
joins.  It must also bind and recheck the exact live model state tree (ordered state
names, dtypes, shapes, and bytes) across all forwards and through final publication.
The result persists the full 18-site native/replacement attention and MLP call ledger,
outer-forward/return closure, and native-call prohibition for every role/arm batch.

Success and failure share one immutable terminal-claim path.  Result, success receipt,
failure receipt, terminal claim, and lock are distinct authority-bound paths.  A
success callback must prove FAILURE absent before the common claim and again before
the receipt; a failure callback must bind any partial-result presence and exact hash.
Both paths recheck the stable lock inode and nonce plus the exact input aggregate
immediately before their receipt-last publication.  These lifecycle repairs change
no row, arm, metric, bootstrap, threshold, or scientific claim.
