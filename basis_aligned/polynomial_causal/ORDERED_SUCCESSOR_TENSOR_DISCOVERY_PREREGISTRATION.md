# Ordered-successor autonomous tensor discovery v1

Status: prospectively frozen discovery design. No SELECT or OOD row tensor, model
forward, or circuit outcome was opened while authoring this document.

## Narrow question and claim boundary

Can the causally implicated natural-text L8H7 contribution be replayed as a fixed,
autonomous QK-plus-two-source tensor and compressed by a weight-only truncated SVD of
its physical OV map without losing its ordered-successor effect or gaining comparable
behavior from a same-price spectral null?

A discovery pass licenses only a separately authorized, once-opened fresh OOD assay.
It does **not** establish a universal successor algorithm, OOD generalization,
minimality, a lexicon-independent symbolic state, upstream writers, or a terminal
selective-removal/extraction certificate. The weekday/month/year results in
`succ_general`/`succ_twin_scale` are historical evidence, not fresh OOD evidence.

This assay is about natural-text L8H7. The distinct six-token list-continuation assay
under `qk_mdl/algo_tasks/successor/` is L8H3-dominant and is excluded.

## Exact fixed tensor

At L8H7 the autonomous score and delivered write are

```text
A_qk = (<R_q Q1 z8_q, R_k K1 z8_k>/128)
     * (<R_q Q2 z8_q, R_k K2 z8_k>/128) * 1[k <= q]

w_q = sum_k A_qk O[(1-lambda)V8 z8_k + lambda b0_k].
```

`z8` is the live normalized L8 attention state. `b0` is the already-projected
`[head=7,dim=128]` slice of block 0's saved first-value bus. The L8 replacement receives
this bus directly from `AttentionEvent.first_value`; it must not reconstruct a 1152-D
block-0 state or apply `V0` a second time. RoPE and causal support are fixed;
the four Q/K RMSNorms are explicit scalar nonlinear edges. No token label, lexicon,
argmax, top-k, regex, family selector, or successor-context mask enters execution.

The weight-only folded physical OV map is

```text
M = O[(1-lambda)V8 | lambda I_128], [1152,1280], rank <= 128.
```

Compute one CPU-float64 SVD before any SELECT forward. For every frozen rank
`s in {8,16,32,64,96,128}`, serialize/hash `U_s diag(sigma_s)` and the
`[s,1152]` current-state plus `[s,128]` saved-bus splits of `V_s^T`. The same-price
null retains the singular values and mismatches the
right singular directions by one pre-frozen fixed-point-free cyclic permutation.
Materialized factors, not an SVD recipe alone, are authority-bound. Repeated-value
ties therefore cannot change the deployed realization after authority.

## Literal price

The target head uses four rank-128 Q/K factors plus one rank-`s` output factor and
two rank-`s` right factors:

```text
P_shared-bus(s) = 4*128*1152 + s*(1152 + 128 + 1152)
                = 589,824 + 2,432s.
```

The exact native shared-bus head stores four Q/K factors, `V8`, and `O`; its saved
bus enters through a fixed identity and does not add a learned `128*128` factor.
Thus `P_native = 589,824 + 147,456 + 147,456 = 884,736` learned floats.

| s | stored floats | fraction of native L8H7 |
|---:|---:|---:|
| 8 | 609,280 | 0.6887 |
| 16 | 628,736 | 0.7106 |
| 32 | 667,648 | 0.7546 |
| 64 | 745,472 | 0.8426 |
| 96 | 823,296 | 0.9306 |
| 128 | 901,120 | 1.0185 |

The current-only and v1-only rank-128 diagnostics cost `884,736` and `753,664`
floats respectively. The score-conditioned OV map alone costs `311,296`, but is not an
autonomous arm and receives no executable credit. The other eight L8 heads are replayed identically in
every nonnative arm and are matched experimental background, not free parts of the
candidate.

These prices are **conditional on the deployed shared first-value bus**. The L0H7
producer slice that mints that bus costs `1152*128 = 147,456` learned floats and is
authority-bound but not duplicated in each L8 candidate. An end-to-end standalone
claim must add that producer once globally: the exact rank-128 total is therefore
`901,120 + 147,456 = 1,048,576`. A joint multi-consumer ledger may amortize the same
producer exactly once, but may not call it free. This discovery replaces only L8
attention and therefore cannot claim standalone extraction. No MDL-bit claim is made;
these are literal unquantized float counts.

The executable backend reports full serialized storage separately from the conditional
target-circuit price. Full replay stores `7,962,689` values. Candidate and deletion
arms use a block-structured background that physically omits the target H7 native V
rows and c_proj columns, so their common background stores `7,667,777` values and an
executable rank-`s` candidate stores `7,667,777 + 2,432s`. Exact tensor shapes certify
that the omitted `294,912` values are absent; all three backend modes report
`storage_closed=true`. This whole-site executable price is not the target-circuit
price and is never hidden in a compression comparison. The current factor class still
cannot materialize the lower source-omission prices without storing zero-valued unused
factors; CURRENT_ONLY and V1_ONLY remain launch-blocking until an omission-aware
factor backend is source-closed.

For runtime reporting, count the target head's projection multiply-adds and the
causal-pair contractions separately. No compute reduction is inferred from storage.

## Immutable arms

Every nonnative arm replaces the entire attention-8 module in the hook-free campaign
runtime, analytically replays the other eight heads, and makes zero native attention-8
or attention-8 Q/K/Q2/K2/V/O calls. All MLPs and other attention sites remain native.

1. `native`.
2. `full_attention8_replay`: exact analytic replay from checkpoint factors.
3. `head8_7_deleted`: exact replay with H7's delivered write zero.
4. For each registered rank, `head8_7_both_r{s}_true`.
5. For each rank, matched `head8_7_both_r{s}_spectral_null`.
6. `head8_7_current_only_r128` and `head8_7_v1_only_r128`, diagnostic and
   nonpromotive.

The true rank-128 arm must replay `full_attention8_replay`; the latter must replay
native. Interpret no behavior unless both maximum absolute final-logit errors are at
most `1e-4` and row-mean `KL(native || replay)` is at most `1e-8`. H7 deletion must be
literal zero, not a mean ablation. Spectral null factors have exactly the true arm's
stored price, rank, singular values, tensor currency, and QK factors.

## Roles, support, and controls

This discovery authority contains SELECT only and cannot name or load OOD. Before the
first forward it must bind the pushed source commit and blobs, exact SELECT row hash
and document count, exact ordered lexicon registry, mask/support hash, pinned model
files, every materialized program/null hash, arm order, and prices. Rows are contiguous
CPU int64 `[N,257]`; logits are unsliced float32 `[N,256,50304]`; positions 64--255 are
scored. All arms execute identical ordered rows in batches of four.

SELECT uses only prospectively registered digit successor pairs and surface forms.
No SELECT outcome may modify lexicons, ranks, SVD factors, gains, null permutation,
cells, or thresholds. The future embargoed OOD role contains whole weekday, month, and
year lexicons with cycle/wrap cases reported separately; it requires a new authority.

For each lexicon, `ordered_successor_masks_v1.py` partitions target occurrences into:

- remote correct predecessor with no target copy (`positive_clean`, primary);
- correct predecessor plus prior target copy;
- copy only;
- remote wrong same-family source (`wrong_source_clean`);
- no family source (`no_source_clean`); and
- local/ambiguous exclusions.

Masks are evaluation-only and never gate a tensor write. Report every cell separately
and pooled. Promotion requires `positive_clean`, `wrong_source_clean`, and
`no_source_clean` each to contain at least 200 positions across at least 30 source
documents; copy cells are mandatory descriptive controls but may be labeled
underpowered. Also report pair-level document/position occupancy so one transition
cannot silently dominate.

## Metrics and inference

For every arm/cell retain per-document float64 sums and integer counts for target CE,
`KL(native || arm)`, top-1 changes, and successor-minus-source-item logit margin. The
estimand is row-position weighted; uncertainty uses a paired source-document cluster
bootstrap with 20,000 shared draws and seed `2026083013`.

All promotive ranks, CE/margin contrasts, two specificity controls, replay gates, and
true-null comparisons form one family. Use one replicatewise maximum absolute centered
error and its sorted zero-based index 18,999 (no interpolation) to form simultaneous
95% bands. Nonlinear recovery ratios are recomputed inside each bootstrap draw; they
are not bootstrapped from precomputed ratios. A zero/nonpositive deletion denominator
makes recovery unevaluable and nonpromotive.

For cell `C`, deletion damage and candidate recovery are

```text
D(C) = CE_deleted(C) - CE_native(C)
R_s(C) = [CE_deleted(C) - CE_true_s(C)] / D(C).
```

A true rank is discovery-promotable only when all common integrity/support gates and:

1. `D(positive_clean)` has simultaneous LCB above zero;
2. each DiD `D(positive_clean)-D(wrong_source_clean)` and
   `D(positive_clean)-D(no_source_clean)` has LCB above zero;
3. CE recovery point is at least 0.80 and simultaneous LCB at least 0.60;
4. successor-margin recovery has the same thresholds and direction;
5. true versus its same-rank spectral null CE benefit has LCB above zero, while the
   null's recovery UCB is below half the true point recovery;
6. candidate-minus-native all-position CE has UCB at most 0.01 nat, and absolute CE
   drift on each powered control cell has simultaneous UCB at most 0.01 nat; and
7. all arm call ledgers, row/support/program hashes, finite currencies, and replay
   identities pass.

Among passing true arms select the lowest literal stored price, then the lower rank.
No rank passes means the fixed native QK plus low-rank OV hypothesis is not promoted.
Current-only/v1-only arms diagnose source omission; they cannot rescue or promote a
failed both-source family under v1.

## Fail-closed lifecycle and missing production slice

The committed Python is intentionally a pure runner scaffold: arm topology, authority
types, hashes, exact call-ledger validation, production tensor currency, and
per-document sufficient statistics. It has no model loader, row loader, CLI, output
path, result scorer, or receipt publisher. A later source-closed adapter must add an
owner-aware lock, pre-forward authority mint, exact site-8 analytic backend, append-only
failure manifest, atomic payload, semantic replay, and receipt-last publication. Until
that separately reviewed slice exists, GPU launch is **NO-GO**.
