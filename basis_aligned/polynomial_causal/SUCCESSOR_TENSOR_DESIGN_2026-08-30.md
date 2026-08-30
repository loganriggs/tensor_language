# Ordered-successor fixed tensor design

Status: prospective, model-agnostic design only. This document grants no row,
checkpoint, GPU, or outcome authority.

## Evidence boundary

The relevant campaign object is natural-text L8H7 (zero-indexed layer 8, head 7),
not every behavior informally called successor. `succ_map.py` shows a weights-readable
digit successor image; `succ_general.py` extends that image to weekdays/months and
reports target damage 0.1478 nat versus 0.00267 elsewhere. Its registered control-head
prediction failed, however. `succ_twin_scale.py` later finds pooled L8H7 removal damage
0.2306 nat across digits/weekdays/months/years while L14H4 is dormant (-0.0043 nat) and
the joint interaction is 0.0069 nat.

These artifacts establish a causal owner and readable weights, not an extracted
program. The weight tests feed RMS-normalized token embeddings into value matrices in
place of live layer states, and the causal tests use mean ablation rather than an exact
head-free parent. They do not measure exact physical replay, OOD extraction, or a
simpler executable component.

There is a separate short-list successor assay under
`basis_aligned/qk_mdl/algo_tasks/successor/` whose dominant route is L8H3 and the
block-0 value bus. It must not be pooled with L8H7 natural-context results: list
continuation and a prior-token-conditioned successor effect are distinct estimands.

## Exact model object

Let `z8_k` be the RMS-normalized live input to attention layer 8 at key position `k`,
and `z0_k` the RMS-normalized block-0 input whose value projection was saved as `v1`.
For L8H7, head width `r=128`, residual width `D=1152`, and learned scalar `lambda`,
the delivered write is

```text
A_qk = <R_q Q1 z8_q, R_k K1 z8_k>/r
     * <R_q Q2 z8_q, R_k K2 z8_k>/r * 1[k <= q]

w_q = sum_k A_qk O [(1-lambda) V8 z8_k + lambda V0 z0_k].
```

`Q1,K1,Q2,K2,V8` are the L8H7 row slices; `V0` is the L0H7 value row
slice on the shared v1 bus; `O` is the L8 c_proj column slice. RoPE `R_q` and the
causal support are fixed tensors. Q/K RMSNorms are explicit scalar nonlinear edges.
There is no semantic successor lookup in this formula: succession is a property of
the learned physical map and its selected states.

The new pure module implements the conditional value/output part

```text
W_q(A,z8,z0) = sum_k A_qk O [(1-lambda)V8 z8_k + lambda V0 z0_k]
```

and exposes its gauge-invariant folded physical map

```text
M = O [(1-lambda)V8 | lambda V0],  shape [D, 2D], rank <= r.
```

Dropping `V0` is not a valid simplification: later layers consume the saved block-0
value path, and `succ_map.py` explicitly mixed both paths. Dropping `V8` is likewise
not licensed by those raw-embedding weight tests.

## Simplicity and price

For a standalone conditional value program, absorb the fixed scalar into the two
right factors and serialize `O`, `(1-lambda)V8`, and `lambda V0`:

```text
P_OV(r) = r(D_out + D_current + D_v1).
P_OV(128) = 128 * (1152 + 1152 + 1152) = 442,368 floating scalars.
```

The shared `GL(r)` factor gauge reduces generic physical dimension by `r^2`, but it
does not reduce literal stored scalars unless a canonical factorization is serialized.
At `r=128`, that generic quotient dimension is `442,368 - 16,384 = 425,984`.
Both raw count and quotient dimension must be reported. Reusing L0H7's `V0` for free
is prohibited in a standalone price; a joint multi-circuit ledger may charge it once.

An autonomous exact head also needs four Q/K factors:

```text
P_full(r) = r(4D + D_current + D_v1 + D_out) = 7Dr
P_full(128) = 1,032,192 floating scalars.
```

That is the native per-head factor count, so exact replay at rank 128 is **not yet a
simpler component**. A genuine compression claim requires a prospectively selected
rank `s<128` for the folded OV map and, for autonomous credit, separately compressed
QK factors or a cheaper fixed selector. Consuming captured native `A_qk` omits `4Dr`
parameters but is teacher-forced conditional extraction, not a zero-native-call
circuit.

## Router boundary

The native continuous QK product is a fixed tensor network and is not a discrete
router. A lexicon table or token-membership mask is also a fixed tensor if explicitly
serialized and contracted with token one-hots. The following are forbidden free
routers:

- Python/regex classification of digits, months, weekdays, or years;
- selecting the nearest or highest-scoring predecessor with argmax/top-k;
- applying the OV map only when an external `successor_context` label is true;
- choosing a family-specific expert from decoded token identity.

Such logic must either remain evaluation-only or be compiled into and priced as a
fixed polynomial/finite-state tensor. Final argmax is an endpoint metric, not part of
the extracted residual-write program.

## Prospective arms and gates

All arms must start from the same exactly replayed residual with the L8H7 delivered
write absent:

1. `NATIVE`: head-free parent plus the exact native L8H7 write.
2. `REMOVE`: head-free parent only. This is literal zero contribution, not mean
   ablation.
3. `CONDITIONAL_OV`: parent plus the candidate two-source OV write under captured
   native scores. Diagnostic only; it localizes failure to WHAT versus WHERE.
4. `FULL_EXTRACT`: parent plus independently executed compressed QK and two-source OV.
   Only this arm may receive autonomous extraction credit.
5. `DERANGED`: same parent plus a pre-frozen same-rank/same-spectrum physical-map null.
6. `L14H4_CONTROL`: report the dormant twin separately; do not silently add it to the
   extracted program.

Required physical identities are native replay, head-free replay, exact current/v1
state and score hashes, arm call ledgers, and zero L8H7 QK/value/output calls in
`FULL_EXTRACT`. The conditional arm must be explicitly labeled teacher-scored. Measure
successor-minus-self/previous logit margins, target CE, per-document global CE/top-1,
and matched same-token non-successor contexts. Promotion requires removal necessity,
held-out recovery LCB, bounded off-target damage (campaign ceiling 0.01 nat), and
failure of the same-price derangement.

The pure helper materializes the deranged physical matrix for verification. An actual
same-price executable must SVD-factor it back into one rank-`s` output factor and two
split right factors and serialize those factors; deploying the dense `[D,2D]` matrix
would invalidate the matched-price control.

## OOD split

- FIT/rank selection: digit transitions 1->2 through 8->9, with source forms divided
  by document and surface tokenization.
- Development only: held-out digit surface forms and distances; no weekday/month/year
  cells may influence rank, gains, nulls, or stopping.
- Final OOD, opened once: entire weekday, month, and year lexicons, reported separately
  and pooled. Hold out cycle/wrap members prospectively because the model need not
  implement cyclic succession.

Rows and documents must be disjoint across roles. Token occurrence is not the
independent unit: inference is source-document clustered. The successor target mask is
evaluation metadata only and must never gate the executable arm.

## Current blockers to a model run

1. No artifact yet freezes live `z8`, saved `z0/v1`, L8H7 scores, and the exact
   head-free parent on disjoint roles.
2. Existing results do not establish a compressed rank for `M`, much less for QK.
3. The fused attention projections make “zero native L8H7 calls” require a typed
   executor that computes the other eight heads without the H7 slices; subtract/add
   hooks are physical interventions but not zero-call extraction.
4. `succ_general`'s control-head gate failed, so new matched spectral and token-context
   nulls are mandatory.
5. Natural successor labels based on “some predecessor within 128 tokens” do not prove
   which source position L8H7 used. An autonomous QK extraction must predict the native
   continuous score/write, not inherit the semantic label as a selector.

Until those blockers close, the honest result is a well-defined conditional fixed
tensor and a causal head owner—not a simpler autonomous successor circuit.
