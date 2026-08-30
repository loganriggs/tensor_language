# Hourly strategic review — 2026-08-30 13:50 UTC

## Bottom line

The completed causal-response factor grid is now frozen before validation. The freeze
contains every training-nondominated rank pair under either the pooled or robust
training order, and every one of its three optimizer seeds: **9 rank pairs and 27
programs**. It binds the exact program bytes and exact training analysis. It does not
read validation or EVAL, select a winner, or identify a semantic circuit.

The main scientific update is still the weighting correction established immediately
before this review. Shared rank 32 reconstructs **65.17% of pooled FIT response
energy**, but the median block-relative recovery over all 36 source-owner/target-owner
interfaces is only **5.57%**. Most of the pooled success comes from the unusually
large `m16 -> m16` response family. Its residual is not unusually high-rank after
normalization. Therefore more universal rank or a larger `m16` private branch is not
the next move; the present candidates must first transport to held-out documents under
both pooled and block-relative metrics.

No strict whole-model ledger moves:

- certified removable storage: 29,196,288 / 545,904,054 = **5.348245316%**;
- named deletion cross-entropy: 0.57968 / 5.30682 = **10.923302467%**;
- unexplained deletion cross-entropy: 4.72714 nat = **89.076697533%**; and
- complete prediction/extraction/removal/OOD circuits: **0/68**.

The old eight-hour plan expired on 2026-08-29. It was inspected as historical evidence,
not treated as a current queue, and no old checkbox was relabelled.

## What was computed

For each intervention phase $p$, source circuit $s$, target circuit $t$, and FIT
document $d$, the measured response is $R_{pstd}$. A candidate tensor program predicts

$$
\widehat R_{pstd}
=\sum_{k=1}^{K_0} a_{pk}b_{sk}c_{tk}h_{dk}
+\sum_g \mathbf 1[s\in g]
\sum_{j=1}^{K_g}a^{(g)}_{pj}b^{(g)}_{sj}c^{(g)}_{tj}h^{(g)}_{dj}.
$$

$K_0$ is shared tensor rank and $K_g$ is private rank for each of six source-owner
groups. This remains an ordinary multilinear tensor program: there is no top-k router
or data-dependent support choice. Its two literal prices are

$$
P=100K_0+355K_g,
\qquad
C=K_0+6K_g,
$$

where $P$ counts persistent stored scalars and $C$ counts document coordinates that
must be inferred. Neither price is collapsed into a hand-chosen scalar simplicity
score.

The new freezer takes the union of two training Pareto frontiers:

1. $(P,C,\text{pooled median FIT MSE})$; and
2. $(P,C,\text{pooled median FIT MSE},\text{median worst-owner NRMSE})$.

Here a Pareto-frontier program is one for which no other program is at least as cheap
and accurate in every listed coordinate and strictly better in one. The union is:

$$
(K_0,K_g)\in
\{(1,0),(2,0),(4,0),(4,1),(8,0),(8,2),(16,0),(16,4),(32,0)\}.
$$

All three seeds for every rank pair are frozen rather than choosing the best training
seed. The resulting 27 artifact hashes are immutable. Six adversarial unit tests
passed in 0.21 seconds. The output reports
`candidate_selected=false`, `validation_values_read=false`, and
`eval_values_read=false`.

This is an anti-selection result, not an interpretation result. It makes the next
held-out comparison credible; it does not itself earn reconstruction, causal, or
whole-model credit.

## How much of the model is actually explained

The honest answer depends on the denominator:

1. On the pooled FIT response tensor, shared rank 32 explains **65.17%** of measured
   response energy.
2. Across an equally weighted median of the 36 owner interfaces, it recovers only
   **5.57%**. The first number is dominated by large-amplitude interfaces.
3. On held-out response prediction, the established fraction is **0%**, because the
   114 validation documents have not been opened for these candidates.
4. On the strict behavioral ledger, no new credit exists: **10.9233%** of deletion CE
   is attached to named behavior and **89.0767%** remains unexplained.
5. Under the terminal standard—prediction, extraction, selective removal, and OOD
   behavior—there are still **0/68** complete circuits.

Thus the response-factor work has found a compact regularity in one measured causal
interface library, not reverse engineered 65% of bilin18.

## Largest remaining gaps and confusing data

1. **No held-out transport.** Document coordinates can memorize training-specific
   response variation. The decisive question is whether a small number of physical
   source interventions predicts untouched source/target cells on new documents.
2. **Pooled versus interface-balanced error disagree sharply.** The 65.17% versus
   5.57% gap means ordinary MSE mostly rewards the large `m16` family. A candidate
   that predicts only that family is not a whole-library compiler.
3. **The failure is amplitude, not clearly tensor rank.** `m16 -> m16` has 9.676 times
   the matched-support residual energy of the next source owner, but only 1.104 times
   its normalized rank-16 unfolding tail. More private rank therefore lacks a current
   mathematical justification.
4. **No compositional interface certificate.** A response program has not yet been
   substituted through RMSNorm, residual addition, the next MLP, and attention while
   preserving its predictions.
5. **No stable semantic coordinates.** Low-rank factors are gauge-dependent: factors
   can permute and rescale without changing the tensor. Cross-seed atom stability and
   conditioning have not been established.
6. **No fresh edit consequence.** Reconstruction has not yet yielded an extracted
   behavior, selective removal with unrelated controls, or OOD prediction.

The only unrelated GPU job observed at the start was `band_head_concentration.py`; it
finished during this review and released the GPU. Its artifact belongs to another
thread and was not promoted or staged here.

## Candidate actions considered and pruned

- **Add more shared or owner-private training rank:** pruned now. It optimizes the
  already misleading pooled FIT objective, costs GPU time, and does not test transport.
- **Immediately fit a block-balanced rank-32 program:** deferred. First score the
  frozen programs under the block-relative metric; otherwise the new weighting could
  be selected after seeing which story is favorable. A future noise floor must also be
  frozen before relative error can safely weight tiny blocks.
- **Interpret CP atoms by inspection:** pruned until cross-seed permutation/scaling
  alignment and conditioning pass. A rotating factor is not a circuit.
- **Plain SAE, HOSVD, or weight-only compression:** pruned at this interface. These
  optimize geometry or local reconstruction and cannot resolve the observed causal
  weighting, composition, or OOD gaps.
- **Top-k or routed sparse program:** pruned from the tensor-program claim unless its
  routing law is itself represented and priced. Dynamic support would otherwise break
  the current multilinear compositional guarantee.
- **Global polynomial expansion of the whole transformer:** pruned for cost and poor
  falsifiability. Exact polynomial/tensor folding remains useful only on bounded
  interfaces after a predictor survives.
- **Arbitrary semantic or DAG search:** pruned until there is a validated primitive
  library and an intervention-based objective. Search now would optimize labels rather
  than causal usefulness.

## Ranked top five

1. **Freeze every training-frontier program before validation — executed.** Highest
   immediate value because it closes the selection leak cheaply and is required for a
   falsifiable held-out result. Numerical outcome: 9 rank pairs, 27 programs, all exact
   artifacts bound, no held-out access.
2. **Score all 27 programs on the sealed 114-document role.** Use unconditional codes
   and calibrated 2/4/8/16-source-arm prediction. Report pooled error, every owner
   block, noise-thresholded block-relative error, CE/KL consequence where defined, and
   inference cost. This directly distinguishes a reusable program from a training code.
3. **Branch prospectively on the validation failure pattern.** If transport works but
   remains amplitude-dominated, compare one fixed block-balanced tensor fit at matched
   $(P,C)$ and bounded pooled regression. If transport fails broadly, reject response
   factorization v1 rather than repairing it indefinitely.
4. **Certify any survivor modulo tensor gauge.** Align all three seeds under
   permutation and reciprocal scaling, measure factor conditioning/Kruskal ranks, and
   require atom-level finite interventions to agree. This tests whether compressed
   coordinates are extractable components rather than an arbitrary basis.
5. **Test whole-model causal composition.** Substitute the survivor across an early
   MLP/RMSNorm/residual boundary, predict fresh interventions, then measure extraction,
   selective removal with unrelated-target controls, and OOD transport. An empirical
   controllability/observability quotient is the best alternate entry point if the
   response program fails because it selects directions by downstream causal effect
   rather than activation variance.

The order is deliberate: validation can kill the entire response-factor branch
cheaply; gauge mathematics cannot rescue a predictor that does not transport; and
whole-model edit claims require both.

## Action completed this review

The first create-only freezer published the correct current 9-rank/27-program census,
but independent audit returned **NO-GO**. It included two per-seed training scores even
though Amendment 14 prohibited candidate scores, did not close mutation windows across
its repeated reads, and lacked post-link semantic replay and publication-race tests.
The exact artifact and failure are preserved as nonpromotive; the namespace was not
silently overwritten.

Prospective Amendment 15 authorized only a lifecycle repair and removal of the score
fields. V2 now binds the same 27 programs using only identity, byte/hash, and literal
price fields. It asserts exact training-analysis/grid identity, snapshots and
revalidates every source/input/program before and after link, semantically reloads the
linked manifest, and passes six focused mutation/publication tests. It published
`causal_response_factorization_v1_candidate_freeze_v2.json` with logical manifest
SHA256 `3b386a38e9bf79f90e01c87ef6471770dfd7bae73ffb89ef3749a182968b5500`.

This is the candidate freeze proposed by the previous review; independent v2 re-audit
is in progress. Validation remains sealed; candidate selection, semantic claims,
ledger promotion, and terminal-circuit credit remain false.
