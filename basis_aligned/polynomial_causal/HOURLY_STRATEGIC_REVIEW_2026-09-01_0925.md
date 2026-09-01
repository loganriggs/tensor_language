# Final strategic review — 2026-09-01 09:25 UTC

## Full goal

Compile bilin18 into a substantially smaller tensor program that is predictive on fresh and shifted text,
composable across independently fit replacements, manipulable under named signed interventions, and literally
simple after every scalar, byte, router state, index, and dependency is counted.  Low CE or a visually attractive
factorization alone is not completion.  Generic input-dependent top-k remains a compute policy; only a small fixed,
priced router state space counts as structural decomposition.

## What changed since the 08:30 review

1. The adopted QK64+MLP{4,0}@p768 point passed global two-byte and original-native signed gates:
   `511,758,646` scalars / `1,023,517,292` bytes / `+.012329` census /43 certs /`.986524` signed cosine.
2. A third preregistered p768 MLP cut reached `509,104,438` scalars and good OOD CE but dropped to38 certs, below
   its >=40 signal bar.  It closed manual prefix extension instead of being rescued by rank/subset tuning.
3. QK72 with the same two MLP cuts produced and fully gated a middle tier:
   `516,264,246` scalars / `1,032,528,492` bytes / `+.009301` /50 certs /`.990600` cosine.
4. The fixed Q/K certificate ray transferred to the distributed MLP programs at cosine `.9988–.9994`, vector R2
   `.9917–.9959`, and count errors1–2.  A second residual mode is predictive but fails its MLP-specificity control;
   retain the one-ray model as the conservative allocator.
5. The all-layer variable-rank model is identified on fit B (factor error1.223, rho.952, order18/18), but exact
   water-filling reselects the adopted `{0,4}@p768` pair.  The local MLP rank grid is closed.
6. Value ranks64–112 are price-closed: even the optimistically bracketed best rank is3.6838x worse per saved scalar
   than the adopted MLP exchange.
7. The invariant-Tucker toy instrument passes perfectly.  The first real MLP0 output-mode screen finds stable,
   non-null bilinear structure (split/embedding top512 overlap `.927/.836`, real-null gap `.078`) but misses the
   absolute p256/p512 energy bar (`.657/.826` versus `.75/.90`).  Structure is mapped; compression is not licensed.

## Fully gated fidelity dial

| Tier | Semantic scalars | Stored bytes | Census CE added | Certificates | Signed cosine |
|---|---:|---:|---:|---:|---:|
| Source-aware BF16 anchor | 545,902,902 | 1,091,805,804 | .000009 | 62/62 | 1.000002 |
| QK72 + MLP0/4 p768 | 516,264,246 | 1,032,528,492 | .009301 | 50/62 | .990600 |
| QK64 + MLP0/4 p768 | 511,758,646 | 1,023,517,292 | .012329 | 43/62 | .986524 |

All compute is fp32 after explicit dequantization.  This table makes no latency or activation-memory claim.

## Confound audit

- **Baseline subtraction / frame mixing:** census and OOD remain compiled-minus-original-native on identical
  positions.  Signed effects are KO-minus-unablated within each model, then compared as signed vectors.  The
  original-native KO is measured before global rounding.
- **Nonlinear composition:** no adopted joint point relies on adding solo CE.  The pair, third-cut miss, mid-tier,
  precision rebuilds, certificates, and signed effects are all physical compositions.  Tail/tax predictions select
  candidates; they do not replace adoption gates.
- **Shared token difficulty / leakage:** MLP selection uses fit-A only and constructs from fit-B; census, fresh
  windows, and monotonically nonoverlapping WT103 spans are untouched by those fits.  Repeated WT103-train segments
  are shifted-population evidence, not independent-domain evidence; that limitation remains explicit.
- **Post-selection:** layers4/0 were chosen by a rule frozen before construction.  Layer2, QK72, variable-rank DP,
  value ranks, and Tucker bars were all registered before their receipts.  Failed bars were not changed.
- **Dead knobs and receipt labels:** rung372/373 inherited stale rung/status/prose labels; rung372's identity clause
  failed honestly.  The generic harness was parameterized and rung377/378 receipts carry corrected physical labels.
  One stale null-key name remains cosmetic and is not used as a physical tripwire.
- **Precision/noise floor:** six independent BF16/FP16 comparisons change census at roughly `1e-4` or less and
  reproduce certificate counts.  Precision and semantic structure are separate price axes at this resolution.
- **R2 as structure:** every structural route has planted and negative controls.  The real Tucker screen is not
  promoted despite stable subspaces because its absolute retained-energy bar failed.  The second certificate mode
  is not called MLP-specific because the value control missed.
- **Gauge / directionality:** raw bilinear units are never scored as identified.  Blocks use invariant contraction
  objects; a DAG remains unidentifiable from a symmetric quadratic map without ordered interventions.
- **Router state:** token/context routers lose to equal-price shared subspaces, and morphology fails prospective
  support.  No combinatorial top-k support table receives tensor-network price credit.

## Is the current path still highest-information?

No further local Q/K, MLP-input, value-rank, router, or identical-sharing experiment is high-information: each is
now measured or computationally closed.  The output-Tucker screen shows that the next promising route must fit a
new joint core rather than project one native mode.  The active object should therefore change.

## Ranked next moves and kill evidence

1. **Joint context–Sobolev Tucker/CP core.** Fit new input/output/product factors under values plus exact bilinear
   JVPs; compare at literal price with input-only RRR, output-only PCA, and their independent composition.  Kill if
   it does not improve every equal-or-cheaper control by20% or derivative subspaces are split-unstable.
2. **Vector-valued intervention-aware RRR.** Preserve a suffix-Jacobian/certificate response subspace rather than
   another scalar consequence weight.  Kill if held-out signed residual does not shrink by25% without CE loss.
3. **Graph-smooth vocabulary residual.** Test whether the distributed rare-token error is smooth on the stored
   embedding geometry.  Kill if unseen-tail repair is<35%, a random graph matches it, or edge storage erases savings.
4. **Asymmetric conditioned contraction algebra.** Only a behavior with independently supported states and ordered
   interventions may reopen hierarchy/DAG structure.  Kill if recovered flags match shuffled-label stability.
5. **Runtime/lower-bit realization.** Measure fused execution, load, and peak memory separately; no semantic-scalar
   credit, and no byte claim without pricing scales/metadata.

The exact formulas, controls, and prospective prices for these directions are in
`NEXT_REPRESENTATION_DIRECTIONS_2026-09-01.md`.  There is no currently licensed unattended GPU follow-up: rung381's
absolute bar failed, and the managed queue is intentionally empty rather than being filled with a tuned p768 arm.
