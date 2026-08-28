# Hourly strategic review — 2026-08-28 19:20 UTC

## UPDATE: what changed in this hour

The main new scientific result is that the sharp composition failure is not uniquely
an attention-1 problem or uniquely an MLP-1 problem. Starting from the healthy
compiled block-0 arm, adding either one of the two layer-1 substitutions produces
almost the entire loss:

| sequential arm | fraction of the live-vs-compiled CE gap recovered |
|---|---:|
| compiled block 0 only (`B0`) | 64.79% |
| `B0` plus compiled attention 1 | 27.40% |
| `B0` plus compiled MLP 1 | 26.06% |
| compiled blocks 0 and 1 (`B1`) | 25.88% |

The losses from adding attention 1, MLP 1, or both are respectively 37.38, 38.73,
and 38.91 percentage points. The sum of the two single-site losses divided by the
joint loss is 1.956; exact redundant explanations would give 2.0, while independent
additive damage would give about 1.0. Thus the two substitutions are nearly redundant
ways to break one layer-1 interface.

There is also a useful difference between their failure mechanisms. Scalar gain
correction rescues the attention-only arm by 23.6 points, but makes the MLP-only arm
3.0 points worse. The same final CE floor can therefore hide different internal
errors. This is why CE alone is a necessary functional score but is not a sufficient
diagnosis.

The MLP0 sparse weight-action discriminator also closed. A signed normalized TopK
dictionary attains held-out action $R^2=0.7386$ and recovers 98.28% of standalone
MLP0 CE. Adding input noise changes neither result meaningfully, and an oracle IHT
encoder adds less than 0.01 $R^2$. The learned subspace is fairly stable across seeds
(rank-64 overlap 0.834), while individual atoms are not (cosine 0.516). This is useful
compression, but not yet a canonical semantic decomposition and not an interface
certificate.

## How much of the model is actually explained

These ledgers answer different questions and must not be averaged:

- **Structural coverage:** 36/36 tensor-network surrogate sites have an executable
  replacement. This means every site has code, not that every computation is
  understood.
- **Certified whole-program simplification:** 5.3481% of storage is proven removable
  under the current whole-program certificate.
- **Older human-readable behavioral account:** 32.1% ± 6.4% on its registered assay.
- **Strict named causal CE account:** 10.923% of the teacher-to-ablated CE gap, leaving
  4.72714 nats unexplained.
- **Current early-MLP compiler gap:** no newly claimed recovery. The final scientific
  role remains unopened and the 68-action ledger remains 0/68.

So the honest answer is: the network is structurally instrumented, but only a small
minority of its behavior is causally named and certified. The strongest recent result
is localization of a failure, not an increase in explained fraction.

## Largest remaining gaps

1. **Layer-1 interface behavior.** We can reproduce attention 1 and MLP 1 locally,
   yet either substitution breaks the composed computation. We do not yet know the
   finite response map that the live layer preserves and the separate fits lose.
2. **Missing observed response backend.** The 22 LL/LT/null causal response arms have
   a fixed schedule, but no sealed model transaction yet executes and binds all 144
   teacher and 3,168 student forwards.
3. **Attention coverage of the causal assay.** The current MLP0-to-MLP1 transport test
   keeps attention 1 live. It can diagnose MLP transport, but cannot alone explain why
   substituting attention 1 produces the same collapse.
4. **Whole-model composition.** High standalone CE recovery, local low rank, and
   per-site reconstruction have repeatedly failed to predict joint deployment.
5. **OOD and editability.** We still lack a promoted program whose extraction,
   selective removal, collateral effects, and OOD transport can be tested honestly.

There is no current data, FineWeb, `rspd`, cache, or GPU-availability blocker. The GPU
is running the preregistered single-site depth sweep. The blockers are scientific
interface identification and the missing sealed execution backend.

## Candidate actions considered and pruned

The following were pruned for now: more SAE optimization, MOD/K-SVD, input-noise
variants, longer local fits, norm minimization before HOSVD, standalone dense CP or
Tucker factorization, and more local moment/routing summaries. They are cheap or
mathematically attractive, but current evidence says they mainly optimize local
reconstruction and are redundant with completed negative discriminators. Scalar
bias/gain remains a nearly free correction when it helps, but the MLP-only failure
shows it cannot be the general explanation.

## Top five priorities

1. **Seal and execute the paired finite-response backend.** Highest causal relevance
   and information gain: it directly tests whether LL, explicit transport LT, or a
   null map preserves the exact teacher's edited response. It is falsifiable and
   composable, although GPU-expensive (3,312 response forwards).
2. **Finish the running single-site depth sweep.** This cheaply distinguishes a real
   layer-1 boundary from the more damaging possibility that adding *any* one compiled
   upper site collapses the arm. It can invalidate the present localization story.
3. **Build a layer-1 joint interaction assay that includes attention.** Measure the
   four live/compiled attention-1 × live/compiled MLP-1 cells under identical finite
   MLP0 edits and decompose main effects from interaction. This addresses what the
   MLP-only transport route cannot.
4. **Complete the 68 observational actions and compensation checks.** Once response
   provenance closes, measure CE, teacher KL, nine token-frequency bins, 18 consumer
   norms, and both deployed/exact MLP2 backgrounds. This tests whether MLP2 is masking
   upstream damage and whether independent simplifications compose.
5. **Only after causal success, fit the joint sparse DAG inside the stable 64D
   subspace.** Price shared dictionary atoms, sparse coefficients, affine corrections,
   and executable calls; then validate extraction, selective removal, OOD transport,
   and prequential/MDL cost. This turns “simplicity” into downstream capability rather
   than an aesthetic parameter count.

## Action executed in this review

Added a pure paired-response reducer and adversarial tests. It now literally freezes
the previously ambiguous response support to every scored position 64--255:

- MLP1 code response shape `[4,192,64]`;
- centered-logit response shape `[4,192,vocabulary]`;
- output-KL response on the same 192 positions;
- positive-minus-baseline and negative-minus-baseline are two pooled occurrences,
  not a central difference;
- output KL uses `KL(exact teacher edit || student edit)` divided later by
  `KL(exact teacher edit || exact teacher baseline)`;
- only detached CPU float64 per-row square, dot, and KL sums can leave the reducer.

Twelve focused response-plan/reduction tests pass. This closes the numerical
definition and catches wrong support, centering, KL orientation, graph-bearing
tensors, malformed identities, and shape mismatches. It does **not** yet execute a
model or authorize final evaluation. Red-team review keeps production execution
NO-GO until the existing observed adapter derives physical edits and action identities
from frozen authority, executes all 48 batches atomically, binds reductions to actual
forward receipts, scrubs raw tensors, and makes the final closure require the resulting
run receipt and structured call ledger.
