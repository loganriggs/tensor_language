# Strategic continuation — 2026-08-28 20:12 UTC

## UPDATE PART

## Bottom line

No honest whole-model “explained percentage” increased in this interval. What did
improve is the experimental map of where the current compiler fails and the machinery
needed to test a proposed explanation causally.

The most consequential new model result is that **MLP5 is an exceptionally expensive
interface to replace**. Starting from B0, replacing MLP5 alone removes 61.2 percentage
points of normalized recovery: about 95% of the advantage B0 had over the fully
compiled model. This does not yet tell us MLP5's semantic algorithm. It tells us that
the present componentwise compiler crosses a crucial interface incorrectly there.

At the same time, a very simple law for pricing replacements has failed. If

\[
C(S)=\text{performance lost after replacing the sites in }S,
\]

then the proposed rule

\[
C(S)=\alpha\sum_{s\in S} C(\{s\})
\]

obtained \(R^2=-1.284\). Here \(R^2=1\) would be perfect prediction, \(R^2=0\)
would merely equal predicting one constant mean, and a negative value is worse than
that constant. Therefore site count, independent site costs, and independent costs
times one global correction are all invalid measures of whole-program difficulty.
Which sites are combined matters.

## What fraction is actually understood?

There is no single scientifically defensible percentage because the existing ledgers
measure different claims.

| Claim | Current evidence | What it does **not** mean |
|---|---:|---|
| Modules with some tested structural surrogate | 36/36 | semantic or causal understanding |
| Certified whole-program storage removed | 5.3481% | 5.35% of behavior explained |
| Older behavior assigned human-readable labels | 32.1% ± 6.4% | a complete executable replacement |
| Strict named causal CE headroom recovered | 10.923% | the remaining model is understood |
| Strict named causal CE still unexplained | 4.72714 nats | local activation error |
| New 68-action coupled experiment completed | 0/68 | the backend work is a scientific pass |

The most honest summary is: every component has been touched structurally; only a
small amount of compression is certified at whole-model level; roughly one tenth of
the older causal CE target is named; and the current early-MLP compiler has not yet
earned any new causal credit.

## New evidence

### 1. Interactions, not contiguity, determine replacement cost

Three programs replaced the same number of deep sites. A contiguous suffix retained
40.6% normalized recovery; two scattered choices retained 44.0% and 40.1%. Packing
replacements next to one another changes recovery by only 1.5 points against the
registered 10-point threshold. Contiguity is not the missing variable.

The complete 34-site table then showed two qualitatively different regimes:

- at layer 1, replacing attention or MLP alone destroys almost the same behavior as
  replacing both, so the two routes are strongly redundant;
- deep replacements are cheap alone but damaging together;
- MLP5 is a special bottleneck: replacing it alone is much worse than replacing the
  attention site at layer 5.

Thus “simplicity” cannot be only parameter count, local rank, or a sum of local
distortions. A useful small program must also have a small **interaction state**: a
compact variable that predicts how an early replacement changes the cost of later
replacements.

### 2. MLP5 is not a small outlier-coordinate circuit

The GPU job kept only selected residual-output coordinates of native MLP5 while using
the compiled version for all other coordinates. The repaired run completed on all
three roles and wrote its result artifact. On the primary role:

| Arm | Absolute top-1 | Normalized gap recovery |
|---|---:|---:|
| B0, native MLP5 | 30.25% | 64.8% |
| K0, compiled MLP5 | 14.47% | 3.6% |
| keep 16 largest native/compiled discrepancy coordinates | 14.50% | 3.7% |
| keep 256 largest native/compiled discrepancy coordinates | 19.29% | 22.3% |
| keep 256 random coordinates | 15.73% | 8.5% |
| keep 256 largest native-output coordinates | 14.10% | 2.1% |

Relative to the B0-to-K0 loss, the 256-coordinate discrepancy choice recovers 30.5%,
versus 8.0% for random coordinates. But 16 discrepancy coordinates recover only 0.2%
of the loss, and at widths 4, 16, and 64 the discrepancy choice does not materially
beat random. Merely keeping the largest-output coordinates fails at every useful
width. The top 16 discrepancy and top 16 output-magnitude sets have zero overlap.

All three registered positive claims fail: the loss is not concentrated in 16
coordinates, the high-output coordinates are not the mechanism, and the two
principled selectors do not consistently beat random. The correct conclusion is not
that MLP5 lacks structure. It is that its present error is distributed or lies in an
input-dependent/non-coordinate subspace. A fixed tiny set of residual axes is pruned.
The qualitative result replicates across the three evaluation roles: DISC16 obtains
3.7%, 2.3%, and 4.4% normalized recovery, while DISC256 obtains 22.3%, 23.8%, and
22.7%. The artifact does not contain a document bootstrap, so these replications and
the very large miss against the 34.2% registered DISC16 threshold—not a formal
confidence interval—support the negative claim.

A completed scaling control rules out the simplest rescue. On the top-16
output-magnitude coordinates, native MLP5 in the B0 stream is 0.899 times—not at
least 2 times—the fully live magnitude. Matching each preserved channel's magnitude
to the compiled row does not help: scaled DISC64 recovers -3.6% of the MLP5 stake and
scaled DISC256 recovers 12.4%, compared with 8.0% and 30.5% unscaled. Thus neither a
small fixed coordinate set nor a missing scalar/per-coordinate gain explains the
MLP5 interface failure.

### 3. The causal response backend is close to a real final measurement

For each four-document batch, the adapter can now run one closed transaction with:

- three shared exact-teacher forwards: baseline, positive edit, negative edit;
- the same three forwards for each of 22 candidate or null programs;
- 69 total forwards per batch, with every response tied to its physical edit and
  action receipt.

It can accumulate all 48 batches, enforcing exactly 144 teacher and 3,168 student
response forwards, and turn the completed run into the final response statistics. The
terminal result now requires the typed completed-run receipt. This closes the earlier
possibility of presenting caller-labelled arrays as if they came from LL, LT, or a
particular null program.

The boundary now also performs a typed join: all 22 response-bearing actions in the
68-action observational bundle must match the completed response run arm by arm and
modality by modality, with common program, support, ordered intervention units, and
receipt identity. A substituted null arm or changed response statistic fails closed.
The complete suffix/observed CPU suite passes 297 tests.

What remains is the production role owner that executes and aggregates 48 batches for
each of all 68 observational actions, emits the nine frequency-bin and 18
consumer-norm reductions plus gauge/SVD/difference-in-differences closure diagnostics,
computes the registered gates, and invokes the existing join once. The 68 actions are
34 early-MLP programs under two MLP2 choices; they are evaluations, not 68
already-discovered circuits.

## What the mathematical review contributed

The useful new move is a **cut-rank test**. Choose eight early replacement masks
\(P_i\), eight late masks \(S_j\), and measure the 8-by-8 cost table

\[
H_{ij}=C(P_i\cup S_j).
\]

Remove pure early and pure late effects:

\[
\Delta_{ij}=H_{ij}-H_{i0}-H_{0j}+H_{00}.
\]

If early/late interactions communicate through only one or two scalar latent
channels across the layer-5 boundary, then \(\Delta\) should be well predicted by a
rank-1 or rank-2 matrix. This is useful beyond reconstruction: fitting some cells must
predict untouched combinations of early and late replacements. It is therefore a
direct test of whether a smaller composable state exists.

A global tensor-train fit was deliberately rejected for now. A length-17,
four-choice-per-layer tensor train has approximately

\[
8R+44R^2
\]

gauge-adjusted parameters: 52 at rank 1 and 192 at rank 2. The current mask evidence
cannot identify a global rank-2 model. More tokens reduce noise in each measured mask
but do not create the missing independent masks. The frozen 8-by-8 cut assay is the
cheapest decisive falsifier before spending on a global tensor program.

The exact registry, untouched split, baselines, CE secondary target, and useful-pass
thresholds are now preregistered in
`COMPILATION_MASK_CUT_RANK_V1_PREREGISTRATION.md`. Its executable registry tests pass.
This is experimental progress, not a positive cut-rank result.

## Largest remaining gaps

1. **Causal equivalence is unmeasured.** We do not yet know whether the compact
   MLP0/MLP1 program responds to edits like the native network.
2. **MLP5 semantics are unnamed.** We know it is a high-cost interface, not what
   information it computes or why downstream layers require it.
3. **Composition remains unexplained.** Cheap single replacements can combine into a
   large failure, and no scalar price predicts that interaction.
4. **OOD and selective editing remain unearned.** Local reconstruction and in-domain
   top-1 agreement do not establish extraction, removal without collateral damage, or
   transport to new distributions.
5. **The 64-dimensional early code is compact but mostly not semantic.** Its causal
   sufficiency is still more important than attaching unstable English labels to
   arbitrary basis coordinates.

## Ranked next five actions

1. **Finish the source-closed 68-action production callback and execute the final
   paired observational/response role.** This has the highest causal relevance and
   whole-model composability. It directly distinguishes local reconstruction from
   preserved edits, logits, CE, MLP2 compensation, and null controls. The blocker is
   remaining implementation and audit, not data, FineWeb, `rspd`, cache, or GPU
   access.
2. **Execute the preregistered layer-5 8-by-8 cut-rank assay.** This asks whether a
   low-dimensional interaction state predicts unmeasured compositions. It is more
   informative than another local PCA/SAE reconstruction and is cleanly falsifiable.
3. **Test MLP5 in learned, input-dependent subspaces rather than coordinate masks.**
   The fixed 16-coordinate hypothesis is dead, but a low-rank map, tensor factor, or
   downstream-weighted subspace could still carry the relevant interaction. Selection
   must be learned without heldout behavior and compared to matched random subspaces.
4. **Conditionally fit MLP0/MLP1/MLP2 jointly against response and CE.** Independent
   decompositions should first be tested for composition; if they fail, a shared
   dictionary or low-rank transport state should be optimized jointly. Local MSE is a
   diagnostic, not the selection objective.
5. **Validate any survivor on doubled data, held-out documents, OOD transport,
   extraction, and selective removal with collateral measurements.** Then compare
   executable parameter/operation cost and prequential description length only among
   behaviorally qualified candidates.

## Ideas pruned or deferred

- Additive site prices, one global interaction multiplier, and contiguity have been
  empirically falsified.
- A global tensor train is underidentified at present.
- A fixed small set of MLP5 output coordinates is falsified by the completed
  concentration assay.
- Scalar or per-coordinate magnitude matching does not rescue that MLP5 coordinate
  hypothesis.
- Another local SAE, HOSVD, or PCA is deferred unless it predicts downstream response
  or enters the joint composition assay; local reconstruction alone has repeatedly
  failed to compose.
- A semantic label for every coordinate is not required. Gauge-equivalent rotations
  are free, so semantics should attach to invariant subspaces or editable functions,
  not arbitrary basis axes.

## Action executed in this review

The layer-5 cut-rank experiment was converted from a mathematical suggestion into a
prospective executable protocol and committed as `a5447493`. The registry fixes all
64 masks, a connected 28-cell training split, 10 validation cells, 11 untouched
held-out cells, additive and nonlinear baselines, document bootstrap requirements,
top-1 and CE targets, and strict predictive gates. Its focused test suite passes
3/3. No held-out outcome has been inspected.

While this review remained active, the unchanged repaired MLP5 channel assay also
closed successfully and was committed as `923d0933`. Its three positive predictions
failed and its control passed, so the fixed-coordinate concentration branch is now
pruned rather than left as provisional evidence.

The response/observation join landed in `d8294813`, with adversarial arm-substitution
and identity tests. The MLP5 magnitude-matching follow-up then completed with all
controls passing and all rescue claims failing. Neither execution changes an
explained-fraction ledger; both narrow the next experiment.

## UPDATE END
