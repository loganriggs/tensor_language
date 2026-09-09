# One-day research update

**Window covered:** 2026-09-07 23:45 UTC through 2026-09-08 23:45 UTC
**Model:** `bilin18`, an 18-layer transformer with residual width 1,152, nine attention heads per
layer, head width 128, and bilinear MLPs with 4,608 hidden factors
**Overall goal:** identify a smaller transparent tensor program that predicts unseen behavior,
composes across tasks, supports selective causal edits, and is literally simpler to store and run

## High-level explanation

Over the last day, the work moved from “there is a useful low-dimensional direction somewhere in
the model” to a substantially more physical account of an `is`-versus-`was` computation.

The strongest current single-task circuit is a four-attention-head program:

```text
L8H1 + L9H1 + L9H4 + L11H3
```

On a token- and endpoint-aligned fresh corpus, swapping these four heads moves the model about 73%
of the way from the base answer toward the donor answer, with almost perfect directional agreement.
Removing the same heads destroys about 72% of the behavior, so the result is both substantially
sufficient and necessary. The four heads are not interchangeable copies: they enter at different
layers and are progressively transformed before arriving at block 11. MLP11 reads a small part of
their effect, but most of the behavior appears to travel through the block-11 residual stream.

A second strong result is the cross-task four-head `H4` interface
`L9H1 + L9H4 + L11H3 + L15H5`. It carries both a temporal command and the `is/was` command on the
same sequence, composes almost additively, transfers to new templates and lexicons, and supports
selective midpoint removal. That result is broader composition evidence; the aligned v23 circuit
above is currently the cleaner object for detailed writer, transport, necessity, and weight-tensor
anatomy.

The day also produced an important negative result about constrained DAS. Optimization can find a
strong target direction, but the complement objective is underdetermined enough to memorize a
construction or parity. Noise/KL regularization can help a particular split, but no fixed setting
yet transfers reliably enough to beat the simpler physical circuit. The research therefore shifted
from “optimize a better subspace” to “identify native causal writers and readers, then restrict the
checkpoint weights to their proven interface.”

That weight-restriction proposal is now implemented. We can construct exact writer, reader,
attention, and bilinear-MLP tensors inside a causally identified subspace. Dataset PCA, SAE, or
hierarchical SAE is kept as a separate description of which legal weight directions are actually
visited. The first four-head weight result suggests one shared residual-context direction plus
head-private adapters and tails, rather than one identical map used by every head. The latest
leave-one-head-out test is being rerun in float64 after its first receipt correctly failed a strict
numerical certificate; its visible scientific values are not being counted yet.

## 1. What counts as a circuit here

The target is not merely a low-rank activation or a set of high-scoring heads. A mature circuit
should eventually satisfy these requirements:

1. State what information is read, what operation is performed, what is written, and who reads it.
2. Group pieces across native modules when they implement one variable, and split modules when only
   some heads/factors participate.
3. Predict effects on held-out lexicons, templates, directions, and construction families.
4. Be sufficient as an executable interface plus an explicitly described background.
5. Be necessary or selectively manipulable under removal, swap, reset, rescue, and dose tests.
6. Compose predictably with other circuits and reuse genuinely shared subcomputations.
7. Survive corpus splits, fitting restarts, and legitimate coordinate gauges.
8. Eventually improve literal storage, arithmetic, edge, and state cost.

We use three evidence levels:

- A **screen** nominates a component, direction, or tensor.
- **Identification** requires prospective held-out behavior and appropriate causal controls.
- **Adoption** requires an executable replacement that also improves the priced program frontier.

No result from this day is a whole-model adoption. The best results are identified partial circuits
and executable local interfaces.

## 2. Core terms and measurements

### Base, donor, and interchange

A **base** prompt expresses one member of a controlled contrast; a **donor** prompt expresses the
other. An **interchange intervention** runs the base computation while replacing a declared internal
state with its donor value. For a component subspace with projector `P`, the standard difference
patch is

```text
x_patch = x_base + P (x_donor - x_base).
```

A complete head patch uses the whole selected 128-dimensional head slice. A subspace patch uses
only `P`; the complement patch uses `I-P`.

### Answer margin and recovery

The behavioral scalar is the correct-answer logit minus the foil logit. If `m_b`, `m_d`, and `m_p`
are base, donor, and patched margins, rowwise recovery is

```text
(m_p - m_b) / (m_d - m_b).
```

For a vector response `delta_p` against target `delta_t`, the main scale statistic is

```text
signed_projection = <delta_p, delta_t> / ||delta_t||^2.
```

Cosine measures orientation; relative residual measures unexplained geometry:

```text
cosine = <delta_p,delta_t> / (||delta_p|| ||delta_t||)
relative_residual = ||delta_p-delta_t|| / ||delta_t||.
```

All three are needed. A vector can have the right scale but a large orthogonal error, or excellent
cosine but insufficient magnitude.

### Controls

`A1` and `A2` are the two target construction families. `P` is a difficult answer-preserving
control designed to share nuisance temporal/lexical structure. `C` is an aligned canonical control.
Target effects must be large and directionally correct while P/C leakage stays small.

### Sufficiency, necessity, reset, and rescue

- **Sufficiency:** installing a component or interface reproduces the target effect.
- **Necessity:** removing it erases the corresponding behavior.
- **Reset:** put a proposed reader back into the source-absent state and measure lost effect.
- **Rescue:** restore that reader's source-present output and require the effect to return.

Reset plus same-module rescue is stronger evidence for a directed physical route than correlation,
weight alignment, or Shapley credit alone.

### Shapley and interactions

For the finite component game `f(S)`, the Möbius/Harsanyi coefficient is

```text
mu(S) = sum_{T subset S} (-1)^(|S|-|T|) f(T).
```

Shapley credit divides every interaction dividend equally among its participating components. It is
useful for redundant circuits where deleting one component from the full set may do nothing. Pair
interaction distinguishes diminishing/redundant effects from complementarity, but neither Shapley
nor a pair interaction proves a directed edge; that still requires physical loss/rescue.

### Bilinear MLP tensor

For residual input `x`, Bilin18's MLP computes

```text
l = Left x
r = Right x
h = l odot r
y = Down h.
```

In indices,

```text
y_o = sum_n Down[o,n] (sum_i Left[n,i] x_i) (sum_j Right[n,j] x_j).
```

The exact third-order tensor is

```text
T[o,i,j] = sum_n Down[o,n] Left[n,i] Right[n,j].
```

This makes it possible to translate a causal state intervention into literal checkpoint algebra
instead of using activation variance as a proxy.

## 3. Midnight: constrained DAS became a construction-memorization result

The day began with a rank-one multi-construction DAS test. Native capability was good, but no fitted
initialization passed both A1 and A2 on every held parity. Pooled numbers looked plausible while an
individual held fold fell to `.72091`, one P row flipped, and the sealed v16 transfer fell to
`.64172/.49549`.

Separate A1 and A2 optimized axes were each internally stable, but their cross-construction cosines
were only about `.41-.67`. A bisector could clear some target bars yet inherited the same P failures
and did not transfer to v16 A2. This showed that the optimizer was learning construction-conditioned
geometry, not one invariant causal coordinate.

A static weight audit also caught an overclaim. The DAS intervention replaced all computed
attention-15 outputs with cached donor outputs, so upstream effects could bypass live L15H5 through
the residual stream. Strong `W_Q/W_K/W_V` alignment therefore nominated L15H5 as a possible reader
but did not prove that native attention 15 read the DAS subspace.

The subsequent live-attention test showed that the upstream effect survived with native attention,
but the full attention-15 reset/rescue atlas found that attention 15 carried only `.057-.081` of the
effect. L15H5 was the strongest head but still only a small, unstable side channel. The dominant
route bypassed attention 15 through residual state.

**DAS conclusion for the day:** complement optimization is not rejected in principle, but its
current target is underidentified. Regularization must be selected across complete construction
environments, preserve target feasibility, keep the complement inert on vocabulary/readers, and
survive new physical writer/reader tests. DIM and exact native interventions remain stronger
baselines.

## 4. Early hours: the direct residual route became explicit

Complete attention/MLP write patches over layers 12--17 recovered only about `.13-.21` of the
upstream effect. By contrast, complete residual-state swaps from entry 12 through post-MLP17 all
recovered exactly `1.0`. This identified a carried residual interface but did not by itself identify
a consumer—equal state at a deterministic suffix trivially gives equal output.

A cross-fitted rank-two A1/A2 subspace at entry 12 was sufficient at roughly `.795-.881`. A pooled
rank-one coordinate failed A1, and removing the full nuisance P span destroyed the target too,
showing genuine target/control overlap rather than an easy orthogonal complement.

Several routers then failed honestly:

- a native-state router could not distinguish paired A1/P rows with identical base text;
- a source-delta nearest-centroid router predicted the off branch everywhere;
- a nested ridge router passed within-parity CV and then failed the held parity because the
  transformation direction reversed;
- a sign-invariant Gram router recovered targets but leaked P;
- a token-pair router worked on v15 but had zero signature coverage on sealed v16.

The downstream computation nevertheless became simple. When all later writes are frozen, the final
residual difference is the entry-12 difference multiplied by the product of the six residual gains:

```text
delta x18 = (product over layers 12..17 of lambda_l[0]) delta x12
           = 1.51363146 delta x12.
```

The exact final RMSNorm, tied unembedding, and soft cap then decode it. A rank-two final-state edit
was sufficient and selectively removable on its tested corpus, but its construction-gain law did
not fully transfer. This is an explicit partial residual program, not yet a universal semantic
router.

## 5. Cross-task composition: a strong four-head H4 interface

A common physical gauge was built for the temporal and `is/was` commands. Their final residual
union has rank 12, but the tasks remain typed rather than collapsing into one subspace. Exact weight
contractions sharply reduced the head candidates.

The first three-head physical union `L9H1 + L11H3 + L15H5` was distributed: no singleton explained
both tasks, yet the union recovered about `.71-.75` of the temporal effect and `.49-.52` of the
`is/was` effect. When both commands were installed simultaneously, interaction was tiny and
later-to-earlier causal influence was exactly zero.

Greedy augmentation selected L9H4, producing H4:

```text
L9H1 + L9H4 + L11H3 + L15H5.
```

H4 recovered `.833/.848` temporal and `.696/.669` `is/was` on the original FIT/HOLDOUT split. It
then transferred without reselection to fresh templates and lexicons at `.835/.833` temporal and
`.730/.739` `is/was`, with cosines `.997-.999`, direction `1.0`, and cross-role collateral below
`.00342`. Simultaneous installation remained almost perfectly additive.

A symmetric-midpoint removal edit moved every native-correct endpoint toward a smaller
correctness-aligned margin while preserving low collateral. This gives H4 composition, OOD, and
selective manipulation evidence. It remains a partial-effect program and is not yet a standalone
replacement.

Within H4, L11H3's value branch explained almost all of the complete L11H3 effect. The static
weight-nominated L15H5 Q/Q2 explanation failed causally: q was negative, q2 small, and the pair
mostly cancelled. This repeated the day's central lesson that weight compatibility proposes a
reader but cannot identify one without intervention.

## 6. Exact L11H3 source computation and upstream writers

The task-specific value source was localized to different semantic regions: the temporal command
uses a bridge region, while `is/was` uses a post-cue region. Most Q/K routing remains recipient
native; donor QK changes are only a small temporal modulation.

For destination `i` and source region `R`, the exact extracted L11H3 preprojection term is

```text
(1-lambda_11) sum_{j in R}
    ((q_i dot k_j)/128)
    ((q2_i dot k2_j)/128)
    delta_v_j.
```

This tensor reproduced the independently patched head delta with maximum error `9.54e-6`. Installing
it directly reproduced selected logits within `8.59e-6` and every original/OOD causal effect at
essentially unit recovery. It is one of the day's clearest executable local computations.

Folding this tensor through the physical L11H3 value weights ranked all 111 earlier writers.
L9H1 and L9H4 passed the complete shared singleton causal gate. A weight-ordered greedy test reduced
the larger candidate set to a shared two-head core `L7H7 + L9H4`, plus temporal-specific L9H1.
However, exact source-term mediation recovered about `.78-.80` of the temporal writer effect but
only `.17-.20` of `is/was`. Full-prefix and full-head L11H3 variants did not repair it. The same
native writers therefore participate in two downstream routes; a shared weight edge is not a shared
end-to-end computation.

## 7. P7, exact Shapley games, and physical readers

A cumulative downstream replacement program selected seven modules:

```text
P7 = A11, M11, M12, M15, M13, M16, M10.
```

P7 plus a residual-identity stream composed to unit recovery on OOD. Ordinary leave-one-module-out
deletion was uninformative because later clamped writes could overwrite any single omission. This
triggered the source-conditioned Shapley construction from `shap_tensor.md`.

The exact seven-player game evaluated all 128 subsets twice, with the source absent and present.
After preserving and repairing one incorrect parent-replay comparator, the valid game assigned
about `.45-.47` of the source effect to direct/background carriage and `.53-.55` to P7. Six modules
had stable signed credit; A11 and M11 were the leading pair, with positive operational interaction.

Physical loss/rescue corrected the game-theoretic interpretation:

- removing A11's reader caused `.222/.224` loss;
- removing M11 caused `.110/.114` loss;
- restoring each module's output rescued its loss exactly;
- their physical joint removal was mildly redundant, not positively complementary.

Within A11, H3 alone recovered about `90.7%/90.3%` of the complete A11 reader loss. A shared
eight-dimensional state `U8` retained about `94.25%` of unseen reader-delta energy and recovered
A11/M11 at `.993/.923` on held-out data. Folding U8 into the checkpoint produced exact writer and
reader maps plus an exact `8 x 8 x 8` M11 bilinear tensor; a frozen top-32 native-factor program
recovered `.809` on HOLDOUT.

Fresh v18 transfer then revealed the distinction between coordinate stability and physical endpoint
stability. The A11 H3 endpoint transferred strongly (`.887/.920`), while some U8 coordinate and M11
factor magnitudes changed by construction. Raw output-energy factor selection failed badly. Pulling
the actual downstream task gradient back through the M11 factors worked much better: a top-16
task-functional subset transferred directionally, but its magnitude required calibration and later
failed P selectivity on sealed v20.

## 8. M11 writer/reader state and the path to v23

The exact M11 task-functional tensor was separated into factor writer state and reader state:

```text
H_n(x) = (Left_n x)(Right_n x)
R_n    = <downstream task gradient, Down[:,n]>
Q      = sum_n H_n R_n.
```

The reader `R` was stable across reporter splits; the writer `H` required six coupled
construction-by-direction states. A scalar-gain model failed. This showed why raw hidden-factor
energy and causal usefulness disagree: the important metric is the writer change after contraction
with the registered downstream reader.

An exhaustive v21 atlas over complete inputs/modules and all 108 heads rediscovered the same four
behavioral heads that had appeared on v15:

```text
L8H1, L9H1, L9H4, L11H3.
```

The v21 atlas itself was not publishable as a circuit verdict. It simultaneously required complete
donor reconstruction and quiet unrelated-donor controls, used an absolute raw-H tolerance in a
gauge-sensitive factor space, and ranked raw H rather than reader-contracted Q. Those measurements
were retained as discovery evidence, and the instrument was repaired prospectively rather than
reinterpreted.

V22 aligned target and P endpoints but its canonical C endpoints were still incompatible. V23 fixed
all 64 base/donor lengths and semantic endpoints before model access. Its capability gate retained
30 jointly correct target rows, and the frozen four-head program passed every registered causal
predicate.

## 9. The aligned v23 four-head result

The v23 union has:

- behavior signed recovery `.72588`, cosine `.99524`, direction `1.0`;
- M11-reader-contracted Q recovery `.67007`, cosine `.97676`, direction `1.0`;
- P leakage `.10033` behavior / `.08344` Q;
- aligned-C leakage `.00396` behavior / `.00640` Q;
- exact self-patch and donor hidden/logit closures;
- all four singletons passing their frozen screens;
- positive Shapley allocation for every head, stable across reporter halves.

The behavior Shapley allocation is approximately:

```text
L9H4  0.2267
L9H1  0.2174
L8H1  0.1484
L11H3 0.1335
```

The Q game gives more weight to L9H4 (`.2632`) and distributes the rest around `.13-.14`. Pair
interactions are small relative to the union. This is a distributed writer program, not a single
head and not a rank selected from activation variance.

Reverse removal gives `.72151` behavior recovery with cosine `.99401`, direction `1.0`, P leakage
`.11861`, C leakage `.00416`, and nearly identical reporter halves. This establishes material
necessity.

Exact layerwise transport distinguishes the heads:

- L9H4 is already substantially aligned after block 9.
- L9H1 needs block-10 rotation and block-11 attention gain.
- L8H1 is progressively oriented through blocks 9 and 10 and still needs block-11 attention.
- Adjacent L11H3 must pass through live RMS normalization. The exact finite response closes to
  `9.3e-7`; the RMS tangent has cosine `.999984`, signed projection `.992374`, and residual `.00945`.

Restoring the entire induced M11 factor change rescues only `.15114` of the removed-head behavior.
The leading occupied M11 mode explains `.87376` of that *small local M11 rescue*, but removing the
mode directly explains only `.18197` of the full four-head effect. The correct conclusion is a
necessary four-head circuit with a minor M11 branch, not an M11-dominated mechanism.

## 10. Breadth campaign and systems trajectory

Alongside the deep circuit, a standardized four-row battery continued testing many linguistic
contrasts. During this window the working ledger grew from the mid-70s to 107 provisional distinct
circuit groups under its current singleton-counting rules. An alternate receipt recount produced
105 because it applied different singleton rules; the discrepancy was explained rather than
silently averaged. These are mostly screens, not 107 finished transparent programs, and the live
v289 separability sweep can still fuse or reject provisional groups.

The breadth work found that many apparent circuits fuse when a larger same-family control set is
used. Cue/readout overlap sometimes explains that fusion, but several preregistered cue theories
were refuted. Exact source-conditioned Shapley tests on three non-preposition behaviors found
`89--100%` direct/background carriage, warning that greedy component sets are often not mediation
chains.

The main systems bottleneck is now separability auditing. Measured per-member cost grows roughly as

```text
seconds/member = 19 + 9.3 * family_size,
```

so fully rechecking every member makes a family sweep quadratic in family size. At family size 44,
one v289-style audit is about 5.2 hours. A proposed efficiency repair would always check new and
weakest members, rotate through the remainder, and retain periodic full sweeps. It has not been
silently applied to the already registered v289 run.

At the close of the window, v289 is the verified live managed GPU job. The queue behind it contains
a Shapley effect game, the seven-forward v23 residual/M11 factorial, the two-forward v24 native
capability gate, and another breadth job. The v23 downstream reader atlas and v24 causal confirmation
are implemented but fail closed until their predecessor hashes pass. No direct competing GPU process
has been launched.

## 11. Important failures and corrections preserved during the day

The research quality improved partly because failed instruments were not converted into favorable
scientific results:

- V16's first capability count inherited a 32-row threshold for a 16-row panel; a separate
  zero-model audit repaired the count without opening causal outcomes.
- DAS's apparent pooled success hid failed construction/parity cells.
- L15H5 weight alignment was initially overinterpreted despite an attention-output clamp that
  bypassed the purported live reader.
- A normalized-reader atlas omitted four reference forwards from its declared price; only the
  accounting was repaired.
- A source-region localization compared two metrics in different frames; the original remained
  invalid and a separate like-for-like audit scoped what could be retained.
- The first P7 source-conditioned game used the wrong all-live parent comparator; its unchanged
  intervention table was rerun under a new version.
- The v21 complete atlas had contradictory closure/selectivity requirements and a gauge-sensitive
  raw-H endpoint; it remained discovery-only.
- V22's canonical controls had unequal endpoints; v23 repaired the corpus rather than filtering or
  relaxing C.
- The necessity result's automatic terminal overstated M11 mediation. A separate audit corrected
  the interpretation using its own measured rescue values.
- The first rank-one shared-context leave-one-out receipt failed its absolute SVD certificate in
  float32 (`.046875` versus `2e-5`). Its attractive scientific values are quarantined while the
  identical test is repaired in float64.

## 12. Current status and next decisions

The strongest current facts are:

1. H4 is a composable, OOD-stable, selectively removable cross-task head interface.
2. The aligned v23 four-head `is/was` program is sufficient, necessary, selective, split-stable,
   and physically differentiated across layers.
3. M11 is a real but minor branch of the v23 effect; the dominant downstream path remains to be
   localized.
4. Exact source tensors and restricted checkpoint operators now connect causal states to real
   weights.
5. Constrained DAS has not yet produced a construction-invariant, selective solution superior to
   the physical circuit.

The next decisions are already frozen:

1. The exact residual-versus-M11 block-11 factorial must close its joint restoration and determine
   the dominant separately rescuing branch.
2. If the residual branch passes, the reciprocal downstream atlas will test every complete
   A12--M17 module by both transfer sufficiency and reset necessity. A passing attention module is
   then split into nine heads.
3. V24's completely new constructions must pass native capability before the fixed four-head circuit
   is tested on all 32 target rows without correctness filtering.
4. Only after a downstream reader is fixed will the shared weight core receive causal core/tail
   swaps and cross-task composition tests.
5. Standalone extraction and literal price comparison come after downstream fidelity, OOD, and
   composition—not before.

## 13. What came of the SAE/subspace-decomposition and fold-it-into-the-weights ideas

This idea has produced concrete code and results, but the three possible meanings are now kept
strictly separate.

### A. Dataset decomposition: what states actually occur

PCA, SAE, hierarchical SAE, clustering, or another activation decomposition can describe the
empirical support of a fixed interface:

- which coordinates are frequently occupied;
- whether activity is sparse or multimodal;
- whether there are discrete construction/task states;
- which combinations never occur on the sampled corpus.

This is valuable for interpretation and for constructing test cases. It is still dataset based: an
unseen but legal checkpoint computation can be absent, while a frequent activation can be causally
irrelevant. Therefore activation decomposition is now called a **reachability/occupancy overlay**,
not the definition of the circuit.

### B. Folding a causal subspace into weights: what computation is possible

For an orthonormal residual interface `U`, the implementation now forms exact restricted operators:

```text
writer: U_out^T W_write
reader: W_read U_in
MLP core: T[a,b,c] on output/input/input interface coordinates
QK core: U^T W_Q^T W_K U
OV core: U_out^T W_O W_V U_in.
```

These arrays describe every computation the corresponding checkpoint weights can perform inside the
admitted interface, independently of a dataset. The writer and reader sides are initially kept
separate; mixing them in one norm would hide a choice of units and relative weighting.

### C. Decomposing the restricted weight tensor

For four component maps `M_h` sharing the physical residual axis, the exact fixed-rank problem is

```text
min over U^T U=I  sum_h w_h ||M_h - U U^T M_h||_F^2.
```

It is solved globally by taking the top left singular vectors of

```text
X = [sqrt(w_1)M_1 | ... | sqrt(w_H)M_H].
```

The implementation returns the common physical basis, each component's private adapter `U^T M_h`,
the common reconstruction, private tail, singular spectrum, boundary gap, and the exact discarded-
singular-value error certificate. It supports attention and MLP readers with unequal private widths.
Component weights are explicit, preventing a wider reader from silently dominating the fit.

It also now reports per-component fit and performs leave-one-component-out transfer. Each fold learns
the common projector from all other heads/tasks/readers and scores the omitted map. This is important:
a pooled optimum can otherwise look shared while sacrificing one circuit.

### What the current four-head tensors say

The exact M11-restricted tensor has broad checkpoint capability but narrow observed use. Weight
reader-mode energy is spread roughly `.362/.215/.213/.210`, whereas `.97137` of realized target
response lies in one frozen occupied mode, stable across reporter halves at cosine `.99998`.

After contracting that mode, the four complete maps are not identical: median literal-map cosine is
only `.0486`, and the leading head-mode energy is `.2972`. The naive private-coordinate comparison
was then red-teamed because each head can rotate its 128-dimensional coordinate gauge. Invariant
context-Gram similarity is high (`.92455`) and the leading context directions overlap around `.96`,
but full-map Procrustes similarity is `.68596` and higher-dimensional context overlaps are much
lower. The current hypothesis is therefore:

```text
one shared low-dimensional residual-context core
    + one private adapter/tail per head,
```

not “all four heads write the same map.”

The first preregistered rank-one leave-one-head-out screen generated large-looking transfer values,
but its absolute float32 SVD certificate failed. Under the project rules, pred A failure makes the
whole scientific receipt invalid. The result and audit are preserved; the same rank, weights,
seeds, random controls, and bars are being rerun with float64 arithmetic. Until that repair passes,
the rank-one transfer values are not evidence.

### What happens next

The decomposition becomes circuit evidence only through opposing interventions:

1. freeze rank and component weights prospectively;
2. install/remove the common maps `UU^T M_h`;
3. install/remove the private tails `(I-UU^T)M_h`;
4. require common plus tail to reconstruct the exact parent response;
5. compare with matched-energy private or rotated controls;
6. require held-out/OOD behavior, reader response, P/C selectivity, and cross-task composition.

If the common core transfers and is selectively necessary, it is a reused circuit variable. If a
matched rotation or private tail works equally well, the shared-looking weight geometry is not the
causal decomposition. After that causal decision, PCA/SAE/hierarchical SAE can still be applied to
the fixed coordinates to explain which legal branches occur and whether their occupancy is sparse,
discrete, or hierarchical.
