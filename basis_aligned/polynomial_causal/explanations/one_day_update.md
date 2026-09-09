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
upstream effect. A separate atlas then made **thirteen independent full residual-state swaps**: at
entry 12 and after each attention and MLP sublayer through post-MLP17. Every boundary recovered
exactly `1.0`. This is not evidence that every layer separately mediates the effect. In particular,
the post-MLP17 swap is a final-state positive control, and any complete donor-state swap followed by
the same deterministic suffix is expected to reproduce the donor output. The informative part is
limited: already at entry 12, replacing the complete residual tensor through the semantic prefix was
sufficient even while `x0` and recurrent attention state `v1` remained native. That establishes a
residual-state interface at entry 12, but neither localizes a downstream consumer nor shows that the
intervening layers actively read the state.

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

This 5.2-hour estimate is **not a Shapley computation and is not exponential in 44 players**. In the
v289 protocol, every candidate member is separately optimized against nearly every sibling as a
control and then evaluated on those siblings. The measured approximation is

```text
total seconds ~= family_size * (19 + 9.3 * family_size),
```

which is quadratic and gives about `5.23` hours at size 44. The scientific question is family
separability under a shared control set, but exhaustive re-auditing is not the right default as the
family grows. New members, previously weakest boundaries, and a rotating sentinel panel should be
tested every cycle; a complete quadratic audit should be occasional calibration rather than the
inner loop.

Exact Shapley accounting has a different cost: an arbitrary `n`-component black-box game needs
`2^n` coalition values. We used it only for small fixed sets. The seven-player P7 game required 258
forwards and ran in about 31--32 seconds; the queued two-case game uses five and six downstream
players and requires 192 forwards. For larger sets, permutation sampling can estimate Shapley values
with confidence intervals, but approximate attribution should only rank follow-ups. It cannot pass
a circuit-identification gate unless the uncertainty is smaller than the decision margin.

Shapley has been useful but secondary. In the P7 game it nominated A11 and M11; the subsequent
physical reader-input loss/output-rescue experiment validated both individual reader interfaces,
although the registered joint-complementarity prediction failed. In the breadth games it showed
that `81--100%` of several source effects traveled through the direct/unlisted background, warning
that greedy downstream unit lists were not mediation chains. Those are useful route-selection and
falsification results. Shapley itself did not identify a residual coordinate, prove a directed edge,
produce an executable circuit, or establish OOD composition. The default decomposition path is
therefore: causal transfer/necessity to fix an interface, reciprocal reader/writer localization,
fold that interface into exact weights, decompose the restricted tensor, and validate common/private
parts by swaps. Shapley is reserved for small ambiguous interaction sets after the pieces are already
operationally defined.

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
but its absolute float32 SVD certificate failed. Under the project rules, pred A failure made that
whole receipt invalid. A post-window float64 repair, with the same rank, weights, seeds, random
controls, and bars, has now passed: held-out captured-energy fractions are `.2291`, `.2687`, `.5022`,
and `.4465` against random 99th percentiles of only `.00204-.00275`; the minimum leave-one-out
projector overlap is `.99543`. This is valid evidence for a shared rank-one **weight-capability**
core, but it is still not evidence of semantic identity, native causal reuse, or the correct adopted
rank.

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

## 14. DAS regularization: what was optimized, what failed, and what remains open

### High-level correction

The current evidence does **not** show that optimized DAS is intrinsically worse than difference in
means (DIM). It shows that the particular losses, regularizers, selection environments, optimizer
budgets, and intervention parameterization tested so far did not yield a construction-invariant,
selective subspace that beat the DIM baseline. The user's concern is therefore live: the
regularization may have been optimized incorrectly, or the primary objective may identify the wrong
solution.

This distinction matters because complement inertness is underidentifying. Many subspaces can leave
a finite complement/control panel nearly unchanged, including subspaces that memorize the training
construction. Passing that test can reject some bad solutions, but cannot by itself prove that the
selected subspace is the intended causal variable.

### The most complete constrained-DAS optimization

The strongest completed test learned a separate orthonormal projector for each of four physical
heads, `L8H1`, `L9H1`, `L9H4`, and `L11H3`. For head `h`, with learned
`U_h in R^(128 x k)` and `U_h^T U_h = I`, the intervention was

```text
P_h = U_h U_h^T
patched head response = base response + P_h(donor response - base response).
```

The projected response was installed as an absolute clamp at every causal prefix position. The
parent attention-15 donor clamp remained fixed, so this experiment optimized the upstream head-local
response interfaces, not an entirely native end-to-end execution.

Rows were split by frozen row-group parity. One parity trained the projector; the opposite parity
selected rank, regularization, and checkpoint, and the direction was then reversed. A2 and C were
sealed until selection was complete. The grid was:

- ranks `k in {1,2,4}`;
- DIM/task-response-SVD and factor-response-SVD initializations;
- Gaussian response noise `sigma in {0,.05,.10}`, scaled by each head's training-response RMS;
- Jacobian/sensitivity weights `lambda_J in {0,.25,1}` through the nonduplicate combinations
  `(0,0)`, `(.05,.25)`, `(.05,1)`, `(.10,.25)`, and `(.10,1)`;
- Adam with learning rate `.03`, eight updates, and checkpoints at steps `0`, `4`, and `8`.

The target side was treated as a hard feasibility constraint: every A1 selection group needed signed
projection at least `.75` and direction fraction at least `.875`. Among feasible candidates, the
secondary score minimized aligned-P full-vocabulary KL plus the weighted local sensitivity penalty.
The implementation used a weight-`100` target barrier and a smooth worst-row P-KL surrogate with
temperature `.05`. Paired-fold selection used the worse held-parity secondary score plus `.25` times
the normalized distance between the two learned projectors. Schematically,

```text
training loss = smooth_worst_P_KL
              + lambda_J * normalized_local_sensitivity
              + 100 * target_feasibility_violations

selection score = worse_held_parity_secondary
                + .25 * cross_fold_projector_distance.
```

This was a substantial search rather than a single fragile fit: 1,738 differentiable forwards,
1,248 backward-forwards, 480 optimizer updates, 28,128 evaluated examples, and 2,048 learned
parameters across the tournament.

### How it was found to be bad

The chosen candidate was rank one, factor-SVD initialized, with `sigma=.05`,
`lambda_J=.25`, and checkpoint step 4 in both folds. It was target-feasible on the A1 selection
folds:

- fold 0: A1 signed projection `.85128`, P mean KL `.001067`, zero P flips;
- fold 1: A1 signed projection `.89229`, P mean KL `.019449`, one P flip.

But the result failed prospectively in two ways. First, sealed A2 target recovery was only `.64888`,
below the `.75` bar. Second, the P panel still had one top-1 flip. The selected regularized candidate
also did not beat matched step-zero DIM on both held parities, producing terminal
`regularization_does_not_beat_dim`. Cross-fold projector stability passed, so this was not merely a
basis-rotation artifact.

The correct diagnosis is **construction-specific generalization/selectivity failure**, not “the
optimizer found no lower training loss” and not “DAS has no better solution.” The training and
selection environments were A1/P parity splits, whereas the decisive target failure occurred on
sealed A2. Thus the experiment can be read as direct evidence that its validation scheme did not
adequately represent the variation we wanted the regularizer to suppress.

For scale, the complete uncompressed parent response recovered A1 `.80535` and A2 `.86829`, but
caused five P flips and three C flips. The best zero-flip exact-factor subset recovered only A1
`.13763` and A2 `.13062`. There is therefore a real target/selectivity tradeoff; the failed optimized
projector did not invent that tradeoff, but it did not solve it either.

### Regularization and selection methods actually tried

| Method | Where it was tested | Outcome |
|---|---|---|
| No explicit regularizer | Single-site, family, nested-construction, and four-head comparisons | Often fit one fold/construction best; did not provide invariant selection. |
| Full-vocabulary KL to the native/control distribution | Single-site and multi-construction tournaments | Sometimes selected and improved pooled control loss, but did not win both folds. |
| Isotropic Gaussian response noise | Alone and with KL; `sigma` including `.05` and `.10` | Helped some folds/configurations, but gains were heterogeneous across constructions. |
| Noise plus KL | Family and nested-construction tournaments | Selected for one construction family; failed to transfer as the universal choice. |
| Local Jacobian/sensitivity penalty | Four-head target-feasible tournament, weights `.25` and `1` | The winning candidate used `.25`, yet failed sealed A2 and retained a P flip. |
| Hard target-feasibility barrier | Four-head tournament | Prevented trivial inert solutions on selected A1 groups, but did not guarantee sealed A2 retention. |
| Early stopping/checkpoint selection | Steps `0/4/8`; broader earlier family checkpoints | The four-head winner stopped at step 4; a nested run selected step zero on one environment. |
| Cross-fold projector-stability penalty | Four-head paired selection, coefficient `.25` | Produced a stable projector but stability did not imply construction invariance. |
| Nested environment selection | Family cross-validation and construction-adaptive tournament | Exposed the problem: one environment chose regularization and another chose no regularization. |

An earlier single-site constrained-DAS run tried no regularization, KL, noise, and noise-plus-KL and
ended null after 3,233 model forwards and 900 updates. A later family tournament selected KL and had
high cross-fold axis cosine (`.88274`); its sealed pooled score improved substantially, but
regularization failed to win both folds. The nested-construction run was even more diagnostic: one
inner environment chose noise `.10` plus KL weight `1`, another chose no regularization, and the
regularized choice scored worse on its outer construction (`1.2052` versus `1.1555`). Its sealed
refit also violated the L15 target-retention bound by `.15642`. These disagreeing choices are
evidence that the current selection target is unstable across environments.

DIM/task-SVD, factor-SVD, exact factor subsets, ranks, and P-complement SVD are **not** counted as
regularizers above. They are initializations, model-class choices, or non-gradient baselines. The
Stiefel constraint `U^T U=I` fixes projector geometry but likewise does not regularize which causal
direction is selected.

### Why the current objective may be wrong

There are several concrete failure modes in the present formulation:

1. Minimizing KL on a finite P panel can memorize that panel instead of learning construction
   invariance. Complement inertness is a necessary diagnostic, not an identifying objective.
2. Row-parity cross-validation changes examples but not the construction. It did not test the A1 to
   A2 environment shift that ultimately failed.
3. Isotropic response noise regularizes every direction equally, while the dangerous perturbations
   may lie specifically along downstream-reader or lexical/construction directions.
4. The Jacobian term is local around the training response. It need not control finite donor swaps or
   top-1 changes after nonlinear downstream computation.
5. A fixed scalarization of P KL, sensitivity, feasibility, and fold distance can prefer a stable
   but semantically wrong subspace. Stability only says the same estimator was found twice.
6. Eight Adam updates and a coarse hyperparameter grid are adequate for comparing the frozen small
   tournament, but not evidence that the best feasible DAS solution was found.
7. Independent head-local fixed-rank projectors and the retained attention-15 clamp may be the wrong
   parameterization if the true variable is distributed jointly across heads or requires native
   downstream adaptation.

### Promising alternatives not yet honestly tested

The following should not be reported as attempted results. They are proposed corrections:

- leave-one-construction-out selection with A1, A2, and genuinely new constructions treated as
  separate environments;
- group-DRO/minimax loss over constructions and lexical groups instead of average or parity loss;
- a target constraint in **every** training environment, followed by a sealed held-construction
  test, rather than A1 feasibility plus sealed A2 discovery;
- KL or logit preservation over a broad native-text corpus, not only the named P/C panels;
- adversarial perturbations aligned to measured downstream readers, replacing or supplementing
  isotropic Gaussian noise;
- finite-intervention robustness and smooth top-1-margin penalties, rather than only a local
  Jacobian proxy;
- explicit distance-to-DIM/geodesic weight decay, ordinary parameter weight decay, head dropout, and
  flatter-minimum or ensemble selection;
- target/control gradient-conflict methods or constrained optimization with separate dual variables,
  avoiding a single hand-weighted scalar objective;
- longer optimization with multiple frozen restarts, while charging the search price and retaining a
  matched DIM baseline.

The next fair DAS claim should therefore be narrower and stronger: test whether environment-level
robust selection can beat DIM on a sealed construction while satisfying target recovery and broad
complement inertness simultaneously. Until that experiment passes, the present result rejects the
tested regularization recipe—not optimized subspace methods as a class.
