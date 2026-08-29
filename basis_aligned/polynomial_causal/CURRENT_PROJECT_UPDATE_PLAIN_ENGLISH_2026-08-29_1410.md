# Current project update, with the computations explained

**Updated:** 2026-08-29 14:10 UTC

## UPDATE: what is genuinely new

The most important new result is about a four-attention-head copy circuit:

`L5H5 + L7H3 + L8H3 + L8H4`.

When all four heads are replaced together, the model becomes much worse at positions
where the preceding text supplies an earlier example of the right continuation. The
copy-position cross-entropy increases by **0.44870 nat**. This is a large and
statistically robust causal effect.

However, the same replacement also worsens ordinary, off-target positions by
**0.02441 nat**. Our replacement was required to keep this damage below `0.01` nat.
It therefore fails as a *selective removal*. We have found an important bundle of
heads, but the replacement removes both their copy-related work and some of their
other work.

A new analysis sharpens this result. Replacing the four heads jointly has
**4.19 times** the copy effect obtained by adding their four individual replacement
effects. In numbers,

$$
\underbrace{0.44870}_{\text{replace all four}}
\quad\text{versus}\quad
\underbrace{0.06205+0.01121+0.02082+0.01297}_{
\text{sum of four individual effects}}
=0.10705.
$$

The difference is

$$
0.44870-0.10705=\mathbf{0.34165}\text{ nat}.
$$

Across 10,000 document-level bootstrap resamples, the simultaneous 95% lower bound
on this difference is `0.20810` nat. Thus ordinary sampling variation does not
plausibly explain the non-additivity.

**Additional update from the 14:20 strategic review:** the complete predicted-token
distribution is even more non-additive on copy positions. Native-to-replacement KL is
`0.38835` for the four-head intervention versus `0.05986` for the singleton sum, a
`6.487x` ratio. Off target, the ratio is only `1.683x`. The copy-minus-matched KL
excess is `0.29373`, with simultaneous lower bound `0.16249`. This makes a uniform
“the joint ablation is simply larger everywhere” account less plausible, but it is not
a residual-vector norm measurement; the next cube must record displacement norm and
alignment directly.

This changes the immediate interpretation. The heads should not be treated as four
independent copy features that can be ranked and removed one by one. Most of the
observed copy effect is a **joint effect** produced when they are changed together.
The next mathematical experiment is designed to determine whether that joint effect
comes from a particular pair, a triple, or a genuinely four-head interaction.

The strict whole-model ledgers have **not** moved. This result identifies a causal
bundle, but it has not yet produced a selective extraction/removal action.

## 1. Exactly what was changed in the model?

Each attention head contributes a vector to the residual stream at every token
position. Call the native output of head $h$ at position $p$

$$
w_h(x,p),
$$

where $x$ is the document. This is a 1,152-dimensional vector after the head has been
written through the attention output projection.

For a chosen set of heads $H$, the intervention computes

$$
w'(x,p)
=w_{\mathrm{native}}(x,p)
-\sum_{h\in H}w_h(x,p)
+\sum_{h\in H}\mu_h(p).
$$

Here:

- $w_{\mathrm{native}}$ is the complete native attention write;
- subtracting $w_h$ removes the chosen head's live, context-dependent output;
- $\mu_h(p)$ is that head's average output at position $p$, estimated on a separate
  fit dataset.

So this is not setting the head to zero. It retains its ordinary position-dependent
mean and removes the part that varies with the current document. This is a sensible
first ablation, but it is still blunt: every context-dependent function of the head is
removed, not just copying.

The tensor structure is useful here because the projected head writes add exactly.
We can subtract any chosen collection of heads and put a replacement vector back at
the same residual-stream interface without changing the rest of the model.

## 2. What are the positive, matched-negative, and off-target cells?

A **cell** is just a predefined class of scored token positions. It is not a neuron,
tensor entry, or learned cluster.

At a position $p$, let the current token be $q=x_p$ and the next token be
$y=x_{p+1}$. We look backward for an earlier occurrence of the same query token $q$.

### Copy-positive cell

A position is positive when an earlier occurrence of $q$ was followed by the same
next token $y$. Schematically,

$$
\ldots,q,y,\ldots,q,\boxed{y}.
$$

The earlier `q -> y` transition makes copying or induction a useful way to predict the
boxed token.

### Matched-negative cell

A matched negative also has an earlier occurrence of $q$, but the earlier occurrence
was not followed by $y$. It is matched to a positive for quantities such as current
position, distance to the earlier query, and query/target frequency. It therefore
asks whether an intervention specifically harms reusable continuations, rather than
merely harming positions with repeated tokens.

### Off-target cell

Off-target positions are all other valid scored positions that are neither selected
positives nor their matched negatives. This large cell measures ordinary collateral
damage.

The selection data contained 192 documents, 303 positive positions, 303 matched
negatives, and 33,570 off-target positions. Uncertainty was resampled at the document
level, so thousands of nearby tokens from one document were not treated as thousands
of independent documents.

## 3. What was the score?

For the correct next token $y$, cross-entropy is

$$
\operatorname{CE}=-\log p(y).
$$

Lower is better. If the intervention changes cross-entropy by

$$
\tau_c
=\operatorname{CE}_{\mathrm{replacement},c}
-\operatorname{CE}_{\mathrm{native},c},
$$

then a positive $\tau_c$ means that replacing the heads damaged prediction in cell
$c$.

For the four-head set, the measured values were:

| Cell | Native CE | Replacement CE | Change $\tau$ | Meaning |
|---|---:|---:|---:|---|
| Copy positive | 0.67495 | 1.12365 | **+0.44870** | Large copy damage |
| Matched negative | 2.41819 | 2.40337 | **-0.01482** | No generic harm on the matched control |
| Off target | 3.21346 | 3.23787 | **+0.02441** | Too much ordinary collateral |

Copy-position top-1 accuracy also falls from **86.14% to 77.89%**. Cross-entropy is
the primary score because it detects probability changes even when the most likely
token does not change.

We define copy specificity as

$$
S=\tau_{+}-\tau_{-},
$$

where $\tau_+$ is the positive effect and $\tau_-$ is the matched-negative effect.
Here,

$$
S=0.44870-(-0.01482)=\mathbf{0.46352}\text{ nat}.
$$

This is strong evidence that the bundle matters especially at copy positions. Its
simultaneous lower confidence bound is `0.28182` nat.

The collateral rule is expressed as the remaining margin

$$
M=0.01-\tau_{\mathrm{off}}.
$$

Here,

$$
M=0.01-0.02441=\mathbf{-0.01441}\text{ nat}.
$$

A passer needed this margin to be positive with its simultaneous uncertainty bound.
It is negative, so no candidate was selected and the protected final/OOD datasets
were not opened.

## 4. What does the new joint-effect computation do?

Let $F$ be the complete four-head set, and let $\{h\}$ mean a single head. Define

$$
v(S)=\operatorname{CE}(\text{replace exactly the heads in }S)
-\operatorname{CE}(\text{native}).
$$

If the heads made independent additive contributions under this intervention, we
would expect

$$
v(F)\approx\sum_{h\in F}v(\{h\}).
$$

Instead the individual positive-cell effects are:

| Head replaced alone | Copy CE change |
|---|---:|
| L5H5 | +0.06205 nat |
| L7H3 | +0.01121 nat |
| L8H3 | +0.02082 nat |
| L8H4 | +0.01297 nat |
| **Sum** | **+0.10705 nat** |
| **All four together** | **+0.44870 nat** |

The descriptive interaction excess is

$$
e=v(F)-\sum_{h\in F}v(\{h\})=0.34165\text{ nat}.
$$

The same calculation for specificity gives an excess of `0.31793` nat, with a
simultaneous lower bound of `0.18438`. Thus most of both the copy damage and its
specificity is missed by an additive head-by-head story.

This does **not** yet tell us that there is a pure four-way interaction. Six pair
interventions and four triple interventions were not measured. For example, one
special pair could account for nearly all the excess. It also does not distinguish
an internal head interaction from nonlinear amplification by RMSNorm and the later
network. It only establishes that the full intervention's behavioral effect cannot be
predicted by summing the four singleton effects.

The uncertainty calculation used 10,000 bootstrap datasets. Each bootstrap dataset
sampled 192 documents with replacement, recomputed the joint-minus-singleton contrast,
and used a simultaneous maximum-deviation radius across positive, matched-negative,
off-target, and specificity contrasts. This is more conservative than placing a
separate interval on each number.

The interaction analysis is post-hoc: it was suggested after seeing the registered E4
result. It is strong descriptive evidence, but the next subset experiment must use
fresh prospectively frozen data before it can select a circuit.

## 5. Why this is progress rather than merely another failed ablation

Before this result, two substantially different stories were plausible:

1. each of four heads independently supplied a small copy signal; or
2. the downstream model used a coordinated computation distributed across them.

The observed 4.19-times ratio strongly favors the second operational description.
This gives us a more precise target than “interpret these heads.” We now need to find
the smallest *interaction term* that carries the copy effect and separate it from the
heads' ordinary work.

That distinction matters for the project's goals:

- **Prediction:** an additive head score would underpredict the four-head ablation by
  `0.34165` nat.
- **Extraction:** copying may require a pair/triple composition, not a list of
  individually strong heads.
- **Removal:** removing full head outputs is too blunt; an input-conditional or
  interaction-specific removal is needed.
- **OOD transport:** a real copy program should transport when the token identities
  and document domain change but the repeated-query relation stays the same.
- **Simplicity:** four head names are not a sufficient program description if their
  effect depends on a conditional composition that the description omits.

It is still not a semantic explanation of the complete model, and it does not move the
strict causal ledger. It is progress in identifying the form that a successful
explanation must have.

## 6. The next experiment: all 16 subsets

Four heads have

$$
2^4=16
$$

possible subsets, including replacing no heads and replacing all four. The existing
result contains the native/no-replacement baseline, the four singletons, and the full
set. The missing interventions are:

$$
\binom{4}{2}=6\text{ pairs},\qquad
\binom{4}{3}=4\text{ triples}.
$$

So only ten new subset arms are required.

Once all subsets are measured on a new role, we can compute the unique discrete
interaction coefficient for every subset $T$:

$$
m(T)=\sum_{S\subseteq T}(-1)^{|T|-|S|}v(S).
$$

This is the Boolean-lattice or Möbius decomposition. It is analogous to decomposing a
polynomial into main effects, pair interactions, triple interactions, and a four-way
interaction. It answers questions such as:

- Is one pair responsible for most of the copy effect?
- Is a triple necessary?
- Is the full four-head effect genuinely irreducible at this intervention interface?
- Are copy effects and off-target damage carried by different interaction terms?

The last question is the most valuable. If one interaction term has a large copy
effect but little collateral, it supplies a much better circuit boundary than an
entire head output.

## 7. The likely successor: a context-conditional replacement

The current replacement is unconditional: it removes the live head output at every
position. But the proposed copy computation is conditional on a relation in the
context—roughly, “the current query appeared earlier and had a reusable successor.”

A small executable context state could be

$$
Z_p=(\text{prior query exists},\text{distance},\text{prior successor},
\text{current query/target relation},\text{position}).
$$

We would then replace only a predicted copy-dependent component $R(Z_p)$ and preserve
the complementary head output. The cheapest useful test compares:

1. the current unconditional mean replacement;
2. a copy-predicate-gated replacement;
3. a random gate activated at exactly the same rate.

The proposal succeeds only if it preserves a substantial fraction of the `0.44870`
copy effect while reducing off-target damage below `0.01` nat. The rate-matched random
gate checks that improvement is due to choosing the right contexts, rather than simply
intervening less often.

This is an example of **causal abstraction**: $Z$ is useful only if interventions
written in the small language have predictable corresponding effects in the full
network. Merely predicting the head activation with low squared error is not enough.

## 8. A second mathematical route if the subset effects are diffuse

If the 16-subset experiment shows that the effect is spread across many interactions,
head identity is probably the wrong coordinate system. The next route is a
**downstream-Fisher active subspace**.

Let $h$ be the concatenated head outputs, let $z=G(h)$ be the final logits, and let

$$
J=\frac{\partial z}{\partial h}
$$

be the downstream Jacobian. $J$ says how a small change in the head interface changes
the final logits. If $p$ is the native next-token distribution, the categorical
Fisher matrix is

$$
F=\operatorname{diag}(p)-pp^\top.
$$

For a small interface error $\delta h$, the induced change in the output distribution
is approximately

$$
\operatorname{KL}\bigl(p(z)\,\|\,p(z+J\delta h)\bigr)
\approx\frac12\delta h^\top J^\top FJ\delta h.
$$

This gives a behavior-weighted notion of direction importance. PCA or HOSVD keeps
directions with large activation energy; the Fisher construction keeps directions to
which the downstream prediction is sensitive. It is therefore better aligned with
functional faithfulness, although it is only a local approximation and must be checked
with finite interventions and rare/OOD data.

## 9. How this fits with MLP0 and Family F

There is no new accepted semantic decomposition of MLP0 in this update. Its Down
outputs still admit useful continuous low-rank descriptions, and shared lexical
structure remains a plausible description, but we do not yet have a sparse semantic
dictionary whose coordinates compose downstream and support selective edits.

The strongest lesson from MLP work is Family F at MLP3. Selecting 512 native product
gates and retaining their native Down columns gives downstream teacher KL `0.05772`.
Refitting the Down decoder to reduce local activation error makes downstream KL worse,
`0.08476`. This shows why the current attention result is being analyzed in behavioral
coordinates rather than by activation reconstruction alone.

The independent native-Down behavioral-port experiment remains high priority. It asks
whether the promising 512-gate program transfers to fresh data and finite edits. It is
complementary to the copy interaction experiment:

- the subset cube investigates a small attention behavior circuit;
- the native-Down port tests a small exact polynomial MLP program;
- the later composition telescope will test whether independently simplified MLP0,
  MLP1, MLP2, and attention components still work when installed together.

## 10. What happened to the simplicity metrics?

The recent results have made the simplicity standard stricter and more useful.

### Metrics that remain valid prices

- stored scalar values or serialized bytes;
- products and multiply-adds per token;
- number of live gates, interaction terms, or conditional program branches;
- the amount of context/state required at execution time.

These measure literal program cost. A scalar rescaling or constant bias can often be
folded into an existing operation and should have negligible marginal price.

### Metrics that validate the cheaper object

- held-out whole-model cross-entropy and native-to-program KL;
- top-1 agreement as a secondary practical metric;
- prediction of unseen intervention compositions;
- OOD transport;
- intended behavioral effect and off-target collateral;
- stability under doubling documents.

### Metrics that are proposal generators, not proof of simplicity

- matrix or tensor rank by itself;
- local activation MSE/NRMSE;
- SAE sparsity on weights or activations;
- HOSVD/CP reconstruction error;
- human-readable clusters without downstream tests.

The operational definition is:

> A representation has earned the label “simpler” only when its lower literal price
> buys a useful capability—prediction, composition, extraction, selective removal,
> OOD transport, certification, or cheaper execution—on untouched data.

The E4 result illustrates this. “Four heads” sounds simple, but it is not yet a useful
simple program: its whole-output removal is not selective, and an additive description
misses most of its effect. A pair/triple interaction or small conditional macrostate
could be both more accurate and more operationally useful.

## 11. Other current evidence, with confidence labels

The strict ledgers remain:

| Ledger | Value | Interpretation |
|---|---:|---|
| Structural interfaces | 36/36 | Every site can be executed/intervened on; not semantic explanation |
| Certified storage removed | **5.348245316%** | Whole-program storage with current consequence certificate |
| Named causal CE | **10.923302467%** | Named effects in the strict causal ledger |
| Unnamed CE | **4.72714 nat = 89.076697533%** | Largest quantitative explanation gap |
| Terminal extraction/removal actions | **0/68** | No practical circuit has passed the full chain yet |

A parallel discovery-only map-rank sweep also reports a useful engineering shape.
Increasing the context-free fallback map from rank 64 to 512 improves all four rare
frequency buckets on all three sampled roles, while the most common `125+` bucket gets
worse. Rank 128 buys much of the rare-token gain at a small fraction of rank 512's
extra storage. But no rank 128/256 “sweet spot” passed its registered requirement, and
the response is not monotone. This is not receipt-backed and does not move any strict
ledger. It says the fallback rank is a deployment tradeoff, not the missing semantic
explanation.

## 12. Current plan and blockers

The ranked plan is now:

1. **Freeze and run the complete 16-subset four-head experiment on a fresh role.**
   This determines the interaction order and whether useful copy effect separates from
   collateral.
2. **Run the rank-512 native-Down behavioral port on fresh rows.** Test CE, KL,
   finite edits, matched controls, and OOD transfer without locally refitting away the
   native decoder geometry.
3. **If the subset cube is sparse, build the conditional copy macrostate and gated
   replacement.** Require off-target damage below `0.01` while retaining at least half
   the observed copy effect in discovery before any final claim.
4. **Build the component composition telescope.** On the same documents, install
   MLP0, MLP1, MLP2, and attention simplifications alone and in combinations, then
   compute the interaction remainder. This directly measures compensation.
5. **If head subsets are diffuse, estimate the downstream-Fisher basis** and compare
   it with PCA/HOSVD at matched rank and executable cost.

There is no missing checkpoint, FineWeb cache, or external software blocker. The
current blockers are scientific:

- the copy intervention is causally strong but not selective;
- the ten pair/triple subset effects are unmeasured;
- MLP simplifications have not yet passed fresh finite-edit and composition tests;
- no learned coordinate system has both semantic/editable meaning and whole-model
  faithfulness.

The mathematical review therefore did lead to a concrete advance: it produced the
joint-minus-singleton test, found a robust `0.34165`-nat non-additive effect, and
converted that observation into a ten-arm falsifiable subset experiment. That is more
specific than another low-rank fit and has a direct route to extraction or selective
removal if copy and collateral occupy different interaction terms.

## 13. Claim boundaries

What we can now say:

- the four-head bundle has a large causal effect on the registered natural-copy cell;
- its effect is specific relative to matched negatives;
- replacing all four live writes by position means causes too much off-target damage;
- the full copy and specificity effects are strongly non-additive relative to the four
  singleton effects.

What we cannot yet say:

- that any single head, pair, or triple is the copy circuit;
- that the excess is a pure four-way internal interaction;
- that we can selectively remove or extract copying;
- that the circuit generalizes to final natural data or OOD code;
- that the copy result increases the fraction of the entire model explained.

The numerical interaction artifact is
`e4_four_head_nonadditivity_descriptive.json`; it is bound to the receipt-backed E4
ledger by SHA256 and is explicitly marked post-hoc/descriptive.

## 14. Retrospective: how did the eight-hour exploration go?

> **Correction:** section 15 at the bottom supersedes this section's novelty claim and
> proposed four-head power-set priority after comparison with the older induction work.

At 04:00 UTC we deliberately stepped away from a single MLP0 fit and committed to four
alternative entry points. The goal was not to make all four work. It was to run cheap,
falsifiable experiments that could tell us which route deserved a larger investment.
We also committed to finish Family F, the downstream-selected MLP3 gate experiment.

The honest summary is:

> The eight hours were successful at **pruning attractive but insufficient ideas** and
> identifying two sharper successors. They were not successful at producing a new
> whole-model explanation, a selectively removable circuit, or an increase in the
> strict explained fraction.

At the 12:00 deadline there were six measured negative evidence cells, three cells
pruned because a stronger prerequisite had already failed, and three E4 cells still
open. The narrowed E4 copy experiment completed shortly after the deadline as a
scientific negative for its exact replacement bank. Plans, tests, row caches, and
unrun runners were not counted as outcomes.

### Preliminary task: Family F

**Question:** Can we choose a small set of exact MLP3 multiplication gates according
to their downstream consequences, then refit a compact output decoder?

**What ran:** The full frozen 480-row fit experiment, including a preserved v1
publication failure and a successful receipt-last v2 recovery.

**Outcome:** The locally refitted 256- and 512-gate programs failed their faithful-port
NRMSE requirement badly: `0.78860` and `0.70275`, versus the required `<=0.20`. They
could not open validation.

**What was useful:** Gate selection worked better than the controls downstream. At
512 gates, consequence selection plus the local decoder gave teacher KL `0.08476`,
versus `0.10077` for matched random support and `0.08862` for activation-selected
support. More importantly, using the selected gates' **native Down columns** gave KL
`0.05772`, even though its local activation NRMSE was worse.

**Decision:** Stop optimizing the local decoder by MSE. Prospectively test the
one-ninth-size native-Down gate program on fresh CE/KL and finite edits. This was the
clearest positive lead produced by the eight-hour window, although it is not yet a
validated program.

### Entry point 1: close the rank-512 stream-map dataflow

**Original hope:** A rank-512 linear map looked very accurate when its input was the
native one-token residual stream. Perhaps it could become the fallback computation for
uncovered tokens.

**Critical question:** Does it still work when its input is produced by the compressed
program itself, rather than supplied by the native model?

**Experiments and outcomes:**

1. Recursive deployment gave deficits of `1.08978 / 1.27276 / 1.26133` nat. The same
   map on native streams had deficits only `0.17427 / 0.21358 / 0.21419`.
2. Refitting the map on the compressed program's own streams did not repair the
   mismatch. Three iterations gave much worse deficits:
   `5.49867 / 5.61939 / 5.59476` nat.
3. The planned layer-by-layer drift localization was pruned. Once direct refitting on
   the deployed inputs failed by roughly five nats, locating the first drift could not
   rescue this particular program within the window.

**Interpretation:** The strong native-stream rank-512 result was an oracle-interface
result, not a closed standalone compiler. Earlier model computation supplies state that
the context-free compressed prefix does not reproduce. A large rank linear map can
read that state but does not create it.

**Decision:** Close this route. Keep embedding/stream rank as an engineering fallback
frontier, but do not treat it as the route to semantic reverse engineering.

### Entry point 2: factor all 36 site maps jointly

**Original hope:** The 36 site-output maps might share one small continuous language.
A common basis could reduce storage and perhaps give stable coordinates that are easier
to name or edit.

**Experiments and outcomes:**

1. One global output dictionary at ranks 64/128/256/512 was compared both with
   independent maps of the same rank and with independent maps of the same storage
   price. Tight ranks 64/128 beat the equal-storage independent allocation by
   `0.022--0.036` nat, so real sharing exists. But every global rank lost to
   independent maps at the same rank, and no rank passed both registered conditions.
2. Separate attention and MLP dictionaries improved over one global basis by only
   `0.00250 / 0.00237 / 0.00004` nat at equal storage, below the `0.01` requirement.
3. Rotating the failed shared projector into sparse coordinates was pruned: rotation
   cannot restore directions missing from the subspace. A stronger shared-trunk plus
   private-residual hierarchy was then tested and also lost to spending the same price
   entirely on private ranks at the large budget.

**Interpretation:** There is genuine shared geometry at tight storage budgets, but
there is not one sufficient universal output language at the tested budgets. The
site-private residuals are too valuable, and the coarse attention/MLP split does not
explain them.

**Decision:** Retain shared bases as a compression tool at tight rank 64/128. Do not
claim semantic coordinates, and do not spend another large run on the same global or
rank-128-trunk hierarchy. A future shared basis should be weighted by downstream
causal sensitivity, not raw output energy.

### Entry point 3: work backward from downstream consequences

**Original hope:** Instead of reconstructing a full residual vector, learn a small
state containing only what later computation needs. This was the system-identification
or minimal-realization route.

**Experiments and outcomes:**

1. Infinitesimal MLP0--2 response panels remained full rank at the tested 32, 64, and
   96 columns. Their spectra were not stable enough across document splits to select a
   trustworthy low-dimensional knee.
2. A finite rank-64 state was tested on an unseen
   `L8 -> L11 -> L14` composition. The composed prediction error was `0.4520`; the
   direct-map error was `0.4861`. Even projecting the **true** destination response
   into the proposed rank-64 space gave error `0.2709`, above the `0.25` gate.
3. Editing one learned coordinate was pruned because the state had already failed
   destination sufficiency. An edit in that coordinate system would have treated a
   failed locator as though it were a valid causal API.

**Interpretation:** Composition itself did not create a mysterious extra collapse—the
direct and chained maps were similarly poor. The main failure was the chosen
64-dimensional pointwise state. It omitted downstream-relevant response before the
maps were composed.

**Decision:** Reject this universal rank-64 state, not system identification in
general. A future state should be behavior-specific, temporal, nonlinear, or selected
using the downstream Fisher metric. It must first predict held-out finite responses
before its coordinates can be called editable variables.

### Entry point 4: begin from a sharply defined behavior

**Original hope:** A short terminal circuit for copying, capitalization, or number
formatting might be easier to extract than a generic component decomposition. A
successful extraction/removal would also tell us which simplicity metric is useful in
practice.

**What actually happened:** This was the most infrastructure-heavy direction. The
window produced fresh document-disjoint roles, exact copy-positive and matched-negative
definitions, a checkpoint-exact per-head attention interface, a physical candidate
dispatcher, fit-only position means, streaming sufficient statistics, and a
receipt-last selection lifecycle. These were prerequisites, not experimental outcomes.

The narrowed attention/copy screen completed shortly after the 12:00 deadline. The
four-head set had a large copy effect, `0.44870` nat, and specificity `0.46352` nat.
But off-target damage was `0.02441` nat, over the `0.01` limit, so no candidate passed
and final/OOD remained unopened.

The later CPU analysis found that the joint effect is strongly non-additive: the
four-head copy effect is `4.19x` the sum of the singleton effects. The complete-output
KL ratio is `6.49x` on copy positives but only `1.68x` off target.

**Interpretation:** Behavior-first analysis did localize a real causal bundle. It did
not yet produce a selective removal because replacing whole context-dependent head
writes is too coarse. This direction supplied the first clear target for a structured
interaction decomposition, which the component-first routes had not supplied.

**Decision:** Continue the behavior-first route, but change the unit of analysis from
whole heads to pair/triple interaction terms and context-conditional components. The
next experiment is the complete 16-subset cube with displacement-magnitude controls.
Capitalization, number formatting, and E4 extraction/removal were not completed and
must not be counted as explored outcomes.

### Did the four-direction exploration pay off?

It did not pay off in the strongest sense: there is still no new certified whole-model
compression percentage, no first terminal action, and no simple semantic program for
MLP0 or the complete model.

It did pay off strategically in five concrete ways:

1. **It falsified the native-state oracle shortcut.** A reader of native hidden state
   is not automatically a self-contained program.
2. **It separated compression sharing from semantic sharing.** A shared basis can be
   economical at tight rank without being a sufficient or interpretable universal
   language.
3. **It located the predictive-state failure at the representation, not an extra
   composition catastrophe.** This narrows the successor design.
4. **It showed that local reconstruction can select the wrong decoder.** Preserving
   native Down geometry is now a concrete, testable alternative.
5. **It found a large, behavior-specific, non-additive attention bundle.** The failure
   is now selective isolation, not absence of a causal signal.

So the search did what a good exploratory portfolio should do: three broad interfaces
were sharply pruned, one produced a promising polynomial successor, and one produced a
promising causal interaction successor. The project should now concentrate full runs
on those two successors—the 512-gate native-Down behavioral port and the four-head
interaction/conditional-copy program—rather than reopening all four broad ideas.

## 15. **UPDATE AND CORRECTION: what was actually new, why progress was slow, and what changes now**

This section answers the follow-up about the eight-hour window. It also corrects the
conclusion immediately above. After rereading the older induction ledger, I agree that
the preceding account made the recent attention result sound substantially newer than
it was.

### 15.1 Where was the answer to the eight-hour question?

It was section 14, beginning with “Retrospective: how did the eight-hour exploration
go?” That section was appended to this file, but it was not visually marked as the new
answer and the most important qualification was missing. This section is the corrected
answer and deliberately appears at the very bottom.

### 15.2 The user's summary is substantially right

The recent E4 experiment did **not** discover the model's copy or induction heads.
It started from an already registered four-head set. Older work had already established
all of the following:

- Induction is distributed and cooperative: single-head or single-layer ablations
  seriously understate the damage caused by ablating a group.
- A few identifiable heads carry much of the *singleton* induction signal, while a
  distributed tail carries much of the total behavior.
- The named four-head copy front end had already been reduced to a computation using
  the embedding table, the heads' own query/key/value/output weights, one affine
  verdict map, and two signed payload scalars. That stand-in recovered approximately
  `0.78` of its held-out intervention stake and transferred to an unseen repeat period.
- A later route-grain extraction retained a cheap architectural broadcast route through
  otherwise removed heads and recovered `79%` of the induction gap. It also exposed a
  real limitation: the same route serves non-copy computation, so extraction and
  selective removal do not coincide automatically.

Those results are recorded in the older bilinear-quotient ledger, especially
sections 953--955, 1257--1261, 1290--1294, and 1314--1316. Therefore “we found a
non-additive four-head copy bundle” is not an adequate account of progress. The broad
mechanistic fact was already known.

The exact recent four-head set, `L5H5, L7H3, L8H3, L8H4`, is also not identical to
every earlier set called a “copy circuit.” The older ledger contains (i) a ranked
collection of identifiable induction heads, (ii) the four-head matcher/fetcher front
end `L2H5, L3H8, L8H3, L8H4`, and (iii) broader closure and transport sets. The recent
experiment used the prior-minimal set frozen in its own preregistration. This is
another reason not to describe it as a fresh localization.

### 15.3 What the recent E4 work genuinely added

Its incremental contribution was a stricter measurement, not a new circuit:

1. It evaluated an already chosen set on document-disjoint natural-text copy-positive
   positions, matched negative positions, and a broad off-target population.
2. It used an exact checkpoint-level intervention that replaces each selected head's
   live output by a mean learned only from fit data.
3. It measured a copy-position CE effect of `0.44870` nat and matched-negative
   specificity of `0.46352` nat.
4. It showed that this **particular mean-replacement bank fails selective removal**:
   off-target damage was `0.02441` nat, above the registered `0.01` limit.
5. A post-hoc calculation quantified the already expected cooperation under this new
   interface: the joint copy CE effect was `4.19x` the singleton sum. For full-output
   KL, the ratio was `6.49x` on copy positions and `1.68x` off target.

This is useful calibration of a new intervention and data contract. It is not a new
explanation of induction.

The terminology matters here. This experiment is **necessity-like**: removing a set
of live writes damages a behavior. It is not a sufficiency test, because it did not run
those components as a standalone program or transplant their computation into a
disabled model. The older `0.78` stand-in and `79%` route-grain extraction are much
closer to sufficiency/extraction evidence. “More sufficient than necessary” was
therefore not the right description of the new result.

### 15.4 The proposed 16-subset power set is not the main next step

For four heads there are $2^4=16$ subsets. Measuring all of them permits a Möbius or
inclusion--exclusion decomposition: singleton, pair, triple, and four-way contrasts.
That can tell us which *ablation combinations* account for the non-additivity.

But it does not automatically decompose a head into an executable copy component. A
Mobius interaction is a contrast between interventions, not a tensor that can be
removed, transplanted, or evaluated on a new input. Because earlier work already
established distributed cooperation, a complete subset cube is now a modest diagnostic,
not a high-return research direction. It should either be skipped or capped at less
than one hour and run only if it selects between two concrete replacement designs.

### 15.5 What was done during the eight hours besides Family F and E4?

Four lines of inquiry did receive numerical tests:

| Line | Actual scientific result | Decision |
|---|---|---|
| Family F | Refitted 256/512-gate local programs failed activation fidelity; the 512-gate **native-Down** version nevertheless reached KL `0.05772` | Preserve the native-Down lead; stop local-MSE decoder tuning |
| E1 | A rank-512 reader that works on native streams failed when recursively supplied compressed streams; refitting made it worse | Closed: it reads state that the small program does not construct |
| E2 | Shared dictionaries helped only at tight equal-storage budgets; one global language and a shared-trunk hierarchy lost to site-private allocation | Keep as economical compression, not a universal semantic basis |
| E3 | A universal rank-64 downstream-response state failed held-out finite transport and destination sufficiency | Reject this state; do not infer that all system-identification approaches fail |
| E4 | The stricter four-head mean replacement was behavior-specific but had excessive off-target damage; its joint effect was non-additive | Exact replacement bank rejected; broad mechanism mostly old knowledge |

Thus it is not literally true that only two computations ran. But it **is** fair to
say that the return per wall-clock hour was poor: three negative interfaces, one
promising but unconfirmed native-Down lead, and one mostly confirmatory attention
result did not justify the amount of process wrapped around them.

### 15.6 What consumed the time?

The model computations were not eight-hour computations. Recorded examples are:

- Family F: `75.26` seconds;
- the main shared-map sweep: about `320` seconds;
- the corrected hierarchy sweep: `421.75` seconds;
- the response-panel calculation: `199.87` seconds;
- the finite transport calculation: `244.1` seconds;
- the final E4 transaction: roughly twenty minutes from authority opening to ledger
  publication, including checks rather than twenty minutes of pure GPU arithmetic.

Data loading was also not the bottleneck: one measured collection path took `0.38`
seconds after checkpoint load.

Most wall time instead went into building fresh data roles, sealing manifests and
hashes, writing receipt-last publication logic, source-closure audits, testing, fixing
device/publication/hash failures, independent review, and waiting behind other GPU
jobs. Some assurance suites alone took about two minutes per invocation, and they were
run repeatedly while their infrastructure changed.

Those controls are appropriate for a final confirmatory result. Applying nearly the
same ceremony to every exploratory branch was an efficiency mistake. In addition, the
older mechanistic ledger was not consulted aggressively enough before E4, so we paid
confirmatory costs to re-establish a result whose qualitative content was already
known. The concern that progress was roughly four times slower than it should have been
is reasonable.

### 15.7 Revised operating rule: discovery first, confirmation second

From now on there should be two visibly separate lanes:

1. **Discovery lane.** Reuse already exposed/cached roles and existing intervention
   adapters. Time-box a probe to `20--45` minutes. Start with 32 documents and double
   to 128 only when the effect is large and stable. Mark the result exploratory; do
   not build a new receipt system for it.
2. **Confirmation lane.** Only a candidate that is both new and actionable earns fresh
   document roles, preregistered gates, OOD data, receipt-last publication, and
   independent audit.

A result is actionable only if it changes an executable program: it selects a tensor
term to keep/remove, improves a composition, predicts a held-out intervention, or
strictly improves the price--behavior frontier. A new plot of a familiar phenomenon
does not qualify.

### 15.8 Revised high-return plan

The earlier recommendation to concentrate on a four-head power set is superseded.
The ranked plan is now:

1. **Resume from the existing copy program rather than rediscovering it.** Reproduce
   the prior `0.78` weights-and-embedding stand-in and `79%` route-grain extraction
   from their preserved scripts/artifacts. This is a baseline audit, not a new claim.
   Then ask exactly which part of the remaining `21--22%` is missing.
2. **Decompose the known heads by exact source-position contributions.** Define the
   value vector of head $h$ at source position $s$ by

   $$
   v_h(s)=W_V^h x_s+b_V^h.
   $$

   Then the complete attention write at destination position $t$ is

   $$
   o(t)=b_O+\sum_h\sum_s o_{h,s}(t),
   \qquad
   o_{h,s}(t)=a_h(t,s)\,W_O^h v_h(s).
   $$

   Here $s$ is a source position, $a_h(t,s)$ is the attention weight from $t$ to $s$,
   $x_s$ is the source residual vector, $W_V^h$ and $W_O^h$ are the head's value and
   output maps, and $b_V^h,b_O$ are the biases. The shared output bias $b_O$ stays
   untouched. This sum is exact for a completed forward pass, including the value and
   output biases that must not be silently dropped. It lets us remove only the
   contribution from the matched source or successor-payload route, while keeping the
   same head's unrelated traffic. That is a direct candidate for selective removal,
   unlike a whole-head mean or a Möbius contrast. We should combine it with the older
   split between the cheap $\lambda v_1$ broadcast route and the fresh-value route.
3. **Run the MLP0/MLP1/MLP2/attention composition telescope already motivated by the
   user's question.** Independently compress each component, then test every prefix
   composition on the same cached rows. The first composition whose CE jumps identifies
   a missing interface. This directly tests whether MLP2 really compensates for an
   MLP0 simplification and whether independently good reductions compose.
4. **Give native-Down K512 one cheap behavioral test, without another infrastructure
   project.** Its KL `0.05772` is the strongest new polynomial lead from the eight-hour
   window. Use existing exposed data for discovery. Only if it beats its controls and
   supports a finite edit should the fresh-role confirmation lifecycle be repaired and
   run. The current fresh-row freezer is paused at a precise NO-GO because its auditor
   still found three time-of-check/time-of-use integrity gaps; that is a publication
   blocker, not a mathematical or exploratory-data blocker.
5. **Return to joint semantic factorization only after the composition telescope says
   which interface needs it.** A shared dictionary, sparse code, tensor factorization,
   or hierarchy should be optimized jointly with the downstream reader at that
   interface and judged by CE, transfer, and selective edits—not merely local MSE.

The immediate scientific priority is item 2. It uses the tensor/linear structure to
turn a known circuit into finer exact additive pieces, and it has a cheap falsifier:
if removing the matched-source terms does not selectively damage copying, or if the
complement causes comparable copy damage, then source-position decomposition is not
the needed grain. That experiment can teach us something new without another day of
infrastructure work.

## 16. **NEW UPDATE: the exact copy-source edge was found**

The priority experiment above has now run, first on 32 documents and then on 128 after
passing its frozen escalation gates. The complete runs took 11.5 and 16.8 seconds.

Suppose the current token at position $p$ last appeared at position $j$. Copying
predicts that the next token should be what followed that earlier occurrence, namely
the already observed token at $j+1$.

For L8 heads H3 and H4, we can write the head output as an exact sum over source
positions. We deleted only the term connecting destination $p$ to source $j+1$ while
leaving every other source through those same heads intact.

On 128 documents and 1,864 natural-text copy positions:

- deleting that one exact edge costs `0.12792` nat;
- deleting the entire H3/H4 writes at the same destinations costs `0.13403` nat;
- therefore the edge accounts for **95.4%** of the matched whole-head CE damage;
- deleting the adjacent wrong edge costs `-0.00057` nat, essentially zero;
- the shared $\lambda_8v_1$ part alone costs `0.11692` nat;
- the context-refined fresh-value part alone costs only `0.00544` nat.

The intervention uses only the input to select the source. Future targets are used
only to score whether copying was correct. On repeat positions where the earlier
successor is *not* the target, deleting the edge improves CE by `0.00837` nat. On
nonrepeat positions, propagated damage is `-0.00024` nat, effectively zero.

This gives a much clearer program-level interpretation:

> L8 H3/H4 are conditional fetchers. Their copy-specific work is almost entirely one
> edge to the earlier occurrence's successor, carrying a mostly static/shared token
> identity payload through the $v_1$ bus.

This is an exact additive tensor component, so it can be removed or replaced without
deleting the heads' other source traffic. The remaining unsimplified part is the
native attention-pattern scalar that decides how strongly to use this edge. The next
test is whether a constant or one affine function of the historical weights-only match
score can replace that scalar on disjoint rows.

The detailed computation, controls, tables, caveats, and artifact names are in
`COPY_SOURCE_EDGE_DISCOVERY_FINDINGS.md`. This remains exploratory because it reused
an exposed selection role; the strict whole-model ledger does not move.

## 17. **NEW UPDATE: the payload is simple; the remaining scalar is a contextual gate**

The promised scalar-replacement test is complete.  It fit on cached documents 1--32
and evaluated on disjoint cached documents 33--128.  The run took 45.6 seconds while
another validation job shared the GPU.

At a current token position $p$, let $j$ be the nearest earlier occurrence of the
same token and let $k=j+1$ be the earlier successor position.  The exact L8 H3/H4
copy-edge write has two relevant ingredients:

1. a scalar $a_h(p,k)$ saying how strongly head $h$ uses that source here;
2. a value vector saying which token information to write.

Replacing only the value vector by the shared, nearly static $\lambda_8v_1$ token
code preserves **95.9%** of copy-positive causal CE.  Aggregate scored CE changes by
`-0.00003` nat.  Thus the context-refined payload is not the important remaining
complexity.

Replacing the scalar by one unconditional constant for each head is different.  Two
constants averaged over every repeat preserve only **27.1%** of the copy effect.  The
native scalar is near zero on many ordinary repeats and larger in magnitude when the
earlier successor is a good prediction.

Two constants fitted only from positive copy examples preserve **81.4%** on the
disjoint evaluation documents.  They are still applied to every evaluation repeat;
evaluation targets are never read by the program.  This works fairly well because it
uses appropriately strong copy coefficients, but it slightly over-copies on negative
repeats.  Older, still stronger synthetic-repeat constants improve copy-positive CE
and top-1 accuracy beyond native, but harm negative repeats enough to worsen overall
CE.

The key conclusion is:

> “What token code gets written?” is almost solved.  “When should this repeat be
> trusted as predicting its successor?” is the main remaining local computation.

That makes the next step specific.  We should reuse the older weights-computed matcher
score and test whether one affine calibration per head can serve as the contextual
gate.  A distance-binned gate is a cheap control.  We should not spend more time
factorizing the already simple payload unless these gate tests reveal a new value-side
failure.

For extraction, the aggressive two-constant program may already be useful: it makes
copying stronger and is tiny.  For faithfulness, prediction, and selective removal,
it needs the gate so that it does not copy in the wrong contexts.  This is exactly why
we keep separate success metrics instead of treating reconstruction alone as the
definition of simplicity.

Full equations, definitions, coefficients, CE/top-1 tables, gates, and caveats are in
`COPY_EDGE_CONSTANT_SCALAR_FINDINGS.md`.
