# Current project update, with the computations explained

**Updated:** 2026-08-29 16:58 UTC

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

## 18. **NEW UPDATE: obvious cheap gates are not enough**

We next tested the two simplest input-side explanations for the remaining scalar.
The experiment fit on 32 documents, evaluated on the next 96, reused hash-pinned
baseline results instead of recomputing them, and took 30.6 seconds.

The first candidate reused the older L2/L3 weights-computed repeat matcher.  In plain
language, it takes the current token and its earlier equal occurrence, runs their
embeddings through two old query/key weight pipelines, and produces one number saying
how strongly they look like a repeat.  Two fitted affine functions—multiply by one
number and add one bias for each L8 head—turn that repeat score into proposed H3/H4
copy strengths.

This is not noise: it recovers **38.7%** of the exact edge effect, versus 27.1% for
unconditional constants, and shifting the score to the wrong token position drops
recovery to 12.9%.  But it misses the 70% bar and explains only 0.7%/3.5% of the two
native scalar variances on held-out positions.  It also retains about 1.18 million
query/key slice values, so 38.7% recovery is poor value for its executable price.

The second candidate stored eight scalar values indexed by repeat distance.  It
recovers only **30.0%** and has negative held-out scalar $R^2$, meaning it predicts the
native scalar worse than a constant mean under squared error.  Distance is pruned.

The conceptual distinction is now clear:

- the old static matcher asks, “is this token a repeat?”;
- the missing L8 gate asks, “in this whole context, should the token after that repeat
  be trusted as the next prediction?”

The latter needs contextual state.  The next approach will compress the native L8
gate itself rather than invent more generic repeat features.  Each native scalar is
the product of two 128-dimensional dot products of normalized query/key vectors.  We
can search for smaller shared/canonical subspaces of those vectors and measure a rank
versus causal-CE curve.  That is a direct tensor/polynomial simplification: if rank 8
or 16 preserves the gate, we obtain a substantially smaller executable computation;
if the curve stays high-rank, we learn that this contextual decision is where the
copy circuit's genuine complexity lives.

Detailed formulas, prices, controls, all metrics, and artifacts are in
`COPY_EDGE_SIMPLE_GATE_FINDINGS.md`.

## 19. **NEW UPDATE: the contextual gate has a faithful rank-64 compression**

We stopped guessing generic features and compressed the native L8 gate directly.
Each of H3 and H4 computes its edge strength as the product of two 128-dimensional
query/key dot products.  Each query or key comes from a $128\times1152$ weight slice;
there are eight slices across the two heads.

For every slice, singular value decomposition rewrites the matrix as ordered
rank-one terms.  Keeping the first $r$ terms gives two smaller executable matrix
multiplications.  We tested ranks 8, 16, 32, 64, 96, and a rank-128 numerical control.
The factors were computed from weights alone—no activation fitting and no next-token
labels.

The causal curve is sharp:

| Rank | Native gate storage | Exact-edge causal recovery |
|---:|---:|---:|
| 8 | 6.9% | 8.8% |
| 16 | 13.9% | 32.6% |
| 32 | 27.8% | 57.6% |
| **64** | **55.6%** | **91.0%** |
| 96 | 83.3% | 94.8% |

Rank 64 is the smallest preregistered faithful point.  It explains 95.4% and 94.7%
of the two held-out native scalar variances, loses only 0.34 percentage point of
copy-position top-1 accuracy, has negligible repeat/nonrepeat collateral, and
slightly improves aggregate CE by `0.00020` nat.  The rank-128 factorized control
matches the native scalar almost exactly, so this is not a numerical artifact.

This gives us the curve we wanted: **stored factor values and multiply width versus
downstream causal faithfulness**, not merely versus weight MSE.  At this interface,
rank is now a validated notion of simplicity because reducing it produces a cheaper
program and the causal curve tells us exactly what behavior is lost.

The result is local but real.  Counting both the Q/K gate and its fixed writer, the
rank-64 program uses 950,272 stored values instead of 1,474,560, a 35.6% reduction.
It still reads the native contextual state entering L8, so it does not yet explain
how earlier MLPs and attention construct that state.

The next mathematical improvement is simultaneous factorization.  Independent SVD
stores a different 1152-dimensional input basis for every projection.  A HOSVD-style
shared input basis can be computed once and reused by all eight.  Because head RMS
normalization makes positive rescaling of each projection functionally irrelevant,
we should first divide slices by their Frobenius norms; otherwise arbitrary gauge
scales distort which directions HOSVD calls important.  This is the concrete real-
model test of the proposed norm-canonicalization-before-HOSVD idea.

Full formulas, CE/KL/top-1 tables, scalar errors, prices, and caveats are in
`COPY_EDGE_LOWRANK_QK_FINDINGS.md`.

## 20. **NEW UPDATE: shared HOSVD makes the faithful gate cheaper**

The independent rank-64 result kept a separate low-dimensional input basis for each
of eight query/key projections.  We have now stacked those weights into one tensor
and found a single input basis shared by all eight.

At shared rank 256, the program first converts the 1152-dimensional L8 contextual
state into one 256-dimensional vector.  All H3/H4 query and key computations reuse
that vector through different small output cores.  This is both cheaper and a cleaner
interface for composition.

The selected program:

- preserves **92.25%** of the exact copy-edge causal effect;
- explains 97.8% and 96.2% of the two held-out native scalar variances;
- changes aggregate CE by `-0.00033` nat and has negligible negative/nonrepeat damage;
- uses 557,056 gate values instead of 655,360 for independent rank 64;
- including the writer, uses 851,968 values instead of 1,474,560 native—a **42.2%**
  reduction for the localized copy program.

The norm-canonicalization-before-HOSVD suggestion was tested directly.  Because each
query/key vector is RMS-normalized, multiplying a whole projection slice by a positive
constant should not change its function.  We therefore normalized slice Frobenius
norms when choosing the shared basis, then restored original scales in the executable
cores.

Canonicalization consistently helps, but only modestly at the selected rank: 92.25%
recovery versus 91.32% raw, a +0.93-point gain.  That fails the preregistered +2-point
bar.  The slice norms were already similar, so this is unsurprising.  The method is
valid; it is not the main discovery here.  The major gain comes from **sharing one
input basis across all eight projections**.

This creates a useful new handle on the rest of the model.  The copy edge does not
need the entire L8 residual stream; it needs the 256-dimensional state
$z=V_{256}^\top x^{(8)}$.  We can use that downstream-defined state as the target of
the MLP0/MLP1/MLP2/attention composition telescope.  Earlier components should be
judged by whether they construct the causally validated $z$ and preserve copy/whole-
model behavior, rather than whether they reconstruct all 1152 residual coordinates.

Full curves, equations, price accounting, canonicalization comparison, and caveats
are in `COPY_EDGE_SHARED_HOSVD_FINDINGS.md`.

## 21. **NEW UPDATE: the MLP0 compression and copy-gate compression compose**

The new 256-dimensional copy state gave us a concrete test of the older C512 MLP0
program.  C512 keeps MLP0's exact input normalization, Left/Right maps, and
coordinatewise products, but replaces its large `Down` matrix by a rank-512 program
that is 72% smaller.

We crossed native/C512 MLP0 with native/shared-HOSVD L8 copy gates on 96 disjoint
cached documents.  All six preregistered gates pass.

C512 preserves the L8 copy state $z$ extremely well:

- $R^2=0.9955$ on all scored positions;
- cosine `0.9985`;
- it removes **99.63%** of the $z$ error caused by deleting MLP0.

Here **deleting MLP0** means replacing MLP0's complete residual-stream write by the
zero vector at every token position.  It deletes the variable bilinear contribution
and MLP0's learned `Down` bias.  The residual bypass, embedding stream, attention0,
and every later component remain live.  This is **not** replacement by the mean
MLP0 write, an optimal constant, or a fitted bias.  It is intentionally a severe
causal scale control.  A mean/optimal-constant replacement would probably be less
damaging, but it was not the denominator used here.

More precisely, if $z_N$ is the native downstream copy state, $z_C$ is the state
under C512, and $z_Z$ is the state when the MLP0 write is zeroed, the reported
quantity is

$$
1-\frac{\lVert z_C-z_N\rVert_2^2}{\lVert z_Z-z_N\rVert_2^2}=99.63\%.
$$

Thus 99.63% refers to recovery of the **squared-error gap in the 256-dimensional
state $z$**.  It is not a claim that C512 recovers exactly 99.63% of the CE damage.
The CE measurements are reported separately below.

This last comparison shows the score is not vacuous: deleting MLP0 raises aggregate
CE by `2.591` nat and copy CE by `2.801` nat.  MLP0 matters enormously, and C512
preserves almost all of what this downstream copy consumer needs.

Behaviorally:

- C512 alone changes aggregate CE by `+0.00220` nat and actually improves copy CE by
  `0.00208` nat;
- the HOSVD gate alone changes aggregate CE by `-0.00020` and copy CE by `+0.01054`;
- both together change aggregate CE by only `+0.00264` and copy CE by `+0.00906`;
- the joint copy top-1 loss is 0.48 percentage point;
- the non-additive interaction is just `+0.00064` nat aggregate and `+0.00061` on
  copy positions.

So these are not merely two individually good approximations.  They work on the same
changed forward trajectory and remain good together.  This is the composability test
we wanted from a simplicity definition.

The next upstream target is MLP1/MLP2.  Prior work found that C512 changes the induced
MLP1 write and that deployed MLP2 attenuates much of the discrepancy.  We should now
judge MLP1/MLP2 candidates by whether they preserve the validated $z$ state, the copy
edge, and whole-model CE under the already composed C512+HOSVD background.  There is
no reason to return to more MLP0 token clustering unless that telescope identifies a
specific missing producer coordinate.

Full computations, definitions, factorial, state metrics, CE/KL/top-1 tables, and
caveats are in `C512_COPY_GATE_COMPOSITION_FINDINGS.md`.

## 22. **NEW UPDATE: the MLP2 downstream-aware selector is fitted, but the finite compression has not yet been tested**

This is the newest result.  It is important to separate two statements that are easy
to blur together:

1. we have now chosen six precise, equal-price candidates for a much smaller MLP2;
2. we have **not yet run those candidates through the complete model and measured
   their loss**.

So this update is evidence about how to choose an MLP2 program.  It is not yet
evidence that the program works.

### 22.1 What is the MLP2 map?

At each token position, the input $x$ is the 1,152-dimensional normalized residual
stream presented to MLP2.  MLP2 computes two linear maps,

$$
u=Lx,\qquad v=Rx,
$$

where $u$ and $v$ each contain 4,608 numbers.  It multiplies matching coordinates,

$$
a_j(x)=u_jv_j=(L_jx)(R_jx),\qquad j=1,\ldots,4608,
$$

and maps the 4,608 products back to a 1,152-dimensional residual-stream write,

$$
y(x)=b+Da(x)=b+\sum_{j=1}^{4608}D_{:j}a_j(x).
$$

Thus one native **product channel** $j$ consists of one row $L_j$, one row $R_j$,
their scalar multiplication, and one output direction $D_{:j}$.  The proposed
program retains only 512 of these 4,608 channels.

For every omitted channel, it keeps that channel's average product
$\mu_j=\mathbb E[a_j]$.  Because an average is constant, all omitted average writes
can be added once to the ordinary bias:

$$
b'=b+\sum_{j\notin K}D_{:j}\mu_j.
$$

The resulting executable is

$$
y_K(x)=b'+\sum_{j\in K}D_{:j}(L_jx)(R_jx),
$$

where $K$ contains exactly 512 channels.  It physically needs only 512 rows of
$L$, 512 rows of $R$, 512 columns of $D$, and one 1,152-number bias.  Its price is

$$
512(1152+1152+1152)+1152=1{,}770{,}624
$$

stored scalar values, which is 11.12% of native MLP2's fixed-grammar value count.
Equivalently, it would remove 88.88% of this local MLP2 price **if** the complete
model validation succeeds.  A rank or value count is useful here because it buys a
literal reduction in stored parameters and executed bilinear products; it is not
being treated as semantic understanding by itself.

### 22.2 How were the 512 channels chosen?

We fitted several selectors at the same size so the downstream-aware method cannot
win merely by spending more.

- **LOCAL** measures each centered product's immediate output-write energy:

  $$
  \operatorname{Var}(a_j)\lVert D_{:j}\rVert_2^2.
  $$

  It asks, “If this channel varies a lot and writes in a large direction, is it
  locally important?”

- **RMS** uses the uncentered second moment:

  $$
  \mathbb E[a_j^2]\lVert D_{:j}\rVert_2^2.
  $$

  It differs from LOCAL by including the constant mean component.

- **MASS** uses weights only:

  $$
  \lVert L_j\rVert_2^2\lVert R_j\rVert_2^2\lVert D_{:j}\rVert_2^2.
  $$

  It does not ask how often natural text activates the channel.

- **HASH_RANDOM** chooses 512 channels by a content-hash ordering.  It is an
  equal-price null control.

- **DERANGED** deliberately mismatches product channels and output directions in a
  gauge-invariant way, then applies the same downstream selection machinery.  It
  tests whether the real product-to-output pairing matters.

- **SUFFIX** is the downstream-aware selector.  “Suffix” means every computation
  after the MLP2 write: later residual additions, RMSNorms, attention blocks, MLPs,
  final normalization, and logits.

The SUFFIX computation introduces one scale $\alpha_j$ for each centered product,
shared across every token position in one document:

$$
y_q(\alpha)=b+Da_q+D\big[(a_q-\mu)\odot(\alpha-1)\big].
$$

At $\alpha=1$, this is exactly native MLP2.  We differentiate a downstream
log-probability score $s$ with respect to each $\alpha_j$:

$$
E_j
=\frac{\partial s}{\partial\alpha_j}
=\sum_q(a_{qj}-\mu_j)D_{:j}^{\mathsf T}
  \frac{\partial s}{\partial y_q}.
$$

The sum over token positions $q$ is essential.  Deleting a channel changes it at
every position, and later attention can move one position's change to another
position.  A local gradient at only the scored position would miss that route.

We used 191 fresh FineWeb documents and eight independent categorical-Fisher probes.
A probe samples output categories from the native model and differentiates their log
probabilities.  This samples sensitivity across the model's predicted distribution
instead of privileging one hand-selected token.  The resulting response tensor had
shape

$$
191\ \text{documents}\times8\ \text{probes}\times4608\ \text{channels}.
$$

Each document was normalized to equal total response energy so a few high-gradient
documents could not dominate.  We then flattened the first two axes, performed a
float64 singular-value decomposition, and computed rank-256 **ridge-leverage
scores**.  In plain language, a leverage score is high when a channel supplies a
direction in downstream-response space that the other channels do not already span.
The 512 highest-scoring native channels form SUFFIX512.  Rank 256 is an analysis
scale used to estimate this shared response geometry; the executable still contains
512 ordinary native bilinear channels.

### 22.3 What was actually measured, and how long did it take?

The mean/control stage used 192 fresh documents and 30,801 eligible token positions.
It made 48 four-document forwards, stopping immediately after MLP2, and took
**10.54 seconds**.

The downstream SUFFIX stage used 191 live documents, eight probes, 48 complete-model
forwards, and 384 backwards.  It took **58.99 seconds**.  The two substantive model
computations therefore took about **69.52 seconds total**.  The later overlap repair
was a source-closed CPU summary correction, not another model run.

The first SUFFIX implementation attempt stopped before producing logits, targets, or
responses because its fused matrix multiplication plus bias was not bit-exact with
the native model's separate operations.  The corrected run used the native operation
order and obtained a bit-exact baseline.  Separately, the first published overlap
summary converted scalar tensors into Python sets incorrectly and printed zero
overlaps.  The stored support tensors and hashes were correct; converting the tensor
indices to ordinary integers repaired only the summary.  These failures are
preserved because they affect how much confidence to place in the pipeline, even
though neither changed the selected supports.

### 22.4 What worked?

The SUFFIX selector is statistically stable on its fit data:

- the channel-score ordering from two independent four-probe halves has Spearman
  correlation `0.94909`;
- their independently selected top-512 sets have Jaccard overlap `0.78397`;
- reciprocal channel rescalings and channel permutations replay the same physical
  selection, so the selector is not exploiting an arbitrary gauge convention.

Here **Spearman correlation** compares two complete rankings: 1 means the rankings
are identical and 0 means no rank association.  **Jaccard overlap** for two sets
$A,B$ is

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

The corrected equal-size support comparisons are:

| Comparison | Shared channels | Jaccard overlap |
|---|---:|---:|
| SUFFIX versus LOCAL | 369 | `0.56336` |
| SUFFIX versus RMS | 364 | `0.55152` |
| SUFFIX versus MASS | 308 | `0.43017` |
| SUFFIX versus DERANGED | 87 | `0.09285` |
| SUFFIX versus HASH_RANDOM | 56 | `0.05785` |

So SUFFIX is not random and not merely a weight-magnitude ranking.  But it is also
not a radically new decomposition: it shares 369 of its 512 channels with LOCAL and
exchanges 143.

Measured in SUFFIX's own ridge-leverage currency, the retained supports capture:

| 512-channel support | Fraction of total SUFFIX score captured |
|---|---:|
| SUFFIX | **25.534%** |
| LOCAL | 23.963% |
| RMS | 23.867% |
| MASS | 22.399% |
| DERANGED | 12.589% |
| HASH_RANDOM | 11.581% |

SUFFIX therefore improves its own fit objective over LOCAL by only about 6.6%
relative.  That is a real, split-stable difference, but it is modest.  It does not
tell us that SUFFIX will have 6.6% lower KL after deleting 4,096 channels at once.
Gradients describe infinitesimal changes around the native model; the proposed
compression is a large finite change, and omitted channels can cancel, interact, or
be amplified by later nonlinearities.

### 22.5 What is the current conclusion and next computation?

The honest conclusion is:

> We now have a stable downstream-aware hypothesis for which 512 native MLP2
> products matter, but no demonstrated 512-product MLP2 program yet.

The decisive next run physically builds all six 512-channel candidates and sends
each changed trajectory through layers 3--17.  It will compare them on:

- **cross-entropy (CE):** probability assigned to the true next token;
- **teacher KL:** the complete candidate output distribution versus the native
  model's distribution;
- **centered-logit NRMSE:** distortion of relative logits after removing the
  irrelevant common logit offset;
- **top-1 agreement:** how often candidate and native model choose the same token;
- copy-positive, repeat-negative, nonrepeat, token-frequency, and per-document cells;
- a margin certificate relating measured logit distortion to top-1 stability;
- small signed edits, checking whether the finite change follows the fitted tangent
  direction rather than reversing it.

The frozen absolute bars include $|\Delta\mathrm{CE}|\leq0.02$, teacher
$\mathrm{KL}\leq0.02$, centered-logit NRMSE $\leq0.10$, top-1 agreement at least
0.90, and no registered cell above 0.02 nat collateral damage.  SUFFIX must also
beat every equal-price control by at least 5% teacher KL with a document-bootstrap
simultaneous lower confidence bound.  Only then would the experiment justify saying
that MLP2 has been compressed at this width.

Before opening the protected validation role, one fit-only calibration still has to
freeze the native-margin epsilon grid and token-frequency strata.  This is a missing
preregistered measurement, not a data or GPU blocker; the RTX 5090 is currently
available.  After that, the finite validator is the critical path.  If every K=512
candidate fails the absolute bars, the useful negative result is that selection of
native products is the wrong grain at this width, and the next move is a shared or
response-conditioned factor basis rather than many nearby K sweeps.

The focused technical receipt is `MLP2_CMR_V1_SUFFIX_FINDINGS.md`.  This section is
the plain-English continuation; the complete current explanation remains this file:
`CURRENT_PROJECT_UPDATE_PLAIN_ENGLISH_2026-08-29_1410.md`.

## UPDATE START — 23. What happened after the MLP2 selector run?

The short answer is: **there is not yet a new MLP2 compression outcome after
Section 22**.  The last scientific model result is still the 58.99-second SUFFIX
selector fit described above.  Since then, the work has been preparing the one
small calibration that must precede the decisive finite K=512 experiment, and
repairing weaknesses found by an independent audit of that calibration code.

This distinction matters.  Writing a runner, projecting a data file, or passing a
test does not show that a compressed MLP works.  The scientific question remains:
after physically replacing native MLP2 by 512 selected product channels, what
happens to the complete model's output distribution and cross-entropy?

### 23.1 What actually ran, and how long did it take?

After the two model computations in Section 22, the only newly executed data job was
a **model-free role projection**.  It copied the already frozen FIT_SELECTOR rows
out of a combined token container into a separate file.  It contained:

- 192 document rows;
- 31,505 eligible next-token positions;
- 191 documents with at least one eligible position;
- exactly one short document with no eligible position.

The observed shell wall time was about **2.7 seconds**.  This time is not stored as a
scientific model-result field, so it should be understood as operational timing,
not a receipted benchmark.  The operation loaded no model, made no forward or
backward calls, and revealed no validation or replication outcome.

The calibration itself has **not run**, and the K=512 finite validator has **not
run**.  Therefore there is no new CE, KL, logit-error, top-1, extraction, removal, or
OOD number to report.  Code review and hardening occupied the remaining interval;
there is no honest single runtime for that human/agent reasoning work.

For reference, the complete substantive MLP2 model timing remains:

| Computation | Model work | Wall time |
|---|---:|---:|
| FIT_MEAN | 48 prefix-only forwards | 10.54 s |
| SUFFIX selector | 48 full forwards + 384 backwards | 58.99 s |
| Role-only projection | no model calls | about 2.7 s |
| Margin/frequency calibration | not run | — |
| Physical K=512 validation | not run | — |

### 23.2 What is a “role,” and why make a separate FIT_SELECTOR file?

The documents were preregistered into disjoint **roles**:

- FIT_MEAN estimates the average product values $\mu_j$;
- FIT_SELECTOR chooses the 512 product channels;
- VALIDATION tests the resulting choices once;
- REPLICATION checks a surviving claim on another held-out sample.

A role is therefore a permission boundary, not a semantic token category.  A
fitting program is allowed to inspect fitting rows but must not deserialize held-out
validation rows, even if it promises not to use them.  The original token file
contained all four roles in one serialized object.  The independent audit correctly
identified that loading that object would technically expose validation and
replication data to calibration code.  The 2.7-second projection created a new
serialized object containing only FIT_SELECTOR.  Its tensor hashes agree exactly
with the earlier frozen rows.

This repair does not improve the model.  It makes the next comparison trustworthy:
the thresholds and reporting cells cannot accidentally adapt to validation data.

### 23.3 What is the missing calibration computing?

It computes only reference quantities from the native model on FIT_SELECTOR.  It
does not choose SUFFIX channels and cannot change the K=512 candidates.

#### Native top-1 margin

Let $z_i\in\mathbb R^{50304}$ be the native final-logit vector at eligible token
position $i$.  If $z_{i,(1)}$ and $z_{i,(2)}$ are its largest and second-largest
entries, the **native margin** is

$$
m_i=z_{i,(1)}-z_{i,(2)}.
$$

A large $m_i$ means the winning token is far ahead of the runner-up.  A small
$m_i$ means a modest logit perturbation can change top-1.  The calibration freezes
an epsilon grid consisting of fixed dyadic values $2^k$, for $k=-10,\ldots,5$,
plus one half of several empirical margin quantiles.  The dyadic values prevent the
data from choosing the entire grid; the quantile values give resolution where this
particular model's decisions actually lie.

In validation, if the candidate's maximum relevant logit disturbance is below a
position's native margin, the native winner is certified unchanged at that
position.  Sweeping the frozen epsilon grid gives a margin-based top-1 stability
curve rather than relying on one convenient threshold selected after seeing the
candidate.

#### Target-frequency cells

For target token type $v$, the calibration counts

$$
c(v)=\sum_i\mathbf 1\{t_i=v\},
$$

where $t_i$ is the true next token at eligible FIT_SELECTOR position $i$.  Positions
are placed into the predeclared count bins

$$
0,\ 1,\ 2\text{--}3,\ 4\text{--}7,\ 8\text{--}15,\ 16\text{--}31,
\ 32\text{--}63,\ 64\text{--}127,\ \ge128.
$$

These are **reporting cells**: subsets on which the same loss metrics are recomputed.
They test whether a candidate looks good only because it preserves common targets
while damaging rare ones.  The bins do not train the selector.

#### Copy and repeat cells

At each eligible position, we look back at most 128 tokens for the nearest earlier
occurrence of the current input token.  If that earlier occurrence was followed by
the current target, the position is **copy-positive**: an induction-like copy rule
could predict it.  If the input repeats but its earlier successor differs from the
current target, it is **repeat-negative**.  If no such earlier input occurs, it is
**nonrepeat**.

Again, these cells do not make the compression.  They reveal whether MLP2 removal
selectively breaks copy-related behavior, confuses arbitrary repetition with useful
copying, or mainly affects ordinary nonrepeat text.

### 23.4 What are $D_2$, teacher KL, and the other validation computations?

For native logits $z_i$ and candidate logits $\hat z_i$, the frozen squared-logit
distortion is

$$
D_2=\frac1N\sum_{i=1}^{N}\sum_{v=1}^{50304}
       (\hat z_{iv}-z_{iv})^2.
$$

The vocabulary dimension is **summed**, not averaged, and the logits are not
centered for this particular certificate.  This exact normalization was missing
from the first draft and is now frozen before validation.

The other whole-model measures answer different questions:

- **CE change** asks how much probability the candidate loses on the true next
  tokens.  It is closest to the model's training objective.
- **Teacher KL** compares the candidate's complete next-token distribution with the
  native model's distribution, including alternatives that are not the true token.
- **Centered-logit NRMSE** measures relative-logit distortion after subtracting the
  irrelevant common offset from every vocabulary logit.
- **Top-1 agreement** asks how often candidate and native select the same winning
  token.  It can remain high while probabilities are badly distorted, which is why
  it is not used alone.
- **Per-document bootstrap bounds** resample whole documents, not individual tokens,
  to test whether SUFFIX's advantage over every equal-price control is robust to
  which documents were sampled.

The finite validator will compare SUFFIX with LOCAL, RMS, MASS, DERANGED, and
HASH_RANDOM at exactly the same 512-channel storage and execution price.  It also
includes the native model and a zero-MLP2 baseline.  This is the first computation
that can convert the selector evidence into a compression claim.

### 23.5 Why did the calibration code need more work?

An independent audit returned **NO-GO** on the first version.  The important issues
were concrete rather than mathematical disagreement:

1. the calibration could deserialize the combined four-role token object;
2. the model-collection function needed a one-use authority token rather than being
   callable as an ordinary public helper;
3. CPU fallback was allowed even though the numerical contract requires the RTX
   5090 in bfloat16;
4. the per-site forward-call ledger and exact row semantics needed stronger checks;
5. crash/receipt publication needed stronger filesystem and hash joins;
6. the $D_2$ normalization had not been stated exactly.

The role-only projection fixes item 1.  The addendum freezes item 6.  The calibration
runner is being repaired for items 2--5 and must pass a second audit before it is
allowed to touch the model.  This is a real quality-control gain, but it also means
the interval produced less scientific information than intended.

### 23.6 Is anything externally blocking progress?

No.  FineWeb rows are cached, the model artifacts are present, and this calibration
does not need new data or user permission.  The blocker is internal and precise:
finish the source-closed calibration runner, pass its second audit, execute its 48
native forwards, and then run the physical validator.

The immediate order is therefore:

1. finish and re-audit the FIT_SELECTOR-only calibration;
2. execute it and publish the frozen margin grid, frequency counts, and copy-cell
   identities;
3. build and test the physical 512-channel MLP2 path, verifying it makes zero native
   MLP2 calls;
4. execute the six equal-price candidates plus native/zero baselines on validation;
5. promote only a candidate that passes both the absolute faithfulness bars and the
   document-bootstrap comparison with all controls.

The key scientific fork remains simple.  A finite SUFFIX win would show that a
downstream-aware native-product decomposition buys a smaller executable MLP2.  A
SUFFIX/LOCAL tie would say the expensive downstream geometry added little.  Failure
of every K=512 arm would say native product coordinates are the wrong atoms, sending
us toward jointly refactored or response-conditioned bases rather than more selector
tuning.

## UPDATE END — 23

## UPDATE START — 24. The fit-only calibration is now complete

Section 23 described the calibration as the immediate blocker.  That blocker is now
closed.  After three independent audit rounds and 18 focused CPU tests, the exact
committed runner executed 48 native model forwards on the RTX 5090.  It took 16.13
seconds wall time and published a receipt; no failure artifact exists.

The result freezes 28 margin thresholds, the complete FIT_SELECTOR target-frequency
reference, and the copy/repeat reporting definitions.  Among 31,505 eligible
positions, 2,420 are copy-positive, 11,224 are repeat-negative, and 17,861 are
nonrepeat.  The native top-1 margin has mean 1.58057, median 0.9375, and 90th
percentile 3.9375.  Its 2% quantile is zero at bfloat16 resolution, so the eventual
margin certificate may cover fewer positions than the direct CE/KL metrics do.

This is still not a compression result: validation and replication remain unopened,
and no 512-channel candidate has executed.  What changed is that the physical
validator can now be implemented against frozen thresholds and cells without any
post-outcome choices.  The next scientific computation remains the equal-price
SUFFIX/LOCAL/RMS/MASS/DERANGED/HASH_RANDOM comparison on actual final logits.

Technical details and hashes are in `MLP2_CMR_V1_CALIBRATION_FINDINGS.md`; the
strategic ranking is in `HOURLY_STRATEGIC_REVIEW_2026-08-29_1735.md`.

## UPDATE END — 24

## UPDATE START — 25. Validation is unblocked, and why more late circuits help

### 25.1 The data-size default

The recent `10.54`-second FIT_MEAN run and `16.13`-second calibration run both used
192 source-document-disjoint examples.  The extra time was calibration work, not a
32-versus-192 document comparison.  Nevertheless, the practical conclusion is
correct: a few extra seconds is too small a price for making 32 documents the
scientific default.

From here, 32 or fewer documents are for code debugging only.  Consequence tests
default to 192 documents and report the nested 48/96/192 prefixes, so we directly see
whether a conclusion survives two data doublings.  A passing or boundary-close result
must then use the separately sealed 192-document replication role.  This does not
make 192 magically sufficient for every rare cell; cell counts and document-level
uncertainty still determine whether a narrow claim is supported.

### 25.2 The validation input and executable are now ready

The model-free VALIDATION projection passed independent audit and completed in 2.46
seconds.  It published only the VALIDATION role: 192 documents, 29,904 scored
positions, 191 documents with nonempty scored support, and one short all-false row.
No model, logits, targets-as-outcomes, candidate result, or REPLICATION data was
opened.

The physical MLP2 program is also implemented.  For a retained set $K$, it stores
only `Left[K]`, `Right[K]`, `Down[:,K]`, and the folded bias

$$
b_K=b+D_{:S}\mu_S.
$$

Its forward pass constructs exactly 512 products rather than constructing all 4,608
and masking them afterward.  The production price test gives exactly 1,770,624
stored scalars and zero calls to the native MLP2 implementation.  The remaining work
before opening finite outcomes is the audited streaming measurement wrapper.

### 25.3 Would more downstream circuits help explain earlier MLPs?

Yes.  The final-logit distribution is the broadest downstream test, but by itself it
does not name the information carried by an early component.  A collection of
causally verified late circuits can serve as a bank of interpretable consumers of
MLP0/1/2.

Mathematically, let $z$ be an early residual state and let

$$
\Phi(z)=\big(f_{\rm copy}(z),f_{\rm capitalization}(z),
f_{\rm numeric}(z),f_{\rm syntax}(z),\ldots\big)
$$

collect the responses of several downstream circuits.  Two early states are
equivalent for this consumer bank when they give the same $\Phi(z)$.  A useful early
MLP decomposition should then make the response tensor

$$
E_{c,j}=\text{effect of early component }j\text{ on consumer }c
$$

sparse or low-complexity across consumers.  Simultaneously factoring this response
tensor can expose:

- components shared by many circuits;
- components specific to copy, capitalization, numbers, or syntax;
- directions no verified consumer reads, which become candidates for removal;
- circuit-specific components that can be extracted or edited with less collateral
  damage.

Capitalization alone is probably too narrow and may reduce mostly to a direct
unembedding/logit direction.  The high-value experiment is a complementary bank:
copy/induction, capitalization, punctuation or syntactic closure, numeric succession,
and entity continuation are plausible starting points.  Each circuit must first have
a localized causal effect and a selective-removal test; otherwise its label is merely
correlational.

This is a semantic follow-on to the current MLP2 finite validation, not a reason to
interrupt it.  The MLP2 test asks whether downstream-aware selection yields a faithful
small executable at all.  Once that is answered, the named-consumer bank is the
highest-return way to attach meaning and editability to retained early components.

## UPDATE END — 25
