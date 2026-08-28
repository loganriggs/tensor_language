# Current best understanding of bilin18, in plain English

Date: 2026-08-28

Latest checkpoint: 14:15 UTC. The MLP1 measurement collector described below is now
implemented and passes its 52-test focused CPU suite. An independent first audit found
eight launch-integrity defects; those defects have been repaired and a second exact-byte
audit is in progress. No MLP1 scientific result has been opened yet.

This is the non-audit version of the project update. It explains what we actually
know, what is still only a hypothesis, why some results seem contradictory, and what
we are doing next. It deliberately does not explain the basic transformer or tensor
architecture.

## The short version

We can reproduce the entire model with our own executable code, but that is mostly
structural ownership, not reverse engineering in the stronger sense.

We have one real whole-model simplification: four related attention projections can
share a 640-dimensional continuous representation. The resulting complete program is
5.35% smaller, remains very close in ordinary prediction, and reproduces most tested
causal changes. It is not exact: it changes the model's top predicted token on about
4% of tested positions.

The remaining size is dominated by the MLPs. MLP0 has clear shared lexical structure
and strong continuous low-dimensional approximations, but no tested MLP0 replacement
has yet survived all downstream composition tests. The main reason is that MLP0,
MLP1, and MLP2 compensate for one another. A locally accurate replacement can move
the state just enough that the next MLP behaves differently.

Our current target is therefore MLP1. We are no longer trying to discover another
arbitrary low-rank coordinate system. We are testing whether a small subset of MLP1's
actual 4,608 multiplication gates accounts for its downstream effects. If it does,
those gates are directly runnable and removable. If it does not, we will switch to a
new factorization of the underlying quadratic function rather than keep searching for
a sparse subset of the checkpoint's gates.

There is no current data, checkpoint, cache, or `rspd` blocker. The GPU is temporarily
occupied by a separate registered run, so we are using that interval for the final
source audit. The source-closed MLP1 measurement program is implemented. The scientific
blocker is unchanged: we do not yet know whether a small, stable set of physical gates
exists, because the protected production measurement has not run.

## What we know with the most confidence

### 1. The complete model is structurally owned, but not semantically explained

All 36 attention/MLP sites have exact executable formulas, and the standalone program
contains all 545,904,054 required values. This means we are not missing a hidden model
operation. It does **not** mean we know the shortest program, the important circuits,
or the meaning of every internal variable.

There is no honest single “percent understood.” The useful current numbers are:

| question | current answer |
|---|---:|
| Can we execute every model operation ourselves? | yes, 36/36 sites |
| How much storage has passed predictive and causal simplification tests? | 5.35% |
| How much behavior has a human-style semantic label in the older semantic ledger? | 32.1% $\pm$ 6.4% |
| How much of the strict named causal ledger is recovered? | 10.923% |
| How much of the model's storage is in the still-dense MLP banks? | 52.51% |

The 32.1% and 10.923% figures use different tests and denominators. They should not be
averaged or treated as competing estimates of one hidden “understanding percentage.”

### 2. The attention simplification is a real but limited success

The complete rank-640 program stores 516,707,766 values instead of 545,904,054. It
removes 29,196,288 values.

On two held-out prediction roles, its extra cross-entropy was only 0.005532 and
0.004449 nat per target. On a fixed causal intervention bank, it recovered 94.442% of
the squared model change on average, with a 92.726% lower confidence bound. Its change
direction had cosine similarity 0.97238 with the original model.

However, its top-token agreement was 95.782% and 96.077%, below our predeclared 98%
bar. Its actual task accuracy was nearly unchanged, but the exact token chosen often
enough to matter was not. The right label is therefore:

> useful, causally tested storage compression; not an exact behavioral clone and not
> yet a semantic explanation of its 640 continuous coordinates.

This is our clearest example of a simplicity measure doing useful work. Smaller stored
size predicted smaller artifacts, and the candidate also retained prediction and
causal response. It did **not** automatically buy exact decision identity or named
concepts.

### 3. MLP0 has shared lexical structure plus continuous refinement

MLP0 is not simply clustering every number, punctuation token, or word form into one
identical output. The best descriptive decomposition is:

$$
\text{MLP0 write}
=\text{shared lexical part}
+\text{token-specific part}
+\text{context-specific part}
+\text{remaining error}.
$$

The shared lexical part is real: meaningful token group assignments beat controls in
which the same group representatives were assigned to the wrong tokens. But tokens in
one group remain distinguishable through the other continuous terms. Thus “numbers
cluster” and “later computation can distinguish different numbers” are compatible.

This lexical decomposition is fitted from observed writes:

- first estimate the average MLP0 write for each token on fit documents;
- average related token codes to get a group component;
- retain each token's difference from its group mean;
- fit a continuous context term to what token identity still misses;
- evaluate every part on different documents.

It is a useful description, not a unique ontology. Different continuous bases can
move information between terms, and the group-based programs did not beat matched-size
continuous maps on the strongest downstream tests. So the existence of lexical
organization has **not** yet established that lexical clusters are the simplest
executable explanation.

Two other MLP0 results matter:

- A rank-64 output subspace contains only about 37% of MLP0's residual energy but
  recovers about 79.9% of its measured causal effect. High variance is therefore not
  the same as downstream importance.
- C512 keeps MLP0's native multiplication features but replaces its final mixing map
  with a rank-512 continuous map. That final map is about 3.6 times smaller and gives
  small ordinary FineWeb output errors. But it has not passed as an independent causal
  replacement.

### 4. MLP0, MLP1, and MLP2 behave like a coupled program

This is the central reason local reconstruction has not turned into a full reverse
engineering result.

When C512 changes MLP0, most of the internal mismatch later appears in MLP1's write.
If the exact MLP1 write is transplanted, most of the exposed error disappears. Yet
with the normal MLP2 present, the final model already suppresses much of that MLP1
write error. Removing MLP2 makes the same mismatch roughly four to five times more
visible under the registered error scale.

The follow-up did **not** find evidence that MLP2 emits one specially aligned “repair
vector.” A shuffled write control was just as good. The most plausible current picture
is distributed damping or compensation through MLP2 and the later suffix, not a neat
single repair circuit.

So C512 can look very accurate at the final output while being internally wrong in a
way that later computation happens to absorb. This is why local mean-squared error and
ordinary final-output error are not sufficient simplicity criteria.

### 5. MLP1 looks moderately low-dimensional, but its apparent directions do not repeat

We measured how small edits after MLP1 affect the whole future output distribution.
In each document, about 10--17 directions contained 95% of the measured response
energy. At first glance, that sounds like a clean small code.

It was not stable. Two independent probe sets applied to the **same document** found
subspaces with average distance 0.562 at rank 16, far above the preregistered 0.15
ceiling. There was also no sharp gap in the spectrum that selected a defensible rank.

This does not prove MLP1 is intrinsically high-dimensional. It says the current
finite-probe response has a smooth spectrum rather than a repeatable set of 16 axes.
More probes might estimate that spectrum more accurately, but merely fitting more
local axes would not directly yield an executable replacement.

This negative result is why the current plan changed from “find a small local basis”
to “test the model's actual multiplication gates.”

## The confusing or weird results

### Low-dimensional energy, unstable directions

A matrix can have most of its energy near a 16-dimensional subspace without having a
stable top-16 cutoff. If singular values 12 through 25 decline smoothly, small sampling
changes can rotate which 16 axes are selected. That appears to be the MLP1 situation.
The energy concentration is real; the claimed discrete 16-dimensional code was not.

### Good final behavior, wrong internal interface

C512 has small FineWeb final-output error, but its changed state makes MLP1 write a
different vector. MLP2 and later computation suppress much of the damage. Therefore
“the final answer is close” does not mean “the replacement has preserved the same
causal interface.” This is the clearest evidence that the early MLPs must be simplified
jointly or against downstream consequences.

### Small cross-entropy, changed top token

Rank 640 changes probabilities only slightly on average, yet changes the winning token
on about 4% of positions. This happens when the top two tokens were already close. It
means cross-entropy is a useful but incomplete success metric.

### A numerically valid MLP2 run remained formally inconclusive

One MLP2 compensation run computed coverage once in float32 and once from exact integer
counts. They differed by roughly $10^{-8}$, while the frozen equality gate required
$10^{-12}$. The model measurements and bootstrap ledger were intact, but the registered
promotion label had to remain inconclusive. We preserved this failure rather than
changing the rule after seeing the outcome. The descriptive compensation pattern is
useful; it is not a promoted causal certificate.

### Several recent hours produced experimental design rather than a new model number

This is a fair criticism of the apparent pace. The earlier 256-backward MLP1 run took
only 92.94 seconds: 67.41 seconds rebuilt and verified the complete program, and 24.75
seconds did the gradients and analysis. The new experiment should also take minutes,
not hours, once launched.

Most recent time went into preventing a fast but invalid result. Before any new model
outcome was opened, audits caught:

- a response tensor whose context and probe axes were reversed;
- a supposed held-out reconstruction that accidentally refit on the held-out data and
  became nearly vacuous at large support;
- a negative control whose answer changed under harmless rescaling of equivalent MLP
  factors;
- an extreme bootstrap confidence tail supported by only about two simulated draws;
- ambiguity about whether fresh rows could be used to fit analysis coefficients.

All five have been repaired. This work increased trustworthiness, but it did not add a
new fact about MLP1. The next priority is now to stop extending the protocol and run it
once the collector is source-closed.

## How the simplicity measures have affected the work

We no longer use one scalar called “simplicity.” Each measure makes a different
promise, and we keep it only if it predicts that promised benefit.

| proposed simplicity | what it should buy | what happened so far |
|---|---|---|
| fewer stored values | a genuinely smaller standalone artifact | validated for rank-640 attention, with a 5.35% whole-model reduction |
| fewer multiplication gates | lower MLP execution cost and a directly removable support | current MLP1 experiment; no result yet |
| low response rank | few downstream-relevant degrees of freedom | energy was concentrated, but the axes failed repeatability; not promoted |
| lexical classes | human-readable shared structure and possibly cheap lookup | real organization, but worse than matched-size continuous maps; descriptive only |
| low local reconstruction error | accurate local state/write | repeatedly failed to predict composition; demoted to a diagnostic |
| gauge-independent dimension | complexity that does not change under arbitrary internal rescaling or basis choices | directly forced the new MLP1 control to be canonicalized before use |
| sparse dependency graph | predictable composition and selective editing | not yet validated; planned only after a gate support survives |
| description length / MDL | better data efficiency and shorter learned descriptions | deliberately deferred until we have comparable runnable candidates |

The central rule is:

> A definition of simplicity earns trust only when being “simpler” predicts something
> useful on untouched tests—smaller execution, better data efficiency, more reliable
> composition, easier extraction, or lower edit collateral—at comparable behavioral
> error.

This rule has already pruned attractive but misleading answers. Token clusters are
not rewarded merely for looking meaningful. Low rank is not rewarded when its axes do
not repeat. A tiny local error is not rewarded when downstream components react
differently.

## What the three-hour mathematical reviews changed

They have led to concrete progress, though mostly by changing and repairing the next
experiment rather than producing a new checkpoint result.

The most important changes were:

1. **From arbitrary latent axes to physical actions.** The failed local MLP1 basis
   prompted the move to scaling actual multiplication gates. A successful result now
   corresponds directly to a runnable smaller MLP.
2. **Two different compression questions.** The reviews separated “can a few gates
   span all observed response patterns?” from “can those gates reproduce the combined
   effect of all native gates?” The first can succeed while the second fails, so the
   new experiment requires both.
3. **Gauge invariance became an executable check.** A proposed shuffled-factor control
   was discovered to depend on arbitrary rescaling of equivalent factors. We now
   normalize and sign-orient each factor pair, and tests confirm that rescaling,
   sign flips, and gate relabeling cannot change the control.
4. **A finite nonlinear action model became the next step after tangent success.** If
   a gate set wins, we will not jump directly to deletion. We first test whether small
   simultaneous changes obey their predicted linear and pairwise interaction law.
5. **Intrinsic polynomial rank became the failure branch.** If no subset of native
   gates works, we will inspect the underlying quadratic tensor for a different,
   smaller factorization. This distinguishes “the checkpoint chose a bad dictionary”
   from “the function itself needs many products.”
6. **Several elegant ideas were pruned.** Information bottlenecks need arbitrary noise
   choices here; token-prefix Hankel methods do not match the continuous nonlinear
   interface; MDL cannot generate useful components by itself; full invariant theory
   is too broad before we isolate the relevant polynomial subgraph.

So the reminders have improved the mathematical target and caught invalid simplicity
claims. They have not yet produced a newly compressed MLP. The frozen physical-gate
measurement is the experiment that now needs to convert that design progress into a
model result.

## The exact current plan

### Step 1: measure every MLP1 gate's downstream effect

For each of 32 fresh documents and each of two independent sets of 32 output probes,
we ask how the complete model changes when one MLP1 gate is scaled infinitesimally at
**all token positions where that gate is used**.

The resulting number is called $E_{c,a,n}$:

- $c$ identifies the document/context;
- $a$ identifies the downstream probability probe;
- $n$ identifies one of the 4,608 MLP1 multiplication gates;
- $E_{c,a,n}$ is the first-order effect of scaling gate $n$ on probe $a$ in context
  $c$.

The 32 documents are split before measurement:

- 16 fit documents may choose gates and fit linear coefficients;
- 16 validation documents may only evaluate that frozen choice;
- the two probe halves test whether the result repeats under new sampled output
  directions.

### Step 2: compare small gate sets at three sizes

We test retaining 32, 128, or 512 gates. The main selector favors gates that preserve
the many different response patterns in $E$. It must beat four alternatives:

- gates with the largest response energy;
- gates with the largest ordinary activation-times-output weight;
- random gates;
- a deliberately mismatched product/output pairing, canonicalized so arbitrary factor
  scaling cannot change it.

Every method gets the same number of gates, the same fit rows, the same numerical
solver, and the same held-out evaluation.

### Step 3: require two kinds of transfer

The selected gates must do both:

1. reconstruct the full collection of individual gate-response patterns using a map
   fitted only on the fit half; and
2. reproduce the combined first-order effect of turning on all 4,608 native gates.

The first asks whether the selected gates are a good response dictionary. The second
asks whether they approximate the actual native MLP operating point. Passing only the
first is not enough for compression.

We also require the selected support to substantially repeat across independent probe
halves, to beat every control on both validation halves, and not to hide a severe
failure on one document. A 20,000-draw document bootstrap gives one simultaneous 95%
lower confidence band across all 48 primary comparisons.

### Step 4A: if no gate budget passes

We stop pursuing sparse subsets of the checkpoint's native gates. The next cheap test
computes the Gram spectrum of their exact quadratic forms. In plain language, this asks
whether the 4,608 native products are algebraically redundant or merely behaviorally
similar on natural text.

- If the native quadratic dictionary is genuinely redundant, we search for an exact
  smaller factorization.
- If it is full-rank but ill-conditioned, we test a stable approximate refactorization.
- If it is full-rank and well-conditioned, exact native-gate deletion is the wrong
  route; we target a joint MLP1/MLP2 causal interface instead.

### Step 4B: if a gate budget passes

We move only 10% of the way from the native MLP toward the sparse candidate and ask
whether the observed probability change matches the first-order prediction. This tests
whether normalization and later nonlinear computation invalidate the tangent result.

Only after that passes do we build a complete finite replacement and test:

- cross-entropy and full-distribution divergence;
- top-token agreement and task accuracy;
- the existing causal intervention bank;
- unseen target-frequency strata and genuinely different data;
- selective removal of a behavior and damage to unrelated behaviors;
- interactions among retained gate packages;
- actual stored bytes, multiplication count, latency, and zero calls to native MLP1.

## What is blocking the next result

No user decision or new authority is currently required.

The frozen mathematics and data split are complete and independently audited. The
remaining implementation must:

1. rebuild and verify the exact admitted rank-640 program;
2. run the complete model while exposing the 4,608 shared MLP1 gate scales;
3. compute the real response and the gauge-safe mismatched-factor control in one-use
   gradient transactions;
4. perform the frozen fit/validation analysis without leaking validation information;
5. publish only the permitted aggregate ledgers and candidate support in create-only
   artifacts;
6. bind every executed source file and program buffer before GPU launch.

That source-closed collector now passes its own focused CPU suite. The first independent
audit found that its artifact write was not crash-safe, some execution counts were
asserted instead of derived, an incomplete numerical arm could silently reduce the
registered 48-comparison confidence family, and failure publication was under-bound.
Those defects have been repaired, including full semantic replay of supports,
coefficients, comparison statistics, confidence bounds, and decisions from the frozen
in-memory responses. A second independent audit is the final launch gate;
then the GPU run should be short. The outcome, rather than another round of protocol
elaboration, will decide the next mathematical branch.

## Symbol and jargon glossary for the recent documents

- $\ell$ is lowercase **ell**, a learned vector on the left side of one multiplication
  gate. It is not the number one and not a loss metric.
- $r$ is the learned vector on the right side of that gate.
- $h_n(z)=(\ell_n^\top z)(r_n^\top z)$ is the continuous output of gate $n$.
- $d_n$ is the output direction multiplied by that gate's value.
- $L$ or “loss” usually means an error being minimized. In the new gate experiment,
  the document loss is a normalized squared response error.
- In the older proposed **L/R/T** comparison, the capital letters were arm names, not
  mathematical standards: **L** trained an MLP0/MLP1 replacement to match local
  internal coordinates, **R** used the same-size replacement but trained it to match
  the final suffix distribution, and **T** added an explicit learned map carrying the
  MLP0 code into the MLP1 code. That comparison was intended to ask whether downstream
  training or a typed cross-layer edge buys better composition. It is not the current
  running experiment; the unstable MLP1 interface made the physical-gate test more
  informative first.
- CE is cross-entropy: how much probability the model assigns to the true next token.
- KL is a full-distribution difference between two predicted token distributions.
- rank is the number of independent continuous directions needed by a specified
  matrix or tensor; it is not automatically the number of human concepts.
- SVD is the standard decomposition used to find directions ordered by squared
  variation or response energy.
- $r_{95}$ is the smallest number of SVD directions containing 95% of measured squared
  response energy.
- CSS means column-subset selection: choosing actual columns, here actual MLP1 gates,
  rather than inventing new mixed directions.
- gauge means an internal rescaling, sign flip, permutation, or basis change that leaves
  the model's function unchanged.
- LCB/UCB mean lower/upper confidence bounds. They express statistical uncertainty,
  not new loss functions.
- tangent means the first derivative for a very small edit. It need not remain correct
  for a large edit.
- bootstrap means repeatedly resampling whole documents to estimate uncertainty. Whole
  documents, not individual tokens, are the independent units in the current assay.
- compiler means a standalone executable replacement that computes the proposed simpler
  program without calling the native component. A fitted description or oracle
  transplant is not a compiler.

## Bottom line

The project is not stuck because MLP0 or MLP1 is completely opaque. We have learned a
fairly specific lesson: MLP0 contains real shared lexical and continuous structure,
but the early MLPs form a coupled compensating computation, and arbitrary low-rank
coordinates do not compose reliably. The next experiment tests a more principled kind
of simplicity—few actual physical multiplication gates chosen by their complete
downstream effect.

The near-term result should be decisive in a useful way. A pass gives us the first
direct route to an executable MLP reduction. A failure rules out sparse native-gate
selection at the registered sizes and sends us to the intrinsic quadratic tensor,
rather than another cycle of token clustering or local PCA.
