# Current project explanation — 2026-08-29 03:34 UTC

## UPDATE: the most important new result

The previous explanation said that most of the error in the context-free program
survived because a linear per-token map was too simple.  That conclusion was based on
a map of rank 64.  A new rank-512 control overturns it.

On three disjoint document samples, an **oracle** rank-512 linear map from the
already-computed one-token residual stream leaves only `0.114`, `0.141`, and `0.142`
nats of error per uncovered position relative to the token-specific ceiling.  The
deployed rank-64 embedding map leaves `0.781`, `0.862`, and `0.840` nats.  Thus the
oracle construction removes approximately 85.4%, 83.7%, and 83.1% of that deficit.

This is much better than expected.  It says that the required token rows are largely
expressible by a moderately high-rank linear map when the input is the right internal
state.  It does **not** yet give us a deployable compression, because the oracle was
fitted on the very uncovered rows it was later asked to reproduce.  The decisive next
test is the same rank-512 stream map fitted only on covered tokens and then evaluated
on unseen tokens.

The other active line is a consequence-fitted simplification of MLP3.  Its mathematical
design and CPU core exist, but no Family-F scientific result exists yet.  Its numerical
runner is deliberately still blocked by correctness checks described below.

## 1. The short version of the whole story

We have learned three different things, at three different levels.

1. **Exact algebra gives us real candidate parts.**  Each bilinear MLP is exactly a
   sum of scalar product gates followed by output vectors.  RMSNorm and residual
   polarization also let us separate exact residual/attention interaction terms.
2. **Some large programs can be compressed predictively.**  A shared-QK whole-model
   program removes 5.3481% of original stored values while passing its registered
   predictive and causal tests.  The context-free token-row program also has a real
   rank/coverage/cross-entropy frontier.
3. **Locally good parts often fail when composed.**  At MLP3, individual compressed
   polynomial pathways preserve 44–85% of their omission effects, yet replacing all
   four together is not faithful.  The downstream network is sign-sensitive and
   nonlinear.  This motivates selecting gates by their effect after Blocks 4–17,
   rather than by local squared error alone.

So the main obstacle is no longer finding *some* low-rank tensor.  It is finding a
small set of interfaces whose simplifications remain correct when connected to one
another.

## 2. Terms and ledgers

### Cross-entropy and nats

For a true next token $y$ and predicted probability $p(y)$, the cross-entropy loss
is

$$
\operatorname{CE}=-\log p(y).
$$

The natural logarithm makes the unit a **nat**.  Lower is better.  A CE increase of
`+0.10` nat means the replacement assigns the observed next tokens less probability
than the native model does.

### KL divergence

When we want to imitate the native model's entire next-token distribution $p$, not
only the observed token, we use

$$
\operatorname{KL}(p\Vert q)=\sum_v p_v\log\frac{p_v}{q_v},
$$

where $q$ is the replacement model's distribution and $v$ ranges over vocabulary
tokens.  KL is zero exactly when the two distributions agree.  In the MLP3 experiment,
the native model is the teacher and the simplified model is the student.

### NRMSE

**Normalized root mean squared error** compares a locally reconstructed tensor or
activation with the native one:

$$
\operatorname{NRMSE}
=\sqrt{\frac{\sum\|\widehat w-w\|^2}{\sum\|w\|^2}}.
$$

Zero is exact.  In the Block-3 convention, one is approximately the bias-only
baseline.  NRMSE is useful for debugging the local port, but it is not the final
criterion: a small local error can have a large nonlinear downstream consequence.

### What fraction of the model is explained?

There is no single honest percentage because the project has several non-equivalent
currencies:

| Question | Current settled answer | Meaning |
|---|---:|---|
| Is every architectural tensor/site inventoried? | 36/36 sites | We know what exists, not what each part means. |
| How much original storage has a whole-program consequence certificate for removal? | 5.3481% | A genuine executable compression certificate. |
| How much of the strict named causal CE headroom is recovered? | 10.923% | 89.077%, or 4.72714 nats in that ledger, is still unnamed. |
| How many terminal extraction/removal/OOD actions have scientific outcomes? | 0/68 | The configurations exist, but the terminal experiment is not complete. |

The `10.923%`, `5.3481%`, and older `32.1% ± 6.4%` figures have different denominators
and must not be added.

## 3. The context-free compiler, with its computation

### What is being compiled

At each of the model's 36 attention/MLP sites, the context-free program replaces the
live output with a vector determined only by the token at the current position.  Call
the desired vector for token $t$ at site $j$

$$
r_j(t)\in\mathbb R^{1152}.
$$

A **covered token** has an explicitly learned or stored row.  An **uncovered token**
needs a fallback rule that predicts its row.  This is where the new result applies.

The **token-specific ceiling** is the performance obtained if we give every scored
token its own correct one-token row.  It is a ceiling only for this context-free
function class; the live model can still do better by using surrounding context.

### The linear fallback calculation

Suppose the input representation for $n$ tokens is the matrix

$$
X\in\mathbb R^{n\times1152}
$$

and their desired output rows at one site are

$$
Y\in\mathbb R^{n\times1152}.
$$

The ridge-regression map is

$$
W=(X^TX+\lambda I)^{-1}X^TY.
$$

If $x_t$ is a new token's input state, its predicted row is

$$
\widehat r_j(t)=x_tW.
$$

To make the map rank $k$, take the singular value decomposition

$$
W=U\Sigma V^T
$$

and keep only the largest $k$ singular directions:

$$
W_k=U_{:k}\Sigma_{:k}V_{:k}^T.
$$

Here **rank 512 means 512 shared continuous directions in this linear map**.  It does
not mean 512 token clusters or 512 semantic categories.  Rotating those directions
usually produces an equivalent map, so a direction does not automatically have a
name such as “city” or “number.”

### Embedding input versus stream input

The embedding input is the token's initial 1,152-dimensional vector.  The
**one-token residual stream entering a site** is obtained by running a sequence of
length one up to that site.  It is still a deterministic function of the token alone,
but it has already passed through earlier model computation.  When predicting a later
site, this state has already been computed, so using it may be cheap at execution time.

The recent sequence of measurements is:

| Fallback | Fit population | Rank | deficit against uncovered-token ceiling |
|---|---|---:|---:|
| embedding map | covered tokens | 64 | 0.781 / 0.862 / 0.840 |
| embedding map | uncovered tokens (oracle) | 64 | 0.651 / 0.722 / 0.700 |
| stream map | uncovered tokens (oracle) | 64 | 0.555 / 0.621 / 0.577 |
| stream map | uncovered tokens (oracle) | 512 | **0.114 / 0.141 / 0.142** |

The last row changes the interpretation.  Rank, choice of input, and fit population
interact strongly; their gains cannot be estimated one at a time while the others are
held at weak settings.

### What this establishes and what it does not

It establishes an **expressibility result**: within the observed uncovered tokens, a
rank-512 linear function of the one-token stream can approximate the required rows far
better than the rank-64 alternatives.

It does not establish transfer to new tokens.  The oracle map used

$$
X=X_{\text{uncovered}},\qquad Y=Y_{\text{uncovered}}
$$

during fitting and then scored those same target rows.  A deployable experiment must
fit $W_{512}$ using only covered-token pairs and score it on disjoint uncovered
tokens.  If that works, we gain a concrete new point on the executable
description-size/CE frontier.  If it fails, the remaining problem is generalization,
not linear expressibility.

That exact deployable test is running while this explanation is being written.  It is
owned by the parallel GPU branch; no outcome is claimed here.

## 4. The Block-3 gate program, with its computation

This is a separate experiment from the context-free compiler.

### A bilinear MLP as a sum of exact gates

For normalized input $z\in\mathbb R^{1152}$, one bilinear MLP can be written as

$$
w(z)=b+\sum_{i=1}^{4608}d_i
\bigl(\ell_i^Tz\bigr)\bigl(r_i^Tz\bigr).
$$

For gate $i$:

- $\ell_i$ is one row of the Left matrix;
- $r_i$ is one row of the Right matrix;
- $h_i(z)=(\ell_i^Tz)(r_i^Tz)$ is one scalar product feature;
- $d_i\in\mathbb R^{1152}$ is one column of the Down matrix;
- $b$ is the output bias.

The native program executes 4,608 such scalar products per token.  A $K=512$
program keeps 512 gates, so the core simplicity question is whether roughly one ninth
of the gates can reproduce the relevant behavior.

### Gauge balancing

The same gate is unchanged by

$$
\ell_i\mapsto c\ell_i,\qquad r_i\mapsto r_i/c.
$$

This is a **gauge freedom**: the parameter values change while the computed function
does not.  Before comparing or selecting gates, we choose a balanced representative so
that reciprocal rescaling cannot make a gate look artificially large or small.

### Polarization into four typed pathways

At Block 3 the normalized input separates as

$$
z=u+v,
$$

where $u$ is the incoming residual contribution and $v$ is the attention
contribution.  Bilinearity gives the exact identity

$$
B(u+v,u+v)=B(u,u)+B(u,v)+B(v,u)+B(v,v).
$$

These are called `uu`, `uv`, `vu`, and `vv`.  They are exact polynomial pathways, not
clusters invented by a probe.

### What Family A measured

Family A selected gates to minimize local reconstruction error in those four terms,
then installed the resulting MLP3 replacement and reran Blocks 4–17 normally.

| program | products/token | exact bytes | local NRMSE | KL / bias-only KL | CE change | native top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| bias only | 0 | 4,608 | 1.000 | 1.000 | +0.13028 | 79.24% |
| Family A, K=256 | 256 | 3,545,600 | 0.7603 | 1.3823 | +0.17998 | 76.34% |
| Family A, K=512 | 512 | 7,086,592 | 0.6842 | 0.7121 | +0.09138 | 82.27% |
| native | 4,608 | 63,705,600 | 0 | 0 | 0 | 100% |

`KL / bias-only KL` divides the candidate's KL by the KL caused by removing the
entire bias-free MLP write.  A value below one beats omission; zero is exact.  K=256
is the confusing case: it locally reconstructs some signal but is behaviorally worse
than preserving only the bias.  This directly demonstrates why local tensor error is
not a sufficient definition of simplicity.

What *did* work was replacing one typed pathway at a time while keeping the other
three native.  At K=512, the fractions of each omission effect recovered were:

- `uu`: 85.08%;
- `uv`: 49.60%;
- `vu`: 44.37%;
- `vv`: 62.99%.

Thus the selected gates contain real pathway signal.  The failure occurs when all four
approximations are connected simultaneously.  This is a composition failure, not an
absence-of-structure result.

### Why the mirror result matters

For diagnosis we also evaluated

$$
w_{\text{mirror}}=2w_{\text{native}}-w_{\text{candidate}}.
$$

The candidate and mirror have equal-magnitude, opposite-sign local error.  The mirror's
KL/bias ratio was `0.198`, much better than the candidate's `0.712`.  The mirror is not
a compression because it needs the native write, but it proves that Blocks 4–17 react
very differently to the two signs of the same local error.  A symmetric local MSE fit
cannot see this asymmetry.

## 5. What pending Family F is trying to do

Family F keeps the same exact gate grammar and the same $K=256,512$ execution
budgets.  It changes the selection objective from local reconstruction to downstream
consequence.

### Stage F1: learn soft gate scores

Start from the sealed pool of 1,024 candidate native gates.  Give each gate a score

$$
0\le s_i\le1,\qquad \sum_{i=1}^{1024}s_i=512.
$$

The student MLP write is

$$
w_s(z)=b+\sum_{i=1}^{1024}s_i d_i h_i(z).
$$

The Down vectors $d_i$ are fixed during this stage.  Otherwise the optimizer could
multiply $s_i$ by $c$, divide $d_i$ by $c$, and make the score meaningless.

For every fit sequence:

1. run native MLP3 and native Blocks 4–17 to obtain teacher logits;
2. replace MLP3 by $w_s$;
3. rerun Blocks 4–17 autonomously, without borrowing hidden teacher states;
4. minimize document-balanced teacher-to-student KL on positions 64–255;
5. project the scores back onto the capped simplex above.

After fitting, sort scores from largest to smallest.  The first 256 gates form K=256
and the first 512 form K=512, so the supports are nested.

### Stage F2: refit the decoder

Once a finite support $S$ is selected, refit a single output decoder from sealed fit
statistics:

$$
\widehat D_S^T=
\left(G_S+10^{-6}\frac{\operatorname{tr}(G_S)}{K}I\right)^{-1}C_S.
$$

Here $G_S$ is the selected gates' feature Gram matrix—roughly, which gates activate
together—and $C_S$ is their cross-moment with the desired MLP output.  The small
diagonal term stabilizes the inverse.  Downstream KL chooses *which* gates survive;
the local statistics then choose the best common output vectors for that fixed support.

### Stage F3: test a nearly free affine correction

For diagnostic arms at K=512, fit

$$
w_{a,c}(z)=b+c+a\,[w(z)-b],
$$

where $a$ is one scalar and $c\in\mathbb R^{1152}$ is a constant correction.  At
deployment, $a$ can be folded into the existing decoder and $c$ into the existing
bias, so this adds no array or runtime operation.  It is execution-cost-free but not
fit-description-free: 1,153 values were optimized.  These calibrated variants are
diagnostics and cannot make Family F pass.

### Controls

Family F uses several controls with distinct purposes:

- **matched random support:** asks whether learned gate selection beats choosing K
  gates at random;
- **same-support permuted cross-moment:** keeps the learned support but destroys the
  correct gate-to-output association;
- **row-reversed teacher:** weakly breaks which teacher label belongs to which row;
- **document-deranged teacher:** pairs every target document with a different donor
  document, a stronger label-information null;
- **Family-A overlap:** measures whether downstream consequence rediscovers the same
  gates selected by local activation reconstruction.

Only the uncalibrated, real-teacher Family-F program can advance to validation.

### Amount of computation

The frozen run has:

- 3 score fits × 480 optimizer steps = 1,440 steps;
- 4 affine fits × 240 optimizer steps = 960 steps;
- total = 2,400 Adam steps and 9,600 two-row backward passes;
- one final 18-arm reporting sweep over all 480 fit rows.

The 480 rows come from 209 source documents.  Rows are weighted inversely by how many
rows their document contributes, so a repeated document cannot dominate the loss.
Fit, validation, and final documents are disjoint.

## 6. Why Family F has not run yet

There is no missing-data, cache, or GPU blocker.  The current blocker is that the
source-closed numerical runner has not passed adversarial audit.  This is deliberate:
after a fit has seen outcomes, silently repairing its rules would make the experiment
post-selected.

The remaining implementation issues are concrete:

1. **Raw-logit known answer:** the independent native replay must compare logits
   before the `30*tanh(raw/30)` softcap as well as after it.  Comparing only the
   softcapped values can hide a large discrepancy in saturated logits.
2. **Fit exactly what will be deployed:** the affine diagnostic currently fits one
   floating-point expression and publishes an algebraically equivalent folded one.
   In float32 they need not be numerically identical.  Fitting or explicitly checking
   the folded executable closes that gap.
3. **Family-A overlap at both K:** the report must include the registered K=256 and
   K=512 overlaps, not only random and label-shuffled overlaps.
4. **Bind every stored program to its selected support:** a valid-looking decoder must
   be rejected if its literal gate indices differ from the score-derived support.
5. **Semantic artifact replay:** after writing programs, results, and the final
   receipt, reload them and reconstruct their important tensors and joins from sealed
   parents.  Schema validation alone is insufficient.
6. **Race and resource guards:** parent files must not change between the last
   verification and the receipt write, and the 45-minute/30-GiB ceilings must be
   checked throughout the run rather than only at the end.

The governing original preregistration was also missing from source closure in the
first audit.  It has now been added locally, but the full corrected runner still needs
tests, a fresh independent audit, commit, and push before it may read Family-F fit
outcomes.

Definitions used here:

- **preregistration:** rules frozen before outcomes are opened;
- **source closure:** the exact committed source files whose hashes define the run;
- **authority:** a create-once artifact proving those rules, inputs, and hashes were
  fixed before execution;
- **receipt:** the last artifact, written only after outputs are reloaded and verified;
- **promotive arm:** an arm allowed by the preregistration to advance to validation;
- **diagnostic arm:** an informative comparison that is forbidden from advancing.

## 7. The 68 actions

The number 68 does not mean 68 discovered circuits.  It means 68 fixed whole-model
intervention configurations:

$$
34\ \text{MLP0/MLP1 configurations}
\times
2\ \text{MLP2 backgrounds}
=68.
$$

The two backgrounds use either the deployed simplified MLP2 or the exact native MLP2.
The 34 configurations include candidate programs, partial removals, transport terms,
false-parent and shuffled controls, and native/deployed baselines.  Running them on
common rows is intended to distinguish a genuine MLP0/MLP1 mechanism from compensation
by MLP2.  Their execution routes are physically defined, but the final scientific
comparison/reducer is incomplete, so the honest ledger remains 0/68.

## 8. What the mathematics has actually bought us

The recurring mathematical reviews have produced operational changes, not a single
magic factorization:

- **Gauge quotients** stopped meaningless rescaling from determining gate importance.
- **Exact polynomial polarization** produced the `uu/uv/vu/vv` causal pathways.
- **Low-rank matrix approximation plus Pareto pricing** turned “simple” into stored
  values and operations versus CE, and exposed rank/coverage interactions.
- **Oracle system-identification tests** separated expressibility from transfer and
  found that rank, input, and fitting compound strongly.
- **Causal suffix fitting** changed the MLP3 objective after the mirror experiment
  proved symmetric local error was misaligned with downstream behavior.
- **Document-level resampling and weighting** made evidence refer to new documents
  rather than repeated rows from the same document.

The norm-minimization/HOSVD, weight-SAE, and generic hierarchical/DAG ideas have not
yet produced a prospectively validated whole-model simplification.  They remain useful
candidate coordinate systems, but local reconstruction alone is now known to be an
insufficient success condition.

## 9. Current plan in priority order

1. **Finish the currently running deployable rank-512 stream fallback test.**  It fits
   on covered tokens and scores disjoint uncovered tokens.  This is the cheapest
   falsifier of whether the strong oracle result transfers into an executable program.
2. **Finish and re-audit the Family-F runner, then execute it once.**  This asks whether
   downstream consequence can turn individually useful MLP3 gates into a composable
   replacement.
3. **If Family F admits a candidate, test its finite-perturbation interface.**  Ordinary
   KL agreement is not enough for selective editing; the compressed port should react
   correctly when its input or write is deliberately changed.
4. **Complete the 68-action semantic reducer.**  This is the main bridge from
   reconstruction to extraction, removal, and MLP2-compensation tests.
5. **Use a hierarchy or context router only after a global support fails.**  If one
   Family-F gate set cannot compose, measure whether document-conditioned score
   gradients lie in a small shared subspace before paying for a more complicated DAG.

The project is making progress, but the strict whole-model claim remains small.  The
new rank-512 result gives a plausible cheap program for a previously difficult
fallback; Family F gives a falsifiable route from exact tensor gates to downstream
faithfulness.  Neither should be counted as full reverse engineering until it passes
transfer, composition, intervention, and OOD tests.

## UPDATE 2 — answers about the linear map and the overall strategy

### 10. What exactly are the linear map's input and output?

There is a separate map for every one of the 36 attention/MLP sites.  For site $j$
and token $t$, the experiment constructs a sequence containing only that token and
runs the native model up to site $j$.

The map input is

$$
x_j(t)\in\mathbb R^{1152},
$$

the residual-stream vector **entering that site** in the native length-one run.  The
target output is

$$
y_j(t)\in\mathbb R^{1152},
$$

the native attention or MLP output produced by that site in the same length-one run.
The fitted rule is

$$
\widehat y_j(t)=x_j(t)W_j,
$$

with $W_j\in\mathbb R^{1152\times1152}$.  A rank-$r$ version factors it as

$$
W_j=A_jB_j,\qquad
A_j\in\mathbb R^{1152\times r},\quad
B_j\in\mathbb R^{r\times1152}.
$$

For covered tokens, the rank-512 map was fitted from their length-one input streams to
their known output rows.  For scored uncovered tokens, the experiment fed their
native length-one input streams through that fitted map.  The resulting output row was
then used by the context-free replacement program whenever that token appeared.

This produced an important new discovery result:

| same-rank fallback | uncovered deficit, three roles |
|---|---:|
| rank-512 map from token embedding | 0.596 / 0.672 / 0.672 nat |
| rank-512 map from covered-fit length-one stream | **0.174 / 0.214 / 0.214 nat** |
| rank-512 stream oracle fitted on its evaluation rows | 0.114 / 0.141 / 0.142 nat |

The covered-token CE is bit-identical between the first two rows.  The input change
therefore improves all-position CE by approximately `0.102/0.117/0.111` nat at the
same map rank.  It preserves about 86–87% of the improvement suggested by the oracle.

#### An important deployability caveat

The script calls this a deployable covered-fit map because its **parameters** were not
fitted on uncovered target outputs.  But its uncovered inputs $x_j(t)$ were obtained
by running the **native length-one model**.  That input is not automatically available
to a standalone compressed program:

- storing all native $x_j(t)$ vectors would itself be another token-by-site table;
- recomputing them with the original model would retain the machinery we intend to
  replace;
- the live contextual residual stream is available during an ordinary forward pass,
  but it is not the same object as the native length-one stream used in this test;
- a recursively compiled length-one stream may be cheaply available, but the current
  result did not test that input distribution.

So the correct status is: **the mapping transfers across tokens, but executable input
closure has not yet been demonstrated**.  The cheapest crucial follow-up is to fit and
evaluate the same map using the stream actually generated by the compressed prefix.
If that works, the result becomes a genuine standalone component.  If it fails, the
present gain depends on an unavailable native feature and is not a valid compression.

### 11. Why rank 512? Why not use an even larger rank?

Rank 512 was the largest point in the existing map-rank sweep.  It was chosen to test
whether the poor rank-64 result was genuinely a limitation of linear maps.  It was not
chosen because 512 has a semantic interpretation or because larger ranks are
mathematically forbidden.

The maximum matrix rank is 1,152.  Larger ranks might reduce error further, but the
factorized storage grows linearly:

$$
36\times(1152r+r1152)=82{,}944r\quad\text{stored floats}.
$$

The exact prices are:

| representation | stored floats |
|---|---:|
| 36 separate rank-512 maps | 42,467,328 |
| 36 separate rank-576 maps | 47,775,744 |
| 36 dense full-rank maps | 47,775,744 |
| 36 separate rank-640 factors | 53,084,160 |
| 36 separate rank-1024 factors | 84,934,656 |

At rank 576, two dense low-rank factors already cost exactly as much as one full
$1152\times1152$ matrix.  Above that crossover, storing the dense map is cheaper than
storing the two nominally low-rank factors.  Thus “try rank 1024” is scientifically
reasonable as an expressibility control but is not a good compressed representation.

A rank-1024 discovery control completed after this section was first written:

| input | rank 512 deficit | rank 1024 deficit | improvement from doubling rank |
|---|---:|---:|---:|
| embedding | 0.596 / 0.672 / 0.672 | 0.587 / 0.665 / 0.664 | 0.008 / 0.008 / 0.008 |
| native length-one stream | 0.174 / 0.214 / 0.214 | 0.147 / 0.177 / 0.190 | 0.028 / 0.037 / 0.024 |

The rank-1024 factorization adds 42,467,328 floats and is more expensive than storing
the dense maps.  Its all-position gain is only about `0.0067` nat in the first role,
roughly 25 times worse per recovered nat than the rank-64-to-512 step.  Rank 512 is
therefore the current operating point.  Input closure is now much higher information
than increasing rank further.

### 12. Is a rank-512 linear map simple or interpretable?

It is simple under one definition and not under several others.

| Definition of simplicity | Rank-512 assessment |
|---|---|
| Algebraic grammar | Simple: it is one linear operator, represented as two matrix multiplies. |
| Literal parameter storage | Moderate/large: 42.47 million floats across 36 sites. |
| Executed arithmetic | Large: 1,179,648 multiplies per site, about 42.47 million across 36 sites per token. |
| Number of learned continuous degrees of freedom | Large. |
| Semantic readability | Weak: 512 dense coordinates have no names. |
| Gauge/canonical uniqueness | Weak at coordinate level: the internal basis can be rotated without changing $W_j$. |
| Editability or selective removal | Unproven. |
| Predictive utility | Strong discovery evidence, subject to input closure. |

So a large linear map is not itself a satisfying interpretation.  Its value is that it
identifies a restricted function class that predicts behavior and can potentially be
compressed further.  In particular, the 36 maps may share an output dictionary.  A
single shared rank-512 output basis with site-specific input maps would cost
21,823,488 floats instead of 42,467,328—saving 48.61% of map storage at the same
per-site multiply count.  The exact simultaneous reduced-rank regression solution for
this proposal is now implemented and passes its CPU proof tests; the model-data test
has not yet run.

This shared basis is also a better possible interpretation point than 36 unrelated
bases.  It would still need sparsity, stable semantic probes, or causal edit tests
before its coordinates deserve names.

### 13. Meta-assessment: are we nearly finished?

No.  We are near several **local decisions**, but we are not close to fully reverse
engineering the model.

The honest whole-project indicators remain:

- 5.3481% of original storage has a strict whole-program removal certificate;
- 10.923% of the strict named causal CE ledger is recovered;
- 0 of 68 terminal extraction/removal/OOD actions have scientific outcomes;
- Family F has not produced a numerical result;
- the new stream map has not passed standalone input closure;
- most components still lack semantic names and reusable causal interfaces.

The volume of recent work is partly real scientific work and partly experimental
infrastructure.  The infrastructure has prevented several false conclusions—most
recently the rank-64 claim that the linear family was exhausted—but it also means many
hours can pass without moving a global percentage.  We should not confuse a hardened
runner with a new model explanation.

#### Is the present strategy likely to pay off soon?

It is likely to pay off soon in the narrower sense of producing two decisive answers:

1. whether the strong stream-map gain survives a genuinely compressed recursive
   input; and
2. whether consequence-selected native gates make MLP3 composably replaceable.

Those answers could arrive after a small number of additional runs because both
hypotheses are concrete and already have implementations.  A positive result would
give a useful new executable component.  A negative result would close a substantial
family cleanly.

It is **not** reasonable to say that one more successful run will yield a complete
human-readable tensor program.  Even a Family-F pass covers one block and ordinary
next-token behavior; it does not automatically give semantics, OOD transport, or safe
editing.  The project needs a strategy that produces reusable interfaces across many
blocks, rather than solving every block independently with a bespoke experiment.

### 14. Alternative entry points ranked by expected return

#### 1. Close the stream-map dataflow, before increasing rank

Use as map input the length-one stream generated recursively by the compressed prefix,
or explicitly test the live contextual stream as a different program.  This is the
highest-return immediate move because it can either validate the new `0.174–0.214`
result as a standalone component or reveal that it depends on native information.
It is cheaper and more decisive than another rank sweep.

#### 2. Factor all 36 maps jointly

Use simultaneous reduced-rank regression with a shared output basis, optionally one
basis for attention and one for MLPs.  This directly tests whether the successful maps
use a common continuous language.  A rank-512 shared basis would nearly halve their
storage without adding multiplies.  Success gives both compression and a common place
to seek sparse/semantic directions; failure tells us the sites require distinct output
spaces.

#### 3. Work backward from downstream consequences

Instead of interpreting MLP0 in isolation, define a vector-valued response interface:
how do chosen logit groups, later residual directions, and finite edits respond to an
early component?  Factor the resulting prefix-by-suffix response matrix as a predictive
state or Hankel system.  Its rank has an operational meaning: the number of reusable
causal state variables required to predict new compositions.  This is more directly
connected to extraction and removal than weight reconstruction or a scalar CE cross.

#### 4. Start with terminal or sharply behavior-anchored circuits

The last few blocks and the unembedding give shorter causal paths than MLP0.  Likewise,
synthetic behaviors such as copying, induction-like continuation, bracket closure,
capitalization, or number formatting provide controllable positive and negative
examples.  Extracting and removing one such circuit on held-out templates and natural
text would move the currently empty action ledger and give empirical evidence about
which simplicity metric is useful.

This sacrifices immediate “whole-model” coverage, but it may be the fastest way to
validate the project's definitions of interpretation.

#### 5. Distill a globally priced tensor program, then analyze its interfaces

Fit several components jointly under a hard budget on stored floats, products, and
shared dictionary atoms.  Score native KL, ground-truth CE, finite edits, and OOD
transport.  This may find a much smaller functionally faithful program faster than
component-by-component interpretation.

The risk is that it produces another opaque student.  It should therefore be used as a
candidate generator whose shared states must pass causal and semantic tests, not as an
interpretation by declaration.

### 15. Recommended strategic adjustment

Keep Family F because it is a clean, nearly ready falsifier of a real composition
failure.  In parallel, stop treating the native length-one stream result as fully
deployable until input closure is tested.  Make the next general method the shared
predictive interface—not another local factorization:

$$
\text{available compressed state}
\longrightarrow
\text{shared low-dimensional code}
\longrightarrow
\text{site output and downstream response}.
$$

Judge that code by four outputs simultaneously: literal price, held-out CE/KL,
prediction of new component compositions, and selective intervention behavior.  That
is the most plausible route from the current collection of local results to a program
that is both smaller and genuinely understandable.

### 16. Doesn't the embedding map also improve at rank 512?

Yes.  It improves materially, and it is important not to attribute the whole gain to
the stream input.

The relevant comparison is:

| input | map rank | uncovered deficit, three roles |
|---|---:|---:|
| token embedding | 64 | 0.781 / 0.862 / 0.840 nat |
| token embedding | 512 | **0.596 / 0.672 / 0.672 nat** |
| native length-one stream | 512 | **0.174 / 0.214 / 0.214 nat** |

Moving the embedding map from rank 64 to rank 512 recovers
`0.185/0.190/0.168` nat per uncovered position.  That is about 20–24% of the
rank-64 embedding deficit.  So rank is a real lever.

Holding rank fixed at 512 and changing only the input from embedding to length-one
stream recovers another `0.421/0.459/0.458` nat.  This is roughly 2.3–2.7 times the
gain obtained from the rank increase.  Thus the larger rank and the richer input both
help, and they interact; neither effect is immaterial.

The rank-512 embedding map is also the cleaner executable result right now.  Its input
is the ordinary token embedding, which is unquestionably available without running
the original model.  The rank-512 stream map is much more accurate, but its current
test obtained uncovered-token inputs from native length-one forwards.  Until the
stream input is generated by the compressed program itself, the practical frontier is:

- **embedding rank 512:** weaker fidelity, but a genuinely available input;
- **stream rank 512:** much stronger fidelity, but standalone input closure unproven.

Both maps have the same factor price—42,467,328 floats across 36 sites—and the same
nominal matrix-multiply count.  If compressed-prefix stream closure succeeds, the
stream map strictly dominates the embedding map at that price.  If closure fails, the
embedding map remains the valid standalone baseline and the stream result is an
informative oracle-like diagnostic rather than a replacement program.
