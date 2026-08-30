# Current project update — plain English

**Time:** 2026-08-30 00:15 UTC  
**Main question:** are we getting closer to a smaller program that predicts, composes,
and can be edited like the native bilin18 model?

## Short answer

We made one useful mathematical advance and one engineering advance.

1. **The layer-10 compression knee is real, but it is not visible in raw tensor
   rank.** Every native MLP tensor slice from layer 0 through layer 17 has numerical
   rank 1,152, with a very smooth singular-value tail. The abrupt difference in how
   much rank the compiled program needs after layer 10 must therefore come from the
   states the model actually visits and what downstream components read—not from late
   weight tensors simply having higher algebraic rank.

2. **The downstream-consequence experiment for MLP2 is finally audit-ready.** Its
   collector passed an independent, outcome-blind audit after 115 tests. The GPU is
   currently finishing the other agent's map-rank job; once free, we run DESIGN,
   freeze the predictor, and then make one sealed HELDOUT prediction.

This is progress, but it is not yet a large increase in the fraction of the native
model explained. The strict score remains about **10.92% of causal CE named**, with
**89.08% unexplained** and **0 of 68** terminal extraction/removal/OOD actions passed.

## 1. What the newest compiled-program results say

The current table-based replacement program does not want the same rank everywhere.

- Its MLP tables want rank 768 at layers 0--9 and rank 1,152 at layers 10--17.
- Raising late MLP rank improves cross-entropy enough to repay the frozen storage
  price. Raising shallow MLP rank does not.
- Attention tables do not show the same effect: rank above 384 is not worth its price.
- The newest map test says the stream map is over-bought at the earliest MLP sites:
  rank 256 rather than 640 at MLP0--5 costs about `0.000125` nat while saving 5.31M
  stored values. Cutting it at all layers is too damaging. This is a compiled-program
  allocation result, not native-model interpretation.

The best compiled allocation is therefore heterogeneous: spend capacity where the
downstream computation uses it. This improves executable compression, but the compiled
program is still much worse than native (`~5.9` CE versus `~3.14`). It does not by
itself reverse engineer the missing context-dependent computation.

## 2. The new mathematical computation

For one bilinear MLP, write the quadratic part as

$$
q(x)=D\big((Lx)\odot(Rx)\big).
$$

Here $x$ is the 1,152-dimensional residual stream, $L$ and $R$ make two linear
projections, $\odot$ multiplies matching coordinates, and $D$ maps the products back
to the residual stream.

To inspect this quadratic map without choosing arbitrary hidden-neuron coordinates,
we use its **symmetric polarization**. Fixing one input to the coordinate vector
$e_0$ gives an ordinary matrix

$$
A_{e_0}=\frac12D\left[
\operatorname{diag}(Re_0)L+\operatorname{diag}(Le_0)R
\right].
$$

Why this matrix matters: an $r$-product quadratic program can give every such matrix
rank at most $r$. Its singular values therefore give both a product-count lower bound
and the best possible coefficient-space error after a rank cut.

We computed all singular values of this $1152\times1152$ matrix for all 18 MLPs from
the exact checkpoint weights in float64. The run took **11.67 seconds**.

Results:

- every layer has numerical slice rank **1,152**;
- the optimal rank-768 relative coefficient error ranges only from `0.1210` to
  `0.1325`;
- MLP10 divided by MLP9 is only `1.0113`;
- the median of layers 10--17 divided by the median of layers 0--9 is only `1.0554`.

The frozen definition required a `1.20` ratio to call a coefficient-rank knee. Both
tests fail. Thus the late layers are not algebraically higher-rank in this slice.

This helps because it prunes a tempting but low-ROI direction: ordinary HOSVD/SVD or
raw-weight rank will not explain why layer 10 is the causal boundary. A useful
simplicity measure has to weight directions by reachable inputs and downstream
consequences.

Detailed result:
[`NATIVE_MLP_POLARIZATION_DEPTH_PROFILE_RESULT.md`](NATIVE_MLP_POLARIZATION_DEPTH_PROFILE_RESULT.md).

## 3. What the MLP2 downstream-consequence experiment computes

We already know that a rank-512 MLP2 program can have tolerable standalone CE yet
compose non-additively with compressed MLP0. Training it on both native and compressed
MLP0 inputs did not solve that interaction: it reduced the interaction by only about
13%, and its advantage over an equal-compute native-only control was statistically
unresolved.

The new experiment asks whether we used the wrong error metric.

For MLP2 input state $z$, native write $f_2(z)$, and compressed program $P(z)$, define
the complete write error

$$
E=P(z)-f_2(z).
$$

We inject small signed fractions of this error,

$$
f_2(z)+\alpha E,
\qquad
\alpha\in\{-1/8,-1/16,+1/16,+1/8\},
$$

then run the exact native suffix. This measures four kinds of downstream consequence:

1. **CE directional derivative:** whether moving along this error initially raises or
   lowers true-token cross-entropy.
2. **Logit Fisher quadratic:** $\Delta\ell^T F\Delta\ell$, the local teacher-KL cost
   of the final-logit change. It measures output-distribution movement while ignoring
   a uniform logit shift.
3. **Attention-5 response energy:** how strongly the full attention-5 write changes.
4. **Attention-6 response energy:** the analogous change at attention 6.

The finite target is the real composition interaction for document $d$ and program
$P$:

$$
i_{d,P}=
[CE_{d,C}(1)-CE_{d,C}(0)]-
[CE_{d,N}(1)-CE_{d,N}(0)],
$$

where $C$ means MLP0 has been replaced by C512 and $N$ means native MLP0. In plain
language: how much more damaging is the MLP2 error after compressing MLP0?

On 32 DESIGN documents, we fit three tiny predictors:

- local MSE only;
- final CE-linear plus Fisher-logit terms;
- final terms plus separate attention-5 and attention-6 response terms.

Program identity is not supplied, so the predictor must describe a shared mechanism
rather than memorize FULL/CONTINUE/ROBUST. Ridge strength is chosen by leaving out one
whole source document at a time. After that, coefficients and normalizers are frozen.
The 32 HELDOUT documents are then opened once.

Two negative controls preserve error size but break correct document matching:

- **DERANGED:** move each complete document error to a different document;
- **COV_RANDOM:** make deterministic mixtures of the other documents' centered
  errors and rescale them to the recipient norm.

The experiment passes only if downstream-consequence features predict the true finite
interaction on HELDOUT better than local MSE and the mismatched controls fail. Small
$\alpha$ must also be consistent between `1/16` and `1/8`, and $\alpha=1$ must exactly
replay the physical compressed program.

## 4. Why the audit took several rounds

The scientific forward pass is simple; most delay was in making the sealed
DESIGN-to-HELDOUT transaction genuinely non-circular. The independent auditor found,
before any model response was opened:

- missing transitive protocol/test files in source closure;
- a predictor bundle that was shape-checked but not recomputed from DESIGN;
- authority/checkpoint race windows;
- failure publication that could fail precisely when a protected input drifted;
- missing adversarial tests for rival terminals and replaced locks;
- finally, a tiny race between the last authority read and collection.

All are now repaired. The exact source closure contains 46 files and **115/115**
transitive tests pass. HELDOUT recomputes the complete predictor from DESIGN and
requires exact agreement. The independent audit status is **GO**, with no row tensor,
model response, bundle, or outcome opened during audit.

This work was slower than the numerical run will be, but it prevents a held-out result
from being accidentally selected after looking at it. The infrastructure is now
reusable for other two-role causal-metric tests.

## 5. Did the mathematics help?

Yes, in three concrete ways.

1. **It supplied a falsifiable simplicity notion.** Error size is not enough;
   downstream consequence is an observable-weighted seminorm. The Rayleigh/Fisher
   quantities operationalize that idea and test whether it predicts finite
   composition, rather than merely reducing local MSE.
2. **It ruled out a broad class of explanations.** Full rank 1,152 at every native MLP
   and smooth coefficient tails mean the layer-10 knee is not explained by ordinary
   raw tensor rank or HOSVD.
3. **It separates global algebra from distributional simplicity.** Globally exact
   rank-512 MLP replacements are impossible for these slices, but low-rank programs
   can still work on the reachable natural-text states and observables. That tells us
   where a successful theorem must live: a reachable, consumer-conditioned quotient,
   not an unweighted identity over all $\mathbb R^{1152}$.

The math has not yet delivered semantic names for MLP0/1/2 coordinates or a complete
editable program. Its value so far is sharper pruning and a better next experiment.

## 6. Current blockers and odd results

- **No scientific blocker to DESIGN remains.** The exact audit is GO. The only
  scheduling blocker at this instant is the shared GPU's map-knee job; DESIGN launches
  when it releases.
- The composition interaction is diffuse across roughly 108--118 effective documents,
  not a small collection of pathological prompts.
- FULL/CONTINUE/ROBUST document interactions correlate `0.843--0.910`, and one shared
  document mode explains `91.2%` of their centered energy. This is evidence for a
  shared failure geometry, not proof of one residual-space direction.
- A two-background local-MSE fit did not fix composition. Repeating local MSE training
  is low priority.
- Attention 5 often behaves like a presence/control interface in compiled settings,
  while attention 6 has small but genuinely high-dimensional content. Treating either
  as a universal one-dimensional semantic code would be wrong.
- The compiled program's improved storage allocation is useful engineering, but its
  large CE gap means it cannot be counted as explaining the native model.

## 7. Ranked next directions by expected return

### 1. Finish Rayleigh DESIGN → frozen predictor → HELDOUT

This is still highest ROI. It directly decides whether a consequence-weighted metric
is useful before we spend another large fit. It is cheap relative to a training sweep,
composition-relevant, and sharply falsifiable.

If it passes, fit one equal-price MLP2 rank-512 program under the successful metric and
test standalone CE, finite MLP0 composition, extraction/removal behavior, and held-out
transport. If it fails, do not train that student.

### 2. Directly model the mixed MLP0 × MLP2 functional

If small-amplitude metrics are valid but fail at $\alpha=1$, fit the interaction term
itself rather than pretending a local quadratic norm contains all RMSNorm/attention
curvature. This is more specialized but directly targets the known error.

### 3. Run a C512 × best-MLP1 × best-MLP2 factorial

MLP1 has not been incorporated cleanly into the MLP0/MLP2 composition story. A full
factorial tells us whether MLP1 transports, amplifies, or compensates the state shift.
It is the best near-term independent early-layer entry point.

### 4. Expand verified downstream consumers, then learn a causal quotient

Additional late circuits can help early-layer interpretation because they provide
more independent readouts of what early writes are for. The useful version is not
“find more heads” in general. It is: verify several functionally distinct downstream
consumers, then retain the smallest early state that predicts all of them and test a
held-out consumer/composition. Only then does a Hankel/minimal-realization calculation
become identifiable.

### 5. Continue heterogeneous shipped-program allocation

This is currently fast and has produced executable savings, including the layer-10
MLP knee and shallow map cut. It is worthwhile in parallel, but lower ROI for native
reverse engineering because the all-table suffix removes the live interfaces we most
need to understand.

## Bottom line

We are not “almost done” with whole-model interpretation: the strict unexplained
fraction is still 89%. But the strategy is more focused than it was several hours ago.
Raw rank, generic HOSVD, another unweighted local-MSE fit, and uniform per-layer rank
sweeps are now pruned or tightly scoped. The immediate question is concrete:

> Does a small set of native downstream-response measurements predict which MLP2
> approximation errors cause the real MLP0×MLP2 composition penalty?

The audited experiment answering that question is next.
