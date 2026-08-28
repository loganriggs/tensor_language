# Hourly strategic review — 2026-08-28 20:40 UTC

## UPDATE PART

## Decision

The project should now optimize for a **small predictive interface**, not a small
coordinate set or a sum of locally simple components. The latest evidence makes that
distinction concrete:

- one replacement, MLP5, removes 61.2 percentage points—95% of B0's advantage;
- its loss is not carried by 16 special coordinates, by high-magnitude coordinates,
  or by a scalar/per-coordinate scale error;
- costs of replacing several sites cannot be predicted by adding the individual
  costs, multiplying that sum by one constant, or checking whether the sites are
  contiguous.

Therefore the next useful simplicity claim must predict **interactions and causal
responses that were not used to fit it**. Local rank or reconstruction remains a
candidate representation, but no longer constitutes success by itself.

One new result strengthens this decision: current-token explained variance predicts
replacement cost with the **wrong sign** (Spearman \(+0.466\)), while depth alone is
much stronger (\(+0.853\)). The most token-determined site, MLP1, is expensive to
replace; the least, attention14, is almost free. The missing metric is therefore much
more likely a **directional downstream response** than local substitution fidelity.
Isotropic random sensitivity has now also failed to beat depth descriptively, although
its registered landing control failed, so that result is provisional rather than a
clean falsification.

## Honest explained fractions

No explained-fraction ledger moved in this review.

| Ledger | Current value | Largest gap |
|---|---:|---|
| Sites with some executable structural surrogate | 36/36 | structure is not semantics or interchangeability |
| Certified whole-program storage removed | 5.3481% | 94.6519% lacks the registered consequence certificate |
| Older behavior covered by human-readable labels | 32.1% ± 6.4% | 67.9% remains unnamed in that ledger |
| Strict named causal CE headroom recovered | 10.923% | 4.72714 nats, or 89.077%, remains unexplained |
| New coupled early-MLP final actions observed | 0/68 | no new compiler causal credit exists yet |

The 36/36 number is inventory coverage. The 5.3481% number is certified executable
compression. The 10.923% number is the strongest strict causal account. They must not
be added or presented as one percentage.

## Largest remaining interfaces and confusing evidence

### 1. The final observational role owner exists, but its real adapter is missing

The response path itself is closed: each batch performs three exact-teacher forwards
and three forwards for each of 22 response arms; 48 batches require exactly 144
teacher and 3,168 student response forwards. The terminal boundary now joins all 22
response-bearing observational arms to that completed run exactly.

The canonical owner now exists. It fixes all 68 actions and all 48 batches, owns typed
row CE/copy/nine-bin frequency receipts and call ledgers, binds backend reductions to
their receipts, and checks that every action used identical frequency support. Its
focused adversarial/backend suite passes 109 tests. This is a closure advance, not
causal credit.

What remains is the adapter that obtains the frozen 192-by-192 frequency assignment
from authority, executes the real final rows and program sources, supplies the exact
denominators, and completes all 18 consumer norms. Only then can the registered
objective/transport gates and gauge/SVD/difference-in-differences/component closure
run. Until then the final role remains 0/68.

### 2. MLP5 is localized as a site, not understood as a computation

The following simple explanations are now separately falsified:

- 16 discrepancy-selected coordinates recover only 0.2% of the MLP5 stake;
- 256 recover 30.5%, so the error is at best moderately—not sparsely—concentrated;
- high-output coordinates are worse than random at every tested width;
- the B0-stream magnitude on those outlier coordinates is 0.899 times fully live,
  not exploded;
- magnitude matching does not rescue the discrepancy or outlier selections.

This points toward an input-dependent subspace, a distributed tensor map, or strong
downstream weighting of otherwise modest errors. It does not identify which one.

### 3. Interaction structure is observed but has no minimal state yet

Layer-1 attention and MLP replacement routes are nearly redundant, while cheap deep
sites become super-additive in groups. A global additive price obtains
\(R^2=-1.284\). The key missing object is a latent state that tells the late network
which early replacements occurred and predicts the cost of combining them.

The frozen layer-5 cut-rank assay is designed to test exactly this: from

\[
H_{ij}=C(P_i\cup S_j),\qquad
\Delta_{ij}=H_{ij}-H_{i0}-H_{0j}+H_{00},
\]

rank 1 or 2 must predict untouched early/late combinations. Its CPU fitter, heldout
boundary, and provenance-safe measurement contract now exist. The latter fixes
row-major ordering of all 64 cells, masks, model/component/program hashes, exact
integer top-1 counts, float64 CE row statistics, one-use call ledgers, and the
terminal per-document bootstrap payload; 8/8 tests pass. What is missing is the thin
GPU adapter that mints authority from the actual committed closure and executes the
64 canonical requests. No mask outcome has yet been observed.

### 4. Independent early-MLP compression has not yet earned composition credit

An existing exploratory runner compares exact MLP0/1/2 factorial effects with PCA0,
PCA1, and exact MLP2 on discovery and heldout documents. Its pure contract tests pass,
but the runner is uncommitted and oracle-scoped. It can answer whether independent
MLP0/1 restrictions retain CE gain and whether exact MLP2 compensates; it cannot prove
that the executable compressed MLP2 or the complete deployed compiler composes.

### 5. Current-token representability fails as an explanation

`token_explained_variance.py` asks whether per-site current-token explained variance
predicts the 34-site replacement-cost table beyond depth. This is a cheap and useful
discriminator between “the table misses contextual variation” and “downstream
sensitivity prices the error.” Its repaired run completed on two roles with maximum
per-site drift 0.0174 and controls passing. All three positive predictions failed:

- explained variance versus cost: Spearman \(+0.466\), versus the registered
  \(-0.70\) target;
- depth versus cost: Spearman \(+0.853\), decisively stronger;
- MLP5 is not the least token-determined site; attention14 is.

MLP1 is the sharp counterexample: it is the most token-determined early site
(0.560 explained fraction) but costs 38.7 points to replace. Attention14 has only
0.104 explained fraction but costs 0.3 points. Thus better local table fidelity can
coexist with greater causal damage. The next instrument must measure what an error
does downstream, not merely its size or representability.

### 6. The correct substitution direction is nonlinear and still does not beat depth

The discovery-only interpolation

\[
y_s(\alpha)=\mu_s(\text{token})+\alpha
\bigl(y_s-\mu_s(\text{token})\bigr)
\]

tests the exact direction in which an ideal current-token table changes site \(s\).
Here \(\alpha=1\) is the live site, \(\alpha=0\) is the empirical token mean, and
\(\alpha=0.9\) is a small local step. The live equality and explained-variance drift
controls pass exactly/within 0.0005, but the registered monotonicity control fails
because several small early steps improve CE slightly.

The scientific predictions are all negative:

- local 10% response versus full response: Spearman \(+0.298\), below \(+0.70\);
- local response versus published cost: \(+0.168\), far below depth \(+0.853\);
- full ideal-table response versus cost: \(+0.851\), essentially tying rather than
  beating depth;
- MLP1, not MLP5, has the largest full CE rise: +0.1935 versus +0.0512 nats.

Thus the actual direction matters more than random noise, but one first-order scalar
per site still does not explain compilation cost. The response is finite-amplitude
and curved. Because the means were estimated and scored on the same discovery rows,
this was an optimistic mechanism test; it cannot establish heldout compiler quality.

## Candidate actions and pruning

### Kept

1. Complete the real adapter/authority/frequency/consumer/gate closure around the new
   canonical observational owner. It directly exposes CE, MLP2 compensation,
   response preservation, frequency failure, and closure for the frozen lattice.
2. The 8-by-8 cut-rank GPU measurement. It makes simplicity predictive: a low-dimensional
   state must forecast unmeasured replacement compositions.
3. The exact-versus-PCA early-MLP factorial. It cheaply tests whether independently
   reduced MLP0/1 and exact MLP2 compose before fitting another joint compressor.
4. A finite-amplitude, downstream-weighted MLP5 subspace. Instead of selecting
   residual axes or using an infinitesimal site scalar, find a
   rank-\(k\) subspace minimizing heldout downstream logit/CE distortion, with matched
   PCA and random-subspace controls. This is the proper next MLP5 test if current-token
   variance does not explain its cost.
5. OOD/extraction/selective-removal tests for candidates that first pass in-domain
   composition.

The running sensitivity sweep has an important interpretation limit. If \(J_s\) is
the downstream Jacobian at site \(s\), isotropic Gaussian noise measures an average
sensitivity proportional to \(\operatorname{tr}(J_s^\top J_s)\). The actual compiler
error \(e_s\) is directional, with local damage closer to

\[
\lVert J_s e_s\rVert^2=e_s^\top J_s^\top J_s e_s.
\]

Multiplying an isotropic sensitivity scalar by RMS error assumes that the compiler
error covariance is isotropic and unaligned with the downstream metric. Consequently,
a failed sweep prunes only the scalar/isotropic sensitivity account, not downstream
sensitivity itself. A pass is evidence for the mechanism but still requires an
error-aligned CE/logit response test; top-1 noise damage alone cannot explain the
remaining 4.72714 CE nats.

The sweep completed during this review, but its registered control failed. Ten-percent
relative isotropic noise changed top-1 by as little as -0.009 percentage points at a
site, so the perturbation did not measurably land everywhere. Sensitivity versus cost
had Spearman \(-0.234\), and sensitivity times scalar RMS error had \(-0.295\), but
neither is a valid negative mechanism result once the landing control fails. The run
prunes this **instrument**—uncalibrated isotropic top-1 noise—not downstream
sensitivity. The next test should use CE/logit change along the actual compiler-error
direction, with amplitude calibrated on separate discovery support.

A 50%-noise CE retry also failed its landing control: the smallest change was 0.0009
nats against the registered 0.005-nat floor. Descriptively, isotropic CE sensitivity
correlates \(+0.464\) with replacement cost, and sensitivity times RMS table error
reaches \(+0.512\); both remain far below depth's \(+0.853\). Because the control is
false, these are not banked mechanism estimates. Together the two runs make a strong
case to stop tuning one scalar per site and measure the compiler-error direction
itself.

### Pruned or deferred

- Fixed-coordinate MLP5 sparsity and simple gain rescue are falsified.
- Additive site prices, one interaction multiplier, and contiguity are falsified.
- A global tensor train remains underidentified: its approximately
  \(8R+44R^2\) parameters cannot be inferred from the current mask count.
- Another local SAE/PCA/HOSVD is deferred unless it predicts a downstream response,
  heldout composition, or executable CE.
- Current-token explained variance is falsified as a cost explanation; it has the
  wrong sign and loses to depth.
- Uncalibrated 10% isotropic-noise top-1 sensitivity is pruned as an instrument: the
  registered perturbation-landing control failed.
- A one-sided 10% derivative along the ideal token-mean error is pruned as a scalar
  cost predictor: it correlates only +0.168 with published cost and does not predict
  the full response. This does not prune finite-amplitude or subspace-valued response.
- OOD transport, selective removal, and MDL pricing remain mandatory validation, but
  applying them to candidates that already fail in-domain composition has low value.

## Priority ranking

1. **Close the real 68-action adapter and final gates.** The canonical observational
   owner is complete; the remaining authority-derived frequency assignment, actual
   batch executor, and 18-consumer norm completer are now the shortest route to final
   causal observations.
2. **Implement the cut-rank GPU adapter and run the 64-cell assay.** The measurement
   contract is complete. This is the highest genuinely new mathematical information:
   it can certify or falsify a two-channel interaction state on untouched combinations.
3. **Audit and run the exact-versus-PCA early composition cube.** It directly
   tests the user's question about whether independent MLP0/MLP1 reductions compose
   and whether MLP2 compensates, at moderate GPU cost.
4. **Learn a finite-amplitude downstream-weighted MLP5 subspace.** The directional
   curve rules out an infinitesimal scalar predictor but not a low-dimensional error
   covariance or nonlinear response surface. Fit on discovery CE/logits and require
   heldout improvement over PCA and random subspaces.
5. **Validate survivors on OOD, extraction, and selective removal.** Use paired
   CE/logit response rather than local MSE. Operational tests decide whether the
   simplification is useful rather than merely compact.

## Action executed during this review

The highest-priority bounded implementation slice is complete at commit `8219f6d2`:
canonical 48-batch aggregation for all 68 actions, typed row CE/copy/frequency
reductions, exact ledgers, backend/receipt binding, and adversarial
skipped/reordered/duplicate/mixed-support tests. The focused suite passes 109 tests.
It deliberately does not fabricate final data, frequency authority, consumer norms,
or scientific outcomes.

The independent cut-rank measurement boundary is complete at commit `99f6c077`.
It freezes all 64 same-wave requests and their provenance through the terminal
per-document bootstrap payload; 8/8 tests pass. It deliberately contains no fitter
and no mask results. The remaining adapter must call the real model exactly once per
canonical request and publish the sealed payload atomically.

The uncommitted early-PCA composition contract's pure tests pass 3/3. It remains
oracle/exploratory and was not promoted or silently added with these changes.

During the review, the repaired token-explained-variance run closed and falsified the
local-fidelity explanation. A registered downstream-sensitivity sweep then entered the
GPU lane while the higher-priority role-owner and cut-rank interfaces continued on
CPU.

That sweep subsequently closed with its perturbation-landing control false. It is
recorded as an invalid instrument result, not evidence against downstream sensitivity.
The larger-noise CE retry also missed its landing floor and remained below depth even
descriptively, so no further scalar/isotropic sensitivity sweep is prioritized. The
GPU is now running the error-aligned interpolation
\(\mu_t+\alpha(y-\mu_t)\), which removes the context-dependent residual along the
actual conditional-mean direction instead of injecting unrelated random noise.

That directional run then closed in 271 seconds. Its small-step response does not
predict either the full response or published replacement cost, the full response
only ties the depth baseline, and MLP1 rather than MLP5 is worst. Exact-live and drift
controls pass, but strict monotonicity fails at tiny amplitudes, so the registered
control is false. This prunes the **first-order scalar** account and moves priority to
finite-amplitude interaction state and subspace-valued response.

No final rows were opened, no scientific outcome was inferred from infrastructure,
and concurrent GPU/job artifacts were left untouched.

## UPDATE END
