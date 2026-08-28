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

A source-closed real batch executor now also exists in `dcc1fa8f`. It deterministically
derives the hash-bound nine-bin frequency plan and joins final context, inherited
programs/bases, one broker/hook, frozen denominators, canonical rows, and the real
program/baseline backends. Its 70-test focused suite passes and it fixed a real bug
that conflated runtime trace identity with semantic action identity.

What remains is production loading/validation of the fit token-count authority and
one-shot final rows, followed by completion of all 18 consumer norms and responses.
Only then can registered objective/transport gates and gauge/SVD/difference-in-
differences/component closure run. Until then the final role remains 0/68.

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
terminal per-document bootstrap payload; 8/8 tests pass. The GPU transaction adapter
now also exists at `ad7fbb22`; its 19/19 tests pass. It fixes one shared program family,
identity gains, no mask-specific refitting, exact 36-native-call/substitution/empty-
gain ledgers, component-tree replay, float64 CE statistics, and receipt-last atomic
publication.

What is missing is the actual bilin18 backend that lazily loads the pinned checkpoint,
constructs the shared context-free/output-nearest-neighbor/rank-64 program from bound
fit rows, hashes its tensors and mask materializations, and executes the hooks. No mask
outcome has yet been observed. Per-mask scalar-gain fitting is excluded from v1 because
it would turn the 64 cells into separate optimization value functions; a separately
named v2 may test that operationally cheap correction without weakening v1's
compositional rank interpretation.

### 4. Independent early-MLP compression has not yet earned composition credit

An existing exploratory runner compares exact MLP0/1/2 factorial effects with PCA0,
PCA1, and exact MLP2 on discovery and heldout documents. Its pure contract tests pass,
but the runner is uncommitted and oracle-scoped. A conceptual audit at commit
`04d52992` gives it a **no-go** for GPU execution. The intervention order, same-row CE
currency, and document-disjoint PCA basis are coherent, and exact MLP2 restoration is
interpretable. However, every PCA correction still calls the full native MLP, so a
pass would show a coupled oracle output-subspace interface—not executable independent
MLP0/1/2 compression.

Launch also requires committed/pushed transitive source closure, exact parent/row
receipt replay before model access, deserialization TOCTOU closure, create-only
publication and lock ownership, finalization after outer return and inertness, runtime
arm/native-call ledgers, and a corrected confidence-interval validator. Its heldout
documents are disjoint but previously exposed by related results, so they are reused
evaluation support rather than pristine confirmation.

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

### 6. The ideal token-mean direction is nonlinear and still does not beat depth

The discovery-only interpolation

\[
y_s(\alpha)=\mu_s(\text{token})+\alpha
\bigl(y_s-\mu_s(\text{token})\bigr)
\]

tests the exact direction in which an **ideal empirical current-token mean** changes
site \(s\). It is not the deployed compiler's length-1 direction; that distinction was
not measured until the following comparison.
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
per site does not explain compilation cost **by the registered standalone rank test**.
The response is finite-amplitude and curved. Because the means were estimated and
scored on the same discovery rows, this was an optimistic mechanism test; it cannot
establish heldout compiler quality.

A post-hoc conditional check prevents over-pruning. In leave-one-layer-out regression,
adding the local response to depth plus attention/MLP type reduces cost-prediction MSE
from 0.01448 to 0.01032 and improves 14 of 17 heldout layers; an exploratory paired
sign-flip test gives \(p=0.0064\). The full response gives only a small, nonsignificant
improvement by the same check. The local CE differences are tiny and this model was
chosen after seeing the outcome, so it is a new hypothesis, not a result. A frozen
heldout-document replication with separately estimated token means and document-level
uncertainty is required.

### 7. The deployed length-1 table and empirical token mean are different objects

The compiler evaluates each site on a one-token sequence; S1840 instead averaged that
site over real contexts for each token. A direct 34-site comparison shows a median
frequency-weighted difference of 0.557 site-output RMS. The largest gaps are MLP4 at
9.920, MLP5 at 2.781, MLP1 at 2.161, and MLP3 at 2.104. Therefore the ideal-mean curve
cannot close the mechanism question for the actual compiler.

The length-1 table's local error correlates \(+0.414\) with published replacement
cost, and the length-1-minus-empirical gap correlates \(+0.472\); both improve on the
ideal error's \(-0.466\) but remain far below depth's \(+0.853\). MLP4 rather than
MLP5 has the largest table mismatch, so this difference does not explain MLP5's
primacy. Controls pass. The cheapest remaining directional falsifier is a curve toward
the actual length-1 table, fit/scored on separated support with document uncertainty.

## Candidate actions and pruning

### Kept

1. Complete the real adapter/authority/frequency/consumer/gate closure around the new
   canonical observational owner. It directly exposes CE, MLP2 compensation,
   response preservation, frequency failure, and closure for the frozen lattice.
2. The 8-by-8 cut-rank GPU measurement. It makes simplicity predictive: a low-dimensional
   state must forecast unmeasured replacement compositions.
3. A heldout curve along the actual length-1 compiler direction, including a frozen
   depth+component-type+local predictor and document bootstrap.
4. Repair the audited exact-versus-PCA early-MLP factorial. It can cheaply test an
   oracle coupled output-subspace interface and exact MLP2 compensation, but cannot
   establish executable independent compression until native MLP calls are removed.
5. A finite-amplitude, downstream-weighted MLP5 subspace. Instead of selecting
   residual axes or using an infinitesimal site scalar, find a
   rank-\(k\) subspace minimizing heldout downstream logit/CE distortion, with matched
   PCA and random-subspace controls. This is the proper next MLP5 test if current-token
   variance does not explain its cost.

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
- A one-sided 10% derivative along the ideal token-mean error is pruned as a
  **standalone rank** predictor: it correlates only +0.168 with published cost and
  does not predict the full response. A post-hoc depth-and-type-conditioned model is
  promising enough to replicate, but cannot be credited on the discovery sample.
- The empirical token mean is pruned as a proxy for the deployed length-1 table: their
  median difference is 0.557 output RMS and reaches 9.920 at MLP4.
- OOD transport, selective removal, and MDL pricing remain mandatory validation, but
  applying them to candidates that already fail in-domain composition has low value.

## Priority ranking

1. **Close the real 68-action adapter and final gates.** The canonical observational
   owner is complete; the remaining authority-derived frequency assignment, actual
   batch executor, and 18-consumer norm completer are now the shortest route to final
   causal observations.
2. **Implement the fixed-program bilin18 backend and run the 64-cell assay.** The
   measurement contract and GPU transaction adapter are complete. This is the highest
   genuinely new mathematical information:
   it can certify or falsify a two-channel interaction state on untouched combinations.
3. **Run a heldout actual-length-1 directional curve.** S1840 used the wrong compiler
   object for this purpose. Separate table fitting from scoring, retain per-document
   CE, and test the frozen depth+type+local model as well as the nonlinear full curve.
4. **Repair, then run, the audited exact-versus-PCA early composition cube.** The
   current runner is no-go: it needs provenance/lifecycle/call-ledger repairs, and a
   pass would only establish an oracle output-subspace interface because it still
   calls the native MLPs. It remains useful for asking whether exact MLP2 compensates,
   but not yet for claiming executable independent compression.
5. **Learn a finite-amplitude downstream-weighted MLP5 subspace, then validate
   survivors on OOD, extraction, and selective removal.** Fit on discovery CE/logits
   and require heldout improvement over PCA/random subspaces before operational tests.

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

That transaction adapter subsequently closed at commit `ad7fbb22`, with 19/19 tests
passing. It enforces identity gains and no mask-specific refit so the measured cut rank
continues to mean composition of one fixed program family. The remaining component is
`compilation_mask_cut_rank_v1_bilin18_backend.py`; historical S1834 code cannot be
imported because it interleaves outcome analysis and per-mask sequential gain fitting.

The uncommitted early-PCA composition contract's pure tests pass 3/3. It remains
oracle/exploratory and was not promoted or silently added with these changes.

A separate conceptual audit closed at commit `04d52992` with 12 tests passing and a
no-go verdict for GPU execution. It did not edit or stage the concurrent runner. The
audit confirms coherent intervention math but identifies incomplete source/lifecycle
closure, previously exposed heldout support, and the decisive semantic limitation
that every projected correction still executes the full native MLP.

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
control is false. This prunes the **standalone first-order scalar** account. A post-hoc
leave-one-layer-out check suggests that the local response may still add information
conditional on depth and component type; that unregistered finding is queued only as
a frozen heldout replication, not as scientific credit.

S1841 then found no sharp nonlinear knee: only 41.1% of MLP5's damage occurs below
\(\alpha=0.2\). Five sites spanning 9x total damage nevertheless share a normalized
curve shape, with half-damage points between 0.21 and 0.31. The measured compiled-layer-0
interaction is only 1.31x, so the earlier proposed cause of MLP5's special S1834 cost
was corrected rather than retained.

Finally, the length-1-versus-empirical comparison shows that S1840's direction is not
the deployed compiler direction. The two tables differ by median 0.557 output RMS;
the gap correlates +0.472 with cost but is largest at MLP4, not MLP5. This redirects
the next cheap directional assay to the actual length-1 table on separated support.

No final rows were opened, no scientific outcome was inferred from infrastructure,
and concurrent GPU/job artifacts were left untouched.

## UPDATE END
