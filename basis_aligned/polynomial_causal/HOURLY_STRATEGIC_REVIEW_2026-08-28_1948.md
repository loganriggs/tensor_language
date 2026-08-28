# Hourly strategic review — 2026-08-28 19:48 UTC

## UPDATE PART

### What is actually explained

There is no honest single percentage for “the model explained,” because three ledgers
measure different things:

- Structural replacement coverage is 36/36 attention/MLP sites, but this only says a
  surrogate exists at every site. It does **not** say the model's computation is
  understood.
- Certified whole-program storage removal remains 5.3481%.
- The older broad behavioral account is 32.1% ± 6.4%, while the strict named causal
  cross-entropy account is 10.923%, leaving 4.72714 nats unexplained under that ledger.
- The frozen final action ledger remains 0/68. No final-role response or OOD claim has
  been opened.

The best current local result is still the MLP0 rank-64/sparse code, which reconstructs
MLP0 well and retains about 98.3% of its standalone CE effect. The central unsolved
fact is that locally good replacements compose badly. Therefore local reconstruction
cannot be converted into a whole-model explained fraction.

### New composition evidence

The count-versus-contiguity test used three sets with exactly 12 compiled sites:

| set | compiled layers | normalized gap recovery |
|---|---|---:|
| TOP13 | 13–17, contiguous | 40.6% |
| SCATTER_ODD | 9, 11, 13, 15, 17 | 44.0% |
| SCATTER_EVEN | 8, 10, 12, 14, 16 | 40.1% |

Making the set non-contiguous changes recovery by only 1.5 percentage points against
the registered 10-point bar. Contiguity is therefore not the missing interaction
variable. The 3.9-point difference between the two scattered sets is more consistent
with depth: the worse set reaches one layer earlier.

This complements the earlier facts:

- replacing either layer-1 attention or layer-1 MLP almost reproduces the entire
  layer-1 pair loss, so those two failure routes are nearly redundant;
- a deep site is cheap alone but deep sites accumulate super-additively in groups;
- count, depth, contiguity, and independent local error each fail as a complete
  simplicity price.

The running GPU job is now measuring all 34 non-layer-0 sites separately and testing
the falsifiable law

\[
  C(S) \approx \alpha \sum_{i\in S} C(\{i\}),
\]

where \(C(S)\) is the behavioral cost of compiling site set \(S\). There are 36 local
site prices including layer 0 and only one fitted interaction multiplier \(\alpha\).
The test is against seven multi-site sets, with a preregistered \(R^2\ge 0.90\) bar.
If it passes, the table predicts the cost of a candidate program without running every
combination. If it fails, interactions depend on site identity and a single global
multiplier is not a useful simplicity measure.

Unit audit: the running log's 39.32% is absolute live-model top-1 accuracy. The often
quoted 64.8% is B0's normalized recovery between the fully compiled baseline (13.55%)
and live model (39.32%). The current B0 value 30.25% gives
\((30.25-13.55)/(39.32-13.55)=64.8\%\), so the controls are comparable.

### Highest-priority action executed

While the GPU measures the site table, the causal paired-response backend was advanced
from separate teacher/student primitives to one atomic batch transaction:

- exactly 3 shared exact-teacher forwards are run: baseline, positive edit, negative
  edit;
- each of 22 candidate simplified actions receives the same three interventions,
  giving 66 student forwards and 69 total forwards;
- MLP1-code response, centered-logit response, and output-KL response are reduced while
  tensors are private;
- every reduction binds the hashes of the actual forwards that produced it;
- no batch receipt exists unless all 69 forwards and the broker ledger close.

The affected suite passes 78/78 tests. This removes the major missing *measurement
interface*: we can now ask whether a simpler early-MLP program transports the causal
effect of an MLP0 edit into MLP1 code and outputs, rather than merely matching ordinary
activations. It is not a scientific result yet; the final role remains sealed.

### Largest remaining gaps

1. The 48 atomic batches are not yet accumulated into a single mandatory run receipt,
   so the registered 144 teacher and 3,168 student forwards cannot yet open final
   statistics.
2. We do not yet know whether any simple site-cost law predicts composition. The GPU
   job directly tests the simplest surviving law.
3. We do not know whether the MLP0 code transports the right causal responses through
   MLP1 and logits, especially OOD.
4. We have no demonstrated independently simplified MLP0/MLP1/MLP2 composition that
   preserves CE and interventions simultaneously.
5. The final 68-action causal ledger, selective-removal tests, collateral-effect tests,
   and OOD transport remain unopened.

## Pruned and ranked next actions

1. **Finish and audit the all-site cost table.** Highest immediate information gain;
   it decides whether one cheap predictive simplicity price survives. The job is
   already running, and its old-result controls are explicit.
2. **Add the ordered 48-batch response accumulator and terminal ledger.** This is the
   shortest route from infrastructure to a falsifiable causal-transport result. It is
   CPU-side and nonredundant with the GPU job.
3. **Run the frozen paired-response experiment, then choose a joint early-MLP model.**
   Favor models that preserve MLP1-code and output responses, not the one with the best
   local MLP0 MSE.
4. **Fit interaction structure only if one multiplier fails.** The next candidate is a
   low-rank pairwise cost matrix or layer-band/DAG factor, cross-validated on held-out
   site sets. Do not jump directly to a large unconstrained interaction model.
5. **Evaluate composed programs by prediction, OOD transport, extraction, and selective
   removal.** This is where MDL/executable size becomes validated simplicity: a smaller
   representation is useful only if it improves prediction of unseen interventions or
   makes edits more selective at matched behavior.

Not pursued now: another standalone SAE/HOSVD fit, a contiguity-aware compiler, or a
larger local reconstruction sweep. They duplicate completed local work or optimize an
axis already shown not to predict composition.

Production remains NO-GO. No new model-explained percentage is claimed by this review.
