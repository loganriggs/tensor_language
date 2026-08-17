# HARNESS: an architecture-agnostic replacement harness

Engineering spec for scaling the replacement benchmark (BENCHMARK.md, Track 2)
beyond bilin18 to arbitrary / larger transformers. Companion skeleton:
`harness_skeleton.py`. The design encodes the instrument lessons paid for in
BILIN18_CONNECTION.md §§104-106 (lambda-mixing capture/apply mismatch; mlp hooks
that never fire in a manual forward) and §§155-160 (the reference ladder,
sequential refit as the frontier lever, joint-only scoring).

## 1. The single-traced-forward contract

Every measurement runs through ONE forward implementation per model, the
"traced forward", which does three jobs in the same code path:

- computes the baseline model;
- APPLIES stand-ins from an explicit `assignment` dict at the exact site where
  the real component's output would be added;
- CAPTURES a requested component's (input, true output) pair, in the exact
  view the stand-in will later consume, under whatever hybrid is installed.

Hard rules, each one a bug we actually shipped:

- **Capture and apply in the same forward.** Never fit on activations recorded
  by one code path and apply the stand-in in another. §105: block input was
  captured before lambda-mixing while stand-ins were fit on post-mixing
  activations; every stand-in was applied to an input it was never fit for, and
  a +0.029 layer masqueraded as a +1.51 catastrophe.
- **No module hooks with a manual forward.** If the traced forward re-implements
  the computation from weights (as it must, to expose intervention sites), hooks
  registered on the original modules silently never fire (§106: an entire
  interchange arm was void). Hooks are permitted only on modules the traced
  forward itself calls, and the self-tests must exercise them; default is no
  hooks at all.
- **Capture returns the TRUE component output** (the component as the model
  computes it on the hybrid input), even when a stand-in is installed at that
  site — this is what sequential refit fits against.
- One assignment format everywhere: `{component: {'kind': 'full'|'const'|
  'linear', 'W','bx','by','rank'}}`. `'full'` must be representable so the
  identity self-test can exercise the assignment plumbing itself.

## 2. Mandatory self-tests (auto-generated, run per model before any experiment)

`ReplacementHarness.self_tests()` is generated from the interface alone and is a
gate: no experiment result is reportable for a model whose adapter has not
passed all of these on the current code.

1. **Identity test.** `assignment = {c: full}` reproduces the empty-assignment
   CE to fp tolerance (<= 1e-6 nats). Exercises the assignment plumbing without
   changing the function.
2. **No-op test.** Empty assignment == raw model: the traced forward's baseline
   CE matches an INDEPENDENT reference implementation of the model (the training
   or original inference code path) to <= 2e-3 nats on the same batches. This is
   the test that catches a wrong traced forward outright.
3. **Mean-ablation cross-check.** A constant-mean stand-in installed via the
   harness matches a reference mean-ablation implemented through a different
   code path (for bilin18: the span-ablation `PATCH` machinery with Q = I) to
   <= 2e-3 nats, with both arms sharing the same mean vector. Catches
   wrong-view application (the §105 class) directly.
4. **Gain-freeze free at zero damage.** The gain-frozen regime (§4 below) with
   an empty assignment matches the free-norm baseline to <= 1e-3 nats. A frozen
   regime that costs anything at zero damage is measuring its own bug (the
   §108 control, promoted to a standing test).
5. **Fit cross-check** (when the adapter wraps existing fit code). The generic
   ridge fit reproduces the model-specific reference fit's map on identical
   captures (max |dW| small). Guards against silent divergence between harness
   and legacy results.

Tolerances are frozen in the harness, not per-experiment.

## 3. Fit protocol: sequential front-to-back refit (default)

The default fit for any multi-component assignment is `refit_sweep`: iterate
components in model order; fit each stand-in (ridge on captured (X, Y), then
SVD-truncate to the assigned rank) on the model WITH ALL UPSTREAM STAND-INS
ALREADY INSTALLED. §158: same architecture, refit vs naive = +1.66 vs +2.68
nats — 36% for free. Naive (fit-on-intact) is available only as an explicitly
labeled ablation arm. Constants are rank 0 of the same fit (by = captured
output mean under the current hybrid).

## 4. Scoring

- **Joint-only.** The score of an assignment is the held-out CE of the model
  with ALL its stand-ins installed at once, minus baseline. Summing
  per-component costs is forbidden as a score (composition drift, §104/§157 —
  modest on clean instruments but real). Per-component numbers may be reported
  only as diagnostics, labeled as such.
- **Both norm regimes shipped.** Every headline joint CE is reported in the
  free-final-norm regime AND the gain-frozen regime (final normalization's
  per-token scale clamped to the clean run's, via a lockstep clean+hybrid dual
  forward). §§116-117: loud components can hide content loss behind final-norm
  compensation; the pair of numbers, not either alone, is the result.
- **Greedy search pattern** (`greedy_allocate`), for finding assignments under
  a budget without O(components x ranks) exact rescoring:
  1. cache one rank-64 refit map per (component, upstream-assignment) key;
  2. estimate each candidate (component, rank) marginal by TRUNCATING the
     cached map to rank r and scoring on the quick eval subset (approximate
     marginal — never reported);
  3. on accepting the best candidate, EXACT joint rescore on the full eval set;
     acceptance is decided on the exact number only;
  4. staleness: the cache key includes the upstream component set, so any
     change upstream of c automatically invalidates c's cached map and forces a
     refit (the fit distribution changed — §104's drift mechanism).
- **Parameter accounting** at the balanced gauge point
  (`../balanced_gauge_spec.md`, WP1 only). Stand-ins: const = d, rank-r linear
  = 2dr. Retained full components are counted after `balance_bilinear` (per
  unit, rescale a_i, b_i, c_i to the geometric-mean norm m_i; zero dead units):
  raw counts are gauge-invariant, but any rank/sparsity/threshold-based credit
  for a kept component is meaningful only at the balanced point — per-unit
  rescalings can game norm thresholds in any other gauge. Report stand-in
  params and total (stand-ins + balanced live-unit count of kept components)
  separately.

## 5. Model-specific vs generic

**Model-specific (one adapter per architecture, ~100 lines):** the traced
forward implementing the `TracedModel` interface — `forward_with(idx,
assignment, capture=None, gain_frozen=False)`, `components()`, `n_layers`,
`d_model`, the independent reference forward for self-test 2, the reference
mean-ablation for self-test 3, `component_params` (balanced-gauge count), and
the train/eval batch iterators. This is the only code that knows about
lambda-mixing, RoPE, norm placement, attention structure, or where a component's
output enters the stream — and it is exactly the code the self-tests validate.

**Generic (written once):** everything else — CE evaluation, capture plumbing,
ridge fit + rank truncation, sequential refit sweep, joint scoring in both
regimes, greedy allocation with map caching/staleness, parameter accounting,
and the auto-generated self-test battery.

**Scaling:** the harness is embarrassingly parallel where it is expensive —
eval batches shard across GPUs; greedy candidates' approximate marginals are
independent forwards; cached maps are per-(component, upstream) and shareable
across search branches. Nothing in the generic layer holds model internals, so
larger models cost only their forward. Porting cost per new architecture: one
adapter plus one green self-test run.
