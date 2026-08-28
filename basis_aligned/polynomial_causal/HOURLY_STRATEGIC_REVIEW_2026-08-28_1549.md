# Hourly strategic review — 2026-08-28 15:49 UTC

## Executive summary

The strongest new result is a matched MLP1/MLP2 negative: neither physical MLP has a
small ordinary coefficient-space Tucker structure. MLP2 is only marginally more
concentrated than MLP1 and remains full rank in every measured mode. This is evidence
against the wrong simplicity metric, not against behavioral compression.

The main mathematical direction is now an empirical fourth-moment test of MLP1. In
plain language, it will weight a polynomial error by how often natural model states
actually exercise that error. A source-closed preregistration and execution addendum
now specify 100k/200k/400k data doubling, independent validation and replication,
native MLP1 states, executable prices, matched controls, and a Gaussian/Wick shortcut
that is usable only if it agrees with held-out empirical geometry.

The whole-program stream also produced a useful simplicity result. Full-rank token
tables are off the measured cost/fidelity Pareto frontier: a rank-16 program for the
top eleven layers used 34M fewer stored reals and gained 5.16 percentage points over
the all-sites full-rank program. The current rank-64 all-sites design was **not**
dominated in that sweep. A finer rank/depth frontier completed after this review was
drafted; its result is recorded below.

## What fraction of the model is actually explained?

There is no single honest percentage because the denominators answer different
questions:

| ledger | value | interpretation |
|---|---:|---|
| executable structural coverage | 36/36 sites | every site has a runnable surrogate; this is not semantic understanding |
| certified whole-program storage reduction | 5.35% | rank-640 attention passed its existing predictive/causal contract |
| older named-behavior coverage | 32.1% ± 6.4% | human-labeled behaviors under that older ledger |
| strict named causal recovery | 10.923% | recovered named intervention effects under its own denominator |
| strict recovery of the current +0.8976 CE ship gap | 0% | no composed replacement has earned whole-model credit |

The last line is the controlling one for “fully reverse engineered.” We can execute
all 36 replacements, but we cannot yet predict and preserve the native computation
through their joint interfaces.

## Largest remaining gaps

1. **Natural MLP complexity is unknown.** Coefficient energy is diffuse, but natural
   inputs may occupy a much smaller non-Gaussian subset. We have not yet measured the
   authoritative empirical fourth-moment loss for MLP1 or MLP2.
2. **Early MLP interfaces do not compose independently.** Compressing MLP0 changes
   what MLP1 writes, and MLP2 is harmful alone but helpful after upstream changes.
   Bottom-up compilation failures survive norm matching, so this is not a scalar
   calibration problem.
3. **Residual CE remains unexplained.** The current fully composed replacement still
   recovers 0% of the strict +0.8976 CE gap despite useful local and named results.
4. **No certified semantic code.** Shared lexical structure in MLP0 is descriptive;
   no overlapping lexical dictionary with sparse downstream readers has yet beaten
   matched continuous alternatives on extraction, editing, OOD, and composition.
5. **OOD and edit claims remain downstream of admission.** No new local factorization
   may receive OOD, removal, or causal credit before it survives natural validation,
   live-suffix transport, and the joint MLP cube.

## Candidate actions considered and pruned

- **Empirical fourth-moment product compression:** retained. It has the highest
  information gain because it directly distinguishes “Euclidean tensor is diffuse”
  from “the used part of the tensor is simple,” is falsifiable by held-out loss and
  data doubling, and can price CP, native-gate, Tucker, Down, affine, and random
  programs in the same currency.
- **Conditional top-down MLP refitting:** retained. Directional compilation and the
  MLP0/1/2 interaction make it causally necessary for whole-model composition.
- **Rank/depth whole-program frontier:** retained. It directly tests
  whether a simpler program enables better fidelity at equal cost; it has already
  removed full-rank tables from the frontier.
- **Direct CP in coefficient Frobenius:** deferred. CP is logically open, but two
  diffuse HOSVD results provide no positive reason to spend optimization budget in
  the wrong metric.
- **More native-gate subset sizes:** pruned. Stable MLP1 supports already failed the
  registered predictive/harm gate at 32/128/512.
- **Coefficient HOSVD across more MLP layers:** pruned for now. MLP1 and MLP2 give the
  same qualitative result; repeating it does not attack CE, composition, or the
  natural metric.
- **Norm minimization followed by HOSVD:** not a separate functional-tensor solution.
  Scalar gate balancing is already fixed before the Down diagnostic, and the folded
  tensor itself is invariant to internal factor gauges. Canonicalizing checkpoint
  factors may aid readability or optimization, but cannot change the measured
  folded coefficient spectrum. It becomes useful only when paired with a new
  activation/consequence metric or a different executable grammar.
- **Lexical SAE alone:** deferred. Overlapping features such as “city” and
  “capitalized” are plausible, but feature sparsity is useful only if MLP1/2 readers
  are sparse in the same code and whole-model interventions preserve unrelated
  behavior.

## Ranked top five

1. **Implement the CPU-only empirical-moment machinery and outcome-blind row
   freezer.** Highest information gain, no current GPU dependency, exact
   falsification gates, and directly targets the surviving mathematical hypothesis.
2. **Run the no-optimizer MLP1 projection and empirical-versus-Wick discriminator.**
   It cheaply decides whether literal low-dimensional input support or a Gaussian
   surrogate is viable before any long CP optimization.
3. **Fit early MLP replacements conditionally and evaluate the complete 8-cell
   MLP0/1/2 cube.** This is the shortest route from local compression to causal
   composability and explicitly measures pair/triple interactions.
4. **Replicate and choose an operating point on the rank/depth Pareto frontier.**
   The discovery sweep now covers ranks 4/8/16/32/64 and four live-prefix depths,
   using both storage and top-1 fidelity rather than local reconstruction alone.
5. **Only after the empirical metric is validated, compare learned CP with
   gauge-standardized native gates, Tucker, Down, and affine controls.** This is the
   expensive stage; it should inherit the 100k/200k/400k and untouched-replication
   gates rather than begin with an unconstrained optimizer search.

## Highest-priority actions executed this hour

1. The MLP1 empirical-moment discriminator was prospectively specified and pushed.
2. A separate readiness audit found underspecified partial-window, native-trajectory,
   cache, PCA, probe, and bootstrap rules before any data were opened. The execution
   addendum closes each ambiguity and was pushed.
3. The independently implemented MLP2 coefficient diagnostic was source-frozen,
   audited before execution, run once, audited afterward, committed, and pushed.
   Its ranks closely match MLP1 and prune dense coefficient-HOSVD/Tucker for MLP2.
4. The whole-program partial frontier completed after two preserved implementation
   failures. It removed full-rank tables from the Pareto frontier while retaining the
   current rank-64 all-sites point; the finer frontier has now completed.

There is no FineWeb, cache, `rspd`, or checkpoint blocker. The immediate blocker to a
large empirical MLP1 run is procedural and now narrow: stage-1 CPU math/tests and an
outcome-blind 6,252-document freezer must be implemented and audited before native
activations are captured. Existing activation rows are exclusion evidence only, not
reusable observations. Expected float32 input/write storage is about 7.37 GB for FIT
plus VALIDATION and 11.06 GB once REPLICATION is licensed.

## Late result: the finer rank/depth frontier

The 20-arm discovery sweep finished in 486.7 seconds. At every fixed depth, rank 64
had the highest accuracy on all three roles, but most of the gain saturated much
earlier. At depth 10, rank 8 was only 0.41/0.65/0.22 percentage points below rank 64
while using 265.61M rather than 270.76M stored reals. The preregistered claim that
rank 8 would lose at least one point therefore failed.

The registered scalar efficiency rule selected the **all-sites rank-8** arm rather
than any partial compile, so the “partial compile is the efficiency optimum”
prediction failed on all three roles. This does not dominate rank-64 all-sites:
rank 8 is cheaper and less accurate, so both remain Pareto points. It also does not
establish a deployable optimum because the sweep was explicitly discovery-only and
the scalar efficiency ratio depends on its chosen baseline and units. The useful
conclusion is narrower: rank and compiled depth are genuinely separate price knobs,
full-rank tables are unnecessary, and rank 8--16 deserves a preregistered held-out
comparison against rank 64.
