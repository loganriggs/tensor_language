# Hourly strategic review: turn MLP1 response sparsity into a falsifiable program claim

Date: 2026-08-28 13:35 UTC

## Executive update

There is no infrastructure blocker. `origin/main` is current through the registry-fresh
MLP1 row freeze, the RTX 5090 is idle, no relevant queue entry or model process is
running, and FineWeb, the checkpoint, the rank-640 program, and its causal and
predictive parents are locally available.

The scientific blocker is narrower: we do not yet know whether MLP1's 4,608 native
bilinear products admit a *shared small physical support* that preserves their effect
on the complete downstream model. The earlier local-frame experiment cannot answer
this. It found moderately low response rank in each context, but the frames did not
repeat across independent probes. A physical product gate, in contrast, is an
executable object shared across every token position, so success would directly yield
a candidate smaller MLP rather than another gauge-dependent coordinate system.

This review executed the highest-priority CPU closure. It froze 32 registry-wide fresh
documents, an explicit row-use boundary, and an exact $2\times2$ gate-response plan:
16 fit documents, 16 untouched validation documents, and two independent 32-probe
halves. It also replaced unconstrained least squares by one registered float64
SVD/Tikhonov solver with conditioning and coefficient-size rejection. The 28 focused
row, plan, graph, response, and numerical tests pass. No model response has been
computed and finite gate scaling remains unauthorized.

## What fraction is actually explained?

These are different denominators and must not be collapsed into one percentage.

| claim currency | current evidence | unresolved part |
|---|---:|---:|
| structural tensor ownership | 36/36 sites and 545,904,054/545,904,054 standalone values | no structural unknown, but ownership is not semantics or minimality |
| prospectively admitted whole-program storage | 516,707,766 values at rank 640 | removes 29,196,288 values, or 5.348245% |
| dense MLP-bank storage | 286,675,200 values, 52.513843% of dense model | no admitted MLP gate removal yet |
| named semantic behavior | 32.1% $\pm$ 6.4% | about 67.9% remains unnamed |
| strict named causal recovery | 0.57968/5.30682 = 10.923% | 4.72714/5.30682 = 89.077% remains outside named strict circuits |
| rank-640 expanded behavioral gates | 5/6 | top-token identity fails: 95.782/96.077% versus the frozen 98% requirement |

Rank 640 remains useful but bounded evidence: cross-task CE harm is
$+0.005532/+0.004449$, live top-1 accuracy retention is 99.437/99.673%, and its
prospective causal-recovery mean is 0.94442 with lower bound 0.92726. It is not an
exact behavioral reconstruction because argmax identity fails.

## Exact new object

For context $c$, categorical-Fisher probe $a$, MLP1 product gate $n$, and token
position $q$, the measured response is

$$
E_{c,a,n}
=\left.\frac{\partial s_{c,a}}{\partial\alpha_{c,n}}\right|_{\alpha=1}
=\sum_q h_n(z_{c,q})\,d_n^\top g_{c,a,q}.
$$

$h_n(z)=(\ell_n^\top z)(r_n^\top z)$ is the native bilinear product, $d_n$ is its
Down column, and $g$ is the gradient arriving from the complete remaining transformer.
One scale $\alpha_{c,n}$ is shared across all positions. Thus each column of $E$ is a
real physical intervention on one runnable product gate, not an arbitrary latent
direction.

Two notions of simplicity are tested separately:

1. **Response-span simplicity (CSS):** can $K$ physical columns reconstruct the full
   response matrix using coefficients fit only on fit-wave/probe-half A?
2. **Executable all-on simplicity:** can those same $K$ gates, with one frozen
   coefficient each, reproduce the tangent effect of all 4,608 native gates?

The second is strictly closer to compression. Neither alone licenses finite removal.
The proposal must transfer without refitting to fit-half B and both fresh validation
halves and beat response energy, activation-times-Down norm, a gauge-canonicalized
factor-product derangement, and hash-random supports. This tests whether the definition of simplicity
buys prediction and composability beyond mere low training error.

## Largest remaining gaps

1. **MLP1 physical interface:** no admitted small support or shared gate dictionary.
2. **Finite validity:** a tangent approximation may fail at even 10% movement because
   RMSNorm, attention, later bilinear MLPs, and residual composition curve the response.
3. **Interactions:** independently good gate packages may collide through quadratic
   and normalized suffixes; no sparse interaction law is admitted.
4. **Whole-model behavioral residual:** rank 640 still changes roughly 4% of top-token
   decisions and named strict circuits explain only 10.923% of the measured causal
   denominator.
5. **OOD and editing:** current fresh FineWeb validation is registry-fresh, not a new
   corpus or task, and no MLP simplification has passed extraction, selective removal,
   collateral, or causal-bank transport.

## Pruned candidate actions

- Repeating the 32-direction local-frame experiment is low information: independent
  halves already reject frame stability.
- Token clustering, feature naming, and coordinate correlations are premature because
  they can be gauge-dependent and do not imply a runnable replacement.
- A larger local-MSE predictor is redundant with failed compiler families and does not
  test the complete suffix.
- Full hard deletion before tangent calibration is unsafe and scientifically ambiguous.
- More attention-only rank sweeps are lower value because attention already owns the
  only admitted storage reduction while dense MLPs dominate the remaining price.
- Prequential MDL is useful only after two executable families have comparable causal
  utility; it cannot rescue a family that fails transfer or composition.

## Ranked next five

1. **Source-close and execute the frozen MLP1 global physical-gate assay.** Highest
   information gain and causal relevance: it directly distinguishes a small shared
   runnable support from merely context-local low rank. It is sharply falsifiable,
   uses fresh held-out documents, and costs about 512 backward passes, a small GPU run.
2. **Native quadratic-form/Gram audit if the support assay fails.** Use bilinear and
   polynomial structure to test whether the irreducible object is a low-rank joint
   activation--Down metric rather than a sparse physical subset. CPU analysis of the
   response bank is cheap and preserves gauge invariance.
3. **Registered $\epsilon=0.1$ candidate-path calibration if a support promotes.** Move
   every omitted gate toward zero and retained gates toward their fitted endpoint;
   require signed tangent prediction to match observed Fisher/KL. This is the first
   causal test that the simpler object survives transformer curvature.
4. **Eight-package quadratic/Möbius interaction law.** Partition the promoted support,
   measure singleton and pair interventions, and test whether a sparse degree-two
   Volterra program predicts compositions. This targets editability and selective
   removal rather than isolated reconstruction.
5. **One-site executable frontier plus genuine-OOD and causal-bank transport.** Compare
   native hard retention and refitted-Down families at complete storage/compute prices,
   then run CE, accuracy, argmax, causal fixtures, coverage strata, and a genuinely new
   corpus/task. Only this can promote response simplicity into whole-model compression.

## Executed action and remaining launch boundary

The fresh row receipt is committed at `39398ff3`. The new row-use authority has file
SHA-256 `9177ca13727e268d4d7ea492d832296b4853ead5b6b4764c4a11444ef3f3b40f`; the
serialized plan has file SHA-256
`4eefcc28ec3ed9fda09b047bb122aa47bc314e29f6a3857bc1da541bf7f5f8b1` after the
independent mathematical audit repairs.
The production collector is intentionally not yet authorized. It must bind these exact
bytes, the fixed `[context, probe, gate]` response axis, the admitted rank-640 buffers,
the gauge-canonicalized zero-leaf factor-product derangement, create-only result and bundle
namespaces, a run lock, and a complete committed source closure. Independent re-audit
of the repaired numerical contract is in progress; any fatal finding is resolved
before committing or opening GPU outcomes.
