# Hourly circuit review — 12 September 09:36

Written at the first safe review boundary after the large audit. Targets remain:
explicit reads/operations/writes; cross-boundary grouping and within-module
splitting; held-out/OOD prediction; extraction with declared ports; selective
removal; composition/reuse; and stable identification. The objective is a simpler
executable model satisfying the four behavioral properties, governed by the
bilinear handoff and later weights-first corrections. No new circuit is promoted.

## What changed

The common MLP15 interface now has two locally converged, nearly identical fits.
Native signed-effect magnitude fidelity still fails. Sixteen rotation planes
find no meaningful escape; the rank-bound gap remains too large for a global
certificate. The original time-limited fit miss remains recorded.

Exact coefficient contractions now handle every background/producer degree
after substituting the repeated MLP15 input. Fixed-frame rescoring does not
resolve a large change from the formal metric. Direct exact-metric gradients
are noisy; an analytic same-pairing correction reduces observed noise by28%
at the small budget but does not produce useful descent. A deliberately larger
two-replica65,536-probe audit also finds no resolved gradient agreement or
improving held-out step. Independent block statistics account for essentially
the full observed gradient norms. This does not prove exact stationarity.
See [primary math and receipts](MIXED_REPEATED_CONTRACTION_V1_MATH.md).

## Direction decision

Stop escalating this same-frame metric search. A long stochastic fit initialized
here is unsupported by the larger audit. Exact one-slot conditional integration
has CPU value/gradient controls, but native cost is unmeasured; it does not earn
another run merely because it is available. Return to the representation question:
shared nonlinear intermediate computations without requiring every branch to
pass through one small linear input bottleneck.

Prior-art checks confirm that generic two-form pencil blocks, projected Gram
rewrites, separate full-input spectral intermediates, and coupled learned
quadratic products already have receipts. They are not new method proposals.
The MLP index has no standalone MLP15 dossier; late-readout records and the recent
MLP15 source/interface receipts provide its current coverage. Missing consolidation
does not imply an unexplored component.

Before opening another fit, the next mathematical test asks whether proposed
oblique maps or native-atom selectors actually change the approximating function
class. The [executed dependency-projection control](DEPENDENCY_PROJECTION_DOMINANCE_V1_CONTROL.json)
already rejects the simple oblique-map variant as a new formal fixed-subspace
solution. It verifies Pythagorean error splitting within9.23e-16. The next
representation comparison must move beyond that restriction and retain the exact
packed parent/partner program as its execution reference. Repeating the old
nonlinear hierarchy fits unchanged is not justified either.

## Why changing a linear map may not change the method

Let $T$ be the output-weighted homogeneous coefficient tensor and let $Q$ be the
orthogonal projector onto an allowed input subspace. A polynomial coefficient
tensor $S$ that depends only on that subspace obeys $S=Q^{\otimes p}S$. Then

$$
\|T-S\|^2=\|T-Q^{\otimes p}T\|^2
+\|Q^{\otimes p}T-S\|^2.
$$

Thus exact coefficient projection is already best for this fixed dependency
subspace, even against arbitrary coefficients inside it. In particular,
$f(Mx)$ for an oblique rank-k map depends only on the rowspace of $M$ and cannot
beat the coefficient projection onto that same rowspace. The dense control uses
two coupled quartic outputs, folded output writers, and eight arbitrary maps.
This is an orthogonal-projection identity, not a claim that the locally fitted
subspace is globally best. It also does not apply unchanged after imposing
repeated native polynomial inputs, nonlinear features or a data-weighted metric.

## Confounds and priorities

Formal independent sources, external RMS/background ports, repeatedly inspected
construction families, output-prior selectivity confounds, and differences between
write levels and signed changes remain material. No text-derived weights or
repaired thresholds were introduced. Gradient noise is no longer treated as
evidence for a large hidden descent direction; the larger sample-size forecast
from only two small replicas was explicitly uncertain and did not materialize.

Priority order: (1) distinguish nonlinear intermediate representations from
already optimized linear-subspace variants; (2) choose a representation with an
explicit shared operation and falsifiable frozen effect test, checking existing
hierarchy receipts first; (3) revisit exact-metric variance reduction only if that
new representation supplies a concrete optimization need. More corpus collection
or outcome-guided fitting remains lower priority under the user's instruction.

## Throughput and workflow gates

[Phase occupancy](HOURLY_PHASE_ACCOUNTING_2026-09-12_0936.json): implementation
24.32minutes, science14.62, review16.40, publication4.67. These are timestamp
categories, not exclusive labor/GPU time. Native direction pilots took14.6 and
14.8seconds; the deliberate large audit took436seconds. CPU derivations and
controls ran alongside it. These are tests of one candidate and its machinery,
not a count of newly identified circuits.

- CIRCUIT_FOCUS: PASS WITH REDIRECTION. Exact composed-input machinery and
  extraction-failure interpretation improved. The common-interface path has
  reached its current decision boundary; further same-frame estimator tuning
  would drift away from identifying a useful computation.
- CEREMONY_BUDGET: PASS WITH LIMITATION. Science plus implementation39minutes
  exceeds review plus publication21minutes. Fine-grained validation is not
  separately recoverable. Keep primary math in one file and use short pointers.
- NOVELTY_LESSON_GATE: PASS. Existing pencil, hierarchy, congruence, module-index
  and source receipts were checked. The executed projection identity prevents a
  redundant oblique-map experiment. No global or semantic claim from local fits.

Next hourly10:36; mathematical review11:00. The full goal remains active. The
claimed dependency-projection CPU test was executed as the immediate review action.
