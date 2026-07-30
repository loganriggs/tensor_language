# Adversarial review of §37 conditional redirect — 2026-07-30

VERDICT: **Defensible with stated caveats — no retraction.** Mechanics verified correct (gate + metric
plumbing clean, no off-by-one; reach mask matches the firing set; scale=10 disclosed and held fixed
across the collateral comparison so 0.614->0.000 is purely gating). The two written caveats honestly
cover the load-bearing weaknesses. Four cheap additions requested; disposition:

- **Concern 2 (per-firing / common-trigger collateral) — ADDRESSED.** §37b freq-sweep already reports
  common-trigger collateral (+0.030 at base rate 3.9%) and the monotone frequency curve. Cross-ref added.
- **Concern 1 (planted best-case reach) — PARTLY ADDRESSED + queued.** §37b shows reach degrades to
  0.175/0.259 for a common (ambiguous-match) trigger. The natural un-planted reach lower bound is
  queued (qk_natural_trigger_redirect.py).
- **Concern 6 (SE + position/token variation) — queued.** The natural-trigger test reports reach with
  standard errors over many distinct natural positions and 3 trigger tokens.
- **Concern 3 (specificity wording) — FIXED.** §37 reworded: direct effect on non-trigger queries is
  zero BY CONSTRUCTION; the measurement bounds the INDIRECT downstream leak at <5e-4. No longer
  conflated.
- **Concern 4 (scale) — credited, no action.** Concern 5 (correctness) — credited clean. Concern 7
  (too-clean collapse) — ruled out as pure destruction (mass lands on chosen token); optional entropy/
  scale-sweep noted as a nicety, folded into the natural-trigger test's SE reporting.

No numbers retracted; §37 promoted to defensible-with-caveats. Not yet promoted to atlas/summary until
the natural-trigger lower bound lands.
