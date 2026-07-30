# Adversarial review of §37c (natural-trigger negative) — 2026-07-30

VERDICT: **Defensible with rewording; the strong CAUSAL claim needs a control run.** The hedged
conclusion ("engages, low yield, NOT an *established* in-the-wild edit") is fair and supported by the
one well-powered point. The affirmative causal claim ("simply do not carry strong induction," "not a
design artifact") was NOT earned. Disposition:

- **F1 (credit).** Payload=token@1 is apples-to-apples with planted §37 and is the right "arbitrary
  payload"; P_payload (plausibility-robust) also stays low, so payload is not carrying the negative.
- **F2 (medium) ACCEPTED.** My "low induction to hijack" used baseline TRUE-NEXT P as the proxy —
  confounded with general LM predictability. Tell: the frequent token has the HIGHEST natural true-next
  (0.24) and the distinctive the LOWEST (0.15), inverting the distinctive=cleanest-induction thesis.
  Fix: measure the natural per-head MATCH coefficient vs planted AINIT (added to the control).
- **F3 (HIGH) ACCEPTED — the settling control.** AINIT calibrated on the planted eval, scale fixed at
  x10, never re-swept on natural text. Weakness could be a recoverable CALIBRATION limit. Fix: SCALE
  sweep on natural triggers (queued qk_natural_redirect_control.py). If P_payload climbs toward ~0.8,
  cause = calibration; if it plateaus low, cause = intrinsic.
- **F4 (HIGH) ACCEPTED.** Powered only in the ambiguous/frequent regime (the pre-flagged reach-worst);
  distinctive n=10 is unpowered (rule-of-three upper bound ~0.3, cannot exclude 30% capture); moderate
  n=18 is a REAL ~2-SE effect, not ~0. Fix: larger slice, triggers with >=40 induction-active
  occurrences (in the control).
- **F5 (credit).** Gate + reach metrics correct.
- **F6 (medium) ACCEPTED.** Section over-generalized; softened to keep only the earned hedge and mark
  the cause OPEN pending the control.

§37c reworded: retracted the affirmative cause, stated power honestly (moderate=real, distinctive=
unpowered), flagged the amplitude/proxy confounds, cause OPEN pending qk_natural_redirect_control.py.
No headline retraction beyond the causal over-claim.
