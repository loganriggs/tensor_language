# Adversarial review of §37d (calibration reversal) — 2026-07-30

VERDICT: **§37d over-corrected in the OPTIMISTIC direction; the "calibration / in-the-wild capability"
headline is retracted.** 4 HIGH findings, all accepted:

- **F1 match-coeff biased.** natcoeff numerator fit on a homogeneous single-trigger-token query set
  (absorbs recency/position into MATCH); denominator fit on ALL planted queries -> not apples-to-apples.
  The 1.62x for a RARE token (more induction than the clean planted repeat) is physically implausible
  = smoking gun. "Induction present at full strength" WITHDRAWN.
- **F2 equal-amplitude contradiction.** At scale 10 (= planted amplitude), natural reach is 0.046/0.065
  vs planted 0.833/0.958 -- 18x/15x shortfall at matched gain. Incompatible with 0.9x-strength induction.
- **F3 brute-force confound (decisive).** scale-160 injects 160*A*v_payload into the residual; nothing
  distinguishes clean repoint from generic injection forcing token@1. CORROBORATED by the collateral
  run: trigger-position dCE blows up 1.94->32.3 nats (P(true-next)~e^-32, logit saturation) -- the
  brute-force signature, not a clean 'prefer payload' repoint. Queued qk_injection_specificity.py
  (inject at non-active queries / non-induction heads -> if reach still climbs, it's generic injection).
- **F4 premature capability claim.** "IS an in-the-wild capability" asserted before collateral measured.

Also: F5 "approaching planted" overstated (0.68/0.73 = 82%/76% at 16x amplitude; rare plateaus 0.27/0.33);
F6 retraction of §37c correct but calibration replacement not earned (cause OPEN); F7 SEs optimistic
(clustered by sequence -> moderate curve still real, rare curve within clustered noise).

DISPOSITION: §37d fully rewritten. Withdrew full-strength-induction + calibration + in-the-wild-capability.
SOLID finding retained: conditional gating keeps PURE (non-trigger) collateral ~0 across the sweep
(<=0.0001 nats at scale 160). In-the-wild clean-repoint question UNRESOLVED pending the injection-
specificity control. Defensible capstone reverts to the CONTROLLED-setting claim (§37/§37b) only.
