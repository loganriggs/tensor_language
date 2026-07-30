# Adversarial review of §42 subject-verb agreement — 2026-07-30

Not ready to enshrine as written (§40/§41 bar); corrections applied.
- **Claim 1 (zero prior / 0.000 / chance) = TAUTOLOGY** (same class as §40): balanced 5×4×2×2 design forces
  mean margin 0.000 / acc 0.50 by sign-cancellation (pair members differ only at pos 1, cancel). Retracted
  as evidence; attention required by architecture (head@1, query@4) + 1.00 incongruent acc instead.
- **Claim 2/3 "exactly why incongruent succeed" CONTRADICTS redundancy**: ablating L11H3 flips NO items
  (acc stays 1.00). Softened to contributory-but-redundant margin dominance (46%, single seed). Noted the
  pos-0 sink weight 0.25; readout is correlational not causal.
- **Claim 4 OMITTED RESULT reconciled**: no_v1 (lambda=0) COLLAPSES agreement (acc 0.40) -> layer-0 value
  cache globally NECESSARY, cutting against "not layer-0". Corrected: single-position swap insufficient to
  flip (~17%) so number is redundant/distributed, NOT a localized swappable payload; "mid-stack" vs
  distributed-layer-0 unresolved. Missing same-number identity control (open follow-up).
- **Claim 5 caveats added**: single template, single seed; tail is attention AND feed-forward (L7-attn 2nd).
SURVIVES: the capability (1.00 incongruent), L11H3 margin dominance (46%, 2.3× gap), directional swap facts.
