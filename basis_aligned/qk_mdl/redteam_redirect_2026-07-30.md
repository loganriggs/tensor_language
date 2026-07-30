# Adversarial review of §36 editing capstone (targeted redirect) — 2026-07-30

Reviewer (GPU-free, read-only) raised 10 findings. Processing:

- **F6 (medium-high) ACCEPTED — headline overturned.** "Soft repoint fails ⇒ target over-determined by
  the full pattern" was NOT isolated from a coefficient-undershoot alternative. Instrumented rerun
  (soft-amplitude sweep) confirms the mundane alternative: the SAME linear edit repoints cleanly when
  scaled ~10× (P_tgt 0.021→0.396, true-next 0.734→0.025), even beating the hard overwrite. RETRACTED
  the "over-determined / overwrite-required" claim; corrected story = coefficient-scale undershoot, and
  the target is controllable through the same linear channel as strength.
- **F5 (medium) ACCEPTED.** Hard path did not re-apply the causal mask → aim@9 could install non-causal
  attention. Fixed (`ph *= mask`) and re-ran; results stable.
- **F4 (medium) ACCEPTED.** rs mass-preservation is a softmax intuition on an unnormalized pattern;
  measured rs is positive on only 59% of active queries (mean 0.227, mean_abs 0.618) — sign-mixed, so
  the hard overwrite is the less principled tool. Now stated; scaled linear repoint is the headline.
- **F3 + F10 (medium) ACCEPTED.** ~35–58% argmax capture with 34–48% residual mass elsewhere (only
  4–22% on neighbours of the aimed column). Softened "genuinely aimable pointer" → "low-yield steer".
- **F7 (medium) ACCEPTED.** Dropped the strawman "+0.316 hard vs +0.006 null-soft" comparison; now
  report the real redirect/collateral tradeoff (hard +0.316@0.24, scaled-linear +0.588@0.40, s30 +2.15).
- **F8 (low-med) ACCEPTED.** Added spanning targets 1/9/30/55 — repoint holds across the sequence
  (P_tgt 0.18–0.27), ruling out a start-of-sequence special case.
- **F2 (low) ACCEPTED.** "16×" was the argmax ratio, not the probability (12.6×); now both stated.
- **F1, F9 (credit).** Double dissociation is clean off a matched ≈0.02 baseline (not a frequency
  artifact); active-query / head / layer restriction is correct. Retained.

Net: the aimability + causal-localization findings survive; the mechanistic "overwrite required"
headline is retracted and replaced by "same linear channel, ~10× amplitude, low-yield steer with real
collateral." §36 and atlas §9 corrected. Another instance of the loop catching an over-claim before
enshrinement.
