# bilin18 — consolidated findings index

The working index. The full ledger (BILIN18_CONNECTION.md, 630+ sections) is an append-only
lab notebook, not a place to keep up with results. **This file is the ~10 things that matter,
their confidence, and what's open.** Update this in place; don't let it grow past a page.

Confidence: **HIGH** = causal test + control + null, reproduced. **MED** = solid but one caveat.
**LOW/known-limit** = suggestive or resolution-limited.

## The established results (most important first)

1. **Redundancy is universal — no necessary component at any grain.** Computation is diffuse over
   MLP units, over depth bands, and over attention heads. The strongest localization the model
   permits is "these components do it and matter more than chance," never "this one is the circuit."
   **HIGH.** §610–616 (units), §633 (clusters), §644 (routing = "front attention", no single head),
   §648 (even the induction heads: top-4 ablation removes only 10%).

2. **Read ≠ write direction.** A supervised probe decodes a feature; the *unembedding row* (write
   axis) steers it; the two are ~orthogonal (cos≈0). Pushing the probe does not steer (even
   reverses). To decode, fit a probe; to intervene, push the write axis. **HIGH.** §619–622.

3. **Block 17 is the DOMINANT frequency calibrator (not the sole one), isolated to a rank-1
   direction.** It suppresses tokens ∝ log-frequency (corr +0.64); the only *net* calibrator by
   whole-block CE (−0.17/+0.69 → +0.43 nat trade). But rank-1 `w_freq` removal finds a calibration
   *component* in **five** layers — L4/5/6 and L16/17 — block 17 dominating 5–10× (§662); calibration
   is distributed across two bands, diluted below net-calibration in the others by their writer roles.
   The calibration = one direction `w_freq` (removing it
   kills it, random doesn't; §650–651), ~40% aligned with the unembedding log-freq axis (cos 0.61,
   cos²≈§627's R² 0.41; §656). Its function→content mass shift (§629) is the SAME w_freq direction —
   "boost rare content" = "suppress frequent function tokens" (removing w_freq also drops rare
   capitalized-writing; §657). Subword-writing is a separate, preserved function. **HIGH.** §624–657.

4. **Depth division of labor (with mechanism).** FRONT (0–2) decides the next-token *class* —
   MLP-dominant, token-local, from the embedding trigger (§634). MIDDLE (7–15) refines *which token
   within the class* (esp. the open content-word slot) — attention+MLP balanced, more context-
   dependent than the class decision (§665); a GENERAL content-word predictor (serves novel words as
   much as repeats — copying/induction is a sub-component, not its defining feature, §666). BACK
   (16–17) calibrates frequency. **MED** (front's +7-nat magnitude inflated by error-compounding;
   middle/back claims clean). §630–632, §634, §665–666.

5. **Circuits bottom out in embedding trigger-geometry, not computed triggers.** Skip all 18 blocks
   and a `.`→newline / prep→the lean is already in embedding∘unembedding. BUT the direct path is a
   *poor* LM (CE 12.65 > uniform 10.83) — it's a relative lean, off-distribution alone; the blocks
   do essentially all real predictive work (+7.48 nats over uniform). **HIGH.** §637, §640.

6. **The blocks' job = context-discrimination/routing of a context-blind trigger.** The `.` bigram
   fires identically at every period; the blocks route among {newline, capitalized, continuation}
   by context, done ~80% by FRONT ATTENTION. Trigger → route → calibrate. **HIGH.** §638–639, §643–644.

7. **Newline circuit (flagship, fully traced).** `.`/`!`/`?` embedding trigger (28× lift) → front
   attention discriminates real line-ends (0.47) from mid-paragraph (0.21) → block-17 calibration.
   **HIGH.** §635, §637, §639, §643–644.

8. **Article circuit (traced; corrected 614).** be-verb→a/an; **preposition→the** (was wrongly
   "a/an"); punctuation→the. Front attention carries the a/an-vs-the *choice*; front MLP carries the
   *magnitude*; block 17 calibrates "the". **HIGH.** §636, §640.

9. **A token "class" can hide two circuits.** Digit: *continuation* (prev digit→digit) vs
   *initiation* (first digit after $/page/word). Initiation is computed (9.4×); the average misleads.
   **MED.** §641–642.

10. **Induction/copying — ALREADY MAPPED in the census (name circuit attn0+attn1 build the copy
    source; "induction-target" motif). This run re-derived it and added:** natural-text induction is
    rare-token-dominant (P 0.33 for rare vs 0.08 frequent) and distance-robust; reader heads L5.H5
    (z+3.99)/L8.H4/H6/L10.H8 attend to the copy-source, BUT under ablation the causal copying is
    **distributed across ~the whole attention stack** — top-16 pattern-heads = 19% of the effect,
    all-attention = 87% (§649). So attention-pattern salience ≠ causal contribution; copying is NOT
    a localizable head-set. **MED / overlaps prior work.** §645–649. ⚠ opened without checking it
    was done — a tracking miss.

### Architecture facts worth keeping
- MLP = `Down[(Lx)·(Rx)] + b`: every output dim is an exact **quadratic form** `xᵀMₖx`. mlp17's
  *output* is rank-8 by **variance** (§615), but its **functional (loss) rank is higher** (§660):
  ~4 quadratic functions recover 75% of its loss (the "~4" answer holds at that level), but the
  top-8 variance dirs recover only 78% — the low-variance tail (last 5% of var) carries ~22% of the
  loss. Variance rank ≠ functional rank (extends §617's variance basis ≠ functional basis). So
  "mlp17 = 4 quadratic functions" is a good ¾-approximation, not an exact reduction. (Q3)
- Residual is rescaled every block (`x = λ₀x + λ₁x₀`); a writer 12 layers back arrives ×∏λ₀ ≈ 2e-4.
- Logits are `30·tanh(lm_head(rmsnorm(x))/30)`.

## Open / focus (hierarchical — go deeper on any)

- **A. Finer component isolation — METHOD WORKS, WITH A SCOPE (§650–652).** Behavior-conditioned
  low-rank + REMOVAL isolates **additive/subtractive** components to rank-1: block-17's calibration
  = one direction `w_freq = cov(mlp17 out, log-freq)` — removing it kills it (103%), random removal
  0–2% (§650–651). But it FAILS on **conditional/routing** computations: removing the rank-1
  `w_route` does nothing to the newline routing (§652), which is distributed in front attention
  (§644) and whose correlational direction is a decode-not-cause readout (read≠write). Rule: isolate
  by (behavior direction + removal + random control); expect rank-1 for biases/calibrations, NOT for
  context-conditional routing. **Scope confirmed (§653):** routing has NO removable linear carrier
  at any rank (top-32 removal = 0%); it's computed by attention + read nonlinearly. So finer-grained
  isolation is answered by component *type*: rank-1 for additive/subtractive biases, not for
  conditional routing (the wall there is nonlinearity/distribution, not redundancy). Arc §650–654,
  now confirmed on **three** behaviors: calibration = rank-1 isolable; newline routing & article
  magnitude = conditional, no low-rank carrier.
- **B. ~~head-SET localization of induction~~ — ANSWERED NO (§649): copying is distributed across
  ~all attention; pattern-selection can't isolate it. → reinforces A (need subspace method).
- **C. Systematic circuit discovery** vs the current opportunistic depth-first tracing (see method note).
- **D. Middle's within-class refinement mechanism** — hit the redundancy wall (§633), unlocalized.
- **E. Reconcile induction reader-heads (L5.H5) with census name-circuit source-builders (attn0/1).**

## Method note (how ideas are generated — honest)
Opportunistic, depth-first: pick a behavior, trace it output→input causally, follow each result to
the next question, generalize/contrast, turn contradictions into experiments. **No systematic
enumeration and no dedup against prior work** — which is how induction got re-run. Fix: consult this
index before opening a "new" thread.
