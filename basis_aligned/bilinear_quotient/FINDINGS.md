# bilin18 — consolidated findings index

The working index. The full ledger (BILIN18_CONNECTION.md, 630+ sections) is an append-only
lab notebook, not a place to keep up with results. **This file is the ~10 things that matter,
their confidence, and what's open.** Update this in place; don't let it grow past a page.

Confidence: **HIGH** = causal test + control + null, reproduced. **MED** = solid but one caveat.
**LOW/known-limit** = suggestive or resolution-limited.

## The established results (most important first)

1. **Redundancy is universal, and decodability ≠ causality — one isolable knob + a distributed
   remainder.** Computation is diffuse over MLP units, depth bands, and attention heads; the
   strongest localization is "these matter more than chance," never "this is the circuit" (§610–616,
   §633, §644, §648). The isolation taxonomy (§650–668): **additive/subtractive biases isolate to a
   removable linear direction** (the frequency calibration = rank-1, §650–651) — but **every
   conditional/predictive/stateful computation** (newline routing §652–653, article magnitude §654,
   content-writing §658, quote-parity register §668) has **no removable linear carrier**, even when
   the feature is strongly *decodable* (quote-parity AUC 0.83 yet causally inert, §668 — read≠write).
   Net: the model is ~**one cleanly-isolable linear knob** (the frequency-calibration bias, ~0.43 of
   the ~7.48-nat loss-benefit over uniform, ~6%) plus a ~94% **distributed remainder** with no
   linear carrier. **HIGH.**

1b. **The right decomposition of the front is a canonical token-class SUBSPACE, not learned
   parts (§737–772).** Chasing a good decomposition: reconstruction-optimal bases (SVD/A-SVD) are
   CE-catastrophic at low rank (front-load loss-irrelevant massive-activation energy, §737/748); a
   learned overcomplete **sparse dictionary** (weight-action SAE) reconstructs faithfully (§750/759,
   needs a reconstruction anchor or pure-CE training destroys weight-faithfulness §758/762) and its
   cross-layer coupling is **weight-only / data-invariant** (§754, A-SVD's is not) — but the SAE
   *atoms* are the **wrong unit**: seed-unstable (recur 0.40, §763), redundant (super-additive §761),
   and monosemanticity is orthogonal to causal importance/stability (§763); co-activation groups
   don't rescue it (§766). The interpretable+causal+stable structure the SAE fails to be **is a
   seed-free SUBSPACE** you compute directly: the **token-conditional-mean** directions of a
   component's output. That subspace is **canonical/data-stable** (0.82 vs atoms 0.40), **necessary**
   (remove top-64 → +1.34 nats, 268× a random subspace §767), **sufficient** (keep only top-64 →
   92% of the layer's loss §770), **low-rank** (64/1152), and **human-nameable**: the front sorts
   what it writes by **token-class** — MLP by LEXICAL categories (determiners/punct/numbers/verbs
   §768), attention by STRUCTURAL/discourse markers (conjunctions/clause-boundaries §771). It is
   **universal** across both component types (§771) and **multiplexed** into **near-orthogonal**
   per-component subspaces (attn0|mlp0 overlap 0.254 vs 0.201 floor §772). *Ground-truth-free recipe
   for "a good component": don't fit a dictionary and hope — compute the canonical token-mean
   subspace, verify it causally (ablate-vs-random AND keep-only-vs-random), read its named axes.*
   **HIGH.**
   **GRAND CAPSTONE (767–795):** keeping ONLY class+position at ALL 36 components at once recovers
   **78%** of the whole model's 9.1-nat contribution (front alone 84%; random subspace 4%),
   **uniformly** across every token-class (function words 0.80, content 0.77 — gap 0.04). Class is a
   computed grammatical-class collapse (identity ~132 eff-dim → class ~24, sharpened ~1.8× vs the
   embedding, nonlinear R²0.44 §780/782); position is a coarse ~2-dim early/late readout of RoPE
   (RoPE ~18 eff-dim → ~2 §786/788). Written to near-orthogonal channels (§772), read amortized by
   later components (§783/784). **Cross-model COMMON across ALL SIX models (§807/§808, corrected
   mean-preserving per-component nat-weighted):** class+position share is bilin18 0.92, gpt2-small 0.91,
   gpt2-large 0.92, pythia-160m 0.81, pythia-410m 0.74, gpt2-medium 0.64 — band 0.64–0.92, every model
   >> its random null (0.02–0.28); robust across bilinear/absolute/rotary and 124M–774M. (bilin18's
   "0.78/four-fifths" is the stricter SIMULTANEOUS metric — all 36 components at once; per-component,
   apples-to-apples with the others, it is 0.92.) **GPT-2-medium was NOT an exception** — §802/803's
   "genuine isolated exception" (0.12) was a DROPPED-MEAN artifact of the centered keep metric (§804/805):
   its mlp0 is 91% constant bias by norm; keep goes −0.13 → +0.63 once the mean is preserved. Large
   constant DC biases are a COMMON early-layer feature (§806: 7/10 early components >0.5), and the
   §800 centered numbers were all slight underestimates; corrected, gpt2-medium is in-family (0.64, the
   least-clean case) with NO genuine exception. (The
   *simultaneous* whole-model metric is bilin18-specific — needs its 30·tanh output clamp — §799).
   The class-SHARPENING mechanism is the one model-specific piece (bilin18/Pythia yes, GPT-2 no §789/796).
   Also generalising to GPT-2/Pythia at the subspace level (§778/781). The distributed remainder is
   **content that makes confident predictions confident** (§798: class+position recovery rises 0.50→1.00
   with token difficulty — the skeleton is all that's left when the model is uncertain). The
   ~22% remainder is a **uniform, diffuse, distributed** computation with no low-rank or class handle
   (§795). So the 546M model is ~4/5 a class+position machine. Per-HEAD concept-naming FAILS (heads
   are broad multi-function; the fold names QK geometry not causal function, §792) — the robust unit
   is the VARIABLE a component reads/writes, not a head label. **HIGH.**

1c. **Whole-stack shape = an amortized compute→maintain→read pipeline, UNIVERSAL across 6 models
   (§809→817).** By loss-benefit the stack is a barbell in PER-COMPONENT ablation (front layers 0–5 =
   81% of benefit, 93% class+position; back MLPs 15–17 read out, 93%; middle low per-component). BUT
   per-component ablation UNDERSTATES redundant regions — CORRECTION §813: ablating the whole middle
   (6–11) at once costs 1.93 nats, ≈4× the 0.49 per-component sum (super-additive = distributed/redundant,
   each component compensable). The middle is class+position MAINTENANCE: simultaneous keep-only
   class+position on 6–11 recovers 0.65 of its collective benefit (random null −1.06, §814). So: FRONT
   computes class+position → MIDDLE redundantly maintains/refreshes it → BACK reads it out into
   predictions. This three-band structure with a SUPER-ADDITIVE class+position-maintenance middle
   REPLICATES in ALL SIX models (§815/§817: middle compounding 1.3–3.9×, class+position keep 0.59–0.78;
   every band ≫ random) — bilin18, GPT-2 small/medium/large, Pythia-160M/410M; bilinear-softmax-free,
   absolute, rotary; 124M–774M. The redundant class+position-MAINTENANCE middle is the robustly universal
   part. Front/back detail is architecture-specific and varies more than n=3 implied (§817): front
   compounding sub-additive in bilin18/pythia, super in GPT-2; pythia-160m is back-heavy; and gpt2-large's
   FRONT is only 0.37 class+position under the SIMULTANEOUS band metric (which compounds — per-component it
   is 0.92, §808 — so its front has strong cross-component interaction structure). Metric note: the
   per-component nat-weighted class+position headline (0.92, item 1b) under-weights the redundant middle,
   so true whole-model share is a bit lower (consistent with the stricter simultaneous 0.78). Middle is
   ~half-compressible (§816: drop 3 of 6 middle layers for 0.27 nats; front convex/irreplaceable). **HIGH.**

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

7. **Newline circuit (flagship, fully traced; causally verified §728).** `.`/`!`/`?` embedding
   trigger (28× lift) → front attention discriminates real line-ends (0.47) from mid-paragraph (0.21)
   → block-17 calibration. **Causal AUC test (§728):** line-end discrimination AUC 0.806 collapses to
   **0.510 (chance)** when front attention is ablated (vs 0.789 random) — front attention carries ~all
   the discrimination. **HIGH.** §635, §637, §639, §643–644, §728.

8. **Article circuit (traced; corrected 614; causally verified §729).** be-verb→a/an; **preposition→the**
   (was wrongly "a/an"); punctuation→the. Front attention carries the a/an-vs-the *choice*; front MLP
   carries the *magnitude*; block 17 calibrates "the". **Causal AUC test (§729):** the-vs-a/an AUC 0.870
   → 0.703 when front attn ablated (drop 0.167, 24× random) > front-mlp 0.737 — **confirms attn=choice,
   mlp=magnitude**, but the choice is more DISTRIBUTED than newline (front-attn ablation ≠ chance). **HIGH.**
   §636, §640, §729.

*Circuit-verification method (§726–729):* every named circuit is now checked by **causal output
selectivity** (ablate → which behavior's CE/AUC collapses), not firing. This verified items 7–8 and
BROKE the false "boundary circuit" (§726–727: only mlp16 causally boundary-selective; block1.attn fires
at boundaries but writes open-vocab continuation). See method note.

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

11. **Stateful context registers exist but are read-correlates.** The model tracks
    counting-based context state — quotation parity (probe AUC 0.83, peaks mid-network then decays;
    §667) and parenthesis depth (AUC 0.92 from block 2, behaviorally 600×; §669) — decodable and
    behaviorally used, but removing the decodable direction does nothing (read≠write): the causal
    mechanism is distributed, per item 1. A capability dimension beyond the token-class/frequency
    machinery. **HIGH.** §667–669.

12. **Massive activations = the rms-norm gain controller (not attention sinks).** A few residual
    dims (persistent 645/990/981) grow to 20–60× the median by block 17 and dominate ~85% of the
    residual sum-of-squares, so their large DC offset *sets the rms-norm scale* for the readout —
    removing that offset costs +1.58 nats (§680). They are NOT token/position sinks (uniform across
    both — this model has no softmax, so no sink mechanism; §678), and they host the frequency-
    calibration direction (88% of `w_freq`; §676). **HIGH.** §676–680.

13. **Softmax-free attention = a two-criterion multiplicative conjunction (mostly positional ×
    content).** Attention is focal (~0.23 eff-keys vs 0.64 random; §681) despite no softmax, because
    each head multiplies TWO QK scores (`pat = s1·s2`): each alone is diffuse (~0.54), the product
    focal (more focal than both in 100% of cases; §682). The two QK circuits are complementary
    (corr ~0, 0/162 redundant; §683) and 44% of heads factorize into one positional (distance-
    selective) × one content QK — nearly all use positional selectivity in ≥1 QK (§684–685). So the
    model does lookup-style attention by AND-ing a positional and a content criterion. **HIGH.** §681–685.

14. **Embedding-dominant residual — the current token is kept present to the readout, distributed.**
    The residual rescale is `x = λ₀x + λ₁x₀` with the embedding re-injected at **λ₁≈8 at every
    block** (a systematic gain, not decay), while λ₀ **resets** the running residual in the front
    (L1 λ₀=0.013, L5=0.064 nearly zero it) and accumulates in the back (§689). Functional confirm:
    the current token's identity stays **linearly recoverable from the FINAL residual** (log-freq
    probe R² 0.91→0.85→**0.73** across depth, slow decay not transformed-away; shuffled null −0.43;
    §690) — unlike a normal transformer that transforms the current token into context. The
    embedding is dimensionally **flat** (per-dim RMS peak 1.5×), so it carries identity in a
    distributed way; it is **NOT** the source of the massive dims — those (peak 58×) are built by
    the blocks (overlap 2/10, corr 0.14; §691). So embedding-dominance (item 14) and the massive-dim
    norm controller (item 12) are **independent** mechanisms sharing the residual stream. Blocks add
    context *on top of* an ever-present embedding. **HIGH.** §689–691.

15. **RSPD functional-rank map of the components (data-conditioned A-SVD, CE-priced).** Decomposing
    each component's decoder map on real activations, priced by held-out cross-entropy (r80 = smallest
    rank recovering ≥80% of the layer's loss-benefit): **attention c_proj maps are very low-rank**
    (block0 r80=2, block1 r80=1 — one direction = a boundary→continuation writer, block2 r80=8), and
    **mlp0/mlp17 are low-rank** (8, 4). But **mlp1/mlp2 are globally high-rank** (r80=128, 256; low-rank
    surrogates worse than ablation). r80 is **data-robust** (mlp0=8 identical across 3k–24k tokens,
    §699). The high-rank layers (mlp1/mlp2) are **genuinely high-rank**. Clustering tokens by
    decoder-output direction and giving each cluster its own low-rank subspace gives a **modest, real
    advantage** over one global subspace at matched low rank (cluster>shuffle>global, data-robust §707),
    but does **NOT** dissolve the high rank: at a fair 80% recovery bar, 7/8 mlp1 clusters still need
    rank ~128 (§709). The strong "union of low-rank circuits" reading of §704–705 was **overstated and
    is corrected** (§708–709): the effect is real but small (even 32 clusters at rank-8 recover ~35%).
    Scope: this is the **decoder map conditioned on real activations**, not the full bilinear-MLP rank.
    Fast A-SVD (normal-equations right-inverse) = 17.5× over the library, N-linear (§700). **HIGH**
    (method validated, controls+nulls; over-optimistic conclusion corrected §709). §694–709.

16. **Decomposition metrics: a learned overcomplete sparse dictionary beats SVD/A-SVD.** SVD/A-SVD are
    faithful (exact at full rank) but **dense** (~120/256 components per datapoint, §740) and **CE-
    catastrophic at low rank** — A-SVD orders by response *energy* (massive-activation directions, §737),
    so its rank-4 reconstruction is worse than ablation (§737, §748). Metrics genuinely differ (weight-SVD
    wins efficiency/composability/monosemanticity, A-SVD marginally wins raw parsimony). The right family
    is a **learned top-k sparse dictionary**: per-datapoint it recovers far more CE than SVD at the same k
    (activation-SAE 86% vs SVD −233% at k=8, §748), validated on ground-truth toys (recovers planted atoms
    perfectly, respects shared/hierarchical structure, doesn't hallucinate; §743–745, §747). The **novel
    weight-action** form — factor `W ≈ D·E`, codes `E·gate` top-k sparse — is faithful to the *weight* and
    CE-faithful on the real mlp1.Down (87% at k=8, §750; soft-L1 failed §749, hard top-k works). Orthogonal
    rotation can't sparsify (§741); overcompleteness is required. Real layers differ: mlp16 (rank-1) is
    genuinely simple (SAE≈SVD), mlp0/mlp1 have rich sparse structure the SAE finds (§746). Metric for
    "right decomposition" = atom-recovery + code-length-matching-true-k on toys; k tuned by the recovery
    peak (§745). **HIGH** (toys + real, controls+nulls). §737–750.

### Architecture facts worth keeping
- MLP = `Down[(Lx)·(Rx)] + b`: every output dim is an exact **quadratic form** `xᵀMₖx`. mlp17's
  *output* is rank-8 by **variance** (§615), but its **functional (loss) rank is higher** (§660):
  ~4 quadratic functions recover 75% of its loss (the "~4" answer holds at that level), but the
  top-8 variance dirs recover only 78% — the low-variance tail (last 5% of var) carries ~22% of the
  loss. Variance rank ≠ functional rank (extends §617's variance basis ≠ functional basis). So
  "mlp17 = 4 quadratic functions" is a good ¾-approximation, not an exact reduction. (Q3)
- Residual is rescaled every block (`x = λ₀x + λ₁x₀`); a writer 12 layers back arrives ×∏λ₀ ≈ 2e-4
  (front L1/L5 λ₀ near-zero reset the stream; embedding re-injected at λ₁≈8 every block — item 14).
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

**Circuit-naming rule (§726–727):** name a circuit by its **causal OUTPUT selectivity** (ablate it,
measure which behavior/token-category the CE-increase concentrates on), **not** by its **firing
pattern** (which input tokens make its coefficient spike). Firing tells you *when* a component
activates; only causal ablation tells you *what it does*. Over-reading firing as function created the
false "boundary→continuation circuit at 3 layers" (§726): of block0.attn-dir1 / block1.attn-rank1 /
mlp16-rank1, only **mlp16** is causally boundary-selective; **block1.attn fires at boundaries but does
general open-vocab continuation** (§727). This is the **fires≠contributes** face of read≠write
(items 1–2). Verify every named circuit this way before trusting the name.
