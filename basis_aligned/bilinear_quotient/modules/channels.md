# Architecture channels — value-residual, x0, massive dims, embedding dominance, clamps

**One line:** bilin18's four standing channels and what each grounds.

## Value residual (v = ½v + ½·v1, block-0 values routed to every layer)
- Block-0's value is EXACTLY static per-token content (R²=1.0 from embedding; static-table
  replacement costs 0.0; shuffled null +0.52; §1076). The channel broadcasts each word's static
  c_v value; content = pooled bag of c_v VALUES (why raw-embedding bags fail, §1072).
- Ablation (lamb=0): +3.3 nats, content-tilted (rare/freq 2.69); per-layer cost concentrates at
  L2-4 (content onset) (§985-987, §1075 [partial dup, corrected]).
- OOD: the ONE channel relied on MORE on code (cost-frac 1.01→1.10; §1081).

## x0 re-injection (x = λ₀x + λ₁x₀ every block; λ₁≈8 saturated)
- Keeps the embedding ever-present (~8/9 of block input); why class is re-derived every block
  (§962); current token stays linearly recoverable at the FINAL residual (R² 0.73; §690).
- Ablation (λ₁=0): +2.3 nats. BOTH channels are content-heavy; x0 only RELATIVELY more
  grammar-weighted (2.96 vs 3.84) — NOT a clean dissociation (§987 correction).
- Front λ₀ near-zero RESETS the stream (L1 0.013, L5 0.064); a writer 12 layers back arrives
  ×∏λ₀ ≈ 2e-4 (§689).

## Massive activations / gain control (dims 645, 990, 981, 880, 750, 111, 373, 43…)
- = the rms-norm GAIN CONTROLLER, not attention sinks (no softmax; §676-680). Removing the DC
  offset costs +1.58. Host w_freq (88%; §676). ~85% of residual sum-of-squares.
- Written by the multiplicative gates collectively (§688-691); delivered as a constant from
  position 0 by mlp4 → head 5.7 (see `attn-sink-5-7.md`); after L5 the residual is mostly one
  fixed vector with text riding on top (prior sink arc: constant = 62-72% of residual magnitude
  L6-8).
- Independent of embedding-dominance (overlap 2/10; §691).

## Output/readout clamps
- Logits = 30·tanh(lm_head(rmsnorm(x))/30). MLP = Down[(Lx)·(Rx)] + b — every output dim an
  exact quadratic form. The 30·tanh clamp is needed for whole-model simultaneous keep metrics
  (§799).

## Gotchas
- Full channel ablations are off-distribution (graded, near-zero nonlinear tail; §987).
- Any experiment touching the residual scale must respect the gain dims (removing/perturbing the
  baseline is catastrophic and NON-LOCAL; partial removals worse than full — §1087/§1091).


### Shared first-value lexical/context test, 10 September 17:48 UTC

Known token-only first-value producer (channels/v173/v174) was tested as a
cross-layer source for current actual-token MLP17 context gates. Cyclic primed
verb at position3; ordinary token stream versus attention0-returned value cache
factorial, preserving attention0 own output. All later native consumers live.
A instrument held, B context source/C joint lexical selectivity failed.
Value-only lexical recovery .84174/.80855 with all32pairs both-endpoint capable,
but gate transfer .48551/.60670/error .55717/.56737; remaining-stream gate
magnitude .53379/.59949. G absCE .70470 (CI .40194–1.07834) fails preservation.
Seven harmed/nine improved; signedCE -.08536 does not rescue the absCE failure.
Edited foil logits were not saved; do not attribute this CE change specifically
to agreement-margin damage. CPU synthetic equal-margin/different-CE witness
and paired audit executed. Full-vocabulary interaction .36498/.32789.
Token-to-V0/noop/saved-u checks exact0;15forwards240seq,1.42524seconds. No smaller
consumer program, OOD, independent extraction or selective circuit promotion.
Primary receipts TOKEN_CONTEXT_BROADCAST_V1_RESULT.json,
TOKEN_CONTEXT_BROADCAST_AUDIT_V1_RESULT.json and
TOKEN_CONTEXT_BROADCAST_CONTROL_CE_V1.json in polynomial_causal; updated dated
gerund_scalar_writes_and_live_feedback.md. Next lexical/form readout separation,
not a head-index slice or rank rescue of the failed whole-source test.


### Lexical/form command reuse and coupling, 10 September18:03 UTC

Fixed lexical command: first-value position3 cyclic donor. Fixed form command:
all36 output e scalars from original-lemma ing cue, reused without adapters for
the cyclic lemma. Native2lemma x2cue grid, both singles in both contexts andjoint.
A instrument held; B/C/D/E failed. A1native16/16everycorner; A2Y/Z15/16, shop
and laugh contexts prefer traveling in their respective four-token comparisons.
L recovery .84174/.84124 and .80855/.78531; form drift .0462–.0976. F recovery
.95175/.95041 and .86341/.86437; lexical drift .1149–.1688 fails. Joint correct
14/16,13/16; four-token error .17660/.19757; full additive error .12298/.08285.
G agreement-margin meanabs .23328(CI .16004–.30963)fails .10 and old CE .70470
replays. This resolves the previous missing-foil limitation; no threshold rescue.
Exact four-reader Hadamard decomposition and MLP17 product fold held1.16e-14.
Actual CPU score-coordinate accounting: interaction contributes58–81% of F's
squared lexical drift, descriptive not causal mediation. Selected-score addition
error .04706/.04069 versus ACTUAL joint command, not native joint target; saved
state addition+exact norm/softcap .04604/.06495 does not fix joint choice failure.
23forwards368seq,1.55786sec. Saved7,476,061bytes reader/product coefficients and
finalstates for CPU reuse; no independent producers or structural saving.
Primary LEXICAL_FORM_INTERCHANGE_V1_RESULT.json, its AUDIT_V1 and
LEXICAL_FORM_DRIFT_COMPONENTS_V1_RESULT.json in polynomial_causal; current dated
gerund_scalar_writes_and_live_feedback.md. Next investigate explicit coupled
lexical/form operations and their producers, not independent-axis relabeling.


### 12 September — Token-only regional producer branch

The known token-only first-value channel contributes44–48%of the value-mediated
regional cue transfer through fixed attention8/9/13 producer readings on four
held-out geographic/spelling families. A48-token table of54scalar readings per
token replaces this branch's contextual capture in the native test; replay passes,
24/24pairs move correctly. Current values supply52–56%, so the original>=90%
current-dominance prediction fails. The two branches' write deltas add exactly;
nonlinear suffix interactions remain<0.2%of effects on tested prefixes.
This is a new reader-specific role, not a new discovery that v1is token-only.
Actual mixing at these layers is4.0,−0.65625,4.1875; historicalone-half notation
above is not the execution rule for this checkpoint. Native routing and remaining
states are still required; no full independent circuit claim.
[Primary maths and receipts](../../polynomial_causal/SHARED_CUBIC_SOURCE_PROJECTION_V1_MATH.md).

The [full-vocabulary endpoint audit](../../polynomial_causal/REGIONAL_FIRST_FULL_VOCAB_V1_RESULT.json)
of this specific regional first-value branch passes its frozen relative spillover
and off-pairTV bars on the80existing prefixes. Off-pair weighted logprob changes
are5.6–8.3%of regional logodds effects; TVmeans0.07–0.29%. Otherregional tokens
and some nonregional tokens change. This doesnot repair the older whole-channel
lexical/form selectivity failures above: the intervention is a different narrow
reader-directed branch with recipient routing/background fixed.

[Fresh behavior controls](../../polynomial_causal/REGIONAL_BEHAVIOR_CONTROLS_V1_RESULT.json)
subsequently pass:0.056nat regionaltransfer(~2.08%fullcuegap), all24tense/number/
factualanswers preserved, meanabscontrastchanges0.00026–0.00297nats. General
embedding/RMS/H execution replaces the restricted48-token table. Norm/Gram-matched
signedcoordinatewrite controls have95thpercentileregionaltransfer0.0023nats.
The small branch is orientation-specific; easycontrolsuccess is not independent
extraction or a claim about every unrelated behavior.
