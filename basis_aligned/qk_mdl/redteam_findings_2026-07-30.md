# Adversarial review of the composition arc (2026-07-30) — ACCEPTED

Ten findings; program responses in brackets. Full agent output summarized.

1. HIGH — LAMBDA-SCALING BUG in the oracle-delta joint swap (qk_chain_deep joint6, qk_joint_polish):
   deltas injected with UNIT coefficients; correct coefficient of m_l at layer 5 is the product of
   downstream lambdas (0.00043 for m0 -> injected ~2300x too large). The 72.5% joint "gap", the
   19-point knob "recovery" (tick ii), and the "exposure-bias reversal" narrative (tick jj) all
   measured a mis-implemented substitution. CO recurrence itself verified CORRECT symbolically; the
   l2_composed arm-C used correct scaling. [FIX QUEUED: qk_joint6_fixed.py with CO-scaled deltas;
   ticks hh/ii/jj claims RETRACTED pending rerun; the knobs "learned" to compensate a bug.]
2. HIGH — "Causal 18-MLP chain 99.8%, zero trained parameters" inflated: attention outputs enter the
   residual UNTRUNCATED; only MLP-input attention components are projected (64/128 per head, bases
   keep ~92% energy); MLP evaluation is the model's own exact tensor. Floor denominator (+18.49 =>
   CE 21.6) exceeds the uniform ceiling ln(V)=10.83 — cosmetically inflating "99.8%". "Zero fitting"
   false (PCA bases fit on cooc). [ACCEPTED: headline reworded to the input-projection statement;
   report vs uniform ceiling (dCE 0.0329 vs 7.74 available) alongside; "zero fitting" -> "no fitted
   parameters beyond PCA basis choice", stated explicitly.]
3. HIGH — "Fully-NAMED analytic TN" false: capstone bases are anonymous PCA at all layers; the named
   archetype basis was abandoned after PCA beat it by ~20 pts (logged, then ignored in the
   headline); MLP outputs re-enter the residual outside the 576-span. Null was weak (full-space
   random). [ACCEPTED: renamed PCA/head-bottleneck; "named" restricted to the 144 layer-0 archetype
   dims; FIX QUEUED: head-span-restricted random null x2 seeds (qk_bottleneck_headnull.py).]
4. MED-HIGH — Composition-vs-data comparisons omit description length: composed forms reference the
   FULL weight tensors (>= component size; cores proven incompressible) vs data programs 27x smaller.
   At MLP0 composition LOSES on dCE (0.114 vs 0.075); MLP2 comparison used a post-polish data number
   against the program's own pre-polish credit rule. [ACCEPTED: reframed as fidelity-vs-compression
   FRONTIER, not supersession; description-length column mandatory; METHODS banner corrected.]
5. MED — "Function-consistency" mechanism asserted, never tested; finding 1 supplies the parsimonious
   alternative. [RETRACTED pending scaled rerun + six-layer causal data-program control.]
6. MED — Gauge identities are architecture tautologies (hold for any weights); they are METHOD
   licenses, not findings; rope/rms commute number (0.0229 bf16) not a verification. [ACCEPTED:
   METHODS reworded; "milestone" framing dropped.]
7. MED — MLP1 routing claim (A0-feeds-through-M0; M0xA1 dominant) is a single-batch variance-share
   claim violating the program's own no-variance-headlines rule; lam1_0=0.0127 makes "A0-direct~0"
   an architectural lambda fact. [DEMOTED to descriptive observation pending dCE block ablation.]
8. LOW-MED — No uncertainty quantification; audit set reused for selection; 0.005-nat contrasts
   uninterpreted. [ACCEPTED: paired SEs for headline contrasts + a held-back audit slice for
   capstone claims -- queued.]
9. LOW-MED — Basis-recipe drift (archetype-16 -> PCA-16 -> PCA-K -> PCA-64) vs METHODS text;
   128-token basis fitting vs 512-token audits unexamined. [ACCEPTED: METHODS S3a updated to actual
   recipe; context-length check queued.]
10. LOW — Clean checks recorded: corpus hygiene sound; CO recurrence correct; floors consistent;
    full-analytic +0.00000 gates genuine.
