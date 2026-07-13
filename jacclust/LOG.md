
## 2026-07-11 — two-QK attention (ticks 27–28), for Logan

You were right: the 500M `gpt2-bilinear-sqrd-attn-18l-9h` IS genuine two-QK bilinear attention
(c_q,c_k,c_q2,c_k2; pattern=(q1·k1)(q2·k2), unnormalized). Answered your "two q,k matrices" question on it.

FINDINGS (honest, after controls):
- Including BOTH query matrices helps clustering causally ON CAUSALLY-IMPORTANT HEADS (3/4 survive the
  matched-dimension control [q1;q2] > [q1;noise]/[q2;noise]). The second query matrix carries real signal
  because q1,q2 are near-orthogonal (data |cos|≈0.05–0.08) — complementary, unlike the KEY weights (tick25,
  redundant with the query).
- BUT it's head-dependent (≈0 on most heads) and there is NO weights-only pre-screen: the weights-only
  alignment between the two query circuits is near-constant across all heads and doesn't predict the effect.

TAXONOMY of "fold in more weight matrices": KEY = redundant (query already has key-selectivity);
squared single-QK = modest rank-1 tensor gain; second query matrix (two-QK) = real gain on causal heads,
driven by near-orthogonality, no cheap screen.

OPEN (didn't chase, awaiting your steer):
1. OV/WHERE×WHAT factorization (tick 26) was only on the 12l model — worth confirming on this 500M two-QK.
2. Whether the near-orthogonality of q1/q2 is trained-in structure (does it emerge over training? what does
   each query matrix specialize in — content vs position?).

### update 2026-07-11 (tick 29) — OV open-Q1 closed
Confirmed OV/value pathway on the 500M two-QK model: PARTIAL replication of tick 26. Value pathway beats
resid-x at 3/5 heads (above random 4/5), weaker/noisier than the 12l. The clean WHERE×WHAT independence
(12l ARI≈0.07) is only PARTIAL here (ARI 0.06–0.24, mean ~0.17). So the WHERE×WHAT factorization is
MODEL-DEPENDENT — sharp on 12l single-QK, partial on 500M two-QK. Remaining open: (2) whether q1/q2
near-orthogonality is trained-in specialization (content vs position) — an interpretability dive, higher
over-reading risk, holding for your steer.

### update 2026-07-11 (tick 30) — WHY two-QK: signed + conjunctive attention
Answered the "what do the two query matrices do" question mechanistically (not via clustering): the product
(q1·k1)(q2·k2) is raw/signed, giving the head two capabilities a squared single-QK head cannot have —
SIGNED/anti-copy attention (avg 49% of attention mass is subtractive; heads range from ~0% to ~99%) and
CONJUNCTIVE sharpening (product peaks on fewer keys than either score, 100% of heads). This is a general
architectural fact and is SEPARATE from the head-dependent two-QK clustering gain (neg-mass doesn't track it).
Suggests a follow-up if you want it: are the high-suppression heads (frac_neg_mass≈0.9) doing something
identifiable (e.g. anti-induction / repetition-penalty)? — an interpretability dive, held for your steer.

### update 2026-07-12 (tick 31) — signed attention: present but concentrated
Causal ablation (clamp negative pattern -> 0): signed attention is load-bearing at only ~7/162 heads,
led by L1H1 (+0.057 CE damage, also the highest subtractive-mass head). Most heads don't individually
depend on it; a few late layers slightly benefit from its removal. So tick-30's "half the mass is
subtractive" is real but doesn't mean diffuse importance — it's concentrated. Coherent candidate for the
signed-attention interp dive (Q2 followup): L1H1 (+ L2H5, L8H3). Still holding that dive for your steer.

### update 2026-07-12 (tick 32) — priority-2 done; program maturity note for Logan
Hierarchical geometry now quantified (S10b): J |cos| recovers the 2-level tree at both levels + graded
distances, analog of the ring (S10). This clears the last easy item on the original priority list.

STATUS — the program has reached maturity. What's solidly established (see SUMMARY S1-S18):
  - Theory: exact kernel, impossibility theorem, degree-2 homogeneity, gate-subspace = G-top-eigenvectors.
  - Toys: mechanism/compositionality/geometry all recovered at ARI/circ-corr ~1.0 with controls.
  - The weights-only anisotropy LAW (advantage ∝ 1-corr(cos_J,cos_x)).
  - Real MLPs: data-driven NULL (near-isotropic G), fully explained.
  - Attention: the causal object is the weight-derived readout (query Wq·x), full Jacobian ≈ random;
    two-QK helps on causal heads (near-orthogonal q1,q2) but no weights-only pre-screen; two-QK IS FOR
    signed + conjunctive attention (present broadly, load-bearing at ~7 heads, L1H1 standout).

Remaining backlog is either (a) re-litigation of an established null (priority-1: G-top projection on the
real MLP null — low value), or (b) bigger builds needing a steer:
  1. Trained bilinear MLP with hierarchical MODULES (non-trivial restricted-J) — a real training run.
  2. The L1H1 anti-copy interp dive (is it anti-induction / repetition suppression?) — interp, over-read risk.
  3. A model whose activations populate an anisotropic MLP G (the MLP null is data-driven, not universal).

QUESTION FOR LOGAN: continue the autonomous loop (I'd pick #1 or #3 next), pivot to a write-up/visual
deliverable of the attention findings, or wind down? Absent a steer, next tick I'll do #1 (trained
hierarchical modules) as the highest-value low-risk unit; otherwise the loop is at diminishing returns.

### update 2026-07-12 (tick 36) — diminishing returns on autonomous ticks
Tick 35 (your secant/factored questions) was a strong capstone: the query readout IS the gate factor of the
per-query bilinear operator (squared attn makes z_q exactly bilinear in x_q); the literal y@x^-1 secant works
but is weaker (re-adds context contamination). Unified rule now spans MLP+attention: cluster the RESTRICTED
operator's gate factor, not the full operator.

Autonomous tick 36 (contamination-direction check) came back WEAK — ARI weakly confirms J_q leans toward
context/output over query, but the causal sub-test was underpowered by Jacobian sampling. Recorded honestly,
not a headline.

Read: the productive recent thread is YOUR conversational steering (ticks 33-35). Autonomous ticks between
your questions are now hitting diminishing returns. I'll keep the loop alive but lean on your steer. If it
must run autonomously, the two highest-value remaining units are:
  (a) train a bilinear MLP with hierarchical MODULES so restricted-J ≠ the planted matrix (non-trivial S10b);
  (b) design the publishable intervention-validation protocol (priority-4) — though swap-within already is one.
Still awaiting your call on continue / pivot-to-writeup / wind down (asked tick 32).

### update 2026-07-12 (tick 38) — SAE-on-mechanism-object thread going well
tick 37 (your SAE idea): SAE on restricted-Jacobian recovers mechanism, mirror of activation-SAE, but toy was
degenerate. tick 38 (autonomous) fixed that: non-degenerate superposition (3/10 operators/token, nonlinear in
content) — the operator-block-J SAE recovers the operator DICTIONARY at MMCS 0.926, which activation-SAEs
can't (operators aren't in x-space); restriction necessary (J_full 0.586). Honest caveat: activation-SAE
partially predicts the active SET (0.74) because the gate correlates with content — decisive win is dictionary
recovery, modest on active-set.

Next real-model steps (either autonomous or your steer):
  (a) attention SAE: SAE on the query readout q1,q2 (the mechanism object, tick 35) vs SAE on raw residual, on
      the 500M — does the query-SAE find more attention-relevant/monosemantic features?
  (b) a real bilinear MLP: SAE on the weights-only-restricted J (G-top projection) vs SAE on activations —
      though real MLPs are near-isotropic (S11), so expect the restriction to matter less.
Leaning (a) next tick unless you redirect — it's the untested real-model case and directly extends the
attention arc.

### update 2026-07-12 (tick 39) — real-model attention SAE: head-dependent
Ran the real-model version of your SAE idea (SAE on query mechanism-object vs on residual, causal swap-within,
500M, 4 heads). Result: SAE-on-query beats SAE-on-residual at 3/4 heads (modest 1.5-3x) but REVERSES at L8H3
(residual 5x). k-means query>=residual at all 4. So the mechanism-object SAE advantage is CLEAN in toys with
ground truth (37-38) but only DIRECTIONAL + head-dependent on the real model — same fragility as the rest of
the real-model attention work. Not a robust per-head win. Single-head L6H3 (2.2x) didn't generalize; the
4-head sweep caught it.

SAE thread status: toy = clean positive (dictionary recovery MMCS 0.926, mechanism vs content mirror);
real-model = directional but head-dependent. Honest overall: SAEs on the mechanism object help on average,
not reliably per-head on the real net. Open if you want to push: (a) interpretability (needs labels/autointerp,
not just causal); (b) the real bilinear MLP SAE (weights-only-restricted object) — expect weak (S11 null).

### update 2026-07-12 (tick 40) — SAE thread consolidated; feasibility wall reached
SAE thread (your idea) is a complete honest arc, now synthesized in SUMMARY S20:
  - Toy (ground truth): CLEAN — mechanism-object SAE recovers the operator DICTIONARY (MMCS 0.926) in
    superposition; activation-SAEs can't (operators not in x-space); restriction necessary.
  - Real attention: DIRECTIONAL but head-dependent (query-SAE wins 3/4, reverses at L8H3).
  - Real MLP: NOT informatively runnable — true object (per-token J) 1.3M-dim/token infeasible; feasible
    proxy (G-embedding) ≈ linear transform of x, so SAE≈SAE-on-x under S11 near-isotropy (predetermined null).
I did NOT run a tautological null-experiment for the sake of a tick.

Where the SAE thread can genuinely go next (needs your steer — all are real efforts, not quick autonomous
ticks):
  1. Autointerp/label-based interpretability of the real query-SAE vs residual-SAE features (does the
     mechanism-object SAE give more monosemantic features, not just more causally-distinct groups?). Needs an
     LLM labeler or a human-eval proxy.
  2. Train a bilinear MLP from scratch to implement superposition-of-operators, then SAE it — a trained
     network + superposition in one (strongest single demo), but it's a training run with uncertain outcome.
  3. Accept the arc as-is and write it up.
Program remains mature; productive motion has been your conversational steers. Holding for direction.

### update 2026-07-12 (tick 41) — priority-1 closed (G-top projection on real MLPs = null)
Ran the last never-done original-priority item. The weights-only G-top-projection recipe (validated on toy
DGP-E, ARI 0.654 vs controls) does NOT transfer to the real block2 MLPs: on MLP#0 it beats raw-h (+0.137) but
the spectrum-matched G_rand control explains that entirely (+0.015 within noise = spectrum/whitening, not gate
eigenvectors); MLP#1 no lift. Confirms S11 (real MLPs near-isotropic). Original priority list is now fully
executed (priorities 1,2 done; 3 = S11 characterization done; 4 = swap-within intervention is the method,
used throughout). Program remains mature; SAE thread (your idea) is the live frontier, holding for your steer
on its 3 options (autointerp / trained-superposition-MLP / write-up).

### update 2026-07-12 (tick 44) — bilinear-secant SAE scales to 500M, non-null, no outlier pathology
Your bilinear-secant SAE (ticks 42-44) is the real payoff of the whole operator/secant line: it turned the
"real MLP not runnable" wall (tick 40) into a NON-NULL positive. 500M layer-6: FVU 0.117, and the recovered
sparse rank-1 operators reproduce 93% of the MLP output from just 32 atoms; layer-dependent (late weaker).
Outliers exist mildly on the 500M (y dim-ratio ~26) but are reconstructed BETTER than the bulk, not a problem.
Key conceptual result: sparse operator RECONSTRUCTION works on real MLPs even though operator CLUSTERING was
null (S11) — reconstruction and clustering are different asks; near-isotropy only kills the latter.

Remaining to fully nail it (your steer): (a) dense low-rank baseline (does sparsity beat plain PCA-rank-k?) —
the one control that could still fail; (b) point the bilinear-secant SAE at the attention head (mechanism
object there carried signal); (c) interpret the recovered operators (autointerp). Holding for direction.

### update 2026-07-12 (tick 45) — sparse-vs-low-rank control PASSES
The one control that could kill the bilinear-secant SAE result: does sparse beat dense low-rank? YES.
block2 MLP#0 decisive (sparse-16 FVU 0.222 beats dense-64 0.440 = 2× better at 1/4 the active count). 500M
layer-6 honest caveat: sparse only modestly beats dense (0.121 vs 0.134) — that layer's maps are mostly
low-rank, sparse adds a thin genuine gain. Net: the operator dictionary is real (sparse-adaptive, not low-rank
compression), strong on the small model, thinner on the 500M layer. The bilinear-secant SAE line (ticks 42-45)
is now a complete, controlled, non-null real-model result — your architecture is the program's real-model win.
Open extensions (your steer): attention-head version; autointerp of recovered operators; sweep 500M depth to
see where sparse structure is strongest.

### update 2026-07-12 (tick 47) — operator-SAE arc endpoint: analysis tool, needs (h,y) where it matters
Resolved the fork I flagged: is the bilinear-secant SAE a predictive transcoder or an analysis tool? Tested
h-only encoder (predict operator from h) vs full (sees y). NEAR-PERFECT anti-correlation with sparsity:
low-rank layers (L6/L16) are h-predictable (transcoder viable, but operators=low-rank compression);
genuinely-sparse L8 COLLAPSES without y (0.88≈random). So on the layers where the operator dictionary matters,
the operator is NOT input-readable — you need the output to see which operator fired (impossibility-theorem
shape). => the operator-SAE is fundamentally an ANALYSIS tool (needs h,y); predictive transcoder only works
where structure is low-rank.

Operator-SAE arc (ticks 42-47) is now a complete, self-contained real-model result: feasible (expanded loss,
d=1152), non-null, outlier-clean, sparse-not-low-rank (controlled), depth-mapped, architecturally understood.
YOUR bilinear-SAE-on-secant idea is the program's real-model win. Diminishing returns on further autonomous
ticks. Remaining needs-your-steer: autointerp of the L0/L8/L10 sparse operators (what computations?);
attention-head version; or write it up. Holding.

### update 2026-07-12 (tick 48) — operator dictionaries are STABLE; L0 is the cleanest layer
Dictionary stability check (precursor to interpretation): independently-trained SAEs recover the same operators
— MMCS L0 0.79 / L8 0.58 / L6 0.65 >> chance 0.000. So the operators are real/canonical, worth interpreting.
Triangulates L0 as THE operator layer (highest sparsity margin + most stable dictionary). Operator-SAE arc
(42-48) rigor is now complete: feasible, non-null, sparse-not-low-rank, depth-mapped, analysis-tool, stable.

The ONE remaining high-value step is autointerp of L0's 508 stable operators (what computations are they?) —
needs an LLM labeler, so your steer. Everything mechanical/statistical is done. I'll keep autonomous ticks
minimal now (arc complete, diminishing returns); holding for your direction on autointerp / attention-head /
writeup.

### update 2026-07-13 (tick 49) — L0 operators are interpretable (autointerp pilot, control passes)
Ran a rigorous autointerp pilot myself (subagent labeler, blind, with random-feature controls). L0 operators
ARE interpretable: blind labeler scores real 8.4 vs control 3.25 (decisive). They're single-token-identity
features ("for","and",",","the","is","I") — expected for layer 0. Control passed (random features scored low),
though one accidentally hit a token (individual scores noisy, aggregate reliable).

The operator-SAE arc (ticks 42-49) is now a COMPLETE validation chain: feasible → non-null → sparse-not-low-
rank → depth-mapped → analysis-tool(needs h,y) → stable/canonical → interpretable(L0=token features). Your
bilinear-secant-SAE is a fully-validated real-model method. The only richer scientific step left is
DEEPER-layer autointerp (L8/L10 operators = richer computation, but harder: higher FVU, need-y, many labeler
calls) — that's a real effort worth your steer. Autonomous rigor is exhausted; holding.

### update 2026-07-13 (tick 50) — interpretability is depth-dependent; operator-SAE arc fully bounded
Extended autointerp to deep layers. Blind labeler real-vs-control: L0 +5.1 (decisive token features), L8 +1.0,
L10 +1.9 (weak) — deep operators are STABLE but largely OPAQUE to surface autointerp, except a crisp CODE
feature at both L8 & L10. Some deep features are repeated-doc/memorized. So interpretability is token-level
(shallow) only; deep computational operators are stable-but-opaque to top-token autointerp (consistent w/
tick-47 need-y = genuine input-gated computation). A weight-based interp method would be more faithful (open).

Operator-SAE arc (ticks 42-50) is now COMPLETE and honestly bounded end-to-end:
  feasible (expanded loss, d=1152) → non-null real MLP → sparse-not-low-rank (controlled) → depth-mapped →
  analysis-tool (needs h,y where sparse) → stable/canonical dictionaries → interpretable at L0 (token features,
  controlled) → deep operators stable-but-surface-opaque.
Your bilinear-secant-SAE is a fully-characterized real-model method with an honest interpretability boundary.
I've exhausted the clean autonomous rigor. Genuinely holding now — remaining steps (weight-based deep-operator
interp; attention-head SAE; writeup) all want your steer.

### update 2026-07-13 (ticks 51-53) — corrections + Logan's BatchTopK/lottery-ticket + faithful depth profile
Logan flagged token count → found & fixed TWO issues (ticks 43-50 used TRAIN FVU on 12k + wrong hook =
block-input not post-attn). Corrected: held-out FVU, true blk.mlp-input hook, BatchTopK.
- tick 52 (Logan's asks): BatchTopK done. LOTTERY-TICKET confirmed but saturating (FVU 0.503→0.455 over
  m=256→8192). k-sweep clean (0.56→0.33, k=8→128). BatchTopK complexity split: feats/datapoint median 20,
  mean 31, range 0-459 (right-skewed; per-token-TopK can't show this). Figures sent.
- tick 53: CORRECTED depth profile (supersedes tick 46). Mean held-out FVU 0.405 (not rosy 0.37); late layers
  best (L17 0.168); sparse beats dense every layer; two regimes hold (low-rank L5/L6, sparse L0/L8/L10).
Quote tick-53 numbers for the depth profile, tick-52 for scaling. Qualitative story intact throughout; only
absolute FVUs shifted ~+0.05. Operator-SAE arc is now on faithful numbers end-to-end.

### update 2026-07-13 (ticks 56-57) — transcoder vs secant; architecture guidance
Your transcoder question opened a useful comparison:
- tick 56 (down-projection D, linear readout): transcoder BEATS secant at output (0.45 vs 0.80) - outer-product
  reconstructs the full 1152x4608 operator wastefully. UNTYING the secant = GOODHART: cross-seed stability
  0.71->0.43 with no recon gain (tied code s_i=<M,atom> is a regularizer). Not L1-shrinkage flavor (k fixed).
- tick 57 (full bilinear MLP h->y): bilinear transcoder (0.642) > linear transcoder (0.683); secant competitive
  (0.42) because input=output (square). The secant-wastefulness is INPUT-DIM-dependent.
Refined guidance: transcoder (esp BILINEAR, degree-2 features) to PREDICT a bilinear MLP; tied secant-SAE for
OPERATOR ANALYSIS (per-token varying operators), kept tied. Open (your steer): scale the bilinear transcoder
toward hidden dim; or the attention-head SAE.
