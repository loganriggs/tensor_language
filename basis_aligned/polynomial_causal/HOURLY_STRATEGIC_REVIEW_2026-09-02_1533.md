# Hourly strategic review — 2026-09-02 15:33 UTC

## Goal and circuit requirements

The goal is still an executable tensor program that explains the model in pieces which correspond to computations,
not merely directions with low reconstruction error. A useful circuit decomposition must establish:

1. what information each piece reads, what operation it performs, what it writes, and what later computation uses it;
2. when parts of different attention heads or MLPs implement the same variable, and when one native module must be
   split because it performs several jobs;
3. prediction of activations and causal effects on held-out and distribution-shifted text;
4. an extractable computation or a precisely stated interface together with any background it requires;
5. selective removal, replacement, or swapping that changes the intended behavior while preserving unrelated
   circuits, including redundancy and interactions;
6. predictable composition and reuse of shared pieces; and
7. stable identification across data splits, corpora, refits, and gauge-equivalent parameterizations, or an
   operational definition based on indistinguishable downstream use.

Rank reduction and quantization do not establish any of these. They can price an already identified computation or
serve as capacity-matched controls, but they are not the discovery objective.

## What changed in the last hour

- Rung491 fully passed discovery and prospective intervention-outcome validation. Attention1 is the unique named
  residual source necessary for both the MLP0 token-only branch T and token-by-context branch I when MLP1 responds.
  It is necessary but not sufficient, and C also uses it inside a larger, still-inadequate source set.
- Rung492 tested whether that exact local output attribution could be turned into a portable attention1-to-MLP1
  input edit. A true attention1 knockout strongly changes T and I, confirming causal upstream dependence. The narrow
  input edit failed: for T its agreement with the knockout was only `.423/.416`, for I it was `-.006/.017`, and in
  every case the same-position edit was worse than a position-shifted control. The edited input `z-a` is not a state
  MLP1 naturally receives; exact algebra did not make the intervention meaningful.
- Rung493 was registered, implemented, CPU-gated, committed, pushed, and launched through the managed GPU runner. It
  replaces two naturally produced branch-absent writes by their arithmetic mean. This stays on the line segment
  between real model states and directly tests whether the T/I distinction is progressively merged between
  attention1 and MLP1.

## Why rung493 is currently highest-information

The earlier exact write-space measurements found that the common part of the T and I responses increases with depth:
about 51% at attention1, 62% on MLP1's direct contribution, and 79% for MLP1's total response. That pattern could be
a real progressive merge or merely a descriptive consequence of changing scale. Rung493 converts it into a causal
test.

For each branch pair, let `Y_p` and `Y_q` be the real attention1 or MLP1 writes when branch `p` or `q` is absent. The
intervention sends both trajectories the same write `(Y_p+Y_q)/2`. It then measures the original final-loss contrast
`x = CE(absent p)-CE(absent q)`, the contrast after merging `y`, and the removed component `r=x-y`. The main statistic
is `<r,x>/||x||^2`: it is positive only when the intervention removes the original contrast in its own direction,
rather than producing a large unrelated perturbation like rung492.

Three modes separate attention1 followed by normal MLP1 recomputation, attention1's direct residual route while
keeping the original MLP1 write, and merging only at MLP1. T/I must beat sixteen position-shift controls, and its
depth gradient must be uniquely larger than all five other branch pairs. This tests grouping and splitting by
downstream causal use, not compression.

## Confounds and controls

- **Generic early-site attenuation:** all six T/C/I/S pairs are measured. T/I cannot pass merely because an earlier
  intervention has more time to affect the network.
- **Large edit mistaken for relevant edit:** the aligned removed fraction evaluates whether the removed final effect
  follows the original pair contrast. Raw RMS is reported separately.
- **Position-generic perturbation:** sixteen shifted-position merges are the direct control that rung492 failed.
- **Write-space description mistaken for a circuit:** the old common-share gradient and the new physical final-loss
  gradient must agree; neither alone is enough.
- **Post-selection:** pair vocabulary, thresholds, discovery halves, and the conditional validation decision were
  frozen before the model outcome was opened. No inconvenient control pair may be dropped.
- **Data scope:** a pass would still be evidence on the current document set. New-corpus prediction and semantic
  preservation remain required before a general/OOD claim.

## Alternative routes, ranked

1. **Finish the site-graded merge test.** A pass proceeds immediately to cross-corpus repetition and preservation of
   unrelated semantic circuits. A null retains attention1 as a causal local attribution but rejects the proposed
   T/I merge boundary.
2. **Exact attention1 decomposition below heads, grouped by downstream use.** Represent each head's two attention-score
   factors and value/output contribution exactly, then group terms across heads only when downstream MLP1 and later
   readers treat them interchangeably. This revisits the old sparse-QK/Tucker work without repeating its per-head,
   reconstruction-first limitation.
3. **Causal test of the scalar composition law.** Existing leave-one-pair-out analysis says a per-token monotone
   function of additive singleton effects predicts held-out pair effects 22–39% better than additivity. Intervene by
   scaling the identified equality direction and test whether that frozen nonlinear readout predicts the resulting
   effects. This is a different route to compositional circuit structure.
4. **Predictive-state grouping across the 62 known circuits.** Give candidate pieces a signature consisting of their
   effects under held-out downstream readers, group pieces with interchangeable signatures, and demand selective
   swap/removal plus unrelated-circuit preservation. This operationalizes simplicity in terms of reusable causal
   computation rather than rank.
5. **New-corpus semantic interpretation of MLP0.** Use natural-text and code corpora with no exact or prefix overlap
   with the current documents to label and test token-only, token-by-context, and context-only groups. This is useful,
   but the current site-graded causal decision should land first because it determines the object to interpret.

## Live continuation

Rung493 is confirmed GPU-active through the managed runner. At landing it must be scored against its frozen gates,
recorded in the ledger and explanation, and followed immediately by the result-conditioned successor above. The next
three-hour mathematical review is due after 16:10 UTC and must reconsider tensor-network identifiability, exact
factorizations, and gauge freedom rather than defaulting to native modules or low rank.
