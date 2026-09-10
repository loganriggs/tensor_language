# Token readers and hierarchical unembedding readers folded through MLP16–17

User-directed 10 September: pursue both individual token vectors and shared
clusters/hierarchies of unembedding vectors, fold beyond the final bilinear
layer, and queue after the current attention OV pullback audit. The CPU audit
is complete before this registration. Original bilinear handoff/pilot governs.

## Prior work and the new question

Read modules/INDEX.md, modules/readout-L15-17.md, MLP17_CURRENT_UNDERSTANDING.md,
MLP16_CURRENT_UNDERSTANDING.md, and MLP_MODULE_DOSSIER_INDEX.md. Quadratic output
forms, frequency calibration and normalization/gated-response expansions are
already known. MLP17's old rank-2/four-output replacement failed clean transfer;
mlp17_channel.py's generic MLP16-channel quadratic fit failed all three bars.
Class-projected forms and consumer/writer folds also predate this test (MLP10
dossier, §§1570–1574/1597). No algebraic novelty or generic rank sweep is claimed.

New questions: can one exact token-reader program propagated through the last
two MLPs predict a specified live intervention despite the intervening attention?
Can a hierarchy found from ALL unembedding rows predict those vocabulary effects
through shared group readers, beyond a global mean or a shuffled grouping?
Failure closes this fixed hierarchy as a sufficient description, not all possible
unembedding structure. Do not tune depth, tokens, layers or gains after outcomes.

## Two views and the further fold

Token view: select every 98th vocabulary row, plus all base/donor answer/foil
IDs in the frozen three panels (sorted unique). This covers at least 512 token
readers, with no model-output selection. Exact folded coefficients for the
complete vocabulary remain representable as U Down; the dense complete
vocabulary-by-product tensor is not stored.

Structured view: all 50,304 raw unembedding rows U_t are clustered by direction.
Normalize each row for clustering only. Four levels of binary spherical k-means,
20 iterations each, deterministic farthest-from-mean/farthest-from-first starts,
produce 16 leaves. No targets, activations or evaluation outputs enter clustering.
Empty children invalidate the instrument; no restart/repair. Raw centroids G_k
remain actual means of U_t, not means of logits. The root and the sequence of
parent-to-child mean differences give a hierarchy. For each token,

    U_t = G_leaf(t) + E_t.

The residual E_t is exact and remains charged. Compare leaf-only effect
prediction with root-only and a fixed shuffled membership control preserving
leaf sizes (seed9111340). Random-group centroids are rescaled to the matching
real centroid norms to avoid a simple norm advantage. No claim that the groups
are semantic until token inspection and functional evidence support it.

Let z16=(Left16 n16)*(Right16 n16), y16=Down16 z16+b16. At the final MLP,
r17=a+lambda*Down16 z16, where a contains the captured residual background,
MLP16 bias, x0 re-entry and attention17 output. Keeping that background fixed,

    Left17 r17 = Left17 a + lambda*(Left17 Down16) z16,
    Right17 r17 = Right17 a + lambda*(Right17 Down16) z16,
    z17 = (Left17 r17)*(Right17 r17)/(mean(r17²)+epsilon),
    U_t h17 = U_t a + lambda*(U_t Down16) z16
              + (U_t Down17) z17 + U_t b17.

Replace U_t by any group reader to obtain the same compiled operation. The
degree is quadratic in z16 before the explicit RMS division; z16 itself is
quadratic in the normalized MLP16 input. In raw residual variables the full
normalized program is rational, not an unrestricted quartic polynomial.
Final RMS and 30*tanh remain exact. Group and residual contributions add BEFORE
the shared normalization/softcap, never as independent capped logits.

## Intervention and price

Reuse all existing recombined A1/A2/C pairs, 16 each, with independent native
endpoint capability checks; no fresh/training-OOD claim. Capture native base
and donor MLP16 products and complete MLP16 outputs. Swap donor MLP16 output
at the recipient semantic position, first with native attention17 live, then
with only that position's attention17 output fixed to its native base value.
Earlier positions remain causal and unchanged. These are four forwards per
panel, 12 body forwards/192 sequence instances, no backwards or updates.

The folded prediction substitutes donor z16 into the equation above using the
base background. Its exactness reference is the attention-fixed intervention.
Its usefulness reference is the LIVE-attention intervention. Report both;
never describe attention-fixed exactness as an independently extracted circuit.

All 545,902,902 native parameters remain required. Two dense 4608x4608 maps are
additional compiler coefficients, not compression. Reader maps add
2*R*4608 values for R selected token/group readers. Report actual dimensions,
temporary coefficient count, hierarchy label/centroid count and runtime. The
model plus maps fit the managed RTX5090 budget; a 900-second alarm is a limit,
not a claim of useful CPU/GPU hours. Save hierarchy labels/centroids as an
artifact; regenerate large folded maps from the bound native weights.

## Frozen predictions

A. Instrument: CPU controls pass; 12/192 body counts; finite values; all 16 leaf
groups nonempty; every semantic position equals its row's final input token;
captured MLP16 products reproduce its donor write (maxabs <=1e-3+1e-5*abs(reference));
compiled state and selected-token logits match the attention-fixed native run
with maxabs<=1e-3 AND relative L2<=1e-5 for logits, state relative L2<=1e-5;
group+token-residual score identity maxabs<=1e-3 and relative L2<=1e-5. Native
captured-final-state logits match the ordinary forward at the same endpoint bar.

B. Native capability/reachability: all 48 native pairs have both own margins>0
and positive cue denominators; MLP16 swap has centered full-vocabulary effect
norm>1e-4 separately in every panel. Report its signed task recovery; no minimum
recovery is imposed or interpreted as target-circuit sufficiency.

C. Further-fold live prediction: A/B and the attention-fixed compiled state's
full-vocabulary effect differs from LIVE MLP16-swap effect by <=10%
relative L2, separately in every panel. (Here the edited module is MLP16;
attention17 is the intervening path.) Failure requires retaining that path.

D. Shared hierarchy: A/B; predicting the LIVE full-vocabulary effect by only
the 16 shared group readers has relative centered L2 error<=.10 in all panels,
and beats both root-only and norm-matched shuffled-membership error by >=.10
in every panel. Its candidate logits equal
softcap(native_base_raw_scores + G_leaf dot delta_normalized_compiled_state).
Evaluate the corresponding attention-fixed error as a separate diagnostic.

E. Signed manipulation prediction: A/B/C; exact folded program predicts LIVE
per-row correct-token CE changes with mean absolute error<=.02 nats in all
panels. This is prediction of the specified MLP16 interchange, not selective
removal or a complete circuit certificate.

Use the same live effect norm for all live errors, floor1e-30. Save per-row
effect/error squared norms, CE changes, margins, exactness checks and cluster
representative token IDs. No behavioral names selected from results. A failure
is invalid instrumentation; C failure refutes this attention-omitting fold;
D failure retains individual residual readers and closes this hierarchy-only
description. Broader four-property goal remains open regardless of this screen.

After results, perform a claimed CPU paired-bootstrap analysis of live-prediction
and hierarchy errors, preserving all nulls and denominators. Update the MLP16/17
dossiers and startup receipt; do not reopen closed rank or calibration families.
