# Probability-metric red team, 10 September 21:41 UTC

Narrow failed claim: the frozen shared-refit MLP17 passes .05-token-CE-MAE/.95
native-top1 preservation. It fails despite1.40% pre-normalization full-U squared
error. Strong alternatives: RMS-induced scale changes, probability-sensitive
directions, and nonlinear finite displacements. This diagnostic fits nothing.

Reuse all three frozen physical modules and the same224validationdocuments /
32positions. Execute native model once per4documents, capture normalized MLP17
input, native MLP output and final residual h. Form candidate delta from compiled
MLP minus native MLP. Replay final scores from h+delta; compare the physical
replacement receipt. With r0=RMS(h), r1=RMS(h+delta), compare full scores with
numerator-only U(h+delta)/r0 and norm-only Uh/r1, each passed through native tanh.
These are diagnostics with native context, not proposed legal replacements.

For delta_h, exact score tangent is

$$
q=Uh/r_0,\quad
\dot q=U\delta_h/r_0-q\frac{h^T\delta_h}{\|h\|^2+d\epsilon},\quad
\dot s=(1-\tanh^2(q/30))\odot\dot q.
$$

The local native-to-candidate KL is approximated by

$$
\frac12\left[\sum_t p_t\dot s_t^2-
\left(\sum_t p_t\dot s_t\right)^2\right],
$$

where p is the native softmax distribution. CPU autodiff/finite-dose/constant-
score-shift controls must hold. Prediction A: exact456bodyforwards1824seq;
all shared native/cached input identities held; replayed document CE-damage
maxabsolute error<=5e-4 and mean full KL relative discrepancy<=1e-3 versus the
physical receipt for all3 candidates. B: Fisher quadratic predicts aggregate
full KL within20% for all3 candidates. C: norm-only KL is>=80% of full KL for
shared-refit. C failing rules out this normalization-only explanation, not all
normalization effects; numerator/norm effects need not add.

Also capture one pre-MLP RMS scalar per stored training/validation input, using
native pre=h-nativeMLP_output. This adds400trainingbodyforwards1600seq to the56
validationbodyforwards224seq above, still0testaccess. Native normalized-input
reconstruction from this pre must have batch-relativeL2<=1e-5. These scalars let
later training reconstruct the incoming residual from the already saved input
directions without storing another150MB matrix. The scalar replay is a numeric
bridge, not exact real-arithmetic recovery of floating-point subtraction.

Price456nativebodyforwards1824seq, local3candidate evaluations, dense full-U
scores/tangents on7168validationpositions, ~234kB scalar cache. Test rows are not
processed. Train cache capture uses no loss/labels; validation diagnosis never
fits or selects coefficients. Managed lane1,900s alarm. Preserve original failure
and reported thresholds. If the local probability metric predicts damage, it
justifies a new probability-aware fitting objective, not immediate adoption.
