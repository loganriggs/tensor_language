# Mean/remainder fold through the following normalized bilinear MLP

The four native head1.8 states expose a substantial interaction already in MLP1. Its exact fold requires both a bilinear mean/remainder cross term and a changing RMS denominator. The cross term alone misses60–81% of the local interaction on the tested, now-opened families. This local identity does not by itself explain the entire downstream loss effect or identify a simple causal circuit.

## Exact object

Let P(x)=D[(Lx) elementwise(Rx)], s_x=mean(x²)+eps, and M(x)=P(x)/s_x+bias. Write the pre-MLP residual as b+m+e, with m the projected head mean and e its projected remainder. Define I=M(b+m+e)-M(b+m)-M(b+e)+M(b). The bias cancels. Set s11=s_(b+m+e), s10=s_(b+m), s01=s_(b+e), s00=s_b. Then

X = D[(Lm) elementwise(Re)+(Le) elementwise(Rm)]/s11,

N = P(b+m)(1/s11-1/s10) + P(b+e)(1/s11-1/s01) + P(b)(1/s00-1/s11),

I=X+N exactly. X freezes the denominator at the both-present corner; N is the exact correction. This decomposition is anchored at that corner, not a unique coordinate-free attribution of normalization.

## Native validation

All four corners run through the actual native model. The direct MLP writes and their four-corner difference agree with the fold. The norm ratio is ||term||/||I||; the signed aligned fraction is <term,I>/||I||². Aligned fractions sum to1; norm ratios need not.

| Family | Cross-only relative error | Cross norm ratio | RMS correction norm ratio | Cross aligned fraction | RMS aligned fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 0.602 | 0.813 | 0.602 | 0.649 | 0.351 |
| arithmetic | 0.814 | 0.269 | 0.814 | 0.205 | 0.795 |
| repetition | 0.650 | 0.895 | 0.650 | 0.689 | 0.311 |
| opened_prose | 0.758 | 0.527 | 0.758 | 0.352 | 0.648 |

The registered20% cross-only error and20% normalization-norm gates both fail. Native-fold replay passes. Interaction is zero before the edited block; after block1 its norm is29–68% of native residual norm across families. Subsequent block-state profiles are response measurements, not causal necessity of those blocks.

## Smaller exact coordinate representation

Both m and e lie in the same head-output space: m=O u, e=O v, with O of shape1152x128. Fold A=L O and B=R O. The cross numerator becomes D[(Au) elementwise(Bv)+(Av) elementwise(Bu)] in128input coordinates. Each folded reader is4608x128 instead of4608x1152. This reduces the representation of this local cross term, not the complete source-to-logit program.

The entire conditional normalized map can be written for z=u+v:

M(b+Oz) = D[(Lb+Az) elementwise(Rb+Bz)] / ((||b||²+2(O^T b)^T z+z^T(O^T O)z)/1152+eps) + bias.

`prepare_head_coordinates` and `evaluate_head_coordinates` implement this rational-quadratic form and share the folded readers, norm geometry and background projections across counterfactuals. Three CPU tests verify direct normalization, independent writer folding, all four coordinates, zero backgrounds and a missing-source control. The conditional background ports remain necessary and must be charged: Lb, Rb, O^T b and ||b||² are input-dependent computations, not free weights. Native numerical conditioning of this newly added coordinate executor has not yet been validated.

## Consequence and limitations

Preserve the denominator geometry while searching for sparse interactions in the128-dimensional head space. Treating the normalization correction as negligible would throw away most of the arithmetic-family interaction. A candidate still has to explain or generate the remainder; declaring it a native port is not extraction of a complete circuit. This result gives a concrete joint object to factor, not evidence that a sparse factorization already exists.

64native forwards,0fits/updates. Same12constructed prompts and4prose rows as v637, now opened. Native writes replay within4.94e-7 and interaction within3.05e-6. The previous full-suffix causal prediction failure remains; no OOD, semantic selectivity or useful component-reuse claim is upgraded by this algebraic validation.

[Native receipt](../bilinear_quotient/circuits/followups/mean_remainder_fold_v638_result.json) · [implementation](normalized_bilinear_face.py) · [prior causal failure](MEAN_HEAD_CAUSAL_PREDICTION_2026-09-20.md)
