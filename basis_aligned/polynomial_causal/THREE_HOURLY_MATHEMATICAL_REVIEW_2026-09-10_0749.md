# Mathematical review: normalization invariants versus an identified bilinear source

Due 07:49 UTC; begun at 07:49 and executed in this block. Previous goal turn made
progress in commit 8c929f718. The original handoff and pilot, including the updated
interpretability criterion, govern. The full goal remains OOD prediction, extraction,
selective manipulation, composition/reuse and lower structural description cost.

## Actual object and accounting

The current partial producer consumes the conditional mixed component of normalized
layer9 residuals, reads it through native value weights and mean routing, and writes
to the common final two positions. Its background and full native suffix are explicit.
There are 32 worlds, each a complete 32-corner cube in five binary factors. The two
selected factors are object number o and attractor kind h. The other three factors
remain conditioned on, never averaged out globally. Residual width d=1152, hidden
width m=4608, nine heads of width128, 18 blocks and vocabulary50304.

RMS is r(x)=x/sqrt(x^T x/d+epsilon), with native FP32 epsilon. MLP8 is
M(u)=D[(L u)*(R u)]+bias, L/R in R^(4608x1152), D in R^(1152x4608).
The value consumer is C=1.65625 W_O P0 W_V Qoh r(x). P0 has nine heads,
two queries and three mature source positions; the to-query cannot read action.
The selected MLP8 source enters x with the actual block9 residual coefficient.
The suffix F(B+C) includes all later nonlinear computation.

The MLP numerator has degree2 in u; with raw RMS input it is a rational quadratic
map (plus bias), not a quadratic polynomial in raw residuals. Product attention has
degree5 in independently supplied normalized query/key/value factors, while actual
factor generation contains more normalizations and RoPE. The full transformer is
not a fixed low-degree polynomial of raw inputs. We retain tied factor dependencies.

Allowed intervention: subtract a specified mixed MLP8 branch only on its path into
this partial value producer, recompute RMS, rebuild C and run the live suffix.
This does not modify all consumers of MLP8. Measure both correct-margin and centered
three-answer effects. Native counterfactual inputs and all545902902parameters remain;
no saving or full-distribution fidelity is claimed. Each source test costs320native
forwards/5120sequence instances; hidden coefficient work is O(N*T*d*m), not a dense
1152-cubed tensor. A full output/input/input FP64 tensor would require about12.23GB
before workspace. Existing low-factor contractions avoid that allocation.

## Literature mappings and limits

[Villar et al., Scalars are universal](https://papers.neurips.cc/paper_files/paper/2021/file/f1b0775946bc0329b35b823b86eeb5f5-Paper.pdf),
Lemma1 and Proposition4, map orthogonally invariant scalar functions to Gram inputs
and equivariant vector functions to invariant scalar combinations of their inputs.
Our raw-removed residual a+ob+hc fits this restriction exactly. Six inner products
cost O(d), followed by four scalar inverse square roots and an O(d) combination.
Only four combinations of the Gram entries enter the four squared norms: diagonal
sum and three cross terms. The invariant representation determines geometry up to
an orthogonal transformation; dependent input vectors need not have unique scalar
coefficients. Native learned weights are not invariant under rotating only inputs:
readers must transform too. Consequently the theorem explains why the CPU identity
works; it does not identify a semantic feature or free us from generating a,b,c.

[Bilinear MLPs enable weight-based mechanistic interpretability](https://arxiv.org/html/2410.08417v2)
maps our ungated MLP and a linear consumer w to the symmetric form
T_w=sym(L^T diag(D^T w) R). Its eigendecomposition is exact for that consumer;
repeated eigenvalues leave basis freedom. It is not an exact eigenbasis for the
nonlinear suffix. Our executable consequence uses the same bilinear contraction
without choosing eigenvectors: conditional parity splits interaction inherited
from u_oh from the cross-product of separate u_o/u_h inputs. A dense eigenproblem
would cost O(d^3) per consumer; evaluating these two branches uses the native
factorized maps. Output bias vanishes under Qoh. No arbitrary rotation is moved
through the coordinatewise product.

[Kolda and Bader](https://www.cs.cornell.edu/courses/cs6241/2020sp/readings/Kolda-Bader-2009-survey.pdf)
give the CP uniqueness condition kA+kB+kC >=2r+2. The supplied unsymmetrized MLP
factorization has4608terms and each mode dimension1152: its maximum possible
left side3456 is below9218. Thus this sufficient criterion cannot certify that
factorization; this is not a proof of nonuniqueness or of its minimal rank.
Reparameterizations preserving weights also do not identify intervention meanings.
We reject a generic CP/TT compression sweep as the next circuit experiment.

[Balle, Panangaden and Precup](https://arxiv.org/abs/1501.06841) connect finite weighted
automata with finite-rank Hankel representations and construct a spectral canonical
form. [Li, Precup and Rabusseau](https://arxiv.org/abs/2010.10029) relate tensor networks,
weighted automata and linear second-order RNNs. Our contextual RMS-normalized state
updates violate that linear-transition restriction. A finite observed prefix/suffix
matrix can be factored by a dense SVD at cubic cost in its square dimension, but its
rank cannot certify finite population rank or recover a closed native transition.
These are not currently solutions to the actual source-identification problem.

## Executable consequence and decision

For actual MLP8 Left/Right outputs, write L0,Lo,Lh,Loh and R0,Ro,Rh,Roh for the
four conditional parity components (including their corner signs). Exactly,

    Qoh M(u) = D[Lo*Rh + Lh*Ro] + D[L0*Roh + Loh*R0].

The first term is created at this product; the second consumes a previously mixed
input. This distinction is operational, not a semantic discovery claim. Existing
planted new-only/inherited-only, rescaling and row-permutation controls pass; maximum
closure error4.45e-16. It was previously used at final-write accounting, which failed
to establish MLP sufficiency. The new test includes the RMS/value path and suffix
that accounting omitted. It therefore asks a different causal question.

Demote native Gram-only replay: it would validate another universal compiler identity,
with little power to distinguish learned mechanisms. Execute the bounded MLP8 origin
test instead. First require a material source effect, then compare new versus inherited
branches to the complete source effect. Separately test whether the complete MLP8
source reproduces the full partial value-component effect. If it does not, do not
name MLP8 the source of the whole circuit. If neither branch passes, retain the joint
operation; no smaller-error winner, head/rank scan, or filtered-world rescue.

The preregistration is MATURE_VALUE_MLP8_ORIGIN_V1_PREREGISTRATION.md; v3 capture exposes
existing source8 inputs/outputs while preserving completed v2. Implementation and
CPU controls are executed in this review block. The GPU runner uses the managed queue.
This advances computational specification and within-module splitting; all four user
properties still need broader evidence. Next mathematical review10:49 UTC; hourly08:14.
