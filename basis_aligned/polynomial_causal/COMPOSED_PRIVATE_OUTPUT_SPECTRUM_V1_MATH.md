# Output-rank limits after folding the private correction through MLP9

13 September 2026. The shared-tail executor's private vector has the form

$$
P_{02}=K_{10}(v_0,\lambda J_9w)-2\gamma\lambda J_{10}m_0,
\qquad J_{10}x=K_{10}(v_0,x).
$$

Here $m_0$ is the pristine bias-free normalized MLP9 output. Its quadratic
numerator is $B_9(z)=D_9[(L_9z)\odot(R_9z)]$. The new target is the composed
quadratic numerator $J_{10}B_9(z)$, with output dimension 1152 and input
dimension 1152. The fixed offset, scalar multipliers and normalization remain
outside this diagnostic. This examines global output compression of the
private computation, rather than again fitting the upstream response map.

For hidden rows $l_k,r_k$, define the symmetric coefficient matrices
$S_k=(l_kr_k^T+r_kl_k^T)/2$. Their exact coefficient Gram is

$$
H_{kl}=\langle S_k,S_l\rangle_F
=\tfrac12[(l_k^Tl_l)(r_k^Tr_l)+(l_k^Tr_l)(r_k^Tl_l)].
$$

The output Gram of MLP9 is $G_9=D_9HD_9^T$; after composition it is
$G=J_{10}G_9J_{10}^T$. Summing the discarded eigenvalues gives the optimal
squared error for any fixed rank-$r$ output subspace in this coefficient norm.
This is a closed-form optimum for that restriction, not an unconverged fit.
A small independently materialized coefficient tensor checks the Gram identity
to relative error $1.57\times10^{-16}$.

| Object | Best rank-64 relative error | Rank needed for 10% error |
|---|---:|---:|
| MLP9 quadratic numerator | 92.40% | 1104 |
| Fixed downstream linear map alone | 79.31% | 806 |
| Composed quadratic numerator | 77.19% | 763 |

Folding improves concentration but fails the declared rank-64/10% target.
The rank requirements refer to their respective norms and objects; they are
not equal-cost executor comparisons. In particular, factoring a 1152-square
linear map into two dense rank-763 factors costs more scalars than storing it
densely. A composed output basis also needs its input computation priced.

The negative result could depend on coefficient weighting rather than
function weighting. For standard Gaussian input, let $t_o$ be the trace of
output quadratic form $o$. The exact output second moment is $2G+tt^T$.
Its rank-64 error is 75.05%, and it needs rank 756 for 10%. Alternatively,
remove each form's radial part $(t_o/1152)I$: the coefficient Gram becomes
$G-tt^T/1152$. This traceless remainder still needs rank 763. Thus neither
Gaussian trace weighting nor separating the radial component rescues a small
global output subspace. These are text-independent counterchecks, not native
data weighting. They do not rule out restricted input domains, selected output
readers, nonlinear shared graphs or sparse product structure.

The spectrum computation and radial countercheck each take about three CPU
seconds. The next search should change the representation or output scope
rather than optimize this same output-rank restriction longer. Preserve the
exact shared-tail implementation and its native validation in the meantime.

Code: `composed_private_output_spectrum_v1.py`, optional `--radial`.
Receipts: `COMPOSED_PRIVATE_OUTPUT_SPECTRUM_V1_RESULT.json` and
`COMPOSED_PRIVATE_OUTPUT_RADIAL_V1_RESULT.json`.
