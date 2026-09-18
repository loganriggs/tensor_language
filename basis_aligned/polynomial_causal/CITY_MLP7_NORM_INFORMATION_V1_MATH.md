# What the three-reader fold does not preserve

The 384 folded reader coordinates do not retain the complete output-norm
information of MLP7 on arbitrary hidden vectors. This is a weight-level
restriction, not a demonstrated counterexample on reachable token states.

Let h=(Lz)⊙(Rz), D be MLP7 Down, b its bias, W the stacked city readers, and
C=WD. The fold returns Ch+Wb. For a supplied other-source background g, the
mixed8 squared norm includes

`||g + lambda*D h||² = ||g||² + 2 lambda*gᵀD h + lambda²*hᵀDᵀD h`.

To replace the last term by a function of Ch alone for arbitrary h, every v
with Cv=0 must also satisfy Dv=0. Otherwise h=t*v leaves Ch unchanged while
the squared norm has positive quadratic coefficient ||Dv||². Equivalently,
no matrix A can satisfy DᵀD=CᵀAC when such a direction exists.

The CPU control constructs a unit v orthogonal to the row span of C. Its
normalized reader residual is3.66e-17, while ||Dv||²=7.8413. With the actual
MLP7 bias, h=−v,0,+v yields squared output norms1050.03,1055.15,1075.94,
respectively. The three readings remain unchanged to numerical precision.
The Down component outside this reader row span has relative Frobenius norm
0.7654. This is not an activation variance fraction or a behavioral error.

Scope matters: this construction treats h as unconstrained. It does not prove
that these three h values are reachable as products of Lz and Rz, let alone
from text. Nor does it prevent calculating the norm from upstream z and the
full weights. It rules out a universal hidden-space replacement based only on
the three projected readings; a restricted-state approximation would need its
own predictive and intervention evidence.

Literal alternatives illustrate the unresolved cost. D stores5,308,416 values;
an explicit hidden Gram DᵀD would store21,233,664. Keeping D and b alongside
the folded-reader program would cost17,696,256 values, versus16,368,768 for
the unfused L/R/D/b plus W calculation. This simple exact norm implementation
therefore erases the local reader-only saving. These are local alternatives,
not a lower bound on every possible complete tensor program.

Consequence: keep mixed8 RMS and query/context inputs explicitly external in
the present reader certificate. Do not infer full-executor storage savings or
norm-port closure from the12,386,688-value reader program. The next full
replacement must account for norm generation and sharing of native factors,
or validate an approximation at that boundary. Failed source composition and
the pending native reader/installed checks are unchanged.

[CPU result](CITY_MLP7_NORM_INFORMATION_V1_RESULT.json) ·
[Control](city_mlp7_norm_information_v1.py) ·
[Saved witness](CITY_MLP7_NORM_INFORMATION_V1_WITNESS.pt).
