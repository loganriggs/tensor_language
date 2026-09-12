# Folding the joint routing/value computation into the last bilinear layer

The new operator includes **QK1 times QK2 times the value**, rather than fitting
one routing factor or treating the entire route as an unspecified scalar.
Dense coefficient and gradient controls pass. A native-weight replay through
attention17, the last bilinear layer and the full unembedding also passes.
This establishes the object to decompose; it does not yet find sparse factors
or a circuit with the four behavioral properties.

## What differs from previous work

[Joint QK/value source ports](JOINT_QK_VALUE_PORTS_V1_PREREGISTRATION.md) measured
the joint biquadratic routing numerator's overlap with source directions read
through OV. [Value-coherent attention folding](SHARED_SOURCE_ATTENTION_QUADRATIC_V1_MATH.md)
composed downstream quadratics with value/output maps while retaining routing
coefficients as ports. Both are relevant prior work. Neither is a factorization
of the full joint routing-times-value coefficient tensor introduced here.
The older pencil and common MLP15-interface methods are separate families.

## Exact joint polynomial and its normalization ports

Let $q\in\mathbb R^{1152}$ be a normalized current-layer query residual and
$s\in\mathbb R^{2304}$ concatenate normalized current-layer and block0-value
source residuals. For a fixed query/source position pair, compile its rotary
matrices into the key maps and define

$$
A_h=Q_{1h}^TR_{q}^TR_s[K_{1h}\;0],\qquad
B_h=Q_{2h}^TR_{q}^TR_s[K_{2h}\;0],
$$

$$
V_h=[(1-\lambda)V_{17,h}\;\lambda V_{0,h}],\qquad M_h=O_hV_h.
$$

The source contribution is

$$
a(q,s;g)=\sum_h g_h(q^TA_hs)(q^TB_hs)M_hs.
$$

For fixed gates $g_h$, this has query degree2 and source degree3. The native
gate is the reciprocal of $128^2$ times all four projected query/key RMS
factors. Those gates depend on actual inputs and remain explicit external
ports; the normalized attention function is not a homogeneous quintic.
The native signed value mixture is $\lambda=-0.0888671875$, not a convex mixture.
There is no softmax, and only causal source positions are included.

The separately symmetric coefficient contraction is

$$
\mathcal A(q_1,q_2;s_1,s_2,s_3;g)
=\frac1{12}\sum_h g_h
\sum_{\tau\in S_2,\,\pi\in S_3}
(q_{\tau(1)}^TA_hs_{\pi(1)})
(q_{\tau(2)}^TB_hs_{\pi(2)})M_hs_{\pi(3)}.
$$

Here $S_2,S_3$ mean the two and six possible permutations. On repeated inputs
this recovers the actual source contribution when its native gate is supplied.
The code keeps the native rank128 query/key/value maps factored, rather than
forming the large $A_h,B_h,M_h$ matrices or a fifth-order tensor.

QK1 and QK2 remain jointly necessary factors in each term. The symmetry is over
input copies; it does not assign one task to QK1 and another to QK2.

## Composing the full unembedding and last bilinear layer

Write $a=\sum_p a(q,s_p;g_p)$ and let $r$ be the remaining input to MLP17.
With its native matrices $L,R,D$, bias $b$, and unembedding $U$, the unembedded
MLP contribution is

$$
\frac{UD}{\nu}\left[
(Lr)\odot(Rr)
+(Lr)\odot(Ra)+(La)\odot(Rr)
+(La)\odot(Ra)\right]+Ub,
$$

$$
\nu=\|r+a\|^2/1152+\epsilon.
$$

These are residual/residual, mixed and attention/attention interactions.
Conditioned on the gates, mixed terms have residual degree1, query degree2 and
source degree3. The same-source attention/attention term has query degree4
and source degree6; different-source terms retain separate source arguments.
This is the route from the unembedding through the last bilinear layer into
joint routing and values. The large $UD$ matrix is implicit: apply $D$ and then
$U$ to each batch of products. Input normalization, final residual background,
final RMS and capped logits have not been eliminated.

We have a full separately multilinear operator for the attention contribution
and an exact diagonal execution of this downstream expansion. A general
multilinear coefficient optimizer for the degree-ten attention/attention term
is not implemented by this replay alone.

## Why tying keys and values can expose additional structure

The [executed cancellation control](JOINT_ROUTING_VALUE_ROLE_CANCELLATION_V1_CONTROL.json)
uses two nonzero heads with joint contributions

$$
q^2s_0s_1s_0-q^2s_0s_0s_1=0.
$$

Their separately symmetric joint tensors cancel exactly. If the value is
treated as an independent variable $v$, the corresponding expression is
$q^2(s_0s_1v_0-s_0^2v_1)$ and does not vanish. The independent-role coefficient
norm is1.2247, while the joint sum has norm0 and each joint head has norm0.57735.
This shows why factorizing routing and value roles separately can miss an
equivalence that appears in the composed function. It is a planted example,
not evidence that trained heads cancel. Different native head normalizers can
also destroy a numerator-only cancellation.

## Executed checks and remaining work

- [Dense coefficient/gradient control](JOINT_ROUTING_VALUE_POLYNOMIAL_V1_CONTROL.json):
  all value, permutation, diagonal and six-map gradient errors below6.45e-16.
- [Native-weight replay](JOINT_ROUTING_VALUE_NATIVE_V1_RESULT.json): query
  position8, sources0/7,12 synthetic normalized ports; attention error below
  2.78e-15, full-unembedding bilinear expansion error3.51e-15, permutation error
  1.51e-15. All registered instrument criteria pass in0.43 CPU seconds.
- No model-body forwards, text fitting, new model parameters or sparse learned
  factors. Native matrices and external ports remain fully charged.

The next structural question is whether joint routing/value computations share
or cancel beyond what their separate maps suggest, and whether those relations
survive restoration of native gates. Any proposed group must then be tested for
held-out/OOD prediction, declared-boundary extraction, selective intervention
and composition/reuse. Exact folding alone does not satisfy those requirements.
