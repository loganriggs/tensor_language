# Mathematical review: a compact value computation still needs its normalization interface

## Current circuit evidence and mathematical object

The four-mode MLP8-to-head9 value path passed native removal, numerator donation and64newline controls. On the newoutput city cues, removal reduces contrast13.38% and donor transfer is12.32%, with all24donor directions correct. Nationality effects are small and opposite; style effects are smaller and mixed in sign. Maximumnewline damage is0.02533nats. These are conditional path-edge interventions, not deletion of the MLP8node, broadcorpusOOD or autonomous extraction.

Let the raw MLP8input be $z_j\in\mathbb R^{1152}$ and $r_j=\|z_j\|^2/1152+\epsilon$, with native $\epsilon>0$. The frozen four-mode computation is

$$
\phi_4(z_j)=\frac{z_j^TM_4z_j}{r_j},\qquad
M_4=U_4\Lambda_4U_4^T,\quad U_4\in\mathbb R^{1152\times4}.
$$

The signed eigenvalues have inertia(onepositive,threenegative). Its physical transmitted write at query position $t$ is

$$
w_t=d_9\sum_{j\le t}\Gamma^{\mathrm{even}}_9(q_t,s_j)
\frac{\lambda_{9,0}}{\rho_{9,j}}\phi_4(z_j).
$$

Here $d_9\in\mathbb R^{1152}$ is the fixed physicalwriter; $q_t,s_j\in\mathbb R^{1152}$ are supplied normalized attention9states; eachQKprojection has128rows; the source-key projector has64columns. Gamma keeps both QK factors, original projected-vector norms, BF16-rounded rotary tables and reflection-even key numerator. The source value is a scalar, and the source positions sum into one scalar write amplitude. Recipient rho9 and routing stay recipient-native under phi donation.

With query, key-source and rawMLP-source treated as independent ports, the unnormalized numerator has degrees(2,2,2), total6. The actual network is not globally a degree6polynomial: RMS, shared dependent inputs and final tanh remain. The target of this review is the local normalized quadratic, with coefficient-Frobenius identities and native phi replay, not a claim about the wholemodel's tensor rank.

## Exact interface obstruction: low numerator rank is not a four-raw-reader circuit

The neighboring concept is a ridgefunction $f(z)=g(Lz)$, a function of a linear projection. Gradient-based active-subspace methods seek useful approximations of this kind under a chosen distribution; near-stationarity is not a global exact representation or semantic-recovery guarantee. We are not starting a data-weighted ridge fit. [Constantine–Eftekhari–Hokanson–Ward](https://arxiv.org/abs/1606.01929).

**Derived obstruction for our object.** If $M\ne0$ and $\epsilon>0$, then

$$
f(z)=\frac{z^TMz}{\|z\|^2/d+\epsilon}
$$

cannot factor globally through a noninjective linear map $L$, even allowing an arbitrary function $g$ after that map.

Proof: take nonzero $w\in\ker L$. Constancy along each fiber would give $f(z+tw)=f(z)$ for all $z,t$. At $z=0$, the positiveepsilon makes $f(0)=0$, so $w^TMw=0$. Multiply the fiber equality by its denominator and compare coefficients of $t^2$:

$$
0=f(z)\frac{\|w\|^2}{d}.
$$

Thus $f$ would vanish identically, contradicting $M\ne0$. This is an exact statement on all rawvectors; it doesnot prove that the counterexample is reachable from text, or that low-dimensional approximation must be poor on a restricted distribution. For already normalized input $x=\operatorname{RMS}(z)$, fourreadings do suffice; the normalization has then been supplied upstream.

The positive resolution is simple: fourlinear readings $U_4^Tz$ **plus one nonlinear norm scalar** $\|z\|^2$ exactly determine phi4. That extra scalar is shareable across all paths using the same nativeRMS, but must be counted and generated. The gradient makes the missing dependence explicit:

$$
\nabla f(z)=\frac{2Mz}{r}-\frac{2(z^TMz)z}{d r^2}.
$$

The second term can point outside the four-reader span. On cachednative inputs its outside-span fraction has median0.83%, maximum4.77%: a global obstruction neednot mean large local error. This prevents both overclaiming exact autonomy and dismissing a useful approximate low-dimensional representation.

## Executed consequence

Construct rawvectors with the same fourreadings and increasing norm in their orthogonal complement. The readings agree within1.06e-15relative, while phi changes by50% and80%. Adding the normport replays the nativeFP32-normalized computation within1.33e-7relative on the cached inputs. The reloaded standalone executor accepts only four scaledreadings and a raw normscalar. [Witness and controls](QUADRATIC_NORM_PORT_V1_RESULT.json) · [Executor](phi4_raw_ports_v1.py) · [Reloaded executor check](PHI4_RAW_PORTS_V1_CONTROL.json).

This is a usable interface specification, not another fit. It supplies a falsifier for any proposed exact extractor that retains only four raw linear coordinates while silently dropping normalization.

## Arithmetic alternatives and prior-art correction

Signed-square pairing alreadyexists in this repository; it is not a new discovery from this cycle. Standard inertia describes the dimensions of positive/negative definite subspaces independently of the chosen sum-of-squares coordinates. [Primary formalization of the inertia statement](https://raw.githubusercontent.com/leanprover-community/mathlib4/master/Mathlib/LinearAlgebra/QuadraticForm/Signature.lean).

For this particular numerator, absorb eigenvalue magnitudes into fourreadings and write $b^2-a^2-c^2-e^2$. Reuse the prior identity:

$$
b^2-a^2-c^2-e^2=(b+a)(b-a)-c^2-e^2.
$$

Three real homogeneous-linear products suffice. They are minimal **within sums of such products**: if fewer thanthree products represented a form with a threedimensional negative-definite subspace, that subspace would contain a nonzero vector annihilating every left factor, forcing the form to be zero there, a contradiction. This is not an unrestricted arithmetic-circuit lower bound with normalization/division or higher-degree cancellation allowed.

Coefficient replay is1.87e-16relative and native numerator replay3.46e-15. The fourstored readers cost4,608scalars. Counting only three multiplications hides four1152Ddotproducts, twoextra additions, the norm's1152squares, a division, attentionrouting and nativeinputgeneration. No measured speedup or wholeprogram-price reduction is claimed. The positive eigenmode can pair with different negative modes; individual product nodes are not thereby uniquely identified semantic circuits. The quadratic function is the better-supported unit.

## Serious alternative: simultaneous congruence diagonalization

For a family of consumer quadratics $M_1,\ldots,M_k$, shared nonorthogonal squaredreaders correspond to simultaneous diagonalization by congruence. He–Kressner's randomized algorithm uses two random combinations; their Theorem7 supplies exact recovery for an exactlySDC family. Regularity means a nonsingular combination exists; robustbounds additionally depend on conditioning/separation. The core generalized eigenproblem is dense cubiccost after quadraticcost combination formation. [Primary paper, definitions and Theorems7/11](https://arxiv.org/html/2402.16557v3).

Mapping here: the numerator/denominator pair is $(M_4,I/d)$, which is regular and alreadydiagonalizable in the known eigenbasis. It doesnot eliminate the denominator's fullrank. Moreover, the paper's stronger PD-family assumption requires everypositive combination to be positive definite; indefinite $M_4$ doesnot satisfy it merely because $I$ is present. There is no justified newoptimization campaign for this already solved pair. Multiple actual consumerforms would be needed to make shared-reader discovery a nontrivial next use. Their simultaneous diagonalizability is not established by the current oneconsumer result, and no uniqueness of semanticfactors follows.

Tensor-train/hierarchical representations could reorganize a larger contraction, but do not by themselves resolve this normalizedinput dependence. A small weighted-automaton/Hankel realization would require an independently verified finite linear predictive-state representation that we do not have. The exact five-port construction dominates those detours for the present circuit.

## Resulting action and remaining gap

The norm-aware raw-port executor and witness have been implemented and tested. Keep the fourmode block and its normalization interface intact when extending backward or comparing arithmetic graphs. The next behavioral question is fresh-context stability and composition with other branches; the next extraction question is generating its declared inputs, not another lower numerator rank. Sourcevalue transfer/newline preservation is positive; nativebackground, prefixgeneration, broaderOOD and generalreuse remain unfinished. Generic normalized-Frobenius guessed-graph search stays deferred as the user requested.

