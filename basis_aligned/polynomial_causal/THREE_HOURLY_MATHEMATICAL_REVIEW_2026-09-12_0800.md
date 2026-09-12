# Mathematical review: deeper paths, source cancellation and coordinate redundancy

12 September 2026, 08:00 checkpoint; written at the first safe boundary after the native source experiment. Previous mathematical review: 05:00. The objective remains a simpler executable model with OOD prediction, extraction, selective removal and composition/reuse. The bilinear handoff and later weights-first instructions govern this work. No circuit is promoted here.

## The actual object

The extracted conditional program has three symmetric matrices $A_0,A_3,A_8\in\mathbb R^{1152\times1152}$ and two residual writers $w_3,w_8\in\mathbb R^{1152}$. They were obtained by weight-only fitting and exact folding through MLP16. Its two branches are

$$
f_j(x)=\frac{(x^TA_0x)(x^TA_jx)}{n_{17}}w_j,
\qquad j\in\{3,8\}.
$$

Here $x=\operatorname{RMS}(r+a)$ is the native MLP16 input, $r$ includes block16 residual re-entry, and $a=O_{16}z$ is attention16's output. The vector $z$ concatenates nine 128-dimensional head outputs; $O_{16}$ is $1152\times1152$. The port $n_{17}$ is the actual downstream mean-square plus native float32 epsilon, not an independently estimated constant. The full output still uses the native residual background, final RMS, all 50,304 unembedding rows and $30\tanh(\cdot/30)$.

The numerator is quartic in the common input. Packed storage is 1,994,688 float32 values for both branches, with approximately three packed quadratic dot products per input and two scalar products followed by residual writes. The original producer has width 4,608. Folding another bilinear producer can raise numerator degree to eight; explicitly expanding a $1152^8$ tensor is unnecessary and infeasible. RMS and QK normalization remain rational/nonpolynomial operations outside such a homogeneous numerator.

Reader/writer rescalings and signs can preserve branch functions, and shared quadratic coordinates can admit alternative representations. Native heads are source labels, not identified semantic units. The allowed native source tuples are constrained by the upstream model; arbitrary independently sampled $r,z$ are an algebraic test domain, not training-distribution examples.

## Exact source algebra and the native decision

With attention gain $g$, each unnormalized reader is

$$
q_j(g)=r^TA_jr+2g\,r^TA_ja+g^2a^TA_ja.
$$

Let $q_0(g)q_j(g)=\sum_{k=0}^4c_{jk}g^k$ and

$$
n_{16}(g)=\frac{\|r+ga\|^2}{1152}+\epsilon.
\qquad
f_j(g)=\frac{\sum_{k=0}^4c_{jk}g^k}{n_{16}(g)^2n_{17}}w_j.
$$

This represents a conditional producer-edge intervention. A whole-model attention ablation would also change other descendants and $n_{17}$. The five signed numerator sectors are not independent energy fractions.

The managed [native source experiment](MATCHED_PARTNER_ATTENTION16_SOURCE_V1_RESULT.json) used all 128 frozen fresh endpoints and 16 body forwards. Native replay errors were at most $9.23\times10^{-7}$; direct/rational gain discrepancy was below $1.70\times10^{-15}$. Residual-only numerator swaps reproduced branch8 within symmetric relative RMS errors of 3.17%, 5.44% and 7.96% on gerund, intervening-adverb and progressive families. All signs agreed. Quoted-control error was 22.85%, and remains reported. Attention-dependent sectors alone failed the registered active-family approximation. Recomputing $n_{16}$ at gain zero retained the same broad conclusion.

The strongest immediate alternative is that individually large attention-degree sectors cancel. The executed [CPU cancellation audit](ATTENTION16_SOURCE_CANCELLATION_V1_RESULT.json) rejects that explanation **between the five aggregated sectors on these endpoints**: summed norms of attention-degree changes were only 2.97%, 5.42% and 7.40% of the full scalar-change norm; their sum-of-norms/net-norm ratios were 1.009–1.030. This does not rule out cancellation among heads, positions or terms inside one sector. Native branch effects are mostly inherited through the residual source on these constructions. That directs the next trace toward MLP15 and earlier residual computations, rather than assuming this branch's attention16 route is the promising one.

## What apparent sparsity after OV folding would mean

This is an elementary derivation for our actual object, separate from a fitted claim. Put $s=(r,z)$ and $T=[I\ O_{16}]$. Then

$$
B_j=T^TA_jT=
\begin{bmatrix}
A_j&A_jO_{16}\\
O_{16}^TA_j&O_{16}^TA_jO_{16}
\end{bmatrix}.
$$

$T$ has full row rank, and every $B_j$ has the common null directions $(-O_{16}v,v)$. This is a 1,152-dimensional redundancy of the 2,304-dimensional source description. Finding that common null block is not discovering a circuit. It says two independently adjustable sources can cancel before this reader.

Set $H=TT^T=I+O_{16}O_{16}^T$ and $E=T^TH^{-1/2}$. Then

$$
E^TE=I,\qquad E^TB_jE=H^{1/2}A_jH^{1/2}.
$$

The observable quotient is therefore an invertible congruence of the original forms. It preserves rank but generally changes eigenvalues, Frobenius norms and orthogonal block structure. The residual/residual block is already $A_j$: the unrestricted source tensor does not eliminate its complexity. A change of sparsity in a chosen source metric can still be useful, but must be distinguished from a smaller computational description with all coordinate maps charged. The CPU audit verifies the quotient, isometry and nullspace identities on a small nonsingular control, with errors below $3\times10^{-15}$.

## Literature mapping and limits

[Maehara–Murota's commutant method](https://www-new.keisu.t.u-tokyo.ac.jp/data/2009/METR09-53.pdf) finds simultaneous orthogonal/unitary blocks of square matrices, with explicit numerical error control. Our direct mapping is the three symmetric quotient matrices $H^{1/2}A_jH^{1/2}$. Common commuting projectors define independent linear subspaces for these forms. This solves a restricted block problem, not arbitrary arithmetic DAG recovery. A dense commutator system has $d^2$ unknowns; explicitly forming its normal operator costs $O(d^4)$ storage, whereas a matrix-free application costs $O(3d^3)$. Exact algebraic decomposition does not provide a semantic uniqueness guarantee or ensure a stable approximate partition in trained, rounded weights. Congruence and nonlinear normalized source reachability further limit transfer of a block claim back to circuits. The 05:00 review already examined commutants; the new point here is removing the specific source duplication before applying such a test.

[Cohen, Sharir and Shashua](https://proceedings.mlr.press/v49/cohen16.html) relate their shallow arithmetic architecture to CP and its hierarchical counterpart to hierarchical Tucker, proving depth-efficiency results for that setting. Our nested quadratic products motivate retaining intermediate computations instead of flattening everything into CP. However, their architecture and input-slot assumptions do not match our repeated residual input, signed re-entry, shared attention values or RMS denominators. Their results do not guarantee recoverability, sparsity, or a small rank for this trained model. They support a representation comparison, not a claim that our optimizer should recover a particular circuit.

## Executable consequence and next direction

The source audit and quotient control are executed consequences, not future promises. They distinguish small net attention due to cancellation from small aggregate attention terms, and identify an artificial null block that must not be counted as discovered structure. Both are more informative here than another unrestricted sparse Tucker fit on duplicated source coordinates.

Next, retain the exact normalized conditional interface and separate the residual stream into the preceding MLP write and its remaining background. For a gain on that write, the same five-sector executor applies without a dense eighth-order tensor. Then compare the shared parent and both partners at the same upstream boundary. A substantial producer-dependent effect would justify folding its weights and searching shared intermediate computations; residual-only preservation would instead send the trace farther upstream. Native replay and original signed effects must be preserved. This changes the computational specification, cross-boundary grouping and extraction evidence; coefficient rank alone is not its success criterion. Full OOD prediction, selective removal and compositional adoption remain incomplete.

Next mathematical checkpoint: 11:00 UTC; hourly checkpoint remains 08:36 UTC.
