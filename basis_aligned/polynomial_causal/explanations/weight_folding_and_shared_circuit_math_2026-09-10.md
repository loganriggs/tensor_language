# Weight folding and shared circuits: mathematical reassessment

10 September 2026, 13:00 UTC.

**Weight-based circuit discovery remains plausible, but the current evidence does
not show that a decomposition makes the desired circuits fall out.** The most
useful part of the original handoff is its proposal to identify shared operations
through their inputs, routing rules, outputs and downstream consumers. Another
is/was axis or generic tensor-rank sweep would have weak justification now.

The documents you recalled are the
[original handoff](bilinear_circuit_reconstruction_codex_handoff.md) and the
[pilot report](bilinear_reconstruction_pilot_report.md). The handoff proposes the
direction; the pilot reports faithful execution and a negative discovery result.
Its explicit recommendation is to stop the tested local sharing search and move
to joint read–route–write operations. The appended success criterion requires a
previously unspecified reusable computation, with structural simplicity and
held-out causal evidence.

This reassessment produced a small executable mathematical control using trained
weights. It shows that two attention descriptions can have **identical folded
QK numerator tensors while their actual normalized routing differs by 53–55%**
on the fixed test inputs. A correct fold retaining the normalization terms agrees
with direct execution. That gives us a concrete way to catch false claims of
shared computation. It does not identify a new language circuit, and it does not
explain away the previous valid negative results, whose formulas already retained
normalization.

**What we can fold through a bilinear MLP.** Let

\[
M(u)=D[(Lu)\odot(Ru)]+b.
\]

Here \(u\) is the normalized residual input, \(L,R\) compute linear input
features, \(\odot\) multiplies corresponding features, and \(D\) writes the
products into the residual stream. For a downstream linear reader \(c\),

\[
c^T M(u)=u^T Q_c u+c^Tb,
\qquad Q_c=\operatorname{sym}(L^T\operatorname{diag}(D^Tc)R).
\]

The operation \(\operatorname{sym}(A)=(A+A^T)/2\) removes a part that vanishes
when both inputs are the same vector. This is an exact way to ask which products
the consumer uses. It avoids constructing the complete three-dimensional weight
tensor. Bilinear weight analysis and output-conditioned spectral methods have
published precedents in [Sharkey](https://arxiv.org/abs/2305.03452) and
[Pearce et al.](https://arxiv.org/abs/2410.08417). Those results motivate candidate
discovery; they do not guarantee a compact explanation of this checkpoint.

If an attention output \(O_hz_h\) enters this MLP, its writer can be folded
into the input maps as \(LO_h\) and \(RO_h\). With two such writes, the
reader's numerator contains the explicit interaction

\[
a^T O_A^T(Q_c+Q_c^T)O_B b.
\]

That says how the two attention computations combine inside the MLP. Terms
involving the background and each write separately also remain. If RMSNorm lies
between the writes and MLP, its denominator must be recomputed after each edit.
Earlier exact polarization and causal tests already cover this algebra; repeating
the derivation alone would add little evidence.

**What a second circuit tells us.** Two consumers provide two functions
\(Q_A,Q_B\) of the same inputs. We can then distinguish:

| Possible relationship | Example | Implication for the program |
|---|---|---|
| Same computed product | Both consume \(x_0x_1\) | Compute that product once, retain both output adapters |
| Same input feature, different products | \(x_0x_1\) and \(x_0x_2\) | Share the \(x_0\) producer, retain two multiplications |
| Coupled private inputs | A consumer needs an A-input times a B-input | Preserve an explicit interaction between the branches |
| Same module location only | Different input dependencies and writes | Location supplies no evidence of a shared algorithm |

These distinctions survive a simple caution: the symmetric matrices for
\(x_0x_1\) and \(x_0x_2\) have zero coefficient inner product despite sharing
\(x_0\). Conversely, similar output directions need not have the same producer.
Thus matrix similarity is an incomplete measure of computational sharing.

Multiple consumers constrain a candidate decomposition, but they do not generally
make it unique. The earlier two-reader MLP1 work already found that a local split
could preserve much of each task's interchange effect while failing selective
removal or controls. The detailed [two-circuit mathematical record](weight_tensor_two_circuit_math_2026-09-09.md)
preserves those findings. They justify testing a new operation rather than tuning
that same split.

**The complete attention fold needs six matrices.** For one head of width
\(d=128\), let \(x,y\) be its query/source residual inputs, and \(Q_i,K_i\)
the learned projections for branch \(i\in\{1,2\}\). Let \(R_t,R_s\) be the
actual rotary matrices at the two positions. Define

\[
A_i(t,s)=Q_i^TR_t^TR_sK_i/d,
\quad G_{Qi}=Q_i^TQ_i/d,
\quad G_{Ki}=K_i^TK_i/d.
\]

The complete scalar routing rule is

\[
p_{ts}(x,y)=
\frac{(x^TA_1y)(x^TA_2y)}
{\sqrt{\prod_{i=1}^2
(x^TG_{Qi}x+\epsilon)(y^TG_{Ki}y+\epsilon)}}.
\]

This follows by substituting each projection into the model's head RMSNorm,
rotary operation, and two dot products. Epsilon is the deployed FP32 value,
\(1.1920928955\times10^{-7}\). The formula describes an allowed causal edge;
future-source edges are masked to zero. Upstream production of \(x,y\) remains
part of the full circuit.

Two numerator matrices and four norm matrices provide a sufficient local
description. Equality of all six is sufficient for equal routing at the stated
positions. It is not a necessary canonical signature: branch exchange and other
function-preserving relations can yield alternative descriptions. Shared values
still need their own reader/writer analysis. The recurrent attention formulation
uses associativity to execute such routing and value updates; its established
background is [Katharopoulos et al.](https://arxiv.org/abs/2006.16236).

**The new trained-weight control.** Freeze layer 0, head 0 before any outcome.
For the first branch, change \(Q_1\) to \(SQ_1\) and \(K_1\) to
\(S^{-T}K_1\). The diagonal matrix \(S\) scales half the rotary planes by 2
and half by 1/2, using the same scale on each plane's paired coordinates.
Consequently \(S\) commutes with every rotary matrix, including its rounded
implementation, and

\[
Q_1'^TR_t^TR_sK_1'=Q_1^TR_t^TR_sK_1.
\]

Both routing numerators are unchanged. But the query and key squared lengths
change, so the normalized routing usually changes. This is a proposed weight
change, not a valid symmetry of the whole normalized head. In contrast, using
paired signs instead of unequal scales preserves both norms and routing.

CPU execution on 64 fixed Gaussian query/source pairs at four causal position
pairs gave:

| Quantity | Measured result |
|---|---:|
| Scaled versus original numerator matrices | Exactly equal in this FP64 computation |
| Complete routing relative L2 change | 0.528–0.553 |
| Same-token routing relative L2 change | 0.534 |
| Valid paired-sign routing change | Exactly zero in this computation |
| Fold versus direct FP64 maximum absolute error | \(9.71\times10^{-17}\) |
| Fold versus direct FP32 maximum absolute error | \(1.80\times10^{-8}\) |
| Independent native rotary-helper bridge maximum error | \(3.73\times10^{-9}\) |

Relative L2 here means the length of the difference vector divided by the length
of the original score vector across 64 pairs. It is neither a percentage of
attention mass nor a language-loss change. All four preregistered predicates
passed. Main execution took 1.81 CPU wall seconds with two threads. These are
continuous local inputs, with no model text forward and no claim of natural
reachability. All original 545,902,902 parameters remain necessary and charged.

A subsequent native-helper audit also measured the effect of rounded positional
tables. The pairs \((31,7)\) and \((24,0)\) have the same lag, but their rotary
products differ by relative L2 0.00178; the corresponding router scores differ
by 0.00216 on this input panel. Ideal real-valued rotary matrices obey a relative
position identity; the deployed BF16 tables do not obey it exactly. An exact
compiler should therefore retain both positions or explicitly account for this
approximation. This is a numerical execution detail, not a new semantic feature.

**How this changes the search.** The appropriate unit to propose is a small
operation with named inputs and multiple consumers. Fold its complete local
dependencies, including normalization, before claiming two implementations are
equivalent. Then test the shared producer and consumer-specific calls separately:
removing a shared producer affects all calls, whereas disabling one call should
retain the others. The pilot's separate editable histories remain relevant.

| Requested property | What this mathematics contributes | Evidence still required |
|---|---|---|
| OOD prediction | An explicit function to evaluate on new inputs | Prediction on fresh constructions and native states |
| Extraction | Exact local producer/reader formulas | A complete executable input path and smaller explained program |
| Selective removal | Precise distinction between producer and call edits | Intended behavioral change with unrelated behaviors preserved |
| Composition/reuse | Shared products and explicit interaction terms | Held-out joint edits with downstream recomputation |

The next semantic application should start from a circuit with a frozen,
replayable interface. For example, the recent correlative v505 receipt reports
successful interchange measurements, but its JSON stores the number of units
without their chosen identities or the fitted direction. That receipt alone
cannot be folded into a weight-level producer. Recovering those artifacts is a
specific dependency; silently refitting them would produce a different candidate.
The present audit leaves that separately owned experimental lane unchanged.

The new code is diagnostic machinery. Each dense matrix is 10.125 MiB, and six
matrices require 60.75 MiB versus 589,824 coefficients in the original four maps.
Keeping factors is often cheaper. This calculation earns no structural savings
and supplies no justification for a broad decomposition sweep. Its contribution
is an executable falsifier of an insufficient sharing criterion.

Receipts: [frozen protocol](../NORMALIZED_ROUTER_FOLD_V1_PREREGISTRATION.md),
[fold helper](../folded_normalized_router_v1.py),
[CPU audit](../audit_normalized_router_fold_v1.py),
[main result](../NORMALIZED_ROUTER_FOLD_V1_RESULT.json), and
[independent native-helper result](../NORMALIZED_ROUTER_NATIVE_BRIDGE_V1_RESULT.json).
