# Better coordinates help fitting; discovering the shared features remains hard

21 September 2026, 13:20 UTC. **An equivalent parameterization improved the native fit slightly, but did not produce an acceptable circuit. More revealingly, fitting works much better on planted examples when the shared input spaces are known than when they must also be discovered.**

This continues the [two-stage overview](research_update_2026-09-21_0104_full_coverage_and_shared_baselines.md). The target is still six quadratic reads feeding three selected components, with native intermediate states supplied. It is not the full folded MLP output or an extracted token-to-output circuit.

**The earlier stall was an optimizer result, not a local-optimum certificate**

Both previous warm-start fits selected their initial checkpoint. An independent finite-difference audit found nonzero gradients and verified that sufficiently small downhill steps improve both objectives. Large steps can worsen them. This rules out interpreting the stalled run as proof that the circuit family has reached its best fit.

We therefore changed the optimization coordinates without changing the represented graph family or its price. Instead of representing each private direction by an orthogonal-complement component plus a potentially large shared coupling, we represent the full private space by an orthonormal basis.

For one pair of quadratic reads, let $P$ be the orthonormal shared input basis, $D$ the orthonormal private basis, and $T_o$ the target coefficient matrix for read $o$. Define

$$
A_o=P^\top T_oP,\qquad W=P^\top D,\qquad J=W^\top W.
$$

The reconstructed form is

$$
\widehat T_o=D C_oD^\top+P(A_o-WC_oW^\top)P^\top.
$$

For fixed input directions, the best symmetric private core satisfies

$$
C_o-JC_oJ=D^\top T_oD-W^\top A_oW.
$$

We solve this conditional problem exactly and differentiate the remaining loss with respect to the input directions. The two native initial functions agree with the previous coordinates to approximately $3\times10^{-14}$. Independent dense solves, full-tensor losses, and finite differences validate the gradients. Degenerate shared/private overlap is rejected rather than hidden by a regularization change.

**Toy success did not translate into a large native improvement**

All toy comparisons below use 1,800 steps, learning rate 0.03 with cosine decay, and two random starts per optimizer. “Recovered” means at most 1% coefficient reconstruction error. These are **five planted instances of the same pairwise-sharing family**, varying dimension and random coefficients—not five distinct circuit architectures.

| Coordinates and information supplied | Adam: cases recovered using better restart | Muon: cases recovered using better restart |
|---|---:|---:|
| Previous private coordinates; shared spaces supplied | 2/5 | 4/5 |
| Orthonormal full private bases; shared spaces supplied | **5/5** | **5/5** |
| Orthonormal bases; shared and private spaces both learned | **0/5** | **0/5** |

For the middle row, Muon recovered all ten individual runs; Adam recovered seven. Muon won the registered aggregate criterion. The bottom row is a separate follow-up experiment with valid oracle reconstruction and gradients: its failure is retained, and no corresponding native joint fit was launched.

The completed native fixed-shared comparison ran four fits: covariance-shaped and original-coordinate coefficient objectives, each from the inherited solution and a random start. All four compiled and replayed successfully. The primary covariance result was:

| Measurement | Previous warm solution | New fit | Required limit |
|---|---:|---:|---:|
| Covariance coefficient error | 7.994% | **7.944%** | 7.734% |
| Original-coordinate coefficient error | 59.73% | **59.61%** | 60.79% |
| Component 1 value error | 2.23% | 2.23% | 2.06% |
| Component 2 value error | 2.36% | **2.34%** | 2.25% |
| Component 3 value error | 10.93% | **10.56%** | 9.69% |

The graph still uses 1,056 products, 1,058,124 floating coefficients and 1,047,648 source multiplications: a 21.26% source-arithmetic saving against the original pair baseline. Fidelity fails. The additional prediction of at least 1% relative fitting-error improvement at both warm starts also fails. All best checkpoints occur at the final step, so this does not establish convergence or global optimality.

**Execution checks do not explain away the fidelity miss**

We audited the actual shared-dictionary executor, rather than only its decoded coefficient matrices. Its input derivatives agree with independent dense formulas within the $10^{-8}$ relative requirement on 16 opened rows. FP32 versus FP64 component drift is at most $2.58\times10^{-5}$ of output variation on the 448 opened states, below the $10^{-4}$ diagnostic bar. All four new graphs pass these checks.

The prior random isotropic run's stricter compiler failure remains recorded; it has not been retrospectively converted into a pass. The FP32 audit is numerical validation, not a quantization or model-compression result. None of these checks is fresh OOD evidence.

**What this changes about the two-stage approach**

With shared spaces supplied, the optimizer can recover the planted computations. Asking the same optimizer to discover those spaces from scratch makes all five cases miss. This is evidence that feature discovery is a distinct difficulty; it is not evidence that known shared directions are sufficient in the trained model.

The next controlled toy test derives each pair's input support from its coefficient matrices, then restricts a reusable dictionary to the intersection of its two consumers' supports. These supports use only the target weights. Five controls verify that they contain the planted directions and preserve the oracle function. Random fitting inside these spaces tests whether a decomposition can provide the missing guidance.

There is a clear limitation: the native quadratic forms are full rank, so their exact supports do not provide this reduction. A successful toy result would motivate approximate subspace discovery; it would not justify pretending we already know the native shared features.

**Evidence**

- [Previous-fit descent audit](../../direct_tensor_match/FREE_PRIVATE_DESCENT_V1.json).
- [Equivalent-coordinate native controls](../../direct_tensor_match/ORTHOGONAL_PRIVATE_NATIVE_PREFLIGHT_V1.json) and [fixed-shared toy comparison](../../direct_tensor_match/ORTHOGONAL_PRIVATE_TOY_V1.json).
- [Completed native comparison](../../direct_tensor_match/ORTHOGONAL_PRIVATE_NATIVE_V1.json), 280.60 seconds, and [actual execution audit](../../direct_tensor_match/ORTHOGONAL_PRIVATE_EXECUTION_AUDIT_V1.json).
- [Joint shared/private toy failure](../../direct_tensor_match/JOINT_ORTHOGONAL_TOY_V1.json), 56.49 seconds.
- [Target-derived support controls](../../direct_tensor_match/JOINT_SUPPORTS_PREFLIGHT_V1.json) and [next fitting plan](../../direct_tensor_match/JOINT_SUPPORTS_TOY_PLAN_V1.json).

No candidate is adopted. Fresh/OOD prediction, stable feature identity, selective manipulation, reuse beyond the fitting interface and full extraction remain outstanding.
