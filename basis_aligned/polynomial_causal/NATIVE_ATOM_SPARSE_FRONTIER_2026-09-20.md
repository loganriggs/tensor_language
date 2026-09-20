# Native folded atoms: pruning cannot reach the desired fidelity

These screens follow v615's obstruction for small shared-input Tucker ranks.
The target remains the full symmetric tensor of C=UD, A=LE, B=RE with
E=[I,lambda17[0] Down16,O17], every vocabulary output and every source port.
There is no data fit or selected-output metric. All errors concern the
unnormalized numerator in the declared Euclidean z coordinates.

## Fixed-dictionary frontier

Define atom t_k=C[:,k] outer sym(A[k] outer B[k]). Its joint Gram is

G_kl = <C_k,C_l> [<A_k,A_l><B_k,B_l> + <A_k,B_l><B_k,A_l>]/2.

For retained set S, squared removal error is 1_removed^T G 1_removed.
This includes every cancellation, unlike summing individual atom energies.
v617 compares greedy removal by exact incremental error, largest atom norms,
and three random orders fixed before results. All coefficients stay native.

| Retained atoms / 4608 | Greedy error | Atom-norm error | Random seed 0 error |
| --- | --- | --- | --- |
| 2304 | 0.618455 | 0.621351 | 0.695684 |
| 4096 | 0.231138 | 0.233873 | 0.324998 |

The registered <=0.10 error at half the terms fails. Greedy cancellation
accounting helps little over simple atom magnitude. Greedy is not globally
optimal, and individual error increments need not be positive.

The sparse runtime cost must include dense input/output factors. The receipt
reports two alternative grammars: explicitly folded C,A,B, or shared U and E
plus retained native L/R/Down columns. Neither includes the computations that
produce z, normalization, or the final output operations. Reduced product
counts alone are not full circuit simplicity.

## A bound covering every subset and scalar refit

Let d_k=||t_k||, and K_kl=G_kl/(d_k d_l). For an approximation with any native
support S and arbitrary scalar coefficients, omitted atoms have fixed residual
coefficients d_k. Therefore

error² >= lambda_min(K) sum_(k not in S) d_k².

For any size-m support, replace the sum by the n-m smallest squared atom norms.
Divide by ||sum_k t_k||² to obtain a relative-error lower bound. This argument
allows arbitrary refitting on retained terms; it is not limited to greedy
selection. It does not allow new input factors or new vector output writers.

v618 recomputes the Gram in FP64 directly from E E^T and U^T U, avoiding the
FP32 QR reduction. The minimum eigenvalue of K is 0.289154794; the maximum is
3.497904353. The energy/diagonal-energy ratio agrees with v617 within 7.35e-9.
After subtracting a stated 1e-6 numerical margin from the minimum eigenvalue:

| Maximum retained atoms | Relative-error lower bound for any scalar refit |
| --- | --- |
| 64 | 0.511338 |
| 512 | 0.475141 |
| 1024 | 0.435282 |
| 2048 | 0.351109 |
| 2304 | 0.328360 |
| 3072 | 0.253356 |
| 4096 | 0.122714 |

These are floating-point numerical bounds, not interval-arithmetic
certificates. The gap to the <=0.10 target at half-budget is substantial.
This obstructs pruning and scalar refitting of the native atoms under the
global ambient metric. It does not obstruct sparse new features, different
output-sharing block terms, polynomial DAG identities, or simpler circuits
for particular behaviors.

## Controls, artifacts, and next action

`test_joint_atom_pruning.py` passes three tests: dense atom/Gram replay with
every greedy step checked, exact cancellation and scale gauge checks, and
exhaustive enumeration of every support in a small problem with optimal
least-squares refitting to validate the bound. Native selected-atom replay
against FP64 original-coordinate contraction is 1.10e-6 relative error.
Both managed GPU jobs completed successfully; the v617 discovery hypothesis
failed, while all v618 registered gates held.

Authoritative receipts:

* [v617 frontier](../bilinear_quotient/circuits/followups/joint_atom_pruning_v617_result.json)
* [v618 spectrum](../bilinear_quotient/circuits/followups/joint_atom_spectrum_v618_result.json)

Next action: change the feature dictionary or output-sharing structure rather
than tune native pruning. Evaluate signed shared block terms / bilinear DAGs
against the same joint objective, using the
[polynomial-quotient formulation](HIERARCHICAL_TUCKER_SHARED_DAG_DIRECTION_2026-09-20.md)
for deeper compositions. Behavioral tests must retain actual denominators and
attention operations, then separately establish OOD prediction, extraction,
selective removal/restoration, and reuse. Those requirements remain open.
