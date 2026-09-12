# Folding the token-only first-state input into four readings

12 September2026,15:10UTC. Full-write/native effect checks pass; the stricter
FP32 child-local check fails. The token-input artifact is a tested candidate,
not a replacement for all previously validated intervention interfaces.

## Exact input fold and price

The earlier package takes source $s=(x,x_{\rm first})$ and uses its first-state
half through four linear readings only: two parent readers and two child readers.
For source token v, the native first attention input is

$$
x_0(v)=\operatorname{RMS}(E_v),\qquad
x_{\rm first}(v)=\operatorname{RMS}(\lambda_{0,0}x_0(v)+\lambda_{0,1}x_0(v)).
$$

Split the four source readers as F=[F_current,F_first], and precompute

$$
T_v=F_{\rm first}x_{\rm first}(v),\qquad
f(x,v)=F_{\rm current}x+T_v.
$$

T has50,304rows and4columns, covering every vocabulary ID. No corpus or fitted
frequencies are used. Native initialization, residual coefficients and RMS
epsilon are preserved. Runtime now needs query1152,current source1152,tokenID
and relative rotary map; it no longer needs a separate1152-dimensional first
state. The other upstream states and final readout remain external.

The package grows from666,656 to863,264scalars: remove4,608first-state reader
weights and add201,216lookup values. This is an explicit storage-for-dependency
tradeoff, not whole-model compression. [Compiler result](COMPILED_TOKEN_SHARED_HEAD2_V1_RESULT.json)
passes independent256-probe write replay8.61e-7 and child partition1.09e-7.

## Native result and preserved precision failure

[Native integration](COMPILED_TOKEN_HEAD2_NATIVE_V1_RESULT.json) passes full-write
and signed-prefix removal effects on48prompts: write<=1.12e-7,effect<=6.50e-6.
The child sum also agrees with the old FP32 joint write within8.74e-8. However,
one source-position child comparison reaches6.76e-5relative error against1e-5.
A/B pass,C fails. Its reported interaction uses the old FP32 joint arm versus
the token child arms, so it is not a pure within-token-package interaction audit.

The [diagnostic capture](TOKEN_HEAD2_CHILD_CAPTURE_V1_RESULT.json) repeats that
miss. Child1 at source position5 has reference write norm0.0166145 and error
norm1.123e-6. The old FP32 package's same child is less accurate:2.51e-4relative.
The earlier very tight child-reference result was for FP64 child programs;
it must not be transferred to arbitrary FP32 child interventions.

[Precision isolation](TOKEN_HEAD2_CHILD_PRECISION_V1_RESULT.json) finds a small
child reading,0.0002481. Original FP64 weights/arithmetic replay within1.30e-12.
Rounded FP32 weights with FP64 arithmetic still err3.35e-5; token lookup raises
that to4.16e-5. Replacing lookup readings by exact captured first-state readings
returns to3.35e-5, isolating additional lookup rounding from parameter rounding.
Original weights plus exact captured readings reproduce the split formula
within2.19e-12. Captured readings are a diagnostic, not a deployable vocabulary
table or a retroactive pass of C.

The executed [field precision test](TOKEN_HEAD2_CHILD_FIELD_PRECISION_V1_RESULT.json)
restores one parameter group at a time. Restoring only the four source readers
to FP64, with all arithmetic FP64, reduces this case to2.85e-8. Restoring dual,
normalization maps, key-atom projections or output writers individually leaves
roughly3.35e-5error. Thus the tiny feature reading is sensitive to reader-weight
rounding. Four high-precision source readers add36,864bytes to the original
package, far less than making every field FP64.

Next: validate a precision-aware reader/lookup implementation on native states
under the unchanged child criterion. This single-case diagnosis does not prove
a mixed-precision package succeeds elsewhere. Full token-input effects pass,
but precise child manipulation remains explicitly unverified for this FP32
candidate. No broader four-property completion follows.
