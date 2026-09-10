# Context dependence survives a change of units; extraction remains open

Two follow-ups clarify the previous calibration result. First, using a common
frequency table does **not** fix the instability between separately fitted
directions. That repair is closed. Second, the scalar's useful variation is
**not mostly an artifact of swapping values between differently scaled texts**.
After converting donor values into the recipient's units, most prediction damage
remains. This strengthens the conditional computation result while leaving its
semantic interpretation and independent extraction unresolved.

The work followed the original bilinear handoff and pilot. We ran one controlled
frequency-reference test, derived and executed a counterexample about scalar
units, tested the resulting control on the trained model, and then checked
uncertainty and the exact upstream dependencies on CPU. No direction was selected
after seeing the results, and no stability threshold was relaxed.

## The frequency-reference repair fails

The previous two fits used separate24-row samples and separate token-frequency
tables. We repeated them with the original48-row frequency table shared by both.
Everything else stayed fixed, including the requirement that each fitted axis
reproduce the original removal effect within20% relative error across the entire
vocabulary on both evaluation corpora.

| Second fit's effect error | FineWeb | Pile |
|---|---:|---:|
| Separate frequency references | 0.340 | 0.308 |
| Shared frequency reference | 0.282 | 0.252 |

The shared reference changes the result, but both errors remain above0.20. Its
first fit has errors0.197/0.179, and mutual axis cosine is0.949. Similar directions
still do not imply interchangeable causal effects. The original-axis, raw-mean
and raw-donor control losses replay exactly, so the mismatch is not execution
drift. This closes reference mismatch as a sufficient repair, not every possible
cause of the instability. The failed result remains in the circuit record.

## Why donor values need explicit units

The original weight fold computes q(u)=u^T Q u+beta, a scalar projection of the
last MLP's output. Let w be its fixed output direction, h the final residual
vector and g=h-qw the retained background. The native final state is g+qw.

Define the background's root mean square magnitude and a dimensionless scalar:

\[
R=\sqrt{\operatorname{mean}(g^2)},\qquad \tau=q/R.
\]

We require finite R>0 rather than silently clipping a zero denominator. If donor
and recipient have the same tau but the donor background is eight times larger,
its raw q is also eight times larger. A raw swap then changes the recipient even
though the two dimensionless coefficients agree. This counterexample was executed:
raw interchange changes the logit vector substantially, while the adjusted edit
agrees with the recipient to8.9e-16 maximum absolute error.

The scale-adjusted donor intervention is

\[
q_{\rm new}=R_{\rm recipient}\frac{q_{\rm donor}}{R_{\rm donor}}.
\]

It keeps the recipient background and changes only the coefficient in the
recipient's units. For unembedding U and the native epsilon, the exact readout is

\[
z(\tau)=30\tanh\!\left(
\frac{U(g/R+\tau w)}
{30\sqrt{\operatorname{mean}((g/R+\tau w)^2)+\epsilon/R^2}}
\right).
\]

The epsilon also changes units. Leaving it unchanged gives a wrong answer for
small residuals; our control explicitly detects that error. This is a different,
precisely specified intervention from the raw swap. The raw result remains valid
for its own intervention and cannot automatically establish scale-independent
semantic transfer.

## Trained-model test: most donor damage remains

We kept the original w, Q, rows and cyclic donor rule: each row receives the next
row's value at the same token position. No evaluation labels enter the donor
computation. We also replaced tau with its mean over the48 fitting rows,18.371.
The table reports increases in mean next-token cross-entropy, in nats. Positive
means worse prediction.

| Intervention | FineWeb | Pile |
|---|---:|---:|
| Original raw mean q | 0.182 | 0.201 |
| Mean dimensionless tau | 0.110 | 0.171 |
| Original raw donor q | 0.287 | 0.395 |
| Donor tau in recipient units | 0.226 | 0.333 |

Adjusted donor damage is **78.7% / 84.4%** of raw donor damage. The hypothesis
that this scale factor explains at least90% of raw damage fails. Both adjusted
edits exceed the registered0.005-nat context-pairing threshold on both corpora,
and each hurts average prediction in all42 FineWeb and all16 Pile rows.

Post-result4,000-draw row-bootstrap intervals for mean-tau replacement are
[0.098,0.122] nats on FineWeb and [0.129,0.215] on Pile. We give no ordinary
independent-row confidence interval for cyclic donor effects because donor pairs
share rows. FineWeb document grouping is unknown; Pile has one row per sampled
document. These are reused evaluation texts and a corpus shift, not pristine new
discovery data or proof of absence from model training.

The original control losses replay exactly. All backgrounds have finite positive
R: approximately599–6161 on FineWeb and611–6010 on Pile. Direct online adjusted
donor edits pass the registered1e-3 absolute /1e-5 relative logit tolerance on
the first four rows of each corpus. All58 rows receive the full offline scoring.
No semantic variable such as confidence or token frequency is identified by this
test alone; other forms of context dependence remain possible.

## The normalized coordinate does not give us extraction for free

The input to the final MLP is the normalized pre-MLP residual a:

\[
\rho=\sqrt{\operatorname{mean}(a^2)+\epsilon},\quad u=a/\rho.
\]

Let P_w=ww^T/(w^Tw) be the projection onto w. Then the retained background and
dimensionless producer are explicitly

\[
g=\rho u+(I-P_w)M(u),\qquad
\tau(u,\rho)=\frac{u^TQ u+\beta}
{\sqrt{\operatorname{mean}([\rho u+(I-P_w)M(u)]^2)}}.
\]

The numerator is our quadratic fold. The denominator still needs the rest of
the MLP output and the upstream residual. As a formal polynomial in (u,rho),
g has degree at most2 and its squared norm degree at most4; valid native inputs
also obey the normalization relation above. We keep the factored computation,
rather than expanding an enormous four-index coefficient tensor.

A CPU control verifies this local producer/readout identity to1.8e-15 logit error.
Omitting the MLP complement from the denominator gives a large error in the same
control. Thus naming tau as one scalar cannot hide the cost of its dependencies.
All545,902,902 native parameters remain. This is conditional manipulation evidence,
not an independently extracted token-to-output circuit or structural cost saving.

The two GPU experiments used32 and34 model-body forwards, with8.69 and11.25
seconds of executor time. The useful next problem is to explain the producer's
input dependencies and identify a stable computation. Further fitting-direction
refinements are closed; neither passing donor test resolves that identification
problem.

Evidence: [shared-reference result](../../CALIBRATION_COMMON_REFERENCE_V1_RESULT.json),
[scale-adjusted preregistration](../../CALIBRATION_DIMENSIONLESS_DONOR_V1_PREREGISTRATION.md),
[native result](../../CALIBRATION_DIMENSIONLESS_DONOR_V1_RESULT.json),
[CPU counterexample](../../CALIBRATION_DIMENSIONLESS_DONOR_V1_CPU_RESULT.json),
[uncertainty and producer audit](../../CALIBRATION_DIMENSIONLESS_AUDIT_V1_RESULT.json).
