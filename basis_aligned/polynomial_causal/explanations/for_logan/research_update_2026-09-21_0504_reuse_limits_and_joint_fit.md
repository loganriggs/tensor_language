# The exact product dictionary does not yet generalize to other readers

21 September 2026,05:04 UTC.

**The exact two-reader simplification remains valid, but its shared products are not a sufficiently accurate dictionary for four additional native readers.** A5% combined-output error initially looked promising; checking each component exposed44% error in one added mode. Small private correction branches helped but did not meet the registered fidelity targets.

This motivates fitting the composed computation jointly. We now have an exact, differentiable coefficient loss for its quartic numerator, checked against explicit tensors. A40-fit toy comparison recovers all five planted structures with suitable rates/restarts. This is preparation for a trained-weight joint fit, not a claim that that fit has already succeeded.

## What did “reuse” mean in this test?

The previous exact program has1,152 shared products derived from two original source quadratic forms. Keep those products fixed and let other readers use new linear combinations of them:

$$
p_k(z)=(\ell_k^\top z)(r_k^\top z),\qquad
\widehat q_j(z)=\sum_k c_{jk}p_k(z).
$$

The extra readers come from the second and third native modes of the same original folded output projection. They are additional computational components, not newly identified semantic concepts.

We solved for the best coefficients $c_{jk}$ under isotropic and covariance-weighted coefficient norms. This is a linear least-squares problem, so initialization and optimizer choice cannot explain the resulting fixed-dictionary limit. No numerical directions were discarded; normal-equation residuals were below $4\times10^{-13}$. The original two readers replayed to roughly $10^{-12}$ coefficient error.

For native-input evaluation, we retained each source form's exact centered constant and linear terms and approximated its quadratic remainder.

| Added source reader | Isotropic coefficient error | Covariance coefficient error | Native source-variation error, covariance fit |
|---|---:|---:|---:|
| Mode2, first reader | 92.41% | 52.22% | 20.85% |
| Mode2, second reader | 92.19% | 38.40% | 16.46% |
| Mode3, first reader | 89.90% | 29.68% | 22.78% |
| Mode3, second reader | 85.46% | 28.09% | 23.61% |

The registered reuse target required each covariance coefficient error to be at most25% and each native source-variation error at most10%. It fails. Covariance helps substantially, but does not make these products a universal source dictionary.

## Why the combined-output score was misleading

Each mode combines two source reads into a normalized scalar feature. The original leading mode remains essentially exact. With the additional fitted readers:

| Quantity | Relative variation error |
|---|---:|
| Leading mode | Approximately zero |
| Mode2 | **14.23%** |
| Mode3 | **43.96%** |
| Sum of all three modes | **4.98%** |

The combined score does not certify the accuracy of its smaller constituents. In particular, it cannot justify selectively manipulating mode3. These are calibration-state scalar errors; no new native or fresh-data confirmation is claimed.

## Do small private branches repair the reuse gap?

We added low-rank quadratic corrections to the four new source forms. The comparison was against fitting private quadratic forms directly, with the same number of added directions. Both preserve the original leading mode exactly.

At16 private directions per new reader:

| Method | Mode2 error | Mode3 error | Combined error |
|---|---:|---:|---:|
| Direct private forms | 6.72% | 37.71% | 3.29% |
| Shared dictionary + private residual forms | 6.83% | **28.01%** | **2.75%** |

The residual version misses the registered15% per-mode tolerance and10% per-source tolerance. It also does not beat the direct version for every reader. Its reuse connections cost4,608 additional coefficients, so the comparison does not treat those connections as free.

This is useful partial sharing, but the strong claim—that the original dictionary supplies accurate reusable components across these readers—is not established. We did not launch a native promotion run for this failed screen.

## A joint objective for the composed quartic numerator

The separately fitted source reads enter

$$
\phi(z,t,s)=\frac{(t-\tfrac12q_a(z)-\alpha s)(q_b(z)-\beta s)}{s^2},
$$

where $t=a^\top h$ and $s$ is the explicit native RMS denominator. Fitting $q_a$ and $q_b$ independently need not optimize their composed effect.

Introduce a constant coordinate $u=1$ and augmented input $x=(z,t,s,u)$. The numerator can be written as a product of two quadratic forms:

$$
N(x)=(x^\top A x)(x^\top B x).
$$

For symmetric matrices $A,B,C,D$, the inner product of the fully input-symmetrized quartic coefficient tensors is

$$
\boxed{
\mathcal I(A,B;C,D)=\frac{
\langle A,C\rangle_F\langle B,D\rangle_F
+\langle A,D\rangle_F\langle B,C\rangle_F
+4\operatorname{tr}(ACBD)}{6}.
}
$$

Thus the exact coefficient squared error is

$$
\|T_{AB}-T_{CD}\|_F^2
=\mathcal I(A,B;A,B)+\mathcal I(C,D;C,D)
-2\mathcal I(A,B;C,D).
$$

This avoids materializing the order-four coefficient tensor. Dense matrix evaluation costs cubic time and quadratic workspace in the input dimension; structured factors can offer further savings.

We independently checked values and gradients against tensors symmetrized over all24 input permutations, with discrepancies below $5\times10^{-16}$. The augmented numerator also replays its direct formula. **This is coefficient Frobenius error, not Gaussian output MSE or native behavioral error.** The constant coordinate and RMS constraints must be restored at execution. Input scaling and any data-informed metric still need explicit treatment.

## Toy optimizer comparison

Five known rank-two structures were used: independent positive forms, signed forms, shared subspaces, opposite forms with cancellation, and forms with separated scales. We tested Adam and Muon at learning rates0.01 and0.05, with two random starts and400 steps per fit:40 fits total.

| Optimizer and rate | Fits below $10^{-3}$ relative coefficient error |
|---|---:|
| Adam,0.01 | 0/10 |
| Adam,0.05 | **9/10** |
| Muon,0.01 | 0/10 |
| Muon,0.05 | 5/10 |

Every planted structure has a recovering fit. Independent explicit-tensor evaluation of the best saved fits gives errors from $1.6\times10^{-8}$ to $4.1\times10^{-5}$. This audit avoids reporting numerical zeros from subtracting nearly equal inner products.

Rate and initialization matter strongly in this finite-step setting. Adam0.05 is a reasonable first configuration for this objective, with restarts. The sweep does not establish a universal optimizer ranking, and a failed400-step fit is not proof of an unavoidable local minimum.

## What changes next?

The next comparison should fit the composed numerator directly and protect the accuracy of each intended component, rather than accept a low error dominated by the leading mode. It must retain isotropic/data-informed metric distinctions and validate native removals or swaps after fitting. The primitive and toy optimization now work; no trained-model improvement from this new objective is claimed yet.

The exact two-reader baseline, native input dependencies, and earlier donor-family failures remain unchanged.

## Evidence

- [Fixed-dictionary reuse and per-mode audit](../../direct_tensor_match/MIDPOINT_ORIGINAL_PRODUCT_REUSE_V1.json).
- [Private residual versus direct controls](../../direct_tensor_match/MIDPOINT_PRODUCT_REUSE_RESIDUAL_SCREEN_V1.json).
- [Quartic contraction implementation](../../direct_tensor_match/quartic_pair_metric.py) and [dense/gradient checks](../../direct_tensor_match/QUARTIC_PAIR_METRIC_CHECK_V1.json).
- [Forty toy fits](../../direct_tensor_match/QUARTIC_NUMERATOR_TOY_FITS_V1.json) and [independent dense recovery audit](../../direct_tensor_match/QUARTIC_NUMERATOR_TOY_AUDIT_V1.json).

All work in this update was CPU analysis. Native-source probes use the original32 calibration documents with64-token contexts; no new text or held-out outcomes entered these fits. The same source matrices were evaluated under both coefficient metrics. The40 toy fits took approximately9seconds. The broader circuit goal remains open.
