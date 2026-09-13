# Folding the four MLP8 readings through MLP7

13 September2026,00:05UTC. **The next backward fold is exact, and its mixed terms are essential on the native city-value contrasts.** This is a computational parent decomposition, not yet a physical identification of those parents. It follows the [causally tested city/later-source split](MLP8_VALUE_FRESH_V1_MATH.md).

## Objects and interfaces

Let $U\in\mathbb R^{1152\times4}$ hold the four frozen MLP8 eigenreaders and $\Lambda\in\mathbb R^{4\times4}$ their signed eigenvalues. MLP8's selected value numerator uses $U^Tz_8$ before its RMS normalization. The previous layer computes

$$
x_7=\operatorname{RMS}(z_7),\qquad
m_7=D_7[(L_7x_7)\odot(R_7x_7)]+b_7,
$$

where $z_7$ is the residual after attention7 and before MLP7. Native block8 then gives

$$
z_8=\ell_{8,0}(z_7+m_7)+\ell_{8,1}x_0+a_8.
$$

Here $a_8$ is the full native attention8 write, including every head; the scalar mixing coefficients are learned and retained. Define four-dimensional parent readings

$$
B=\ell_{8,0}U^Tz_7+\ell_{8,1}U^Tx_0+\ell_{8,0}U^Tb_7,
$$

$$
Q=\ell_{8,0}C_7[(L_7x_7)\odot(R_7x_7)],\qquad C_7=U^TD_7,
\qquad H=U^Ta_8.
$$

Thus $U^Tz_8=B+Q+H$. The folded coefficient matrix $C_7$ has shape $4\times4608$. Equivalently each coordinate of $Q$ is a quadratic form with symmetric matrix

$$
M_{7,i}=\operatorname{sym}(L_7^T\operatorname{diag}(C_{7,i,:})R_7).
$$

The matrices have shape $4\times1152\times1152$. We verified that representation but stored the smaller coefficient interface instead. The saved program has23,050 scalars; **10,616,832 native L7/R7 weights plus all contextual input generation remain external and charged**. This is not an autonomous23k-parameter circuit.

## Six interaction paths

Let $r_8=\|z_8\|^2/1152+\epsilon$ be the squared native RMS denominator. Then

$$
\phi_4=\frac{(B+Q+H)^T\Lambda(B+Q+H)}{r_8}
$$

has six terms:

$$
\phi_4=\frac{B^T\Lambda B+Q^T\Lambda Q+H^T\Lambda H
+2B^T\Lambda Q+2B^T\Lambda H+2Q^T\Lambda H}{r_8}.
$$

$Q$ is quadratic in the normalized MLP7 input, so its square is quartic in that input. The whole native expression is not a global polynomial in raw text states: RMS denominators and the attention computation remain. $B,Q,H$ are coupled native parents, not independent random variables. A future parent-edge intervention must specify which recipient values and normalizers remain fixed.

The CPU coefficient identity agrees within $3.86\times10^{-15}$; the six-term sum within $2.68\times10^{-16}$. [Weight-fold control](MLP7_PHI_READERS_FOLD_V1_CONTROL.json).

## Native evidence

A managed96-forward cache reuses the24 original city and72 fresh prefixes. Native endpoint replay is exact; folded Q versus the native bias-free MLP7 projection differs at most $1.36\times10^{-6}$ relative; raw-reader reconstruction differs at most $4.42\times10^{-7}$, and phi reconstruction at most $1.02\times10^{-6}$. These small errors include native FP32 rounding. Six-path accounting is accurate to $5.72\times10^{-16}$. Execution took2.34seconds.

Omitting all three mixed terms causes relative error of58.03%,68.51%,65.23%,70.72% in the native city-position paired phi contrast (original/fronted/near-quote/distant groups). All preregistered A/B/C criteria pass. Thus separately explaining each parent's square would miss much of the observed computation. These are computational contrast errors, not fractions of final behavior lost under removal. [Native receipt](MLP7_PHI_PARENTS_V1_RESULT.json).

Using the previously verified routing and normalization cache, six-path donor differences reconstruct the full donor write field within $1.74\times10^{-6}$. Selected aligned fractions are:

| Path | Original city source | Fronted city source | Near-quote city source | Distant city source |
|---|---:|---:|---:|---:|
| $B^T\Lambda B$ |33.67%|28.36%|33.83%|35.56%|
| $2B^T\Lambda H$ |43.67%|33.87%|32.41%|38.10%|
| All Q-containing terms |15.17%|35.77%|36.93%|31.17%|

In the **near-quote later-source** write, Q-containing terms jointly have116.31% aligned fraction, opposed by the Q-free remainder. Individually, Q-squared contributes−122.41%, B–Q contributes156.18%, and Q–H contributes82.55%. This is a cancellation structure; the large signed fractions are not probabilities or independently established causal shares. [Full six-path allocation](MLP7_PHI_PARENT_PATHS_V1_RESULT.json).

## Next causal distinction

A principled two-parent grouping is

$$
\phi_{\text{Q-free}}=\frac{(B+H)^T\Lambda(B+H)}{r_8},\qquad
\phi_{\text{Q-containing}}=\frac{Q^T\Lambda Q+2Q^T\Lambda(B+H)}{r_8}.
$$

It keeps all mixed terms involving MLP7 together instead of treating the MLP7 square alone as its computation. Physical donation/removal of these generated value-path contributions can test whether that grouping explains the stable city effect and the later reversal, and whether the two contributions predictably compose. This test is not yet executed. The exact algebra and native allocations justify it; they do not substitute for it. No factors were fitted to this panel, and the earlier failed broad city-circuit claim remains failed.
