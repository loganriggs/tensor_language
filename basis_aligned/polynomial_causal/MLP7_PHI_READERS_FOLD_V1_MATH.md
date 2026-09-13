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


## 00:10 — Physical path-group donation and composition pass

The two-group test above is now executed on96 reused prefixes. It donates each **generated composite value** from the native donor, retaining recipient head9 routing and normalization. Thus a Q-containing donation changes the generated expression involving donor Q, B, H and RMS8 together; it is not a Q-node-only transplant. Native/full-source anchors replay exactly and the grouped write fields sum to the full source field within $2.87\times10^{-7}$ relative error.672 full forwards took9.59seconds.

| Group | City Q-containing | City Q-free | Later Q-containing | Later Q-free |
|---|---:|---:|---:|---:|
| Original |0.48%|2.60%|0.51%|8.45%|
| Fronted |0.99%|1.78%|−2.07%|8.31%|
| Near quote |1.70%|2.96%|−10.75%|2.15%|
| Distant |0.45%|1.00%|2.82%|6.09%|

Percentages are directed donor logit effects divided by mean native paired contrast. Both city groups have24/24 correct directions in each construction. Near-quote later Q-containing donation has24/24 opposing directions; its effect differs from full later-source donation by41.11% relative error, within the registered50% criterion. The Q-free near-quote later group has only16/24 correct directions despite a positive mean, so it is not a uniformly positive substitute.

The sum of separately measured Q-containing and Q-free logit effects predicts full-source donation within0.41–0.69% for city sources and0.89–2.53% for later sources. All A/B/C criteria pass. This supports local physical composition of the declared value-path groups, with native contextual inputs and suffix supplied. It doesnot certify unseen-context reuse or whole-MLP7 causality. [Native path-group receipt](MLP7_PHI_PATH_DONATION_V1_RESULT.json).

## Counter-review: does a Q-containing path carry changing information from Q?

Not necessarily. Write $S=B+H$, $R=r_8$ and

$$
F(Q,S,R)=\frac{Q^T\Lambda Q+2Q^T\Lambda S}{R}.
$$

The composite transplant changes all three arguments. We evaluated all eight recipient/donor corners and exactly allocated the full change by averaging each argument's marginal change across the other arguments' two settings (three-input Shapley allocation). The allocations sum within $2.49\times10^{-16}$; their routed fields replay the composite intervention within $7.57\times10^{-16}$.

For the reversed near-quote **later-source final-position write**, aligned fractions are Q8.16%, S92.05%, R−0.20%. Swapping Q alone with recipient S/R misses93.75% of the full composite write; swapping S alone misses18.16%. The registered20% Q-only hypothesis fails. This is a computational input-localization test, not an independently measured native one-port intervention. In particular, the causal composite result must not be summarized as “MLP7 supplies the reversal signal.” Q participates in the operation while much of the changing information enters through its partner. [Eight-corner input audit](MLP7_QPATH_INPUT_ALLOCATION_V1_RESULT.json).

With recipient Q and R fixed, the partner change splits exactly as

$$
\Delta_S F=\frac{2Q^T\Lambda\Delta B}{R}
+\frac{2Q^T\Lambda\Delta H}{R}.
$$

For near-quote later sources, the aligned allocation is40.66% to B and59.34% to attention8 H. Neither alone reproduces the partner-only write closely: B-only error59.39%, H-only40.74%. Across other constructions, attention8 supplies roughly69–76% of the later partner-only allocation. Accounting agrees within $4.10\times10^{-16}$. These are all-head attention8 and aggregate residual/reentry readings; no individual attention head is identified by this result. [Partner split](MLP7_QPATH_PARTNER_SPLIT_V1_RESULT.json).

The next causal discriminator should transplant the specified Q, B and H input readings with explicit recipient normalizers. It should test the changing input signal rather than infer it from which module's weights appear in a polynomial. Retain the successful composite path and its local composition evidence while testing that stronger mechanistic interpretation.


## 00:18 — Individual raw-input swaps confirm the partner signal

The input-port discriminator is now physically executed. For the Q-containing function F, it swaps Q, B, H or R while retaining all other recipient inputs, and applies the resulting value change only at post-city source positions of the selected head9 edge. It also tests B+H and Q+B+H together. This is an input-port intervention inside the generated path, not a whole native-module swap.768 full forwards took10.79seconds; native/full-path replay is exact and B/H write-field addition agrees within $1.68\times10^{-16}$.

Near-quote results:

| Swapped inputs | Directed transfer | Opposing directions | Relative effect error versus full path |
|---|---:|---:|---:|
| All Q/B/H/R |−10.75%|24/24|reference|
| Q alone |−0.82%|14/24|94.16%|
| B alone |−4.06%|24/24|62.43%|
| H alone |−5.98%|24/24|46.22%|
| B+H |−9.97%|24/24|17.33%|
| Q+B+H, recipient R |−10.77%|24/24|0.63%|

All preregistered A/B/C criteria pass: the partner inputs explain this physical effect substantially better than Q alone. Separately measured B/H logit effects sum to their joint effect within0.35–1.00% relative error across all four groups. That is local compositional evidence for these small interventions; exact algebra alone would not guarantee it after the nonlinear suffix.

Partner-only is not a universal approximation: its effect errors relative to the full Q-containing path are85.96%,79.64%,17.33%,71.79% across original/fronted/near-quote/distant groups. The result identifies the changing input signal in the problematic construction; it doesnot authorize dropping Q elsewhere. Likewise keeping recipient R while swapping Q+B+H has only0.63–1.22% effect error on these paired swaps, but that doesnot remove the general need to compute normalization for arbitrary inputs. [Native raw-port receipt](MLP7_QPATH_PORT_DONATION_V1_RESULT.json).

## Exact next fold: attention8 heads into the four readings

The physical H-only effect motivates decomposing H, rather than assuming a particular head supplies it. For head h, define

$$
C_h=U^TO_{8,h}\in\mathbb R^{4\times128},
\qquad A_h=(1-\mu_8)C_hV_{8,h},
\qquad A_{0,h}=\mu_8C_hV_{0,h}.
$$

Both A maps have shape $4\times1152$. Here $O_{8,h}$ is that head's physical output slice, $V_{8,h}$ its current-stream value map, $V_{0,h}$ the corresponding shared first-layer value map, and $\mu_8$ the actual learned value-mixing parameter. If $\gamma_{8,h}(j,k)$ denotes that head's full product of the two normalized, rotated QK scores, then

$$
H_j=\sum_{h=1}^{9}\sum_{k\le j}\gamma_{8,h}(j,k)
\left[A_h x_{8,k}+A_{0,h}x_{\mathrm{attn0},k}\right].
$$

$x_{8,k}$ and $x_{\mathrm{attn0},k}$ are the actual normalized attention inputs. The first-layer input is not silently replaced by a raw embedding. Folding values doesnot eliminate either QK factor or its native normalization.

Substituting this expression into $2Q_j^T\Lambda H_j/R_j$ exposes an explicit two-attention interaction path: attention8 brings information from k to j; MLP7-derived Q modulates the four readings there; head9 carries the resulting value from j to the target t. The output/value coefficients are fixed weights, while the inputs, QK routing and normalization are still contextual.

The CPU fold stores87,553 scalars and matches direct head-wise value projection and summed physical output readings within $1.47\times10^{-15}$. This is an exact linear algebra control, not a measured model speedup, a standalone87k-parameter circuit, or evidence ranking the nine heads. The next native head-reading cache must replay H and then test individual head contributions under the already specified H-input intervention. [Fold and price receipt](ATTENTION8_PHI_READER_FOLD_V1_CONTROL.json).
