# Three-term compression of a composed rational interaction

13 September 2026. Weight-derived candidate; native preservation test registered.

The exact five-vector representation remains useful, but exact minimality and useful approximation are different questions. Here we keep the normalization denominator exact and approximate only a polynomial numerator. This uses the actual upstream producer constraint, rather than fitting an arbitrary bilinear input tensor.

## Derivation and dimensions

For a fixed pristine context, write the generated residual response as

$$
r(a)=-a v_0-\frac{a}{\rho(a)}v_1+\frac{a^2}{2\rho(a)}v_2,
\qquad \rho(a)=\rho_0-2\beta a+\gamma a^2.
$$

The three vectors have width 1152 and already include the learned block10 re-entry coefficient. The scalars come from the pre-MLP9 state and fixed intervention writer. See [the response basis](RESPONSE_PRODUCT_BASIS_V2_MATH.md). Define

$$
A_0=-\rho_0v_0-v_1,\qquad
A_1=2\beta v_0+v_2/2,\qquad A_2=-\gamma v_0.
$$

Then

$$
r(a)=\frac{a}{\rho(a)}(A_0+aA_1+a^2A_2).
$$

For the symmetric downstream bilinear map

$$
K(x,y)=D_{10}[(L_{10}x)\odot(R_{10}y)+(L_{10}y)\odot(R_{10}x)],
$$

the exact self interaction is

$$
K(r(a),r(a))=\frac{a^2}{\rho(a)^2}\sum_{k=0}^{4}a^kN_k,
$$

where

$$
\begin{aligned}
N_0&=K(A_0,A_0),\\
N_1&=2K(A_0,A_1),\\
N_2&=K(A_1,A_1)+2K(A_0,A_2),\\
N_3&=2K(A_1,A_2),\\
N_4&=K(A_2,A_2).
\end{aligned}
$$

Each coefficient is a 1152-vector. The candidate keeps only $N_0,N_1,N_2$. This is **not** simply deleting two original conic-bank nodes: the coefficients mix those nodes, and $N_2$ retains one cross term involving $A_2$.

We use dimensionless strength $t=a/s$, with $s=\sqrt{\rho_0/\gamma}$, and coefficients $\widetilde N_k=s^kN_k$. Thus $|t|$ measures the writer displacement relative to the pristine state's RMS scale, including epsilon. It does not certify that synthetic contexts are in distribution.

For degree $m$, an explicit pointwise error bound is

$$
\|K-\widehat K_m\|_2
\leq \frac{a^2}{\rho(a)^2}
\sum_{k=m+1}^{4}|t|^k\|\widetilde N_k\|_2.
$$

This is a local output bound, not a guarantee on nonlinear suffix behavior. It is exact in real arithmetic; floating-point evaluation needs a rounding allowance.

Five independent coefficient vectors imply that any exact representation by amplitude-dependent scalar coefficients and fixed output vectors needs at least five vectors over an open strength interval. This lower bound concerns that representation class at a fixed context, not general arithmetic-circuit complexity. Independent polynomial coefficients span the same output space as function evaluations away from zero.

## Executed weights-only check

[The CPU control](CONIC_AMPLITUDE_POLYNOMIAL_V1_CONTROL.json) uses actual model weights and 32 seeded Gaussian contexts, with 65 strengths per declared interval. The exact numerator identity agrees within $2.09\times10^{-16}$ relative error. Equilibrated coefficient matrices have smallest/largest singular-value ratios 0.176–0.344: these examples support five-dimensional exact output span, rather than another exact four-vector identity.

Nevertheless, retaining three coefficients gives maximum aggregate relative error across contexts of approximately:

| Strength interval | Three-term error |
|---|---:|
| $|t|\leq0.01$ | $1.85\times10^{-8}$ |
| $|t|\leq0.1$ | $1.76\times10^{-5}$ |
| $|t|\leq0.3$ | 0.0317% |
| $|t|\leq1$ | 0.194% |

These are interval-grid aggregate errors, not maximum pointwise relative guarantees. The triangle bound supplies pointwise absolute information. A rank-matched SVD on the same grid is a diagnostic lower bound and can fit better; its fitted directions and grid-specific coefficient functions are not automatically a cheaper executor.

The initial fixed absolute floating-point bound assertion failed by $6.35\times10^{-6}$ on large synthetic outputs. The preserved receipt reports this and the normalized excess, $4.55\times10^{-16}$. The mathematical bound was unchanged; the implementation check now includes scale. This correction does not alter approximation errors.

## Execution price and native test

The direct compiler forms three hidden-width coefficient arrays and performs three Down writes. It never prepares the full five-bank. Compared with the cached exact implementation, this reduces varying output vectors from four to three, but also changes which global terms are represented. Full context, reader projections, background and suffix remain charged. Runtime improvement has not been measured.

The full branch retains background/background and background/response products exactly and substitutes half the approximate self interaction before dividing by the actual MLP10 RMS denominator. The managed test uses the same 160 historical prefixes and original two-edit interaction, with unchanged native-fidelity bars. This specifically checks whether small local approximation errors survive subtraction and nonlinear propagation; no fresh/OOD or semantic-selectivity claim is made.

## Native result and cost countercheck

[The managed test](POLYNOMIAL_MLP10_NATIVE_V1_RESULT.json) completed in 10.81 seconds with all three registered predicates true and exact native-reference replay. Regional original-interaction target errors are 0.051–0.081%; controls 0.155–0.217%. FineWeb target errors are 7.799%, 0.381%, 3.574%, 0.640%, with controls 7.887%, 1.141%, 1.801%, 1.355%. The largest changed endpoint difference from the exact five-bank predictor is 0.0000658.

[Incremental compression error](POLYNOMIAL_MLP10_NATIVE_V1_AUDIT.json), measured against that five-bank predictor rather than the original model, is 0.050–0.075% for regional targets and 0.497–7.267% for FineWeb targets. There are five sign reversals against the native interaction, all below the declared 0.00001 material-effect threshold. No material sign reversals occur. This preserves the distinction between total predictor error and error added by compression.

[The CPU cost countercheck](POLYNOMIAL_MLP10_COST_V1_RESULT.json) does **not** establish adoption as a faster full branch. The candidate still executes the same three large dense maps per input. Including preparation, it costs 9.60/10.14/15.15/30.72 ms at batches 1/3/16/64, versus direct evaluation at 3.48/5.03/6.94/19.11 ms. Reused-bank execution also fails the registered 10% speedup bar. The comparison includes the earlier uncached exact-five branch; it is not a benchmark against the best globally cached exact-five preparation.

Stored output vectors fall from five per context to three; compared with the improved exact representation's four varying vectors plus one globally shared vector, the asymptotic varying-bank saving is 25%. Counting stored left/right basis projections, the current three-term full-branch wrapper holds 31,105 extra scalars versus the earlier uncached wrapper's 33,408, only a 6.9% reduction. All dense weights and other producer/background state remain outside that small difference and must still be charged.

The timing panel also exposes 0.87–1.36% full-branch approximation errors on its synthetic backgrounds and amplitudes. Those examples have a different cancellation structure from the native panel. Native success therefore must not be promoted to arbitrary-background preservation. The local numerator bound does not control relative error after background cancellation or nonlinear suffix amplification.

The result is a smaller, weight-derived conditional interaction representation with historical native preservation. It is not a faster whole-model replacement, independent circuit extraction, selective manipulation, or fresh/OOD validation. The next useful structural target is the expensive background/response computation, rather than repeatedly shaving the now-small self-product bank.

## Fresh-template prediction and own-term preservation, 13 September10:30

[The fresh test](THREE_TERM_FRESH_TEMPLATE_V1_RESULT.json) evaluates48new prompts across four templates and the same six lexical spelling contrasts. Their token sequences are disjoint from1037sequences in48checked row inventories. This is a constructed template shift, not unseen vocabulary or corpus-wide OOD. No fitting or rank selection uses the new examples.

The upstream child/parent fields are generated from live native states by the assembled executor, rather than loaded from cached fresh examples. Historical anchors at rows0/24/48/72 reproduce child fields and postMLP9 states exactly; parent-field error is at most $4.84\times10^{-17}$. The prefix and suffix remain native dependencies, so this does not establish an autonomous small language model.

All four registered predicates pass in5.18seconds. Compressed original-interaction target errors are0.092–0.191%; controls0.417–0.758%. Incremental error versus the exact five-bank program is0.051–0.114%target and0.234–0.601%control. All four families have positive native paired cue capability, and no material interaction sign reversals occur. The preserved quantity is the original two-edit interaction, not the entire regional task effect.

A further [native self-term removal](THREE_TERM_FRESH_SELF_REMOVAL_V1_RESULT.json) guards against surrounding computation hiding approximation error. In every changed branch, remove the entire $K(r,r)/(2\rho_{10})$ contribution, keeping background and mixed terms. Native/exact references replay identically. The removal changes target interaction norms by10.26–15.81%, and controls by12.34–29.31%. Thus the approximated component is behaviorally material on this panel.

Define its removal-based contribution as $E=I_{\mathrm{exact}}-I_{\mathrm{removed}}$, and its reconstructed contribution as $\widehat E=I_{\mathrm{three}}-I_{\mathrm{removed}}$. Then the stricter error is

$$
\frac{\|\widehat E-E\|_2}{\|E\|_2}
=\frac{\|I_{\mathrm{three}}-I_{\mathrm{exact}}\|_2}
{\|I_{\mathrm{exact}}-I_{\mathrm{removed}}\|_2}.
$$

It is0.413–1.050%for targets and1.094–2.968%for controls, passing10%in every family. This uses actual nonlinear suffix evaluations; no additivity of suffix effects is assumed. The removal receipt reuses the fresh panel and is a countercheck, not a second independent transfer panel. Its inherited third-arm field names refer to the removed-self arm; `own_groups` explicitly records the compression comparison with the original fresh receipt.

[Leaving out each lexical pair in turn](THREE_TERM_FRESH_SELF_PAIR_V1_AUDIT.json) raises the worst error to2.83%target and4.75%control, still below10%. There are no own-component sign reversals, including below the material threshold. These descriptive checks address concentration in one concept; they are not statistical guarantees for arbitrary text.

This is the strongest current evidence for the three-term representation: fewer conditional output vectors, prediction on new templates with live-generated fields, and preservation of a material component's own effect. Selective semantic manipulation, independent extraction of the full input/background generator, general composition with other replacements, and whole-program runtime/storage gains remain unproven.

## Reuse across signed intervention strengths, 13 September10:35

[The full strength-grid test](THREE_TERM_STRENGTH_GRID_V1_RESULT.json) reuses the48template prompts with independently varied child and remainder multipliers in $\{-1,0,1,2\}$. Negative strength adds the declared writer contribution; strength2removes twice its original amount. These are intervention-domain changes, not an additional fresh-text panel.

One prepared context and one coefficient bank serve every strength pair. For each native/exact/compressed variant, the16joint branch outputs are evaluated once and reused to form16four-corner interactions. This requires2304suffix evaluations rather than separately recomputing all four corners of every query. The same caching opportunity applies to the exact baseline: this is computational reuse, not a speedup attributable solely to compression or semantic reuse across different tasks.

All registered predicates pass in14.31seconds. Across four families and nine nonzero strength pairs each, compressed target errors are0.033–0.191%; controls0.079–1.413%. Incremental three-term versus five-bank error is at most0.150%target and0.792%control. The unit-strength reference replays exactly, and there are no material target/control sign reversals. Zero-strength interactions cancel exactly through shared-state lookup; this is an explicit consistency construction rather than an independent numerical discovery.

A [quadratic-scaling countercheck](THREE_TERM_STRENGTH_GRID_V1_QUADRATIC_AUDIT.json) asks whether the test is too easy. Supply the native unit-strength interaction and predict

$$
I(\alpha,\beta)\approx\alpha\beta I(1,1).
$$

This comparator has access to the native reference at unit strength and is not a weights-only extracted predictor. Excluding the trivial unit pair, its median relative errors are17.77%target and15.60%control, with maxima40.74%and170.15%. It fails the2%target criterion in26of32family/strength cells and the5%control criterion in25of32. Thus accurate prediction over this grid is not explained by merely scaling one cached bilinear effect. The composed response, attention and normalization dependencies retained by the program matter collectively; this comparison does not separately attribute the improvement to each of them.

The supported claim is prediction by a shared conditional program across signed intervention strengths. Arbitrary composition with other circuit replacements, semantic selectivity, corpus-wide OOD, autonomous prefix extraction and whole-program efficiency remain open.
