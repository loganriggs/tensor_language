# Research update: an executable interaction path, and where composition breaks

**13 September 2026, 02:00 UTC.** This covers work since [the previous full update](research_update_2026-09-13_0033_interaction_paths.md), which covered results through 00:29 UTC. Updated before publication to include the native MLP9 test completed at 02:01:38 UTC and its subsequent CPU counter-review.

## 1. High-level overview

**We now have a more explicit, removable interaction path spanning several modules. We also found the next obstacle: its final behavioral effect depends on what other components remain in the model.**

Previously, we had traced an important interaction between MLP7 and attention head 8.2, read through MLP8 and head9. The new work split head8.2's input into its current-state and shared first-layer value contributions. The latter can be generated directly from token embeddings and weights. Combining that token contribution with a four-dimensional contextual reading from MLP7 gives a specific cross-module computation.

This is useful progress toward your proposed unit of analysis: a composed interaction path. It specifies which two quantities interact, how attention transports them, and where the result is written. We can remove this term without removing an entire head or needing a paired donor sentence.

The main events were:

1. **Separated token information from contextual modulation.** The shared first-layer values have an exact token-only generator. Their effect still depends on contextual MLP7 readings and attention routing.
2. **Tested new sentence constructions.** The path's opposing effect near quotations replicated, but a predicted positive effect in a distant construction failed on six prefixes. A physical interaction test explained how the contextual input changes that sign; it did not erase the failure.
3. **Removed the absolute component and checked an unrelated behavior.** Removal changes regional spelling contrasts modestly and preserves newline prediction closely on new FineWeb prefixes.
4. **Built and validated an executable implementation.** It calculates its internal readings and routing from declared weights and supplied model states, reproducing the earlier native interventions to numerical precision.
5. **Established a parent/child hierarchy.** The path is a child of a larger, previously studied head9 value component. Joint child-plus-remainder removal exactly matches parent removal.
6. **Found that separate behavioral effects do not add accurately enough.** The discrepancy comes substantially from computations inside the remaining model, rather than just its final unembedding and loss.
7. **Derived an exact forward fold through MLP9.** This avoids an optimization problem and includes changing RMS normalization. Weight-level and native local-response checks pass. The completed mediation test shows that correcting this first layer alone explains only a limited part of the final interaction.

My assessment is that this strengthens the direction as a way to describe and manipulate specific computations. It does not yet establish a general unsupervised decomposition whose factors automatically become independent circuits. The strongest advances are explicit computation, limited extraction, and a concrete hierarchy with measured interaction effects.

## 2. What the path actually computes

All layer and head numbers here follow the repository's zero-based indexing. The residual width is 1,152; each bilinear MLP has 4,608 product channels. The model's attention multiplies two QK scores rather than applying softmax. The implementation retains both QK factors and their position dependence.

The starting object is a scalar value reader downstream of MLP8. Folding it through that bilinear layer produces a symmetric interaction matrix. Four selected eigenmodes give

$$
M_8^{(4)}=U\Lambda U^T,
\qquad U\in\mathbb R^{1152\times4},\quad
\Lambda\in\mathbb R^{4\times4}.
$$

These four readings are a useful approximation boundary, not four independently identified semantic concepts. The earlier report explained how their inputs split into residual, MLP7, and attention8 contributions. Squaring their sum produces cross terms.

The component pursued here is one of those cross terms: the interaction between the MLP7 contribution and the first-layer values transported by head8.2. To make the computation readable, let:

- \(Q_j\in\mathbb R^4\) be the four readings of the MLP7-generated contribution at position \(j\), including the appropriate learned residual scaling. This capital \(Q\) is **not** an attention query matrix.
- \(F_k\in\mathbb R^4\) be four weight-derived readings of the shared first-layer value associated with token \(k\), including the value mixture and output projection relevant to this path.
- \(\Gamma_8(j,k)\) be the full signed attention coefficient for head8.2 from source \(k\) to destination \(j\).
- \(s_{8,j}\) be the squared RMS denominator before MLP8.

Then

$$
H^{\mathrm{first}}_j=\sum_{k\leq j}\Gamma_8(j,k)F_k,
\qquad
G_j=\frac{2Q_j^T\Lambda H^{\mathrm{first}}_j}{s_{8,j}}.
$$

The factor of two is the ordinary cross term in a symmetric quadratic expansion. It is not fitted to intervention data.

Head9 transports this scalar contribution and writes it along a fixed residual direction \(w\in\mathbb R^{1152}\):

$$
c_t=\alpha_9\sum_{j\leq t}
\Gamma_9(t,j)\frac{G_j}{\rho_{9,j}},
\qquad \text{child write at }t=c_tw.
$$

Here \(\rho_{9,j}\) is the RMS divisor at the head9 input, \(\alpha_9\) is the learned residual re-entry coefficient, and \(\Gamma_9\) is the selected joint head9 routing component inherited from the parent program. Head8 routing is full; head9 routing is the already selected component. This distinction matters when interpreting the scope of the extracted path.

The computation has two source indices: head8 reads token position \(k\), then head9 reads intermediate position \(j\). The **all-source** component sums both sets of causal sources. It needs neither a city-position label nor another sentence to define its removal.

The token generator is exact because the first block's pre-attention value input depends only on the token embedding and the block's initial normalization/scaling. This does **not** make the whole path token-only: \(Q\), both attention patterns, and the normalization factors remain contextual. An earlier attempt to generate the entire contextual MLP8 value from a token alone failed; that was a different, much larger object.

[Primary derivation and receipts](../../ATTENTION8_PHI_VALUE_ROUTING_V1_MATH.md).

## 3. What new contexts taught us

We froze a panel of 24 old anchors plus 72 new prefixes: two new near-quotation constructions and one distant construction. Cities and spelling endpoints were reused. These are construction holdouts, not a new corpus or new vocabulary.

The native model had the expected regional contrast on all 36 new paired comparisons. The first-value-only path opposed the donor's expected spelling direction on all 48 new near-quotation prefixes, with mean directed transfers of −3.19% and −2.40% of the native regional contrast. In the distant construction, mean transfer was +0.864%, but only 18 of 24 signs were positive. The registered requirement was at least 20. That prediction failed.

All six exceptions were Baltimore recipients. The full head8.2 path had the same signs, so dropping the current-value contribution was not the explanation for this failure.

We then asked which contextual input changes the interpretation of the fixed token-value difference. An algebraic allocation across four context inputs pointed to the MLP7 readings. Because swapping computational inputs can create unnatural combinations, we followed that diagnostic with actual model interventions.

For a bilinear interaction, changing both inputs produces an extra term:

$$
G(Q+\Delta Q,F+\Delta F)-G(Q+\Delta Q,F)
-G(Q,F+\Delta F)+G(Q,F)
\propto \Delta Q^T\Lambda\Gamma_8\Delta F.
$$

We injected the two single-input changes, their joint change, and the isolated mixed term through the specified head9 path. The isolated term predicted the measured behavioral difference-of-differences within **1.29–2.09%** across the four constructions.

Donating Sheffield's MLP7 readings to Baltimore recipients changed all six previously negative token effects to positive. The counter-review matters: donating Baltimore's readings to Sheffield flipped those six effects in the opposite direction. We transferred the contextual asymmetry; we did not improve the overall sign count. This supports a context-dependent interaction, not a universally supportive “British spelling circuit.”

[Fresh-context result](../../FIRST_TOKEN_PATH_FRESH_V1_RESULT.json) · [Physical interaction result](../../FIRST_TOKEN_Q_INTERACTION_V1_RESULT.json) · [Counter-review](../../FIRST_TOKEN_Q_INTERACTION_V1_COUNTER_REVIEW.json).

## 4. Removal and preservation of another behavior

A donor swap asks what happens when a quantity changes between two examples. Absolute removal asks whether we have defined a component of a single model execution. We now remove the all-source child write \(c_tw\) directly.

| Construction | Change in native regional contrast after removal |
|---|---:|
| Old near-quotation anchors | +4.42% |
| New near-message construction | +2.70% |
| New near-person construction | +1.52% |
| New distant-note construction | −1.33% |

Positive means regional contrast becomes stronger after removal. Thus this component inhibits the contrast in the near constructions and supports it modestly in the distant one. Every near pair increased; every distant pair decreased. Half-strength removals predicted full-strength effects within 0.20–0.32% relative error on these panels.

To test selectivity, we removed the same annotation-free component from FineWeb prefixes whose next token is a newline. **Cross-entropy (CE)** is negative log probability of that actual next token, measured in nats. Signed positive CE change means damage. Absolute CE change measures disturbance regardless of whether prediction improves or worsens.

On 32 new-to-this-path prefixes, split into two families:

| New FineWeb family | Mean absolute CE change | Maximum absolute CE change | Whole-head8.2 removal: mean absolute CE change |
|---|---:|---:|---:|
| Family 2 | 0.000278 | 0.001415 | 0.03274 |
| Family 3 | 0.000214 | 0.000686 | 0.05091 |

The small component preserves this newline behavior much more closely than removing the whole head. All registered checks for this new panel passed. These prefixes come from disjoint indices in an existing FineWeb cache; document independence and corpus OOD are not established.

There was also an avoidable earlier control-design mistake. I required whole-head removal to change CE by more than 0.1 nats despite a published fixture showing about 0.038. That registered check failed and remains failed. A shared preflight now checks proposed control thresholds against the existing fixture. The new panel used a separately registered sensitivity criterion; it does not retroactively repair the old result.

[Regional removal](../../FIRST_TOKEN_ABSOLUTE_REMOVAL_V1_RESULT.json) · [New FineWeb controls](../../FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_RESULT.json).

## 5. What “executable” now means

The new implementation receives token IDs, three normalized model-state arrays, and two normalization arrays. It computes the MLP7 readings, the token value readings, head8 routing, selected head9 routing, and the final child field itself from declared weights. Earlier versions supplied some of those computed intermediates directly.

On 96 regional and 64 FineWeb prefixes, its aggregate scalar-field errors were about \(8\times10^{-7}\) relative to the native calculations. Its physical removal outcomes matched the previous interventions within \(4\times10^{-7}\) aggregate relative error. All registered replay checks passed.

That is a useful extraction milestone: the computation has an explicit executable interface and reproduces an actual intervention. However, the upstream model must still generate the supplied states, and the remaining model must still turn the write into a final answer. Including embedding and MLP7 input weights, the implementation declares **69,970,952 tensor scalars**. It is not a tiny standalone spelling model.

[Executable package and dependency description](../../extracted_circuits/crossfirst_state_executor_v1/README.md) · [Native validation](../../CROSSFIRST_STATE_EXECUTOR_V1_RESULT.json).

## 6. A real hierarchy, with a behavioral composition limit

The child is contained in a previously studied head9 current-value component. Calling that parent \(P\), this child \(C\), and everything else in the parent \(R\), we have

$$
P=C+R.
$$

This equality is at the injected scalar-field/write interface. The full MLP8 weight expansion, including the matrix outside the four selected modes and the output bias, verifies the containment. The remainder is an exact accounting object; it has not been identified as one semantic circuit.

Physically removing \(C\) and \(R\) together gave exactly the same measured outputs as removing \(P\). The implementation also rejects selecting the parent and its child together, which would accidentally remove the child twice.

But the model after this interface is nonlinear. Let \(Y_N\) be the native outcome and \(Y_C,Y_R,Y_P\) the outcomes after those removals. Independent effect addition would predict

$$
Y_P-Y_N\approx (Y_C-Y_N)+(Y_R-Y_N).
$$

The discrepancy is

$$
I=Y_P-Y_C-Y_R+Y_N.
$$

We measured its norm relative to the **small child's effect**, so a large parent effect cannot hide the error:

| Panel group | Additive-effect error relative to child |
|---|---:|
| Regional old near | 5.29% |
| Regional near-message | 11.98% |
| Regional near-person | 9.53% |
| Regional distant | 20.24% |
| FineWeb old family 0 | 4.54% |
| FineWeb old family 1 | 20.81% |
| FineWeb new family 2 | 8.94% |
| FineWeb new family 3 | 47.43% |

Four of eight cells missed the 10% criterion. Relative to the larger parent, the errors would look like only 0.21–2.77%.

The counter-review limits the pessimistic interpretation. The child's marginal-effect sign survived all 96 regional prefixes even when the remainder was absent. Its magnitude changed more than its direction. The largest absolute CE composition discrepancy was only 0.000646 nats. We therefore retain both conclusions: the hierarchy executes exactly, and independently adding its final effects is insufficient at the requested child-relative precision.

[Hierarchy experiment](../../CROSSFIRST_HIERARCHY_V1_RESULT.json) · [Counter-review](../../CROSSFIRST_HIERARCHY_V1_COUNTER_REVIEW.json).

## 7. Why we are folding forward through MLP9

A cheap possible explanation was that all the nonadditivity arose at the final RMS normalization, unembedding, score soft-cap, or CE calculation. We tested it by caching the final residual states for native, child, remainder, and parent removals.

Let \(f\) be the actual final readout and define \(h_{\mathrm{add}}=h_C+h_R-h_N\). Then the observed nonadditivity splits exactly:

$$
I=
\underbrace{f(h_P)-f(h_{\mathrm{add}})}_{\text{internal state interaction}}
+
\underbrace{f(h_{\mathrm{add}})-f(h_C)-f(h_R)+f(h_N)}_{\text{final readout interaction}}.
$$

On the regional groups, the first term's norm was **87–101%** of the total nonadditivity norm. Norm ratios can exceed 100% when the two contributions partially cancel; they are not percentages of variance explained. The exact final-readout correction left 4.62–19.23% child-relative errors. FineWeb corrections sometimes helped and sometimes hurt. The remaining model's internal computation matters.

The first nonlinear layer after the edited head9 write is MLP9. Both child and remainder write along the same direction \(w\), with different scalar amplitudes. At each position their joint input edit is simply \(z\mapsto z-(a+b)w\). This special structure lets us derive the entire local response without fitting a general tensor decomposition.

Write its bias-free bilinear function as

$$
\mathcal B(x)=D[(Lx)\odot(Rx)],
\quad L,R\in\mathbb R^{4608\times1152},
\quad D\in\mathbb R^{1152\times4608}.
$$

Let \(m_0=\mathcal B(z/\sqrt{\rho_0})\), where

$$
\rho_a=\frac{\|z-aw\|^2}{1152}+\epsilon,
\qquad \epsilon=1.1920928955078125\times10^{-7}.
$$

Expanding the two bilinear inputs gives

$$
\mathcal B(z-aw)=\mathcal B(z)-aJ_wz+\frac{a^2}{2}J_ww,
$$

with the fixed weight map

$$
J_w=D[\operatorname{diag}(Rw)L+\operatorname{diag}(Lw)R]
\in\mathbb R^{1152\times1152}.
$$

Including the residual connection and the **changed** RMS denominator, the exact output change is

$$
\Delta(a)=-aw+
\left(\frac{\rho_0}{\rho_a}-1\right)m_0
-\frac{a}{\rho_a}J_wz+
\frac{a^2}{2\rho_a}J_ww.
$$

The output bias cancels. This is not a small-edit approximation. For a fixed context, every amplitude response lies in the span of four vectors: \(w,m_0,J_wz,J_ww\). In the two-edit interaction

$$
\Delta(a+b)-\Delta(a)-\Delta(b),
$$

the linear writer term cancels, leaving at most three contextual directions. This is an equation-specific simplification of the kind you suggested looking for. It is not a discovery of three semantic factors, nor a universal three-dimensional basis across all contexts.

Actual-weight FP64 checks match direct MLP9 evaluation to about \(10^{-15}\) relative error. We reused the earlier directional-MLP method rather than treating this as a new theorem. The compiled map and writer store 1,328,256 scalars; the native baseline state and baseline MLP output remain required.

The native experiment asked two questions: does this formula reproduce local responses on the existing text panels, and does removing its predicted mixed state eliminate at least half the final behavioral nonadditivity in every group? It used 160 existing prefixes and 800 full forwards, taking 8.86 seconds of experiment-body execution. **It completed while this report was being finalized.**

Native anchor replay was exact. Aggregate individual local-response errors were **0.000808% regional and 0.002231% FineWeb**, comfortably below the registered 1% bar. The formula therefore works on these native states, not only on random FP64 controls.

However, subtracting the predicted MLP9 mixed state left **84.6–89.7% of regional nonadditivity** and **65.1–99.9% of FineWeb nonadditivity**. All eight groups missed the requirement to leave at most 50%. Replay and local response pass; the proposed single-layer mediation explanation fails.

The CPU counter-review found that the remaining regional interaction points almost exactly in its original direction: cosine similarity 0.9986–0.9997, where 1 means the same direction across examples. The removed contribution also aligns positively, with cosine 0.910–0.989. Thus this is a coherent partial correction, rather than a large correction that merely rotates or reverses the error. No coefficients were fitted for this analysis.

The local mixed-state prediction itself has 0.128–0.212% relative error on the regional groups and 0.293–1.313% on FineWeb. Those are small, but local error alone does not bound its amplification downstream. A precision-robust mediation control remains untested. The supported conclusion is that this implemented MLP9 correction is insufficient; we have not proved an exact causal allocation of all remaining interaction to particular later layers.

[Native MLP9 result](../../MLP9_CROSSFIRST_RESPONSE_V1_RESULT.json) · [Executed outcome counter-review](../../MLP9_CROSSFIRST_RESPONSE_V1_COUNTER_REVIEW.json).

[Final-readout experiment](../../CROSSFIRST_READOUT_SPLIT_V1_RESULT.json) · [MLP9 derivation, controls, and pending test](../../MLP9_CROSSFIRST_RESPONSE_V1_MATH.md).

## 8. Efficiency, math cycles, and the four desired properties

The current gains come from using the actual equation and an already specified intervention boundary. We are not running another unconverged CP or LL1 fit in this interval. Exact bilinear expansion determines the interaction coefficients, and native text tests their consequences.

The latest completed three-hour mathematical review predates this reporting window. I therefore would not attribute all these results to a newly completed scheduled math cycle. The useful mathematical work since the last report was the effective contextual reader, the two-input mixed-term identity, exact parent containment, the readout/internal partition, and the fixed-writer response derivation. Each narrowed a concrete experiment or implementation.

For repeated edit strengths, the response evaluator now computes \(J_wz\) once per context and reuses it. On a CPU comparison with 24 signed amplitudes and 32 synthetic contexts, this reduced kernel time from 0.0290 to 0.00529 seconds, about **5.49×**, while retaining numerical equivalence. That excludes native baseline generation and the remaining model; it is not a whole-model speedup. The frozen queued runner was not modified by this optimization.

Representative completed native tests were inexpensive once running: 320 forwards for executor replay took 6.04 seconds, 800 for hierarchy testing took 9.27 seconds, and 640 plus final-readout evaluations took 8.08 seconds. These are recorded experiment-body times, not end-to-end research latency. Queue waits, loading, derivation, and implementation still matter. The managed runner was healthy: the colleague's preceding job finished at 02:01:27 and the MLP9 test finished at 02:01:38. The CPU counter-review above was performed after that result arrived.

| Desired property | Current evidence | Remaining gap |
|---|---|---|
| OOD prediction | New construction tests reveal reproducible context dependence; some frozen predictions pass | Distant sign rule failed; no broad corpus-OOD effect predictor |
| Extraction | Explicit weights-and-states executor reproduces the native intervention | Requires native contextual states, baseline computation, and downstream model |
| Selective removal | Absolute component removal works; new FineWeb newline controls are closely preserved | Modest task effect; broader unrelated behaviors untested |
| Composition and reuse | Exact child/parent containment and joint write accounting; causal two-input interaction prediction | Separate final effects are not sufficiently additive; exact MLP9 response passes but its correction is insufficient |

The new result favors tracing where the remaining interaction arises farther downstream, while checking that rounding in the local mixed-state subtraction is not materially distorting the mediation result. The next experiment should measure layerwise interaction creation and physically test a candidate site, rather than infer causality from activation norms alone. The completed CPU direction check is the first follow-up; a new downstream GPU experiment has not yet been registered.

Your normalized Frobenius/tensor-similarity proposal remains a possible method for comparing candidate arithmetic structures when a well-matched problem calls for it. No implementation or experiment with that proposal was added in this interval.
