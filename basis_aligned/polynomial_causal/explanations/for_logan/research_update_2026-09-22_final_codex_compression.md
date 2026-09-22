# Final Codex compression report: what was compressed, what survived, and what remains open

22 September 2026. This report closes the Codex weights-first compression thread through the completed hybrid local-quartic experiment and feature-stability audit. It does not cover Claude's separate attention-compressibility work.

## Executive summary

We built and tested smaller arithmetic programs for one precisely defined computation through the final two bilinear MLPs of the 18-block `bilin18` model. The work produced real compression and several useful executable approximations, but it did **not** recover a small, stable, reusable circuit that satisfies all component, intervention, and out-of-distribution requirements.

The most compact early program reduced a selected quartic approximation from 656 to 384 variable-variable products and from 903,168 to 322,048 floating coefficients while replaying that fitted approximation exactly. That was a successful graph rewrite, not an exact reconstruction of the native model computation. On actual model-derived targets, compact candidates often achieved low pooled error while leaving smaller output coordinates 40–70% wrong. Native removal tests confirmed that those hidden component errors matter.

Increasing capacity and improving the fitting metric helped. A 1,536-product CP program with calibration-informed Gaussian fitting reduced pooled error on 16,384 opened text states to roughly 7.4%. Learning 96 additional output-local quartic atoms improved the twelve weak outputs by about 7% relative, but still left approximately 53% value error and 54% matched-response error. L-BFGS and Muon did not repair the gap. Oracle readouts using evaluation answers showed that the learned feature spans themselves were insufficient; the failure was not just in their final coefficients.

The final experiment combined exact Gaussian weight matching with native-sensitivity-weighted text fitting. It improved the twelve weak outputs further, to 46.6–46.7% value error and 48.8–49.2% matched-response error. Those are real gains, including better residual tails, but both preregistered scientific gates failed: the improvement was below 15%, and only 54.9–55.6% of the Gaussian baseline's explained residual energy was retained. An independent evaluator reproduced all reported value and response errors to numerical precision.

The final stability audit is decisive for interpretation. Two matched hybrid fits reached similar aggregate errors but did not learn the same eight-dimensional feature spaces: every one of the twelve output-specific dictionaries had at least one poorly aligned direction, and aggregate correction-function cosines ranged from 0.44 to 0.80 under the Gaussian metric and 0.70 to 0.93 on text. Similar performance therefore does not identify a reusable circuit. No feature merge or semantic naming is licensed.

The project ends with a narrower and better-supported conclusion: **the selected native computation has compressible functional approximations, but the tested low-product dictionaries do not provide stable component-level circuit recovery.** The next justified design change is to model propagated source/context geometry while retaining all output coordinates and explicit normalization. More optimizer, readout, or objective-mixture micro-sweeps within the current local dictionary family are not supported by the evidence.

## 1. What computation was studied?

The model has 18 transformer blocks, residual width 1,152, and bilinear MLP width 4,608. For a normalized input state $x$ entering MLP16, a bilinear MLP has the polynomial form

$$
B_\ell(x)=D_\ell[(L_\ell x)\odot(R_\ell x)].
$$

Let the transported, bias-free MLP16 contribution be

$$
m(x)=\lambda D_{16}[(L_{16}x)\odot(R_{16}x)].
$$

If the pre-normalization input to MLP17 is written as $b+m(x)$, its bilinear numerator contains a background term, two cross terms, and the selected pure quartic branch

$$
F_4(x)=D_{17}[(L_{17}m(x))\odot(R_{17}m(x))].
$$

Because $m(x)$ is quadratic, $F_4(x)$ is degree four in the 1,152 input coordinates. The experiments measure this branch along 16 fixed output directions. Those directions are learned writer coordinates for an approximation. They are not 16 examples, 16 native neurons, or 16 established semantic concepts.

The selected branch is only one term in the last two blocks. Attention, residual backgrounds, biases, MLP17 cross terms, normalization denominators, final RMSNorm, unembedding, and logit softcapping remain separate native computations. Successful approximation of $F_4$ does not by itself replace either block or establish a whole-model circuit.

The wider project also contains a full single-layer third-order tensor objective. This quartic branch study is a focused test of the proposed two-stage method—continuous tensor-style feature discovery followed by discrete arithmetic-graph simplification—not a substitute for the full tensor goal.

## 2. What “compression” means here

The original small candidate computed learned quadratic intermediates and then shared their products across 16 output writers:

$$
q_i(x)=\sum_{k=1}^4(u_{ik}^{\top}x)(v_{ik}^{\top}x),
\qquad
h_g(x)=\sum_{i\le j}A_{gij}q_i(x)q_j(x),
\qquad
\widehat F_4(x)=\sum_g w_g h_g(x).
$$

The graph compiler reduced this fitted program from 656 to 384 variable-variable products and from 903,168 to 322,048 stored floating coefficients, plus indices. That exact rewrite preserved the candidate's function. It did not remove the candidate's error relative to $F_4$.

Later CP programs used products of four learned linear forms. A 1,536-product parent required 2,385,920 floating coefficients. Output-local residual corrections added eight quartic atoms to each of outputs 4–15:

$$
\widehat F_g(x)=F_g^{\mathrm{parent}}(x)+
\sum_{k=1}^{8}c_{gk}\prod_{s=1}^{4}(a_{gks}^{\top}x).
$$

The correction added 288 products and 442,464 coefficients. It was a capacity and discovery test, so it made the program larger. It was never claimed as a final compressed circuit.

Throughout the study, product counts, coefficient storage, graph reuse, and approximation quality were tracked separately. An exact rewrite can compress an approximation without improving native fidelity; a larger discovery model can improve fidelity without being a useful final compression.

## 3. How the evidence developed

### 3.1 Algebraic compression worked locally

The compiler found exact reuse inside fitted candidates. The strongest early rewrite reduced 656 products to 384. Later exact reader reuse reduced linear storage by about 6.1%, and exact product reassociation saved a further 9–10 products after that reader edit. These are valid arithmetic savings with numerical replay controls.

The limitation is equally clear: graph simplification cannot recover computations absent from the fitted feature dictionary. Several compact graphs retained low aggregate error but failed individual-coordinate and finite-response tests.

### 3.2 Coefficient geometry and text behavior disagreed

Pure coefficient matching under an isotropic metric performed poorly on text. For the 1,536-product CP family, coefficient-only fits produced roughly 39–41% relative error on the 16 text-state outputs. Refitting the same dictionary under a Gaussian with empirical mean and covariance reduced that error to roughly 9.8–10.0%. Jointly moving the input directions reduced it further to about 6.3–6.6% on the earlier panel.

This large shift showed that hidden-state geometry matters. It also created a warning: a metric that predicts pooled values well can still miss specific output coordinates, matched changes, or installed model effects. A quartic squared-error objective depends on moments through degree eight; mean and covariance determine those moments only under the Gaussian assumption.

### 3.3 Pooled errors concealed component failures

The larger evaluation panel contains 16,384 token-position states from 256 documents and 16 scalar targets per state. A matched response is the difference between outputs at two states with the same current token and position but different documents. It is an observational finite-difference diagnostic, not a causal intervention. On this panel, the fixed CP parent has about 7.38% pooled relative error, but the twelve smaller outputs have 56.58% RMS value error and 58.09% matched-response error. The small pooled number is dominated by the largest output coordinates.

The errors are not confined to a handful of examples. Median document-level small-output error is about 52%, the 90th percentile about 67%, and the worst 10% of states account for about half of the squared residual. A paired whole-document bootstrap later put the Gaussian Adam correction's relative small-output improvement at only about 7.1%, with narrow conditional intervals on this opened panel.

This distinction drove the evaluation policy: all 16 coordinates, each smaller coordinate, matched-state changes, document distributions, and native interventions were reported separately. Favorable pooled scores could not replace failed component gates.

### 3.4 Native removal exposed normalization-sensitive error

Conditional full and lean programs failed their registered finite-removal criteria. For five failing lean cases, the raw quartic scalar error was below 10%, yet error exceeded 10% after the native local normalization denominator and remained high through final RMSNorm, unembedding, and softcap.

Across states, dividing prediction and reference by the same denominator does not cancel in a pooled relative metric; it reweights the states. In the FineWeb failure strata, the lowest-denominator 10% of states carried roughly 46–50% of denominator-weighted residual energy but only 28% of reference energy. This explains part of the amplification without reducing the failure to one simple denominator threshold.

The practical consequence is that raw polynomial fidelity is insufficient for installed behavior. Normalization and downstream metrics must stay explicit in any future circuit candidate.

### 3.5 New local features helped, but the feature class remained inadequate

The Gaussian residual learner added 96 quartic atoms while keeping the CP parent fixed. On the larger panel:

| candidate | small-output value error | small-output matched-response error |
|---|---:|---:|
| fixed CP parent | 56.58% | 58.09% |
| Adam, seed 25001 | 52.59% | 54.08% |
| Adam, seed 25002 | 52.54% | 54.13% |
| Muon, seed 25001 | 56.58% | 58.08% |
| Muon, seed 25002 | 56.57% | 58.07% |
| L-BFGS, seed 25001 | 53.99% | 55.47% |
| L-BFGS, seed 25002 | 54.10% | 55.65% |

The registered target required at least 15% relative improvement in both values and responses for both starts of one optimizer. Every method failed. Adam was best in this tested configuration, but still improved only about 7%.

The optimizer diagnosis was controlled rather than assumed. Scaled L-BFGS recovered 8 of 10 planted quartic toy cases below 5% error, yet did not transfer that advantage to the native target. Muon also depended strongly on schedule in toys and made negligible native progress at the frozen budget. These outcomes do not establish a universal optimizer ranking; they show that optimizer substitution did not solve this native representation problem.

Evaluation-informed oracle readouts provided a stronger capacity check. Allowing every weak output to read all 96 learned products improved the best Adam banks to about 45% value error and 47% response error, but large errors remained. Because these oracles used evaluation answers separately for values and responses, they are diagnostics rather than deployable programs. They show that readout refitting alone cannot rescue the frozen learned spans.

## 4. The final hybrid experiment

The final registered experiment changed the metric while holding the correction class, seeds, optimizer, and 250-update budget fixed. Its objective mixed:

1. exact Gaussian weight-contraction loss; and
2. calibration text error weighted by native logit sensitivity.

The sensitivity term includes local normalization effects at the native background. It is a tangent metric, not a guarantee for finite removals. Fitting used 6,144 states from 96 documents; the larger opened evaluation used 16,384 states and 2,494 same-token, same-position pairs from different documents.

| seed | Gaussian-only value error | hybrid value error | Gaussian-only response error | hybrid response error | Gaussian explained-residual retention |
|---|---:|---:|---:|---:|---:|
| 25001 | 52.59% | **46.69%** | 54.08% | **49.17%** | 54.88% |
| 25002 | 52.54% | **46.55%** | 54.13% | **48.80%** | 55.59% |

The hybrid improves values by 11.23% and 11.39% relative, and responses by 9.09% and 9.85%. Both improvements are real but below the registered 15% threshold. The separate requirement to preserve at least 90% of the Gaussian fit's explained residual energy also fails badly. The hybrid moves along a text-versus-Gaussian tradeoff rather than finding a uniformly better circuit.

Residual tails improve: the worst 1% of states carry 12.8–13.5% of small-output squared error, down from roughly 21% for the Gaussian seed-25001 baseline; the worst 10% carry 43.3–43.6%, down from about 50%. Substantial error remains across the rest of the panel.

Integrity checks pass. Export drift is below $4\times10^{-7}$; an independent CPU implementation reproduces all 16 value and response errors within $1.2\times10^{-16}$; outputs 0–3 are unchanged exactly. The two fits used 237.8 GPU-seconds. These checks support the negative interpretation by excluding the obvious export and accounting failures.

## 5. Why the hybrid features were not adopted

The preregistered follow-up compared the two learned dictionaries without assuming their columns would have the same ordering, signs, or scale. For each output it measured canonical correlations between the two eight-feature spaces under both the Gaussian and text metrics, and it compared the full correction functions.

Every output-specific dictionary failed the requirement that all canonical correlations exceed 0.9. The smallest correlations were about 0.00003 under the Gaussian metric and 0.00070 on text. This does not mean the spaces have no overlap; it means the complete eight-dimensional spaces are not the same.

Aggregate correction functions were more similar but still unstable. Cosines ranged from 0.435 to 0.797 under the Gaussian metric and from 0.701 to 0.929 on text. Only two of twelve text outputs exceeded 0.9, and none did under the Gaussian metric. The registered all-output stability gate therefore failed.

Two models can have similar prediction error while using different internal functions. The study repeatedly observed this non-identifiability: product atoms varied across restarts, individually improved feature fits failed to establish common dictionaries, and output readouts could conceal missed components. A stable aggregate prediction is useful as an approximation but is not enough to name, merge, or reuse internal units as circuit components.

## 6. Mathematical and structural conclusions

Several negative results narrow the next search space:

- A separate audit of the full single-MLP quadratic target showed that its rank bound cannot be evaded by hiding quadratic behavior in higher-degree multiplication chains. For a division-free polynomial DAG, the degree-two coefficient at each multiplication obeys
  $$H_2(ab)=a_0H_2(b)+b_0H_2(a)+H_1(a)H_1(b),$$
  so the established quadratic coefficient-rank floor still applies. Fifty random exact-integer DAGs and 1,750 node identities passed the implementation check. This is a bound on polynomial coefficient structure, not on arbitrary non-polynomial or distribution-specific functions.
- At the tested budget, output-local spectral decompositions can improve centered quadratic error, but cross-output sharing remains important. An adaptive 4,608-product allocation lowered pooled centered error to 17.44% while leaving a 42.83% worst output; a rotated shared-output construction improved it further but approached native storage.
- Exact reader and product reuse produce modest arithmetic savings, but do not repair missing native behavior.
- Token identity explains only about 6–7% of residual squared error beyond a constant on the opened panel; position does not help, and a combined token/position risk predictor failed its registered concentration criterion. The missing computation is not a simple token or position lookup.
- Error metrics induce materially different geometries on the same frozen feature spans. Native-sensitivity relative eigenvalues versus the Gaussian metric ranged from 0.037 to 0.874, and evaluation versus calibration from 0.305 to 1.423. Weighted least-squares intuition does not automatically transfer among these correlated, adaptive text-state metrics.

These findings reject a broad family of easy repairs: more readout fitting, minor objective reweighting, a different optimizer at the same local feature class, or semantic labeling of unstable atoms.

## 7. What is established and what is not

### Established

- The selected MLP16→MLP17 quartic branch admits useful compressed approximations and exact graph rewrites.
- Hidden-state distribution and native-effect weighting materially change which approximations look good.
- Low pooled error does not imply accurate output components or faithful finite interventions.
- The tested local quartic corrections produce repeatable aggregate gains but remain quantitatively inaccurate.
- The final hybrid tradeoff and its failed stability are numerically well audited.
- The learned internal dictionaries are not identified well enough for circuit merging, reuse, or semantic claims.

### Not established

- A complete replacement for MLP16, MLP17, or their surrounding transformer blocks.
- A small program that passes all native removal, composition, selectivity, and fresh OOD gates.
- Sixteen native semantic features, or a capitalization/newline circuit. Newline association was useful for testing, but native selectivity was insufficient across domains.
- A general failure of Tucker, hierarchical Tucker, CP, or arithmetic-DAG approaches. The negative results apply to specific ranks, objectives, feature classes, and budgets.
- A lower bound for arbitrary distribution-specific or non-polynomial computation.
- Completion of the broader full third-order tensor objective.

## 8. Recommended continuation

If this research resumes, the best-supported next experiment is the distinct route already identified at the cutoff: construct features from **propagated source/context geometry** rather than another output-local residual dictionary. The design should:

1. preserve all output coordinates instead of protecting the four large outputs and repairing only 4–15;
2. carry the actual normalization and downstream metric through the objective;
3. distinguish source-state features from context-dependent transport;
4. report literal product, addition, index, and coefficient costs;
5. freeze discovery before any fresh intervention or OOD test; and
6. require cross-start functional stability before naming or merging features.

The existing evidence argues against further small mixture-weight searches, readout-only sweeps, or optimizer substitutions within the current 96-atom correction family. Those questions have been tested with matched controls and oracle diagnostics.

## 9. Reproducibility and final state

The terminal hybrid run exited successfully. Its primary receipt, exported programs, independent replay, and stability audit are committed and pushed. No Codex GPU job remains queued or running, and the proposed source/context-geometry experiment was not started. The automated mathematical reviewer was disabled at finalization to prevent further credit use; its former cron file is preserved as `/etc/cron.d/bilin18-mathematical.disabled` and can be restored by renaming it.

Primary artifacts:

- [Hybrid result](../../direct_tensor_match/HYBRID_LOCAL_QUARTIC_NATIVE_V1.json)
- [Independent export audit](../../direct_tensor_match/HYBRID_EXPORT_AUDIT_V1.json)
- [Hybrid protocol](../../direct_tensor_match/HYBRID_LOCAL_QUARTIC_PLAN_V1.md)
- [Feature-stability protocol](../../direct_tensor_match/HYBRID_FEATURE_STABILITY_PLAN_V1.md)
- [Feature-stability result](../../direct_tensor_match/HYBRID_FEATURE_STABILITY_V1.json)
- [Final hybrid result note](research_update_2026-09-22_0822_hybrid_result.md)
- [Model and target context](research_update_2026-09-22_0000_model_context_and_research_trajectory.md)
- [Native optimizer comparison](research_update_2026-09-22_0602_optimizer_comparison.md)
- [Normalization-stage audit](research_update_2026-09-22_0559_normalization_error.md)
- [Document-bootstrap uncertainty](research_update_2026-09-22_0736_document_uncertainty.md)

All preregistered failures remain recorded as failures. The final result is useful because it closes several plausible explanations without overstating a partial improvement: the metric matters, the hybrid improves text behavior, and the current dictionaries still do not recover a stable reusable circuit.
