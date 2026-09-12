# Folded producer structure and its native behavioral failure

12 September, results through16:32UTC. This follows the [four-reading source interface](REGIONAL_COMPETING_CUES_V1_MATH.md), [failed shared cubic fit](FOLDED_PRODUCER_CUBIC_NATIVE_V1_RESULT.json), and [requested report section10](explanations/for_logan/research_update_2026-09-12_1536_interaction_decomposition.md#10-latest-follow-up-folding-the-regional-readers-into-their-upstream-attention-producers). The report's pending behavioral test is now complete; its timestamped account remains unchanged.

## Native verdict

The shared linear parent recovered by two cubic source fits is almost exactly head13.0's leading folded value reader. Retaining that reader with full native QK1×QK2 captures63–65% of this head's folded coefficient energy, but it **fails native effect fidelity and the registered meaningful-transfer threshold**.

The managed cache executed12 native batches in1.38seconds, with producer-fold error<=3.14e-7 and bit-exact replay of saved contextual states. Its 685KB artifact holds the contribution to four downstream readings from attention8+9+13, head13.0, and its rank1 value part. CPU interventions retain recipient query, projected key norms, first-token lookup, other producers and downstream background. They swap donor contributions divided by recipient residual RMS into all four readings. This is a conditional producer-to-reader edge intervention, not recursive replacement of a native head.

| City assignment / clause order | Full group mean directed transfer | Whole head13.0 | Euclidean value component | Consumer-weighted component |
|---|---:|---:|---:|---:|
| Original / editor first | .16969 | −.01051 | .00428 | .00508 |
| Original / editor second | .13859 | −.00888 | .00333 | .00392 |
| Reversed / editor first | .24815 | .00919 | .00323 | .00398 |
| Reversed / editor second | .19902 | .00847 | .00331 | .00407 |

Transfer is the change in British-minus-American token margin, signed toward the donor cue and averaged across24 directed swaps per cell. Each reverse direction is included; these are not24 independent semantic concepts.

The Euclidean component preserves the sign on95/96 swaps, with unrelated transfer below the registered50%relative ceiling, but provides only1.3–2.5%of the full producer-group transfer rather than the required10%. Its full-head effect error is59–141%, rather than<=10%. The complement opposes the selected component for the original city assignment. Component-plus-complement effect nonadditivity is only.24–.41%of full-head effect norm. Thus suffix nonlinearity does not explain away the disagreement. The head's city-pair-dependent sign does not establish a stable regional semantic unit.

[Registered bars](STRUCTURED_PRODUCER_NATIVE_V1_PREREGISTRATION.md) · [native cache](STRUCTURED_PRODUCER_CACHE_V1_RESULT.json) · [behavioral result](STRUCTURED_PRODUCER_EFFECT_CPU_V1_RESULT.json).

## Executed negative-result audit: coordinate dependence

The frozen downstream source polynomial is

$$
F(q,f)=\beta_0(q)f_0f_1f_2+\beta_1(q)f_0f_1f_3.
$$

For diagonal nonsingular S=diag(s0,...,s3), replacing f by Sf and dividing beta_a by s0*s1*s(2+a) leaves the function identical. The compiled executor realizes this by rescaling readings and compensating the two dual rows.

An ordinary leading singular component of the four-output value matrix M is not invariant under this freedom. After rescaling one reader100-fold, the whole branch replays bit-for-bit, but the leading source-reader cosine can drop to.00577 and apparent retained energy rises above99.8%. This is a limitation of treating arbitrary intermediate coordinates as a Euclidean output space. It does not invalidate the registered fit in its chosen coordinates or establish that natural unit-reader coordinates are useless.

[Gauge audit](FOLDED_PRODUCER_READER_GAUGE_V1_RESULT.json) implements this counterexample on the actual frozen component.

## A weights-only invariant alternative, solved exactly

Define a consumer pullback metric

$$
H=\mathbb E\left[J_fF(q,f)^T J_fF(q,f)\right].
$$

Here J is the derivative of the downstream polynomial output with respect to the four readings. The expectation is over independent isotropic Gaussian query/current inputs, uniform complete-vocabulary first-token initialization, and32 relative positions. There is no language-data or task-label fitting. This choice is an explicit distributional assumption; real contextual correlations, finite-change nonlinearities, final-suffix sensitivity and QK normalization gates are omitted from the discovery metric. Native validation retains the actual gates.

Write f=Cx+t, where x is standard Gaussian and t ranges over the full token lookup. Let Sigma=CC^T. The required fourth moments are the token fourth moment, six products of Sigma with token second moments, and the three Gaussian covariance pairings. Query-dependent beta functions are quadratic, so their output Gram also uses exact fourth moments. These Gaussian pairings follow [Isserlis's original moment formula](https://academic.oup.com/biomet/article-abstract/12/1-2/134/193428). We implement the finite token sum and independent Gaussian parts directly.

For a fixed metric, solve

$$
\min_{\operatorname{rank}(\widehat M)\le1}
\|H^{1/2}(M-\widehat M)\|_F^2.
$$

When H is positive definite, this is ordinary truncated SVD after the invertible left transformation H^(1/2), a special separable weighted low-rank problem rather than general elementwise weighted fitting. See [Markovsky's treatment of weighted low-rank approximation](https://imarkovs.github.io/book/book2e-2x1.pdf). If v is the leading right singular vector of H^(1/2)M, the solution is Mhat=Mvv^T. No iterative convergence issue remains for this fixed matrix problem.

Under the exact re-encoding, H'=S^(-T)HS^(-1), M'=SM, so

$$
M'^T H' M'=M^T H M.
$$

The selected source direction is invariant when its leading eigenvalue is isolated. The equivalent four-output projector P=Mhat M^+ is generally oblique; row-valued cached contributions must multiply P^T. We charge the full native QK and state dependencies as before.

Independent controls pass: metric covariance error<=5.63e-16; back-transformed selected-map error<=3.54e-13; Monte Carlo checks of the exact fourth moments and query Gram disagree by1.50%and.784%, respectively. H has eigenvalues .175,.182,.199,.444 after trace normalization. Thus the original natural coordinates were only moderately anisotropic under this particular consumer metric.

## The invariant metric does not rescue this candidate

The weighted source direction has cosine.99916 with the original one. It gives96/96 signed swaps in the expected direction but only1.6–3.0%of the producer-group transfer. Full-head fidelity error remains51–148%. Both B/C predictions fail under their unchanged thresholds. The new method solves an actual coordinate-dependence problem; the behavioral miss survives it.

[Metric controls](CONSUMER_PULLBACK_V1_CONTROL.json) · [frozen comparison](CONSUMER_PULLBACK_V1_PREREGISTRATION.md) · [native weighted result](CONSUMER_PULLBACK_EFFECT_CPU_V1_RESULT.json).

**Narrow conclusion:** the leading folded value component of this coefficient-dominant producer head is not a sufficient regional producer at the registered boundary. Neither a high coefficient fraction nor gauge invariance establishes circuit significance. This does not rule out more complex shared QK/value structure, other heads, or sparse mixed producer–consumer paths. The next discriminating object is the exact cubic interaction among producer contributions and background, before treating any one producer direction as the unit of computation.

## 16:36 — Exact interaction paths carry the group effect

Write the four downstream readings as f=B+H+O: B is the recipient background after subtracting the attention8/9/13 group, H is wholehead13.0's contribution, and O is the rest of that group. Contributions include recipient RMS scaling. For each child, expanding the three ordered reader slots and collecting equal source-group multisets gives ten paths:

$$
B^3,\ B^2H,\ B^2O,\ BH^2,\ BHO,\ BO^2,\ H^3,\ H^2O,\ HO^2,\ O^3.
$$

These symbols mean the sum of all assignments to the parent0, parent1 and child reader slots with that multiset. For example BHO includes six assignments; it is not six times one ordered product when the reader roles differ. All terms retain the original query consumer and normalization. The decomposition is an exact algebraic diagnostic, not necessarily a cheaper implementation than the original two shared cubic features.

The summed writes replay within1.03e-7 and the full group-swap effect exactly matches the previous CPU test. The seven mixed paths reproduce that effect within**.16–.29%** in each of the four cells (registered10%bar passes). But the three paths containing both H and O carry only**.8–1.9%**of full effect norm, missing the registered10%cross-producer contribution threshold. This is mainly **background–producer mixing**, not strong cooperation specifically between head13.0 and the other selected producers.

Keeping only B^2H+B^2O incurs**11.8–14.1%**effect error. BO^2 is the largest remaining contribution, with mean directed effect .0160–.0224. Individual path margin effects sum to the full effect within .048–.108%, although additivity was not assumed by the test. Pure producer cubes are small under this particular partition and panel; this does not establish that all cubic paths are unimportant or that the background is independently generated.

The mixed-path pass is a conditional local result and may partly reflect small producer contributions relative to background. No behavioral fit selected these ten terms, but the producer group itself came from earlier behavioral tracing. A broader unsupervised decomposition cannot be inferred from this one cut. [Exact executor and all cells](PRODUCER_INTERACTION_CUT_CPU_V1_RESULT.json).

## 16:43–16:48 — Two of27 fixed components pass; common output is a new hypothesis

The complete frozen bank produces two exploratory passers, head8.2 and head9.8. Across all four cells, their component/fullhead effect errors are1.66–2.55%and1.84–3.72%; their group-relative mean directed transfer fractions are27.7–31.7%and60.1–72.0%. Both pass the fixed sign and unrelated-effect bars; the other25 do not pass both criteria. Head13.0 exactly replays its earlier failed weighted test. [All27 scored candidates](STRUCTURED_PRODUCER_BANK_EFFECT_V1_RESULT.json).

This is stronger than continuing to inspect the coefficient-dominant head, but remains a reused-panel, multiple-candidate screen. Head8.2's existing newline-setter dossier must be retained. Current evidence concerns its regional downstream edge, not the whole head's selectivity or a newly discovered whole-head role.

The passers' output writers have ordinary cosine.999252 and consumer-metric cosine.999491. Their combined source-reader cosine has magnitude.5162; current and first sectors differ substantially. Their current states live at different layers. A common downstream output with separate source/QK computations is therefore a plausible grouping, not an established common input feature. The weight-fixed shared-output proposal changes each writer by about1.6%in the consumer metric and saves only two output scalars locally; its value would be a reusable interface, not major compression.

[Fresh confirmation preregistration](STRUCTURED_PRODUCER_PASSERS_CONFIRMATION_V1_PREREGISTRATION.md) and [48fixed rows](STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json) now exist. They use four new cities, six new spelling pairs and two new templates. Tokenization and pair controls pass; native capability and individual/joint/merged effects are **not yet evaluated**. A separate newline-capability panel is still needed before claiming preservation of the known8.2 service. This is the actively prepared next experiment, with no confirmation-data refit.

## 17:02 — Fresh confirmation passes and a scalar producer package executes

The fresh48-row panel passes native capability: all24cue contrasts positive, template means3.09and1.99. Individual component/fullhead effect errors are2.34–2.98%. The pair retains92.53%and89.18%of the selected producer-group transfer. Common-output merging changes individual effects by.37–.40%and joint effects by.227–.245%. All registered fresh A/B/C predicates pass. Joint effects differ from the sum of individual effects by1.62–2.47%, so composition was evaluated directly. Removing the unmerged pair reduces the full native cue contrast by.263and.152, approximately8.5%and7.6%of those native template contrasts; group-relative89–93%must not be described as explaining89–93%of the whole behavior. [Fresh results](PRODUCER_FRESH_CONFIRMATION_EFFECT_V1_RESULT.json).

The new [scalar producer package](extracted_circuits/regional_shared_producers_8_2_9_8_v1/README.md) directly executes each head's complete QK1×QK2 routing and a single value reading, with a shared downstream output. Its first-value input is replaced by a complete50,304-token table compiled from native embedding and block0 re-entry/RMS weights. It requires two distinct native normalized input sequences at layers8and9. Output contributions include residual propagation scales and must be divided by recipient residual17 RMS at the downstream normalized reader interface.

The package has1,282,566scalars/5,541,936tensorbytes;1,179,648scalars are still full QK maps. Native individual write replay error<=3.02e-7, joint8.75e-8, first-token lookup8.68e-8. CPU selected-margin swap/removal replay error<=6.86e-5passes1e-4. This validates execution of the approximate shared-output candidate; its semantic approximation error remains the separately measured.23–.25%joint value above. [Native check](SCALAR_PRODUCERS_NATIVE_V1_RESULT.json) · [effect replay](SCALAR_PRODUCERS_EFFECT_REPLAY_V1_RESULT.json).

This is a conditional cross-boundary grouping with controlled fresh lexical/template evidence. Independent current-state generation, broader corpus OOD, and a dedicated newline capability/selectivity panel remain unfinished. The next test addresses that control without refitting the passers or replacing failed examples.

## 17:30 — Newline control inconclusive; native writer lift verified

[NewlineV1](SCALAR_PRODUCERS_NEWLINE_V1_RESULT.json) passes implementation replay but fails native capability and wholehead-zero positive control on authored lists/forms. Native newline CE9.01/10.86; wholeheadzero changes CE by−.618/−.813. Small regional-edge changes do not establish service preservation. Prior dossier used FineWeb targets and mean replacement. [V2](SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_PREREGISTRATION.md) retains authored rows and adds32fixed natural prefixes, crossing zero/mean interventions; rows exist, runner not yet implemented.

[Physical lift](SCALAR_PRODUCER_NATIVE_LIFT_V1_RESULT.json) computes each unmerged component’s native residual writer from its original OV/current-first value maps. Residual-scaled downstream projection replays frozen writers within4.3e-16; physical writer cosine.899 versus downstream~.999. This enables a principled recursive component-removal test but is not an executed behavioral intervention. The norm-aware interface control and all new findings are explained in the [requested17:30report](explanations/for_logan/research_update_2026-09-12_1730_shared_producer_interactions.md).

## 17:40 — Natural newline control passes; context explains the authored miss

[Factorial V2](SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_RESULT.json) completes150bodyforwards in12.99seconds, A/B/Cpass. Authored V1 CE and effects replay bit-for-bit. Both authored families still fail native capability and both mean/zero controls improve newline CE; changing intervention alone does not fix those prompts.

On the two fixed16-row FineWeb halves, native newline CE is.8933/.6095 and all32newline-minus-comma margins are positive. Mean wholehead8.2 replacement increases newline CE by.03332/.02480, with12/16and13/16positive damage. Zero removal also increases mean CE(.02875/.02010). Therefore the context difference is the strongest explanation supported by this factorial audit. It does not establish that zero and mean interventions are equivalent generally.

Conditional joint regional-edge removal has meanabs newline CE change.000260/.000347, maximum.000967/.002065, well inside the unchanged.02mean/.1max preservation bars. All individual candidates pass too. This establishes a training-domain natural newline control for the consumer-specific regional edges, not global head/component selectivity. Source examples were selected by true next-token labels, without model-score filtering; factors remain weight-frozen. The old V1 failure remains recorded.

The next [recursive physical-removal test](SCALAR_PRODUCERS_RECURSIVE_REMOVAL_V1_PREREGISTRATION.md) removes original unmerged source components at their own producer outputs and recomputes every subsequent layer. It uses the natural OV-derived residual lift, with original QK routing and source readers; it does not lift the common four-reader approximation arbitrarily. Its implementation is committed. Queue gate initially rejected dict-key syntax before model execution; explicit quoted prediction keys resolve that parser requirement with unchanged predictions.

## 17:46 — Physical removal is stronger, but not fully selective

[Recursive removal](SCALAR_PRODUCERS_RECURSIVE_REMOVAL_V1_RESULT.json) completes496forwards in6.61seconds. Baseline and head-subtraction checks pass. Native pair removal reduces cue contrasts by2.172/1.343, or70.3/67.4%of the native contrasts. Individual/joint reductions have the expected direction on all24paired examples. These are reused regional validation contexts, not newOOD. The physical unmerged source components are removed at their actual outputs and every subsequent layer recomputes.

Registered wholehead fidelity fails: individual errors8.2=39.4/31.7%,9.8=7.47/21.7%; joint10.48/15.22%, against10%. Unrelated-margin ratios.08–.11 pass. Joint nonadditivity is17.7/32.8%, much larger than the former consumer-edge result. This is neither a failed component effect nor a completed selective circuit: the component effect is large, but it is not the same as wholehead removal.

Natural newline meanabs changes for joint removal are.00995/.01476, within.02. Maximum changes.0494/.1665 fail the.1bar in the secondhalf. The rowwise audit identifies one fixed sourcechunk177 (baselineNLCE.2996): component8 damage.10834,component9.07473,joint.16654,meanwholeheadcontrol.07898. Every row remains included. This is an actual collateral failure, not an invalid capability control. [Audit](SCALAR_PRODUCERS_RECURSIVE_AUDIT_V1_RESULT.json).

### Serial component dependence explains much of the interaction diagnostic

[Serial audit](SCALAR_PRODUCERS_SERIAL_INTERACTION_V1_RESULT.json),144forwards3.13sec, passes A/B/C. Let e8,e9,eJ be individual/joint baseline-subtracted margin effects and I=e8+e9-eJ. Repeating joint removal while subtracting head9's pristine-native component field gives eF. The change D=eF-eJ has cosine.958/.998 with I; its norm is1.286/1.093times I. The residual norm ||I-D||/||I|| is.436/.113. Baseline/dynamicjoint replay is exact and recovered scalar fields match native within1.64e-6.

This indicates that changing the second removed component after the first removal explains a substantial aligned part of the measured nonadditivity. It does not isolate the direct8->9edge: MLP8 and norms intervene. Freezing a baseline subtraction after its source changes is a hybrid counterfactual and may over-remove a component; it is not adopted as the circuit intervention. Newline selectivity remains failed.

### Exact composed bridge through the intervening bilinear MLP

Let z be the residual entering MLP8, d the physical writer of the8.2component, and a its scalar output at this position. Removing it gives z'=z-ad. Define rho²=mean(z²)+epsilon, rho'²=mean(z'²)+epsilon, and the bias-free quadratic M(z)=D[(Lz) elementwise (Rz)]. The MLP bias cancels between the two states. The exact output change is

$$
\Delta x_8=-ad+
\left(\frac{1}{\rho'^2}-\frac{1}{\rho^2}\right)M(z)
-\frac{a}{\rho'^2}D\big[(Lz)\odot(Rd)+(Ld)\odot(Rz)\big]
+\frac{a^2}{\rho'^2}D\big[(Ld)\odot(Rd)\big].
$$

For the layer9 raw read stack E9 and residual coefficient lambda9, the next interface change is lambda9 E9 Delta x8. This is a direct producer term, a background term changed by normalization, a mixed producer/MLP term, and an amplitude-square term. After computing it, layer9 input RMS and projected QK normalization still need their actual norms; the expression alone does not close their generation.

[CPU actual-weight algebra test](SCALAR_PRODUCER_MLP_BRIDGE_V1_RESULT.json) replays all513rawQK/value reads to2.33e-15aggregate relative error on32independent probes at perturbation/input norm ratios0,.01,.1,1. Dropping quadratic or normalization terms produces nonzero probe errors; these random-domain numbers are not native mediation estimates. The exact bridge is a new composed-path tool, not another fitted low-rank representation. Native z8/a8 evaluation is the next required evidence before attributing the regional interaction to these terms.

## 17:54 — One fixed linear map exactly represents the directional MLP response

For the fixed physical producer writer d, define

$$
J_d=D\left[\operatorname{diag}(Rd)L+\operatorname{diag}(Ld)R\right].
$$

This is a constant matrix, not a Jacobian frozen at one contextual state. It maps any z to the mixed bilinear term for that fixed direction. The pure quadratic term is not independent:

$$
D[(Ld)\odot(Rd)]=\tfrac12J_dd.
$$

Let u=M(z)/rho² be the original bias-free normalized MLP output, supplied as an explicit background input. The exact change in the residual-plus-MLP computation is

$$
\Delta x_8=-ad+
\left(\frac{\rho^2}{\rho'^2}-1\right)u
-\frac{a}{\rho'^2}J_d\left(z-\frac a2d\right).
$$

This packages the full finite-amplitude response as one linear map with a shared amplitude and norm terms. It is an exact interaction-path representation, without low-rank approximation or behavioral fitting. [Actual-weight CPU control](SCALAR_PRODUCER_DIRECTIONAL_MLP_V1_RESULT.json) verifies the quadratic identity to1.97e-15, complete response to4.17e-14inFP64 and6.46e-6inFP32 on independent probes. [Frozen program](SCALAR_PRODUCER_DIRECTIONAL_MLP_V1_PROGRAM.pt) contains J_d and d:1,328,256scalars/10,626,048FP64tensorbytes, compared with15,926,400scalars in the original full MLP.

The comparison is conditional: z, the native background u, and amplitude a still require generation, and the rest of the native model remains charged. It is not a12fold whole-model compression claim. The square term is already implicit in this program; removing it is an approximation, not an equivalent rewrite.

[Native-state preregistration](SCALAR_PRODUCER_MLP_BRIDGE_NATIVE_V1_PREREGISTRATION.md) and cache/scorer implementations are committed. At17:54 the96-forward native cache is queued behind the confirmed live peer663 process. It will capture pristine z8, pristine/changed raw r9 and scalar fields on48reusedregional rows; no native bridge result exists yet. The fixed CPU comparison tests exact four-term replay, direct+mixed sufficiency, and whether direct-only misses. It must complete before assigning native importance to any of the algebraic terms.

## 18:10 — The composed interaction predicts native scalar and behavioral changes

The queued [native cache](SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_RESULT.json) completed96forwards in2.28seconds after the peer job finished. All cache bars pass; native/physical8 margins replay exactly. No duplicate job was launched during the live wait.

[Native four-term bridge](SCALAR_PRODUCER_MLP_BRIDGE_NATIVE_V1_RESULT.json) passes A/B/C. Exact residual-change errors are1.38e-5/1.37e-5, including FP32 native rounding; exact scalar9-change errors are7.98e-7/5.35e-7. Direct+mixed scalar prediction errors are1.32%/.50%; direct-only33.45/36.20%, confirming that the MLP response matters. The changed-normalizer background term is small on this panel, but actual normalization remains in all executions. Median perturbation norm is.57%of native z8norm, maximum2.06%; the native regime differs from the earlier random large-amplitude probes.

[Directional runtime](SCALAR_PRODUCER_DIRECTIONAL_NATIVE_V1_RESULT.json) also passes both FP64 and FP32 native checks. FP32 scalar-change error<=9.37e-7 with the original bias-free MLP output supplied as background. This verifies the fixed constant-map program on actual states; it does not eliminate the cost of producing those states or background.

[Signed end-effect test](SCALAR_PRODUCER_MLP_BRIDGE_EFFECT_V1_RESULT.json),240forwards4.08seconds, passes A/B/C. Comparisons use the difference between dynamic joint removal and the prior frozen-second-write joint removal, preserving the serial-effect denominator. Exact bridge effect errors are4.75e-6/4.78e-6. Direct+mixed errors are1.143%/.377%; direct-only33.724%in bothfamilies. Thus the shortened interaction path predicts the measured signed native serial effect, not only intermediate activity. These are reused48regional rows, not another fresh or corpusOOD test.

### Newline collateral survives the fixed country audit

[Country audit](SCALAR_PRODUCERS_NEWLINE_COUNTRY_AUDIT_V1_RESULT.json),30forwards1.37seconds, reproduces the original failure exactly and passes capability/mean-head control for allsixvariants. Joint newline CE changes are Australia.16654, Britain.14274, Canada.10615, America.09792, France.12756, Germany.12234. Five exceed the original.1max preservation threshold. The range is41.2%of Australia's damage, missing the registered50%country-dependence bar. This is appreciable variation, not no country effect, but it does not support an Australia-only explanation. All variants derive from one postselected paragraph; they are not six independent natural observations. The original32-row failure stays included.

### Joint QK remains necessary; value-only transfer is not generally sufficient

[Four-corner routing/value auditV2](SCALAR_PRODUCER_JOINT_QK_VALUE_V2_RESULT.json) keeps QK1*QK2 together. Let gamma0,gamma1 be the original/changed complete routing operators and v0,v1 the scalar value fields after component8 removal. Then

$$
\Delta a_9=
\gamma_0\Delta v+\Delta\gamma\,v_0+\Delta\gamma\,\Delta v.
$$

The three terms are value-only, routing-only, and their interaction. Native endpoints replay<=1.78e-7. V1 defined the mixed term by subtraction, making its identity check tautological; V2 independently contracts (gamma1-gamma0)(v1-v0) and reproduces the full change within7.43e-16. The numerical B/C misses are unchanged.

Value-only scalar-change error is56.61%in template0 and9.925%in template1; routing-only errors136.46/105.30%. The mixed term has19.32/7.50%of the full-change norm. Routing/value effect cosines are−.681/−.319: contributions partly oppose, so component norms cannot be read as additive percentages of explanation. The registered general value-only and small-interaction claims fail. This does not assign a task to QK1 versus QK2 or prove disjoint task input spaces. The supported bridge retains complete joint routing. Its end-task routing/value partition is a next test, not yet measured.

The new supported object is a conditional arithmetic interaction across head8.2, MLP8 and head9.8, with a common downstream writer available at the regional consumer. Independent input generation, broadOOD and robust selective removal remain unresolved. The explicit native newline collateral prevents a four-property promotion.

## 18:23 — Fresh contextual removal and interaction transfer pass

[Signed routing/value test](SCALAR_PRODUCER_JOINT_QK_VALUE_EFFECT_V1_RESULT.json) completes240forwards4.12sec, A/Cpass,Bmiss. Value-only generation has35.15/6.19%serial-effect error; routing-only139.50/102.01%. Joint-field replay<=5.24e-6. This confirms at the behavioral boundary that the general value-only shortcut fails; bothQK factors remain together in the retained routing computation.

The new [48-row email/letter panel](SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_ROWS.json) changes templates, cities (Glasgow/Phoenix,Cambridge/Detroit) and six endpoint pairs (neighbours,organise,realise,labelled,defence,metre versus US spellings). Novelty checks covered existing regional/producer row files, not pretraining data. Factors and the directional MLP map were frozen before evaluation.

[Fresh result](SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_RESULT.json),336forwards5.29sec, passes A/B/C. Both template families have12/12positive native cue contrasts, means2.646/2.971. Physical pair removal reduces70.49/65.10%of those contrasts; every individual/joint removal reduces all24paired contrasts. Unrelated margin ratios.050–.065 remain below the fixed.5bar. This strengthens controlled contextual generalization; it does not repair the prior natural newline max-error miss.

The composed direct+mixed MLP response predicts the signed serial effect within.866/.929%, while the exact directional response is within3.51e-6/3.31e-6. Exact scalar-change replay<=3.10e-7. Pair nonadditivity is31.9/38.5%on this panel, so success is actual predicted composition rather than an assumption that individual effects add. Native z8,a8,u8,r9 and the remaining fullQK/background/suffix are still external dependencies.

[Post-result subgroup audit](SCALAR_PRODUCERS_CONTEXT_AUDIT_V1_RESULT.json) reports allfourtemplate/city strata: mean joint coverage64.2–72.2%, allsixpairs positive in each. Individual concept fractions range53.7–87.9%. This is descriptive heterogeneity evidence, not a new pre-registered criterion or six independent domain shifts. No endpoint or city was removed.

The controlling handoff's appended success criterion was reread: memory savings alone are not interpretation, and a large matrix cannot be called one explained operation. The supported structural result is the explicit shared-writer/serial interaction with native correspondence and held-out effect prediction. The1.33Mdirectional matrix is still arbitrary numerical content, and background generation is still charged. No four-property or whole-model completion claim.

### Next intervention is donor interchange, not another removal-only validation

[Interchange preregistration](SCALAR_PRODUCERS_INTERCHANGE_V1_PREREGISTRATION.md) and [all-row matched donor map](SCALAR_PRODUCERS_INTERCHANGE_V1_DONORS.json) are now prepared. Replace component8/9scalar fields with the opposite-cue donor's fields, singly and jointly. The bridge then receives signed amplitude a8_recipient-a8_donor instead of the earlier zero-removal amplitude. Preserve native re-execution and compare exact/directmixed generated9effects against the explicit frozen-recipient9 control. This tests whether the operation supports new manipulations, not just its original lesion. Runner implementation and native outcomes are still pending.

## 18:55 — Interchange passes; source-sector selectivity misses

[Interchange](SCALAR_PRODUCERS_INTERCHANGE_V1_RESULT.json) passes all registered bars: joint donor transfer72.4/67.7% of native cue contrast, every48directed row positive for each individual/joint arm, self-donor bit-exact. The direct+mixed directional response predicts the signed serial effect within1.43/2.33%. These are new manipulations on known email/letter contexts, not another OOD panel.

[Four-sector factorial](SCALAR_VALUE_SECTOR_FACTORIAL_V1_RESULT.json) executes1312forwards in17.40seconds. Native/fullmask/control replay is exact. Split each frozen value into current and first-state sectors, preserving the same full joint QK routing and physical writer. Masks6/7/14/15 meet regional coverage; no mask meets regional plus newline criteria. Head8-first plus head9-current covers67.8/61.0% but has maximum newline CE change.13709 against.1. The full pair remains.16654. No bar or example changed.

[Exact finite-intervention expansion](SCALAR_SECTOR_MOBIUS_V1_RESULT.json) reconstructs all16masks exactly. For the selected two-sector pair, regional interaction norm is~30% of joint effect with negative mean. On preserved newline row30, damage.137089=.060216+.076724+.000150; interaction is only.11% of joint damage. [Loss/readout audit](SCALAR_SECTOR_LOSS_SPACE_AUDIT_V1_RESULT.json) finds2.66% interaction fraction for newline-comma margin on that same row, so mostly additive collateral is not solely the CE readout. Descriptive post-result audit, no selectivity rescue or independent causal ownership claim.

The [requested full update](explanations/for_logan/research_update_2026-09-12_1855_composed_circuit_interactions.md) explains the computations and scope. Finer consumer paths and joint-routing input spaces remain hypotheses, not completed decompositions.

## 19:12 — Complete joint-key path bank and position robustness

[Serial query/source algebra](SCALAR_SERIAL_JOINT_PORTS_V1_RESULT.json) independently contracts jointquery(q1 tensor q2), jointkey(k1 tensor k2), and scalarvalue deltas. Native scalar-change replay<=8.24e-7; source-only error24.1/5.3% misses10%both. Complete signed-effect test is queued, not scored.

[Weight-only bank](SCALAR_JOINT_KEY_BANK_V1_RESULT.json) averages the exact joint-numerator source influence over32relative positions for8.2/9.8. All256key directions are retained in four64D bands; ten unordered pairs retain bothkeyslot assignments and native fullnormalizers. This is complete path accounting, not ranktruncation or semantic identification. [Native9control](SCALAR_JOINT_KEY_PATHS_V1_CONTROL.json) sum replay<=1.64e-7, change<=5.75e-7. The frozen120candidate regional/newline screen is committed and queued, with prior50%coverage/.02mean/.1max preservation bars and all failed rows retained.

[Position robustness](SCALAR_JOINT_KEY_POSITION_AUDIT_V1_RESULT.json) passes its narrowly registered leading-path tests. Near/far weight-only reconstructions retain.966–.980top64overlap for9.8 and.969–.970for8.2; native9leading-path field errors2.1–6.6%. Orthogonal coordinate rotation within bands changes paths5.70e-16. Smaller paths show much larger relative field differences, so these results do not identify everyband as stable. [Denominator audit](SCALAR_JOINT_KEY_POSITION_SCALE_AUDIT_V1_RESULT.json) reports everydifference relative to fullnative scalar field as well; smallabsolute discrepancies do not rescue individual unstablepathidentity. No basis or candidate was replaced after these audits; no newOOD or selectivity result.

## 19:17 — Shared output does not mean the same joint-QK numerator

[Cross-head weight geometry](SCALAR_JOINT_KEY_CROSSHEAD_V1_RESULT.json) compares the complete separately symmetric joint-QK numerator functions of8.2/9.8 in common formal query/source coordinates. Full and leading-band numerators have cosine−.0093to−.0142 at threefixedrelativepositions. Independent dense cross-inner-product and symmetric-support controls pass<=1.12e-16. Leading64D linearspace overlap is.1311; symmetric-product support envelope overlap.01747. The actual layerstates differ; normalization is absent from this coefficientmetric. Thus distinct formalnumerators feeding a sharedwriter are supported, not disjointtaskspaces or equivalentcontextualcomputations.

[Random-support audit](SCALAR_JOINT_KEY_CROSSHEAD_NULL_V1_RESULT.json) contextualizes the smallabsolute product overlap using32seeded Haar64Dspaces. Actualsupportsharing exceeds random; retain both absolute and matched-null numbers rather than calling the spaces disjoint. No queuedcandidatechanged and no additionalbehavioralscreenresult yet.
