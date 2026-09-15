# September 15 research update: a multiplicative upstream response in the subject-number circuit

## High-level summary

The compact subject-number circuit has a stable output direction at attention head 11.3, but its amplitude was still unexplained. The new result localizes that missing amplitude to an interaction between two scalars:

1. $z_b$: where the current L11H3 write lies on the head's native output-weight axis;
2. $s_b$: how far that same scalar moves when the upstream MLP6/7 number source is switched, with background $b$ held fixed.

The useful program is multiplicative:

$$
\widehat\alpha_b=\beta_0+\beta_z z_b+\beta_s s_b+\beta_{zs}z_bs_b.
$$

On leave-one-construction-out prediction, $z_b$ alone has relative $L_2$ error `.6093`. The response alone reaches `.4449`, and the additive pair reaches `.4515`; both miss the preregistered `.40` ceiling. Adding the interaction $z_bs_b$ reaches `.3769`, improves on the native baseline by `.2324`, and passes. This says the MLP6/7 input does not merely add a fixed number signal to L11H3. Its effect depends on the head's existing native state.

A follow-up tried to remove the row-specific opposite-number donor. It averaged the MLP6/7 displacement into one fixed 1,152-dimensional vector per answer direction, using only the other syntactic construction. Those two vectors predict the exact response scalar surprisingly well: cosine `.9659` and relative $L_2$ error `.2880`. But the error is amplified by multiplication with $z_b$; the complete coefficient program has error `.5159`, missing its `.45` ceiling and improving over the native baseline by only `.0935` rather than the required `.10`.

So the interaction is a positive localization result, while the simplest donor-free realization is a valid null. The next circuit step is a vector-valued recipient-side predictive state. We will not turn the two-vector proxy into a lexical or background lookup table. The next weight-folding step remains the independent regional path: split head9.8 QK1's dominant carry×carry interaction into embedding and layer-0–7 self/cross terms.

![Prediction errors for the response interaction and donor-free proxy](assets/research_update_2026-09-15_subject_response_interaction.png)

*Figure 1. Lower is better. Left: only the exact multiplicative response model passes the `.40` coefficient-error ceiling. Right: the two-vector proxy passes its response-scalar gate but fails after composition into the coefficient program.*

## Terms and computation

**Native output axis.** Let $u$ be the top left singular direction of L11H3's native output-projection weight slice. It is checkpoint-defined and was frozen before these experiments. Let $H(x)$ be the exact reconstructed L11H3 projected write when raw state $x$ is installed at the MLP8 input and propagated through the registered path.

**Background.** The MLP8 input is decomposed into earlier source groups. $b$ denotes one of the 16 subsets of four already identified background groups, E/A/U/W. For example, $b=\mathrm{EA}$ installs E and A from the opposite-number prompt while leaving U and W recipient-native.

**Native coordinate.** For the raw MLP8 input $x_b$ under background $b$,

$$
z_b=u^\top H(x_b).
$$

This is an activation coordinate on a native weight direction. It is not a learned probe.

**Head-local causal-response coordinate.** Let $x_{b,YZ}$ additionally switch the grouped MLP6/7 source, called `YZ`, to its opposite-number value. Then

$$
s_b=u^\top\left(H(x_{b,YZ})-H(x_b)\right).
$$

This measures the signed response of the head's native scalar, rather than the downstream answer-logit response. No answer logits or behavioral outcomes are used to construct it.

**Frozen model comparison.** The target $\alpha(d,|b|)$ is the coefficient of the already validated rank-one L11H3 write program for answer direction $d$ and background cardinality $|b|$. Four OLS forms were fixed in advance:

$$
\begin{aligned}
\text{native} &: [1,z_b],\\
\text{response} &: [1,s_b],\\
\text{additive} &: [1,z_b,s_b],\\
\text{interaction} &: [1,z_b,s_b,z_bs_b].
\end{aligned}
$$

Each form was trained on one syntactic construction and evaluated on the other, in both directions. The interaction form was selected only because it was the smallest response form with relative $L_2\leq .40$ and improvement over native of at least `.10`.

| Form | Cosine | Relative $L_2$ | Sign agreement | Decision |
|---|---:|---:|---:|---|
| Native `[1,z]` | `.7968` | `.6093` | `.9609` | baseline |
| Response `[1,s]` | `.8962` | `.4449` | `.9727` | misses error gate |
| Additive `[1,z,s]` | `.8936` | `.4515` | `.9785` | misses error gate |
| Interaction `[1,z,s,zs]` | `.9265` | `.3769` | `.9551` | selected |

The collections $z$ and $s$ have cosine `-.8704`, so they are strongly correlated. The additive model nevertheless fails and their product passes, which is evidence for state-dependent gain rather than two interchangeable readouts.

## Donor-free proxy test

For each held-out construction and direction $d$, the follow-up averaged only the training construction's raw MLP6/7 displacement:

$$
p_d=\mathbb E_{i\in\mathrm{train}(d)}[x_{i,YZ}-x_i].
$$

It then estimated the response on a held-out row without its donor state:

$$
\hat s_b=u^\top\left(H(x_b+p_d)-H(x_b)\right).
$$

The test reused the already frozen interaction coefficients, with no recalibration:

$$
\widehat\alpha_b=[1,z_b,\hat s_b,z_b\hat s_b]\,\beta.
$$

| Quantity | Relative $L_2$ | Cosine | Gate | Result |
|---|---:|---:|---:|---|
| Proxy $\hat s$ versus exact $s$ | `.2880` | `.9659` | error $\leq .50$, cosine $\geq .85$ | pass |
| Proxy coefficient versus target | `.5159` | `.8746` | error $\leq .45$ | fail |
| Exact-response coefficient versus target | `.3769` | `.9265` | audit reference | replayed |

The proxy coefficient improves by `.09346` over the native baseline, below the `.10` gate, and loses `.13896` relative to the exact-response oracle, above the `.10` degradation gate. This is a useful distinction: a compressed intermediate variable can transfer well while still being too inaccurate after a high-gain interaction.

## What is established

The missing subject-number amplitude is partly organized by a specific interaction at the L11H3 interface. The evidence is cross-construction, outcome-blind, and uses a native weight axis. The exact response variable remains donor-dependent, so this does not yet explain how a normal forward pass computes the amplitude.

The two-vector proxy shows that much of the MLP6/7 response is shared across lexical rows. It is not accurate enough for the complete coefficient program. The registered rule closes lexical- and background-specific prototype expansion; the next representation must preserve more recipient-side state in a compact vector and earn fresh causal substitution.

## Appendix: experiment details

### Dataset examples and split

Both experiments reused a frozen 32-row subject–verb-number authority with two syntactic templates, 16 lexical/attractor groups, and both number directions. Each row has recipient, opposite-number same-lemma, and same-number different-lemma endpoints. Representative rows are:

```text
Near the commander beyond the governor, the member
Near the commander beyond the governor, the members
Near the commander beyond the governor, the dealer

Beyond the commander near the governor, the member
Beyond the commander near the governor, the members
Beyond the commander near the governor, the dealer

Near the critic beyond the journalists, the neighbor
Near the critic beyond the journalists, the neighbors
Near the critic beyond the journalists, the fisherman
```

The subject is always at token position 8. Discovery used leave-one-construction-out fits between `near_beyond` and `beyond_near`. The donor-free assay trained each direction prototype on eight rows from one construction and tested the 16 rows of the other construction, then reversed the split. Every row was evaluated under all 16 E/A/U/W background subsets, giving 512 examples.

### Hooks and interventions

- MLP8 raw input at the subject position: source state $x_b$.
- Grouped MLP6/7 source: `YZ`.
- L11H3 projected write: exact function $H(x)$.
- Reader: checkpoint-native top output-projection singular axis $u$.
- Discovery arms: $H(x_b)$ and $H(x_{b,YZ})$.
- Proxy arms: $H(x_b)$, $H(x_{b,YZ})$, and $H(x_b+p_d)$.

### Fits, hyperparameters, and thresholds

The discovery performed ordinary least squares with `numpy.linalg.lstsq` and no ridge penalty, intercept regularization, feature normalization, seed, gradient, or parameter update. It made 12 fits: two construction folds plus one all-row fit for each of four forms. Selection required relative $L_2\leq .40$ and an absolute improvement of at least `.10` over `[1,z]`.

The donor-free test made zero fits. It froze each fold's interaction coefficients from discovery and constructed four direction prototypes total. Its response gates were cosine $\geq .85$ and relative $L_2\leq .50$. Its program gates were relative $L_2\leq .45$, improvement over native $\geq .10$, and degradation from exact response $\leq .10$.

Discovery used one physical model forward over 96 role sequences and 1,024 row-wise offline head evaluations. The proxy test used one physical forward, 96 role sequences, 1,536 offline head evaluations, and four stored 1,152-dimensional prototypes. Both used native float32 model states, exact registered algebra, no answer logits, no behavioral effects, no downstream causal outcomes, no backward passes, no updates, and no quantization. State and normalized-state closures were exactly zero.

The first discovery runner failed before scientific output because it read a dtype from a component dictionary. The corrected runner changed only the dtype source. The first proxy receipt was implementation-invalid because construction-sliced float32 aggregation differed from the parent metric by $3.24\times10^{-9}$ against an unnecessarily tight $10^{-10}$ audit. The corrected $10^{-8}$ audit remains far below the project closure tolerance; all scientific gates were unchanged.

### Code and primary receipts

- [Response-coordinate preregistration](../../SUBJECT_NUMBER_NATIVE_HEAD_RESPONSE_COORDINATE_DISCOVERY_V1_PREREGISTRATION.md)
- [Response-coordinate runner](../../../bilinear_quotient/ops/run_subject_number_native_head_response_coordinate_discovery_v2.py)
- [Response-coordinate result](../../../bilinear_quotient/circuits/fast_screens/subject_number_native_head_response_coordinate_discovery_v2_result.json)
- [Donor-free proxy preregistration](../../SUBJECT_NUMBER_DONOR_FREE_HEAD_RESPONSE_PROXY_V1_PREREGISTRATION.md)
- [Donor-free proxy correction](../../SUBJECT_NUMBER_DONOR_FREE_HEAD_RESPONSE_PROXY_V2_CORRECTION.md)
- [Donor-free proxy runner](../../../bilinear_quotient/ops/run_subject_number_donor_free_head_response_proxy_v2.py)
- [Donor-free proxy result](../../../bilinear_quotient/circuits/fast_screens/subject_number_donor_free_head_response_proxy_v2_result.json)
- [Computation-path registry](../../../bilinear_quotient/COMPUTATION_PATH_REGISTRY.md)
- [L11H3 module dossier](../../../bilinear_quotient/circuits/MODULE_DOSSIERS.md)
