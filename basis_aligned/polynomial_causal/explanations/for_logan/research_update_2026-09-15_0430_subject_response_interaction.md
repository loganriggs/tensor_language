# September 15 research update: subject response and a compact regional interaction path

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

A frozen rank-2 version of that recipient-state idea has now also returned a null.
It predicts two grouped MLP6/7 displacement modes from two recipient-state PCs, but
improves response error only `.28804→.28722` and coefficient error only
`.51585→.51072`. This closes ordinary activation/displacement PCA on these opened
rows. A future circuit coordinate must be selected by the head-response operator,
not by activation energy.

The alternating weight-folding track has also produced a clean result. In the
regional UK/US spelling path, the dominant QK1 input to attention head 9.8 can be
written as 289 exact query-source × key-source interactions. No individual pair
dominates. But four late sources—attention5 and MLP5/6/7—form a useful group.
Their self-interaction has change-norm ratio `.4548`; interactions crossing between
this group and the other 13 sources jointly reach `.4409`. Keeping the three blocks
that touch the late group reproduces the complete carry×carry path with `.1119`
relative error. This is a compact attribution of the selected path, and the next
test is a fresh causal routing intervention.

![Prediction errors for the response interaction and donor-free proxy](assets/research_update_2026-09-15_subject_response_interaction.png)

*Figure 1. Lower is better. Left: only the exact multiplicative response model passes the `.40` coefficient-error ceiling. Right: the two-vector proxy passes its response-scalar gate but fails after composition into the coefficient program.*

![Four exact late/remainder QK1 interaction blocks](assets/research_update_2026-09-15_regional_late_group.png)

*Figure 2. Change norm of each exact QK1 block relative to the carry×carry parent. The blocks are correlated, so these ratios need not sum to one. The late self-block leads, and the two ordered cross-boundary blocks are both substantial.*

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

## Rank-2 recipient-state test

The next assay retained the grouped MLP6/7 source but added two recipient-state
coordinates. On each training construction and answer direction, it took the top
two principal directions $V_x$ of the recipient MLP8 input and the top two directions
$V_\delta$ of the grouped source displacement. A fixed $2\times2$ ridge map predicted
displacement scores:

$$
r_b=(x_b-\bar x)V_x,
\qquad
\widehat\delta_b=\bar\delta+r_bA V_\delta^\top.
$$

The four fold/direction maps retained `.366–.432` of recipient-state energy and
`.491–.539` of displacement energy. Despite that extra state, the response error
fell by only `.00083`, far below the preregistered `.05` improvement. The complete
coefficient error remained `.51072`; its native-baseline improvement was `.09859`
and its oracle degradation was `.13383`, missing the `.10` gates on both sides.
The result argues against choosing the next basis by activation variance.

## Regional head9.8 interaction-path fold

For the separate UK/US spelling path, let $c$ be the exact residual carry entering
block 9 after learned residual coefficients are folded in. It decomposes into the
embedding and 16 attention/MLP writes from layers 0–7:

$$
c=\sum_{i=1}^{17}c_i.
$$

Head 9.8's first routing score is bilinear in its query and key inputs. Holding
the native RMS denominator fixed therefore gives an exact expansion

$$
B(c,c)=\sum_{i=1}^{17}\sum_{j=1}^{17}B(c_i,c_j).
$$

The 289-term census was distributed: its ten largest terms still had `.66738`
relative replay error, and its largest term, MLP6-query × attention5-key, was
only `.04839` of the parent change norm. We then froze

$$
D=A_5+M_5+M_6+M_7,
\qquad
R=c-D,
$$

and evaluated the exact four-block identity

$$
B(c,c)=B(D,D)+B(D,R)+B(R,D)+B(R,R).
$$

| Ordered block | Change-norm ratio | Aligned fraction | Family range |
|---|---:|---:|---:|
| $D\times D$ | `.45480` | `.45284` | `.42180–.48700` |
| $D\times R$ | `.25335` | `.25227` | `.23494–.27360` |
| $R\times D$ | `.19041` | `.18844` | `.15197–.20647` |
| $R\times R$ | `.11186` | `.10645` | `.09656–.12851` |

Here **change-norm ratio** means the $L_2$ norm of one block's paired UK/US
change vector divided by the corresponding norm for the complete carry×carry
parent. **Aligned fraction** is its dot product with the parent change, divided
by the parent's squared norm. Since the four vectors can reinforce or cancel,
the norm ratios are interaction magnitudes rather than additive percentages.

All preregistered descriptive predictions passed. $D\times D$ leads globally and
within every family. The two cross-boundary terms have joint change-norm ratio
`.44088`, and

$$
B(D,D)+B(D,R)+B(R,D)
$$

replays the parent with relative error `.11186`. Carry reconstruction error was
$8.39\times10^{-8}$; QK1 score closure and downstream folded closure were
$1.51\times10^{-16}$ and $1.07\times10^{-16}$. These results show that the
computation is compact at grouped interaction grain, despite being diffuse at
individual module-pair grain. They do not yet show causal sufficiency.

## What is established

The missing subject-number amplitude is partly organized by a specific interaction at the L11H3 interface. The evidence is cross-construction, outcome-blind, and uses a native weight axis. The exact response variable remains donor-dependent, so this does not yet explain how a normal forward pass computes the amplitude.

The two-vector proxy shows that much of the MLP6/7 response is shared across lexical rows. It is not accurate enough for the complete coefficient program. Rank-2 activation/displacement PCA adds negligible useful information. The registered rules close lexical/background prototype expansion and activation-rank sweeps on this authority; the next representation must use response-oriented coordinates and earn fresh causal substitution.

For the regional path, exact folding now connects the output reader backward
through MLP17, MLP16, attention9 head 9.8, QK1, and the layer-0–7 carry sources.
The late group is a compact interaction handle. A selective routing edit on fresh
rows is required before treating those three late-touching blocks as a circuit.

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

The rank-2 assay fit four upstream predictive maps, one per construction fold and
answer direction. Each used 128 training background examples, two recipient PCs,
two displacement PCs, and ridge

$$
\lambda=10^{-3}\frac{\operatorname{tr}(R^\top R)}{2}.
$$

There was no rank sweep and no coefficient refit. It used one physical forward,
96 role sequences, and 2,048 offline head-function evaluations.

Discovery used one physical model forward over 96 role sequences and 1,024 row-wise offline head evaluations. The proxy test used one physical forward, 96 role sequences, 1,536 offline head evaluations, and four stored 1,152-dimensional prototypes. Both used native float32 model states, exact registered algebra, no answer logits, no behavioral effects, no downstream causal outcomes, no backward passes, no updates, and no quantization. State and normalized-state closures were exactly zero.

The first discovery runner failed before scientific output because it read a dtype from a component dictionary. The corrected runner changed only the dtype source. The first proxy receipt was implementation-invalid because construction-sliced float32 aggregation differed from the parent metric by $3.24\times10^{-9}$ against an unnecessarily tight $10^{-10}$ audit. The corrected $10^{-8}$ audit remains far below the project closure tolerance; all scientific gates were unchanged.

### Regional-fold dataset, computation, and price

The regional experiment reused 96 frozen sequences: 48 matched UK/US pairs over
four prompt families (`distant_note`, `near_message`, `near_person`, and
`old_near_anchor`). Each pair differs at the registered regional cue while the
answer reader is the unembedding-row difference for the matched UK and US answer
tokens. The 96 rows were processed in 14 length-bucketed physical forwards.

The runner reconstructed 17 native carry sources, projected the fixed late and
remainder sums through head 9.8's QK1 weights, retained QK2 and the value stream
at their native values, and folded each resulting 128-dimensional head output
through the attention output projection, residual coefficients, the exact
MLP16 × MLP17 bilinear term, and the row-specific unembedding reader. It used
float64 for the offline contractions, zero fits, zero backward passes, zero
parameter updates, and no hyperparameter or group search. The late group and all
thresholds were frozen from the preceding 289-term census.

### Code and primary receipts

- [Response-coordinate preregistration](../../SUBJECT_NUMBER_NATIVE_HEAD_RESPONSE_COORDINATE_DISCOVERY_V1_PREREGISTRATION.md)
- [Response-coordinate runner](../../../bilinear_quotient/ops/run_subject_number_native_head_response_coordinate_discovery_v2.py)
- [Response-coordinate result](../../../bilinear_quotient/circuits/fast_screens/subject_number_native_head_response_coordinate_discovery_v2_result.json)
- [Donor-free proxy preregistration](../../SUBJECT_NUMBER_DONOR_FREE_HEAD_RESPONSE_PROXY_V1_PREREGISTRATION.md)
- [Donor-free proxy correction](../../SUBJECT_NUMBER_DONOR_FREE_HEAD_RESPONSE_PROXY_V2_CORRECTION.md)
- [Donor-free proxy runner](../../../bilinear_quotient/ops/run_subject_number_donor_free_head_response_proxy_v2.py)
- [Donor-free proxy result](../../../bilinear_quotient/circuits/fast_screens/subject_number_donor_free_head_response_proxy_v2_result.json)
- [Rank-2 recipient-state preregistration](../../SUBJECT_NUMBER_RANK2_RECIPIENT_STATE_RESPONSE_PROXY_V1_PREREGISTRATION.md)
- [Rank-2 recipient-state result](../../../bilinear_quotient/circuits/fast_screens/subject_number_rank2_recipient_state_response_proxy_v2_result.json)
- [Computation-path registry](../../../bilinear_quotient/COMPUTATION_PATH_REGISTRY.md)
- [L11H3 module dossier](../../../bilinear_quotient/circuits/MODULE_DOSSIERS.md)
- [Carry-source census result](../../../bilinear_quotient/circuits/fast_screens/setting2_regional_head9_8_qk1_carry_source_fold_v1_result.json)
- [Late-group preregistration](../../SETTING2_REGIONAL_HEAD9_8_QK1_LATE_GROUP_FOLD_V1_PREREGISTRATION.md)
- [Late-group runner](../../../bilinear_quotient/ops/run_setting2_regional_head9_8_qk1_late_group_fold_v1.py)
- [Late-group result](../../../bilinear_quotient/circuits/fast_screens/setting2_regional_head9_8_qk1_late_group_fold_v1_result.json)
