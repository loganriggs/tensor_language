# Research update: predicting composed circuit interactions

**12 September 2026. Results checked at 18:55 UTC, with a subsequent CPU loss/readout audit during preparation. Requested full update for Logan.**

This covers Codex's work since the [17:30 full update](research_update_2026-09-12_1730_shared_producer_interactions.md). It does not count that report's discoveries again or include Claude's separate candidate campaign. Head8.2 means head 2 in block 8, using zero-based numbering.

## High-level overview

**The interaction-path direction has made concrete progress: we now have a weight-derived formula for how one extracted attention component changes another through the intervening bilinear MLP. It predicts the measured interaction on fresh prompts and under donor swaps.** We also found a clearer division between the input sources used by the two components.

The main limitation is now equally concrete: **the components are useful for regional spelling, but removing them at their actual sources also damages newline prediction.** Splitting their value inputs helped explain their roles but did not produce a subset passing both behavior and preservation criteria.

The sequence of events was:

1. **We established a valid newline control.** The old authored prompts were unsuitable for demonstrating the model's newline service. Natural FineWeb prefixes passed capability and whole-head intervention controls. Removing only the regional consumer edges preserved that service.
2. **We moved the interventions upstream to the actual producers.** This exposed much larger regional effects—roughly 67–70% of the native cue contrast on the first panel—but also nonlinear composition and a real newline-preservation failure.
3. **We derived and executed the missing interaction through MLP8.** A direct write alone misses about a third of the serial effect. Including its mixed interaction with the MLP predicts that effect to approximately 0.4–1.1% error on the original panel.
4. **We tested fresh contexts and a different manipulation.** On 48 new email/letter prompts, source removal retains 65–70% regional coverage and the interaction approximation has less than 1% error. Donor swaps then move the behavior in the expected direction on every tested row; the same formula predicts their serial effects within 1.4–2.3%.
5. **We checked the attention computation more carefully.** Complete QK1 × QK2 routing must generally change along with the values. A value-only account fails one template family substantially.
6. **We split each value computation into current-state and first-state contributions.** Head8.2's first-state contribution and head9.8's current-state contribution carry most of the regional effect. We evaluated all 15 nonempty subsets of the four contributions. None meets both the registered regional-coverage and newline-preservation criteria.
7. **We explained the latest selectivity miss.** On the preserved worst newline case, the two smaller collateral effects mostly add. A large interaction is not what pushes this case over the limit.

This is progress on a **particular composed computation**, rather than a new successful global decomposition of the whole unembedding. We have not yet obtained a general unsupervised method from which many independently executable circuits simply fall out.

## What changed relative to the previous report?

The previous report's roughly **8%** whole-cue coverage and this report's roughly **70%** are different interventions, not a sudden improvement in the same factorization.

- **Consumer-edge removal:** remove a producer's contribution only where it enters the selected downstream regional computation. Other uses of that producer remain available.
- **Physical source removal:** subtract the component's actual residual-stream write at its producer and recompute every subsequent layer. This changes all downstream uses, including its influence on the second producer.

The source intervention is more powerful and less selective. That difference is central to the idea that interaction paths, rather than whole modules or even whole producer components, may be the right units.

For a regional prompt, the score margin is the model's British-spelling score minus its American-spelling score. The **cue contrast** compares that margin between matched British-city and American-city prompts. Coverage here is

$$
\mathrm{coverage}
=\frac{\mathrm{native\ cue\ contrast}-\mathrm{removed\ cue\ contrast}}
{\mathrm{native\ cue\ contrast}}.
$$

It measures how much of this controlled regional effect the intervention removes. It is not a percentage of overall language ability, nor proof that the component represents only regional spelling.

[Physical-removal result](../../SCALAR_PRODUCERS_RECURSIVE_REMOVAL_V1_RESULT.json).

## The recovered composed computation

The two frozen producers each compute a scalar attention output and write it along a fixed residual direction. Schematically,

$$
\text{head8.2 component}
\longrightarrow \text{residual + normalized bilinear MLP8}
\longrightarrow \text{head9.8 input and routing}
\longrightarrow \text{shared regional consumer}.
$$

The original grouping arose because the two components have almost the same writer in the downstream consumer's four-reading space. Their physical residual writers are less similar: cosine approximately 0.899, versus 0.99949 downstream. Consequently, a common downstream interface does not mean identical effects everywhere else in the model.

### Folding the producer direction through the bilinear layer

Let $z\in\mathbb R^{1152}$ be the residual entering MLP8, $d\in\mathbb R^{1152}$ the fixed physical writer of the head8.2 component, and $a$ its scalar amplitude at this position. Removal changes the input to

$$
z'=z-ad.
$$

Write the bias-free bilinear map as

$$
M(z)=D[(Lz)\odot(Rz)],
$$

where $L,R$ map 1152 residual coordinates into 4608 product coordinates, $D$ maps back, and $\odot$ is elementwise multiplication. The actual model normalizes before the MLP. With

$$
\rho^2=\operatorname{mean}(z^2)+\epsilon,
\qquad
\rho'^2=\operatorname{mean}((z-ad)^2)+\epsilon,
$$

its bias-free MLP output is $M(z)/\rho^2$. The fixed MLP bias cancels in the difference.

Expanding the bilinear product gives an exact finite change:

$$
\begin{aligned}
\Delta x_8={}&-ad
+\left(\frac1{\rho'^2}-\frac1{\rho^2}\right)M(z)\\
&-\frac a{\rho'^2}D[(Lz)\odot(Rd)+(Ld)\odot(Rz)]\\
&+\frac{a^2}{\rho'^2}D[(Ld)\odot(Rd)].
\end{aligned}
$$

These are four explicit paths: the direct removed write; the old MLP background rescaled by the changed normalization; the mixed product of the producer direction and background; and the producer's self-product. Both cross terms are required. This is the principled symmetrization for a bilinear layer evaluated on the same residual in its two input slots.

There is a useful shared intermediate. Define the constant matrix

$$
J_d=D[\operatorname{diag}(Rd)L+\operatorname{diag}(Ld)R].
$$

It maps any $z$ to the mixed term for this fixed producer direction. It is **not** a Jacobian frozen at one observed prompt. Moreover,

$$
D[(Ld)\odot(Rd)]=\frac12J_dd.
$$

Thus, supplying the original bias-free MLP output $u=M(z)/\rho^2$, the entire response becomes

$$
\boxed{\Delta x_8=-ad+
\left(\frac{\rho^2}{\rho'^2}-1\right)u
-\frac a{\rho'^2}J_d\left(z-\frac a2d\right).}
$$

The next block's learned residual coefficient multiplies this change; its actual input normalization, projected QK normalization and positional operations are then recomputed.

This is a concrete instance of your proposed folding: **a producer direction composed with a bilinear layer admits a simpler conditional operation than retaining every MLP product separately.** No text-fitting or nonlinear optimization was needed for this identity.

The matrix and direction contain **1,328,256 scalars**, versus 15,926,400 in the original full MLP. That is a conditional representation price, not a whole-model compression claim: $z$, $u$, the amplitude, the other producer's background and the remaining model still need to be generated. The dense matrix itself remains substantial unexplained numerical content.

[Derivation and native evidence](../../FOLDED_PRODUCER_NATIVE_V1_MATH.md) · [Exact weight control](../../SCALAR_PRODUCER_DIRECTIONAL_MLP_V1_RESULT.json) · [Executable response](../../directional_mlp_bridge_v1.py).

## Does the formula predict behavior, or just algebra?

We tested both. Native residual-change replay is within roughly $1.4\times10^{-5}$ relative error, and the resulting scalar head9.8 change within $8\times10^{-7}$. These are implementation fidelity checks against the real model.

The more consequential test compares **signed token-margin effects**. The serial reference is the difference between recomputing the second removed component after the first removal and freezing that component at its pristine value. This isolates a diagnostic of serial dependence. The frozen case is a hybrid counterfactual, not our adopted intervention.

Errors below are relative vector errors on that serial effect, not on the much larger baseline logits:

| Prediction | Original 48 prompts, two families | Fresh 48 email/letter prompts | Donor swaps on email/letter prompts |
|---|---:|---:|---:|
| Exact response | Below 0.001% | Below 0.001% | Below 0.001% |
| Direct write + mixed MLP term | 1.14% / 0.38% | 0.87% / 0.93% | 1.43% / 2.33% |
| Direct write alone | 33.7% / 33.7% | Not tested in this panel | Not tested in this panel |

“Direct + mixed” drops the changed-normalizer background term and the explicit self-product from the response formula; actual subsequent normalization is still computed. It is an approximation supported on these panels, not an exact global simplification. Original removal amplitudes were small: median perturbation norm about 0.57% of the native residual norm, maximum 2.06%. Donor swaps reached 5.68%.

The math cycle therefore helped materially: explicit normalization and bilinear expansion led to a reusable computation that predicts a measured interaction. The best result came from changing the mathematical object to a **directional composed response**, rather than adding another low-rank fit.

[Original behavioral test](../../SCALAR_PRODUCER_MLP_BRIDGE_EFFECT_V1_RESULT.json) · [Fresh-context test](../../SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_RESULT.json) · [Donor interchange](../../SCALAR_PRODUCERS_INTERCHANGE_V1_RESULT.json).

## Fresh generalization and donor interchange

The fresh panel changed templates to emails and letters, cities to Glasgow/Phoenix and Cambridge/Detroit, and spelling pairs to neighbours/neighbors, organise/organize, realise/realize, labelled/labeled, defence/defense and metre/meter. Factors and the directional map were frozen before evaluation. There are 48 prompts, organized into 24 matched cue pairs, not 48 independent concepts.

The native cue contrasts averaged 2.646 and 2.971. Physical pair removal reduced them by **70.5% and 65.1%**. Each individual and joint removal moved all 24 paired contrasts in the expected direction. All four template/city strata had mean coverage between 64.2% and 72.2%; no subgroup was dropped.

For **interchange**, we replaced a recipient's scalar component with the matched opposite-city donor's scalar field. We tested either producer alone and both together. This is stronger evidence of manipulability than merely zeroing a component. The formula accepts a signed amplitude,

$$
a=a_{8,\mathrm{recipient}}-a_{8,\mathrm{donor}},
$$

so the same response code applies without a new fit.

Every individual and joint swap moved all 48 directed rows toward the donor's spelling preference. Joint transfer was **72.4% and 67.7%** of the native cue contrast. Self-donor swaps were exactly unchanged. The small unrelated-margin control passed, but that does not override the separate newline failure.

Simply adding the two individual swap effects differs from the directly evaluated joint effect by **43.5% and 45.7%**. The successful composed prediction therefore matters: the parts are not independent additive contributions.

These results support controlled lexical/template generalization and a new manipulation on those contexts. They are not broad corpus OOD prediction or evidence that the prompts were absent from training.

## What happened with QK1 × QK2?

We kept both QK factors together throughout. For a head, let $\gamma$ be its complete position-aware product routing operator and $v$ its scalar value field. The output is $a=\gamma v$. When the preceding component changes, both can change:

$$
\Delta a=\gamma_0\Delta v+\Delta\gamma\,v_0+\Delta\gamma\,\Delta v.
$$

The three terms mean changed values under old routing, changed routing over old values, and their interaction. We independently evaluated all terms. An initial check constructed the mixed term by subtraction, making the identity check tautological; the corrected check contracts $\Delta\gamma\Delta v$ directly and preserves the same numerical verdict.

A value-only account has **35.1% / 6.2% error** on the signed serial behavior in the two original template families. Routing-only has **139.5% / 102.0% error**. The contributions can oppose each other, so their separate norms are not percentages that sum to 100%.

This supports retaining joint routing. It has **not yet** identified separate regional and newline subspaces inside the full QK1 × QK2 product. The newer split below concerns value sources while preserving complete routing; it should not be mistaken for that more ambitious QK decomposition.

[Independent contraction audit](../../SCALAR_PRODUCER_JOINT_QK_VALUE_V2_RESULT.json) · [Behavioral test](../../SCALAR_PRODUCER_JOINT_QK_VALUE_EFFECT_V1_RESULT.json).

## Selectivity: a valid negative, followed by a useful split

The natural newline control uses 32 FineWeb prefixes whose true next token is newline. Selection used dataset labels rather than model scores. The model predicts newline well, and replacing whole-head8.2 output with its mean damages newline prediction in both fixed halves. FineWeb is the training domain; this is not a new-corpus test.

Consumer-edge removal changes newline cross-entropy by at most about **0.0021**, passing the registered preservation limit. Physical source-pair removal reaches **0.1665**, failing the maximum permitted absolute change of **0.1**, although both halves pass the mean-absolute-change limit of 0.02.

Cross-entropy is $-\log p(\text{correct token})$; an increase is damage. In the preserved worst paragraph, native newline probability falls from approximately **0.741 to 0.627**. This is a real capable example, not the invalid authored control. Replacing the paragraph's country with five other countries leaves five of six variants over the limit; the registered country-dependence prediction also misses. The source example remains included.

### Splitting the values by input source

Each scalar producer naturally separates as

$$
a_i=\gamma_i v_{i,\mathrm{current}}+\gamma_i v_{i,\mathrm{first}}.
$$

“Current” reads the contextual state at that layer. “First” uses the model's shared first-state value source, compiled as a token lookup. The first-state contribution still has contextual routing through $\gamma_i$; it is not a context-free token effect.

Two producers give four removable contributions. We evaluated all $2^4=16$ masks, including native, with later layers recomputed. Both source sectors of a producer use the same physical writer and the same complete joint QK routing.

| Removed contributions | Regional coverage, two families | Maximum absolute newline CE change across the panel |
|---|---:|---:|
| Head8.2 current only | 3.8% / 4.2% | 0.0191 |
| Head8.2 first only | 35.5% / 37.1% | 0.0602 |
| Head9.8 current only | 52.9% / 42.8% | 0.0767 |
| Head9.8 first only | Approximately 0% / 0.9% | 0.0036 |
| Head8.2 first + head9.8 current | **67.8% / 61.0%** | **0.1371** |
| All four | 70.5% / 65.1% | 0.1665 |

The desired screen required at least 50% regional coverage **in each family**, plus direction and unrelated-margin controls, together with newline preservation. Four subsets passed the regional criteria. **None passed both sets of criteria.** Head9.8-current alone misses the regional bar in the second family; we do not lower it after seeing the result.

The split gives a more explicit computational description: the first-state value branch of one producer and the contextual value branch of the other dominate this regional effect. It does not yet separate the regional and newline services. This is exploratory selection over 15 subsets and would need fresh confirmation even if a joint passer had appeared.

[All-mask factorial result](../../SCALAR_VALUE_SECTOR_FACTORIAL_V1_RESULT.json) · [Original preservation failure](../../SCALAR_PRODUCERS_RECURSIVE_REMOVAL_V1_RESULT.json) · [Country audit](../../SCALAR_PRODUCERS_NEWLINE_COUNTRY_AUDIT_V1_RESULT.json).

### Is the remaining newline failure caused by interaction?

For a measured change $E(S)$ after removing subset $S$, define the two-part finite interaction as

$$
I_{AB}=E(\{A,B\})-E(\{A\})-E(\{B\}),\qquad E(\varnothing)=0.
$$

The full Boolean expansion, also called a Möbius transform, applies this subtraction to every subset. It exactly reconstructs the 16 observed interventions. This is bookkeeping for finite interventions, not a unique allocation of causal ownership.

For $A=$ head8.2-first and $B=$ head9.8-current, the worst newline case is

$$
\underbrace{0.137089}_{\mathrm{joint\ damage}}
=\underbrace{0.060216}_{A}
+\underbrace{0.076724}_{B}
+\underbrace{0.000150}_{\mathrm{interaction}}.
$$

The interaction is only **0.11%** of joint CE damage on this example. A further CPU audit checked the newline-minus-comma score margin: its interaction is **2.66%** of the joint margin change. Thus the mostly additive collateral is not solely a peculiarity of the CE readout on this case.

The regional effects behave differently. Their interaction norm is about **30%** of the joint effect, and the mean interaction is negative: their individual regional effects partly overlap. We should not expect that finding and deleting a large newline-specific interaction term will automatically repair selectivity. A more precise consumer/path boundary or further joint-routing split is the better hypothesis to test.

[Möbius result](../../SCALAR_SECTOR_MOBIUS_V1_RESULT.json) · [Loss/readout audit](../../SCALAR_SECTOR_LOSS_SPACE_AUDIT_V1_RESULT.json).

## Status against the project goal

| Requirement | Evidence now | Remaining gap |
|---|---|---|
| OOD prediction | Frozen components and composed response transfer to new controlled vocabulary/templates. | Broad natural-text and corpus-shift prediction; independently generated inputs. |
| Extraction | Executable scalar producers plus an exact directional MLP response; strong native replay. | Native contextual states, background, full QK maps and suffix remain dependencies. |
| Selective removal/interchange | Large directed regional effects; donor swaps and self-controls pass. | Physical source removal fails valid newline preservation. No sector subset repairs it. |
| Composition and reuse | Shared downstream writer; explicit cross-module response predicts nonadditive removal and swap effects. | Multiple independently extracted tasks sharing a closed program; robust joint selective manipulation. |
| Structural simplicity | One reusable directional map replaces an explicit list of MLP response products conditionally. | Dense arbitrary weights and all input/background adapters remain charged; no general minimal DAG. |

The latest interval used exact weight algebra and validation of frozen components. It did not add a new global CP/Tucker fit, fit the factors to a million-token corpus, or solve general sparse hierarchy discovery. The original full-unembedding and deeper-path proposal remains broader than this demonstrated motif.

The most useful next question is whether we can isolate **which downstream uses of the two value sectors** carry regional spelling and which carry newline prediction, retaining their joint QK computation. The current source-sector screen rules out only a coarse four-piece split, not the existence of finer structure. Any finer candidate must retain the failed examples and be confirmed on new data rather than optimized to erase this particular outlier.

## Execution and evidence notes

The GPU experiments were managed through lane1 `bqrunner`. The major new native tests used roughly 1–17 seconds of timed model-forward execution each; these timings exclude loading, queue waits, CPU derivations, implementation and reporting. The largest sector screen ran 1,312 body forwards in 17.4 seconds. Raw GPU execution was not the principal bottleneck in this interval.

At the 18:55 process check, both managed runners were running; lane1 was executing Claude's separate v665 job. The decomposition jobs described above had finished. This report does not imply that a further decomposition GPU job was running. The additional loss/readout audit was actually performed on CPU during report preparation.

The [primary running math note](../../FOLDED_PRODUCER_NATIVE_V1_MATH.md) links the durable implementation and experiment receipts. The existing [producer package](../../extracted_circuits/regional_shared_producers_8_2_9_8_v1/README.md) documents the conditional interface. New component findings also belong in the existing specialist-head and MLP dossiers; head8.2's newline service was prior knowledge, not a new discovery claimed here.
