# Three-hour mathematical circuit review — 2026-09-05 17:02 UTC

## Decision

The new Task14 evidence supports a sparse **causal interaction graph**, not a low-rank approximation. At MLP8's subject position, earlier MLP writes from layers 4--7 are the stable source family that combines with both the embedding and attention histories. Layers 0--3 are a small correction at this resolution. The next experiment should therefore split MLP4--7 while preserving the embedding--MLP and attention--MLP interactions; an isolated layer-ablation ranking would discard the computation we just identified.

This advances the controlling goal: find reusable units that predict task behavior, can be exchanged or removed selectively, and compose into multiple circuits. It does not optimize activation reconstruction, variance, parameter count, quantization, or rank.

## Exact computation

Immediately before MLP8, the raw subject state is decomposed operationally as

$$
x = E + A + U + V,
$$

where $E$ is the propagated embedding/skip contribution, $A$ is the sum of attention writes through attention layer 8, $U$ is the sum of MLP0--MLP3 writes, and $V$ is the sum of MLP4--MLP7 writes. Fixed floating-point remainders are bookkeeping terms attached to their preregistered source; they are not interpreted as features.

MLP8 first normalizes this state and then applies its bilinear map:

$$
N(x)=\operatorname{RMSNorm}(x),
$$

$$
z(x)=\bigl(LN(x)\bigr)\odot\bigl(RN(x)\bigr),
$$

$$
\operatorname{MLP8}(x)=Dz(x)+b.
$$

For every nonempty subset $S\subseteq\{E,A,U,V\}$, the experiment takes the sources in $S$ from an opposite-number donor and the remaining sources from the recipient, normalizes the resulting raw state, computes the exact MLP8 response, propagates it through the fixed layer-11-head-3 interface, and measures answer-directed margin and full-vocabulary CE. The same $2^4$ factorial is run with a different noun of the same grammatical number as the lexical control.

For any task outcome $F(S)$, its finite causal interaction for a source set $T$ is the Boolean-lattice Möbius coefficient

$$
\mu(T)=\sum_{S\subseteq T}(-1)^{|T|-|S|}F(S).
$$

This is an exact finite intervention statistic. For example,

$$
\mu(E,V)=F(EV)-F(E)-F(V)+F(\varnothing)
$$

measures whether the task effect of the layer-4--7 history changes when the embedding history is also transplanted. Because RMSNorm and the downstream model are nonlinear, these task-level interactions are not identical to coefficients of the bilinear weight tensor.

## What the two factorials establish

The coarser $E/A/M$ experiment found:

- only about 6--7% interaction in MLP8's cross response;
- about 45--49% interaction in its quadratic response;
- embedding--MLP interaction of about 23--26%;
- attention--MLP interaction of about 14--22%; and
- small embedding--attention and three-way terms.

Those same MLP-centered interactions cancel roughly 74--76% of singleton behavioral effects for plural-to-singular transfer and create roughly 66% of the singular-to-plural effect. CE and answer margin agree, so this is not an artifact of one outcome definition.

The refined $E/A/U/V$ experiment then passed its frozen late-depth prediction. Across both sentence templates, both transfer directions, and both CE and margin, $V=M4\ldots M7$ supplies at least 70% of the parent $M$, $E\!M$, and $A\!M$ aggregates, while the corresponding marginal $U=M0\ldots M3$ terms remain below 25%. The winner is $V$ in both directions. The global cross-depth prediction does not pass, but one important exception prevents us from deleting $U$: in the singular-to-plural full-response parent-$M$ aggregate, the disjoint contributions are about $-9\%$ from $U$ alone, $52\%$ from $V$ alone, and $57\%$ from the $U\times V$ interaction. Same-number lexical effects remain below 18.8% of the opposite-number scale.

The valid result is `task14_head11_3_fresh_matched_subject_mlp8_mlp_depth_source_factorial_v1_result.json`, SHA256 `429812569df68b1581f4f6632c704b8d034f65ed115c0f9f7d78ca8bb37ec817`.

## Tensor-network interpretation

If normalization were absent and the downstream measurement were linear, substituting

$$
x=E+A+U+V

$$

into the bilinear MLP would produce only singleton quadratic terms and pairwise source products. There could be no genuine third- or fourth-order source term. RMSNorm and later nonlinear layers can create higher-order causal interactions, but the measured higher-order terms are small and cross-depth $U$--$V$ composition fails the registered bar.

That suggests a useful local model: a sparse factor graph whose important edges are centered on $V$,

$$
E\longleftrightarrow V,
\qquad
A\longleftrightarrow V,
$$

with $U$ usually a smaller marginal source but still retained as a live context because of the singular-to-plural $U\times V$ exception. This is not yet a semantic decomposition. It becomes one only if finer parts of $V$ retain the same causal predictions, transfer beyond the current sentences, and can be removed without damaging unrelated circuits.

## Alternative mathematical paths

Three routes remain live and test different notions of “same computation”:

1. **Recursive finite interaction split.** Split $V$ into MLP4--5 and MLP6--7 while retaining $E$, $A$, and $U$ as live factors. This is the immediate route because it exactly refines the causal object that passed without erasing the exceptional $U\times V$ effect.
2. **Weight contraction after causal localization.** Once the responsible source pair is small, contract its donor differences through $L$, $R$, and $D$ to identify which bilinear functions implement the causal interaction. This translates the intervention result into weights without choosing product coordinates as the scientific basis.
3. **Downstream-equivalence quotient.** Treat two MLP8 output directions as equivalent when every registered downstream circuit responds to them the same way. The growing circuit corpus can define this quotient empirically and may merge different native coordinates that implement the same reusable computation.

The second and third routes should not replace the finite causal split yet: without the source localization they would again optimize a geometric proxy rather than the circuit's task behavior.

## Immediate continuation and falsifiers

The next split is the complete $2^5$ factorial over $E$, $A$, $U$, $W=\mathrm{MLP4}+\mathrm{MLP5}$, and $X=\mathrm{MLP6}+\mathrm{MLP7}$. It must retain CE, answer margin, all parent-regrouping closure checks, and the same-number lexical control. It should recurse to individual MLPs only if one half dominates or if the interaction between halves is small; otherwise the pair remains the causal unit and must be split conditionally. The frozen computational price should be 4 forwards, 6,080 example evaluations, 2,976 interventions, and no gradients or parameter updates.

The current conclusion is falsified or redirected if the layer-4--7 arm is not independently task-live, if its apparent dominance fails either metric or direction, if a same-number lexical donor produces a comparable effect, or if finer halves interact so strongly that neither can be manipulated separately. The downstream-weight contraction opens only after a finer causal source survives these checks.
