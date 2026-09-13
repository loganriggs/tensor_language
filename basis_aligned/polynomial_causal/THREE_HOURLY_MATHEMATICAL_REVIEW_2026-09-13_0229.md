# Mathematical review: joint attention products and inherited mixed inputs

13 September2026,02:29UTC review. Previous review23:28:50; next due05:29 at the first safe boundary. This review follows the corrected bilinear handoff/pilot and the four-property goal, not the stale better-math objective text.

## Current evidence and precise object

The fixed head17.2 hypothesis passes: this known regional head's direct mixed write predicts the full attention17 direct mixed effect within1.25–1.62%across four regional constructions. Joint head writes match the independent preMLP-state reference within0.128%regional and0.417%FineWeb; native output anchors replay exactly. Separate head effects add within0.56–1.03%regional. This reconnects a path to a known head, not necessarily the same within-head source component. FineWeb head2errors37.8–73.5%are descriptive and prevent a universal singlehead statement.

For final target position t and head h=2, define two actual normalized rotated QK score vectors a,b in R^T and mixed values V in R^(T x128):

$$
a_s=\langle q_{1,t},k_{1,s}\rangle/128,\qquad
b_s=\langle q_{2,t},k_{2,s}\rangle/128,
\qquad y_t=\sum_{s\le t}a_sb_sV_s.
$$

W_h in R^(1152x128) maps y_t into residual space. Q/K projection matrices are128x1152; values use the current normalized state and shared first-layer value with native signed mixture. Input-state width1152, nine heads, source count<=248 in the current test. All four corner executions share x0 and first-layer values; current states, QKnormalizers and attention-inputRMS differ.

The contraction graph is a source-index star linking score1,score2,value, then an output projection. It is trilinear in these declared ports. Before normalization, in independent query/source variables, its numerator has query degree2 and source degree3 (total5, treating the first-value input as an additional linear source port). The actual model is not a degree5polynomial: RMS includes square roots, shared-state dependence matters, and the final readout includes tanh. Fixed native RoPE rounding is part of the contract.

Target: predict the finite mixed output yA-yC-yR+yN from explicit local joint products and inherited port nonadditivity. N is native, C/R are individual path removals, A is the additive incoming-state execution. This is not the original joint-removal P trajectory. Approximation is initially zero in FP64 algebra, then native relative write/effect error plus absolute tiny-control error. No text-weighted coefficient fitting.

## Literature match and limits

Kuo, Sloan, Wasilkowski and Wozniakowski formulate multivariate decompositions using commuting variable-eliminating projections. Fixing selected inputs to anchors gives a unique expansion under the associated vanishing conditions; orthogonality requires additional structure. Here each projection fixes one declared attention port to its chosen reference. The free-port domain is closed under these substitutions, so the operator identity applies componentwise. Native reachable states need not be closed under those substitutions, so this algebra does not establish on-manifold causal identifiability. The generic three-port anchored expansion needs eight corner evaluations; a general d-port expansion can need2^d. We exploit the known product rather than treating the model as a black box. [Primary paper](https://web.maths.unsw.edu.au/~fkuo/pubs/preprint/ksww09-decomp.pdf).

The concrete finite-edit expansion below is derived directly for this product and its four observed port states. It is not a claim that the paper already identifies our circuit or supplies a globally canonical semantic basis.

Tensor-train methods map a finite coefficient array to a chain whose attainable bond ranks are governed by unfoldings, with SVD-based approximation. Here the three-port source contraction is already cheap and exact. A TT of the expanded four-state product table could compress its bookkeeping, but does not identify which normalized input generator matters. Dense TT-SVD also requires access to the finite array; its cost depends on unfolding sizes and is not a solution for the full normalized network. Basis/gauge nonuniqueness prevents semantic naming from cores alone. No new TT fit is warranted. [Oseledets](https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf).

Weighted-automaton realization instead targets functions on token strings through Hankel prefix/suffix matrices and finite linear state updates. It would address autonomous input generation if a small closed realization were established. No such closure or finite-rank certificate exists for these normalized, position-dependent transformer states. Building a sampled Hankel matrix now would introduce data fitting and a different expensive target without answering the local product question. The bounded contraction below has greater immediate circuit value. [Balle and Mohri](https://cs.nyu.edu/~mohri/pub/cai.pdf).

## Derived exact decomposition

For each port x in {a,b,V}, define

$$
c_x=x_C-x_N,\qquad r_x=x_R-x_N,\qquad
\bar x=x_N+c_x+r_x,\qquad e_x=x_A-\bar x.
$$

The e terms are inherited mixed changes inside the port generators. They need not vanish: normalized QK scores are nonlinear even when the incoming residual state is additive.

Let P(a,b,V)=sum_s a_s b_s V_s. The desired output mixed change is

$$
I=P(a_A,b_A,V_A)-P(a_C,b_C,V_C)-P(a_R,b_R,V_R)+P(a_N,b_N,V_N).
$$

It splits into

$$
I=I_{\mathrm{cross}}+I_{\mathrm{defect}},
$$

$$
I_{\mathrm{cross}}=P(\bar a,\bar b,\bar V)-P(a_C,b_C,V_C)-P(a_R,b_R,V_R)+P(a_N,b_N,V_N),
$$

$$
I_{\mathrm{defect}}=P(\bar a+e_a,\bar b+e_b,\bar V+e_V)-P(\bar a,\bar b,\bar V).
$$

Icross contains exactly12monomials: choose one of native,childchange,remainderchange in each of the three factors, retaining only choices containing both childchange and remainderchange. Idefect has seven grouped terms indexed by nonempty subsets of the three defect ports; use e on the selected subset and additive bars elsewhere. This groups the37 fully expanded monomials containing at least one inherited mixed defect. Twelve plus37 gives49 surviving terms out of the original4^3 expansion; neither input's pure effect is counted twice.

The implementation sums small differences directly instead of subtracting four large completed products. Both QK factors remain jointly multiplied. Swapping their labels or exchanging child and remainder leaves the total invariant. Reciprocal score scaling changes coordinates but preserves the product; grouping under other arbitrary reparameterizations is not a semantic uniqueness guarantee. No assumption assigns one task to QK1 and another to QK2.

## Executed consequence and falsifier

`joint_attention_mixed_ports_v1.py` implements the12cross monomials and7defect groups for source-summed128dimensional values. Its CPUcontrol has five synthetic contexts and13sources. Replayerror is3.90e-15, child/remainder exchange2.05e-16, QKexchange1.21e-16.

The planted falsifier makes only score1 change under the child and only score2 change under the remainder. Every port's own mixed defect isexactlyzero, yet the joint response has norm1.8176 and is recovered within6.68e-16. Thus inspecting independent port mixed changes alone can miss the entire interaction. This directly supports keeping the joint QK/value product as the unit of analysis.

Price:19source contractions, eachO(T*128) at the final target, plus O(T*128) port differences and the existing128x1152 output projection. No fitting iterations, full tensor or extra learned coefficients. N/C/R/A port generation and native background still cost model execution. Processing all targets would costO(T^2*128); current final-block readout needs only the final target. Streaming terms avoids retaining19fullsourcearrays.

[Executed control](JOINT_ATTENTION_MIXED_PORTS_V1_CONTROL.json), [kernel](joint_attention_mixed_ports_v1.py), [control script](joint_attention_mixed_ports_control_v1.py).

## Decision

This mathematical route beats another rank/sparsity search for the immediate circuit question. It gives an exact, cheap, symmetric decomposition with two opposing hypotheses: most mixed write is generated by products of otherwise additive port changes, or most is inherited from nonlinear port generation. Next capture native head17.2 score1/score2/value corners, replay the joint contraction, and test these groups through the same final-state intervention. Keep source/value/QK subgroups only when their effects and fresh predictions justify them. Known head recurrence is not within-head component identity, and borrowed port states are not autonomous extraction.

The derivation and executable falsifier are completed; native port decomposition has not yet run. No promotion against the four desired properties follows from this math control alone.
