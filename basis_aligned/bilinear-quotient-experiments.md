# Information Loss as Structure: Experiments and DGPs for Bilinear Layers and Bilinear Transformers

Working document. The organizing claim: a bilinear layer's computation is characterized by what it collapses, and "what it collapses" is relative to who reads it. Every experiment below has a known DGP, a weight-space prediction, and a causal verification step, in that order.

## 0. Shared machinery and notation

A bilinear layer y = D(Lx ⊙ Rx) with x ∈ ℝ^d, hidden width h, output dim m. Per-output interaction matrices:

Q_i = ½ Σ_j D_ij (l_j r_jᵀ + r_j l_jᵀ)

so y_i = xᵀQ_i x = ⟨Q_i, xxᵀ⟩. The layer factors as the Veronese lift x ↦ xxᵀ followed by a linear map W: Sym²(V) → ℝ^m whose rows are vec(Q_i). All learned content lives in W.

A reader is a downstream linear functional u on the output; its path-form is Q_u = Σ_i u_i Q_i. Readers come from the circuit: rows of the next layer's L and R, columns of W_Q, W_K, W_V, unembedding directions. The reader-weighted second moment is

M = Σ_u w_u vec(Q_u) vec(Q_u)ᵀ

with w_u the reader's gain. Eigenvectors of M are the layer's shared form-vocabulary; the spectrum measures reuse. Equivalent tensor view: stack the family into a 3-tensor (readers × input × input) and Tucker/CP/dictionary-learn on it, with the reader mode kept explicit.

Stratification of a form direction, given downstream tolerance ε: preserved (transmitted above ε through some reader), linearized (intermediate; replace bilinear contribution with first-order approximation), dead (below every reader's floor). The middle band is where the linearization idea lives, and it is per-path, not per-layer.

Metrics. All reconstruction and model-selection error is measured functionally, through readers, using the Λ-weighted tensor similarity loss, never raw Frobenius. Where data enters, whiten by input covariance Σ (RMSNorm removes scale and makes the identity component of every Q pure bias, restricting the effective domain to traceless symmetric matrices; it does not remove anisotropy, so whitening is still needed).

### Null baselines (run for every experiment)

Three nulls of increasing strength, plus a permutation test:

1. Random weights. Same architecture at init, full pipeline. Anything found here is methodological artifact. Marchenko–Pastur gives analytic expectations for M-spectra.
2. Gauge-scrambled trained weights. Random orthogonal rotation in the Hadamard hidden space, random reader rotations. Function-preserving, basis-destroying. Findings that survive are gauge-invariant properties of the mechanism; findings that die were basis-reading. ODT canonicalization makes this null natural to construct.
3. Task-shuffled model. Train on label-permuted or structure-destroyed data to matched loss where achievable. Separates "structure any training run produces" from "structure this task induces."
4. Reader-shuffle permutation test. Randomly reassign which reader gets which form, re-measure spectral concentration of M. Genuine reuse collapses; generic spectral decay does not.

### Model selection protocol

Candidate structures: ∗-algebra blocks (strongest), sparse dictionary of forms, Tucker core, CP/Waring components, plain PCA of {Q_u} (weakest). For each, sweep the budget (block sizes, atom count r and sparsity s, core shape, rank) and plot functional error against description length. Structure A beats B when its frontier dominates uniformly. Crossing frontiers are a finding, not a failure: dictionary-beats-blocks at small budget with reversal at large budget is the signature of coarse-to-fine hierarchy. Every claimed structure must clear all four nulls.

---

## Part A: Pure bilinear layers

### A1. Parity / XOR: kernel verification

DGP. Inputs are k-bit strings embedded in ℝ^d (k ≤ d, random orthogonal embedding plus optional distractor dimensions with continuous noise). Target: parity of designated bit-pairs, or full k-bit parity built from pairwise chunks. XOR is exactly degree 2: x₁ + x₂ − 2x₁x₂, so a single bilinear layer suffices and the ground-truth Q's are known pairwise interaction matrices.

Predictions. (i) Recovered Q_i match planted interaction matrices up to gauge. (ii) ker(W) contains every magnitude direction and every distractor direction: the even/odd quotient made literal. (iii) Effective rank of W equals the number of planted pairwise interactions.

Verification. The falsifiable blindness claim: perturb inputs along directions whose lift lands in ker(W), confirm zero downstream change; perturb along row-space directions, confirm predicted change. This is the cleanest demo that weight analysis yields a crisp causal prediction.

Knobs. Distractor count, embedding anisotropy (tests the whitening story), label noise.

### A2. Modular addition: ∗-algebra blocks and grokking

DGP. Standard (a + b) mod p with one or two bilinear layers. Known Fourier ground truth: the solution lives in 2×2 rotation blocks, one per frequency.

Predictions. (i) The retained {Q_i} simultaneously block-diagonalize; Maehara–Murota recovers frequency blocks from weights alone, no probing. (ii) Block structure is invariant under the gauge-scramble null. (iii) During grokking, block crystallization in weight space precedes or coincides with the generalization transition; tensor-similarity between checkpoints tracks it.

Verification. Ablate one recovered block, confirm the corresponding frequency's contribution to logits vanishes; splice blocks between independently trained models after canonicalization, confirm function transfer.

Why this is the highest-value first target: known ground truth, existing literature for comparison, and it exercises the block machinery, which is the genuinely new tool relative to the CP pipeline. Existing checkpoint-similarity infrastructure applies directly.

### A3. Planted low-rank teacher: CP calibration

DGP. y_i = Σ_k c_ik (a_kᵀx)(b_kᵀx) with known {a_k, b_k}, small true rank K, Gaussian or structured x. Train a bilinear student.

Predictions. Partially-symmetric CP (equivalently signed Waring on the symmetrized forms) recovers planted directions up to the gauge group: sign flips for tied factors, the (c·a, b/c) scale gauge for untied pairs.

Measurements. Recovery quality as a function of K relative to h and d, pushing K into superposition; noise level; correlation between planted directions. Output is a calibration curve: the regime where CP components are trustworthy as variables, and the regime where the dictionary or block view must take over. This experiment exists to bound the other experiments' claims.

### A4. Planted quotient DGP: the stratification and the linearization band

DGP. Construct explicit equivalence structure on inputs. Hard version: binning (all x with feature value in [0.1, 0.2] must map identically); discrete quotient like A1 but with continuous inputs. Soft version: target functions that logarithmically compress some directions and expand others, so small differences in one region matter more than large differences in another. Mixed version: some directions exactly collapsed, some contracted, some preserved.

Predictions. (i) Difference vectors of equivalent inputs land in ker(W) after the lift (hard case). (ii) The contraction spectrum, singular values of W in the whitened metric, reproduces the planted compression profile (soft case). (iii) Stratification thresholds calibrated by the effective noise floor of a downstream readout correctly classify each planted direction as preserved / linearized / dead.

Intervention on the middle band. For directions with intermediate singular value, compare three surgeries: full keep, full prune, replace-with-first-order-linearization. Prediction: linearization dominates pruning on functional error at equal parameter savings across a nontrivial band, and the band's location moves with the downstream tolerance ε. This is the direct test of the linearize-the-middle idea, and the per-path refinement (a feature linearized for one reader, kept bilinear for another) gets tested in A6.

### A5. Shared vs private subcomputation: the reader-weighted global view

DGP. One shared planted subcomputation (a specific quadratic form or small block) feeding three distinct readers, plus one private subcomputation per reader. Readers are explicit downstream linear maps with controlled gains.

Predictions. (i) Reader-weighted spectral analysis (eigenvectors of M) puts the shared form at the top of the spectrum with eigenvalue ≈ 3× the private forms, scaling with reader multiplicity and gains. (ii) Sparse dictionary learning over {Q_u} recovers shared and private atoms as separate dictionary elements, with the correct sparse mixing vectors per reader. (iii) Plain per-path decomposition finds the shared form three times in mutually incompatible gauges, demonstrating concretely why global-fit-then-explain-paths beats decompose-per-path-then-stitch. (iv) Under the reader-shuffle null, the spectral gap between shared and private forms collapses.

Also the natural home for testing PCA's orthogonality failure: plant a hierarchical structure (coarse form plus overlapping refinements) and show PCA smears it while the dictionary and block methods recover it, with the frontier-crossing signature from the model selection protocol.

### A6. Two-hop routing table: path-relative quotients

DGP. Bilinear layer 1 computes features A and B. Two downstream bilinear readers: reader 1 must use A and be provably blind to B; reader 2 vice versa. Constructed by choosing targets so that each reader's task depends on exactly one feature, with capacity limits preventing incidental leakage.

Predictions. (i) ker(Q_u) for reader 1's directions contains B's lifted subspace and excludes A's, and symmetrically. (ii) Kernels grow along paths: the effective path kernel equals ker(W) plus the transmitted-but-ignored complement, verified against the planted routing table. (iii) The mechanism ledger for this model, features × readers with entries preserved/linearized/dead, exactly matches the construction.

Verification. Path patching: perturb along B within layer 1's row space, confirm reader-1 output invariance and reader-2 output change. Circuit pruning by norm tests on composed forms (Λ-weighted norm of Q_u below threshold ⇒ path declared dead) validated against activation patching as ground truth, measuring agreement rate. This is the calibration of "norm tests replace patching for discovery, patching remains verification."

---

## Part B: Bilinear transformers with bilinear attention

Architecture under study: bilinear MLPs plus multiplicative attention, two QK matrix sets whose attention patterns element-wise multiply. Both placement variants matter and have different theory.

### B0. Two multiplication placements

Logit-level product. Combined score s(q,k) = (qᵀW₁k)(qᵀW₂k). With linear Q/K maps this is already degree (2,2) across positions; with bilinear residual-stream features feeding it, higher. Rank-decompose each W_i = Σ_a σ_a u_a v_aᵀ and the product is a sum over pairs of matching conditions: key must satisfy a condition from set 1 AND a condition from set 2. Architectural prior toward conjunctive, composable predicates.

Post-softmax product. A = normalize(A₁ ⊙ A₂): soft intersection of two attention distributions, a product-of-experts over positions. Decompose each pattern's QK circuit separately, then quantify interaction: per head-pair, how much the composite differs from either factor (KL to each factor, entropy drop).

### B1. Degeneracy taxonomy (real and toy models)

Three sharp null hypotheses per head, each with a scalar test:

(a) Factor collapse: one factor near-uniform, model effectively uses one QK circuit. Test: entropy of each factor's pattern; H(A₂) ≈ log n.
(b) Factor alignment: W₁ ≈ cW₂ up to gauge. Test: tensor similarity between the two QK circuits directly.
(c) Genuine conjunction: each factor individually broad, product sharp. Test: H(A₁), H(A₂) both high while H(A₁ ⊙ A₂ normalized) low. The "individually vague, jointly precise" entropy signature is the smoking gun, measurable per head across training time.

Deliverable: the (a)/(b)/(c) census over heads and checkpoints, with the prediction that conjunction (c) emerges preferentially on tasks whose DGP is itself conjunctive (B2), and degeneracies (a)/(b) dominate elsewhere.

### B2. Conjunctive retrieval DGP

DGP. Sequences of tokens each carrying two independent planted properties: a type feature (one of T types, linearly decodable) and a position-parity or timing feature. The query token defines a conjunction; exactly one key token matches both properties; distractors match exactly one property each, in equal numbers, so single-property strategies score at chance against distractors.

Predictions. (i) Multiplicative attention solves the task with each factor linear in one property: decomposing W₁ and W₂ separately recovers the two planted properties, one per factor, up to gauge. (ii) Standard softmax attention at matched capacity either fails or solves it by learning conjunction features, visible as the planted product-feature in its QK eigenstructure, with a measurable capacity penalty. (iii) The entropy signature (c) appears for multiplicative heads on this task and not on a control task where a single property suffices.

Verification. Swap planted property embeddings between factor circuits (after canonicalization), confirm attention follows the swap. Ablate one factor, confirm degradation to single-property chance performance against distractors.

Knobs. Distractor ratio, correlation between the two properties (correlated properties should induce drift from (c) toward (b)), number of types T, sequence length.

### B3. Full-stack testbed: planted quotient upstream of conjunctive attention

DGP. Compose A6 and B2. Bilinear layer 1 computes features A, B, C from token inputs with a known quotient (some input distinctions collapsed, some contracted). Downstream, a multiplicative attention head whose factor 1 reads feature A and factor 2 reads feature B, while an MLP path reads C. Ground truth is a complete routing table plus a known conjunction.

Predictions, end to end. (i) Path-relative kernels of layer 1 through each attention factor and through the MLP path reproduce the routing table. (ii) The end-to-end preserved information is the intersection of layer-1 row space with reader sensitivities, strictly smaller than any per-layer analysis suggests; measured effective rank matches the construction. (iii) The mechanism ledger recovered purely from weights (norm tests on composed forms, reader-weighted decomposition, degeneracy census) matches the planted design, with activation patching as the audit.

This is the centerpiece testbed: every claim in Parts A and B has a component here with known ground truth, and it is the strongest known-DGP environment the framework admits. Success criterion for the whole agenda: the weight-only pipeline reconstructs the planted routing table and conjunction with high fidelity and its errors are diagnosable (attributable to superposition level, gauge residue, or threshold miscalibration measured in A3/A4).

### B4. Bilinear attention decomposition on trained language models (stretch)

Once B1's census and B2's calibration exist: apply the conjunctive decomposition to the ~500M bilinear GPT. Targets: identify heads in regime (c); extract their factored matching conditions; attempt natural-language characterization of each factor separately (the architectural bet being that factors are simpler than their product); check whether the (a)/(b)/(c) proportions shift with depth and training compute.

---

## Ordering and dependencies

A2 first (highest value, exercises the new block machinery, existing infra). A1 and A3 in parallel as cheap calibrations. A4 next (tests the linearization idea directly). A5 and A6 build the reader-relative machinery. B1/B2 can start independently of Part A once the multiplicative-attention toy infra exists; B3 requires A6 and B2. Every experiment ships with its four nulls; a result without its nulls is not a result.
