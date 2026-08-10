# Memorization-in-Bilinear-Layers Blog Post: Experiment & Figure Handoff

Purpose: produce every figure and empirical claim needed by the draft post "Memorization in Bilinear Layers". Self-contained. All experiments are small (CPU-viable except possibly the 2-layer scan). Every figure the post needs is listed with its acceptance criterion.

## Global conventions

- Bilinear layer: y = D((Lx) * (Rx)), elementwise product over hidden dim H.
- Folded tensor: T[i,j,c] = sum_h D[c,h] * sym(L[h,i] * R[h,j]), i.e. per-class interaction matrix B_c = T[:,:,c], symmetrized. Report both symmetric eigendecompositions and asymmetric L/R factorizations where relevant.
- Boolean inputs: note x_i^2 = x_i, so DIAGONAL entries of B_c act as linear terms. Always plot diagonals distinctly (e.g. outlined cells) in interaction-matrix heatmaps.
- Seeds: 5 per trained result; report mean and range; flag sign flips.
- Save all matrices as .npy plus rendered heatmaps (diverging colormap centered at 0, same color scale within each figure panel).

## Part 1: The 3-class toy (Dog / Cat / Catfish)

Features: [furry, happy, whiskers]. Classes: Dog = furry AND happy; Cat = furry AND whiskers; Catfish = whiskers AND happy. Dataset: all 8 boolean inputs, labels by the rules (inputs matching no rule or multiple rules: use only the 3 single-class inputs plus the all-zeros "none" case for training v1; if degenerate, train on the 3 positive examples with softmax CE and report what happens on the other 5 inputs as part of the generalization figure).

### 1a. Hand-coded clean model
Construct H=3 solution: unit h_c has L-row = first feature of class c, R-row = second feature, D = identity-ish.
- FIGURE F1: three 3x3 interaction matrices (one per class), clean case. Acceptance: each matrix has a single symmetric off-diagonal pair, nothing else.
- Verify rank claims for the post: each B_c symmetric rank 2 with eigenvalues +-1/2 on (e_a +- e_b)/sqrt2; asymmetric L/R rank 1. Print both.

### 1b. Trained model, overcomplete (H = 8)
Train with softmax CE, AdamW, small weight decay sweep {0, 1e-3, 1e-2}.
- FIGURE F2: trained interaction matrices next to F1. Pre-registered prediction to check: positive off-diagonal on the class's feature pair PLUS negative structure involving the class's absent feature (negative off-diagonals with the absent feature and/or negative diagonal on it, i.e. linear "not-whiskers" logic via the diagonal). Report which channel SGD used, and whether weight decay changes it.
- FIGURE F3 (optional callout): bar chart of diagonal entries per class showing the linear-logic-via-diagonal effect.

### 1c. Undercomplete variant
Either H=2 with the 3 classes, or scale to 6 features / 6 classes with H=4 if H=2 fails to train. Goal: force shared hidden units.
- FIGURE F4: undercomplete interaction matrices ("things are more mixed up").
- FIGURE F5: the key Gram matrix. Keys: per class, the input pattern z_c (or in the general setup, the stored key). Plot G = Z^T Z (and the C-weighted version if a retain-weighted metric is used). Two panels: overcomplete case (near diagonal) vs undercomplete (off-diagonal mass).
- CLAIM CHECK C1 (the post's centerpiece): for each single-class edit (zero class c's slice, or KKT-edit class c to uniform), PREDICT collateral damage to each other class from the Gram off-diagonals BEFORE measuring, then measure actual accuracy/logit damage. FIGURE F6: scatter of predicted vs measured collateral, one point per (edit, victim) pair, both regimes. Acceptance: strong rank correlation; report r.

### 1d. Edits on the toy
- Pull-out: slice T[:,:,dog]; reconstruct standalone dog-detector; verify exact functional match on all 8 inputs. Then CP-decompose the slice back to L/R/D form; confirm rank-1 asymmetric recovery.
- Two-class sub-model: slice two classes, re-decompose, report hidden dim needed. Also compute the discriminator matrix T[:,:,c1] - T[:,:,c2] and its eigendecomposition. FIGURE F7: discriminator heatmap + top eigenvectors labeled by feature.
- Path removal: add the 4th class per the draft (Human: hands + dog-ears; Dog gains a second path furry + dog-ears, requires extending features to [furry, happy, whiskers, hands, dog-ears]). Remove ONE computation path to Dog (zero the (furry,dog-ears) interaction) while preserving (hands,dog-ears)->Human and the original (furry,happy)->Dog path. Verify surgical success in the overcomplete case and measure collateral in the undercomplete case. FIGURE F8: before/after matrices + a small table of per-class accuracy deltas.

## Part 2: 100 random facts (the Linda & Lucius setup)

Setup: n-bit random keys (n=20), 100 facts mapping keys to random classes (e.g. 10 classes), no structure. Single bilinear layer sized so memorization succeeds but with meaningful key overlap (sweep H).

### 2a. Construction vs SGD
- Implement the closed-form/ALS construction: fix D = -I (or the convention from prior work), alternate exact convex block solves for L and R; and implement the rank-1 KKT edit for adding a single fact: delta = (y* - f(z*)) applied via the C^{-1}-weighted key direction, C = sum_k z_k z_k^T over the fact list (weights + fact list only, NO corpus; state this explicitly in outputs).
- Train the same architecture with sparsity-regularized SGD from scratch, 5 seeds.
- FIGURE F9: comparison of the SGD solution and the construction. Three panels: (i) weight-statistic histograms, (ii) folded-tensor similarity (Frobenius inner product over Gaussian and over on-distribution boolean inputs) between SGD and construction, with a random-init baseline and a permuted-fact baseline, (iii) D matrix of the SGD run (checking D approx -I emerges).
- Acceptance: similarity of SGD-vs-construction clearly separated from both baselines.

### 2b. Edits at scale
- Extraction: recover individual facts as rank-1-ish components of the folded tensor; report recovery rate as a function of Gram overlap (bin facts by max off-diagonal Gram entry; plot recovery vs overlap). FIGURE F10.
- Removal: unlearn a random set of 10 facts via the KKT edit (target uniform). Measure: (i) forgetting success on the 10, (ii) collateral on the 90, (iii) predicted collateral from Gram vs measured (extends F6 to scale; add these points to the F6 scatter or make F11).
- Naive-removal baseline: subtract the fact's rank-1 component without re-tensioning; show it is worse, quantify.

### 2c. Generalization bound
- The trained/constructed model is a known degree-2 polynomial. Compute, for a large sample (or exhaustively for n<=20 via smart enumeration or bound derivation), the maximum output margin on inputs OUTSIDE the fact set, and compare against the analytic bound in terms of Gram overlap with stored keys. FIGURE F12: histogram of off-fact-set max margins with the bound overlaid. This is the "behavior guarantee on all 2^n inputs" figure; the post's claim is exactly this and no more.

## Part 3: 200 facts in 2 layers (verify the cross-layer question)

Two bilinear blocks with residual stream, 200 facts, sized so one layer alone fails but two suffice.
- Question 1: does SGD store facts disjointly per layer (clean per-layer folded slices) or use the degree-4 composed terms?
- Measurements: (i) per-layer folded tensors; attribute each fact to a layer by single-layer evaluation of that fact's key; (ii) magnitude of cross terms: evaluate the composed model minus the sum of per-layer contributions on fact keys (the degree-4 interference); (iii) ablation: zero layer 2 (resp. 1) and count surviving facts.
- FIGURE F13: fact-attribution histogram (layer 1 / layer 2 / mixed) + cross-term magnitude distribution.
- Also check the draft's "negation for every fact" hunch: look for interference-cancellation mass, i.e. weight structure whose removal selectively breaks facts stored in the OTHER layer.
- Report honestly if results are messy; this section of the post is allowed to end with "verified: it is/isn't cross-layer" plus one figure.

## Part 4: Structure interpolation (stretch goal, only if time permits)

Interpolate between random facts and structured facts (e.g. some bits deterministically indicate class regions, remaining bits random). Sweep structure fraction; measure capacity (max facts memorized at fixed H). Prediction from the draft: structure -> compressible -> more facts stored. FIGURE F14: capacity vs structure fraction. Keep minimal; one plot.

## Deliverables checklist

- figures/ F1-F14 as PNG + SVG, matrices as .npy
- results.md summarizing every claim-check with numbers (especially C1 predicted-vs-measured correlation, F9 similarity separations, D approx -I confirmation)
- A short "surprises.md" for anything that contradicted the pre-registered predictions in 1b, Part 3, or Part 4; the post will quote these directly.
- All code seeded and re-runnable end to end from one script per part.

## Pitfalls

- Do not symmetrize before extracting asymmetric L/R rank claims; do both factorizations explicitly.
- Boolean diagonal-as-linear effect: always separate diagonal from off-diagonal when quantifying "interaction" structure, or the linear logic contaminates interaction statistics.
- The Gram-based collateral prediction must be committed (values written to disk) BEFORE the measurement runs; keep the timestamped prediction file.
- For F9, similarity must be compared against BOTH baselines (random init AND permuted-fact construction); beating only random init is not evidence of mechanism identity.
- If the 3-class toy is degenerate under softmax CE with only 3-4 training points, add label noise or the 4th class rather than silently changing the loss; note whatever was done in results.md.
