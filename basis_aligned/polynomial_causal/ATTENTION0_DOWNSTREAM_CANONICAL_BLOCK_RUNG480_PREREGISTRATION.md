# Rung 480 preregistration — downstream-canonical directions inside the continuous attention0 block

Registered after rung479's frozen strong null and before rebuilding the attention0 block or collecting any new
response. The30 odd-root circuit tags, documents500:1000, and the attention0 SEALED consequence family remain
unopened.

## Goal and non-duplication boundary

Rungs424/425 already identified and independently replicated a continuous attention0 interface. On unseen documents,
six varying modes for each of the two score branches and32 varying modes for the head-by-output payload reproduce99.03% of the
head-summed response-metric edge signal, retain98.53% of the routed U16 signal, give99.19--99.45% R2 at six named
downstream readers, and add only`.000200` nat CE. Permuting which heads' branches multiply destroys the result.

This rung does **not** rerun:

- rung418's pairwise folded-Q/K subspace comparison;
- rungs426/430's sparse Q/K token vocabularies;
- raw Q/K weight SAEs or Tucker/HOSVD;
- rung431's failed direct unnormalized bilinear generator; or
- rank/mode-count selection.

Instead, it asks whether downstream circuit behavior chooses reproducible directions *inside* the already-fixed
6×6×32 continuous interface. This targets a split of one useful attention computation by what later computation does
with it.

## Exact affine-trilinear object

Deterministically rebuild rung424's selected joint fit on its original96 FIT documents and reproduce its SELECT and
rung425 fresh-row metrics before using it. Rung424's reconstructions are **affine** projectors: each contains a fixed
offset in addition to its6/6/32 varying directions. A pre-implementation algebra audit on2026-09-02 caught that the
original 1,152-triplet wording omitted these offsets and therefore could not be exact. No model or response outcome
had been opened. This registration is amended before implementation to include one distinguished constant coordinate
per mode while leaving all learned ranks unchanged.

Let `B1[h,i]`, `B2[h,j]`, and `BV[h,u,k]` be the fitted varying bases for score branch1 (`i=1..6`), score branch2
(`j=1..6`), and the nine-head by16-output payload (`k=1..32`, `u=1..16`). Define augmented bases whose coordinate0
is the fixed affine offset and whose remaining coordinates are these varying bases. For realized edge `e`, define
augmented coordinates `a_e=[1,(score1_e-mean1)B1]`, `b_e=[1,(score2_e-mean2)B2]`, and
`c_e=[1,(payload_e-meanV)BV]`. The exact fitted-block contribution of one augmented triplet is

`edge[e,i,j,k,u] = a_e[i] b_e[j] c_e[k] sum_h B1_aug[h,i] B2_aug[h,j] BV_aug[h,u,k]`.

Summing all `7×7×33=1,617` augmented triplets must reproduce the fitted joint block's U16 edge output, and summing causal
source edges must reproduce its query output. The native remainder is kept separate and unchanged.

Only the6/6/32 varying coordinates can rotate; the constant coordinate is distinguished and cannot mix with them.
An individual varying mode is therefore not meaningful until a downstream statistic fixes its projector and that
projector transfers.

## Discovery response tensor

Use the same32 even-root discovery circuit tags and documents0:500 as rung477b, with the corrected row250 split,
member masks, and matched in-slice controls. Use two downstream trajectories: the native matcher and the frozen
transplanted matcher. These trajectories change later use of attention0 but do not refit the attention0 block.

For a selected target token at query position `q`, differentiate its CE with respect to the fitted block's16-number
attention0 query write. For each latent triplet, contract that gradient with the sum of the triplet's causal-source
edge contributions into `q`. Average separately over circuit members and matched controls. The response tensor is

`R[half,source,mask,circuit,i,j,k]`,

with shape `2×2×2×32×7×7×33`. Coordinate0 in each latent mode is the fixed affine offset. The
member-minus-control tensor is the task response used below. This is a first-order, query-local proposal. No mode is
a causal circuit until later exact removal.

Run a second deterministic joint-block fit on the two original FIT document halves. Align each refit's three latent
spaces to the main fit by orthogonal Procrustes maps on `B1`, `B2`, and `BV`; transform its response tensor by those
maps before comparing it. This comparison removes harmless latent rotations without forcing individual axes to
match.

## Downstream-chosen projectors

Fit only the native-source documents0:250 member-minus-control tensor. For each latent mode, restrict that mode to
its varying coordinates and contract over every augmented coordinate in the other two modes and the32 circuits to
form a response Gram matrix:

- `G1[i,i'] = sum_(c,j=0..6,k=0..32) R[c,i,j,k] R[c,i',j,k]`, for varying `i,i'=1..6`;
- `G2[j,j'] = sum_(c,i=0..6,k=0..32) R[c,i,j,k] R[c,i,j',k]`, for varying `j,j'=1..6`; and
- `GV[k,k'] = sum_(c,i=0..6,j=0..6) R[c,i,j,k] R[c,i,j,k']`, for varying `k,k'=1..32`.

The leading eigenvector of each Gram defines one one-dimensional projector. This is a downstream-response direction,
not a claim that rank one is sufficient. No other eigenvector, rank, cutoff, or subset is tried.

The three proposed **slabs** retain the leading projector in one mode's varying subspace and every augmented
coordinate in the other two. For example, the branch1 slab is `P1 R`, not the single Cartesian triplet
`P1×P2×PV`. Its complement is the full exact affine block minus that slab, so fixed-offset terms remain there. A
slab can therefore contain many compositions while testing whether one varying input/output direction has a stable
downstream role.

## Controls

1. **Activation-only basis:** form the leading varying-coordinate direction from the same causal-edge coordinates,
   without CE gradients or circuit labels. It has the same per-mode dimension but cannot use downstream computation.
   For B, compare the main downstream projector's refit/half stability with its overlap with this activation-only
   direction; the downstream stability must be larger by`.10`.
2. **Circuit-label control:** for16 frozen seeds, independently permute the32 circuit coordinates of the second
   document/source view after fitting. This preserves response magnitudes and latent geometry while destroying the
   claim that a slab means the same downstream thing.
3. **Matched circuit controls:** keep the existing in-slice matched positions as a separately reported response, not
   merely as a subtraction.
4. **Independent-refit gauge control:** compare after legal Procrustes alignment and also after a deliberately wrong
   permutation of the three latent refit maps.

## Frozen predictions

### A — exact block and response instrument

- all rung479, rung424, rung425, circuit-authority, row, source, and model hashes match;
- the rebuilt block reproduces rung424/425 metrics and fixed6/6/32 sizes;
- all1,617 augmented-triplet sums reproduce block edge/query outputs to relative squared error at most`1e-8`;
- native replay, row250 allocation, support, gradient path, forward/backward counts, and all live controls pass; and
- odd-root, documents500:1000, FINAL, and SEALED outcomes remain unopened.

### B — downstream use fixes at least two latent projectors

At least two of branch1, branch2, and payload have fit leading/second response-Gram eigenvalue ratio at least`1.50`;
their leading projectors overlap at least`.80` with the aligned independent-refit projectors and at least`.80` with
projectors fit on each half of the native fitting documents. The downstream projector's minimum aligned-refit/fit-half
overlap must exceed its overlap with the corresponding activation-only direction by at least`.10`.

### C — at least one slab has a stable circuit-labelled response

For one slab, its centered32-circuit member-minus-control response profile has cosine at least`.70` between document
halves and at least`.70` between matcher sources. Its minimum cross-view cosine exceeds the95th percentile of the16
circuit-label controls by at least`.15`, and its member response norm is at least1.5 times its matched-control norm in
every view.

### D — the slab is not driven by one discovery family

After omitting each of the six even top-level circuit families, at least five omissions retain minimum cross-view
profile cosine at least`.60` and retain the same winning slab.

### E — the proposed split distinguishes downstream uses

The winning slab and its orthogonal complement each have nonzero response in every view; their32-circuit profiles
have absolute cosine at most`.70` in every view; and at least ten discovery circuits have opposite signed
slab-versus-complement effects in both document halves. This is the prospective evidence that the continuous block
contains distinguishable computations rather than one amplitude direction.

## Strong null and routing

The strong null fires if A fails, fewer than two projectors pass B, or no slab beats the circuit-label control in C.
A+B+C+D+E is still only a screen. It licenses exact projector-defined removal of the winning slab on the reserved
odd-root circuits and documents500:1000, with unrelated-circuit preservation and later recomputation.

With A and a strong null, retain rungs424/425 only as a compact continuous predictive interface: do not name its
individual axes, change6/6/32, sweep Tucker/CP ranks, or rerun sparse atoms. Switch the active line to the independent
MLP0 token-only / token×context / context-only functional decomposition using its finite vocabulary and exact folded
weights.

## Price

Discovery-only GPU response collection plus deterministic block refits and CPU projector analysis. Zero deployed
parameters saved or added. Report the full1,617-triplet affine response tensor, exact identities, refit alignments, projector
spectra, slab profiles, control distributions, runtime, and literal stored values. Save no raw rows, tokens, logits,
or hidden states.
