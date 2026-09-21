# Pair-pencil blocks as proposals for reusable input spaces

21 September 2026. This is a bounded mathematical follow-up; it does not reset the scheduled three-hour review clock.

For pair j, two symmetric coefficient matrices share a representation

$$
T_{jo}=F_j\operatorname{diag}(A_{jo},C_{jo})F_j^\top,
\qquad F_j=[P_j\ D_j],\quad o\in\{1,2\}.
$$

The shared columns P_j contain the two dictionaries reused by j's neighboring consumers. The private columns D_j are local. F_j is full column rank on the pair's input support, even though the union of dictionaries across all consumers need not be globally independent.

Restrict to that support. If a pencil base B=T_1+T_2 is nonsingular, then K=(T_1-.37T_2)B^{-1} is similar through F to a block diagonal operator. Thus the shared and private column spaces are invariant subspaces of K. With a regular diagonalizable pencil and separated blocks, candidate shared spaces can be assembled from real eigenblocks (including real two-dimensional blocks for complex conjugate eigenvalues). Repeated or defective spectra require additional treatment; this implementation is not a general canonical form.

Our existing pair compiler supplies such blocks and their coefficient-space input directions. Enumerate block subsets at the registered shared width. A proposed shared space must intersect each neighboring pair's input support in at least the registered edge width. After one unique proposal remains per pair, intersect the proposed common spaces to recover the three reusable edge dictionaries. Planted directions are used only to audit the result.

Related primary work:

- [Fang, Huang and Huang, simultaneous block diagonalization by congruence](https://arxiv.org/html/2503.01166v1) studies the block structure of collections of symmetric forms using their center. Our use here is the pair-level congruence structure; their general result does not establish uniqueness of our coupled, overlapping dictionaries.
- [Cai and Li, identification of matrix joint block diagonalization](https://proceedings.mlr.press/v130/cai21a.html) formulates a common full-column-rank mixing matrix with block-diagonal cores. Our local pairs have that form, but the complete overcomplete set of shared/private inputs is not one global independent mixing matrix. We cannot import its identification guarantees without that mapping.
- [Evert and De Lathauwer, block-term decomposition and joint block diagonalization](https://www.tensorlabplus.net/papers/evert2023camsap.html) connects algebraic BTD algorithms with block diagonalization. It motivates the candidate-generation route, not a claim that this implementation solves their general problem.

The tested enumeration costs O(2^b) subset visits per pair for b pencil blocks, plus polynomial-size QR/SVD checks. The five fixtures have only 3–8 blocks per pair and require 3–35 dimension-matched subsets, except one with 22. Native 352-dimensional pairs can have hundreds of blocks: exhaustive enumeration is unsuitable. Moreover, full-rank native targets give no exact support restriction. A scalable approximate proposal and a reconstruction test remain necessary.

Results: all five original fixtures recover their planted shared spaces and coefficient tensors (maximum error 4.55e-14). Fifteen input-coordinate/output-mixing controls pass actual shared-graph execution and cost comparisons. The saved graph uses the same number of nonlinear products as independent pair compilation; its savings come from shared linear projections.

The fixed noise screen retains a negative result. At relative pair noise 1e-8, all five have unique candidates, but only three meet reconstruction <=10*noise and shared-projector discrepancy <=100*noise. Cases 1 and 4 amplify coefficient error by about16.7 and20.1. At1e-4 none is uniquely accepted by the unchanged exact procedure. Exact recovery is therefore not native robustness.

Next consequence: use fixed architecture-width support truncation and score approximate block compatibility instead of relying on exact intersections. Candidate directions remain proposals; continuous refitting must improve the full coefficient error at the same graph cost. This is a new approximation procedure, not a relaxed pass for the exact-noise screen.
