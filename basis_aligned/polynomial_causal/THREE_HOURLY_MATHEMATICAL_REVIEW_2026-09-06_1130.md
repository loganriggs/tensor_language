# Three-hour mathematical circuit review — 2026-09-06 11:30 UTC

## Decision

The current object is not a tensor-rank search. It is an exact, architecture-native decomposition of one already localized attention contraction. The strongest applicable theorem is Boolean-lattice Möbius inversion: four interventions uniquely determine the two branch main effects and their downstream interaction. That theorem does not identify the branches by itself; identification comes from the checkpoint equation that separately names the local layer-9 and carried layer-0 value paths. The executable consequence is to unit-test the Möbius accounting (including efficiency) and then run the already preregistered four-arm, two-orientation experiment. No tensor-train, CP, or graph-width method offers more information at lower cost here.

## Exact current object

The registered population has batch index (b\in[16]), selected head (h\in\{1,4\}), final query (q_b), task-indexed carrier set (K_b) of size two (`is/was`) or three (`has/had`), and head coordinate (j\in[128]). Model width is 1,152 with nine heads. At layer 9 the exact selected-head carrier vector is

\[
t_{bhj}=\sum_{k\in K_b}P^{(9)}_{bhq_bk}
\left[(1-\lambda_9)V^{(9)}_{bkhj}+\lambda_9V^{(1)}_{bkhj}\right].
\]

Here (P^{(9)}=S^{(1)}S^{(2)}) is the elementwise product of two causal masked query-key score tensors. (V^{(9)}=c_v^{(9)}(N(x_9))), while (V^{(1)}=c_v^{(0)}(N(x_0))) is retained once and tied into every later attention layer. The checkpoint fixes one scalar (lambda_9=-0.656\) approximately, shared over all rows, tokens, heads, and head coordinates.

For each side (s\in\{b,d\}), define routing mass (p^s_{bh}=\sum_{k\in K^s_b}P^s_{bhqk}) and side-native normalized weights (a^s_{bhk}=P^s_{bhqk}/p^s_{bh}). The parent experiment's content tensor is

\[
\Delta C_{bhj}=p^b_{bh}\left(\sum_{k\in K^d_b}a^d_{bhk}u^d_{bkhj}
-\sum_{k\in K^b_b}a^b_{bhk}u^b_{bkhj}\right),
\quad u=(1-\lambda_9)V^{(9)}+\lambda_9V^{(1)}.
\]

The new factorial uses the exact linear identity

\[
\Delta C=\Delta C_{9}+\Delta C_{1},
\]

where (Delta C_9) substitutes ((1-\lambda_9)V^{(9)}) for (u), and (Delta C_1) substitutes (lambda_9V^{(1)}). Each selected (Delta C_ell) is added before layer-9 `c_proj`; all unselected heads, all routing patterns, the projection, residual/MLP suffix, RMS normalization, unembedding, and logit soft-cap remain native.

The contraction graph is `tokens -> x0 -> layer-0 c_v -> tied V1 branch` in parallel with `tokens -> native blocks 0:9 -> x9 -> layer-9 c_v -> local V9 branch`; the two meet through a scalar weighted sum, contract with the fixed observed carrier pattern over (k), concatenate across heads, pass through `c_proj`, and then through the native blocks 9:18 and output head. Conditioned on normalized layer-9 states, each score is bilinear in query/key coordinates, their product is quartic, and multiplication by a value is degree five. As a function of residual states the RMS denominators make the expression algebraic rather than polynomial; the final `tanh` soft-cap makes the whole outcome map analytic but non-polynomial. Therefore polynomial-decomposition theorems do not directly identify the end-to-end map.

Tied parameters are the single (V^{(1)}) tensor and scalar (lambda_9) reused by all heads/positions. Source order inside a semantic carrier set is a permutation symmetry if patterns and values are permuted jointly. Each head admits the usual simultaneous hidden-coordinate gauge when its value coordinates and corresponding `c_proj` block transform inversely. The local/carried labeling is checkpoint-native, not invariant under an arbitrary mixing of the two branches; separate interventions make it operationally identifiable only relative to the fixed architecture.

Allowed inputs are the two native matched-task prompt batches, their raw layer-0/layer-9 value projections, native patterns, fixed carrier roles, and checkpoint weights. Required outputs are the four target-token logits, normalized task-support recovery, direction fraction, capability cells, exact tensor closures, and the parent content effect. Internal approximation is judged in max absolute coordinate error (\le10^{-4}); behavioral preservation is within 0.05 normalized recovery plus exact registered direction bars. Literal price is two native captures plus eight subset-orientation interventions: 10 forwards, 160 example evaluations, zero fitted scalars, zero search, zero backwards, and zero updates. No program parameters are stored by this diagnostic beyond hashes and scalar results.

## Exact theorem and algorithm mappings

### Möbius inversion on the intervention lattice

Rota's incidence-algebra formulation gives a unique inverse for the zeta transform on a locally finite partially ordered set ([Rota, 1964](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/rota1.pdf)). Map the Boolean lattice (2^{\{9,1\}}) to the four causal arms and let (f(S)) be mean normalized donor recovery after installing branches in (S). The unique Möbius coefficients are

\[
g(\varnothing)=f(\varnothing),\quad
g(9)=f(9)-f(\varnothing),\quad
g(1)=f(1)-f(\varnothing),
\]

\[
g(9,1)=f(9,1)-f(9)-f(1)+f(\varnothing).
\]

The two-factor Shapley allocation is then (\phi_9=g(9)+g(9,1)/2), (\phi_1=g(1)+g(9,1)/2), so the exact efficiency identity is (\phi_9+\phi_1=f(9,1)-f(\varnothing)). Complexity is (O(2^n)) arm evaluations and an (O(n2^n)) transform; here (n=2), so it is exactly the registered four arms per orientation. The guarantee is uniqueness of the set-function interaction accounting. It assumes only complete values on the finite poset. It does not say that the neural branch labels are unique, causal, or architecture invariant; those properties must come from the separately verified internal tensor identity and intervention sites.

### Bilinear complexity of the carrier contraction

For a carrier of fixed cardinality (K), the generic map ((p,U)\mapsto y), (y_j=\sum_{k=1}^Kp_kU_{kj}), has structure tensor

\[
T=\sum_{k=1}^{K}\sum_{j=1}^{128}e_k^*\otimes e_{kj}^*\otimes e_j.
\]

Flattening the (U) mode against (p\otimes y) has rank (128K), giving the same lower bound on bilinear tensor rank; the literal algorithm uses (128K) scalar products. Kruskal develops tensor-rank lower bounds and sufficient essential-uniqueness conditions for trilinear decompositions, with applications to arithmetic complexity ([Kruskal, 1977](https://www.sciencedirect.com/science/article/pii/0024379577900696)). This mapping says there is no generic exact multiplication-saving rewrite of an arbitrary carrier contraction. It does not certify a lower bound on this checkpoint's restricted activation manifold, and it does not determine whether (Delta C_9) or (Delta C_1) matters behaviorally. Consequently it supports preserving the exact contraction but does not replace the intervention.

### CP, tensor train, and hierarchical Tucker identification

Kruskal's essential-uniqueness condition for an (R)-term CP decomposition requires sufficient k-rank across three factor matrices (in the familiar form, their sum at least (2R+2)). No such CP generative model or k-rank certificate has been established for the row-by-head-by-coordinate response tensor; unequal carrier arity and tied (V^{(1)}) further violate a naive independent-factor interpretation. CP uniqueness therefore does not identify our branches.

TT-SVD constructs tensor-train factors from sequential unfoldings and supplies controlled Frobenius approximation through truncation ([Oseledets, 2011](https://epubs.siam.org/doi/10.1137/090752286)). Hierarchical Tucker similarly builds nested tensor-product subspaces on a binary tree and truncates using SVDs ([Hackbusch and Kühn, 2009](https://files-www.mis.mpg.de/mpi-typo3/preprints/2009/preprint2009_2.pdf)). Their objects are explicit high-order arrays and their guarantees concern exact ranks or Euclidean approximation. Our target norm is a nonlinear downstream causal-response functional, the relevant tensor has only two labeled additive branches, and the checkpoint tying must remain intact. Either method would optimize the wrong norm and add decomposition choices without causal identification.

### Graph-width contraction

Markov and Shi show that a tensor network with (T) gates and graph treewidth (d) can be contracted in (T^{O(1)}\exp(O(d))) time ([Markov and Shi, 2008](https://arxiv.org/abs/quant-ph/0511069)). The localized carrier subgraph has a trivial contraction order: scale two value tensors, add them, and contract one source index. The expensive operation is loading and executing the native transformer suffix, which the theorem does not remove because its nonlinearities and observed activations are required. Graph-width optimization therefore has no useful advantage at this scale.

## Executable consequence and route comparison

Before GPU execution, factor the two-arm Möbius calculation into a pure helper and test on a synthetic nonlinear set function that:

1. the recovered interaction equals inclusion-exclusion exactly;
2. Shapley efficiency holds exactly;
3. the local retained fraction uses the registered joint endpoint; and
4. the enumerated arm order is exactly empty, local, carried, both.

Then execute the hash-bound experiment. Opposing outcomes remain preregistered: local (V^{(9)}) dominance licenses it as the major carrier-content writer; carried (V^{(1)}) dominance changes the graph to a direct lexical-value bypass; a large interaction retains both branches and forbids an additive behavioral simplification. Exact internal closure with behavioral non-additivity is coherent because the native suffix is nonlinear.

This route dominates a CP/TT fit or contraction-order search: it costs one ten-forward receipt, exactly tests the architecture's two named edges, and every terminal outcome changes the circuit graph. The alternatives either require unverified low-rank assumptions, optimize Frobenius rather than causal error, or address a contraction cost that is already negligible. The live plan therefore stays on the effective-value branch factorial, with the Möbius unit test as the only added preexecution requirement.

The next mathematical review is due around **2026-09-06 14:30 UTC**. The next hourly strategic review remains due around **12:17 UTC**.
