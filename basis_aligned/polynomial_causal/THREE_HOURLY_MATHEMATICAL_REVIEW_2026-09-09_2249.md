# Mathematical checkpoint — 2026-09-09 22:49 UTC

The target is an explicit reusable circuit computation, with fresh/OOD
full-output prediction, extraction, accurately predicted removals and joint
use, and a smaller structural description charging all adapters and opaque
weights. Neither a good task margin nor a coordinate decomposition satisfies it.

## Exact object and current boundary

The live model is now actual bilin18:18 layers, residual dimension1152,
nine attention heads of width128, bilinear MLP width4608, vocabulary50304.
Its545902902 parameters are unchanged. Residual state r_l and original
embedding e enter u_l=lambda_l0*r_l+lambda_l1*e. Values are
V_l=(1-alpha_l)*W_Vl RMS(u_l)+alpha_l*v0, with token-local v0 formed before
attention. Each source read uses (q1 dot k1)(q2 dot k2)/128² and a causal
prefix mask. Input/head RMS, BF16-rounded RoPE tables and final RMS/softcap
remain explicit. With fixed normalizers, an attention term has degree5 in
its query/source factors and a bilinear MLP degree2; the full normalized
network is not a polynomial of that degree or a known finite linear automaton.

The graph has one token-only value producer feeding many contextual readers,
inside a sequential residual/MLP computation. Our fixed consumers are
L9H1/H4,L11H3,L15H5. The intervention deletes their shared or contextual value
inputs at all positions, with live later recomputation. It does not delete
v0 globally or assume a shared producer entails interchangeable consumers.

The current restricted input domain is32 opened worlds with two Boolean
command bits, all four cells per world. Output E_ab(t) is the50304-vector
of centered logits after shared-value deletion minus native logits. Source
positions align within each world; there are3180 scored token-cell positions.
We preserve the fixed output vocabulary frame and remove only the logit-shift
gauge. Error is relative full-vector RMS with floor1e-6; native teacher KL
is separately measured. No rank selection, fitted coordinates or opaque
weights disappear. The two latest GPU runs cost3.238 and1.765seconds; the
largest explicitly shaped analysis logit tensor is43.46MB.

## Primary theorem mapping

[O'Donnell, Chapter1](https://www.cs.cmu.edu/~odonnell/papers/Analysis-of-Boolean-Functions-by-Ryan-ODonnell.pdf)
gives the orthogonal character basis and Parseval identity on the Boolean
cube. Here n=2, with vector-valued functions handled coordinatewise. The
four coefficients M_uv=(1/4)sum_ab(-1)^(ua+vb)E_ab uniquely reconstruct the
observed function. This is degree<=2 in two discrete command bits, not a
low-degree representation of the entire transformer. Cost is O(2^n*n*K)
for K output entries with a fast transform, linear in K at n=2; general
intervention cubes still grow exponentially.

[Elesedy and Zaidi, Lemma1 and Proposition3](https://proceedings.mlr.press/v139/elesedy21a/elesedy21a.pdf)
identify group averaging as the unique least-squares invariant approximation.
Our group is Z2 x Z2 acting on command cells; the measure is uniform within
each world and the output representation is trivial. These assumptions
hold on the registered finite domain. Thus M00 is the best possible response
independent of both commands, even allowing arbitrary world/token-dependent
baselines. The discarded-mode norm is an exact function-class error floor.
Their Gaussian linear-regression generalization theorem is not imported:
our response is nonlinear, non-invariant, and these worlds are already open.

[Geiger et al., Definitions25 and41](https://www.jmlr.org/papers/volume26/23-0058/23-0058.pdf)
require an explicit state map and intervention correspondence, evaluated
after intervened mechanisms run. Our consumer component cuts specify a
limited intervention family; they do not supply a reduced high-level state
map. Non-additive output effects do not disprove compositional circuits:
a correct circuit can explicitly compute interactions. It is the independent
contribution approximation that failed. Joint component masks are passed in
one call; arbitrary nesting of Python hooks is not an intervention-algebra
implementation or a proved commutativity claim.

The neighboring searches revisited [CLUE](https://arxiv.org/abs/2004.11961),
[weighted tree automata](https://proceedings.mlr.press/v51/rabusseau16.html),
tensor-train/Hankel realization, and [contraction treewidth](https://arxiv.org/abs/quant-ph/0511069).
Invariant-space closure is relevant to supplied linear transition operators,
and automaton minimality to a supplied multilinear realization. We have
neither for the RMS/softcap network. Contraction ordering evaluates an
existing graph; it does not identify a simpler causal program. These are
not serious new reduction candidates without the missing object mappings.

## Executed consequences

The full-response mode run has native KL replay0, reconstruction1.78e-15,
Parseval discrepancy3.64e-12, and future-command modes exactly0 at the earlier
query. Minimum invariant-response errors are .374/.373 all-token,
.245/.223 temporal and .478/.514 iswas, FIT/HOLDOUT. Small answer/foil
effects therefore do not establish a task-independent shared contribution.

The new integer incidence audit checks795 aligned source positions across
all32 worlds. Every position has count00+count11=count01+count10, so ANY
token-only map phi, including v0, has zero mixed command mode. Constant and
independent-token controls pass; an XOR-token fixture correctly fails.
The native shared-removal effect nevertheless has a mixed mode. Its M11
norm gives an additive-response error floor .07276/.07215 all-token and
.13696/.15365 at the later iswas query. These are full-response L2 bounds,
not KL bounds; no lower bound is asserted at the earlier causal-zero query.

This locates the interaction downstream of the shared producer. It does
not yet locate one attention reader or prove native read mixing suffices.

## Concrete next operation: crossed routing and payload modes

At a fixed head/query let S_ab=P_ab U_ab, contracting over source positions,
where U_ab is the shared value projected through that head's output matrix.
Character multiplication gives an exact convolution of their four modes:

    S_hat11 = P_hat11 U_hat00 + P_hat01 U_hat10
              + P_hat10 U_hat01 + P_hat00 U_hat11.

The incidence certificate makes U_hat11 zero. The remaining terms distinguish
a genuinely two-command native router from crossed single-command routing
and payload terms. This is an executable cross-boundary computation with
explicit inputs and output, not a request for another rank sweep.

Next capture the native shared reads at all four fixed heads and both command
queries. Validate the convolution and native read, then test whether crossed
terms alone predict the mixed read within1%. Keep any joint-router term
needed by the measured full-vector error. No output/head subset selection
or structural credit from a four-cell table. Native routing-state production
and all weights remain charged. A passing read decomposition would still
need causal removal/interchange and new-world execution before identification.

The finite projection theorem dominates fitting an invariant baseline: it
already gives that family's best possible error. The source-incidence proof
now makes the reader-convolution test more informative than more global
shared-value ablations. Its algebraic primitive and fixed protocol are the
immediate next work. Next mathematical checkpoint is due01:49 UTC on Sep10;
the independent hourly checkpoint remains due22:58 UTC on Sep9.
