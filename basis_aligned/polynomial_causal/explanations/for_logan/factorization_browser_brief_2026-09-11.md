# Self-contained browser-Codex brief: joint factorization of a bilinear layer and its unembedding

**Context snapshot: 11 September 2026, approximately 13:50 UTC.** Paste this entire file into browser Codex. The request and mathematical object below are self-contained; no local repository access is assumed.

## What I want help with

I want to reverse-engineer a trained language model into simpler, reusable computations. Please assess the structural factorization choices and propose strong, scalable optimization methods. In particular, assess **output-sharing symmetric LL1/block-term decomposition**, including adaptive block counts/ranks. Do not merely suggest increasing CP iterations or gathering more text.

Discover structure from **weights first**, freeze candidates, then validate on data. The model was trained on **FineWeb**. Pile would be a separate distribution-shift check; historical Pile-adapted candidates cannot claim untouched Pile validation. The eventual circuits must support OOD prediction, extraction/sufficiency, selective removal, and composition/reuse. Reconstruction or compression alone is not success.

## Exact object

The model has 18 blocks, residual width d=1152, bilinear width n=4608, and V=50304 output vocabulary rows. We currently focus on the last bilinear MLP, not the full attention-to-logit model.

At its already-normalized input x:

$$
q(x)=D[(Lx)\odot(Rx)]+b,
\quad L,R\in\mathbb R^{4608\times1152},
\quad D\in\mathbb R^{1152\times4608},
\quad U\in\mathbb R^{50304\times1152}.
$$

Folding the entire unembedding U into D gives a partially symmetric third-order coefficient tensor:

$$
T_{vij}=\sum_{k=1}^{4608}(UD)_{vk}
\frac{L_{ki}R_{kj}+R_{ki}L_{kj}}2,
\qquad F_v(x)=x^\top T_vx.
$$

Output-first axis order is (vocabulary,input,input). Both input modes act on the **same x**. Only the symmetric part contributes. Factors are real and signed. Bias is retained separately. Actual logits also require residual addition, final RMS normalization and a 30*tanh(logit/30) cap. Therefore the full model is not this quadratic tensor.

Default discovery objective:

$$
\frac{\sum_v\|T_v-\widehat T_v\|_F^2}{\sum_v\|T_v\|_F^2}.
$$

It can be computed implicitly using U^T U and

$$
\langle\operatorname{sym}(ab^\top),\operatorname{sym}(cd^\top)\rangle
=\tfrac12[(a^\top c)(b^\top d)+(a^\top d)(b^\top c)].
$$

Do not propose materializing the approximately66.8billion-entry tensor. The output span has dimension at most1152; metric-preserving output coordinates remove the vocabulary dimension without changing the objective. This does **not** bound product rank by1152. At most1152 unrestricted dense quadratic functions already suffice, and the native4608products are already fewer than vocabulary size; neither observation discovers simple input arithmetic.

## Structural alternatives to distinguish

1. **Shared products:** F=sum_k c_k(a_k^T x)(b_k^T x). A general symmetric product is a tied pair of ordinary CP terms. A square is one partially symmetric CP term. Compare literal costs, not these counts interchangeably.
2. **Output-sharing symmetric LL1:** F=sum_g c_g(x^T Q_g x), Q_g symmetric with rank<=L_g. This has mode ranks bounded by (1,L_g,L_g) with output first, or conventional (L_g,L_g,1) with output last. Each group allows multiple input directions/interactions but one output vector. Input subspaces and token loadings may overlap. Parameterize Q_g=A_g H_g A_g^T with symmetric possibly indefinite H_g. If instead Q_g is a sum of p two-reader symmetric products, its rank is at most2p, not p. Symmetrizing an arbitrary AB^T can likewise double its rank.
3. **Shared-input groups, the newest implemented family:** F=sum_g(a_g^T x)M_gx, rank(M_g)<=r_g, M_g=W_gV_g. One input reader is shared across several partner products and outputs. After symmetrization, mode-rank bounds are (r_g,r_g+1,r_g+1), with additional structure. This is not output-sharing LL1.
4. **Shared feature dictionary:** F=UD[(C_L Bx)⊙(C_R Bx)]. Many products reuse linear features; savings need not come from reducing product count. More general overlapping blocks or multiple sparse stages may expose reuse beyond CP rank.

Example: F=c1(x1*x2+x3*x4)+c2*x1*x3 has three products, two output-sharing scalar blocks, or shared-input groups x1(c1*x2+c2*x3)+x3(c1*x4). Different decompositions can expose different meaningful reuse.

## What has actually been tried

- Free shared product banks, signed squares, sparse token-function dictionaries, orthogonal/nonorthogonal input dictionaries, small overlapping/core blocks, and prescribed sparse linear stages. Some fit locally; others remain unconverged. These are not a matched-capacity leaderboard or an exhaustive LL1 search.
- Strongest completed native shared-feature program:2304features,128connections per each of9216left/rightreaders,4608product slots. Direct full-tensor variable projection solves output weights exactly at each L-BFGS update. Two hour-long starts capture64.685/64.687% of squared coefficient energy, but miss joint convergence. One connection-exchange sweep reachesabout65.00%; complete-function cosine between starts remains.747. “Capture” is1-relative squared error, not accuracy.
- A component penalty cuts large component energies while preservingcapture, but both20minstarts remain unconverged and complete-function cosineabout.745. No global optimum claim.
- A matched-size2645native-product baseline with exact output refitting captures68.63%, beating the dictionary on weights. Frozen validation at128historically openedFineWebpositions reverses the ranking: addedCE +.3546nats for native selection versus+.0119/+.0358 for the dictionaries; KL .2940 versus.0212/.0224. No data fitting, fresh/OODclaim, or identified selective circuit follows.
- One scalar function agrees across fits and partly recovers known pronoun-related products. It is not a new gender circuit; prior sufficiency/removal failures and a punctuation association remain relevant.
- New shared-input group tests are **planted only**. Exact conditional partner SVD and unit-sphere reader updates pass dense controls. Some starts recover, others fail. A joint small dense-Jacobian trust-region fit develops summedgroupenergy1.24milliontimes targetenergy while retaining4.82%error.
- Whole-group penalty eta*sum||G_g||², eta=.01, prevents that growth and is invariant to internal partner-basis changes. A bad warm start reaches smallgradient yetpoor objective. Across8randomstarts,3recover low-objective plantedgroups; best selected solely by weight objective hasrelativeerror.0001604 andmatchedgroupcos.999752. Recovery-rate target4/8fails. Local stationarity is not enough.
- A native64shared-input-group/rank8pilot is in draft, not yet queued at this snapshot. A broad adaptive-rank output-sharing symmetricLL1whole-tensorfit has **not** been established as exhausted.

## Questions to answer

1. Is symmetric output-sharing LL1 the right mathematical family for “multiple input interactions produce one output variable”? What alternatives capture shared input/output spaces with fewer unnecessary restrictions?
2. Map ordinary LL1/BTD definitions and actual recovery/uniqueness assumptions onto this signed, same-input symmetric tensor. Which guarantees survive symmetry, overcomplete/overlapping blocks, and the1152-dimensional outputspan? Distinguish sufficient-condition failure from nonidentifiability.
3. Propose the strongest practical solvers and initializations for this implicit tensor: generalized eigenvalue/block-diagonalization approaches where assumptions apply; structured nonlinear least squares/Gauss–Newton; variable projection; hierarchical rank selection or group sparsity; suitable preconditioning and restart strategies. Give complexity and what must be materialized. We have a good GPU; use it where beneficial.
4. How should block countG and ranksL_g be selected without hiding dense input complexity inside a small number of outputs? Compare storage, arithmetic, shared readers, and gauge freedoms. Avoid claiming a globally minimal decomposition from an ordinary local optimizer.
5. How should we prevent or diagnose diverging cancellations, stationary bad basins, and unstable factor identities? Propose planted controls that genuinely challenge recovery and meaningful native convergence checks.
6. Give a concrete prioritized experiment plan, with weight-only discovery first and frozen FineWeb validation afterward. Explain how candidate groups could progress to extraction, selective manipulation, OODprediction and composition, rather than ending with a reconstruction score.

Please cite primary literature and map the algorithms to these equations. Separate implemented methods, promising literature leads, and unverified guarantees.

## Primary starting references

- [Tensorlab LL1 definition and algorithms](https://tensorlab.net/doc/ll1.html).
- [Tensorlab constrained/coupled LL1 examples](https://tensorlab.net/doc/sdf-examples.html).
- [Kolda and Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/koba09/).
- [Rontogiannis, Kofidis and Giampouras, Block-Term Tensor Decomposition: Model Selection and Computation](https://arxiv.org/abs/2002.09759). A candidate adaptive block-count/rank method; it has not been implemented for this model.

The remaining open problem is not whether some tensor factorization can be written down. It is whether we can reliably find a simpler computational organization of these learned weights and then demonstrate that its groups are useful circuits.

**13:54 implementation addendum:** a small CPU symmetric LL1 conditional-update control now exists. For fixed c, contract E against c/||c||² and keep the largest-absolute signed eigenvalues; for fixed Q, c_o=<E_o,Q>/||Q||². Numerical checks pass and two near-initialized planted blocks recover to4.63e-13error. This does not supply native/global recovery. Robust zero cases, implicit native contractions, adaptive rank selection and reliable initialization remain to implement.
# Follow-up context: hierarchy and DAG discovery

Logan now asks whether reusable arithmetic should be discovered jointly with factorization, and whether dense or sparse Tucker must come first. Our proposed answer: fit an explicit acyclic graph of shared linear nodes, products of two linear nodes, and shared linear combinations of quadratic nodes directly against the implicit folded weight tensor. Alternate continuous coefficient fitting with discrete graph changes and basis changes. Price each computed node once, including constants, connections, additions, products, adapters, and storage. Sparse Tucker is a comparator, not a prerequisite; fixed-basis core sparsity misses distributive structure such as $(a+b)(c+d)$.

An exact five-variable CPU toy transformed $(a+b)(c+d)$ and $(a+b)e$ by invertible input/output mixing. Both output slices then had all15quadratic monomials. Symbolic factoring recovered two products and their common linear parent exactly. This is a restricted exact integer control, not a native floating-point search or unique recovery of the historical graph. Real weight sharing will generally be approximate and basis-dependent.

Current native matched pilots (about1.25Mparameters each): output-sharing LL1 captures11.66/11.64%, shared-input groups8.53/8.54%; allfour120-second fits are unconverged. These restricted families do not yet implement general joint DAG discovery. Questions for browser Codex: which tractable graph proposals discover approximate common linear/quadratic subexpressions? How should we combine basis optimization, graph-aware extraction of exact rewrites, and continuous refitting without mistaking a sparse representation for an identified circuit? Which planted overlapping/noisy examples discriminate the solvers? Keep discovery weights-first; freeze before FineWeb validation.
