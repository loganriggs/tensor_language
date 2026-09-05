# Three-hour mathematical circuit review — 2026-09-05 23:35 UTC

## Decision

The useful mathematical object is now a composed finite intervention program with a two-site mediator lattice, not a tensor-rank factorization. The upstream program is indexed by grammatical direction and source cardinality; the downstream mediator response is adequately indexed only by direction. Exact finite differences identify MLP15, MLP17, and their interaction operationally, while a six-scalar law predicts their task effect from the frozen reader. The next theorem-to-experiment consequence is prospective transfer of this composed law on a fourth authority.

## Explicit program and response tensor

Let direction be (d\in\{-1,+1\}), upstream background cardinality (k\in\{0,1,2,3,4\}), and (p_{d,k}\in\mathbb R^{1152}) the fixed projected head-11.3 write. Let (r_d\in\mathbb R^{1152}) be the fixed downstream reader. The extracted upstream/readout program predicts

\[
\widehat q(d,k)=\langle r_d,p_{d,k}\rangle,
\]

with ten write vectors and two readers already fixed before the third-corpus causal outcomes. The live model installs (p_{d,k}) at the L11H3 projected-write interface and yields a contextual behavioral effect (q_i) for row/background cell (i).

For mediator clamp set (S\subseteq\{15,17\}), define

\[
G_i(S)=\bigl[y_i(\text{program},\operatorname{clamp}_S)-y_i(\text{base},\operatorname{clamp}_S)\bigr],
\]

where (y_i) is the signed answer-minus-foil margin and each clamped MLP output is set to its matched base-program final-token value. The operational effects are

\[
m_{15,i}=G_i(\varnothing)-G_i(\{15\}),\quad
m_{17,i}=G_i(\varnothing)-G_i(\{17\}),
\]

\[
m_{15,17,i}=G_i(\varnothing)-G_i(\{15,17\}),\quad
I_i=m_{15,17,i}-m_{15,i}-m_{17,i}.
\]

The screened scalar composition is

\[
\widehat m_{15,i}=a_{15,d_i}\widehat q_i,\quad
\widehat m_{17,i}=a_{17,d_i}\widehat q_i,\quad
\widehat I_i=a_{I,d_i}\widehat q_i,
\]

and (\widehat m_{15,17,i}=\widehat m_{15,i}+\widehat m_{17,i}+\widehat I_i). The six fitted gains are approximately `(0.03436, 0.05862)` for MLP15, `(0.12297, 0.20482)` for MLP17, and `(-0.01094, -0.05627)` for the interaction in singular-to-plural and plural-to-singular directions respectively. These are operational causal-effect gains, not reconstructions of MLP activation vectors.

The literal stored candidate has 13,824 upstream-vector scalars, 2,304 reader scalars, and six mediator scalars. Its contraction graph is: choose ((d,k)), add (p_{d,k}) at the projected head interface, contract with (r_d) for (widehat q), multiply by the three direction gains, and sum the mediator terms. The native model remains necessary to supply the contextual base state and to test the intervention, so this is an interface program rather than a whole-model replacement.

## Exact theorem mapping

Rota's incidence-algebra Möbius inversion applies exactly to the Boolean poset (2^{\{15,17\}}) ([Rota, 1964, primary paper](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/rota1.pdf)). The zeta values are the four experimentally observed (G_i(S)). The singleton finite differences and the mixed coefficient

\[
\Delta_{15}\Delta_{17}G_i=G_i(\{15,17\})-G_i(\{15\})-G_i(\{17\})+G_i(\varnothing)
\]

are unique for this operational clamp lattice; our signed interaction is (I_i=-\Delta_{15}\Delta_{17}G_i). No smoothness, linearity, or neural-network approximation assumption is required. Complete four-corner coverage and deterministic intervention semantics are sufficient.

The causal-mediation literature supplies an important boundary rather than an identification shortcut. Robins and Greenland show that direct and indirect effects are generally not separated merely by randomizing an exposure and that controlled intervention on the mediator requires explicit assumptions ([Robins & Greenland, 1992](https://pubmed.ncbi.nlm.nih.gov/1576220/)). Pearl formalizes path-specific direct and indirect effects in nonlinear models ([Pearl, 2001](https://arxiv.org/abs/1301.2300)). Our mapping does **not** claim a natural indirect effect: the upstream write and MLP output clamps are both directly executed, so (G_i(S)) is an interventional engineering quantity on a fully specified neural computation. It identifies what the clamp changes, not a unique semantic causal pathway under alternate mediator definitions.

The scalar law is an approximation in the endpoint norm, not an exact consequence of Möbius inversion. For a held-out set (H), its error is

\[
\frac{\lVert m_H-X_Ha\rVert_2}{\lVert m_H\rVert_2},
\]

where (X_H) contains the sealed (widehat q_i) gated by direction. Row-held-out evaluation gave joint cosine `0.95337`, relative error `0.30185`, and perfect signs. Adding cardinality to (X) reduced SSE only `0.1018%`; hence the 30-scalar model is not justified by the chosen norm.

## What is and is not identified

The complete mediator lattice identifies the total controlled effect of replacing final-token MLP15 and MLP17 outputs with matched base values, their singleton effects, and their mixed finite difference. It establishes that both singleton effects recur and that MLP17 is larger in the measured endpoint norm. It does not establish unique internal features, necessity under all alternative paths, equivalence to natural language number semantics, invariance to a different clamp definition, or a low-dimensional activation subspace.

The six gains identify a predictive quotient only on the span of the fixed ten program writes and the signed Task14 endpoint. Their success cannot be promoted to ambient vector equality. Conversely, generic SVD/Tucker/tensor-train compression would optimize a coordinate norm without answering whether a factor predicts these controlled endpoint effects, so it remains the wrong immediate tool.

## Executable consequence and falsifier

The next experiment must freeze the six gains above and construct a fourth authority before any new mediator outcome is opened:

1. choose 32 new one-token singular/plural noun pairs disjoint from all prior Task14 corpora and use a syntax template absent from the third corpus;
2. establish native capability per direction×template cell;
3. install each frozen (p_{d,k}), compute only the frozen-reader (\widehat q), and seal all (\widehat m_{15}), (\widehat m_{17}), (\widehat I), and joint predictions;
4. execute the same eight-corner within-batch clamp lattice;
5. require exact base/group replay and apply the unchanged v2 component, joint, sign, and per-group bars without refitting gains.

Failure of native capability or replay invalidates the instrument. Passing capability but missing the joint or group bars falsifies prospective scalar composition and retains the six gains as a retrospective screen only. Passing promotes a fixed upstream-write/readout/mediator program across corpora. No gain recalibration, cardinality restoration, row filtering, or new rank fit is allowed after outcomes.

## Route comparison

Exact four-corner intervention dominates observational mediation because both mediator values are controllable and every lattice corner is affordable. Direction-only scalar composition dominates direction×cardinality because the larger model supplied essentially zero held-out SSE benefit. A full native-weight contraction through MLP15 and MLP17 would become worthwhile only after the scalar endpoint law transfers; before then it would add algebra without a stable target. Cross-circuit composition remains the next broader theorem test after fourth-corpus promotion: the mixed finite difference of two independently fixed programs will directly test whether their operational state transitions commute or interact.
