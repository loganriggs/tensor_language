# Three-hour mathematical circuit review — 2026-09-06 08:30 UTC

## Decision

The aspectual-anchor evidence is best represented as a **controllability/observability factorization**, not as equality of read and write coordinates. One identified rank-one actuator, `q_has`, is selectively controllable by the fixed `has`/`had` reader and is observable through both `has`/`had` and `is`/`was` margins. Conversely, a direction fitted directly to `is`/`was` is not selective: it also moves the answer-preserving temporal-paraphrase family by 0.644–0.659. This asymmetry is compatible with a shared actuator seen by several readouts and does not license a shared upstream reader, a second circuit, or a rank increase.

The next experiment should therefore keep rank one and solve a **selective constrained identification** problem for `is`/`was`: maximize held-in causal recovery while explicitly suppressing response on the registered P and C nuisance families. Its opposing predictions distinguish a genuinely shared writer from output-gradient contamination. This changes circuit grouping/reuse and selective-manipulation evidence; it is not a compression sweep.

## Current mathematical object

Let the checkpoint width be \(D=1152\), with query-position states \(x_{10},x_{18}\in\mathbb R^D\). Let \(N(z)=z/\sqrt{D^{-1}\|z\|_2^2+\epsilon}\) be the model's exact RMS normalization and \(s\) its exact soft-cap. With fixed unembedding rows \(w_t,w_f\), define the signed output functional

\[
m_{t,f}(x)=s(w_t^\top N(x))-s(w_f^\top N(x)).
\]

Program v12 computes, for direction \(d\in\{\mathrm{present\to past},\mathrm{past\to present}\}\),

\[
c_h(x_{10})=s(w_{has}^\top N(x_{10}))-s(w_{had}^\top N(x_{10})),
\qquad
\alpha_d=a_d+b_dc_h(x_{10}),
\qquad
x'_{18}=x_{18}+\alpha_dq_h,
\]

where \(q_h\in\mathbb R^{1152}\) is unit norm and the four fitted coefficients are fixed. The two tested output pairs are \(h=(has,had)\) and \(i=(is,was)\). For row \(r\), output family \(k\), and direction \(d\), the finite causal response is

\[
R_{kdr}(q,\alpha)=m_k(x_{18,r}+\alpha_{dr}q)-m_k(x_{18,r}).
\]

The contraction graph is `raw text -> native transformer prefix -> x10 -> fixed local head contrast -> two-scalar affine controller -> alpha*q_h write at x18 -> exact RMS/head output`. The retained interface does not replace the native prefix or background suffix. Its stored program state is 1,157 scalars: 1,152 for \(q_h\), four affine coefficients, and one inherited budget scalar. Runtime beyond the native background is two fixed-row head contractions, two scalar affine operations, and one 1,152-coordinate scaled write; there is no grid, donor activation, confirmation-margin lookup, row-outcome lookup, backward pass, or parameter update.

The controller is degree one in the scalar \(c_h\); with fixed \(q_h\), the write is affine in \(\alpha\). The complete map is nonlinear because \(N\) and \(s\) occur at both read and output. If both \(q\) and \(\alpha\) were variables, their product would be bilinear. Its elementary gauge is \((q,\alpha)\mapsto(-q,-\alpha)\). More generally, an invertible hidden-state change of basis is observationally irrelevant only if all readers, writes, and native suffix maps transform jointly. The operational quotient identifies directions by their complete registered causal-response vector, not by coordinates or cosine alone.

Allowed program inputs are the native \(x_{10}\), native \(x_{18}\), the requested direction, the fixed checkpoint head rows, and frozen parameters. Preserved outputs are signed A1/A2 recovery, P reflection, C invariance, direction fractions, and exact native-head replay on every retained row. Approximation is scored by population means and directions under the registered normalized causal margins, not hidden-state reconstruction. The intervention price is one rank-one edit at one site per example.

## What the response geometry says

For small dose \(\alpha\),

\[
R_{kdr}(q,\alpha)=\alpha\,\nabla m_k(x_{18,r})^\top q+O(\alpha^2).
\]

Thus a single write direction can be visible through several output functionals whenever their local gradients have nonzero projection on it. This does **not** imply that those readouts define the same direction, nor that a direction optimized against one readout is selective. The finite-dose result is stronger than this tangent analogy: v12's 0.5/1.0/1.5 doses produced ordered effects, and the unchanged \(q_h\) transferred to two capability-qualified `is`/`was` lexicons.

On its native prospective `has`/`had` v5 panel, \(q_h\) gave A1/A2 0.854/0.858, P 1.003, and C 0.00214. Through the distinct `is`/`was` functional it gave A1/A2 0.377/0.366 and P 0.373 on v2, and 0.387/0.358 and 0.389 on v3, with C 0.00214. In contrast, the independently fitted \(q_i\) gave strong `is`/`was` A recovery (held-out A1 0.975/1.003 and A2 0.606/0.611) but P 0.644/0.659. Its cosine 0.221 with \(q_h\) is descriptive only because \(q_i\) failed identification.

The correct local analogy is \(x^+=Ax+Bu\), \(y_k=C_kx\): \(q_h\) is a one-column intervention map \(B\), while each output margin supplies a state-dependent observation row \(C_k(x)=\nabla m_k(x)^\top\). A nonzero \(C_i(x)q_h\) explains cross-readout action without requiring \(C_i=C_h\), and an unconstrained target fit can absorb a nuisance direction that also has large P response.

## Exact theorem mappings and their limits

### Linear realization and controllability/observability

Kalman's original state-space treatment separates controllability, a property of the state dynamics and input map, from observability, a property of state dynamics and output map, and makes both invariant under algebraic changes of state coordinates ([Kalman, 1963](https://people.duke.edu/~hpgavin/SystemID/References/Kalman-JSIAM-1963.pdf)). Under a finite-dimensional LTI model with complete input/output experiments, a minimal realization is the reachable-and-observable quotient and is unique up to similarity.

Object mapping: transformer residual state maps to system state; a registered residual edit maps to an input column; output-margin derivatives map to observation rows; and the empirical intervention/readout table maps locally to Markov parameters. The theorem assumes linear state transition, linear time-invariant input/output maps, complete excitation, and exact access to the sequence response. We have a nonlinear transformer, state-dependent RMS/soft-cap readouts, one checkpoint, two sites, finite doses, and a sparse registered population. Therefore no Kalman minimality or global uniqueness result applies. What does survive exactly is the conceptual separation of actuator and readout and the sign/similarity gauge warning.

### Predictive-state representations

Littman, Sutton, and Singh show that controlled stochastic systems admit linear predictive-state representations whose state is a set of action-conditional predictions, with dimension no greater than a minimal POMDP state count ([NIPS 2001](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)). Object mapping: our causal-response coordinates are predictions under registered residual interventions, and a sufficient circuit state would predict all retained A/P/C responses. Their result requires a controlled dynamical-system/test structure and a complete core set of predictions. Our prompt-to-single-site intervention table is neither a proven controlled process nor complete, so it supplies a design principle—define state by intervention predictions—not a dimension certificate.

### Hankel-rank minimal realization

For a rational series computed by a weighted finite automaton, the infinite prefix/suffix Hankel matrix has rank equal to the number of states of a minimal realization; a complete finite sub-block and rank factorization recover a minimal automaton (the construction is summarized with formulas in [Arrivault et al., 2017](https://proceedings.mlr.press/v57/arrivault16.html), building on the classical theorem). Object mapping: registered cue/intervention histories would index prefixes, output/control probes would index suffixes, and their causal responses would fill a Hankel-like matrix. The guarantee requires a rational series, linear symbol transitions, and a complete basis. None is established for this nonlinear fixed-site transformer experiment. Consequently, SVD rank of the present response table would be only a probe statistic, not circuit identification or a lower bound on transformer state.

### Causal abstraction

Interchange-intervention training aligns high-level variables with neural representations and evaluates matched counterfactual behavior; zero interchange loss gives the stated causal-abstraction guarantee under the alignment ([Geiger et al., ICML 2022](https://proceedings.mlr.press/v162/geiger22a.html)). Object mapping: the scalar controller, direction input, and rank-one write are proposed high-level variables/mechanisms, while residual replacement is the low-level intervention. The theorem's universal matched-intervention condition is not met by finite A/P/C panels, and v12 uses an additive soft intervention rather than a full interchange value. Hence current evidence identifies a bounded operational abstraction, not a universal causal abstraction.

No reviewed theorem exactly solves the Theseus object. The closest exact restriction is a local linear causal-response system, and its useful output is a falsifiable selective-factor test—not a global rank claim.

## Executable consequence: selective rank-one identification

Construct fit-set differentiable response losses using capability-qualified `is`/`was` rows. Let \(L_A(q)\) reward the signed A target effect, while \(L_P(q)\) and \(L_C(q)\) measure squared finite responses on answer-preserving and unrelated controls, all normalized exactly as in evaluation. Solve one preregistered constrained problem

\[
\max_{\|q\|_2=1} L_A(q)
\quad\text{subject to}\quad
L_P(q)\le \tau_P,\;L_C(q)\le\tau_C,
\]

using a fixed augmented-Lagrangian schedule, not a hyperparameter/rank sweep. Fit rows and held-out rows remain disjoint; neither \(q_h\), cross-transfer outcomes, nor held-out outcomes enter optimization. Before GPU execution, a CPU toy must verify that the solver recovers an actuator shared across two observation rows while rejecting an A-correlated nuisance direction.

Opposing registered outcomes:

1. **Shared selective writer:** the constrained \(q_i^*\) passes held-out `is`/`was` A1/A2 and P/C; then \(|q_i^{*\top}q_h|\ge0.50\) and frozen reciprocal transfer to `has`/`had` is at least 0.25. This would group a shared actuator across output vocabularies while retaining distinct readers.
2. **One-way shared sensitivity:** \(q_i^*\) is selective on `is`/`was`, but cosine is at most 0.20 and reciprocal transfer at most 0.20. Then `is`/`was` observes \(q_h\), but independently controllable selective writers differ.
3. **No selective standalone rank-one writer:** held-out A remains below 0.50 or P exceeds 0.20. This preserves the present boundary and forbids a rank increase as a response.

The constrained test is more informative than another unconstrained fit, cosine audit, or response-matrix SVD because every branch changes grouping, selective-manipulation, or reuse evidence. It is cheaper than broad construction OOD work and directly targets the failure just observed. The live plan therefore redirects from immediate reciprocal transfer to CPU solver falsification, preregistration, then one managed GPU fit if the toy and static gates pass.

The next mathematical review is due around **2026-09-06 11:30 UTC**. The next hourly strategic review remains due around **09:17 UTC**.
