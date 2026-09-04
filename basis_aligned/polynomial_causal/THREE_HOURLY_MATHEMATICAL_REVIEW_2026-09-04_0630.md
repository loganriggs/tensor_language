# Three-hour mathematical review — 2026-09-04 06:30 UTC

## Goal and current boundary

The goal is not to replace a module by fewer dimensions.  It is to recover a smaller executable tensor program whose
parts have a computational meaning and which:

1. predicts the computation on held-out and out-of-distribution inputs;
2. groups pieces of different heads or MLPs when later computation treats them as the same variable, and splits one
   module when its pieces have different uses;
3. reproduces the target computation when extracted or installed;
4. permits selective swaps and removals without comparably changing unrelated circuits;
5. composes and reuses recovered computations across tasks; and
6. is stable across data splits, fitting runs, and changes of internal coordinates.

Task 17 gave a valid capability failure.  The independently reviewed task-21 authority now tests local verbatim
repetition: the correct next token is the token immediately before the answer position.  It is a useful strict-pipeline
anchor, but a capability pass would not establish induction, remote retrieval, or an attention-head circuit.  A direct
embedding-to-residual path can already carry the needed token identity.

The mathematical step-back is therefore: **define the state by everything the model will observably do with it, rather
than by a head boundary, an eigenvector cutoff, or a convenient rank.**

## Predictive equivalence gives a basis-independent state

Let $u$ denote the text seen so far.  Choose a registered set of future tests $\mathcal T$.  A test can be an ordinary
continuation, a donor swap, or a later-reader intervention.  Let

$$
F(u,t)
$$

be the resulting observable output, such as the vector of candidate-token logit differences, for test $t\in\mathcal
T$.  Two histories are the same operational state exactly when

$$
u\sim v
\quad\Longleftrightarrow\quad
F(u,t)=F(v,t)\quad\text{for every registered }t\in\mathcal T.
$$

This is the useful notion of “the same thing.”  It can merge outputs from different heads if every later test treats
them identically, and split one head if two of its directions have different future effects.  Predictive-state
representations similarly define state by predictions of observable tests, rather than an arbitrary hidden coordinate
([Singh, James, and Rudary](https://arxiv.org/abs/1207.4167)).  Computational mechanics calls pasts equivalent when
they imply the same future distribution and proves minimality and uniqueness properties for the resulting causal-state
representation ([Shalizi and Crutchfield](https://arxiv.org/abs/cond-mat/9907176)).

The choice of tests is substantive, not automatic.  Candidate-token logits alone identify what is needed for the next
answer.  Later-reader clamps, swaps, and unrelated-circuit controls are needed to distinguish two implementations that
make the same answer but interact differently with the rest of the model.

## The Hankel construction and why this is not generic rank reduction

For a string-to-output function $f$, form a Hankel object indexed by prefixes $p$ and suffixes $s$:

$$
H(p,s)=f(ps).
$$

For a weighted finite automaton, finite Hankel rank is equivalent to the existence of a finite-dimensional linear
realization, and that rank is the size of a minimal realization.  A complete finite block can be factorized as

$$
H=PS,\qquad M_a=P^+H_aS^+,
$$

where $H_a(p,s)=f(pas)$, $P^+$ and $S^+$ are pseudoinverses, and $M_a$ is the state update for symbol $a$.  This is the
spectral weighted-automaton construction described by
[Arrivault et al.](https://proceedings.mlr.press/v57/arrivault16.pdf); black-box language models can also be
approximately minimized through sampled Hankel blocks
([Lacroce, Panangaden, and Rabusseau](https://proceedings.mlr.press/v153/lacroce21a.html)).

This use of rank differs from the rejected 768-eigenvector route.  There, rank measured how many activation directions
were retained without specifying a computation.  Here, rank measures the number of linear predictive coordinates
needed for one declared input-output function over declared future tests.  Equality of rows has an operational
meaning: no registered continuation or intervention distinguishes them.  It therefore bears directly on prediction,
grouping, and stable identification.  It still does not by itself locate or selectively remove a physical circuit.

There is also a direct tensor-network connection.  Linear second-order recurrent networks and vector-valued weighted
automata are expressively equivalent; their Hankel tensors have tensor-train structure and can be recovered under
complete-basis and rank assumptions
([Rabusseau, Li, and Precup](https://proceedings.mlr.press/v89/rabusseau19a.html)).  This supplies exact mathematics for
an appropriate linear sequential function.  It does **not** make the entire bilin18 transformer a weighted automaton:
attention, finite context, and RMS normalization make the network nonlinear, and a small sampled Hankel block can
underestimate the state required outside the sampled language.

## Exact task-21 consequence

Task 21 uses 21 registered one-token candidates per phase.  For each evaluated prompt side, a future localization
instrument should retain the complete candidate response

$$
r_a(u)=\ell_a(u)-\frac{1}{21}\sum_{b=1}^{21}\ell_b(u),\qquad a=1,\ldots,21,
$$

where $\ell_a$ is candidate $a$'s logit.  Subtracting the mean removes the irrelevant freedom to add the same constant
to every candidate logit.  The current capability result retains only the correct-answer logit and the largest foil,
which is enough to decide capability but cannot reveal whether two prompts have the same full response.

For an ideal previous-token copier, there are 21 distinguishable semantic states—one for each preceding token.  Their
mean-centered one-hot response vectors span 20 linear dimensions because the 21 coordinates sum to zero.  Those two
numbers answer different questions: 21 is the number of discrete answer identities; 20 is the dimension of their
logit-difference representation.  Neither is a fitted cutoff.

The linked panels then test what else is present:

- A1 changes the entire repeated run;
- A2 changes the newest run while leaving an older conflicting token visible;
- P changes a leading filler while preserving the answer; and
- C changes repeat count while preserving answer identity.

If C changes only a common logit offset, mean-centering removes it.  If it changes confidence along the same token-ID
pattern, it is a continuous strength variable.  If it changes the full response in another direction or alters later
reader effects, it is an additional predictive variable.  This gives a concrete, falsifiable meaning to the earlier
“continuous function” idea.

Retaining 21 `float32` values for all 168 FIT row-sides would cost

$$
168\times21\times4=14{,}112\text{ raw bytes},
$$

which is negligible next to model execution.  This is a prospective localization requirement only.  It does not alter
the already frozen 1,344-byte capability screen and does not authorize opening localization before capability passes.

## Mapping the predictive state back to weights

If capability passes, the physical question is whether some model subspace carries the candidate-response state.
For a proposed residual projector $P$ and matched recipient/donor prompts $x,y$, interchange only

$$
Pz(x)\leftarrow Pz(y)
$$

and require the entire 21-candidate response vector—and the selected later-reader responses—to move toward the donor.
Necessity then clamps that state back to the recipient value; sufficiency injects it without the upstream donor.  P/C
controls test whether answer identity is separated from irrelevant context and repeat strength.

At a bilinear MLP reader, the observed normalized-state change $\delta=\bar z_1-\bar z_0$ translates exactly to weights:

$$
\begin{aligned}
M(\bar z_1)-M(\bar z_0)=W_D\big[&
(W_L\delta)\odot(W_R\bar z_0)
+(W_L\bar z_0)\odot(W_R\delta)\\
&+(W_L\delta)\odot(W_R\delta)\big].
\end{aligned}
$$

Thus the predictive state proposes *what* must be preserved; interchange and the exact quadratic expansion establish
*where and how* the model implements it.  The state is accepted only if the same frozen map predicts held-out/OOD
responses, survives multiple valid donor choices, permits selective edits, and composes with another recovered state.

## Falsifiers and ranked decision

This route is rejected for task 21 if:

1. native capability fails;
2. the 21-candidate response does not generalize under the frozen A1/A2/P/C organization;
3. a proposed state works for one donor but not other valid donors with the same high-level change;
4. direct final-token residual information explains the effect and the proposed attention path adds no selective
   predictive value;
5. the state cannot be injected and clamped with matching signed effects; or
6. the intervention changes unrelated circuit controls comparably to the target.

The ranked action remains:

1. build and independently review the task-21 model-facing capability adapter;
2. run the frozen eight-forward capability screen through the managed queue only after that approval;
3. if it passes, preregister a localization experiment retaining the full 21-candidate response and explicitly compare
   the direct residual path against attention/MLP alternatives;
4. if it fails, close it unchanged and move to the richer subject–verb agreement authority; and
5. use the same predictive-equivalence construction on a remote-retrieval behavior, where a direct previous-token path
   cannot solve the task.

The immediate action taken at this review boundary is concrete: the approved authority/compiler has been handed to a
fresh CPU-only adapter build, while execution remains closed.  The separate localization requirements below prevent a
future capability pass from silently collapsing back into head attribution or arbitrary dimension cutting.

