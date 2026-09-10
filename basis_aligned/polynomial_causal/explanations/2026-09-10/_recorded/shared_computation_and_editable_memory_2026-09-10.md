# Shared computation can require separate editable memories

September 10, 2026. This is a mathematical follow-up to the original
[handoff](bilinear_circuit_reconstruction_codex_handoff.md) and
[pilot report](bilinear_reconstruction_pilot_report.md).

**We have not recovered a new trained-model circuit.** The latest prefix-binding
candidate failed its explanatory tests, and the older history-lookup branch had
already failed its stronger semantic compiler tests. Those branches stay closed.
This follow-up instead makes one useful idea from the pilot executable: share a
calculation when its inputs and operation agree, but split its accumulated history
when independent consumer edits require that distinction.

The high-level sequence was: check the old reports to avoid repeating their failed
proposals; derive the state requirement from the allowed edits and actual readers;
implement an exact compiler; and check independent and joint removals. These are
CPU algebra calculations, with no model training, fitting, or GPU forwards.

## The distinction that matters

Suppose three consumers use the same four quantities computed from each source:

\[
\phi(x,y)=(x^3,x^2y,xy^2,y^3).
\]

These four products can be computed once. Each consumer then adds them to its own
running total. A source mask says which consumers receive a particular update;
for example, `(1,1,0)` sends it to A and B but removes it from C.

| Allowed source histories | Required linear memory coordinates in the tested fixture |
|---|---:|
| All three consumers always receive the same updates | 4 |
| A and B stay tied; C can receive different updates | 8 |
| All three consumers can receive independent updates | 12 |
| Independent updates, but readers only use two of the four features per consumer | 6 |

The feature library still has four entries in every row. Memory width and shared
arithmetic are different quantities. The final row also shows why actual consumers
matter: a stored difference can be discarded if no allowed query can read it.

This generalizes the pilot's existing four-versus-eight example; it does not
rediscover its result or establish learned semantic structure.

## The exact compilation rule

Let the physical memory be a vector `s` with `n` entries. An allowed edit setting
`g` selects an update matrix `U_g`, and each source updates the memory by

\[
s' = s+U_g\phi(x),\qquad s_0=0.
\]

The feature producer may be nonlinear. Here it is supplied explicitly and does
not depend on the memory being reduced. Let `H` stack all coefficient rows used
by the allowed query readers. For example, if a reader uses a quadratic query,
stack its coefficients for every distinct query monomial, then evaluate those
monomials after reading the memory.

Let `E` contain a basis of the columns of all the `U_g` matrices. This is the
**reachable linear space**: all directions that source updates can generate.
Then the minimum memory dimension among linear encoders preserving these reads
on that space is

\[
m=\operatorname{rank}(HE).
\]

Here rank counts independent observable directions; it is a consequence of the
declared computation and edit policy, not a rank hyperparameter to tune.
Select `m` independent rows of `HE`, and let `C` be the corresponding rows of
`H`. There is an exact decoder `D` such that `HE=DCE`. The compiled program is

\[
z'=z+(CU_g)\phi(x),\qquad \text{reader coefficients}=Dz.
\]

It never reconstructs the original physical memory. Since every reachable state
is `s=Ea`, the identity `Hs=DCs` proves read fidelity. Additivity proves closure
after each allowed update, removal, or donor-source substitution. Conversely, a
linear encoder with fewer than `rank(HE)` coordinates cannot retain all these
independent linear reads. This minimality claim is restricted to the reachable
linear domain and linear encoders; arbitrary nonlinear encodings are outside it.

This is an additive special case of reachable/observable state reduction, rather
than a new general realization theorem. In the linear-system notation the state
transition is the identity matrix. See
[Frazzoli's MIT lecture on minimal realizations](https://ocw.mit.edu/courses/6-241j-dynamic-systems-and-control-spring-2011/c4b2f9d80e51e07a7f1ff6e7be4308e7_MIT6_241JS11_lec21.pdf).

For `k` consumers sharing an `r`-dimensional feature space, update matrices have
the form `g ⊗ I_r`: each entry of the mask multiplies a copy of the feature vector.
If masks can be selected independently at each source and the source features
span their stated domain, the reachable width is `r × rank(G)`, where `G` stacks
the allowed mask vectors. Applying `H` can reduce the observable width further.

## Executed checks and limits

[The compiler](../edit_policy_memory_compiler_v1.py) passes
[17 exact rational checks](../EDIT_POLICY_MEMORY_COMPILER_V1_CONTROLS.json),
including 57 stream-prefix replays, a dense change of physical state coordinates,
separate and joint branch removals, shared-producer removal, and rejection of an
unregistered independent edit. Exact means rational identities, with no numerical
rank tolerance. Four Vandermonde feature evaluations certify the fixture's feature
span, so the required widths are not inferred from a few correlated activations.

The same construction has now been executed on the **original pilot fixtures**,
using their existing 128-coordinate state changes and actual quadratic-query
reader coefficients. Exact update/read identities pass for all three:

| Original pilot fixture | Reachable width with independent edits | Compiled observable memory |
|---|---:|---:|
| Planted shared cubic updates | 8 | 8 |
| One key coefficient perturbed by 2^-20 | 11 | 11 |
| Shared values, independent routers | 38 | 37 |

The last row turns the pilot's previously reported observable rank into an
executable quotient: the unused state direction is physically absent from the
compiled accumulator. It does **not** turn independent routers into one shared
router. The update/read coefficient identities prove fidelity for every source
and query in the fixtures' polynomial domain, including independent source edits.
No trained checkpoint is involved. The compiled constant counts are respectively
960, 1,320, and 4,440; original feature/query producers remain required.

[The original-fixture receipt](../EDIT_POLICY_ORIGINAL_PILOT_V1_RESULT.json)
and [reproduction script](../audit_edit_policy_original_pilot_v1.py) preserve a
computational correction: generic symbolic elimination was stopped after more
than 120 CPU seconds, with two fixtures finished. Exact rational-domain
elimination and a square pivot solve completed all three in approximately
2.32 seconds. The coefficients and scientific identity were unchanged.

One necessary exception is also tested. If a mask is fixed for the entire run and
given to the decoder as external metadata, one four-coordinate history can serve
every fixed mask. The 12-coordinate claim concerns independently varying source
masks and a fixed encoder/reader contract. Changing available side information
changes the question. Likewise, a retrospective edit needs its original source
information or replay; that information is not free storage supplied by this proof.

Dense compilation is only a reference construction. For `a` masks, feature width
`r`, reader coefficient count `p`, and reduced width `m`, storing all compiled
updates and the decoder costs `a m r + p m` arbitrary coefficients. Feature and
query producers, mask metadata, and any retained background cost extra. Ordinary
elimination is cubic at its dense worst case, with additional rational coefficient
growth. No advantage over a competent hand-factored baseline is claimed here.

## Consequence for the research direction

When two candidate circuits use one module, test two different interventions:
remove the shared producer, which should affect all dependent consumers; and
remove a branch's updates, which should preserve the other branch's local history.
A decomposition that matches ordinary outputs can fail the second test because
it merged those histories. Conversely, failure of a shared-memory proposal does
not disprove a shared feature computation.

For attention this rule applies to an additive accumulator only after its key,
value, query, positional, and normalization producers are specified. In the full
transformer those producers depend on other state. The compiler does not prove
their closure or replace them with semantic inputs. It therefore supplies a
testable module-splitting rule and exact manipulation machinery, while OOD model
prediction, independent trained-circuit extraction, selective behavioral removal,
and a structurally simpler model remain unestablished.
