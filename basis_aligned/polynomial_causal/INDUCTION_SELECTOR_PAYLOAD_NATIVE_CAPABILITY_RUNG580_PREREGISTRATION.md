# Rung 580 preregistration: repaired induction selector × payload native capability

**Frozen:** 2026-09-03 18:50 UTC, after review of R578 and before opening any R580 model output

## Question and stopping rule

Does the unmodified bilin18 model implement the selector × payload behavior in the repaired R578 three-source rows?
This is a behavior-level capability test. It does not choose a head, site, subspace, rank, or regularizer.

Only FIT and SELECT may be opened. FINAL_TEST and OOD remain closed. If any scientific gate below fails, R580 writes
a complete scientific-null result with every failed cell and stops; it does not search model sites or crash merely
because a prediction failed. Integrity failures such as a hash mismatch, missing row, duplicate sequence evaluation,
wrong split, or wrong forward count remain hard errors.

Frozen inputs:

- R578 rows SHA-256: `8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6`
- R578 receipt SHA-256: `9e4e63ebd98503d6aa5daa27617a20fea595829c5a372f27b1ce4371d7c05b45`
- R578 construction preregistration SHA-256:
  `276d801bbf5795e6421488dd4971b3a2d2dcb56e4fc7c4bc7ecdd2f61a73e9ce`
- R578 builder SHA-256: `d47bb3d46bd2c6061132c13b356e58ba9dfe2a56a2629f8b49a03f280d290bbd`
- R578 focused tests SHA-256: `9d795df358dfef9c5d17a539307f8e781f2a4debeb4909078858a242b3dfc512`
- model checkpoint weights SHA-256 is the value enforced by `bilin18_observed_model_facade`.

## Exact measurements

For every group, let `B,D` be the two target payloads and define

$$z(x)=\operatorname{logit}_x(B)-\operatorname{logit}_x(D).$$

For a prompt whose correct target payload is `a` and other target payload is `a'`, define

$$m(x)=\operatorname{logit}_x(a)-\operatorname{logit}_x(a').$$

The four-cell interaction is

$$I_g=\tfrac14(z_{g,00}-z_{g,10}-z_{g,01}+z_{g,11}).$$

For each selected-match-break row, define

$$d=m_\text{base}-m_\text{broken}.$$

Pair that row by exact group and `(S,P)` cell with the neutral-source and neutral-payload edits. Define

$$
g=d-\max\bigl(|m_\text{base}-m_\text{neutral-source}|,
              |m_\text{base}-m_\text{neutral-payload}|\bigr).
$$

This compares selected-match necessity with two endpoint-neutral edits. The contrast-target-source edit is reported
separately because it changes the context immediately before the competing target payload; it is not an invariance
gate.

## Frozen scientific gates

All gates apply separately to FIT and SELECT.

1. **Four factorial cells.** In every `s0p0,s0p1,s1p0,s1p1` cell, at least 75% of groups have `m>0`, and the 95%
   bootstrap lower bound on mean `m` is strictly above zero.
2. **Selector × payload interaction.** The 95% bootstrap lower bound on mean group interaction `I_g` is strictly
   above zero.
3. **Relation-preserving controls.** For each of neutral-source, neutral-payload, filler, and lag changes, for every
   `(S,P)` cell and for base and donor endpoints separately, at least 75% of groups have `m>0`, and the 95% bootstrap
   lower mean margin is strictly above zero.
4. **Selected-match necessity.** Across the four selected-match rows per group, at least 70% of rows have `d>0`, and
   the group-bootstrap 95% lower bound on mean `d` is strictly above zero.
5. **Selected versus neutral controls.** The group-bootstrap 95% lower bound on mean paired gap `g` is strictly above
   zero.

Every clause must pass in both splits. Equality at a threshold fails. Contrast-source effects are summarized by
split and `(S,P)` cell, with mean signed margin change and a two-sided bootstrap interval, but never enter the pass
decision.

For compatibility with the managed runner, the five clauses are also reported under three top-level predicates. This
is only a grouping of the frozen clauses, not a change to a threshold:

- `pred_a_native_factorial_and_controls` is true exactly when clauses 1 and 3 pass in both splits;
- `pred_b_selector_payload_interaction` is true exactly when clause 2 passes in both splits;
- `pred_c_selected_match_necessity_and_neutral_selectivity` is true exactly when clauses 4 and 5 pass in both splits.

The overall capability gate remains the conjunction of these three predicates. This runner-interface clarification was
frozen at 2026-09-03 19:07 UTC, after the queue preflight rejected the otherwise unexecuted script for having no
top-level predicate fields and before any R580 model output existed.

## Exact bootstrap

Every interval uses 2,000 group-cluster bootstrap replicates. Group IDs are sorted lexicographically. For cell ID
`c`, replicate `b`, and draw `k`, choose cluster index

$$
\operatorname{int}\bigl(\operatorname{SHA256}(
\texttt{a8-r580-group-bootstrap-v1:c:b:k})[0:8]\bigr)\bmod n,
$$

where the first eight digest bytes are interpreted as an unsigned big-endian integer. A chosen group contributes all
of its rows in that cell. The lower bound is NumPy quantile `0.025, method="lower"`; the contrast diagnostic also
reports `0.975, method="higher"`. Cell IDs and ordered group IDs are saved so an audit can reproduce every draw.

## Saved evidence and literal price

The implementation must save:

- one native sequence measurement for each unique FIT/SELECT prompt, including stable sequence/group/split IDs,
  length, `B,D` logits, both target-token cross-entropies, and the log-normalizer;
- all 3,240 FIT/SELECT row measurements, including row/group/family/cell IDs, base and donor margins and
  cross-entropies, and the paired effect;
- all group-level factorial margins, signed `z` coordinates, `I_g`, selected drops, neutral effects, paired gaps,
  and contrast effects;
- every aggregate, bootstrap cell ID, ordered cluster IDs, interval, pass flag, failed clause, and terminal decision;
- exact input/code/preregistration/checkpoint/result hashes, model forwards/backwards, and opened splits.

The 108 FIT/SELECT groups contain exactly 3,024 unique prompts. Each is evaluated exactly once in batches of 32, for
exactly `ceil(3024/32)=95` model forwards, zero backwards, and no weight updates. A dry run must exercise authority,
sequence ownership, planted passing and scientific-null decisions, raw-evidence construction, and the literal
95-forward price without loading a model or touching CUDA.

Passing licenses an independently preregistered adaptation of R557/R558 to the R578 rows. It is not itself circuit
identification. A null blocks factor/site search on these synthetic rows while preserving the complete raw evidence.
