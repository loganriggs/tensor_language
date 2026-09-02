# Rung497 preregistration — finite causal-action archive coverage

**Registered:** 2026-09-02 17:42 UTC

**Parent:** lawful rung496 strong null (`A=true, B=false`)

**Claim level:** CPU provenance and experiment-design audit. This rung does not identify a circuit.

## Question

Can the existing post-02:19 archive support an intervention-conditional causal quotient across attention and MLP
boundaries without another model run? A usable quotient must group candidates by what the model does under several
finite actions, not by one activation vector, derivative fingerprint, rank, or reconstruction score.

The audit asks which candidate-by-action cells already contain comparable per-example effects, document splits,
dedicated task conditions, and downstream circuit conditions. It then chooses the smallest prospective collection
needed to test operational grouping with a known positive and a known negative control.

## Frozen candidate families

The audit includes five families fixed from completed results:

1. **Equality attention terms:** the four L5H5/L7H3/L8H3/L8H4 equality terms and the score-versus-output split.
2. **Downstream equality corrections:** MLP8/MLP9/MLP12 current-query writes and attention14/MLP17 suppression.
3. **MLP0 branches through block1:** exact T/C/I/S contributions and their attention1/MLP1 responses.
4. **MLP1 write adjustments:** T/I own restoration and cross-document donor substitution.
5. **Attention1 factor-attributed writes:** the rung495b score/value pieces and rung496 Q1/K1/Q2/K2/V pieces.

The source manifest freezes the exact completed receipts supporting each family. It may not add a favorable artifact
after calculating coverage. An artifact with a failed instrument is excluded; a lawful strong null remains valid
coverage of the action it actually performed.

## Frozen action alphabet

The minimum operational alphabet is:

- `remove`: delete one candidate from the relevant model trajectory;
- `restore`: reinstall the candidate's own finite contribution into its absent background;
- `substitute`: install another candidate's contribution or computation in its place, in both directions when
  equivalence is claimed; and
- `compose`: apply at least two candidate changes together and retain the joint result.

`local_derivative` is recorded but does not satisfy any finite-action requirement. Pairwise equalization is recorded
as `equalize`; it does not count as bidirectional substitution because it replaces both states with their average.

## Required observations

A family is archive-ready only if all four finite actions above are available with:

1. effects retained separately for individual examples or documents;
2. at least two fixed document splits;
3. the same action semantics across the candidates being compared;
4. complete downstream recomputation after the intervention;
5. a dedicated task condition where the proposed computation is live; and
6. the existing downstream circuit battery, with discovery and held-out circuit families or an unopened held-out
   role that can be used prospectively.

Aggregate summaries do not become per-example data. A circuit-gradient contraction does not become a finite action.
Missing cells are reported as missing and never imputed.

## Registered predictions and routing

### A — audit integrity

Every frozen receipt exists, reports a completed lawful instrument, and matches its registered bundle hash when a
bundle is claimed. The manifest has no duplicate evidence IDs and every action belongs to the frozen alphabet.

### B — existing archive is already action-complete

At least one candidate family satisfies all four finite actions and all six observation requirements above.

### C — existing archive supports transition refinement

At least one archive-ready family also has a finite action applied after another action, so equivalence can be
checked for closure under the action alphabet rather than only at the starting state.

### D — successor choice

- If A fails, repair only the audit; no scientific successor is licensed.
- If A/B/C pass, build the quotient and freeze discovery/confirmation partitions without a new GPU collection.
- If A passes but B or C fails, preregister the smallest collection filling the common missing cells. The collection
  must contain a known-positive interchangeable computation and a matched known-negative split, so the quotient
  method has an assay-sensitivity test before it searches for new groups.

The pre-outcome expected result is that A passes while B/C fail: the archive contains strong but heterogeneous
interventions, and no family combines finite actions, per-example outputs, task masks, and downstream circuits under
one shared design. This expectation does not change the gates.

## Literal price

CPU only. Read JSON receipts and metadata; inspect only bundle keys/shapes and verify bundle hashes. Do not load the
model, tokens, logits, hidden states, or any sealed role. Output one JSON receipt and one CSV evidence matrix. The
audit saves and adds zero deployed parameters.

## Anti-rank and anti-duplication rule

No clustering, matrix factorization, SAE, rank choice, similarity threshold, or missing-cell fit is allowed in this
rung. The old MLP0 quotient and generic balanced-truncation/Hankel proposals are not rerun. The only question is
whether finite operational evidence already forms a closed action table and, if not, exactly what must be collected.
