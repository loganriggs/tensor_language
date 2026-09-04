# Task-21 predictive-state localization requirements — 2026-09-04

Status: prospective design requirements only.  These requirements neither open localization nor authorize GPU work.
They apply only if the frozen task-21 FIT capability screen passes its unchanged gate.

## Required observable

For every retained FIT row-side, save the logits of all 21 registered phase candidates in frozen candidate-ID order.
The primary response is the mean-centered vector

$$
r_a=\ell_a-\frac{1}{21}\sum_b\ell_b.
$$

This makes two states equivalent only when they predict the same relative preference over every registered candidate;
answer-versus-largest-foil margin alone is insufficient for localization.  The raw evidence price is exactly
$168\times21\times4=14{,}112$ bytes if every current FIT row-side is retained as `float32`.

## Required high-level variables and controls

- Token identity: the immediately preceding token, with 21 possible values per phase.
- Older-token conflict: A2 must distinguish the newest run from the older visible target.
- Irrelevant prefix: P must preserve the response after changing only the registered leading filler.
- Repeat strength: C may reveal a continuous confidence variable but must preserve token identity.
- Direct-path alternative: always compare the final-token embedding/residual path with any proposed attention or MLP
  path.  A native-head effect is not evidence of remote copying.

## Required causal tests

For each proposed subspace, use multiple valid recipient/donor pairs implementing the same token-identity change.
Require the complete candidate response to move toward the donor under interchange.  Separately test necessity by
clamping the proposed state back to the recipient and sufficiency by injecting the donor state without the upstream
donor.  Evaluate held-out prompts, out-of-distribution prompts, and unrelated circuit controls.  At bilinear readers,
reconstruct the finite write change exactly from the left, right, and joint terms before attaching a weight-level
interpretation.

## Non-claims

Response-matrix dimension, Hankel rank, activation rank, retained variance, and CE preservation are not circuit
identification on their own.  A task-21 result cannot be called induction or remote retrieval.  The goal is an
operationally defined state plus a physical, selectively manipulable implementation that generalizes and composes.

