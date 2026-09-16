# Research update: an OOD sparse source graph above the equality score

The DCT briefing's most useful instruction here was to keep context slots open
and judge decompositions by interventions.  We therefore did not distribute RMS
normalization across sources or rank sources by activation norm.  We first
extracted the complete L5H5 computation from one residual-state port, then
selected sources only by the resulting equality-edge score.

## What is now extracted

The verified path is:

`L2/L3/L4 native writes -> L5 residual context -> four frozen L5H5 Q/K projections -> Q/K RMS + rotary -> causal product score -> frozen scalar adapter -> exact reversible L8H4 equality node`.

The full residual-to-score executor has zero measured score and downstream-logit
error and learns no parameters.  It reuses 589,824 native Q/K weight values; it
is a port/execution extraction, not yet weight compression.

## Sparse source result

We decomposed the pre-L5 residual exactly into embedding and five layer-write
groups, `E,L0,...,L4`.  All 63 nonempty subsets were evaluated on 192 natural
documents without looking at behavior.  The frozen rule chose the smallest
support below `.15` score error and above `.98` cosine.  It selected exactly:

`L2 + L3 + L4`.

Natural score error/cosine was `.14462/.99197`.  Without reselection on 192 code
documents, error/cosine was `.15268/.99233`, stable across halves.  Installing
that score through the already extracted downstream graph recovered `.90458`
of the equality-removal stake; every registered copy cell and half passed, and
noncopy mean damage was `.00172` nat.  The complete source Möbius expansion
closed to `3.94e-9`, so its interactions form an explicit reusable graph rather
than an assumed additive story.

## Positive and negative red-team

The provenance ledger initially retained a `.00365`-norm FP32/BF16 correction
port.  Because Q/K RMS can amplify direction independently of amplitude, we did
not accept its small norm as proof of irrelevance.  A preregistered audit zeroed
it, repeated all 63 natural subsets, and tested an equal-norm one-position roll.

The correction-free run selected the same `L2+L3+L4` support, slightly improved
code score error to `.15255`, and changed recovery by only `-.00020`.  Rolling
the correction changed recovery by `.00287`.  All six controls passed.  The
sparse graph is therefore not secretly using the roundoff direction.

This follows the earlier causal-mask episode in the other red-team direction:
an apparent `1.0932` score failure was a missing-mask diagnostic bug, caught by
its simultaneous zero logit error.  After correction the score error was exactly
zero.  Both episodes support treating contradictions as implementation evidence
before calling a scientific null.

## Honest remaining boundary

`L2`, `L3`, and `L4` are still native frozen write ports.  We have not yet
recursively extracted the heads/MLPs and source positions that generate those
writes.  L8H4's raw payload is also still native.  The next useful experiment is
a prospective within-layer factorial for the three selected write groups,
starting at L4 and requiring frozen code transfer and source-interaction closure;
another local L5 score basis would not advance the graph.
