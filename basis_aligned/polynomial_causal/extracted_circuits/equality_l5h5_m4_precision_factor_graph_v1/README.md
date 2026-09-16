# Equality L5H5 M4 precision factor graph V1

This package replaces the 15-term Boolean port lattice with four exact typed
nodes.  For derived/native first QK factors `a0,a1` and second factors `b0,b1`,
the algebraic nodes are `(a1-a0)b0`, `a0(b1-b0)`, and
`(a1-a0)(b1-b0)`.  A fourth node stores the native BF16 arithmetic residual
between that real-arithmetic expansion and the actual child score.

The residual is essential, not cosmetic.  Omitting it caused `.339/.353`
natural/code closure error in the invalid V1 instrument.  With it explicit,
score closure and behavioral replay are exactly zero-error on 192 natural and
192 code-OOD documents.

All four node removals are behaviorally active and selective.  On code, their
copy-positive effect norms are `.834` (first), `.859` (second), `.688`
(algebraic cross), and `.848` (arithmetic residual) relative to exact
interaction removal.  Their absolute noncopy mean changes are at most
`.000507` nat.  The arithmetic node's one-query roll preserves full score norm
exactly but has only `.511` behavioral-effect cosine to true removal.

The algebraic cross illustrates why norm pruning is unsafe: it is only `.00122`
of correction-score norm on code, yet its removal has a `.688` behavioral
effect.  The package has zero learned parameters and composes exactly, but it
still consumes the 16 native Q/K projections and leaves upstream factor ports
external.
