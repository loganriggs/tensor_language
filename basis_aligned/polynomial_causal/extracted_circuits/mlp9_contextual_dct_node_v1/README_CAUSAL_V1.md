# Causal use of the rank-16 contextual DCT node

The packaged `execute.py` response can be installed or removed as an explicit
local MLP9 interaction.  For the prospectively tested diagonal case, let
`p = scale² * execute(z9, program)[direction, direction]`, let `h2` be the
native MLP9 state under the combined input intervention, and let `ha` be the
sum of the two singleton states minus the base state.  Then:

- installation state: `ha + p`;
- removal state: `h2 - p`.

With frozen `direction=2` and `scale=32`, a new 64-prefix panel gives `.00567`
relative error for `p` versus the exact finite local interaction and `.00430`
for its exact-suffix all-logit installation effect.  Removal leaves `.03696`
of the native all-logit mixed effect and `.01397` at the frozen token reader.

This is an induced, synthetic local-interaction interface.  It retains native
`z9` and the native downstream suffix.  It is not evidence for a naturally
occurring semantic feature or a complete input-to-output circuit.  Stronger
structure-matched controls support reader specificity, but alternate DCT pairs
can produce up to `.9643` as much total all-logit change, so global residual
specificity is not established.
