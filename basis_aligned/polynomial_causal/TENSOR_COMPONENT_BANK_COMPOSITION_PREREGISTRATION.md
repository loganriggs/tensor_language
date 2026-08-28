# Simultaneous attention/MLP component-bank identity

Date: 2026-08-28 10:03 UTC

Status: role-free composition gate. No corpus, fitting, compression, selection, or
promotion authority.

The attention and MLP banks passed separate dense identity gates. This protocol tests
their missing interface: install all 36 owned component programs simultaneously through
`forward_with_dispatch`, physically replace all 36 native attention/MLP objects, and
require bitwise equality to a native reference on the deterministic production-shaped
fixture.

Frozen gates:

1. each attention write, v1 bus, and MLP write equals its native-trajectory reference;
2. final logits, logit hash, and synthetic CE are bitwise identical;
3. attention and MLP transactions each dispatch sites 0..17 exactly once with exact
   block and v1 identity closure;
4. literal native attention and MLP calls are both zero, replacements restore exactly,
   and attention/MLP/native tensor storage sets are mutually disjoint;
5. complete component storage equals the sum of the two frozen dense denominators,
   430,003,602 values, including attention lambdas/rotary constants and MLP Down biases.

This is a 36-component-core identity, not yet an entirely owned model. The facade still
executes the checkpoint's token embedding, 18 residual lambda pairs, RMSNorm operations,
final unembedding, and softcap. The result must list those unowned exact interfaces so
they cannot disappear from later simplicity prices.
