# Ordering the global residual and shared-value gauges

The V/O compiler already quotients one `GL(128)` value-coordinate gauge per head,
shared across all 18 layers by the `v0` bus. The newer global residual gauge acts on
the other leg of the same matrices. For one head,

\[
V_l\mapsto G V_l Q,\qquad
O_l\mapsto Q^\top O_l G^{-1},
\]

where `Q in O(1152)` changes residual coordinates and `G in GL(128)` changes value
coordinates. The actions commute because they act on different tensor legs:

\[
G(V_lQ)=(GV_l)Q,
\qquad
(Q^\top O_l)G^{-1}=Q^\top(O_lG^{-1}).
\]

On the generic stratum, the combined continuous gauge dimension is

\[
\frac{1152(1151)}2+9(128)^2=662{,}976+147{,}456=810{,}432.
\]

This is a parameter-orbit dimension, not a bit discount.

## Canonicalization order

Although the group actions commute, the existing value canonical section projects
lexicographically ordered ambient residual coordinate axes into the row space of
`V0`. It therefore presupposes a fixed residual frame. The correct compiler order is:

1. fix the global residual `O(1152)` frame from the versioned embedding anchor;
2. transform every residual reader and writer, including all V/O maps;
3. fix one shared-depth `GL(128)` section per value head;
4. produce Q/K route bytes in the same globally canonical residual frame;
5. sort complete `(route bytes, V/O bytes)` pairs under the one common `S9` action.

`nested_residual_value_gauge.py` implements steps 1--3 and connects them to the
existing Q/K-keyed V/O container for step 5. Randomized CPU tests apply an arbitrary
global orthogonal rotation, a different nonorthogonal value gauge to each head, and
a common head permutation. Both canonical tensors and final quantized V/O container
bytes remain identical.

The first adapter treated route keys as already canonical external identities. The
follow-up `nested_attention_gauge_codec.py` now closes that boundary: it transforms
Q/K/Q2/K2 reader maps into the same embedding-anchored residual frame before invoking
the production dense or low-rank head codec. Dense and low-rank route bytes are
identical after an arbitrary global rotation. The full wrapper then uses those bytes
to bind the V/O payload and fix the common head permutation.

A joint randomized test simultaneously applies global residual rotation, existing
even-parity/branch-swap Q/K gauges, independent headwise nonorthogonal value gauges,
and common head permutation. The route multiset and final route-bound V/O bytes are
identical. This closes the continuous/discrete gauge-order proof for the CPU codec
stack. It does not retrofit these bytes into an already frozen production artifact or
license inherited behavioral scores; that requires a new versioned rewrap and replay.

## Accounting consequence

The existing V/O `GL(128)^9` quotient remains valid; it was not made obsolete by the
global residual gauge. But its bytes become globally quotient-comparable only after
the residual frame is fixed first. Conversely, fixing the global frame does not allow
subtracting another independent value gauge per layer: `v0` still ties the 18 layers.

This is the tensor-network canonical-form discipline in concrete compiler form:
shared virtual-leg gauges are fixed at their true fan-out scope, and canonical sections
are composed in dependency order. No operational score, checkpoint price, or bit
saving is promoted by the CPU proof alone.
