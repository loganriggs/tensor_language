# Carry key-span reuse through the composed response

Use unchanged native parent and rank64 directional-response factors. Replace the
577-coordinate shared reader by513coordinates (Q1,K1,Q2,K2,value), deriving its
former64basis reads from existing key outputs with C256x64. Remove the same64rows
from folded read_left and read_direction, retain all norm/background/factor terms
and inside adapters. This is exact versus the old compiled response, not versus
the native body when its rank64 producer approximation was inaccurate.

A: expanded features/scalars and supplied norm identities<=1e-10relative over
all72existing cached contexts. B: total serialized tensor scalar count reduces
at least7%versus871363, charging C and adapters. CPU120seconds/two threads, no
fitting or body forwards. If passed, price real execution before stronger
runtime claims. No fresh/OOD or new selective-circuit claim.
