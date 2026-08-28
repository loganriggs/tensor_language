# Tensor-preserving bilinear MLP identity gate

Date: 2026-08-28 09:45 UTC

Status: role-free executable identity protocol. No corpus, fit, selection, final, or
promotion authority.

Construct all 18 MLP programs densely from the pinned checkpoint and execute

$$
y=\operatorname{Down}(\operatorname{Left}(x)\odot\operatorname{Right}(x))
+b_{\rm Down}
$$

directly through `forward_with_dispatch`. During the program arm every native MLP
object is physically replaced by a forbidden module. The gate passes only if:

1. all 18 offline same-input writes and all 18 sequential program-trajectory writes
   equal native writes bitwise;
2. final unsliced logits and deterministic synthetic CE equal native bitwise;
3. every MLP program dispatches exactly once, literal native MLP calls are zero, and
   all native attention calls remain exactly once in both arms;
4. bank order, block identity, closure, replacement restoration, and storage
   disjointness pass;
5. complete storage includes Left, Right, Down, and the load-bearing Down bias at every
   site, including the exact `Down_bias`, with total input support and no token table
   or fallback.

This gate earns only an executable dense identity denominator. It licenses no MLP
compression or claim about the failed unseen-token table program.
