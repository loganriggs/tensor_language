# Is there a scalable extractor? (layer-by-layer / tensor networks)

A depth-`L` bilinear net computes a polynomial of degree `2ᴸ`, whose full tensor has
`n^(2ᴸ)` entries — exponential in **both** `n` and depth. You can never materialise it
past tiny cases. But you don't have to, and your instinct (layer-by-layer + the
canonicalisation giving a "more global view") is exactly right. Here's the theory.

## What the extractor actually needs: contractions, not the tensor

Both pieces of the matrix/tensor method are **contractions**, and a contraction of a
tensor given in **factored (per-layer) form is polynomial** — you contract the network
leg-by-leg, never building the full object:

- **subspace** = top eigenvectors of a covariance of slices;
- **CP / Jennrich** = `eigh(M_a M_b⁻¹)` of two **slices** `M_a = T(·,·,a,…,a)`.

For 2 layers we verified the slice is exact from the `n×n` factored forms (`Acheck`,
`Bcheck`), `O(h2·n²)` each, with **no `n⁴` tensor** (`bilinear2_scalable.py`, err 1e-15):

    M_a = Σ_p Wo[p][ (1/6)(A_p (aᵀB_p a) + B_p (aᵀA_p a)) + (1/3)((A_p a)(B_p a)ᵀ + (B_p a)(A_p a)ᵀ) ]

That is the "layer-by-layer" computation: a forward pass with most legs frozen to `a`.
It generalises — contracting the depth-`L` tensor network with `a` on its legs is a
sweep through the layers, **polynomial in `n`, the widths, and depth**. So *computing*
the objects the extractor needs is scalable at any depth; the exponential only appears
if you insist on writing the tensor out.

## The catch: it's the bond dimension, not the compute

Jennrich/CP recovers the secrets only if the tensor is (close to) the clean low-rank
form `Σ_s sₛ⊗…⊗sₛ` — rank ≈ #secrets. The relevant quantity is the **bond dimension**:
the rank of the network across each cut between layers (how much of the secret subspace
one layer mixes before the next reads it).

- low bond dim ⇒ the factored slices are low-rank ⇒ Jennrich bites (the ideal tensor:
  16/16, even non-orthogonal);
- high bond dim ⇒ the slices are full-rank / messy ⇒ Jennrich fails (our **trained**
  organisms: 0/16, the subspace captures only ~50% — the layers smear the secrets across
  many directions).

So the bottleneck isn't compute, it's whether the *trained model* keeps a low-bond-
dimension representation. The messiness we keep hitting **is** high bond dimension.

## Why naive layer-by-layer fails, and what fixes it (= your canonicalisation point)

A purely **local** layer-1 analysis weights all of layer 1's output directions equally.
But layer 2 only *reads* certain combinations, weighted by what flows to the output — so
"directions layer 1 writes that layer 2 barely reads" get over-counted locally. The
global object correctly down-weights them.

The reconciliation is **bond canonicalisation** (the MPS/DMRG canonical form, and
`../basic_circuits/two_layer/` thread #4): SVD each inter-layer bond and absorb the gauge
so that each layer's local forms are expressed in the basis that reflects **downstream
importance**. Then a layer-by-layer sweep *is* the global view — you propagate the
output's sensitivity backward through the bonds (a backward contraction), exactly as DMRG
sweeps a tensor train. This costs `O(poly · bond_dim)`; it is efficient precisely when
the bond dimension is small.

## Summary

| | cost | when it works |
|---|---|---|
| materialise full tensor | `n^(2ᴸ)` (exponential) | never, past toys |
| factored slice / covariance (sweep) | `poly(n, width, depth)` | always computable |
| recover the secrets (CP/Jennrich on the slices) | `poly` | only if **bond dimension is small** (clean, low-rank storage) |

So: a scalable layer-by-layer extractor exists and is exactly the tensor-network / DMRG
picture with bond canonicalisation. It succeeds iff the trained organism stores its
secrets at low bond dimension — which, empirically, naturally-trained nets do **not**
(they smear across the bond), which is why they resist detection.
