**Equivalent quartic representatives recover compact planted hierarchies**

All five scalar controls pass. Each canonical minimum-Frobenius Gram matrix has rank15, while signed affine/rank projection recovers the planted rank1 or rank2 representation below1e-10 relative symmetric coefficient error in every one of four starts. Convergence takes37–253iterations. Independent polynomial execution confirms canonical and planted representations agree below1e-12.

The five cases are radial square, rotated dense quadratic square, signed difference of squares, product of two quadratics, and a shared quadratic factor multiplying a sum. They are controlled structural examples with supplied target ranks, not a claim of arbitrary circuit recovery. The last family tests algebraic reuse but overlaps expressively with the quadratic-product family.

The construction writes f(x)=z(x)^T K z(x), where z contains orthonormal quadratic monomials. A linear map converts K to fully symmetric quartic coefficients. Alternating projection changes K within the function-equivalence class and truncates its signed spectrum. Unlike truncating one canonical unfolding, this explicitly searches polynomial identities induced by repeated inputs. It has no convergence or global-minimum guarantee.

A rank-r K proposes r quadratic features with signed squared outputs. This is a root-level initialization, not the full arithmetic cost: a dense quadratic feature still needs its own decomposition. Sharing those leaf computations and coordinating multiple output banks remain necessary. Native model nonlinearities are outside this polynomial representation.

[Gram spectrahedra of ternary quartics](https://arxiv.org/abs/2112.10533) provides the relevant affine Gram/PSD context. Our matrices may be indefinite, so PSD existence and rank results are not used as guarantees for signed model outputs.

A native follow-up is registered on all16 saved five-input contexts, four scalar output coordinates independently. These are restricted amplitude targets, not the full1152input tensor. Results belong in NATIVE_QUARTIC_GRAM_SEARCH_V1.json when execution finishes; no native success is inferred from the controls.

[Toy results](QUARTIC_GRAM_SEARCH_TOYS_V1.json) · [Plan and native successor](QUARTIC_GRAM_SEARCH_PLAN_V1.md) · [Algorithm](quartic_gram_search.py).
