# Suffix-transport v1 graph-identity and large-logit integrity boundary

Status: **prospective, nonauthorizing performance closure.** This note changes no
row, tensor program, basis, optimizer, selector, gate, or scientific interpretation.
It removes a prohibitive implementation cost before any fresh suffix role is loaded.

Student and autonomous teacher logits have shape `[4,256,50257]` and
`[4,192,50257]`. Copying and cryptographically hashing each complete tensor at both
issuance and consumption transfers roughly 393 MiB per student training transaction,
before the teacher copy. Across L/R/S/T and null trials this is multiple TiB of
unnecessary CPU traffic.

The transaction already has a narrower trustworthy boundary:

1. the observed model adapter owns the only student/teacher forward callbacks;
2. it passes the returned tensor directly into a sealed one-use transaction;
3. it closes and consumes that transaction before releasing a caller alias;
4. source hashes bind the adapter and capability implementation.

The abstract capability deliberately accepts reduced vocabulary dimensions in CPU
tests. The real observed adapter must additionally require the exact production
dimension 50,257 before binding either student or teacher logits. This check belongs
to the adapter because that is the source-closed boundary that knows the concrete
model contract.

Within that boundary, full logits use a process-local graph/storage witness rather
than a content digest. The witness binds exact Python tensor identity, storage pointer,
PyTorch mutation version, shape, dtype, device, stride, storage offset,
`requires_grad`, grad-function presence/type, and finiteness at issuance and
consumption. The deterministic closure hash binds this logical descriptor and the
full trace identity; it is explicitly a **logical graph binding**, not a
cryptographic digest of all logit bytes.

Rank-64 code tensors and coordinate labels remain byte-hashed because they are small.
Autonomous OON teacher logits use the same structural witness after an owned detached
clone. Public ordinary in-place mutation, storage/shape replacement, detach, graph
replacement, and nonfinite `.data` mutation fail. Finite hostile `.data` mutation in
the private mint-to-consume interval is outside the public threat surface and would
require a source-closure change; the real adapter must not expose that interval.

This optimization does not address the separate extra-backward connectivity probe.
That probe remains required until benchmarked or prospectively replaced.
