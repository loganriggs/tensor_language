# Causal-response factorization v1 — amendment 5

Status: controlling and frozen before any FIT response bundle is deserialized. This
amendment defines the source-closed training-input transaction. It does not authorize
execution until an independent audit bound to the exact source closure returns GO.

## Transaction

The production entrypoint accepts no arguments. Under one create-only owner lock it:

1. validates the completed FIT receipt as opaque bytes and computes the outcome-blind
   parent binding;
2. replays the exact published analysis source closure and independent audit;
3. publishes factor-training authority before `torch.load`, binding the parent,
   protocol, paths, and zero outcome access;
4. spends a one-use loader capability, stable-loading only the exact receipt-named
   bundle bytes with `BytesIO(...), map_location="cpu", weights_only=True`;
5. replays full bundle semantics, manifest summary, checkpoint/model-state/ledger
   joins, and the terminal parent binding;
6. passes the private payload once to the training-only adapter, then deletes the
   payload and raw-byte aliases;
7. publishes a sanitized 229-document training artifact create-only after an adjacent
   authority/source/parent guard;
8. publishes its manifest; and
9. publishes either a same-inode shared terminal plus receipt, or shared terminal plus
   failure.

The sanitized artifact contains signed response, validity, training document IDs and
their original FIT indices, sealed axes/owner topology, exact parent identities, and
tensor digests. Its exact schema forbids validation responses or IDs, EVAL values, the
full FIT payload, tokens, targets, activations, logits, and models.

## Capability and claim boundary

The loader poisons itself before its first file lookup. A failed attempt cannot be
retried. The training authority and receipt authorize the sanitized artifact only as
a parent for a separately audited candidate-fitting transaction. They do not
authorize candidate selection, validation, EVAL, semantic naming, circuit removal, or
strict-ledger credit.

Candidate fitting may later read the sanitized training artifact without receiving
the original 343-document bundle. Candidate topology, seeds, optimizer, and prices
remain frozen by the preregistration and amendments 1–4. Validation stays in a later
source closure after all candidate programs and health results are frozen.

## Gate

Production execution requires an immutable independent GO that verifies every source
hash, authority-before-load ordering, exact-byte joins, role exclusion, create-only
publication, owner races, and failure semantics. Synthetic green tests are not
self-authorization.
